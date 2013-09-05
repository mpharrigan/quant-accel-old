'''
Created on Sep 5, 2013

@author: harrigan
'''
from fuzzy import classic, analysis, get_data

from matplotlib import pyplot as pp
import os
import numpy as np
from msmbuilder import clustering, MSMLib as msml
import sqlite3 as sql
from msmaccelerator.core import markovstatemodel
from scipy import optimize
import pickle

class ToySim(object):
    def __init__(self, directory, distance_cutoff=0.25, n_medoid_iters=1, n_timescales=1):
        self.directory = directory
        self.distance_cutoff = distance_cutoff
        self.n_medoid_iters = n_medoid_iters
        self.n_timescales = n_timescales


    def do_clustering(self):
        trajs = get_data.get_shimtraj_from_trajlist(self.traj_list)
        metric = classic.Euclidean2d()

        clustering.logger.setLevel('WARNING')
        hkm = clustering.HybridKMedoids(metric, trajs, k=None, distance_cutoff=self.distance_cutoff, local_num_iters=self.n_medoid_iters)
        self.clusterer = hkm
        self.n_clusters = hkm.get_generators_as_traj()['XYZList'].shape[0]

        print "Clustered data into {:,} clusters".format(self.n_clusters)

    def build_msm(self, lag_time):
        """Build an MSM from the loaded trajectories."""
        counts = msml.get_count_matrix_from_assignments(self.clusterer.get_assignments(), self.n_clusters, lag_time)
        rev_counts, t_matrix, populations, mapping = msml.build_msm(counts)
        return t_matrix

    def get_implied_timescales(self, lag_time, n_timescales):
        """Get implied timescales at a particular lag time."""
        t_matrix = self.build_msm(lag_time)
        implied_timescales = analysis.get_implied_timescales(t_matrix, n_timescales, lag_time * self.load_stride)
        print "Calculated implied timescale at lag time {}".format(lag_time)
        return implied_timescales

    def get_error_vs_time(self, start_percent=5, n_points=10, load_stride=1):
        percents = np.linspace(start_percent / 100., 1.0, n_points)

        errors = np.zeros((len(percents), 2))
        for i in xrange(len(percents)):
            percent = percents[i]
            self.load_trajs(load_stride=load_stride, load_up_to_this_percent=percent)
            self.do_clustering()
            its = self.get_implied_timescales(lag_time=self.good_lag_time, n_timescales=self.n_timescales)

            # Calculate error
            evec = np.log(its) - np.log(self.gold_its)
            errors[i][1] = np.sqrt(np.dot(evec, evec))
            errors[i][0] = self.wall_steps

            print "Calculated error from using {}% of all data".format(percent * 100.)

        self.errors = errors

    def __getstate__(self):
        state = dict(self.__dict__)
        try:
            del state['clusterer'], state['traj_list']
        except KeyError:
            pass
        return state


class Gold(ToySim):

    def load_trajs(self, load_stride, load_up_to_this_percent=1.0, verbose=True):
        self.load_stride = load_stride
        assert load_up_to_this_percent <= 1.0, 'Must load less than 100%'

        traj_list = get_data.get_trajs(directory=os.path.join(self.directory, 'trajs/'), dim=2)
        load_end = int(load_up_to_this_percent * traj_list[0].shape[0])
        print "Ending at index {:,}".format(load_end)
        traj_list = [traj[:load_end:load_stride, ...] for traj in traj_list]

        # Stats
        if verbose:
            n_trajs = len(traj_list)
            traj_len = traj_list[0].shape[0]
            print "{} trajs x {:,} length = {:,} frames".format(n_trajs, traj_len, n_trajs * traj_len)

        self.traj_list = traj_list
        self.wall_steps = load_end


    def get_implied_timescales_vs_lt(self, range_tuple, n_timescales=4):
        """Get implied timescales vs lag time."""
        lt_range = range(*range_tuple)
        implied_timescales = np.zeros((len(lt_range), n_timescales))

        for i in xrange(len(lt_range)):
            t_matrix = self.build_msm(lt_range[i])
            implied_timescales[i] = analysis.get_implied_timescales(t_matrix, n_timescales, lt_range[i] * self.load_stride)
            print "Calculated implied timescale at lag time {}".format(lt_range[i])

        self.implied_timescales = implied_timescales
        self.lt_range = lt_range


    def plot_implied_timescales(self):
        for i in xrange(self.implied_timescales.shape[1]):
            pp.scatter(self.lt_range, self.implied_timescales[:, i])
        pp.yscale('log')

    def fit_error(self):
        assert self.errors is not None, 'Calculate errors before fitting.'

        # Our functional form
        def error_func_model(x, a, b):
            return 1.e5 * a * np.exp(-b * x)

        # Optimize
        popt, _ = optimize.curve_fit(error_func_model, self.errors[:, 1], self.errors[:, 0])
        print "Optimized parameters", popt

        # Save the function
        def gold_walltime(error):
            return error_func_model(error, *popt)
        self.gold_walltime = gold_walltime

    def plot_fit(self):
        pp.plot(self.errors[:, 1], self.errors[:, 0], 'bo')
        xmin, xmax = pp.xlim()
        xs = np.linspace(xmin, xmax)
        pp.plot(xs, self.gold_walltime(xs), 'r')
        pp.xlim(xmin, xmax)
        pp.xlabel('Error')
        pp.ylabel('Walltime')



class LPT(ToySim):

    def __init__(self, directory):
        super(LPT, self).__init__(directory)
        self.models = None

    def get_models(self, db_fn='db.sqlite'):
        connection = sql.connect(os.path.join(self.directory, db_fn))
        cursor = connection.cursor()
        cursor.execute('select path from models order by time asc')
        model_fns = cursor.fetchall()

        models = list()
        for fn in model_fns:
            # Unpack tuple
            fn = fn[0]
            # Get relative path
            fn = fn[fn.find(self.directory):]
            # Load model
            model = markovstatemodel.MarkovStateModel.load(fn)

            for i in xrange(len(model.traj_filenames)):
                tfn = model.traj_filenames[i]
                model.traj_filenames[i] = tfn[tfn.find(self.directory):]

            models.append(model)

        self.models = models


    def load_trajs(self, load_stride, load_up_to_this_percent=1.0, verbose=True):
        self.load_stride = load_stride
        assert load_up_to_this_percent <= 1.0, 'Must load less than 100%'
        assert self.models is not None, 'Please load models first'

        model_i = int(load_up_to_this_percent * (len(self.models) - 1))
        print "Using trajectories after round {}".format(model_i + 1)
        model = self.models[model_i]

        traj_list = get_data.get_trajs_from_fn_list(model.traj_filenames)
        traj_list = [traj[::load_stride] for traj in traj_list]


        # Stats
        traj_len = traj_list[0].shape[0]
        if verbose:
            n_trajs = len(traj_list)
            print "{} trajs x {:,} length = {:,} frames".format(n_trajs, traj_len, n_trajs * traj_len)

        self.traj_list = traj_list
        self.wall_steps = traj_len * (model_i + 1)

    # Don't pickle unneccesary things
    def __getstate__(self):
        state = super(LPT, self).__getstate__()
        try:
            del state['models']
        except KeyError:
            pass
        return state


class Compare(object):

    def __init__(self, good_lag_time):
        self.good_lag_time = good_lag_time

    def calculate_gold(self):
        gold = Gold('quant/gold-run/gold/')
        gold.good_lag_time = self.good_lag_time
        gold.load_trajs(load_stride=100, load_up_to_this_percent=1.0)
        gold.do_clustering()
        gold_it = gold.get_implied_timescales(lag_time=gold.good_lag_time, n_timescales=1)[0]
        gold.gold_its = [gold_it]

        gold.get_error_vs_time(load_stride=1000, n_points=5, start_percent=10)
        gold.fit_error()

        self.gold = gold

    def calculate_lpt(self):
        lpt_dirs = [
                    'lpt-250-120',
                    'lpt-500-60',
                    'lpt-1000-30',
                    'lpt-2000-15',
                    'lpt-6000-5',
                    'lpt-30000-1'
                    ]
        lpt_list = list()
        for i in xrange(len(lpt_dirs)):
            lpt = LPT(os.path.join('quant/lpt-run/', lpt_dirs[i]))
            lpt.good_lag_time = self.good_lag_time
            lpt.gold_its = self.gold.gold_its
            lpt.get_models()
            lpt.get_error_vs_time(start_percent=0.0, n_points=5, load_stride=1)
            lpt_list.append(lpt)

        self.lpt_list = lpt_list


def main():
    c = Compare(20)
    c.calculate_gold()
    c.calculate_lpt()

    with open('quant_results.pickl', 'w') as f:
        pickle.dump(c, f)


if __name__ == "__main__":
    main()








