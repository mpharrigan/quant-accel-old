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
import re

ERROR_CUTOFF = 2.0

def _error_vs_time(time, a, tau):
    return a * np.exp(-time / (tau * 1.e2))

def _time_vs_error(err, a, tau):
    return -tau * 1.e2 * np.log(err / a)

class ToySim(object):
    def __init__(self, directory, n_timescales, distance_cutoff=0.25, n_medoid_iters=1):
        self.directory = directory
        self.distance_cutoff = distance_cutoff
        self.n_medoid_iters = n_medoid_iters
        self.n_timescales = n_timescales


    def do_clustering(self, traj_list):
        trajs = get_data.get_shimtraj_from_trajlist(traj_list)
        metric = classic.Euclidean2d()

        clustering.logger.setLevel('WARNING')
        hkm = clustering.HybridKMedoids(metric, trajs, k=None, distance_cutoff=self.distance_cutoff, local_num_iters=self.n_medoid_iters)
        self.n_clusters = hkm.get_generators_as_traj()['XYZList'].shape[0]

        print "Clustered data into {:,} clusters".format(self.n_clusters)
        return hkm

    def build_msm(self, clusterer, lag_time):
        """Build an MSM from the loaded trajectories."""
        counts = msml.get_count_matrix_from_assignments(clusterer.get_assignments(), self.n_clusters, lag_time)
        rev_counts, t_matrix, populations, mapping = msml.build_msm(counts)
        return t_matrix

    def get_implied_timescales(self, clusterer, lag_time, n_timescales):
        """Get implied timescales at a particular lag time."""
        t_matrix = self.build_msm(clusterer, lag_time)
        implied_timescales = analysis.get_implied_timescales(t_matrix, n_timescales, lag_time * self.load_stride)
        print "Calculated implied timescale at lag time {}".format(lag_time)
        return implied_timescales

    def get_error_vs_time(self, start_percent=5, n_points=10, load_stride=1):
        percents = np.linspace(start_percent / 100., 1.0, n_points)

        errors = np.zeros((len(percents), 2))
        for i in xrange(len(percents)):
            percent = percents[i]
            traj_list = self.load_trajs(load_stride=load_stride, load_up_to_this_percent=percent)
            clusterer = self.do_clustering(traj_list)
            its = self.get_implied_timescales(clusterer, lag_time=self.good_lag_time, n_timescales=self.n_timescales)

            # Calculate error
            evec = np.log(its) - np.log(self.gold_its)
            errors[i][1] = np.sqrt(np.dot(evec, evec))
            errors[i][0] = self.wall_steps

            print "Calculated error from using {}% of all data".format(percent * 100.)

        self.errors = errors

    def get_implied_timescales_vs_lt(self, clusterer, range_tuple, n_timescales=4):
        """Get implied timescales vs lag time."""
        lt_range = range(*range_tuple)
        implied_timescales = np.zeros((len(lt_range), n_timescales))

        for i in xrange(len(lt_range)):
            implied_timescales[i] = self.get_implied_timescales(clusterer, lag_time=lt_range[i], n_timescales=n_timescales)
            print "Calculated implied timescale at lag time {}".format(lt_range[i])

        self.implied_timescales = implied_timescales
        self.lt_range = lt_range


    def plot_implied_timescales_vs_lt(self):
        for i in xrange(self.implied_timescales.shape[1]):
            pp.scatter(self.lt_range, self.implied_timescales[:, i])
        pp.yscale('log')

    def plot_errors(self):
        pp.plot(self.errors[:, 0], self.errors[:, 1], 'bo')
        pp.xlabel('Walltime')
        pp.ylabel('Error')


    def __getstate__(self):
        state = dict(self.__dict__)
        try:
            del state['gold_walltime']
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


        self.wall_steps = load_end
        return traj_list

    def get_name(self):
        return "Gold"



class LPT(ToySim):

    def __init__(self, directory, n_timescales):
        super(LPT, self).__init__(directory, n_timescales)
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

    def cleanup(self):
        for model in self.models:
            model.close()


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
        self.traj_len = traj_list[0].shape[0]
        self.n_trajs = len(traj_list)
        if verbose:
            print "{} trajs x {:,} length = {:,} frames".format(self.n_trajs, self.traj_len, self.n_trajs * self.traj_len)


        self.wall_steps = self.traj_len * (model_i + 1)
        return traj_list

    def get_name(self):
        fn = self.directory
        fn = fn[fn.rfind('/') + 1:]
        return fn

    # Don't pickle unneccesary things
    def __getstate__(self):
        state = super(LPT, self).__getstate__()
        try:
            del state['models']
        except KeyError:
            pass
        return state


class Compare(object):

    def __init__(self, lag_time, n_timescales):
        self.good_lag_time = lag_time
        self.n_timescales = n_timescales

    def calculate_gold(self):
        gold = Gold('quant/gold-run/gold/', self.n_timescales)
        load_stride = 10
        gold.good_lag_time = self.good_lag_time // load_stride
        traj_list = gold.load_trajs(load_stride=load_stride, load_up_to_this_percent=1.0)
        clusterer = gold.do_clustering(traj_list)
        gold.gold_its = gold.get_implied_timescales(clusterer, lag_time=gold.good_lag_time, n_timescales=self.n_timescales)

        gold.get_error_vs_time(load_stride=load_stride, n_points=25, start_percent=5)

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
            try:
                lpt = LPT(os.path.join('quant/lpt-run/', lpt_dirs[i]), self.n_timescales)
                lpt.good_lag_time = self.good_lag_time
                lpt.gold_its = self.gold.gold_its
                lpt.get_models()
                lpt.get_error_vs_time(start_percent=0.0, n_points=10, load_stride=1)
                lpt.cleanup()
                lpt_list.append(lpt)
            except:
                print "+++ Errored in lpt +++", lpt_dirs[i]
                pass

        self.lpt_list = lpt_list

    def calculate_ll(self):
        ll_dirs = os.listdir('quant/parallelism-run')
        ll_dirs = [ll  for ll in ll_dirs if ll.startswith('ll-')]
        def sort_by_end(s):
            return int(s[s.rfind('-') + 1:])
        ll_dirs = sorted(ll_dirs, key=sort_by_end)

        ll_list = list()
        for i in xrange(len(ll_dirs)):
            try:
                ll = LPT(os.path.join('quant/parallelism-run/', ll_dirs[i]), self.n_timescales)
                ll.good_lag_time = self.good_lag_time
                ll.gold_its = self.gold.gold_its
                ll.get_models()
                ll.get_error_vs_time(start_percent=0.0, n_points=10, load_stride=1)
                ll.cleanup()
                ll_list.append(ll)
            except:
                print "+++ Errored in ll +++", ll_dirs[i]
                pass

        self.ll_list = ll_list

    def calculate_repeat(self):
        stat_dirs = os.listdir('quant/stat-run')
        stat_dirs = [ss for ss in stat_dirs if ss.startswith('stat-')]

        stat_list = list()
        for i in xrange(len(stat_dirs)):
            try:
                ss = LPT(os.path.join('quant/stat-run', stat_dirs[i]), self.n_timescales)
                ss.good_lag_time = self.good_lag_time
                ss.gold_its = self.gold.gold_its
                ss.get_models()
                ss.get_error_vs_time(start_percent=0.0, n_points=10, load_stride=1)
                ss.cleanup()
                stat_list.append(ss)
            except:
                print "+++ Errored in stat+++", stat_dirs[i]
                pass
        self.stat_list = stat_list


def lpt_dir_to_x(fn):
    fn = fn[fn.rfind('/') + 1:]
    reres = re.search('(?<=-)[0-9]+(?=-)', fn)
    x = int(reres.group(0))
    return np.log(x)

def ll_dir_to_x(fn):
    fn = fn[fn.rfind('/') + 1:]
    reres = re.search('(?<=-)[0-9]+', fn)
    x = int(reres.group(0))
    return np.log(x)


def get_speedup2(toy, popt_gold, p0=None, error_val=0.4):
    # Optimize
    popt_toy, _ = optimize.curve_fit(_error_vs_time, toy.errors[:, 0], toy.errors[:, 1], p0=p0)
    speedup = _time_vs_error(error_val, *popt_gold) / _time_vs_error(error_val, *popt_toy)
    return speedup

def plot_and_fit(toy, p0=None):
    ax = pp.gca()
    popt_toy, _ = optimize.curve_fit(_error_vs_time, toy.errors[:, 0], toy.errors[:, 1], p0=p0)
    xs = np.linspace(1, toy.errors[-1, 0])
    ys = _error_vs_time(xs, *popt_toy)
    ax.plot(toy.errors[:, 0], toy.errors[:, 1], 'o', label=toy.get_name())
    ax.plot(xs, ys, color=ax.lines[-1].get_color())
    print popt_toy
    return popt_toy

def plot_speedup_bar(toys, popt_gold, xlabel, directory_to_x_func=None, width=0.4, p0=None, error_val=0.4):
    speedups = list()
    xvals = list()
    xlabels = list()
    colors = list()
    for toy in toys:
        speedups.append(get_speedup2(toy, popt_gold, p0, error_val))

        if directory_to_x_func is not None:
            xvals.append(directory_to_x_func(toy.directory))
            xlabels.append(toy.directory)

    if directory_to_x_func is None:
        xvals = np.linspace(0, 10.0, len(toys))

    pp.bar(xvals, speedups, bottom=0, width=width, log=True)
    pp.xticks(np.array(xvals) + width / 2, np.exp(xvals))
    xmin, xmax = pp.xlim()
    pp.hlines(1.0, xmin, xmax)  # Break even
    pp.xlim(xmin, xmax)
    pp.ylabel("Speedup")
    pp.xlabel(xlabel)
    print xlabels
    print speedups


def main():
    c = Compare(lag_time=60, n_timescales=3)
    output_fn = "quant_results.lt{}.it{}.pickl".format(c.good_lag_time, c.n_timescales)
    c.calculate_gold()
    c.calculate_lpt()
    c.calculate_ll()
    c.calculate_repeat()

    with open(output_fn, 'w') as f:
        pickle.dump(c, f)


if __name__ == "__main__":
    main()








