'''
Created on Sep 5, 2013

@author: harrigan
'''
from fuzzy import classic, analysis, get_data
from matplotlib import pyplot as pp
from msmaccelerator.core import markovstatemodel
from msmbuilder import clustering, MSMLib as msml, msm_analysis as msma
from scipy import optimize
import numpy as np
import os
import pickle
import re
import sqlite3 as sql
import sys


# Constants (filenames)
GOLD_TMAT = 'results-gold.pickl'
RESULTS = 'results.pickl'
LPT_FORMAT = 'results-lpt-%s.pickl'
LL_FORMAT = 'results-ll-%s.pickl'

def get_implied_timescales(t_matrix, lag_time, n_timescales):
    """Get implied timescales at a particular lag time."""
    implied_timescales = analysis.get_implied_timescales(t_matrix, n_timescales, lag_time)
    print "Calculated implied timescale at lag time {}".format(lag_time)
    return implied_timescales

class ToySim(object):
    """Contains data for one toy simulation run."""

    def __init__(self, directory, clusterer):
        """Initialize with direcory and clusterer."""
        self.directory = directory
        self.clusterer = clusterer

        # Other Attributes
        self.traj_list = list()
        self.good_lag_time = None
        self.load_stride = 1
        
        self.t_matrices = None
        self.errors = None



    def build_msm(self, lag_time=None):
        """Build an MSM from the loaded trajectories."""
        if lag_time is None:
            lag_time = self.good_lag_time
        else:
            self.good_lag_time = lag_time

        # Do assignment
        trajs = get_data.get_shimtraj_from_trajlist(self.traj_list)
        metric = classic.Euclidean2d()

        # Allocate array
        n_trajs = len(self.traj_list)
        max_traj_len = max([t.shape[0] for t in self.traj_list])
        assignments = -1 * np.ones((n_trajs, max_traj_len), dtype='int')

        # Prepare generators
        pgens = metric.prepare_trajectory(self.clusterer.get_generators_as_traj())

        for i, traj in enumerate(trajs):
            ptraj = metric.prepare_trajectory(traj)

            for j in xrange(len(traj)):
                d = metric.one_to_all(ptraj, pgens, j)
                assignments[i, j] = np.argmin(d)

        counts = msml.get_count_matrix_from_assignments(assignments, n_states=None, lag_time=lag_time)
        rev_counts, t_matrix, populations, mapping = msml.build_msm(counts, ergodic_trimming=False)
        return t_matrix
    
    def calculate_errors(self, n_eigen, gold_tmatrix):        
        gvals, gvecs = msma.get_eigenvectors(gold_tmatrix, n_eigs=n_eigen)
        errors = np.zeros((len(self.t_matrices), 2))
        
        for i, (wall_steps, t_matrix) in enumerate(self.t_matrices):
            vals, vecs = msma.get_eigenvectors(t_matrix, n_eigs=n_eigen)
            if gvecs.shape[0] != vecs.shape[0]:
                print "Error: Vectors are not the same shape!"
            
            error = 0.0
            for j in xrange(n_eigen):
                diff = vecs[:, j] - gvecs[:, j]
                error += np.dot(diff, diff)
                
            errors[i, 0] = wall_steps
            errors[i, 1] = error
        
        self.errors = errors
            
            

    def __getstate__(self):
        """Delete irrelevant variables from pickling."""
        state = dict(self.__dict__)
        try:
            del state['traj_list']
        except KeyError:
            pass
        return state


class Gold(ToySim):
    """Simulation that you can load partial data. Also store the clusterer."""

    def load_trajs(self, load_stride, load_up_to_this_percent=1.0, verbose=True):
        """Load trajectories by percentage.

        Returns number of wall steps.
        """
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

        wall_steps = load_end
        self.traj_list = traj_list
        return wall_steps

    def calculate_tmatrices(self, n_points=15, lag_time=None):
        """Calculate transition matrices at n_points percentages of data.

        Returns transition matrices for saving so that we don't blow up
        memory.
        """
        if lag_time is None:
            lag_time = self.good_lag_time
        else:
            self.good_lag_time = lag_time

        percents = np.linspace(0.01, 1.0, n_points)

        t_matrices = list()
        for percent in percents:
            try:
                wall_steps = self.load_trajs(load_stride=1, load_up_to_this_percent=percent)
                t_matrix = self.build_msm(lag_time)
                t_matrices.append((wall_steps, t_matrix))

                print "Built MSM from gold using {:.2f}% of data".format(100.*percent)
            except:
                print "Couldn't build msm {} using {:.2f}% of data".format(self.get_name(), 100.*percent)

        self.t_matrices = t_matrices

    def get_name(self):
        """Display name."""
        return "Gold"



class LPT(ToySim):
    """LPT or ll simulations."""

    def __init__(self, directory, clusterer):
        """Initialize and generate filenames."""
        super(LPT, self).__init__(directory, clusterer)

        self.rounds = None
        self._get_trajs_fns()

    def _get_trajs_fns(self, db_fn='db.sqlite'):
        """Save trajectory filenames per round of simulation.

        This function loads the msmaccelerator database file
        and uses the models table to enumerate rounds. From each
        model, the trajectories generated up to that point
        are saved.
        """
        connection = sql.connect(os.path.join(self.directory, db_fn))
        cursor = connection.cursor()
        cursor.execute('select path from models order by time asc')
        model_fns = cursor.fetchall()

        rounds = list()
        for fn in model_fns:
            # Unpack tuple
            fn = fn[0]
            # Get relative path
            fn = fn[fn.find(self.directory):]
            # Load model
            model = markovstatemodel.MarkovStateModel.load(fn)
            # Get relative paths
            traj_filenames = [tfn[tfn.find(self.directory):] for tfn in model.traj_filenames]
            # Save trajectory filenames
            rounds.append(traj_filenames)
            # Cleanup
            model.close()

        self.rounds = rounds


    def load_trajs(self, load_stride, round_i, verbose=True):
        """Load trajectories up to round_i.

        Returns number of wall steps.
        """
        self.load_stride = load_stride

        assert self.rounds is not None, 'Please load rounds first'
        assert round_i < len(self.rounds), 'Round index out of range %d' % round_i

        print "Using trajectories after round {}".format(round_i + 1)

        traj_list = get_data.get_trajs_from_fn_list(self.rounds[round_i])
        traj_list = [traj[::load_stride] for traj in traj_list]

        # Stats
        traj_len = traj_list[0].shape[0]
        n_trajs = len(traj_list)
        if verbose:
            print "{} trajs x {:,} length = {:,} frames".format(n_trajs, traj_len, n_trajs * traj_len)


        self.traj_list = traj_list
        wall_steps = traj_len * (round_i + 1)
        return wall_steps

    def calculate_tmatrices(self, lag_time=None):
        """Build transition matrices for each round.

        Returns a list for pickling to save memory.
        """
        if lag_time is None:
            lag_time = self.good_lag_time
        else:
            self.good_lag_time = lag_time

        t_matrices = list()
        for round_i in xrange(len(self.rounds)):
            try:
                wall_steps = self.load_trajs(load_stride=1, round_i=round_i)
                t_matrix = self.build_msm(lag_time)
                t_matrices.append((wall_steps, t_matrix))

                print "Built MSM from round {}".format(round_i + 1)
            except:
                print "Couldn't build msm {} after round {}".format(self.get_name(), round_i + 1)

        self.t_matrices = t_matrices

    def get_name(self):
        """Get a display name. We take this from the directory name."""
        fn = self.directory
        fn = fn[fn.rfind('/') + 1:]
        return fn



class Compare(object):
    """Class for managing all the simulations."""

    def __init__(self):
        """Initialize."""
        self.gold = None
        self.n_clusters = None
        self.implied_timescales = None
        
        self.ll_list = None
        self.lpt_list = None

    def do_clustering(self, distance_cutoff, n_medoid_iters):
        """Perform clustering on the gold run and save the clusterer."""
        gold = Gold('quant/gold-run/gold', clusterer=None)
        gold.load_trajs(1, 1.0)
        trajs = get_data.get_shimtraj_from_trajlist(gold.traj_list)
        metric = classic.Euclidean2d()

        clustering.logger.setLevel('WARNING')
        hkm = clustering.HybridKMedoids(metric, trajs, k=None, distance_cutoff=distance_cutoff, local_num_iters=n_medoid_iters)
        self.n_clusters = hkm.get_generators_as_traj()['XYZList'].shape[0]

        print "Clustered data into {:,} clusters".format(self.n_clusters)
        gold.clusterer = hkm
        self.gold = gold


    def calculate_implied_timescales(self, lag_times, n_timescales):
        """Calculate implied timescales of the gold run at range specified
        by lag_times.
        """
        implied_timescales = list()
        for lag_time in lag_times:
            t_matrix = self.gold.build_msm(lag_time)
            it = analysis.get_implied_timescales(t_matrix, n_timescales, lag_time)
            implied_timescales.append((lag_time, it))
            print "Calculated lag time at time {}".format(lag_time)

        self.implied_timescales = implied_timescales

    def calculate_all_tmatrices(self, lag_time):
        """Calculate all the transition matrices."""

        # Gold
        self.gold.calculate_tmatrices(lag_time)

        # Do length per traj
        lpt_list = list()
        lpt_dirs = self.get_lpt_dirs()
        for lpt_dir in lpt_dirs:
            # Load and calculate
            lpt = LPT(lpt_dir, self.gold.clusterer)
            lpt.calculate_tmatrices(lag_time)
            lpt_list.append(lpt)

        # Do parallel
        ll_dirs = self.get_ll_dirs()
        ll_list = list()
        for ll_dir in ll_dirs:
            # Load and calculate
            ll = LPT(ll_dir, self.gold.clusterer)
            ll.calculate_tmatrices(lag_time)
            ll_list.append(ll)
            
        self.lpt_list = lpt_list
        self.ll_list = ll_list
            
    def calculate_all_errors(self, n_eigens):
        """Calculate errors in each toy simulation."""
        
        _, gold_tmatrix = self.gold.t_matrices[-1]
        self.gold.calculate_errors(n_eigens, gold_tmatrix)
        
        for toysim in self.ll_list:
            toysim.calculate_errors(n_eigens, gold_tmatrix)
            
        for toysim in self.lpt_list:
            toysim.calculate_errors(n_eigens, gold_tmatrix)


    def get_lpt_dirs(self):
        lpt_dirs = [
                    'lpt-250-120',
                    'lpt-500-60',
                    'lpt-1000-30',
                    'lpt-2000-15',
                    'lpt-6000-5',
                    'lpt-30000-1'
                    ]

        lpt_dirs = [os.path.join('quant/lpt-run/', lpt) for lpt in lpt_dirs]
        return lpt_dirs

    def get_ll_dirs(self):
        ll_dirs = os.listdir('quant/parallelism-run')
        ll_dirs = [ll  for ll in ll_dirs if ll.startswith('ll-')]
        def sort_by_end(s):
            return int(s[s.rfind('-') + 1:])
        ll_dirs = sorted(ll_dirs, key=sort_by_end)
        ll_dirs = [os.path.join('quant/parallelism-run/', ll) for ll in ll_dirs]

        return ll_dirs


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


# def get_speedup2(toy, popt_gold, p0=None, error_val=0.4):
#     # Optimize
#     popt_toy, _ = optimize.curve_fit(_error_vs_time, toy.errors[:, 0], toy.errors[:, 1], p0=p0)
#     speedup = _time_vs_error(error_val, *popt_gold) / _time_vs_error(error_val, *popt_toy)
#     return speedup
# 
# def plot_and_fit(toy, p0=None):
#     ax = pp.gca()
#     popt_toy, _ = optimize.curve_fit(_error_vs_time, toy.errors[:, 0], toy.errors[:, 1], p0=p0)
#     xs = np.linspace(1, toy.errors[-1, 0])
#     ys = _error_vs_time(xs, *popt_toy)
#     ax.plot(toy.errors[:, 0], toy.errors[:, 1], 'o', label=toy.get_name())
#     ax.plot(xs, ys, color=ax.lines[-1].get_color())
#     print popt_toy
#     return popt_toy

# def plot_speedup_bar(toys, popt_gold, xlabel, directory_to_x_func=None, width=0.4, p0=None, error_val=0.4):
#     speedups = list()
#     xvals = list()
#     xlabels = list()
#     for toy in toys:
#         speedups.append(get_speedup2(toy, popt_gold, p0, error_val))
# 
#         if directory_to_x_func is not None:
#             xvals.append(directory_to_x_func(toy.directory))
#             xlabels.append(toy.directory)
# 
#     if directory_to_x_func is None:
#         xvals = np.linspace(0, 10.0, len(toys))
# 
#     pp.bar(xvals, speedups, bottom=0, width=width, log=True)
#     pp.xticks(np.array(xvals) + width / 2, np.exp(xvals))
#     xmin, xmax = pp.xlim()
#     pp.hlines(1.0, xmin, xmax)  # Break even
#     pp.xlim(xmin, xmax)
#     pp.ylabel("Speedup")
#     pp.xlabel(xlabel)
#     print xlabels
#     print speedups


def main(options):
    if os.path.exists('results.pickl'):
        with open('results.pickl', 'rb') as f:
            c = pickle.load(f)
    else:
        c = Compare()

    print "You specified options {}".format(options)

    if '1' in options:
        c.do_clustering(distance_cutoff=0.25, n_medoid_iters=10)
    if '2' in options:
        c.calculate_implied_timescales(lag_times=xrange(1, 100, 5), n_timescales=3)
    if '3' in options:
        c.calculate_all_tmatrices(lag_time=20)
    if '4' in options:
        c.calculate_all_errors(n_eigens=2)

    with open('results.pickl', 'wb') as f:
        pickle.dump(c, f, protocol=2)

def parse():
    print """Quantitative analysis for adaptive sampling on the muller potential.

        Include the following options:
         1 - Do clustering
         2 - Calculate ITs
         3 - Calculate transition matrices (x3)
         4 - Apply func to transition matricies to get error
    """
    if len(sys.argv) > 1:
        main(sys.argv[1])


if __name__ == "__main__":
    parse()








