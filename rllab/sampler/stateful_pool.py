

from joblib.pool import MemmapingPool
import multiprocessing as mp
from rllab.misc import logger
import pyprind
import time
import traceback
import sys
from gym.envs.dart.dynamic_models import *
from sklearn.neighbors import KNeighborsRegressor
import joblib

class ProgBarCounter(object):
    def __init__(self, total_count):
        self.total_count = total_count
        self.max_progress = 1000000
        self.cur_progress = 0
        self.cur_count = 0
        if not logger.get_log_tabular_only():
            self.pbar = pyprind.ProgBar(self.max_progress)
        else:
            self.pbar = None

    def inc(self, increment):
        if not logger.get_log_tabular_only():
            self.cur_count += increment
            new_progress = self.cur_count * self.max_progress / self.total_count
            if new_progress < self.max_progress:
                self.pbar.update(new_progress - self.cur_progress)
            self.cur_progress = new_progress

    def stop(self):
        if self.pbar is not None and self.pbar.active:
            self.pbar.stop()


class SharedGlobal(object):
    def __init__(self):
        # put things about model parameter re-sampling here for now (perhaps forever...)
        self.mp_resamp = {}
        self.mp_resamp['use_model_resample'] = False
        self.mp_resamp['use_adjusted_resample'] = False
        if self.mp_resamp['use_model_resample']:
            logger.log('Use model resample!')
        else:
            logger.log('Not using model resample!')
        self.mp_resamp['mr_buffer'] = []
        self.mp_resamp['mr_buffer_size'] = 300
        self.mp_resamp['mr_iteration_num'] = 25
        self.mp_resamp['mr_current_iteration'] = 0
        self.mp_resamp['mr_store_percentage'] = 0.05
        self.mp_resamp['mr_activated'] = False
        self.mp_resamp['mr_probability'] = 0.02
        self.mp_resamp['iter'] = 0
        self.mp_resamp['mr_minimum'] = []

        self.ensemble_dynamics = {}
        self.ensemble_dynamics['use_ens_dyn'] = False
        #self.ensemble_dynamics['dyn_models'] = [joblib.load('data/trained/dyn_models.pkl')[0]]
        self.ensemble_dynamics['dyn_models'] = [LinearDynamicModel()]
        self.ensemble_dynamics['transition_locator'] = KNeighborsRegressor(n_neighbors=1, weights='distance')
        self.ensemble_dynamics['dyn_model_choice'] = 0
        self.ensemble_dynamics['training_buffer_x'] = []
        self.ensemble_dynamics['training_buffer_y'] = []
        self.ensemble_dynamics['base_paths'] = []
        self.ensemble_dynamics['baseline'] = None

class StatefulPool(object):
    def __init__(self):
        self.n_parallel = 1
        self.pool = None
        self.queue = None
        self.worker_queue = None
        self.G = SharedGlobal()

    def initialize(self, n_parallel):
        self.n_parallel = n_parallel
        if self.pool is not None:
            print("Warning: terminating existing pool")
            self.pool.terminate()
            self.queue.close()
            self.worker_queue.close()
            self.G = SharedGlobal()
        if n_parallel > 1:
            self.queue = mp.Queue()
            self.worker_queue = mp.Queue()
            self.pool = MemmapingPool(
                self.n_parallel,
                temp_folder="/tmp",
            )

    def run_each(self, runner, args_list=None):
        """
        Run the method on each worker process, and collect the result of execution.
        The runner method will receive 'G' as its first argument, followed by the arguments
        in the args_list, if any
        :return:
        """
        if args_list is None:
            args_list = [tuple()] * self.n_parallel
        assert len(args_list) == self.n_parallel
        if self.n_parallel > 1:
            results = self.pool.map_async(
                _worker_run_each, [(runner, args) for args in args_list]
            )
            for i in range(self.n_parallel):
                self.worker_queue.get()
            for i in range(self.n_parallel):
                self.queue.put(None)
            return results.get()
        return [runner(self.G, *args_list[0])]

    def run_map(self, runner, args_list):
        if self.n_parallel > 1:
            return self.pool.map(_worker_run_map, [(runner, args) for args in args_list])
        else:
            ret = []
            for args in args_list:
                ret.append(runner(self.G, *args))
            return ret

    def run_imap_unordered(self, runner, args_list):
        if self.n_parallel > 1:
            for x in self.pool.imap_unordered(_worker_run_map, [(runner, args) for args in args_list]):
                yield x
        else:
            for args in args_list:
                yield runner(self.G, *args)

    def run_collect(self, collect_once, threshold, args=None, show_prog_bar=True):
        """
        Run the collector method using the worker pool. The collect_once method will receive 'G' as
        its first argument, followed by the provided args, if any. The method should return a pair of values.
        The first should be the object to be collected, and the second is the increment to be added.
        This will continue until the total increment reaches or exceeds the given threshold.

        Sample script:

        def collect_once(G):
            return 'a', 1

        stateful_pool.run_collect(collect_once, threshold=3) # => ['a', 'a', 'a']

        :param collector:
        :param threshold:
        :return:
        """
        if args is None:
            args = tuple()
        if self.pool:
            manager = mp.Manager()
            counter = manager.Value('i', 0)
            lock = manager.RLock()
            results = self.pool.map_async(
                _worker_run_collect,
                [(collect_once, counter, lock, threshold, args)] * self.n_parallel
            )
            if show_prog_bar:
                pbar = ProgBarCounter(threshold)
            last_value = 0
            while True:
                time.sleep(0.1)
                with lock:
                    if counter.value >= threshold:
                        if show_prog_bar:
                            pbar.stop()
                        break
                    if show_prog_bar:
                        pbar.inc(counter.value - last_value)
                    last_value = counter.value
            return sum(results.get(), [])
        else:
            count = 0
            results = []
            if show_prog_bar:
                pbar = ProgBarCounter(threshold)
            while count < threshold:
                result, inc = collect_once(self.G, *args)
                for path in result:
                    results.append(path)
                count += inc
                if show_prog_bar:
                    pbar.inc(inc)
            if show_prog_bar:
                pbar.stop()
            return results


singleton_pool = StatefulPool()


def _worker_run_each(all_args):
    try:
        runner, args = all_args
        # signals to the master that this task is up and running
        singleton_pool.worker_queue.put(None)
        # wait for the master to signal continuation
        singleton_pool.queue.get()
        return runner(singleton_pool.G, *args)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def _worker_run_collect(all_args):
    try:
        collect_once, counter, lock, threshold, args = all_args
        collected = []
        while True:
            with lock:
                if counter.value >= threshold:
                    return collected
            result, inc = collect_once(singleton_pool.G, *args)
            collected += result
            with lock:
                counter.value += inc
                if counter.value >= threshold:
                    return collected
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def _worker_run_map(all_args):
    try:
        runner, args = all_args
        return runner(singleton_pool.G, *args)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
