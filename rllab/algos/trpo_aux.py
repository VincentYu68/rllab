__author__ = 'yuwenhao'

from rllab.algos.npo import NPO
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable

from rllab.sampler import parallel_sampler
from rllab.sampler.base import BaseSampler
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.policies.base import Policy

import numpy as np

class SimpleReplayPoolAux(object):
    def __init__(
            self, max_pool_size, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._max_pool_size = max_pool_size
        self.input_data = np.zeros(
            (max_pool_size, input_dim),
        )
        self.output_data = np.zeros(
            (max_pool_size, output_dim),
        )
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, input, output):
        self.input_data[self._top] = input
        self.output_data[self._top] = output
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        if self._size < batch_size:
            batch_size = self._size
        indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(self._bottom, self._bottom + self._size) % self._max_pool_size
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            indices[count] = index
            count += 1
        return dict(
            inputs=self.input_data[indices],
            outputs=self.output_data[indices],
        )

    @property
    def size(self):
        return self._size

class TRPOAux(NPO):
    """
    Trust Region Policy Optimization with Auxiliary tasks
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            aux_pred_step = 3,
            pool_batch_size=10000,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        self.pool_batch_size = pool_batch_size
        self.aux_pred_step = aux_pred_step
        super(TRPOAux, self).__init__(optimizer=optimizer, **kwargs)
        self.reward_pool = SimpleReplayPoolAux(10000, self.env.observation_space.shape[0] * self.aux_pred_step, 1)
        self.termination_pool = SimpleReplayPoolAux(10000, self.env.observation_space.shape[0] * self.aux_pred_step, 1)
        self.com_pool = SimpleReplayPoolAux(10000, self.env.observation_space.shape[0] * self.aux_pred_step, 3)


    def storeAuxData(self, paths):
        for path in paths:
            for step in range(len(path['observations'])-self.aux_pred_step):
                obs_hist = path['observations'][step:step + self.aux_pred_step]
                obs_hist = np.reshape(obs_hist, (obs_hist.shape[0] * obs_hist.shape[1],))
                reward = path['rewards'][step + self.aux_pred_step]
                termination = path['env_infos']['done_return'][step + self.aux_pred_step]
                com_pred = path['env_infos']['com_foot_offset'][step + self.aux_pred_step]
                if termination:
                    self.termination_pool.add_sample(obs_hist, int(0))
                else:
                    self.reward_pool.add_sample(obs_hist, int(np.sign(reward)))
                self.com_pool.add_sample(obs_hist, com_pred)

    def optimize_aux_tasks(self):
        reward_data = self.reward_pool.random_batch(int(self.pool_batch_size/2))
        termination_data = self.termination_pool.random_batch(int(self.pool_batch_size/2))
        reward_termination_data = dict(
            inputs=np.concatenate([reward_data['inputs'], termination_data['inputs']]),
            outputs=np.concatenate([reward_data['outputs'], termination_data['outputs']]),
        )
        reward_termination_target = reward_termination_data['outputs']
        reward_termination_target = np.reshape(reward_termination_target, (reward_termination_target.shape[0] * reward_termination_target.shape[1],))

        com_data = self.com_pool.random_batch(self.pool_batch_size)

        self.policy.train_aux_reward(reward_termination_data['inputs'], reward_termination_target, 1, 32)
        self.policy.train_aux_com(com_data['inputs'], com_data['outputs'], 1, 32)

    def train(self, continue_learning=False):
        self.start_worker()
        if not continue_learning:
            self.init_opt()
        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths = self.sampler.obtain_samples(itr)
                samples_data = self.sampler.process_samples(itr, paths)
                self.log_diagnostics(paths)
                self.storeAuxData(paths)
                self.optimize_aux_tasks()
                self.optimize_policy(itr, samples_data)
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                self.current_itr = itr + 1
                params["algo"] = self
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")

        self.shutdown_worker()