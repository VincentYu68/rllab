__author__ = 'yuwenhao'

from rllab.algos.npo import NPO
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable

from rllab.sampler import parallel_sampler
from rllab.sampler.base import BaseSampler
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.policies.base import Policy
from rllab.misc import ext
import theano
import theano.tensor as TT
import lasagne

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
            aux_pred_dim = 4,
            pool_batch_size=50000,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        self.pool_batch_size = pool_batch_size
        self.aux_pred_step = aux_pred_step
        super(TRPOAux, self).__init__(optimizer=optimizer, **kwargs)

        self.aux_pred_dim = aux_pred_dim
        self.aux_pred_pool = SimpleReplayPoolAux(50000, self.env.observation_space.shape[0] * self.aux_pred_step, aux_pred_dim)


    def storeAuxData(self, paths):
        for path in paths:
            for step in range(len(path['observations'])-self.aux_pred_step):
                obs_hist = path['observations'][step:step + self.aux_pred_step]
                obs_hist = np.reshape(obs_hist, (obs_hist.shape[0] * obs_hist.shape[1],))
                #reward = path['rewards'][step + self.aux_pred_step]
                #termination = path['env_infos']['done_return'][step + self.aux_pred_step]
                if 'aux_pred' in path['env_infos']:
                    aux_pred = path['env_infos']['aux_pred'][step + self.aux_pred_step]
                    self.aux_pred_pool.add_sample(obs_hist, aux_pred)


    def optimize_aux_tasks(self, epoch):
        aux_pred_data = self.aux_pred_pool.random_batch(int(self.pool_batch_size))

        self.policy.train_aux(aux_pred_data['inputs'], aux_pred_data['outputs'], epoch, 32)

    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        # aux net
        aux_input_var = self.policy._aux_pred_network.input_layer.input_var
        aux_target_var = TT.matrix('aux_targets')
        prediction = self.policy._aux_pred_network._output
        surr_loss += 0.01 * TT.mean(TT.square(aux_target_var - prediction))
        '''loss = lasagne.objectives.squared_error(prediction, aux_target_var)
        loss = loss.mean()
        grads = theano.grad(surr_loss, wrt=self.policy.get_params(trainable=True), disconnected_inputs='warn')
        abcd'''
        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ]  + state_info_vars_list + old_dist_info_vars_list+ [aux_input_var, aux_target_var]
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))

        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)

        aux_pred_data = self.aux_pred_pool.random_batch(int(self.pool_batch_size))

        all_input_values += tuple([np.array(aux_pred_data['inputs'])]) + tuple([aux_pred_data['outputs']])

        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer.loss(all_input_values)

        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)

        pred_loss = self.policy.aux_loss(aux_pred_data['inputs'], aux_pred_data['outputs'])

        if itr == 0:
            self.optimize_aux_tasks(epoch=100)
        '''loss_after = self.optimizer.loss(all_input_values)
        param_before = np.copy(self.policy.get_param_values(trainable=True))
        aux_net_param_before = np.copy(self.policy._aux_pred_network.get_param_values(trainable=True))
        if itr == 0:
            auxstep_size = 0
            self.optimize_aux_tasks(epoch=100)
        else:
            self.optimize_aux_tasks(1)
            policy_direction = self.policy.get_param_values(trainable=True) - param_before
            aux_net_direction = self.policy._aux_pred_network.get_param_values(trainable=True) - aux_net_param_before
            auxstep_size = 1
            for line_step in range(20):
                self.policy.set_param_values(param_before + auxstep_size * policy_direction, trainable=True)
                temp_kl = self.optimizer.constraint_val(all_input_values)
                temp_loss = self.optimizer.loss(all_input_values)
                if temp_loss < loss_after+abs(loss_after)*0.001 and temp_kl < self.step_size:
                    break
                auxstep_size *= 0.6
            self.policy._aux_pred_network.set_param_values(aux_net_param_before + auxstep_size * aux_net_direction,trainable=True)'''

        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('Prediction Loss', pred_loss)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

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