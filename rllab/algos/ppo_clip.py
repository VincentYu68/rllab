__author__ = 'yuwenhao'

from rllab.algos.npo import NPO
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.core.serializable import Serializable

from rllab.sampler import parallel_sampler
from rllab.sampler.base import BaseSampler
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.policies.base import Policy
from rllab.misc import ext
import theano
import theano as T
import theano.tensor as TT
import lasagne

import collections

import numpy as np
import lasagne.layers as L
import copy

class PPO_Clip_Sym(NPO):
    """
    Trust Region Policy Optimization with Auxiliary tasks
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            observation_permutation = None,
            action_permutation = None,
            sym_loss_weight = 0.0001,
            clip_param = 0.2,
            adam_batchsize = 128,
            adam_epochs = 10,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = LbfgsOptimizer(**optimizer_args)
        super(PPO_Clip_Sym, self).__init__(optimizer=optimizer, **kwargs)

        self.observation_permutation = observation_permutation
        self.action_permutation = action_permutation
        self.sym_loss_weight = sym_loss_weight

        self.clip_param = clip_param

        self.adam_batchsize = adam_batchsize
        self.adam_epochs = adam_epochs

        self.obs_perm_mat = np.zeros((len(observation_permutation), len(observation_permutation)))
        self.act_per_mat = np.zeros((len(action_permutation), len(action_permutation)))
        for i, perm in enumerate(self.observation_permutation):
            self.obs_perm_mat[i][int(np.abs(perm))] = np.sign(perm)
        for i, perm in enumerate(self.action_permutation):
            self.act_per_mat[i][int(np.abs(perm))] = np.sign(perm)

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
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            std_advar = (advantage_var - TT.mean(advantage_var))/TT.std(advantage_var)
            surr_loss = - TT.mean(TT.min([lr * std_advar, TT.clip(lr, 1-self.clip_param, 1+self.clip_param) * std_advar]))

        # symmetry loss
        mirrored_obs_var = self.env.observation_space.new_tensor_variable(
            'mirrored_obs',
            extra_dims=1 + is_recurrent,
        )

        mean_act_collected = L.get_output(self.policy._l_mean, obs_var)
        mean_act_mirrored = L.get_output(self.policy._l_mean, mirrored_obs_var)
        sym_loss = self.sym_loss_weight * TT.mean(TT.square(TT.dot(mean_act_collected, self.act_per_mat.T)-mean_act_mirrored))
        surr_loss += sym_loss

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ]  + state_info_vars_list + old_dist_info_vars_list+ [mirrored_obs_var]
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            inputs=input_list,
        )

        self._f_sym_loss = ext.compile_function(
                inputs=[obs_var, mirrored_obs_var],
                outputs=[sym_loss]
            )

        grad = theano.grad(
            surr_loss, wrt=self.policy.get_params(trainable=True), disconnected_inputs='warn')

        self._f_grad = ext.compile_function(
            inputs=input_list,
            outputs=grad,
        )

        self._f_loss = ext.compile_function(input_list+list(),
            surr_loss
        )

        self.m_prev = []
        self.v_prev = []
        for i in range(len(self.policy.get_params(trainable=True))):
            self.m_prev.append(np.zeros(self.policy.get_params(trainable=True)[i].get_value().shape, dtype=self.policy.get_params(trainable=True)[i].get_value().dtype))
            self.v_prev.append(np.zeros(self.policy.get_params(trainable=True)[i].get_value().shape,
                                        dtype=self.policy.get_params(trainable=True)[i].get_value().dtype))
        self.t_prev = 0

        self.optimizer.update_opt(surr_loss, self.policy, input_list)

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

        mirrored_obs = copy.deepcopy(samples_data["observations"])
        for i in range(len(mirrored_obs)):
            mirrored_obs[i] = np.dot(self.obs_perm_mat, mirrored_obs[i])
        all_input_values += tuple([np.array(mirrored_obs)])

        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        loss_before = self.optimizer.loss(all_input_values)

        for iepoch in range(self.adam_epochs):
            sortinds = np.random.permutation(len(all_input_values[0]))
            for istart in range(0, len(all_input_values[0]), self.adam_batchsize):
                input_data = []
                for j in range(len(all_input_values)):
                    input_data.append(all_input_values[j][sortinds[istart:istart+self.adam_batchsize]])
                self.adam_updates(input_data)

        sym_loss = self._f_sym_loss(samples_data["observations"], mirrored_obs)

        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('Symmetry Loss', sym_loss)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
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

    def adam_updates(self, data, learning_rate=0.0003, beta1=0.9,
                     beta2=0.999, epsilon=1e-8):

        t = self.t_prev + 1
        a_t = learning_rate * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        gradient = self._f_grad(*data)
        for i in range(len(gradient)):
            m_t = beta1 * self.m_prev[i] + (1 - beta1) * gradient[i]
            v_t = beta2 * self.v_prev[i] + (1 - beta2) * gradient[i] ** 2
            step = a_t * m_t / (np.sqrt(v_t) + epsilon)

            self.m_prev[i] = m_t
            self.v_prev[i] = v_t
            self.policy.get_params(trainable=True)[i].set_value(self.policy.get_params(trainable=True)[i].get_value() - step)

        self.t_prev = t
