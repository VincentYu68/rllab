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
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(TT.min([lr * advantage_var, TT.clip(lr, 1-self.clip_param, 1+self.clip_param) * advantage_var]))

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

        self.train = theano.function(input_list, surr_loss,
                                     updates=self.policy.get_params()
                                             + self.adam_updates(surr_loss, self.policy.get_params(), learning_rate=0.001).items())

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

        #self.optimizer.optimize(all_input_values)
        self.train(all_input_values)

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

    def adam_updates(self, loss, params, learning_rate=0.001, beta1=0.9,
                     beta2=0.999, epsilon=1e-8):

        all_grads = T.grad(loss, params)
        t_prev = theano.shared(np.array(0, dtype=np.float32))
        updates = collections.OrderedDict()

        t = t_prev + 1
        a_t = learning_rate * TT.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

        for param, g_t in zip(params, all_grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

            m_t = beta1 * m_prev + (1 - beta1) * g_t
            v_t = beta2 * v_prev + (1 - beta2) * g_t ** 2
            step = a_t * m_t / (TT.sqrt(v_t) + epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

        updates[t_prev] = t
        return updates