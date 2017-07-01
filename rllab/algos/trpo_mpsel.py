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



class TRPOMPSel(NPO):
    """
    Trust Region Policy Optimization with Auxiliary tasks
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            mp_dim = 2,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)

        self.mp_dim = mp_dim

        # genearte data for weight entropy related objective terms
        self.base = np.zeros(4)
        ent_input = []
        for i in range(1000):
            ent_input.append(np.concatenate([self.base, np.random.random(self.mp_dim)]).tolist())
        self.ent_input = [np.array(ent_input)]


        super(TRPOMPSel, self).__init__(optimizer=optimizer, **kwargs)



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

        entropy_input_var = TT.matrix('entropy_inputs')

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

        # entropy of blended weights
        surr_loss += 1.0 * self.policy.bw_entropy(entropy_input_var) - 1.0 * self.policy.bw_choice_entropy(entropy_input_var)
        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ]  + state_info_vars_list + old_dist_info_vars_list + [entropy_input_var]
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
        # update the weight entropy input list
        ent_input = []
        for i in range(1000):
            ent_input.append(np.concatenate([self.base, np.random.random(self.mp_dim)]).tolist())
        self.ent_input = [np.array(ent_input)]

        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))

        ooo = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )

        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)

        all_input_values += tuple(self.ent_input)

        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer.loss(all_input_values)

        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)

        ent_input = tuple(self.ent_input)
        blend_weight_entropy = self.policy._f_weightentropy(ent_input[0])[0]
        blend_choice_entropy = self.policy._f_choiceentropy(ent_input[0])[0]

        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('Blend Weight Entropy', blend_weight_entropy)
        logger.record_tabular('Blend Choice Entropy', blend_choice_entropy)
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