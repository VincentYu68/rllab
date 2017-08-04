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
import theano as T
import theano.tensor as TT
import lasagne

import numpy as np



class TRPOSplit(NPO):
    """
    Trust Region Policy Optimization with Auxiliary tasks
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            mp_dim = 2,
            split_weight = 0.1,
            split_importance = None,
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
        self.split_weight = split_weight
        self.split_importance = split_importance

        super(TRPOSplit, self).__init__(optimizer=optimizer, **kwargs)



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

        # temporary code
        split_params = []
        sim_loss = None
        total_param_size = len(self.policy._mean_network.get_param_values())
        for splitid in range(2):
            split_params.append(self.policy._mean_network.get_split_parameter(splitid))

        for splitid in range(2):
            for splitid2 in range(splitid+1, 2):
                for pid in range(len(split_params[0])):
                    weight_mat = 1
                    if self.split_importance is not None:
                        print(len(self.split_importance), len(split_params[0]))
                        weight_mat = np.clip(1.0/self.split_importance[pid], 0, 1e2)
                    if sim_loss is None:
                        sim_loss = self.split_weight / total_param_size * TT.sum(weight_mat*((split_params[splitid][pid] - split_params[splitid2][pid])**2))
                    else:
                        sim_loss += self.split_weight / total_param_size * TT.sum(weight_mat*((split_params[splitid][pid] - split_params[splitid2][pid])**2))
        if self.split_weight > 0.1:
            surr_loss += sim_loss
            #self.step_size += 0.01


        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ]  + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        self.f_sim_loss = T.function([], sim_loss)

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

        print('Similarity Loss: ', self.f_sim_loss())

        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer.loss(all_input_values)

        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)


        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)

        return dict()