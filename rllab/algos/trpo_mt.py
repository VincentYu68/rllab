from rllab.algos.npo import NPO
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
import rllab.misc.logger as logger
from rllab.misc import ext
import theano
import theano.tensor as TT
import numpy as np
from rllab.misc.ext import sliced_fun

class TRPO_MultiTask(NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            task_num = 1,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        self.task_num = task_num
        self.kl_weights = np.ones(self.task_num)
        super(TRPO_MultiTask, self).__init__(optimizer=optimizer, **kwargs)


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

        task_obs_var = []
        task_old_dist_info_vars_list = []
        task_kls = []
        for i in range(self.task_num):
            task_obs_var.append(self.env.observation_space.new_tensor_variable(
                'obs_task%d'%(i),
                extra_dims=1 + is_recurrent,
            ))
            temp_dist_info_var = self.policy.dist_info_sym(task_obs_var[-1], state_info_vars)
            temp_old_dist_info_vars = {
                k: ext.new_tensor(
                    'task%d_old_%s' % (i,k),
                    ndim=2 + is_recurrent,
                    dtype=theano.config.floatX
                ) for k in dist.dist_info_keys
                }
            task_old_dist_info_vars_list += [temp_old_dist_info_vars[k] for k in dist.dist_info_keys]
            task_kls.append(dist.kl_sym(temp_old_dist_info_vars, temp_dist_info_var))

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        kl_weight_var = ext.new_tensor(
                'kl_weight',
                ndim=1,
                dtype=theano.config.floatX
            )

        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            weighted_kls = []
            '''for i, one_task_kl in enumerate(task_kls):
                weighted_kls.append(TT.mean(one_task_kl * kl_weight_var[i]))
            mean_kl = TT.mean(weighted_kls)'''
            for i, one_task_kl in enumerate(task_kls):
                weighted_kls.append((one_task_kl * kl_weight_var[i]))
            mean_kl = TT.mean(TT.concatenate(weighted_kls))
            surr_loss = - TT.mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list + task_obs_var + task_old_dist_info_vars_list + [kl_weight_var]
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

        self.f_constraints=[]
        self.f_constraints.append(ext.compile_function(
                    inputs=input_list,
                    outputs=TT.mean(kl),
                    log_name="kl_div_task",
                ))
        for i in range(self.task_num):
            self.f_constraints.append(ext.compile_function(
                    inputs=input_list,
                    outputs=TT.mean(task_kls[i]),
                    log_name="kl_div_task%d"%i,
                ))

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
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        # multitask related
        task_obs = []
        task_old_dist_info_list = []
        task_old_dist_info = []
        for i in range(self.task_num):
            task_obs.append([])
            task_old_dist_info_list.append([])
            task_old_dist_info.append([])
            for k in self.policy.distribution.dist_info_keys:
                task_old_dist_info_list[i].append([])
        for i in range(len(samples_data["observations"])):
            taskid = samples_data["env_infos"]["state_index"][i]
            task_obs[taskid].append(samples_data["observations"][i])
            for j, k in enumerate(self.policy.distribution.dist_info_keys):
                task_old_dist_info_list[taskid][j].append(samples_data["agent_infos"][k][i])
        for i in range(self.task_num):
            for j, k in enumerate(self.policy.distribution.dist_info_keys):
                task_old_dist_info[i].append(np.array(task_old_dist_info_list[i][j]))
            task_obs[i] = np.array(task_obs[i])

        for i in range(self.task_num):
            all_input_values += tuple([task_obs[i]])
        for i in range(self.task_num):
            all_input_values += tuple(task_old_dist_info[i])
        all_input_values += tuple([self.kl_weights])

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

        # update the weights for the kl divergence
        kl_divs = []
        for constraint in self.f_constraints:
            kl_divs.append(sliced_fun(constraint, 1)(all_input_values))
        for i in range(1, len(kl_divs)):
            if kl_divs[i] < 0.2*self.step_size:
                self.kl_weights[i-1] /= 1.0
            elif kl_divs[i] > 1.0*self.step_size:
                self.kl_weights[i-1] *= 1.2
            else: # move 10% towards 1
                self.kl_weights[i-1] = self.kl_weights[i-1] + 0.1*(1-self.kl_weights[i-1])
        '''self.kl_weights /= np.sum(self.kl_weights)
        self.kl_weights *= self.task_num'''
        print('Current kl divergence weight: ', self.kl_weights)

        return dict()

    def mean_kl(self, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))

        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        # multitask related
        task_obs = []
        task_old_dist_info_list = []
        task_old_dist_info = []
        for i in range(self.task_num):
            task_obs.append([])
            task_old_dist_info_list.append([])
            task_old_dist_info.append([])
            for k in self.policy.distribution.dist_info_keys:
                task_old_dist_info_list[i].append([])
        for i in range(len(samples_data["observations"])):
            taskid = samples_data["env_infos"]["state_index"][i]
            task_obs[taskid].append(samples_data["observations"][i])
            for j, k in enumerate(self.policy.distribution.dist_info_keys):
                task_old_dist_info_list[taskid][j].append(samples_data["agent_infos"][k][i])
        for i in range(self.task_num):
            for j, k in enumerate(self.policy.distribution.dist_info_keys):
                task_old_dist_info[i].append(np.array(task_old_dist_info_list[i][j]))
            task_obs[i] = np.array(task_obs[i])

        for i in range(self.task_num):
            all_input_values += tuple([task_obs[i]])
        for i in range(self.task_num):
            all_input_values += tuple(task_old_dist_info[i])
        all_input_values += tuple([self.kl_weights])

        kl_divs = []
        for constraint in self.f_constraints:
            kl_divs.append(sliced_fun(constraint, 1)(all_input_values))
        return kl_divs

    def get_gradient(self, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))

        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        # multitask related
        task_obs = []
        task_old_dist_info_list = []
        task_old_dist_info = []
        for i in range(self.task_num):
            task_obs.append([])
            task_old_dist_info_list.append([])
            task_old_dist_info.append([])
            for k in self.policy.distribution.dist_info_keys:
                task_old_dist_info_list[i].append([])
        for i in range(len(samples_data["observations"])):
            taskid = np.random.randint(self.task_num) # fake the taskid to satisfy the calculation requirement, very ugly
            task_obs[taskid].append(samples_data["observations"][i])
            for j, k in enumerate(self.policy.distribution.dist_info_keys):
                task_old_dist_info_list[taskid][j].append(samples_data["agent_infos"][k][i])
        for i in range(self.task_num):
            for j, k in enumerate(self.policy.distribution.dist_info_keys):
                task_old_dist_info[i].append(np.array(task_old_dist_info_list[i][j]))
            task_obs[i] = np.array(task_obs[i])

        for i in range(self.task_num):
            all_input_values += tuple([task_obs[i]])
        for i in range(self.task_num):
            all_input_values += tuple(task_old_dist_info[i])
        all_input_values += tuple([self.kl_weights])

        grad = sliced_fun(self.optimizer._opt_fun["f_grads"], 1)(
            (all_input_values))

        return grad