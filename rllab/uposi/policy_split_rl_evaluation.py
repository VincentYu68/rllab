import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np

import rllab
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP, MLP_SplitAct
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import theano.tensor as TT
import theano as T
import theano

import joblib
from rllab.misc.ext import iterate_minibatches_generic
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from rllab.misc import ext
from rllab.misc.ext import sliced_fun
from rllab.algos.trpo import TRPO
from rllab.algos.trpo_mt import TRPO_MultiTask
from rllab.algos.trpo_imb import TRPO_ImbalanceMultiTask
from rllab.algos.trpo_mpsel import TRPOMPSel
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from gym import error, spaces
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import random
import time


def get_gradient(algo, samples_data, trpo_split=False):
    if trpo_split:
        return algo.get_gradient(samples_data)

    all_input_values = tuple(ext.extract(
        samples_data,
        "observations", "actions", "advantages"
    ))
    agent_infos = samples_data["agent_infos"]
    state_info_list = [agent_infos[k] for k in algo.policy.state_info_keys]
    dist_info_list = [agent_infos[k] for k in algo.policy.distribution.dist_info_keys]
    all_input_values += tuple(state_info_list) + tuple(dist_info_list)

    grad = sliced_fun(algo.optimizer._opt_fun["f_grads"], 1)(
        tuple(all_input_values), tuple())

    return grad


def get_samples(algo, task_size, adaptive_sample=False, imbalance_sample=False, batch_size = 0, sample_ratio=[]):
    if adaptive_sample:
        paths = []
        reward_paths = []
        for t in range(task_size):
            paths += algo.sampler.obtain_samples(0, t)
            reward_paths += algo.sampler.obtain_samples(0)
    elif imbalance_sample:
        paths = []
        reward_paths = []
        for t in range(task_size):
            algo.batch_size = batch_size * sample_ratio[t]
            task_path = algo.sampler.obtain_samples(0, t)
            paths += task_path
            if t == 0:
                reward_paths += task_path
            task_reward = 0
            for path in task_path:
                task_reward += np.sum(path["rewards"])
            task_reward /= len(task_path)
            print('Reward for task ', t, ' is ', task_reward)
    else:
        paths = algo.sampler.obtain_samples(0)
        reward_paths = []
    return paths, reward_paths

def perform_evaluation(num_parallel,
                       hidden_size,
                       batch_size,
                       pathlength,
                       random_split,
                       prioritized_split,
                       adaptive_sample,
                       initialize_epochs,
                       grad_epochs,
                       test_epochs,
                       append,
                       task_size,
                       load_init_policy,
                       load_split_data,
                       alternate_update,
                       accumulate_gradient,
                       imbalance_sample,
                       sample_ratio,
                       split_percentages,
                       env_name,
                       seed,
                       test_num=1,
                       param_update_start = 50,
                       param_update_frequency = 50,
                       param_update_end = 200,
                       use_param_variance = 0,
                       param_variance_batch = 10000,
                       param_variance_sample = 100):
    reps = 1

    learning_curves = []
    kl_divergences = []
    for i in range(len(split_percentages)):
        learning_curves.append([])
        kl_divergences.append([])

    performances = []

    diretory = 'data/trained/gradient_temp/rl_split_' + append

    if not os.path.exists(diretory):
        os.makedirs(diretory)
        os.makedirs(diretory + '/policies')


    for testit in range(test_num):
        print('======== Start Test ', testit, ' ========')
        env = normalize(GymEnv(env_name, record_log=False, record_video=False))
        dartenv = env._wrapped_env.env.env
        if env._wrapped_env.monitoring:
            dartenv = dartenv.env

        np.random.seed(testit * 3 + seed)
        random.seed(testit * 3 + seed)

        pre_training_learning_curve = []

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=hidden_size,
            # append_dim=2,
            net_mode=0,
        )
        baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

        if load_init_policy:
            policy = joblib.load(diretory + '/init_policy.pkl')

        algo = TRPO(  # _MultiTask(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=pathlength,
            n_itr=5,

            discount=0.995,
            step_size=0.02,
            gae_lambda=0.97,
            whole_paths=False,
            # task_num=task_size,
        )
        algo.init_opt()

        from rllab.sampler import parallel_sampler

        parallel_sampler.initialize(n_parallel=num_parallel)
        parallel_sampler.set_seed(0)

        algo.start_worker()

        if not load_init_policy:
            for i in range(initialize_epochs):
                print('------ Iter ', i, ' in Init Training --------')
                if adaptive_sample:
                    paths = []
                    reward_paths = []
                    for t in range(task_size):
                        paths += algo.sampler.obtain_samples(0, t)
                        reward_paths += algo.sampler.obtain_samples(0)
                elif imbalance_sample:
                    paths = []
                    reward_paths = []
                    for t in range(task_size):
                        algo.batch_size = batch_size * sample_ratio[t]
                        task_path = algo.sampler.obtain_samples(0, t)
                        paths += task_path
                        if t == 0:
                            reward_paths += task_path
                else:
                    paths = algo.sampler.obtain_samples(0)
                samples_data = algo.sampler.process_samples(0, paths)
                opt_data = algo.optimize_policy(0, samples_data)
                pol_aft = (policy.get_param_values())
                print(algo.mean_kl(samples_data))

                print(dict(logger._tabular)['AverageReturn'])
                pre_training_learning_curve.append(dict(logger._tabular)['AverageReturn'])
            joblib.dump(policy, diretory + '/init_policy.pkl', compress=True)

        print('------- initial training complete ---------------')
        if not load_split_data:
            split_data = []
            net_weights = []
            net_weight_values = []
            for i in range(grad_epochs):
                cur_param_val = np.copy(policy.get_param_values())
                cur_param = copy.deepcopy(policy.get_params())

                cp = []
                for param in policy._mean_network.get_params():
                    cp.append(np.copy(param.get_value()))
                net_weights.append(cp)
                net_weight_values.append(np.copy(policy.get_param_values()))

                if adaptive_sample:
                    paths = []
                    reward_paths = []
                    for t in range(task_size):
                        paths += algo.sampler.obtain_samples(0, t)
                        reward_paths += algo.sampler.obtain_samples(0)
                elif imbalance_sample:
                    paths = []
                    reward_paths = []
                    for t in range(task_size):
                        algo.batch_size = batch_size * sample_ratio[t]
                        task_path = algo.sampler.obtain_samples(0, t)
                        paths += task_path
                        if t == 0:
                            reward_paths += task_path
                else:
                    paths = algo.sampler.obtain_samples(0)
                split_data.append(paths)

                samples_data = algo.sampler.process_samples(0, paths)
                opt_data = algo.optimize_policy(0, samples_data)
                pre_training_learning_curve.append(dict(logger._tabular)['AverageReturn'])
            joblib.dump(split_data, diretory + '/split_data.pkl', compress=True)
            joblib.dump(net_weights, diretory + '/net_weights.pkl', compress=True)
            joblib.dump(net_weight_values, diretory + '/net_weight_values.pkl', compress=True)
            joblib.dump(pre_training_learning_curve, diretory + '/pretrain_learningcurve.pkl', compress=True)
        else:
            split_data = joblib.load(diretory + '/split_data.pkl')
            net_weights = joblib.load(diretory + '/net_weights.pkl')
            net_weight_values = joblib.load(diretory + '/net_weight_values.pkl')
            pre_training_learning_curve = joblib.load(diretory + '/pretrain_learningcurve.pkl')

        task_grads = []
        variance_grads = []
        for i in range(task_size):
            task_grads.append([])
        for i in range(grad_epochs):
            policy.set_param_values(net_weight_values[i])
            task_paths = []
            for j in range(task_size):
                task_paths.append([])
            for path in split_data[i]:
                taskid = path['env_infos']['state_index'][-1]
                task_paths[taskid].append(path)

            for j in range(task_size):
                samples_data = algo.sampler.process_samples(0, task_paths[j], False)
                grad = get_gradient(algo, samples_data, False)
                task_grads[j].append(grad)
            if use_param_variance == 1 and i == grad_epochs-1:
                for j in range(param_variance_sample):
                    cur_batch = 0
                    sampled_rollouts = []
                    indices = np.arange(len(split_data[i]))
                    np.random.shuffle(indices)
                    for k in range(len(indices)):
                        sampled_rollouts.append(split_data[i][indices[k]])
                        cur_batch += len(split_data[i][indices[k]]['env_infos']['state_index'])
                        if cur_batch > param_variance_batch:
                            break
                    samples_data = algo.sampler.process_samples(0, sampled_rollouts, False)
                    grad = get_gradient(algo, samples_data, False)
                    variance_grads.append(grad)
            algo.sampler.process_samples(0, split_data[i])

        weight_variances = []
        for i in range(len(task_grads[0][0]) - 1):
            weight_variances.append(np.zeros(task_grads[0][0][i].shape))
        if use_param_variance == 1:
            for k in range(len(task_grads[0][0]) - 1):
                one_grad = []
                for g in range(len(variance_grads)):
                    one_grad.append(np.asarray(variance_grads[g][k]))
                weight_variances[k] += np.var(one_grad, axis=0)

        print('------- collected gradient info -------------')

        split_counts = []
        for i in range(len(task_grads[0][0]) - 1):
            split_counts.append(np.zeros(task_grads[0][0][i].shape))

        for i in range(len(task_grads[0])):
            for k in range(len(task_grads[0][i]) - 1):
                region_gradients = []
                for region in range(len(task_grads)):
                    region_gradients.append(task_grads[region][i][k])
                region_gradients = np.array(region_gradients)
                if not random_split:
                    split_counts[k] += np.var(region_gradients,
                                              axis=0)  # * np.abs(net_weights[i][k])# + 100 * (len(task_grads[0][i])-k)
                elif prioritized_split:
                    split_counts[k] += np.random.random(split_counts[k].shape) * (len(task_grads[0][i]) - k)
                else:
                    split_counts[k] += np.random.random(split_counts[k].shape)

        for j in range(len(split_counts)):
            plt.figure()
            plt.title(policy._mean_network.get_params()[j].name)
            if len(split_counts[j].shape) == 2:
                plt.imshow(split_counts[j])
                plt.colorbar()
            elif len(split_counts[j].shape) == 1:
                plt.plot(split_counts[j])

            plt.savefig(diretory + '/' + policy._mean_network.get_params()[j].name + '.png')

            if use_param_variance:
                plt.figure()
                plt.title(policy._mean_network.get_params()[j].name)
                if len(weight_variances[j].shape) == 2:
                    plt.imshow(weight_variances[j])
                    plt.colorbar()
                elif len(weight_variances[j].shape) == 1:
                    plt.plot(weight_variances[j])

                plt.savefig(diretory + '/' + policy._mean_network.get_params()[j].name + '_variances.png')

        algo.shutdown_worker()

        # organize the metric into each edges and sort them
        split_metrics = []
        metrics_list = []
        variance_list = []
        for k in range(len(task_grads[0][0])):
            for index, value in np.ndenumerate(split_counts[k]):
                split_metrics.append([k, index, value, weight_variances[k][index]])
                metrics_list.append(value)
                variance_list.append(weight_variances[k][index])
        if use_param_variance == 0:
            split_metrics.sort(key=lambda x: x[2], reverse=True)
        else:
            split_metrics.sort(key=lambda x: x[3], reverse=False)

        # test the effect of splitting
        total_param_size = len(policy._mean_network.get_param_values())

        pred_list = []
        # use the optimized network
        init_param_value = np.copy(policy.get_param_values())

        for split_id, split_percentage in enumerate(split_percentages):
            split_param_size = split_percentage * total_param_size
            masks = []
            for k in range(len(task_grads[0][0]) - 1):
                masks.append(np.zeros(split_counts[k].shape))

            if split_percentage <= 1.0:
                for i in range(int(split_param_size)):
                    masks[split_metrics[i][0]][split_metrics[i][1]] = 1
            else:
                threshold = np.mean(metrics_list) + np.std(metrics_list)
                print('threashold,', threshold)
                for i in range(len(split_metrics)):
                    if split_metrics[i][2] < threshold:
                        break
                    else:
                        masks[split_metrics[i][0]][split_metrics[i][1]] = 1

            mask_split_flat = np.array([])
            for k in range(int((len(task_grads[0][0]) - 1) / 2)):
                for j in range(task_size):
                    mask_split_flat = np.concatenate(
                        [mask_split_flat, np.array(masks[k * 2]).flatten(), np.array(masks[k * 2 + 1]).flatten()])
            mask_share_flat = np.ones(len(mask_split_flat))
            mask_share_flat -= mask_split_flat
            if np.abs(split_percentage - 1.0) < 0.0001:
                mask_split_flat = np.concatenate([mask_split_flat, np.ones(dartenv.act_dim * task_size)])
                mask_share_flat = np.concatenate([mask_share_flat, np.zeros(dartenv.act_dim * task_size)])
            else:
                mask_split_flat = np.concatenate([mask_split_flat, np.zeros(dartenv.act_dim)])
                mask_share_flat = np.concatenate([mask_share_flat, np.ones(dartenv.act_dim)])

            policy.set_param_values(init_param_value)
            if split_param_size != 0:
                if dartenv.avg_div != task_size:
                    dartenv.avg_div = task_size
                    dartenv.obs_dim += dartenv.avg_div
                    high = np.inf * np.ones(dartenv.obs_dim)
                    low = -high
                    dartenv.observation_space = spaces.Box(low, high)
                    env._wrapped_env._observation_space = rllab.envs.gym_env.convert_gym_space(
                        dartenv.observation_space)
                    env.spec = rllab.envs.env_spec.EnvSpec(
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                    )

                split_policy = GaussianMLPPolicy(
                    env_spec=env.spec,
                    # The neural network policy should have two hidden layers, each with 32 hidden units.
                    hidden_sizes=hidden_size,
                    # append_dim=2,
                    net_mode=8,
                    split_num=task_size,
                    split_masks=masks,
                    split_init_net=policy,
                    split_std=np.abs(split_percentage - 1.0) < 0.0001,
                )
            else:
                split_policy = copy.deepcopy(policy)

            if split_param_size == 0:
                baseline_add = 0
            else:
                baseline_add = task_size  # use 0 for now, though task_size should in theory improve performance more
            split_baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=baseline_add)

            new_batch_size = batch_size
            if (split_param_size != 0 and alternate_update) or adaptive_sample:
                new_batch_size = int(batch_size / task_size)
            split_algo = TRPO(  # _MultiTask(
                env=env,
                policy=split_policy,
                baseline=split_baseline,
                batch_size=new_batch_size,
                max_path_length=pathlength,
                n_itr=5,

                discount=0.995,
                step_size=0.02,
                gae_lambda=0.97,
                whole_paths=False,
                # task_num=task_size,
            )
            split_algo.init_opt()

            parallel_sampler.initialize(n_parallel=num_parallel)
            parallel_sampler.set_seed(0)

            split_algo.start_worker()
            if split_param_size != 0:
                parallel_sampler.update_env_params({'avg_div': dartenv.avg_div, 'obs_dim': dartenv.obs_dim,
                                                    'observation_space': dartenv.observation_space})

            print('Network parameter size: ', total_param_size, len(split_policy.get_param_values()))

            split_init_param = np.copy(split_policy.get_param_values())
            avg_error = 0.0

            avg_learning_curve = []
            for rep in range(int(reps)):
                split_policy.set_param_values(split_init_param)
                learning_curve = []
                kl_div_curve = []
                for i in range(test_epochs):
                    # if not split
                    if split_param_size == 0:
                        paths, _ = get_samples(split_algo, task_size, adaptive_sample, imbalance_sample, batch_size, sample_ratio)
                        # sanity check
                        samp_num = 0
                        for p in paths:
                            samp_num += len(p['observations'])
                        print('samp_num: ', samp_num, adaptive_sample, imbalance_sample)
                        samples_data = split_algo.sampler.process_samples(0, paths)
                        opt_data = split_algo.optimize_policy(0, samples_data)

                        if adaptive_sample or imbalance_sample:
                            reward = 0
                            for path in reward_paths:
                                reward += np.sum(path["rewards"])
                            reward /= len(reward_paths)
                        else:
                            reward = float((dict(logger._tabular)['AverageReturn']))
                        kl_div_curve.append(split_algo.mean_kl(samples_data))
                        print('reward: ', reward)
                        print(split_algo.mean_kl(samples_data))
                    elif alternate_update:
                        reward = 0
                        total_traj = 0
                        task_rewards = []
                        for j in range(task_size):
                            paths = split_algo.sampler.obtain_samples(0, j)
                            # split_algo.sampler.process_samples(0, paths)
                            samples_data = split_algo.sampler.process_samples(0, paths)
                            opt_data = split_algo.optimize_policy(0, samples_data)
                            reward += float((dict(logger._tabular)['AverageReturn'])) * float(
                                (dict(logger._tabular)['NumTrajs']))
                            total_traj += float((dict(logger._tabular)['NumTrajs']))
                            task_rewards.append(dict(logger._tabular)['AverageReturn'])
                        reward /= total_traj
                        print('reward for different tasks: ', task_rewards, reward)
                    elif accumulate_gradient:
                        paths, _ = get_samples(split_algo, task_size, adaptive_sample, imbalance_sample, batch_size, sample_ratio)

                        task_paths = []
                        task_rewards = []
                        for j in range(task_size):
                            task_paths.append([])
                            task_rewards.append([])
                        for path in paths:
                            taskid = path['env_infos']['state_index'][-1]
                            task_paths[taskid].append(path)
                            task_rewards[taskid].append(np.sum(path['rewards']))
                        pre_opt_parameter = np.copy(split_policy.get_param_values())

                        # compute the split gradient first
                        split_policy.set_param_values(pre_opt_parameter)
                        accum_grad = np.zeros(pre_opt_parameter.shape)
                        processed_task_data = []
                        for j in range(task_size):
                            if len(task_paths[j]) == 0:
                                processed_task_data.append([])
                                continue
                            split_policy.set_param_values(pre_opt_parameter)
                            # split_algo.sampler.process_samples(0, task_paths[j])
                            samples_data = split_algo.sampler.process_samples(0, task_paths[j], False)
                            processed_task_data.append(samples_data)
                            split_algo.optimize_policy(0, samples_data)

                            # if j == 1:
                            accum_grad += split_policy.get_param_values() - pre_opt_parameter
                        # sanity check
                        samp_num = 0
                        for p in paths:
                            samp_num += len(p['observations'])
                        print('samp_num: ', samp_num)

                        # compute the gradient together
                        split_policy.set_param_values(pre_opt_parameter)
                        all_data = split_algo.sampler.process_samples(0, paths)
                        if adaptive_sample or imbalance_sample:
                            reward = 0
                            for path in reward_paths:
                                reward += np.sum(path["rewards"])
                            reward /= len(reward_paths)
                        else:
                            reward = float((dict(logger._tabular)['AverageReturn']))

                        #split_algo.optimize_policy(0, all_data)
                        #all_data_grad = split_policy.get_param_values() - pre_opt_parameter

                        # do a line search to project the udpate onto the constraint manifold
                        sum_grad = accum_grad# * mask_split_flat + all_data_grad * mask_share_flat

                        ls_steps = []
                        loss_before = split_algo.loss(all_data)

                        for s in range(40):
                            ls_steps.append(0.95 ** s)
                        for step in ls_steps:
                            split_policy.set_param_values(pre_opt_parameter + sum_grad * step)
                            if split_algo.mean_kl(all_data)[0] < split_algo.step_size:  # and split_algo.loss(all_data)[0] < loss_before[0]:
                                break
                        # step=1

                        split_policy.set_param_values(pre_opt_parameter + sum_grad * step)

                        for j in range(task_size):
                            task_rewards[j] = np.mean(task_rewards[j])

                        print('reward for different tasks: ', task_rewards, reward)
                        print('mean kl: ', split_algo.mean_kl(all_data), ' step size: ', step)
                        task_mean_kls = []
                        for j in range(task_size):
                            if len(processed_task_data[j]) == 0:
                                task_mean_kls.append(0)
                            else:
                                task_mean_kls.append(split_algo.mean_kl(processed_task_data[j])[0])
                        print('mean kl for different tasks: ', task_mean_kls)
                        kl_div_curve.append(np.concatenate([split_algo.mean_kl(all_data), task_mean_kls]))
                    else:
                        paths = split_algo.sampler.obtain_samples(0)
                        reward = float((dict(logger._tabular)['AverageReturn']))
                        task_paths = []
                        task_rewards = []
                        for j in range(task_size):
                            task_paths.append([])
                            task_rewards.append([])
                        for path in paths:
                            taskid = path['env_infos']['state_index'][-1]
                            task_paths[taskid].append(path)
                            task_rewards[taskid].append(np.sum(path['rewards']))
                        pre_opt_parameter = np.copy(split_policy.get_param_values())
                        # optimize the shared part
                        # split_algo.sampler.process_samples(0, paths)
                        samples_data = split_algo.sampler.process_samples(0, paths)
                        for layer in split_policy._mean_network._layers:
                            for param in layer.get_params():
                                if 'split' in param.name:
                                    layer.params[param].remove('trainable')
                        split_policy._cached_params = {}
                        split_policy._cached_param_dtypes = {}
                        split_policy._cached_param_shapes = {}
                        split_algo.init_opt()
                        print('Optimizing shared parameter size: ', len(split_policy.get_param_values(trainable=True)))
                        split_algo.optimize_policy(0, samples_data)

                        # optimize the tasks
                        for layer in split_policy._mean_network._layers:
                            for param in layer.get_params():
                                if 'split' in param.name:
                                    layer.params[param].add('trainable')
                                if 'share' in param.name:
                                    layer.params[param].remove('trainable')

                        # shuffle the optimization order
                        opt_order = np.arange(task_size)
                        np.random.shuffle(opt_order)
                        split_policy._cached_params = {}
                        split_policy._cached_param_dtypes = {}
                        split_policy._cached_param_shapes = {}
                        split_algo.init_opt()
                        for taskid in opt_order:
                            # split_algo.sampler.process_samples(0, task_paths[taskid])
                            samples_data = split_algo.sampler.process_samples(0, task_paths[taskid])
                            print('Optimizing parameter size: ', len(split_policy.get_param_values(trainable=True)))
                            split_algo.optimize_policy(0, samples_data)
                        for layer in split_policy._mean_network._layers:
                            for param in layer.get_params():
                                if 'share' in param.name:
                                    layer.params[param].add('trainable')

                        for j in range(task_size):
                            task_rewards[j] = np.mean(task_rewards[j])
                        print('reward for different tasks: ', task_rewards, reward)

                    learning_curve.append(reward)
                    if (i+initialize_epochs+grad_epochs) % param_update_frequency == 0 and (i+initialize_epochs+grad_epochs) < param_update_end and (i+initialize_epochs+grad_epochs) > param_update_start:
                        print("Updating model parameters...")
                        parallel_sampler.update_env_params({'task_expand_flag': True})
                    print('============= Finished ', split_percentage, ' Rep ', rep, '   test ', i, ' ================')
                    print(diretory)
                    joblib.dump(split_policy, diretory + '/policies/policy_' + str(rep) + '_' + str(i) + '_' + str(
                        split_percentage) + '.pkl',
                                compress=True)
                avg_learning_curve.append(learning_curve)
                kl_divergences[split_id].append(kl_div_curve)
                joblib.dump(split_policy, diretory + '/policies/final_policy_' + str(split_percentage) + '.pkl',
                            compress=True)

                avg_error += float(reward)
            pred_list.append(avg_error / reps)
            print(split_percentage, avg_error / reps)
            split_algo.shutdown_worker()
            print(avg_learning_curve)
            avg_learning_curve = np.mean(avg_learning_curve, axis=0)
            learning_curves[split_id].append(avg_learning_curve)
            # output the learning curves so far
            joblib.dump(learning_curves, diretory + '/learning_curve.pkl', compress=True)
            avg_learning_curve = []
            for lc in range(len(learning_curves)):
                avg_learning_curve.append(np.mean(learning_curves[lc], axis=0))
            plt.figure()
            for lc in range(len(learning_curves)):
                plt.plot(avg_learning_curve[lc], label=str(split_percentages[lc]))
            plt.legend(bbox_to_anchor=(0.3, 0.3),
                       bbox_transform=plt.gcf().transFigure, numpoints=1)
            plt.savefig(diretory + '/split_learning_curves.png')

            if len(kl_divergences[0]) > 0:
                #print('kldiv:', kl_divergences)
                avg_kl_div = []
                for i in range(len(kl_divergences)):
                    if len(kl_divergences[i]) > 0:
                        avg_kl_div.append(np.mean(kl_divergences[i], axis=0))
                #print(avg_kl_div)
                for i in range(len(avg_kl_div)):
                    one_perc_kl_div = np.array(avg_kl_div[i])
                    #print(i, one_perc_kl_div)
                    plt.figure()
                    for j in range(len(one_perc_kl_div[0])):
                        append = 'task%d' % j
                        if j == 0:
                            append = 'all'
                        plt.plot(one_perc_kl_div[:, j], label=str(split_percentages[i]) + append, alpha=0.3)
                    plt.legend(bbox_to_anchor=(0.3, 0.3),
                               bbox_transform=plt.gcf().transFigure, numpoints=1)
                    plt.savefig(diretory + '/kl_div_%s.png' % str(split_percentages[i]))
        performances.append(pred_list)

    np.savetxt(diretory + '/performance.txt', performances)
    plt.figure()
    plt.plot(split_percentages, np.mean(performances, axis=0))
    plt.savefig(diretory + '/split_performance.png')

    avg_learning_curve = []
    for i in range(len(learning_curves)):
        avg_learning_curve.append(np.mean(learning_curves[i], axis=0))
    plt.figure()
    for i in range(len(split_percentages)):
        plt.plot(avg_learning_curve[i], label=str(split_percentages[i]))
    plt.legend(bbox_to_anchor=(0.3, 0.3),
               bbox_transform=plt.gcf().transFigure, numpoints=1)
    plt.savefig(diretory + '/split_learning_curves.png')
    np.savetxt(diretory + '/learning_curves.txt', avg_learning_curve)

    if len(kl_divergences[0]) > 0:
        avg_kl_div = []
        for i in range(len(kl_divergences)):
            avg_kl_div.append(np.mean(kl_divergences[i], axis=0))
        for i in range(len(avg_kl_div)):
            one_perc_kl_div = np.array(avg_kl_div[i])
            plt.figure()
            for j in range(len(one_perc_kl_div[0])):
                append = 'task%d' % j
                if j == 0:
                    append = 'all'
                plt.plot(one_perc_kl_div[:, j], label=str(split_percentages[i]) + append, alpha=0.3)
            plt.legend(bbox_to_anchor=(0.3, 0.3),
                       bbox_transform=plt.gcf().transFigure, numpoints=1)
            plt.savefig(diretory + '/kl_div_%s.png' % str(split_percentages[i]))

    plt.close('all')

    print(diretory)
