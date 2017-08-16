__author__ = 'yuwenhao'

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
from rllab.algos.trpo_mpsel import TRPOMPSel
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from gym import error, spaces
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import random


def get_gradient(algo, samples_data, trpo_split = False):
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

if __name__ == '__main__':
    env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))
    dartenv = env._wrapped_env.env.env
    if env._wrapped_env.monitoring:
        dartenv = dartenv.env
    dartenv.avg_div = 0
    dartenv.split_task_test = True

    num_parallel = 2

    hidden_size = (64, 64)
    batch_size = 8000
    pathlength = 500

    random_split = False
    prioritized_split = False
    adaptive_sample = True

    initialize_epochs = 0
    grad_epochs = 1
    test_epochs = 300
    append = 'hopper_split_test_adaptsample_sd3_%dk_%d_%d_unweighted'%(batch_size/1000, initialize_epochs, grad_epochs)

    task_size = 2

    reps = 1
    if random_split:
        append += '_rand'
        if prioritized_split:
            append += '_prio'

    load_init_policy = False
    load_split_data = False

    alternate_update = False
    accumulate_gradient = True

    if alternate_update:
        append += '_alternate_update'
    if accumulate_gradient:
        append += '_accumulate_gradient'

    #split_percentages = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0]
    split_percentages = [0.0]
    learning_curves = []
    for i in range(len(split_percentages)):
        learning_curves.append([])

    test_num = 1
    performances = []
    kl_divergences = []

    diretory = 'data/trained/gradient_temp/rl_split_' + append

    if not os.path.exists(diretory):
        os.makedirs(diretory)

    average_metric_list = []

    for testit in range(test_num):
        print('======== Start Test ', testit, ' ========')
        env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))
        dartenv = env._wrapped_env.env.env
        if env._wrapped_env.monitoring:
            dartenv = dartenv.env
        dartenv.avg_div = 0
        dartenv.split_task_test = True

        np.random.seed(testit*3+3)
        random.seed(testit*3+3)

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

        algo = TRPO_MultiTask(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=pathlength,
            n_itr=5,

            discount=0.995,
            step_size=0.01,
            gae_lambda=0.97,

            task_num=2,
        )
        algo.init_opt()

        from rllab.sampler import parallel_sampler
        parallel_sampler.initialize(n_parallel=num_parallel)
        algo.start_worker()

        if not load_init_policy:
            for i in range(initialize_epochs):
                val_bf = np.copy(policy.get_param_values())
                paths = algo.sampler.obtain_samples(0)
                # if not split
                samples_data = algo.sampler.process_samples(0, paths)
                opt_data = algo.optimize_policy(0, samples_data)
                pol_aft = (policy.get_param_values())
                print(algo.mean_kl(samples_data))

                print(dict(logger._tabular)['AverageReturn'])
            joblib.dump(policy, diretory + '/init_policy.pkl', compress=True)

        print('------- initial training complete ---------------')

        init_param_value = np.copy(policy.get_param_values())

        task_grads = []
        for i in range(task_size):
            task_grads.append([])

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

                paths = algo.sampler.obtain_samples(0)
                split_data.append(paths)

                algo.sampler.process_samples(0, paths)
                samples_data = algo.sampler.process_samples(0, paths)
                opt_data = algo.optimize_policy(0, samples_data)
            joblib.dump(split_data, diretory + '/split_data.pkl', compress=True)
            joblib.dump(net_weights, diretory + '/net_weights.pkl', compress=True)
            joblib.dump(net_weight_values, diretory + '/net_weight_values.pkl', compress=True)
        else:
            split_data = joblib.load(diretory + '/split_data.pkl')
            net_weights = joblib.load(diretory + '/net_weights.pkl')
            net_weight_values = joblib.load(diretory + '/net_weight_values.pkl')

        for i in range(grad_epochs):
            policy.set_param_values(net_weight_values[i])
            task_paths = []
            for j in range(task_size):
                task_paths.append([])
            for path in split_data[i]:
                taskid = 0
                taskid = path['env_infos']['state_index'][-1]
                task_paths[taskid].append(path)

            for j in range(task_size):
                algo.sampler.process_samples(0, task_paths[j])
                samples_data = algo.sampler.process_samples(0, task_paths[j])
                grad = get_gradient(algo, samples_data, True)
                task_grads[j].append(grad)

        print('------- collected gradient info -------------')

        split_counts = []
        for i in range(len(task_grads[0][0])-1):
            split_counts.append(np.zeros(task_grads[0][0][i].shape))

        for i in range(len(task_grads[0])):
            for k in range(len(task_grads[0][i])-1):
                region_gradients = []
                for region in range(len(task_grads)):
                    region_gradients.append(task_grads[region][i][k])
                region_gradients = np.array(region_gradients)
                if not random_split:
                    split_counts[k] += np.var(region_gradients, axis=0)# * np.abs(net_weights[i][k])# + 100 * (len(task_grads[0][i])-k)
                elif prioritized_split:
                    split_counts[k] += np.random.random(split_counts[k].shape) * (len(task_grads[0][i])-k)
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
        algo.shutdown_worker()

        # organize the metric into each edges and sort them
        split_metrics = []
        metrics_list = []
        for k in range(len(task_grads[0][0])-1):
            for index, value in np.ndenumerate(split_counts[k]):
                split_metrics.append([k, index, value])
                metrics_list.append(value)
        split_metrics.sort(key=lambda x:x[2], reverse=True)

        # test the effect of splitting
        total_param_size = len(policy._mean_network.get_param_values())

        for i in range(int(len(split_counts))):
            split_counts[i] *= 0

        pred_list = []
        # use the optimized network
        init_param_value = np.copy(policy.get_param_values())

        for split_id, split_percentage in enumerate(split_percentages):
            split_param_size = split_percentage * total_param_size
            masks = []
            for k in range(len(task_grads[0][0])-1):
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

            policy.set_param_values(init_param_value)
            if split_param_size != 0:
                if dartenv.avg_div != task_size:
                    dartenv.avg_div = task_size
                    dartenv.obs_dim += dartenv.avg_div
                    high = np.inf*np.ones(dartenv.obs_dim)
                    low = -high
                    dartenv.observation_space = spaces.Box(low, high)
                    env._wrapped_env._observation_space = rllab.envs.gym_env.convert_gym_space(dartenv.observation_space)
                    env.spec = rllab.envs.env_spec.EnvSpec(
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                        )
                print(masks[0].shape)
                split_policy = GaussianMLPPolicy(
                    env_spec=env.spec,
                    # The neural network policy should have two hidden layers, each with 32 hidden units.
                    hidden_sizes=hidden_size,
                    #append_dim=2,
                    net_mode=8,
                    split_num=task_size,
                    split_masks=masks,
                    split_init_net=policy,
                )
            else:
                split_policy = copy.deepcopy(policy)

            split_baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

            new_batch_size = batch_size
            if split_param_size != 0 and (alternate_update or adaptive_sample):
                new_batch_size = int(batch_size / task_size)
            split_algo = TRPO_MultiTask(
                env=env,
                policy=split_policy,
                baseline=split_baseline,
                batch_size=new_batch_size,
                max_path_length=pathlength,
                n_itr=5,

                discount=0.995,
                step_size=0.01,
                gae_lambda=0.97,

                task_num=2,
            )
            split_algo.init_opt()

            parallel_sampler.initialize(n_parallel=num_parallel)
            split_algo.start_worker()
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
                        if adaptive_sample:
                            paths = []
                            for t in range(task_size):
                                paths += split_algo.sampler.obtain_samples(0, t)
                        else:
                            paths = split_algo.sampler.obtain_samples(0)
                        samples_data = split_algo.sampler.process_samples(0, paths)
                        opt_data = split_algo.optimize_policy(0, samples_data)
                        reward = float((dict(logger._tabular)['AverageReturn']))
                        kl_div_curve.append(split_algo.mean_kl(samples_data))
                    elif alternate_update:
                        reward = 0
                        total_traj = 0
                        task_rewards = []
                        for j in range(task_size):
                            paths = split_algo.sampler.obtain_samples(0, j)
                            split_algo.sampler.process_samples(0, paths)
                            samples_data = split_algo.sampler.process_samples(0, paths)
                            opt_data = split_algo.optimize_policy(0, samples_data)
                            reward += float((dict(logger._tabular)['AverageReturn'])) * float((dict(logger._tabular)['NumTrajs']))
                            total_traj += float((dict(logger._tabular)['NumTrajs']))
                            task_rewards.append(dict(logger._tabular)['AverageReturn'])
                        reward /= total_traj
                        print('reward for different tasks: ', task_rewards, reward)
                    elif accumulate_gradient:
                        paths = split_algo.sampler.obtain_samples(0)
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

                        split_algo.sampler.process_samples(0, paths)
                        all_data = split_algo.sampler.process_samples(0, paths)
                        reward = float((dict(logger._tabular)['AverageReturn']))
                        split_algo.optimize_policy(0, all_data)
                        all_data_grad = split_policy.get_param_values() - pre_opt_parameter

                        accum_grad = np.zeros(pre_opt_parameter.shape)
                        processed_task_data = []
                        for j in range(task_size):
                            if len(task_paths[j]) == 0:
                                continue
                            split_policy.set_param_values(pre_opt_parameter)
                            split_algo.sampler.process_samples(0, task_paths[j])
                            samples_data = split_algo.sampler.process_samples(0, task_paths[j])
                            processed_task_data.append(samples_data)
                            split_algo.optimize_policy(0, samples_data)
                            accum_grad += split_policy.get_param_values() - pre_opt_parameter
                        split_policy.set_param_values(pre_opt_parameter + accum_grad * split_percentage + (1-split_percentage)*all_data_grad)

                        for j in range(task_size):
                            task_rewards[j] = np.mean(task_rewards[j])

                        print('reward for different tasks: ', task_rewards, reward)
                        print('mean kl: ', split_algo.mean_kl(all_data))
                        task_mean_kls = []
                        for j in range(task_size):
                            task_mean_kls.append(split_algo.mean_kl(processed_task_data[j]))
                        print('mean kl for different tasks: ', task_mean_kls)
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
                        split_algo.sampler.process_samples(0, paths)
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
                            split_algo.sampler.process_samples(0, task_paths[taskid])
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

                    print('============= Finished ', split_percentage, ' Rep ', rep, '   test ', i, ' ================')
                avg_learning_curve.append(learning_curve)
                kl_divergences.append(kl_div_curve)
                joblib.dump(split_policy, diretory + '/final_policy_'+str(split_percentage)+'.pkl', compress=True)

                avg_error += float(reward)
            pred_list.append(avg_error / reps)
            print(split_percentage, avg_error / reps)
            split_algo.shutdown_worker()
            print(avg_learning_curve)
            avg_learning_curve = np.mean(avg_learning_curve, axis=0)
            learning_curves[split_id].append(avg_learning_curve)
            # output the learning curves so far
            avg_learning_curve = []
            for lc in range(len(learning_curves)):
                avg_learning_curve.append(np.mean(learning_curves[lc], axis=0))
            plt.figure()
            for lc in range(len(learning_curves)):
                plt.plot(avg_learning_curve[lc], label=str(split_percentages[lc]))
            plt.legend(bbox_to_anchor=(0.3, 0.3),
            bbox_transform=plt.gcf().transFigure, numpoints=1)
            plt.savefig(diretory + '/split_learning_curves.png')
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

    if len(kl_divergences[0]) > 0:
        for i in range(len(kl_divergences)):
            one_perc_kl_div = np.array(kl_divergences[i])
            plt.figure()
            for j in range(len(one_perc_kl_div[0])):
                append='task%d'%j
                if j == 0:
                    append = 'all'
                plt.plot(one_perc_kl_div[:, j], label=str(split_percentages[i])+append)
            plt.legend(bbox_to_anchor=(0.3, 0.3),
            bbox_transform=plt.gcf().transFigure, numpoints=1)
            plt.savefig(diretory + '/kl_div_%s.png'%str(split_percentages[i]))


    plt.close('all')

