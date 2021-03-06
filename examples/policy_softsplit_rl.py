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
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from rllab.misc import ext
from rllab.misc.ext import sliced_fun
from rllab.algos.trpo import TRPO
from rllab.algos.trpo_split import TRPOSplit
from rllab.algos.trpo_mpsel import TRPOMPSel
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from gym import error, spaces
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


def get_gradient(algo, samples_data):
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
    hidden_size = (8,)
    batch_size = 2000
    dartenv = env._wrapped_env.env.env
    if env._wrapped_env.monitoring:
        dartenv = dartenv.env
    dartenv.avg_div = 0
    dartenv.split_task_test = True

    random_split = False
    prioritized_split = False

    initialize_epochs = 10
    grad_epochs = 50
    test_epochs = 100
    append = 'hopper_0802_sd3_%dk_%d_%d_unweighted'%(batch_size/1000, initialize_epochs, grad_epochs)

    reps = 1
    if random_split:
        append += '_rand'
        if prioritized_split:
            append += '_prio'

    load_init_policy = False
    load_split_data = False

    #split_percentages = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0]
    split_percentages = [0.0, 0.1, 0.01]
    learning_curves = []
    for i in range(len(split_percentages)):
        learning_curves.append([])

    test_num = 1
    performances = []

    if not os.path.exists('data/trained/gradient_temp/rl_split_' + append):
        os.makedirs('data/trained/gradient_temp/rl_split_' + append)

    average_metric_list = []

    for testit in range(test_num):
        print('======== Start Test ', testit, ' ========')
        np.random.seed(testit*3+2)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=hidden_size,
            # append_dim=2,
            net_mode=0,
        )
        baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

        if load_init_policy:
            policy = joblib.load('data/trained/gradient_temp/rl_split_' + append + '/init_policy.pkl')

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=500,
            n_itr=5,

            discount=0.995,
            step_size=0.01,
            gae_lambda=0.97,
        )
        algo.init_opt()
        from rllab.sampler import parallel_sampler
        parallel_sampler.initialize(n_parallel=8)
        algo.start_worker()

        if not load_init_policy:
            for i in range(initialize_epochs):
                paths = algo.sampler.obtain_samples(0)
                # if not split
                samples_data = algo.sampler.process_samples(0, paths)
                opt_data = algo.optimize_policy(0, samples_data)
                print(dict(logger._tabular)['AverageReturn'])
            joblib.dump(policy, 'data/trained/gradient_temp/rl_split_' + append + '/init_policy.pkl', compress=True)

        print('------- initial training complete ---------------')

        init_param_value = np.copy(policy.get_param_values())

        task_grads = []
        for i in range(2):
            task_grads.append([])

        if not load_split_data:
            split_data = []
            net_weights = []
            for i in range(grad_epochs):
                cur_param_val = np.copy(policy.get_param_values())
                cur_param = copy.deepcopy(policy.get_params())

                cp = []
                for param in policy._mean_network.get_params():
                    cp.append(np.copy(param.get_value()))
                net_weights.append(cp)

                paths = algo.sampler.obtain_samples(0)
                split_data.append(paths)

                algo.sampler.process_samples(0, paths)
                samples_data = algo.sampler.process_samples(0, paths)
                opt_data = algo.optimize_policy(0, samples_data)
            joblib.dump(split_data, 'data/trained/gradient_temp/rl_split_' + append + '/split_data.pkl', compress=True)
            joblib.dump(net_weights, 'data/trained/gradient_temp/rl_split_' + append + '/net_weights.pkl', compress=True)
        else:
            split_data = joblib.load('data/trained/gradient_temp/rl_split_' + append + '/split_data.pkl')
            net_weights = joblib.load('data/trained/gradient_temp/rl_split_' + append + '/net_weights.pkl')

        for i in range(grad_epochs):
            # if not split
            task_paths = [[], []]
            for path in split_data[i]:
                taskid = 0
                taskid = path['env_infos']['state_index'][-1]
                task_paths[taskid].append(path)

            for j in range(2):
                algo.sampler.process_samples(0, task_paths[j])
                samples_data = algo.sampler.process_samples(0, task_paths[j])
                grad = get_gradient(algo, samples_data)
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

            plt.savefig('data/trained/gradient_temp/rl_split_' + append + '/' + policy._mean_network.get_params()[j].name + '.png')

        algo.shutdown_worker()

        # test the effect of splitting
        total_param_size = len(policy._mean_network.get_param_values())

        split_indices = []
        metrics_lsit = []
        ord = 0
        for p in range(int(len(split_counts)/2)):
            for col in range(split_counts[p*2].shape[1]):
                split_metric = np.mean(split_counts[p*2][:, col]) + split_counts[p*2+1][col]
                split_indices.append([[p, col], split_metric])
                metrics_lsit.append([ord, split_metric])
                ord+=1
        split_indices.sort(key=lambda x:x[1], reverse=True)

        metrics_lsit = np.array(metrics_lsit)
        plt.figure()
        plt.plot(metrics_lsit[:,0], metrics_lsit[:, 1])
        plt.savefig('data/trained/gradient_temp/rl_split_' + append + '/metric_rank.png')
        average_metric_list.append(metrics_lsit)

        pred_list = []
        # use the optimized network
        init_param_value = np.copy(policy.get_param_values())
        individual_test = False

        for split_id, split_percentage in enumerate(split_percentages):
            split_param_size = split_percentage * total_param_size
            policy.set_param_values(init_param_value)
            if split_param_size != 0:
                if dartenv.avg_div != 2:
                    dartenv.avg_div = 2
                    dartenv.obs_dim += dartenv.avg_div
                    high = np.inf*np.ones(dartenv.obs_dim)
                    low = -high
                    dartenv.observation_space = spaces.Box(low, high)
                    env._wrapped_env._observation_space = rllab.envs.gym_env.convert_gym_space(dartenv.observation_space)
                    env.spec = rllab.envs.env_spec.EnvSpec(
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                        )
                split_policy = GaussianMLPPolicy(
                    env_spec=env.spec,
                    # The neural network policy should have two hidden layers, each with 32 hidden units.
                    hidden_sizes=hidden_size,
                    #append_dim=2,
                    net_mode=7,
                    split_num=2,
                    split_init_net=policy,
                )
            else:
                split_policy = copy.deepcopy(policy)

            split_baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)
            if split_param_size != 0:
                split_algo = TRPOSplit(
                    env=env,
                    policy=split_policy,
                    baseline=split_baseline,
                    batch_size=batch_size,
                    max_path_length=500,
                    n_itr=5,

                    discount=0.995,
                    step_size=0.02,
                    gae_lambda=0.97,
                    split_weight=split_percentage,
                    split_importance = split_counts,
                )
            else:
                split_algo = TRPO(
                    env=env,
                    policy=split_policy,
                    baseline=split_baseline,
                    batch_size=batch_size,
                    max_path_length=500,
                    n_itr=5,

                    discount=0.995,
                    step_size=0.01,
                    gae_lambda=0.97,
                )
            split_algo.init_opt()

            parallel_sampler.initialize(n_parallel=8)
            split_algo.start_worker()
            print('Network parameter size: ', total_param_size, len(split_policy.get_param_values()))

            split_init_param = np.copy(split_policy.get_param_values())
            split_init_pms = copy.deepcopy(split_policy.get_params())
            avg_error = 0.0

            avg_learning_curve = []
            for rep in range(int(reps)):
                split_policy.set_param_values(split_init_param)
                learning_curve = []
                for i in range(test_epochs):
                    paths = split_algo.sampler.obtain_samples(0)
                    # if not split
                    samples_data = split_algo.sampler.process_samples(0, paths)
                    opt_data = split_algo.optimize_policy(0, samples_data)
                    reward = float((dict(logger._tabular)['AverageReturn']))
                    learning_curve.append(reward)
                    print('============= Finished Rep ', rep, '   test ', i, ' ================')
                avg_learning_curve.append(learning_curve)

                avg_error += float(reward)
                '''print('parameters: ', split_policy.get_params()[0].get_value()-split_init_pms[0].get_value())
                if split_percentage > 0:
                    print(split_policy.get_params()[2].get_value()-split_init_pms[0].get_value())'''
            pred_list.append(avg_error / reps)
            print(split_percentage, avg_error / reps)
            split_algo.shutdown_worker()
            print(avg_learning_curve)
            avg_learning_curve = np.mean(avg_learning_curve, axis=0)
            if not individual_test:
                learning_curves[split_id].append(avg_learning_curve)
        performances.append(pred_list)



    if individual_test:
        plt.figure()
        average_metric_list = np.mean(average_metric_list, axis=0)
        plt.plot(average_metric_list[:,0], -average_metric_list[:, 1])
        plt.savefig('data/trained/gradient_temp/rl_split_' + append + '/metric_rank.png')

    np.savetxt('data/trained/gradient_temp/rl_split_' + append + '/performance.txt', performances)
    plt.figure()
    plt.plot(split_percentages, np.mean(performances, axis=0))
    plt.savefig('data/trained/gradient_temp/rl_split_' + append + '/split_performance.png')

    if not individual_test:
        avg_learning_curve = []
        for i in range(len(learning_curves)):
            avg_learning_curve.append(np.mean(learning_curves[i], axis=0))
        plt.figure()
        for i in range(len(split_percentages)):
            plt.plot(avg_learning_curve[i], label=str(split_percentages[i]), alpha=0.4)
        plt.legend(bbox_to_anchor=(0.3, 0.3),
        bbox_transform=plt.gcf().transFigure, numpoints=1)
        plt.savefig('data/trained/gradient_temp/rl_split_' + append + '/split_learning_curves.png')

    plt.close('all')

