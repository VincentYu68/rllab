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
from rllab.algos.trpo_imb import TRPO_ImbalanceMultiTask
from rllab.algos.trpo_mpsel import TRPOMPSel
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from gym import error, spaces
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import random

def get_flat_gradient(algo, samples_data, trpo_split = False):
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

    grad = sliced_fun(algo.optimizer._opt_fun["f_grad"], 1)(
        tuple(all_input_values), tuple())

    return grad



if __name__ == '__main__':
    print('======== Start Test ========')
    env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))
    dartenv = env._wrapped_env.env.env
    if env._wrapped_env.monitoring:
        dartenv = dartenv.env

    np.random.seed(3)
    random.seed(3)

    num_parallel = 2

    hidden_size = (64, 32)
    batch_size = 20000
    pathlength = 1000

    random_split = False
    prioritized_split = False
    adaptive_sample = False

    initialize_epochs = 50
    append = ''

    var_test_time = 10

    variances = []

    diretory = 'data/trained/gradient_temp/rl_split_' + append

    if not os.path.exists(diretory):
        os.makedirs(diretory)

    policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=hidden_size,
            # append_dim=2,
            net_mode=0,
        )

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=pathlength,
        n_itr=5,

        discount=0.995,
        step_size=0.01,
        gae_lambda=0.97,

        #task_num=task_size,
    )
    algo.init_opt()

    from rllab.sampler import parallel_sampler
    parallel_sampler.initialize(n_parallel=num_parallel)
    parallel_sampler.set_seed(0)

    algo.start_worker()

    for i in range(initialize_epochs):
        print('------ Iter ',i,' in Init Training ',diretory,'--------')
        paths = algo.sampler.obtain_samples(0)
        samples_data = algo.sampler.process_samples(0, paths)
        opt_data = algo.optimize_policy(0, samples_data)
        pol_aft = (policy.get_param_values())
        print(algo.mean_kl(samples_data))
        print(dict(logger._tabular)['AverageReturn'])

    data_perc_list = [0.999, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]

    testpaths = algo.sampler.obtain_samples(0)
    for perc in data_perc_list:
        sampnum = int(batch_size*perc)
        grads = []
        for i in range(var_test_time):
            idx = np.random.choice(len(testpaths), len(testpaths))
            algo.sampler.process_samples(0, testpaths)
            selected_paths = []
            current_sample_num = 0
            for id in idx:
                selected_paths.append(testpaths[id])
                current_sample_num += len(testpaths[id]["observations"])
                if current_sample_num > sampnum:
                    break
            print(len(testpaths), len(selected_paths))
            samp_data = algo.sampler.process_samples(0, selected_paths, False)
            grad = get_flat_gradient(algo, samp_data)
            grads.append(grad)
        variances.append(np.mean(np.var(grads, axis=1)))

    algo.shutdown_worker()
    plt.figure()
    plt.plot(np.array(data_perc_list)*batch_size, variances)

    plt.savefig(diretory + '/variances.png')

    plt.close('all')

