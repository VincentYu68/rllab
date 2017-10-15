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
from rllab.algos.trpo_sym import TRPO_Symmetry
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from gym import error, spaces
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import random
import time

def get_curriculum_estimations(paths):
    curriculum_candidates = paths[0]['env_infos']['curriculum_candidates'][0]
    return_collections = []
    for i in range(len(curriculum_candidates)):
        return_collections.append([])
    for path in paths:
        path_return = np.sum(path['rewards'])
        return_collections[path['env_infos']['curriculum_id'][0]].append(path_return)
    avg_returns = []
    for rc in return_collections:
        avg_returns.append(np.mean(rc))
    return curriculum_candidates, avg_returns

def evaluate_policy(env, policy, reps=6):
    avg_return = 0
    for i in range(reps):  # average performance over 10 trajectories
        o = env.reset()
        while True:
            o, rew, done, _ = env.step(policy.get_action(o)[0])
            avg_return += rew
            if done:
                break
    return avg_return / reps


def binary_search_curriculum(env, policy, anchor, direction, threshold, max_step):
    current_min = 0.0
    if anchor[0] / np.linalg.norm(anchor) < direction[0]:
        current_max = np.abs(anchor[0] / direction[0])
    else:
        current_max = np.abs(anchor[1] / direction[1])
    bound_point = anchor + direction * current_max
    env.set_param_values({'anchor_kp':bound_point})
    bound_performance = evaluate_policy(env, policy)
    if (bound_performance - threshold) < np.abs(threshold * 0.1) and bound_performance > threshold:
        return bound_point

    for i in range(max_step):
        current_step = 0.5 * (current_max + current_min)
        current_point = anchor + current_step * direction
        env.set_param_values({'anchor_kp': current_point})
        curr_perf = evaluate_policy(env, policy)
        if (curr_perf - threshold) < np.abs(threshold * 0.1) and curr_perf > threshold:
            return current_point
        if curr_perf > threshold:
            current_min = current_step
        if curr_perf < threshold:
            current_max = current_step
    return anchor + current_min * direction


if __name__ == '__main__':
    num_parallel = 8

    batch_size = 60000

    total_iterations = 1000

    # load pre-trained policy
    policy = joblib.load('data/trained/walker3d/DartWalker3d-v1_sd0_ancthres0.7_progthres0.5separate_testing/policy_990.pkl')
    init_curriculum = np.array([100, 40])

    env_name = "DartWalker3d-v1"
    seed = 0

    separate_testing = True # use binary search to find candidates
    anchor_threshold = 0.7
    progress_threshold = 0.5

    ref_policy = joblib.load(
        'data/local/experiment/walker3d_symmetry1_sd13_1alivebonus_2velrew_targetvelocity1_15frameskip_5en1absenergypenalty_spd20002000/policy.pkl')
    ref_curriculum = np.array([2000, 2000])

    append = env_name+'_sd' + str(seed) + '_ancthres' + str(anchor_threshold) + '_progthres' + str(progress_threshold)
    if separate_testing:
        append += 'separate_testing'
    append += '_cont'
    learning_curves = []
    diretory = 'data/trained/walker3d/'+append

    if not os.path.exists(diretory):
        os.makedirs(diretory)
        os.makedirs(diretory + '/policies')

    env = normalize(GymEnv(env_name, record_log=False, record_video=False))
    dartenv = env._wrapped_env.env.env
    if env._wrapped_env.monitoring:
        dartenv = dartenv.env

    np.random.seed(seed)
    random.seed(seed)

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

    algo = TRPO_Symmetry(  # _MultiTask(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=env.horizon,
        n_itr=5,

        discount=0.99,
        step_size=0.02,
        gae_lambda=0.97,

        observation_permutation=np.array(
            [0.0001, -1, 2, -3, -4, -5, -6, 7, 14, -15, -16, 17, 18, -19, 8, -9, -10, 11, 12, -13,
             20, 21, -22, 23, -24, -25, -26, -27, 28, 35, -36, -37, 38, 39, -40, 29, -30, -31, 32, 33, -34, 42, 41]),
        action_permutation=np.array([-0.0001, -1, 2, 9, -10, -11, 12, 13, -14, 3, -4, -5, 6, 7, -8]),

        sym_loss_weight=1.0,
        whole_paths=False,
    )
    algo.init_opt()

    from rllab.sampler import parallel_sampler

    parallel_sampler.initialize(n_parallel=num_parallel)
    parallel_sampler.set_seed(seed)

    algo.start_worker()

    curriculum_evolution = []

    parallel_sampler.update_env_params({'anchor_kp':ref_curriculum})
    ref_score = evaluate_policy(env, ref_policy, 20)
    reference_score = ref_score * progress_threshold
    reference_anchor_score = ref_score * anchor_threshold
    parallel_sampler.update_env_params({'anchor_kp':init_curriculum})

    learning_curve = []
    for i in range(total_iterations):
        print('------ Iter ', i, ' in Training --------')
        paths = algo.sampler.obtain_samples(0)
        candidates, scores = get_curriculum_estimations(paths)
        print('Reference score: ', reference_score, reference_anchor_score)
        if not separate_testing:
            current_candidate = None
            current_min_dist = 10000
            if scores[0] > reference_anchor_score:
                for j in range(1, len(candidates)):
                    if scores[j] > reference_score and np.linalg.norm(candidates[j]) < current_min_dist:
                        current_candidate = candidates[j]
            if current_candidate is not None:
                parallel_sampler.update_env_params({'anchor_kp':current_candidate})
                curriculum_evolution.append(current_candidate)
                print('Current curriculum: ', current_candidate)
        else:
            if scores[0] > reference_anchor_score:
                directions = [np.array([-1, 0]), np.array([0, -1]), -candidates[0]/np.linalg.norm(candidates[0])]
                int_d1 = directions[0] + directions[2]
                int_d2 = directions[1] + directions[2]
                directions.append(int_d1/np.linalg.norm(int_d1))
                directions.append(int_d2/np.linalg.norm(int_d2))
                candidate_next_anchors = []
                closest_candidate = None
                for direction in directions:
                    found_point = binary_search_curriculum(env, policy, candidates[0], direction, reference_score, 6)
                    candidate_next_anchors.append(found_point)
                    if closest_candidate is None:
                        closest_candidate = np.copy(found_point)
                    elif np.linalg.norm(closest_candidate) > np.linalg.norm(found_point):
                        closest_candidate = np.copy(found_point)
                parallel_sampler.update_env_params({'anchor_kp': closest_candidate})
                curriculum_evolution.append(closest_candidate)
                print('Candidate points: ', candidate_next_anchors)
                print('Current curriculum: ', closest_candidate)

        samples_data = algo.sampler.process_samples(0, paths)
        opt_data = algo.optimize_policy(0, samples_data)
        print(dict(logger._tabular)['AverageReturn'])
        if i % 10 == 0:
            joblib.dump(policy, diretory + '/policy_'+str(i)+'.pkl', compress=True)
        if len(curriculum_evolution) > 0:
            plt.figure()
            curriculum_evolution_np = np.array(curriculum_evolution)
            plt.plot(curriculum_evolution_np[:,0], curriculum_evolution_np[:,1])
            plt.savefig(diretory + '/curriculum_evolution.png')
            np.savetxt(curriculum_evolution_np, diretory + '/curriculum_evolution.txt')
        learning_curve.append(dict(logger._tabular)['AverageReturn'])
        plt.figure()
        plt.plot(learning_curve)
        plt.savefig(diretory + '/learning_curve.png')

    algo.shutdown_worker()
    joblib.dump(policy, diretory + '/policy.pkl', compress=True)

    plt.close('all')

    print(diretory)
