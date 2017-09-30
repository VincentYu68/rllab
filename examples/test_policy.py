__author__ = 'yuwenhao'

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_hrl_prop_policy import GaussianHMLPPropPolicy
import gym
import sys, os, time

import joblib
import numpy as np

import matplotlib.pyplot as plt
from gym import wrappers

np.random.seed(15)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartWalker3dRestricted-v1')

    if hasattr(env.env, 'disableViewer'):
        env.env.disableViewer = False
        '''if hasattr(env.env, 'resample_MP'):
        env.env.resample_MP = False'''

    record = False
    if len(sys.argv) > 3:
        record = int(sys.argv[3]) == 1
    if record:
        env_wrapper = wrappers.Monitor(env, 'data/videos/', force=True)
    else:
        env_wrapper = env

    #dyn_models = joblib.load('data/trained/dyn_models.pkl')
    #env.env.dyn_models = dyn_models
    #env.env.dyn_model_id = 0

    '''if hasattr(env.env, 'param_manager'):
        #env.env.param_manager.resample_parameters()
        #env.env.param_manager.set_simulator_parameters([0.7, 0.45])
        #env.env.resample_MP = True
        print('Model Parameters: ', env.env.param_manager.get_simulator_parameters())'''


    policy = None
    if len(sys.argv) > 2:
        policy = joblib.load(sys.argv[2])

    o = env_wrapper.reset()

    rew = 0

    actions = []

    traj = 1
    ct = 0
    action_pen = []
    while ct < traj:
        if policy is not None:
            a, ainfo = policy.get_action(o)
            act = a#ainfo['mean']
        else:
            act = env.action_space.sample()
        actions.append(act)
        if hasattr(policy, '_lowlevelnetwork'):
            lowa = policy.lowlevel_action(o, act)
            o, r, d, env_info = env_wrapper.step(lowa)
        else:
            o, r, d, env_info = env_wrapper.step(act)

        if 'action_pen' in env_info:
            action_pen.append(env_info['action_pen'])

        rew += r

        env_wrapper.render()

        #time.sleep(0.1)

        if d:
            ct += 1
            print('reward: ', rew)
            o=env_wrapper.reset()
            #break
    print('avg rew ', rew / traj)


    #plt.plot(thigh_torque_1)
    #plt.plot(thigh_torque_2)
    #plt.show()
    if len(actions[0]) < 20 and len(actions[0]) > 12:
        rendergroup = [[0,1,2], [3,4,5, 9,10,11], [6,12], [7,8, 12,13]]
        for rg in rendergroup:
            plt.figure()
            for i in rg:
                plt.plot(np.array(actions)[:, i])
        plt.figure()
        plt.plot(action_pen)
        plt.show()
