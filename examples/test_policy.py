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

    #env_wrapper = wrappers.Monitor(env, 'data/videos/', force=True, video_callable=False)
    env_wrapper = env

    if hasattr(env.env, 'disableViewer'):
        env.env.disableViewer = False
    if hasattr(env.env, 'resample_MP'):
        env.env.resample_MP = False

    dyn_models = joblib.load('data/trained/dyn_models.pkl')
    env.env.dyn_models = dyn_models
    env.env.dyn_model_id = 0

    if hasattr(env.env, 'param_manager'):
        #env.env.param_manager.resample_parameters()
        env.env.param_manager.set_simulator_parameters([0.7, 0.45])
        #env.env.resample_MP = True
        print('Model Parameters: ', env.env.param_manager.get_simulator_parameters())


    policy = None
    if len(sys.argv) > 2:
        policy = joblib.load(sys.argv[2])

    o = env_wrapper.reset()

    rew = 0

    for i in range(1500):
        if policy is not None:
            a, ainfo = policy.get_action(o)
            act = a#ainfo['mean']
        else:
            act = env.action_space.sample()

        if hasattr(policy, '_lowlevelnetwork'):
            lowa = policy.lowlevel_action(o, act)
            o, r, d, env_info = env_wrapper.step(lowa)
        else:
            o, r, d, env_info = env_wrapper.step(act)

        rew += r

        env_wrapper.render()

        #time.sleep(0.1)

        if d:
            print('reward: ', rew)
            o=env_wrapper.reset()
            #break

    #plt.plot(thigh_torque_1)
    #plt.plot(thigh_torque_2)
    #plt.show()
