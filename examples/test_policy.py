__author__ = 'yuwenhao'

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_hrl_prop_policy import GaussianHMLPPropPolicy
import gym
import sys

import joblib
import numpy as np

import matplotlib.pyplot as plt

np.random.seed(13)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartWalker3dRestricted-v1')

    #if hasattr(env.env, 'disableViewer'):
    #    env.env.disableViewer = False
    if hasattr(env.env, 'resample_MP'):
        env.env.resample_MP = False

    env.env.param_manager.set_simulator_parameters([0.59])


    if len(sys.argv) > 2:
        policy = joblib.load(sys.argv[2])

    o = env.reset()

    rew = 0


    for i in range(1000):
        a, ainfo = policy.get_action(o)
        act = ainfo['mean']
        if hasattr(policy, '_lowlevelnetwork'):
            lowa = policy.lowlevel_action(o, act)
            o, r, d, env_info = env.step(lowa)
        else:
            o, r, d, env_info = env.step(act)

        rew += r

        #env.render()
        if d:
            print('reward: ', rew)
            break

    #plt.plot(thigh_torque_1)
    #plt.plot(thigh_torque_2)
    #plt.show()
