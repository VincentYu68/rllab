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

np.random.seed(1)

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
    vel_rew = []
    action_pen = []
    deviation_pen = []
    rew_seq = []
    com_z = []
    x_vel = []
    foot_contacts = []
    d=False
    init_p = env.env.robot_skeleton.q[0]
    while ct < traj:
        if policy is not None:
            a, ainfo = policy.get_action(o)
            act = ainfo['mean']
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
        if 'vel_rew' in env_info:
            vel_rew.append(env_info['vel_rew'])
        rew_seq.append(r)
        if 'deviation_pen' in env_info:
            deviation_pen.append(env_info['deviation_pen'])

        com_z.append(o[1])
        foot_contacts.append(o[-2:])

        rew += r

        env_wrapper.render()

        #time.sleep(0.1)
        if len(o) > 25:
            x_vel.append(env.env.robot_skeleton.dq[0])

        if d:
            ct += 1
            print('reward: ', rew)
            print('travelled dist: ', env.env.robot_skeleton.q[0] - init_p)
            o=env_wrapper.reset()
            init_p = env.env.robot_skeleton.q[0]
            #break
    print('avg rew ', rew / traj)

    if sys.argv[1] == 'DartWalker3d-v1':
        rendergroup = [[0,1,2], [3,4,5, 9,10,11], [6,12], [7,8, 12,13]]
        for rg in rendergroup:
            plt.figure()
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    if sys.argv[1] == 'DartHumanWalker-v1':
        rendergroup = [[0,1,2, 6,7,8], [3,9], [4,5,10,11], [12,13,14]]
        for rg in rendergroup:
            plt.figure()
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    plt.figure()
    plt.title('rewards')
    plt.plot(rew_seq, label='total rew')
    plt.plot(action_pen, label='action pen')
    plt.plot(vel_rew, label='vel rew')
    plt.plot(deviation_pen, label='dev pen')
    plt.legend()
    plt.figure()
    plt.title('com z')
    plt.plot(com_z)
    plt.figure()
    plt.title('x vel')
    plt.plot(x_vel)
    foot_contacts = np.array(foot_contacts)
    plt.figure()
    plt.title('foot contacts')
    plt.plot(1-foot_contacts[:, 0])
    plt.plot(1-foot_contacts[:, 1])
    plt.show()



