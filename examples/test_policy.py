__author__ = 'yuwenhao'

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_hmlp_policy import GaussianHMLPPolicy
from rllab.policies.gaussian_hrl_prop_policy import GaussianHMLPPropPolicy

import joblib
import numpy as np

np.random.seed(1)

env = normalize(GymEnv("DartWalker2d-v1"))

policy = GaussianHMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(64,16),
    #subnet_split1=[5, 6, 7, 8, 9, 21, 22, 23, 24, 25],
    #subnet_split2=[10, 11, 12, 13, 14, 26, 27, 28, 29, 30],
    #sub_out_dim=6,
    #option_dim=4,
    sub_out_dim=3,
    option_dim=2,
)

#policy = joblib.load('data/local/Walker2d-alt-proprio/Walker2d_alt_proprio_2017_04_03_12_33_22_0001/policy_240.pkl')

o = env.reset()

print(policy.get_hidden_sig(o))

'''a, ainfo = policy.get_action(o)
print(ainfo['mean'])
policy.set_use_proprioception(False)
a, ainfo = policy.get_action(o)
print(ainfo['mean'])
policy.set_use_proprioception(True)
a, ainfo = policy.get_action(o)
print(ainfo['mean'])
'''
aa
rew = 0

for i in range(1000):
    a, ainfo = policy.get_action(o)
    act = a
    print(policy.get_option_layer_val(o))
    if hasattr(policy, '_lowlevelnetwork'):
        lowa = policy.lowlevel_action(o, act)
        o, r, d, env_info = env.step(lowa)
    else:
        o, r, d, env_info = env.step(act)

    rew += r

    env.render()
    if d:
        print('reward: ', rew)
        break