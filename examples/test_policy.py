__author__ = 'yuwenhao'

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_hmlp_policy import GaussianHMLPPolicy

import numpy as np

np.random.seed(13)

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

print(policy.get_params())

o = env.reset()

for i in range(1000):
    a, ainfo = policy.get_action(o)
    act = ainfo['mean']
    #print(act)
    o, rew, done, info = env.step(act)
    env.render()
    #if done:
    #    break