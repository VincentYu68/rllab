from rllab.algos.trpo import TRPO
from rllab.algos.trpo_mpsel import TRPOMPSel
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import joblib
import sys
import numpy as np
import copy
from rllab.misc import ext
from rllab.misc.ext import sliced_fun
import matplotlib.pyplot as plt

def get_reward(env, policy, traj_num = 10):
    reward = 0

    for i in range(traj_num):
        obs = env.reset()
        done = False
        while not done:
            act, act_info = policy.get_action(obs)
            obs, rew, done, _ = env.step(act_info['mean'])
            reward += rew

    reward /= traj_num

    return reward


if __name__ == '__main__':
    env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(10, 5),
        # append_dim=2,
        net_mode=0,
    )

    policy = joblib.load(
        'data/local/experiment/hopper_restfoot_seed8_50ksamp_2000finish/policy.pkl')

    values = []
    for pm in policy._mean_network.get_params(trainable=True):
        values += np.abs(pm.get_value()).flatten().tolist()
        print(pm.name, np.max(np.abs(pm.get_value())), np.min(np.abs(pm.get_value())))

    print(np.max(values), np.min(values))

    rewards = []

    total_param = len(values)
    values.sort()
    perc = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
    threasholds = [values[-int(0.01*total_param)], values[-int(0.1*total_param)], values[-int(0.15*total_param)], values[-int(0.2*total_param)], values[-int(0.25*total_param)], values[-int(0.3*total_param)]]

    for threashold in threasholds:
        for pm in policy._mean_network.get_params(trainable=True):
            weight = pm.get_value()
            weight[np.abs(weight) > threashold] += np.random.uniform(-0.01, 0.01)
            pm.set_value(weight)

        rewards.append(get_reward(env, policy, 10))

    print(threasholds)
    plt.plot(perc, rewards)
    plt.show()






