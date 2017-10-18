from rllab.algos.trpo import TRPO
from rllab.algos.trpo_sym import TRPO_Symmetry
from rllab.algos.trpo_mpsel import TRPOMPSel
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_rbf_policy import GaussianRBFPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

import joblib
import numpy as np
import random

def run_task(*_):
    env = normalize(GymEnv("DartHumanWalker-v1", record_log=False, record_video=False))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(128,64),

        net_mode=0,
    )
    #policy = joblib.load('data/local/experiment/humanwalker_symmetry1_sd11_1alivebonus_2velrew_targetvelocity1_15frameskip_5en1absenergypenalty_spd20002000/policy.pkl')

    # increase policy std a bit for exploration
    #policy.get_params()[-1].set_value(policy.get_params()[-1].get_value() + 0.5)

    print('trainable parameter size: ', policy.get_param_values(trainable=True).shape)

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)


    algo = TRPO_Symmetry(
        env=env,
        policy=policy,
        baseline=baseline,

        batch_size=50000,

        max_path_length=env.horizon,
        n_itr=1000,

        discount=0.99,
        step_size=0.02,
        gae_lambda=0.97,
        observation_permutation=np.array([0.0001,-1,2,-3,-4, -11,12,-13,14,15,16, -5,6,-7,8,9,10, -17,18, -19, -24,25,-26,27, -20,21,-22,23,\
                                          28,29,-30,31,-32,-33, -40,41,-42,43,44,45, -34,35,-36,37,38,39, -46,47, -48, -53,54,-55,56, -49,50,-51,52, 58,57]),
        action_permutation=np.array([-6,7,-8, 9, 10,11,  -0.001,1,-2, 3, 4,5, -12,13, -14, -19,20,-21,22, -15,16,-17,18]),

        sym_loss_weight=1.0,
        action_reg_weight=0.0,
        whole_paths=False,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=4,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used

    seed=11,
    exp_name='humanwalker_symmetry1_sd11_25alivebonus_2velrew_targetvelocity1_15frameskip_3en1absenergypenalty_spd2000_200_bodyspd',

    # plot=True
)
