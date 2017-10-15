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
    env = normalize(GymEnv("DartWalker3d-v1", record_log=False, record_video=False))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(128,64),

        net_mode=0,
    )
    #policy = joblib.load('data/local/experiment/walker3d_symmetry1_sd4_1alivebonus_2velrew_targetvelocity2_15frameskip_2en1absenergypenalty_spd2k2k/policy.pkl')

    # increase policy std a bit for exploration
    #policy.get_params()[-1].set_value(policy.get_params()[-1].get_value() + 0.5)

    print('trainable parameter size: ', policy.get_param_values(trainable=True).shape)

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)


    algo = TRPO_Symmetry(
        env=env,
        policy=policy,
        baseline=baseline,

        batch_size=6000,

        max_path_length=env.horizon,
        n_itr=500,

        discount=0.99,
        step_size=0.02,
        gae_lambda=0.97,
        observation_permutation=np.array([0.0001,-1, 2,-3,-4, -5,-6,7, 14,-15,-16, 17, 18,-19, 8,-9,-10, 11, 12,-13,\
                                          20,21,-22, 23,-24,-25, -26,-27,28, 35,-36,-37, 38, 39,-40, 29,-30,-31, 32, 33,-34, 42, 41]),
        #observation_permutation=np.array([0.0001, 1, 5,6,7, 2,3,4, 8,9,10, 14,15,16, 11,12,13]),
        #action_permutation=np.array([3,4,5, 0.00001,1,2]),
        action_permutation=np.array([-0.0001, -1, 2, 9, -10, -11, 12, 13, -14, 3, -4, -5, 6, 7, -8]),

        sym_loss_weight=1.0,
        whole_paths=False,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=8,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used

    seed=11,
    exp_name='walker3d_symmetry1_sd11_1alivebonus_2velrew_targetvelocity1_15frameskip_5en1absenergypenalty_spd20002000',

    # plot=True
)
