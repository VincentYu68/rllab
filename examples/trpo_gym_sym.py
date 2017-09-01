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
    env = normalize(GymEnv("DartWalker2d-v1"))#, record_log=False, record_video=False))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32,32),

        net_mode=0,
    )

    print('trainable parameter size: ', policy.get_param_values(trainable=True).shape)

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)


    algo = TRPO_Symmetry(
        env=env,
        policy=policy,
        baseline=baseline,

        batch_size=30000,

        max_path_length=env.horizon,
        n_itr=300,

        discount=0.995,
        step_size=0.01,
        gae_lambda=0.97,
        observation_permutation=np.array([0,1, 5,6,7, 2,3,4, 8,9,10, 14,15,16, 11,12,13]),
        action_permutation=np.array([3,4,5,0,1,2]),
        sym_loss_weight=0.0,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=2,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=3,
    exp_name='walker2d_vanilla',

    # plot=True,
)
