from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_rbf_policy import GaussianRBFPolicy
from rllab.policies.gaussian_hmlp_policy import GaussianHMLPPolicy
from rllab.policies.gaussian_hmlp_phase_policy import GaussianHMLPPhasePolicy

import numpy as np
import joblib

def run_task(*_):
    env = normalize(GymEnv("DartWalker3dRestricted-v1", record_log=False, record_video=False))

    policy = GaussianHMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32,32),
        subnet_split1=[8, 9, 10, 11, 12, 13, 29, 30, 31, 32, 33, 34],
        subnet_split2=[14, 15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 40],
        sub_out_dim=6,
        option_dim=2,
        hlc_output_dim=3,
    )

    #policy = joblib.load('data/local/experiment/Walker3d_waist_onlyconcatoption3/policy.pkl')

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        max_path_length=env.horizon,
        n_itr=500,
        discount=0.995,
        step_size=0.01,
        epopt_epsilon = 1.0,
        epopt_after_iter = 0,
        gae_lambda=0.97,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
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
    seed=1,
    exp_name='Walker3d_restricted_hrl'
    # plot=True
)
