from rllab.algos.trpo import TRPO
from rllab.algos.trpo_mpsel import TRPOMPSel
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_rbf_policy import GaussianRBFPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

import joblib
import numpy as np
import random

def run_task(*_):
    env = normalize(GymEnv("DartHopperRSS-v1", record_log=False, record_video=False))

    mp_dim = 1
    #policy_pre = joblib.load('data/trained/gradient_temp/rl_split_hopper_3models_taskinput_6432net_sd4_splitstd_maskedgrad_specbaseline_40k_70_30_unweighted_accumulate_gradient/final_policy_0.1.pkl')
    split_dim = 0
    policy = GaussianGRUPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64,),
    )

    print('trainable parameter size: ', policy.get_param_values(trainable=True).shape)

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

    #policy = params['policy']
    #baseline = params['baseline']

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,

        batch_size=75000,

        max_path_length=env.horizon,
        n_itr=800,

        discount=0.99,
        step_size=0.01,
        gae_lambda=0.98,
        #mp_dim = mp_dim,
        #epopt_epsilon = 1.0,
        #epopt_after_iter = 0,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
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
    seed=12,
    exp_name='hopper_rss_e2e_friction0310_energy003_ab2',

    # plot=True,
)
