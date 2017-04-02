from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_rbf_policy import GaussianRBFPolicy
from rllab.policies.gaussian_hmlp_policy import GaussianHMLPPolicy
from rllab.policies.gaussian_hlc_policy import GaussianHLCPolicy

import numpy as np

def run_task(*_):
    env = normalize(GymEnv("DartWalker3d-v1", record_video=False))

    policy_int = GaussianHMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64,16),
        subnet_split1=[5, 6, 7, 8, 9, 21, 22, 23, 24, 25],
        subnet_split2=[10, 11, 12, 13, 14, 26, 27, 28, 29, 30],
        sub_out_dim=6,
        option_dim=4,
    )

    policy_sep = GaussianHLCPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64,16),
        subnet_split1=[5, 6, 7, 8, 9, 21, 22, 23, 24, 25],
        subnet_split2=[10, 11, 12, 13, 14, 26, 27, 28, 29, 30],
        sub_out_dim=6,
        option_dim=4,
    )


    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo1 = TRPO(
        env=env,
        policy=policy_int,
        baseline=baseline,
        batch_size=500,
        max_path_length=env.horizon,
        n_itr=2,
        discount=0.99,
        step_size=0.01,
        epopt_epsilon = 1.0,
        epopt_after_iter = 0,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    algo2 = TRPO(
        env=env,
        policy=policy_sep,
        baseline=baseline,
        batch_size=500,
        max_path_length=env.horizon,
        n_itr=2,
        discount=0.99,
        step_size=0.01,
        epopt_epsilon = 1.0,
        epopt_after_iter = 0,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    # copy parameter from integrated controller to separate controller
    def int2sep():
        # sync the weights
        hrl_pol_param = policy_int._mean_network.get_params()
        hlc_param = policy_sep._mean_network.get_params()
        llc_param = policy_sep._lowlevelnetwork.get_params()

        for param in hlc_param:
            for hrl_param in hrl_pol_param:
                if param.name == hrl_param.name:
                    param.set_value(hrl_param.get_value(borrow=True))

        for param in llc_param:
            for hrl_param in hrl_pol_param:
                if param.name == hrl_param.name:
                    param.set_value(hrl_param.get_value(borrow=True))

    # copy parameter from separate controller to integrated controller
    def sep2int():
        hrl_pol_param = policy_int._mean_network.get_params()
        hlc_param = policy_sep._mean_network.get_params()
        llc_param = policy_sep._lowlevelnetwork.get_params()
        for param in hrl_pol_param:
            for hrl_param in hlc_param:
                if param.name == hrl_param.name:
                    param.set_value(hrl_param.get_value(borrow=True))

        for param in hrl_pol_param:
            for hrl_param in llc_param:
                if param.name == hrl_param.name:
                    param.set_value(hrl_param.get_value(borrow=True))

    for i in range(100):
        algo1.current_itr=0
        algo2.current_itr=0
        algo2.train(continue_learning=(i > 0))
        sep2int()

        algo1.train(continue_learning=(i > 0))
        int2sep()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=0,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    exp_prefix='Walker3d_async_hrl'
    # plot=True
)
