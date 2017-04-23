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
import joblib

def run_task(*_):
    env = normalize(GymEnv("DartWalker2d-v1", record_video=False))

    policy_sep = GaussianHLCPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64,32),
        sub_out_dim=3,
        option_dim=2,
        #init_std=0.1,
    )

    policy_sep = joblib.load('data/local/experiment/Walker2d_hlc_2/policy_0.pkl')

    '''# copy parameter from integrated controller to separate controller
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
                param.set_value(hrl_param.get_value(borrow=True))'''


    baseline = LinearFeatureBaseline(env_spec=env.spec)

    '''o = np.random.random(17)*0
    o[0]=1.25
    a, ainfo = policy_int.get_action(o)
    a2, a2info = policy_sep.get_action(o)
    action1 = ainfo['mean']
    action2 = policy_sep.lowlevel_action(o, a2)
    print(action1)
    print(action2)
    abc'''

    algo2 = TRPO(
        env=env,
        policy=policy_sep,
        baseline=baseline,
        batch_size=15000,
        max_path_length=env.horizon,
        n_itr=200,
        discount=0.99,
        step_size=0.01,
        epopt_epsilon = 1.0,
        epopt_after_iter = 0,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo2.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=2,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    exp_name='Walker2d_hlc_cont',
    # plot=True
)
