from rllab.algos.trpo import TRPO
from rllab.algos.trpo_mpsel import TRPOMPSel
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.linear_feature_baseline_mc import LinearFeatureBaselineMultiClass
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_rbf_policy import GaussianRBFPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

import joblib
import numpy as np

def run_task(*_):
    env = normalize(GymEnv("DartWalker3dRestricted-v1"))#, record_log=False, record_video=False))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 50, 25),
        #append_dim=2,
        mp_dim=2,
        mp_sel_hid_dim=32,
        mp_sel_num=4,
    )
    '''policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
    )'''

    #policy = joblib.load('data/local/experiment/hopper_reststrength_seed6_cont_cont/policy.pkl')
    '''policy_prev = joblib.load('data/trained/policy_2d_restfoot_sd6_cont_cont.pkl')
    
    params = policy_prev.get_params(trainable=True)
    for paramid in range(len(params)):
        if paramid == 0:
            n_class = env._wrapped_env.env.env.sampling_selector.n_class
            param_value = params[paramid].get_value(borrow=True)
            param_value = np.vstack([param_value]*n_class)
            policy.get_params(trainable=True)[paramid].set_value(param_value)
        else:
            policy.get_params(trainable=True)[paramid].set_value(params[paramid].get_value(borrow=True))'''


    baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    #policy = params['policy']
    #baseline = params['baseline']

    algo = TRPOMPSel(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=150000,
        max_path_length=env.horizon,
        n_itr=1000,
        discount=0.995,
        step_size=0.01,
        gae_lambda=0.97,
        #epopt_epsilon = 1.0,
        #epopt_after_iter = 0,
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
    seed=6,
    exp_name='hopper_reststrength_seed6_mpsel_entpen',
    # plot=True,
)
