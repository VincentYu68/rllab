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

import joblib
import numpy as np
import random

def run_task(*_):
    env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))

    mp_dim = 1
    #policy_pre = joblib.load('data/trained/gradient_temp/backpack_slope_sd7_3seg_vanillagradient_unweighted_1200start/policy_cont.pkl')
    split_dim = 0
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(2,),

        net_mode=0,
    )

    init_policy = joblib.load('data/local/experiment/hopper_torso0110_sd3_additionaldim_threetask/policy_0.pkl')
    masks = []
    params = init_policy.get_params()
    for k in range(len(params) - 1):
        masks.append(np.zeros(params[k].get_value().shape))
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(2,),
        # append_dim=2,
        net_mode=8,
        split_num=2,
        split_masks=masks,
        split_init_net=init_policy,
    )

    #policy = joblib.load('data/local/experiment/hopper_torso0110_sd3_additionaldim_threetask/policy_0.pkl')
    print('trainable parameter size: ', policy.get_param_values(trainable=True).shape)


    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=2)

    #policy = params['policy']
    #baseline = params['baseline']

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,

        batch_size=30000,
        max_path_length=env.horizon,
        n_itr=100,

        discount=0.995,
        step_size=0.01,
        gae_lambda=0.97,
        #mp_dim = mp_dim,
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
    seed=3,
    exp_name='hopper_torso0110_sd3_additionaldim_twotask_splitpolicy_addbaseline_2',

    # plot=True,
)
