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

def run_task(*_):
    env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))

    mp_dim = 1
    #policy_pre = joblib.load('data/trained/gradient_temp/backpack_slope_sd7_3seg_vanillagradient_unweighted_1200start/policy_cont.pkl')
    split_dim = 0

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(128, 64),
        #append_dim=2,

        net_mode=0,

        mp_dim=mp_dim,
        mp_sel_hid_dim=12,
        mp_sel_num=split_dim,
        #wc_net_path='data/trained/2d_weightconverter.pkl',
        learn_segment = False,
        split_layer=[2],
        split_num=split_dim,
        #split_units=joblib.load('data/trained/gradient_temp/backpack_slope_sd7_3seg_vanillagradient_unweighted_1200start/split_scheme_backpack_slope_sd7_3seg_vanillagradient_unweighted_1200start_orth_0.5.pkl'),
        #split_init_net=policy_pre,
    )

    #policy = joblib.load('data/local/experiment/walker3d-2d_cont/policy.pkl')

    print('trainable parameter size: ', policy.get_param_values(trainable=True).shape)
    '''policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
    )'''

    '''policy_prev = joblib.load('data/trained/policy_2d_footstrength_sd4_1000.pkl')


    params = policy_prev.get_params(trainable=True)
    for paramid in range(len(params)):
        if paramid == 0:
            n_class = env._wrapped_env.env.env.sampling_selector.n_class
            param_value = params[paramid].get_value(borrow=True)
            #obs_dim = param_value.shape[0] - 2  # hard-coded!!!
            #param_value = np.vstack([param_value[0:obs_dim, :], param_value[obs_dim:, :].tolist() * n_class])
            param_value = np.vstack([param_value] * n_class)
            policy.get_params(trainable=True)[paramid].set_value(np.array(param_value, dtype=np.float32))
        else:
            policy.get_params(trainable=True)[paramid].set_value(params[paramid].get_value(borrow=True))
    '''

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

    #policy = params['policy']
    #baseline = params['baseline']

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,

        batch_size=20000,
        max_path_length=env.horizon,
        n_itr=200,


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
    seed=1,
    exp_name='hopper_torso01_20ktest_sd1_mixbaseline',

    # plot=True,
)
