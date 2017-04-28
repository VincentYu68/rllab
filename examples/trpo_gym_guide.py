from rllab.algos.trpo_guide import TRPOGuide
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_aux_policy import GaussianMLPAuxPolicy

import joblib

def run_task(*_):
    env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))

    policy = GaussianMLPAuxPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100,50,25),
        aux_pred_step = 1,
        aux_pred_dim = env.action_space.shape[0],
        skip_last=0,
        copy_output=True,
    )

    guidepolicy = joblib.load('data/gp/policy_fric0.pkl')
    #previouspolicy = joblib.load('data/trained/hopper_2d_adjust.pkl')
    #policy.set_param_values(previouspolicy.get_param_values(trainable=True), trainable=True)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPOGuide(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=100000,
        max_path_length=env.horizon,
        n_itr=1000,
        discount=0.995,
        step_size=0.01,
        epopt_epsilon = 1.0,
        epopt_after_iter = 0,
        gae_lambda=0.97,
        guiding_policies=[guidepolicy],
        guiding_policy_mps=[[ 0.        ,  0.17809725]],
        guiding_policy_weight=0.1,
        guiding_policy_pool_size=100000,
        guiding_policy_sample_size=20000,
        guiding_policy_batch_sizes=[15000],
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel= 8,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    exp_name='hopper_cap_frictorso_guide',
    # plot=True,
)
