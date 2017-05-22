from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_rbf_policy import GaussianRBFPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

import joblib

def run_task(*_):
    env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 50, 25),
    )
    '''policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
    )'''

    policy = joblib.load('data/local/experiment/hopper_footmass_0005/policy.pkl')

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=75000,
        max_path_length=env.horizon,
        n_itr=500,

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
    seed=3,
    exp_name='hopper_footmass_0005_2',
    # plot=True,
)
