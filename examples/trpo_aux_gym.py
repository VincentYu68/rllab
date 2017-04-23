__author__ = 'yuwenhao'

from rllab.algos.trpo_aux import TRPOAux
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_rbf_policy import GaussianRBFPolicy
from rllab.policies.gaussian_mlp_aux_policy import GaussianMLPAuxPolicy
import lasagne.nonlinearities as NL

import joblib

def run_task(*_):
    env = normalize(GymEnv("DartWalker3d-v1", record_log=False, record_video=False))

    policy = GaussianMLPAuxPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 50, 25),
        aux_pred_step = 3,
        aux_pred_dim = 7,
    )

    #policy = joblib.load('data/local/experiment/walker_aux/policy.pkl')

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPOAux(
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
        aux_pred_step=3,
        aux_pred_dim = 7,
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
    exp_name='walker_7daux_seed3_pred',
    # plot=True,
)
