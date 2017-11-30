from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv

stub(globals())

env = TfEnv(normalize(GymEnv("DartHopperRSS-v1", record_log=False, record_video=False)))

policy = GaussianLSTMPolicy(
    name="policy",
    env_spec=env.spec,
    lstm_layer_cls=L.TfBasicLSTMLayer,
    # gru_layer_cls=L.GRULayer,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=75000,
    max_path_length=env.horizon,
    n_itr=800,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)
run_experiment_lite(
    algo.train(),
    n_parallel=4,
    seed=1,
    exp_name='hopper_rss_e2e_friction0310_energy003_ab2',
)
