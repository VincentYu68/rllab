from rllab.algos.trpo import TRPO
from rllab.algos.trpo_mpsel import TRPOMPSel
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import joblib
import sys
import numpy as np
import copy
from rllab.misc import ext
from rllab.misc.ext import sliced_fun
import matplotlib.pyplot as plt

def get_gradient(algo, samples_data):
    all_input_values = tuple(ext.extract(
        samples_data,
        "observations", "actions", "advantages"
    ))
    agent_infos = samples_data["agent_infos"]
    state_info_list = [agent_infos[k] for k in algo.policy.state_info_keys]
    dist_info_list = [agent_infos[k] for k in algo.policy.distribution.dist_info_keys]
    all_input_values += tuple(state_info_list) + tuple(dist_info_list)

    flat_g = sliced_fun(algo.optimizer._opt_fun["f_grad"], 1)(
        tuple(all_input_values), tuple())

    return flat_g

def average_error(env, policy, batch_size, gt_gradient):
    np.random.seed(0)

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

    init_param = policy.get_param_values()

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length= env.horizon,
        n_itr=5,

        discount=0.995,
        step_size=0.01,
        gae_lambda=0.97,
    )

    gradients_vanilla = []
    gradients_randwalk = []

    gradient_error_vanilla = []
    gradient_error_randwalk = []

    env.wrapped_env.env.env.perturb_MP = True
    algo.start_worker()
    algo.init_opt()
    for i in range(20):
        policy.set_param_values(init_param)  # reset the policy parameters
        paths = algo.sampler.obtain_samples(0)
        samples_data = algo.sampler.process_samples(0, paths)
        samples_data = algo.sampler.process_samples(0, paths)
        grad = get_gradient(algo, samples_data)

        gradients_randwalk.append(grad)

        gradient_error_randwalk.append(np.linalg.norm(grad - gt_gradient))

    algo.shutdown_worker()

    env.wrapped_env.env.env.perturb_MP = False
    algo.start_worker()
    algo.init_opt()
    for i in range(20):
        policy.set_param_values(init_param)  # reset the policy parameters
        paths = algo.sampler.obtain_samples(0)
        samples_data = algo.sampler.process_samples(0, paths)
        samples_data = algo.sampler.process_samples(0, paths)
        grad = get_gradient(algo, samples_data)

        gradients_vanilla.append(grad)

        gradient_error_vanilla.append(np.linalg.norm(grad - gt_gradient))

    algo.shutdown_worker()

    print(np.std(gradients_vanilla, axis=0).shape)
    print(np.linalg.norm(np.mean(gradients_vanilla, axis=0)), np.mean(np.std(gradients_vanilla, axis=0)))
    print(np.mean(gradient_error_vanilla))

    print('randwalk')
    print(np.linalg.norm(np.mean(gradients_randwalk, axis=0)), np.mean(np.std(gradients_randwalk, axis=0)))
    print(np.mean(gradient_error_randwalk))

    return np.mean(gradient_error_vanilla), np.mean(gradient_error_randwalk)


if __name__ == '__main__':
    env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(10, 5),
        # append_dim=2,
        net_mode=0,
    )

    policy = joblib.load(
        'data/local/experiment/hopper_footstrength_rest1_sd4_boundedrandwalk_2000finish/policy_1500.pkl')

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

    init_param = policy.get_param_values()

    ###### get baseline gradient ###################################
    '''algobase = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2000000,
        max_path_length=env.horizon,
        n_itr=5,

        discount=0.995,
        step_size=0.01,
        gae_lambda=0.97,
    )

    from rllab.sampler import parallel_sampler

    parallel_sampler.initialize(n_parallel=4)

    env.wrapped_env.env.env.perturb_MP = False
    algobase.start_worker()
    algobase.init_opt()
    for i in range(1):
        policy.set_param_values(init_param)  # reset the policy parameters
        paths = algobase.sampler.obtain_samples(0)
        samples_data = algobase.sampler.process_samples(0, paths)
        samples_data = algobase.sampler.process_samples(0, paths)
        grad = get_gradient(algobase, samples_data)

        gt_gradient = grad

    algobase.shutdown_worker()
    joblib.dump(gt_gradient, 'data/trained/baseline_gradient.pkl', compress=True)'''
    gt_gradient = joblib.load('data/trained/baseline_gradient.pkl')
    #########################################################################

    vanilla_errors = []
    randwalk_errors = []
    batch_list = [1000, 5000, 10000, 30000, 50000, 100000, 150000]
    for batch in batch_list:
        ve, re = average_error(env, policy, batch, gt_gradient)
        vanilla_errors.append(ve)
        randwalk_errors.append(re)

    joblib.dump([vanilla_errors, randwalk_errors], 'data/trained/vere.pkl', compress=True)

    plt.plot(batch_list, vanilla_errors,'r-')
    plt.plot(batch_list, randwalk_errors,'g-')
    plt.show()










