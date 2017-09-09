__author__ = 'yuwenhao'

from  rllab.uposi.policy_split_rl_evaluation import *
import joblib
import numpy as np


def get_flat_gradient(algo, samples_data):
    all_input_values = tuple(ext.extract(
        samples_data,
        "observations", "actions", "advantages"
    ))
    agent_infos = samples_data["agent_infos"]
    state_info_list = [agent_infos[k] for k in algo.policy.state_info_keys]
    dist_info_list = [agent_infos[k] for k in algo.policy.distribution.dist_info_keys]
    all_input_values += tuple(state_info_list) + tuple(dist_info_list)

    grad = sliced_fun(algo.optimizer._opt_fun["f_grad"], 1)(
        tuple(all_input_values), tuple())

    return grad

if __name__ == '__main__':
    num_parallel = 4

    directory = 'data/trained/gradient_temp/rl_split_reacher_3modelsexp1_alivepenalty_tasksplit_taskinput_6432net_sd1_vanbaseline_splitstd_accumgrad_40k_0_1_unweighted_accumulate_gradient/'
    policy = joblib.load(directory + '/final_policy_0.0.pkl')
    test_trajs = 50
    pathlength = 1000

    env_name = "DartReacher3d-v1"

    env = normalize(GymEnv(env_name, record_log=False, record_video=False))

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

    algo = TRPO(  # _MultiTask(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=40000,
        max_path_length=pathlength,
        n_itr=5,

        discount=0.995,
        step_size=0.02,
        gae_lambda=0.97,
        whole_paths=False,
    )
    algo.init_opt()

    from rllab.sampler import parallel_sampler

    parallel_sampler.initialize(n_parallel=num_parallel)
    parallel_sampler.set_seed(0)

    algo.start_worker()

    paths = algo.sampler.obtain_samples(0)
    algo.sampler.process_samples(0, paths)
    samples_data_ori = algo.sampler.process_samples(0, paths)

    print('Original return', dict(logger._tabular)['AverageReturn'])

    sample_grads = []
    sample_grad_van = []
    for i in range(100):
        samples_data = {}
        indices = np.arange(len(samples_data_ori['observations']))
        np.random.shuffle(indices)
        samples_data["observations"] = samples_data_ori["observations"][indices[0:35000]]
        samples_data["actions"] = samples_data_ori["actions"][indices[0:35000]]
        samples_data["rewards"] = samples_data_ori["rewards"][indices[0:35000]]
        samples_data["advantages"] = samples_data_ori["advantages"][indices[0:35000]]
        samples_data["agent_infos"] = {}
        samples_data["agent_infos"]["log_std"] = samples_data_ori["agent_infos"]["log_std"][
            indices[0:35000]]
        samples_data["agent_infos"]["mean"] = samples_data_ori["agent_infos"]["mean"][
            indices[0:35000]]
        grad = get_flat_gradient(algo, samples_data)
        sample_grads.append(grad)
        sample_grad_van.append(get_gradient(algo, samples_data, False))

    grad_variance = np.var(sample_grads, axis=0)

    sorted_grad_var = np.copy(grad_variance)
    sorted_grad_var.sort()

    mat_grads = []
    for k in range(len(sample_grad_van[0])):
        one_grad = []
        for i in range(len(sample_grad_van)):
            one_grad.append(sample_grad_van[i][k])
        mat_grads.append(np.var(one_grad, axis=0))

    for j in range(len(mat_grads)):
        plt.figure()
        plt.title(policy.get_params()[j].name)
        if len(mat_grads[j].shape) == 2:
            plt.imshow(mat_grads[j])
            plt.colorbar()
        elif len(mat_grads[j].shape) == 1:
            plt.plot(mat_grads[j])
        plt.savefig(directory + policy.get_params()[
            j].name + '_grad_variance.png')

    max_var = np.max(grad_variance)

    perturbed_performances = []
    old_params = np.copy(policy.get_param_values())
    for i in range(20):
        start = sorted_grad_var[int(0.05 * i * len(grad_variance))]
        end = sorted_grad_var[int(0.05 * (i+1) * len(grad_variance))-1]
        new_params = np.copy(old_params)
        num = 0
        for j in range(len(new_params)):
            if grad_variance[j] >= start and grad_variance[j] < end:
                new_params[j] += np.random.normal(0, np.sqrt(max_var))
                num += 1
        policy.set_param_values(new_params)
        print('NUM: ', num, max_var)
        paths = algo.sampler.obtain_samples(0)
        algo.sampler.process_samples(0, paths, False)
        perturbed_performances.append(dict(logger._tabular)['AverageReturn'])

    plt.figure()
    plt.plot(perturbed_performances)
    plt.savefig(directory+'/perturbed_performances.png')

    algo.shutdown_worker()
















