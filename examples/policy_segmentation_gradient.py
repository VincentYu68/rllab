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

    grad = sliced_fun(algo.optimizer._opt_fun["f_grads"], 1)(
        tuple(all_input_values), tuple())

    return grad


if __name__ == '__main__':
    env = normalize(GymEnv("DartCartPoleSwingUp-v1", record_log=False, record_video=False))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(10, 5),
        # append_dim=2,
        net_mode=0,
    )

    policy = joblib.load('data/local/experiment/cartpoleswingup_mass_seed15_rotreward_all_highsamp/policy.pkl')

    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

    init_param = policy.get_param_values()
    init_param_obj = copy.deepcopy(policy.get_params())

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=150000,
        max_path_length=env.horizon,
        n_itr=5,

        discount=0.995,
        step_size=0.01,
        gae_lambda=0.97,
    )

    from rllab.sampler import parallel_sampler

    parallel_sampler.initialize(n_parallel=4)

    env.wrapped_env.env.env.perturb_MP = False
    algo.start_worker()
    algo.init_opt()

    total_grad_left = []
    total_grad_right = []
    total_grad_both = []
    for i in range(50):
        init_param = policy.get_param_values()
        init_param_obj = copy.deepcopy(policy.get_params())

        #####   get data ###################
        policy.set_param_values(init_param)  # reset the policy parameters
        paths = algo.sampler.obtain_samples(0)

        path_left = []
        path_right = []
        for path in paths:
            mp = path['env_infos']['model_parameters'][-1]
            if mp[0] < 0.5:
                path_left.append(path)
            else:
                path_right.append(path)
        #####################################

        samples_data = algo.sampler.process_samples(0, path_left)
        samples_data = algo.sampler.process_samples(0, path_left)
        #grad_left = get_gradient(algo, samples_data)
        algo.optimize_policy(0, samples_data)
        grad_left = []
        for j in range(len(init_param_obj)):
            grad_left.append(policy.get_params()[j].get_value() - init_param_obj[j].get_value())

        samples_data = algo.sampler.process_samples(0, path_right)
        samples_data = algo.sampler.process_samples(0, path_right)
        #grad_right = get_gradient(algo, samples_data)
        policy.set_param_values(init_param)  # reset the policy parameters
        algo.optimize_policy(0, samples_data)
        grad_right = []
        for j in range(len(init_param_obj)):
            grad_right.append(policy.get_params()[j].get_value() - init_param_obj[j].get_value())

        # if not split
        samples_data = algo.sampler.process_samples(0, paths)
        samples_data = algo.sampler.process_samples(0, paths)
        # grad_left = get_gradient(algo, samples_data)
        policy.set_param_values(init_param)  # reset the policy parameters
        algo.optimize_policy(0, samples_data)
        grad_both = []
        for j in range(len(init_param_obj)):
            grad_both.append(policy.get_params()[j].get_value() - init_param_obj[j].get_value())


        total_grad_left.append(grad_left)
        total_grad_right.append(grad_right)
        total_grad_both.append(grad_both)

    algo.shutdown_worker()

    joblib.dump([total_grad_left, total_grad_right, total_grad_both], 'data/trained/gradient_temp/total_gradients.pkl', compress=True)

    #total_grad_left, total_grad_right, total_grad_both = joblib.load('data/trained/gradient_temp/total_gradients.pkl')

    split_counts = []
    for i in range(len(total_grad_left[0])):
        split_counts.append(np.zeros(total_grad_left[0][i].shape))
    for i in range(len(total_grad_left)):
        value_list = []
        for j in range(len(total_grad_left[i])):
            value_list += (np.abs(total_grad_right[i][j] - total_grad_left[i][j])/np.abs(total_grad_both[i][j])).flatten().tolist()
        value_list.sort()
        threshold = value_list[-int(0.3*(len(value_list)))]
        for j in range(len(total_grad_left[i])):
            split_indices = (np.abs(total_grad_right[i][j] - total_grad_left[i][j])/np.abs(total_grad_both[i][j])) > threshold
            split_counts[j][split_indices] += 1

    for j in range(len(split_counts)):
        plt.figure()
        plt.title(policy.get_params()[j].name)
        if len(split_counts[j].shape) == 2:
            plt.imshow(split_counts[j])
            plt.colorbar()
        elif len(split_counts[j].shape) == 1:
            plt.plot(split_counts[j])
        plt.savefig('data/trained/gradient_temp/' + policy.get_params()[j].name + '.png')
    plt.show()

    '''grad_discrepencies = []
    for j in range(len(policy.get_params())):
        grad_disc = np.abs(total_grad_left[0][j] - total_grad_right[0][j]) / np.abs(total_grad_both[0][j])
        for p in range(1, len(total_grad_left)):
            grad_disc += np.abs(total_grad_left[p][j] - total_grad_right[p][j]) / np.abs(total_grad_both[p][j])
        grad_discrepencies.append(grad_disc)

    for j in range(len(grad_discrepencies)):
        plt.figure()
        plt.title(policy.get_params()[j].name)
        if len(grad_discrepencies[j].shape) == 2:
            plt.imshow(grad_discrepencies[j])
            plt.colorbar()
        elif len(grad_discrepencies[j].shape) == 1:
            plt.plot(grad_discrepencies[j])
        plt.savefig('data/trained/gradient_temp/'+policy.get_params()[j].name+'.png')

    plt.show()'''










