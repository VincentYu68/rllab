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
    env = normalize(GymEnv("DartHopper-v1", record_log=False, record_video=False))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(10, 5),
        # append_dim=2,
        net_mode=0,
    )


    policy = joblib.load(
        'data/trained/policy_2d_footstrength_sd34_1600.pkl')

    # generate data
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

    total_grads = [[],[],[],[]]
    total_grad_all = []
    pol_weights = []
    for i in range(50):
        init_param = policy.get_param_values()
        init_param_obj = copy.deepcopy(policy.get_params())

        #####   get data ###################
        policy.set_param_values(init_param)  # reset the policy parameters
        paths = algo.sampler.obtain_samples(0)

        path_collections = [[],[],[],[]]
        for path in paths:
            mp = path['env_infos']['model_parameters'][-1]
            if mp[0] < 0.5 and mp[1] < 0.5:
                path_collections[0].append(path)
            elif mp[0] < 0.5 and mp[1] >= 0.5:
                path_collections[1].append(path)
            elif mp[0] >= 0.5 and mp [1] < 0.5:
                path_collections[2].append(path)
            elif mp[0] >= 0.5 and mp[1] >= 0.5:
                path_collections[3].append(path)
        #####################################

        for pid in range(len(path_collections)):
            samples_data = algo.sampler.process_samples(0, path_collections[pid])
            samples_data = algo.sampler.process_samples(0, path_collections[pid])
            algo.optimize_policy(0, samples_data)

            tempgrad = []
            for j in range(len(init_param_obj)):
                tempgrad.append(policy.get_params()[j].get_value() - init_param_obj[j].get_value())
            total_grads[pid].append(tempgrad)

        # if not split
        samples_data = algo.sampler.process_samples(0, paths)
        samples_data = algo.sampler.process_samples(0, paths)
        # grad_left = get_gradient(algo, samples_data)
        policy.set_param_values(init_param)  # reset the policy parameters
        algo.optimize_policy(0, samples_data)
        for j in range(len(init_param_obj)):
            pol_weights.append(init_param_obj[j].get_value())
            grad_all = []
        for j in range(len(init_param_obj)):
            grad_all.append(policy.get_params()[j].get_value() - init_param_obj[j].get_value())

        total_grad_all.append(grad_all)

    algo.shutdown_worker()

    joblib.dump([total_grads, total_grad_all, pol_weights], 'data/trained/gradient_temp/total_gradients.pkl', compress=True)

    #total_grads, total_grad_all, pol_weights = joblib.load('data/trained/gradient_temp/total_gradients.pkl')


    split_counts = []
    for i in range(len(total_grads[0][0])):
        split_counts.append(np.zeros(total_grads[0][0][i].shape))
    for i in range(len(total_grads[0])):
        value_list = []
        for j in range(len(total_grads)):
            for k in range(len(total_grads[j][i])):
                curr_weight = total_grads[j][i][k]
                avg_weight = curr_weight * 0
                for l in range(3):
                    avg_weight += total_grads[(j+l)%4][i][k]
                avg_weight /= 3.0
                split_counts[k] += np.abs(curr_weight - avg_weight) * np.abs(pol_weights[i*len(total_grads[j][i])+k])

    split_indices = []
    for p in range(int(len(split_counts)/2)):
        for col in range(split_counts[p*2].shape[1]):
            split_metric = np.mean(split_counts[p*2][:, col]) + split_counts[p*2+1][col]
            split_indices.append([[p, col], split_metric])
    split_indices.sort(key=lambda x:x[1], reverse=True)

    for i in range(int(len(split_counts))):
        split_counts[i] *= 0

    total_param_size = len(policy.get_param_values())
    split_param_size = 0.2 * total_param_size
    current_split_size = 0
    split_layer_units = []
    for i in range(len(split_indices)):
        pm = split_indices[i][0][0]
        col = split_indices[i][0][1]
        split_counts[pm*2][:, col] = 1
        split_counts[pm*2+1][col] = 1
        current_split_size += split_counts[pm*2].shape[0]+1
        split_layer_units.append([pm, col])
        if current_split_size > int(split_param_size):
            break

    split_layer_units.sort(key=lambda x: (x[0], x[1]))


    joblib.dump(split_layer_units, 'data/trained/gradient_temp/split_scheme_4p.pkl', compress=True)

    for j in range(len(split_counts)):
        plt.figure()
        plt.title(policy.get_params()[j].name)
        if len(split_counts[j].shape) == 2:
            plt.imshow(split_counts[j])
            plt.colorbar()
        elif len(split_counts[j].shape) == 1:
            plt.plot(split_counts[j])
        plt.savefig('data/trained/gradient_temp/' + policy.get_params()[j].name + '.png')
    #plt.show()








