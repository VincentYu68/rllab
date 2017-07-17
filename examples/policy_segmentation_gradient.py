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
from UPSelector import UPSelector

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

def get_mp_rew_pairs(paths):
    mp_rew_raw = []
    for path in paths:
        mp_rew_raw.append([np.array(path['env_infos']['model_parameters'][-1]), path['rewards'].sum()])
    mp_rew_raw.sort(key=lambda x: str(x[0]))
    mp_rew = []
    i = 0
    while True:
        if i >= len(mp_rew_raw) - 1:
            break
        cur_mp = mp_rew_raw[i][0]
        cur_rew = mp_rew_raw[i][1]
        cur_mp_num = 1
        for j in range(i + 1, len(mp_rew_raw)):
            if (mp_rew_raw[j][0] - cur_mp).any():
                break
            cur_rew += mp_rew_raw[j][1]
            cur_mp_num += 1
        i += cur_mp_num
        mp_rew.append([np.array(cur_mp), cur_rew * 1.0 / cur_mp_num])
    mp_rew.sort(key=lambda x: x[1])
    return mp_rew

def get_UPSelector_training_data(total_paths):
    data = []
    for paths in total_paths:
        data += get_mp_rew_pairs(paths)
    total_training_data = []
    total_training_target = []
    for d in data:
        total_training_data.append(d[0])
        total_training_target.append(d[1])
    return total_training_data, total_training_target



def estimate_splitability(algo, policy, total_paths, init_params, pol_weights, segment_num, loc_weight, training_x, training_y):
    selector = UPSelector()
    selector.num_class=segment_num
    selector.loc_weight = loc_weight

    selector.train(training_x, training_y)

    path_count = [0] * segment_num
    for i in range(len(total_paths)):
        for path in total_paths[i]:
            path_count[selector.classify([path['env_infos']['model_parameters'][-1]])] += 1.0
    path_perc = np.array(path_count) / np.sum(path_count)
    if (path_perc < 0.5/segment_num).any():
        return 0, None, None

    total_grads = []
    for i in range(segmentation_num):
        total_grads.append([])

    for i in range(len(total_paths)):
        policy.set_param_values(init_params[i])
        path_collections = []
        for _ in range(segment_num):
            path_collections.append([])
        for path in total_paths[i]:
            path_collections[selector.classify([path['env_infos']['model_parameters'][-1]])].append(path)

        for pid in range(len(path_collections)):
            samples_data = algo.sampler.process_samples(0, path_collections[pid])
            samples_data = algo.sampler.process_samples(0, path_collections[pid])
            policy.set_param_values(init_params[i])
            algo.optimize_policy(0, samples_data)

            tempgrad = []
            for j in range(len(policy.get_params())):
                tempgrad.append(policy.get_params()[j].get_value() - pol_weights[i*len(policy.get_params())+j])
            total_grads[pid].append(tempgrad)

    split_counts = []
    for i in range(len(total_grads[0][0])):
        split_counts.append(np.zeros(total_grads[0][0][i].shape))
    # use variance instead
    for i in range(len(total_grads[0])):
        for k in range(len(total_grads[0][i])):
            region_gradients = []
            for j in range(len(total_grads)):
                region_gradients.append(total_grads[j][i][k])
            region_gradients = np.array(region_gradients)
            split_counts[k] += np.var(region_gradients, axis=0) * np.abs(pol_weights[i * len(total_grads[0][i]) + k])

    split_metrics = []
    for p in range(int(len(split_counts) / 2)):
        for col in range(split_counts[p * 2].shape[1]):
            split_metric = np.mean(split_counts[p * 2][:, col]) + split_counts[p * 2 + 1][col]
            split_metrics.append(split_metric)

    return np.sum(split_metrics), selector, split_counts


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
        'data/trained/policy_3d_rest1_sd1_400.pkl')

    folder_name = '3d_rest1_sd1_8seg'
    segmentation_num = 8
    loc_weights = [0.0, 0.2, 0.5]
    load_path_from_file = True
    load_metric_from_file = True
    split_percentage = 0.2


    # generate data
    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=300000,
        max_path_length=env.horizon,
        n_itr=5,

        discount=0.995,
        step_size=0.01,
        gae_lambda=0.97,
    )

    algo.init_opt()
    if not load_path_from_file:
        init_param = policy.get_param_values()
        init_param_obj = copy.deepcopy(policy.get_params())

        from rllab.sampler import parallel_sampler

        parallel_sampler.initialize(n_parallel=7)

        env.wrapped_env.env.env.perturb_MP = False
        algo.start_worker()

        pol_weights = []
        all_paths = []
        policy_params = []
        for i in range(50):
            init_param = policy.get_param_values()
            init_param_obj = copy.deepcopy(policy.get_params())
            policy_params.append(np.copy(init_param))

            #####   get data ###################
            policy.set_param_values(init_param)  # reset the policy parameters
            paths = algo.sampler.obtain_samples(0)
            all_paths.append(paths)

            # if not split
            samples_data = algo.sampler.process_samples(0, paths)
            if i == 0:
                samples_data = algo.sampler.process_samples(0, paths)
            # grad_left = get_gradient(algo, samples_data)
            policy.set_param_values(init_param)  # reset the policy parameters
            algo.optimize_policy(0, samples_data)
            for j in range(len(init_param_obj)):
                pol_weights.append(init_param_obj[j].get_value())

        algo.shutdown_worker()

        joblib.dump(all_paths, 'data/trained/gradient_temp/'+folder_name+'/all_paths.pkl', compress=True)
        joblib.dump([policy_params, pol_weights], 'data/trained/gradient_temp/'+folder_name+'/policy_params.pkl', compress=True)
    else:
        all_paths = joblib.load('data/trained/gradient_temp/' + folder_name + '/all_paths.pkl')
        policy_params, pol_weights = joblib.load('data/trained/gradient_temp/'+folder_name+'/policy_params.pkl')

    if not load_metric_from_file:
        # compute the optimal segmentation
        UPSelector_X, UPSelector_Y = get_UPSelector_training_data(all_paths)
        splitability_list=[]
        max_splitability = -1
        best_seletor = None
        best_split_counts = None
        for loc_weight in loc_weights:
            splitability, selector, split_counts = estimate_splitability(algo, policy, all_paths, policy_params, pol_weights, segmentation_num, loc_weight, UPSelector_X, UPSelector_Y)
            if splitability > max_splitability:
                max_splitability = splitability
                best_seletor = copy.deepcopy(selector)
                best_split_counts = copy.deepcopy(split_counts)
            splitability_list.append([loc_weight, splitability])
            print(loc_weight, splitability)
        joblib.dump(best_seletor, 'data/trained/gradient_temp/' + folder_name + '/UPSelector_'+folder_name+'.pkl', compress=True)
        joblib.dump(best_split_counts, 'data/trained/gradient_temp/' + folder_name + '/split_metric.pkl', compress=True)
        print('best selector location weight: ', best_seletor.loc_weight)
        print(splitability_list)
    else:
        best_seletor = joblib.load('data/trained/gradient_temp/' + folder_name + '/UPSelector_'+folder_name+'.pkl')
        best_split_counts = joblib.load('data/trained/gradient_temp/' + folder_name + '/split_metric.pkl')


    split_indices = []
    for p in range(int(len(best_split_counts)/2)):
        for col in range(best_split_counts[p*2].shape[1]):
            split_metric = np.mean(best_split_counts[p*2][:, col]) + best_split_counts[p*2+1][col]
            split_indices.append([[p, col], split_metric])
    split_indices.sort(key=lambda x:x[1], reverse=True)

    for i in range(int(len(best_split_counts))):
        best_split_counts[i] *= 0

    total_param_size = len(policy.get_param_values())
    split_param_size = split_percentage * total_param_size
    current_split_size = 0
    split_layer_units = []
    for i in range(len(split_indices)):
        pm = split_indices[i][0][0]
        col = split_indices[i][0][1]
        best_split_counts[pm*2][:, col] = 1
        best_split_counts[pm*2+1][col] = 1
        current_split_size += best_split_counts[pm*2].shape[0]+1
        split_layer_units.append([pm, col])
        if current_split_size > int(split_param_size):
            break

    split_layer_units.sort(key=lambda x: (x[0], x[1]))

    joblib.dump(split_layer_units, 'data/trained/gradient_temp/'+folder_name+'/split_scheme_'+folder_name+'_orth_'+str(split_percentage)+'.pkl', compress=True)

    for j in range(len(best_split_counts)):
        plt.figure()
        plt.title(policy.get_params()[j].name)
        if len(best_split_counts[j].shape) == 2:
            plt.imshow(best_split_counts[j])
            plt.colorbar()
        elif len(best_split_counts[j].shape) == 1:
            plt.plot(best_split_counts[j])
        plt.savefig('data/trained/gradient_temp/'+folder_name+'/' + policy.get_params()[j].name + '.png')

    mp_dim = best_seletor.models[0]._fit_X.shape[1]
    if mp_dim == 2:
        coordx = []
        coordy = []
        pred_data = []
        for i in np.arange(0, 1, 0.05):
            for j in np.arange(0, 1, 0.05):
                reps = []
                for rep in range(1):
                    reps.append(best_seletor.classify(np.array([[i, j]]), stoch=False))
                pred = np.mean(reps)
                coordx.append(i)
                coordy.append(j)
                pred_data.append(pred)

        plt.imshow(np.reshape(pred_data, (20, 20)))
        plt.colorbar()
        plt.savefig('data/trained/gradient_temp/'+folder_name+'/mp_segmentation.png')
    #plt.show()








