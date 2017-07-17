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
from sklearn.cluster import KMeans

def get_gradient(algo, samples_data, flat = False):
    all_input_values = tuple(ext.extract(
        samples_data,
        "observations", "actions", "advantages"
    ))
    agent_infos = samples_data["agent_infos"]
    state_info_list = [agent_infos[k] for k in algo.policy.state_info_keys]
    dist_info_list = [agent_infos[k] for k in algo.policy.distribution.dist_info_keys]
    all_input_values += tuple(state_info_list) + tuple(dist_info_list)

    if flat:
        grad = sliced_fun(algo.optimizer._opt_fun["f_grad"], 1)(
            tuple(all_input_values), tuple())
    else:
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


    policy = joblib.load('data/trained/policy_2d_restfoot_sd33_1500.pkl')

    folder_name = 'restfoot_sd3_test'
    segmentation_num = 2
    load_path_from_file = False
    load_metric_from_file = True
    split_percentage = 0.2

    # generate data
    baseline = LinearFeatureBaseline(env_spec=env.spec, additional_dim=0)

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

    algo.init_opt()

    one_iter_grad = []
    mps = []
    if not load_path_from_file:
        init_param = policy.get_param_values()
        init_param_obj = copy.deepcopy(policy.get_params())

        from rllab.sampler import parallel_sampler

        parallel_sampler.initialize(n_parallel=7)

        env.wrapped_env.env.env.perturb_MP = False
        pol_weights = []
        all_paths = []
        policy_params = []
        init_param = np.copy(policy.get_param_values())
        algo.start_worker()
        for i in range(100):
            policy.set_param_values(init_param)
            #####   get data ###################
            for it in range(1):
                paths = algo.sampler.obtain_samples(i)
                algo.sampler.process_samples(0, paths)
                samples_data = algo.sampler.process_samples(0, paths)
                algo.optimize_policy(0, samples_data)
                #all_paths.append(paths)
            print('iter ', iter)
            mps.append(paths[0]['env_infos']['model_parameters'][-1])
            one_iter_grad.append(policy.get_param_values() - init_param)

        algo.shutdown_worker()

        #joblib.dump(all_paths, 'data/trained/gradient_temp/'+folder_name+'/all_paths.pkl', compress=True)

        joblib.dump([one_iter_grad, mps], 'data/trained/gradient_temp/' + folder_name + '/pointwise_grad.pkl',
                    compress=True)
    else:
        #all_paths = joblib.load('data/trained/gradient_temp/' + folder_name + '/all_paths.pkl')
        one_iter_grad, mps = joblib.load('data/trained/gradient_temp/' + folder_name + '/pointwise_grad.pkl')

    '''if not load_metric_from_file:
        # compute policy gradient for each path
        one_iter_grad = []
        mps = []
        path_composite = []
        for iter in range(int(len(all_paths))):
            path_composite += all_paths[iter]
        init_param = np.copy(policy.get_param_values())
        for iter in range(int(len(all_paths))):
            policy.set_param_values(init_param)
            algo.sampler.process_samples(0, all_paths[iter])
            samples_data = algo.sampler.process_samples(0, all_paths[iter])
            #algo.optimize_policy(0, samples_data)
            #samp_gradient = policy.get_param_values() - init_param
            samp_gradient = get_gradient(algo, samples_data, flat=True)

            one_iter_grad.append(samp_gradient)
            mps.append(all_paths[iter][0]['env_infos']['model_parameters'][-1])
        mps = np.array(mps)
        joblib.dump([one_iter_grad, mps], 'data/trained/gradient_temp/' + folder_name + '/pointwise_grad.pkl', compress=True)
    else:
        one_iter_grad, mps = joblib.load('data/trained/gradient_temp/' + folder_name + '/pointwise_grad.pkl')'''

    for p in range(len(one_iter_grad)):
        one_iter_grad[p] /= np.linalg.norm(one_iter_grad[p])

    kmeans = KMeans(n_clusters=segmentation_num)
    kmeans.fit(one_iter_grad)
    pred_class = kmeans.predict(one_iter_grad)

    mps = np.array(mps)

    plt.figure()
    plt.scatter(mps[:, 0], mps[:, 1], c=pred_class)
    plt.show()










