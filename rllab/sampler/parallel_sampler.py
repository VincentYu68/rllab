from rllab.sampler.utils import rollout
from rllab.sampler.stateful_pool import singleton_pool, SharedGlobal
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc import tensor_utils
import pickle
import joblib
import numpy as np
import copy


def _worker_init(G, id):
    if singleton_pool.n_parallel > 1:
        import os
        os.environ['THEANO_FLAGS'] = 'device=cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    G.worker_id = id


def initialize(n_parallel):
    singleton_pool.initialize(n_parallel)
    singleton_pool.run_each(_worker_init, [(id,) for id in range(singleton_pool.n_parallel)])


def _get_scoped_G(G, scope):
    if scope is None:
        return G
    if not hasattr(G, "scopes"):
        G.scopes = dict()
    if scope not in G.scopes:
        G.scopes[scope] = SharedGlobal()
        G.scopes[scope].worker_id = G.worker_id
    return G.scopes[scope]


def _worker_populate_task(G, env, policy, scope=None):
    G = _get_scoped_G(G, scope)
    G.env = pickle.loads(env)
    G.policy = pickle.loads(policy)


def _worker_terminate_task(G, scope=None):
    G = _get_scoped_G(G, scope)
    if getattr(G, "env", None):
        G.env.terminate()
        G.env = None
    if getattr(G, "policy", None):
        G.policy.terminate()
        G.policy = None


def populate_task(env, policy, scope=None):
    logger.log("Populating workers...")
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(
            _worker_populate_task,
            [(pickle.dumps(env), pickle.dumps(policy), scope)] * singleton_pool.n_parallel
        )
    else:
        # avoid unnecessary copying
        G = _get_scoped_G(singleton_pool.G, scope)
        G.env = env
        G.policy = policy
    logger.log("Populated")


def terminate_task(scope=None):
    singleton_pool.run_each(
        _worker_terminate_task,
        [(scope,)] * singleton_pool.n_parallel
    )


def _worker_set_seed(_, seed):
    logger.log("Setting seed to %d" % seed)
    ext.set_seed(seed)


def set_seed(seed):
    singleton_pool.run_each(
        _worker_set_seed,
        [(seed + i,) for i in range(singleton_pool.n_parallel)]
    )


def _worker_set_policy_params(G, params, scope=None):
    G = _get_scoped_G(G, scope)
    G.policy.set_param_values(params)

def _worker_set_env_params(G,params,scope=None):
    G = _get_scoped_G(G, scope)
    G.env.set_param_values(params)

def _worker_collect_one_path(G, max_path_length, scope=None):
    G = _get_scoped_G(G, scope)
    dartenv = G.env._wrapped_env.env.env
    if G.env._wrapped_env.monitoring:
        dartenv = dartenv.env
    dartenv.iter = G.mp_resamp['iter']
    if hasattr(dartenv, 'param_manager') and G.mp_resamp['use_adjusted_resample']:
        if dartenv.train_UP:
            dartenv.param_manager.resample_parameters()
            dartenv.resample_MP = False
            model_parameter = dartenv.param_manager.get_simulator_parameters()
            if G.mp_resamp['mr_activated']:
                model_parameter = G.mp_resamp['mr_buffer'][np.random.randint(0, int(len(G.mp_resamp['mr_buffer'])), 1)[0]]
                #model_parameter += np.random.normal(0, 0.005, len(model_parameter))
                #model_parameter = np.clip(model_parameter, 0, 1)

            sample_num = 0
            sampled_paths = []
            while sample_num <= G.env.horizon * 0.9:
                path = rollout(G.env, G.policy, max_path_length, resample_mp=model_parameter+np.random.uniform(-0.005, 0.005, len(model_parameter)))
                sampled_paths.append(path)
                sample_num += len(path["rewards"])
            return sampled_paths, sample_num

    if G.ensemble_dynamics['use_ens_dyn']:
        dartenv.dyn_models = G.ensemble_dynamics['dyn_models']
        dartenv.dyn_model_id = G.ensemble_dynamics['dyn_model_choice']
        if len(G.ensemble_dynamics['base_paths']) > 0:
            dartenv.base_path = G.ensemble_dynamics['base_paths'][np.random.randint(0, len(G.ensemble_dynamics['base_paths']))]
            dartenv.transition_locator = G.ensemble_dynamics['transition_locator']

    if hasattr(dartenv, 'param_manager') and len(G.mp_resamp['mr_buffer']) > 0 and not dartenv.resample_MP:
        if np.random.random() < 1.0 / len(G.mp_resamp['mr_buffer']):
            model_parameter = np.random.uniform(low=-0.05, high = 1.05, size=len(G.mp_resamp['mr_buffer'][0]))
        else:
            model_parameter = np.copy(G.mp_resamp['mr_buffer'][np.random.randint(0, len(G.mp_resamp['mr_buffer']))])
            model_parameter += np.random.uniform(low=-0.01, high=0.01, size=len(model_parameter))
        path = rollout(G.env, G.policy, max_path_length, resample_mp=model_parameter)
    else:
        path = rollout(G.env, G.policy, max_path_length)
    return [path], len(path["rewards"])


def _worker_update_mr(G, paramname, newval, scope):
    G = _get_scoped_G(G, scope)
    G.mp_resamp[paramname] = newval

def _worker_update_dyn(G, paramname, newval, scope):
    G = _get_scoped_G(G, scope)
    G.ensemble_dynamics[paramname] = newval

def sample_paths(
        policy_params,
        max_samples,
        max_path_length=np.inf,
        env_params=None,
        scope=None,
        iter = 0):
    """
    :param policy_params: parameters for the policy. This will be updated on each worker process
    :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
    might be greater since all trajectories will be rolled out either until termination or until max_path_length is
    reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """


    singleton_pool.run_each(
        _worker_set_policy_params,
        [(policy_params, scope)] * singleton_pool.n_parallel
    )
    if env_params is not None:
        singleton_pool.run_each(
            _worker_set_env_params,
            [(env_params, scope)] * singleton_pool.n_parallel
        )


    if singleton_pool.G.ensemble_dynamics['use_ens_dyn'] and iter > 0:
        singleton_pool.run_each(_worker_update_dyn, [('dyn_model_choice',
                                                             0, scope)] * singleton_pool.n_parallel)
        result1 = singleton_pool.run_collect(
            _worker_collect_one_path,
            threshold=max_samples * (1.0/2.0),
            args=(max_path_length, scope),
            show_prog_bar=True
        )

        singleton_pool.run_each(_worker_update_dyn, [('dyn_model_choice',
                                                             1, scope)] * singleton_pool.n_parallel)
        singleton_pool.run_each(_worker_update_dyn, [('base_paths',
                                                             result1, scope)] * singleton_pool.n_parallel)

        result2 = singleton_pool.run_collect(
            _worker_collect_one_path,
            threshold=max_samples * (1.0/2.0),
            args=(max_path_length, scope),
            show_prog_bar=True
        )

        result = result1 + result2
    else:
        result = singleton_pool.run_collect(
            _worker_collect_one_path,
            threshold=max_samples,
            args=(max_path_length, scope),
            show_prog_bar=True
        )


    logger.log('Collected Traj Num: '+str(len(result)))

    if singleton_pool.G.ensemble_dynamics['use_ens_dyn']:
        dyn_training_x = []
        dyn_training_y = []
        dyn_training_result = result
        if iter > 0:
            dyn_training_result = result1
        for path in dyn_training_result:
            for state_act in path['env_infos']['state_act']:
                dyn_training_x.append(state_act)
            for next_state in path['env_infos']['next_state']:
                dyn_training_y.append(next_state)
        singleton_pool.G.ensemble_dynamics['training_buffer_x'] += dyn_training_x
        singleton_pool.G.ensemble_dynamics['training_buffer_y'] += dyn_training_y
        if len(singleton_pool.G.ensemble_dynamics['training_buffer_x']) > 100000:
            singleton_pool.G.ensemble_dynamics['training_buffer_x'] = singleton_pool.G.ensemble_dynamics['training_buffer_x'][-100000:]
            singleton_pool.G.ensemble_dynamics['training_buffer_y'] = singleton_pool.G.ensemble_dynamics['training_buffer_y'][-100000:]
        singleton_pool.G.ensemble_dynamics['dyn_models'][0].fit(dyn_training_x, dyn_training_y)
        singleton_pool.G.ensemble_dynamics['transition_locator'].fit(singleton_pool.G.ensemble_dynamics['training_buffer_x'], singleton_pool.G.ensemble_dynamics['training_buffer_y'])
        print('fitted dynamic models and transition locator')
        singleton_pool.run_each(_worker_update_dyn, [('dyn_models',
                                                             singleton_pool.G.ensemble_dynamics['dyn_models'], scope)] * singleton_pool.n_parallel)
        singleton_pool.run_each(_worker_update_dyn, [('transition_locator',
                                                             singleton_pool.G.ensemble_dynamics['transition_locator'], scope)] * singleton_pool.n_parallel)
        joblib.dump(singleton_pool.G.ensemble_dynamics['dyn_models'], 'data/trained/dyn_models.pkl', compress=True)




    if 'model_parameters' in result[0]['env_infos'] and logger._snapshot_dir is not None:
        mp_rew_raw = []
        for path in result:
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
        filename = logger._snapshot_dir + '/mp_rew_' + str(iter) + '.pkl'
        pickle.dump(mp_rew, open(filename, 'wb'))

    return result


def truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is exactly equal to max_samples. This is done by
    removing extra paths at the end of the list, and make the last path shorter if necessary
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while len(paths) > 0 and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    if len(paths) > 0:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(last_path["rewards"]) - (total_n_samples - max_samples)
        for k, v in last_path.items():
            if k in ["observations", "actions", "rewards"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_list(v, truncated_len)
            elif k in ["env_infos", "agent_infos"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(v, truncated_len)
            else:
                raise NotImplementedError
        paths.append(truncated_last_path)
    return paths
