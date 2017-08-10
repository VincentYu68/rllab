import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False, resample_mp = None, target_task = None):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()

    dartenv = env._wrapped_env.env.env
    if env._wrapped_env.monitoring:
        dartenv = dartenv.env
    if resample_mp is not None:
        dartenv.param_manager.set_simulator_parameters(resample_mp)

    if target_task is not None:
        while dartenv.state_index != target_task:
            o = env.reset()

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        if hasattr(agent, '_lowlevelnetwork'):
            lowa = agent.lowlevel_action(o, a)
            next_o, r, d, env_info = env.step(lowa)
            #print('lowlevela:',lowa)
        else:
            #print('normal a:',agent_info['mean'])
            next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
        #if path_length > 5:
        #    abc
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
