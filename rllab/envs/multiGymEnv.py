__author__ = 'yuwenhao'
import gym
import gym.wrappers
import gym.envs
import gym.spaces
import traceback
import logging
import numpy as np

try:
    from gym.wrappers.monitoring import logger as monitor_logger

    monitor_logger.setLevel(logging.WARNING)
except Exception as e:
    traceback.print_exc()

import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger


def convert_gym_space(space, box_additional_dim = 0):
    if isinstance(space, gym.spaces.Box):
        if box_additional_dim != 0:
            low = np.concatenate([space.low, [-np.inf]*box_additional_dim])
            high = np.concatenate([space.high, [np.inf]*box_additional_dim])
            return Box(low=low, high=high)
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    else:
        raise NotImplementedError


class CappedCubicVideoSchedule(object):
    # Copied from gym, since this method is frequently moved around
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0


class FixedIntervalVideoSchedule(object):
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, count):
        return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False

# load multiple gym envs
class MultiGymEnv(Env, Serializable):
    def __init__(self, env_names, record_video=True, video_schedule=None, log_dir=None, record_log=True,
                 force_reset=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        envs = []
        for name in env_names:
            envs.append(gym.envs.make(name))
        self.envs = envs
        self.envs_id = []
        for env in envs:
            self.envs_id.append(env.spec.id)

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        # NOTE: the observation_space, action_space and horizon are assumed to be the same across the multiple loaded envs
        self._observation_space = convert_gym_space(envs[0].observation_space, len(self.envs))
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(envs[0].action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = envs[0].spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        self._log_dir = log_dir
        self._force_reset = force_reset
        self._current_activated_env = 0

        self.split_task_test = True
        self.avg_div = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        if self.split_task_test:
            self._current_activated_env = np.random.randint(len(self.envs))

        if self._force_reset and self.monitoring:
            from gym.wrappers.monitoring import _Monitor
            assert isinstance(self.env, _Monitor)
            recorder = self.envs[self._current_activated_env].stats_recorder
            if recorder is not None:
                recorder.done = True

        init_obs = self.envs[self._current_activated_env].reset()
        if self.split_task_test:
            split_vec = np.zeros(len(self.envs))
            split_vec[self._current_activated_env] = 1
            init_obs = np.concatenate([init_obs, split_vec])
        if self.avg_div != 0:
            split_vec = np.zeros(len(self.envs))
            split_vec[self._current_activated_env] = 1
            init_obs = np.concatenate([init_obs, split_vec])
        return init_obs

    def step(self, action):
        next_obs, reward, done, info = self.envs[self._current_activated_env].step(action)

        if self.split_task_test:
            split_vec = np.zeros(len(self.envs))
            split_vec[self._current_activated_env] = 1
            next_obs = np.concatenate([next_obs, split_vec])
        if self.avg_div != 0:
            split_vec = np.zeros(len(self.envs))
            split_vec[self._current_activated_env] = 1
            next_obs = np.concatenate([next_obs, split_vec])
        info['state_index'] = self._current_activated_env

        return Step(next_obs, reward, done, **info)

    def render(self):
        self.envs[self._current_activated_env].render()

    def terminate(self):
        if self.monitoring:
            self.envs[self._current_activated_env]._close()
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py %s

    ***************************
                """ % self._log_dir)

    def set_param_values(self, env_params):
        for k,v in env_params.items():
            if hasattr(self, k):
                setattr(self, k, v)
                if k == 'avg_div' and v != 0:
                    self._observation_space = convert_gym_space(self.envs[0].observation_space, len(self.envs)+v)

        '''dartenv = self.env.env
        if self.monitoring:
            dartenv = dartenv.env
        for k,v in env_params.items():
            if hasattr(dartenv, k):
                setattr(dartenv, k, v)'''

