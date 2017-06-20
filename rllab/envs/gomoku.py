__author__ = 'yuwenhao'

from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Discrete, Box
import numpy as np
from rllab.core.serializable import Serializable


class Gomoku(Env, Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())
        self.n_size = 10
        self.state = [-1]*(self.n_size * self.n_size)
        self.side = 0 # 0 is black, 1 is white
        self.opponent_policy = None # opponent to play against

    @property
    def observation_space(self):
        return Box(low=-1, high=1, shape=(self.n_size*self.n_size,))

    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(self.n_size*self.n_size,))

    def reset(self):
        self.state = [-1]*(self.n_size * self.n_size)
        observation = np.copy(self.state)
        self.side = np.random.randint(0, 2)

        if self.side == 1:
            self.opponent_move()
        return observation

    def opponent_move(self):
        if self.opponent_policy is not None:
            oact, _ = self.opponent_policy.get_action(np.copy(self.state))
        else:
            oact = self.action_space.sample()
        loc = np.argmax(oact[np.array(self.state) == -1])
        self.state[loc] = 1-self.side

    def step(self, action):
        reward = 0
        done = False

        copied_act = np.copy(action)
        loc = np.argmax(copied_act[np.array(self.state) == -1])

        self.state[loc] = self.side

        if not -1 in self.state:
            done = True
        else:
            self.opponent_move()

        next_observation = np.copy(self.state)


        if not -1 in self.state:
            done = True
        # search per row for 5 consecutive ones
        found_win = -1
        for i in range(self.n_size):
            consec = 0
            consec_side = -1
            for j in range(self.n_size):
                if self.state[i * self.n_size + j] != -1:
                    if self.state[i * self.n_size + j] == consec_side:
                        consec += 1
                        if consec == 5:
                            found_win = consec_side
                            break
                    else:
                        consec = 1
                        consec_side = self.state[i * self.n_size + j]
                else:
                    consec = 0
                    consec_side = -1
            if found_win != -1:
                break
        if found_win == -1:
            for i in range(self.n_size):
                consec = 0
                consec_side = -1
                for j in range(self.n_size):
                    if self.state[j * self.n_size + i] != -1:
                        if self.state[j * self.n_size + i] == consec_side:
                            consec += 1
                            if consec == 5:
                                found_win = consec_side
                                break
                        else:
                            consec = 1
                            consec_side = self.state[j * self.n_size + i]
                    else:
                        consec = 0
                        consec_side = -1
                if found_win != -1:
                    break
        if found_win == -1:
            for i in range(self.n_size):
                for j in range(self.n_size):
                    if self.n_size - i >= 5 and self.n_size - j >= 5:
                        if self.state[(i) * self.n_size + j] == -1:
                            continue
                        identical = True
                        for id in range(4):
                            if self.state[(i+id) * self.n_size + j+id] != self.state[(i+id+1) * self.n_size + j+id+1]:
                                identical = False
                                break
                        if identical:
                            found_win = self.state[(i) * self.n_size + j]
                            break
                    if i >= 4 and self.n_size - j >= 5:
                        if self.state[(i) * self.n_size + j] == -1:
                            continue
                        identical = True
                        for id in range(4):
                            if self.state[(i-id) * self.n_size + j+id] != self.state[(i-id-1) * self.n_size + j+id+1]:
                                identical = False
                                break
                        if identical:
                            found_win = self.state[(i) * self.n_size + j]
                            break

                if found_win != -1:
                    break

        if found_win != -1:
            done = True
            if found_win == self.side:
                reward = 1
            else:
                reward = -1

        #    print('done', reward, self.side)

        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print(np.reshape(self.state, (self.n_size, self.n_size))+1)
        print()