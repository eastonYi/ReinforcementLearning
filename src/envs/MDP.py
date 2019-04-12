from .env import Environment
import numpy as np
import random


class MDP(Environment):
    """
    A->A->A->...->A
    each move will generate various and deterministic rewards
    the agent needs to search a best path walking through.

    It is a simple implement of lamguage model.
    """
    def __init__(self, size, size_state, args=None):
        self.size = size
        self.size_state = size_state
        self.list_weights = []
        self.cur_state = 0
        self.pre_action = 0
        self.s = [self.size] * size_state
        [self.expand_state() for _ in range(8)]
        super().__init__(args)

    def reset(self):
        self.cur_state = 0
        self.pre_action = 0
        self.s = [self.size] * self.size_state

    def step(self, action):
        reward = self.list_weights[self.cur_state][self.pre_action][action]
        # noise = np.random.standard_normal(size=1)
        noise = 0

        self.cur_state += 1
        self.pre_action = action
        self.s.append(action)
        del self.s[0]

        return reward + noise, self.s.copy()

    def expand_state(self):
        i = len(self.list_weights)
        w = np.random.randint(10, size=(self.size, self.size))
        w[i][i+1] = 100
        self.list_weights.append(w)

    def reveal(self):
        import pdb; pdb.set_trace()
        return [np.max(w) for w in self.list_weights]
