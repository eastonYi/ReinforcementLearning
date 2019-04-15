import numpy as np
import random
import copy

from .env import Environment
from src.tools.state import State

class MDP(Environment):
    """
    A->A->A->...->A
    each move will generate various and deterministic rewards
    the agent needs to search a best path walking through.

    It is a simple implement of lamguage model.

    size: size of action space
    num_state: num of different rewards distributions
    """
    def __init__(self, size, num_state, args=None):
        self.size = size
        self.num_state = num_state
        self.list_weights = []
        self.cur_state = 0
        self.pre_action = 0
        self.s = [self.size] * num_state
        [self.expand() for _ in range(num_state)]
        super().__init__(args)

    def reset(self):
        self.cur_state = 0
        self.pre_action = 0
        self.s = [self.size] * self.num_state

    def save_state(self):
        return self.cur_state, self.pre_action, copy.deepcopy(self.s)

    def load_state(self, env_state):
        self.cur_state, self.pre_action, self.s = env_state

    def isEnd(self):
        return self.cur_state >= self.num_state -1

    def step(self, action):
        try:
            reward = self.list_weights[self.cur_state][self.pre_action][action]
        except:
            print(self.cur_state, self.pre_action, action)
            import pdb; pdb.set_trace()
            print(self.cur_state, self.pre_action, action)
        # noise = np.random.standard_normal(size=1)
        noise = 0

        self.cur_state += 1
        done = True if self.cur_state >= self.size-1 else False
        self.pre_action = action
        self.s.append(action)
        del self.s[0]

        return reward + noise, self.get_state(), done

    def expand(self):
        i = len(self.list_weights)
        w = np.random.randint(10, size=(self.size, self.size))
        w[i][i+1] = 100
        self.list_weights.append(w)

    def get_state(self):

        return State(self.s)

    def reveal(self):
        import pdb; pdb.set_trace()
        return [np.max(w) for w in self.list_weights]
