import random
import numpy as np


class Sample_Collector(object):
    '''
    pool: [{}, {}, {}]
    '''
    def __init__(self, capacity):
        self.pool = []
        self.capacity = capacity

    def __call__(self, pack):
        self.collect(pack)

    def __len__(self):
        return len(self.pool)

    def collect(self, pack):
        s, a, r, q, u = pack
        sample = {}
        sample['state'] = s
        sample['action'] = a
        sample['reward'] = r
        sample['Q'] = q
        sample['UCB'] = u

        self.pool.append(sample)
        if len(self) > self.capacity:
            del self.pool[0]

    def get_batch(self, batch_size):
        samples = [random.choice(self.pool) for _ in range(batch_size)]
        state = np.array([sample['state'] for sample in samples])
        action = np.array([sample['action'] for sample in samples])
        reward = np.array([sample['reward'] for sample in samples])

        return state, action, reward


class Memory_Collector(object):
    '''
    pool: [{}, {}, {}]
    '''
    def __init__(self, capacity):
        self.list_a = []
        self.list_r = []
        self.pool = []
        self.capacity = capacity

    def __call__(self, pack):
        self.collect(pack)

    def __len__(self):
        return len(self.pool)

    def collect(self, pack):
        a, r = pack
        self.list_a.append(a)
        self.list_r.append(r)

    def wrapper(self):
        sample = {}
        sample['actions'] = self.list_a
        sample['lengths'] = len(self.list_a)
        sample['rewards'] = self.list_r
        self.list_a = []
        self.list_r = []

        self.pool.append(sample)
        if len(self) > self.capacity:
            del self.pool[0]

    def get_batch(self, batch_size):
        samples = [random.choice(self.pool) for _ in range(batch_size)]
        actions = np.array([sample['actions'] for sample in samples])
        lengths = np.array([sample['lengths'] for sample in samples])
        rewards = np.array([sample['rewards'] for sample in samples])

        return actions, lengths, rewards
