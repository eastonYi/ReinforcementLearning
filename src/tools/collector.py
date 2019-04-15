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
        s, a, r, pi, q = pack
        sample = {}
        sample['state'] = s
        sample['action'] = a
        sample['reward'] = r
        sample['pi'] = pi
        sample['Q'] = q
        # sample['UCB'] = u

        self.pool.append(sample)
        if len(self) > self.capacity:
            del self.pool[random.randint(0, len(self)-1)]

    def get_batch(self, batch_size):
        samples = [random.choice(self.pool) for _ in range(batch_size)]

        state = np.array([sample['state'].data for sample in samples])
        pi = np.array([sample['pi'] for sample in samples])

        return state, pi


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
