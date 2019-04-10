from abc import ABCMeta, abstractmethod


class Environment(object):
    __metaclass__ = ABCMeta

    def __init__(self, args):
        self.args = args

    def __call__(self, action):
        return self.step(action)

    @abstractmethod
    def step(self, action):
        '''
        '''
        # return next_state, reward, done, info

    def reset(self):
        '''
        '''
        return
