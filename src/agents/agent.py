'''
the template of agents
'''
import numpy as np
import random
from abc import ABCMeta, abstractmethod


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, size_action):
        self.size_action = size_action

    def __call__(self):
        return self.forward()

    @abstractmethod
    def forward(self):
        """
        """

    def exploration(self):
        return random.randint(0, self.size_action-1)

    def exploitation(self, prob):
        return np.random.multinomial(1, prob)
