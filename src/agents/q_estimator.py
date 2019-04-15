import numpy as np
import random

from .agent import Agent

np.set_printoptions(precision=2)


class Q_Estimator(Agent):
    def __init__(self, size, epsilon=0, q_init=10, UCB_param=None, gradient=None):
        self.q_estimation = np.ones(size) * q_init
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.action_count = np.zeros([size], dtype=np.int32)
        self.epsilon = epsilon
        super().__init__(size)

    def move(self):
        """
        from state to action
        """
        # action
        if self.UCB_param:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(np.sum(self.action_count)) / (self.action_count + 1e-5))
            # print('self.action_count: ', self.action_count)
            # print('UCB_estimation: ', UCB_estimation)
            action = np.argmax(UCB_estimation)
        else:
            if random.random() < self.epsilon:
                action = self.exploitation()
            else:
                action = self.exploration()

        return action

    def exploitation(self):
        if self.gradient:
            return np.random.multinomial(self.q_estimation)
        else:
            return np.argmax(self.q_estimation)
