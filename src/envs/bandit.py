from .env import Environment
import numpy as np
import random

class Bandit(Environment):
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, true_rewards, args=None):
        self.true_rewards = true_rewards
        self.size = len(true_rewards)
        self.best_actions = np.argmax(self.true_rewards)
        super().__init__(args)

    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = random.random() + self.true_rewards[action]

        return reward
