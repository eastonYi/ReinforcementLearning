import numpy as np

from src.agents.q_estimator import Q_Estimator as Agent
from src.envs.bandit import Bandit as ENV

np.set_printoptions(precision=2)

def main1():
    round = 10
    lr = 0.1
    env = ENV(true_rewards=np.arange(10))
    agent = Agent(size=10, epsilon=0.9)

    for _ in range(1000):
        average_reward = 0
        for i in range(round):
            action = agent.forward()
            reward = env.step(action)

            average_reward = i/(i+1) * average_reward + reward/(i+1)
            agent.q_estimation[action] += lr * (reward - agent.q_estimation[action])
            # print(agent.q_estimation)
        print('r: {:.3f}, a: {}'.format(average_reward, action))

def test_UCB():
    round = 50
    lr = 0.1
    env = ENV(true_rewards=np.arange(10))
    agent = Agent(size=10, UCB_param=1.0)

    for _ in range(10):
        average_reward = 0
        for i in range(round):
            action = agent.forward()
            reward = env.step(action)

            average_reward = i/(i+1) * average_reward + reward/(i+1)
            agent.q_estimation[action] += lr * (reward - agent.q_estimation[action])
            agent.action_count[action] += 1
            # print(agent.q_estimation)
        print('r: {:.3f}, a: {}'.format(average_reward, action))


if __name__ == '__main__':
    # main1()
    test_UCB()
