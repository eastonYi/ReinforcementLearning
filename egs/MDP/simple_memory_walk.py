import numpy as np
import tensorflow as tf
import random

from src.agents.walker import Walker as Agent
from src.envs.MDP import MDP as ENV
from src.tools import Sample_Collector

np.set_printoptions(precision=2)
random.seed(0)
np.random.seed(0)

def main():
    round = 8
    lr = 0.0002
    env = ENV(size=10)
    # agent = Agent(size_action=10, num_cell_units=50, num_layers=3, UCB_param=1.0)
    agent = Agent(size_action=10, num_cell_units=150, num_layers=3, epsilon=0.9)
    op_loss, op_optimize = optimize(agent, lr)
    collector = Sample_Collector(50)

    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    agent.sess = sess

    for t in range(3000):
        reward = 0
        agent.clear_memory()
        env.reset()

        for i in range(round):
            a, q_estimation = agent.forward()
            r = env.step(a)

            agent.action_count[a] += 1
            collector((a, r))
            # print(a, q_estimation)
            # reward = i/(i+1) * reward + reward/(i+1)
            reward += r

        collector.wrapper()
        if len(collector) > 20:
            loss = training(sess, collector, agent, op_loss, op_optimize)
            print('loss: {:.3f}, reward: {:.3f}'.format(loss, reward))
    # print('env potatial: {:.2f}, agent gets: {:.2f}'.format(sum(env.reveal()), sum(reward)))
    test(agent, env, round)

def optimize(agent, lr):
    op_loss = agent.compute_loss()
    optimizer = tf.train.GradientDescentOptimizer(lr)
    gradients = optimizer.compute_gradients(op_loss)
    op_optimize = optimizer.apply_gradients(gradients)

    return op_loss, op_optimize


def training(sess, collector, agent, op_loss, op_optimize):
    batch_a, batch_l, batch_r = collector.get_batch(10)
    feed_dict = {agent.pl_actions: batch_a,
                 agent.pl_lengths: batch_l,
                 agent.pl_rewards: batch_r}
    loss, _ = sess.run([op_loss, op_optimize], feed_dict=feed_dict)

    return loss

def test(agent, env, round):
    agent.clear_memory()
    env.reset()

    for _ in range(round):
        a, q_estimation = agent.forward()
        print(a, q_estimation)



if __name__ == '__main__':
    main()
