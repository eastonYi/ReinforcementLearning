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
    lr = 0.000001
    size_state = 8
    env = ENV(size=10, size_state=size_state)
    agent = Agent(size_action=10, size_window=size_state, num_cell_units=500, num_layers=5, UCB_param=1.0)
    op_loss, op_optimize = optimize(agent, lr)
    collector = Sample_Collector(1000)

    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    agent.sess = sess

    for t in range(1000):
        reward = 0
        env.reset()
        s = env.s.copy()
        for i in range(round):
            a, q, u = agent.forward(s)
            r, _s = env.step(a)
            if i < 2:
                collector((s, a, r, q, u))
            s = _s
            reward += r

        if len(collector) > 100:
            # import pdb; pdb.set_trace()
            loss = training(sess, collector, agent, op_loss, op_optimize)
            print('reward: {:.3f}'.format(reward))
            # [print(i) for i in collector.pool]
            # print('loss: {:.3f}, reward: {:.3f}'.format(loss, reward))
            # print(collector.pool[0])
            # test(agent, env, round)
    # print('env potatial: {:.2f}, agent gets: {:.2f}'.format(sum(env.reveal()), sum(reward)))
            test(sess, agent, env, round)

def optimize(agent, lr):
    op_loss = tf.reduce_sum(agent.compute_loss())
    optimizer = tf.train.GradientDescentOptimizer(lr)
    gradients = optimizer.compute_gradients(op_loss)
    op_optimize = optimizer.apply_gradients(gradients)

    return op_loss, op_optimize


def training(sess, collector, agent, op_loss, op_optimize):
    batch_s, batch_a, batch_r = collector.get_batch(100)
    feed_dict = {agent.pl_state: batch_s,
                 agent.pl_reward: batch_r,
                 agent.pl_action: batch_a}
    loss, _ = sess.run([op_loss, op_optimize], feed_dict=feed_dict)

    return loss

def test(sess, agent, env, round):
    # s = [10, 10, 10, 10, 10, 10, 10, 10]
    # a, q, u = agent.forward(s)
    # print(s, a, q)
    #
    # s = [10, 10, 10, 10, 10, 10, 10, 1]
    # a, q, u = agent.forward(s)
    # print(s, a, q)

    batch_s = np.array([[10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 10, 10, 10, 10, 10, 10, 1]], dtype=np.float32)
    batch_r = np.array([100, 1], dtype=np.float32)
    batch_a = np.array([1, 1], dtype=np.float32)
    feed_dict = {agent.pl_state: batch_s,
                 agent.pl_reward: batch_r,
                 agent.pl_action: batch_a}
    loss, q, x = sess.run([agent.compute_loss(), agent.q_estimation, agent.x], feed_dict=feed_dict)
    print(loss, q)

    # s = [10, 10, 10, 10, 10, 10, 1, 2]
    # a, q_estimation = agent.forward(s)
    # print(s, a, q_estimation)


if __name__ == '__main__':
    main()
