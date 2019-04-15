import numpy as np
import tensorflow as tf
import random
import time
import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

from src.agents.walker import Walker as Agent
from src.envs.MDP import MDP as ENV
from src.tools.collector import Sample_Collector
from src.tools.MCTS import MCTS


np.set_printoptions(precision=1)
random.seed(0)
np.random.seed(0)

def main():
    round = 7
    num_simulation = 50
    lr = 0.001
    num_state = 8
    env = ENV(size=10, num_state=num_state)
    agent = Agent(size_action=10, size_window=num_state, num_cell_units=500, num_layers=5, UCB_param=1.0)
    op_loss, op_optimize = optimize(agent, lr)
    collector = Sample_Collector(100)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    agent.sess = sess

    for t in range(2000):
        reward = 0
        env.reset()
        s = env.get_state()
        for i in range(round):
            env_state = env.save_state()
            a, pi, q = agent.move_with_MCTS(s, env, num=num_simulation, tau=0.5)
            env.load_state(env_state)
            # logging.info("pi: {}".format(pi))
            r, _s, done = env.step(a)
            # print(s.data, a, r)
            collector((s, a, r, pi, q))
            s = _s
            reward += r

        if len(collector) > 10:
            # import pdb; pdb.set_trace()
            loss = training(sess, collector, agent, op_loss, op_optimize)
            print('reward: {:.3f}, loss: {:.3f}'.format(reward, loss))
            print(collector.pool[0]['state'].data)
            # time.sleep(1)
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
    batch_s, batch_pi = collector.get_batch(10)
    feed_dict = {agent.pl_state: batch_s,
                 agent.pl_pi: batch_pi}
    loss, _ = sess.run([op_loss, op_optimize], feed_dict=feed_dict)

    return loss

def test(sess, agent, env, round):
    # s = [10, 10, 10, 10, 10, 10, 10, 10]
    # a, q, u = agent.move(s)
    # print(s, a, q)
    #
    # s = [10, 10, 10, 10, 10, 10, 10, 1]
    # a, q, u = agent.move(s)
    # print(s, a, q)

    batch_s = np.array([[10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 10, 10, 10, 10, 10, 10, 1],
                        [10, 10, 10, 10, 10, 10, 1, 2],
                        [10, 10, 10, 10, 10, 1, 2, 3]], dtype=np.float32)
    feed_dict = {agent.pl_state: batch_s}
    policy = sess.run(agent.policy, feed_dict=feed_dict)
    print(policy)

    # s = [10, 10, 10, 10, 10, 10, 1, 2]
    # a, q_estimation = agent.move(s)
    # print(s, a, q_estimation)


if __name__ == '__main__':
    '''
    reward: 700.000, loss: 8.992
    (10, 10, 1, 2, 3, 4, 5, 6)
    [[0.  0.3 0.2 0.1 0.1 0.1 0.1 0.1 0.  0. ]
     [0.  0.2 0.3 0.1 0.1 0.1 0.1 0.1 0.  0. ]
     [0.  0.1 0.1 0.3 0.1 0.1 0.1 0.1 0.  0. ]
     [0.  0.1 0.1 0.1 0.4 0.1 0.1 0.1 0.  0. ]]
    '''
    main()
