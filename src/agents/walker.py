import numpy as np
import random
import tensorflow as tf
from collections import defaultdict

from .agent import Agent
from tfModels.layers import make_multi_cell, dense
from tfTools.tfTools import state2tensor, tensor2state

np.set_printoptions(precision=2)


class Walker(Agent):
    def __init__(self, size_action, size_window, sess=None, epsilon=0, num_cell_units=10, num_layers=3, UCB_param=None, name='walker'):
        self.dict_actionCount = defaultdict(lambda: np.zeros([size_action], dtype=np.int32))
        super().__init__(size_action, UCB_param, epsilon, name)

        self.num_cell_units = num_cell_units
        self.num_layers = num_layers
        self.size_window = size_window
        self.size_embedding = size_action+1
        self.build_inputs()
        self.q_estimation = self.build_NN()

    def forward(self, state):
        """
        from state to action
        state: [1,2,3]
        """
        # action
        dict_feed = {self.pl_state: np.array([state], dtype=np.int32)}
        q_estimation = self.sess.run(self.q_estimation, feed_dict=dict_feed)[0]

        state_str = ''.join(map(str, state))
        if self.UCB_param:
            UCB_estimation = q_estimation + \
                self.UCB_param * np.sqrt(np.log(np.sum(self.dict_actionCount[state_str])+1.1)/(self.dict_actionCount[state_str]+1e-10))
            action = np.argmax(UCB_estimation)
        else:
            if random.random() < self.epsilon:
                action = np.argmax(q_estimation)
            else:
                action = random.randint(0, self.size_action-1)

        self.dict_actionCount[state_str][action] += 1

        return action, q_estimation, UCB_estimation

    def build_inputs(self):
        self.pl_state = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.pl_reward = tf.placeholder(shape=[None], dtype=tf.float32)
        self.pl_action = tf.placeholder(shape=[None], dtype=tf.int32)

    def build_NN(self, reuse=False):
        '''
        self.pl_state: [batch, num]
        x: [batch, num, size]

        q_estimation: [batch, size]
        '''
        batch_size = tf.shape(self.pl_state)[0]
        x = tf.one_hot(self.pl_state, self.size_embedding, dtype=tf.float32)
        x = tf.reshape(x, [batch_size, self.size_window * self.size_embedding])
        self.x = x

        with tf.variable_scope(self.name, reuse=reuse):
            for i in range(self.num_layers):
                x = tf.layers.dense(
                    inputs=x,
                    units=self.num_cell_units,
                    activation=tf.nn.relu,
                    use_bias=True,
                    name='dense_{}'.format(i))

            q_estimation = tf.layers.dense(
                inputs=x,
                units=self.size_action,
                activation=None,
                use_bias=False,
                name='fully_connected')

        return q_estimation

    def compute_loss(self):

        q_estimation = self.build_NN(reuse=True)

        size_batch = tf.shape(q_estimation)[0]
        predicts = tf.gather_nd(q_estimation, tf.stack([tf.range(size_batch), self.pl_action], -1))

        loss = tf.pow(predicts - self.pl_reward, 2)

        return loss
