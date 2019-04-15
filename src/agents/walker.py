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
        self.policy = tf.nn.softmax(self.build_NN())

    def move(self, state):
        """
        from state to action
        state: [1,2,3]
        """
        # action
        policy = self.eval(state)
        action = np.argmax(policy)

        self.dict_actionCount[state.idx][action] += 1

        return action

    def eval(self, state):
        dict_feed = {self.pl_state: np.array([state.data], dtype=np.int32)}
        policy = self.sess.run(self.policy, feed_dict=dict_feed)[0]

        return policy

    def build_inputs(self):
        self.pl_state = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.pl_pi = tf.placeholder(shape=[None, None], dtype=tf.float32)

    def build_NN(self, reuse=False):
        '''
        self.pl_state: [batch, num]
        x: [batch, num, size]

        policy: [batch, size]
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

            logit = tf.layers.dense(
                inputs=x,
                units=self.size_action,
                activation=None,
                use_bias=False,
                name='fully_connected')

        return logit

    def compute_loss(self):

        logit = self.build_NN(reuse=True)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logit, labels=self.pl_pi)

        return loss
