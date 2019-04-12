import numpy as np
import random
import tensorflow as tf

from .agent import Agent
from tfModels.layers import make_multi_cell, dense
from tfTools.tfTools import state2tensor, tensor2state

np.set_printoptions(precision=2)


class Walker(Agent):
    def __init__(self, size_action, sess=None, epsilon=0, num_cell_units=10, num_layers=3, UCB_param=None, name='walker'):
        self.action_count = np.zeros([size_action], dtype=np.int32)
        self.pre_action = size_action
        super().__init__(size_action, UCB_param, epsilon, name)

        self.num_cell_units = num_cell_units
        self.num_layers = num_layers
        self.cell = self.build_cell()
        self.build_inputs()
        self.q_estimation, self.new_memory = self.build_NN()

    def clear_memory(self):
        self.memory = np.zeros([1, 2*self.num_layers, self.num_cell_units], np.float32)
        self.pre_action = self.size_action # start action id

    def forward(self):
        """
        from state to action
        """
        # action
        dict_feed = {self.pl_memory: self.memory,
                     self.pl_action: np.array([self.pre_action], dtype=np.float32)}
        q_estimation, self.memory = \
            self.sess.run([self.q_estimation, self.new_memory], feed_dict=dict_feed)


        if self.UCB_param:
            UCB_estimation = q_estimation + \
                self.UCB_param * np.sqrt(np.log(np.sum(self.action_count)+1.1) / (self.action_count+1e-10))
            action = np.argmax(UCB_estimation)
        else:
            if random.random() < self.epsilon:
                action = np.argmax(q_estimation)
            else:
                action = random.randint(0, self.size_action-1)

        self.pre_action = action

        return action, q_estimation

    def build_inputs(self):
        self.pl_action = tf.placeholder(shape=[1], dtype=tf.int32)
        self.pl_memory = tf.placeholder(shape=[None, 2*self.num_layers, self.num_cell_units], dtype=tf.float32)
        self.pl_actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.pl_lengths = tf.placeholder(shape=[None], dtype=tf.int32)
        self.pl_rewards = tf.placeholder(shape=[None, None], dtype=tf.float32)

    def build_cell(self):
        cell = make_multi_cell(
            num_cell_units=self.num_cell_units,
            num_layers=self.num_layers,
            is_train=True,
            keep_prob=0.9)

        return cell

    def build_NN(self):

        with tf.variable_scope(self.name):
            hidden_output, memory = tf.contrib.legacy_seq2seq.rnn_decoder(
                decoder_inputs=[tf.one_hot(self.pl_action, self.size_action+1, dtype=tf.float32)],
                initial_state=tensor2state(self.pl_memory),
                cell=self.cell)
            cur_logit = tf.layers.dense(
                inputs=hidden_output[0],
                units=self.size_action,
                activation=None,
                use_bias=False,
                name='fully_connected')

        return cur_logit, state2tensor(memory)

    def compute_loss(self):
        with tf.variable_scope(self.name, reuse=True):
            hidden_output, _ = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=tf.one_hot(self.pl_actions, self.size_action+1, dtype=tf.float32),
                sequence_length=self.pl_lengths,
                dtype=tf.float32)
            q_estimations = tf.layers.dense(
                inputs=hidden_output,
                units=self.size_action,
                activation=None,
                use_bias=False,
                name='fully_connected')

            size_batch, len_time = tf.shape(q_estimations)[0], tf.shape(q_estimations)[1]

            n = tf.range(size_batch, dtype=tf.int32)[:, None]
            index_batch = tf.tile(n, [1, len_time])

            m = tf.range(len_time, dtype=tf.int32)[None, :]
            index_time = tf.tile(m, [size_batch, 1])

            predicts = tf.gather_nd(q_estimations, tf.stack([index_batch, index_time, self.pl_actions], -1))
            loss = tf.pow(self.pl_rewards - predicts, 2)
            loss = tf.reduce_sum(loss)

        return loss
