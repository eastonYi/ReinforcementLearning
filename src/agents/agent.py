'''
the template of agents
'''
import numpy as np
import random
from abc import ABCMeta, abstractmethod
import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

from src.tools import MCTS


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, size_action, UCB_param, epsilon, name):
        self.size_action = size_action
        self.UCB_param = UCB_param
        self.epsilon = epsilon
        self.name = name
        self.MCTree = None

    def __call__(self):
        return self.forward()

    @abstractmethod
    def move(self):
        """
        """

    @abstractmethod
    def eval(self):
        """
        """

    def exploration(self):
        return random.randint(0, self.size_action-1)

    def exploitation(self, prob):
        return np.random.multinomial(1, prob)

    def move_with_MCTS(self, state, env, num, tau):
        # logging.info('move with MCTS ...')
        if self.MCTree == None or state.id not in self.MCTree.nodes.keys():
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        self.simulate(num, env)

        pi, values = self.getResults(tau=tau)
        action = np.argmax(pi)

        return action, pi, values

    def buildMCTS(self, state):
        # logging.info('build MCTS ...')
        self.root = MCTS.Node(state)
        self.MCTree = MCTS.MCTS(self.root, cpuct=1.0)

    def changeRootMCTS(self, state):
        self.MCTree.root = self.MCTree.nodes[state.id]

    def simulate(self, num, env):
        env_state = env.save_state()
        for i in range(num):
            env.load_state(env_state)
            # logging.info('simulate {}-th from {} ...'.format(i, env.cur_state))
            leafNode, value, done, breadcrumbs, env = self.MCTree.moveToLeaf(env)
            # logging.info("\tmoveToLeaf: env.state_{}, done: {}".format(env.cur_state, done))
            if not done:
                # logging.info("\tevaluateLeaf: env.state_{}".format(env.cur_state))
                self.evaluateLeaf(leafNode, env)
            self.MCTree.backup(leafNode, value, breadcrumbs)

    def evaluateLeaf(self, leafNode, env):
        probs = self.get_preds(leafNode.state)
        env_state = env.save_state()
        for action in range(self.size_action):
            env.load_state(env_state)
            r, newState, done = env.step(action)
            if newState.id not in self.MCTree.nodes.keys():
                node = MCTS.Node(newState)
                self.MCTree.addNode(node)
            else:
                node = self.MCTree.nodes[newState.id]

            newEdge = MCTS.Edge(leafNode, node, probs[action], action)
            leafNode.edges.append((action, newEdge))

        return

    def get_preds(self, state):
        #predict the leaf
        policy = self.eval(state)

        # allowedActions = state.allowedActions()
        #
        # mask = np.ones(policy.shape, dtype=bool)
        # mask[allowedActions] = False
        # policy[mask] = 0.0

        return policy

    def getResults(self, tau):
        edges = self.MCTree.root.edges
        pi = np.zeros(self.size_action, dtype=np.float32)
        values = np.zeros(self.size_action, dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1/tau)
            values[action] = edge.stats['Q']

        pi /= np.sum(pi)
        return pi, values
