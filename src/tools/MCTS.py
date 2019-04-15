import numpy as np
import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

class Node():
	def __init__(self, state):
		self.state = state
		self.id = state.id
		self.edges = []

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True


class Edge():
	def __init__(self, inNode, outNode, prior, action):
		self.id = inNode.state.id + '|' + outNode.state.id
		self.inNode = inNode
		self.outNode = outNode
		self.action = action
		self.stats =  {'N': 0, 'W': 0, 'Q': 0, 'P': prior}


class MCTS():
	def __init__(self, root, cpuct):
		self.root = root
		self.nodes = {}
		self.cpuct = cpuct
		self.addNode(root)

	def __len__(self):
		return len(self.nodes)

	def moveToLeaf(self, env):
		breadcrumbs = []
		currentNode = self.root

		done = False
		value = 0
		while (not currentNode.isLeaf()) and (not env.isEnd()):
			maxQU = -99999

			Nb = 1+1e-5
			for action, edge in currentNode.edges:
				Nb += edge.stats['N']

			for idx, (action, edge) in enumerate(currentNode.edges):

				U = self.cpuct * edge.stats['P'] * np.sqrt(Nb/(edge.stats['N']+0.001))
				Q = edge.stats['Q']

				if Q + U > maxQU:
					maxQU = Q + U
					simulationAction = action
					simulationEdge = edge
			value, newState, done = env.step(simulationAction)
			# logging.info("\tenv.cur_state: {}, done: {}".format(env.cur_state, done))
			currentNode = simulationEdge.outNode
			breadcrumbs.append(simulationEdge)

		return currentNode, value, done, breadcrumbs, env

	def backup(self, leaf, value, breadcrumbs):
		for edge in breadcrumbs:
			edge.stats['N'] += 1
			edge.stats['W'] += value
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

	def addNode(self, node):
		self.nodes[node.id] = node
