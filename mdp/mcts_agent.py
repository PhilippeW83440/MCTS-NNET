import collections
import os
import random
import math
import pdb
import time
from mdp import *
import dqn_agent as dqn

import argparse

## Inference (Algorithms)

class MctsAgent():
	def __init__(self, mdp, piRollout, depth=12, iters=100, explorConst=1.0, tMaxRollouts=200, reuseTree=True, nnet=None):
		self.depth = depth
		self.iters = iters
		self.c = explorConst # param controlling amount of exploration
		self.tMax = tMaxRollouts # max steps used to estimate Qval via rollout
		self.reuseTree = reuseTree
		self.mdp = mdp
		self.nnet = nnet
		self._resetTree()
		self.pi0 = piRollout

	def _resetTree(self):
		self.Tree = set()
		self.Nsa = {}
		self.Ns = {}
		self.Q = {}

	def act(self, state):
		if self.reuseTree is False:
			self._resetTree()
		action = self._selectAction(state, self.depth, self.iters, self.c)
		#dqn_action = self.nnet.act(state)
		#if dqn_action != action:
		#	pdb.set_trace()
		return action

	def _selectAction(self, s, d, iters, c):
		for _ in range(iters):
			self._simulate(s, d, self.pi0)
		q, action = max([(self.Q[(s,a)], a) for a in self.mdp.actions(s)]) # argmax
		print("q={}".format(q))
		return action

	def _simulate(self, s, d, pi0):
		if self.mdp.isEnd(s)[0]:
			return 0
		if d == 0: # we stop exploring the tree, just estimate Qval here
			return self._rollout(s, self.tMax, pi0)
		if s not in self.Tree:
			if self.nnet is not None:
				for a in self.mdp.actions(s):
					self.Nsa[(s,a)], self.Ns[s], self.Q[(s,a)] =  1, 1, self.nnet.getQ(s, a)
			else:
				for a in self.mdp.actions(s):
					self.Nsa[(s,a)], self.Ns[s], self.Q[(s,a)] =  0, 1, 0. # could use expert knowledge as well
			self.Tree.add(s)
			# use tMax instead of d: we want to rollout deeper
			return self._rollout(s, self.tMax, pi0)

		#a = max([(self.Q[(s,a)]+self.c*math.sqrt(math.log(self.Ns[s])/(1e-5 + self.Nsa[(s,a)])), a) for a in self.mdp.actions(s)])[1] # argmax
		qa_tab = ([(self.Q[(s,a)]+self.c*math.sqrt(math.log(self.Ns[s])/(1e-5 + self.Nsa[(s,a)])), a) for a in self.mdp.actions(s)]) # argmax
		qbest, a = max(qa_tab)
		qworst, _ = min(qa_tab)
		if abs(qbest - qworst) <= .1 or qbest <= -.55: # use exploration constant 1
			#pdb.set_trace()
			a = max([(self.Q[(s,a)] + 0.35 * math.sqrt(math.log(self.Ns[s])/(1e-5 + self.Nsa[(s,a)])), a) for a in self.mdp.actions(s)])[1] # argmax
		sp, r = self.mdp.sampleSuccReward(s, a)
		q = r + self.mdp.discount() * self._simulate(sp, d-1, pi0)
		self.Nsa[(s,a)] += 1
		self.Ns[s] += 1
		self.Q[(s,a)] += (q-self.Q[(s,a)])/self.Nsa[(s,a)]
		return q

	def _rollout(self, s, d, pi0):
		if self.mdp.isEnd(s)[0]:
			return 0
		elif self.nnet is not None:
			return self.nnet.getV(s)
		elif d == 0:
			return 0
		else:
			a = pi0(s)
			sp, r = self.mdp.sampleSuccReward(s, a)
			return r + self.mdp.discount() * self._rollout(sp, d-1, pi0)

