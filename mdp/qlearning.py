import numpy as np
import random
from collections import namedtuple, deque, defaultdict
import matplotlib.pyplot as plt
from mdp import *
import pdb
import util

import json
import pprint
import logging
import utils_nn as utils

BUFFER_SIZE = int(1e4)
BATCH_SIZE = 1
LR = 1e-3

class ReplayBuffer:
	def __init__(self, buffer_size, batch_size, seed):
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.seed = random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		e = (state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		experiences = random.sample(self.memory, k=self.batch_size)
		return experiences

	def __len__(self):
		return len(self.memory)

# Performs Q-learning.	Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
	def __init__(self, actions, discount, featureExtractor, mdp, explorationProb=0.2):
		self.actions = actions
		self.discount = discount
		self.featureExtractor = featureExtractor
		self.explorationProb = explorationProb
		self.weights = defaultdict(float)
		self.numIters = 0
		self.mdp = mdp

	# Return the Q function associated with the weights and features
	def getQ(self, state, action):
		score = 0
		for f, v in self.featureExtractor(state, action, self.mdp):
			score += self.weights[f] * v
		return score

	# This algorithm will produce an action given a state.
	# Here we use the epsilon-greedy algorithm: with probability
	# |explorationProb|, take a random action.
	def getAction(self, state, eps):
		self.numIters += 1
		#if random.random() < self.explorationProb:
		if random.random() < eps: # align qlearning and dqn exploration strategy
			return random.choice(self.actions(state))
		else:
			return max((self.getQ(state, action), action) for action in self.actions(state))[1]

	# Call this function to get the step size to update the weights.
	def getStepSize(self):
		return LR
		return 1e-4 / math.sqrt(self.numIters)

	# We will call this function with (s, a, r, s'), which you should use to update |weights|.
	# Note that if s is a terminal state, then s' will be None.  Remember to check for this.
	# You should update the weights using self.getStepSize(); use
	# self.getQ() to compute the current estimate of the parameters.
	def incorporateFeedback(self, state, action, reward, newState, done=False):
		if newState is None or done:
			error = self.getQ(state, action) - reward
		else:
			error = self.getQ(state, action) - (reward + self.discount * max([self.getQ(newState, a) for a in self.actions(newState)]))
		loss = error
		#print("error={}".format(error))
		error = min(10, error)
		error = max(-10, error)
		error *= self.getStepSize()
		for f, v in self.featureExtractor(state, action, self.mdp):
			self.weights[f] = self.weights[f] - error * v
		return loss

	def dumpWeights(self):
		pprint.pprint(json.loads(json.dumps(self.weights)), weightsFile)
		#print(dict(self.weights))

def actFeatureExtractor(state, action, mdp):
	features = []

	order = 1 # polynomial approx

	dmax = 200
	vmax = 30
	amax = 2
	ttcmax = 100

	pos, speed, ttc_info = state[1], state[3], mdp._get_smallest_TTC(state)
	ttc, nobj = ttc_info
	idx = 4+nobj*4
	ttcX, ttcY, ttcVx, ttcVy = state[idx:idx+4]
	ttcX, ttcY, ttcVx, ttcVy = ttcX/dmax, ttcY/dmax, ttcVx/vmax, ttcVy/vmax

	features.append(('bias', 1))

	# NB: trying to play with these features. I had to lower donw the learning rate (cf LR)
	#for i in range(1,order+1):
	#	features.append(('ttcX'+str(i), ttcX**i))
	#	features.append(('ttcY'+str(i), ttcY**i))
	#	features.append(('ttcVx'+str(i), ttcVx**i))
	#	features.append(('ttcVy'+str(i), ttcVy**i))

	#features.append(('ttcR', 1 - math.exp(-ttc/100.)))
	#features.append(('speedR', 1 - abs((speed-20.)/20.)))

	# normalize features, otherwise it does not work at all
	ttc = min(ttc,ttcmax)
	pos, speed, ttc, action = pos/dmax, speed/vmax, ttc/ttcmax, action/amax

	for i in range(1,order+1):
		#features.append(('pos'+str(i), pos**i))
		features.append(('speed'+str(i), speed**i))
		features.append(('ttc'+str(i), ttc**i))
		features.append(('action'+str(i), action**i))

	return features



def qlearning(mdp, n_epochs=20, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

	rl = QLearningAlgorithm(mdp.actions, mdp.discount(), actFeatureExtractor, mdp, 0.2)
	memory = ReplayBuffer(BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0)

	best_score = -math.inf

	mean_score = -math.inf
	avg_tr_loss = 0
	eps = eps_start
	iters = 0
	for num_epoch in range(n_epochs):
		random.shuffle(mdp.train_set)
		tr_scores_window = deque(maxlen=100) # last 100 scores
		for num_s, s in enumerate(mdp.train()):
			score = 0
			for t in range(max_t):
				iters += 1
				#a = agent.act(mdp.reduce_state(s), eps) # a is an index !!!
				a = rl.getAction(s, eps)
				sp, r = mdp.sampleSuccReward(s, a)
				done = mdp.isEnd(sp)[0]

				memory.add(s, a, r, sp, done)
				if len(memory) > BATCH_SIZE:
					samples = memory.sample()
					for sample in samples:
						state, action, reward, next_state, isDone = sample
						l = rl.incorporateFeedback(state, action, reward, next_state, isDone)
				else:
					l = rl.incorporateFeedback(s, a, r, sp, done)
				avg_tr_loss += l

				score += r
				if done:
					break
				s = sp
				if iters%100 == 99:
					logging.info("Epoch no {}: sample {} iter {} avg_tr_loss: {:0.4f} tr_mean_score: {:.2f}".format(num_epoch, num_s, iters, avg_tr_loss/100, mean_score))
					avg_tr_loss = 0
			tr_scores_window.append(score)
			mean_score = np.mean(tr_scores_window)
			eps = max(eps_end, eps_decay*eps)

		dev_scores_window = deque(maxlen=100) # last 100 scores
		for num_s, s in enumerate(mdp.dev()):
			score = 0
			for t in range(max_t):
				#a = agent.act(mdp.reduce_state(s), eps=0.) # a is an index !!!
				a = rl.getAction(s, eps)
				sp, r = mdp.sampleSuccReward(s, a)
				done = mdp.isEnd(sp)[0]
				score += r
				if done:
					break
				s = sp
			dev_scores_window.append(score)
		dev_mean_score = np.mean(dev_scores_window)
		logging.info("Epoch no {}: dev_mean_score: {:.2f}".format(num_epoch, dev_mean_score))
		if dev_mean_score > best_score:
			weightsFile.write("Epoch {} dev_mean_score: {:.2f}\n".format(num_epoch, dev_mean_score))
			rl.dumpWeights()
			best_score = dev_mean_score


	# scores_window = deque(maxlen=100) # last 100 scores
	# eps = eps_start
	# for i_episode in range(1, n_episodes+1):
	# 	s = mdp.startState()
	# 	score = 0
	# 	for t in range(max_t):
	# 		#a = agent.act(s, eps)
	# 		a = rl.getAction(s, eps)

	# 		#pdb.set_trace()
	# 		sp, r = mdp.sampleSuccReward(s, a)
	# 		done = mdp.isEnd(sp)[0]

	# 		#agent.step(s, a, r, sp, done)
	# 		memory.add(s, a, r, sp, done)
	# 		if len(memory) > BATCH_SIZE:
	# 			samples = memory.sample()
	# 			for sample in samples:
	# 				state, action, reward, next_state, isDone = sample
	# 				rl.incorporateFeedback(state, action, reward, next_state, isDone)
	# 		else:
	# 			rl.incorporateFeedback(s, a, r, sp, done)

	# 		score += r
	# 		if done:
	# 			break
	# 		s = sp
	# 	scores_window.append(score)
	# 	eps = max(eps_end, eps_decay*eps)
	# 	avg_sliding_score = np.mean(scores_window)
	# 	print("Episode {} Average sliding score: {:.2f}".format(i_episode, avg_sliding_score))
	# 	if avg_sliding_score > -10:
	# 		weightsFile.write("Episode {} Average sliding score: {:.2f}\n".format(i_episode, avg_sliding_score))
	# 		rl.dumpWeights()

utils.set_logger('qlearning.log')

weightsFile = open("models/qlearning.weights", "a")
mdp = ActMDP()
qlearning(mdp)
