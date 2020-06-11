import numpy as np
from collections import deque
from mdp import *
from dqn_agent import *
import pdb

import argparse
import logging
import utils_nn as utils

def train_dqn(mdp, args, n_epochs=20, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

	#agent = Agent(7, mdp.action_size(), mdp.discount(), args, seed=0)
	#agent = DqnAgent(mdp, args, seed=0, model_state='reduce')
	if args.nn == 'cnn':
		agent = DqnAgent(mdp, args, seed=0, model_state='grid')
	else:
		agent = DqnAgent(mdp, args, seed=0, model_state='project')

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
				a = agent.act(s, eps)
				sp, r = mdp.sampleSuccReward(s, a)
				done = mdp.isEnd(sp)[0]
				l = agent.step(s, a, r, sp, done)
				if l is not None:
					avg_tr_loss += l.item()
				ttc, _ = mdp._get_smallest_TTC(sp)
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
			#print(num_s)
			score = 0
			for t in range(max_t):
				a = agent.act(s, eps=0.)
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
			agent.save(num_epoch, dev_mean_score)
			best_score = dev_mean_score

utils.set_logger('logs/train.log')

# run python3 dqn.py or python3 dqn.py --restore best or python3 dqn.py -nn cnn
parser = argparse.ArgumentParser()
parser.add_argument('--nn', default='dnn', help="dnn or cnn")
parser.add_argument('--restore', default=None, help="Optional, file in models containing weights to reload before training")

args = parser.parse_args()

mdp = ActMDP()
train_dqn(mdp, args)
