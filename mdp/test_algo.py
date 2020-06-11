# Misc
import numpy as np
import pdb
import argparse
import logging
import utils_nn as utils
import matplotlib.pyplot as plt
import time
from matplotlib import collections as mc
import os, imageio
from scipy.interpolate import interp1d

# MDP
from mdp import *

# Agents
from baseline_agent import *
from dqn_agent import *
from mcts_agent import *
from mpc_agent import *
from human_agent import *


def plot(step_to_goal, steps_to_coll, algo='Baseline'):
	fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
	# We can set the number of bins with the `bins` kwarg
	axs[0].set_title(algo)
	axs[0].hist(steps_to_coll)
	axs[0].set_ylabel('Number')
	axs[0].set_xlabel('Steps to collision')

	axs[1].set_title(algo)
	axs[1].hist(step_to_goal)
	axs[1].set_ylabel('Number')
	axs[1].set_xlabel('Steps to goal')
	plt.show()



def test_algo(mdp, args, max_t=1000):
	if args.algo == 'dqn':
		logging.info("agent: DqnAgent(mdp, args, seed=0)")
		if 'P3' in args.restore:
			agent = DqnAgent(mdp, args, seed=0, model_state='project')
		else:
			agent = DqnAgent(mdp, args, seed=0)
	elif args.algo == 'mcts':
		logging.info(
			"agent: MctsAgent(mdp, mdp.pi0, depth=12, iters=100, "
			"explorConst=1.0, tMaxRollouts=200, reuseTree=False, nnet=None)")
		agent = MctsAgent(mdp, mdp.pi0, depth=12, iters=100, explorConst=1.0,
						  tMaxRollouts=200, reuseTree=False, nnet=None)
	elif args.algo == 'mcts-nnet':
		if 'P3' in args.restore:
			pdb.set_trace()
			dqn = DqnAgent(mdp, args, seed=0, model_state='project')
		else:
			dqn = DqnAgent(mdp, args, seed=0)
		logging.info(
			"agent = MctsAgent(mdp, mdp.pi0, depth=24, iters=200, "
			"explorConst=0, tMaxRollouts=10, reuseTree=False, nnet=dqn)")
		#	 "agent: MctsAgent(mdp, mdp.pi0, depth=20, iters=100, "
		#	 "explorConst=0.25, tMaxRollouts=10, reuseTree=False, nnet=dqn)")
		#		agent = MctsAgent(mdp, mdp.pi0, depth=20, iters=100,
		#		explorConst=0.25, tMaxRollouts=10, reuseTree=False, nnet=dqn)
		agent = MctsAgent(mdp, mdp.pi0, depth=24, iters=200, explorConst=0.,
						  tMaxRollouts=10, reuseTree=True, nnet=dqn)
	elif args.algo == 'mpc':
		agent = MpcAgent(mdp)
	elif args.algo == 'human':
		agent = HumanAgent(mdp)
	else:
		agent = BaselineAgent(mdp)

	metric_scores = []
	metric_hardbrakes = []
	metric_steps_to_goal = []
	metric_steps_to_collision = []
	metric_speed_at_collision = []
	metric_runtime = []
	success = 0

	for num_s, s in enumerate(mdp.dev()):
		if args.test is not None and args.test != str(num_s+1):
			continue
		print("[TEST] Test {}".format(num_s + 1))
		mdp.resetCache()
		s = tuple(s)  # just to make it hashable
		score, hardbrakes, speed_at_collision = 0, 0, None
		sequence = []
		ttc = []
		runtimes = []
		for steps in range(1, max_t):
			start_act = time.time()
			a = agent.act(s)
			end_act = time.time()
			runtimes.append(end_act - start_act)
			sequence.append(a)
			if a <= -4.:
				hardbrakes += 1
			sp, r = mdp.sampleSuccReward(s, a)
			ttcp = round(mdp._get_smallest_TTC(sp)[0], 1)
			ttc_s = mdp._get_smallest_TTC(s)
			print("ego: act={}, ttc={}, s={}".format(a, ttc_s, s))
			if type(agent) is MctsAgent:
				q_values = [agent.Q[(s, aa)] for aa in mdp.actions(s)]
				print("Q values: {}".format(q_values))
			ttc.append(ttcp)
			done, info = mdp.isEnd(sp)
			score += r
			if done:
				if info == 'goal':
					success += 1
					metric_steps_to_goal.append(steps)
				elif info == 'collision':
					metric_steps_to_collision.append(steps)
					speed_at_collision = math.sqrt(sp[2]**2+sp[3]**2)
					metric_speed_at_collision.append(speed_at_collision)
				break
			if args.visu:
				mdp.visu(s, a, r, num_s+1, args.algo, steps, score)
			s = sp

		runtime = np.mean(runtimes)
		metric_runtime.append(runtime) # Time to make 1 decision (agent.act runtime)

		logging.info(
			"Test {}: score {:.3f} hardbrakes {} steps {} runtime {} collision_speed {}".format(
				num_s + 1, score, hardbrakes, steps, runtime, speed_at_collision))
		logging.info("	actions {}".format(sequence))
		logging.info("	ttc		{}".format(ttc))
		metric_scores.append(score)
		metric_hardbrakes.append(hardbrakes)

	logging.info(
		"METRICS mean values => score: {:.3f}, success_rate {:.2f}, runtime: "
		"{:.6f} sec, hardbrakes: {:.2f}, steps_to_goal: {:.2f}, "
		"steps_to_collision {:.2f}, "
		"speed_at_collision {:.2f}".format(
			np.mean(metric_scores), success / len(metric_scores), np.mean(metric_runtime),
			np.mean(metric_hardbrakes), np.mean(metric_steps_to_goal),
			np.mean(metric_steps_to_collision), np.mean(metric_speed_at_collision)))

	plot(metric_steps_to_goal, metric_steps_to_collision, args.algo)


## MAIN starts here ###

utils.set_logger('logs/test.log')
# Parser
# python3 test_algo.py
# python3 test_algo.py --algo dqn
parser = argparse.ArgumentParser()
parser.add_argument('--algo', default='baseline',
					help="baseline, qlearning, dqn, mcts, mcts-nnet, mpc, human")
parser.add_argument('--nn', default='dnn', help="dnn or cnn")
parser.add_argument('--restore', default=None,
					help="Optional, file in models containing weights to load")
parser.add_argument('--visu', default=False,
					help="Optional, used to debug and visualize tests")
parser.add_argument('--test', default=None,
					help="Optional, used to run a single test")
args = parser.parse_args()

if 'dqn' == args.algo or 'mcts-nnet' == args.algo:
	args.restore = 'dnn-0.42'  # best one so far
	args.restore = 'dnn-0.31'  # best one so far
	args.restore = 'dnnP3ch-0.38'  # best one so far
	#args.restore = 'dnnP3-0.31'  # best one so far
	args.restore = 'dnn-0.31'  # best one so far
	logging.info("dqn: dnn-0.31")

if 'mcts' == args.algo:
	mdp = ActMDP(restrict_actions=True)
else:
	mdp = ActMDP()

test_algo(mdp, args)
