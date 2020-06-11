import collections
import os
import random
import math
import pdb
from mdp import *
import time


# -------------------
# Inference with VI
# -------------------
def valueIteration(mdp):
	V = collections.defaultdict(float)

	def Q(s,a):
		return sum([proba*(reward + mdp.discount()*V[sp]) for sp,proba,reward in mdp.succProbReward(s,a)])

	step = 1
	while True:
		start = time.time()
		newV = collections.defaultdict(float)
		for s in mdp.states():
			if mdp.isEnd(s)[0]:
				newV[s] = 0.
			else:
				newV[s] = max([Q(s,a) for a in mdp.actions(s)])
		residual = max([abs(V[s]-newV[s]) for s in mdp.states()])
		if residual < 1e-10:
			break
		V = newV

		# read out policy
		pi = {}
		for s in mdp.states():
			if mdp.isEnd(s)[0]:
				pi[s] = 'none'
			else:
				pi[s] = max([(Q(s,a), a) for a in mdp.actions(s)])[1]

		end = time.time()
		# print results
		os.system('clear')
		print('{:20} {:20} {:20}'.format('s', 'V(s)', 'pi(s)'))
		s_num = 0
		for s in mdp.states():
			#print('{:>0} {:>20} {:>20}'.format(s, V[s], pi[s]))
			print('{} {} {}'.format(s_num, V[s], pi[s]))
			s_num += 1
		print("Iter {} of VI took {} sec: residual = {}".format(step, end-start, residual))
		step += 1
		#input()
	print("VI done")


#mdp = TransportationMDP(N=10, tram_fail=0.5)
#print(mdp.actions(3))
#print(mdp.succProbReward(3, 'walk'))
#print(mdp.succProbReward(3, 'tram'))

#mdp = TransportationMDP(N=10)

mdp = ActMDP() # with a partially expanded tree (depth of 6)
valueIteration(mdp)
