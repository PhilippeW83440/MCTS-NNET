import collections
import os
import random
import math
import pdb
import numpy as np
import copy
import matplotlib.pyplot as plt


# Return i in [0, ..., len(probs)-1] with probability probs[i].
def sample(probs):
	target = random.random()
	accum = 0
	for i, prob in enumerate(probs):
		accum += prob
		if accum >= target: return i
	raise Exception("Invalid probs: %s" % probs)


### Model (MDP problem)

class TransportationMDP(object):
	def __init__(self, N=10, tram_fail=0.5, discount=1, sort=False):
		self.N = N
		self.tram_fail = tram_fail
		self.gamma = discount
		self.sort = sort

	def startState(self):
		return 1

	def isEnd(self, s):
		return s >= self.N, 'done'

	def states(self):
		return range(1, self.N+1)

	def actions(self, s):
		results = []
		if s+1 <= self.N:
			results.append('walk')
		if 2*s <= self.N:
			results.append('tram')
		return results

	def succProbReward(self, s, a):
		# returns sp, proba=T(s,a,sp), R(s,a,sp) 
		results = []
		if a == 'walk':
			results.append((s+1, 1., -1.))
		elif a == 'tram':
			results.append((2*s, 1. - self.tram_fail, -2.))
			results.append((s, self.tram_fail, -2.))
		return results

	def sampleSuccReward(self, s, a): # G(s,a) for mcts
		transitions = self.succProbReward(s, a)
		if self.sort: transitions = sorted(transitions)
		# sample a random transition
		i = sample([prob for newState, prob, reward in transitions])
		sp, prob, r = transitions[i]
		return (sp, r)

	def discount(self):
		return self.gamma

	def states(self):
		return range(1, self.N+1)

	def pi0(self, s):
		#print("pi0({})".format(s))
		return random.choice(self.actions(s))



def get_dist(obj1, obj2):
	return math.sqrt((obj1[0]-obj2[0])**2 + (obj1[1]-obj2[1])**2)

# Transition with Constant Acceleration model
def transition_ca(s, a, dt):
	Ts = np.matrix([[1.0, 0.0, dt,	0.0],
					[0.0, 1.0, 0.0, dt],
					[0.0, 0.0, 1.0, 0.0],
					[0.0, 0.0, 0.0, 1.0]])
	Ta = np.matrix([[0.5*dt**2, 0.0],
					[0.0, 0.5*dt**2],
					[dt, 0.0],
					[0.0, dt]])
	return np.dot(Ts, s) + np.dot(Ta, a)

# mean	= [x, y, vx, vy]
# sigma = [sigma_x, sigma_y, sigma_vx, sigma_vy]
def mvNormal(mean, sigma):
	return np.random.multivariate_normal(mean, np.diag(sigma))


class ActMDP(object): # Anti Collision Tests problem
	# actions are accelerations
	# Easier for qlearning.py (so we deal with something normalized between -1 and 1)
	def __init__(self, nobjs=10, dist_collision=10, dt=0.25, action_set=[-4., -2., -1., 0., +1., +2.], discount=1, restrict_actions=False, seed=0, single_crossing_point=False, acc_scenario=False, cross_scenario=False, intersection_scenario=False):
		self.intersection_scenario = intersection_scenario
		self.cross_scenario = cross_scenario
		self.acc_scenario = acc_scenario
		self.single_crossing_point = single_crossing_point
		self.seed = random.seed(seed)
		self.nobjs = nobjs
		self.dist_collision = dist_collision
		self.dt = dt
		self.action_set = action_set
		self.restrict_actions = restrict_actions # will restrict actions(s) to safe_actions(s)
		# x, y, vx, vy
		self.startEgo = np.array([100.0,	0.0,  0.0,		20.0], dtype=float)
		self.goal  = np.array([100.0, 200.0, 0.0, 0.0], dtype=float) # down from 200 to 50
		self.start = self._randomStartState()
		self.vdes = 20 # desired speed 20ms-1
		self.gamma = discount
		self.create_validation_sets()
		self.next_states = {} # SPEED OPTIM (works only for deterministic cases)
		self.restricted_actions = {} # SPEED OPTIM
		self.reachable_states = []
		#depth = 5
		#print("Expand MDP states: for a depth of {} ...".format(depth))
		#self._expand(self.start, depth)
		#print("Expand MDP states: expanded {} states".format(len(self.reachable_states)))

	def resetCache(self):
		self.next_states = {} # SPEED OPTIM (works only for deterministic cases)
		self.restricted_actions = {} # SPEED OPTIM

	def create_validation_sets(self):
		self.train_set = [self._randomStartState() for _ in range(10000)]
		self.dev_set = [self._randomStartState() for _ in range(100)]
		#self.train_set = [self._randomStartState() for _ in range(5*10000)]
		self.test_set = [self._randomStartState() for _ in range(100)]

	def train(self):
		return self.train_set
	def dev(self):
		return self.dev_set
	def test(self):
		return self.test_set

	# enforce a single crossing point
	def _singleCrossingPoint(self, s):
		ttc, crossobj = self._get_smallest_TTC(s)
		idx = 4
		for n in range(int((len(s)-4)/4)):
			if n != crossobj: # set speed to 0
				s[idx+2] = 0
				s[idx+3] = 0
			idx += 4
		return s

	def _randomStartState(self):
		state = copy.deepcopy(self.startEgo)
		if self.intersection_scenario:
			assert self.nobjs == 10
			for n in range(4): # from the right
				x = float(random.randint(150, 200))
				y = 102
				vx = -float(random.randint(10, 25))
				vy = 0
				obj = np.array([x, y, vx, vy])
				state = np.append(state, obj)
			for n in range(4): # from the left
				x = float(random.randint(0, 50))
				y = 98
				vx = float(random.randint(10, 25))
				vy = 0
				obj = np.array([x, y, vx, vy])
				state = np.append(state, obj)
			for n in range(2): # in front of ego
				x = 100
				y = float(random.randint(25, 100))
				vx = 0
				vy = float(random.randint(10, 15))
				obj = np.array([x, y, vx, vy])
				state = np.append(state, obj)
		elif self.cross_scenario:
			yy = np.array([97.5, 100.12, 101.43, 102.74, 105.35])
			for i in range(self.nobjs):
				x = float(random.randint(0, 50))
				vx = float(random.randint(10, 25))
				vy = 0
				y = yy[i % len(yy)]
				obj = np.array([x, y, vx, vy])
				state = np.append(state, obj)
		elif self.acc_scenario:
			for n in range(int(self.nobjs)):
				x = 100
				y = float(random.randint(25, 190))
				vx = 0
				vy = float(random.randint(10, 15))
				obj = np.array([x, y, vx, vy])
				state = np.append(state, obj)
		else:
			for n in range(int(self.nobjs/2)):
				x = float(random.randint(0, 50))
				y = float(random.randint(25, 190))
				vx = float(random.randint(10, 25))
				vy = float(random.randint(0, 5))
				obj = np.array([x, y, vx, vy])
				state = np.append(state, obj)

			for n in range(int(self.nobjs/2)):
				x = float(random.randint(150, 200))
				y = float(random.randint(25, 190))
				vx = - float(random.randint(10, 25))
				vy = - float(random.randint(0, 5))
				obj = np.array([x, y, vx, vy])
				state = np.append(state, obj)
			if self.single_crossing_point:
				state = self._singleCrossingPoint(state)
		return state

	def startState(self):
		return tuple(self.start) # to make it hashable

	def isEnd(self, s):
		if s[1] >= self.goal[1]:
			return True, 'goal'
		elif self._get_dist_nearest_obj(s) < self.dist_collision:
			return True, 'collision'
		else:
			return False, 'ongoing'

	###########################################
	# Hard Constraint w.r.t. Time To Collision
	###########################################
	def _get_TTC(self, ego, obj, radius):
		x1, y1, vx1, vy1 = ego[0], ego[1], ego[2], ego[3]
		x2, y2, vx2, vy2 = obj[0], obj[1], obj[2], obj[3]
		a = (vx1 - vx2) **2 + (vy1 - vy2) **2
		b = 2 * ((x1 - x2) * (vx1 - vx2) + (y1 - y2) * (vy1 - vy2))
		c = (x1 - x2) **2 + (y1 - y2) **2 - radius **2
		if a == 0 and b == 0:
			if c == 0:
				return 0
			else:
				return np.inf
		if a == 0 and b != 0:
			t = -c / b
			if t < 0:
				return np.inf
			else:
				return t
		delta = b **2 - 4 * a * c
		if delta < 0:
			return np.inf
		t1 = (-b - np.sqrt(delta)) / (2 * a)
		t2 = (-b + np.sqrt(delta)) / (2 * a)
		if t1 < 0:
			t1 = np.inf
		if t2 < 0:
			t2 = np.inf
		return min(t1, t2)

	def _get_smallest_TTC(self, s):
		radius = self.dist_collision
		ego = s[0:4]
		smallest_TTC = np.Inf
		smallest_TTC_obj = -1
		idx = 4
		for n in range(int((len(s)-4)/4)):
			obj = s[idx:idx+4]
			TTC = self._get_TTC(ego, obj, radius)
			if TTC < smallest_TTC:
				smallest_TTC = TTC
				smallest_TTC_obj = n
			idx += 4
		return smallest_TTC, smallest_TTC_obj

	def _get_sorted_TTC(self, s):
		ego = s[0:4]
		idx = 4
		sorted_ttcs = []
		for n in range(int((len(s)-4)/4)):
			obj = s[idx:idx+4]
			ttc = self._get_TTC(ego, obj, self.dist_collision)
			if ttc < 10:
				_, y_obj, _, vy_obj = obj
				ycol = y_obj + ttc * vy_obj
				sorted_ttcs.append((ycol, ttc))
			idx += 4
		sorted_ttcs.sort(key=lambda tup: tup[1])
		#pdb.set_trace()
		return sorted_ttcs


	def _sort_by_TTC(self, s):
		radius = self.dist_collision
		ego = s[0:4]
		objs = []
		idx = 4
		for n in range(int((len(s)-4)/4)):
			obj = s[idx:idx+4]
			TTC = self._get_TTC(ego, obj, radius)
			objs.append((TTC,obj))
			idx += 4
		#minTTC = min(objs)[0]
		objs.sort(key=lambda tup: tup[0])
		state = np.zeros_like(s)
		state[0:4] = s[0:4]
		for i in range(self.nobjs):
			state[(i+1)*4:(i+2)*4]=objs[i][1]
		#state[0] = minTTC # x does not change
		#state[0] = min(objs[0][0],100.)/100. # x does not change
		#state[1] /= 200.
		#state[2] = self._get_dist_nearest_obj(s)/200. # vx = 0
		#state[3] /= 30.
		#pdb.set_trace()
		return state

	def _get_vego(self, s):
		#return math.sqrt(s[2]**2+s[3]**2)
		return s[3]

	def _get_dist_nearest_obj(self, s):
		nobjs = int(len(s)/4 - 1)
		ego = s[0:4]

		dist_nearest_obj = math.inf
		num_nearest_obj = -1

		idx = 4
		for n in range(nobjs):
				obj = s[idx:idx+4]
				dist = get_dist(ego, obj)

				if dist < dist_nearest_obj:
						dist_nearest_obj = dist
						num_nearest_obj = n
				idx += 4

		#return dist_nearest_obj, num_nearest_obj
		return dist_nearest_obj


	# CA model for the ego vehicle and CV model for other cars
	def _step(self, state, action):

		# Speed optim (works only for deterministic cases)
		key = tuple(np.append(state, action))
		if key in self.next_states:
			sp, reward = self.next_states[key]
			return sp ,reward
		#pdb.set_trace()
		sp = np.zeros_like(self.start)

		s = state[0:4]
		a = np.array([0.0, action])
		sp[0:4] = transition_ca(s, a, self.dt) # ego is known - under control

		idx = 4
		for n in range(self.nobjs):
			s_obj = state[idx:idx+4]
			a_obj = np.array([0.0, 0.0]) # CV model so far
			sp[idx:idx+4] = transition_ca(s_obj, a_obj, self.dt)
			idx += 4

		# We have 3 possible type of rewards (checks them all, with impact on metrics)

		dist_nearest_obj = self._get_dist_nearest_obj(sp)
		# collision or driving backward (negative speed)
		if dist_nearest_obj < self.dist_collision or sp[3] < 0:
			reward = -1
		elif action <= -4:
			reward = -0.002
		else:
			reward = -0.001

		self.next_states[key] = (sp, reward)
		return sp, reward # reward in [-1,0] or [0,1] ? with [-1,0] easier to track the learning trend

		ttc, _ = self._get_smallest_TTC(sp)
		vego = self._get_vego(sp)
		if vego < 0:
			reward = -1 # with raw Q-learning we end up with negative speeds => penalize this
		else:
			# make sure reward is in [0,1] for UCB-1
			# take into account safety via ttc, efficiency via vego Comfort is missing
			#reward = 1 - (0.9*math.exp(-ttc/10) + 0.1 * abs((vego-self.vdes)/self.vdes))
			# reward = 1 - math.exp(-ttc/100) # /100 instead /10 to penalize even more risky ttc
			# if reward == 1:
			# 	reward = 1 - abs((vego-self.vdes)/self.vdes)
			# reward -= 1 # reward in [-1,0] or [0,1] ? with [-1,0] easier to track the learning trend

			reward_saf = 1 - math.exp(-ttc/100) # /100 instead /10 to penalize even more risky ttc
			reward_eff = 1 - abs((vego-self.vdes)/self.vdes)
			reward = (0.8 * reward_saf + 0.2 * reward_eff) - 1

		return sp, reward # reward in [-1,0] or [0,1] ? with [-1,0] easier to track the learning trend

	def actions(self, s):
		if self.restrict_actions:
			tuple_s = tuple(s)
			if tuple_s in self.restricted_actions:
				return self.restricted_actions[tuple_s]
			sTTC, _ = self._get_smallest_TTC(s)
			#print("sTTC {}".format(sTTC))
			safe_action_set = []
			spTTC_set = []
			for a in self.action_set:
				sp, r = self._step(s, a)
				spTTC, _ = self._get_smallest_TTC(sp)
				spTTC_set.append((spTTC, a))
				#print("spTTC {}".format(spTTC))
				#if spTTC >= 10 or (spTTC < 10 and spTTC > sTTC):
				if spTTC >= sTTC:
					safe_action_set.append(a)

			if safe_action_set == []:
				#print("NO SAFE ACTIONS !!!")
				res = [max(spTTC_set)[1]] # make it iterable => array single elt
				#return self.action_set
			else:
				#print("SAFE ACTIONS !!!")
				res = safe_action_set
			self.restricted_actions[tuple_s] = res
			return res
		else:
			return self.action_set

	def action_index(self, action):
		return self.action_set.index(action)

	def _expand(self, s, d):
		self.reachable_states.append(tuple(s)) # tuple to make it hashable
		if d == 0:
			return
		for a in self.actions(s):
			sp, r = self._step(s, a)
			self._expand(sp, d-1)

	def states(self):
		return self.reachable_states


	def reduce_state(self, s):
		features = []
		dmax = 200.
		vmax = 30.
		ttcmax = 100.

		pos, speed, ttc_info = s[1], s[3], self._get_smallest_TTC(s)
		ttc, nobj = ttc_info
		idx = 4+nobj*4
		ttcX, ttcY, ttcVx, ttcVy = s[idx:idx+4]
		ttcX, ttcY, ttcVx, ttcVy = ttcX/dmax, ttcY/dmax, ttcVx/vmax, ttcVy/vmax

		ttc = min(ttc,ttcmax)
		pos, speed, ttc = pos/dmax, speed/vmax, ttc/ttcmax

		features = [pos, speed, ttc, ttcX, ttcY, ttcVx, ttcVy]
		nstate = np.array(features)
		return nstate

	def _get_obstacles(self, s):
		x_ego, y_ego, vx_ego, vy_ego = s[0:4]
		idx = 4
		obstacles = []
		for n in range(int((len(s)-4)/4)):
			x_obj, y_obj, vx_obj, vy_obj = s[idx:idx+4]
			if vx_obj == 0: # No static objs here
				continue
			t_cross = (x_ego - x_obj) / vx_obj
			if t_cross > 0 and t_cross < 10:
				y_cross = y_obj + vy_obj * t_cross
				if y_cross >= y_ego and y_cross < self.goal[1]:
					obstacles.append((y_cross, t_cross))
			idx += 4
		# Sort by increasing y_cross ...
		obstacles.sort(key=lambda tup: tup[0])
		return obstacles

	def grid_state(self, s): # MDP state spec for offline VI
		# Spatio temporal occupancy grid which is a timeview of the ego path: 200m x 10sec
		# 200 meters x 10 seconds (start with simple discretization) x 2 states (authorized or forbidden) 
		vmax = 30.

		y_ego, v_ego = s[1], s[3]
		obstacles = self._get_obstacles(s)
		#obstacles = self._get_sorted_TTC(s)

		#features = np.zeros(1 + self.nobjs * 2)
		grid = np.zeros((200,10))
		for obstacle in obstacles:
			y_cross, t_cross = obstacle
			#t_cross *= 4
			grid[int(y_cross - y_ego), int(t_cross)] = 1
		grid[0,0] = v_ego / vmax
		return grid

		return None

	def project_state(self, s):
		dmax = 200.
		vmax = 30.
		tmax = 10.

		y_ego, v_ego = s[1], s[3]
		#obstacles = self._get_obstacles(s)
		#pdb.set_trace()
		obstacles = self._get_sorted_TTC(s)

		features = np.zeros(2 + self.nobjs * 2)
		features[0] = y_ego / dmax
		features[1] = v_ego / vmax
		nfeat = 2
		for obstacle in obstacles:
			y_cross, t_cross = obstacle
			#pdb.set_trace()
			#features[nfeat] = (y_cross - y_ego) / dmax
			features[nfeat] = y_cross / dmax
			features[nfeat+1] = t_cross / tmax
			nfeat+=2
		#return features
		return features[0:8]# just keep ego + 3 objs for now on

	def normalize_state(self, state):
		s = self._sort_by_TTC(state)
		dmax = 200.
		vmax = 30.
		ns = np.zeros_like(s)
		for i in range(int(len(s)/4)):
			ns[i*4] = s[i*4]/dmax
			ns[i*4+1] = s[i*4+1]/dmax
			ns[i*4+2] = s[i*4+2]/vmax
			ns[i*4+3] = s[i*4+3]/vmax
		return ns[0:12] # Too Complex

	def succProbReward(self, s, a, actionIndex=False):
		if actionIndex:
			a = self.action_set[a] # convert action index to action value (bugfix for DQN)
		# we can't return a list
		#raise NotImplementedError("Continuous state space")
		# Running on a subset of states with a single transition per state ... 
		# returns sp, proba=T(s,a,sp), R(s,a,sp) 
		results = []
		sp, r = self._step(s, a)
		results.append((tuple(sp), 1., r))
		return results

	def sampleSuccReward(self, s, a, actionIndex=False): # G(s,a) for mcts or Q-learning
		if actionIndex:
			a = self.action_set[a] # convert action index to action value (bugfix for DQN)
		sp, r = self._step(s, a)
		return (tuple(sp), r) # make the state hashable

	def discount(self):
		return self.gamma

	def piRandom(self, s):
		#print("pi0({})".format(s))
		return random.choice(self.action_set)
		return random.choice(self.actions(s))

	def pi0(self, s):
		return 0. # no accel, no decel

	def action_size(self):
		return len(self.action_set)
	def state_size(self):
		return len(self.start)

	def indexToAction(self, a):
		return self.action_set[a]

	def actionToIndex(self, a):
		return self.action_set.index(a)

	def visu(self, s, a=42, r=0, num_test=-1, algo="human", steps=-1, score=0, fsize=10):
		ttc = self._get_smallest_TTC(s)

		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		plt.xlim(0, 300)
		plt.ylim(0, 250)
		plt.grid()
		plt.title('Agent %s Test %d: step=%d a=%d r=%.3f ttc=%.2f score=%.3f' % (algo, num_test, steps, a, r, ttc[0], score), fontsize=12)
		dmin = self._get_dist_nearest_obj(s)
		plt.annotate("dmin=%.2f meters" % dmin, (105, 225), fontsize=fsize)
		x, y, vx, vy = s[0:4]
		plt.arrow(x, y, vx, vy, head_width=3.5, head_length=3.5, fc='lightblue', ec='blue')
		v = math.sqrt(vx**2+vy**2)
		plt.annotate("v=%.1f m.$s^{-1}$, y=%.2f m" % (v, y), (210, y), fontsize=fsize, color='blue')
		idx = 4
		for n in range(int((len(s)-4)/4)):
			x, y, vx, vy = s[idx:idx+4]
			v = math.sqrt(vx**2+vy**2)
			if n == ttc[1]:
				plt.arrow(x, y, vx, vy, head_width=3.5, head_length=3.5, fc='lightblue', ec='red')
				plt.annotate("v = %.1f m.$s^{-1}$" % v, (x-15, y-10), fontsize=fsize)
			else:
				plt.arrow(x, y, vx, vy, head_width=3.5, head_length=3.5, fc='lightblue', ec='green')
				if vx <=0:
					plt.annotate("v=%.1f m.$s^{-1}$" % v, (x, y+2), fontsize=fsize)
				elif (v >= 16 and v <=17) or (v>=22 and v<=23):
					plt.annotate("v=%.1f m.$s^{-1}$" % v, (x+1, y-7), fontsize=fsize)
				else:
					plt.annotate("v=%.1f m.$s^{-1}$" % v, (x+1, y+5), fontsize=fsize)
			idx += 4

		obstacles = self._get_sorted_TTC(s)
		for obstacle in obstacles:
			y_cross, t_cross = obstacle
			plt.scatter(s[0], y_cross, marker='o', color='r')
			plt.annotate("tcol=%.3f sec, dcol=%.2f m" % (t_cross, y_cross-s[1]), (210, y_cross), fontsize=fsize, color='r')

		plt.savefig('visu/%s_test%d_step%d.png' % (algo, num_test, steps))
		plt.show()

#random.seed(30)
#
#problem = ActProblem()
#start = problem.startState()
#print("start state: {}".format(start))
#print(problem.isEnd(start))

