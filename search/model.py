import numpy as np
import copy
import random
import math
import pdb

# Example solved and checked UCS cost < 1 at 100 m
# REVERSE HISTORY:  (0.017000000000000008, [(None, 0), (2.0, 0.001), (2.0, 0.001), (2.0, 0.001), (2.0, 0.001), (1.0, 0.001), (2.0, 0.001), (2.0, 0.001), (2.0, 0.001), (2.0, 0.001), (2.0, 0.001), (2.0, 0.001), (2.0, 0.001), (1.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001)])
# n_states explored: 92560
# time: 141.81043004989624 sec

start2 = (100.0, 0.0, 0.0, 20.0, 49.0, 145.0, 21.0, 1.0, 50.0, 124.0, 13.0, 4.0, 2.0, 153.0, 10.0, 5.0, 25.0, 76.0, 25.0, 3.0, 12.0, 119.0, 21.0, 0.0, 187.0, 126.0, -10.0, -4.0, 195.0, 179.0, -16.0, -2.0, 194.0, 108.0, -17.0, -5.0, 170.0, 116.0, -17.0, -0.0, 169.0, 119.0, -14.0, -3.0)


# Unsolvable: cost > 1 at 125 meters
# REVERSE HISTORY:  (1.009, [(None, 0), (0.0, 0.001), (2.0, 0.001), (1.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-4.0, 1)])
# n_states explored: 37927
# time: 58.85057854652405 sec
start4 = (100.0, 0.0, 0.0, 20.0, 24.0, 62.0, 18.0, 4.0, 43.0, 176.0, 13.0, 1.0, 23.0, 50.0, 12.0, 5.0, 33.0, 126.0, 20.0, 1.0, 42.0, 167.0, 16.0, 3.0, 165.0, 116.0, -12.0, -0.0, 191.0, 138.0, -17.0, -2.0, 167.0, 59.0, -23.0, -3.0, 162.0, 46.0, -14.0, -1.0, 172.0, 151.0, -11.0, -2.0)

# Unsolvable: cost > 1 at 100 meters
# REVERSE HISTORY:  (1.01, [(None, 0), (-2.0, 0.001), (1.0, 0.001), (0.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-4.0, 1)])
# n_states explored: 26792
# time: 41.29521918296814 sec
start5 = (100.0, 0.0, 0.0, 20.0, 9.0, 67.0, 15.0, 4.0, 14.0, 111.0, 21.0, 5.0, 27.0, 97.0, 19.0, 2.0, 35.0, 75.0, 20.0, 5.0, 36.0, 47.0, 21.0, 4.0, 162.0, 68.0, -15.0, -4.0, 168.0, 70.0, -11.0, -2.0, 180.0, 121.0, -16.0, -2.0, 194.0, 62.0, -23.0, -1.0, 165.0, 73.0, -21.0, -3.0)

# Unsolvable cost > 1 at 100 meters
# REVERSE HISTORY:  (1.011, [(None, 0), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-1.0, 0.001), (-4.0, 1)])
# n_states explored: 28918
# time: 45.73861479759216 sec
start10 = (100.0, 0.0, 0.0, 20.0, 29.0, 119.0, 24.0, 2.0, 34.0, 57.0, 20.0, 3.0, 37.0, 103.0, 19.0, 3.0, 11.0, 135.0, 22.0, 1.0, 30.0, 189.0, 16.0, 5.0, 159.0, 72.0, -17.0, -5.0, 187.0, 161.0, -16.0, -4.0, 180.0, 138.0, -21.0, -2.0, 185.0, 114.0, -20.0, -0.0, 158.0, 97.0, -21.0, -2.0)

# Unsolvable cost > 1 at 100 meters
# start state: (100.0, 0.0, 0.0, 20.0, 0.0, 159.0, 24.0, 1.0, 32.0, 56.0, 10.0, 4.0, 49.0, 127.0, 21.0, 5.0, 17.0, 124.0, 20.0, 2.0, 1.0, 32.0, 12.0, 2.0, 179.0, 84.0, -25.0, -1.0, 156.0, 165.0, -11.0, -3.0, 161.0, 48.0, -23.0, -0.0, 156.0, 104.0, -15.0, -0.0, 153.0, 139.0, -15.0, -1.0)
# False
# REVERSE HISTORY:  (1.008, [(None, 0), (2.0, 0.001), (1.0, 0.001), (0.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-4.0, 1)])
# n_states explored: 10279
# time: 15.673994302749634 sec
start28 = (100.0, 0.0, 0.0, 20.0, 0.0, 159.0, 24.0, 1.0, 32.0, 56.0, 10.0, 4.0, 49.0, 127.0, 21.0, 5.0, 17.0, 124.0, 20.0, 2.0, 1.0, 32.0, 12.0, 2.0, 179.0, 84.0, -25.0, -1.0, 156.0, 165.0, -11.0, -3.0, 161.0, 48.0, -23.0, -0.0, 156.0, 104.0, -15.0, -0.0, 153.0, 139.0, -15.0, -1.0)

# Unsolvable cost > 1 at 100 meters
# REVERSE HISTORY:  (1.01, [(None, 0), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-4.0, 1)])
# n_states explored: 21115
# time: 32.95072078704834 sec
start33 = (100.0, 0.0, 0.0, 20.0, 44.0, 76.0, 18.0, 1.0, 47.0, 30.0, 18.0, 2.0, 12.0, 32.0, 18.0, 1.0, 5.0, 132.0, 18.0, 1.0, 35.0, 82.0, 14.0, 3.0, 161.0, 97.0, -10.0, -1.0, 200.0, 144.0, -10.0, -1.0, 164.0, 101.0, -14.0, -1.0, 183.0, 169.0, -13.0, -1.0, 155.0, 64.0, -18.0, -5.0)

# Unsolvable cost > 1 at 100 meters
# REVERSE HISTORY:  (1.008, [(None, 0), (2.0, 0.001), (2.0, 0.001), (-1.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-4.0, 1)])
# n_states explored: 12025
# time: 18.1905460357666 sec
start80 = (100.0, 0.0, 0.0, 20.0, 12.0, 133.0, 18.0, 0.0, 33.0, 44.0, 17.0, 3.0, 44.0, 112.0, 11.0, 1.0, 40.0, 91.0, 25.0, 5.0, 48.0, 69.0, 21.0, 3.0, 170.0, 89.0, -21.0, -1.0, 164.0, 61.0, -22.0, -5.0, 168.0, 47.0, -15.0, -0.0, 167.0, 44.0, -17.0, -0.0, 154.0, 63.0, -25.0, -4.0)

# Unsolvable cost > 1 at 100 meters
# REVERSE HISTORY:  (1.009, [(None, 0), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-2.0, 0.001), (-4.0, 1)])
# n_states explored: 14668
# time: 22.722214698791504 sec
start90 = (100.0, 0.0, 0.0, 20.0, 34.0, 48.0, 23.0, 0.0, 28.0, 169.0, 12.0, 4.0, 38.0, 147.0, 15.0, 5.0, 1.0, 68.0, 11.0, 2.0, 29.0, 104.0, 23.0, 4.0, 184.0, 144.0, -10.0, -4.0, 189.0, 83.0, -17.0, -5.0, 174.0, 80.0, -24.0, -4.0, 192.0, 139.0, -15.0, -1.0, 154.0, 120.0, -18.0, -2.0)

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


# -------------
# Model
# -------------
# we frame the problem as a search problem, without uncertainties (cf CS221 W3 Search)
# - driving models are known
# - sensors are perfect (we know exactly the different state vectors, [x,y,vx,vy] for all cars)
# This is going to be used by our Oracle

class ActProblem:  # Anti Collision Tests problem
	# actions are accelerations
	def __init__(self, nobjs=10, dist_collision=10, dt=0.25, actions=[-4., -2., -1., 0., +1., +2.], start=None):
		self.nobjs = nobjs
		self.dist_collision = dist_collision
		self.dt = dt
		self.actions = actions
		#self.goal  = np.array([100.0, 90.0, 0.0, 0.0], dtype=float) # down from 200 to 50
		self.goal  = np.array([100.0, 100.0, 0.0, 0.0], dtype=float) # down from 200 to 50
		self.goal  = np.array([100.0, 200.0, 0.0, 0.0], dtype=float) # down from 200 to 50
		if start is not None:
			self.start = self._readStartState(start)
		else:
			# x, y, vx, vy
			self.start = np.array([100.0, 0.0, 0.0, 20.0], dtype=float)
			self.start = self._randomStartState()
			self.start = np.array(start2)
		self.vdes = 20 # desired speep 20 ms-1

	def _readStartState(self, startStateNumber):
		"""reads start state from the startState.txt file"""
		start_state = np.zeros(shape=(1, 44), dtype=float)
		with open("startStates.txt") as states_file:
			for line in states_file:
				if line.find("Test " + str(startStateNumber)) > -1:
					from_idx = line.find("(")
					start_state = np.array([float(x.strip()) for x in line[from_idx + 1:-2].split(',')], dtype=float)
					break
		return start_state

		# stase is R44: 1 ego + 10 cars, 4 coordonates (x,y,vx,vy) each
	def _randomStartState(self):
		state = copy.deepcopy(self.start)
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
		return state

	def startState(self):
		return tuple(self.start) # to make it hashable

	def isEnd(self, s):
		if s[1] >= self.goal[1]:
			return True
		elif self._get_dist_nearest_obj(s) < self.dist_collision:
			return True
		else:
			return False

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

	def _get_vego(self, s):
		#return math.sqrt(s[2]**2+s[3]**2)
		return s[3]

	def _get_dist_nearest_obj(self, s):
		nobjs = int(len(s)/4 - 1)
		ego = s[0:4]

		dist_nearest_obj = math.inf
		num_nearest_obj = -1

		idx = 4
		# TODO rewrite this in a more pythonic way
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

		dist_nearest_obj = self._get_dist_nearest_obj(sp)
		# collision or driving backward (negative speed)
		if dist_nearest_obj < self.dist_collision or sp[3] < 0:
			cost = 1
		elif action <= -4.:
			cost = 0.002
		else:
			cost = 0.001
		return sp, cost

	def succAndCost(self, s):
		res = [] # (action, nextState, cost)
		for a in self.actions:
			sp, cost = self._step(s, a)
			res.append((a, tuple(sp), cost))
		return res

#random.seed(30)
#
#problem = ActProblem()
#start = problem.startState()
#print("start state: {}".format(start))
#print(problem.isEnd(start))

