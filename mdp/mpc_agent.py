import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pdb

NORMALIZE = [100., 200., 20., 20.]
NORMALIZE = [1., 1., 1., 1.]

class MpcAgent():
	def __init__(self, mdp, xr = np.array([100.0, 200.0, 0.0, 20.0]), T=20, dt=0.250):
		self.mdp = mdp

		self.T = T # Prediction horizon
		self.dt = dt
		self.Ad = np.matrix([[1.0, 0.0, dt,	0.0],
						[0.0, 1.0, 0.0, dt],
						[0.0, 0.0, 1.0, 0.0],
						[0.0, 0.0, 0.0, 1.0]])
		self.Bd = np.matrix([[0.5*dt**2, 0.0],
						[0.0, 0.5*dt**2],
						[dt, 0.0],
						[0.0, dt]])
		self.nx, self.nu = self.Bd.shape
		# Constraints (static ones)
		self.umin = np.array([0.0, -4.0]) # for ax,ay
		self.umax = np.array([0.0,  2.0]) # for ax,ay
		self.xmin = np.array([100.0,   0.0, 0.0, 0.0]) # for x,y,vx,vy
		self.xmax = np.array([100.0, 10000.0, 25.0, 25.0]) # for x,y,vx,vy

		# Objective function: QP
		self.Q = np.diag([1.0, 1.0, 50.0, 50.0])
		self.QT = self.Q
		self.R = np.diag([0.001, 0.001])
		self.xr = xr

		self.DECELERATE = -4.
		self.ACCELERATE = 1

	def _run_mpc_backup(self, x_init, xr, obstacles=[], dsafety=10):
		print("mpc: x_init={} xr={} obstacles={}".format(x_init, xr, obstacles))
		x = cp.Variable((self.nx, self.T+1))
		u = cp.Variable((self.nu, self.T))

		objective = 0
		constraints = [x[:,0] == x_init]
		for t in range(self.T):
			objective += cp.quad_form(x[:,t] - xr, self.Q) # + cp.quad_form(u[:,t], self.R)
			constraints += [x[:,t+1] == self.Ad*x[:,t] + self.Bd*u[:,t]]
			constraints += [self.xmin <= x[:,t], x[:,t] <= self.xmax]
			constraints += [self.umin <= u[:,t], u[:,t] <= self.umax]

		# Dynamic obstacles constraints
		for obstacle in obstacles:
			#pdb.set_trace()
			tcross = obstacle[1]
			tcrossd = int(tcross/self.dt) # ttc discrete
			if tcrossd <= self.T:
				ycross = obstacle[0]
				constraints += [x[1,tcrossd] >= ycross + 0.5*dsafety] # AFTER !!!

		objective += cp.quad_form(x[:,self.T] - xr, self.QT)

		prob = cp.Problem(cp.Minimize(objective), constraints)

		start = time.time()
		prob.solve(solver=cp.ECOS, verbose=False) # ECOS, SCS
		#prob.solve(solver=cp.OSQP, verbose=False) # ECOS, SCS
		runtime = time.time() - start
		print("runtime {:.2f} sec".format(runtime))

		if prob.status == cp.OPTIMAL:
			ay = u.value[1,0]
			print("OPTIMAL BACKUP solution: ay={:.2f}".format(ay))
			return u[:,0].value
		else:
			#pdb.set_trace()
			print("OPTIMAL BACKUP solution not found")
			return None


	def _run_mpc(self, x_init, xr, obstacles=[], dsafety=10):
		print("mpc: x_init={} xr={} obstacles={}".format(x_init, xr, obstacles))
		x = cp.Variable((self.nx, self.T+1))
		u = cp.Variable((self.nu, self.T))

		objective = 0
		constraints = [x[:,0] == x_init]
		for t in range(self.T):
			objective += cp.quad_form(x[:,t] - xr, self.Q) # + cp.quad_form(u[:,t], self.R)
			constraints += [x[:,t+1] == self.Ad*x[:,t] + self.Bd*u[:,t]]
			constraints += [self.xmin <= x[:,t], x[:,t] <= self.xmax]
			constraints += [self.umin <= u[:,t], u[:,t] <= self.umax]

		# Dynamic obstacles constraints
		for obstacle in obstacles:
			#pdb.set_trace()
			tcross = obstacle[1]
			tcrossd = int(tcross/self.dt) # ttc discrete
			if tcrossd <= self.T:
				ycross = obstacle[0]
				constraints += [x[1,tcrossd] <= ycross - 2*dsafety] # y <= something

		objective += cp.quad_form(x[:,self.T] - xr, self.QT)

		prob = cp.Problem(cp.Minimize(objective), constraints)

		start = time.time()
		prob.solve(solver=cp.ECOS, verbose=False) # ECOS, SCS
		#prob.solve(solver=cp.OSQP, verbose=False) # ECOS, SCS
		runtime = time.time() - start
		print("runtime {:.2f} sec".format(runtime))

		if prob.status == cp.OPTIMAL:
			ay = u.value[1,0]
			print("OPTIMAL solution: ay={:.2f}".format(ay))
			return u[:,0].value
		else:
			#pdb.set_trace()
			return self._run_mpc_backup(x_init, self.xr, obstacles)
			return None


	def act(self, state):
		ttc, nobj = self.mdp._get_smallest_TTC(state)
		#pdb.set_trace()
		if ttc <= self.T * self.dt:
			idx = 4 + nobj * 4
			ttcY, ttcVy = state[idx+1], state[idx+3]
			ycol = ttcY + ttc * ttcVy
			obstacles = [[ycol, ttc, nobj]]
		else:
			obstacles = []

		obstacles = self.mdp._get_sorted_TTC(state)
		# no improvemements with below code => commented
		#obstacles = self.mdp._get_obstacles(state)

		x0 = np.asarray(state[0:4])
		cmd = self._run_mpc(x0, self.xr, obstacles[0:3])
		if cmd is None:
			ttc, nobj = self.mdp._get_smallest_TTC(state)
			if ttc <= 10:
				return self.DECELERATE
			else:
				return 0 # self.ACCELERATE
		else:
			ax, ay = cmd
			return ay

