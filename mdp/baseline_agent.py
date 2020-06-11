import numpy as np
import pdb

class BaselineAgent():
	def __init__(self, mdp):
		self.mdp = mdp
		self.DECELERATE = -4. # in actions [-2., -1., 0, 1., 2.]
		self.ACCELERATE = 1 # in actions [-2., -1., 0, 1., 2.]

	def act(self, state):
		ttc, _ = self.mdp._get_smallest_TTC(state)
		if ttc <= 10:
			return self.DECELERATE
		else:
			return self.ACCELERATE
