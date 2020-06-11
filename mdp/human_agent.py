import numpy as np
import pdb

class HumanAgent():
	def __init__(self, mdp):
		self.mdp = mdp

	def act(self, state):
		self.mdp.visu(state)
		a = 5
		while a not in self.mdp.actions(state):
			a = input("Which action (-4, -2, -1, 0, 1, 2) ? ")
			a = int(float(a))
		return a

