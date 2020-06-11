import collections, random

############################################################

# An algorithm that solves an MDP (i.e., computes the optimal
# policy).
class MDPAlgorithm:
	# Set:
	# - self.pi: optimal policy (mapping from state to action)
	# - self.V: values (mapping from state to best values)
	def solve(self, mdp): raise NotImplementedError("Override me")

############################################################
class ValueIteration(MDPAlgorithm):
	'''
	Solve the MDP using value iteration.  Your solve() method must set
	- self.V to the dictionary mapping states to optimal values
	- self.pi to the dictionary mapping states to an optimal action
	Note: epsilon is the error tolerance: you should stop value iteration when
	all of the values change by less than epsilon.
	The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
	'''
	def solve(self, mdp, epsilon=0.001):
		mdp.computeStates()
		def computeQ(mdp, V, state, action):
			# Return Q(state, action) based on V(state).
			return sum(prob * (reward + mdp.discount() * V[newState]) \
							for newState, prob, reward in mdp.succAndProbReward(state, action))

		def computeOptimalPolicy(mdp, V):
			# Return the optimal policy given the values V.
			pi = {}
			for state in mdp.states:
				pi[state] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
			return pi

		V = collections.defaultdict(float)	# state -> value of state
		numIters = 0
		while True:
			newV = {}
			for state in mdp.states:
				# This evaluates to zero for end states, which have no available actions (by definition)
				newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
			numIters += 1
			if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
				V = newV
				break
			V = newV

		# Compute the optimal policy now
		pi = computeOptimalPolicy(mdp, V)
		print(("ValueIteration: %d iterations" % numIters))
		self.pi = pi
		self.V = V

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
	# Return the start state.
	def startState(self): raise NotImplementedError("Override me")

	# Return set of actions possible from |state|.
	def actions(self, state): raise NotImplementedError("Override me")

	# Return a list of (newState, prob, reward) tuples corresponding to edges
	# coming out of |state|.
	# Mapping to notation from class:
	#	state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
	# If IsEnd(state), return the empty list.
	def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

	def discount(self): raise NotImplementedError("Override me")

	# Compute set of states reachable from startState.	Helper function for
	# MDPAlgorithms to know which states to compute values and policies for.
	# This function sets |self.states| to be the set of all states.
	def computeStates(self):
		self.states = set()
		queue = []
		self.states.add(self.startState())
		queue.append(self.startState())
		while len(queue) > 0:
			state = queue.pop()
			for action in self.actions(state):
				for newState, prob, reward in self.succAndProbReward(state, action):
					if newState not in self.states:
						self.states.add(newState)
						queue.append(newState)
		# print "%d states" % len(self.states)
		# print self.states

############################################################

# A simple example of an MDP where states are integers in [-n, +n].
# and actions involve moving left and right by one position.
# We get rewarded for going to the right.
class NumberLineMDP(MDP):
	def __init__(self, n=5): self.n = n
	def startState(self): return 0
	def actions(self, state): return [-1, +1]
	def succAndProbReward(self, state, action):
		return [(state, 0.4, 0),
				(min(max(state + action, -self.n), +self.n), 0.6, state)]
	def discount(self): return 0.9

############################################################




############################################################

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.	The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
	# Your algorithm will be asked to produce an action given a state.
	def getAction(self, state): raise NotImplementedError("Override me")

	# We will call this function when simulating an MDP, and you should update
	# parameters.
	# If |state| is a terminal state, this function will be called with (s, a,
	# 0, None). When this function is called, it indicates that taking action
	# |action| in state |state| resulted in reward |reward| and a transition to state
	# |newState|.
	def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")

# An RL algorithm that acts according to a fixed policy |pi| and doesn't
# actually do any learning.
class FixedRLAlgorithm(RLAlgorithm):
	def __init__(self, pi): self.pi = pi

	# Just return the action given by the policy.
	def getAction(self, state): return self.pi[state]

	# Don't do anything: just stare off into space.
	def incorporateFeedback(self, state, action, reward, newState): pass

############################################################

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,
			 sort=False):

	evaluatedStates = {}

	# Return i in [0, ..., len(probs)-1] with probability probs[i].
	def sample(probs):
		target = random.random()
		accum = 0
		for i, prob in enumerate(probs):
			accum += prob
			if accum >= target: return i
		raise Exception("Invalid probs: %s" % probs)

	totalRewards = []  # The rewards we get on each trial
	for trial in range(numTrials):
		state = mdp.startState()
		sequence = [state]
		totalDiscount = 1
		totalReward = 0
		for _ in range(maxIterations):
			action = rl.getAction(state)
			transitions = mdp.succAndProbReward(state, action)
			if sort: transitions = sorted(transitions)
			if len(transitions) == 0:
				rl.incorporateFeedback(state, action, 0, None)
				break

			# Choose a random transition
			i = sample([prob for newState, prob, reward in transitions])
			newState, prob, reward = transitions[i]
			sequence.append(action)
			sequence.append(reward)
			sequence.append(newState)

			evaluatedStates[state] = 1
			rl.incorporateFeedback(state, action, reward, newState)
			totalReward += totalDiscount * reward
			totalDiscount *= mdp.discount()
			state = newState
		if verbose:
			print(("Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)))
		totalRewards.append(totalReward)

	print("Number of evaluatedStates: {} ".format(len(evaluatedStates)))
	return totalRewards
