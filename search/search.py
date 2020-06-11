import time
import util
import argparse
import pdb
import sys

sys.setrecursionlimit(10000)

import model
from model import *


# ---------------------------
# Algo: Backtracking search
# ---------------------------
def backtrackingSearch(problem):
    best = {'cost': float('inf'), 'history': None}

    def recurse(s, history, totalCost):
        if problem.isEnd(s):
            if totalCost < best['cost']:
                best['history'] = history
                best['cost'] = totalCost
            return totalCost
        for a, sp, cost in problem.succAndCost(s):
            recurse(sp, history + [(a, totalCost + cost, s)],
                    totalCost + cost)

    recurse(problem.startState(), [], 0)
    return best


# -------------
# Print
# -------------
def printBestPath(best, problem):
    print("totalCost={}".format(best['cost']))
    for (a, totalCost, s) in best['history']:
        print("[STEP] a={}, s={}, cost={}".format(a, s, totalCost))
        if type(problem) is ActProblem:
            ttc = problem._get_smallest_TTC(s)
            print("[TTC] ttc={}".format(ttc))


# ---------------------------
# Algo: Dynamic Programming
# ---------------------------
def dynamicProgramming(problem):
    futureCost = {}

    def recurse(s):
        if s in futureCost:
            return futureCost[s][0]
        if problem.isEnd(s):
            return 0
        futureCost[s] = min(
            [(cost + recurse(sp), a, sp, cost) for a, sp, cost in
             problem.succAndCost(s)])

        return futureCost[s][0]

    minCost = recurse(problem.startState())

    # recover history
    print("n_states explored: {}".format(len(futureCost)))
    history = []
    s = problem.startState()
    while not problem.isEnd(s):
        totalCost, a, sp, cost = futureCost[s]
        history.append((a, totalCost, s))
        s = sp
    return {'cost': minCost, 'history': history}


# ---------------------------
# Algo: Uniform Cost Search
# ---------------------------
def uniformCostSearch(problem):
    frontier = util.PriorityQueue()
    explored = {}
    previous = {}

    frontier.update(problem.startState(), 0)
    # state, action, cost
    previous[problem.startState()] = None, None, 0
    while True:
        s, pastCost = frontier.removeMin()
        prev_s, prev_a, prev_cost = previous[s]
        explored[s] = (pastCost, prev_s, prev_a, prev_cost)
        if problem.isEnd(s):
            minCost = pastCost
            break
        for a, sp, cost in problem.succAndCost(s):
            if sp in explored:
                continue
            if frontier.update(sp, pastCost + cost):
                previous[sp] = s, a, cost

    print("EXPLORED: ", explored)

    history = []
    while s is not None:
        pastCost, prev_s, prev_a, prev_cost = explored[s]
        # history.append((prev_a, s, prev_cost))
        # action, cost, state
        if prev_s is not None:
            history.append((prev_a, pastCost, prev_s))
        s = prev_s
    history.reverse()
    return {'cost': minCost, 'history': history}


# -------------
# Sample Model
# -------------
# just to validate the algorithms
class TransportProblem():
    def __init__(self, N):
        self.end = N

    def startState(self):
        return 1

    def isEnd(self, s):
        return (s >= self.end)

    def succAndCost(self, s):
        res = []  # (action, nextState, cost)
        if s + 1 <= self.end:
            res.append(('walk', s + 1, 1))
        if 2 * s <= self.end:
            res.append(('tram', 2 * s, 2))
        return res

#problem = TransportProblem(N=1000)

parser = argparse.ArgumentParser()
parser.add_argument('--start', default=None,
                    help="start state number as defined in startStates.txt. "
                         "If missing, the random state is taken or problem "
                         "size if it is the transport problem")
parser.add_argument('--algo', default='ucs',
                    choices=["ucs", "dp", "backtracking"],
                    help="used algorithm")
parser.add_argument('--problem', default='act', choices=["act", "transport"],
                    help="problem to solve")
args = parser.parse_args()

# for debug purpose uncomment
# args.problem = "transport"
# args.start = 10
# args.algo = "backtracking"

random.seed(30)

# instantiate the problem object
if args.problem == "act":
    problem = ActProblem(start=args.start)
else:
    problem = TransportProblem(N=int(args.start))

start = problem.startState()
print("start state: {}".format(start))
print(problem.isEnd(start))

if args.algo == "ucs":
    search_algo = uniformCostSearch
elif args.algo == "dp":
    search_algo = dynamicProgramming
else:
    search_algo = backtrackingSearch

start = time.time()
result = search_algo(problem)
end = time.time()

printBestPath(result, problem)
print("time: {} sec".format(end - start))
