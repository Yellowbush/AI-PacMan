# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    stack = util.Stack()
    start_state = problem.getStartState()
    stack.push((start_state, []))

    visited = set()
    while not stack.isEmpty():
        state, path = stack.pop()

        if problem.isGoalState(state):
            return path
        
        visited.add(state)
        successors = problem.getSuccessors(state)
        for successor, action, cost in successors:
            if successor not in visited:
                new_path = path + [action]
                stack.push((successor, new_path))
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    queue = util.Queue()
    start_state = problem.getStartState()
    queue.push((start_state, [], 0))

    visited = []
    queue.push((start_state, [], 0))

    while not queue.isEmpty():
        successor, actions, costs = queue.pop()
        if not successor in visited:
            visited.append(successor)
            if problem.isGoalState(successor):
                return actions
            for state, action, cost in problem.getSuccessors(successor):
                if not state in visited:
                    queue.push((state, actions + [action], cost))
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    priority_queue.push((start_state, []), 0)
    total_cost = {start_state: 0}

    visited = set()

    while not priority_queue.isEmpty():
        successor, actions = priority_queue.pop()
        if problem.isGoalState(successor):
            return actions
        
        if successor not in visited:
            visited.add(successor)
            successors = problem.getSuccessors(successor)
            for state in successors:
                next_state, action, step_cost = state
                total_cost_to_reach_successor = total_cost[successor] + step_cost
                if next_state not in visited or total_cost_to_reach_successor < total_cost[next_state]:
                    total_cost[next_state] = total_cost_to_reach_successor
                    priority_queue.push((next_state, actions + [action]), total_cost_to_reach_successor)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    priority_queue.push((start_state, [], 0), 0)
    visited = []

    while not priority_queue.isEmpty():
        successor, actions, costs = priority_queue.pop()
        if not successor in visited:
            visited.append(successor)
            if problem.isGoalState(successor):
                return actions
            for state, action, cost in problem.getSuccessors(successor):
                if not state in visited:
                    heuristic_cost = costs + cost + heuristic(state, problem)
                    priority_queue.push((state, actions + [action], costs + cost), heuristic_cost)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
