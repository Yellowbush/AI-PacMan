# AI-PacMan

### 1 Search:
1.
Implemented depth-first search using stack to explore deepest node in the
search tree first and returns a list of actions that reach the goal state.

2.
Implemented breadth-first search using queue to explore shallowest nodes
in the search tree first and returns a list of actions that reach the goal
state.

3.
Implemented uniform cost search using priority queue to search for
solutions in a graph represented by the problem parameter based on the
total cost of actions taken to reach each state. The priority queue will
prioritize states with lower total cost.

4.
Implemented A*search using priority queue to search for solutions in a
graph represented by the problem parameter based on the combined cost of
actions taken to reach each state. The priority queue will prioritize
states with lower combined cost.

5.
No code added to constructor. Added return as start state for
getStartState(). For isGoalState() added condition so that goal state is
reached when there are no remaining corners to visit. For getSuccessors()
considered all possible actions(north, south, east, west) to calculate the
new position of pacman and see if he hits walls.

6.
Calculates the estimated distance from the current state to the goal state
by summing the Manhattan distances from the current position to the
remaining corners. The total heuristic value will be the lower bound on
the shortest path from current state to the goal.
Used while loop to iterate corners, distance_list is created to store the
Manhattan distances. So that it can later be used to find corner with the
minimum Manhattan distance from current position.

7.
Search problem and heuristic for pacman to eat all active dots on board.
Calculates the estimated distance from the current state to the closest
food using Breadth-First Search and returns the maximum of these distances
as the heuristic value.
Used for loop to iterate over each food and used PositionSearchProblem to create new problem where food coordinate is
the goal state. Then performed breadth-first search to find shortest path.

8.
Used breadth-first search to find closest dot.
Filled in goal state that completes the problem.

### 2 MultiAgent Search:
1.
Implemented improved ReflexAgent to consider both food locations and ghost locations. Aiming to improve Pacman’s performance.

2.
Implemented a MinimaxAgent in multiAgents.py to work with any number of ghosts, expanding the game tree to an arbitrary depth using self.depth and self.evaluationFunction.

3.
Implemented AlphaBetaAgent in AlphaBetaAgent class in multiAgents.py, enhancing minimax search with alpha-beta pruning to efficiently explore the game tree.

4.
Implemented ExpectimaxAgent in ExpectimaxAgent class in multiAgents.py to model probabilistic behavior when facing non-optimal adversaries, particularly random ghosts.

5.
Implemented improved Pacman evaluation function in betterEvaluationFunction to evaluate states for optimal decision-making, considering factors like food, ghost, and capsule proximity.

### 3 Reinforcement Learning:
1. 
Implemented value Iteration agent for offline planning in valueIterationAgents.py. It runs k-step value iteration, computes optimal actions, and Q-values based on self.values.

2. 
Implemented BridgeGrid environment, adjust either the discount or noise parameter in analysis.py's question2() to encourage the agent to cross the bridge.

3. 
Implemented policies for discount grid.

4. 
Implemented Asynchronous Value Iteration Agent in valueIterationAgents.py, performing cyclic value iteration for a specified number of iterations.

5. 
Implemented PrioritizedSweepingValueIterationAgent in valueIterationAgents.py. It updates state values by prioritizing states with high errors, using a min heap.

6. 
Implemented Q-learning in qlearningAgents.py. You must code update, computeValueFromQValues, getQValue, and computeActionFromQValues. Break ties randomly for computeActionFromQValues.

7.
Implemented epsilon-greedy action selection in getAction for the Q-learning agent in qlearningAgents.py, using random actions with probability epsilon

8.
Implemented an Approximate Q-learning agent in qlearningAgents.py as the ApproximateQAgent class, using feature extractors for state-action pairs.

