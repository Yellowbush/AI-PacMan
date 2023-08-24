# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Convert newFood to as list
        foodList = newFood.asList()
        # Initialize a variable to store the minimum manhattean distance to food
        minFood = float("inf")
        # Iterate through each food and update distance
        for food in foodList:
            minFood = min(minFood, manhattanDistance(newPos, food))
        # Check if new position is too close to ghost
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                # Indicate losing move by returning negative infinity
                return float('-inf')
        
        return successorGameState.getScore() + 1.0/minFood

        

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maximizer(state, depth):
            # Loop for reaching max search depth or game is over, return state evaluation
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # Initialize max value to negative inf
            value = float("-inf")
            #Get legal moves for current state
            legalMoves = state.getLegalActions()
            for action in legalMoves:
                value = max(value, minimizer(state.generateSuccessor(0, action), depth, 1))
            return value
        
        def minimizer(state, depth, agentIndex):
            # Loop for reaching max search depth or game is over, return state evaluation
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            value = float("inf")
            legalMoves = state.getLegalActions(agentIndex)
            # If current agen is last one, loop through all legal moves
            if agentIndex == state.getNumAgents()-1:
                for action in legalMoves:
                    value = min(value, maximizer(state.generateSuccessor(agentIndex, action), depth+1))
            # If the current agen is not the last one, loop through all legal moves
            else:
                for action in legalMoves:
                    value = min(value, minimizer(state.generateSuccessor(agentIndex, action), depth, agentIndex+1))
            return value
        
        legalMoves = gameState.getLegalActions()
        move = Directions.STOP
        value = float("-inf")
        for action in legalMoves:
            temp = minimizer(gameState.generateSuccessor(0, action), 0, 1)
            if temp > value:
                value = temp
                move = action

        # return the best move obtained from calling minimizer on all legal moves
        return move
    
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maximizer(state, agentIndex, depth, alpha, beta):
            agentIndex = 0
            legalMoves = state.getLegalActions(agentIndex)
            if not legalMoves or depth == self.depth:
                return self.evaluationFunction(state)
            
            value = float("-inf")
            currentAlpha = alpha

            # For each legal action, generate the successor state and call the minimizer
            for action in legalMoves:
                value = max(value, minimizer(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1, currentAlpha, beta))
                # If the value is greater than beta, we can prune the remaining branches and return the value.
                if value > beta:
                    return value
                currentAlpha = max(currentAlpha, value)
            # return max value
            return value
        
        def minimizer(state, agentIndex, depth, alpha, beta):
            agentCount = gameState.getNumAgents()
            legalMoves = state.getLegalActions(agentIndex)

            if not legalMoves:
                return self.evaluationFunction(state)
            
            value = float("inf")
            currentBeta = beta
            # if current agen is last ghost player, we need to call the maximizer function on the next agent
            if agentIndex == agentCount -1:
                for action in legalMoves:
                    value = min(value, maximizer(state.generateSuccessor(agentIndex, action), agentIndex, depth, alpha, currentBeta))
                    if value < alpha:
                        return value
                    currentBeta = min(currentBeta, value)
            # If the current agent is not the last ghost player, call the minimizer
            else:
                for action in legalMoves:
                    value = min(value, minimizer(state.generateSuccessor(agentIndex, action), agentIndex +1, depth, alpha, currentBeta))
                    if value < alpha:
                        return value
                    currentBeta = min(currentBeta, value)
            return value

        
        actions = gameState.getLegalActions(0)
        alpha = float("-inf")
        beta = float("inf")

        allActions = {}
        for action in actions:
            value = minimizer(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)
            allActions[action] = value

            if value > beta:
                return action
            alpha = max(value, alpha)

        return max(allActions, key = allActions.get)
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expected(state, agentIndex, depth):
            agentCount = gameState.getNumAgents()
            legalMoves = state.getLegalActions(agentIndex)

            # IF no legal actions or max depth reached, return evaluation
            if not legalMoves:
                return self.evaluationFunction(state)
            
            value = 0
            probability = 1.0/len(legalMoves)

            # For each legal action, calculate the expected value
            for action in legalMoves:
                if agentIndex == agentCount - 1:
                    # if last agent, max
                    currentValue = maximizer(state.generateSuccessor(agentIndex, action), agentIndex, depth)
                else:
                    # else, expected
                    currentValue = expected(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                value += currentValue * probability
            return value
        
        def maximizer(state, agentIndex, depth):
            agentIndex = 0
            legalMoves = state.getLegalActions(agentIndex)

            #If no legal actions or max depth reached
            if not legalMoves or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Calculate max expected value among all legal actions
            value = max(expected(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1) for action in legalMoves)
            return value
        
        actions = gameState.getLegalActions(0)
        allActions = {}
        for action in actions:
            allActions[action] = expected(gameState.generateSuccessor(0, action), 1, 1)

        return max(allActions, key = allActions.get)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPosition = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    
    
    score = currentGameState.getScore()
    value = float("-inf")

    # Calcualte distance to all remaining food
    distancesToFoodList = [util.manhattanDistance(currentPosition, foodPosition) for foodPosition in currentFood.asList()]
    # If remaining, add reciprocal of min distance to food weight
    if len(distancesToFoodList) > 0:
        score += 1/min(distancesToFoodList)
    # Else, add wegith for food
    else:
        score += 1
    
    # iterate through all ghost in game state
    for ghost in currentGhostStates:
        distance = manhattanDistance(currentPosition, ghost.getPosition())
        #if ghost is not same position as agent, add appropraite weight based on ghost state
        if distance > 0:
            if ghost.scaredTimer > 0:
                score += 1/distance
            else:
                score += -1/distance
        else:
            return value
        
    return score

# Abbreviation
better = betterEvaluationFunction
