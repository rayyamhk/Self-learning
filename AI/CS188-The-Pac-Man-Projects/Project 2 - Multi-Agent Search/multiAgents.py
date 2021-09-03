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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        prevPos = currentGameState.getPacmanPosition()
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]

        def getClosest(curNode, nodeList):
          if len(nodeList) == 0:
            return None
          closest = nodeList[0]
          shortestDistance = manhattanDistance(curNode, closest)
          for node in nodeList[1:]:
            distance = manhattanDistance(curNode, node)
            if distance < shortestDistance:
              closest = node
              shortestDistance = distance
          return (shortestDistance, closest)

        def getTotalDistance(curNode, nodeList):
          clone = nodeList.copy()
          total = 0
          node = curNode
          while len(clone) != 0:
            distance, closest = getClosest(node, clone)
            total += distance
            node = closest
            clone.remove(node)
          return total
        
        if prevPos == newPos: # avoid stationary
          return -999999

        for i in range(len(newScaredTimes)): # ignore ghosts when they're scared
          if newScaredTimes[i] == 0:
            ghostPos = newGhostPositions[i]
            if manhattanDistance(newPos, ghostPos) <= 1:
              return -999999
        
        newFoodList = newFood.asList()

        if len(newFoodList) == 0:
          return 999999
        #TODO: sometimes the pacman will stuck
        return getTotalDistance(newPos, newFoodList) * -1

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
        def getMinMax(gameState, agentIndex, height):
          # terminal test
          if gameState.isWin() or gameState.isLose() or height == 0:
            return (self.evaluationFunction(gameState), [])

          # list of legal actions of current state with respect to the agent (pacman or ghost)
          actions = gameState.getLegalActions(agentIndex)

          # new states generated by all legal actions with respect to the agent
          nextStates = [gameState.generateSuccessor(agentIndex, action) for action in actions]

          numAgents = gameState.getNumAgents() # 1 + number of ghosts
          nextAgentIndex = (agentIndex + 1) % numAgents

          # get all values from the child nodes in the minimax
          values = [getMinMax(state, nextAgentIndex, height - 1)[0] for state in nextStates]

          if agentIndex == 0: # if pacman, maximize the utility
            return (max(values), values)
          else: # if ghost, minimize the utility
            return (min(values), values)

        numAgents = gameState.getNumAgents()
        depth = self.depth
        height = numAgents * depth # height of the minimax

        value, values = getMinMax(gameState, 0, height)
        index = values.index(value)

        return gameState.getLegalActions(0)[index]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def AlphaBeta(gameState, agentIndex, height, alpha, beta):
          # Returns (Best option of a node, action index)
          # -1 means that no actions, i.e. leaf node
          if gameState.isWin() or gameState.isLose() or height == 0:
            return (self.evaluationFunction(gameState), -1)

          actions = gameState.getLegalActions(agentIndex)
          nextAgentIndex = (agentIndex + 1) % numAgents

          nodeValue = None # Value of current node
          nodeValueIndex = -1 # Action index
          alphaIndex = -1 # Action index
          betaIndex = -1 # Action index

          for idx, action in enumerate(actions):
            nextState = gameState.generateSuccessor(agentIndex, action)
            # Recursion, get best option of each child node
            value = AlphaBeta(nextState, nextAgentIndex, height - 1, alpha, beta)[0]
            
            if nodeValue == None: # initialize the node value
              nodeValue = value
              nodeValueIndex = 0

            if agentIndex == 0: # pacman, Max Node

              if value >= nodeValue: # maximize child's utility
                nodeValue = value
                nodeValueIndex = idx

              if value >= alpha: # maximize best Max option
                alpha = value
                alphaIndex = idx

            else: # ghost, Min Node

              if value <= nodeValue: # minimize child's utility
                nodeValue = value
                nodeValueIndex = idx

              if value <= beta: # minimize best Min option
                beta = value
                betaIndex = idx

            if alpha > beta: # beta-pruning, ignore equality
              return (beta, betaIndex)

          return (nodeValue, nodeValueIndex)

        numAgents = gameState.getNumAgents()
        depth = self.depth
        height = numAgents * depth # height of the minimax tree

        value, index = AlphaBeta(gameState, 0, height, -999999, 999999)
        return gameState.getLegalActions(0)[index]        

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
        def expectimax(gameState, agentIndex, height):
          if gameState.isWin() or gameState.isLose() or height == 0:
            return (self.evaluationFunction(gameState), -1)

          actions = gameState.getLegalActions(agentIndex)
          numChild = len(actions)
          numAgents = gameState.getNumAgents()
          nextAgentIndex = (agentIndex + 1) % numAgents
          nodeValue = None
          actionIndex = -1
          for idx, action in enumerate(actions):
            nextState = gameState.generateSuccessor(agentIndex, action)
            childValue = expectimax(nextState, nextAgentIndex, height - 1)[0]

            if agentIndex == 0: # pacman
              if nodeValue == None: # initialize node value
                nodeValue = childValue
                actionIndex = 0
              if childValue > nodeValue: # maximize child utility
                nodeValue = childValue
                actionIndex = idx
            else: # ghost
              if nodeValue == None:
                nodeValue = 0
              nodeValue += childValue / numChild # calculate the expected utility

          return (nodeValue, actionIndex)

        numAgents = gameState.getNumAgents()
        depth = self.depth
        height = numAgents * depth
        nodeValue, actionIndex = expectimax(gameState, 0, height)
        return gameState.getLegalActions(0)[actionIndex]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    import random
    def getClosestDistance(curNode, nodeList):
      if len(nodeList) == 0:
        return None
      closest = manhattanDistance(curNode, nodeList[0])
      for node in nodeList[1:]:
        dis = manhattanDistance(curNode, node)
        if dis < closest:
          closest = dis
      return closest

    if currentGameState.isWin():
      return 999999
    if currentGameState.isLose():
      return -999999

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()

    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = [ghostState.getPosition() for ghostState in ghostStates]

    scaredGhosts = []
    normalGhosts = []
    for i in range(len(ghostStates)):
      ghost = ghostPositions[i]
      if scaredTimes[i] == 0:
        normalGhosts.append(ghost)
      else:
        scaredGhosts.append(ghost)

    closestScaredGhost = getClosestDistance(pos, scaredGhosts) or 0
    closestNormalGhost = getClosestDistance(pos, normalGhosts)
    if closestNormalGhost == None:
      closestNormalGhost = 0
    elif closestNormalGhost <= 1:
      closestNormalGhost = -999999

    numFood = len(foodList)
    numCapsule = len(capsuleList)
    closestFood = getClosestDistance(pos, foodList) or 0
    currentScore = scoreEvaluationFunction(currentGameState)
    random = random.randint(-2, 2) # avoid stationary

    factors = [
      (currentScore, 10),
      (random, 1),
      (closestNormalGhost, 0.01),
      (closestScaredGhost, -5),
      (numFood, -30),
      (numCapsule, -50),
      (closestFood, -1.75)
    ]

    score = 0
    for value, weight in factors:
      score += value * weight
    return score

# Abbreviation
better = betterEvaluationFunction

