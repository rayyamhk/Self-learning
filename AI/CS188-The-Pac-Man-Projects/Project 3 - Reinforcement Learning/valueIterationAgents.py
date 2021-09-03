import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(iterations):
            nextValues = util.Counter()
            for state in mdp.getStates():
                if mdp.isTerminal(state):
                    continue
                maxQ = None
                for action in mdp.getPossibleActions(state):
                    QValue = self.getQValue(state, action)
                    if maxQ == None or QValue > maxQ:
                        maxQ = QValue
                nextValues[state] = maxQ
            self.values = nextValues # batch update

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        statesAndProbs = mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0
        for nextState, prob in statesAndProbs:
            reward = mdp.getReward(state, action, nextState)
            discount = self.discount
            value = self.getValue(nextState)
            QValue += prob * (reward + discount * value)
        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        if mdp.isTerminal(state):
            return None
        actions = mdp.getPossibleActions(state)
        maxQ = self.getQValue(state, actions[0])
        bestAction = actions[0]
        for action in actions[1:]:
            QValue = self.getQValue(state, action)
            if QValue > maxQ:
                maxQ = QValue
                bestAction = action
        return bestAction
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
