import numpy as np
# import env

class RewardFunctionTerminalPenalty():
    def __init__(self, aliveBouns, deathPenalty, isTerminal):
        self.aliveBouns = aliveBouns
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
    def __call__(self, state, action):
        reward = self.aliveBouns 
        if self.isTerminal(state):
            reward = self.deathPenalty
        return reward

class RewardFunction():
    def __init__(self, aliveBouns):
        self.aliveBouns = aliveBouns
    def __call__(self, state, action):
        reward = self.aliveBouns 
        return reward

def euclideanDistance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))

class RewardFunctionCompete():
    def __init__(self, aliveBouns, catchReward, disDiscountFactor, minXDis):
        self.aliveBouns = aliveBouns
        self.catchReward = catchReward
        self.disDiscountFactor = disDiscountFactor
        self.minXDis = minXDis
    def __call__(self, state, action):
        # print("state", state)
        pos0 = state[0][2:4]
        pos1 = state[1][2:4]
        distance = euclideanDistance(pos0, pos1)
        # print(pos0, pos1, distance)

        if distance <= 2 * self.minXDis:
            catchReward = self.catchReward
        else:
            catchReward = 0

        distanceReward = self.disDiscountFactor * distance

        # reward = np.array([distanceReward - catchReward, -distanceReward + catchReward])
        # print("reward", reward)
        reward = distanceReward - catchReward
        return reward

class CartpoleRewardFunction():
    def __init__(self, aliveBouns):
        self.aliveBouns = aliveBouns
    def __call__(self, state, action):
        distanceBonus = (0.21 - abs(state[2])) / 0.21 + (2.4 - abs(state[0])) / 2.4  
        reward = self.aliveBouns + distanceBonus
        return reward

