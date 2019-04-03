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


class CartpoleRewardFunction():
    def __init__(self, aliveBouns):
        self.aliveBouns = aliveBouns
    def __call__(self, state, action):
        distanceBonus = (0.21 - abs(state[2])) / 0.21 + (2.4 - abs(state[0])) / 2.4  
        reward = self.aliveBouns + distanceBonus
        return reward
