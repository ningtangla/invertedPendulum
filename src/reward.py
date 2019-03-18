import numpy as np

class RewardFunction():
    def __init__(self, aliveBouns):
        self.aliveBouns = aliveBouns
    def __call__(self, oldState, action):
        reward = self.aliveBouns 
        return reward


    
