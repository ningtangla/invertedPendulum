import unittest
from ddt import ddt, data, unpack
import pandas as pd
import numpy as np
import itertools as it
import math
import reward as targetCode

class NewQState():
    def __init__(self, qvel):
        self.qvel = qvel

@ddt
class TestRewardFunctionWithAllParaEqualOne(unittest.TestCase):
    def setUp(self): 
        self.rewardFunction = targetCode.RewardFunction(1,1,1,1,1,1,np.array([0,0,1.2]))
        
    @data((-1,[NewQState([0,0,0]),np.array([0,0,1.2])],-1),
            (0,[NewQState([0,-1,0]),np.array([0,0,1.2])],-1),
            (0,[NewQState([0,0,-1]),np.array([0,0,1.2])], -1),
            (0,[NewQState([0,0,0]),np.array([-1,0,1.2])],-1),
            (0,[NewQState([0,0,0]),np.array([0,-1,1.2])],-1),
            (0,[NewQState([0,0,0]),np.array([0,0,0.2])],-1),
            (-2,[NewQState([0,0,0]),np.array([0,0,1.2])],-4),
            (0,[NewQState([0,-2,0]),np.array([0,0,1.2])],-4),
            (0,[NewQState([0,0,-2]),np.array([0,0,1.2])], -4),
            (0,[NewQState([0,0,0]),np.array([-2,0,1.2])],-4),
            (0,[NewQState([0,0,0]),np.array([0,-2,1.2])],-4),
            (0,[NewQState([0,0,0]),np.array([0,0,-0.8])],-4)) 
    @unpack
    def testRewardFunction(self, action, newState, rewardGroundTruth):
        oldState = []
        reward = self.rewardFunction(oldState, action, newState)
        self.assertEqual(reward, rewardGroundTruth)
        
    def tearDown(self):
        pass

@ddt
class TestRewardFunctionWithAllParaEqualTwo(unittest.TestCase):
    def setUp(self): 
        self.rewardFunction = targetCode.RewardFunction(2,2,2,2,2,2,np.array([0,0,1.2]))
        
    @data((-1,[NewQState([0,0,0]),np.array([0,0,1.2])],-2),
            (0,[NewQState([0,-1,0]),np.array([0,0,1.2])],-2),
            (0,[NewQState([0,0,-1]),np.array([0,0,1.2])], -2),
            (0,[NewQState([0,0,0]),np.array([-1,0,1.2])],-2),
            (0,[NewQState([0,0,0]),np.array([0,-1,1.2])],-2),
            (0,[NewQState([0,0,0]),np.array([0,0,0.2])],-2)) 
    @unpack
    def testRewardFunction(self, action, newState, rewardGroundTruth):
        oldState = []
        reward = self.rewardFunction(oldState, action, newState)
        self.assertEqual(reward, rewardGroundTruth)
        
    def tearDown(self):
        pass
if __name__ == '__main__':
    print('1') 
    rewardFunctionFirstSuit = unittest.TestLoader().loadTestsFromTestCase(TestRewardFunctionWithAllParaEqualOne)
    rewardFunctionSecondSuit = unittest.TestLoader().loadTestsFromTestCase(TestRewardFunctionWithAllParaEqualTwo)
    unittest.TextTestRunner(verbosity = 2).run(rewardFunctionFirstSuit) 
    unittest.TextTestRunner(verbosity = 2).run(rewardFunctionSecondSuit) 
    

