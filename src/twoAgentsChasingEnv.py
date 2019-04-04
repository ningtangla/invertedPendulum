import mujoco_py as mujoco
import os
import numpy as np

#np.random.seed(123)
class Reset():
    def __init__(self, modelName, qPosInitNoise, qVelInitNoise): 
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
    def __call__(self, numAgent):
        numQPos = len(self.simulation.data.qpos)
        numQVel = len(self.simulation.data.qvel)
        numQPosEachAgent = int(numQPos/numAgent)
        numQVelEachAgent = int(numQVel/numAgent)

        qPos = self.simulation.data.qpos + np.random.uniform(low = -self.qPosInitNoise, high = self.qPosInitNoise, size = numQPos)
        qVel = self.simulation.data.qvel + np.random.uniform(low = -self.qVelInitNoise, high = self.qVelInitNoise, size = numQVel)
        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qvel[:] = qVel
        self.simulation.forward()
        xPos = np.concatenate(self.simulation.data.body_xpos[-numAgent: , :numQPosEachAgent])
        startState = np.array([np.concatenate([qPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)], xPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)],
            qVel[numQVelEachAgent * agentIndex : numQVelEachAgent * (agentIndex + 1)]]) for agentIndex in range(numAgent)]) 
        return startState

class TransitionFunction():
    def __init__(self, modelName, renderOn): 
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.numQPos = len(self.simulation.data.qpos)
        self.numQVel = len(self.simulation.data.qvel)
        self.renderOn = renderOn
        if self.renderOn:
            self.viewer = mujoco.MjViewer(self.simulation)
    def __call__(self, allAgentOldState, allAgentAction, renderOpen = False, numSimulationFrames = 1):
        numAgent = len(allAgentOldState)
        numQPosEachAgent = int(self.numQPos/numAgent)
        numQVelEachAgent = int(self.numQVel/numAgent)

        self.simulation.data.ctrl[:] = allAgentAction.flatten()
        
        for i in range(numSimulationFrames):
            self.simulation.step()
            if self.renderOn:
                self.viewer.render()
        newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
        newXPos = np.concatenate(self.simulation.data.body_xpos[-numAgent: , :numQPosEachAgent])
        newState = np.array([np.concatenate([newQPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)], newXPos[numQPosEachAgent * agentIndex : numQPosEachAgent * (agentIndex + 1)],
            newQVel[numQVelEachAgent * agentIndex : numQVelEachAgent * (agentIndex + 1)]]) for agentIndex in range(numAgent)]) 
        return newState

class IsTerminal():
    def __init__(self, maxQPos):
        self.maxQPos = maxQPos
    def __call__(self, state):
        terminal = False
        return terminal   

if __name__ == '__main__':
    transite = TransitionFunction('twoAgentsChasing', renderOn = True)
    for i in range(50000):
        aa = transite([np.zeros(2), np.zeros(2), np.array([0, 0, -0.6]), np.zeros(3)], 0.001)
