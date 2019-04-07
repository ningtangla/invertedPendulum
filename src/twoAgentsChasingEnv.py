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

class TransitionFunctionNaivePredator():
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

        # print("state", allAgentOldState)
        # print("old action", allAgentAction)

        preyState = allAgentOldState[0][numQPosEachAgent: numQPosEachAgent + 2]
        predatorState = allAgentOldState[1][numQPosEachAgent: numQPosEachAgent + 2]
        # print("prey state", preyState)
        # print("predator state", predatorState)
        predatorAction = preyState - predatorState
        # print("action", predatorAction)
        allAgentAction[1] = predatorAction
        # rand1 = np.random.random()
        # rand2 = np.random.random()
        # allAgentAction[0][0] *= rand1
        # allAgentAction[0][1] *= rand2
        print("new action", allAgentAction)

        allAgentOldQPos = allAgentOldState[:, 0:numQPosEachAgent].flatten()
        allAgentOldQVel = allAgentOldState[:, -numQVelEachAgent:].flatten()

        self.simulation.data.qpos[:] = allAgentOldQPos
        self.simulation.data.qvel[:] = allAgentOldQVel
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

        allAgentOldQPos = allAgentOldState[:, 0:numQPosEachAgent].flatten()
        allAgentOldQVel = allAgentOldState[:, -numQVelEachAgent:].flatten()

        self.simulation.data.qpos[:] = allAgentOldQPos
        self.simulation.data.qvel[:] = allAgentOldQVel
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

def euclideanDistance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))

class IsTerminal():
    def __init__(self, minXDis):
        self.minXDis = minXDis
    def __call__(self, state):
        # Assume only two agents. get x position
        pos0 = state[0][2:4]
        pos1 = state[1][2:4]
        distance = euclideanDistance(pos0, pos1)
        # print(state, type(state), len(state))
        # print("state", state)
        # print("state 0", pos0)
        # print("state 1", pos1)
        # print("distance", distance)
        terminal = (distance <= 2 * self.minXDis)
        # terminal = False
        return terminal   

if __name__ == '__main__':
    transite = TransitionFunction('twoAgentsChasing', renderOn = True)
    for i in range(50000):
        aa = transite([np.zeros(2), np.zeros(2), np.array([0, 0, -0.6]), np.zeros(3)], 0.001)
