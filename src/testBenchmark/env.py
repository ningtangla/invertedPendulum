import mujoco_py as mujoco
import os
import numpy as np

class Reset():
    def __init__(self, modelName, qPosInitNoise, qVelInitNoise): 
        model = mujoco.load_model_from_path('xmls/' + modelName + '.xml')
        self.simulation = mujoco.MjSim(model)
        self.qPosInitNoise = qPosInitNoise
        self.qVelInitNoise = qVelInitNoise
    def __call__(self):
        qPos = self.simulation.data.qpos + np.random.uniform(low = -self.qPosInitNoise, high = self.qPosInitNoise, size = len(self.simulation.data.qpos))
        qVel = self.simulation.data.qvel + np.random.uniform(low = -self.qVelInitNoise, high = self.qVelInitNoise, size = len(self.simulation.data.qvel))
        self.simulation.data.qpos[:] = qPos
        self.simulation.data.qpos[:] = qVel
        self.simulation.forward()
        startState = np.concatenate([qPos, qVel])
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
    def __call__(self, oldState, action, renderOpen = False, numSimulationFrames = 1):
        oldQPos = oldState[0 : self.numQPos]
        oldQVel = oldState[self.numQPos : self.numQPos + self.numQVel]
        self.simulation.data.qpos[:] = oldQPos
        self.simulation.data.qvel[:] = oldQVel
        self.simulation.data.ctrl[:] = action
        for i in range(numSimulationFrames):
            self.simulation.step()
            if self.renderOn:
                self.viewer.render()
        newQPos, newQVel = self.simulation.data.qpos, self.simulation.data.qvel
        newState = np.concatenate([newQPos, newQVel])
        return newState

class IsTerminal():
    def __init__(self, maxQPos):
        self.maxQPos = maxQPos
    def __call__(self, state):
        terminal = not (np.isfinite(state).all() and np.abs(state[1]) <= self.maxQPos)
        return terminal   

if __name__ == '__main__':
    #transite = TransitionFunction('inverted_pendulum')
    #aa = transite([np.zeros(2), np.zeros(2), np.array([0, 0, -0.6]), np.zeros(3)], 0.001)
    reset = Reset('inverted_pendulum', 0.001, 0.001)
    bb = reset()
