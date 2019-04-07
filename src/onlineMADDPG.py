import tensorflow as tf
import numpy as np
import functools as ft
import twoAgentsChasingEnv as env
import cartpole_env
import reward
import computationGraph as cg
import dataSave 
import random

def approximatePolicyEvaluation(ownStateBatch, actorModel):
    graph = actorModel.graph
    oweState_ = graph.get_tensor_by_name('inputs/ownState_:0')
    evaOwnAction_ = graph.get_tensor_by_name('outputs/evaOwnAction_:0')
    evaOwnActionBatch = actorModel.run(evaOwnAction_, feed_dict = {oweState_ : ownStateBatch})
    return evaOwnActionBatch

def approximatePolicyTarget(ownStateBatch, actorModel):
    graph = actorModel.graph
    oweState_ = graph.get_tensor_by_name('inputs/ownState_:0')
    tarOwnAction_ = graph.get_tensor_by_name('outputs/tarOwnAction_:0')
    tarOwnActionBatch = actorModel.run(tarOwnAction_, feed_dict = {oweState_ : ownStateBatch})
    return tarOwnActionBatch

def approximateQTarget(allAgentStateBatch, ownActionBatch, otherActionBatch, criticModel):
    graph = criticModel.graph
    allAgentState_ = graph.get_tensor_by_name('inputs/allAgentState_:0')
    ownAction_ = graph.get_tensor_by_name('inputs/ownAction_:0')
    otherAction_ = graph.get_tensor_by_name('inputs/otherAction_:0')
    tarQ_ = graph.get_tensor_by_name('outputs/tarQ_:0')
    tarQBatch = criticModel.run(tarQ_, feed_dict = {allAgentState_ : allAgentStateBatch,
                                                   ownAction_ : ownActionBatch,
                                                   otherAction_ : otherActionBatch
                                                   })
    return tarQBatch

def gradientPartialOwnActionFromQEvaluation(allAgentStateBatch, ownActionBatch, otherActionBatch, criticModel):
    graph = criticModel.graph
    allAgentState_ = graph.get_tensor_by_name('inputs/allAgentState_:0')
    ownAction_ = graph.get_tensor_by_name('inputs/ownAction_:0')
    otherAction_ = graph.get_tensor_by_name('inputs/otherAction_:0')
    gradientQPartialOwnAction_ = graph.get_tensor_by_name('outputs/gradientQPartialOwnAction_/evaluationHidden/MatMul_1_grad/MatMul:0')
    gradientQPartialOwnAction = criticModel.run([gradientQPartialOwnAction_], feed_dict = {allAgentState_ : allAgentStateBatch,
                                                                                           ownAction_ : ownActionBatch,
                                                                                           otherAction_ : otherActionBatch
                                                                                          })
    return gradientQPartialOwnAction

class AddActionNoise():
    def __init__(self, actionNoise, noiseDecay, actionLow, actionHigh):
        self.actionNoise = actionNoise
        self.noiseDecay = noiseDecay
        self.actionLow, self.actionHigh = actionLow, actionHigh
        
    def __call__(self, actionPerfect, episodeIndex):
        noisyAction = np.random.normal(actionPerfect, self.actionNoise * (self.noiseDecay ** episodeIndex))
        action = np.clip(noisyAction, self.actionLow, self.actionHigh)
        return action

class Memory():
    def __init__(self, memoryCapacity):
        self.memoryCapacity = memoryCapacity
    def __call__(self, replayBuffer, timeStep):
        replayBuffer.append(timeStep)
        if len(replayBuffer) > self.memoryCapacity:
            numDelete = len(replayBuffer) - self.memoryCapacity
            del replayBuffer[:numDelete]
        return replayBuffer

class TrainActorTensorflow():
    def __init__(self, numAgent, actorWriter):
        self.numAgent = numAgent
        self.actorWriter = actorWriter
    def __call__(self, miniBatch, evaActor, gradientEvaCritic, actorModel, agentIndex):

        if agentIndex == 0:
            return actorModel

        numBatch = len(miniBatch)
        allAgentStates, allAgentActions, allAgentNextStates = list(zip(*miniBatch))

        ownStates = np.array(allAgentStates)[ : , agentIndex]
        ownActions = np.array(allAgentActions)[ : , agentIndex]
        otherActions = np.array(allAgentActions)[ : , list(range(agentIndex)) + list(range(agentIndex + 1, self.numAgent))]
        
        ownStateBatch, allAgentStateBatch, otherActionBatch = np.array(ownStates).reshape(numBatch, -1), np.array(allAgentStates).reshape(numBatch, -1), np.array(otherActions).reshape(numBatch, -1)
        evaActorOwnActionBatch = evaActor(ownStateBatch)
        
        gradientQPartialOwnAction = gradientEvaCritic(allAgentStateBatch, evaActorOwnActionBatch, otherActionBatch)

        graph = actorModel.graph
        state_ = graph.get_tensor_by_name('inputs/ownState_:0')
        gradientQPartialOwnAction_ = graph.get_tensor_by_name('inputs/gradientQPartialOwnAction_:0')
        gradientQPartialActorParameter_ = graph.get_tensor_by_name('outputs/gradientQPartialActorParameter_/evaluationHidden/dense/MatMul_grad/MatMul:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        gradientQPartialActorParameter_, trainOpt = actorModel.run([gradientQPartialActorParameter_, trainOpt_], feed_dict = {state_ : ownStateBatch,
                                                                                                                              gradientQPartialOwnAction_ : gradientQPartialOwnAction[0]  
                                                                                                                              })
        numParams_ = graph.get_tensor_by_name('outputs/numParams_:0')
        numParams = actorModel.run(numParams_)
        updateTargetParameter_ = [graph.get_tensor_by_name('outputs/assign'+str(paramIndex_)+':0') for paramIndex_ in range(numParams)]
        actorModel.run(updateTargetParameter_)
        self.actorWriter.flush()
        return actorModel

class TrainCriticBootstrapTensorflow():
    def __init__(self, numAgent, criticWriter, decay, rewardFunction):
        self.numAgent = numAgent
        self.criticWriter = criticWriter
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, miniBatch, tarActors, tarCritic, criticModel, agentIndex):

        if agentIndex == 0:
            return criticModel
        
        numBatch = len(miniBatch)
        allAgentStates, allAgentActions, allAgentNextStates = list(zip(*miniBatch))

        # TODO: QTarget for mse.
        allAgentRewards = np.array([self.rewardFunction(allAgentState, allAgentAction) for allAgentState, allAgentAction in zip(allAgentStates, allAgentActions)])
        ownRewards = allAgentRewards[ : , agentIndex] 
        
        nextActionsOnEveryTarActor = np.array([tarActors[agent](np.array(allAgentNextStates)[ : , agent].reshape(numBatch, -1)) for agent in range(self.numAgent)])
        tarActorAllAgentNextActions = list(zip(*nextActionsOnEveryTarActor))

        tarActorOwnNextActions = np.array(tarActorAllAgentNextActions)[ : , agentIndex]
        tarActorOtherNextActions = np.array(tarActorAllAgentNextActions)[ : , list(range(agentIndex)) + list(range(agentIndex + 1, self.numAgent))]

        tarActorOwnNextActionBatch = np.array(tarActorOwnNextActions).reshape(numBatch, -1)
        tarActorOtherNextActionBatch = np.array(tarActorOtherNextActions).reshape(numBatch, -1)
         
        allAgentNextStateBatch = np.array(allAgentNextStates).reshape(numBatch, -1)
        tarNextQBatch = tarCritic(allAgentNextStateBatch, tarActorOwnNextActionBatch, tarActorOtherNextActionBatch)
        ownRewardBatch = np.array(ownRewards).reshape(numBatch, -1)
        QAimBatch = ownRewardBatch + self.decay * tarNextQBatch
        
        # TODO: learn by mse.
        ownActions = np.array(allAgentActions)[ : , agentIndex]
        otherActions = np.array(allAgentActions)[ : , list(range(agentIndex)) + list(range(agentIndex + 1, self.numAgent))]

        ownActionBatch = np.array(ownActions).reshape(numBatch, -1)
        otherActionBatch = np.array(otherActions).reshape(numBatch, -1)
        
        graph = criticModel.graph
        allAgentState_ = graph.get_tensor_by_name('inputs/allAgentState_:0')
        ownAction_ = graph.get_tensor_by_name('inputs/ownAction_:0')
        otherAction_ = graph.get_tensor_by_name('inputs/otherAction_:0')
        QAim_ = graph.get_tensor_by_name('inputs/QAim_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict = {allAgentState_ : allAgentNextStateBatch,
                                                                          ownAction_ : ownActionBatch,
                                                                          otherAction_ : otherActionBatch,
                                                                          QAim_ : QAimBatch
                                                                          })
        
        numParams_ = graph.get_tensor_by_name('outputs/numParams_:0')
        numParams = criticModel.run(numParams_)
        updateTargetParameter_ = [graph.get_tensor_by_name('outputs/assign'+str(paramIndex_)+':0') for paramIndex_ in range(numParams)]
        criticModel.run(updateTargetParameter_)
        
        self.criticWriter.flush()
        return criticModel


class MADDPG():
    def __init__(self, numAgent, maxEpisode, maxTimeStep, numMiniBatch, transitionFunction, isTerminal, reset, addActionNoise, savePathActors, savePathCritics, saveRate):
        self.numAgent = numAgent
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.numMiniBatch = numMiniBatch
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset
        self.addActionNoise = addActionNoise
        self.savePathActors = savePathActors
        self.savePathCritics = savePathCritics
        self.saveRate = saveRate

    def __call__(self, actorModels, criticModels, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget, gradientPartialActionFromQEvaluation,
            memory, trainCritic, trainActor):
        replayBuffer = []
        for episodeIndex in range(self.maxEpisode):
            print("Starting episode", episodeIndex)
            allAgentOldState = self.reset(self.numAgent)
            # print("state", allAgentOldState)
            for timeStepIndex in range(self.maxTimeStep):
                evaActors = [lambda ownState: approximatePolicyEvaluation(ownState, actorModel) for actorModel in actorModels]
                allAgentActionBatch = [evaActor(ownState.reshape(1, -1)) for evaActor, ownState in zip(evaActors, allAgentOldState)]
                # print("all agent batch", allAgentActionBatch)
                allAgentActionPerfect = [actionBatch[0] for actionBatch in allAgentActionBatch]
                # print("action perfects", allAgentActionPerfect)
                allAgentAction = np.array([self.addActionNoise(actionPerfect, episodeIndex) for actionPerfect in allAgentActionPerfect])
                # print("final actions", allAgentAction)
                allAgentNewState = self.transitionFunction(allAgentOldState, allAgentAction)
                # print("state", allAgentNewState)
                timeStep = [allAgentOldState, allAgentAction, allAgentNewState]
                replayBuffer = memory(replayBuffer, timeStep)

                # Start training
                if len(replayBuffer) >= self.numMiniBatch and len(replayBuffer) % 100 == 0:
                    miniBatch = random.sample(replayBuffer, self.numMiniBatch)
                    tarActors = [lambda ownState: approximatePolicyEvaluation(ownState, actorModel) for actorModel in actorModels]
                    tarCritics = [lambda allAgentState, ownAction, otherAction: approximateQTarget(allAgentState, ownAction, otherAction, criticModel) for criticModel in criticModels]
                    criticModels = [trainCritic(miniBatch, tarActors, tarCritic, criticModel, agentIndex) for tarCritic, criticModel, agentIndex in 
                            zip(tarCritics, criticModels, range(self.numAgent))]
                    gradientEvaCritics = [lambda allAgentState, ownAction, otherAction: gradientPartialActionFromQEvaluation(allAgentState, ownAction, otherAction,
                            criticModel) for criticModel in criticModels]
                    actorModels = [trainActor(miniBatch, evaActor, gradientEvaCritic, actorModel, agentIndex) for evaActor, gradientEvaCritic, actorModel, agentIndex in
                            zip(evaActors, gradientEvaCritics, actorModels, range(self.numAgent))]

                if self.isTerminal(allAgentOldState):
                    if timeStepIndex != self.maxTimeStep - 1:
                        print("Target caught at time step: ", timeStepIndex)
                    break
                allAgentOldState = allAgentNewState
                

            # Save checkpoints
            if episodeIndex % self.saveRate == 0:
                for i, actorModel in enumerate(actorModels):
                    with actorModel.as_default():
                        with actorModel.graph.as_default():
                            # print(tf.global_variables())
                            actorSaver = tf.train.Saver(tf.global_variables())
                        actorSaver.save(actorModel, self.savePathActors[i])
                
                for i, criticModel in enumerate(criticModels):
                    with criticModel.as_default():
                        with criticModel.graph.as_default():
                            criticSaver = tf.train.Saver(tf.global_variables())
                        criticSaver.save(criticModel, self.savePathCritics[i])
                
                print("Saved models in episode", episodeIndex)

        return actorModels, criticModels

def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)
    
    numAgent = 2
    numActionSpace = 2
    numStateSpace = 6
    actionLow = np.array([-10, -10])
    actionHigh = np.array([10, 10])
    actionRatio = (actionHigh - actionLow) / 2.
    actionNoise = np.array([100.0, 100.0])
    noiseDecay = 0.999

    # numAgents = 1
    # numOtherActionSpace = (numAgents - 1) * numActionSpace
    
    envModelName = 'twoAgentsChasing'
    renderOn = True
    restore = False
    maxTimeStep = 500
    minXDis = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    catchReward = 50
    disRewardDiscount = 0.2
    rewardDecay = 0.99

    memoryCapacity = 100000
    numMiniBatch = 2500

    maxEpisode = 100000
    saveRate = 100

    savePathActors = ['data/tmpModelActor{}.ckpt'.format(agentIndex) for agentIndex in range(numAgent)]
    savePathCritics = ['data/tmpModelCritic{}.ckpt'.format(agentIndex) for agentIndex in range(numAgent)]
 
    actorModels = [cg.createDDPGActorGraph(numStateSpace, numActionSpace, actionRatio, agentIndex) for agentIndex in range(numAgent)]  
    criticModels = [cg.createDDPGCriticGraph(numStateSpace, numActionSpace, numAgent, agentIndex) for agentIndex in range(numAgent)]  
    
    # print("actor models", actorModels)

    # Restore if needed.
    if restore:
        for actorIndex, actorModel in enumerate(actorModels):
            with actorModel.as_default():
                with actorModel.graph.as_default():
                    actorSaver = tf.train.Saver(tf.global_variables())
                actorSaver.restore(actorModel, savePathActors[actorIndex])
        
        for criticIndex, criticModel in enumerate(criticModels):
            with criticModel.as_default():
                with criticModel.graph.as_default():
                    criticSaver = tf.train.Saver(tf.global_variables())
                criticSaver.restore(criticModel, savePathCritics[criticIndex])
            
        print("Restored models from previous training")

    # TODO
    actorWriter = tf.summary.FileWriter('tensorBoard/actorOnlineMADDPG', graph = actorModels[0].graph)
    criticWriter = tf.summary.FileWriter('tensorBoard/criticOnlineMADDPG', graph = criticModels[0].graph)
    
    transitionFunction = env.TransitionFunctionNaivePredator(envModelName, renderOn)
    isTerminal = env.IsTerminal(minXDis)
    reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh)
     
    rewardFunction = reward.RewardFunctionCompete(aliveBouns, catchReward, disRewardDiscount, minXDis)
    
    memory = Memory(memoryCapacity)
 
    trainCritic = TrainCriticBootstrapTensorflow(numAgent, criticWriter, rewardDecay, rewardFunction)
    
    trainActor = TrainActorTensorflow(numAgent, actorWriter) 

    maddpg = MADDPG(numAgent, maxEpisode, maxTimeStep, numMiniBatch, transitionFunction, isTerminal, reset, addActionNoise, savePathActors, savePathCritics, saveRate)

    trainedActorModel, trainedCriticModel = maddpg(actorModels, criticModels, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget,
            gradientPartialOwnActionFromQEvaluation, memory, trainCritic, trainActor)


if __name__ == "__main__":
    main()



