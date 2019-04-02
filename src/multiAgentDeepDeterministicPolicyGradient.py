import tensorflow as tf
import numpy as np
import functools as ft
import compuationGraph as cg
import env
import cartpole_env
import reward
import dataSave 
import tensorflow_probability as tfp
import random

def approximatePolicyEvaluation(stateBatch, actorModel):
    graph = actorModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    evaAction_ = graph.get_tensor_by_name('outputs/evaAction_:0')
    evaActionBatch = actorModel.run(evaAction_, feed_dict = {state_ : stateBatch})
    return evaActionBatch

def approximatePolicyTarget(stateBatch, actorModel):
    graph = actorModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    tarAction_ = graph.get_tensor_by_name('outputs/tarAction_:0')
    tarActionBatch = actorModel.run(tarAction_, feed_dict = {state_ : stateBatch})
    return tarActionBatch

def approximateQTarget(stateBatch, actionBatch, criticModel):
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    action_ = graph.get_tensor_by_name('inputs/action_:0')
    tarQ_ = graph.get_tensor_by_name('outputs/tarQ_:0')
    tarQBatch = criticModel.run(tarQ_, feed_dict = {state_ : stateBatch,
                                                   action_ : actionBatch
                                                   })
    return tarQBatch

def gradientPartialActionFromQEvaluation(stateBatch, actionBatch, criticModel):
    criticGraph = criticModel.graph
    state_ = criticGraph.get_tensor_by_name('inputs/state_:0')
    action_ = criticGraph.get_tensor_by_name('inputs/action_:0')
    gradientQPartialAction_ = criticGraph.get_tensor_by_name('outputs/gradientQPartialAction_/evaluationHidden/MatMul_1_grad/MatMul:0')
    gradientQPartialAction = criticModel.run([gradientQPartialAction_], feed_dict = {state_ : stateBatch,
                                                                                     action_ : actionBatch,
                                                                                     })
    return gradientQPartialAction


class AddActionNoise():
    def __init__(self, actionNoise, noiseDecay, actionLow, actionHigh):
        self.actionNoise = actionNoise
        self.noiseDecay = noiseDecay
        self.actionLow, self.actionHigh = actionLow, actionHigh
        
    def __call__(self, actionPerfect, episodeIndex):
        noisyAction = np.random.normal(actionPerfect, self.actionNoise * (self.noiseDecay ** episodeIndex))
        action = np.clip(noisyAction, self.actionLow, self.actionHigh)
        return action

class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset, addActionNoise):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset
        self.addActionNoise = addActionNoise

    def __call__(self, evaActor, episodeIndex): 
        oldState = self.reset()
        trajectory = []
        
        for time in range(self.maxTimeStep): 
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = evaActor(oldStateBatch) 
            actionPerfect = actionBatch[0]
            action = self.addActionNoise(actionPerfect, episodeIndex)
            #action = actionNoNoise
            # actionBatch shape: batch * action Dimension; only keep action Dimention in shape
            newState = self.transitionFunction(oldState, action) 
            trajectory.append([oldState, action, newState])
            terminal = self.isTerminal(oldState)
            if terminal:
                break
            oldState = newState
            
        return trajectory

class Memory():
    def __init__(self, memoryCapacity):
        self.memoryCapacity = memoryCapacity
    def __call__(self, replayBuffer, episode):
        #noLastStateEpisode = [trajectory[ : -1] for trajectory in episode]
        #mergedNoLastStateEpisode = np.concatenate(noLastStateEpisode)
        #states, actions = list(zip(*mergedNoLastStateEpisode)) 
        #
        #noFirstStateEpisode = [trajectory[1 : ] for trajectory in episode]
        #mergedNoFirstStateEpisode = np.concatenate(noFirstStateEpisode)
        #nextStates, nextActions = list(zip(*mergedNoFirstStateEpisode))
        #episode = list(zip(states, actions, nextStates))
        replayBuffer.extend(np.concatenate(episode))

        if len(replayBuffer) > self.memoryCapacity:
            numDelete = len(replayBuffer) - self.memoryCapacity
            del replayBuffer[numDelete : ]
        return replayBuffer

class SampleMiniBatch():
    def __init__(self, numMiniBatch):
        self.numMiniBatch = numMiniBatch
    def __call__(self, replayBuffer):
        numSample = self.numMiniBatch
        if len(replayBuffer) < self.numMiniBatch:
            numSample = len(replayBuffer)
        miniBatch = random.sample(replayBuffer, numSample)
        return miniBatch
        
class TrainCriticBootstrapTensorflow():
    def __init__(self, criticWriter, decay, rewardFunction):
        self.criticWriter = criticWriter
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, miniBatch, tarActor, tarCritic, criticModel):
        
        states, actions, nextStates = list(zip(*miniBatch))
        stateBatch, actionBatch, nextStateBatch = np.vstack(states), np.vstack(actions), np.vstack(nextStates)
        
        nextTargetActionBatch = tarActor(nextStateBatch)

        nextTargetQBatch = tarCritic(nextStateBatch, nextTargetActionBatch)
        
        rewardsEpisode = np.array([self.rewardFunction(state, action) for state, action in zip(stateBatch, actionBatch)])
        rewardBatch = np.vstack(rewardsEpisode)
        QTargetBatch = rewardBatch + self.decay * nextTargetQBatch
        
        criticGraph = criticModel.graph
        state_ = criticGraph.get_tensor_by_name('inputs/state_:0')
        action_ = criticGraph.get_tensor_by_name('inputs/action_:0') 
        QTarget_ = criticGraph.get_tensor_by_name('inputs/QTarget_:0')
        loss_ = criticGraph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = criticGraph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                          action_ : actionBatch,
                                                                          QTarget_ : QTargetBatch
                                                                          })
        
        numParams_ = criticGraph.get_tensor_by_name('outputs/numParams_:0')
        numParams = criticModel.run(numParams_)
        updateTargetParameter_ = [criticGraph.get_tensor_by_name('outputs/assign'+str(paramIndex_)+':0') for paramIndex_ in range(numParams)]
        criticModel.run(updateTargetParameter_)
        
        self.criticWriter.flush()
        return criticModel

class TrainActorTensorflow():
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, miniBatch, evaActor, gradientEvaCritic, actorModel):

        states, actions, nextStates = list(zip(*miniBatch))
        stateBatch = np.vstack(states)
        evaActorActionBatch = evaActor(stateBatch)
        
        gradientQPartialAction = gradientEvaCritic(stateBatch, evaActorActionBatch)

        actorGraph = actorModel.graph
        state_ = actorGraph.get_tensor_by_name('inputs/state_:0')
        gradientQPartialAction_ = actorGraph.get_tensor_by_name('inputs/gradientQPartialAction_:0')
        gradientQPartialActorParameter_ = actorGraph.get_tensor_by_name('outputs/gradientQPartialActorParameter_/evaluationHidden/dense/MatMul_grad/MatMul:0')
        trainOpt_ = actorGraph.get_operation_by_name('train/adamOpt_')
        gradientQPartialActorParameter, trainOpt = actorModel.run([gradientQPartialActorParameter_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                                                                              gradientQPartialAction_ : gradientQPartialAction[0]  
                                                                                                                              })
        numParams_ = actorGraph.get_tensor_by_name('outputs/numParams_:0')
        numParams = actorModel.run(numParams_)
        updateTargetParameter_ = [actorGraph.get_tensor_by_name('outputs/assign'+str(paramIndex_)+':0') for paramIndex_ in range(numParams)]
        actorModel.run(updateTargetParameter_)
        self.actorWriter.flush()
        return actorModel

class DeepDeterministicPolicyGradient():
    def __init__(self, numTrajectory, maxEpisode):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
    def __call__(self, actorModels, criticModels, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget, gradientPartialActionFromQEvaluation, sampleTrajectory,
            memory, sampleMiniBatch, trainCritic, trainActor):
        replayBuffer = []
        for episodeIndex in range(self.maxEpisode):
            evaActors = [lambda state: approximatePolicyEvaluation(state, actorModel) for actorModel in actorModels]
            episode = [sampleTrajectory(evaActors, episodeIndex) for index in range(self.numTrajectory)]
            replayBuffer = memory(replayBuffer, episode)
            miniBatch = sampleMiniBatch(replayBuffer) 
            tarActors = [lambda state: approximatePolicyEvaluation(state, actorModel) for actorModel in actorModels]
            tarCritics = [lambda state, action: approximateQTarget(state, action, criticModel) for criticModel in criticModels]
            criticModels = [trainCritic(miniBatch, tarActor, tarCritic, criticModel) for tarActor, tarCriti, criticModel in zip(tarActors, tarCritics, criticModels)]
            gradientEvaCritics = [lambda state, action: gradientPartialActionFromQEvaluation(state, action, criticModel) for criticModel in criticModels]
            actorModels = trainActor(miniBatch, evaActor, gradientEvaCritic, actorModel) for evaActor, gradientEvaCritic, actorModel in zip(evaActors, gradientEvaCritics, actorModels]
            print(np.mean([len(episode[index]) for index in range(self.numTrajectory)]))
        return actorModels, criticModels

def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)

    numActionSpace = 1
    numStateSpace = 4
    actionLow = -3
    actionHigh = 3
    actionRatio = (actionHigh - actionLow) / 2.

    envModelName = 'inverted_pendulum'
    renderOn = False
    maxTimeStep = 200
    maxQPos = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    rewardDecay = 0.99

    memoryCapacity = 100000
    numMiniBatch = 64

    numTrajectory = 1 
    maxEpisode = 100000

    learningRateActor = 0.0001
    learningRateCritic = 0.001
 
    savePathActor = 'data/tmpModelActor.ckpt'
    savePathCritic = 'data/tmpModelCritic.ckpt'
    
    softReplaceRatio = 0.01

    numAgent = 1

    actorModels = [cg.createDDPGActorGraph(numStateSpace, numActionSpace, softReplaceRatio, agentIndex) for agentIndex in range(numAgent)]
    criticModels = [cg.createDDPGCriticGraph(numStateSpace, numActionSpace, softReplaceRatio, agentIndex) for agentIndex in range(numAgent)]
    
    #transitionFunction = env.TransitionFunction(envModelName, renderOn)
    #isTerminal = env.IsTerminal(maxQPos)
    #reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    transitionFunction = cartpole_env.Cartpole_continuous_action_transition_function(renderOn = False)
    isTerminal = cartpole_env.cartpole_done_function
    reset = cartpole_env.cartpole_get_initial_state
    
    sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)
    
    #rewardFunction = reward.RewardFunction(aliveBouns)
    rewardFunction = reward.CartpoleRewardFunction(aliveBouns)
    #accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)
 
    memory = Memory(memoryCapacity)

    sampleMiniBatch = SampleMiniBatch(numMiniBatch)
    
    trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay, rewardFunction)
    
    trainActor = TrainActorTensorflow(actorWriter) 

    deepDeterministicPolicyGradient = DeepDeterministicPolicyGradient(numTrajectory, maxEpisode)

    trainedActorModels, trainedCriticModels = deepDeterministicPolicyGradient(actorModels, criticModels, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget,
            gradientPartialActionFromQEvaluation, sampleTrajectory, memory, sampleMiniBatch, trainCritic, trainActor)

    #with actorModel.as_default():
    #    actorSaver.save(trainedActorModel, savePathActor)
    #with criticModel.as_default():
    #    criticSaver.save(trainedCriticModel, savePathCritic)

    transitionPlay = cartpole_env.Cartpole_continuous_action_transition_function(renderOn = True)
    samplePlay = SampleTrajectory(maxTimeStep, transitionPlay, isTerminal, reset)
    actors = [lambda state: approximatePolicy(state, trainedActorModel) for trainedActorModel in trainedActorModels]
    playEpisode = [samplePlay(actors) for index in range(5)]
    print(np.mean([len(playEpisode[index]) for index in range(5)]))

if __name__ == "__main__":
    main()


