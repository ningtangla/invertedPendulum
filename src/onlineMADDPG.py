import tensorflow as tf
import numpy as np
import functools as ft
# import env
import cartpole_env
import reward
import dataSave 
# import tensorflow_probability as tfp
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

def approximateQTarget(stateBatch, actionBatch, criticModel, agentIndex):
    ownActionBatch = [actions[agentIndex] for actions in actionBatches]
    otherActionBatch = [[actions[actionIndex] for actionIndex in range(len(actions)) if actionIndex != agentIndex] for actions in actionBatches]
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    own_action_ = graph.get_tensor_by_name('inputs/own_action_:0')
    other_action_ = criticGraph.get_tensor_by_name('inputs/other_action_:0')
    tarQ_ = graph.get_tensor_by_name('outputs/tarQ_:0')
    tarQBatch = criticModel.run(tarQ_, feed_dict = {state_ : stateBatch,
                                                   own_action_ : ownActionBatch,
                                                   other_action_ : otherActionBatch
                                                   })
    return tarQBatch

def gradientPartialActionFromQEvaluation(stateBatch, actionBatch, criticModel, agentIndex):
    ownActionBatch = [actions[agentIndex] for actions in actionBatches]
    otherActionBatch = [[actions[actionIndex] for actionIndex in range(len(actions)) if actionIndex != agentIndex] for actions in actionBatches]
    criticGraph = criticModel.graph
    state_ = criticGraph.get_tensor_by_name('inputs/state_:0')
    own_action_ = criticGraph.get_tensor_by_name('inputs/own_action_:0')
    other_action_ = criticGraph.get_tensor_by_name('inputs/other_action_:0')
    gradientQPartialAction_ = criticGraph.get_tensor_by_name('outputs/gradientQPartialAction_/evaluationHidden/MatMul_1_grad/MatMul:0')
    gradientQPartialAction = criticModel.run([gradientQPartialAction_], feed_dict = {state_ : stateBatch,
                                                                                     own_action_ : ownActionBatch,
                                                                                     other_action_ : otherActionBatch
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
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, miniBatch, evaActor, gradientEvaCritic, actorModel, agentIndex):

        numBatch = len(miniBatch)
        states, actions, nextStates = list(zip(*miniBatch))
        ownState = [state[agentIndex] for state in states]
        ownStateBatch = np.array(ownState).reshape(numBatch, -1)
        evaActorActionBatch = evaActor(ownStateBatch)
        
        gradientQPartialAction = gradientEvaCritic(ownStateBatch, evaActorActionBatch)

        actorGraph = actorModel.graph
        state_ = actorGraph.get_tensor_by_name('inputs/state_:0')
        gradientQPartialAction_ = actorGraph.get_tensor_by_name('inputs/gradientQPartialAction_:0')
        gradientQPartialActorParameter_ = actorGraph.get_tensor_by_name('outputs/gradientQPartialActorParameter_/evaluationHidden/dense/MatMul_grad/MatMul:0')
        trainOpt_ = actorGraph.get_operation_by_name('train/adamOpt_')
        gradientQPartialActorParameter_, trainOpt = actorModel.run([gradientQPartialActorParameter_, trainOpt_], feed_dict = {state_ : ownStateBatch,
                                                                                                                              gradientQPartialAction_ : gradientQPartialAction[0]  
                                                                                                                              })
        numParams_ = actorGraph.get_tensor_by_name('outputs/numParams_:0')
        numParams = actorModel.run(numParams_)
        updateTargetParameter_ = [actorGraph.get_tensor_by_name('outputs/assign'+str(paramIndex_)+':0') for paramIndex_ in range(numParams)]
        actorModel.run(updateTargetParameter_)
        self.actorWriter.flush()
        return actorModel

class TrainCriticBootstrapTensorflow():
    def __init__(self, criticWriter, decay, rewardFunction):
        self.criticWriter = criticWriter
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, miniBatch, tarActor, tarCritic, criticModel, agentIndex):
        
        numBatches = len(miniBatch)
        states, actions, nextStates = list(zip(*miniBatch))

        # separate own action and other actions
        ownActions = [action[agentIndex] for action in actions]
        otherActions = [[action[actionIndex] for actionIndex in range(len(action)) if actionIndex != agentIndex] for action in actions]

        rewards = np.array([self.rewardFunction(state, action_n[agentIndex]) for state, action_n in zip(states, actions)])
        
        # TODO: expand the state to include observation of all agents, then make into batch.
        stateBatch = np.array(states).reshape(numMiniBatch, -1)
        nextStateBatch = np.array(nextStates).reshape(numMiniBatch, -1)
        rewardBatch = np.array(rewards).reshape(numMiniBatch, -1)
        ownActionBatch = np.array(ownActions).reshape(numMiniBatch, -1)
        otherActionBatch = np.array(otherActions).reshape(numMiniBatch, -1)

        nextTargetActionBatch = tarActor(nextStateBatch)

        nextTargetQBatch = tarCritic(nextStateBatch, nextTargetActionBatch)
        
        QTargetBatch = rewardBatch + self.decay * nextTargetQBatch
        
        criticGraph = criticModel.graph
        state_ = criticGraph.get_tensor_by_name('inputs/state_:0')
        own_action_ = criticGraph.get_tensor_by_name('inputs/own_action_:0') 
        other_action_ = criticGraph.get_tensor_by_name('inputs/other_action_:0') 
        QTarget_ = criticGraph.get_tensor_by_name('inputs/QTarget_:0')
        loss_ = criticGraph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = criticGraph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                          own_action_ : actionBatch,
                                                                          other_action_ : otherActionBatch,
                                                                          QTarget_ : QTargetBatch
                                                                          })
        
        numParams_ = criticGraph.get_tensor_by_name('outputs/numParams_:0')
        numParams = criticModel.run(numParams_)
        updateTargetParameter_ = [criticGraph.get_tensor_by_name('outputs/assign'+str(paramIndex_)+':0') for paramIndex_ in range(numParams)]
        criticModel.run(updateTargetParameter_)
        
        self.criticWriter.flush()
        return criticModel


class MADDPG():
    def __init__(self, maxEpisode, maxTimeStep, numMiniBatch, transitionFunction, isTerminal, reset, addActionNoise):
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.numMiniBatch = numMiniBatch
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset
        self.addActionNoise = addActionNoise

    def __call__(self, actorModels, criticModels, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget, gradientPartialActionFromQEvaluation,
            memory, trainCritic, trainActor):
        replayBuffer = []
        for episodeIndex in range(self.maxEpisode):
            oldState = self.reset()
            for timeStepIndex in range(self.maxTimeStep):
                evaActors = [lambda state: approximatePolicyEvaluation(state, actorModel) for actorModel in actorModels]
                actionBatches = [evaActor(ownState.reshape(1, -1)) for evaActor, ownState in zip(evaActors, oldState)]
                actionPerfects = [actionBatch[0] for actionBatch in actionBatches]
                actions = [self.addActionNoise(actionPerfect, episodeIndex) for actionPerfect in actionPerfects]
                newState = self.transitionFunction(oldState, actions)
                timeStep = [oldState, actions, newState]
                replayBuffer = memory(replayBuffer, timeStep)

                # Start training
                if len(replayBuffer) >= self.numMiniBatch:
                    miniBatch = random.sample(replayBuffer, self.numMiniBatch)
                    tarActors = [lambda state: approximatePolicyEvaluation(state, actorModel) for actorModel in actorModels]
                    tarCritics = [lambda state, action: approximateQTarget(state, action[agentIndex], criticModel, agentIndex) for agentIndex, criticModel in enumerate(criticModels)]
                    criticModels = [trainCritic(miniBatch, tarActor, tarCritic, criticModel, agentIndex) for agentIndex, (tarActor, tarCritic, criticModel) in enumerate(zip(tarActors, tarCritics, criticModels))]
                    gradientEvaCritics = [lambda state, action: gradientPartialActionFromQEvaluation(state, action, criticModel, agentIndex) for agentIndex, criticModel in enumerate(criticModels)]
                    actorModels = [trainActor(miniBatch, evaActor, gradientEvaCritic, actorModel, agentIndex) for agentIndex, (evaActor, gradientEvaCritic, actorModel) in enumerate(zip(evaActors, gradientEvaCritics, actorModels))]

                if self.isTerminal(oldState):
                    break
                oldState = newState

        return actorModels, criticModels

def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)

    numActionSpace = 1
    numStateSpace = 4
    actionLow = -2
    actionHigh = 2
    actionRatio = (actionHigh - actionLow) / 2.
    actionNoise = 0.1
    noiseDecay = 0.999

    numAgents = 1
    numOtherActionSpace = (numAgents - 1) * numActionSpace

    envModelName = 'inverted_pendulum'
    renderOn = False
    maxTimeStep = 200
    maxQPos = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    deathPenalty = -20
    rewardDecay = 0.99

    memoryCapacity = 100000
    numMiniBatch = 2

    maxEpisode = 100000

    numActorFC1Unit = 20
    numActorFC2Unit = 20
    numCriticFC1Unit = 100
    numCriticFC2Unit = 100
    learningRateActor = 0.0001
    learningRateCritic = 0.001
    l2DecayCritic = 0.0000001

    savePathActor = 'data/tmpModelActor.ckpt'
    savePathCritic = 'data/tmpModelCritic.ckpt'
    
    softReplaceRatio = 0.001

    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.variable_scope("inputs"):
            state_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="state_"))
            gradientQPartialAction_ = tf.placeholder(tf.float32, [None, numActionSpace], name="gradientQPartialAction_")

        with tf.variable_scope("evaluationHidden"):
            evaFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = numActorFC1Unit, activation = tf.nn.relu))
            evaFullyConnected2_ = tf.layers.batch_normalization(tf.layers.dense(inputs = evaFullyConnected1_, units = numActorFC2Unit, activation = tf.nn.relu))
            evaActionActivation_ = tf.layers.dense(inputs = evaFullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
            
        with tf.variable_scope("targetHidden"):
            tarFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = numActorFC1Unit, activation = tf.nn.relu))
            tarFullyConnected2_ = tf.layers.batch_normalization(tf.layers.dense(inputs = tarFullyConnected1_, units = numActorFC2Unit, activation = tf.nn.relu))
            tarActionActivation_ = tf.layers.dense(inputs = tarFullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
        
        with tf.variable_scope("outputs"):        
            evaParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluationHidden')
            tarParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            numParams_ = tf.constant(len(evaParams_), name = 'numParams_')
            updateTargetParameter_ = [tf.assign(tarParam_, (1 - softReplaceRatio) * tarParam_ + softReplaceRatio * evaParam_, name = 'assign'+str(paramIndex_)) for paramIndex_,
                tarParam_, evaParam_ in zip(range(len(evaParams_)), tarParams_, evaParams_)]
            evaAction_ = tf.multiply(evaActionActivation_, actionRatio, name = 'evaAction_')
            tarAction_ = tf.multiply(tarActionActivation_, actionRatio, name = 'tarAction_')
            gradientQPartialActorParameter_ = tf.gradients(ys = evaAction_, xs = evaParams_, grad_ys = gradientQPartialAction_, name = 'gradientQPartialActorParameter_')

        with tf.variable_scope("train"):
            #-learningRate for ascent
            trainOpt_ = tf.train.AdamOptimizer(-learningRateActor, name = 'adamOpt_').apply_gradients(zip(gradientQPartialActorParameter_, evaParams_))
        actorInit = tf.global_variables_initializer()
        
        actorSummary = tf.summary.merge_all()
        actorSaver = tf.train.Saver(tf.global_variables())

    actorWriter = tf.summary.FileWriter('tensorBoard/actorOnlineDDPG', graph = actorGraph)
    actorModel = [tf.Session(graph = actorGraph)]
    actorModel[0].run(actorInit)    
    
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.variable_scope("inputs"):
            state_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="state_"))
            own_action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, numActionSpace]), name='own_action_')
            other_action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, numOtherActionSpace]), name='other_action_')
            QTarget_ = tf.placeholder(tf.float32, [None, 1], name="QTarget_")

        with tf.variable_scope("evaluationHidden"):
            evaFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = numCriticFC1Unit, activation = tf.nn.relu))
            numFullyConnected2Units = numCriticFC2Unit
            evaStateFC1ToFullyConnected2Weights_ = tf.get_variable(name='evaStateFC1ToFullyConnected2Weights', shape = [numCriticFC1Unit, numFullyConnected2Units])
            evaOwnActionToFullyConnected2Weights_ = tf.get_variable(name='evaOwnActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units])
            evaOtherActionToFullyConnected2Weights_ = tf.get_variable(name='evaOtherActionToFullyConnected2Weights', shape = [numOtherActionSpace, numFullyConnected2Units])
            evaFullyConnected2Bias_ = tf.get_variable(name = 'evaFullyConnected2Bias', shape = [numFullyConnected2Units])
            evaFullyConnected2_ = tf.nn.relu(tf.matmul(evaFullyConnected1_, evaStateFC1ToFullyConnected2Weights_) + tf.matmul(own_action_, evaOwnActionToFullyConnected2Weights_) + tf.matmul(other_action_, evaOtherActionToFullyConnected2Weights_) + evaFullyConnected2Bias_ )
            evaQActivation_ = tf.layers.dense(inputs = evaFullyConnected2_, units = 1, activation = None, )

        with tf.variable_scope("targetHidden"):
            tarFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = numCriticFC1Unit, activation = tf.nn.relu))
            numFullyConnected2Units = numCriticFC2Unit
            tarStateFC1ToFullyConnected2Weights_ = tf.get_variable(name='tarStateFC1ToFullyConnected2Weights', shape = [numCriticFC1Unit, numFullyConnected2Units])
            tarOwnActionToFullyConnected2Weights_ = tf.get_variable(name='tarOwnActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units])
            tarOtherActionToFullyConnected2Weights_ = tf.get_variable(name='tarOtherActionToFullyConnected2Weights', shape = [numOtherActionSpace, numFullyConnected2Units])
            tarFullyConnected2Bias_ = tf.get_variable(name = 'tarFullyConnected2Bias', shape = [numFullyConnected2Units])
            tarFullyConnected2_ = tf.nn.relu(tf.matmul(tarFullyConnected1_, tarStateFC1ToFullyConnected2Weights_) + tf.matmul(own_action_, tarOwnActionToFullyConnected2Weights_) + tf.matmul(other_action_, tarOtherActionToFullyConnected2Weights_) + tarFullyConnected2Bias_ )
            tarQActivation_ = tf.layers.dense(inputs = tarFullyConnected2_, units = 1, activation = None)
        
        with tf.variable_scope("outputs"):        
            evaParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluationHidden')
            tarParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            numParams_ = tf.constant(len(evaParams_), name = 'numParams_')
            updateTargetParameter_ = [tf.assign(tarParam_, (1 - softReplaceRatio) * tarParam_ + softReplaceRatio * evaParam_, name = 'assign'+str(paramIndex_)) for paramIndex_,
                tarParam_, evaParam_ in zip(range(len(evaParams_)), tarParams_, evaParams_)]
            evaQ_ = tf.multiply(evaQActivation_, 1, name = 'evaQ_')
            tarQ_ = tf.multiply(tarQActivation_, 1, name = 'tarQ_')
            diff_ = tf.subtract(QTarget_, evaQ_, name = 'diff_')
            loss_ = tf.reduce_mean(tf.square(diff_), name = 'loss_')
            gradientQPartialAction_ = tf.gradients(evaQ_, own_action_, name = 'gradientQPartialAction_')
            criticLossSummary = tf.summary.scalar("CriticLoss", loss_)
        with tf.variable_scope("train"):
            trainOpt_ = tf.contrib.opt.AdamWOptimizer(weight_decay = l2DecayCritic, learning_rate = learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticWriter = tf.summary.FileWriter('tensorBoard/criticOnlineDDPG', graph = criticGraph)
    criticModel = [tf.Session(graph = criticGraph)]
    criticModel[0].run(criticInit)   
     
    #transitionFunction = env.TransitionFunction(envModelName, renderOn)
    #isTerminal = env.IsTerminal(maxQPos)
    #reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    transitionFunction = cartpole_env.Cartpole_continuous_action_transition_function(renderOn = False)
    isTerminal = cartpole_env.cartpole_done_function
    reset = cartpole_env.cartpole_get_initial_state
    addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh)
     
    rewardFunction = reward.RewardFunctionTerminalPenalty(aliveBouns, deathPenalty, isTerminal)
    #rewardFunction = reward.CartpoleRewardFunction(aliveBouns)
    
    memory = Memory(memoryCapacity)
 
    trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay, rewardFunction)
    
    trainActor = TrainActorTensorflow(actorWriter) 

    maddpg = MADDPG(maxEpisode, maxTimeStep, numMiniBatch, transitionFunction, isTerminal, reset, addActionNoise)

    trainedActorModel, trainedCriticModel = maddpg(actorModel, criticModel, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget,
            gradientPartialActionFromQEvaluation, memory, trainCritic, trainActor)

    with actorModel.as_default():
        actorSaver.save(trainedActorModel, savePathActor)
    with criticModel.as_default():
        criticSaver.save(trainedCriticModel, savePathCritic)

if __name__ == "__main__":
    main()



