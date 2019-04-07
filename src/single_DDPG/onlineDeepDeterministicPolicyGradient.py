import tensorflow as tf
import numpy as np
import twoAgentsChasingEnv as env
import reward
import random
import time

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

class Memory():
    def __init__(self, memoryCapacity):
        self.memoryCapacity = memoryCapacity
    def __call__(self, replayBuffer, timeStep):
        replayBuffer.append(timeStep)
        if len(replayBuffer) > self.memoryCapacity:
            numDelete = len(replayBuffer) - self.memoryCapacity
            del replayBuffer[numDelete : ]
        return replayBuffer

class TrainCriticBootstrapTensorflow():
    def __init__(self, criticWriter, decay, rewardFunction):
        self.criticWriter = criticWriter
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, miniBatch, tarActor, tarCritic, criticModel):
        
        states, actions, nextStates = list(zip(*miniBatch))
        numBatch = len(miniBatch)
        rewards = np.array([self.rewardFunction(state, action) for state, action in zip(states, actions)])
        stateBatch, actionBatch, nextStateBatch = np.array(states).reshape(numBatch, -1), np.array(actions).reshape(numBatch, -1), np.array(nextStates).reshape(numBatch, -1),
        rewardBatch = np.array(rewards).reshape(numBatch, -1)
        
        nextTargetActionBatch = tarActor(nextStateBatch)

        nextTargetQBatch = tarCritic(nextStateBatch, nextTargetActionBatch)
        
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
        return loss, criticModel

class TrainActorTensorflow():
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, miniBatch, evaActor, gradientEvaCritic, actorModel):

        states, actions, nextStates = list(zip(*miniBatch))
        numBatch = len(miniBatch)
        stateBatch = np.array(states).reshape(numBatch, -1)
        evaActorActionBatch = evaActor(stateBatch)
        
        gradientQPartialAction = gradientEvaCritic(stateBatch, evaActorActionBatch)

        actorGraph = actorModel.graph
        state_ = actorGraph.get_tensor_by_name('inputs/state_:0')
        gradientQPartialAction_ = actorGraph.get_tensor_by_name('inputs/gradientQPartialAction_:0')
        gradientQPartialActorParameter_ = actorGraph.get_tensor_by_name('outputs/gradientQPartialActorParameter_/evaluationHidden/dense/MatMul_grad/MatMul:0')
        trainOpt_ = actorGraph.get_operation_by_name('train/adamOpt_')
        gradientQPartialActorParameter_, trainOpt = actorModel.run([gradientQPartialActorParameter_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                                                                              gradientQPartialAction_ : gradientQPartialAction[0]  
                                                                                                                              })
        numParams_ = actorGraph.get_tensor_by_name('outputs/numParams_:0')
        numParams = actorModel.run(numParams_)
        updateTargetParameter_ = [actorGraph.get_tensor_by_name('outputs/assign'+str(paramIndex_)+':0') for paramIndex_ in range(numParams)]
        actorModel.run(updateTargetParameter_)
        self.actorWriter.flush()
        return gradientQPartialActorParameter_, actorModel

class OnlineDeepDeterministicPolicyGradient():
    def __init__(self, maxEpisode, maxTimeStep, numMiniBatch, transitionFunction, isTerminal, reset, addActionNoise, savePathActor, savePathCritic, saveRate):
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.numMiniBatch = numMiniBatch
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset
        self.addActionNoise = addActionNoise
        self.savePathActor = savePathActor
        self.savePathCritic = savePathCritic
        self.saveRate = saveRate

    def __call__(self, actorModel, criticModel, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget, gradientPartialActionFromQEvaluation,
            memory, trainCritic, trainActor):
        catches = 0
        replayBuffer = []
        for episodeIndex in range(self.maxEpisode):
            print("Starting episode", episodeIndex)
            oldState = self.reset(numAgent=2)
            for timeStepIndex in range(self.maxTimeStep):
                evaActor = lambda state: approximatePolicyEvaluation(state, actorModel)
                actionBatch = evaActor(oldState.reshape(1, -1))
                actionPerfect = actionBatch[0]
                action = self.addActionNoise(actionPerfect, episodeIndex)
                newState = self.transitionFunction(oldState, action)
                timeStep = [oldState, action, newState] 
                replayBuffer = memory(replayBuffer, timeStep)
                if len(replayBuffer) >= self.numMiniBatch:
                    miniBatch = random.sample(replayBuffer, self.numMiniBatch)
                    tarActor = lambda state: approximatePolicyEvaluation(state, actorModel)
                    tarCritic = lambda state, action: approximateQTarget(state, action, criticModel)
                    QLoss, criticModel = trainCritic(miniBatch, tarActor, tarCritic, criticModel)
                    gradientEvaCritic = lambda state, action: gradientPartialActionFromQEvaluation(state, action, criticModel)
                    gradientQPartialActorParameter, actorModel = trainActor(miniBatch, evaActor, gradientEvaCritic, actorModel)
                if self.isTerminal(oldState):
                    if episodeIndex != self.maxEpisode - 1:
                        print("Target caught at episode", episodeIndex)
                        catches += 1
                    else:
                        print("Episode terminated. Not caught.")
                    break

                # time.sleep(.002)
                oldState = newState

            # Save checkpoints
            if episodeIndex != 0 and episodeIndex % self.saveRate == 0:
                with actorModel.as_default():
                    with actorModel.graph.as_default():
                        actorSaver = tf.train.Saver(tf.global_variables())
                    actorSaver.save(actorModel, self.savePathActor)
                
                with criticModel.as_default():
                    with criticModel.graph.as_default():
                        criticSaver = tf.train.Saver(tf.global_variables())
                    criticSaver.save(criticModel, self.savePathCritic)
                
                print("Saved models in episode", episodeIndex, "average catch rate", catches * 1.0/episodeIndex)
                

        return actorModel, criticModel

def main():
    numAgent = 2
    numActionSpace = 2
    numStateSpace = 12
    actionLow = np.array([-10, -10])
    actionHigh = np.array([10, 10])
    actionRatio = (actionHigh - actionLow) / 2.
    actionNoise = np.array([0.1, 0.1])
    noiseDecay = 0.999

    # numAgents = 1
    # numOtherActionSpace = (numAgents - 1) * numActionSpace
    
    envModelName = 'twoAgentsChasing'
    renderOn = False
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
    numMiniBatch = 250

    maxEpisode = 100000
    saveRate = 100

    numActorFC1Unit = 64
    numActorFC2Unit = 64
    numCriticFC1Unit = 128
    numCriticFC2Unit = 128
    learningRateActor = 0.001
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
    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.variable_scope("inputs"):
            state_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="state_"))
            action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, numActionSpace]), name='action_')
            QTarget_ = tf.placeholder(tf.float32, [None, 1], name="QTarget_")

        with tf.variable_scope("evaluationHidden"):
            evaFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = numCriticFC1Unit, activation = tf.nn.relu))
            numFullyConnected2Units = numCriticFC2Unit
            evaStateFC1ToFullyConnected2Weights_ = tf.get_variable(name='evaStateFC1ToFullyConnected2Weights', shape = [numCriticFC1Unit, numFullyConnected2Units])
            evaActionToFullyConnected2Weights_ = tf.get_variable(name='evaActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units])
            evaFullyConnected2Bias_ = tf.get_variable(name = 'evaFullyConnected2Bias', shape = [numFullyConnected2Units])
            evaFullyConnected2_ = tf.nn.relu(tf.matmul(evaFullyConnected1_, evaStateFC1ToFullyConnected2Weights_) + tf.matmul(action_, evaActionToFullyConnected2Weights_) + evaFullyConnected2Bias_ )
            evaQActivation_ = tf.layers.dense(inputs = evaFullyConnected2_, units = 1, activation = None, )

        with tf.variable_scope("targetHidden"):
            tarFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = numCriticFC1Unit, activation = tf.nn.relu))
            numFullyConnected2Units = numCriticFC2Unit
            tarStateFC1ToFullyConnected2Weights_ = tf.get_variable(name='tarStateFC1ToFullyConnected2Weights', shape = [numCriticFC1Unit, numFullyConnected2Units])
            tarActionToFullyConnected2Weights_ = tf.get_variable(name='tarActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units])
            tarFullyConnected2Bias_ = tf.get_variable(name = 'tarFullyConnected2Bias', shape = [numFullyConnected2Units])
            tarFullyConnected2_ = tf.nn.relu(tf.matmul(tarFullyConnected1_, tarStateFC1ToFullyConnected2Weights_) + tf.matmul(action_, tarActionToFullyConnected2Weights_) + tarFullyConnected2Bias_ )
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
            gradientQPartialAction_ = tf.gradients(evaQ_, action_, name = 'gradientQPartialAction_')
            criticLossSummary = tf.summary.scalar("CriticLoss", loss_)
        with tf.variable_scope("train"):
            trainOpt_ = tf.contrib.opt.AdamWOptimizer(weight_decay = l2DecayCritic, learning_rate = learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticWriter = tf.summary.FileWriter('tensorBoard/criticOnlineDDPG', graph = criticGraph)
    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(criticInit)   

    if restore:
        with actorModel.as_default():
            with actorModel.graph.as_default():
                actorSaver = tf.train.Saver(tf.global_variables())
            actorSaver.restore(actorModel, savePathActor)
        
        with criticModel.as_default():
            with criticModel.graph.as_default():
                criticSaver = tf.train.Saver(tf.global_variables())
            criticSaver.restore(criticModel, savePathCritic)
            
        print("Restored models from previous training")
     
    transitionFunction = env.TransitionFunctionNaivePredator(envModelName, renderOn)
    isTerminal = env.IsTerminal(minXDis)
    reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    addActionNoise = AddActionNoise(actionNoise, noiseDecay, actionLow, actionHigh)
     
    rewardFunction = reward.RewardFunctionCompete(aliveBouns, catchReward, disRewardDiscount, minXDis)
    
    memory = Memory(memoryCapacity)
 
    trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay, rewardFunction)
    
    trainActor = TrainActorTensorflow(actorWriter) 

    deepDeterministicPolicyGradient = OnlineDeepDeterministicPolicyGradient(maxEpisode, maxTimeStep, numMiniBatch, transitionFunction, isTerminal, reset, addActionNoise, savePathActor, savePathCritic, saveRate)

    
    trainedActorModel, trainedCriticModel = deepDeterministicPolicyGradient(actorModel, criticModel, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget,
            gradientPartialActionFromQEvaluation, memory, trainCritic, trainActor)
    


if __name__ == "__main__":
    main()



