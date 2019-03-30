import tensorflow as tf
import numpy as np
import functools as ft
import env
import cartpole_env
import reward
import dataSave 
import tensorflow_probability as tfp

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

class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset
    def __call__(self, evaActor, episodeIndex): 
        oldState = self.reset()
        trajectory = []
        
        for time in range(self.maxTimeStep): 
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = evaActor(oldStateBatch) 
            actionNoNoise = actionBatch[0]
            action = np.clip(np.random.normal(actionNoNoise, 0.01*(0.9995 ** episodeIndex)), -3, 3)
            #action = actionNoNoise
            # actionBatch shape: batch * action Dimension; only keep action Dimention in shape
            newState = self.transitionFunction(oldState, action)
            trajectory.append((oldState, action))
            
            terminal = self.isTerminal(newState)
            if terminal:
                break
            oldState = newState
            
        return trajectory

class TrainCriticBootstrapTensorflow():
    def __init__(self, criticWriter, decay, rewardFunction):
        self.criticWriter = criticWriter
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, episode, tarActor, tarCritic, criticModel):
        
        noLastStateEpisode = [trajectory[ : -1] for trajectory in episode]
        mergedNoLastStateEpisode = np.concatenate(noLastStateEpisode)
        states, actions = list(zip(*mergedNoLastStateEpisode)) 
        
        noFirstStateEpisode = [trajectory[1 : ] for trajectory in episode]
        mergedNoFirstStateEpisode = np.concatenate(noFirstStateEpisode)
        nextStates, nextActions = list(zip(*mergedNoFirstStateEpisode)) 
 
        stateBatch, actionBatch, nextStateBatch = np.vstack(states), np.vstack(actions), np.vstack(nextStates)
        
        nextTargetActionBatch = tarActor(nextStateBatch)

        nextTargetQBatch = tarCritic(nextStateBatch, nextTargetActionBatch)
        
        rewardsEpisode = np.array([self.rewardFunction(state, action) for state, action in mergedNoLastStateEpisode])
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
        return loss, criticModel

class TrainActorTensorflow():
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, episode, evaActor, gradientEvaCritic, actorModel):

        mergedEpisode = np.concatenate(episode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch, actionBatch = np.vstack(stateEpisode), np.vstack(actionEpisode)
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

class OfflineDeterministicPolicyGradient():
    def __init__(self, numTrajectory, maxEpisode):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
    def __call__(self, actorModel, criticModel, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget, gradientPartialActionFromQEvaluation, sampleTrajectory, trainCritic, trainActor):
        for episodeIndex in range(self.maxEpisode):
            evaActor = lambda state: approximatePolicyEvaluation(state, actorModel)
            episode = [sampleTrajectory(evaActor, episodeIndex) for index in range(self.numTrajectory)]
            tarActor = lambda state: approximatePolicyEvaluation(state, actorModel)
            tarCritic = lambda state, action: approximateQTarget(state, action, criticModel)
            QLoss, criticModel = trainCritic(episode, tarActor, tarCritic, criticModel)
            gradientEvaCritic = lambda state, action: gradientPartialActionFromQEvaluation(state, action, criticModel)
            gradientQPartialActorParameter, actorModel = trainActor(episode, evaActor, gradientEvaCritic, actorModel)
            print(np.mean([len(episode[index]) for index in range(self.numTrajectory)]))
        return actorModel, criticModel

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

    numTrajectory = 100 
    maxEpisode = 1000

    learningRateActor = 0.0001
    learningRateCritic = 0.001
 
    savePathActor = 'data/tmpModelActor.ckpt'
    savePathCritic = 'data/tmpModelCritic.ckpt'
    
    softReplaceRatio = 0.01

    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.variable_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            gradientQPartialAction_ = tf.placeholder(tf.float32, [None, numActionSpace], name="gradientQPartialAction_")

        with tf.variable_scope("evaluationHidden"):
            evaFullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu)
            evaFullyConnected2_ = tf.layers.dense(inputs = evaFullyConnected1_, units = 20, activation = tf.nn.relu)
            evaActionActivation_ = tf.layers.dense(inputs = evaFullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
            
        with tf.variable_scope("targetHidden"):
            tarFullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu)
            tarFullyConnected2_ = tf.layers.dense(inputs = tarFullyConnected1_, units = 20, activation = tf.nn.relu)
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

    actorWriter = tf.summary.FileWriter('tensorBoard/actorDDPG', graph = actorGraph)
    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.variable_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, numActionSpace]), name='action_')
            QTarget_ = tf.placeholder(tf.float32, [None, 1], name="QTarget_")

        with tf.variable_scope("evaluationHidden"):
            numFullyConnected1Units = 100
            evaStateToFullyConnected1Weights_ = tf.get_variable(name='evaStateToFullyConnected1Weights', shape = [numStateSpace, numFullyConnected1Units])
            evaActionToFullyConnected1Weights_ = tf.get_variable(name='evaActionToFullyConnected1Weights', shape = [numActionSpace, numFullyConnected1Units])
            evaFullyConnected1Bias_ = tf.get_variable(name = 'evaFullyConnected1Bias1Bias', shape = [numFullyConnected1Units])
            evaFullyConnected1_ = tf.nn.relu(tf.matmul(state_, evaStateToFullyConnected1Weights_) + tf.matmul(action_, evaActionToFullyConnected1Weights_) + evaFullyConnected1Bias_ )
            evaQActivation_ = tf.layers.dense(inputs = evaFullyConnected1_, units = 1, activation = None)

        with tf.variable_scope("targetHidden"):
            numFullyConnected1Units = 100
            tarStateToFullyConnected1Weights_ = tf.get_variable(name='tarStateToFullyConnected1Weights', shape = [numStateSpace, numFullyConnected1Units])
            tarActionToFullyConnected1Weights_ = tf.get_variable(name='tarActionToFullyConnected1Weights', shape = [numActionSpace, numFullyConnected1Units])
            tarFullyConnected1Bias_ = tf.get_variable(name = 'tarFullyConnected1Bias1Bias', shape = [numFullyConnected1Units])
            tarFullyConnected1_ = tf.nn.relu(tf.matmul(state_, tarStateToFullyConnected1Weights_) + tf.matmul(action_, tarActionToFullyConnected1Weights_) + tarFullyConnected1Bias_ )
            tarQActivation_ = tf.layers.dense(inputs = tarFullyConnected1_, units = 1, activation = None)
        
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
            trainOpt_ = tf.train.AdamOptimizer(learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticWriter = tf.summary.FileWriter('tensorBoard/criticDDPG', graph = criticGraph)
    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(criticInit)   
     
    transitionFunction = env.TransitionFunction(envModelName, renderOn)
    isTerminal = env.IsTerminal(maxQPos)
    reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    #transitionFunction = cartpole_env.Cartpole_continuous_action_transition_function(renderOn = False)
    #isTerminal = cartpole_env.cartpole_done_function
    #reset = cartpole_env.cartpole_get_initial_state
    
    sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)
    
    rewardFunction = reward.RewardFunction(aliveBouns)
    #accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)
 
    trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay, rewardFunction)
    
    trainActor = TrainActorTensorflow(actorWriter) 

    deterministicPolicyGradient = OfflineDeterministicPolicyGradient(numTrajectory, maxEpisode)

    trainedActorModel, trainedCriticModel = deterministicPolicyGradient(actorModel, criticModel, approximatePolicyEvaluation, approximatePolicyTarget, approximateQTarget,
            gradientPartialActionFromQEvaluation, sampleTrajectory, trainCritic, trainActor)

    with actorModel.as_default():
        actorSaver.save(trainedActorModel, savePathActor)
    with criticModel.as_default():
        criticSaver.save(trainedCriticModel, savePathCritic)

    transitionPlay = cartpole_env.Cartpole_continuous_action_transition_function(renderOn = True)
    samplePlay = SampleTrajectory(maxTimeStep, transitionPlay, isTerminal, reset)
    actor = lambda state: approximatePolicy(state, trainedActorModel)
    playEpisode = [samplePlay(actor) for index in range(5)]
    print(np.mean([len(playEpisode[index]) for index in range(5)]))

if __name__ == "__main__":
    main()

