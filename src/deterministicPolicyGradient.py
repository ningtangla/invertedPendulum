import tensorflow as tf
import numpy as np
import functools as ft
import env
import cartpole_env
import reward
import dataSave 
import tensorflow_probability as tfp

def approximatePolicy(stateBatch, actorModel):
    graph = actorModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    action_ = graph.get_tensor_by_name('outputs/action_:0')
    actionBatch = actorModel.run(action_, feed_dict = {state_ : stateBatch})
    return actionBatch

class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset
    def __call__(self, actor, episodeIndex): 
        oldState = self.reset()
        trajectory = []
        
        for time in range(self.maxTimeStep): 
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = actor(oldStateBatch) 
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
    def __call__(self, episode, criticModel, actor):
        
        noLastStateEpisode = [trajectory[ : -1] for trajectory in episode]
        mergedNoLastStateEpisode = np.concatenate(noLastStateEpisode)
        states, actions = list(zip(*mergedNoLastStateEpisode)) 
        
        noFirstStateEpisode = [trajectory[1 : ] for trajectory in episode]
        mergedNoFirstStateEpisode = np.concatenate(noFirstStateEpisode)
        nextStates, nextActions = list(zip(*mergedNoFirstStateEpisode)) 
 
        stateBatch, actionBatch, nextStateBatch, nextActionBatch = np.vstack(states), np.vstack(actions), np.vstack(nextStates), np.vstack(nextActions)
        nextActorActionBatch = actor(nextStateBatch)

        graph = criticModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0') 
        Q_ = graph.get_tensor_by_name('outputs/Q_/BiasAdd:0')
        nextQBatch = criticModel.run(Q_, feed_dict = {state_ : nextStateBatch, action_ : nextActorActionBatch})
        #nextQBatch = criticModel.run(Q_, feed_dict = {state_ : nextStateBatch, action_ : nextActionBatch})
        
        rewardsEpisode = np.array([self.rewardFunction(state, action) for state, action in mergedNoLastStateEpisode])
        trajectoryLengthes = [len(trajectory) for trajectory in noLastStateEpisode]
        lastStateIndex = np.cumsum(trajectoryLengthes) - 1
        #rewardsEpisode[lastStateIndex] = -20
        rewardBatch = np.vstack(rewardsEpisode)
        QTargetBatch = rewardBatch + self.decay * nextQBatch

        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0') 
        QTarget_ = graph.get_tensor_by_name('inputs/QTarget_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                          action_ : actionBatch,
                                                                          QTarget_ : QTargetBatch
                                                                          })
        self.criticWriter.flush()
        return loss, criticModel

class TrainActorTensorflow():
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, episode, actorModel, criticModel, actor):

        mergedEpisode = np.concatenate(episode)
        states, actions = list(zip(*mergedEpisode))
        stateBatch = np.vstack(stateEpisode)
        actorActionBatch = actor(stateBatch)
        
        criticGraph = criticModel.graph
        state_ = criticGraph.get_tensor_by_name('inputs/state_:0')
        action_ = criticGraph.get_tensor_by_name('inputs/action_:0')
        gradientQPartialAction_ = criticGraph.get_tensor_by_name('outputs/gradientQPartialAction_/hidden/MatMul_1_grad/MatMul:0')
        gradientQPartialAction = criticModel.run([gradientQPartialAction_], feed_dict = {state_ : stateBatch,
                                                                                         action_ : actorActionBatch,
                                                                                         })
        
        actorGraph = actorModel.graph
        state_ = actorGraph.get_tensor_by_name('inputs/state_:0')
        gradientQPartialAction_ = actorGraph.get_tensor_by_name('inputs/gradientQPartialAction_:0')
        gradientQPartialActorParameter_ = actorGraph.get_tensor_by_name('outputs/gradientQPartialActorParameter_/hidden/dense/MatMul_grad/MatMul:0')
        trainOpt_ = actorGraph.get_operation_by_name('train/adamOpt_')
        gradientQPartialActorParameter_, trainOpt = actorModel.run([gradientQPartialActorParameter_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                                                                              gradientQPartialAction_ : gradientQPartialAction[0]  
                                                                                                                              })
        self.actorWriter.flush()
        return gradientQPartialActorParameter_, actorModel

class OfflineDeterministicPolicyGradient():
    def __init__(self, numTrajectory, maxEpisode):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
    def __call__(self, actorModel, criticModel, approximatePolicy, sampleTrajectory, trainCritic, trainActor):
        for episodeIndex in range(self.maxEpisode):
            actor = lambda state: approximatePolicy(state, actorModel)
            episode = [sampleTrajectory(actor, episodeIndex) for index in range(self.numTrajectory)]
            valueLoss, criticModel = trainCritic(episode, criticModel, actor)
            gradientQPartialActorParameter, actorModel = trainActor(episode, actorModel, criticModel, actor)
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
    
    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="state_"))
            gradientQPartialAction_ = tf.placeholder(tf.float32, [None, numActionSpace], name="gradientQPartialAction_")

        with tf.name_scope("hidden"):
            fullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu))
            fullyConnected2_ = tf.layers.batch_normalization(tf.layers.dense(inputs = fullyConnected1_, units = 10, activation = tf.nn.relu))
            actionActivation_ = tf.layers.batch_normalization(tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.tanh))

        with tf.name_scope("outputs"):        
            action_ = tf.multiply(actionActivation_, actionRatio, name = 'action_')
            actorParameter_ = tf.trainable_variables()
            gradientQPartialActorParameter_ = tf.gradients(ys = action_, xs = actorParameter_, grad_ys = gradientQPartialAction_, name = 'gradientQPartialActorParameter_')

        with tf.name_scope("train"):
            #-learningRate for ascent
            trainOpt_ = tf.train.AdamOptimizer(-learningRateActor, name = 'adamOpt_').apply_gradients(zip(gradientQPartialActorParameter_, actorParameter_))
        actorInit = tf.global_variables_initializer()
        
        actorSummary = tf.summary.merge_all()
        actorSaver = tf.train.Saver(tf.global_variables())

    actorWriter = tf.summary.FileWriter('tensorBoard/actorDPG', graph = actorGraph)
    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="state_"))
            action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, numActionSpace]), name='action_')
            QTarget_ = tf.placeholder(tf.float32, [None, 1], name="QTarget_")

        with tf.name_scope("hidden"):
            fullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = 300, activation = tf.nn.relu))
            numFullyConnected2Units = 200
            stateFC1ToFullyConnected2Weights_ = tf.get_variable(name='stateFC1ToFullyConnected2Weights', shape = [300, numFullyConnected2Units])
            actionToFullyConnected2Weights_ = tf.get_variable(name='actionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units])
            fullyConnected2Bias_ = tf.get_variable(name = 'fullyConnected1Bias1Bias', shape = [numFullyConnected2Units])
            fullyConnected2_ = tf.nn.relu(tf.matmul(fullyConnected1_, stateFC1ToFullyConnected2Weights_) + tf.matmul(action_, actionToFullyConnected2Weights_) + fullyConnected2Bias_ )
        
        with tf.name_scope("outputs"):        
            Q_ = tf.layers.dense(inputs = fullyConnected2_, units = 1, activation = None, name = 'Q_')
            diff_ = tf.subtract(QTarget_, Q_, name = 'diff_')
            loss_ = tf.reduce_mean(tf.square(diff_), name = 'loss_')
            gradientQPartialAction_ = tf.gradients(Q_, action_, name = 'gradientQPartialAction_')
            criticLossSummary = tf.summary.scalar("CriticLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticWriter = tf.summary.FileWriter('tensorBoard/criticDPG', graph = criticGraph)
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

    #trainCritic = TrainCriticMonteCarloTensorflow(criticWriter, accumulateRewards)
    #trainActor = TrainActorMonteCarloTensorflow(actorWriter) 
    
    trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay, rewardFunction)
    
    trainActor = TrainActorTensorflow(actorWriter) 

    deterministicPolicyGradient = OfflineDeterministicPolicyGradient(numTrajectory, maxEpisode)

    trainedActorModel, trainedCriticModel = deterministicPolicyGradient(actorModel, criticModel, approximatePolicy, sampleTrajectory, trainCritic, trainActor)

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

