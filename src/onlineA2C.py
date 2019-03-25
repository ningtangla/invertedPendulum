import tensorflow as tf
import numpy as np
import functools as ft
#import env
import cartpole_env
import reward
import dataSave 
import tensorflow_probability as tfp

def approximatePolicy(stateBatch, actorModel):
    graph = actorModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    actionSample_ = graph.get_tensor_by_name('outputs/actionSample_:0')
    actionBatch = actorModel.run(actionSample_, feed_dict = {state_ : stateBatch})
    return actionBatch

class SampleOneStep():
    def __init__(self, transitionFunction, rewardFunction):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction

    def __call__(self, oldstate, actor): 
        sample = [] 
        oldStateBatch = oldState.reshape(1, -1)
        actionBatch = actor(oldStateBatch) 
        action = actionBatch[0]
        # actionBatch shape: batch * action Dimension; only keep action Dimention in shape
        newState = self.transitionFunction(oldState, action)
        reward = self.rewardFunction(oldState, action)
        sample.append((oldState, action, newState, reward))
            
        return sample

def approximateValue(stateBatch, criticModel):
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
    valueBatch = criticModel.run(value_, feed_dict = {state_ : stateBatch})
    return valueBatch

class TrainCriticBootstrapTensorflow():
    def __init__(self, criticWriter, decay, rewardFunction):
        self.criticWriter = criticWriter
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, samples, criticModel):
        
        states, actions, nextStates, rewards = list(zip(*samples))
        stateBatch, nextStateBatch, rewardBatch = np.vstack(states), np.vstack(nextStates), np.vstack(rewards)

        graph = criticModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
        nextStateValueBatch = criticModel.run(value_, feed_dict = {state_ : nextStateBatch})
        
        valueTargetBatch = rewardBatch + self.decay * nextStateValueBatch

        state_ = graph.get_tensor_by_name('inputs/state_:0')
        valueTarget_ = graph.get_tensor_by_name('inputs/valueTarget_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                          valueTarget_ : valueTargetBatch
                                                                          })
        self.criticWriter.flush()
        return loss, criticModel

class EstimateAdvantageBootstrap():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, samples, critic):
        
        states, actions, nextStates, rewards = list(zip(*samples))
        stateBatch, actionBatch, nextStateBatch, rewardBatch = np.vstack(states), np.vstack(actions), np.vstack(nextStates), np.vstack(rewards)
         
        advantageBatch = rewardBatch + self.decay * critic(nextStateBatch) - critic(stateBatch)
        advantages = np.concatenate(advantageBatch)
        return advantages

class EstimateAdvantageMonteCarlo():
    def __init__(self, decay, accumulateRewards):
        self.decay = decay
        self.accumulateRewards = accumulateRewards
    def __call__(self, samples, critic):
        
        states, actions, nextStates, rewards = list(zip(*samples))
        stateBatch, actionBatch, nextStateBatch, rewardBatch = np.vstack(states), np.vstack(actions), np.vstack(nextStates), np.vstack(rewards)
         
        advantageBatch = rewardBatch + self.decay * critic(nextStateBatch) - critic(stateBatch)
        advantages = np.concatenate(advantageBatch)
        return advantages

class TrainActorTensorflow():
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, samples, advantages, critic):
        
        states, actions, nextStates, rewards = list(zip(*samples))
        stateBatch, actionBatch, nextStateBatch, rewardBatch = np.vstack(states), np.vstack(actions), np.vstack(nextStates), np.vstack(rewards)
        
        graph = actorModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0')
        advantages_ = graph.get_tensor_by_name('inputs/advantages_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = actorModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                         action_ : actionBatch,
                                                                         advantages_ : advantages       
                                                                         })
        self.actorWriter.flush()
        return loss, actorModel

class offlineAdvantageActorCritic():
    def __init__(self, numTrajectory, maxEpisode):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
    def __call__(self, actorModel, criticModel, approximatePolicy, sampleTrajectory, trainCritic, approximateValue, estimateAdvantage, trainActor):
        for episodeIndex in range(self.maxEpisode):
            actor = lambda state: approximatePolicy(state, actorModel)
            episode = [sampleTrajectory(actor) for index in range(self.numTrajectory)]
            valueLoss, criticModel = trainCritic(episode, criticModel)
            critic = lambda state: approximateValue(state, criticModel)
            advantages = estimateAdvantage(episode, critic)
            policyLoss, actorModel = trainActor(episode, advantages, actorModel)
            print(np.mean([len(episode[index]) for index in range(self.numTrajectory)]))
        return actorModel, criticModel

def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)

    numActionSpace = 1
    numStateSpace = 4
    actionLow = -2
    actionHigh = 2
    actionRatio = (actionHigh - actionLow) / 2.

    envModelName = 'inverted_pendulum'
    renderOn = True
    maxTimeStep = 200
    maxQPos = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    rewardDecay = 0.99

    numTrajectory = 200 
    maxEpisode = 1000

    learningRateActor = 0.001
    learningRateCritic = 0.01
 
    savePathActor = 'data/tmpModelActor.ckpt'
    savePathCritic = 'data/tmpModelCritic.ckpt'
    
    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            action_ = tf.placeholder(tf.float32, [None, numActionSpace], name="action_")
            advantages_ = tf.placeholder(tf.float32, [None, ], name="advantages_")

        with tf.name_scope("hidden"):
            fullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu)
            fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = 20, activation = tf.nn.relu)
            actionMean_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
            actionVariance_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.softplus)

        with tf.name_scope("outputs"):        
            actionDistribution_ = tfp.distributions.MultivariateNormalDiag(actionMean_ * actionRatio, actionVariance_ + 1e-8, name = 'actionDistribution_')
            actionSample_ = tf.clip_by_value(actionDistribution_.sample(), actionLow, actionHigh, name = 'actionSample_')
            negLogProb_ = - actionDistribution_.log_prob(action_, name = 'negLogProb_')
            loss_ = tf.reduce_sum(tf.multiply(negLogProb_, advantages_), name = 'loss_')
        actorLossSummary = tf.summary.scalar("ActorLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateActor, name = 'adamOpt_').minimize(loss_)

        actorInit = tf.global_variables_initializer()
        
        actorSummary = tf.summary.merge_all()
        actorSaver = tf.train.Saver(tf.global_variables())

    actorWriter = tf.summary.FileWriter('tensorBoard/actor', graph = actorGraph)
    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")

        with tf.name_scope("hidden1"):
            fullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu)

        with tf.name_scope("outputs"):        
            value_ = tf.layers.dense(inputs = fullyConnected1_, units = 1, activation = None, name = 'value_')
            diff_ = tf.subtract(valueTarget_, value_, name = 'diff_')
            loss_ = tf.reduce_mean(tf.square(diff_), name = 'loss_')
        criticLossSummary = tf.summary.scalar("CriticLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticWriter = tf.summary.FileWriter('tensorBorad/critic', graph = criticGraph)
    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(criticInit)    
     
    #transitionFunction = env.TransitionFunction(envModelName, renderOn)
    #isTerminal = env.IsTerminal(maxQPos)
    #reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    transitionFunction = cartpole_env.Cartpole_continuous_action_transition_function(renderOn = False)
    isTerminal = cartpole_env.cartpole_done_function
    reset = cartpole_env.cartpole_get_initial_state
    
    sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)
    
    rewardFunction = reward.RewardFunction(aliveBouns)
    #rewardFunction = reward.CartpoleRewardFunction(aliveBouns) 
    accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)

    trainCritic = TrainCriticMonteCarloTensorflow(criticWriter, accumulateRewards)
    estimateAdvantage = EstimateAdvantageMonteCarlo(accumulateRewards)
    trainActor = TrainActorMonteCarloTensorflow(actorWriter) 
    
    #trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay, rewardFunction)
    #estimateAdvantage = EstimateAdvantageBootstrap(rewardDecay, rewardFunction)
    #trainActor = TrainActorBootstrapTensorflow(actorWriter) 

    actorCritic = offlineAdvantageActorCritic(numTrajectory, maxEpisode)

    trainedActorModel, trainedCriticModel = actorCritic(actorModel, criticModel, approximatePolicy, sampleTrajectory, trainCritic,
            approximateValue, estimateAdvantage, trainActor)

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

