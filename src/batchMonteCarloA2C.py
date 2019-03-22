import tensorflow as tf
import numpy as np
import gym
import functools as ft
#import env
import cartpole_env
import reward
import dataSave 

def approximatePolicy(state, actorModel):
    graph = model.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    actionSample_ = graph.get_tensor_by_name('outputs/actionSample_:0')
    action = actorModel.run(actionSample_, feed_dict = {state_ : state.reshape(1, -1)})
    return action

class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, actor): 
        oldState = self.reset()
        trajectory = []
        
        for time in range(self.maxTimeStep): 
            action = actor(oldState) 
            newState = self.transitionFunction(oldState, action)
            trajectory.append((oldState, action))
            
            terminal = self.isTerminal(newState)
            if terminal:
                break
            oldState = newState
            
        return trajectory

class AccumulateRewards():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = [ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))]
        return accumulatedRewards

class TrainCriticTensorflow():
    def __init__(self, criticWriter):
        self.criticWriter = criticWriter
    def __call__(self, episode, accumulatedRewardsEpisode, criticModel):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateBatch, actionBatch = list(zip(*mergedEpisode))
        accumulatedRewardsBatch = np.concatenate(accumulatedRewardsEpisode)

        graph = criticModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        valueTarget_ = graph.get_tensor_by_name('inputs/valueTarget_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        __import__('ipdb').set_trace()
        loss, trainOpt = model.run([loss_, trainOpt_], feed_dict = {state_ : np.vstack(stateBatch),
                                                                    valueTarget_ : np.vstack(accumulatedRewardsBatch)
                                                                    })
        self.summaryWriter.flush()
        return loss, criticModel

def approximateValue(state, criticModel):
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    value_ = graph.get_tensor_by_name('outputs/value_:0')
    value = actorModel.run(value_, feed_dict = {state_ : state.reshape(1, -1)})
    return value

class EstimateAdvantage():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = repwardFunction
    def __call__(trajectory, critic):
        statesOrigin, actionsOrigin = list(zip(*trajectory))
        nextStates = statesOrigin[1 : ]
        states, actions = statesOrigin[ : -1], actionsOrigin[ : -1]
        rewards = np.array([self.rewardFunction(state, action) for state, action in trajectory[ : -1]])
        valueDifferences = np.array([self.decay * critic(state) - critic(nextState) for state, nextState in zip(states, nextStates)])
        advantages = rewards + valueDifferences
        return advantages

class TrainActorTensorflow():
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, episode, advantagesEpisode, actorModel):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateBatch, actionBatch = list(zip(*mergedEpisode))
        advantagesBatch = np.concatenate(advantagesEpisode)

        graph = actorModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0')
        advantages_ = graph.get_tensor_by_name('inputs/advantages_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = actorModel.run([loss_, trainOpt_], feed_dict = {state_ : np.vstack(stateBatch),
                                                                    action_ : np.vstack(actionBatch),
                                                                    advantages_ : advantagesBatch
                                                                    })
        self.summaryWriter.flush()
        return loss, actorModel

class BatchMontoCarloAdvantageActorCritic():
    def __init__(self, numTrajectory, maxEpisode):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
    def __call__(self, actorModel, criticModel, approximatePolicy, sampleTrajectory, accumulateRewards, trainCritic, approximateValue, estimateAdvantage, trainActor):
        for episodeIndex in range(self.maxEpisode):
            actor = lambda state: approximatePolicy(state, actorModel)
            episode = [sampleTrajectory(actor) for index in range(self.numTrajectory)]
            accumulatedRewardsEpisode = [accumulateRewards(trajectory) for trajectory in episode]
            valueLoss, criticModel = trainCritic(episode, accumulatedRewardsEpisode, criticModel)
            critic = lambda state: approximateValue(state, criticModel)
            advantagesEpisode = [estimateAdvantage(trajectory) for trajectory in episode]
            policyLoss, actorModel = train(episode, advantagesEpisode, actorModel)
            print(np.mean([len(episode[index]) for index in range(self.numTrajectory)]))
        return actorModel, critiModel

def main():
    numActionSpace = 1
    numStateSpace = 4
    actionLow = -1
    actionHigh = 1
    actionRatio = (actionHigh - actionLow) / 2.

    envModelName = 'inverted_pendulum'
    renderOpen = False
    maxTimeStep = 200
    maxQPos = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    rewardDecay = 1

    numTrajectory = 100 
    maxEpisode = 3000

    learningRateActor = 0.001
    learningRateCritic = 0.01
 
    savePathActor = 'data/tmpModelActor.ckpt'
    savePathCritic = 'data/tmpModelCritic.ckpt'
    
    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            action_ = tf.placeholder(tf.float32, [None, numActionSpace], name="action_")
            accumulatedRewards_ = tf.placeholder(tf.float32, [None, ], name="accumulatedRewards_")

        with tf.name_scope("hidden"):
            fullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu)
            fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = 20, activation = tf.nn.relu)
            actionMean_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
            actionVariance_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.softplus)

        with tf.name_scope("outputs"):        
            actionDistribution_ = tf.distributions.Normal(tf.squeeze(actionMean_) * actionRatio, tf.squeeze(actionVariance_) + 1e-8, name = 'actionDistribution_')
            actionSample_ = tf.clip_by_value((actionDistribution_.sample(1)), actionLow, actionHigh, name = 'actionSample_')
            neg_log_prob_ = - actionDistribution_.log_prob(action_)
            loss_ = tf.reduce_sum(tf.multiply(neg_log_prob_, accumulatedRewards_), name = 'loss_')
        actorLossSummary = tf.summary.scalar("ActorLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateActor, name = 'adamOpt_').minimize(loss_)

        actorSummary = tf.summary.merge_all()
        actorSaver = tf.train.Saver()

    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")

        with tf.name_scope("hidden1"):
            fullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu)
            fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = 20, activation = tf.nn.relu)

        with tf.name_scope("outputs"):        
            value_ = tf.layers.dense(inputs = fullyConnected2_, units = 1, activation = None, name = 'value_')
            loss_ = tf.square((valueTarget_ - value_), name = 'loss_')
        criticLossSummary = tf.summary.scalar("CriticLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver()

    actorWriter = tf.Summary.FileWriter('tensorBoard/actor', graph = actorGraph)
    criticWriter = tf.Summary.FileWriter('tensorBorad/critic', graph = criticGraph)

    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(tf.global_variables_initializer())    

    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(tf.global_variables_initializer())    
    
    """
    transitionFunction = env.TransitionFunction(envModelName, renderOpen)
    isTerminal = env.IsTerminal(maxQPos)
    reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    """
    transitionFunction = cartpole_env.cartpole_continuous_action_transition_function
    isTerminal = cartpole_env.cartpole_done_function
    reset = cartpole_env.cartpole_get_initial_state
    sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)

    rewardFunction = reward.RewardFunction(aliveBouns) 
    accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)

    trainCritic = TrainCriticTensorflow(criticWriter)
    
    estimateAdvantage = EstimateAdvantage(rewardDecay, rewardFunction)
    trainActor = TrainActorTensorflow(actorWriter) 

    batchMontoCarloAdvantageActorCritic = BatchMontoCarloAdvantageActorCritic(numTrajectory, maxEpisode)

    trainedActorModel, trainedCriticModel = batchMontoCarloAdvantageActorCritic(actorModel, criticModel, approximatePolicy, sampleTrajectory, accumulateRewards, trainCritic,
            approximateValue, trainActor)

    actorSaver(trainedActorModel, savePathActor)
    criticSaver(trainedCriticModel, savePathCritic)

    transitionPlay = env.TransitionFunction(envModelName, renderOpen = True)
    samplePlay = SampleTrajectory(maxTimeStep, transitionPlay, isTerminal, reset)
    actor = lambda state: approximatePolicy(state, model)
    playEpisode = [samplePlay(actor) for index in range(5)]
    print(np.mean([len(playEpisode[index]) for index in range(5)]))

if __name__ == "__main__":
    main()
