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

def approximateValue(state, criticModel):
    graph = model.graph
    state_ = graph.get_tensor_by_name('inputs/oldState_:0')
    value_ = graph.get_tensor_by_name('outputs/value_:0')
    value = actorModel.run(value_, feed_dict = {state_ : state.reshape(1, -1)})
    return value

class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, policyFunction): 
        oldState = self.reset()
        trajectory = []
        
        for time in range(self.maxTimeStep): 
            action = policyFunction(oldState) 
            newState = self.transitionFunction(oldState, action)
            trajectory.append((oldState, action))
            
            terminal = self.isTerminal(newState)
            if terminal:
                break
            oldState = newState
            
        return trajectory

class ():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = [ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))]
        return accumulatedRewards

class TrainTensorflow():
    def __init__(self, summaryPath):
        self.summaryWriter = tf.summary.FileWriter(summaryPath)
    def __call__(self, episode, normalizedAccumulatedRewardsEpisode, model):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateBatch, actionBatch = list(zip(*mergedEpisode))
        accumulatedRewardsBatch = np.concatenate(normalizedAccumulatedRewardsEpisode)
        
        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0')
        accumulatedRewards_ = graph.get_tensor_by_name('inputs/accumulatedRewards_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = model.run([loss_, trainOpt_], feed_dict = {state_ : np.vstack(stateBatch),
                                                                    action_ : np.vstack(actionBatch),
                                                                    accumulatedRewards_ : accumulatedRewardsBatch
                                                                    })
        self.summaryWriter.flush()
        return loss, model

class PolicyGradient():
    def __init__(self, numTrajectory, maxEpisode):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
    def __call__(self, model, approximatePolicy, sampleTrajectory, accumulateRewards, train):
        for episodeIndex in range(self.maxEpisode):
            policy = lambda state: approximatePolicy(state, model)
            episode = [sampleTrajectory(policy) for index in range(self.numTrajectory)]
            normalizedAccumulatedRewardsEpisode = [normalize(accumulateRewards(trajectory)) for trajectory in episode]
            loss, model = train(episode, normalizedAccumulatedRewardsEpisode, model)
            print(np.mean([len(episode[index]) for index in range(self.numTrajectory)]))
        return model

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

    learningRate = 0.01
    summaryPath = 'tensorBoard/1'

    savePath = 'data/tmpModelGaussian.ckpt'
    
    with tf.name_scope("inputs"):
        state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
        action_ = tf.placeholder(tf.float32, [None, numActionSpace], name="action_")
        accumulatedRewards_ = tf.placeholder(tf.float32, [None, ], name="accumulatedRewards_")

    with tf.name_scope("hidden1"):
        fullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu)
        fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = 20, activation = tf.nn.relu)
        actionMean_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
        actionVariance_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.softplus)

    with tf.name_scope("outputs"):        
        actionDistribution_ = tf.distributions.Normal(tf.squeeze(actionMean_) * actionRatio, tf.squeeze(actionVariance_) + 1e-8, name = 'actionDistribution_')
        actionSample_ = tf.clip_by_value((actionDistribution_.sample(1)), actionLow, actionHigh, name = 'actionSample_')
        neg_log_prob_ = - actionDistribution_.log_prob(action_)
        loss_ = tf.reduce_sum(tf.multiply(neg_log_prob_, accumulatedRewards_), name = 'loss_')
    tf.summary.scalar("Loss", loss_)

    with tf.name_scope("train"):
        trainOpt_ = tf.train.AdamOptimizer(learningRate, name = 'adamOpt_').minimize(loss_)

    mergedSummary = tf.summary.merge_all()
    
    model = tf.Session()
    model.run(tf.global_variables_initializer())    

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

    train = TrainTensorflow(summaryPath) 

    policyGradient = PolicyGradient(numTrajectory, maxEpisode)

    trainedModel = policyGradient(model, approximatePolicy, sampleTrajectory, accumulateRewards, train)

    saveModel = dataSave.SaveModel(savePath)
    modelSave = saveModel(model)

    transitionPlay = env.TransitionFunction(envModelName, renderOpen = True)
    samplePlay = SampleTrajectory(maxTimeStep, transitionPlay, isTerminal, reset)
    policy = lambda state: approximatePolicy(state, model)
    playEpisode = [samplePlay(policy) for index in range(5)]
    print(np.mean([len(playEpisode[index]) for index in range(5)]))
if __name__ == "__main__":
    main()
