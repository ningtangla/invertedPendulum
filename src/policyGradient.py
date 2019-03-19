import tensorflow as tf
import numpy as np
import gym
import functools as ft
import env
import reward
import dataSave 

class ApproximatePolicy():
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
        self.numActionSpace = len(self.actionSpace)
    def __call__(self, state, model):
        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        actionDistribution_ = graph.get_tensor_by_name('outputs/actionDistribution_:0')
        actionDistribution = model.run(actionDistribution_, feed_dict = {state_ : state.reshape(1, -1)})
        actionIndex = np.random.choice(range(self.numActionSpace), p = actionDistribution.ravel())  # select action w.r.t the actions prob
        action = np.array(self.actionSpace[actionIndex])
        return action

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

class AccumulateRewards():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = [ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))]
        return accumulatedRewards

def normalize(accumulatedRewards):
    normalizedAccumulatedRewards = (accumulatedRewards - np.mean(accumulatedRewards)) / np.std(accumulatedRewards)
    return normalizedAccumulatedRewards

class TrainTensorflow():
    def __init__(self, actionSpace, summaryPath):
        self.actionSpace = actionSpace
        self.numActionSpace = len(actionSpace)
        self.summaryWriter = tf.summary.FileWriter(summaryPath)
    def __call__(self, episode, normalizedAccumulatedRewardsEpisode, model):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateBatch, actionBatch = list(zip(*mergedEpisode))
        actionIndexBatch = np.array([list(self.actionSpace).index(action) for action in actionBatch])
        actionLabelBatch = np.zeros([numBatch, self.numActionSpace])
        actionLabelBatch[np.arange(numBatch), actionIndexBatch] = 1
        accumulatedRewardsBatch = np.concatenate(normalizedAccumulatedRewardsEpisode)
        
        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        actionLabel_ = graph.get_tensor_by_name('inputs/actionLabel_:0')
        accumulatedRewards_ = graph.get_tensor_by_name('inputs/accumulatedRewards_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = model.run([loss_, trainOpt_], feed_dict = {state_ : np.vstack(stateBatch),
                                                                    actionLabel_ : np.vstack(actionLabelBatch),
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
        return model

def main():
    actionSpace = [np.array(action) for action in np.arange(-3, 3.01, 0.1)]
    numActionSpace = len(actionSpace)
    numStateSpace = 4
    
    envModelName = 'inverted_pendulum'
    renderOpen = False
    maxTimeStep = 300
    maxQPos = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    rewardDecay = 1

    numTrajectory = 5
    maxEpisode = 200

    learningRate = 0.01
    summaryPath = 'tensorBoard/1'

    savePath = 'data/tmpModel.ckpt'
    
    with tf.name_scope("inputs"):
        state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
        actionLabel_ = tf.placeholder(tf.int32, [None, numActionSpace], name="actionLabel_")
        accumulatedRewards_ = tf.placeholder(tf.float32, [None, ], name="accumulatedRewards_")

    with tf.name_scope("hidden"):
        fullyConnected1_ = tf.layers.dense(inputs = state_, units = 100, activation = tf.nn.relu)
        fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = numActionSpace, activation = tf.nn.relu)
        fullyConnected3_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = None)

    with tf.name_scope("outputs"):
        actionDistribution_ = tf.nn.softmax(fullyConnected3_, name = 'actionDistribution_')
        neg_log_prob_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fullyConnected3_, labels = actionLabel_)
        loss_ = tf.reduce_sum(tf.multiply(neg_log_prob_, accumulatedRewards_), name = 'loss_')
    tf.summary.scalar("Loss", loss_)

    with tf.name_scope("train"):
        trainOpt_ = tf.train.AdamOptimizer(learningRate, name = 'adamOpt_').minimize(loss_)

    mergedSummary = tf.summary.merge_all()
    
    model = tf.Session()
    model.run(tf.global_variables_initializer())    

    approximatePolicy = ApproximatePolicy(actionSpace)

    transitionFunction = env.TransitionFunction(envModelName, renderOpen)
    isTerminal = env.IsTerminal(maxQPos)
    reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)

    rewardFunction = reward.RewardFunction(aliveBouns) 
    accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)

    train = TrainTensorflow(actionSpace, summaryPath) 

    policyGradient = PolicyGradient(numTrajectory, maxEpisode)

    trainedModel = policyGradient(model, approximatePolicy, sampleTrajectory, accumulateRewards, train)

    saveModel = dataSave.SaveModel(savePath)
    modelSave = saveModel(model)

    transitionPlay = env.TransitionFunction(envModelName, renderOpen = True)
    samplePlay = SampleTrajectory(maxTimeStep, transitionPlay, isTerminal, reset)
    policy = lambda state: approximatePolicy(state, model)
    playEpisode = [samplePlay(policy) for index in range(5)]
    accumulatedRewardsEpisode = [accumulateRewards(trajectory) for trajectory in playEpisode]

if __name__ == "__main__":
    main()
