import tensorflow as tf
import numpy as np
import gym
import functools as ft
import env
import reward
import dataSave 

def fowardModel(inputValue, inputVar, outputVar, model):
    outputValue = model.run(outputVar, feed_dict={inputVar: inputValue.reshape(1, -1)})
    return outputValue 

class ApproximatePolicy():
    def __init__(self, actionSpace, inputVar, outputVar):
        self.actionSpace = actionSpace
        self.numActionSpace = len(self.actionSpace)
        self.inputVar = inputVar
        self.outputVar = outputVar
    def __call__(self, state, model):
        #__import__('ipdb').set_trace()
        actionDistribution = fowardModel(state, self.inputVar['state_'], self.outputVar['actionDistribution_'], model)
        actionLabel = np.random.choice(range(self.numActionSpace), p = actionDistribution.ravel())  # select action w.r.t the actions prob
        action = self.actionSpace[actionLabel]
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
    def __init__(self, actionSpace, inputVar, outputVar, summaryPath):
        self.actionSpace = actionSpace
        self.numActionSpace = len(actionSpace)
        self.summaryWriter = tf.summary.FileWriter(summaryPath)
        self.inputVar = inputVar
        self.outputVar = outputVar
    def __call__(self, episode, normalizedAccumulatedRewardsEpisode, model):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateBatch, actionBatch = list(zip(*mergedEpisode))
        actionIndexBatch = np.array([list(self.actionSpace).index(action) for action in actionBatch])
        actionLabelBatch = np.zeros([numBatch, self.numActionSpace])
        actionLabelBatch[np.arange(numBatch), actionIndexBatch] = 1 
        accumulatedRewardsBatch = np.concatenate(normalizedAccumulatedRewardsEpisode)
        loss, trainOpt = model.run([self.outputVar['loss_'], self.outputVar['trainOpt_']], feed_dict={self.inputVar['state_']: np.vstack(np.array(stateBatch)),
                                                                                                    self.inputVar['actionLabel_']: np.vstack(np.array(actionLabelBatch)),
                                                                                                    self.inputVar['accumulatedRewards_']: accumulatedRewardsBatch
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
    actionSpace = np.arange(-3, 3.01, 0.1)
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
        accumulatedRewards_ = tf.placeholder(tf.float32, [None,], name="accumulatedRewards_")

        # Add this placeholder for having this variable in tensorboard
        #mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

        with tf.name_scope("fc1"):
            fc1 = tf.contrib.layers.fully_connected(inputs = state_,
                                                    num_outputs = 100,
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("fc2"):
            fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
                                                    num_outputs = numActionSpace,
                                                    activation_fn= tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("fc3"):
            fc3 = tf.contrib.layers.fully_connected(inputs = fc2,
                                                    num_outputs = numActionSpace,
                                                    activation_fn= None,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("softmax"):
            actionDistribution_ = tf.nn.softmax(fc3)

        with tf.name_scope("loss"):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actionLabel_)
            loss_ = tf.reduce_sum(neg_log_prob * accumulatedRewards_)
        tf.summary.scalar("Loss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRate).minimize(loss_)
    
    mergedSummary = tf.summary.merge_all()
    
    inputApproximatePolicy = {'state_' : state_}
    outputApproximatePolicy = {'actionDistribution_' : actionDistribution_}
    
    inputTrain = {'state_' : state_, 'actionLabel_' : actionLabel_, 'accumulatedRewards_' : accumulatedRewards_}
    outputTrain = {'loss_' : loss_, 'trainOpt_' : trainOpt_}

    with tf.Session() as model:
        model.run(tf.global_variables_initializer())    

        approximatePolicy = ApproximatePolicy(actionSpace, inputApproximatePolicy, outputApproximatePolicy)

        transitionFunction = env.TransitionFunction(envModelName, renderOpen)
        isTerminal = env.IsTerminal(maxQPos)
        reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
        sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)

        rewardFunction = reward.RewardFunction(aliveBouns) 
        accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)

        train = TrainTensorflow(actionSpace, inputTrain, outputTrain,summaryPath) 

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
