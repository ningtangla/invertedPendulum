import tensorflow as tf
import numpy as np
import functools as ft
#import env
import cartpole_env
import reward
import dataSave 


def fowardModel(inputValue, inputVariable, outputVariable, model):
    outVar = [outputVar[varName] for varName in outputVariable.keys()]
    inputVariableValuePair = dict((inputVariable[varName], np.vstack(inputValue[varName])) for varName in inputVariable.keys())
    outValue = model.run(outVar, feed_dict = inputVariableValuePair)
    outputVarValuePair = dict((outVaiable[varIndex], outValue[varIndex]) for varIndex in range(len(outVar))) 
    return outputValue 

class ApproximatePolicy():
    def __init__(self, variance, inputVar, outputVar):
        self.actionVaricen = actionVariance
        self.inputVar = inputVar
        self.outputVar = outputVar
    def __call__(self, state, model):
        actionMean = model.run(self.outputVar['actionMean_'], feed_dict={self.inputVar['state_']: state, self.inputVar['variance_']: self.variance})
        action = np.random.normal(mean, self.variance)
        print(action)
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
    def __init__(self, inputVar, outputVar, summaryPath):
        self.summaryWriter = tf.summary.FileWriter(summaryPath)
        self.inputVar = inputVar
        self.outputVar = outputVar
    def __call__(self, episode, normalizedAccumulatedRewardsEpisode, model):
        mergedEpisode = np.concatenate(episode)
        stateBatch, actionBatch = list(zip(*mergedEpisode))
        accumulatedRewardsBatch = np.concatenate(normalizedAccumulatedRewardsEpisode)
        loss, trainOpt = model.run([self.outputVar['loss_'], self.outputVar['trainOpt_']], feed_dict={self.inputVar['state_']: np.vstack(np.array(stateBatch)),
                                                                                                    self.inputVar['action_']: np.vstack(np.array(actionBatch)),
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
    variance = 1.0
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
        action_ = tf.placeholder(tf.float32, [None, 1], name="action_")
        variance_ = tf.placeholder(tf.float32, name="variance")
        accumulatedRewards_ = tf.placeholder(tf.float32, [None,], name="accumulatedRewards_")

        # Add this placeholder for having this variable in tensorboard
        #mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

        with tf.name_scope("fc1"):
            fc1 = tf.contrib.layers.fully_connected(inputs = state_,
                                                    num_outputs = 32,
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("fc2"):
            fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
                                                    #num_outputs = numActionSpace,
                                                    num_outputs = 32,
                                                    activation_fn= tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("fc3"):
            fc3 = tf.contrib.layers.fully_connected(inputs = fc2,
                                                    num_outputs = 1,
                                                    activation_fn= tf.nn.sigmoid,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())
        
        with tf.name_scope("mean"):
            mean_ = fc3*2.0 - 1
        
        with tf.name_scope("loss"):
            #neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actionLabel_)
            neg_prob = - (mean_ - action_) / variance
            loss_ = tf.reduce_sum(neg_prob * accumulatedRewards_)
        tf.summary.scalar("Loss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRate).minimize(loss_)
    
    mergedSummary = tf.summary.merge_all()
    
    inputApproximatePolicy = {'state_' : state_, "variance" : variance_}
    outputApproximatePolicy = {'mean_' : mean_}
    
    inputTrain = {'state_' : state_, 'action_' : action_, 'accumulatedRewards_' : accumulatedRewards_}
    outputTrain = {'loss_' : loss_, 'trainOpt_' : trainOpt_}

    with tf.Session() as model:
        model.run(tf.global_variables_initializer())    

        approximatePolicy = ApproximatePolicy(variance, inputApproximatePolicy, outputApproximatePolicy)

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

        train = TrainTensorflow(inputTrain, outputTrain,summaryPath) 

        policyGradient = PolicyGradient(numTrajectory, maxEpisode)

        trainedModel = policyGradient(model, approximatePolicy, sampleTrajectory, accumulateRewards, train)

        saveModel = dataSave.SaveModel(savePath)
        modelSave = saveModel(model)

        """
        transitionPlay = env.TransitionFunction(envModelName, renderOpen = True)
        """
        transitionPlay = lambda s,a: cartpole_env.cartpole_continuous_action_transition_function(s,a,renderOn=True)
        samplePlay = SampleTrajectory(maxTimeStep, transitionPlay, isTerminal, reset)
        policy = lambda state: approximatePolicy(state, trainedModel)
        playEpisode = [samplePlay(policy) for index in range(5)]
        accumulatedRewardsEpisode = [accumulateRewards(trajectory) for trajectory in playEpisode]
        print([rewards[0] for rewards in accumulatedRewardsEpisode])

if __name__ == "__main__":
    main()
