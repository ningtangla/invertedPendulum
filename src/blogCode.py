import tensorflow as tf
import numpy as np
import gym
import functools as ft

env = gym.make('CartPole-v0')
#env = env.unwrapped
# Policy gradient has high variance, seed for reproducability
#env.seed(1)

actionSpace = np.arange(-1, 1.01, 2)
numActionSpace = len(actionSpace)
numStateSpace = 4

envModelName = 'inverted_pendulum'
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]
maxTimeStep = 500
maxCartXPos = np.array([5, 0, 0])
qPosInitNoise = 0.01
qVelInitNoise = 0.01
maxTimeStep = 3000

sitePenaltyPara = 0
actionPenaltyPara = 0
aliveBouns = 1
siteTargetXPos = np.array([0, 0, 0.6])
rewardDecay = 1

numTrajectory = 300
maxEpisode = 500

learningRate = 0.01

def fowardComputationModel(input, output, model):
    outputValue = model.run(output, feed_dict={input_: input})
    return outputValue 

class ApproximatePolicy():
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
    def __call__(self, state, model):
        reshapedState = np.ravel(state)
        actionDistribution = model(state)
        actionLabel = np.random.choice(actionSpace, p=actionDistribution.ravel())  # select action w.r.t the actions prob
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

with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, numStateSpace], name="input_")
    actions = tf.placeholder(tf.int32, [None, numActionSpace], name="actions")
    accumulatedRewards = tf.placeholder(tf.float32, [None,], name="accumulatedRewards")

    # Add this placeholder for having this variable in tensorboard
    #mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs = input_,
                                                num_outputs = 10,
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
        actionDistribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
        # If you have single-class labels, where an object can only belong to one class, you might now consider using
        # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
        loss = tf.reduce_sum(neg_log_prob * accumulatedRewards)


    with tf.name_scope("train"):
        trainOpt = tf.train.AdamOptimizer(learningRate).minimize(loss)

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("tensorboard/1")

## Losses
tf.summary.scalar("Loss", loss)

## Reward mean
#tf.summary.scalar("Reward_mean", mean_reward_)

write_op = tf.summary.merge_all()
#
#sess = tf.Session() 
#sess.run(tf.global_variables_initializer())
#saver = tf.train.Saver()
#
#approximatePolicy = ApproximatePolicy(actionSpace)
#
#transitionFunction = env.TransitionFunction(modelName)
#isTerminal = env.IsTerminal(maxCartXPos)
#reset = env.Reset(modelName, qPosInitNoise, qVelInitNoise)
#sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, isTerminal, reset)
#
#rewardFunction = reward.RewardFunction(sitePenaltyPara, actionPenaltyPara, aliveBouns, siteTargetXPos) 
#accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)

gamma = 0.95
def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    return discounted_episode_rewards

episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for episode in range(maxEpisode):
        if episode % 100 == 0:
            print(episode)
        episode_rewards_sum = 0

        # Launch the game
        state = env.reset()
         
        while True:
            
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action_probability_distribution = sess.run(actionDistribution, feed_dict={input_: state.reshape([1,4])})
            action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

            # Perform a
            new_state, reward, done, info = env.step(action)

            # Store s, a, r
            episode_states.append(state)
                        
            # For actions because we output only one (the index) we need 2 (1 is for the action taken)
            # We need [0., 1.] (if we take right) not just the index
            action_ = np.zeros(numActionSpace)
            action_[action] = 1
            
            episode_actions.append(action_)
            
            episode_rewards.append(reward)
            if done:
                # Calculate sum reward
                episode_rewards_sum = np.sum(episode_rewards)
                
                # Calculate discounted reward
                discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)
                
                # Feedforward, gradient and backpropagation
                loss_, _ = sess.run([loss, trainOpt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                 actions: np.vstack(np.array(episode_actions)),
                                                                 accumulatedRewards: discounted_episode_rewards 
                                                                })
                writer.flush()

                # Reset the transition stores
                episode_states, episode_actions, episode_rewards = [],[],[]

                break

            state = new_state

    # Restore variables from disk.
    saver.save(sess, "data/tepModel.ckpt")

    for i in range(5):
        episode_rewards_sum = 0

        # Launch the game
        state = env.reset()
        
        env.render()

        for j in range(1000):

            env.render()
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action_probability_distribution = sess.run(actionDistribution, feed_dict={input_: state.reshape([1,4])})

            action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

            # Perform a
            new_state, reward, done, info = env.step(action)

            # Store s, a, r
            episode_states.append(state)

            # For actions because we output only one (the index) we need 2 (1 is for the action taken)
            # We need [0., 1.] (if we take right) not just the index
            action_ = np.zeros(numActionSpace)
            action_[action] = 1

            episode_actions.append(action_)

            episode_rewards.append(reward)
            if done:
                # Calculate sum reward
                episode_rewards_sum = np.sum(episode_rewards)
                print(episode_rewards_sum)
            
                episode_states, episode_actions, episode_rewards = [],[],[]

                break

            state = new_state
