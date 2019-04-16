import tensorflow as tf
import tensorflow_probability as tfp


class modelInPolicyGradientContinuous:
	def __init__(self, numStateSpace, numActionSpace, actionLow, actionHigh, learningRate):
		self.numStateSpace = numStateSpace
		self.numActionSpace = numActionSpace
		self.actionLow = actionLow
		self.actionHigh = actionHigh
		self.learningRate = learningRate
		self.actionRatio = (actionHigh - actionLow) / 2.
		self.graph = None

	def __call__(self):
		with tf.name_scope("inputs"):
			state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
			action_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="action_")
			accumulatedRewards_ = tf.placeholder(tf.float32, [None, ], name="accumulatedRewards_")

		with tf.name_scope("hidden"):
			fullyConnected1_ = tf.layers.dense(inputs=state_, units=30, activation=tf.nn.relu)
			fullyConnected2_ = tf.layers.dense(inputs=fullyConnected1_, units=20, activation=tf.nn.relu)
			actionMean_ = tf.layers.dense(inputs=fullyConnected2_, units=self.numActionSpace, activation=tf.nn.tanh)
			actionVariance_ = tf.layers.dense(inputs=fullyConnected2_, units=self.numActionSpace, activation=tf.nn.softplus)

		with tf.name_scope("outputs"):
			actionDistribution_ = tfp.distributions.MultivariateNormalDiag(actionMean_ * self.actionRatio,
			                                                               actionVariance_ + 1e-8,
			                                                               name='actionDistribution_')
			actionSample_ = tf.clip_by_value(actionDistribution_.sample(), self.actionLow, self.actionHigh, name='actionSample_')
			negLogProb_ = - actionDistribution_.log_prob(action_, name='negLogProb_')
			loss_ = tf.reduce_sum(tf.multiply(negLogProb_, accumulatedRewards_), name='loss_')
			tf.summary.scalar("Loss", loss_)

		with tf.name_scope("train"):
			trainOpt_ = tf.train.AdamOptimizer(self.learningRate, name='adamOpt_').minimize(loss_)

		mergedSummary = tf.summary.merge_all()
		model = tf.Session()
		model.run(tf.global_variables_initializer())
		self.graph = model.graph
		return model


class modelInOfflineA2C:
	def __init__(self, numStateSpace, numActionSpace, actionLow, actionHigh, learningRateActor, learningRateCritic):
		self.numStateSpace = numStateSpace
		self.numActionSpace = numActionSpace
		self.actionLow = actionLow
		self.actionHigh = actionHigh
		self.learningRateActor = learningRateActor
		self.learningRateCritic = learningRateCritic
		self.actionRatio = (actionHigh - actionLow) / 2.

	def __call__(self):
		actorGraph = tf.Graph()
		with actorGraph.as_default():
			with tf.name_scope("inputs"):
				state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
				action_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="action_")
				advantages_ = tf.placeholder(tf.float32, [None, ], name="advantages_")

			with tf.name_scope("hidden"):
				fullyConnected1_ = tf.layers.dense(inputs=state_, units=30, activation=tf.nn.relu)
				fullyConnected2_ = tf.layers.dense(inputs=fullyConnected1_, units=20, activation=tf.nn.relu)
				actionMean_ = tf.layers.dense(inputs=fullyConnected2_, units=self.numActionSpace, activation=tf.nn.tanh)
				actionVariance_ = tf.layers.dense(inputs=fullyConnected2_, units=self.numActionSpace,
				                                  activation=tf.nn.softplus)

			with tf.name_scope("outputs"):
				actionDistribution_ = tfp.distributions.MultivariateNormalDiag(actionMean_ * self.actionRatio,
				                                                               actionVariance_ + 1e-8,
				                                                               name='actionDistribution_')
				actionSample_ = tf.clip_by_value(actionDistribution_.sample(), self.actionLow, self.actionHigh,
				                                 name='actionSample_')
				negLogProb_ = - actionDistribution_.log_prob(action_, name='negLogProb_')
				loss_ = tf.reduce_sum(tf.multiply(negLogProb_, advantages_), name='loss_')
			actorLossSummary = tf.summary.scalar("ActorLoss", loss_)

			with tf.name_scope("train"):
				trainOpt_ = tf.train.AdamOptimizer(self.learningRateActor, name='adamOpt_').minimize(loss_)

			actorInit = tf.global_variables_initializer()

			actorSummary = tf.summary.merge_all()

		actorModel = tf.Session(graph=actorGraph)
		actorModel.run(actorInit)

		criticGraph = tf.Graph()
		with criticGraph.as_default():
			with tf.name_scope("inputs"):
				state_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="state_")
				valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")

			with tf.name_scope("hidden1"):
				fullyConnected1_ = tf.layers.dense(inputs=state_, units=30, activation=tf.nn.relu)

			with tf.name_scope("outputs"):
				value_ = tf.layers.dense(inputs=fullyConnected1_, units=1, activation=None, name='value_')
				diff_ = tf.subtract(valueTarget_, value_, name='diff_')
				loss_ = tf.reduce_mean(tf.square(diff_), name='loss_')
			criticLossSummary = tf.summary.scalar("CriticLoss", loss_)

			with tf.name_scope("train"):
				trainOpt_ = tf.train.AdamOptimizer(self.learningRateCritic, name='adamOpt_').minimize(loss_)

			criticInit = tf.global_variables_initializer()

			criticSummary = tf.summary.merge_all()


		criticModel = tf.Session(graph=criticGraph)
		criticModel.run(criticInit)
		return actorModel, criticModel, actorGraph, criticGraph