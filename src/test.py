import tensorflow as tf


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

