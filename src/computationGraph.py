import tensorflow as tf
import numpy as np

def createDDPGActorGraph(numStateSpace, numActionSpace, softReplaceRatio, agentIndex):

    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.variable_scope("inputs"):
            state_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="state_"))
            gradientQPartialAction_ = tf.placeholder(tf.float32, [None, numActionSpace], name="gradientQPartialAction_")

        with tf.variable_scope("evaluationHidden"):
            evaFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = 20, activation = tf.nn.relu))
            evaFullyConnected2_ = tf.layers.batch_normalization(tf.layers.dense(inputs = evaFullyConnected1_, units = 20, activation = tf.nn.relu))
            evaActionActivation_ = tf.layers.dense(inputs = evaFullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
            
        with tf.variable_scope("targetHidden"):
            tarFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = 20, activation = tf.nn.relu))
            tarFullyConnected2_ = tf.layers.batch_normalization(tf.layers.dense(inputs = tarFullyConnected1_, units = 20, activation = tf.nn.relu))
            tarActionActivation_ = tf.layers.dense(inputs = tarFullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
        
        with tf.variable_scope("outputs"):        
            evaParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluationHidden')
            tarParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            numParams_ = tf.constant(len(evaParams_), name = 'numParams_')
            updateTargetParameter_ = [tf.assign(tarParam_, (1 - softReplaceRatio) * tarParam_ + softReplaceRatio * evaParam_, name = 'assign'+str(paramIndex_)) for paramIndex_,
                tarParam_, evaParam_ in zip(range(len(evaParams_)), tarParams_, evaParams_)]
            evaAction_ = tf.multiply(evaActionActivation_, actionRatio, name = 'evaAction_')
            tarAction_ = tf.multiply(tarActionActivation_, actionRatio, name = 'tarAction_')
            gradientQPartialActorParameter_ = tf.gradients(ys = evaAction_, xs = evaParams_, grad_ys = gradientQPartialAction_, name = 'gradientQPartialActorParameter_')

        with tf.variable_scope("train"):
            #-learningRate for ascent
            trainOpt_ = tf.train.AdamOptimizer(-learningRateActor, name = 'adamOpt_').apply_gradients(zip(gradientQPartialActorParameter_, evaParams_))
        actorInit = tf.global_variables_initializer()
        
        actorSummary = tf.summary.merge_all()
        actorSaver = tf.train.Saver(tf.global_variables())

    actorWriter = tf.summary.FileWriter('tensorBoard/' + str(agentIndex) + 'actorDDPG', graph = actorGraph)
    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    return actorModel

def createDDPGCriticGraph(numStateSpace, numActionSpace, softReplaceRatio, agentIndex):
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.variable_scope("inputs"):
            state_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="state_"))
            action_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, numActionSpace]), name='action_')
            QTarget_ = tf.placeholder(tf.float32, [None, 1], name="QTarget_")

        with tf.variable_scope("evaluationHidden"):
            evaFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = 300, activation = tf.nn.relu))
            numFullyConnected2Units = 100
            evaStateFC1ToFullyConnected2Weights_ = tf.get_variable(name='evaStateFC1ToFullyConnected2Weights', shape = [300, numFullyConnected2Units])
            evaActionToFullyConnected2Weights_ = tf.get_variable(name='evaActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units])
            evaFullyConnected2Bias_ = tf.get_variable(name = 'evaFullyConnected2Bias', shape = [numFullyConnected2Units])
            evaFullyConnected2_ = tf.nn.relu(tf.matmul(evaFullyConnected1_, evaStateFC1ToFullyConnected2Weights_) + tf.matmul(action_, evaActionToFullyConnected2Weights_) + evaFullyConnected2Bias_ )
            evaQActivation_ = tf.layers.dense(inputs = evaFullyConnected2_, units = 1, activation = None)

        with tf.variable_scope("targetHidden"):
            tarFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = state_, units = 300, activation = tf.nn.relu))
            numFullyConnected2Units = 100
            tarStateFC1ToFullyConnected2Weights_ = tf.get_variable(name='tarStateFC1ToFullyConnected2Weights', shape = [300, numFullyConnected2Units])
            tarActionToFullyConnected2Weights_ = tf.get_variable(name='tarActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units])
            tarFullyConnected2Bias_ = tf.get_variable(name = 'tarFullyConnected2Bias', shape = [numFullyConnected2Units])
            tarFullyConnected2_ = tf.nn.relu(tf.matmul(tarFullyConnected1_, tarStateFC1ToFullyConnected2Weights_) + tf.matmul(action_, tarActionToFullyConnected2Weights_) + tarFullyConnected2Bias_ )
            tarQActivation_ = tf.layers.dense(inputs = tarFullyConnected2_, units = 1, activation = None)
        
        with tf.variable_scope("outputs"):        
            evaParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluationHidden')
            tarParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            numParams_ = tf.constant(len(evaParams_), name = 'numParams_')
            updateTargetParameter_ = [tf.assign(tarParam_, (1 - softReplaceRatio) * tarParam_ + softReplaceRatio * evaParam_, name = 'assign'+str(paramIndex_)) for paramIndex_,
                tarParam_, evaParam_ in zip(range(len(evaParams_)), tarParams_, evaParams_)]
            evaQ_ = tf.multiply(evaQActivation_, 1, name = 'evaQ_')
            tarQ_ = tf.multiply(tarQActivation_, 1, name = 'tarQ_')
            diff_ = tf.subtract(QTarget_, evaQ_, name = 'diff_')
            loss_ = tf.reduce_mean(tf.square(diff_), name = 'loss_')
            gradientQPartialAction_ = tf.gradients(evaQ_, action_, name = 'gradientQPartialAction_')
            criticLossSummary = tf.summary.scalar("CriticLoss", loss_)
        with tf.variable_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticWriter = tf.summary.FileWriter('tensorBoard/' + str(agentIndex) + 'criticDDPG', graph = criticGraph)
    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(criticInit)   
    return criticModel
