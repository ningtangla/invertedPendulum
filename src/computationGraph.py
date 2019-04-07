import tensorflow as tf
import numpy as np


numActorFC1Unit = 128
numActorFC2Unit = 128
numCriticFC1Unit = 128
numCriticFC2Unit = 128
learningRateActor = 0.01
learningRateCritic = 0.01
l2DecayCritic = 0#0.0001

softReplaceRatio = 0.01

def createDDPGActorGraph(numStateSpace, numActionSpace, actionRatio, agentIndex):

    if agentIndex == 0:
        # prey
        # initializer = tf.random_normal_initializer(mean=-1, stddev=0.1)
        initializer = tf.random_uniform_initializer(minval=-1, maxval=0)
    else:
        #predator
        initializer = tf.random_uniform_initializer(minval=0, maxval=1)
        # initializer = tf.random_normal_initializer(mean=1, stddev=0.1)

    bias_initializer = tf.random_normal_initializer(mean=-10.0, stddev=1.0)

    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.variable_scope("inputs"):
            ownState_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace], name="ownState_"))
            gradientQPartialOwnAction_ = tf.placeholder(tf.float32, [None, numActionSpace], name="gradientQPartialOwnAction_")

        with tf.variable_scope("evaluationHidden"):
            evaFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = ownState_, units = numActorFC1Unit, activation = tf.nn.relu, kernel_initializer=initializer))
            evaFullyConnected2_ = tf.layers.batch_normalization(tf.layers.dense(inputs = evaFullyConnected1_, units = numActorFC2Unit, activation = tf.nn.relu, kernel_initializer=initializer))
            evaActionActivation_ = tf.layers.dense(inputs = evaFullyConnected2_, units = numActionSpace, activation = tf.nn.tanh, kernel_initializer=initializer, bias_initializer=bias_initializer)
            
        with tf.variable_scope("targetHidden"):
            tarFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = ownState_, units = numActorFC1Unit, activation = tf.nn.relu, kernel_initializer=initializer))
            tarFullyConnected2_ = tf.layers.batch_normalization(tf.layers.dense(inputs = tarFullyConnected1_, units = numActorFC2Unit, activation = tf.nn.relu, kernel_initializer=initializer))
            tarActionActivation_ = tf.layers.dense(inputs = tarFullyConnected2_, units = numActionSpace, activation = tf.nn.tanh, kernel_initializer=initializer, bias_initializer=bias_initializer)
        
        with tf.variable_scope("outputs"):        
            evaParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluationHidden')
            tarParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            numParams_ = tf.constant(len(evaParams_), name = 'numParams_')
            updateTargetParameter_ = [tf.assign(tarParam_, (1 - softReplaceRatio) * tarParam_ + softReplaceRatio * evaParam_, name = 'assign'+str(paramIndex_)) for paramIndex_,
                tarParam_, evaParam_ in zip(range(len(evaParams_)), tarParams_, evaParams_)]
            evaOwnAction_ = tf.multiply(evaActionActivation_, actionRatio, name = 'evaOwnAction_')
            tarOwnAction_ = tf.multiply(tarActionActivation_, actionRatio, name = 'tarOwnAction_')
            gradientQPartialActorParameter_ = tf.gradients(ys = evaOwnAction_, xs = evaParams_, grad_ys = gradientQPartialOwnAction_, name = 'gradientQPartialActorParameter_')

        with tf.variable_scope("train"):
            #-learningRate for ascent
            trainOpt_ = tf.train.AdamOptimizer(-learningRateActor, name = 'adamOpt_').apply_gradients(zip(gradientQPartialActorParameter_, evaParams_))
        actorInit = tf.global_variables_initializer()
        # actorInit = None
        
        actorSummary = tf.summary.merge_all()
        actorSaver = tf.train.Saver(tf.global_variables())

    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    return actorModel

def createDDPGCriticGraph(numStateSpace, numActionSpace, numAgent, agentIndex):

    if agentIndex == 0:
        # prey
        # initializer = tf.random_normal_initializer
        initializer = tf.random_uniform_initializer(minval=-1, maxval=0)
    else:
        #predator
        initializer = tf.random_uniform_initializer(minval=0, maxval=1)

    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.variable_scope("inputs"):
            allAgentState_ = tf.layers.batch_normalization(tf.placeholder(tf.float32, [None, numStateSpace * numAgent], name="allAgentState_"))
            ownAction_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, numActionSpace]), name='ownAction_')
            otherAction_ = tf.stop_gradient(tf.placeholder(tf.float32, [None, numActionSpace * (numAgent -1)]), name='otherAction_')
            QAim_ = tf.placeholder(tf.float32, [None, 1], name="QAim_")

        with tf.variable_scope("evaluationHidden"):
            evaFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = allAgentState_, units = numCriticFC1Unit, activation = tf.nn.relu, kernel_initializer=initializer))
            numFullyConnected2Units = numCriticFC2Unit
            evaStateFC1ToFullyConnected2Weights_ = tf.get_variable(name='evaStateFC1ToFullyConnected2Weights', shape = [numCriticFC1Unit, numFullyConnected2Units], initializer=initializer)
            evaOwnActionToFullyConnected2Weights_ = tf.get_variable(name='evaOwnActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units], initializer=initializer)
            evaOtherActionToFullyConnected2Weights_ = tf.get_variable(name='evaOtherActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units], initializer=initializer)
            evaFullyConnected2Bias_ = tf.get_variable(name = 'evaFullyConnected2Bias', shape = [numFullyConnected2Units])
            evaFullyConnected2_ = tf.nn.relu(tf.matmul(evaFullyConnected1_, evaStateFC1ToFullyConnected2Weights_) + tf.matmul(ownAction_, evaOwnActionToFullyConnected2Weights_) +
                    tf.matmul(otherAction_, evaOtherActionToFullyConnected2Weights_) + evaFullyConnected2Bias_ )
            evaQActivation_ = tf.layers.dense(inputs = evaFullyConnected2_, units = 1, activation = None, )

        with tf.variable_scope("targetHidden"):
            tarFullyConnected1_ = tf.layers.batch_normalization(tf.layers.dense(inputs = allAgentState_, units = numCriticFC1Unit, activation = tf.nn.relu, kernel_initializer=initializer))
            numFullyConnected2Units = numCriticFC2Unit
            tarStateFC1ToFullyConnected2Weights_ = tf.get_variable(name='tarStateFC1ToFullyConnected2Weights', shape = [numCriticFC1Unit, numFullyConnected2Units], initializer=initializer)
            tarOwnActionToFullyConnected2Weights_ = tf.get_variable(name='tarOwnActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units], initializer=initializer)
            tarOtherActionToFullyConnected2Weights_ = tf.get_variable(name='tarOtherActionToFullyConnected2Weights', shape = [numActionSpace, numFullyConnected2Units], initializer=initializer)
            tarFullyConnected2Bias_ = tf.get_variable(name = 'tarFullyConnected2Bias', shape = [numFullyConnected2Units])
            tarFullyConnected2_ = tf.nn.relu(tf.matmul(tarFullyConnected1_, tarStateFC1ToFullyConnected2Weights_) + tf.matmul(ownAction_, tarOwnActionToFullyConnected2Weights_) +
                    tf.matmul(otherAction_, tarOtherActionToFullyConnected2Weights_) + tarFullyConnected2Bias_ )
            tarQActivation_ = tf.layers.dense(inputs = tarFullyConnected2_, units = 1, activation = None)
        
        with tf.variable_scope("outputs"):        
            evaParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluationHidden')
            tarParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetHidden')
            numParams_ = tf.constant(len(evaParams_), name = 'numParams_')
            updateTargetParameter_ = [tf.assign(tarParam_, (1 - softReplaceRatio) * tarParam_ + softReplaceRatio * evaParam_, name = 'assign'+str(paramIndex_)) for paramIndex_,
                tarParam_, evaParam_ in zip(range(len(evaParams_)), tarParams_, evaParams_)]
            evaQ_ = tf.multiply(evaQActivation_, 1, name = 'evaQ_')
            tarQ_ = tf.multiply(tarQActivation_, 1, name = 'tarQ_')
            diff_ = tf.subtract(QAim_, evaQ_, name = 'diff_')
            loss_ = tf.reduce_mean(tf.square(diff_), name = 'loss_')
            gradientQPartialOwnAction_ = tf.gradients(evaQ_, ownAction_, name = 'gradientQPartialOwnAction_')
            criticLossSummary = tf.summary.scalar("CriticLoss", loss_)
        with tf.variable_scope("train"):
            trainOpt_ = tf.contrib.opt.AdamWOptimizer(weight_decay = l2DecayCritic, learning_rate = learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        # criticInit = None
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(criticInit)   
    return criticModel
