import nn as nn
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import functools as ft
import env as env
import cartpole_env as cartpole_env
import reward as reward
import dataSave as dataSave
import offlineA2C as offlineA2C
import policyGradientContinuous as pgc
import getopt
import sys


class runPolicyGradientContinuoues:
	def __init__(self, maxTimeStep, aliveBouns, rewardDecay, numTrajectory, maxEpisode, model, summaryPath):
		self.maxTimeStep = maxTimeStep
		self.aliveBouns = aliveBouns
		self.rewardDecay = rewardDecay
		self.numTrajectory = numTrajectory
		self.maxEpisode = maxEpisode
		self.model = model
		self.summaryPath = summaryPath
	def __call__(self):
		summaryWriter = tf.summary.FileWriter(self.summaryPath, graph=self.model.graph)
		rewardFunction = reward.RewardFunction(self.aliveBouns)
		accumulateRewards = pgc.AccumulateRewards(self.rewardDecay, rewardFunction)

		train = pgc.TrainTensorflow(summaryWriter)

		policyGradient = pgc.PolicyGradient(self.numTrajectory, self.maxEpisode)

		trainedModel = policyGradient(self.model, pgc.approximatePolicy, pgc.sampleTrajectory, accumulateRewards, train)

		transitionPlay = cartpole_env.Cartpole_continuous_action_transition_function(renderOn=True)
		samplePlay = pgc.SampleTrajectory(self.maxTimeStep, transitionPlay, self.isTerminal, self.reset)
		policy = lambda state: pgc.approximatePolicy(state, self.model)
		playEpisode = [samplePlay(policy) for index in range(5)]
		print(np.mean([len(playEpisode[index]) for index in range(5)]))


class runTaskByMethod:
	def __init__(self, task):
		self.numActionSpace = 1
		self.numStateSpace = 4
		self.actionLow = -2
		self.actionHigh = 2
		self.maxTimeStep = 200
		self.aliveBouns = 1
		self.rewardDecay = 1
		self.numTrajectory = 1000
		self.maxEpisode = 100
		self.learningRate = 0.01
		self.learningRateActor = 0.01
		self.learningRateCritic = 0.01
		self.renderOn = False
		self.deathPenalty = -20
		self.summaryPath = 'test/'
		self.savePath = 'data/tmpModelGaussian.ckpt'
		self.savePathActor = 'data/tmpModelActor.ckpt'
		self.savePathCritic = 'data/tmpModelCritic.ckpt'
		if task == 'inverted_pendulum':
			self.envModelName = 'inverted_pendulum'
			self.maxQPos = 0.2
			self.qPosInitNoise = 0.001
			self.qVelInitNoise = 0.001
			self.transitionFunction = env.TransitionFunction(self.envModelName, self.renderOn)
			self.isTerminal = env.IsTerminal(self.maxQPos)
			self.reset = env.Reset(self.envModelName, self.qPosInitNoise, self.qVelInitNoise)
			self.rewardFunction = reward.RewardFunctionTerminalPenalty(self.aliveBouns, self.deathPenalty,
			                                                           self.isTerminal)
		elif task == 'cart_pole':
			self.transitionFunction = cartpole_env.Cartpole_continuous_action_transition_function(renderOn = self.renderOn)
			self.isTerminal = cartpole_env.cartpole_done_function
			self.reset = cartpole_env.cartpole_get_initial_state
			self.rewardFunction = reward.CartpoleRewardFunction(self.aliveBouns)

	def __call__(self, method):
		benchMark = None
		if method == 'offlineA2C':
			mIOA2C = nn.modelInOfflineA2C(self.numStateSpace, self.numActionSpace, self.actionLow, self.actionHigh,
								 self.learningRateActor, self.learningRateCritic)
			actorModel, criticModel, actorGraph, criticGraph = mIOA2C()
			criticWriter = tf.summary.FileWriter('tensorBorad/critic', graph=criticGraph)
			actorWriter = tf.summary.FileWriter('tensorBoard/actor', graph=actorGraph)
			# actorSaver = tf.train.Saver(tf.global_variables())
			# criticSaver = tf.train.Saver(tf.global_variables())

			sampleTrajectory = offlineA2C.SampleTrajectory(self.maxTimeStep, self.transitionFunction, self.isTerminal, self.reset)

			accumulateRewards = offlineA2C.AccumulateRewards(self.rewardDecay, self.rewardFunction)

			trainCritic = offlineA2C.TrainCriticMonteCarloTensorflow(criticWriter, accumulateRewards)

			trainCritic = offlineA2C.TrainCriticBootstrapTensorflow(criticWriter, self.rewardDecay, self.rewardFunction)
			estimateAdvantage = offlineA2C.EstimateAdvantageBootstrap(self.rewardDecay, self.rewardFunction)
			trainActor = offlineA2C.TrainActorBootstrapTensorflow(actorWriter)

			actorCritic = offlineA2C.offlineAdvantageActorCritic(self.numTrajectory, self.maxEpisode)

			trainedActorModel, trainedCriticModel, benchMark = actorCritic(actorModel, criticModel, offlineA2C.approximatePolicy,
			                                                    sampleTrajectory, trainCritic,
			                                                    offlineA2C.approximateValue, estimateAdvantage, trainActor)

			# with actorModel.as_default():
			# 	actorSaver.save(trainedActorModel, self.savePathActor)
			# with criticModel.as_default():
			# 	criticSaver.save(trainedCriticModel, self.savePathCritic)

			# transitionPlay = cartpole_env.Cartpole_continuous_action_transition_function(renderOn=True)
			# samplePlay = offlineA2C.SampleTrajectory(self.maxTimeStep, transitionPlay, self.isTerminal, self.reset)
			# actor = lambda state: offlineA2C.approximatePolicy(state, trainedActorModel)
			# playEpisode = [samplePlay(actor) for index in range(5)]
			# print(np.mean([len(playEpisode[index]) for index in range(5)]))
		elif method == 'policyGradientContinuous':
			mIPGC = nn.modelInPolicyGradientContinuous(self.numStateSpace, self.numActionSpace, self.actionLow, self.actionHigh, self.learningRate)
			pgcModel = mIPGC()
			sampleTrajectory = pgc.SampleTrajectory(self.maxTimeStep, self.transitionFunction, self.isTerminal, self.reset)
			accumulateRewards = pgc.AccumulateRewards(self.rewardDecay, self.rewardFunction)
			summaryWriter = tf.summary.FileWriter(self.summaryPath, graph=pgcModel.graph)
			train = pgc.TrainTensorflow(summaryWriter)

			policyGradient = pgc.PolicyGradient(self.numTrajectory, self.maxEpisode)

			trainedModel, benchMark = policyGradient(pgcModel, pgc.approximatePolicy, sampleTrajectory, accumulateRewards, train)

			saveModel = dataSave.SaveModel(self.savePath)
			modelSave = saveModel(pgcModel)

			# transitionPlay = cartpole_env.Cartpole_continuous_action_transition_function(renderOn=True)
			# samplePlay = pgc.SampleTrajectory(self.maxTimeStep, transitionPlay, self.isTerminal, self.reset)
			# policy = lambda state: pgc.approximatePolicy(state, pgcModel)
			# playEpisode = [samplePlay(policy) for index in range(5)]
			# print(np.mean([len(playEpisode[index]) for index in range(5)]))

		print("BenchMark: %d" % benchMark)
		return benchMark


def main(argv):
	methods = ['policyGradientContinuous', 'offlineA2C']
	tasks = ['cart_pole', 'inverted_pendulum']
	method = None
	task = None
	times = 50
	try:
		opts, args = getopt.getopt(argv, "lhm:t:", ["list", "help", "method=", "task="])
	except getopt.GetoptError:
		print('python3 file.py -m <method> -t <task>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-h', "--help"):
			print('python3 file.py -m <method> -t <task> -l')
			sys.exit()
		elif opt in ("-m", "--method"):
			method = arg
			if method not in methods:
				print('Unavailable method')
				sys.exit(2)
		elif opt in ("-t", "--task"):
			task = arg
			if task not in tasks:
				print('Unavilable task')
				sys.exit(2)
		elif opt in ("-l", "--list"):
			print("methods:")
			print(methods)
			print("tasks:")
			print(tasks)
	if method is None or task is None:
		print("Specify at least one method and one task.")
		sys.exit(3)
	# task = 'cart_pole'
	# method = 'policyGradientContinuous'
	# task = 'cart_pole'
	# method = 'offlineA2C'
	test = runTaskByMethod(task)
	benchMarks = np.array([test(method) for i in range(0, times)])
	print(np.mean(benchMarks))


if __name__ == '__main__':
	main(sys.argv[1:])