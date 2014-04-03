import UML

from nose.tools import *
import numpy.testing

from UML.interfaces.custom_learner import CustomLearner


@raises(TypeError)
def testCustomLearnerValidationNoType():
	""" Test  CustomLearner's validation for the problemType class attribute """
	class NoType(CustomLearner):
		def train(self, trainX, trainY):
			return None
		def apply(self, testX):
			return None
	CustomLearner.validateSubclass(NoType)

@raises(TypeError)
def testCustomLearnerValidationWrongParamsTrain():
	""" Test CustomLearner's validation of required train() parameters """
	class WrongArgs(CustomLearner):
		problemType = 'unknown'
		def train(self, trainZ, foo):
			return None
		def apply(self, testX):
			return None
	CustomLearner.validateSubclass(WrongArgs)

@raises(TypeError)
def testCustomLearnerValidationWrongParamsIncTrain():
	""" Test CustomLearner's validation of required incrementalTrain() parameters """
	class WrongArgs(CustomLearner):
		problemType = 'unknown'
		def train(self, trainX, trainY):
			return None
		def incrementalTrain(self, trainZ, foo):
			return None
		def apply(self, testX):
			return None
	CustomLearner.validateSubclass(WrongArgs)

@raises(TypeError)
def testCustomLearnerValidationWrongParamsApply():
	""" Test CustomLearner's validation of required apply() parameters """
	class WrongArgs(CustomLearner):
		problemType = 'unknown'
		def train(self, trainX, trainY):
			return None
		def apply(self, testZ):
			return None
	CustomLearner.validateSubclass(WrongArgs)

@raises(TypeError)
def testCustomLearnerValidationNoTrainOrIncTrain():
	""" Test CustomLearner's validation of requiring either train() or incrementalTrain() """
	class NoTrain(CustomLearner):
		problemType = 'unknown'
		def apply(self, testX):
			return None
	CustomLearner.validateSubclass(NoTrain)

@raises(TypeError)
def testCustomLearnerValidationGetScoresParamsMatch():
	""" Test CustomLearner's validation of the match between getScores() param names and apply()"""
	class NoType(CustomLearner):
		problemType = 'classification'

		def train(self, trainX, trainY):
			return None
		def apply(self, testX, foo):
			return None
		def getScores(self, testX):
			return None
	CustomLearner.validateSubclass(NoType)

@raises(TypeError)
def testCustomLearnerValidationInitNoParams():
	""" Test CustomLearner's validation of __init__'s params """
	class TooMany(CustomLearner):
		problemType = 'classification'
		def __init__(self, so, many, params):
			super(TooMany, self).__init__()
		def train(self, trainX, trainY):
			return None
		def apply(self, testX, foo):
			return None
	CustomLearner.validateSubclass(TooMany)

@raises(TypeError)
def testCustomLearnerValidationInstantiates():
	""" Test CustomLearner's validation actually tries to instantiation the subclass """
	class NoApp(CustomLearner):
		problemType = 'classification'
		def train(self, trainX, trainY):
			return None
	CustomLearner.validateSubclass(NoApp)


class LoveAtFirstSightClassifier(CustomLearner):
	""" Always predicts the value of the first class it ever sees """
	problemType = 'classification'
	def incrementalTrain(self, trainX, trainY):
		if hasattr(self, 'scope'):
			self.scope = numpy.intersect1d(self.scope, trainY.copyAs('numpyarray'))
		else:
			self.scope = numpy.unique(trainY.copyAs('numpyarray'))
		if not hasattr(self, 'prediction'):
			self.prediction = trainY[0,0]
	def apply(self, testX):
		ret = []
		for point in testX.pointIterator():
			ret.append([self.prediction])
		return UML.createData("Matrix", ret)
	def getScores(self, testX):
		ret = []
		for point in testX.pointIterator():
			currScores = []
			for value in self.scope:
				if value == self.prediction:
					currScores.append(1)
				else:
					currScores.append(0)
			ret.append(currScores)
		return UML.createData("Matrix", ret)

def testCustomLearnerGetScores():
	""" Test that a CustomLearner with getScores can actually call that method """
	data = [[1,3],[2,-5],[1,44]]
	labels = [[0],[2],[1]]

	trainObj = UML.createData('Matrix', data)
	labelsObj = UML.createData('Matrix', labels)

	tdata = [[23,2343],[23,22],[454,-44]]
	testObj = UML.createData('Matrix', tdata)

	UML.registerCustomLearner(LoveAtFirstSightClassifier)

	name = 'Custom.LoveAtFirstSightClassifier'
	preds = UML.trainAndApply(name, trainX=trainObj, trainY=labelsObj, testX=testObj, scoreMode='label')
	assert preds.pointCount == 3
	assert preds.featureCount == 1
	best = UML.trainAndApply(name, trainX=trainObj, trainY=labelsObj, testX=testObj, scoreMode='bestScore')
	assert best.pointCount == 3
	assert best.featureCount == 2
	allScores = UML.trainAndApply(name, trainX=trainObj, trainY=labelsObj, testX=testObj, scoreMode='allScores')
	assert allScores.pointCount == 3
	assert allScores.featureCount == 3


def testCustomLearnerIncTrainCheck():
	""" Test that a CustomLearner with incrementalTrain() but no train() works as expecte """
	data = [[1,3],[2,-5],[1,44]]
	labels = [[0],[2],[1]]

	trainObj = UML.createData('Matrix', data)
	labelsObj = UML.createData('Matrix', labels)

	tdata = [[23,2343],[23,22],[454,-44]]
	testObj = UML.createData('Matrix', tdata)

	UML.registerCustomLearner(LoveAtFirstSightClassifier)

	name = 'Custom.LoveAtFirstSightClassifier'
	tlObj = UML.train(name, trainX=trainObj, trainY=labelsObj, )

	prevAllScores = tlObj.apply(testX=testObj, scoreMode='allScores')
	tlObj.incrementalTrain()

	assert False
