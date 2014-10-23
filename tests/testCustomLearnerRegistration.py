
import numpy
import copy
from nose.tools import raises

import UML
from UML.exceptions import ArgumentException
from UML.customLearners import CustomLearner
from UML.customLearners.ridge_regression import RidgeRegression


class LoveAtFirstSightClassifier(CustomLearner):
	""" Always predicts the value of the first class it sees in the most recently trained data """
	learnerType = 'classification'
	
	def incrementalTrain(self, trainX, trainY):
		if hasattr(self, 'scope'):
			self.scope = numpy.union1d(self.scope, trainY.copyAs('numpyarray').flatten())
		else:
			self.scope = numpy.unique(trainY.copyAs('numpyarray'))
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

class UncallableLearner(CustomLearner):
	learnerType = 'classification'
	
	def train(self, trainX, trainY, foo):
		return None
	
	def apply(self, testX):
		return None


@raises(ArgumentException)
def testCustomPackageNameCollision():
	""" Test registerCustomLearner raises an exception when the given name collides with a real package """
	avail = UML.interfaces.available
	nonCustom = None
	for inter in avail:
		if not isinstance(inter, UML.interfaces.CustomLearnerInterface):
			nonCustom = inter
			break
	if nonCustom is not None:
		UML.registerCustomLearner(nonCustom.getCanonicalName(), LoveAtFirstSightClassifier)
	else:
		raise ArgumentException("Can't test, just pass")


@raises(ArgumentException)
def testDeregisterBogusPackageName():
	""" Test deregisterCustomLearner raises an exception when passed a bogus customPackageName """
	UML.deregisterCustomLearner("BozoPackage", 'ClownClassifier')


@raises(ArgumentException)
def testDeregisterFromNonCustom():
	""" Test deregisterCustomLearner raises an exception when trying to deregister from a non-custom interface """
	UML.deregisterCustomLearner("Mlpy", 'KNN')


@raises(ArgumentException)
def testDeregisterBogusLearnerName():
	""" Test deregisterCustomLearner raises an exception when passed a bogus learnerName """
	#save state before registration, to check against final state
	saved = copy.copy(UML.interfaces.available)
	try:
		UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
		UML.deregisterCustomLearner("Foo", 'BogusNameLearner')
		# should never get to here
		assert False
	finally:
		UML.deregisterCustomLearner("Foo", "LoveAtFirstSightClassifier")
		assert saved == UML.interfaces.available


def testMultipleCustomPackages():
	""" Test registerCustomLearner correctly instantiates multiple custom packages """

	#save state before registration, to check against final state
	saved = copy.copy(UML.interfaces.available)

	UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
	UML.registerCustomLearner("Bar", RidgeRegression)
	UML.registerCustomLearner("Baz", UncallableLearner)

	assert UML.listLearners("Foo") == ["LoveAtFirstSightClassifier"]
	assert UML.listLearners("Bar") == ["RidgeRegression"]
	assert UML.listLearners("Baz") == ["UncallableLearner"] 

	assert UML.learnerParameters("Foo.LoveAtFirstSightClassifier") == [[]]
	assert UML.learnerParameters("Bar.RidgeRegression") == [['lamb']]
	assert UML.learnerParameters("Baz.UncallableLearner") == [['foo']]

	UML.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
	UML.deregisterCustomLearner("Bar", 'RidgeRegression')
	UML.deregisterCustomLearner("Baz", 'UncallableLearner')

	assert saved == UML.interfaces.available


def testMultipleLearnersSinglePackage():
	#TODO test is not isolated from above.

	#save state before registration, to check against final state
	saved = copy.copy(UML.interfaces.available)

	UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
	UML.registerCustomLearner("Foo", RidgeRegression)
	UML.registerCustomLearner("Foo", UncallableLearner)

	learners = UML.listLearners("Foo")
	assert len(learners) == 3
	assert "LoveAtFirstSightClassifier" in learners
	assert "RidgeRegression" in learners
	assert "UncallableLearner" in learners

	assert UML.learnerParameters("Foo.LoveAtFirstSightClassifier") == [[]]
	assert UML.learnerParameters("Foo.RidgeRegression") == [['lamb']]
	assert UML.learnerParameters("Foo.UncallableLearner") == [['foo']]


	UML.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')

	learners = UML.listLearners("Foo")
	assert len(learners) == 2
	assert "RidgeRegression" in learners
	assert "UncallableLearner" in learners

	UML.deregisterCustomLearner("Foo", 'RidgeRegression')
	UML.deregisterCustomLearner("Foo", 'UncallableLearner')

	assert saved == UML.interfaces.available
