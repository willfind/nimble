"""
Integration tests to demonstrate consistency between output of different methods
of a single interface. All tests are general, testing knowledge guaranteed by
the UniversalInterface api.

"""

from nose.tools import raises

import UML

from UML.exceptions import ArgumentException
from UML.interfaces.universal_interface import UniversalInterface
from UML.umlHelpers import generateClusteredPoints

def generateClassificationData(labels):
	"""
	Randomly generate sensible data for a classification problem. Returns a tuple of tuples,
	where the first value is a tuple containing (trainX, trainY) and the second value is
	a tuple containing (testX ,testY)

	"""
	clusterCount = labels
	pointsPer = 10
	featuresPer = 5

	#add noise to the features only
	trainData, trainLabels, noiselessTrainLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer, addFeatureNoise=True, addLabelNoise=False, addLabelColumn=False)
	testData, testLabels, noiselessTestLabels = generateClusteredPoints(clusterCount, 1, featuresPer, addFeatureNoise=True, addLabelNoise=False, addLabelColumn=False)

	return ((trainData, noiselessTrainLabels), (testData, noiselessTestLabels))

def checkFormat(scores, numLabels):
	"""
	Check that the provided UML data typed scores structurally match either a one vs
	one or a one vs all formatting scheme.

	"""
	if scores.featureCount != numLabels and scores.featureCount != (numLabels * (numLabels-1))/2:
		raise RuntimeError("_getScores() must return scores that are either One vs One or One vs All formatted")


def checkFormatRaw(scores, numLabels):
	"""
	Check that the provided numpy typed scores structurally match either a one vs
	one or a one vs all formatting scheme.

	"""
	if scores.shape[1] != numLabels and scores.shape[1] != (numLabels * (numLabels-1))/2:
		raise RuntimeError("_getScores() must return scores that are either One vs One or One vs All formatted")

def test__getScoresFormat():
	"""
	Automatically checks the _getScores() format for as many classifiers we can identify in each
	interface.
	"""
	data2 = generateClassificationData(2)
	((trainX2, trainY2), (testX2, testY2)) = data2
	data4 = generateClassificationData(4)
	((trainX4, trainY4), (testX4, testY4)) = data4
	for interface in UML.interfaces.available:
		interfaceName = interface.getCanonicalName()

		if interfaceName == 'shogun': # TODO - remove
			continue

		learners = interface.listLearners()
		for lName in learners:
			fullName = interfaceName + '.' + lName
			if UML.learnerType(fullName) == 'classifier':
				try:
					tl2 = UML.train(fullName, trainX2, trainY2)
				except ArgumentException:
					# this is to catch learners that have required arguments.
					# we have to skip them in that case
					continue
				(ign1, ign2, transTestX2, ign3) = interface._inputTransformation(lName, None, None, testX2, {}, tl2.customDict)
				try:
					scores2 = interface._getScores(tl2.backend, transTestX2, {}, tl2.customDict)
				except ArgumentException:
					# this is to catch learners that cannot output scores
					continue
				checkFormatRaw(scores2, 2)

				try:
					tl4 = UML.train(fullName, trainX4, trainY4)
				except:
					# some classifiers are binary only
					continue
				(ign1, ign2, transTestX4, ign3) = interface._inputTransformation(lName, None, None, testX4, {}, tl4.customDict)
				scores4 = interface._getScores(tl4.backend, transTestX4, {}, tl4.customDict)
				checkFormatRaw(scores4, 4)

def testGetScoresFormat():
	"""
	Automatically checks the TrainedLearner getScores() format for as many classifiers we
	can identify in each interface
	
	"""
	data2 = generateClassificationData(2)
	((trainX2, trainY2), (testX2, testY2)) = data2
	data4 = generateClassificationData(4)
	((trainX4, trainY4), (testX4, testY4)) = data4
	for interface in UML.interfaces.available:
		interfaceName = interface.getCanonicalName()

		if interfaceName == 'shogun': # TODO - remove
			continue

		learners = interface.listLearners()
		for lName in learners:
			fullName = interfaceName + '.' + lName
			if UML.learnerType(fullName) == 'classifier':
				try:
					tl2 = UML.train(fullName, trainX2, trainY2)
				except ArgumentException:
					# this is to catch learners that have required arguments.
					# we have to skip them in that case
					continue
				try:
					scores2 = tl2.getScores(testX2)
				except ArgumentException:
					# this is to catch learners that cannot output scores
					continue
				checkFormat(scores2, 2)

				try:
					tl4 = UML.train(fullName, trainX4, trainY4)
				except:
					# some classifiers are binary only
					continue
				scores4 = tl4.getScores(testX4)
				checkFormat(scores4, 4)


# TODO
#def testGetParamsOverListLearners():
#def testGetParamDefaultsOverListLearners():


# comparison between UML.learnerType and interface learnerType
