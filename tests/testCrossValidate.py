"""Tests for the user facing functions for cross validation and
the backend helpers they rely on.

"""

import sys
import numpy

import nose
from nose.plugins.attrib import attr

import UML

from UML import crossValidate
from UML import crossValidateReturnAll
from UML import crossValidateReturnBest
from UML import createData

from UML.calculate import *
from UML.randomness import pythonRandom
from UML.helpers import computeMetrics




def _randomLabeledDataSet(dataType='Matrix', numPoints=50, numFeatures=5, numLabels=3):
	"""returns a tuple of two data objects of type dataType
	the first object in the tuple contains the feature information ('X' in UML language)
	the second object in the tuple contains the labels for each feature ('Y' in UML language)
	"""
	if numLabels is None:
		labelsRaw = [[pythonRandom.random()] for _x in xrange(numPoints)]
	else:  # labels data set
		labelsRaw = [[int(pythonRandom.random()*numLabels)] for _x in xrange(numPoints)]

	rawFeatures = [[pythonRandom.random() for _x in xrange(numFeatures)] for _y in xrange(numPoints)]

	return (createData(dataType, rawFeatures), createData(dataType, labelsRaw))

def test_crossValidate_XY_unchanged():
	"""assert that after running cross validate on datasets passed to 
	X and Y, the original data is unchanged"""
	classifierAlgo = 'Custom.KNNClassifier'
	X, Y = _randomLabeledDataSet(numLabels=5)
	copyX = X.copy()
	copyY = Y.copy()
	result = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
	assert X.hashCode() == copyX.hashCode()
	assert Y.hashCode() == copyY.hashCode()



def test_crossValidate_runs():
	"""tests that crossValidate gives results (in form of float) 
	for different algorithms
	and different data types (children of base)
	"""
	#just scrap data to make sure it doesn't crash
	numLabelsInSet = 3
	numPointsInSet = 50
	#todo add other data types - currently crashes in sklearn interface for list and sparse
	for dType in ['Matrix',]:
		X, Y = _randomLabeledDataSet(numPoints=numPointsInSet, numLabels=numLabelsInSet, dataType=dType)	
		classifierAlgos = ['Custom.KNNClassifier']
		for curAlgo in classifierAlgos:
			result = crossValidate(curAlgo, X, Y, fractionIncorrect, {}, numFolds=3)
			assert isinstance(result, float)


		#With regression dataset (no repeated labels)
		X, Y = _randomLabeledDataSet(numLabels=None, dataType=dType)	
		classifierAlgos = ['Custom.RidgeRegression']
		for curAlgo in classifierAlgos:
			result = crossValidate(curAlgo, X, Y, meanAbsoluteError, {}, numFolds=3)
			assert isinstance(result, float)
		

def _assertClassifierErrorOnRandomDataPlausible(actualError, numLabels, tolerance=.1):
	"""assert the actual error on a labeled data set (for a classifier)
	is plausible, given the number of (evenly distributed) labels in hte data set
	"""
	idealFractionIncorrect = 1.0 - 1.0/numLabels
	error = abs(actualError - idealFractionIncorrect)
	assert error <= tolerance


def test_crossValidate_reasonable_results():
	"""Assert that crossValidate returns reasonable errors for known algorithms
	on cooked data sets:
	crossValidate should do the following:
	classifiers:
		have no error when there is only one label in the data set
		classify random data at roughly the accuracy of 1/numLabels
	regressors:
		LinearRegression - have no error when the dataset all lies on one plane
	"""
	classifierAlgo = 'Custom.KNNClassifier'
	#assert that when whole dataset has the same label, crossValidated score 
	#reflects 100% accruacy (with a classifier)
	X, Y = _randomLabeledDataSet(numLabels=1)
	result = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
	assert result < 0.000001  # 0 incorrect ever

	#assert that a random dataset will have accuracy roughly equal to 1/numLabels
	numLabelsList = [2,3,5]
	for curNumLabels in numLabelsList:
		X, Y = _randomLabeledDataSet(numPoints=50, numLabels=curNumLabels)
		result = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
		_assertClassifierErrorOnRandomDataPlausible(result, curNumLabels, tolerance=(1.0/curNumLabels))

	#assert that for an easy dataset (no noise, overdetermined linear hyperplane!), 
	#crossValidated error is perfect 
	#for all folds, with simple LinearRegression
	regressionAlgo = 'Custom.RidgeRegression'

	#make random data set where all points lie on a linear hyperplane
	numFeats = 3
	numPoints = 50
	points = [[pythonRandom.gauss(0,1) for _x in xrange(numFeats)] for _y in xrange(numPoints)]
	labels = [[sum(featVector)] for featVector in points]
	X = createData('Matrix', points)
	Y = createData('Matrix', labels)
	
	#run in crossValidate
	result = crossValidate(regressionAlgo, X, Y, meanAbsoluteError, {}, numFolds=5)
	#assert error essentially zero since there's no noise
	assert result < .001 

@attr('slow')
@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def test_crossValidateShuffleSeed():
	"""Assert that for a dataset, the same algorithm will generate the same model 
	(and have the same accuracy) when presented with identical random state (and
	therefore identical folds).
	Assert that the model is different when the random state is different
	"""
	numTrials = 5
	for _ in xrange(numTrials):
		X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
		seed = UML.randomness.pythonRandom.randint(0, sys.maxint)
		UML.setRandomSeed(seed)
		resultOne = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, {}, numFolds=3)
		UML.setRandomSeed(seed)
		resultTwo = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, {}, numFolds=3)
		assert resultOne == resultTwo

	resultThree = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, {}, numFolds=3)
	#assert that models have diffeerent errors when different random state is available.
	#the idea being that different seeds create different folds
	#which create different models, which create different accuracies
	#for sufficiently large datasets.
	assert resultOne != resultThree


def _randomLabeledDataSet(dataType='Matrix', numPoints=100, numFeatures=5, numLabels=3):
	"""returns a tuple of two data objects of type dataType
	the first object in the tuple contains the feature information ('X' in UML language)
	the second object in the tuple contains the labels for each feature ('Y' in UML language)
	"""
	if numLabels is None:
		labelsRaw = [[pythonRandom.random()] for _x in xrange(numPoints)]
	else:  # labels data set
		labelsRaw = [[int(pythonRandom.random()*numLabels)] for _x in xrange(numPoints)]

	rawFeatures = [[pythonRandom.random() for _x in xrange(numFeatures)] for _y in xrange(numPoints)]

	return (createData(dataType, rawFeatures), createData(dataType, labelsRaw))


@attr('slow')
@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def test_crossValidateReturnAll():
	"""assert that KNeighborsClassifier generates results with default arguments
	assert that having the same function arguments yields the same results.
	assert that return all gives a cross validated performance for all of its 
	parameter permutations
	"""
	X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
	#try with no extra arguments at all:
	result = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect)
	assert result
	assert 1 == len(result)
	assert result[0][0] == {}
	#try with some extra elements but all after default
	result = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, k=(1,2,3))
	assert result
	assert 3 == len(result)

	#since the same seed is used, and these calls are effectively building the same arguments, (p=2 is default for algo)
	#the scores in results list should be the same (though the keys will be different (one the second will have 'p':2 in the keys as well))
	seed = UML.randomness.pythonRandom.randint(0, sys.maxint)
	UML.setRandomSeed(seed)
	resultDifferentNeighbors = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, k=(1,2,3,4,5))
	UML.setRandomSeed(seed)
	resultDifferentNeighborsButSameCombinations = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, k=(1,2,3,4,5))
	#assert the the resulting SCORES are identical
	#uncertain about the order
	resultOneScores = [curEntry[1] for curEntry in resultDifferentNeighbors]
	resultTwoScores = [curEntry[1] for curEntry in resultDifferentNeighborsButSameCombinations]
	resultsOneSet = set(resultOneScores)
	resultsTwoSet = set(resultTwoScores)
	assert resultsOneSet == resultsTwoSet

	#assert results have the expected data structure:
	#a list of tuples where the first entry is the argument dict
	#and second entry is the score (float)
	assert isinstance(resultDifferentNeighbors, list)
	for curResult in resultDifferentNeighbors:
		assert isinstance(curResult, tuple)
		assert isinstance(curResult[0], dict)
		assert isinstance(curResult[1], float)


@attr('slow')
@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def test_crossValidateReturnBest():
	"""test that the 'best' ie fittest argument combination is chosen.
	test that best tuple is in the 'all' list of tuples.
	"""
	#assert that it returns the best, enforce a seed?
	X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
	#try with no extra arguments at all:
	shouldMaximizeScores = False

	# want to have a predictable random state in order to control 
	seed = UML.randomness.pythonRandom.randint(0, sys.maxint)
	UML.setRandomSeed(seed)
	resultTuple = crossValidateReturnBest('Custom.KNNClassifier', X, Y, fractionIncorrect, maximize=shouldMaximizeScores, k=(1,2,3))
	assert resultTuple

	UML.setRandomSeed(seed)
	allResultsList = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, k=(1,2,3))
	#since same args were used (except return all doesn't have a 'maximize' parameter,
	# the best tuple should be in allResultsList
	allArguments = [curResult[0] for curResult in allResultsList]
	allScores = [curResult[1] for curResult in allResultsList]
	assert resultTuple[0] in allArguments
	assert resultTuple[1] in allScores

	#crudely verify that resultTuple was in fact the best in allResultsList
	for curError in allScores:
		#assert that the error is not 'better' than our best error:
		if shouldMaximizeScores:
			assert curError <= resultTuple[1]
		else:
			assert curError >= resultTuple[1]

def test_crossValidateReturnEtc_withDefaultArgs():
	"""Assert that return best and return all work with default arguments as predicted
	ie generating scores for '{}' as the arguments
	"""
	X, Y = _randomLabeledDataSet(numPoints=20, numFeatures=5, numLabels=5)
	#run with default arguments
	bestTuple = crossValidateReturnBest('Custom.KNNClassifier', X, Y, fractionIncorrect, )
	assert bestTuple
	assert isinstance(bestTuple, tuple)
	assert bestTuple[0] == {}
	#run return all with default arguments
	allResultsList = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, )
	assert allResultsList
	assert 1 == len(allResultsList)
	assert allResultsList[0][0] == {}
