"""
Wrapper module, allowing for easy access to the major functionality of the package that
is static.

"""

import numpy

from UML import run
from UML import data
from .utility import ArgumentException


# run() with a return type of the predicted labels added back into the object?


def loadTrainingAndTesting(fileName, labelID, fractionForTestSet, fileType, loadType="DenseMatrixData"):
	"""this is a helpful function that makes it easy to do the common task of loading a dataset and splitting it into training and testing sets.
	It returns training X, training Y, testing X and testing Y"""
	trainX = data(loadType, fileName, fileType=fileType)
	testX = trainX.extractPoints(start=0, end=trainX.points(), number=int(round(fractionForTestSet*trainX.points())), randomize=True)	#pull out a testing set
	trainY = trainX.extractFeatures(labelID)	#construct the column vector of training labels
	testY = testX.extractFeatures(labelID)	#construct the column vector of testing labels
	return trainX, trainY, testX, testY



def normalize(algorithm, trainData, testData=None, dependentVar=None, arguments={}, mode=True):
	"""
	Calls on the functionality of a package to train on some data and then modify both
	the training data and a set of test data accroding to the produced model.


	"""
	# single call normalize, combined data
	if mode and testData is not None:
		testLength = testData.points()
		# glue training data at the end of test data
		testData.appendPoints(trainData)
		try:
			normalizedAll = run(algorithm, trainData, testData, dependentVar=dependentVar, arguments=arguments)
		except ArgumentException:
			testData.extractPoints(start=testLength, end=normalizedAll.points())
		# resplit normalized
		normalizedTrain = normalizedAll.extractPoints(start=testLength, end=normalizedAll.points())
		normalizedTest = normalizedAll
	# two call normalize, no data combination
	else:
		normalizedTrain = run(algorithm, trainData, trainData, dependentVar=dependentVar, arguments=arguments)
		if testData is not None:
			normalizedTest = run(algorithm, trainData, testData, dependentVar=dependentVar, arguments=arguments)
		
	# modify references for trainData and testData
	trainData.copyReferences(normalizedTrain)
	if testData is not None:
		testData.copyReferences(normalizedTest)


# def runWithPerformance()  # same as run, with extra parameter?

#combinations() -- maybe

#listAllAlgorithms()

