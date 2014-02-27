"""
Module containing most of the user facing functions for the top level uml import.

"""

import random
import numpy
import scipy.sparse
import inspect
import operator
import re 
import datetime
import os

import UML
from UML.exceptions import ArgumentException

from UML.logger import UmlLogger
from UML.logger import LogManager
from UML.logger import Stopwatch

from UML.runners import run

from UML.umlHelpers import findBestInterface
from UML.umlHelpers import _loadSparse
from UML.umlHelpers import _loadMatrix
from UML.umlHelpers import _loadList
from UML.umlHelpers import executeCode
from UML.umlHelpers import _incrementTrialWindows
from UML.umlHelpers import _learnerQuery

from UML.umlHelpers import computeMetrics
from UML.umlHelpers import ArgumentIterator
from UML.data.dataHelpers import DEFAULT_SEED


UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def createRandomizedData(retType, numPoints, numFeatures, sparcity, numericType="int", featureNames=None, name=None):
	if numPoints < 1:
		raise ArgumentException("must specify a positive nonzero number of points")
	if numFeatures < 1:
		raise ArgumentException("must specify a positive nonzero number of features")
	if sparcity < 0 or sparcity >=1:
		raise ArgumentException("sparcity must be greater than zero and less than one")
	if numericType != "int" and numericType != "float":
		raise ArgumentException("numericType may only be 'int' or 'float'")

	randData = numpy.zeros((numPoints,numFeatures))
	for i in xrange(numPoints):
		for j in xrange(numFeatures):
			if random.random() > sparcity:
				if numericType == 'int':
					randData[i,j] = numpy.random.randint(1,100)
				else:
					randData[i,j] = numpy.random.rand()

	return createData(retType, data=randData, featureNames=featureNames, name=name)


def splitData(toSplit, fractionForTestSet, labelID=None):
	"""this is a helpful function that makes it easy to do the common task of loading a dataset and splitting it into training and testing sets.
	It returns training X, training Y, testing X and testing Y"""
	testXSize = int(round(fractionForTestSet*toSplit.pointCount))
	#pull out a testing set
	testX = toSplit.extractPoints(start=0, end=toSplit.pointCount, number=testXSize, randomize=True)	
	trainY = None
	testY = None
	if labelID is not None:
		trainY = toSplit.extractFeatures(labelID)	#construct the column vector of training labels
		testY = testX.extractFeatures(labelID)	#construct the column vector of testing labels
	return toSplit, trainY, testX, testY



def normalizeData(learnerName, trainX, trainY=None, testX=None, arguments={}, mode=True):
	"""
	Calls on the functionality of a package to train on some data and then modify both
	the training data and a set of test data accroding to the produced model.

	"""
	# single call normalize, combined data
	if mode and testX is not None:
		testLength = testX.pointCount
		# glue training data at the end of test data
		testX.appendPoints(trainX)
		try:
			normalizedAll = run(learnerName, trainX, trainY, testX, arguments=arguments)
		except ArgumentException:
			testX.extractPoints(start=testLength, end=normalizedAll.pointCount)
		# resplit normalized
		normalizedTrain = normalizedAll.extractPoints(start=testLength, end=normalizedAll.pointCount)
		normalizedTest = normalizedAll
	# two call normalize, no data combination
	else:
		normalizedTrain = run(learnerName, trainX, trainY, trainX, arguments=arguments)
		if testX is not None:
			normalizedTest = run(learnerName, trainX, trainY, testX, arguments=arguments)
		
	# modify references for trainX and testX
	trainX.referenceDataFrom(normalizedTrain)
	if testX is not None:
		testX.referenceDataFrom(normalizedTest)


def learnerParameters(name):
	"""
	Takes a string of the form 'package.learnerName' and returns a list of
	strings which are the names of the parameters when calling package.learnerName

	If the name cannot be found within the package, then an exception will be thrown.
	If the name is found, be for some reason we cannot determine what the parameters
	are, then we return None. Note that if we have determined that there are no
	parameters, we return an empty list. 

	"""
	return _learnerQuery(name, 'parameters')

def learnerDefaultValues(name):
	"""
	Takes a string of the form 'package.learnerName' and returns a returns a
	dict mapping of parameter names to their default values when calling
	package.learnerName

	If the name cannot be found within the package, then an exception will be thrown.
	If the name is found, be for some reason we cannot determine what the parameters
	are, then we return None. Note that if we have determined that there are no
	parameters, we return an empty dict. 

	"""
	return _learnerQuery(name, 'defaults')


def listLearners(package=None):
	"""
	Takes the name of a package, and returns a list of learners that are callable through that
	package's run() interface.

	"""
	results = []
	if package is None:
		for interface in UML.interfaces.available:
			packageName = interface.getCanonicalName()
			currResults = interface.listLearners()
			for learnerName in currResults:
				results.append(packageName + "." + learnerName)
	else:
		interface = findBestInterface(package)
		currResults = interface.listLearners()
		for learnerName in currResults:
			results.append(learnerName)

	return results


def listDataFunctions():
	methodList = dir(UML.data.Base)
	visibleMethodList = []
	for methodName in methodList:
		if not methodName.startswith('_'):
			visibleMethodList.append(methodName)

	ret = []
	for methodName in visibleMethodList:
		currMethod = eval("UML.data.Base." + methodName)
		(args, varargs, keywords, defaults) = inspect.getargspec(currMethod)

		retString = methodName + "("
		for i in xrange(0, len(args)):
			if i != 0:
				retString += ", "
			retString += args[i]
			if defaults is not None and i >= (len(args) - len(defaults)):
				retString += "=" + str(defaults[i - (len(args) - len(defaults))])
			
		# obliterate the last comma
		retString += ")"
		ret.append(retString)

	return ret


def listUMLFunctions():
	methodList = dir(UML)

	visibleMethodList = []
	for methodName in methodList:
		if not methodName.startswith('_'):
			visibleMethodList.append(methodName)

	ret = []
	for methodName in visibleMethodList:
		currMethod = eval("UML." + methodName)
		if "__call__" not in dir(currMethod):
			continue
		(args, varargs, keywords, defaults) = inspect.getargspec(currMethod)

		retString = methodName + "("
		for i in xrange(0, len(args)):
			if i != 0:
				retString += ", "
			retString += args[i]
			if defaults is not None and i >= (len(args) - len(defaults)):
				retString += "=" + str(defaults[i - (len(args) - len(defaults))])
			
		# obliterate the last comma
		retString += ")"
		ret.append(retString)

	return ret


def createData(retType, data=None, featureNames=None, fileType=None, name=None, sendToLog=True):
	automatedRetType = False
	# determine if its a file we have to read; we assume if its a string its a path
	if isinstance(data, basestring):
		#we may log this event
		if sendToLog is not None and sendToLog is not False:
			if isinstance(sendToLog, UmlLogger):
				logger = sendToLog
			else:
				logger = LogManager()
			#logger.logLoad(retType, data, name)
		# determine the extension, call the appropriate helper
		split = data.rsplit('.', 1)
		extension = None
		if len(split) > 1:
			extension = split[1].lower()
		# if fileType is specified, it takes priority, otherwise, use the extension
		if fileType is None:
			fileType = extension
		if fileType is None:
			raise ArgumentException("The file must be recognizable by extension, or a type must be specified")
		
		if retType is None:
			automatedRetType = True
			if fileType == 'csv':
				retType = 'Matrix'
			if fileType == 'mtx':
				retType = 'Sparse'

	# if data is not a path to a file, then we don't care about the value of this flag;
	# instead we use it as an indicator flag to directly instantiate
	else:
		fileType = None
		if retType is None:
			if isinstance(data, list):
				retType = 'List'
			if isinstance(data, numpy.matrix):
				retType = 'Matrix'
			if isinstance(data, numpy.array):
				retType = 'Matrix'
			if scipy.sparse.issparse(data):
				retType = 'Sparse'


	# these should be lowercase to avoid ambiguity
	retType = retType.lower()
	sparseAlias = ["sparse"]
	matrixAlias = ['matrix']
	listAlias = ["list"]
	if retType in sparseAlias:
		ret = _loadSparse(data, featureNames, fileType, automatedRetType)
	elif retType in matrixAlias:
		ret = _loadMatrix(data, featureNames, fileType, automatedRetType)
	elif retType in listAlias:
		ret = _loadList(data, featureNames, fileType)
	else:
		msg = "Unknown data type, cannot instantiate. Only allowable inputs: "
		msg += "'List' for data in python lists, 'Matrix' for a numpy matrix, "
		msg += "'Sparse' for a scipy sparse coo_matrix, and None for automated choice."
		raise ArgumentException(msg)

	if name is not None:
		ret.nameData(name)
	return ret


#todo add support for Y as an index in X
#todo add logging
#todo add seed specification support to UML.foldIterator() to avoid 
#using two harmonious iterators (methods of base) and zip()
def crossValidate(learningAlgorithm, X, Y, performanceFunction, argumentsForAlgorithm={}, numFolds=10, scoreMode='label', negativeLabel=None, sendToLog=False, foldSeed=DEFAULT_SEED):
	"""
	K-fold cross validation.
	Returns mean performance (float) across numFolds folds on a X Y.

	Parameters:

	learningAlgorithm (string) - UML compliant algorithm name in the form 
	'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

	X (UML.Base subclass) - points/features data

	Y (UML.Base subclass) - labels/data about points in X

	performanceFunction (function) - Look in UML.metrics for premade options.
	Function used by computeMetrics to generate a performance score for the run.
	function is of the form:
	def func(knownValues, predictedValues, negativeLabel).

	argumentsForAlgorithm (dict) - dictionary mapping argument names (strings)
	to their values. This parameter is sent to run()
	example: {'dimensions':5, 'k':5}

	numFolds (int) - the number of folds used in the cross validation. Can't
	exceed the number of points in X, Y

	scoreMode - used by computeMetrics

	negativeLabel - used by computeMetrics

	sendToLog (bool) - send results/timing to log

	foldSeed - seed used to generate the folds, if you want to ensure the same
	folds for two different sets of points, provided the data has the same
	number of points, using the same seed will generate the same folds.
	"""
	#using the same seed (to ensure idetical folds in X and Y and thus accurate
	#linking between according points) make iterators containing folds
	if not X.pointCount == Y.pointCount:
		#todo support indexing if Y is an index for X instead
		raise ArgumentException("X and Y must contain the same number of points.")
	foldedXIterator = X.foldIterator(numFolds, seed=foldSeed)
	foldedYIterator = Y.foldIterator(numFolds, seed=foldSeed)
	assert len(foldedXIterator.foldList) == len(foldedYIterator.foldList)

	performanceListOfFolds = []
	#for each fold get train and test sets
	for XFold, YFold in zip(foldedXIterator, foldedYIterator):
		
		curTrainX, curTestingX = XFold
		curTrainY, curTestingY = YFold

		#run algorithm on the folds' training and testing sets
		curRunResult = run(learningAlgorithm=learningAlgorithm, trainX=curTrainX, trainY=curTrainY, testX=curTestingX, arguments=argumentsForAlgorithm, scoreMode=scoreMode, sendToLog=sendToLog)
		#calculate error of prediction, according to performanceFunction
		curPerformance = computeMetrics(curTestingY, None, curRunResult, performanceFunction, negativeLabel)

		performanceListOfFolds.append(curPerformance)

	if len(performanceListOfFolds) == 0:
		raise(ZeroDivisionError("crossValidate tried to average performance of ZERO runs"))
		
	#else average score from each fold (works for one fold as well)
	averagePerformance = sum(performanceListOfFolds)/float(len(performanceListOfFolds))
	return averagePerformance

#todo improve docstring
def crossValidateReturnAll(learningAlgorithm, X, Y, performanceFunction, numFolds=10, scoreMode='label', negativeLabel=None, sendToLog=False, foldSeed=DEFAULT_SEED, **arguments):
	"""Returns a list of tuples, where every tuple contains
	a dict representing the argument sent to run, and
	a float represennting the cross validated error associated
	with that argument dict.
	example list element: ({'arg1':2, 'arg2':'max'}, 89.0000123)
	"""

	#get an iterator for the argumet combinations- iterator
	#handles case of arguments being {}
	argumentCombinationIterator = ArgumentIterator(arguments)

	performanceList = []
	#make sure that the folds are identical for all trials, so that no argument combination gets
	#a lucky/easy fold set
	commonFoldSeed = foldSeed

	for curArgumentCombination in argumentCombinationIterator:
		#calculate cross validated performance, given the current argument dict
		errorForArgument = crossValidate(learningAlgorithm, X, Y, performanceFunction, curArgumentCombination, numFolds, scoreMode, negativeLabel, sendToLog, foldSeed=commonFoldSeed)
		#store the tuple with the current argument and cross validated performance	
		performanceList.append((curArgumentCombination, errorForArgument))
	#return the list of tuples - tracking the performance of each argument
	return performanceList

#todo get T's code to sense if maximize or minimize
#todo improve docstring
def crossValidateReturnBest(learningAlgorithm, X, Y, performanceFunction, numFolds=10, scoreMode='label', negativeLabel=None, sendToLog=False, foldSeed=DEFAULT_SEED, maximize=False, **arguments):
	"""For each argument combination in arguments, crossValidateReturnBest runs crossValidate to compute
	and mean error for the argument combination. crossValidateReturnBest then 
	RETURNS the best argument and error as a tuple:
	(argument_as_dict, cross_validated_performance_float)
	"""
	resultsAll = crossValidateReturnAll(learningAlgorithm, X, Y, performanceFunction, numFolds, scoreMode, negativeLabel, sendToLog, foldSeed, **arguments)
	bestArgumentAndScoreTuple = None
	for curResultTuple in resultsAll:
		curArgument, curScore = curResultTuple
		#if curArgument is the first or best we've seen: 
		#store its details in bestArgumentAndScoreTuple
		if bestArgumentAndScoreTuple is None:
			bestArgumentAndScoreTuple = curResultTuple
		else:
			if (maximize and curScore > bestArgumentAndScoreTuple[1]) or ((not maximize) and curScore < bestArgumentAndScoreTuple[1]):
				bestArgumentAndScoreTuple = curResultTuple

	return bestArgumentAndScoreTuple





