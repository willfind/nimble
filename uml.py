"""
Module containing most of the user facing functions for the top level uml import.

"""

import random
import numpy
import inspect
import operator
import re 
import datetime
import os

import UML
from UML.exceptions import ArgumentException

from UML.logging import UmlLogger
from UML.logging import LogManager
from UML.logging import Stopwatch

from UML.runners import run

from UML.umlHelpers import _loadSparse
from UML.umlHelpers import _loadMatrix
from UML.umlHelpers import _loadList
from UML.umlHelpers import executeCode
from UML.umlHelpers import _incrementTrialWindows


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
	testXSize = int(round(fractionForTestSet*toSplit.points()))
	#pull out a testing set
	testX = toSplit.extractPoints(start=0, end=toSplit.points(), number=testXSize, randomize=True)	
	trainY = None
	testY = None
	if labelID is not None:
		trainY = toSplit.extractFeatures(labelID)	#construct the column vector of training labels
		testY = testX.extractFeatures(labelID)	#construct the column vector of testing labels
	return toSplit, trainY, testX, testY



def normalizeData(algorithm, trainData, testData=None, dependentVar=None, arguments={}, mode=True):
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
			normalizedAll = run(algorithm, trainData, dependentVar, testData, arguments=arguments)
		except ArgumentException:
			testData.extractPoints(start=testLength, end=normalizedAll.points())
		# resplit normalized
		normalizedTrain = normalizedAll.extractPoints(start=testLength, end=normalizedAll.points())
		normalizedTest = normalizedAll
	# two call normalize, no data combination
	else:
		normalizedTrain = run(algorithm, trainData, dependentVar, trainData, arguments=arguments)
		if testData is not None:
			normalizedTest = run(algorithm, trainData, dependentVar, testData, arguments=arguments)
		
	# modify references for trainData and testData
	trainData.referenceDataFrom(normalizedTrain)
	if testData is not None:
		testData.referenceDataFrom(normalizedTest)


def listLearningFunctions(package):
	package = package.lower()
	results = None
	if package == 'mahout':
		import UML.interfaces.mahout_interface
		results = UML.interfaces.mahout_interface.listMahoutAlgorithms()
	elif package == 'regressor':
		import UML.interfaces.regressors_interface
		results = UML.interfaces.regressors_interface.listRegressorAlgorithms()
	elif package == 'scikitlearn':
		import UML.interfaces.scikit_learn_interface
		results = UML.interfaces.scikit_learn_interface.listSciKitLearnAlgorithms()
	elif package == 'mlpy':
		import UML.interfaces.mlpy_interface
		results = UML.interfaces.mlpy_interface.listMlpyAlgorithms()
	elif package == 'shogun':
		import UML.interfaces.shogun_interface
		results = UML.interfaces.shogun_interface.listShogunAlgorithms()

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
	# if data is not a path to a file, then we don't care about the value of this flag;
	# instead we use it as an indicator flag to directly instantiate
	else:
		fileType = None

	# these should be lowercase to avoid ambiguity
	retType = retType.lower()
	sparseAlias = ["sparse"]
	matrixAlias = ['matrix']
	listAlias = ["list"]
	if retType in sparseAlias:
		ret = _loadSparse(data, featureNames, fileType)
	elif retType in matrixAlias:
		ret = _loadMatrix(data, featureNames, fileType)
	elif retType in listAlias:
		ret = _loadList(data, featureNames, fileType)
	else:
		msg = "Unknown data type, cannot instantiate. Only allowable inputs: "
		msg += "'List' for data in python lists, 'Matrix' for a numpy matrix, "
		msg += "and 'Sparse' for a scipy sparse coo_matrix"
		raise ArgumentException(msg)

	if name is not None:
		ret.nameData(name)
	return ret



def functionCombinations(functionText):
	"""process the text of the given python code, returning a list of different versions of the text. Any occurence of, say, <41|6|3>
		generate three different versions of the code, one with 41, one with 6, one with 3. This notation <...|...|...> can have any number of vertical bars
		and can be used anywhere in the text. The ... portions can be anything."""
	resultsSoFar = [functionText]	#as we process the <...|...|...> patterns one at a time, we keep adding all versions of the text so far to this list.
	while True:	#we'll keep searching the text until we can't find a <...|...|...> pattern
		newResults = []	#our next round of results that we'll generate in a moment
		done = False
		for text in resultsSoFar:
			result = re.search("(<[^\|]+(?:\|[^\|>]+)*>)", text) #an incomprehensable regular expression that searches for strings of the form <...|...|...>
			if result == None: #no more occurences, so we're done
				done = True
				break
			start = result.start()+1
			end = result.end()-1
			pieces = text[start:end].split("|")
			for piece in pieces:
				newText = text[:start-1] + str(piece) + text[end+1:]
				newResults.append(newText)
		if not done: resultsSoFar = newResults
		if done: break
	return resultsSoFar



def orderedCrossValidateReturnBest(X, Y, functionsToApply, mode, orderedFeature, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize, extraParams={}):
	"""
	Runs ordered cross validation on the functions whose text is in the list functionsToApply, and
	returns the text of the best (of averaged resutls) performer together with its performance
	"""
	if mode == 'min':
		minimize = True
	elif mode == 'max':
		minimize = False
	else:
		raise Exception("mode must be either 'min' or 'max' depending on the desired solution")
	resultsHash = orderedCrossValidateAveragedResults(X,Y, functionsToApply, orderedFeature, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize, extraParams)
	if minimize: bestPerformance = float('inf')
	else: bestPerformance = float('-inf')
	bestFuncText = None
	for functionText in resultsHash.keys():
		performance = resultsHash[functionText]
		if (minimize and performance < bestPerformance) or (not minimize and performance > bestPerformance): #if it's the best we've seen so far
				bestPerformance = performance
				bestFuncText = functionText
	return bestFuncText, bestPerformance

def orderedCrossValidateAveragedResults(X, Y, functionsToApply, orderedFeature, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize, extraParams={}):
	results = orderedCrossValidate(X, Y, functionsToApply, orderedFeature, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize, extraParams, False)
	ret = {}
	for function in results:
		sumResults = 0
		numResults = 0
		for value in results[function]:
			sumResults += value
			numResults += 1
		ret[function] = (float(sumResults)/float(numResults))

	return ret


def orderedCrossValidate(X, Y, functionsToApply, orderedFeature, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize, extraParams={}, extraInfo=True):
	"""
	Performs iterative testing of ordered subsets of the data for each function in the list
	functionsToApply, collecting the output of each into a dict that maps output to function text.
	It is important to note that the training and testing subdivisions of the data will never
	split points with the same value of the orderedFeature; 

	X: the data which will be subdivided into training and testing sets.

	Y: the labels corresponding to the data points in X

	functionsToApply: a list of functions which is called on each subdivision of the data

	orderedFeature: an ID of a feature in X over which the points of X will be sorted by. This
	defines the ordering which is maintained during the creation of subdivisions.

	minTrainSize: the minimum valid training size allowed by a subdivision of the data.

	maxTrainSize: the maximum valid training size allowed by a subdivision of the data. After
	the first subdivion, the training window will grow (roughly) in proportion to step size,
	tending towards maxTrainSize, but there is no guarantee that a trial with maxTrainSize will
	ever be reached.

	stepSize: roughly how far forward we shift the training and testing windows of each
	iterative subdivision. The training and testing windows are all defined off of the anchor
	point of the end of the training set. As long as there is more (high valued according to the
	ordered feature) data, step size is used to define the new end point of the training set:
	newTrainEndPoint = oldTrainEndPoint + stepSize (or an equivalent operation if a timedelta
	object is provided) though this endpoint may be shifted forward in order to not split
	data with the same value in the orderedFeature.

	gap: rouhly the distance between the end of the training window and the begining of the
	testing window.

	minTestSize: the minimum valid testing size allowed by a subdivision of the data. As
	large as possible testing sets are preferred, but as we approach the end of the data,
	the testing set size will shrink towards the minimum.

	maxTestSize: the maximum valid testing size allowed by a subdivision of the data. As
	long as there is enough data to do so, we will always construct maximum sized test sets.

	extraParams: name to value mapping of parameters used by functions in the functionsToApply list

	"""
	if not (isinstance(minTrainSize,int) and isinstance(maxTrainSize,int)):
		if not (isinstance(minTrainSize,datetime.timedelta) and isinstance(maxTrainSize,datetime.timedelta)):
			raise ArgumentException("minTrainSize and maxTrainSize must be specified using the same type")
	if not (isinstance(minTestSize,int) and isinstance(maxTestSize,int)):
		if not (isinstance(minTestSize,datetime.timedelta) and isinstance(maxTestSize,datetime.timedelta)):
			raise ArgumentException("minTestSize and maxTestSize must be specified using the same type")

	# we have to combine all the data into a new object before we sort
	allData = X.copy()
	allData.appendFeatures(Y)

	# gran the ID with which to split the data later
	labelName = Y.featureNamesInverse[0]

	# we sort the data according to the values of the 'orderedFeature' feature
	allData.sortPoints(sortBy=orderedFeature)

	tempResults = {}
	# populate the dict with empty lists to hold the results for each trial
	for function in functionsToApply:
		tempResults[function] = []

#	import pdb; pdb.set_trace()

	# the None input indicates this is the first call
	endPoints = _incrementTrialWindows(allData, orderedFeature, None, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize)

	# we need to be able to run at least one trial
	if endPoints is None:
		raise ArgumentException("Check input, couldn't run a trial")

	# the helper function will return None if it is impossible to construct trials out of the remaining
	# data
	while (endPoints is not None):
		# it's not None, so we can unpack it
		(startOfTrainSet, endOfTrainSet, startOfTestSet, endOfTestSet) = endPoints

		trainSet = allData.copyPoints(start=startOfTrainSet, end=endOfTrainSet)
		trainY = trainSet.extractFeatures(labelName)
		trainX = trainSet

		testSet = allData.copyPoints(start=startOfTestSet, end=endOfTestSet)
		testY = testSet.extractFeatures(labelName)
		testX = testSet

		# we have our train and test data, we call each of the given functions
		for function in functionsToApply:
			dataHash = extraParams
			dataHash["trainX"] = trainX; dataHash["testX"] = testX
			dataHash["trainY"] = trainY; dataHash["testY"] = testY
			currResult = executeCode(function, dataHash)
			if extraInfo:
				currEntry = {}
				currEntry["result"] = currResult
				currEntry["startTrain"] = startOfTrainSet
				currEntry["endTrain"] = endOfTrainSet
				currEntry["startTest"] = startOfTestSet
				currEntry["endTest"] = endOfTestSet
				tempResults[function].append(currEntry)
			else:
				tempResults[function].append(currResult)

		endPoints = _incrementTrialWindows(allData, orderedFeature, endOfTrainSet, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize)

	return tempResults




def crossValidate(X, Y, functionsToApply, numFolds=10, extraParams={}, sendToLog=True):
	"""applies crossValidation using numFolds folds, applying each function in the list functionsToApply (which are text of python functions)
	one by one. Assumes that functionsToApply is a list of the text functions, that use the variables trainX, trainY, testX, testY within
	their text."""
	aggregatedResults = {}
	if sendToLog:
		timerMap = {}
	for function in functionsToApply:
		curResults = []
		XIterator = X.foldIterator(numFolds=numFolds)	#assumes that the dataMatrix class has a function foldIterator that returns an iterator
														#each time you call .next() on this iterator, it gives you the next (trainData, testData)
														#pair, one for each fold. So for 10 fold cross validation, next works 10 times before failing.
		YIterator = Y.foldIterator(numFolds=numFolds)
		
		#if we are loggin this, create a timer
		if sendToLog:
				timer = Stopwatch()
				timerMap[function] = timer

		while True: #need to add a test here for when iterator .next() is done
			try:
				curTrainX, curTestX = XIterator.next()
				curTrainY, curTestY = YIterator.next()
			except StopIteration:	#once we've gone through all the folds, this exception gets thrown and we're done!
					break
			dataHash = extraParams
			dataHash["trainX"] = curTrainX; dataHash["testX"] = curTestX	#assumes that the function text in functionsToApply uses these variables
			dataHash["trainY"] = curTrainY; dataHash["testY"] = curTestY

			if sendToLog:
				timer.start('learning')

			curResults.append(executeCode(function, dataHash))

			if sendToLog:
				timer.stop('learning')

		if sendToLog:
			timer.cumulativeTimes['learning'] /= numFolds

		# average across all folds
		avg = 0.
		denom = 0.
		for result in curResults:
			for v in result.values():
				avg += v
				denom += 1
		aggregatedResults[function] = avg/denom #NOTE: this could be bad if the sets have different size!!

		if sendToLog:
			logger = LogManager()
			sortedResults = sorted(aggregatedResults.iteritems(), key=operator.itemgetter(1))
			for result in sortedResults:
				logger.logRun(X, None, function, {function:aggregatedResults[function]}, timer, numFolds=numFolds)
	return aggregatedResults


def crossValidateReturnBest(X, Y, functionsToApply, mode, numFolds=10, extraParams={}, sendToLog=True):
	"""runs cross validation on the functions whose text is in the list functionsToApply, and returns the text of the best performer together with
	its performance"""
	if mode == 'min':
		minimize = True
	elif mode == 'max':
		minimize = False
	else:
		raise Exception("mode must be either 'min' or 'max' depending on the desired solution")

	resultsHash = crossValidate(X,Y, functionsToApply=functionsToApply, numFolds=numFolds, extraParams=extraParams, sendToLog=sendToLog)

	if minimize: bestPerformance = float('inf')
	else: bestPerformance = float('-inf')
	bestFuncText = None
	for functionText in resultsHash.keys():
		performance = resultsHash[functionText]
		if (minimize and performance < bestPerformance) or (not minimize and performance > bestPerformance): #if it's the best we've seen so far
				bestPerformance = performance
				bestFuncText = functionText

	return bestFuncText, bestPerformance






