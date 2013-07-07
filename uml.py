"""
Module containing most of the user facing functions for the top level uml import.

"""

import random
import numpy
import inspect
import operator
import re 
import datetime

import UML
from UML.exceptions import ArgumentException
from UML.uml_logging.uml_logger import UmlLogger
from UML.uml_logging.log_manager import LogManager
from UML.uml_logging.stopwatch import Stopwatch
from UML.data import Base


from UML.umlHelpers import computeMetrics
from UML.umlHelpers import _loadSparse
from UML.umlHelpers import _loadDense
from UML.umlHelpers import _loadList
from UML.umlHelpers import countWins
from UML.umlHelpers import extractWinningPredictionLabel
from UML.umlHelpers import extractWinningPredictionIndex
from UML.umlHelpers import extractWinningPredictionIndexAndScore
from UML.umlHelpers import extractConfidenceScores
from UML.umlHelpers import copyLabels
from UML.umlHelpers import executeCode
from UML.umlHelpers import _incrementTrialWindows
from UML.umlHelpers import _jumpBack
from UML.umlHelpers import _jumpForward
from UML.umlHelpers import _diffLessThan
from UML.umlHelpers import generateAllPairs







def randomizedData(retType, numPoints, numFeatures, sparcity, numericType="int", featureNames=None, name=None):
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

	return create(retType, data=randData, featureNames=featureNames, name=name)


def loadTrainingAndTesting(fileName, labelID, fractionForTestSet, fileType, loadType="DenseMatrixData"):
	"""this is a helpful function that makes it easy to do the common task of loading a dataset and splitting it into training and testing sets.
	It returns training X, training Y, testing X and testing Y"""
	trainX = create(loadType, fileName, fileType=fileType)
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


def listAlgorithms(package):
	package = package.lower()
	results = None
	if package == 'mahout':
		import UML.interfaces.mahout_interface
		results = UML.interfaces.mahout_interface.listAlgorithms()
	elif package == 'regressor':
		import UML.interfaces.regressors_interface
		results = UML.interfaces.regressors_interface.listAlgorithms()
	elif package == 'scikitlearn':
		import UML.interfaces.scikit_learn_interface
		results = UML.interfaces.scikit_learn_interface.listAlgorithms()
	elif package == 'mlpy':
		import UML.interfaces.mlpy_interface
		results = UML.interfaces.mlpy_interface.listAlgorithms()
	elif package == 'shogun':
		import UML.interfaces.shogun_interface
		results = UML.interfaces.shogun_interface.listAlgorithms()

	return results

def listDataRepresentationMethods():
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


def run(algorithm, trainData, testData, dependentVar=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', sendToLog=True):
	if scoreMode != 'label' and scoreMode != 'bestScore' and scoreMode != 'allScores':
		raise ArgumentException("scoreMode may only be 'label' 'bestScore' or 'allScores'")
	multiClassStrategy = multiClassStrategy.lower()
	if multiClassStrategy != 'default' and multiClassStrategy != 'OneVsAll'.lower() and multiClassStrategy != 'OneVsOne'.lower():
		raise ArgumentException("multiClassStrategy may only be 'default' 'OneVsAll' or 'OneVsOne'")
	if not isinstance(arguments, dict):
		raise ArgumentException("The 'arguments' parameter must be a dictionary")
	splitList = algorithm.split('.',1)
	if len(splitList) < 2:
		raise ArgumentException("The algorithm must be prefaced with the package name and a dot. Example:'mlpy.KNN'")
	package = splitList[0]
	algorithm = splitList[1]

	if sendToLog:
		timer = Stopwatch()
	else:
		timer = None

	if package == 'mahout':
		results = UML.interfaces.mahout(algorithm, trainData, testData, dependentVar, arguments, output, timer)
	elif package == 'regressor':
		results = UML.interfaces.regressor(algorithm, trainData, testData, dependentVar, arguments, output, timer)
	elif package == 'sciKitLearn':
		results = UML.interfaces.sciKitLearn(algorithm, trainData, testData, dependentVar, arguments, output, scoreMode, multiClassStrategy, timer)
	elif package == 'mlpy':
		results = UML.interfaces.mlpy(algorithm, trainData, testData, dependentVar, arguments, output, scoreMode, multiClassStrategy, timer)
	elif package == 'shogun':
		results = UML.interfaces.shogun(algorithm, trainData, testData, dependentVar, arguments, output, scoreMode, multiClassStrategy, timer)
	elif package == 'self':
		raise ArgumentException("self modification not yet implemented")
	else:
		raise ArgumentException("package not recognized")

	if sendToLog:
			logManager = LogManager()
			if package == 'regressor':
				funcString = 'regressors.' + algorithm
			else:
				funcString = package + '.' + algorithm
			logManager.logRun(trainData, testData, funcString, None, timer, extraInfo=arguments)

	return results


def create(retType, data=None, featureNames=None, fileType=None, name=None, sendToLog=True):
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
	denseAlias = ["densematrixdata", 'dmd', 'dense']
	listAlias = ["list"]
	if retType in sparseAlias:
		ret = _loadSparse(data, featureNames, fileType)
	elif retType in denseAlias:
		ret = _loadDense(data, featureNames, fileType)
	elif retType in listAlias:
		ret = _loadList(data, featureNames, fileType)
	else:
		raise ArgumentException("Unknown data type, cannot instantiate")

	if name is not None:
		ret.setName(name)
	return ret



def runAndTest(algorithm, trainX, testX, trainDependentVar, testDependentVar, arguments, performanceMetricFuncs, scoreMode='label', negativeLabel=None, sendToLog=True):
	"""
		Calls on run() to train and evaluate the learning algorithm defined in 'algorithm,'
		then tests its performance using the metric function(s) found in
		performanceMetricFunctions

		algorithm: training algorithm to be called, in the form 'package.algorithmName'.

		trainX: data set to be used for training (as some form of Base object)

		testX: data set to be used for testing (as some form of Base object)
		
		trainDependentVar: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (as a Base object) or an index (numerical or string) 
		that defines their locale in the trainX object
		
		testDependentVar: used to retreive the known class labels of the test data. Either
		contains the labels themselves (as a Base object) or an index (numerical or string) 
		that defines their locale in the testX object.  If left blank, runAndTest() assumes 
		that testDependentVar is the same as trainDependentVar.
		
		arguments: optional arguments to be passed to the function specified by 'algorithm'
		
		negativeLabel: Argument required if performanceMetricFuncs contains proportionPercentPositive90
		or proportionPercentPositive50.  Identifies the 'negative' label in the data set.  Only
		applies to data sets with 2 class labels.
		
		sendToLog: optional boolean valued parameter; True meaning the results should be logged
	"""
	#Need to make copies of all data, in case it will be modified before a classifier is trained
	trainX = trainX.duplicate()
	testX = testX.duplicate()
	
	#if testDependentVar is empty, attempt to use trainDependentVar
	if testDependentVar is None and isinstance(trainDependentVar, (str, unicode, int)):
		testDependentVar = trainDependentVar

	trainDependentVar = copyLabels(trainX, trainDependentVar)
	testDependentVar = copyLabels(testX, testDependentVar)

	#if we are logging this run, we need to start the timer
	if sendToLog:
		timer = Stopwatch()
		timer.start('train')

	#rawResults contains predictions for each version of a learning function in the combos list
	rawResult = run(algorithm, trainX, testX, dependentVar=trainDependentVar, arguments=arguments, scoreMode=scoreMode, sendToLog=False)

	#if we are logging this run, we need to stop the timer
	if sendToLog:
		timer.stop('train')
		timer.start('errorComputation')

	#now we need to compute performance metric(s) for all prediction sets
	results = computeMetrics(testDependentVar, None, rawResult, performanceMetricFuncs, negativeLabel)

	if sendToLog:
		timer.stop('errorComputation')

	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results

def runAndTestOneVsOne(algorithm, trainX, testX, trainDependentVar, testDependentVar=None, arguments={}, performanceMetricFuncs=None, negativeLabel=None, sendToLog=True):
	"""
		Wrapper class for runOneVsOne.  Useful if you want the entire process of training,
		testing, and computing performance measures to be handled.  Takes in a learning algorithm
		and training and testing data sets, trains a learner, passes the test data to the 
		computed model, gets results, and calculates performance based on those results.  

		Arguments:

			algorithm: training algorithm to be called, in the form 'package.algorithmName'.

			trainX: data set to be used for training (as some form of Base object)
		
			testX: data set to be used for testing (as some form of Base object)
		
			trainDependentVar: used to retrieve the known class labels of the traing data. Either
			contains the labels themselves (in a Base object of the same type as trainX) 
			or an index (numerical or string) that defines their locale in the trainX object.
		
			testDependentVar: used to retreive the known class labels of the test data. Either
			contains the labels themselves or an index (numerical or string) that defines their locale
			in the testX object.  If not present, it is assumed that testDependentVar is the same
			as trainDependentVar.  
			
			arguments: optional arguments to be passed to the function specified by 'algorithm'

			performanceMetricFuncs: iterable collection of functions that can take two collections
			of corresponding labels - one of true labels, one of predicted labels - and return a
			performance metric.
		
			sendToLog: optional boolean valued parameter; True meaning the results should be printed 
			to log file.

		Returns: A dictionary associating the name or code of performance metrics with the results
		of those metrics, computed using the predictions of 'algorithm' on testX.  
		Example: { 'classificationError': 0.21, 'numCorrect': 1020 }
	"""
	if sendToLog:
		timer = Stopwatch()
	else:
		timer = None

	if testDependentVar is None:
		if not isinstance(trainDependentVar, (str, int, long)):
			raise ArgumentException("testDependentVar is missing in runOneVsOne")
		else:
			testDependentVar = testX.extractFeatures([trainDependentVar])
	else:
		if isinstance(testDependentVar, (str, int, long)):
			testDependentVar = testX.extractFeatures([testDependentVar])

	predictions = runOneVsOne(algorithm, trainX, testX, trainDependentVar, testDependentVar, arguments, scoreMode='label', sendToLog=False, timer=timer)

	#now we need to compute performance metric(s) for the set of winning predictions
	results = computeMetrics(testDependentVar, None, predictions, performanceMetricFuncs, negativeLabel)

	# Send this run to the log, if desired
	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results


def runOneVsOne(algorithm, trainX, testX, trainDependentVar, testDependentVar=None, arguments={}, scoreMode='label', sendToLog=True, timer=None):
	"""
	Calls on run() to train and evaluate the learning algorithm defined in 'algorithm.'  Assumes
	there are multiple (>2) class labels, and uses the one vs. one method of splitting the 
	training set into 2-label subsets. Tests performance using the metric function(s) found in 
	performanceMetricFunctions.

		algorithm: training algorithm to be called, in the form 'package.algorithmName'.

		trainX: data set to be used for training (as some form of Base object)
		
		testX: data set to be used for testing (as some form of Base object)
		
		trainDependentVar: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (in a Base object of the same type as trainX) 
		or an index (numerical or string) that defines their locale in the trainX object.
		
		testDependentVar: used to retreive the known class labels of the test data. Either
		contains the labels themselves or an index (numerical or string) that defines their locale
		in the testX object.  If not present, it is assumed that testDependentVar is the same
		as trainDependentVar.  
		
		arguments: optional arguments to be passed to the function specified by 'algorithm'

		scoreMode:  a flag with three possible values:  label, bestScore, or allScores.  If
		labels is selected, this function returns a single column with a predicted label for 
		each point in the test set.  If bestScore is selected, this function returns an object
		with two columns: the first has the predicted label, the second  has that label's score.  
		If allScores is selected, returns a Base object with each row containing a score for 
		each possible class label.  The class labels are the featureNames of the Base object, 
		so the list of scores in each row is not sorted by score, but by the order of class label
		found in featureNames.
		
		sendToLog: optional boolean valued parameter; True meaning the results should be logged
	"""
	trainX = trainX.duplicate()
	testX = testX.duplicate()

	if isinstance(trainDependentVar, Base):
		trainX.appendFeatures(trainDependentVar)
		trainDependentVar = trainX.features() - 1

	# If testDependentVar is missing, assume it is because it's the same as trainDependentVar
	if testDependentVar is None:
		testDependentVar = trainDependentVar

	#Remove true labels from testing set
	if isinstance(testDependentVar, (str, int, long)):
		testTrueLabels = testX.extractFeatures(testDependentVar)
	else:
		testTrueLabels = testDependentVar

	# Get set of unique class labels, then generate list of all 2-combinations of
	# class labels
	labelVector = trainX.copyFeatures([trainDependentVar])
	labelVector.transpose()
	labelSet = list(set(labelVector.toListOfLists()[0]))
	labelPairs = generateAllPairs(labelSet)

	#if we are logging this run, we need to start the timer
	if sendToLog:
		if timer is None:
			timer = Stopwatch()

		timer.start('train')

	# For each pair of class labels: remove all points with one of those labels,
	# train a classifier on those points, get predictions based on that model,
	# and put the points back into the data object
	rawPredictions = None
	predictionFeatureID = 0
	for pair in labelPairs:
		#get all points that have one of the labels in pair
		pairData = trainX.extractPoints(lambda point: (point[trainDependentVar] == pair[0]) or (point[trainDependentVar] == pair[1]))
		pairTrueLabels = pairData.extractFeatures(trainDependentVar)
		#train classifier on that data; apply it to the test set
		partialResults = run(algorithm, pairData, testX, output=None, dependentVar=pairTrueLabels, arguments=arguments, sendToLog=False)
		#put predictions into table of predictions
		if rawPredictions is None:
			rawPredictions = partialResults.toList()
		else:
			partialResults.renameFeatureName(0, 'predictions-'+str(predictionFeatureID))
			rawPredictions.appendFeatures(partialResults.toList())
		pairData.appendFeatures(pairTrueLabels)
		trainX.appendPoints(pairData)
		predictionFeatureID +=1

	if sendToLog:
		timer.stop('train')

	#set up the return data based on which format has been requested
	if scoreMode.lower() == 'label'.lower():
		return rawPredictions.applyFunctionToEachPoint(extractWinningPredictionLabel)
	elif scoreMode.lower() == 'bestScore'.lower():
		#construct a list of lists, with each row in the list containing the predicted
		#label and score of that label for the corresponding row in rawPredictions
		predictionMatrix = rawPredictions.toListOfLists()
		tempResultsList = []
		for row in predictionMatrix:
			scores = countWins(row)
			sortedScores = sorted(scores, key=scores.get, reverse=True)
			bestLabel = sortedScores[0]
			tempResultsList.append([bestLabel, scores[bestLabel]])

		#wrap the results data in a List container
		featureNames = ['PredictedClassLabel', 'LabelScore']
		resultsContainer = create("List", tempResultsList, featureNames=featureNames)
		return resultsContainer
	elif scoreMode.lower() == 'allScores'.lower():
		columnHeaders = sorted([str(i) for i in labelSet])
		labelIndexDict = {str(v):k for k, v in zip(range(len(columnHeaders)), columnHeaders)}
		predictionMatrix = rawPredictions.toListOfLists()
		resultsContainer = []
		for row in predictionMatrix:
			finalRow = [0] * len(columnHeaders)
			scores = countWins(row)
			for label, score in scores.items():
				finalIndex = labelIndexDict[str(int(label))]
				finalRow[finalIndex] = score
			resultsContainer.append(finalRow)

		return create(rawPredictions.getType(), resultsContainer, featureNames=columnHeaders)
	else:
		raise ArgumentException('Unknown score mode in runOneVsOne: ' + str(scoreMode))


	
def runOneVsAll(algorithm, trainX, testX, trainDependentVar, testDependentVar=None, arguments={}, scoreMode='label', sendToLog=True, timer=None):
	"""
	Calls on run() to train and evaluate the learning algorithm defined in 'algorithm.'  Assumes
	there are multiple (>2) class labels, and uses the one vs. all method of splitting the 
	training set into 2-label subsets. Tests performance using the metric function(s) found in 
	performanceMetricFunctions.

		algorithm: training algorithm to be called, in the form 'package.algorithmName'.

		trainX: data set to be used for training (as some form of Base object)
		
		testX: data set to be used for testing (as some form of Base object)
		
		trainDependentVar: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (in a Base object of the same type as trainX) 
		or an index (numerical or string) that defines their locale in the trainX object.
		
		testDependentVar: used to retreive the known class labels of the test data. Either
		contains the labels themselves or an index (numerical or string) that defines their locale
		in the testX object.  If not present, it is assumed that testDependentVar is the same
		as trainDependentVar.  
		
		arguments: optional arguments to be passed to the function specified by 'algorithm'

		scoreMode:  a flag with three possible values:  label, bestScore, or allScores.  If
		labels is selected, this function returns a single column with a predicted label for 
		each point in the test set.  If bestScore is selected, this function returns an object
		with two columns: the first has the predicted label, the second  has that label's score.  
		If allScores is selected, returns a Base object with each row containing a score for 
		each possible class label.  The class labels are the featureNames of the Base object, 
		so the list of scores in each row is not sorted by score, but by the order of class label
		found in featureNames.
		
		sendToLog: optional boolean valued parameter; True meaning the results should be logged
	"""
	trainX = trainX.duplicate()
	testX = testX.duplicate()


	# If testDependentVar is missing, assume it is because it's the same as trainDependentVar
	if testDependentVar is None and isinstance(trainDependentVar, (str, int, long)):
		testDependentVar = trainDependentVar
	elif testDependentVar is None:
		raise ArgumentException("Missing testDependentVar in runAndTestOneVsAll")

	#Remove true labels from from training set, if not already separated
	if isinstance(trainDependentVar, (str, int, long)):
		trainDependentVar = trainX.extractFeatures(trainDependentVar)

	#Remove true labels from test set, if not already separated
	if isinstance(testDependentVar, (str, int, long)):
		testDependentVar = testX.extractFeatures(testDependentVar)

	# Get set of unique class labels
	labelVector = trainX.copyFeatures([trainDependentVar])
	labelVector.transpose()
	labelSet = list(set(labelVector.toListOfLists()[0]))

	#if we are logging this run, we need to start the timer
	if sendToLog:
		if timer is None:
			timer = Stopwatch()

	timer.start('train')

	# For each class label in the set of labels:  convert the true
	# labels in trainDependentVar into boolean labels (1 if the point
	# has 'label', 0 otherwise.)  Train a classifier with the processed
	# labels and get predictions on the test set.
	rawPredictions = None
	for label in labelSet:
		def relabeler(point):
			if point[0] != label:
				return 0
			else: return 1
		trainLabels = trainDependentVar.applyFunctionToEachPoint(relabeler)
		oneLabelResults = run(algorithm, trainX, testX, output=None, dependentVar=trainLabels, arguments=arguments, sendToLog=False)
		#put all results into one Base container, of the same type as trainX
		if rawPredictions is None:
			rawPredictions = oneLabelResults
			#as it's added to results object, rename each column with its corresponding class label
			rawPredictions.renameFeatureName(0, str(label))
		else:
			#as it's added to results object, rename each column with its corresponding class label
			oneLabelResults.renameFeatureName(0, str(label))
			rawPredictions.appendFeatures(oneLabelResults)

	if scoreMode.lower() == 'label'.lower():
		winningPredictionIndices = rawPredictions.applyFunctionToEachPoint(extractWinningPredictionIndex).toListOfLists()
		indexToLabelMap = rawPredictions.featureNamesInverse
		winningLabels = []
		for winningIndex in winningPredictionIndices:
			winningLabels.append([indexToLabelMap[winningIndex]])
		return create(rawPredictions.getType, winningLabels, featureNames='winningLabel')

	elif scoreMode.lower() == 'bestScore'.lower():
		#construct a list of lists, with each row in the list containing the predicted
		#label and score of that label for the corresponding row in rawPredictions
		predictionMatrix = rawPredictions.toListOfLists()
		labelMapInverse = rawPredictions.featureNamesInverse
		tempResultsList = []
		for row in predictionMatrix:
			scores = extractWinningPredictionIndexAndScore(row, labelMapInverse)
			scores = sorted(scores, key=operator.itemgetter(1))
			bestLabelAndScore = scores[0]
			tempResultsList.append([[bestLabelAndScore[0], bestLabelAndScore[1]]])
		#wrap the results data in a List container
		featureNames = ['PredictedClassLabel', 'LabelScore']
		resultsContainer = create("List", tempResultsList, featureNames=featureNames)
		return resultsContainer

	elif scoreMode.lower() == 'allScores'.lower():
		#create list of Feature Names/Column Headers for final return object
		columnHeaders = sorted([str(i) for i in labelSet])
		#create map between label and index in list, so we know where to put each value
		labelIndexDict = {v:k for k, v in zip(range(len(columnHeaders)), columnHeaders)}
		featureNamesInverse = rawPredictions.featureNamesInverse
		predictionMatrix = rawPredictions.toListOfLists()
		resultsContainer = []
		for row in predictionMatrix:
			finalRow = [0] * len(columnHeaders)
			scores = extractConfidenceScores(row, featureNamesInverse)
			for label, score in scores.items():
				#get numerical index of label in return object
				finalIndex = labelIndexDict[label]
				#put score into proper place in its row
				finalRow[finalIndex] = score
			resultsContainer.append(finalRow)
		#wrap data in Base container
		return create(rawPredictions.getType(), resultsContainer, featureNames=columnHeaders)
	else:
		raise ArgumentException('Unknown score mode in runOneVsAll: ' + str(scoreMode))

def runAndTestOneVsAll(algorithm, trainX, testX, trainDependentVar, testDependentVar=None, arguments={}, performanceMetricFuncs=None, negativeLabel=None, sendToLog=True):
	"""
	Calls on run() to train and evaluate the learning algorithm defined in 'algorithm.'  Assumes
	there are multiple (>2) class labels, and uses the one vs. all method of splitting the 
	training set into 2-label subsets. Tests performance using the metric function(s) found in 
	performanceMetricFunctions.

		trainX: data set to be used for training (as some form of Base object)
		
		testX: data set to be used for testing (as some form of Base object)
		
		trainDependentVar: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (in a Base object of the same type as trainX) 
		or an index (numerical or string) that defines their locale in the trainX object.
		
		testDependentVar: used to retreive the known class labels of the test data. Either
		contains the labels themselves or an index (numerical or string) that defines their locale
		in the testX object.  If not present, it is assumed that testDependentVar is the same
		as trainDependentVar.  
		
		arguments: optional arguments to be passed to the function specified by 'algorithm'
		
		performanceMetricFuncs: iterable collection of functions that can take two collections
		of corresponding labels - one of true labels, one of predicted labels - and return a
		performance metric.
		
		sendToLog: optional boolean valued parameter; True meaning the results should be logged
	"""

	if sendToLog:
		timer = Stopwatch()

	if testDependentVar is None:
		if not isinstance(trainDependentVar, (str, int, long)):
			raise ArgumentException("testDependentVar is missing in runOneVsOne")
		else:
			testDependentVar = testX.extractFeatures([trainDependentVar])
	else:
		if isinstance(testDependentVar, (str, int, long)):
			testDependentVar = testX.extractFeatures([testDependentVar])

	predictions = runOneVsAll(algorithm, trainX, testX, trainDependentVar, testDependentVar, arguments, scoreMode='label', sendToLog=False, timer=timer)

	#now we need to compute performance metric(s) for the set of winning predictions
	results = computeMetrics(testDependentVar, None, predictions, performanceMetricFuncs, negativeLabel)

	# Send this run to the log, if desired
	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results



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
	allData = X.duplicate()
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






