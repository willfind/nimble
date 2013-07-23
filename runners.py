"""
All the functions which involve calling interfaces to run some algorithm on
the provided data. 

"""

import operator

import UML

from UML.exceptions import ArgumentException
from UML.logging import UmlLogger
from UML.logging import LogManager
from UML.logging import Stopwatch
from UML.data import Base
from UML.interfaces import shogun
from UML.interfaces import mahout
from UML.interfaces import regressor
from UML.interfaces import sciKitLearn
from UML.interfaces import mlpy

from UML.umlHelpers import computeMetrics
from UML.umlHelpers import _loadSparse
from UML.umlHelpers import _loadMatrix
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


def _validScoreMode(scoreMode):
	scoreMode = scoreMode.lower()
	if scoreMode != 'label' and scoreMode != 'bestscore' and scoreMode != 'allscores':
		raise ArgumentException("scoreMode may only be 'label' 'bestScore' or 'allScores'")


def _validMultiClassStrategy(multiClassStrategy):
	multiClassStrategy = multiClassStrategy.lower()
	if multiClassStrategy != 'default' and multiClassStrategy != 'OneVsAll'.lower() and multiClassStrategy != 'OneVsOne'.lower():
		raise ArgumentException("multiClassStrategy may only be 'default' 'OneVsAll' or 'OneVsOne'")


def _unpackAlgorithm(algorithm):
	splitList = algorithm.split('.',1)
	if len(splitList) < 2:
		raise ArgumentException("The algorithm must be prefaced with the package name and a dot. Example:'mlpy.KNN'")
	package = splitList[0]
	algorithm = splitList[1]
	return (package, algorithm)


def _validArguments(arguments):
	if not isinstance(arguments, dict):
		raise ArgumentException("The 'arguments' parameter must be a dictionary")

def _validData(trainX, trainY, testX, testY):
	if not isinstance(trainX, Base):
		raise ArgumentException("trainX may only be an object derived from Base")
	if trainY is not None:
		if not (isinstance(trainY, Base) or isinstance(trainY, basestring) or isinstance(trainY, int)):
			raise ArgumentException("trainY may only be an object derived from Base, or an ID of the feature containing labels in testX")
	if testX is None:
		raise ArgumentException("Despite it being an optional parameter, testX must not be None. textX may only be an object derived from Base")
	if not isinstance(testX, Base):
		raise ArgumentException("testX may only be an object derived from Base")
	if testY is not None:
		if not (isinstance(testY, Base) or isinstance(testY, basestring) or isinstance(testY, int)):
			raise ArgumentException("testY may only be an object derived from Base, or an ID of the feature containing labels in testX")


def run(algorithm, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', sendToLog=True):
	(package, algorithm) = _unpackAlgorithm(algorithm)
	_validData(trainX, trainY, testX, None)
	_validScoreMode(scoreMode)
	_validMultiClassStrategy(multiClassStrategy)
	_validArguments(arguments)

	if sendToLog:
		timer = Stopwatch()
	else:
		timer = None

	if package == 'mahout':
		results = mahout(algorithm, trainX, trainY, testX, arguments, output, timer)
	elif package == 'regressor':
		results = regressor(algorithm, trainX, trainY, testX, arguments, output, timer)
	elif package == 'sciKitLearn':
		results = sciKitLearn(algorithm, trainX, trainY, testX, arguments, output, scoreMode, multiClassStrategy, timer)
	elif package == 'mlpy':
		results = mlpy(algorithm, trainX, trainY, testX, arguments, output, scoreMode, multiClassStrategy, timer)
	elif package == 'shogun':
		results = shogun(algorithm, trainX, trainY, testX, arguments, output, scoreMode, multiClassStrategy, timer)
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
			logManager.logRun(trainX, testX, funcString, None, timer, extraInfo=arguments)

	return results



def runAndTest(algorithm, trainX, trainY, testX, testY, arguments, performanceMetricFuncs, scoreMode='label', negativeLabel=None, sendToLog=True):
	"""
		Calls on run() to train and evaluate the learning algorithm defined in 'algorithm,'
		then tests its performance using the metric function(s) found in
		performanceMetricFunctions

		algorithm: training algorithm to be called, in the form 'package.algorithmName'.

		trainX: data set to be used for training (as some form of Base object)

		testX: data set to be used for testing (as some form of Base object)
		
		trainY: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (as a Base object) or an index (numerical or string) 
		that defines their locale in the trainX object
		
		testY: used to retreive the known class labels of the test data. Either
		contains the labels themselves (as a Base object) or an index (numerical or string) 
		that defines their locale in the testX object.  If left blank, runAndTest() assumes 
		that testY is the same as trainY.
		
		arguments: optional arguments to be passed to the function specified by 'algorithm'
		
		negativeLabel: Argument required if performanceMetricFuncs contains proportionPercentPositive90
		or proportionPercentPositive50.  Identifies the 'negative' label in the data set.  Only
		applies to data sets with 2 class labels.
		
		sendToLog: optional boolean valued parameter; True meaning the results should be logged
	"""
	_validData(trainX, trainY, testX, testY)

	#Need to make copies of all data, in case it will be modified before a classifier is trained
	trainX = trainX.copy()
	testX = testX.copy()
	
	#if testY is empty, attempt to use trainY
	if testY is None and isinstance(trainY, (str, unicode, int)):
		testY = trainY

	trainY = copyLabels(trainX, trainY)
	testY = copyLabels(testX, testY)

	#if we are logging this run, we need to start the timer
	if sendToLog:
		timer = Stopwatch()
		timer.start('train')

	#rawResults contains predictions for each version of a learning function in the combos list
	rawResult = run(algorithm, trainX, trainY, testX, arguments=arguments, scoreMode=scoreMode, sendToLog=False)

	#if we are logging this run, we need to stop the timer
	if sendToLog:
		timer.stop('train')
		timer.start('errorComputation')

	#now we need to compute performance metric(s) for all prediction sets
	results = computeMetrics(testY, None, rawResult, performanceMetricFuncs, negativeLabel)

	if sendToLog:
		timer.stop('errorComputation')

	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results

def runAndTestOneVsOne(algorithm, trainX, trainY, testX, testY=None, arguments={}, performanceMetricFuncs=None, negativeLabel=None, sendToLog=True):
	"""
		Wrapper class for runOneVsOne.  Useful if you want the entire process of training,
		testing, and computing performance measures to be handled.  Takes in a learning algorithm
		and training and testing data sets, trains a learner, passes the test data to the 
		computed model, gets results, and calculates performance based on those results.  

		Arguments:

			algorithm: training algorithm to be called, in the form 'package.algorithmName'.

			trainX: data set to be used for training (as some form of Base object)
		
			testX: data set to be used for testing (as some form of Base object)
		
			trainY: used to retrieve the known class labels of the traing data. Either
			contains the labels themselves (in a Base object of the same type as trainX) 
			or an index (numerical or string) that defines their locale in the trainX object.
		
			testY: used to retreive the known class labels of the test data. Either
			contains the labels themselves or an index (numerical or string) that defines their locale
			in the testX object.  If not present, it is assumed that testY is the same
			as trainY.  
			
			arguments: optional arguments to be passed to the function specified by 'algorithm'

			performanceMetricFuncs: iterable collection of functions that can take two collections
			of corresponding labels - one of true labels, one of predicted labels - and return a
			performance metric.
		
			sendToLog: optional boolean valued parameter; True meaning the results should be printed 
			to log file.

		Returns: A dictionary associating the name or code of performance metrics with the results
		of those metrics, computed using the predictions of 'algorithm' on testX.  
		Example: { 'fractionIncorrect': 0.21, 'numCorrect': 1020 }
	"""
	_validData(trainX, trainY, testX, testY)

	if sendToLog:
		timer = Stopwatch()
	else:
		timer = None

	if testY is None:
		if not isinstance(trainY, (str, int, long)):
			raise ArgumentException("testY is missing in runOneVsOne")
		else:
			testY = testX.extractFeatures([trainY])
	else:
		if isinstance(testY, (str, int, long)):
			testY = testX.extractFeatures([testY])

	predictions = runOneVsOne(algorithm, trainX, trainY, testX, testY, arguments, scoreMode='label', sendToLog=False, timer=timer)

	#now we need to compute performance metric(s) for the set of winning predictions
	results = computeMetrics(testY, None, predictions, performanceMetricFuncs, negativeLabel)

	# Send this run to the log, if desired
	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results


def runOneVsOne(algorithm, trainX, trainY, testX, testY=None, arguments={}, scoreMode='label', sendToLog=True, timer=None):
	"""
	Calls on run() to train and evaluate the learning algorithm defined in 'algorithm.'  Assumes
	there are multiple (>2) class labels, and uses the one vs. one method of splitting the 
	training set into 2-label subsets. Tests performance using the metric function(s) found in 
	performanceMetricFunctions.

		algorithm: training algorithm to be called, in the form 'package.algorithmName'.

		trainX: data set to be used for training (as some form of Base object)
		
		testX: data set to be used for testing (as some form of Base object)
		
		trainY: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (in a Base object of the same type as trainX) 
		or an index (numerical or string) that defines their locale in the trainX object.
		
		testY: used to retreive the known class labels of the test data. Either
		contains the labels themselves or an index (numerical or string) that defines their locale
		in the testX object.  If not present, it is assumed that testY is the same
		as trainY.  
		
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
	_validData(trainX, trainY, testX, testY)

	trainX = trainX.copy()
	testX = testX.copy()

	if isinstance(trainY, Base):
		trainX.appendFeatures(trainY)
		trainY = trainX.features() - 1

	# If testY is missing, assume it is because it's the same as trainY
	if testY is None:
		testY = trainY

	#Remove true labels from testing set
	if isinstance(testY, (str, int, long)):
		testTrueLabels = testX.extractFeatures(testY)
	else:
		testTrueLabels = testY

	# Get set of unique class labels, then generate list of all 2-combinations of
	# class labels
	labelVector = trainX.copyFeatures([trainY])
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
		pairData = trainX.extractPoints(lambda point: (point[trainY] == pair[0]) or (point[trainY] == pair[1]))
		pairTrueLabels = pairData.extractFeatures(trainY)
		#train classifier on that data; apply it to the test set
		partialResults = run(algorithm, pairData, pairTrueLabels, testX, output=None, arguments=arguments, sendToLog=False)
		#put predictions into table of predictions
		if rawPredictions is None:
			rawPredictions = partialResults.toList()
		else:
			partialResults.setFeatureName(0, 'predictions-'+str(predictionFeatureID))
			rawPredictions.appendFeatures(partialResults.toList())
		pairData.appendFeatures(pairTrueLabels)
		trainX.appendPoints(pairData)
		predictionFeatureID +=1

	if sendToLog:
		timer.stop('train')

	#set up the return data based on which format has been requested
	if scoreMode.lower() == 'label'.lower():
		return rawPredictions.applyToEachPoint(extractWinningPredictionLabel)
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
		resultsContainer = UML.createData("List", tempResultsList, featureNames=featureNames)
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

		return UML.createData(rawPredictions.getTypeString(), resultsContainer, featureNames=columnHeaders)
	else:
		raise ArgumentException('Unknown score mode in runOneVsOne: ' + str(scoreMode))


	
def runOneVsAll(algorithm, trainX, trainY, testX, testY=None, arguments={}, scoreMode='label', sendToLog=True, timer=None):
	"""
	Calls on run() to train and evaluate the learning algorithm defined in 'algorithm.'  Assumes
	there are multiple (>2) class labels, and uses the one vs. all method of splitting the 
	training set into 2-label subsets. Tests performance using the metric function(s) found in 
	performanceMetricFunctions.

		algorithm: training algorithm to be called, in the form 'package.algorithmName'.

		trainX: data set to be used for training (as some form of Base object)
		
		testX: data set to be used for testing (as some form of Base object)
		
		trainY: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (in a Base object of the same type as trainX) 
		or an index (numerical or string) that defines their locale in the trainX object.
		
		testY: used to retreive the known class labels of the test data. Either
		contains the labels themselves or an index (numerical or string) that defines their locale
		in the testX object.  If not present, it is assumed that testY is the same
		as trainY.  
		
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
	_validData(trainX, trainY, testX, testY)
	trainX = trainX.copy()
	testX = testX.copy()


	# If testY is missing, assume it is because it's the same as trainY
	if testY is None and isinstance(trainY, (str, int, long)):
		testY = trainY
	elif testY is None:
		raise ArgumentException("Missing testY in runAndTestOneVsAll")

	#Remove true labels from from training set, if not already separated
	if isinstance(trainY, (str, int, long)):
		trainY = trainX.extractFeatures(trainY)

	#Remove true labels from test set, if not already separated
	if isinstance(testY, (str, int, long)):
		testY = testX.extractFeatures(testY)

	# Get set of unique class labels
	labelVector = trainX.copyFeatures([trainY])
	labelVector.transpose()
	labelSet = list(set(labelVector.toListOfLists()[0]))

	#if we are logging this run, we need to start the timer
	if sendToLog:
		if timer is None:
			timer = Stopwatch()

	timer.start('train')

	# For each class label in the set of labels:  convert the true
	# labels in trainY into boolean labels (1 if the point
	# has 'label', 0 otherwise.)  Train a classifier with the processed
	# labels and get predictions on the test set.
	rawPredictions = None
	for label in labelSet:
		def relabeler(point):
			if point[0] != label:
				return 0
			else: return 1
		trainLabels = trainY.applyToEachPoint(relabeler)
		oneLabelResults = run(algorithm, trainX, trainLabels, testX, output=None, arguments=arguments, sendToLog=False)
		#put all results into one Base container, of the same type as trainX
		if rawPredictions is None:
			rawPredictions = oneLabelResults
			#as it's added to results object, rename each column with its corresponding class label
			rawPredictions.setFeatureName(0, str(label))
		else:
			#as it's added to results object, rename each column with its corresponding class label
			oneLabelResults.setFeatureName(0, str(label))
			rawPredictions.appendFeatures(oneLabelResults)

	if scoreMode.lower() == 'label'.lower():
		winningPredictionIndices = rawPredictions.applyToEachPoint(extractWinningPredictionIndex).toListOfLists()
		indexToLabelMap = rawPredictions.featureNamesInverse
		winningLabels = []
		for winningIndex in winningPredictionIndices:
			winningLabels.append([indexToLabelMap[winningIndex]])
		return UML.createData(rawPredictions.getTypeString(), winningLabels, featureNames='winningLabel')

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
		resultsContainer = UML.createData("List", tempResultsList, featureNames=featureNames)
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
		return UML.createData(rawPredictions.getTypeString(), resultsContainer, featureNames=columnHeaders)
	else:
		raise ArgumentException('Unknown score mode in runOneVsAll: ' + str(scoreMode))

def runAndTestOneVsAll(algorithm, trainX, trainY, testX, testY=None, arguments={}, performanceMetricFuncs=None, negativeLabel=None, sendToLog=True):
	"""
	Calls on run() to train and evaluate the learning algorithm defined in 'algorithm.'  Assumes
	there are multiple (>2) class labels, and uses the one vs. all method of splitting the 
	training set into 2-label subsets. Tests performance using the metric function(s) found in 
	performanceMetricFunctions.

		trainX: data set to be used for training (as some form of Base object)
		
		testX: data set to be used for testing (as some form of Base object)
		
		trainY: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (in a Base object of the same type as trainX) 
		or an index (numerical or string) that defines their locale in the trainX object.
		
		testY: used to retreive the known class labels of the test data. Either
		contains the labels themselves or an index (numerical or string) that defines their locale
		in the testX object.  If not present, it is assumed that testY is the same
		as trainY.  
		
		arguments: optional arguments to be passed to the function specified by 'algorithm'
		
		performanceMetricFuncs: iterable collection of functions that can take two collections
		of corresponding labels - one of true labels, one of predicted labels - and return a
		performance metric.
		
		sendToLog: optional boolean valued parameter; True meaning the results should be logged
	"""
	_validData(trainX, trainY, testX, testY)

	if sendToLog:
		timer = Stopwatch()

	if testY is None:
		if not isinstance(trainY, (str, int, long)):
			raise ArgumentException("testY is missing in runOneVsOne")
		else:
			testY = testX.extractFeatures([trainY])
	else:
		if isinstance(testY, (str, int, long)):
			testY = testX.extractFeatures([testY])

	predictions = runOneVsAll(algorithm, trainX, trainY, testX, testY, arguments, scoreMode='label', sendToLog=False, timer=timer)

	#now we need to compute performance metric(s) for the set of winning predictions
	results = computeMetrics(testY, None, predictions, performanceMetricFuncs, negativeLabel)

	# Send this run to the log, if desired
	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results

