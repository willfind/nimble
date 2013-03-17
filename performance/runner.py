import time
from UML.combinations.Combinations import executeCode
import performance_interface
from UML.processing import BaseData
from UML.logging.log_manager import LogManager
from UML.logging.stopwatch import Stopwatch
from UML.interfaces.interface_helpers import generateAllPairs
from UML import run


def runAndTest(algorithm, trainX, testX, trainDependentVar, testDependentVar, arguments, performanceMetricFuncs, sendToLog=True):
	"""
		Calls on run() to train and evaluate the learnin algorithm defined in 'algorithm,'
		then tests its performance using the metric function(s) found in
		performanceMetricFunctions

		trainX: data set to be used for training (as some form of BaseData object)
		testX: data set to be used for testing (as some form of BaseData object)
		trainDependentVar: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves or an index (numerical or string) that defines their locale
		in the trainX object
		testDependentVar: used to retreive the known class labels of the test data. Either
		contains the labels themselves or an index (numerical or string) that defines their locale
		in the testX object
		arguments: optional arguments to be passed to the function specified by 'algorithm'
		sendToLog: optional boolean valued parameter; True meaning the results should be logged
	"""
	#Need to make copies of all data, in case it will be modified before a classifier is trained
	trainX = trainX.duplicate()
	testX = testX.duplicate()
	
	if testDependentVar is None and isinstance(trainDependentVar, (str, unicode, int)):
		testDependentVar = trainDependentVar

	trainDependentVar = copyLabels(trainX, trainDependentVar)
	testDependentVar = copyLabels(testX, testDependentVar)

	#if we are logging this run, we need to start the timer
	if sendToLog:
		timer = Stopwatch()
		timer.start('train')

	#rawResults contains predictions for each version of a learning function in the combos list
	rawResult = run(algorithm, trainX, testX, dependentVar=trainDependentVar, arguments=arguments)

	#if we are logging this run, we need to stop the timer
	if sendToLog:
		timer.stop('train')

	#now we need to compute performance metric(s) for all prediction sets
	results = performance_interface.computeMetrics(testDependentVar, None, rawResult, performanceMetricFuncs)

	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results


def runOneVsOne(algorithm, trainX, testX, trainDependentIndicator, testDependentIndicator, arguments, performanceMetricFuncs, sendToLog=True):
	"""
	TODO: add docstring
	"""

	#TODO DUPLICATE DATA BEFORE CALLING RUN
	trainX = trainX.duplicate()
	testX = testX.duplicate()

	# If testDependentIndicator is missing, assume it is because it's the same as trainDependentIndicator
	if testDependentIndicator is None:
		testDependentIndicator = trainDependentIndicator

	#Remove true labels from testing set
	testTrueLabels = testX.extractFeatures(testDependentIndicator)

	# Get set of unique class labels, then generate list of all 2-combinations of
	# class labels
	labelVector = trainX.copyFeatures([trainDependentIndicator])
	labelVector.transpose()
	labelSet = list(set(labelVector.toListOfLists()[0]))
	labelPairs = generateAllPairs(labelSet)

	#if we are logging this run, we need to start the timer
	if sendToLog:
		timer = Stopwatch()
		timer.start('train')

	# For each pair of class labels: remove all points with one of those labels,
	# train a classifier on those points, get predictions based on that model,
	# and put the points back into the data object
	rawPredictions = None
	predictionFeatureID = 0
	for pair in labelPairs:
		filteringFunc = filterFuncGenerator(trainDependentIndicator, pair)
		#get all points that have one of the labels in pair
		pairData = trainX.extractPoints(filteringFunc)
		pairTrueLabels = pairData.extractFeatures(trainDependentIndicator)
		#train classifier on that data; apply it to the test set
		partialResults = run(algorithm, pairData, testX, output=None, dependentVar=pairTrueLabels, arguments=arguments, sendToLog=False)
		#put predictions into table of predictions
		if rawPredictions is None:
			rawPredictions = partialResults.toRowListData()
		else:
			partialResults.renameFeatureName(0, 'predictions-'+str(predictionFeatureID))
			rawPredictions.appendFeatures(partialResults.toRowListData())
		pairData.appendFeatures(pairTrueLabels)
		trainX.appendPoints(pairData)
		predictionFeatureID +=1

	finalPredictions = rawPredictions.applyFunctionToEachPoint(extractWinningPrediction)

	#if we are logging this run, we need to stop the timer
	if sendToLog:
		timer.stop('train')

	#now we need to compute performance metric(s) for the set of winning predictions
	results = performance_interface.computeMetrics(testTrueLabels, None, finalPredictions, performanceMetricFuncs)

	# Send this run to the log, if desired
	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results

	
def filterFuncGenerator(featureIndicator, desiredLabelPair):
	"""
		Generate a function, whose only argument is a point (vector), that will
		return true if the value in point indicated by featureIndicator is equal
		to one of the two values in desiredLabelPair.  Useful for extracting points
		from a BaseData object based on class/prediction value.

		Inputs:
			featureIndicator - either an int or a string, representing an index in a
			point 

			desiredLabelPair - pair of values (either as a tuple or a list).  If
			the value found in point at the index indicated by featureIndicator is 
			equal to either of these two values, the generated function will return
			True.  Otherwise, it will return false.

		Output:
			function that checks for equality of a feature in a point, designated by
			featureIndicator, and 2 possible values, found in desiredLabelPair
	"""
	desiredLabel1 = desiredLabelPair[0]
	desiredLabel2 = desiredLabelPair[1]

	def tmp(point):
		if (point[featureIndicator] == desiredLabel1) or (point[featureIndicator] == desiredLabel2):
			return True
		else: return False

	return tmp

def valueCounter(values):
	"""
	Given an iterable list of values, count how many times each individual value occurs
	and return a dictionary containing value:valueCount for each value in values.
	"""
	valueCounts = {}
	for value in values:
		if value in valueCounts:
			valueCounts[value] += 1
		else:
			valueCounts[value] = 1

	return valueCounts

def extractWinningPrediction(predictions):
	"""
	TODO: write docstring
	"""
	#Count how many times each class won
	#predictionCounts = valueCounter(predictions)

	predictionCounts = {}
	for prediction in predictions:
		if prediction in predictionCounts:
			predictionCounts[prediction] += 1
		else:
			predictionCounts[prediction] = 1

	#get the class that won the most tournaments
	#TODO: what if there are ties?
	return max(predictionCounts.iterkeys(), key=(lambda key: predictionCounts[key]))



#TODO this is a helper, move to utilities package?
def copyLabels(dataSet, dependentVar):
	"""
		A helper function to simplify the process of obtaining a 1-dimensional matrix of class
		labels from a data matrix.  Useful in functions which have an argument that may be
		a column index or a 1-dimensional matrix.  If 'dependentVar' is an index, this function
		will return a copy of the column in 'dataSet' indexed by 'dependentVar'.  If 'dependentVar'
		is itself a column (1-dimensional matrix w/shape (nx1)), dependentVar will be returned.

		dataSet:  matrix containing labels and, possibly, features.  May be empty if 'dependentVar'
		is a 1-column matrix containing labels.

		dependentVar: Either a column index indicating which column in dataSet contains class labels,
		or a matrix containing 1 column of class labels.

		returns A 1-column matrix of class labels
	"""
	if isinstance(dependentVar, BaseData):
		#The known Indicator argument already contains all known
		#labels, so we do not need to do any further processing
		labels = dependentVar
	elif isinstance(dependentVar, (str, unicode, int)):
		#known Indicator is an index; we extract the column it indicates
		#from knownValues
		labels = dataSet.copyFeatures([dependentVar])
	else:
		raise ArgumentException("Missing or improperly formatted indicator for known labels in computeMetrics")

	return labels
