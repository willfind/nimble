from UML.performance.performance_interface import computeMetrics
from UML.processing import BaseData
from UML.logging.log_manager import LogManager
from UML.logging.stopwatch import Stopwatch
from UML.interfaces.interface_helpers import generateAllPairs
from UML.utility import ArgumentException
from UML import run


def runAndTest(algorithm, trainX, testX, trainDependentVar, testDependentVar, arguments, performanceMetricFuncs, sendToLog=True):
	"""
		Calls on run() to train and evaluate the learning algorithm defined in 'algorithm,'
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
	results = computeMetrics(testDependentVar, None, rawResult, performanceMetricFuncs)

	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results


def runAndTestOneVsOne(algorithm, trainX, testX, trainDependentVar, testDependentVar=None, arguments={}, performanceMetricFuncs=None, sendToLog=True):
	"""
	Calls on run() to train and evaluate the learning algorithm defined in 'algorithm.'  Assumes
	there are multiple (>2) class labels, and uses the one vs. one method of splitting the 
	training set into 2-label subsets. Tests performance using the metric function(s) found in 
	performanceMetricFunctions.

		trainX: data set to be used for training (as some form of BaseData object)
		
		testX: data set to be used for testing (as some form of BaseData object)
		
		trainDependentVar: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (in a BaseData object of the same type as trainX) 
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

	#TODO DUPLICATE DATA BEFORE CALLING RUN
	trainX = trainX.duplicate()
	testX = testX.duplicate()

	if isinstance(trainDependentVar, BaseData):
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
			rawPredictions = partialResults.toRowListData()
		else:
			partialResults.renameFeatureName(0, 'predictions-'+str(predictionFeatureID))
			rawPredictions.appendFeatures(partialResults.toRowListData())
		pairData.appendFeatures(pairTrueLabels)
		trainX.appendPoints(pairData)
		predictionFeatureID +=1

	finalPredictions = rawPredictions.applyFunctionToEachPoint(extractWinningPredictionLabel)

	#if we are logging this run, we need to stop the timer
	if sendToLog:
		timer.stop('train')

	#now we need to compute performance metric(s) for the set of winning predictions
	results = computeMetrics(testTrueLabels, None, finalPredictions, performanceMetricFuncs)

	# Send this run to the log, if desired
	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results

def runAndTestOneVsAll(algorithm, trainX, testX, trainDependentVar, testDependentVar=None, arguments={}, performanceMetricFuncs=None, sendToLog=True):
	"""
	Calls on run() to train and evaluate the learning algorithm defined in 'algorithm.'  Assumes
	there are multiple (>2) class labels, and uses the one vs. all method of splitting the 
	training set into 2-label subsets. Tests performance using the metric function(s) found in 
	performanceMetricFunctions.

		trainX: data set to be used for training (as some form of BaseData object)
		
		testX: data set to be used for testing (as some form of BaseData object)
		
		trainDependentVar: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves (in a BaseData object of the same type as trainX) 
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

	#TODO DUPLICATE DATA BEFORE CALLING RUN
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
	else:
		testTrueLabels = testDependentVar

	# Get set of unique class labels
	labelVector = trainX.copyFeatures([trainDependentVar])
	labelVector.transpose()
	labelSet = list(set(labelVector.toListOfLists()[0]))

	#if we are logging this run, we need to start the timer
	if sendToLog:
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
		#put all results into one BaseData container, of the same type as trainX
		if rawPredictions is None:
			rawPredictions = oneLabelResults
			#as it's added to results object, rename each column with its corresponding class label
			rawPredictions.renameFeatureName(0, str(label))
		else:
			#as it's added to results object, rename each column with its corresponding class label
			oneLabelResults.renameFeatureName(0, str(label))
			rawPredictions.appendFeatures(oneLabelResults)


	# We now have a data object with a confidence score for each possible label.  We want to reduce
	# that to a prediction: predict whichever label has the highest score.  First we get the index 
	# of the winning label for each row.  Then we translate that index into the label from the original
	# data set.
	predictionLabelIndices = rawPredictions.applyFunctionToEachPoint(extractWinningPredictionIndex)
	predictionLabels = predictionLabelIndices.applyFunctionToEachPoint(lambda point: rawPredictions.featureNamesInverse[int(point[0])])


	#if we are logging this run, we need to stop the timer
	if sendToLog:
		timer.stop('train')

	#now we need to compute performance metric(s) for the set of winning predictions
	results = computeMetrics(testTrueLabels, None, predictionLabels, performanceMetricFuncs)

	# Send this run to the log, if desired
	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, timer, extraInfo=arguments)

	return results

def extractWinningPredictionLabel(predictions):
	"""
	Provided a list of tournament winners (class labels) for one point/row in a test set,
	choose the label that wins the most tournaments.  Returns the winning label.
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


def extractWinningPredictionIndex(predictionScores):
	"""
	Provided a list of confidence scores for one point/row in a test set,
	return the index of the column (i.e. label) of the highest score.  If
	no score in the list of predictionScores is a number greater than negative
	infinity, returns None.  
	"""

	maxScore = float("-inf")
	maxScoreIndex = -1
	for i in range(len(predictionScores)):
		score = predictionScores[i]
		if score > maxScore:
			maxScore = score
			maxScoreIndex = i

	if maxScoreIndex == -1:
		return None
	else:
		return maxScoreIndex


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
