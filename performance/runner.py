import time
from UML.combinations.Combinations import executeCode
import performance_interface
from UML.processing import BaseData
from UML.logging.log_manager import LogManager
from UML import run


def runAndTest(algorithm, trainX, testX, trainDependentVar, testDependentVar, arguments, performanceMetricFuncs, sendToLog=True):
	"""
		Calls on run() to train and evaluate the learnin algorithm defined in 'algorithm,'
		then tests its performance using the metric function(s) found in
		performanceMetricFunctions

		trainX: data set to be used for training (as some form of BaseData object)
		testX: data set to be used for testing (as some form of BaseData object)
		trainDependentVar: used to retrieve the known class labels of the traing data. Either
		contains the labels themselves or an index (numerical or string) that defines their local
		in the trainX object
		testDependentVar: used to retrive the known class labels of the test data. Either
		contains the labels themselves or an index (numerical or string) that defines their local
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
		startTime = time.clock()

	#rawResults contains predictions for each version of a learning function in the combos list
	rawResult = run(algorithm, trainX, testX, dependentVar=trainDependentVar, arguments=arguments)

	#if we are logging this run, we need to stop the timer
	if sendToLog:
		stopTime = time.clock()

	#now we need to compute performance metric(s) for all prediction sets
	results = performance_interface.computeMetrics(testDependentVar, None, rawResult, performanceMetricFuncs)

	if sendToLog:
		logManager = LogManager()
		logManager.logRun(trainX, testX, algorithm, results, stopTime - startTime, extraInfo=arguments)
	return results


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
