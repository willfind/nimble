import ..combinations.Combinations
import ..performance.performance_interface


def runAndTest(trainData, testData, trainLabelIndicator, testLabelIndicator, function, performanceMetricFuncs):
	"""
		Trains a classifier using the function defined in 'function' using trainingData, then
		tests the performance of that classifier using the metric function(s) found in
		performanceMetricFunctions

		trainData: data set to be used for training (as some form of BaseData object)
		testData: data set to be used for testing (as some form of BaseData object)
		labelIndicator: used to retrieve the known class labels of the data.  Either contains
		the labels themselves or an index (numerical or string) that
	"""
	#TODO resolve problem w/labelIndicator:  if it is a matrix, how do you match it w/two data sets?
	if testLabelIndicator is None and isinstance(trainLabelIndicator, (str, unicode, int):
		testLabelIndicator = trainLabelIndicator

	trainLabels = getLabels(testData, trainLabelIndicator)
	testLabels = getLabels(trainData, testLabelIndicator)
	functionArgs = {'trainData':trainData,
					'testData':testData,
					'dependentVar':trainLabels
					}

	#rawResults contains predictions for each version of a learning function in the combos list
	rawResult = Combinations.applyCodeVersions(function, functionArgs)

	#now we need to compute performance metric(s) for all prediction sets
	predictedLabels = result.extractColumns([0])
	results = {}
	for perfMetric in performanceMetricFuncs:
		results[perfMetric] = performance_interface.computeMetrics(testLabels, knownValues=None, predictedLabels, perfomanceMetricFuncs)
	return results

def getLabels(dataSet, labelIndicator):
	if isinstance(labelIndicator, BaseData):
		#The known Indicator argument already contains all known
		#labels, so we do not need to do any further processing
		knownLabels = labelIndicator
	elif isinstance(labelIndicator, (str, unicode, int):
		#known Indicator is an index; we extract the column it indicates
		#from knownValues
		knownLabels = knownValues.extractColumns([knownIndicator])
	else:
		raise ArgumentException("Missing or improperly formatted indicator for known labels in computeMetrics")

	return knownLabels
