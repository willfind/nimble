
# Layers:
#	top- a crossvalidatereturnbest equivalent
#	middle - main func to deal with each input function and calculating the sizes for each window etc
#	bottom - main func also collects and aggregates the output

#TODO re-add logging
#TODO copyPoints need to be able to take ranges
#TODO we should be able to query our error metric in order to aggregate the return values

import datetime

import Combinations
from ..processing.base_data import DEFAULT_PREFIX
from ..utility import ArgumentException

from nose.tools import set_trace

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
			currResult = Combinations.executeCode(function, dataHash)
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



def _incrementTrialWindows(allData, orderedFeature, currEndTrain, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize):
	"""
	Helper which will calculate the start and end of the training and testing sizes given the current
	position in the full data set. 

	"""
#	set_trace()
	# determine the location of endTrain.
	if currEndTrain is None:
		# points are zero indexed, thus -1 for the num of points case
#		set_trace()
		endTrain = _jumpForward(allData, orderedFeature, 0, minTrainSize, -1)
	else:
		endTrain =_jumpForward(allData, orderedFeature, currEndTrain, stepSize)

	# the value we don't want to split from the training set
	nonSplit = allData[endTrain,orderedFeature]
	# we're doing a lookahead here, thus -1 from the last possible index, and  +1 to our lookup
	while (endTrain < allData.points() - 1 and allData[endTrain+1,orderedFeature] == nonSplit):
		endTrain += 1

	if endTrain == allData.points() -1:
		return None

	# we get the start for training by counting back from endTrain
	startTrain = _jumpBack(allData, orderedFeature, endTrain, maxTrainSize, -1)
	if startTrain < 0:
		startTrain = 0
#	if _diffLessThan(allData, orderedFeature, startTrain, endTrain, minTrainSize):
#		return _incrementTrialWindows(allData, orderedFeature, currEndTrain+1, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize)
#		return None

	# we get the start and end of the test set by counting forward from endTrain
	# speciffically, we go forward by one, and as much more forward as specified by gap
	startTest = _jumpForward(allData, orderedFeature, endTrain+1, gap)
	if startTest >= allData.points():
		return None

	endTest = _jumpForward(allData, orderedFeature, startTest, maxTestSize, -1)
	if endTest >= allData.points():
		endTest = allData.points() - 1
	if _diffLessThan(allData, orderedFeature, startTest, endTest, minTestSize):
#		return _incrementTrialWindows(allData, orderedFeature, currEndTrain+1, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize)
		return None

	return (startTrain, endTrain, startTest, endTest)


def _jumpBack(allData, orderedFeature, start, delta, intCaseOffset=0):
	if isinstance(delta,datetime.timedelta):
		endPoint = start
		startVal = datetime.timedelta(float(allData[start,orderedFeature]))
		# loop as long as we don't run off the end of the data
		while (endPoint > 0):
			if (startVal - datetime.timedelta(float(allData[endPoint-1,orderedFeature])) > delta):
				break
			endPoint = endPoint - 1 
	else:
		endPoint = start - (delta + intCaseOffset)

	return endPoint


def _jumpForward(allData, orderedFeature, start, delta, intCaseOffset=0):
	if isinstance(delta,datetime.timedelta):
		endPoint = start
		startVal = datetime.timedelta(float(allData[start,orderedFeature]))
		# loop as long as we don't run off the end of the data
		while (endPoint < allData.points() - 1):
			if (datetime.timedelta(float(allData[endPoint+1,orderedFeature])) - startVal > delta):
				break
			endPoint = endPoint + 1 
	else:
		endPoint = start + (delta + intCaseOffset)

	return endPoint



def _diffLessThan(allData, orderedFeature, startPoint, endPoint, delta):
	if isinstance(delta,datetime.timedelta):
		startVal = datetime.timedelta(float(allData[startPoint,orderedFeature]))
		endVal = datetime.timedelta(float(allData[endPoint,orderedFeature]))
		return (endVal - startVal) < delta
	else:
		return (endPoint - startPoint + 1) < delta
