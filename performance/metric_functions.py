from math import sqrt
from ..utility.custom_exceptions import ArgumentException


def computeError(knownValues, predictedValues, loopFunction, compressionFunction):
	"""
		A generic function to compute different kinds of error metrics.  knownValues
		is a numpy array with the
		TODO: finish docstring
	"""
	if knownValues is None or len(knownValues.data) == 0:
		raise ArgumentException("Empty 'knownValues' argument in error calculator")
	elif predictedValues is None or len(predictedValues.data) == 0:
		raise ArgumentException("Empty 'predictedValues' argument in error calculator")

	n=0.0
	runningTotal=0.0
	for i in xrange(predictedValues.points()):
		pV = predictedValues.data[i][0]
		aV = knownValues.data[i][0]
		runningTotal = loopFunction(aV, pV, runningTotal)
		n += 1
	if n > 0:
		try:
			runningTotal = compressionFunction(runningTotal, n)
		except ZeroDivisionError:
			raise ZeroDivisionError('Tried to divide by zero when calculating performance metric')
			return
	else:
		raise ArgumentException("Empty argument(s) in error calculator")

	return runningTotal

def rmse(knownValues, predictedValues):
	"""
		Compute the root mean square error
	"""
	return computeError(knownValues, predictedValues, lambda x,y,z: z + (y - x)**2, lambda x,y: sqrt(x/y))

def meanAbsoluteError(knownValues, predictedValues):
	"""
		Compute mean absolute error.
	"""
	return computeError(knownValues, predictedValues, lambda x,y,z: z + abs(y - x), lambda x,y: x/y)

def classificationError(knownValues, predictedValues):
	"""
		Compute the proportion of incorrect predictions within a set of
		instances.
	"""
	return computeError(knownValues, predictedValues, lambda x,y,z: z if x == y else z + 1, lambda x,y: x/y)


def bottomPercentError(knownValues, labelScoreList, negativeLabel, proportionToScore=0.10, decisionThreshold=0.0):
	"""
		Note: this error function is only appropriate for binary classification
		situations.  If there are more than two labels in the labelScoreMap,
		it will break.
		Compute the proportion of incorrect predictions in the bottom 10% of
		predictions.  Bottom 10% is defined by sorting all predictions by
		the following metric: positiveLabelScore - negativeLabelScore, then
		computing the classification error only for those points whose metric
		fall within the lowest proportionToScore of points.
	"""
	if proportionToScore <= 0.0 or proportionToScore > 1.0:
		raise ArgumentException("proportionToScore must be between 0 and 1.0")

	#figure out the positive label
	firstLabelScoreMap = labelScoreList[0]
	positiveLabel = ''
	if len(firstLabelScoreMap) != 2:
		raise ArgumentException("bottomPercentError requires a set of precisely two predicted label scores for each point")

	for label in firstLabelScoreMap.keys():
		if label != negativeLabel:
			positiveLabel = label


	scoreDifferenceList = []
	for labelScoreMap in labelScoreList:
		positiveScore = labelScoreMap[positiveLabel]
		negativeScore = labelScoreMap[negativeLabel]
		scoreDiff = negativeScore - positiveScore
		scoreDifferenceList.append(scoreDiff)

	scoreDiffAndKnown = zip(scoreDifferenceList, knownValues)

	scoreDiffAndKnown.sort(key=lambda score: score[0])

	bottomProportionIndex = int(proportionToScore * len(scoreDiffAndKnown))

	sortedBottomProportion = scoreDiffAndKnown[0:bottomProportionIndex]

	sortedScoreDiffAndKnown = ([scoreDiff for scoreDiff,known in sortedBottomProportion], [known for scoreDiff,known in sortedBottomProportion])

	winningLabels = []
	for scoreDiff in sortedScoreDiffAndKnown[0]:
		if scoreDiff <= decisionThreshold:
			winningLabels.append(negativeLabel)
		else:
			winningLabels.append(positiveLabel)

	return classificationError(sortedScoreDiffAndKnown[1], winningLabels)





