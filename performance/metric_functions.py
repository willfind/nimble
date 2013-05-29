from math import sqrt
from UML import data
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

def proportionPercentNegative90(knownValues, predictedValues, negativeLabel):
	"""
		Wrapper function for proportionPercentError.  Computes the proportion
		of posts that fall in the 90% most likely to be in the positive class
		that are actually in the negative class.
				
	"""
	return proportionPercentNegative(knownValues, predictedValues, negativeLabel, proportionToScore=0.90)

def proportionPercentNegative50(knownValues, predictedValues, negativeLabel):
	"""
		Wrapper function for proportionPercentError.  Computes the proportion
		of posts that fall in the 50% most likely to be in the positive class
		that are actually in the negative class.
	"""
	return proportionPercentNegative(knownValues, predictedValues, negativeLabel, proportionToScore=0.50)

def bottomProportionPercentNegative10(knownValues, predictedValues, negativeLabel):
	"""
		Wrapper function for proportionPercentError.  Computes the proportion
		of posts that fall in the 50% most likely to be in the positive class
		that are actually in the negative class.
	"""
	return proportionPercentNegative(knownValues, predictedValues, negativeLabel, proportionToScore=0.10, reverseSort=True)

def proportionPercentNegative(knownValues, labelScoreList, negativeLabel, proportionToScore=0.90, reverseSort=False):
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
	labelNames = labelScoreList.featureNames
	positiveLabel = ''
	if len(labelNames) != 2:
		raise ArgumentException("proportionPercentNegative requires a set of precisely two predicted label scores for each point")

	for labelName in labelNames.keys():
		if labelName == negativeLabel:
			continue
		else:
			positiveLabel = labelName

	negativeLabelIndex = labelScoreList.featureNames[negativeLabel]
	positiveLabelIndex = labelScoreList.featureNames[positiveLabel]

	#Compute the score difference (positive label score - negative label score) for
	#all entries in labelScoreList
	scoreDifferenceList = []
	labelScoreList = labelScoreList.toListOfLists()
	for i in range(len(labelScoreList)):
		positiveScore = labelScoreList[i][positiveLabelIndex]
		negativeScore = labelScoreList[i][negativeLabelIndex]
		scoreDiff = positiveScore - negativeScore
		scoreDifferenceList.append(scoreDiff)

	#convert knownValues to list of lists; drop first row, which has featureNames
	listOfKnownLabels = knownValues.toListOfLists()
	knownLabels = listOfKnownLabels[0:]
	for i in range(len(knownLabels)):
		knownLabel = knownLabels[i][0]
		knownLabels[i] = knownLabels[i][0]

	#Put together score differences and known labels, then sort by score difference,
	#so we have a list ranked, in descending order, by most likely to have negative label
	scoreDiffAndKnown = zip(scoreDifferenceList, knownLabels)

	if reverseSort == True:
		scoreDiffAndKnown.sort(key=lambda score: score[0])
	else:
		scoreDiffAndKnown.sort(key=lambda score: score[0], reverse=True)

	#Get bottom of list (lowest score differences, which may be negative)
	topProportionIndex = int(round(proportionToScore * len(scoreDiffAndKnown)))
	sortedTopProportion = scoreDiffAndKnown[0:topProportionIndex]

	#Unzip into two lists
	sortedScoreDiffAndKnown = ([scoreDiff for scoreDiff,known in sortedTopProportion], [known for scoreDiff,known in sortedTopProportion])

	#get newly sorted known labels
	sortedKnownValues = sortedScoreDiffAndKnown[1]

	#compute number of negative labels present in specified proportion of posts that
	#are predicted to be more likely to be positive labels
	numNegativeLabels = 0
	for knownLabel in sortedKnownValues:
		if str(knownLabel) == negativeLabel:
			numNegativeLabels += 1

	#return proportion of top posts that are negative
	return float(numNegativeLabels) / float(len(sortedKnownValues))

def bottomPercentError(knownValues, labelScoreList, negativeLabel):
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
	#figure out the positive label
	labelNames = labelScoreList.featureNames
	positiveLabel = ''
	if len(labelNames) != 2:
		raise ArgumentException("proportionPercentNegative requires a set of precisely two predicted label scores for each point")

	for labelName in labelNames.keys():
		if labelName == negativeLabel:
			continue
		else:
			positiveLabel = labelName

	negativeLabelIndex = labelScoreList.featureNames[negativeLabel]
	positiveLabelIndex = labelScoreList.featureNames[positiveLabel]

	#Compute the score difference (positive label score - negative label score) for
	#all entries in labelScoreList
	scoreDifferenceList = []
	labelScoreList = labelScoreList.toListOfLists()
	for i in range(len(labelScoreList)):
		positiveScore = labelScoreList[i][positiveLabelIndex]
		negativeScore = labelScoreList[i][negativeLabelIndex]
		scoreDiff = positiveScore - negativeScore
		scoreDifferenceList.append(scoreDiff)

	#convert knownValues to list of lists; drop first row, which has featureNames
	listOfKnownLabels = knownValues.toListOfLists()
	knownLabels = listOfKnownLabels[0:][0]

	#Put together score differences and known labels, then sort by score difference,
	#so we have a list ranked, in descending order, by most likely to have negative label
	scoreDiffAndKnown = zip(scoreDifferenceList, knownLabels)

	scoreDiffAndKnown.sort(key=lambda score: score[0])

	#Get bottom of list (lowest score differences, which may be negative)
	topProportionIndex = int(round(0.10 * len(scoreDiffAndKnown)))
	sortedTopProportion = scoreDiffAndKnown[0:]

	#Unzip into two lists
	sortedScoreDiffAndKnown = ([scoreDiff for scoreDiff,known in sortedTopProportion], [known for scoreDiff,known in sortedTopProportion])

	#get newly sorted known labels
	sortedKnownValues = sortedScoreDiffAndKnown[1]

	#compute number of negative labels present in specified proportion of posts that
	#are predicted to be more likely to be positive labels
	winningLabels = []
	for scoreDiff in sortedScoreDiffAndKnown[0]:
		if scoreDiff <= 0.0:
			winningLabels.append(negativeLabel)
		else:
			winningLabels.append(positiveLabel)

	correctPredictions = 0
	for i in range(len(winningLabels)):
		if sortedKnownValues[i] == winningLabels[i]:
			correctPredictions += 1

	proportionCorrect = float(correctPredictions) / float(len(sortedKnownValues))

	return proportionCorrect

