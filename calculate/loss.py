

import UML
from UML.data import Base
from UML.data import Matrix
from math import sqrt

from UML.exceptions import ArgumentException

def _validatePredictedAsLabels(predictedValues):
	if not isinstance(predictedValues, UML.data.Base):
		raise ArgumentException("predictedValues must be derived class of UML.data.Base")
	if predictedValues.featureCount > 1:
		raise ArgumentException("predictedValues must be labels only; this has more than one feature")

def _computeError(knownValues, predictedValues, loopFunction, compressionFunction):
	"""
		A generic function to compute different kinds of error metrics.  knownValues
		is a 1d Base object with one known label (or number) per row. predictedValues is a 1d Base
		object with one predictedLabel (or score) per row.  The ith row in knownValues should refer
		to the same point as the ith row in predictedValues. loopFunction is a function to be applied
		to each row in knownValues/predictedValues, that takes 3 arguments: a known class label,
		a predicted label, and runningTotal, which contains the successive output of loopFunction.
		compressionFunction is a function that should take two arguments: runningTotal, the final
		output of loopFunction, and n, the number of values in knownValues/predictedValues.
	"""
	if knownValues is None or not isinstance(knownValues, Base) or knownValues.pointCount == 0:
		raise ArgumentException("Empty 'knownValues' argument in error calculator")
	elif predictedValues is None or not isinstance(predictedValues, Base) or predictedValues.pointCount == 0:
		raise ArgumentException("Empty 'predictedValues' argument in error calculator")

	if not isinstance(knownValues, Matrix):
		knownValues = knownValues.copyAs(format="Matrix")

	if not isinstance(predictedValues, Matrix):
		predictedValues = predictedValues.copyAs(format="Matrix")

	n = 0.0
	runningTotal = 0.0
	#Go through all values in known and predicted values, and pass those values to loopFunction
	for i in xrange(predictedValues.pointCount):
		pV = predictedValues[i,0]
		aV = knownValues[i,0]
		runningTotal = loopFunction(aV, pV, runningTotal)
		n += 1
	if n > 0:
		try:
			#provide the final value from loopFunction to compressionFunction, along with the
			#number of values looped over
			runningTotal = compressionFunction(runningTotal, n)
		except ZeroDivisionError:
			raise ZeroDivisionError('Tried to divide by zero when calculating performance metric')
			return
	else:
		raise ArgumentException("Empty argument(s) in error calculator")

	return runningTotal


def rootMeanSquareError(knownValues, predictedValues):
	"""
		Compute the root mean square error.  Assumes that knownValues and predictedValues contain
		numerical values, rather than categorical data.
	"""
	_validatePredictedAsLabels(predictedValues)
	return _computeError(knownValues, predictedValues, lambda x,y,z: z + (y - x)**2, lambda x,y: sqrt(x/y))

def meanFeaturewiseRootMeanSquareError(knownValues, predictedValues):
	"""For 2d prediction data, compute the RMSE of each feature, then average
	the results.
	"""
	if knownValues.featureCount != predictedValues.featureCount:
		raise ArgumentException("The known and predicted data must have the same number of features")
	if knownValues.pointCount != predictedValues.pointCount:
		raise ArgumentException("The known and predicted data must have the same number of points")

	results = []
	for i in xrange(knownValues.featureCount):
		currKnown = knownValues.copyFeatures(i)
		currPred = predictedValues.copyFeatures(i)
		results.append(rootMeanSquareError(currKnown, currPred))

	return float(sum(results)) / knownValues.featureCount


def meanAbsoluteError(knownValues, predictedValues):
	"""
		Compute mean absolute error. Assumes that knownValues and predictedValues contain
		numerical values, rather than categorical data.
	"""
	_validatePredictedAsLabels(predictedValues)
	return _computeError(knownValues, predictedValues, lambda x,y,z: z + abs(y - x), lambda x,y: x/y)

def fractionIncorrect(knownValues, predictedValues):
	"""
		Compute the proportion of incorrect predictions within a set of
		instances.  Assumes that values in knownValues and predictedValues are categorical.
	"""
	_validatePredictedAsLabels(predictedValues)
	return _computeError(knownValues, predictedValues, lambda x,y,z: z if x == y else z + 1, lambda x,y: x/y)

def fractionIncorrectBottom10(knownValues, labelScoreList, negativeLabel):
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
	positiveLabelString = ''
	if len(labelNames) != 2:
		raise ArgumentException("fractionIncorrectNegative requires a set of precisely two predicted label scores for each point")

	if labelScoreList.getTypeString() == "Matrix":
		negativeLabelString = str(float(negativeLabel))
	else:
		negativeLabelString = str(negativeLabel)

	for labelName in labelNames.keys():
		if labelName == negativeLabelString:
			continue
		else:
			positiveLabelString = labelName

	negativeLabelIndex = labelScoreList.featureNames[negativeLabelString]
	positiveLabelIndex = labelScoreList.featureNames[positiveLabelString]

	#Compute the score difference (positive label score - negative label score) for
	#all entries in labelScoreList
	scoreDifferenceList = []
	labelScoreList = labelScoreList.copyAs(format="python list")
	for i in range(len(labelScoreList)):
		positiveScore = labelScoreList[i][positiveLabelIndex]
		negativeScore = labelScoreList[i][negativeLabelIndex]
		scoreDiff = positiveScore - negativeScore
		scoreDifferenceList.append(scoreDiff)

	#convert knownValues to list of lists; drop first row, which has featureNames
	listOfKnownLabels = knownValues.copyAs(format="python list")
#	knownLabels = listOfKnownLabels[0:][0]
	knownLabels = listOfKnownLabels[0:]
	for i in range(len(knownLabels)):
		knownLabels[i] = knownLabels[i][0]

	#Put together score differences and known labels, then sort by score difference,
	#so we have a list ranked, in descending order, by most likely to have negative label
	scoreDiffAndKnown = zip(scoreDifferenceList, knownLabels)

	scoreDiffAndKnown.sort(key=lambda score: score[0])

	#Get bottom of list (lowest score differences, which may be negative)
	topProportionIndex = int(round(0.10 * len(scoreDiffAndKnown)))
	sortedTopProportion = scoreDiffAndKnown[0:topProportionIndex]

	#Unzip into two lists
	sortedScoreDiffAndKnown = ([scoreDiff for scoreDiff,known in sortedTopProportion], [known for scoreDiff,known in sortedTopProportion])

	#get newly sorted known labels
	sortedKnownValues = sortedScoreDiffAndKnown[1]

	#compute number of negative labels present in specified proportion of posts that
	#are predicted to be more likely to be positive labels
	winningLabels = []
	for scoreDiff in sortedScoreDiffAndKnown[0]:
		if scoreDiff <= 0.0:
			winningLabels.append(negativeLabelString)
		else:
			winningLabels.append(positiveLabelString)

	incorrectPredictions = 0
	for i in range(len(winningLabels)):
		if str(sortedKnownValues[i]) != winningLabels[i]:
			incorrectPredictions += 1

	proportionCorrect = float(incorrectPredictions) / float(len(sortedKnownValues))

	return proportionCorrect

