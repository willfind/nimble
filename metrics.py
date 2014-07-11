"""
Contains various error metrics to be used for evaluating the results
of supervised learners.


"""
import inspect
import UML
import numpy
import math
from math import sqrt
from UML.exceptions import ArgumentException
from UML.umlHelpers import computeError

from UML.umlRandom import npRandom


def _validatePredictedAsLabels(predictedValues):
	if not isinstance(predictedValues, UML.data.Base):
		raise ArgumentException("predictedValues must be derived class of UML.data.Base")
	if predictedValues.featureCount > 1:
		raise ArgumentException("predictedValues must be labels only; this has more than one feature")


def cosineSimilarity(knownValues, predictedValues):
	_validatePredictedAsLabels(predictedValues)

	known = knownValues.copyAs(format="numpy array").flatten()
	predicted = predictedValues.copyAs(format="numpy array").flatten()

	numerator = (numpy.dot(known, predicted))
	denominator = (numpy.linalg.norm(known) * numpy.linalg.norm(predicted))

	return numerator / denominator

def rootMeanSquareError(knownValues, predictedValues):
	"""
		Compute the root mean square error.  Assumes that knownValues and predictedValues contain
		numerical values, rather than categorical data.
	"""
	_validatePredictedAsLabels(predictedValues)
	return computeError(knownValues, predictedValues, lambda x,y,z: z + (y - x)**2, lambda x,y: sqrt(x/y))

def meanAbsoluteError(knownValues, predictedValues):
	"""
		Compute mean absolute error. Assumes that knownValues and predictedValues contain
		numerical values, rather than categorical data.
	"""
	_validatePredictedAsLabels(predictedValues)
	return computeError(knownValues, predictedValues, lambda x,y,z: z + abs(y - x), lambda x,y: x/y)

def fractionIncorrect(knownValues, predictedValues):
	"""
		Compute the proportion of incorrect predictions within a set of
		instances.  Assumes that values in knownValues and predictedValues are categorical.
	"""
	_validatePredictedAsLabels(predictedValues)
	return computeError(knownValues, predictedValues, lambda x,y,z: z if x == y else z + 1, lambda x,y: x/y)

def fractionTrueNegativeTop90(knownValues, predictedValues, negativeLabel):
	"""
		Wrapper function for fractionTrueNegative.  Computes the proportion
		of posts that fall in the 90% most likely to be in the positive class
		that are actually in the negative class.  Assumes there are only 2 classes
		in the data set, and that predictedValues contains a score for each of
		the 2 labels.  Sorts by (positive label score - negative label score), in
		ascending order, and looks at highest 90% of values.
				
	"""
	return fractionTrueNegative(knownValues, predictedValues, negativeLabel, proportionToScore=0.90)

def fractionTrueNegativeTop50(knownValues, predictedValues, negativeLabel):
	"""
		Wrapper function for fractionTrueNegative.  Computes the proportion
		of posts that fall in the 50% most likely to be in the positive class
		that are actually in the negative class.  Assumes there are only 2 classes
		in the data set, and that predictedValues contains a score for each of
		the 2 labels.  Sorts by (positive label score - negative label score), in
		ascending order, and looks at highest 50% of values.
	"""
	return fractionTrueNegative(knownValues, predictedValues, negativeLabel, proportionToScore=0.50)

def fractionTrueNegativeBottom10(knownValues, predictedValues, negativeLabel):
	"""
		Wrapper function for fractionTrueNegative.  Computes the proportion
		of posts that fall in the 50% most likely to be in the positive class
		that are actually in the negative class.  Assumes there are only 2 classes
		in the data set, and that predictedValues contains a score for each of
		the 2 labels.  Sorts by (positive label score - negative label score), in
		ascending order, and looks at lowest 10% of values.
	"""
	return fractionTrueNegative(knownValues, predictedValues, negativeLabel, proportionToScore=0.10, reverseSort=True)

def fractionTrueNegative(knownValues, labelScoreList, negativeLabel, proportionToScore=0.90, reverseSort=False):
	"""
		Computes the proportion
		of posts that fall in the x% most likely to be in the positive class
		that are actually in the negative class.  Assumes there are only 2 classes
		in the data set, and that predictedValues contains a score for each of
		the 2 labels.  Sorts by (positive label score - negative label score), in
		ascending order, and looks at highest x proportion of values, where x is defined
		by proportionToScore.  If reverseSort is True, looks at lowest x proportion of values.
	"""
	#proportion must fall between 0 and 1
	if proportionToScore <= 0.0 or proportionToScore > 1.0:
		raise ArgumentException("proportionToScore must be between 0 and 1.0")

	if labelScoreList.getTypeString() == "Matrix":
		negativeLabelString = str(float(negativeLabel))
	else:
		negativeLabelString = str(negativeLabel)

	#use featureNames in labelScoreList to discover what the positiveLabelString is
	labelNames = labelScoreList.featureNames
	positiveLabelString = ''

	#Enforce requirement that there be only 2 classes present
	if len(labelNames) != 2:
		raise ArgumentException("fractionTrueNegative requires a set of precisely two predicted label scores for each point")

	#look through featureNames; whichever isn't the negative label must be
	#the positive label
	for labelName in labelNames.keys():
		if labelName == negativeLabelString:
			continue
		else:
			positiveLabelString = labelName
			break

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

	#convert knownValues to list of lists
	listOfKnownLabels = knownValues.copyAs(format="python list")
	knownLabels = listOfKnownLabels[0:]
	for i in range(len(knownLabels)):
		knownLabels[i] = knownLabels[i][0]

	#Put together score differences and known labels, then sort by score difference,
	#so we have a list ranked by likelihood of having positiveLabelString.  Generally will
	#be in descending order, so we can look at those points that are most likely to 
	#be positive.  
	scoreDiffAndKnown = zip(scoreDifferenceList, knownLabels)

	if reverseSort is True:
		scoreDiffAndKnown.sort(key=lambda score: score[0])
	else:
		scoreDiffAndKnown.sort(key=lambda score: score[0], reverse=True)

	#Get some proportion of list based on proportionToScore
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
		if knownLabel == negativeLabel:
			numNegativeLabels += 1

	#return proportion of top posts that are negative
	return float(numNegativeLabels) / float(len(sortedKnownValues))

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
		raise ArgumentException("fractionTrueNegative requires a set of precisely two predicted label scores for each point")

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



def detectBestResult(functionToCheck):
	"""
	Provides sample data to the function in question and evaluates the results
	to determine whether the returned value associates correctness with
	minimum values or maximum values.

	functionToCheck may only take two or three arguments. In the two argument
	case, the first must be a vector of desired values and the second must be
	a vector of predicted values. In the second case, the first argument must
	be a vector of known labes, the second argument must be an object
	containing confidence scores for different labels, and the third argument
	must be the value of a label value present in the data. In either cause,
	the functions must return a float value.

	"""
	(args, varargs, keywords, defaults) = inspect.getargspec(functionToCheck)
	# indicates whether functionToCheck takes a 3rd param: negativeLabel
	# wrapper function used to call functionToCheck by providing all of the
	# parameters it might need for the 3 param case, even if functionToCheck
	# is only takes knowns and predicted values.
	wrapper = None

	if len(args) == 2:
		def twoParam(knowns, predicted, negLabel):
			return functionToCheck(knowns, predicted)
		wrapper = twoParam
	elif len(args) == 3:
		def threeParam(knowns, predicted, negLabel):
			return functionToCheck(knowns, predicted, negLabel)
		wrapper = threeParam
	else:
		raise ArgumentException("functionToCheck takes wrong number of parameters, unable to do detection")

	resultsByType = [None, None, None]
	trialSize = 10
	# we can't know which format is expected in the predicted values param,
	# so we try all three and then interpret the results
	# 0 if predicted labels, 1 if bestScores, 2 if allScores
	for predictionType in range(3):
		try:
			bestResultList = []
			# going to run trials with three types of known values: all ones, all
			# zeros, and some mixture
			for knownsType in xrange(4):
				if knownsType == 0:
					knowns = _generateAllZeros(trialSize)
				elif knownsType == 1:
					knowns = _generateAllOnes(trialSize)
				else:
					knowns = _generateAllCorrect(trialSize)
				# range over fixed confidence values for a trial
				confidenceTrials = 7
				# in a trial with predicted labels, there are no confidences, so only 1
				# trial is needed
				if predictionType == 0:
					confidenceTrials = 1
				for i in xrange(confidenceTrials):
					predicted = _generatePredicted(knowns, predictionType)
					# range over possible negative label value
					for negLabel in xrange(2):
						best = _runTrialGivenParameters(wrapper, knowns.copy(), predicted.copy(), negLabel, predictionType)
						bestResultList.append(best)

			# check that all of values in bestResultList are equal.
			firstNonNan = None
			for result in bestResultList:
				if result != 'nan':
					if firstNonNan is None:
						firstNonNan = result
					elif result != firstNonNan:
						assert False
						raise ArgumentException("functionToCheck may not be a performance function. The best result was inverted due to factors other than correctness")
			resultsByType[predictionType] = firstNonNan
		except Exception as e:
			resultsByType[predictionType] = e

	best = None
	for result in resultsByType:
		if not isinstance(result, Exception):
			if best is not None and result != best:
				raise ArgumentException("Unable to determine formatting for 2nd parameter to funciton to check, different inputs gave inconsistent results. The best solution would be to verify the formatting of the input and throw an exception if it is not as expected")
			best = result

	if best is None:
		raise ArgumentException("Unable to determine formatting for 2nd parameter to funciton to check, none of the possible inputs produced accepted return values. We must conclude that it is not a performance function")

	return best


def _runTrialGivenParameters(functionWrapper, knowns, predicted, negativeLabel, predictionType):
	"""
	

	"""
	allCorrectScore = functionWrapper(knowns, predicted, negativeLabel)
	# this is going to hold the output of the function. Each value will
	# correspond to a trial that contains incrementally less correct predicted values
	scoreList = [allCorrectScore]
	# range over the indices of predicted values, making them incorrect one
	# by one
	# TODO randomize the ordering
	for index in xrange(predicted.pointCount):
		_makeIncorrect(predicted, predictionType, index)
		scoreList.append(functionWrapper(knowns, predicted, negativeLabel))

	# defining our error message in case of unexpected scores
	errorMsg = "functionToCheck produces return values that do not "
	errorMsg += "correspond with correctness. We must conclude that it "
	errorMsg += "is not a performance function"	
	allWrongScore = scoreList[len(scoreList)-1]

	for score in scoreList:
		if math.isnan(score):
			return "nan"

	if allWrongScore < allCorrectScore:
		prevScore = allCorrectScore + 1
		for score in scoreList:
			if score < allWrongScore or score > allCorrectScore or score > prevScore:
				raise ArgumentException(errorMsg)
			prevScore = score
		return "max"
	elif allWrongScore > allCorrectScore:
		prevScore = allCorrectScore - 1
		for score in scoreList:
			if score > allWrongScore or score < allCorrectScore or score < prevScore:
				raise ArgumentException(errorMsg)
			prevScore = score
		return "min"
	# allWrong and allCorrect must not be equal, otherwise it cannot be a measure
	# of correct performance
	else:
		raise ArgumentException("functionToCheck produced the same values for trials including all correct and all incorrect predictions. We must conclude that it is not a performance function")


def _generateAllZeros(length):
	correct = []
	for i in xrange(length):
		correct.append(0)
	correct = numpy.array(correct)
#	correct = numpy.zeros(length, dtype=int)
	correct = numpy.matrix(correct)
	correct = correct.transpose()
	correct = UML.createData(retType="List", data=correct)
	return correct

def _generateAllOnes(length):
	correct = []
	for i in xrange(length):
		correct.append(1)
	correct = numpy.array(correct)
#	correct = numpy.ones(length, dtype=int)
	correct = numpy.matrix(correct)
	correct = correct.transpose()
	correct = UML.createData(retType="List", data=correct)
	return correct

def _generateAllCorrect(length):
	while True:
		correct = npRandom.randint(2, size=length)
		if numpy.any(correct) and not numpy.all(correct):
			break
#	correct = npRandom.randint(2, size=length)
	correct = numpy.matrix(correct)
	correct = correct.transpose()
	correct = UML.createData(retType="List", data=correct)
	return correct

def _generatePredicted(knowns, predictionType):
	"""
	Predicted may mean any of the three kinds of output formats for trainAndApply:
	predicted labels, bestScores, or allScores. If confidences are involved, they are
	randomly generated, yet consistent with correctness

	"""
	workingCopy = knowns.copy()
	workingCopy.setFeatureName(0,'PredictedClassLabel')
	if predictionType == 0:	
		return workingCopy
	elif predictionType == 1:
		scores = npRandom.randint(2, size=workingCopy.pointCount)
		scores = numpy.matrix(scores)
		scores = scores.transpose()
		scores = UML.createData(retType="List", data=scores, featureNames=['LabelScore'])
		workingCopy.appendFeatures(scores)
		return workingCopy
	else:
		dataToFill = []
		for i in xrange(workingCopy.pointCount):
			currConfidences = [None,None]
			winner = npRandom.randint(10) + 10 + 2
			loser = npRandom.randint(winner - 2) + 2 
			if knowns.data[i][0] == 0:
				currConfidences[0] = winner
				currConfidences[1] = loser
			else:
				currConfidences[0] = loser
				currConfidences[1] = winner
			dataToFill.append(currConfidences)

		scores = UML.createData(retType="List", data=dataToFill, featureNames=['0', '1'])
		return scores


def _makeIncorrect(predicted, predictionType, index):
	if predictionType in [0,1]:
		predicted.data[index][0] = math.fabs(predicted.data[index][0] - 1)
	else:
		temp = predicted.data[index][0]
		predicted.data[index][0] = predicted.data[index][1]
		predicted.data[index][1] = temp



