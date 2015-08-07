
import inspect
import numpy
import math

import UML
from UML.exceptions import ArgumentException
from UML.randomness import numpyRandom



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

	if len(args) != 2:
		msg = "functionToCheck takes wrong number of parameters, unable to do "
		msg += "detection"
		raise ArgumentException()

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
					best = _runTrialGivenParameters(functionToCheck, knowns.copy(), predicted.copy(), predictionType)
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


def _runTrialGivenParameters(toCheck, knowns, predicted, predictionType):
	"""
	

	"""
	allCorrectScore = toCheck(knowns, predicted)
	# this is going to hold the output of the function. Each value will
	# correspond to a trial that contains incrementally less correct predicted values
	scoreList = [allCorrectScore]
	# range over the indices of predicted values, making them incorrect one
	# by one
	# TODO randomize the ordering
	for index in xrange(predicted.pointCount):
		_makeIncorrect(predicted, predictionType, index)
		scoreList.append(toCheck(knowns, predicted))

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
	correct = UML.createData(returnType="List", data=correct)
	return correct

def _generateAllOnes(length):
	correct = []
	for i in xrange(length):
		correct.append(1)
	correct = numpy.array(correct)
#	correct = numpy.ones(length, dtype=int)
	correct = numpy.matrix(correct)
	correct = correct.transpose()
	correct = UML.createData(returnType="List", data=correct)
	return correct

def _generateAllCorrect(length):
	while True:
		correct = numpyRandom.randint(2, size=length)
		if numpy.any(correct) and not numpy.all(correct):
			break
#	correct = numpyRandom.randint(2, size=length)
	correct = numpy.matrix(correct)
	correct = correct.transpose()
	correct = UML.createData(returnType="List", data=correct)
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
		scores = numpyRandom.randint(2, size=workingCopy.pointCount)
		scores = numpy.matrix(scores)
		scores = scores.transpose()
		scores = UML.createData(returnType="List", data=scores, featureNames=['LabelScore'])
		workingCopy.appendFeatures(scores)
		return workingCopy
	else:
		dataToFill = []
		for i in xrange(workingCopy.pointCount):
			currConfidences = [None,None]
			winner = numpyRandom.randint(10) + 10 + 2
			loser = numpyRandom.randint(winner - 2) + 2 
			if knowns.data[i][0] == 0:
				currConfidences[0] = winner
				currConfidences[1] = loser
			else:
				currConfidences[0] = loser
				currConfidences[1] = winner
			dataToFill.append(currConfidences)

		scores = UML.createData(returnType="List", data=dataToFill, featureNames=['0', '1'])
		return scores


def _makeIncorrect(predicted, predictionType, index):
	if predictionType in [0,1]:
		predicted.data[index][0] = math.fabs(predicted.data[index][0] - 1)
	else:
		temp = predicted.data[index][0]
		predicted.data[index][0] = predicted.data[index][1]
		predicted.data[index][1] = temp


