import math
import scipy.stats
import numpy

import UML
from UML.exceptions import ArgumentException

def confidenceIntervalHelper(errors, transform, confidence=0.95):
	"""Helper to calculate the confidence interval, given a vector of errors
	and a monotonic transform to be applied after the calculation. 

	"""
	if errors.featureCount != 1:
		raise ArgumentException("The errors vector may only have one feature")

	if transform is None:
		wrappedTransform = lambda x: x
	# we want to ensure that the transform will scale negative values, even
	# if negative values are not within its domain
	else:
		def wrappedTransform(value):
			isNegative = value < 0
			pos = abs(value)
			ret = transform(pos)
			if isNegative:
				ret = -ret
			return ret

	halfConfidence = 1 - ((1-confidence)/2.0)
	boundaryOnNormalScale = scipy.stats.norm.ppf(halfConfidence)

	sqrtN = float(math.sqrt(errors.pointCount))
	mean = errors.featureStatistics('mean')[0,0]
	std = errors.featureStatistics('SampleStandardDeviation')[0,0]

	low = wrappedTransform(mean - (boundaryOnNormalScale * (std / sqrtN)))
	high = wrappedTransform(mean + (boundaryOnNormalScale * (std / sqrtN)))

	return (low, high)


def rootMeanSquareErrorConfidenceInterval(known, predicted, confidence=0.95):
	errors = known - predicted
	errors.elementwisePower(2)
	return confidenceIntervalHelper(errors, math.sqrt, confidence)

def meanAbsoluteErrorConfidenceInterval(known, predicted, confidence=0.95):
	errors = known - predicted
	return confidenceIntervalHelper(errors, None, confidence)

def fractionIncorrectConfidenceInterval(known, predicted, confidence=0.95):
	rawErrors = known.copyAs('numpyarray') - predicted.copyAs('numpyarray')
	rawErrors = numpy.array((-(known == predicted)), dtype=int)
	errors = UML.createData("Matrix", rawErrors)

	return confidenceIntervalHelper(errors, None, confidence)
