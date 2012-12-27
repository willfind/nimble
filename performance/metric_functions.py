import numpy

from math import sqrt
from ..utility import ArgumentException


def rmse(predictedValues, actualValues):
	"""
		Compute the root mean square error
	"""
	n=0
	runningSum=0.0
	for i in xrange(predictedValues.numRows()):
		pV = predictedValues[i,0]
		aV = actualValues[i,0]
		runningSum += (pV - aV)**2
		n += 1
	if n > 0:
		runningSum /= n
	else:
        raise ArgumentException("Empty argument(s) in rmse calculator")

    return math.sqrt(runningSum)

def meanAbsoluteError(predictedValues, actualValues):
	"""

	"""
	n=0
	runningSum=0.0
	for i in xrange(predictedValues.numRows()):
		pV = predictedValues[i,0]
		aV = actualValues[i,0]
		runningSum += abs(pV - aV)
		n += 1
	if n > 0:
		runningSum /= n
	else:
		raise ArgumentException("Empty argument(s) in mean absolute error calculator")

	return runningSum

def error(predictedValues, actualValues):
	"""

	"""
	n=0
	runningSum=0.0
	for i in xrange(predictedValues.numRows()):
		pV = predictedValues[i,0]
		aV = actualValues[i,0]
		if pV != aV:
			runningSum += 1.0
		n += 1
	if n > 0:
		runningSum /= n
	else:
		raise ArgumentException("Empty argument(s) in error calculator")

	return runningSum

