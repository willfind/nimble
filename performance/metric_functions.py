from math import sqrt
from ..utility.custom_exceptions import ArgumentException


def computeError(knownValues, predictedValues, loopFunction, compressionFunction):
	"""
		A generic function to compute different kinds of error metrics.  knownValues
		is a numpy array with the
	"""
	if knownValues is None or predictedValues is None or knownValues.data.size == 0 or predictedValues.data.size == 0:
		raise ArgumentException("Empty argument(s) in error calculator")

	n=0.0
	runningTotal=0.0
	for i in xrange(predictedValues.points()):
		pV = predictedValues.data[i,0]
		aV = knownValues.data[i,0]
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
