import numpy
from nose.tools import *
from ..metric_functions import *

###########################
# Root mean squared error #
###########################


@raises(ArgumentException)
def testRmseEmptyKnownValues():
	"""
		Check that the rmse calculator correctly throws an
		exception if knownLabels vector is empty
	"""
	knownLabels = array([])
	predictedLabels = array([1, 2, 3])

	rmseRate = rmse(predictedLabels, knownLabels)

@raises(ArgumentException)
def testRmseEmptyPredictedValues():
	"""
		Check that the rmse calculator correctly throws an
		exception if predictedLabels vector is empty
	"""
	predictedLabels = array([])
	knownLabels = array([1, 2, 3])

	rmseRate = rmse(predictedLabels, knownLabels)


def testRmseAllZero():
	"""
		Check that the rmse calculator works correctly when
		all inputs are zero.
	"""
	predictedLabels = array([0,0,0])
	knownLabels = array([0,0,0])

	rmseRate = rmse(predictedLabels, knownLabels)
	assert rmseRate == 0.0

def testRmseAllRight():
	"""
		Check that the rmse calculator works correctly if all
		predictions are exactly the same as all known values,
		and are non-zero
	"""
	predictedLabels = array([1.0, 2.0, 3.0])
	knownLabels = array([1.0, 2.0, 3.0])

	rmseRate = rmse(predictedLabels, knownLabels)
	assert rmseRate == 0.0


#######################
# Mean Absolute Error #
#######################
@raises(ArgumentException)
def testMeanAbsoluteErrorEmptyKnownValues():
	"""
		Check that the mean absolute error calculator correctly throws an
		exception if knownLabels vector is empty
	"""
	knownLabels = array([])
	predictedLabels = array([1, 2, 3])

	maeRate = meanAbsoluteError(predictedLabels, knownLabels)

@raises(ArgumentException)
def testMeanAbsoluteErrorEmptyPredictedValues():
	"""
		Check that the mean absolute error calculator correctly throws an
		exception if predictedLabels vector is empty
	"""
	predictedLabels = array([])
	knownLabels = array([1, 2, 3])

	maeRate = meanAbsoluteError(predictedLabels, knownLabels)

def testMeanAbsoluteErrorAllZero():
	"""
		Check that the mean absolute error calculator works correctly when
		all inputs are zero.
	"""
	predictedLabels = array([0,0,0])
	knownLabels = array([0,0,0])

	maeRate = meanAbsoluteError(predictedLabels, knownLabels)
	assert maeRate == 0.0

def testMeanAbsoluteErrorAllRight():
	"""
		Check that the rmse calculator works correctly if all
		predictions are exactly the same as all known values,
		and are non-zero
	"""
	predictedLabels = array([1.0, 2.0, 3.0])
	knownLabels = array([1.0, 2.0, 3.0])

	maeRate = meanAbsoluteError(predictedLabels, knownLabels)
	assert maeRate == 0.0

