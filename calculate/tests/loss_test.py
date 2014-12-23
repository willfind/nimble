
import numpy

from nose.tools import *

import UML
from UML import createData
from UML.exceptions import ArgumentException
from UML.calculate import meanAbsoluteError
from UML.calculate import rootMeanSquareError
from UML.calculate import fractionIncorrectBottom10

def testfractionIncorrectBottom10SanityCheck():
	"""An all correct and all incorrec check on fractionIncorrectBottom10 """
	knownsData = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
	correctData = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
	wrongData = [[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]]

	knowns = createData('List', data=knownsData, sendToLog=False)
	correct = createData('List', data=correctData, featureNames=['0','1'], sendToLog=False)
	wrong = createData('List', data=wrongData, featureNames=['0','1'], sendToLog=False)

	correctScore = fractionIncorrectBottom10(knowns, correct, negativeLabel=0)
	wrongScore = fractionIncorrectBottom10(knowns, wrong, negativeLabel=0)

	assert correctScore == 0
	assert wrongScore == 1


#################
# _computeError #
#################

@raises(ArgumentException)
def testGenericErrorCalculatorEmptyKnownInput():
	"""
		Test that _computeError raises an exception if knownLabels is empty
	"""
	knownLabels = numpy.array([])
	predictedLabels = numpy.array([1,2,3])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	UML.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x,y,z: z, lambda x,y: x)

@raises(ArgumentException)
def testGenericErrorCalculatorEmptyPredictedInput():
	"""
		Test that _computeError raises an exception if predictedLabels is empty
	"""
	knownLabels = numpy.array([1,2,3])
	predictedLabels = numpy.array([])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	UML.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x,y,z: z, lambda x,y: x)

@raises(ZeroDivisionError)
def testGenericErrorCalculatorDivideByZero():
	"""
		Test that _computeError raises a divide by zero exception if the outerFunction argument
		would lead to division by zero.
	"""
	knownLabels = numpy.array([1,2,3])
	predictedLabels = numpy.array([1,2,3])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	UML.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x,y,z: z, lambda x,y: y/x)

def testGenericErrorCalculator():
	knownLabels = numpy.array([1.0, 2.0, 3.0])
	predictedLabels = numpy.array([1.0, 2.0, 3.0])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	sameRate = UML.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x,y,z: z, lambda x,y: x)
	assert sameRate == 0.0




#######################
# Mean Absolute Error #
#######################

@raises(ArgumentException)
def testMeanAbsoluteErrorEmptyKnownValues():
	"""
		Check that the mean absolute error calculator correctly throws an
		exception if knownLabels vector is empty
	"""
	knownLabels = numpy.array([])
	predictedLabels = numpy.array([1, 2, 3])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)

@raises(ArgumentException)
def testMeanAbsoluteErrorEmptyPredictedValues():
	"""
		Check that the mean absolute error calculator correctly throws an
		exception if predictedLabels vector is empty
	"""
	predictedLabels = numpy.array([])
	knownLabels = numpy.array([1, 2, 3])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)

def testMeanAbsoluteError():
	"""
		Check that the mean absolute error calculator works correctly when
		all inputs are zero, or predictions are exactly the same as all known
		values, and are non-zero
	"""
	predictedLabels = numpy.array([0,0,0])
	knownLabels = numpy.array([0,0,0])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)
	knownLabelsMatrix.transpose()
	predictedLabelsMatrix.transpose()

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
	assert maeRate == 0.0

	predictedLabels = numpy.array([1.0, 2.0, 3.0])
	knownLabels = numpy.array([1.0, 2.0, 3.0])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	knownLabelsMatrix.transpose()
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)
	predictedLabelsMatrix.transpose()

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
	assert maeRate == 0.0

	predictedLabels = numpy.array([1.0, 2.0, 3.0])
	knownLabels = numpy.array([1.5, 2.5, 3.5])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	knownLabelsMatrix.transpose()
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)
	predictedLabelsMatrix.transpose()

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
	assert maeRate > 0.49
	assert maeRate < 0.51


###########################
# Root mean squared error #
###########################

@raises(ArgumentException)
def testRmseEmptyKnownValues():
	"""
		Check that the rootMeanSquareError calculator correctly throws an
		exception if knownLabels vector is empty
	"""
	knownLabels = numpy.array([])
	predictedLabels = numpy.array([1, 2, 3])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	rootMeanSquareErrorRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)

@raises(ArgumentException)
def testRmseEmptyPredictedValues():
	"""
		Check that the rootMeanSquareError calculator correctly throws an
		exception if predictedLabels vector is empty
	"""
	predictedLabels = numpy.array([])
	knownLabels = numpy.array([1, 2, 3])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)


def testRmse():
	"""
		Check that the rootMeanSquareError calculator works correctly when
		all inputs are zero, and when all known values are
		the same as predicted values.
	"""
	predictedLabels = numpy.array([[0],[0],[0]])
	knownLabels = numpy.array([[0],[0],[0]])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
	assert rmseRate == 0.0

	predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
	knownLabels = numpy.array([[1.0], [2.0], [3.0]])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
	assert rmseRate == 0.0

	predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
	knownLabels = numpy.array([[1.5], [2.5], [3.5]])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
	assert rmseRate > 0.49
	assert rmseRate < 0.51
