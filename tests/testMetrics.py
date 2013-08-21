import numpy
import math
from nose.tools import *
from UML.metrics import rootMeanSquareError, fractionIncorrect, computeError
from UML.metrics import meanAbsoluteError, fractionTrueNegativeTop50
from UML.metrics import fractionTrueNegativeTop90, fractionTrueNegativeBottom10
from UML.metrics import fractionIncorrectBottom10, detectBestResult, cosineSimilarity
from UML.umlHelpers import computeMetrics
from UML import createData
from UML.exceptions import ArgumentException


def testfractionIncorrectBottom10SanityCheck():
	"""An all correct and all incorrec check on fractionIncorrectBottom10 """
	knownsData = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
	correctData = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
	wrongData = [[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]]

	knowns = createData('List', knownsData, sendToLog=False)
	correct = createData('List', correctData, featureNames=['0','1'], sendToLog=False)
	wrong = createData('List', wrongData, featureNames=['0','1'], sendToLog=False)

	correctScore = fractionIncorrectBottom10(knowns, correct, negativeLabel=0)
	wrongScore = fractionIncorrectBottom10(knowns, wrong, negativeLabel=0)

	assert correctScore == 0
	assert wrongScore == 1

def testFractionTrueNegative():
	"""
	Unit test for fractionTrueNegativeTop50/Top90
	"""
	knownLabelsOne = [[1], [2], [2], [2], [1], [1], [1], [2], [2], [2], [1], [2], [2], [2], [1], [1], [1], [2], [2], [2]]
	knownLabelsTwo = [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
	knownLabelsThree = [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]]
	knownLabelsFour = [[2], [1], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]]

	knownLabelsOneBase = createData('Matrix', knownLabelsOne, sendToLog=False)
	knownLabelsTwoBase = createData('Matrix', knownLabelsTwo, sendToLog=False)
	knownLabelsThreeBase = createData('Matrix', knownLabelsThree, sendToLog=False)
	knownLabelsFourBase = createData('Matrix', knownLabelsFour, sendToLog=False)

	predictedScoreList = []
	for i in range(20):
		oneScore = i * 0.05
		twoScore = 1.0 - i * 0.05
		predictedScoreList.append([oneScore, twoScore])

	predictedScoreListBase = createData('Matrix', predictedScoreList, ['1.0', '2.0'])

	topHalfProportionNegativeOne = fractionTrueNegativeTop50(knownLabelsOneBase, predictedScoreListBase, negativeLabel=1)
	topNinetyProportionNegativeOne = fractionTrueNegativeTop90(knownLabelsOneBase, predictedScoreListBase, negativeLabel=1)
	topHalfProportionNegativeTwo = fractionTrueNegativeTop50(knownLabelsTwoBase, predictedScoreListBase, negativeLabel=1)
	topNinetyProportionNegativeTwo = fractionTrueNegativeTop90(knownLabelsTwoBase, predictedScoreListBase, negativeLabel=1)
	topHalfProportionNegativeThree = fractionTrueNegativeTop50(knownLabelsThreeBase, predictedScoreListBase, negativeLabel=1)
	topNinetyProportionNegativeThree = fractionTrueNegativeTop90(knownLabelsThreeBase, predictedScoreListBase, negativeLabel=1)
	topHalfProportionNegativeFour = fractionTrueNegativeTop50(knownLabelsFourBase, predictedScoreListBase, negativeLabel=1)
	topNinetyProportionNegativeFour = fractionTrueNegativeTop90(knownLabelsFourBase, predictedScoreListBase, negativeLabel=1)
	
#	print topHalfProportionNegativeOne
	assert topHalfProportionNegativeOne == 0.4
	assert topNinetyProportionNegativeOne >= 0.443 and topNinetyProportionNegativeOne <= 0.445
	assert topHalfProportionNegativeTwo == 0.0
	assert topNinetyProportionNegativeTwo >= 0.443 and topNinetyProportionNegativeTwo <= 0.445
	assert topHalfProportionNegativeThree == 0.0
	assert topNinetyProportionNegativeThree == 0.0
	assert topHalfProportionNegativeFour == 0.10
	assert topNinetyProportionNegativeFour >= 0.0554 and topNinetyProportionNegativeFour <= 0.0556


#####################################
# performance combinations function #
#####################################
def testPerfCombinations():
	knownLabels = numpy.array([[1.0],[2.0],[3.0]])
	predictedLabels = numpy.array([[1.0],[2.0],[3.0]])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	metricFunctions = [rootMeanSquareError, meanAbsoluteError, fractionIncorrect]
	results = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
#	print results
	assert results[0] == 0.0
	assert results[1] == 0.0
	assert results[2] == 0.0

	knownLabels = numpy.array([[1.5],[2.5],[3.5]])
	predictedLabels = numpy.array([[1.0],[2.0],[3.0]])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	metricFunctions = [rootMeanSquareError, meanAbsoluteError, fractionIncorrect]
	results = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
	assert results[0] > 0.49
	assert results[0] < 0.51
	assert results[1] > 0.49
	assert results[1] < 0.51

############################
# generic error calculator #
############################
@raises(ArgumentException)
def testGenericErrorCalculatorEmptyKnownInput():
	"""
		Test that computeError raises an exception if knownLabels is empty
	"""
	knownLabels = numpy.array([])
	predictedLabels = numpy.array([1,2,3])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x,y,z: z, lambda x,y: x)

@raises(ArgumentException)
def testGenericErrorCalculatorEmptyPredictedInput():
	"""
		Test that computeError raises an exception if predictedLabels is empty
	"""
	knownLabels = numpy.array([1,2,3])
	predictedLabels = numpy.array([])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x,y,z: z, lambda x,y: x)

@raises(ZeroDivisionError)
def testGenericErrorCalculatorDivideByZero():
	"""
		Test that computeError raises a divide by zero exception if the outerFunction argument
		would lead to division by zero.
	"""
	knownLabels = numpy.array([1,2,3])
	predictedLabels = numpy.array([1,2,3])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x,y,z: z, lambda x,y: y/x)

def testGenericErrorCalculator():
	knownLabels = numpy.array([1.0, 2.0, 3.0])
	predictedLabels = numpy.array([1.0, 2.0, 3.0])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	sameRate = computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x,y,z: z, lambda x,y: x)
	assert sameRate == 0.0

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

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	rootMeanSquareErrorRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)

@raises(ArgumentException)
def testRmseEmptyPredictedValues():
	"""
		Check that the rootMeanSquareError calculator correctly throws an
		exception if predictedLabels vector is empty
	"""
	predictedLabels = numpy.array([])
	knownLabels = numpy.array([1, 2, 3])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)


def testRmse():
	"""
		Check that the rootMeanSquareError calculator works correctly when
		all inputs are zero, and when all known values are
		the same as predicted values.
	"""
	predictedLabels = numpy.array([[0],[0],[0]])
	knownLabels = numpy.array([[0],[0],[0]])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
	assert rmseRate == 0.0

	predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
	knownLabels = numpy.array([[1.0], [2.0], [3.0]])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
	assert rmseRate == 0.0

	predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
	knownLabels = numpy.array([[1.5], [2.5], [3.5]])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
	assert rmseRate > 0.49
	assert rmseRate < 0.51

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

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)

@raises(ArgumentException)
def testMeanAbsoluteErrorEmptyPredictedValues():
	"""
		Check that the mean absolute error calculator correctly throws an
		exception if predictedLabels vector is empty
	"""
	predictedLabels = numpy.array([])
	knownLabels = numpy.array([1, 2, 3])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)

def testMeanAbsoluteError():
	"""
		Check that the mean absolute error calculator works correctly when
		all inputs are zero, or predictions are exactly the same as all known
		values, and are non-zero
	"""
	predictedLabels = numpy.array([0,0,0])
	knownLabels = numpy.array([0,0,0])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	predictedLabelsMatrix = createData('Matrix', predictedLabels)
	knownLabelsMatrix.transpose()
	predictedLabelsMatrix.transpose()

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
	assert maeRate == 0.0

	predictedLabels = numpy.array([1.0, 2.0, 3.0])
	knownLabels = numpy.array([1.0, 2.0, 3.0])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	knownLabelsMatrix.transpose()
	predictedLabelsMatrix = createData('Matrix', predictedLabels)
	predictedLabelsMatrix.transpose()

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
	assert maeRate == 0.0

	predictedLabels = numpy.array([1.0, 2.0, 3.0])
	knownLabels = numpy.array([1.5, 2.5, 3.5])

	knownLabelsMatrix = createData('Matrix', knownLabels)
	knownLabelsMatrix.transpose()
	predictedLabelsMatrix = createData('Matrix', predictedLabels)
	predictedLabelsMatrix.transpose()

	maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
	assert maeRate > 0.49
	assert maeRate < 0.51


########################
# Classification Error #
########################

#@raises(ArgumentException)
#def testClassificationErrorEmptyKnownValues():

####################
# cosineSimilarity #
####################

def test_cosineSimilarity():
	orig = numpy.array([[1],[0]])
	orth = numpy.array([[0],[1]])
	neg = numpy.array([[-1],[0]])

	origMatrix = createData('Matrix', orig)
	orthMatrix = createData('Matrix', orth)
	negMatrix = createData('Matrix', neg)

	result0 = cosineSimilarity(origMatrix, origMatrix)
	result1 = cosineSimilarity(origMatrix, orthMatrix)
	result2 = cosineSimilarity(origMatrix, negMatrix)

	assert result0 == 1
	assert result1 == 0
	assert result2 == -1

def test_cosineSimilarityZeros():
	zeros = [[0],[0]]

	zerosMatrix = createData('Matrix', zeros)

	result0 = cosineSimilarity(zerosMatrix, zerosMatrix)

	assert math.isnan(result0)

####################
# detectBestResult #
####################

def testDectection_rootMeanSquareError():
	result = detectBestResult(rootMeanSquareError)
	assert result == 'min'

def testDectection_meanAbsoluteError():
	result = detectBestResult(meanAbsoluteError)
	assert result == 'min'

def testDectection_fractionIncorrect():
	result = detectBestResult(fractionIncorrect)
	assert result == 'min'

def testDetection_cosineSimilarity():
	result = detectBestResult(cosineSimilarity)
	assert result == 'max'

@raises(ArgumentException)
def testDectection_fractionTrueNegativeTop90():
	result = detectBestResult(fractionTrueNegativeTop90)

@raises(ArgumentException)
def testDectection_fractionTrueNegativeTop50():
	result = detectBestResult(fractionTrueNegativeTop50)

@raises(ArgumentException)
def testDectection_fractionTrueNegativeBottom10():
	result = detectBestResult(fractionTrueNegativeBottom10)

def testDectection_fractionIncorrectBottom10():
	result = detectBestResult(fractionIncorrectBottom10)
	assert result == 'min'
