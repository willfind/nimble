import numpy
from nose.tools import *

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.calculate import meanAbsoluteError
from nimble.calculate import rootMeanSquareError
from nimble.calculate import meanFeaturewiseRootMeanSquareError
from nimble.calculate import fractionCorrect
from nimble.calculate import fractionIncorrect
from nimble.calculate import rSquared
from nimble.calculate import varianceFractionRemaining
from tests.helpers import noLogEntryExpected

#################
# _computeError #
#################

@raises(InvalidArgumentValue)
def testGenericErrorCalculatorEmptyKnownInput():
    """
    Test that _computeError raises an exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    nimble.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x, y, z: z, lambda x, y: x)


@raises(InvalidArgumentValue)
def testGenericErrorCalculatorEmptyPredictedInput():
    """
    Test that _computeError raises an exception if predictedLabels is empty
    """
    knownLabels = numpy.array([1, 2, 3])
    predictedLabels = numpy.array([])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    nimble.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x, y, z: z, lambda x, y: x)


@raises(ZeroDivisionError)
def testGenericErrorCalculatorDivideByZero():
    """
    Test that _computeError raises a divide by zero exception if the
    outerFunction argument would lead to division by zero.
    """
    knownLabels = numpy.array([1, 2, 3])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    nimble.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x, y, z: z, lambda x, y: y / x)


def testGenericErrorCalculator():
    knownLabels = numpy.array([1.0, 2.0, 3.0])
    predictedLabels = numpy.array([1.0, 2.0, 3.0])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    sameRate = nimble.calculate.loss._computeError(
        knownLabelsMatrix, predictedLabelsMatrix, lambda x, y, z: z, lambda x, y: x)
    assert sameRate == 0.0


#######################
# Mean Absolute Error #
#######################

@raises(InvalidArgumentValue)
def testMeanAbsoluteErrorEmptyKnownValues():
    """
    Check that the mean absolute error calculator correctly throws an
    exception if knownLabels vector is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValue)
def testMeanAbsoluteErrorEmptyPredictedValues():
    """
    Check that the mean absolute error calculator correctly throws an
    exception if predictedLabels vector is empty
    """
    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testMeanAbsoluteError():
    """
    Check that the mean absolute error calculator works correctly when
    all inputs are zero, or predictions are exactly the same as all known
    values, and are non-zero
    """
    predictedLabels = numpy.array([0, 0, 0])
    knownLabels = numpy.array([0, 0, 0])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate == 0.0

    predictedLabels = numpy.array([1.0, 2.0, 3.0])
    knownLabels = numpy.array([1.0, 2.0, 3.0])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate == 0.0

    predictedLabels = numpy.array([1.0, 2.0, 3.0])
    knownLabels = numpy.array([1.5, 2.5, 3.5])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels, useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate > 0.49
    assert maeRate < 0.51


###########################
# Root mean squared error #
###########################

@raises(InvalidArgumentValue)
def testRmseEmptyKnownValues():
    """
    rootMeanSquareError calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    rootMeanSquareErrorRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValue)
def testRmseEmptyPredictedValues():
    """
    rootMeanSquareError calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testRmse():
    """
    Check that the rootMeanSquareError calculator works correctly when
    all inputs are zero, and when all known values are
    the same as predicted values.
    """
    predictedLabels = numpy.array([[0], [0], [0]])
    knownLabels = numpy.array([[0], [0], [0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.5], [2.5], [3.5]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate > 0.49
    assert rmseRate < 0.51


######################################
# meanFeaturewiseRootMeanSquareError #
######################################

@raises(InvalidArgumentValueCombination)
def testMFRMSE_badshapePoints():
    predictedLabels = numpy.array([[0, 2], [0, 2], [0, 2], [0, 2]])
    knownLabels = numpy.array([[0, 0], [0, 0], [0, 0]])

    knowns = nimble.data('Matrix', source=knownLabels)
    predicted = nimble.data('Matrix', source=predictedLabels)

    meanFeaturewiseRootMeanSquareError(knowns, predicted)


@raises(InvalidArgumentValueCombination)
def testMFRMSE_badshapeFeatures():
    predictedLabels = numpy.array([[0], [0], [0], [0]])
    knownLabels = numpy.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    knowns = nimble.data('Matrix', source=knownLabels)
    predicted = nimble.data('Matrix', source=predictedLabels)

    meanFeaturewiseRootMeanSquareError(knowns, predicted)

@noLogEntryExpected
def testMFRMSE_simpleSuccess():
    predictedLabels = numpy.array([[0, 2], [0, 2], [0, 2], [0, 2]])
    knownLabels = numpy.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    knowns = nimble.data('Matrix', source=knownLabels, useLog=False)
    predicted = nimble.data('Matrix', source=predictedLabels, useLog=False)

    mfrmseRate = meanFeaturewiseRootMeanSquareError(knowns, predicted)
    assert mfrmseRate == 1.0

###################
# fractionCorrect #
###################

@raises(InvalidArgumentValue)
def testFractionCorrectEmptyKnownValues():
    """
    fractionCorrect calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValue)
def testFractionCorrectEmptyPredictedValues():
    """
    fractionCorrect calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testFractionCorrect():
    """
    Check that the fractionCorrect calculator works correctly when
    all inputs are zero, and when all known values are
    the same as predicted values.
    """
    predictedLabels = numpy.array([[0], [0], [0]])
    knownLabels = numpy.array([[0], [0], [0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 1.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 1.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = numpy.array([[1.0], [2.0], [1.0], [2.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 0.5

#####################
# fractionIncorrect #
#####################

@raises(InvalidArgumentValue)
def testFractionIncorrectEmptyKnownValues():
    """
    fractionIncorrect calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValue)
def testFractionIncorrectEmptyPredictedValues():
    """
    fractionIncorrect calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testFractionIncorrect():
    """
    Check that the fractionIncorrect calculator works correctly when
    all inputs are zero, and when all known values are
    the same as predicted values.
    """
    predictedLabels = numpy.array([[0], [0], [0]])
    knownLabels = numpy.array([[0], [0], [0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = numpy.array([[1.0], [2.0], [1.0], [2.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.5

#############################
# varianceFractionRemaining #
#############################

@raises(InvalidArgumentValueCombination)
def testVarianceFractionRemainingEmptyKnownValues():
    """
    varianceFractionRemaining calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValueCombination)
def testVarianceFractionRemainingEmptyPredictedValues():
    """
    varianceFractionRemaining calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testVarianceFractionRemaining():
    """
    Check that the varianceFractionRemaining calculator works correctly.
    """
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)
    assert vfr == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = numpy.array([[1.0], [2.0], [4.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)
    assert vfr == 0.4

############
# rSquared #
############

@raises(InvalidArgumentValueCombination)
def testRSquaredEmptyKnownValues():
    """
    rSquared calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValueCombination)
def testRSquaredEmptyPredictedValues():
    """
    rSquared calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testRSquared():
    """
    Check that the rSquared calculator works correctly.
    """
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)
    assert rsq == 1.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = numpy.array([[1.0], [2.0], [4.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)
    assert rsq == 0.6
