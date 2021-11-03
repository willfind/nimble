import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.calculate import meanAbsoluteError
from nimble.calculate import rootMeanSquareError
from nimble.calculate import meanFeaturewiseRootMeanSquareError
from nimble.calculate import fractionCorrect
from nimble.calculate import fractionIncorrect
from nimble.calculate import rSquared
from nimble.calculate import varianceFractionRemaining
from nimble.calculate.loss import _computeError
from nimble.calculate.utility import performanceFunction
from tests.helpers import raises
from tests.helpers import noLogEntryExpected

#######################
# performanceFunction #
#######################

@performanceFunction('min')
def generic(knownValues, predictedValues):
    return 1

def testGenericPerformanceFunctionEmpty():
    """
    Test that performanceFunction raises an exception if data is empty
    """
    emptyArray = np.array([])
    nonEmptyArray = np.array([1, 2, 3])
    emptyMatrix = nimble.data('Matrix', source=emptyArray)
    nonEmptyMatrix = nimble.data('Matrix', source=nonEmptyArray)

    with raises(InvalidArgumentValue):
        perf = generic(emptyMatrix, emptyMatrix)

    with raises(InvalidArgumentValue):
        perf = generic(nonEmptyMatrix, emptyMatrix)

    with raises(InvalidArgumentValue):
        perf = generic(emptyMatrix, nonEmptyMatrix)

@raises(InvalidArgumentValue)
def testGenericPerformanceFunctionNansNoData():
    """
    Test that performanceFunction raises an exception if nans prevent calculation.
    """
    knownLabels = np.array([1, np.nan, 3])
    predictedLabels = np.array([np.nan, 3, np.nan])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    perf = generic(knownLabelsMatrix, predictedLabelsMatrix)

def testGenericPerformanceFunctionInvalidInputs():
    """
    Test that performanceFunction raises an exception for invalid types
    """
    with raises(InvalidArgumentType):
        knownLabels = np.array([[1], [2], [3]])
        predictedLabels = np.array([[1], [2], [3]])

        predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

        perf = generic(knownLabels, predictedLabelsMatrix)

    with raises(InvalidArgumentType):
        knownLabels = np.array([[1], [2], [3]])
        predictedLabels = np.array([[1], [2], [3]])

        knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)

        perf = generic(knownLabelsMatrix, predictedLabels)

def testGenericPerformanceFunctionShapeMismatch():
    """
    Test that performanceFunction raises an exception if shapes don't match
    """
    with raises(InvalidArgumentValueCombination):
        knownLabels = np.array([[1], [2], [3]])
        predictedLabels = np.array([[1], [2]])

        knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
        predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

        perf = generic(knownLabelsMatrix, predictedLabelsMatrix)

    with raises(InvalidArgumentValueCombination):
        knownLabels = np.array([[1], [2]])
        predictedLabels = np.array([[1], [2], [3]])

        knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
        predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

        perf = generic(knownLabelsMatrix, predictedLabelsMatrix)

    with raises(InvalidArgumentValueCombination):
        knownLabels = np.array([[1, 1], [2, 2], [3, 3]])
        predictedLabels = np.array([[1], [2], [3]])

        knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
        predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

        perf = generic(knownLabelsMatrix, predictedLabelsMatrix)

    with raises(InvalidArgumentValueCombination):
        knownLabels = np.array([[1], [2], [3]])
        predictedLabels = np.array([[1, 1], [2, 2], [3, 3]])

        knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
        predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

        perf = generic(knownLabelsMatrix, predictedLabelsMatrix)


def testGenericPerformanceFunction2D():
    """
    Test that performanceFunction checks allowed input shape
    """

    with raises(InvalidArgumentValue):

        knownLabels = np.array([[1, 2], [2, 3], [3, 4]])
        predictedLabels = np.array([[1, 2], [2, 3], [3, 4]])

        knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
        predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

        perf = generic(knownLabelsMatrix, predictedLabelsMatrix)

    @performanceFunction('min', requires1D=False)
    def generic2D(knownValues, predictedValues):
        return 1

    knownLabels = np.array([[1, 2], [2, 3], [3, 4]])
    predictedLabels = np.array([[1, 2], [2, 3], [3, 4]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    perf = generic2D(knownLabelsMatrix, predictedLabelsMatrix)
    assert perf == 1

def testGenericPerformanceFunctionWithValidData():
    """
    Test that performanceFunction wrapper works as expected
    """
    knownLabels = [[1.0], [3.0], [0.0], [4.0]]
    predictedLabels = [[1.0], [2.0], [0.0], [3.0]]

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    @performanceFunction('min')
    def performance(knownValues, predictedValues):
        return _computeError(knownValues, predictedValues,
                             lambda x, y, z: z + (x - y), lambda x, y: x / y)

    assert generic.optimal == 'min'
    perf = performance(knownLabelsMatrix, predictedLabelsMatrix)
    assert perf == 0.5

def testGenericPerformanceFunctionSkipValidation():
    """
    Test that a performance function that does not require validation
    """
    knownLabels = np.array([[1, 2], [3, 4]])
    predictedLabels = np.array([[1], [2], [3]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    @performanceFunction('max', validate=False)
    def noValidation(knownValues, predictedValues):
        return 1

    assert noValidation.optimal == 'max'
    perf = noValidation(knownLabelsMatrix, predictedLabelsMatrix)
    assert perf == 1

#################
# _computeError #
#################

@raises(ZeroDivisionError)
def testGenericErrorCalculatorDivideByZero():
    """
    Test that _computeError raises a divide by zero exception if the
    outerFunction argument would lead to division by zero.
    """
    knownLabels = np.array([1, 2, 3])
    predictedLabels = np.array([1, 2, 3])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    _computeError(knownLabelsMatrix, predictedLabelsMatrix,
                  lambda x, y, z: z, lambda x, y: y / x)


def testGenericErrorCalculator():
    knownLabels = np.array([1.0, 2.0, 3.0])
    predictedLabels = np.array([1.0, 2.0, 3.0])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    sameRate = _computeError(
        knownLabelsMatrix, predictedLabelsMatrix,
        lambda x, y, z: z, lambda x, y: x)
    assert sameRate == 0.0


#######################
# Mean Absolute Error #
#######################

@noLogEntryExpected
def testMeanAbsoluteError():
    """
    Check that the mean absolute error calculator works correctly when
    all inputs are zero, or predictions are exactly the same as all known
    values, and are non-zero
    """
    assert meanAbsoluteError.optimal == 'min'

    predictedLabels = np.array([0, 0, 0])
    knownLabels = np.array([0, 0, 0])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate == 0.0

    predictedLabels = np.array([1.0, 2.0, 3.0])
    knownLabels = np.array([1.0, 2.0, 3.0])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate == 0.0

    predictedLabels = np.array([1.0, 2.0, 3.0])
    knownLabels = np.array([1.5, 2.5, 3.5])

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
@noLogEntryExpected
def testRmse():
    """
    Check that the rootMeanSquareError calculator works correctly when
    all inputs are zero, and when all known values are
    the same as predicted values.
    """
    assert rootMeanSquareError.optimal == 'min'

    predictedLabels = np.array([[0], [0], [0]])
    knownLabels = np.array([[0], [0], [0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.5], [2.5], [3.5]])

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
    predictedLabels = np.array([[0, 2], [0, 2], [0, 2], [0, 2]])
    knownLabels = np.array([[0, 0], [0, 0], [0, 0]])

    knowns = nimble.data('Matrix', source=knownLabels)
    predicted = nimble.data('Matrix', source=predictedLabels)

    meanFeaturewiseRootMeanSquareError(knowns, predicted)


@raises(InvalidArgumentValueCombination)
def testMFRMSE_badshapeFeatures():
    predictedLabels = np.array([[0], [0], [0], [0]])
    knownLabels = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    knowns = nimble.data('Matrix', source=knownLabels)
    predicted = nimble.data('Matrix', source=predictedLabels)

    meanFeaturewiseRootMeanSquareError(knowns, predicted)

@noLogEntryExpected
def testMFRMSE_simpleSuccess():
    predictedLabels = np.array([[0, 2], [0, 2], [0, 2], [0, 2]])
    knownLabels = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    knowns = nimble.data('Matrix', source=knownLabels, useLog=False)
    predicted = nimble.data('Matrix', source=predictedLabels, useLog=False)

    mfrmseRate = meanFeaturewiseRootMeanSquareError(knowns, predicted)
    assert mfrmseRate == 1.0

###################
# fractionCorrect #
###################

@noLogEntryExpected
def testFractionCorrect():
    """
    Check that the fractionCorrect calculator works correctly when
    all inputs are zero, and when all known values are
    the same as predicted values.
    """
    assert fractionCorrect.optimal == 'max'

    predictedLabels = np.array([[0], [0], [0]])
    knownLabels = np.array([[0], [0], [0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 1.0

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 1.0

    predictedLabels = np.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = np.array([[1.0], [2.0], [1.0], [2.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 0.5

#####################
# fractionIncorrect #
#####################

@noLogEntryExpected
def testFractionIncorrect():
    """
    Check that the fractionIncorrect calculator works correctly when
    all inputs are zero, and when all known values are
    the same as predicted values.
    """
    assert fractionIncorrect.optimal == 'min'

    predictedLabels = np.array([[0], [0], [0]])
    knownLabels = np.array([[0], [0], [0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = np.array([[1.0], [2.0], [1.0], [2.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.5

#############################
# varianceFractionRemaining #
#############################

@noLogEntryExpected
def testVarianceFractionRemaining():
    """
    Check that the varianceFractionRemaining calculator works correctly.
    """
    assert varianceFractionRemaining.optimal == 'min'

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)
    assert vfr == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = np.array([[1.0], [2.0], [4.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)
    assert vfr == 0.4

############
# rSquared #
############

@noLogEntryExpected
def testRSquared():
    """
    Check that the rSquared calculator works correctly.
    """
    assert rSquared.optimal == 'max'

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)
    assert rsq == 1.0

    predictedLabels = np.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = np.array([[1.0], [2.0], [4.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels,
                                        useLog=False)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)
    assert rsq == 0.6
