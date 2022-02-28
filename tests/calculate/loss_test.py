import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.calculate import meanAbsoluteError
from nimble.calculate import rootMeanSquareError
from nimble.calculate import meanFeaturewiseRootMeanSquareError
from nimble.calculate import fractionCorrect
from nimble.calculate import fractionIncorrect
from nimble.calculate import rSquared
from nimble.calculate import varianceFractionRemaining
from nimble.calculate.loss import _computeError
from tests.helpers import raises
from tests.helpers import noLogEntryExpected

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

    knownLabelsMatrix = nimble.data(source=knownLabels)
    predictedLabelsMatrix = nimble.data(source=predictedLabels)

    _computeError(knownLabelsMatrix, predictedLabelsMatrix,
                  lambda x, y, z: z, lambda x, y: y / x)


def testGenericErrorCalculator():
    knownLabels = np.array([1.0, 2.0, 3.0])
    predictedLabels = np.array([1.0, 2.0, 3.0])

    knownLabelsMatrix = nimble.data(source=knownLabels)
    predictedLabelsMatrix = nimble.data(source=predictedLabels)

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

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate == 0.0

    predictedLabels = np.array([1.0, 2.0, 3.0])
    knownLabels = np.array([1.0, 2.0, 3.0])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate == 0.0

    predictedLabels = np.array([1.0, 2.0, 3.0])
    knownLabels = np.array([1.5, 2.5, 3.5])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels, useLog=False)
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

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.5], [2.5], [3.5]])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
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

    knowns = nimble.data(source=knownLabels)
    predicted = nimble.data(source=predictedLabels)

    meanFeaturewiseRootMeanSquareError(knowns, predicted)


@raises(InvalidArgumentValueCombination)
def testMFRMSE_badshapeFeatures():
    predictedLabels = np.array([[0], [0], [0], [0]])
    knownLabels = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    knowns = nimble.data(source=knownLabels)
    predicted = nimble.data(source=predictedLabels)

    meanFeaturewiseRootMeanSquareError(knowns, predicted)

@noLogEntryExpected
def testMFRMSE_simpleSuccess():
    predictedLabels = np.array([[0, 2], [0, 2], [0, 2], [0, 2]])
    knownLabels = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    knowns = nimble.data(source=knownLabels, useLog=False)
    predicted = nimble.data(source=predictedLabels, useLog=False)

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

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 1.0

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 1.0

    predictedLabels = np.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = np.array([[1.0], [2.0], [1.0], [2.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
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

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0]])
    knownLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = np.array([[1.0], [2.0], [1.0], [2.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
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

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)
    assert vfr == 0.0

    predictedLabels = np.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = np.array([[1.0], [2.0], [4.0], [3.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
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

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)
    assert rsq == 1.0

    predictedLabels = np.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = np.array([[1.0], [2.0], [4.0], [3.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels,
                                        useLog=False)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)
    assert rsq == 0.6
