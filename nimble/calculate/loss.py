"""
Definitions for functions that can be used as performance functions by
nimble. Specifically, this only contains those functions that measure
loss; or in other words, those where smaller values indicate a higher
level of correctness in the predicted values.
"""

from math import sqrt

import numpy

import nimble
from nimble.data import Base
from nimble.data import Matrix
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination

def _validatePredictedAsLabels(predictedValues):
    if not isinstance(predictedValues, nimble.data.Base):
        msg = "predictedValues must be derived class of nimble.data.Base"
        raise InvalidArgumentType(msg)
    if len(predictedValues.features) > 1:
        msg = "predictedValues must be labels only; this has more than "
        msg += "one feature"
        raise InvalidArgumentValue(msg)


def _computeError(knownValues, predictedValues, loopFunction, compressionFunction):
    """
    A generic function to compute different kinds of error metrics.  knownValues
    is a 1d Base object with one known label (or number) per row. predictedValues is a 1d Base
    object with one predictedLabel (or score) per row.  The ith row in knownValues should refer
    to the same point as the ith row in predictedValues. loopFunction is a function to be applied
    to each row in knownValues/predictedValues, that takes 3 arguments: a known class label,
    a predicted label, and runningTotal, which contains the successive output of loopFunction.
    compressionFunction is a function that should take two arguments: runningTotal, the final
    output of loopFunction, and n, the number of values in knownValues/predictedValues.
    """
    knownIsEmpty = len(knownValues.points) == 0 or len(knownValues.features) == 0
    predIsEmpty = len(predictedValues.points) == 0 or len(predictedValues.features) == 0
    if knownValues is None or not isinstance(knownValues, Base):
        msg = "knownValues must be derived class of nimble.data.Base"
        raise InvalidArgumentType(msg)
    if knownIsEmpty:
        msg = "Empty 'knownValues' object in error calculator"
        raise InvalidArgumentValue(msg)
    if predictedValues is None or not isinstance(predictedValues, Base):
        msg = "predictedValues must be derived class of nimble.data.Base"
        raise InvalidArgumentType(msg)
    if predIsEmpty:
        msg = "Empty 'predictedValues' object in error calculator"
        raise InvalidArgumentValue(msg)

    if len(knownValues.points) != len(predictedValues.points):
        msg = "The knownValues and predictedValues must have the same number "
        msg += "of points"
        raise InvalidArgumentValueCombination(msg)

    if len(knownValues.features) != len(predictedValues.features):
        msg = "The knownValues and predictedValues must have the same number "
        msg += "of features"
        raise InvalidArgumentValueCombination(msg)

    if not isinstance(knownValues, Matrix):
        knownValues = knownValues.copy(to="Matrix")

    if not isinstance(predictedValues, Matrix):
        predictedValues = predictedValues.copy(to="Matrix")

    n = 0.0
    runningTotal = 0.0
    #Go through all values in known and predicted values, and pass those values to loopFunction
    for i in range(len(predictedValues.points)):
        pV = predictedValues[i, 0]
        aV = knownValues[i, 0]
        runningTotal = loopFunction(aV, pV, runningTotal)
        n += 1
    if n > 0:
        try:
            #provide the final value from loopFunction to compressionFunction, along with the
            #number of values looped over
            runningTotal = compressionFunction(runningTotal, n)
        except ZeroDivisionError:
            raise ZeroDivisionError('Tried to divide by zero when calculating performance metric')

    else:
        raise InvalidArgumentValue("Empty argument(s) in error calculator")

    return runningTotal


def rootMeanSquareError(knownValues, predictedValues):
    """
    Compute the root mean square error.  Assumes that knownValues and
    predictedValues contain numerical values, rather than categorical
    data.
    """
    _validatePredictedAsLabels(predictedValues)
    return _computeError(knownValues, predictedValues, lambda x, y, z: z + (y - x) ** 2, lambda x, y: sqrt(x / y))


rootMeanSquareError.optimal = 'min'

def meanFeaturewiseRootMeanSquareError(knownValues, predictedValues):
    """
    For 2d prediction data, compute the RMSE of each feature, then
    average the results.
    """
    if len(knownValues.features) != len(predictedValues.features):
        raise InvalidArgumentValueCombination("The known and predicted data must have the same number of features")
    if len(knownValues.points) != len(predictedValues.points):
        raise InvalidArgumentValueCombination("The known and predicted data must have the same number of points")

    results = []
    for i in range(len(knownValues.features)):
        currKnown = knownValues.features.copy(i, useLog=False)
        currPred = predictedValues.features.copy(i, useLog=False)
        results.append(rootMeanSquareError(currKnown, currPred))

    return float(sum(results)) / len(knownValues.features)


meanFeaturewiseRootMeanSquareError.optimal = 'min'


def meanAbsoluteError(knownValues, predictedValues):
    """
        Compute mean absolute error. Assumes that knownValues and predictedValues contain
        numerical values, rather than categorical data.
    """
    _validatePredictedAsLabels(predictedValues)
    return _computeError(knownValues, predictedValues, lambda x, y, z: z + abs(y - x), lambda x, y: x / y)


meanAbsoluteError.optimal = 'min'


def fractionIncorrect(knownValues, predictedValues):
    """
    Compute the proportion of incorrect predictions within a set of
    instances.  Assumes that values in knownValues and predictedValues
    are categorical.
    """
    _validatePredictedAsLabels(predictedValues)
    return _computeError(knownValues, predictedValues, lambda x, y, z: z if x == y else z + 1, lambda x, y: x / y)


fractionIncorrect.optimal = 'min'


def varianceFractionRemaining(knownValues, predictedValues):
    """
    Calculate the how much variance is has not been correctly predicted in the
    predicted values. This will be equal to 1 - nimble.calculate.rsquared() of
    the same inputs.
    """
    if len(knownValues.points) != len(predictedValues.points):
        raise InvalidArgumentValueCombination("Objects had different numbers of points")
    if len(knownValues.features) != len(predictedValues.features):
        raise InvalidArgumentValueCombination(
        "Objects had different numbers of features. Known values had " + str(
            len(knownValues.features)) + " and predicted values had " + str(len(predictedValues.features)))
    diffObject = predictedValues - knownValues
    rawDiff = diffObject.copy(to="numpy array")
    rawKnowns = knownValues.copy(to="numpy array")
    assert rawDiff.shape[1] == 1
    avgSqDif = numpy.dot(rawDiff.T, rawDiff)[0, 0] / float(len(rawDiff))
    return avgSqDif / float(numpy.var(rawKnowns))


varianceFractionRemaining.optimal = 'min'
