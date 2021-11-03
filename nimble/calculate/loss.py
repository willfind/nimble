"""
Definitions for functions that can be used as performance functions by
nimble. Specifically, this only contains those functions that measure
loss; or in other words, those where smaller values indicate a higher
level of correctness in the predicted values.
"""

from math import sqrt

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from .utility import performanceFunction


def _computeError(knownValues, predictedValues, loopFunction,
                  compressionFunction):
    """
    A generic function to compute different kinds of error metrics.
    knownValues is a 1d Base object with one known label (or number) per
    row. predictedValues is a 1d Base object with one predictedLabel
    (or score) per row.  The ith row in knownValues should refer to the
    same point as the ith row in predictedValues. loopFunction is a
    function to be applied to each row in knownValues/predictedValues,
    that takes 3 arguments: a known class label, a predicted label, and
    runningTotal, which contains the successive output of loopFunction.
    compressionFunction is a function that should take two arguments:
    runningTotal, the final output of loopFunction, and n, the number of
    values in knownValues/predictedValues.
    """
    n = 0.0
    runningTotal = 0.0
    # Go through all values in known and predicted values, and pass those
    # values to loopFunction
    for i in range(len(predictedValues.points)):
        pVal = predictedValues[i, 0]
        aVal = knownValues[i, 0]
        runningTotal = loopFunction(aVal, pVal, runningTotal)
        n += 1
    if n > 0:
        try:
            #provide the final value from loopFunction to compressionFunction,
            #along with the number of values looped over
            runningTotal = compressionFunction(runningTotal, n)
        except ZeroDivisionError as e:
            msg = 'Tried to divide by zero when calculating performance metric'
            raise ZeroDivisionError(msg) from e

    else:
        raise InvalidArgumentValue("Empty argument(s) in error calculator")

    return runningTotal


@performanceFunction('min')
def rootMeanSquareError(knownValues, predictedValues):
    """
    Compute the root mean square error.  Assumes that knownValues and
    predictedValues contain numerical values, rather than categorical
    data.
    """
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z + (y - x) ** 2,
                         lambda x, y: sqrt(x / y))

@performanceFunction('min', requires1D=False)
def meanFeaturewiseRootMeanSquareError(knownValues, predictedValues):
    """
    For 2d prediction data, compute the RMSE of each feature, then
    average the results.
    """
    if len(knownValues.features) != len(predictedValues.features):
        msg = "The known and predicted data must have the same number of "
        msg += "features"
        raise InvalidArgumentValueCombination(msg)

    results = []
    for i in range(len(knownValues.features)):
        currKnown = knownValues.features.copy(i, useLog=False)
        currPred = predictedValues.features.copy(i, useLog=False)
        results.append(rootMeanSquareError(currKnown, currPred))

    return float(sum(results)) / len(knownValues.features)

@performanceFunction('min')
def meanAbsoluteError(knownValues, predictedValues):
    """
    Compute mean absolute error. Assumes that knownValues and
    predictedValues contain numerical values, rather than categorical
    data.
    """
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z + abs(y - x),
                         lambda x, y: x / y)

@performanceFunction('min')
def fractionIncorrect(knownValues, predictedValues):
    """
    Compute the proportion of incorrect predictions within a set of
    instances.  Assumes that values in knownValues and predictedValues
    are categorical.
    """
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z if x == y else z + 1,
                         lambda x, y: x / y)

@performanceFunction('min')
def varianceFractionRemaining(knownValues, predictedValues):
    """
    Calculate the how much variance has not been correctly predicted in
    the predicted values. This will be equal to
    1 - nimble.calculate.rSquared() of the same inputs.
    """
    diffObj = predictedValues - knownValues

    avgSqDiff = diffObj.T.matrixMultiply(diffObj)[0, 0] / float(len(diffObj))

    return avgSqDiff / float(nimble.calculate.variance(knownValues, False))
