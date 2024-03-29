
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

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


@performanceFunction('min', 0)
def rootMeanSquareError(knownValues, predictedValues):
    """
    Compute the root mean square error.  Assumes that knownValues and
    predictedValues contain numerical values, rather than categorical
    data.
    """
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z + (y - x) ** 2,
                         lambda x, y: sqrt(x / y))

@performanceFunction('min', 0, requires1D=False)
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

@performanceFunction('min', 0)
def meanAbsoluteError(knownValues, predictedValues):
    """
    Compute mean absolute error. Assumes that knownValues and
    predictedValues contain numerical values, rather than categorical
    data.
    """
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z + abs(y - x),
                         lambda x, y: x / y)

@performanceFunction('min', 0)
def fractionIncorrect(knownValues, predictedValues):
    """
    Compute the proportion of incorrect predictions within a set of
    instances.  Assumes that values in knownValues and predictedValues
    are categorical.
    """
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z if x == y else z + 1,
                         lambda x, y: x / y)

@performanceFunction('min', 0)
def varianceFractionRemaining(knownValues, predictedValues):
    """
    Calculate the how much variance has not been correctly predicted in
    the predicted values. This will be equal to
    1 - nimble.calculate.rSquared() of the same inputs.
    """
    diffObj = predictedValues - knownValues

    avgSqDiff = diffObj.T.matrixMultiply(diffObj)[0, 0] / float(len(diffObj))

    return avgSqDiff / float(nimble.calculate.variance(knownValues, False))

# TODO: Work in progress - performance metrics for clustering
# NOTES:
#    1) Is this a good metric? Will averageDistanceToClusterCenter always
#       imporove when increasing the number of clusters?
#       Better options are probably silhoutte score or davies-bouldin
#    2) I do not think that the validation protocols used for tuning are setup
#       for unsupervised learning. They always expect Y data.
#    3) If usable, a test is commented out in tests/calculate/loss

# def _predictedClusterCenters(trainedLearner, knownValues, arguments=None):
#     """
#     Return the cluster center for each point in the knownValues.
#
#     Will use the stored cluster centers if possible, otherwise will
#     calculate them.
#     """
#     clusters = trainedLearner.apply(knownValues, arguments, useLog=False)
#     centers = trainedLearner.getAttributes().get('cluster_centers_', None)
#     if centers is None:
#         clustered = knownValues.copy()
#         clustered.features.append(clusters, useLog=False)
#         centers = {i: cluster.features.statistics('mean') for i, cluster
#                    in clustered.groupByFeature(-1, useLog=False).items()}
#     else:
#         centers = {i: center for i, center in enumerate(centers)}
#
#     return nimble.data([centers[i] for i in clusters], useLog=False)

# @performanceFunction('min', 0, predict=_predictedClusterCenters,
#                      requires1D=False)
# def averageDistanceToClusterCenter(knownValues, clusterCenters):
#     """
#     The average euclidean distance of the clusters from their center.
#
#     The knownValues and clusterCenters should have the same shape. The
#     clusterCenters must be the center of the cluster assigned to each
#     each point.
#     """
#     squaredDiff = (knownValues - clusterCenters) ** 2
#     rootSquaredSum = squaredDiff.points.calculate(lambda pt: sqrt(sum(pt)),
#                                                   useLog=False)
#
#     return sum(rootSquaredSum) / len(knownValues.points)
