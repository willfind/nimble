
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
Similarity calculations.
"""

from math import sqrt

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.calculate.loss import fractionIncorrect
from nimble.calculate.loss import varianceFractionRemaining
from nimble.core.data._dataHelpers import createDataNoValidation
from .utility import performanceFunction


@performanceFunction('max', 1)
def cosineSimilarity(knownValues, predictedValues):
    """
    Calculate the cosine similarity between known and predicted values.
    """
    numerator = knownValues.T.matrixMultiply(predictedValues)[0, 0]
    denominator = (sqrt(sum(knownValues ** 2))
                   * sqrt(sum(predictedValues ** 2)))

    return numerator / denominator


def _nanCovCorr(X, X_T=None, sample=True, corr=False):
    """
    Calculate the covariance or correlation between points in X when
    the data contains missing values. If X_T is not provided, a copy of
    X will be made in this function.
    """
    # pylint: disable=invalid-name
    if X_T is None:
        X_T = X.T

    arrX = X.copy('numpy array').astype(float)
    arrXT = X_T.copy('numpy array').astype(float)
    maskX = np.isfinite(arrX)
    maskXT = np.isfinite(arrXT)

    if maskX.all() and maskXT.all(): # no missing data
        means = np.mean(arrX, axis=1, keepdims=True)
        results = np.dot(arrX - means, arrXT - means.T)
        if sample:
            results /= (arrX.shape[1] - 1)
        else:
            results /= arrX.shape[1]
        if corr:
            results /= np.dot(np.std(arrX, axis=1, keepdims=True),
                              np.std(arrXT, axis=0, keepdims=True))
    else: # contains missing values
        numPts = len(arrX)
        results = np.empty((numPts,) * 2)

        for i in range(numPts):
            for j in range(i, numPts):
                use = maskX[i] & maskXT[:, j]
                if not use.any():
                    val = np.nan
                else:
                    row = arrX[i][use]
                    col = arrXT[:, j][use]
                    val = np.dot(row - np.mean(row), col - np.mean(col))
                    if sample:
                        denominator = len(row) - 1
                    else:
                        denominator = len(row)
                    if denominator:
                        val /= denominator
                    else:
                        val = np.nan
                    if corr:
                        corrDenominator = (np.std(row) * np.std(col))
                        if corrDenominator:
                            val /= corrDenominator
                        else:
                            val = np.nan

                results[i, j] = val
                if i != j:
                    results[j, i] = val

    names = X.points._getNamesNoGeneration()
    return createDataNoValidation(X.getTypeString(), results, pointNames=names,
                                  featureNames=names)

def correlation(X, X_T=None):
    """
    Calculate the Pearson correlation coefficients between points in X.

    If X_T is not provided, a copy of X will be made in this function.
    """
    # pylint: disable=invalid-name
    return _nanCovCorr(X, X_T, False, True)

def covariance(X, X_T=None, sample=True):
    """
    Calculate the covariance between points in X.

    If X_T is not provided, a copy of X will be made in this function.
    """
    # pylint: disable=invalid-name
    return _nanCovCorr(X, X_T, sample)

# The following two performance functions make calls to validated functions
# so do not repeat the validation.
@performanceFunction('max', 1, validate=False)
def fractionCorrect(knownValues, predictedValues):
    """
    Calculate how many values in predictedValues are equal to the
    values in the corresponding positions in knownValues. The return
    will be a float between 0 and 1 inclusive.
    """
    return 1 - fractionIncorrect(knownValues, predictedValues)

@performanceFunction('max', 1, validate=False)
def rSquared(knownValues, predictedValues):
    """
    Calculate the r-squared (or coefficient of determination) of the
    predictedValues given the knownValues. This will be equal to 1 -
    nimble.calculate.varianceFractionRemaining() of the same inputs.
    """
    return 1.0 - varianceFractionRemaining(knownValues, predictedValues)


def confusionMatrix(knownValues, predictedValues, labels=None,
                    convertCountsToFractions=False):
    """
    Generate a confusion matrix for known and predicted label values.

    The confusion matrix contains the counts of observations that
    occurred for each known/predicted pair. Features represent the
    known classes and points represent the predicted classes.
    Optionally, these can be output as fractions instead of counts.

    Parameters
    ----------
    knownValues : nimble Base object
        The ground truth labels collected for some data.
    predictedValues : nimble Base object
        The labels predicted for the same data.
    labels : dict, list
        As a dictionary, a mapping of from the value in ``knownLabels``
        to a more specific label. A list may also be used provided the
        values in ``knownLabels`` represent an index to each value in
        the list. The labels will be used to create the featureNames and
        pointNames with the prefixes `known_` and `predicted_`,
        respectively.  If labels is None, the prefixes will be applied
        directly to the unique values found in ``knownLabels``.
    convertCountsToFractions : bool
        If False, the default, elements are counts. If True, the counts
        are converted to fractions by dividing by the total number of
        observations.

    Returns
    -------
    Base
        A confusion matrix nimble object matching the type of
        ``knownValues``.

    Notes
    -----
    Metrics for binary classification based on a confusion matrix,
    like truePositive, recall, precision, etc., can also be found in
    the nimble.calculate module.

    Examples
    --------
    Confusion matrix with and without alternate labels.

    >>> known = [[0], [1], [2],
    ...          [0], [1], [2],
    ...          [0], [1], [2],
    ...          [0], [1], [2]]
    >>> pred = [[0], [1], [2],
    ...         [0], [1], [2],
    ...         [0], [1], [2],
    ...         [1], [0], [2]]
    >>> knownObj = nimble.data(known)
    >>> predObj = nimble.data(pred)
    >>> cm = confusionMatrix(knownObj, predObj)
    >>> cm
    <Matrix 3pt x 3ft
                   known_0  known_1  known_2
                 ┌──────────────────────────
     predicted_0 │    3        1        0
     predicted_1 │    1        3        0
     predicted_2 │    0        0        4
    >
    >>> labels = {0: 'cat', 1: 'dog', 2: 'fish'}
    >>> cm = confusionMatrix(knownObj, predObj, labels=labels)
    >>> cm
    <Matrix 3pt x 3ft
                      known_cat  known_dog  known_fish
                    ┌─────────────────────────────────
      predicted_cat │     3          1          0
      predicted_dog │     1          3          0
     predicted_fish │     0          0          4
    >

    Label objects can have string values and here we output fractions.

    >>> known = [['cat'], ['dog'], ['fish'],
    ...          ['cat'], ['dog'], ['fish'],
    ...          ['cat'], ['dog'], ['fish'],
    ...          ['cat'], ['dog'], ['fish']]
    >>> pred = [['cat'], ['dog'], ['fish'],
    ...         ['cat'], ['dog'], ['fish'],
    ...         ['cat'], ['dog'], ['fish'],
    ...         ['dog'], ['cat'], ['fish']]
    >>> knownObj = nimble.data(known)
    >>> predObj = nimble.data(pred)
    >>> cm = confusionMatrix(knownObj, predObj,
    ...                      convertCountsToFractions=True)
    >>> cm
    <DataFrame 3pt x 3ft
                      known_cat  known_dog  known_fish
                    ┌─────────────────────────────────
      predicted_cat │   0.250      0.083      0.000
      predicted_dog │   0.083      0.250      0.000
     predicted_fish │   0.000      0.000      0.333
    >
    """
    if not (isinstance(knownValues, nimble.core.data.Base)
            and isinstance(predictedValues, nimble.core.data.Base)):
        msg = 'knownValues and predictedValues must be nimble data objects'
        raise InvalidArgumentType(msg)
    if not knownValues.shape[1] == predictedValues.shape[1] == 1:
        msg = 'knownValues and predictedValues must each be a single feature'
        raise InvalidArgumentValue(msg)
    if knownValues.shape[0] != predictedValues.shape[0]:
        msg = 'knownValues and predictedValues must have the same number of '
        msg += 'points'
        raise InvalidArgumentValue(msg)
    if not isinstance(labels, (type(None), dict, list)):
        msg = 'labels must be a dictionary mapping values from knownValues to '
        msg += 'a label or a list if the unique values in knownValues are in '
        msg += 'the range 0 to len(labels)'
        raise InvalidArgumentType(msg)

    if isinstance(labels, dict):
        confusionMtx, knownLabels = _confusionMatrixWithLabelsDict(
            knownValues, predictedValues, labels)
    elif labels is not None:
        confusionMtx, knownLabels = _confusionMatrixWithLabelsList(
            knownValues, predictedValues, labels)
    else:
        confusionMtx, knownLabels = _confusionMatrixNoLabels(
            knownValues, predictedValues)

    if convertCountsToFractions:
        confusionMtx = confusionMtx.astype(float) / len(knownValues.points)

    asType = knownValues.getTypeString()
    fNames = ['known_' + str(label) for label in knownLabels]
    pNames = ['predicted_' + str(label) for label in knownLabels]

    return createDataNoValidation(asType, confusionMtx, pNames, fNames,
                                  reuseData=True)

###########
# Helpers #
###########

_intMapCache = {} # increase efficiency by caching
def _mapInt(val):
    if val in _intMapCache:
        return _intMapCache[val]

    try:
        if val % 1 == 0:
            _intMapCache[val] = int(val)
            return int(val)
        return val
    except TypeError:
        return val

def _validateIndex(idx, numLabels, sourceArg):
    errorType = None
    if not isinstance(idx, int):
        errorType = InvalidArgumentValue
    elif not 0 <= idx < numLabels:
        errorType = IndexError
    if errorType is not None:
        msg = f'{sourceArg} contains an invalid value: {idx}. All values must '
        msg += f'be equal to integers 0 through {numLabels-1} (inclusive) '
        msg += 'indicating an index value for the labels argument'
        raise errorType(msg)

def _confusionMatrixWithLabelsList(knownValues, predictedValues, labels):
    numLabels = len(labels)
    toFill = np.zeros((numLabels, numLabels), dtype=int)
    validLabels = set() # to prevent repeated validation of same label
    for kVal, pVal in zip(knownValues, predictedValues):
        kVal = _mapInt(kVal)
        if kVal not in validLabels:
            _validateIndex(kVal, numLabels, 'knownValues')
            validLabels.add(kVal)
        pVal = _mapInt(pVal)
        if pVal not in validLabels:
            _validateIndex(pVal, numLabels, 'predictedValues')
            validLabels.add(pVal)
        toFill[pVal, kVal] += 1

    return toFill, labels

def _validateKey(key, labels, sourceArg):
    if key not in labels:
        msg = f'{key} was found in {sourceArg} but is not a key in labels'
        raise KeyError(msg)

def _confusionMatrixWithLabelsDict(knownValues, predictedValues, labels):
    sortedLabels = sorted(labels)
    numLabels = len(labels)
    toFill = np.zeros((numLabels, numLabels), dtype=int)
    labelsIdx = {}
    for kVal, pVal in zip(knownValues, predictedValues):
        # trigger KeyError if label not present
        if kVal not in labelsIdx:
            _validateKey(kVal, labels, 'knownValues')
            labelsIdx[kVal] = sortedLabels.index(kVal)
        if pVal not in labelsIdx:
            _validateKey(pVal, labels, 'predictedValues')
            labelsIdx[pVal] = sortedLabels.index(pVal)
        toFill[labelsIdx[pVal], labelsIdx[kVal]] += 1

    knownLabels = [labels[key] for key in sortedLabels]

    return toFill, knownLabels

def _confusionMatrixNoLabels(knownValues, predictedValues):
    knownLabels = set()
    confusionDict = {}
    # get labels and positions first then we will sort before creating matrix
    for kVal, pVal in zip(knownValues, predictedValues):
        knownLabels.add(kVal)
        if (kVal, pVal) in confusionDict:
            confusionDict[(kVal, pVal)] += 1
        else:
            confusionDict[(kVal, pVal)] = 1

    knownLabels = sorted(list(map(_mapInt, knownLabels)))
    labelsIdx = {}
    length = len(knownLabels)
    toFill = np.zeros((length, length), dtype=int)

    for (kVal, pVal), count in confusionDict.items():
        if kVal not in labelsIdx:
            labelsIdx[kVal] = knownLabels.index(kVal)
        if pVal not in labelsIdx:
            labelsIdx[pVal] = knownLabels.index(pVal)
        toFill[labelsIdx[pVal], labelsIdx[kVal]] = count

    return toFill, knownLabels
