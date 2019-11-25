from __future__ import absolute_import

import numpy

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.calculate import fractionIncorrect
from nimble.calculate import varianceFractionRemaining


def _validatePredictedAsLabels(predictedValues):
    if not isinstance(predictedValues, nimble.data.Base):
        msg = "predictedValues must be derived class of nimble.data.Base"
        raise InvalidArgumentType(msg)
    if len(predictedValues.features) > 1:
        msg = "predictedValues must be labels only; this has more than "
        msg += "one feature"
        raise InvalidArgumentValue(msg)


def cosineSimilarity(knownValues, predictedValues):
    _validatePredictedAsLabels(predictedValues)
    if not isinstance(knownValues, nimble.data.Base):
        msg = "knownValues must be derived class of nimble.data.Base"
        raise InvalidArgumentType(msg)

    known = knownValues.copy(to="numpy array").flatten()
    predicted = predictedValues.copy(to="numpy array").flatten()

    numerator = (numpy.dot(known, predicted))
    denominator = (numpy.linalg.norm(known) * numpy.linalg.norm(predicted))

    return numerator / denominator


cosineSimilarity.optimal = 'max'


def correlation(X, X_T=None):
    """
    Calculate the correlation between points in X. If X_T is not
    provided, a copy of X will be made in this function.

    """
    if X_T is None:
        X_T = X.T
    stdVector = X.points.statistics('populationstd')
    stdVector_T = stdVector.T

    cov = covariance(X, X_T, False)
    stdMatrix = stdVector.matrixMultiply(stdVector_T)
    ret = cov / stdMatrix

    return ret


def covariance(X, X_T=None, sample=True):
    """
    Calculate the covariance between points in X. If X_T is not
    provided, a copy of X will be made in this function.

    """
    if X_T is None:
        X_T = X.T
    pointMeansVector = X.points.statistics('mean')
    fill = lambda x: [x[0]] * len(X.features)
    pointMeans = pointMeansVector.points.calculate(fill, useLog=False)
    pointMeans_T = pointMeans.T

    XminusEofX = X - pointMeans
    X_TminusEofX_T = X_T - pointMeans_T

    # doing sample covariance calculation
    if sample:
        divisor = len(X.features) - 1
    # doing population covariance calculation
    else:
        divisor = len(X.features)

    ret = (XminusEofX.matrixMultiply(X_TminusEofX_T)) / divisor
    return ret


def fractionCorrect(knownValues, predictedValues):
    """
    Calculate how many values in predictedValues are equal to the
    values in the corresponding positions in knownValues. The return
    will be a float between 0 and 1 inclusive.

    """
    return 1 - fractionIncorrect(knownValues, predictedValues)


fractionCorrect.optimal = 'max'


def rSquared(knownValues, predictedValues):
    """
    Calculate the r-squared (or coefficient of determination) of the
    predictedValues given the knownValues. This will be equal to 1 -
    nimble.calculate.varianceFractionRemaining() of the same inputs.

    """
    return 1.0 - varianceFractionRemaining(knownValues, predictedValues)


rSquared.optimal = 'max'


def confusionMatrix(knownValues, predictedValues, labels=None,
                    outputFractions=False):
    """
    Confusion Matrix

    Parameters
    ----------

    Returns
    -------

    See Also
    --------

    Examples
    --------
    """
    if not (isinstance(knownValues, nimble.data.Base)
            and isinstance(predictedValues, nimble.data.Base)):
        msg = 'knownValues and predictedValues must be a nimble data objects'
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

    knownLabels = set()
    confusionDict = {}
    for kVal, pVal in zip(knownValues.elements, predictedValues.elements):
        knownLabels.add(kVal)
        if (kVal, pVal) in confusionDict:
            confusionDict[(kVal, pVal)] += 1
        else:
            confusionDict[(kVal, pVal)] = 1

    knownLabels = sorted(list(knownLabels))
    confusionMtx = []
    for pLabel in knownLabels:
        point = []
        for kLabel in knownLabels:
            try:
                val = confusionDict[(kLabel, pLabel)]
                if outputFractions:
                    point.append(val / len(knownValues.points))
                else:
                    point.append(val)
            except KeyError:
                point.append(0)
        confusionMtx.append(point)

    asType = knownValues.getTypeString()
    if labels is not None and len(labels) != len(knownLabels):
        msg = 'labels contained {0} labels '.format(len(labels))
        msg += 'but knownValues contained {0}. '.format(len(knownLabels))
        msg += 'The labels identified in knownValues were '
        msg += str(knownLabels)
        raise InvalidArgumentValue(msg)
    if isinstance(labels, dict):
        try:
            knownLabels = [labels[l] for l in knownLabels]
        except KeyError:
            msg = 'labels contained keys which were not identified in '
            msg += 'knownValues. The labels identified in knownValues were '
            msg += str(knownLabels)
            raise KeyError(msg)
    elif isinstance(labels, list):
        if knownLabels != list(range(len(knownLabels))):
            msg = 'A list can only be used for labels if the labels in '
            msg += 'knownValues represent index values (they are in range '
            msg += '0 to len(labels)) for the labels list . The labels '
            msg += 'identified in knownValues were ' + str(knownLabels)
            raise IndexError(msg)
        knownLabels = labels

    fNames = ['known_' + str(label) for label in knownLabels]
    pNames = ['predicted_' + str(label) for label in knownLabels]
    if outputFractions:
        eType = float
    else:
        eType = int

    return nimble.createData(asType, confusionMtx, pNames, fNames,
                             elementType=eType)
