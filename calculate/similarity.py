from __future__ import absolute_import
import UML
import numpy

from UML.exceptions import ArgumentException
from UML.calculate import fractionIncorrect
from UML.calculate import varianceFractionRemaining


def _validatePredictedAsLabels(predictedValues):
    if not isinstance(predictedValues, UML.data.Base):
        raise ArgumentException("predictedValues must be derived class of UML.data.Base")
    if len(predictedValues.features) > 1:
        raise ArgumentException("predictedValues must be labels only; this has more than one feature")


def cosineSimilarity(knownValues, predictedValues):
    _validatePredictedAsLabels(predictedValues)

    known = knownValues.copyAs(format="numpy array").flatten()
    predicted = predictedValues.copyAs(format="numpy array").flatten()

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
        X_T = X.copy()
        X_T.transpose()
    stdVector = X.pointStatistics('populationstd')
    stdVector_T = stdVector.copy()
    stdVector_T.transpose()

    cov = covariance(X, X_T, False)
    stdMatrix = stdVector * stdVector_T
    ret = cov / stdMatrix

    return ret


def covariance(X, X_T=None, sample=True):
    """
    Calculate the covariance between points in X. If X_T is not
    provided, a copy of X will be made in this function.

    """
    if X_T is None:
        X_T = X.copy()
        X_T.transpose()
    pointMeansVector = X.pointStatistics('mean')
    fill = lambda x: [x[0]] * len(X.features)
    pointMeans = pointMeansVector.points.calculate(fill)
    pointMeans_T = pointMeans.copy()
    pointMeans_T.transpose()

    XminusEofX = X - pointMeans
    X_TminusEofX_T = X_T - pointMeans_T

    # doing sample covariance calculation
    if sample:
        divisor = len(X.features) - 1
    # doing population covariance calculation
    else:
        divisor = len(X.features)

    ret = (XminusEofX * X_TminusEofX_T) / divisor
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
    UML.calculate.varianceFractionRemaining() of the same inputs.

    """
    return 1.0 - varianceFractionRemaining(knownValues, predictedValues)


rSquared.optimal = 'max'
