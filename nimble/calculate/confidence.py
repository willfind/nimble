"""
Confidence Intervals for error metrics.
"""
import numpy as np

import nimble
from nimble.exceptions import PackageException
from nimble._utility import scipy

def _confidenceIntervalHelper(mean, standardDeviation, sampleSize,
                              confidence, transform=None):
    """
    Helper to calculate the confidence interval, given a vector of
    errors and a monotonic transform to be applied after the
    calculation.
    """
    if not scipy.nimbleAccessible():
        msg = 'To call this function, scipy must be installed.'
        raise PackageException(msg)

    halfConfidence = 1 - ((1 - confidence) / 2.0)
    boundaryOnNormalScale = scipy.stats.t.ppf(halfConfidence, sampleSize - 1)
    sqrtN = np.sqrt(sampleSize)

    low = mean - (boundaryOnNormalScale * (standardDeviation / sqrtN))
    high = mean + (boundaryOnNormalScale * (standardDeviation / sqrtN))
    if transform is not None:
        low, high = transform(low), transform(high)

    return low, high

def rootMeanSquareErrorConfidenceInterval(known, predicted, confidence=0.95):
    """
    Estimate a confidence interval for the root mean square error.

    This is an estimation that relies on the Central Limit Theorem and
    may be inaccurate for smaller samples. A minimum sample size of 100
    is recommended.
    """
    errors = (known - predicted) ** 2
    numSamples = len(errors)
    mean = nimble.calculate.mean(errors)
    standardDeviation = nimble.calculate.standardDeviation(errors)

    return _confidenceIntervalHelper(mean, standardDeviation, numSamples,
                                     confidence, np.sqrt)


def meanAbsoluteErrorConfidenceInterval(known, predicted, confidence=0.95):
    """
    Estimate a confidence interval for the mean absolute error.

    This is an estimation that relies on the Central Limit Theorem and
    may be inaccurate for smaller samples. A minimum sample size of 100
    is recommended.
    """
    errors = abs(known - predicted)
    numSamples = len(errors)
    mean = nimble.calculate.mean(errors)
    standardDeviation = nimble.calculate.standardDeviation(errors)

    return _confidenceIntervalHelper(mean, standardDeviation, numSamples,
                                     confidence)


def fractionIncorrectConfidenceInterval(known, predicted, confidence=0.95):
    """
    Estimate a confidence interval for the fraction incorrect.

    This is an estimation that relies on the Central Limit Theorem and
    may be inaccurate for smaller samples. A minimum sample size of 100
    is recommended.
    """
    rawKnown = known.copy(to='numpyarray')
    rawPredicted = predicted.copy(to='numpyarray')
    errors = rawKnown[:] != rawPredicted[:]
    numSamples = len(errors)
    fracIncorrect = np.mean(errors)
    standardDeviation = np.sqrt((fracIncorrect) * (1 - fracIncorrect))

    return _confidenceIntervalHelper(fracIncorrect, standardDeviation,
                                     numSamples, confidence)
