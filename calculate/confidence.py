import math
import scipy.stats
import numpy

import UML
from UML.exceptions import ArgumentException


def confidenceIntervalHelper(errors, transform, confidence=0.95):
    """Helper to calculate the confidence interval, given a vector of errors
    and a monotonic transform to be applied after the calculation.

    """
    if errors.featureCount != 1:
        raise ArgumentException("The errors vector may only have one feature")

    if transform is None:
        transform = lambda x: x

    halfConfidence = 1 - ((1 - confidence) / 2.0)
    boundaryOnNormalScale = scipy.stats.norm.ppf(halfConfidence)

    sqrtN = float(math.sqrt(errors.pointCount))
    mean = errors.featureStatistics('mean')[0, 0]
    std = errors.featureStatistics('SampleStandardDeviation')[0, 0]

    low = transform(mean - (boundaryOnNormalScale * (std / sqrtN)))
    high = transform(mean + (boundaryOnNormalScale * (std / sqrtN)))

    return (low, high)


def rootMeanSquareErrorConfidenceInterval(known, predicted, confidence=0.95):
    errors = known - predicted
    errors.elementwisePower(2)

    def wrappedSqrt(value):
        if value < 0:
            return 0
        return math.sqrt(value)

    return confidenceIntervalHelper(errors, wrappedSqrt, confidence)


def meanAbsoluteErrorConfidenceInterval(known, predicted, confidence=0.95):
    errors = known - predicted
    return confidenceIntervalHelper(errors, None, confidence)


def fractionIncorrectConfidenceInterval(known, predicted, confidence=0.95):
    rawErrors = known.copyAs('numpyarray') - predicted.copyAs('numpyarray')
    rawErrors = numpy.absolute(rawErrors)
    errors = UML.createData("Matrix", rawErrors)

    return confidenceIntervalHelper(errors, None, confidence)
