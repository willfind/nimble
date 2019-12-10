from __future__ import absolute_import
import math

import numpy

import nimble
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble.utility import ImportModule

scipy = ImportModule('scipy')

def confidenceIntervalHelper(errors, transform, confidence=0.95):
    """Helper to calculate the confidence interval, given a vector of errors
    and a monotonic transform to be applied after the calculation.
    """
    if not scipy:
        msg = 'To call this function, scipy must be installed.'
        raise PackageException(msg)

    if len(errors.features) != 1:
        msg = "The errors vector may only have one feature"
        raise ImproperObjectAction(msg)

    if transform is None:
        transform = lambda x: x

    halfConfidence = 1 - ((1 - confidence) / 2.0)
    boundaryOnNormalScale = scipy.stats.norm.ppf(halfConfidence)

    sqrtN = float(math.sqrt(len(errors.points)))
    mean = errors.features.statistics('mean')[0, 0]
    std = errors.features.statistics('SampleStandardDeviation')[0, 0]

    low = transform(mean - (boundaryOnNormalScale * (std / sqrtN)))
    high = transform(mean + (boundaryOnNormalScale * (std / sqrtN)))

    return (low, high)


def rootMeanSquareErrorConfidenceInterval(known, predicted, confidence=0.95):
    errors = (known - predicted) ** 2

    def wrappedSqrt(value):
        if value < 0:
            return 0
        return math.sqrt(value)

    return confidenceIntervalHelper(errors, wrappedSqrt, confidence)


def meanAbsoluteErrorConfidenceInterval(known, predicted, confidence=0.95):
    errors = known - predicted
    return confidenceIntervalHelper(errors, None, confidence)


def fractionIncorrectConfidenceInterval(known, predicted, confidence=0.95):
    rawErrors = known.copy(to='numpyarray') - predicted.copy(to='numpyarray')
    rawErrors = numpy.absolute(rawErrors)
    errors = nimble.createData("Matrix", rawErrors)

    return confidenceIntervalHelper(errors, None, confidence)
