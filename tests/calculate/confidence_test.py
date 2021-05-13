try:
    from unittest import mock #python >=3.3
except ImportError:
    import mock

import numpy as np
from nose.tools import raises
from nose.plugins.attrib import attr

import nimble
from nimble.random import numpyRandom
from nimble.exceptions import PackageException
from tests.helpers import noLogEntryExpected
from nimble.calculate.confidence import _confidenceIntervalHelper
from nimble.calculate import (
    rootMeanSquareErrorConfidenceInterval,
    meanAbsoluteErrorConfidenceInterval,
    fractionIncorrectConfidenceInterval,
    )

def fractionOfTimeInCI(getActual, getPredictions, ciFunc, expError):
    confidence = numpyRandom.randint(90, 99) / 100
    results = []
    for _ in range(1000):
        actual = getActual(300)
        predicted = getPredictions(actual)
        lower, upper = ciFunc(actual, predicted, confidence)
        isInCI = expError > lower and expError < upper
        results.append(isInCI)

    assert abs(np.mean(results) - confidence) <= 0.015

@raises(PackageException)
@mock.patch('nimble.calculate.confidence.scipy.nimbleAccessible', new=lambda: False)
def testCannotImportSciPy():
    _ = _confidenceIntervalHelper(None, None, None, None)

#########################################
# rootMeanSquareErrorConfidenceInterval #
#########################################
@attr('slow')
@noLogEntryExpected
def test_rootMeanSquareErrorConfidenceInterval():
    def getActual(n):
        return nimble.data('Matrix', numpyRandom.normal(size=(n, 1)),
                           useLog=False)

    def getPredictions(actual):
        return actual + 0.1 * getActual(len(actual))

    fractionOfTimeInCI(getActual, getPredictions,
                       rootMeanSquareErrorConfidenceInterval, 0.1)

#######################################
# meanAbsoluteErrorConfidenceInterval #
#######################################
@attr('slow')
@noLogEntryExpected
def test_meanAbsoluteErrorConfidenceInterval():
    def getActual(n):
        return nimble.data('Matrix', numpyRandom.normal(size=(n,1)),
                           useLog=False)

    def getPredictions(actual):
        return actual + 0.1 * getActual(len(actual))

    expectedError = np.mean(abs(0.1 * getActual(1000000)))

    fractionOfTimeInCI(getActual, getPredictions,
                       meanAbsoluteErrorConfidenceInterval, expectedError)

#######################################
# fractionIncorrectConfidenceInterval #
#######################################
@attr('slow')
@noLogEntryExpected
def test_fractionIncorrectConfidenceInterval():
    def getActual(n):
        return nimble.data('Matrix',
                           numpyRandom.binomial(1, 0.5, size=(n,1)),
                           useLog=False)

    def getPredictions(actual):
        def predict(pt):
            rand = numpyRandom.rand()
            if rand < 0.8:
                return pt[0]
            return int(not pt[0])

        return actual.points.calculate(predict, useLog=False)

    fractionOfTimeInCI(getActual, getPredictions,
                       fractionIncorrectConfidenceInterval, 0.2)
