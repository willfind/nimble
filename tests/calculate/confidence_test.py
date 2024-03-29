
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

import numpy as np
import pytest

import nimble
from nimble.random import numpyRandom
from nimble.exceptions import PackageException
from nimble.calculate.confidence import _confidenceIntervalHelper
from nimble.calculate import (
    rootMeanSquareErrorConfidenceInterval,
    meanAbsoluteErrorConfidenceInterval,
    fractionIncorrectConfidenceInterval,
    )
from tests.helpers import noLogEntryExpected
from tests.helpers import raises, patch

def fractionOfTimeInCI(getActual, getPredictions, ciFunc, expError):
    # different random states can lead to failures by a very small margin so we
    # will use a random state we expect to be successful
    with nimble.random.alternateControl(1, useLog=False):
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
@patch(nimble.calculate.confidence.scipy, 'nimbleAccessible', lambda: False)
def testCannotImportSciPy():
    _ = _confidenceIntervalHelper(None, None, None, None)

#########################################
# rootMeanSquareErrorConfidenceInterval #
#########################################
@pytest.mark.slow
@noLogEntryExpected
def test_rootMeanSquareErrorConfidenceInterval():
    def getActual(n):
        return nimble.data(numpyRandom.normal(size=(n, 1)),
                           useLog=False)

    def getPredictions(actual):
        return actual + 0.1 * getActual(len(actual))

    fractionOfTimeInCI(getActual, getPredictions,
                       rootMeanSquareErrorConfidenceInterval, 0.1)

#######################################
# meanAbsoluteErrorConfidenceInterval #
#######################################
@pytest.mark.slow
@noLogEntryExpected
def test_meanAbsoluteErrorConfidenceInterval():
    def getActual(n):
        return nimble.data(numpyRandom.normal(size=(n,1)),
                           useLog=False)

    def getPredictions(actual):
        return actual + 0.1 * getActual(len(actual))

    expectedError = np.mean(abs(0.1 * getActual(1000000).copy(to='numpyarray')))

    fractionOfTimeInCI(getActual, getPredictions,
                       meanAbsoluteErrorConfidenceInterval, expectedError)

#######################################
# fractionIncorrectConfidenceInterval #
#######################################
@pytest.mark.slow
@noLogEntryExpected
def test_fractionIncorrectConfidenceInterval():
    def getActual(n):
        return nimble.data(numpyRandom.binomial(1, 0.5, size=(n,1)),
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
