
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
