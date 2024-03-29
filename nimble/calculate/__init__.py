
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
Functions that perform calculations on Nimble-defined objects.

This includes functions that can be used as performance functions in the
Nimble testing and cross-validation API. Some similar functionality may
be available as methods off of data objects; the versions here are
functions, and take any inputs as arguments.
"""

from .loss import fractionIncorrect
from .loss import meanAbsoluteError
from .loss import meanFeaturewiseRootMeanSquareError
from .loss import rootMeanSquareError
from .loss import varianceFractionRemaining
from .matrix import elementwiseMultiply
from .matrix import elementwisePower
from .similarity import correlation
from .similarity import cosineSimilarity
from .similarity import covariance
from .similarity import fractionCorrect
from .similarity import rSquared
from .similarity import confusionMatrix
from .statistic import count
from .statistic import maximum
from .statistic import mean
from .statistic import median
from .statistic import medianAbsoluteDeviation
from .statistic import mode
from .statistic import minimum
from .statistic import uniqueCount
from .statistic import proportionMissing
from .statistic import proportionZero
from .statistic import quartiles
from .statistic import residuals
from .statistic import standardDeviation
from .statistic import variance
from .statistic import sum # pylint: disable=redefined-builtin
from .utility import detectBestResult
from .utility import performanceFunction
from .linalg import inverse
from .linalg import pseudoInverse
from .linalg import solve
from .linalg import leastSquaresSolution
from .binary import truePositive
from .binary import trueNegative
from .binary import falsePositive
from .binary import falseNegative
from .binary import recall
from .binary import precision
from .binary import specificity
from .binary import balancedAccuracy
from .binary import f1Score
from .normalize import meanNormalize
from .normalize import meanStandardDeviationNormalize
from .normalize import range0to1Normalize
from .normalize import percentileNormalize
from .confidence import rootMeanSquareErrorConfidenceInterval
from .confidence import meanAbsoluteErrorConfidenceInterval
from .confidence import fractionIncorrectConfidenceInterval
from .._utility import _setAll

__all__ = _setAll(vars())
