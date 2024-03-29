
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
Metrics available for binary classification.
"""

from nimble.exceptions import InvalidArgumentValue
from .similarity import confusionMatrix
from .utility import performanceFunction

@performanceFunction('max')
def truePositive(knownValues, predictedValues):
    """
    Number of predicted positive values that were known to be positive.
    """
    confMtx = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getTruePositive(confMtx)

@performanceFunction('max')
def trueNegative(knownValues, predictedValues):
    """
    Number of predicted negative values that were known to be negative.
    """
    confMtx = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getTrueNegative(confMtx)

@performanceFunction('min')
def falsePositive(knownValues, predictedValues):
    """
    Number of predicted positive values that were known to be negative.
    """
    confMtx = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getFalsePositive(confMtx)

@performanceFunction('min')
def falseNegative(knownValues, predictedValues):
    """
    Number of predicted negative values that were known to be positive.
    """
    confMtx = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getFalseNegative(confMtx)

@performanceFunction('max')
def recall(knownValues, predictedValues):
    """
    The ratio of true positive values to known positive values.
    """
    confMtx = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getRecall(confMtx)

@performanceFunction('max')
def precision(knownValues, predictedValues):
    """
    The ratio of true positive values to predicted positive values.
    """
    confMtx = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getPrecision(confMtx)

@performanceFunction('max')
def specificity(knownValues, predictedValues):
    """
    The ratio of true negative values to known negative values.
    """
    confMtx = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getSpecificity(confMtx)

@performanceFunction('max')
def balancedAccuracy(knownValues, predictedValues):
    """
    Accurracy measure accounting for imbalances in the label counts.
    """
    confMtx = _getBinaryConfusionMatrix(knownValues, predictedValues)
    recall_ = _getRecall(confMtx)
    specificity_ = _getSpecificity(confMtx)

    return (recall_ + specificity_) / 2

@performanceFunction('max')
def f1Score(knownValues, predictedValues):
    """
    The harmonic mean of precision and recall.
    """
    confMtx = _getBinaryConfusionMatrix(knownValues, predictedValues)
    recall_ = _getRecall(confMtx)
    precision_ = _getPrecision(confMtx)

    return 2 * ((precision_ * recall_) / (precision_ + recall_))


###########
# Helpers #
###########

def _getBinaryConfusionMatrix(knownValues, predictedValues):
    """
    Providing labels [False, True] to confusionMatrix will validate that
    the provided values are binary. The known labels are features and
    the predicted labels are points. Result is always:

                       known_False      known_True
                    +----------------+----------------+
    predicted_False | True Negative  | False Negative |
                    +----------------+----------------+
    predicted_True  | False Positive | True Positive  |
                    +----------------+----------------+
    """
    try:
        return confusionMatrix(knownValues, predictedValues, [False, True])
    except IndexError as e:
        # catch this error to clarify the error since we provided the labels
        msg = 'The values provided are not binary, this function allows '
        msg += 'only 1, 0, True, and False as valid values'
        raise InvalidArgumentValue(msg) from e

# these helper functions avoid need to recreate confusion matrix each time

def _getTruePositive(confMtx):
    return confMtx[1, 1]

def _getTrueNegative(confMtx):
    return confMtx[0, 0]

def _getFalsePositive(confMtx):
    return confMtx[1, 0]

def _getFalseNegative(confMtx):
    return confMtx[0, 1]

def _getKnownPositive(confMtx):
    return sum(confMtx.features[1])

def _getKnownNegative(confMtx):
    return sum(confMtx.features[0])

def _getPredictedPositive(confMtx):
    return sum(confMtx.points[1])

def _getPredictedNegative(confMtx):
    return sum(confMtx.points[0])

def _getRecall(confMtx):
    truePos = _getTruePositive(confMtx)
    knownPos = _getKnownPositive(confMtx)
    return truePos / knownPos

def _getPrecision(confMtx):
    truePos = _getTruePositive(confMtx)
    predPos = _getPredictedPositive(confMtx)

    return truePos / predPos

def _getSpecificity(confMtx):
    trueNeg = _getTrueNegative(confMtx)
    knownNeg = _getKnownNegative(confMtx)

    return trueNeg / knownNeg
