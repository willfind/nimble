"""
Metrics available for binary classification.
"""

from nimble.exceptions import InvalidArgumentValue
from .similarity import confusionMatrix

def truePositive(knownValues, predictedValues):
    """
    Number of predicted positive values that were known to be positive.
    """
    cm = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getTruePositive(cm)

truePositive.optimal = 'max'

def trueNegative(knownValues, predictedValues):
    """
    Number of predicted negative values that were known to be negative.
    """
    cm = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getTrueNegative(cm)

trueNegative.optimal = 'max'

def falsePositive(knownValues, predictedValues):
    """
    Number of predicted positive values that were known to be negative.
    """
    cm = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getFalsePositive(cm)

falsePositive.optimal = 'min'

def falseNegative(knownValues, predictedValues):
    """
    Number of predicted negative values that were known to be positive.
    """
    cm = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getFalseNegative(cm)

falseNegative.optimal = 'min'

def recall(knownValues, predictedValues):
    """
    The ratio of true positive values to known positive values.
    """
    cm = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getRecall(cm)

recall.optimal = 'max'

def precision(knownValues, predictedValues):
    """
    The ratio of true positive values to predicted positive values.
    """
    cm = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getPrecision(cm)

precision.optimal = 'max'

def specificity(knownValues, predictedValues):
    """
    The ratio of true negative values to known negative values.
    """
    cm = _getBinaryConfusionMatrix(knownValues, predictedValues)
    return _getSpecificity(cm)

specificity.optimal = 'max'

def balancedAccuracy(knownValues, predictedValues):
    """
    Accurracy measure accounting for imbalances in the label counts.
    """
    cm = _getBinaryConfusionMatrix(knownValues, predictedValues)
    recall = _getRecall(cm)
    specificity = _getSpecificity(cm)

    return (recall + specificity) / 2

balancedAccuracy.optimal = 'max'

def f1Score(knownValues, predictedValues):
    """
    The harmonic mean of precision and recall.
    """
    cm = _getBinaryConfusionMatrix(knownValues, predictedValues)
    recall = _getRecall(cm)
    precision = _getPrecision(cm)

    return 2 * ((precision * recall) / (precision + recall))

f1Score.optimal = 'max'

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
    except IndexError:
        # catch this error to clarify the error since we provided the labels
        msg = 'The values provided are not binary, this function allows '
        msg += 'only 1, 0, True, and False as valid values'
        raise InvalidArgumentValue(msg)

# these helper functions avoid need to recreate confusion matrix each time

def _getTruePositive(cm):
    return cm[1, 1]

def _getTrueNegative(cm):
    return cm[0, 0]

def _getFalsePositive(cm):
    return cm[1, 0]

def _getFalseNegative(cm):
    return cm[0, 1]

def _getKnownPositive(cm):
    return sum(cm.features[1])

def _getKnownNegative(cm):
    return sum(cm.features[0])

def _getPredictedPositive(cm):
    return sum(cm.points[1])

def _getPredictedNegative(cm):
    return sum(cm.points[0])

def _getRecall(cm):
    truePos = _getTruePositive(cm)
    knownPos = _getKnownPositive(cm)
    return truePos / knownPos

def _getPrecision(cm):
    truePos = _getTruePositive(cm)
    predPos = _getPredictedPositive(cm)

    return truePos / predPos

def _getSpecificity(cm):
    trueNeg = _getTrueNegative(cm)
    knownNeg = _getKnownNegative(cm)

    return trueNeg / knownNeg
