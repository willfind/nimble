"""

"""
from nimble.exceptions import InvalidArgumentValue
from .similarity import confusionMatrix

def truePositive(knownValues, predictedValues):
    cm = getValidConfusionMatrix(knownValues, predictedValues)
    return getTruePositive(cm)

truePositive.optimal = 'max'

def trueNegative(knownValues, predictedValues):
    cm = getValidConfusionMatrix(knownValues, predictedValues)
    return getTrueNegative(cm)

trueNegative.optimal = 'max'

def falsePositive(knownValues, predictedValues):
    cm = getValidConfusionMatrix(knownValues, predictedValues)
    return getFalsePositive(cm)

falsePositive.optimal = 'min'

def falseNegative(knownValues, predictedValues):
    cm = getValidConfusionMatrix(knownValues, predictedValues)
    return getFalseNegative(cm)

falseNegative.optimal = 'min'

def recall(knownValues, predictedValues):
    cm = getValidConfusionMatrix(knownValues, predictedValues)
    return getRecall(cm)

recall.optimal = 'max'

def precision(knownValues, predictedValues):
    cm = getValidConfusionMatrix(knownValues, predictedValues)
    return getPrecision(cm)

precision.optimal = 'max'

def specificity(knownValues, predictedValues):
    cm = getValidConfusionMatrix(knownValues, predictedValues)
    return getSpecificity(cm)

specificity.optimal = 'max'

def balancedAccuracy(knownValues, predictedValues):
    cm = getValidConfusionMatrix(knownValues, predictedValues)
    recall = getRecall(cm)
    specificity = getSpecificity(cm)

    return (recall + specificity) / 2

balancedAccuracy.optimal = 'max'

def f1Score(knownValues, predictedValues):
    cm = getValidConfusionMatrix(knownValues, predictedValues)
    recall = getRecall(cm)
    precision = getPrecision(cm)

    return 2 * ((precision * recall) / (precision + recall))


###########
# Helpers #
###########

def getValidConfusionMatrix(knownValues, predictedValues):
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

def getTruePositive(cm):
    return cm[1, 1]

def getTrueNegative(cm):
    return cm[0, 0]

def getFalsePositive(cm):
    return cm[1, 0]

def getFalseNegative(cm):
    return cm[0, 1]

def getKnownPositive(cm):
    return sum(cm.features[1])

def getKnownNegative(cm):
    return sum(cm.features[0])

def getPredictedPositive(cm):
    return sum(cm.points[1])

def getPredictedNegative(cm):
    return sum(cm.points[0])

def getRecall(cm):
    truePos = getTruePositive(cm)
    knownPos = getKnownPositive(cm)
    return truePos / knownPos

def getPrecision(cm):
    truePos = getTruePositive(cm)
    predPos = getPredictedPositive(cm)

    return truePos / predPos

def getSpecificity(cm):
    trueNeg = getTrueNegative(cm)
    knownNeg = getKnownNegative(cm)

    return trueNeg / knownNeg
