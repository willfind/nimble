'''
Created on Dec 11, 2012

@author: rossnoren
'''

import inspect

from ..processing.base_data import BaseData
from ..combinations.Combinations import executeCode
from ..utility.custom_exceptions import ArgumentException


def computeMetrics(dependentVar, knownData, predictedData, performanceFunctions):
    """
        Calculate one or more error metrics, given a list of known labels and a list of
        predicted labels.  Return as a dictionary associating the performance metric with
        its numerical result.

        dependentVar: either an int/string representing a column index in knownData
        containing the known labels, or a matrix that contains the known labels

        knownData: matrix containing the known labels of the test set, as well as the
        features of the test set. Can be None if 'knownIndicator' contains the labels,
        and none of the performance functions features as input.

        predictedData: Matrix containing predicted class labels for a testing set.
        Assumes that the predicted label in the nth row of predictedLabels is associated
        with the same data point/instance as the label in the nth row of knownLabels.

        performanceFunctions: list of functions that compute some kind of error metric.
        Functions must be either a string that defines a proper function, a one-liner
        function (see Combinations.py), or function code.  Also, they are expected to
        take at least 2 arguments:  a vector or known labels and a vector of predicted
        labels.  Optionally, they may take the features of the test set as a third argument,
        as a matrix.

        Returns: a dictionary associating each performance metric with the (presumably)
        numerical value computed by running the function over the known labels & predicted labels
    """
    if isinstance(dependentVar, BaseData):
        #The known Indicator argument already contains all known
        #labels, so we do not need to do any further processing
        knownLabels = dependentVar
    elif dependentVar is not None:
        #known Indicator is an index; we extract the column it indicates
        #from knownValues
        knownLabels = knownData.copyColumns([dependentVar])
    else:
        raise ArgumentException("Missing indicator for known labels in computeMetrics")

    results = {}
    #TODO make this hash more generic - what if function args are not knownValues and predictedValues
    parameterHash = {"knownValues":knownLabels, "predictedValues":predictedData}
    for func in performanceFunctions:
        #print inspect.getargspec(func).args
        if len(inspect.getargspec(func).args) == 2:
            #the metric function only takes two arguments: we assume they
            #are the known class labels and the predicted class labels
            results[func] = executeCode(func, parameterHash)
        elif len(inspect.getargspec(func).args) == 3:
            #the metric function takes three arguments:  known class labels,
            #features, and predicted class labels. add features to the parameter hash
            #divide X into labels and features
            #TODO correctly separate known labels and features in all cases
            parameterHash["features"] = knownData
            results[func] = executeCode(func, parameterHash)
        else:
            raise Exception("One of the functions passed to computeMetrics has an invalid signature: "+func.__name__)
    return results

def confusion_matrix_generator(knownY, predictedY):
    """ Given two vectors, one of known class labels (as strings) and one of predicted labels,
    compute the confusion matrix.  Returns a 2-dimensional dictionary in which outer label is
    keyed by known label, inner label is keyed by predicted label, and the value stored is the count
    of instances for each combination.  Works for an indefinite number of class labels.
    """
    confusionCounts = {}
    for known, predicted in zip(knownY, predictedY):
        if confusionCounts[known] is None:
            confusionCounts[known] = {predicted:1}
        elif confusionCounts[known][predicted] is None:
            confusionCounts[known][predicted] = 1
        else:
            confusionCounts[known][predicted] += 1
    
    #if there are any entries in the square matrix confusionCounts,
    #then there value must be 0.  Go through and fill them in.
    for knownY in confusionCounts:
        if confusionCounts[knownY][knownY] is None:
            confusionCounts[knownY][knownY] = 0
    
    return confusionCounts

def print_confusion_matrix(confusionMatrix):
    """ Print a confusion matrix in human readable form, with
    rows indexed by known labels, and columns indexed by predictedlabels.
    confusionMatrix is a 2-dimensional dictionary, that is also primarily
    indexed by known labels, and secondarily indexed by predicted labels,
    with the value at confusionMatrix[knownLabel][predictedLabel] being the
    count of posts that fell into that slot.  Does not need to be sorted.
    """
    #print heading
    print "*"*30 + "Confusion Matrix"+"*"*30
    print "\n\n"
    
    #print top line - just the column headings for
    #predicted labels
    spacer = " "*15
    sortedLabels = sorted(confusionMatrix.iterKeys())
    for knownLabel in sortedLabels:
        spacer += " "*(6 - len(knownLabel)) + knownLabel
    
    print spacer
    totalPostCount = 0
    for knownLabel in sortedLabels:
        outputBuffer = knownLabel+" "*(15 - len(knownLabel))
        for predictedLabel in sortedLabels:
            count = confusionMatrix[knownLabel][predictedLabel]
            totalPostCount += count
            outputBuffer += " "*(6 - len(count)) + count
        print outputBuffer
    
    print "Total post count: " + totalPostCount
    
def checkPrintConfusionMatrix():
    X = {"classLabel": ["A", "B", "C", "C", "B", "C", "A", "B", "C", "C", "B", "C", "A", "B", "C", "C", "B", "C"]}
    Y = ["A", "C", "C", "A", "B", "C", "A", "C", "C", "A", "B", "C", "A", "C", "C", "A", "B", "C"]
    functions = [confusion_matrix_generator]
    classLabelIndex = "classLabel"
    confusionMatrixResults = computeMetrics(classLabelIndex, X, Y, functions)
    confusionMatrix = confusionMatrixResults["confusion_matrix_generator"]
    print_confusion_matrix(confusionMatrix)
