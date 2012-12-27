'''
Created on Dec 11, 2012

@author: rossnoren
'''

import inspect
import numpy
from ..combinations.Combinations import executeCode


def computeMetrics(knownIndicator, knownValues, predictedValues, performanceFunctions):
    """

    """
    results = {}
    for func in performanceFunctions:
        print inspect.getargspec(func).args
        if len(inspect.getargspec(func).args) == 2:
            trueLabels = knownValues[:, trueLabelIndex]
            results[func] = (func(trueLabels, predictedValues))
        elif len(inspect.getargspec(func).args) == 3:
            #divide X into labels and features
            #TODO
            results[func] = (func(knownValues, predictedValues))
        else:
            raise Exception("One of the functions passed to computeMetrics has an invalid signature: "+func.__name__)
    return results

def confusion_matrix_generator(knownLabels, predictedLabels):
    """ Given two vectors, one of known class labels (as strings) and one of predicted labels,
    compute the confusion matrix.  Returns a 2-dimensional dictionary in which outer label is 
    keyed by known label, inner label is keyed by predicted label, and the value stored is the count
    of instances for each combination.  Works for an indefinite number of class labels.
    """
    confusionCounts = {}
    for known, predicted in zip(knownLabels, predictedLabels):
        if confusionCounts[known] is None:
            confusionCounts[known] = {predicted:1}
        elif confusionCounts[known][predicted] is None:
            confusionCounts[known][predicted] = 1
        else:
            confusionCounts[known][predicted] += 1
    
    #if there are any entries in the square matrix confusionCounts,
    #then there value must be 0.  Go through and fill them in.
    for knownLabel in confusionCounts:
        if confusionCounts[knownLabel][knownLabel] is None:
            confusionCounts[knownLabel][knownLabel] = 0
    
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
    confusionMatrixResults = computeMetrics_v2(classLabelIndex, X, Y, functions)
    confusionMatrix = confusionMatrixResults["confusion_matrix_generator"]
    print_confusion_matrix(confusionMatrix)

"""
ignore this

def computeMetrics_v1(X, Y, functions, classLabelName):
    results = {}
    for func in functions:
        if len(inspect.getargspec(func).args) == 2:
            #get just label column out of X
            #TODO 
            results[func.__name__] = (func(X, Y))
        elif len(inspect.getargspec(func).args) == 3:
            #divide X into labels and features
            #TODO
            results[func.__name__] = (func(X, Y))
        else:
            raise Exception("One of the functions passed to computeMetrics has an invalid signature: "+func.__name__)
    return results


if __name__ == "__main__":
    testMetric()
"""