import math

import numpy as np

import nimble
from nimble.calculate import cosineSimilarity
from nimble.calculate import confusionMatrix
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from tests.helpers import raises
from tests.helpers import noLogEntryExpected
from tests.helpers import getDataConstructors

####################
# cosineSimilarity #
####################
@noLogEntryExpected
def test_cosineSimilarity():
    assert cosineSimilarity.optimal == 'max'

    orig = np.array([[1], [0]])
    orth = np.array([[0], [1]])
    neg = np.array([[-1], [0]])

    origMatrix = nimble.data(source=orig, useLog=False)
    orthMatrix = nimble.data(source=orth, useLog=False)
    negMatrix = nimble.data(source=neg, useLog=False)

    result0 = cosineSimilarity(origMatrix, origMatrix)
    result1 = cosineSimilarity(origMatrix, orthMatrix)
    result2 = cosineSimilarity(origMatrix, negMatrix)

    assert result0 == 1
    assert result1 == 0
    assert result2 == -1

def test_cosineSimilarityZeros():
    zeros = [[0], [0]]

    zerosMatrix = nimble.data(source=zeros)

    result0 = cosineSimilarity(zerosMatrix, zerosMatrix)

    assert math.isnan(result0)

@raises(InvalidArgumentType)
def test_cosineSimilarityKnownWrongType():
    orig = np.array([[1], [0]])
    origMatrix = nimble.data(source=orig)

    result = cosineSimilarity(orig, origMatrix)

@raises(InvalidArgumentType)
def test_cosineSimilarityPredictedWrongType():
    orig = np.array([[1], [0]])
    origMatrix = nimble.data(source=orig)

    result = cosineSimilarity(origMatrix, orig)

@raises(InvalidArgumentValueCombination)
def test_cosineSimilarityPredictedWrongShape():
    orig = np.array([[1], [0]])
    origMatrix = nimble.data(source=orig)
    pred = np.array([[1, 1], [0, 0]])
    predMatrix = nimble.data(source=pred)

    result = cosineSimilarity(origMatrix, predMatrix)

def test_confusionMatrix_exception_wrongType():
    known = [[1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4]]
    pred = [[1], [2], [3], [4],
            [1], [2], [3], [4],
            [1], [2], [3], [4],
            [4], [3], [2], [1],]

    knownObj = nimble.data(known, useLog=False)
    predObj = nimble.data(pred, useLog=False)

    with raises(InvalidArgumentType):
        cm = confusionMatrix(known, predObj)

    with raises(InvalidArgumentType):
        cm = confusionMatrix(knownObj, pred)

def test_confusionMatrix_exception_labelsMissingKnown():
    known = [[0], [1], [2], [3],
             [0], [1], [2], [3],
             [0], [1], [2], [3],
             [0], [1], [2], [3]]
    pred = [[0], [1], [2], [3],
            [0], [1], [2], [3],
            [0], [1], [2], [3],
            [3], [2], [1], [0]]

    knownObj = nimble.data(known, useLog=False)
    predObj = nimble.data(pred, useLog=False)

    # short
    with raises(IndexError):
        labels = ['zero', 'one', 'two']
        cm = confusionMatrix(knownObj, predObj, labels=labels)

    with raises(KeyError):
        labels = {0:'zero', 1:'one', 2:'two'}
        cm = confusionMatrix(knownObj, predObj, labels=labels)

@raises(InvalidArgumentValue)
def test_confusionMatrix_exception_labelListInvalid_wrongType():
    known = [['dog'], ['cat'], ['fish'], ['bear'],
             ['dog'], ['cat'], ['fish'], ['bear'],
             ['dog'], ['cat'], ['fish'], ['bear'],
             ['dog'], ['cat'], ['fish'], ['bear']]
    pred = [['dog'], ['cat'], ['fish'], ['bear'],
            ['dog'], ['cat'], ['fish'], ['bear'],
            ['dog'], ['cat'], ['fish'], ['bear'],
            ['cat'], ['dog'], ['bear'], ['fish']]

    knownObj = nimble.data(known, useLog=False)
    predObj = nimble.data(pred, useLog=False)

    labels = ['zero', 'one', 'two', 'three']
    cm = confusionMatrix(knownObj, predObj, labels=labels)

@raises(IndexError)
def test_confusionMatrix_exception_labelListInvalid_outOfRange():
    known = [[1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4]]
    pred = [[1], [2], [3], [4],
            [1], [2], [3], [4],
            [1], [2], [3], [4],
            [4], [3], [2], [1]]

    knownObj = nimble.data(known, useLog=False)
    predObj = nimble.data(pred, useLog=False)

    labels = ['zero', 'one', 'two', 'three']
    cm = confusionMatrix(knownObj, predObj, labels=labels)

@raises(KeyError)
def test_confusionMatrix_exception_labelDictInvalidKey():
    known = [[1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4]]
    pred = [[1], [2], [3], [4],
            [1], [2], [3], [4],
            [1], [2], [3], [4],
            [4], [3], [2], [1]]

    knownObj = nimble.data(known, useLog=False)
    predObj = nimble.data(pred, useLog=False)

    labels = {0:'zero', 1:'one', 2:'two', 3:'three'}
    cm = confusionMatrix(knownObj, predObj, labels=labels)

@noLogEntryExpected
def test_confusionMatrix_noLabels():
    known = [[1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4]]
    pred = [[1], [2], [3], [4],
            [1], [2], [3], [4],
            [1], [2], [3], [4],
            [4], [3], [2], [1]]

    for constructor in getDataConstructors():
        knownObj = constructor(known, useLog=False)
        predObj = constructor(pred, useLog=False)

        cm = confusionMatrix(knownObj, predObj)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        featureNames = ['known_' + str(i) for i in range(1, 5)]
        pointNames = ['predicted_' + str(i) for i in range(1, 5)]
        expObj = constructor(expData, pointNames, featureNames,
                             useLog=False)

        assert cm.isIdentical(expObj)

@noLogEntryExpected
def test_confusionMatrix_noLabels_consistentOutput():
    known1 = [[1], [2], [3], [4],
              [1], [2], [3], [4],
              [1], [2], [3], [4],
              [1], [2], [3], [4]]
    pred1 = [[1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4],
             [4], [3], [2], [1]]
    # same result as known1 but in different order
    known2 = [[4], [3], [2], [1],
              [4], [3], [2], [1],
              [4], [3], [2], [1],
              [4], [3], [2], [1],]
    pred2 = [[4], [3], [2], [1],
             [4], [3], [2], [1],
             [4], [3], [2], [1],
             [1], [2], [3], [4],]

    for constructor in getDataConstructors():
        knownObj1 = constructor(known1, useLog=False)
        predObj1 = constructor(pred1, useLog=False)
        cm1 = confusionMatrix(knownObj1, predObj1)

        knownObj2 = constructor(known2, useLog=False)
        predObj2 = constructor(pred2, useLog=False)
        cm2 = confusionMatrix(knownObj2, predObj2)

        assert cm1 == cm2

@noLogEntryExpected
def test_confusionMatrix_withLabelsDict():
    known = [[1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4]]
    pred = [[1], [2], [3], [4],
            [1], [2], [3], [4],
            [1], [2], [3], [4],
            [4], [3], [2], [1]]

    for constructor in getDataConstructors():
        knownObj = constructor(known, useLog=False)
        predObj = constructor(pred, useLog=False)

        labels = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        cm = confusionMatrix(knownObj, predObj, labels=labels)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        sortedLabels = [labels[i] for i in range(1, 5)]
        featureNames = ['known_' + l for l in sortedLabels]
        pointNames = ['predicted_' + l for l in sortedLabels]
        expObj = constructor(expData, pointNames, featureNames,
                             useLog=False)

        assert cm.isIdentical(expObj)

@noLogEntryExpected
def test_confusionMatrix_withLabelsDict_consistentOutput():
    known1 = [[1], [2], [3], [4],
              [1], [2], [3], [4],
              [1], [2], [3], [4],
              [1], [2], [3], [4]]
    pred1 = [[1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4],
             [4], [3], [2], [1]]
    # same result as known1 but in different order
    known2 = [[4], [3], [2], [1],
              [4], [3], [2], [1],
              [4], [3], [2], [1],
              [4], [3], [2], [1],]
    pred2 = [[4], [3], [2], [1],
             [4], [3], [2], [1],
             [4], [3], [2], [1],
             [1], [2], [3], [4],]

    for constructor in getDataConstructors():
        knownObj1 = constructor(known1, useLog=False)
        predObj1 = constructor(pred1, useLog=False)
        labels1 = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        cm1 = confusionMatrix(knownObj1, predObj1, labels=labels1)

        knownObj2 = constructor(known2, useLog=False)
        predObj2 = constructor(pred2, useLog=False)
        labels2 = {4: 'four', 3: 'three', 2: 'two', 1: 'one'}
        cm2 = confusionMatrix(knownObj2, predObj2, labels=labels2)

        assert cm1 == cm2

@noLogEntryExpected
def test_confusionMatrix_withLabelsList():
    known = [[3], [2], [1], [0],
             [3], [2], [1], [0],
             [3], [2], [1], [0],
             [3], [2], [1], [0]]
    pred = [[3], [2], [1], [0],
            [3], [2], [1], [0],
            [3], [2], [1], [0],
            [0], [1], [2], [3]]

    for constructor in getDataConstructors():
        knownObj = constructor(known, useLog=False)
        predObj = constructor(pred, useLog=False)

        labels = ['three', 'two', 'one', 'zero']
        cm = confusionMatrix(knownObj, predObj, labels=labels)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        featureNames = ['known_' + l for l in labels]
        pointNames = ['predicted_' + l for l in labels]
        expObj = constructor(expData, pointNames, featureNames,
                             useLog=False)

        assert cm.isIdentical(expObj)

@noLogEntryExpected
def test_confusionMatrix_additionalLabelsProvided():

    # 3 never found in known or predicted but will be in labels
    known = [[0], [1], [2], [4],
             [0], [1], [2], [4],
             [0], [1], [2], [4],
             [0], [1], [2], [4]]
    pred = [[0], [1], [2], [4],
            [0], [1], [2], [4],
            [0], [1], [2], [4],
            [4], [2], [1], [0]]

    for constructor in getDataConstructors():
        knownObj = constructor(known, useLog=False)
        predObj = constructor(pred, useLog=False)

        labelsList = ['zero', 'one', 'two', 'three', 'four']
        cm1 = confusionMatrix(knownObj, predObj, labels=labelsList)

        labelsDict = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four'}
        cm2 = confusionMatrix(knownObj, predObj, labels=labelsDict)

        expData = [[3, 0, 0, 0, 1],
                   [0, 3, 1, 0, 0],
                   [0, 1, 3, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 3]]

        featureNames = ['known_' + lbl for lbl in labelsList]
        pointNames = ['predicted_' + lbl for lbl in labelsList]
        expObj = constructor(expData, pointNames, featureNames,
                             useLog=False)

        assert cm1 == expObj
        assert cm1 == cm2

@noLogEntryExpected
def test_confusionMatrix_convertCountsToFractions():
    known = [[1], [2], [3],
             [1], [2], [3],
             [1], [2], [3],
             [1], [2], [3]]
    pred = [[1], [2], [3],
            [1], [2], [3],
            [3], [2], [1],
            [3], [2], [1]]

    for constructor in getDataConstructors():
        knownObj = constructor(known, useLog=False)
        predObj = constructor(pred, useLog=False)

        cm = confusionMatrix(knownObj, predObj, convertCountsToFractions=True)

        expData = [[(1 / 6), 0, (1 / 6)],
                   [0, (1 / 3), 0],
                   [(1 / 6), 0, (1 / 6)]]

        featureNames = ['known_' + str(i) for i in range(1, 4)]
        pointNames = ['predicted_' + str(i) for i in range(1, 4)]
        expObj = constructor(expData, pointNames, featureNames, useLog=False)

        assert cm.isIdentical(expObj)

@noLogEntryExpected
def test_confusionMatrix_strings():
    known = [['dog'], ['cat'], ['fish'], ['bear'],
             ['dog'], ['cat'], ['fish'], ['bear'],
             ['dog'], ['cat'], ['fish'], ['bear'],
             ['dog'], ['cat'], ['fish'], ['bear']]
    pred = [['dog'], ['cat'], ['fish'], ['bear'],
            ['dog'], ['cat'], ['fish'], ['bear'],
            ['dog'], ['cat'], ['fish'], ['bear'],
            ['cat'], ['dog'], ['bear'], ['fish']]

    for constructor in getDataConstructors(includeSparse=False):
        knownObj = constructor(known, useLog=False)
        predObj = constructor(pred, useLog=False)

        cm = confusionMatrix(knownObj, predObj)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        featureNames = ['known_' + lab for lab in ['bear', 'cat', 'dog', 'fish']]
        pointNames = ['predicted_' + lab for lab in ['bear', 'cat', 'dog', 'fish']]
        expObj = constructor(expData, pointNames, featureNames, useLog=False)

        assert cm.isIdentical(expObj)

@noLogEntryExpected
def test_covariance_sample():
    x = [[1, 2, 3, 4],
         [2, 4, 6, 8],
         [-2, -4, -6, -8]]
    exp = [[(5/3), (10/3), (-10/3)],
           [(10/3), (20/3), (-20/3)],
           [(-10/3), (-20/3), (20/3)]]

    for constructor in getDataConstructors():
        pnames = ['a', 'b', 'c']
        toTest = constructor(x, pointNames=pnames, useLog=False)
        expObj = constructor(exp, pointNames=pnames, featureNames=pnames,
                             useLog=False)
        cov = nimble.calculate.covariance(toTest)
        assert cov.isApproximatelyEqual(expObj)

@noLogEntryExpected
def test_covariance_population():
    x = [[1, 2, 3, 4],
         [2, 4, 6, 8],
         [-2, -4, -6, -8]]
    exp = [[(5/4), (5/2), (-5/2)],
           [(5/2), 5, -5],
           [(-5/2), -5, 5]]

    for constructor in getDataConstructors():
        pnames = ['a', 'b', 'c']
        toTest = constructor(x, pointNames=pnames, useLog=False)
        expObj = constructor(exp, pointNames=pnames, featureNames=pnames,
                             useLog=False)
        cov = nimble.calculate.covariance(toTest, sample=False)
        assert cov.isApproximatelyEqual(expObj)

@noLogEntryExpected
def test_covariance_withMissing():
    x = [[1, np.nan, 3, 4],
         [2, 4, np.nan, 8],
         [-2, -4, -6, -8]]
    exp = [[(7/3), 9, (-14/3)],
           [9, (28/3), (-28/3)],
           [(-14/3), (-28/3), (20/3)]]

    for constructor in getDataConstructors():
        pnames = ['a', 'b', 'c']
        toTest = constructor(x, pointNames=pnames, useLog=False)
        expObj = constructor(exp, pointNames=pnames, featureNames=pnames,
                             useLog=False)
        cov = nimble.calculate.covariance(toTest)
        assert cov.isApproximatelyEqual(expObj)

@noLogEntryExpected
def test_correlation():
    x = [[1, 2, 3, 4],
         [2, 4, 6, 8],
         [-2, -4, -6, -8]]
    exp = [[1, 1, -1],
           [1, 1, -1],
           [-1, -1, 1]]

    for constructor in getDataConstructors():
        pnames = ['a', 'b', 'c']
        toTest = constructor(x, pointNames=pnames, useLog=False)
        expObj = constructor(exp, pointNames=pnames, featureNames=pnames,
                             useLog=False)
        cov = nimble.calculate.correlation(toTest)
        assert cov.isApproximatelyEqual(expObj)

@noLogEntryExpected
def test_correlation_withMissing():
    x = [[1, np.nan, 3, 4],
         [2, 4, np.nan, 8],
         [-2, -4, -6, -8]]
    exp = [[1, 1, -1],
           [1, 1, -1],
           [-1, -1, 1]]

    for constructor in getDataConstructors():
        pnames = ['a', 'b', 'c']
        toTest = constructor(x, pointNames=pnames, useLog=False)
        expObj = constructor(exp, pointNames=pnames, featureNames=pnames,
                             useLog=False)
        cov = nimble.calculate.correlation(toTest)
        assert cov.isApproximatelyEqual(expObj)
