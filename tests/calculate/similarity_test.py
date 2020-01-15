import math

import numpy
from nose.tools import raises

import nimble
from nimble import createData
from nimble.calculate import cosineSimilarity
from nimble.calculate import rSquared
from nimble.calculate import confusionMatrix
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from ..assertionHelpers import noLogEntryExpected

####################
# cosineSimilarity #
####################
@noLogEntryExpected
def test_cosineSimilarity():
    orig = numpy.array([[1], [0]])
    orth = numpy.array([[0], [1]])
    neg = numpy.array([[-1], [0]])

    origMatrix = createData('Matrix', data=orig, useLog=False)
    orthMatrix = createData('Matrix', data=orth, useLog=False)
    negMatrix = createData('Matrix', data=neg, useLog=False)

    result0 = cosineSimilarity(origMatrix, origMatrix)
    result1 = cosineSimilarity(origMatrix, orthMatrix)
    result2 = cosineSimilarity(origMatrix, negMatrix)

    assert result0 == 1
    assert result1 == 0
    assert result2 == -1

def test_cosineSimilarityZeros():
    zeros = [[0], [0]]

    zerosMatrix = createData('Matrix', data=zeros)

    result0 = cosineSimilarity(zerosMatrix, zerosMatrix)

    assert math.isnan(result0)

@raises(InvalidArgumentType)
def test_cosineSimilarityKnownWrongType():
    orig = numpy.array([[1], [0]])
    origMatrix = createData('Matrix', data=orig)

    result = cosineSimilarity(orig, origMatrix)

@raises(InvalidArgumentType)
def test_cosineSimilarityPredictedWrongType():
    orig = numpy.array([[1], [0]])
    origMatrix = createData('Matrix', data=orig)

    result = cosineSimilarity(origMatrix, orig)

@raises(InvalidArgumentValue)
def test_cosineSimilarityPredictedWrongShape():
    orig = numpy.array([[1], [0]])
    origMatrix = createData('Matrix', data=orig)
    pred = numpy.array([[1, 1], [0, 0]])
    predMatrix = createData('Matrix', data=pred)

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

    knownObj = createData('Matrix', known, useLog=False)
    predObj = createData('Matrix', pred, useLog=False)

    try:
        cm = confusionMatrix(known, predObj)
        assert False # expected InvalidArgumentType
    except InvalidArgumentType:
        pass

    try:
        cm = confusionMatrix(knownObj, pred)
        assert False # expected InvalidArgumentType
    except InvalidArgumentType:
        pass

def test_confusionMatrix_exception_labelsMissingKnown():
    known = [[0], [1], [2], [3],
             [0], [1], [2], [3],
             [0], [1], [2], [3],
             [0], [1], [2], [3]]
    pred = [[0], [1], [2], [3],
            [0], [1], [2], [3],
            [0], [1], [2], [3],
            [3], [2], [1], [0]]

    knownObj = createData('Matrix', known, useLog=False)
    predObj = createData('Matrix', pred, useLog=False)

    # short
    try:
        labels = ['zero', 'one', 'two']
        cm = confusionMatrix(knownObj, predObj, labels=labels)
        assert False # expected IndexError
    except IndexError:
        pass

    try:
        labels = {0:'zero', 1:'one', 2:'two'}
        cm = confusionMatrix(knownObj, predObj, labels=labels)
        assert False # expected KeyError
    except KeyError:
        pass

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

    knownObj = createData('Matrix', known, useLog=False)
    predObj = createData('Matrix', pred, useLog=False)

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

    knownObj = createData('Matrix', known, useLog=False)
    predObj = createData('Matrix', pred, useLog=False)

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

    knownObj = createData('Matrix', known, useLog=False)
    predObj = createData('Matrix', pred, useLog=False)

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

    for t in nimble.data.available:
        knownObj = createData(t, known, useLog=False)
        predObj = createData(t, pred, useLog=False)

        cm = confusionMatrix(knownObj, predObj)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        featureNames = ['known_' + str(i) for i in range(1, 5)]
        pointNames = ['predicted_' + str(i) for i in range(1, 5)]
        expObj = createData(t, expData, pointNames, featureNames,
                            elementType=int, useLog=False)

        assert cm.isIdentical(expObj)

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

    for t in nimble.data.available:
        knownObj = createData(t, known, useLog=False)
        predObj = createData(t, pred, useLog=False)

        labels = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        cm = confusionMatrix(knownObj, predObj, labels=labels)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        sortedLabels = [labels[i] for i in range(1, 5)]
        featureNames = ['known_' + l for l in sortedLabels]
        pointNames = ['predicted_' + l for l in sortedLabels]
        expObj = createData(t, expData, pointNames, featureNames,
                            elementType=int, useLog=False)

        assert cm.isIdentical(expObj)

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

    for t in nimble.data.available:
        knownObj = createData(t, known, useLog=False)
        predObj = createData(t, pred, useLog=False)

        labels = ['three', 'two', 'one', 'zero']
        cm = confusionMatrix(knownObj, predObj, labels=labels)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        featureNames = ['known_' + l for l in labels]
        pointNames = ['predicted_' + l for l in labels]
        expObj = createData(t, expData, pointNames, featureNames,
                            elementType=int, useLog=False)

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

    for t in nimble.data.available:
        knownObj = createData(t, known, useLog=False)
        predObj = createData(t, pred, useLog=False)

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
        expObj = createData(t, expData, pointNames, featureNames,
                            elementType=int, useLog=False)

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

    for t in nimble.data.available:
        knownObj = createData(t, known, useLog=False)
        predObj = createData(t, pred, useLog=False)

        cm = confusionMatrix(knownObj, predObj, convertCountsToFractions=True)

        expData = [[(1 / 6), 0, (1 / 6)],
                   [0, (1 / 3), 0],
                   [(1 / 6), 0, (1 / 6)]]

        featureNames = ['known_' + str(i) for i in range(1, 4)]
        pointNames = ['predicted_' + str(i) for i in range(1, 4)]
        expObj = createData(t, expData, pointNames, featureNames, useLog=False)

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

    for t in nimble.data.available:
        knownObj = createData(t, known, useLog=False)
        predObj = createData(t, pred, useLog=False)

        cm = confusionMatrix(knownObj, predObj)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        featureNames = ['known_' + lab for lab in ['bear', 'cat', 'dog', 'fish']]
        pointNames = ['predicted_' + lab for lab in ['bear', 'cat', 'dog', 'fish']]
        expObj = createData(t, expData, pointNames, featureNames, elementType=int,
                            useLog=False)

        assert cm.isIdentical(expObj)
