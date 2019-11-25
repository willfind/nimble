from __future__ import absolute_import
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

    knownObj = createData('Matrix', known)
    predObj = createData('Matrix', pred)

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

def test_confusionMatrix_exception_wrongLabelLength():
    known = [[0], [1], [2], [3],
             [0], [1], [2], [3],
             [0], [1], [2], [3],
             [0], [1], [2], [3]]
    pred = [[0], [1], [2], [3],
            [0], [1], [2], [3],
            [0], [1], [2], [3],
            [3], [2], [1], [0]]

    knownObj = createData('Matrix', known)
    predObj = createData('Matrix', pred)

    # long
    try:
        labels = ['zero', 'one', 'two', 'three', 'four']
        cm = confusionMatrix(knownObj, predObj, labels=labels)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

    try:
        labels = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four'}
        cm = confusionMatrix(knownObj, predObj, labels=labels)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

    # short
    try:
        labels = ['zero', 'one', 'two']
        cm = confusionMatrix(knownObj, predObj, labels=labels)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

    try:
        labels = {0:'zero', 1:'one', 2:'two'}
        cm = confusionMatrix(knownObj, predObj, labels=labels)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

@raises(IndexError)
def test_confusionMatrix_exception_labelListInvalid():
    known = [[1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4],
             [1], [2], [3], [4]]
    pred = [[1], [2], [3], [4],
            [1], [2], [3], [4],
            [1], [2], [3], [4],
            [4], [3], [2], [1]]

    knownObj = createData('Matrix', known)
    predObj = createData('Matrix', pred)

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

    knownObj = createData('Matrix', known)
    predObj = createData('Matrix', pred)

    labels = {0:'zero', 1:'one', 2:'two', 3:'three'}
    cm = confusionMatrix(knownObj, predObj, labels=labels)

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
        knownObj = createData(t, known, elementType=int)
        predObj = createData(t, pred, elementType=int)

        cm = confusionMatrix(knownObj, predObj)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        featureNames = ['known_' + str(i) for i in range(1, 5)]
        pointNames = ['predicted_' + str(i) for i in range(1, 5)]
        expObj = createData(t, expData, pointNames, featureNames, elementType=int)

        assert cm.isIdentical(expObj)

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
        knownObj = createData(t, known, elementType=int)
        predObj = createData(t, pred, elementType=int)

        labels = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        cm = confusionMatrix(knownObj, predObj, labels=labels)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        sortedLabels = [labels[i] for i in range(1, 5)]
        featureNames = ['known_' + l for l in sortedLabels]
        pointNames = ['predicted_' + l for l in sortedLabels]
        expObj = createData(t, expData, pointNames, featureNames, elementType=int)

        assert cm.isIdentical(expObj)

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
        knownObj = createData(t, known, elementType=int)
        predObj = createData(t, pred, elementType=int)

        labels = ['three', 'two', 'one', 'zero']
        cm = confusionMatrix(knownObj, predObj, labels=labels)

        expData = [[3, 0, 0, 1],
                   [0, 3, 1, 0],
                   [0, 1, 3, 0],
                   [1, 0, 0, 3]]

        featureNames = ['known_' + l for l in labels]
        pointNames = ['predicted_' + l for l in labels]
        expObj = createData(t, expData, pointNames, featureNames, elementType=int)

        assert cm.isIdentical(expObj)
