"""
Tests for binary metrics based on a confusion matrix.
"""

import nimble
from nimble import train
from nimble.exceptions import InvalidArgumentValue
from nimble.calculate import (
    truePositive, trueNegative, falsePositive, falseNegative, recall,
    precision, specificity, balancedAccuracy, f1Score, confusionMatrix)
from tests.helpers import noLogEntryExpected, raises

def test_binary_confusionMatrix_nonBinary():
    known = [[1], [2], [1], [2],
             [1], [2], [1], [2],
             [1], [2], [1], [2],
             [1], [2], [1], [2],
             [1], [2], [1], [2]]
    pred = [[1], [2], [1], [2],
            [1], [1], [1], [1],
            [1], [1], [1], [1],
            [2], [1], [2], [1],
            [2], [1], [2], [1]]

    knownObj = nimble.data(known, useLog=False)
    predObj = nimble.data(pred, useLog=False)

    # check that confusionMatrix raises IndexError but using the binary
    # functions raises InvalidArgumentValue because confusionMatrix error
    # would be confusing given the user did not provide the labels
    with raises(IndexError):
        confusionMatrix(knownObj, predObj, [False, True])

    funcs = [truePositive, trueNegative, falsePositive, falseNegative,
             recall, precision, specificity, balancedAccuracy, f1Score]
    for func in funcs:
        with raises(InvalidArgumentValue):
            func(knownObj, predObj)

@noLogEntryExpected
def test_binary_confusionMatrixValues():
    known = [[1], [1], [1], [1], [1], [1],
             [0], [0], [0], [0], [0], [0], [0], [0],
             [0], [0],
             [1], [1], [1], [1]]
    pred = [[1], [1], [1], [1], [1], [1],           # 6 TP
            [1], [1], [1], [1], [1], [1], [1], [1], # 8 FP
            [0], [0],                               # 2 TN
            [0], [0], [0], [0]]                     # 4 FN

    knownObj = nimble.data(known, useLog=False)
    predObj = nimble.data(pred, useLog=False)

    expTP = 6
    expTN = 2
    expFP = 8
    expFN = 4

    assert truePositive(knownObj, predObj) == expTP
    assert trueNegative(knownObj, predObj) == expTN
    assert falsePositive(knownObj, predObj) == expFP
    assert falseNegative(knownObj, predObj) == expFN

@noLogEntryExpected
def test_binary_confusionMatrixMetrics():
    known = [[1], [1], [1], [1], [1], [1],
             [0], [0], [0], [0], [0], [0], [0], [0],
             [0], [0],
             [1], [1], [1], [1]]
    pred = [[1], [1], [1], [1], [1], [1],           # 6 TP
            [1], [1], [1], [1], [1], [1], [1], [1], # 8 FP
            [0], [0],                               # 2 TN
            [0], [0], [0], [0]]                     # 4 FN

    knownObj = nimble.data(known, useLog=False)
    predObj = nimble.data(pred, useLog=False)

    expRecall = .6
    expPrecision = 6 / 14
    expSpecificity = .2
    expBalancedAcc = .4
    expF1 = .5

    assert recall(knownObj, predObj) == expRecall
    assert precision(knownObj, predObj) == expPrecision
    assert specificity(knownObj, predObj) == expSpecificity
    assert balancedAccuracy(knownObj, predObj) == expBalancedAcc
    assert f1Score(knownObj, predObj) == expF1

@noLogEntryExpected
def test_binary_metricsAsPerformanceFunction():
    rawTrain = [[0, 0], [1, 1]]
    rawTest = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],                 # 6 TP
               [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], # 8 FP
               [0, 0], [0, 0],                                                 # 2 TN
               [1, 0], [1, 0], [1, 0], [1, 0]]                                 # 4 FN

    trainData = nimble.data(rawTrain, useLog=False)
    testData = nimble.data(rawTest, useLog=False)

    tl = train('nimble.KNNClassifier', trainData, 0, arguments={'k': 1}, useLog=False)
    score1 = tl.test(truePositive, testData, 0, useLog=False)
    assert score1 == 6
    assert truePositive.optimal == 'max'

    score2 = tl.test(falsePositive, testData, 0, useLog=False)
    assert score2 == 8
    assert falsePositive.optimal == 'min'

    score3 = tl.test(trueNegative, testData, 0, useLog=False)
    assert score3 == 2
    assert trueNegative.optimal == 'max'

    score4 = tl.test(falseNegative, testData, 0, useLog=False)
    assert score4 == 4
    assert falseNegative.optimal == 'min'

    score5 = tl.test(recall, testData, 0, useLog=False)
    assert score5 == .6
    assert recall.optimal == 'max'

    score6 = tl.test(precision, testData, 0, useLog=False)
    assert score6 == 6 / 14
    assert precision.optimal == 'max'

    score7 = tl.test(specificity, testData, 0, useLog=False)
    assert score7 == .2
    assert specificity.optimal == 'max'

    score8 = tl.test(balancedAccuracy, testData, 0, useLog=False)
    assert score8 == .4
    assert balancedAccuracy.optimal == 'max'

    score9 = tl.test(f1Score, testData, 0, useLog=False)
    assert score9 == .5
    assert f1Score.optimal == 'max'
