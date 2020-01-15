"""
Tests for binary metrics based on a confusion matrix.
"""

from nimble import createData, train
from nimble.exceptions import InvalidArgumentValue
from nimble.calculate import (
    truePositive, trueNegative, falsePositive, falseNegative, recall,
    precision, specificity, balancedAccuracy, f1Score, confusionMatrix)
from ..assertionHelpers import noLogEntryExpected

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

    knownObj = createData('Matrix', known, useLog=False)
    predObj = createData('Matrix', pred, useLog=False)

    # check that confusionMatrix raises IndexError but using the binary
    # functions raises InvalidArgumentValue because confusionMatrix error
    # would be confusing given the user did not provide the labels
    try:
        confusionMatrix(knownObj, predObj, [False, True])
        assert False # expected IndexError
    except IndexError:
        pass

    funcs = [truePositive, trueNegative, falsePositive, falseNegative,
             recall, precision, specificity, balancedAccuracy, f1Score]
    for func in funcs:
        try:
            func(knownObj, predObj)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

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

    knownObj = createData('Matrix', known, useLog=False)
    predObj = createData('Matrix', pred, useLog=False)

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

    knownObj = createData('Matrix', known, useLog=False)
    predObj = createData('Matrix', pred, useLog=False)

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

    trainData = createData('Matrix', rawTrain, useLog=False)
    testData = createData('Matrix', rawTest, useLog=False)

    tl = train('Custom.KNNClassifier', trainData, 0, arguments={'k': 1}, useLog=False)
    score1 = tl.test(testData, 0, truePositive, useLog=False)
    assert score1 == 6
    assert truePositive.optimal == 'max'

    score2 = tl.test(testData, 0, falsePositive, useLog=False)
    assert score2 == 8
    assert falsePositive.optimal == 'min'

    score3 = tl.test(testData, 0, trueNegative, useLog=False)
    assert score3 == 2
    assert trueNegative.optimal == 'max'

    score4 = tl.test(testData, 0, falseNegative, useLog=False)
    assert score4 == 4
    assert falseNegative.optimal == 'min'

    score5 = tl.test(testData, 0, recall, useLog=False)
    assert score5 == .6
    assert recall.optimal == 'max'

    score6 = tl.test(testData, 0, precision, useLog=False)
    assert score6 == 6 / 14
    assert precision.optimal == 'max'

    score7 = tl.test(testData, 0, specificity, useLog=False)
    assert score7 == .2
    assert specificity.optimal == 'max'

    score8 = tl.test(testData, 0, balancedAccuracy, useLog=False)
    assert score8 == .4
    assert balancedAccuracy.optimal == 'max'

    assert not hasattr(f1Score, 'optimal')
    f1Score.optimal = 'max'
    score9 = tl.test(testData, 0, f1Score, useLog=False)
    assert score9 == .5
    assert f1Score.optimal == 'max'
