"""
Tests for binary metrics based on a confusion matrix.
"""

from nimble import createData, train
from nimble.exceptions import InvalidArgumentValue
from nimble.calculate import (
    truePositive, trueNegative, falsePositive, falseNegative, recall,
    precision, specificity, balancedAccuracy, f1Score, confusionMatrix)

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

    knownObj = createData('Matrix', known, elementType=int)
    predObj = createData('Matrix', pred, elementType=int)

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

def test_binary_confusionMatrixValues():
    known = [[1], [1], [1], [1], [1], [1],
             [0], [0], [0], [0], [0], [0], [0], [0],
             [0], [0],
             [1], [1], [1], [1]]
    pred = [[1], [1], [1], [1], [1], [1],           # 6 TP
            [1], [1], [1], [1], [1], [1], [1], [1], # 8 FP
            [0], [0],                               # 2 TN
            [0], [0], [0], [0]]                     # 4 FN

    knownObj = createData('Matrix', known, elementType=int)
    predObj = createData('Matrix', pred, elementType=int)

    expTP = 6
    expTN = 2
    expFP = 8
    expFN = 4

    assert truePositive(knownObj, predObj) == expTP
    assert trueNegative(knownObj, predObj) == expTN
    assert falsePositive(knownObj, predObj) == expFP
    assert falseNegative(knownObj, predObj) == expFN

def test_binary_confusionMatrixMetrics():
    known = [[1], [1], [1], [1], [1], [1],
             [0], [0], [0], [0], [0], [0], [0], [0],
             [0], [0],
             [1], [1], [1], [1]]
    pred = [[1], [1], [1], [1], [1], [1],           # 6 TP
            [1], [1], [1], [1], [1], [1], [1], [1], # 8 FP
            [0], [0],                               # 2 TN
            [0], [0], [0], [0]]                     # 4 FN

    knownObj = createData('Matrix', known, elementType=int)
    predObj = createData('Matrix', pred, elementType=int)

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

def test_binary_metricsAsPerformanceFunction():
    rawTrain = [[0, 0], [1, 1]]
    # 6 TP, 8 FP, 2 TN, 4 FN
    rawTest = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],                 # 6 TP
               [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], # 8 FP
               [0, 0], [0, 0],                                                 # 2 TN
               [1, 0], [1, 0], [1, 0], [1, 0]]                                 # 4 FN

    trainData = createData('Matrix', rawTrain)
    testData = createData('Matrix', rawTest)

    tl = train('Custom.KNNClassifier', trainData, 0, arguments={'k': 1})
    score1 = tl.test(testData, 0, truePositive)
    assert score1 == 6
    assert truePositive.optimal == 'max'

    score2 = tl.test(testData, 0, falsePositive)
    assert score2 == 8
    assert falsePositive.optimal == 'min'

    score3 = tl.test(testData, 0, trueNegative)
    assert score3 == 2
    assert trueNegative.optimal == 'max'

    score4 = tl.test(testData, 0, falseNegative)
    assert score4 == 4
    assert falseNegative.optimal == 'min'

    score5 = tl.test(testData, 0, recall)
    assert score5 == .6
    assert recall.optimal == 'max'

    score6 = tl.test(testData, 0, precision)
    assert score6 == 6 / 14
    assert precision.optimal == 'max'

    score7 = tl.test(testData, 0, specificity)
    assert score7 == .2
    assert specificity.optimal == 'max'

    score8 = tl.test(testData, 0, balancedAccuracy)
    assert score8 == .4
    assert balancedAccuracy.optimal == 'max'

    assert not hasattr(f1Score, 'optimal')
    f1Score.optimal = 'max'
    score9 = tl.test(testData, 0, f1Score)
    assert score9 == .5
    assert f1Score.optimal == 'max'
