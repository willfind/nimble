from __future__ import absolute_import
import sys
from unittest import mock

import six
from six.moves import range
from nose.tools import raises

import UML as nimble
from UML import createData
from UML import train
from UML import trainAndApply
from UML import trainAndTest
from UML.calculate import fractionIncorrect
from UML.randomness import pythonRandom
from UML.exceptions import InvalidArgumentValue
from UML.exceptions import InvalidArgumentValueCombination
from .assertionHelpers import logCountAssertionFactory, oneLogEntryExpected

def test_trainAndApply_dataInputs():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _ in range(numPoints)]
    trainObj = createData('Matrix', data=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = createData('Matrix', data=testData, featureNames=variables)
    testObjNoLabels = testObj[:, :2]

    learner = 'Custom.KNNClassifier'
    # Expected outcomes
    exp = nimble.trainAndApply(learner, trainObjData, trainObjLabels, testObjNoLabels)
    expSelf = nimble.trainAndApply(learner, trainObjData, trainObjLabels, trainObjData)
    # trainY is ID, testX does not contain labels; test int
    out = nimble.trainAndApply(learner, trainObj, 3, testObjNoLabels)
    assert out == exp
    # trainY is ID, testX does not contain labels; test string
    out = nimble.trainAndApply(learner, trainObj, 'label', testObjNoLabels)
    assert out == exp
    # trainY is Base; testX None
    out = nimble.trainAndApply(learner, trainObjData, trainObjLabels, None)
    assert out == expSelf
    # trainY is ID; testX None
    out = nimble.trainAndApply(learner, trainObj, 3, None)
    assert out == expSelf
    # Exception trainY is ID; testX contains labels
    try:
        out = nimble.trainAndApply(learner, trainObj, 3, testObj)
        assert False # expected ValueError
    except ValueError:
        pass
    try:
        out = nimble.trainAndApply(learner, trainObj, 'label', testObj)
        assert False # expected ValueError
    except ValueError:
        pass
    # Exception trainY is Base; testX contains labels
    try:
        out = nimble.trainAndApply(learner, trainObjData, trainObjLabels, testObj)
        assert False # expected ValueError
    except ValueError:
        pass
    # Exception trainY is ID; testX bad shape
    try:
        out = nimble.trainAndApply(learner, trainObj, 3, testObj[:, 2:])
        assert False # expected ValueError
    except ValueError:
        pass

def test_trainAndTest_dataInputs():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _pt in range(numPoints)]
    trainObj = createData('Matrix', data=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = createData('Matrix', data=testData, featureNames=variables)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    learner = 'Custom.KNNClassifier'
    # Expected outcomes
    exp = nimble.trainAndTest(learner, trainObjData, trainObjLabels, testObjData, testObjLabels, fractionIncorrect)
    # trainX and testX contain labels
    out1 = nimble.trainAndTest(learner, trainObj, 3, testObj, 3, fractionIncorrect)
    out2 = nimble.trainAndTest(learner, trainObj, 'label', testObj, 'label', fractionIncorrect)
    assert out1 == exp
    assert out2 == exp
    # trainX contains labels
    out3 = nimble.trainAndTest(learner, trainObj, 3, testObjData, testObjLabels, fractionIncorrect)
    assert out3 == exp
    # testX contains labels
    out4 = nimble.trainAndTest(learner, trainObjData, trainObjLabels, testObj, 3, fractionIncorrect)
    assert out4 == exp

#todo set seed and verify that you can regenerate error several times with
#crossValidate.bestArguments, trainAndApply, and your own computeMetrics
def test_trainAndTest():
    """Assert valid results returned for different arguments to the algorithm:
    with default ie no args
    with one argument for the algorithm
    with multiple values for one argument for the algorithm (triggers CV)
    with multiple values and a small dataset (triggers CV with intelligent folding)
    """
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data1 = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _pt in range(numPoints)]
    # data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    trainObj1 = createData('Matrix', data=data1, featureNames=variables)

    testData1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj1 = createData('Matrix', data=testData1)

    #with default ie no args
    runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect)
    assert isinstance(runError, float)

    #with one argument for the algorithm
    runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect, k=1)
    assert isinstance(runError, float)

    #with multiple values for one argument for the algorithm
    runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3,
                            fractionIncorrect, k=nimble.CV([1, 2]), numFolds=3)
    assert isinstance(runError, float)

    #with small data set
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2]]
    trainObj1 = createData('Matrix', data=data1, featureNames=variables)
    runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3,
                            fractionIncorrect, k=nimble.CV([1, 2]), numFolds=3)
    assert isinstance(runError, float)


def test_multioutput_learners_callable_from_all():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.createData('Matrix', data)

    trainY0 = trainY.features.copy(0)
    trainY1 = trainY.features.copy(1)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.createData('Matrix', data)

    data = [[555, -555], [1, -1]]
    testY = nimble.createData('Matrix', data)

    testY0 = testY.features.copy(0)
    testY1 = testY.features.copy(1)

    testName = 'Custom.MultiOutputRidgeRegression'
    wrappedName = 'Custom.RidgeRegression'

    metric = nimble.calculate.meanFeaturewiseRootMeanSquareError

    # trainAndApply()
    ret_TA_multi = nimble.trainAndApply(testName, trainX=trainX, trainY=trainY, testX=testX, lamb=1)
    ret_TA_0 = nimble.trainAndApply(wrappedName, trainX=trainX, trainY=trainY0, testX=testX, lamb=1)
    ret_TA_1 = nimble.trainAndApply(wrappedName, trainX=trainX, trainY=trainY1, testX=testX, lamb=1)

    #train, then
    TLmulti = nimble.train(testName, trainX=trainX, trainY=trainY, lamb=1)
    TL0 = nimble.train(wrappedName, trainX=trainX, trainY=trainY0, lamb=1)
    TL1 = nimble.train(wrappedName, trainX=trainX, trainY=trainY1, lamb=1)

    # tl.apply()
    ret_TLA_multi = TLmulti.apply(testX)
    ret_TLA_0 = TL0.apply(testX)
    ret_TLA_1 = TL1.apply(testX)

    # trainAndTest()
    ret_TT_multi = nimble.trainAndTest(testName, trainX=trainX, trainY=trainY, testX=testX, testY=testY,
                                    performanceFunction=metric, lamb=1)
    ret_TT_0 = nimble.trainAndTest(wrappedName, trainX=trainX, trainY=trainY0, testX=testX, testY=testY0,
                                performanceFunction=metric, lamb=1)
    ret_TT_1 = nimble.trainAndTest(wrappedName, trainX=trainX, trainY=trainY1, testX=testX, testY=testY1,
                                performanceFunction=metric, lamb=1)

    # trainAndTestOnTrainingData()
    ret_TTTD_multi = nimble.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                                    lamb=1)
    ret_TTTD_0 = nimble.trainAndTestOnTrainingData(wrappedName, trainX=trainX, trainY=trainY0, performanceFunction=metric,
                                                lamb=1)
    ret_TTTD_1 = nimble.trainAndTestOnTrainingData(wrappedName, trainX=trainX, trainY=trainY1, performanceFunction=metric,
                                                lamb=1)

    # Control randomness for each cross-validation so folds are consistent
    nimble.randomness.startAlternateControl(seed=0)
    ret_TTTD_multi_cv = nimble.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                                       lamb=1, crossValidationError=True)
    nimble.randomness.setRandomSeed(0)
    ret_TTTD_0_cv = nimble.trainAndTestOnTrainingData(wrappedName, trainX=trainX, trainY=trainY0, performanceFunction=metric,
                                                   lamb=1, crossValidationError=True)
    nimble.randomness.setRandomSeed(0)
    ret_TTTD_1_cv = nimble.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY1, performanceFunction=metric,
                                                   lamb=1, crossValidationError=True)
    nimble.randomness.endAlternateControl()

    # tl.test()
    ret_TLT_multi = TLmulti.test(testX, testY, metric)
    ret_TLT_0 = TL0.test(testX, testY0, metric)
    ret_TLT_1 = TL1.test(testX, testY1, metric)

    # confirm consistency

    # individual columns in multioutput returns should match their single output
    # counterparts
    assert ret_TA_multi[0, 0] == ret_TA_0[0]
    assert ret_TA_multi[0, 1] == ret_TA_1[0]
    assert ret_TA_multi[1, 0] == ret_TA_0[1]
    assert ret_TA_multi[1, 1] == ret_TA_1[1]

    assert ret_TLA_multi[0, 0] == ret_TLA_0[0]
    assert ret_TLA_multi[0, 1] == ret_TLA_1[0]
    assert ret_TLA_multi[1, 0] == ret_TLA_0[1]
    assert ret_TLA_multi[1, 1] == ret_TLA_1[1]

    assert ret_TT_multi == ret_TT_0
    assert ret_TT_multi == ret_TT_1

    assert ret_TTTD_multi == ret_TTTD_0
    assert ret_TTTD_multi == ret_TTTD_1

    assert ret_TTTD_multi_cv == ret_TTTD_0_cv
    assert ret_TTTD_multi_cv == ret_TTTD_1_cv

    assert ret_TLT_multi == ret_TLT_0
    assert ret_TLT_multi == ret_TLT_1

    # using trainAndApply vs getting a trained learner shouldn't matter
    assert ret_TA_multi[0, 0] == ret_TLA_0[0]
    assert ret_TA_multi[0, 1] == ret_TLA_1[0]
    assert ret_TA_multi[1, 0] == ret_TLA_0[1]
    assert ret_TA_multi[1, 1] == ret_TLA_1[1]

    assert ret_TLA_multi[0, 0] == ret_TA_0[0]
    assert ret_TLA_multi[0, 1] == ret_TA_1[0]
    assert ret_TLA_multi[1, 0] == ret_TA_0[1]
    assert ret_TLA_multi[1, 1] == ret_TA_1[1]

    assert ret_TT_multi == ret_TLT_0
    assert ret_TT_multi == ret_TLT_1

    assert ret_TLT_multi == ret_TT_0
    assert ret_TLT_multi == ret_TT_1


@raises(InvalidArgumentValueCombination)
def test_train_multiclassStrat_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'

    TLmulti = nimble.train(testName, trainX=trainX, trainY=trainY, multiClassStrategy='OneVsOne', lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndApply_scoreMode_disallowed_multiOutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.createData('Matrix', data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'

    nimble.trainAndApply(testName, trainX=trainX, trainY=trainY, testX=testX, scoreMode="allScores", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndApply_multiClassStrat_disallowed_multiOutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.createData('Matrix', data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'

    nimble.trainAndApply(testName, trainX=trainX, trainY=trainY, testX=testX, multiClassStrategy="OneVsOne", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTest_scoreMode_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.createData('Matrix', data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.createData('Matrix', data)

    data = [[555, -555], [1, -1]]
    testY = nimble.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'
    metric = nimble.calculate.meanFeaturewiseRootMeanSquareError

    nimble.trainAndTest(testName, trainX=trainX, trainY=trainY, testX=testX, testY=testY, performanceFunction=metric,
                     scoreMode="allScores", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTestOnTrainingData_scoreMode_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'
    metric = nimble.calculate.meanFeaturewiseRootMeanSquareError

    nimble.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                   scoreMode="allScores", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTest_multiclassStrat_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.createData('Matrix', data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.createData('Matrix', data)

    data = [[555, -555], [1, -1]]
    testY = nimble.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'
    metric = nimble.calculate.meanFeaturewiseRootMeanSquareError

    nimble.trainAndTest(testName, trainX=trainX, trainY=trainY, testX=testX, testY=testY, performanceFunction=metric,
                     multiClassStrategy="OneVsOne", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTestOnTrainingData_multiclassStrat_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'
    metric = nimble.calculate.meanFeaturewiseRootMeanSquareError

    nimble.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                   multiClassStrategy="OneVsOne", lamb=1)


def test_trainFunctions_cv_triggered_errors():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 10
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _pt in range(numPoints)]
    trainObj = createData('Matrix', data=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = createData('Matrix', data=testData, featureNames=variables)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    learner = 'Custom.KNNClassifier'
    # no performanceFunction (only train and trainAndApply; required in Test functions)
    try:
        nimble.train(learner, trainObjData, trainObjLabels, k=nimble.CV([1, 3]))
        assert False # expect InvalidArgumentValueCombination
    except InvalidArgumentValueCombination as iavc:
        assert "performanceFunction" in str(iavc)
    try:
        nimble.trainAndApply(learner, trainObjData, trainObjLabels, testObjData,
                             k=nimble.CV([1, 3]))
        assert False # expect InvalidArgumentValueCombination
    except InvalidArgumentValueCombination as iavc:
        assert "performanceFunction" in str(iavc)

    # numFolds too large
    try:
        nimble.train(learner, trainObjData, trainObjLabels,
                     performanceFunction=fractionIncorrect, k=nimble.CV([1, 3]), numFolds=11)
        assert False # expect InvalidArgumentValueCombination
    except InvalidArgumentValueCombination as iavc:
        assert "numFolds" in str(iavc)
    try:
        nimble.trainAndApply(learner, trainObjData, trainObjLabels, testObjData,
                             performanceFunction=fractionIncorrect, k=nimble.CV([1, 3]), numFolds=11)
        assert False # expect InvalidArgumentValueCombination
    except InvalidArgumentValueCombination as iavc:
        assert "numFolds" in str(iavc)
    try:
        nimble.trainAndTest(learner, trainObjData, trainObjLabels, testObjData,
                            testObjLabels, performanceFunction=fractionIncorrect,
                            k=nimble.CV([1, 3]), numFolds=11)
        assert False # expect InvalidArgumentValueCombination
    except InvalidArgumentValueCombination as iavc:
        assert "numFolds" in str(iavc)
    try:
        # training error
        nimble.trainAndTestOnTrainingData(learner, trainObjData, trainObjLabels,
                                          performanceFunction=fractionIncorrect,
                                          k=nimble.CV([1, 3]), numFolds=11)
        assert False # expect InvalidArgumentValueCombination
    except InvalidArgumentValueCombination as iavc:
        assert "numFolds" in str(iavc)
    try:
        # cross-validation error
        nimble.trainAndTestOnTrainingData(learner, trainObjData, trainObjLabels,
                                          performanceFunction=fractionIncorrect,
                                          crossValidationError=True, numFolds=11)
        assert False # expect InvalidArgumentValueCombination
    except InvalidArgumentValue as iavc:
        # different exception since this triggers crossValidation directly
        assert "folds" in str(iavc)

class CVWasCalledException(Exception):
    pass

def cvBackgroundCheck(*args, **kwargs):
    raise CVWasCalledException()

@mock.patch('UML.uml.crossValidate', cvBackgroundCheck)
def test_frontend_CV_triggering():
    #with small data set
    variables = ["x1", "x2", "x3"]
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    labels = [[1], [2], [3], [1], [2]]
    trainObj = createData('Matrix', data=data, featureNames=variables)
    labelsObj = createData("Matrix", data=labels)

    # confirm that the calls are being made
    try:
        try:
            train('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                  performanceFunction=fractionIncorrect, k=nimble.CV([1, 2]), numFolds=5)
            assert False # expected CVWasCalledException
        except CVWasCalledException:
            pass

        try:
            trainAndApply('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                          performanceFunction=fractionIncorrect, testX=trainObj, k=nimble.CV([1, 2]),
                          numFolds=5)
            assert False # expected CVWasCalledException
        except CVWasCalledException:
            pass

        try:
            trainAndTest('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                         testX=trainObj, testY=labelsObj, performanceFunction=fractionIncorrect,
                         k=nimble.CV([1, 2]), numFolds=5)
            assert False # expected CVWasCalledException
        except CVWasCalledException:
            pass
    except Exception:
        einfo = sys.exc_info()
        six.reraise(*einfo)

def test_frontend_CV_triggering_success():
    #with small data set
    variables = ["x1", "x2", "x3"]
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    labels = [[1], [2], [3], [1], [2]]
    trainObj = createData('Matrix', data=data, featureNames=variables)
    labelsObj = createData("Matrix", data=labels)

    tl = train('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
               performanceFunction=fractionIncorrect, k=nimble.CV([1, 2]), numFolds=5)
    assert hasattr(tl, 'apply')
    assert tl.crossValidation is not None
    assert tl.crossValidation.performanceFunction == fractionIncorrect
    assert tl.crossValidation.numFolds == 5

    result = trainAndApply('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                           testX=trainObj, performanceFunction=fractionIncorrect, k=nimble.CV([1, 2]),
                           numFolds=5)
    assert isinstance(result, nimble.data.Matrix)

    error = trainAndTest('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                         testX=trainObj, testY=labelsObj, performanceFunction=fractionIncorrect,
                         k=nimble.CV([1, 2]), numFolds=5)
    assert isinstance(error, float)


def test_train_trainAndApply_perfFunc_reqForCV():
    #with small data set
    variables = ["x1", "x2", "x3"]
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    labels = [[1], [2], [3], [1], [2]]
    trainObj = createData('Matrix', data=data, featureNames=variables)
    labelsObj = createData("Matrix", data=labels)

    # Default value of performanceFunction is None, which since we're doing
    # CV should fail
    try:
        tl = train('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj, k=nimble.CV([1, 2]))
        assert False
    except InvalidArgumentValueCombination:
        pass

    try:
        result = trainAndApply('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                               testX=trainObj, k=nimble.CV([1, 2]))
        assert False
    except InvalidArgumentValueCombination:
        pass

def back_logCount(toCall):
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(),
             int(pythonRandom.random() * 3) + 1] for _pt in range(numPoints)]
    trainObj = createData('Matrix', data=data, featureNames=variables, useLog=False)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = createData('Matrix', data=testData, featureNames=variables, useLog=False)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    out = toCall('Custom.KNNClassifier', trainObjData, trainObjLabels, testObjData,
           testObjLabels, fractionIncorrect)

@oneLogEntryExpected
def test_train_logCount_noCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.train(learner, trainX, trainY)
    back_logCount(wrapped)

@oneLogEntryExpected
def test_trainAndApply_logCount_noCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndApply(learner, trainX, trainY, testX, performanceFunction)
    back_logCount(wrapped)

@oneLogEntryExpected
def test_trainAndTest_logCount_noCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTest(learner, trainX, trainY, testX, testY, performanceFunction)
    back_logCount(wrapped)

@oneLogEntryExpected
def test_trainAndTestOnTrainingData_logCount_noCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTestOnTrainingData(learner, trainX, trainY, performanceFunction)
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_train_logCount_withCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.train(learner, trainX, trainY, performanceFunction=performanceFunction, k=nimble.CV([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_trainAndApply_logCount_withCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndApply(learner, trainX, trainY, testX, performanceFunction, k=nimble.CV([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_trainAndTest_logCount_withCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTest(learner, trainX, trainY, testX, testY, performanceFunction, k=nimble.CV([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_trainAndTestOnTrainingData_logCount_withCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTestOnTrainingData(learner, trainX, trainY, performanceFunction, k=nimble.CV([1, 2]))
    back_logCount(wrapped)

