from __future__ import absolute_import
from nose.tools import raises
import sys

import UML

from UML import createData
from UML import train
from UML import trainAndApply
from UML import trainAndTest

from UML.calculate import fractionIncorrect
from UML.randomness import pythonRandom
from UML.exceptions import InvalidArgumentValueCombination
import six
from six.moves import range
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
    exp = UML.trainAndApply(learner, trainObjData, trainObjLabels, testObjNoLabels)
    expSelf = UML.trainAndApply(learner, trainObjData, trainObjLabels, trainObjData)
    # trainY is ID, testX does not contain labels; test int
    out = UML.trainAndApply(learner, trainObj, 3, testObjNoLabels)
    assert out == exp
    # trainY is ID, testX does not contain labels; test string
    out = UML.trainAndApply(learner, trainObj, 'label', testObjNoLabels)
    assert out == exp
    # trainY is Base; testX None
    out = UML.trainAndApply(learner, trainObjData, trainObjLabels, None)
    assert out == expSelf
    # trainY is ID; testX None
    out = UML.trainAndApply(learner, trainObj, 3, None)
    assert out == expSelf
    # Exception trainY is ID; testX contains labels
    try:
        out = UML.trainAndApply(learner, trainObj, 3, testObj)
        assert False # expected ValueError
    except ValueError:
        pass
    try:
        out = UML.trainAndApply(learner, trainObj, 'label', testObj)
        assert False # expected ValueError
    except ValueError:
        pass
    # Exception trainY is Base; testX contains labels
    try:
        out = UML.trainAndApply(learner, trainObjData, trainObjLabels, testObj)
        assert False # expected ValueError
    except ValueError:
        pass
    # Exception trainY is ID; testX bad shape
    try:
        out = UML.trainAndApply(learner, trainObj, 3, testObj[:, 2:])
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
    exp = UML.trainAndTest(learner, trainObjData, trainObjLabels, testObjData, testObjLabels, fractionIncorrect)
    # trainX and testX contain labels
    out1 = UML.trainAndTest(learner, trainObj, 3, testObj, 3, fractionIncorrect)
    out2 = UML.trainAndTest(learner, trainObj, 'label', testObj, 'label', fractionIncorrect)
    assert out1 == exp
    assert out2 == exp
    # trainX contains labels
    out3 = UML.trainAndTest(learner, trainObj, 3, testObjData, testObjLabels, fractionIncorrect)
    assert out3 == exp
    # testX contains labels
    out4 = UML.trainAndTest(learner, trainObjData, trainObjLabels, testObj, 3, fractionIncorrect)
    assert out4 == exp

#todo set seed and verify that you can regenerate error several times with
#crossValidateReturnBest, trainAndApply, and your own computeMetrics
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
    runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect, k=(1, 2))
    assert isinstance(runError, float)

    #with small data set
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2]]
    trainObj1 = createData('Matrix', data=data1, featureNames=variables)
    runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect, k=(1, 2))
    assert isinstance(runError, float)


def test_multioutput_learners_callable_from_all():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = UML.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = UML.createData('Matrix', data)

    trainY0 = trainY.features.copy(0)
    trainY1 = trainY.features.copy(1)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = UML.createData('Matrix', data)

    data = [[555, -555], [1, -1]]
    testY = UML.createData('Matrix', data)

    testY0 = testY.features.copy(0)
    testY1 = testY.features.copy(1)

    testName = 'Custom.MultiOutputRidgeRegression'
    wrappedName = 'Custom.RidgeRegression'

    metric = UML.calculate.meanFeaturewiseRootMeanSquareError

    # trainAndApply()
    ret_TA_multi = UML.trainAndApply(testName, trainX=trainX, trainY=trainY, testX=testX, lamb=1)
    ret_TA_0 = UML.trainAndApply(wrappedName, trainX=trainX, trainY=trainY0, testX=testX, lamb=1)
    ret_TA_1 = UML.trainAndApply(wrappedName, trainX=trainX, trainY=trainY1, testX=testX, lamb=1)

    #train, then
    TLmulti = UML.train(testName, trainX=trainX, trainY=trainY, lamb=1)
    TL0 = UML.train(wrappedName, trainX=trainX, trainY=trainY0, lamb=1)
    TL1 = UML.train(wrappedName, trainX=trainX, trainY=trainY1, lamb=1)

    # tl.apply()
    ret_TLA_multi = TLmulti.apply(testX)
    ret_TLA_0 = TL0.apply(testX)
    ret_TLA_1 = TL1.apply(testX)

    # trainAndTest()
    ret_TT_multi = UML.trainAndTest(testName, trainX=trainX, trainY=trainY, testX=testX, testY=testY,
                                    performanceFunction=metric, lamb=1)
    ret_TT_0 = UML.trainAndTest(wrappedName, trainX=trainX, trainY=trainY0, testX=testX, testY=testY0,
                                performanceFunction=metric, lamb=1)
    ret_TT_1 = UML.trainAndTest(wrappedName, trainX=trainX, trainY=trainY1, testX=testX, testY=testY1,
                                performanceFunction=metric, lamb=1)

    # trainAndTestOnTrainingData()
    ret_TTTD_multi = UML.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                                    lamb=1)
    ret_TTTD_0 = UML.trainAndTestOnTrainingData(wrappedName, trainX=trainX, trainY=trainY0, performanceFunction=metric,
                                                lamb=1)
    ret_TTTD_1 = UML.trainAndTestOnTrainingData(wrappedName, trainX=trainX, trainY=trainY1, performanceFunction=metric,
                                                lamb=1)

    # Control randomness for each cross-validation so folds are consistent
    UML.randomness.startAlternateControl(seed=0)
    ret_TTTD_multi_cv = UML.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                                       lamb=1, crossValidationError=True)
    UML.randomness.setRandomSeed(0)
    ret_TTTD_0_cv = UML.trainAndTestOnTrainingData(wrappedName, trainX=trainX, trainY=trainY0, performanceFunction=metric,
                                                   lamb=1, crossValidationError=True)
    UML.randomness.setRandomSeed(0)
    ret_TTTD_1_cv = UML.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY1, performanceFunction=metric,
                                                   lamb=1, crossValidationError=True)
    UML.randomness.endAlternateControl()

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
    trainX = UML.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = UML.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'

    TLmulti = UML.train(testName, trainX=trainX, trainY=trainY, multiClassStrategy='OneVsOne', lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndApply_scoreMode_disallowed_multiOutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = UML.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = UML.createData('Matrix', data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = UML.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'

    UML.trainAndApply(testName, trainX=trainX, trainY=trainY, testX=testX, scoreMode="allScores", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndApply_multiClassStrat_disallowed_multiOutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = UML.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = UML.createData('Matrix', data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = UML.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'

    UML.trainAndApply(testName, trainX=trainX, trainY=trainY, testX=testX, multiClassStrategy="OneVsOne", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTest_scoreMode_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = UML.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = UML.createData('Matrix', data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = UML.createData('Matrix', data)

    data = [[555, -555], [1, -1]]
    testY = UML.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'
    metric = UML.calculate.meanFeaturewiseRootMeanSquareError

    UML.trainAndTest(testName, trainX=trainX, trainY=trainY, testX=testX, testY=testY, performanceFunction=metric,
                     scoreMode="allScores", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTestOnTrainingData_scoreMode_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = UML.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = UML.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'
    metric = UML.calculate.meanFeaturewiseRootMeanSquareError

    UML.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                   scoreMode="allScores", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTest_multiclassStrat_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = UML.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = UML.createData('Matrix', data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = UML.createData('Matrix', data)

    data = [[555, -555], [1, -1]]
    testY = UML.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'
    metric = UML.calculate.meanFeaturewiseRootMeanSquareError

    UML.trainAndTest(testName, trainX=trainX, trainY=trainY, testX=testX, testY=testY, performanceFunction=metric,
                     multiClassStrategy="OneVsOne", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTestOnTrainingData_multiclassStrat_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = UML.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = UML.createData('Matrix', data)

    testName = 'Custom.MultiOutputRidgeRegression'
    metric = UML.calculate.meanFeaturewiseRootMeanSquareError

    UML.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                   multiClassStrategy="OneVsOne", lamb=1)


def test_frontend_CV_triggering():
    #with small data set
    variables = ["x1", "x2", "x3"]
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    labels = [[1], [2], [3], [1], [2]]
    trainObj = createData('Matrix', data=data, featureNames=variables)
    labelsObj = createData("Matrix", data=labels)

    class CVWasCalledException(Exception):
        pass

    def cvBackgroundCheck():
        raise CVWasCalledException()

    temp = UML.helpers.crossValidateBackend
    UML.helpers.crossValidateBackend = cvBackgroundCheck

    # confirm that the calls are being made
    try:
        try:
            train('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                  performanceFunction=fractionIncorrect, k=(1, 2))
        except CVWasCalledException:
            pass

        try:
            trainAndApply('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                          performanceFunction=fractionIncorrect, testX=trainObj, k=(1, 2))
        except CVWasCalledException:
            pass

        try:
            trainAndTest('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                         testX=trainObj, testY=labelsObj, performanceFunction=fractionIncorrect,
                         k=(1, 2))
        except CVWasCalledException:
            pass
    except Exception:
        einfo = sys.exc_info()
        six.reraise(*einfo)
    finally:
        UML.helpers.crossValidateBackend = temp

    # demonstrate some succesful calls
    tl = train('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
               performanceFunction=fractionIncorrect, k=(1, 2))
    assert hasattr(tl, 'apply')

    result = trainAndApply('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                           testX=trainObj, performanceFunction=fractionIncorrect, k=(1, 2))
    assert isinstance(result, UML.data.Matrix)

    error = trainAndTest('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                         testX=trainObj, testY=labelsObj, performanceFunction=fractionIncorrect,
                         k=(1, 2))
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
        tl = train('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj, k=(1, 2))
        assert False
    except InvalidArgumentValueCombination:
        pass

    try:
        result = trainAndApply('Custom.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                               testX=trainObj, k=(1, 2))
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
        return UML.train(learner, trainX, trainY)
    back_logCount(wrapped)

@oneLogEntryExpected
def test_trainAndApply_logCount_noCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return UML.trainAndApply(learner, trainX, trainY, testX, performanceFunction)
    back_logCount(wrapped)

@oneLogEntryExpected
def test_trainAndTest_logCount_noCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return UML.trainAndTest(learner, trainX, trainY, testX, testY, performanceFunction)
    back_logCount(wrapped)

@oneLogEntryExpected
def test_trainAndTestOnTrainingData_logCount_noCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return UML.trainAndTestOnTrainingData(learner, trainX, trainY, performanceFunction)
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_train_logCount_withCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return UML.train(learner, trainX, trainY, performanceFunction=performanceFunction, k=(1,2))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_trainAndApply_logCount_withCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return UML.trainAndApply(learner, trainX, trainY, testX, performanceFunction, k=(1,2))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_trainAndTest_logCount_withCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return UML.trainAndTest(learner, trainX, trainY, testX, testY, performanceFunction, k=(1,2))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_trainAndTestOnTrainingData_logCount_withCV():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return UML.trainAndTestOnTrainingData(learner, trainX, trainY, performanceFunction, k=(1,2))
    back_logCount(wrapped)

