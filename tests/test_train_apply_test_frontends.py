import nimble
from nimble import train
from nimble import trainAndApply
from nimble import trainAndTest
from nimble import Tuning
from nimble.calculate import fractionIncorrect
from nimble.calculate import performanceFunction
from nimble.random import pythonRandom
from nimble.learners import KNNClassifier
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from tests.helpers import raises, assertCalled
from tests.helpers import logCountAssertionFactory, oneLogEntryExpected

def test_trainAndApply_dataInputs():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _ in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables)
    testObjNoLabels = testObj[:, :2]

    for learner in ['nimble.KNNClassifier', KNNClassifier]:
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
        with raises(ValueError):
            out = nimble.trainAndApply(learner, trainObj, 3, testObj)
        with raises(ValueError):
            out = nimble.trainAndApply(learner, trainObj, 'label', testObj)
        # Exception trainY is Base; testX contains labels
        with raises(ValueError):
            out = nimble.trainAndApply(learner, trainObjData, trainObjLabels, testObj)
        # Exception trainY is ID; testX bad shape
        with raises(ValueError):
            out = nimble.trainAndApply(learner, trainObj, 3, testObj[:, 2:])

def test_trainAndTest_dataInputs():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _pt in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    learner = 'nimble.KNNClassifier'
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

def test_TrainedLearnerTest_dataInputs():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _pt in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    learner = 'nimble.KNNClassifier'
    tl = nimble.train(learner, trainObjData, trainObjLabels)
    # Expected outcome
    exp = nimble.trainAndTest(learner, trainObjData, trainObjLabels, testObjData,
                              testObjLabels, fractionIncorrect)
    # testX contains labels
    out1 = tl.test(testObj, 3, fractionIncorrect)
    out2 = tl.test(testObj, 'label', fractionIncorrect)
    assert out1 == exp
    assert out2 == exp
    # testX no labels
    out3 = tl.test(testObjData, testObjLabels, fractionIncorrect)
    assert out3 == exp


class VariablePointPredictor(nimble.CustomLearner):
    """
    This will be used to test that point name preservation for
    TrainedLearner apply is based on the number of points returned
    by the learner.
    """
    learnerType = 'unknown'

    def train(self, trainX, trainY, matchTestPoints=True):
        self.matchTestPoints = matchTestPoints

    def apply(self, testX):
        # returned object will have no axis names
        if self.matchTestPoints:
            retData = [[0]] * len(testX.points)
        else:
            retData = [0]
        return nimble.data(retData)

def test_TrainedLearnerApply_pointNamePreservation():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    getRandom = pythonRandom.random
    data = [[getRandom(), getRandom(), getRandom(), int(getRandom() * 3) + 1]
             for _pt in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testPtNames = ['test1', 'test2', 'test3']
    testObj = nimble.data(source=testData, featureNames=variables,
                          pointNames=testPtNames)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    tl1 = nimble.train(VariablePointPredictor, trainObjData, trainObjLabels)

    exp1 = nimble.data([[0], [0], [0]], pointNames=testPtNames)

    out1 = tl1.apply(testObjData)
    assert out1 == exp1

    tl2 = nimble.train(VariablePointPredictor, trainObjData, trainObjLabels,
                       matchTestPoints=False)

    exp2 = nimble.data([0])

    out2 = tl2.apply(testObjData)
    assert out2 == exp2

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
    trainObj1 = nimble.data(source=data1, featureNames=variables)

    testData1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj1 = nimble.data(source=testData1)

    #with default ie no args
    runError = trainAndTest('nimble.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect)
    assert isinstance(runError, float)

    #with one argument for the algorithm
    runError = trainAndTest('nimble.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect, k=1)
    assert isinstance(runError, float)

    #with multiple values for one argument for the algorithm
    runError = trainAndTest('nimble.KNNClassifier', trainObj1, 3, testObj1, 3,
                            fractionIncorrect, k=nimble.Tune([1, 2]),
                            tuning=Tuning(folds=3))
    assert isinstance(runError, float)

    #with small data set
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2]]
    trainObj1 = nimble.data(source=data1, featureNames=variables)
    runError = trainAndTest('nimble.KNNClassifier', trainObj1, 3, testObj1, 3,
                            fractionIncorrect, k=nimble.Tune([1, 2]),
                            tuning=Tuning(folds=3))
    assert isinstance(runError, float)


def test_multioutput_learners_callable_from_all():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.data(data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.data(data)

    trainY0 = trainY.features.copy(0)
    trainY1 = trainY.features.copy(1)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.data(data)

    data = [[555, -555], [1, -1]]
    testY = nimble.data(data)

    testY0 = testY.features.copy(0)
    testY1 = testY.features.copy(1)

    testName = 'nimble.MultiOutputRidgeRegression'
    wrappedName = 'nimble.RidgeRegression'

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
    with nimble.random.alternateControl(seed=0):
        ret_TTTD_multi_cv = nimble.trainAndTestOnTrainingData(
            testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
            lamb=1, crossValidationError=True)
    with nimble.random.alternateControl(seed=0):
        ret_TTTD_0_cv = nimble.trainAndTestOnTrainingData(
            wrappedName, trainX=trainX, trainY=trainY0,
            performanceFunction=metric, lamb=1, crossValidationError=True)
    with nimble.random.alternateControl(seed=0):
        ret_TTTD_1_cv = nimble.trainAndTestOnTrainingData(
            testName, trainX=trainX, trainY=trainY1,
            performanceFunction=metric, lamb=1, crossValidationError=True)

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
    trainX = nimble.data(data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.data(data)

    testName = 'nimble.MultiOutputRidgeRegression'

    TLmulti = nimble.train(testName, trainX=trainX, trainY=trainY, multiClassStrategy='OneVsOne', lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndApply_scoreMode_disallowed_multiOutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.data(data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.data(data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.data(data)

    testName = 'nimble.MultiOutputRidgeRegression'

    nimble.trainAndApply(testName, trainX=trainX, trainY=trainY, testX=testX, scoreMode="allScores", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndApply_multiClassStrat_disallowed_multiOutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.data(data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.data(data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.data(data)

    testName = 'nimble.MultiOutputRidgeRegression'

    nimble.trainAndApply(testName, trainX=trainX, trainY=trainY, testX=testX, multiClassStrategy="OneVsOne", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTest_scoreMode_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.data(data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.data(data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.data(data)

    data = [[555, -555], [1, -1]]
    testY = nimble.data(data)

    testName = 'nimble.MultiOutputRidgeRegression'
    metric = nimble.calculate.meanFeaturewiseRootMeanSquareError

    nimble.trainAndTest(testName, trainX=trainX, trainY=trainY, testX=testX, testY=testY, performanceFunction=metric,
                     scoreMode="allScores", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTestOnTrainingData_scoreMode_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.data(data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.data(data)

    testName = 'nimble.MultiOutputRidgeRegression'
    metric = nimble.calculate.meanFeaturewiseRootMeanSquareError

    nimble.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                   scoreMode="allScores", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTest_multiclassStrat_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.data(data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.data(data)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.data(data)

    data = [[555, -555], [1, -1]]
    testY = nimble.data(data)

    testName = 'nimble.MultiOutputRidgeRegression'
    metric = nimble.calculate.meanFeaturewiseRootMeanSquareError

    nimble.trainAndTest(testName, trainX=trainX, trainY=trainY, testX=testX, testY=testY, performanceFunction=metric,
                     multiClassStrategy="OneVsOne", lamb=1)


@raises(InvalidArgumentValueCombination)
def test_trainAndTestOnTrainingData_multiclassStrat_disallowed_multioutput():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2], [0, 1, 0],
            [0, 0, 2], ]
    trainX = nimble.data(data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10], [2, -2], [1200, -1200], [222, -222], [10, -10],
            [2, -2]]
    trainY = nimble.data(data)

    testName = 'nimble.MultiOutputRidgeRegression'
    metric = nimble.calculate.meanFeaturewiseRootMeanSquareError

    nimble.trainAndTestOnTrainingData(testName, trainX=trainX, trainY=trainY, performanceFunction=metric,
                                   multiClassStrategy="OneVsOne", lamb=1)


def test_trainFunctions_Tune_triggered_errors():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 10
    data = [[pythonRandom.random(), pythonRandom.random(),
             pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _pt in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    learner = 'nimble.KNNClassifier'
    # no performanceFunction (only train and trainAndApply; required in Test functions)
    with raises(InvalidArgumentValue, match="performanceFunction"):
        nimble.train(learner, trainObjData, trainObjLabels, k=nimble.Tune([1, 3]))
    with raises(InvalidArgumentValue, match="performanceFunction"):
        nimble.trainAndApply(learner, trainObjData, trainObjLabels, testObjData,
                             k=nimble.Tune([1, 3]))

    # folds too large
    with raises(InvalidArgumentValueCombination, match="folds"):
        nimble.train(learner, trainObjData, trainObjLabels,
                     performanceFunction=fractionIncorrect,
                     k=nimble.Tune([1, 3]),
                     tuning=Tuning(folds=11))
    with raises(InvalidArgumentValueCombination, match="folds"):
        nimble.trainAndApply(learner, trainObjData, trainObjLabels, testObjData,
                             performanceFunction=fractionIncorrect,
                             k=nimble.Tune([1, 3]),
                             tuning=Tuning(folds=11))
    with raises(InvalidArgumentValueCombination, match="folds"):
        nimble.trainAndTest(learner, trainObjData, trainObjLabels, testObjData,
                            testObjLabels, performanceFunction=fractionIncorrect,
                            k=nimble.Tune([1, 3]),
                            tuning=Tuning(folds=11))
    with raises(InvalidArgumentValueCombination, match="folds"):
        # training error
        nimble.trainAndTestOnTrainingData(learner, trainObjData, trainObjLabels,
                                          performanceFunction=fractionIncorrect,
                                          k=nimble.Tune([1, 3]),
                                          tuning=Tuning(folds=11))
    with raises(InvalidArgumentValueCombination, match="folds"):
        # cross-validation error
        nimble.trainAndTestOnTrainingData(learner, trainObjData, trainObjLabels,
                                          performanceFunction=fractionIncorrect,
                                          crossValidationError=True, folds=11,
                                          k=5)

def test_frontend_Tune_triggering():
    #with small data set
    variables = ["x1", "x2", "x3"]
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    labels = [[1], [2], [3], [1], [2]]
    trainObj = nimble.data(source=data, featureNames=variables)
    labelsObj = nimble.data(source=labels)

    calledTune = assertCalled(nimble.core.tune.Tuning, 'tune')
    # confirm that the calls are being made
    with calledTune:
        train('nimble.KNNClassifier', trainX=trainObj, trainY=labelsObj,
              performanceFunction=fractionIncorrect, k=nimble.Tune([1, 2]))

    with calledTune:
        trainAndApply('nimble.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                      performanceFunction=fractionIncorrect, testX=trainObj,
                      k=nimble.Tune([1, 2]))

    with calledTune:
        trainAndTest('nimble.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                     testX=trainObj, testY=labelsObj,
                     performanceFunction=fractionIncorrect,
                     k=nimble.Tune([1, 2]))

def test_frontend_Tune_triggering_success():
    #with small data set
    variables = ["x1", "x2", "x3"]
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    labels = [[1], [2], [3], [1], [2]]
    trainObj = nimble.data(source=data, featureNames=variables)
    labelsObj = nimble.data(source=labels)

    tl = train('nimble.KNNClassifier', trainX=trainObj, trainY=labelsObj,
               performanceFunction=fractionIncorrect, k=nimble.Tune([1, 2]))
    assert hasattr(tl, 'apply')
    assert tl.tuning is not None
    assert tl.tuning.bestArguments == {'k': 1} or tl.tuning.bestArguments == {'k': 2}
    assert tl.tuning.validator.performanceFunction == fractionIncorrect
    assert tl.tuning.validator.folds == 5

    result = trainAndApply('nimble.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                           testX=trainObj, performanceFunction=fractionIncorrect,
                           k=nimble.Tune([1, 2]))
    assert isinstance(result, nimble.core.data.Matrix)

    error = trainAndTest('nimble.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                         testX=trainObj, testY=labelsObj, performanceFunction=fractionIncorrect,
                         k=nimble.Tune([1, 2]))
    assert isinstance(error, float)


def test_train_trainAndApply_perfFunc_reqForTune():
    #with small data set
    variables = ["x1", "x2", "x3"]
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    labels = [[1], [2], [3], [1], [2]]
    trainObj = nimble.data(source=data, featureNames=variables)
    labelsObj = nimble.data(source=labels)

    # Default value of performanceFunction is None, which since we're doing
    # Tune should InvalidArgumentValue
    with raises(InvalidArgumentValue):
        tl = train('nimble.KNNClassifier', trainX=trainObj, trainY=labelsObj,
                   k=nimble.Tune([1, 2]))

    with raises(InvalidArgumentValue):
        result = trainAndApply('nimble.KNNClassifier', trainX=trainObj,
                               trainY=labelsObj, testX=trainObj,
                               k=nimble.Tune([1, 2]))

def back_logCount(toCall):
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(),
             int(pythonRandom.random() * 3) + 1] for _pt in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables, useLog=False)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables, useLog=False)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    out = toCall('nimble.KNNClassifier', trainObjData, trainObjLabels, testObjData,
           testObjLabels, fractionIncorrect)

@oneLogEntryExpected
def test_train_logCount_noTune():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.train(learner, trainX, trainY)
    back_logCount(wrapped)

@oneLogEntryExpected
def test_trainAndApply_logCount_noTune():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndApply(learner, trainX, trainY, testX,
                                    performanceFunction=performanceFunction)
    back_logCount(wrapped)

@oneLogEntryExpected
def test_trainAndTest_logCount_noTune():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTest(learner, trainX, trainY, testX, testY, performanceFunction)
    back_logCount(wrapped)

@oneLogEntryExpected
def test_trainAndTestOnTrainingData_logCount_noTune():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTestOnTrainingData(learner, trainX, trainY, performanceFunction)
    back_logCount(wrapped)

@logCountAssertionFactory(12)
def test_train_logCount_withTune_deep():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.train(learner, trainX, trainY, performanceFunction=performanceFunction, k=nimble.Tune([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(12)
def test_trainAndApply_logCount_withTune_deep():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndApply(learner, trainX, trainY, testX,
                                    performanceFunction=performanceFunction,
                                    k=nimble.Tune([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(12)
def test_trainAndTest_logCount_withTune_deep():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTest(learner, trainX, trainY, testX, testY, performanceFunction, k=nimble.Tune([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(12)
def test_trainAndTestOnTrainingData_logCount_withTune_deep():
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTestOnTrainingData(learner, trainX, trainY, performanceFunction, k=nimble.Tune([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_train_logCount_withTune_noDeep():
    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'False')
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.train(learner, trainX, trainY,
                            performanceFunction=performanceFunction,
                            k=nimble.Tune([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_trainAndApply_logCount_withTune_noDeep():
    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'False')
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndApply(learner, trainX, trainY, testX,
                                    performanceFunction=performanceFunction,
                                    k=nimble.Tune([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_trainAndTest_logCount_withTune_noDeep():
    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'False')
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTest(learner, trainX, trainY, testX, testY, performanceFunction, k=nimble.Tune([1, 2]))
    back_logCount(wrapped)

@logCountAssertionFactory(2)
def test_trainAndTestOnTrainingData_logCount_withTune_noDeep():
    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'False')
    def wrapped(learner, trainX, trainY, testX, testY, performanceFunction):
        return nimble.trainAndTestOnTrainingData(learner, trainX, trainY, performanceFunction, k=nimble.Tune([1, 2]))
    back_logCount(wrapped)

@assertCalled(nimble.core.interfaces.TrainedLearner, '_validTestData')
def test_trainAndApply_testXValidation():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _ in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables)
    testObjNoLabels = testObj[:, :2]

    learner = 'nimble.KNNClassifier'
    # Expected outcomes
    # trainY is ID, testX does not contain labels; test int
    out = nimble.trainAndApply(learner, trainObj, 3, testObjNoLabels)

@assertCalled(nimble.core.interfaces.TrainedLearner, '_validTestData')
def test_trainAndTest_testXValidation():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _ in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    learner = 'nimble.KNNClassifier'
    perfFunc = nimble.calculate.fractionIncorrect
    # Expected outcomes
    # trainY is ID, testX does not contain labels; test int
    out = nimble.trainAndTest(learner, trainObj, trainObjLabels, testObjData,
                              testObjLabels, perfFunc)

@assertCalled(nimble.core.interfaces.TrainedLearner, '_validTestData')
def test_TL_apply_testXValidation():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _ in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables)
    testObjNoLabels = testObj[:, :2]

    learner = 'nimble.KNNClassifier'
    # Expected outcomes
    # trainY is ID, testX does not contain labels; test int
    tl = nimble.train(learner, trainObj, 3)
    out = tl.apply(testObjNoLabels)

@assertCalled(nimble.core.interfaces.TrainedLearner, '_validTestData')
def test_TL_test_testXValidation():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _ in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    learner = 'nimble.KNNClassifier'
    perfFunc = nimble.calculate.fractionIncorrect
    # Expected outcomes
    # trainY is ID, testX does not contain labels; test int
    tl = nimble.train(learner, trainObj, trainObjLabels)
    out = tl.test(testObjData, testObjLabels, perfFunc)

@assertCalled(nimble.core.interfaces.TrainedLearner, '_validTestData')
def test_TL_getScores_testXValidation():
    variables = ["x1", "x2", "x3", "label"]
    numPoints = 20
    data = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random() * 3) + 1]
             for _ in range(numPoints)]
    trainObj = nimble.data(source=data, featureNames=variables)
    trainObjData = trainObj[:, :2]
    trainObjLabels = trainObj[:, 3]

    testData = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testObj = nimble.data(source=testData, featureNames=variables)
    testObjData = testObj[:, :2]
    testObjLabels = testObj[:, 3]

    learner = 'nimble.KNNClassifier'
    perfFunc = nimble.calculate.fractionIncorrect
    # Expected outcomes
    # trainY is ID, testX does not contain labels; test int
    tl = nimble.train(learner, trainObj, trainObjLabels)
    out = tl.getScores(testObjData)


def test_trainAndTestOneVsOne():
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    data2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 1, 1, 4], [0, 1, 1, 4], [0, 1, 1, 4], [0, 1, 1, 4],
             [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1],
             [0, 0, 1, 2]]
    trainObj1 = nimble.data(source=data1, featureNames=variables)
    trainObj2 = nimble.data(source=data2, featureNames=variables)

    testData1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testData2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 1, 1, 2]]
    testObj1 = nimble.data(source=testData1)
    testObj2 = nimble.data(source=testData2)

    metricFunc = fractionIncorrect

    results1 = trainAndTest('nimble.KNNClassifier', trainObj1, trainY=3, testX=testObj1, testY=3,
                            performanceFunction=metricFunc, multiClassStrategy='OneVsOne')
    results2 = trainAndTest('nimble.KNNClassifier', trainObj2, trainY=3, testX=testObj2, testY=3,
                            performanceFunction=metricFunc, multiClassStrategy='OneVsOne')

    assert results1 == 0.0
    assert results2 == 0.25

def test_trainAndTestOneVsAll():
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    data2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 1, 1, 4], [0, 1, 1, 4], [0, 1, 1, 4], [0, 1, 1, 4],
             [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1],
             [0, 0, 1, 2]]
    trainObj1 = nimble.data(source=data1, featureNames=variables)
    trainObj2 = nimble.data(source=data2, featureNames=variables)

    testData1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    testData2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 1, 1, 2]]
    testObj1 = nimble.data(source=testData1)
    testObj2 = nimble.data(source=testData2)

    metricFunc = fractionIncorrect

    results1 = trainAndTest('nimble.KNNClassifier', trainObj1, trainY=3, testX=testObj1, testY=3,
                            performanceFunction=metricFunc, multiClassStrategy='OneVsAll')
    results2 = trainAndTest('nimble.KNNClassifier', trainObj2, trainY=3, testX=testObj2, testY=3,
                            performanceFunction=metricFunc, multiClassStrategy='OneVsAll')

    assert results1 == 0.0
    assert results2 == 0.25

def test_trainAndApplyOneVsAll():
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]

    trainObj1 = nimble.data(source=data1, featureNames=variables)

    testData1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    testObj1 = nimble.data(source=testData1)

    results1 = trainAndApply('nimble.KNNClassifier', trainObj1, trainY=3,
                             testX=testObj1, scoreMode='label', multiClassStrategy='OneVsAll')
    results2 = trainAndApply('nimble.KNNClassifier', trainObj1, trainY=3,
                             testX=testObj1, scoreMode='bestScore', multiClassStrategy='OneVsAll')
    results3 = trainAndApply('nimble.KNNClassifier', trainObj1, trainY=3,
                             testX=testObj1, scoreMode='allScores', multiClassStrategy='OneVsAll')

    assert results1.copy(to="python list")[0][0] >= 0.0
    assert results1.copy(to="python list")[0][0] <= 3.0

    assert results2.copy(to="python list")[0][0]


def test_trainAndApplyOneVsOne():
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    trainObj1 = nimble.data(source=data1, featureNames=variables)

    testData1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    testObj1 = nimble.data(source=testData1)

    results1 = trainAndApply('nimble.KNNClassifier', trainObj1, trainY=3,
                             testX=testObj1, scoreMode='label', multiClassStrategy='OneVsOne')
    results2 = trainAndApply('nimble.KNNClassifier', trainObj1, trainY=3,
                             testX=testObj1, scoreMode='bestScore', multiClassStrategy='OneVsOne')
    results3 = trainAndApply('nimble.KNNClassifier', trainObj1, trainY=3,
                             testX=testObj1, scoreMode='allScores', multiClassStrategy='OneVsOne')

    assert results1._data[0][0] == 1
    assert results1._data[1][0] == 2
    assert results1._data[2][0] == 3
    assert len(results1._data) == 3

    assert results2._data[0][0] == 1
    assert results2._data[0][1] == 2
    assert results2._data[1][0] == 2
    assert results2._data[1][1] == 2
    assert results2._data[2][0] == 3
    assert results2._data[2][1] == 2

    results3FeatureMap = results3.features.getNames()
    for i in range(len(results3._data)):
        row = results3._data[i]
        for j in range(len(row)):
            score = row[j]
            # because our input data was matrix, we have to check feature names
            # as they would have been generated from float data
            if i == 0:
                if score == 2:
                    assert results3FeatureMap[j] == str(float(1))
            elif i == 1:
                if score == 2:
                    assert results3FeatureMap[j] == str(float(2))
            else:
                if score == 2:
                    assert results3FeatureMap[j] == str(float(3))


def test_trainAndTest_scoreModes():
    trainX = nimble.data([[0, 0], [2, 2], [-2, -2]] * 10)
    trainY = nimble.data([0, 1, 2] * 10).T
    testX = nimble.data([[0, 0], [1, 1], [2, 2], [1, 1], [-1, -2]])
    testY = nimble.data([0, 0, 1, 1, 2]).T

    # with bestScore
    @performanceFunction('min', 0, requires1D=False, sameFtCount=False)
    def votesWhenIncorrect(knownValues, predictedValues):
        incorrect = 0
        for true, best in zip(knownValues, predictedValues.points):
            if true != best[0]:
                incorrect += best[1]
        return incorrect

    performance = nimble.trainAndTest(
        'nimble.KNNClassifier', trainX, trainY, testX, testY,
        votesWhenIncorrect, scoreMode='bestScore', k=3)
    assert performance == 2 # 1 incorrect label receiving 2 votes

    # with allScores
    @performanceFunction('max', 1, requires1D=False, sameFtCount=False)
    def correctVoteRatio(knownValues, predictedValues):
        cumulative = 0
        totalVotes = 0
        for true, votes in zip(knownValues, predictedValues.points):
            cumulative += votes[true]
            totalVotes += sum(votes)
        return cumulative / totalVotes

    trainX = nimble.data([[0, 0], [2, 2], [-2, -2]] * 10)
    trainY = nimble.data([0, 1, 2] * 10).T
    testX = nimble.data([[0, 0], [1, 1], [2, 2], [1, 1], [-1, -2]])
    testY = nimble.data([0, 0, 1, 1, 2]).T
    performance = nimble.trainAndTest(
        'nimble.KNNClassifier', trainX, trainY, testX, testY, correctVoteRatio,
        scoreMode='allScores', k=3)
    assert performance == 0.8 # 12/15 votes correct
