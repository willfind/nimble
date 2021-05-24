import numpy as np

import nimble
from nimble import CustomLearner
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.learners import RidgeRegression, KNNClassifier
from nimble.core.interfaces.custom_learner import validateCustomLearnerSubclass
from tests.helpers import raises
from tests.helpers import noLogEntryExpected


@raises(TypeError)
def testCustomLearnerValidationNoType():
    """ Test  CustomLearner's validation for the learnerType class attribute """

    class NoType(CustomLearner):
        def train(self, trainX, trainY):
            return None

        def apply(self, testX):
            return None

    validateCustomLearnerSubclass(NoType)


@raises(TypeError)
def testCustomLearnerValidationWrongParamsTrain():
    """ Test CustomLearner's validation of required train() parameters """

    class WrongArgs(CustomLearner):
        learnerType = 'unknown'

        def train(self, trainZ, foo):
            return None

        def apply(self, testX):
            return None

    validateCustomLearnerSubclass(WrongArgs)


@raises(TypeError)
def testCustomLearnerValidationWrongParamsIncTrain():
    """ Test CustomLearner's validation of required incrementalTrain() parameters """

    class WrongArgs(CustomLearner):
        learnerType = 'unknown'

        def train(self, trainX, trainY):
            return None

        def incrementalTrain(self, trainZ, foo):
            return None

        def apply(self, testX):
            return None

    validateCustomLearnerSubclass(WrongArgs)


@raises(TypeError)
def testCustomLearnerValidationWrongParamsApply():
    """ Test CustomLearner's validation of required apply() parameters """

    class WrongArgs(CustomLearner):
        learnerType = 'unknown'

        def train(self, trainX, trainY):
            return None

        def apply(self, testZ):
            return None

    validateCustomLearnerSubclass(WrongArgs)


@raises(TypeError)
def testCustomLearnerValidationNoTrainOrIncTrain():
    """ Test CustomLearner's validation of requiring either train() or incrementalTrain() """

    class NoTrain(CustomLearner):
        learnerType = 'unknown'

        def apply(self, testX):
            return None

    validateCustomLearnerSubclass(NoTrain)


@raises(TypeError)
def testCustomLearnerValidationGetScoresParamsMatch():
    """ Test CustomLearner's validation of the match between getScores() param names and apply()"""

    class NoType(CustomLearner):
        learnerType = 'classification'

        def train(self, trainX, trainY):
            return None

        def apply(self, testX, foo):
            return None

        def getScores(self, testX):
            return None

    validateCustomLearnerSubclass(NoType)


@raises(TypeError)
def testCustomLearnerValidationInitNoParams():
    """ Test CustomLearner's validation of __init__'s params """

    class TooMany(CustomLearner):
        learnerType = 'classification'

        def __init__(self, so, many, params):
            super(TooMany, self).__init__()

        def train(self, trainX, trainY):
            return None

        def apply(self, testX, foo):
            return None

    validateCustomLearnerSubclass(TooMany)


@raises(TypeError)
def testCustomLearnerValidationInstantiates():
    """ Test CustomLearner's validation actually tries to instantiation the subclass """

    class NoApp(CustomLearner):
        learnerType = 'classification'

        def train(self, trainX, trainY):
            return None

    validateCustomLearnerSubclass(NoApp)


class LoveAtFirstSightClassifier(CustomLearner):
    """ Always predicts the value of the first class it sees in the most recently trained data """
    learnerType = 'classification'

    def incrementalTrain(self, trainX, trainY):
        if hasattr(self, 'scope'):
            self.scope = np.union1d(self.scope, trainY.copy(to='numpyarray').flatten())
        else:
            self.scope = np.unique(trainY.copy(to='numpyarray'))
        self.prediction = trainY[0, 0]

    def apply(self, testX):
        ret = []
        for point in testX.points:
            ret.append([self.prediction])
        return nimble.data("Matrix", ret)

    def getScores(self, testX):
        ret = []
        for point in testX.points:
            currScores = []
            for value in self.scope:
                if value == self.prediction:
                    currScores.append(1)
                else:
                    currScores.append(0)
            ret.append(currScores)
        return nimble.data("Matrix", ret)


def testCustomLearnerGetScores():
    """ Test that a CustomLearner with getScores can actually call that method """
    data = [[1, 3], [2, -5], [1, 44]]
    labels = [[0], [2], [1]]

    trainObj = nimble.data('Matrix', data)
    labelsObj = nimble.data('Matrix', labels)

    tdata = [[23, 2343], [23, 22], [454, -44]]
    testObj = nimble.data('Matrix', tdata)

    name = LoveAtFirstSightClassifier
    preds = nimble.trainAndApply(name, trainX=trainObj, trainY=labelsObj, testX=testObj, scoreMode='label')
    assert len(preds.points) == 3
    assert len(preds.features) == 1
    best = nimble.trainAndApply(name, trainX=trainObj, trainY=labelsObj, testX=testObj, scoreMode='bestScore')
    assert len(best.points) == 3
    assert len(best.features) == 2
    allScores = nimble.trainAndApply(name, trainX=trainObj, trainY=labelsObj, testX=testObj, scoreMode='allScores')
    assert len(allScores.points) == 3
    assert len(allScores.features) == 3


def testCustomLearnerIncTrainCheck():
    """ Test that a CustomLearner with incrementalTrain() but no train() works as expected """
    data = [[1, 3], [2, -5], [1, 44]]
    labels = [[0], [2], [1]]
    trainObj = nimble.data('Matrix', data)
    labelsObj = nimble.data('Matrix', labels)

    tdata = [[23, 2343], [23, 22], [454, -44]]
    testObj = nimble.data('Matrix', tdata)

    def verifyScores(scores, currPredIndex):
        for rowNum in range(len(scores.points)):
            for featNum in range(len(scores.features)):
                value = scores[rowNum, featNum]
                if featNum == currPredIndex:
                    assert value == 1
                else:
                    assert value == 0

    name = LoveAtFirstSightClassifier
    tlObj = nimble.train(name, trainX=trainObj, trainY=labelsObj)
    origBackend = tlObj._backend
    assert all(val in origBackend.scope for val in [0, 1, 2])
    origAllScores = tlObj.apply(testX=testObj, scoreMode='allScores')
    # label set [0,1,2] with 0 as constant prediction value
    verifyScores(origAllScores, 0)

    extendData = [[-343, -23]]
    extendLabels = [[3]]
    extTrainObj = nimble.data("Matrix", extendData)
    extLabelsObj = nimble.data("Matrix", extendLabels)

    tlObj.incrementalTrain(extTrainObj, extLabelsObj)
    # check incrementalTrain does not instantiate a new backend
    assert tlObj._backend == origBackend
    assert 3 in origBackend.scope
    incAllScores = tlObj.apply(testX=testObj, scoreMode='allScores')
    # label set now [0,1,2,3] with 3 as constant prediction value
    verifyScores(incAllScores, 3)

    reData = [[11, 12], [13, 14], [-22, -48]]
    reLabels = [[-1], [-1], [-2]]
    reTrainObj = nimble.data("Matrix", reData)
    reLabelsObj = nimble.data('Matrix', reLabels)

    tlObj.retrain(reTrainObj, reLabelsObj)
    # retrain instantiates a new trained backend on only the retrain data,
    # so the learner is trained from scratch even if incrementalTrain is used
    assert tlObj._backend != origBackend
    assert all(val not in tlObj._backend.scope for val in origBackend.scope)
    reAllScores = tlObj.apply(testX=testObj, scoreMode='allScores')
    # label set now [-2,-1] with -1 as constant prediction value
    verifyScores(reAllScores, 1)

class OneOrZeroClassifier(CustomLearner):
    """ Classifies all data as either one or zero based on predictZero argument """
    learnerType = 'classification'

    def train(self, trainX, trainY, predictZero=False):
        if predictZero:
            self.prediction = 0
        else:
            self.prediction = 1

    def apply(self, testX):
        preds = [[self.prediction] for _ in range(len(testX.points))]
        return nimble.data("Matrix", preds)

def test_retrain_withArg():
    trainObj = nimble.random.data('Matrix', 4, 3, 0)
    testObj = nimble.data('Matrix', [[0, 0], [1, 1]])
    expZeros = nimble.zeros('Matrix', 2, 1)
    expOnes = nimble.ones('Matrix', 2, 1)

    tl = nimble.train(OneOrZeroClassifier, trainObj, 0)
    predOnes1 = tl.apply(testObj)
    assert predOnes1 == expOnes

    tl.retrain(trainObj, 0, predictZero=True)
    predZeros1 = tl.apply(testObj)
    assert predZeros1 == expZeros

@raises(InvalidArgumentValue)
def test_retrain_invalidArg():
    trainObj = nimble.random.data('Matrix', 4, 3, 0)
    testObj = nimble.data('Matrix', [[0, 0], [1, 1]])
    expZeros = nimble.zeros('Matrix', 2, 1)
    expOnes = nimble.ones('Matrix', 2, 1)

    tl = nimble.train(OneOrZeroClassifier, trainObj, 0)
    predOnes1 = tl.apply(testObj)
    assert predOnes1 == expOnes

    tl.retrain(trainObj, 0, foo=True)

@raises(InvalidArgumentValue)
def test_retrain_CVArg():
    trainObj = nimble.random.data('Matrix', 4, 3, 0)
    testObj = nimble.data('Matrix', [[0, 0], [1, 1]])
    expZeros = nimble.zeros('Matrix', 2, 1)
    expOnes = nimble.ones('Matrix', 2, 1)

    tl = nimble.train(OneOrZeroClassifier, trainObj, 0)
    predOnes1 = tl.apply(testObj)
    assert predOnes1 == expOnes

    tl.retrain(trainObj, 0, predictZero=nimble.CV([True, False]))


class UncallableLearner(CustomLearner):
    learnerType = 'classification'

    def train(self, trainX, trainY, foo, bar=None):
        return None

    def apply(self, testX):
        return None


def testCustomPackage():
    """ Test learner registration 'custom' CustomLearnerInterface """

    data = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    labels = [[1], [2], [0]]
    trainX = nimble.data('Matrix', data)
    trainY = nimble.data('Matrix', labels)

    nimble.train(LoveAtFirstSightClassifier, trainX, trainY)
    nimble.train(UncallableLearner, trainX, trainY, foo='foo')

    assert 'LoveAtFirstSightClassifier' in nimble.listLearners("custom")
    assert 'UncallableLearner' in nimble.listLearners("custom")

    assert nimble.learnerParameters("custom.LoveAtFirstSightClassifier") == []
    assert nimble.learnerParameters("custom.UncallableLearner") == ['bar', 'foo']

def testNimblePackage():
    """ Test learner registration 'nimble' CustomLearnerInterface """

    data = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    labels = [[1], [2], [0]]
    trainX = nimble.data('Matrix', data)
    trainY = nimble.data('Matrix', labels)

    nimble.train(RidgeRegression, trainX, trainY, lamb=1)
    nimble.train(KNNClassifier, trainX, trainY, k=1)

    assert 'RidgeRegression' in nimble.listLearners("nimble")
    assert 'KNNClassifier' in nimble.listLearners("nimble")

    assert nimble.learnerParameters("nimble.RidgeRegression") == ['lamb']
    assert nimble.learnerParameters("nimble.KNNClassifier") == ['k']

def test_learnersAvailableOnImport():
    """ Test that the auto registration helper correctly registers learners """
    nimbleLearners = nimble.listLearners('nimble')
    assert 'KNNClassifier' in nimbleLearners
    assert 'RidgeRegression' in nimbleLearners
    assert 'MultiOutputRidgeRegression' in nimbleLearners
    assert 'MultiOutputLinearRegression' in nimbleLearners

    customLearners = nimble.listLearners('custom')
    assert not customLearners

@noLogEntryExpected
def test_logCount():

    data = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    labels = [[1], [2], [0]]
    trainX = nimble.data('Matrix', data, useLog=False)
    trainY = nimble.data('Matrix', labels, useLog=False)

    nimble.train(LoveAtFirstSightClassifier, trainX, trainY, useLog=False)
    lst = nimble.listLearners("custom")
    params = nimble.learnerParameters("custom.LoveAtFirstSightClassifier")
    defaults = nimble.learnerDefaultValues("custom.LoveAtFirstSightClassifier")
    lType = nimble.learnerType("custom.LoveAtFirstSightClassifier")

def test_listLearnersDirectFromModule():
    nimbleLearners = nimble.listLearners(nimble)
    assert 'KNNClassifier' in nimbleLearners
    assert 'RidgeRegression' in nimbleLearners
    assert 'MultiOutputRidgeRegression' in nimbleLearners
    assert 'MultiOutputLinearRegression' in nimbleLearners

    try:
        import sklearn
        sklearnLearners = nimble.listLearners(sklearn)
        assert 'LinearRegression' in sklearnLearners
        assert 'LogisticRegression' in sklearnLearners
        assert 'KNeighborsClassifier' in sklearnLearners
    except ImportError:
        pass

def test_learnerQueries():
    params = nimble.learnerParameters(UncallableLearner)
    defaults = nimble.learnerDefaultValues(UncallableLearner)
    lType = nimble.learnerType(UncallableLearner)
    lst = nimble.listLearners("custom")

    assert params == ['bar', 'foo']
    assert defaults == {'bar': None}
    assert lType == 'classification'
    assert lst == ['UncallableLearner']

    params = nimble.learnerParameters(KNNClassifier)
    defaults = nimble.learnerDefaultValues(KNNClassifier)
    lType = nimble.learnerType(KNNClassifier)

    assert params == ['k']
    assert defaults == {'k': 5}
    assert lType == 'classification'

def test_redefinedLearner():
    data = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    labels = [[1], [2], [0]]
    trainX = nimble.data('Matrix', data)
    trainY = nimble.data('Matrix', labels)
    testX = nimble.data('Matrix', [[0, 0, 0], [1, 2, 3]])

    class Redefine(LoveAtFirstSightClassifier):
        pass

    tl1 = nimble.train(Redefine, trainX, trainY)
    out1 = tl1.apply(testX)
    assert out1 is not None

    class Redefine(UncallableLearner):
        pass

    # we should now be able to train this learner with a foo argument
    tl2 = nimble.train(Redefine, trainX, trainY, foo='foo')
    # we should not be able to apply the learner because it returns None
    try:
        tl2.apply(testX)
        assert False # expected InvalidArgumentType
    except InvalidArgumentType:
        pass

class MeanConstant(CustomLearner):
    learnerType = 'regression'

    def train(self, trainX, trainY):
        self.mean = trainY.features.statistics('mean')[0, 0]

    def apply(self, testX):
        raw = np.zeros(len(testX.points))
        np.ndarray.fill(raw, self.mean)

        ret = nimble.data("Matrix", raw, useLog=False)
        ret.transpose(useLog=False)
        return ret


def testMeanConstantSimple():
    """ Test MeanConstant by checking the ouput given simple hand made inputs """

    dataX = [[0, 0, 0], [1, 10, 10], [0, -1, 4], [1, 0, 20], [0, 1, 0], [1, 2, 3]]
    trainX = nimble.data('Matrix', dataX)

    dataY = [[0], [1], [0], [1], [0], [1]]
    trainY = nimble.data('Matrix', dataY)

    for value in [MeanConstant, 'custom.MeanConstant']:
        ret = nimble.trainAndApply(value, trainX=trainX, trainY=trainY, testX=trainX)

        assert len(ret.points) == 6
        assert len(ret.features) == 1

        assert ret[0] == .5
        assert ret[1] == .5
        assert ret[2] == .5
        assert ret[3] == .5
        assert ret[4] == .5
        assert ret[5] == .5

        customLearners = nimble.listLearners('custom')
        assert 'MeanConstant' in customLearners
