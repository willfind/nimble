import copy

import numpy

import nimble
from nimble import CustomLearner
from nimble.learners import RidgeRegression, KNNClassifier
from tests.helpers import configSafetyWrapper
from tests.helpers import noLogEntryExpected

class LoveAtFirstSightClassifier(CustomLearner):
    """ Always predicts the value of the first class it sees in the most recently trained data """
    learnerType = 'classification'

    def incrementalTrain(self, trainX, trainY):
        if hasattr(self, 'scope'):
            self.scope = numpy.union1d(self.scope, trainY.copy(to='numpyarray').flatten())
        else:
            self.scope = numpy.unique(trainY.copy(to='numpyarray'))
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

    @classmethod
    def options(cls):
        return ['option']


class UncallableLearner(CustomLearner):
    learnerType = 'classification'

    def train(self, trainX, trainY, foo, bar=None):
        return None

    def apply(self, testX):
        return None

    @classmethod
    def options(cls):
        return ['option']

@configSafetyWrapper
def testCustomPackage():
    """ Test learner registration 'custom' CustomLearnerInterface """

    data = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    labels = [[1], [2], [0]]
    testX = nimble.data('Matrix', data)
    testY = nimble.data('Matrix', labels)

    nimble.train(LoveAtFirstSightClassifier, testX, testY)
    nimble.train(UncallableLearner, testX, testY, foo='foo')

    assert 'LoveAtFirstSightClassifier' in nimble.listLearners("custom")
    assert 'UncallableLearner' in nimble.listLearners("custom")

    assert nimble.learnerParameters("custom.LoveAtFirstSightClassifier") == []
    assert nimble.learnerParameters("custom.UncallableLearner") == ['bar', 'foo']

@configSafetyWrapper
def testNimblePackage():
    """ Test learner registration 'nimble' CustomLearnerInterface """

    data = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    labels = [[1], [2], [0]]
    testX = nimble.data('Matrix', data)
    testY = nimble.data('Matrix', labels)

    nimble.train(RidgeRegression, testX, testY, lamb=1)
    nimble.train(KNNClassifier, testX, testY, k=1)

    assert 'RidgeRegression' in nimble.listLearners("nimble")
    assert 'KNNClassifier' in nimble.listLearners("nimble")

    assert nimble.learnerParameters("nimble.RidgeRegression") == ['lamb']
    assert nimble.learnerParameters("nimble.KNNClassifier") == ['k']


# test that registering a sample custom learner with option names
# will affect nimble.settings but not the config file
@configSafetyWrapper
def testRegisterLearnersWithOptionNames():
    """ Test the availability of a custom learner's options """

    data = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    labels = [[1], [2], [0]]
    testX = nimble.data('Matrix', data)
    testY = nimble.data('Matrix', labels)

    nimble.train(LoveAtFirstSightClassifier, testX, testY)
    nimble.train(UncallableLearner, testX, testY, foo=0)

    options = ['LoveAtFirstSightClassifier.option',
               'UncallableLearner.option']
    for option in options:
        assert nimble.settings.get('custom', option) == ""
        with open(nimble.settings.path) as config:
            for line in config.readlines():
                assert 'custom' not in line
                assert option not in line


@configSafetyWrapper
def test_learnersAvailableOnImport():
    """ Test that the auto registration helper correctly registers learners """
    nimbleLearners = nimble.listLearners('nimble')
    assert 'KNNClassifier' in nimbleLearners
    assert 'RidgeRegression' in nimbleLearners
    assert 'MeanConstant' in nimbleLearners
    assert 'MultiOutputRidgeRegression' in nimbleLearners
    assert 'MultiOutputLinearRegression' in nimbleLearners

    customLearners = nimble.listLearners('custom')
    assert not customLearners

@configSafetyWrapper
@noLogEntryExpected
def test_logCount():

    data = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    labels = [[1], [2], [0]]
    testX = nimble.data('Matrix', data, useLog=False)
    testY = nimble.data('Matrix', labels, useLog=False)

    nimble.train(LoveAtFirstSightClassifier, testX, testY, useLog=False)
    lst = nimble.listLearners("custom")
    params = nimble.learnerParameters("custom.LoveAtFirstSightClassifier")
    defaults = nimble.learnerDefaultValues("custom.LoveAtFirstSightClassifier")
    lType = nimble.learnerType("custom.LoveAtFirstSightClassifier")

def test_listLearnersDirectFromModule():
    nimbleLearners = nimble.listLearners(nimble)
    assert 'KNNClassifier' in nimbleLearners
    assert 'RidgeRegression' in nimbleLearners
    assert 'MeanConstant' in nimbleLearners
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

@configSafetyWrapper
def test_learnerQueries():
    data = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
    labels = [[1], [2], [0]]
    testX = nimble.data('Matrix', data, useLog=False)
    testY = nimble.data('Matrix', labels, useLog=False)

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
