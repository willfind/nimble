import re

from pytest import fixture

import nimble
from nimble.core.tune import Validator, CrossValidator, HoldoutValidator
from nimble.core.tune import KFold, LeaveOneOut, LeaveOneGroupOut
from nimble.core.tune import HoldoutData, HoldoutProportion
from nimble.calculate import fractionCorrect, fractionIncorrect
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination

from tests.helpers import raises, logCountAssertionFactory, assertCalled

@fixture
def X():
    return nimble.random.data(100, 9, 0)

@fixture
def Y():
    return nimble.random.data(100, 1, 0, elementType='int') // 25

class GenericValidator(Validator):
    name = "generic"
    def _validate(self, arguments, useLog):
        return 1 if arguments['foo'] == 'high' else 0

def test_Validator(X, Y):
    # since _validate does not call train we do not need a real learner
    gv = GenericValidator("test.Learner", X, Y, fractionIncorrect, None)
    assert gv.learnerName == 'test.Learner'
    assert gv.X == X
    assert gv.Y == Y
    assert gv.performanceFunction == fractionIncorrect
    assert gv.performanceFunction.optimal == 'min'
    assert gv.randomSeed is not None
    gv.validate(foo='high')
    gv.validate(foo='low')
    assert gv._results == [1,  0]
    assert gv._arguments == [{'foo': 'high'}, {'foo': 'low'}]
    assert gv._best == (0, {'foo': 'low'})
    gv = GenericValidator("test.Learner", X, Y, fractionCorrect, None)
    gv.validate(foo='high')
    gv.validate(foo='low')
    assert gv._results == [1,  0]
    assert gv._arguments == [{'foo': 'high'}, {'foo': 'low'}]
    assert gv._best == (1, {'foo': 'high'})

    string = str(gv)
    rep = repr(gv)
    assert string == rep
    exp = (r'GenericValidator\("test.Learner", '
           + r'performanceFunction=fractionCorrect, randomSeed=[0-9]+\)')
    assert re.match(exp, rep)

    gv = GenericValidator("test.Learner", X, 0, fractionCorrect, 23, bar='baz')
    assert gv.Y != 0 and isinstance(gv.Y, nimble.core.data.Base)
    assert gv.randomSeed == 23
    string = str(gv)
    rep = repr(gv)
    assert string == rep
    assert rep == ('GenericValidator("test.Learner", '
                   + 'performanceFunction=fractionCorrect, randomSeed=23, '
                   + 'bar="baz")')

    with raises(InvalidArgumentValue):
        gv = GenericValidator("test.Learner", X, None, fractionCorrect, None)

    with raises(InvalidArgumentValueCombination):
        gv = GenericValidator("test.Learner", X, Y.points[5:], fractionCorrect,
                              None)

class CountingPerformance:
    """
    Used to count the number of times that the performance function is
    used. This allows for checking that number of folds is as expected
    and provides performance values that are predefined.
    """
    def __init__(self, max, optimal):
        self.range = iter(range(max))
        self.optimal = optimal
        self.__name__ = 'CountingPerformance'
        # prevent an additional use of the function to calculate the overall
        # folds performance so that the max value can be set based on the
        # number of folds.
        self.avgFolds = True

    def __call__(self, known, predicted):
        return next(self.range) # StopIteration -> count more than expected

    def reachedMax(self):
        try:
            next(self.range)
            return False # count less than expected
        except StopIteration:
            return True

@logCountAssertionFactory(9) # 3 folds for each of 3 calls to validate
def test_KFold(X, Y):
    cnt9 = CountingPerformance(9, 'max')
    kf = KFold("nimble.KNNClassifier", X, Y, cnt9, folds=3)
    assert kf.folds == 3
    assert isinstance(kf, CrossValidator)
    kf.validate({'k': 7})
    kf.validate({'k': 5})
    kf.validate({'k': 1})

    assert kf._results == [1.0, 4.0, 7.0]
    assert kf._arguments == [{'k': 7}, {'k': 5}, {'k': 1}]
    assert kf._deepResults == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert kf._best == (7.0, {'k': 1})
    assert cnt9.reachedMax()

    with assertCalled(nimble.core.tune.KFoldIterator, '__init__'):
        kf = KFold("nimble.KNNClassifier", X, Y, cnt9, folds=3)

def test_LeaveOneOut(X, Y):
    cnt200 = CountingPerformance(200, 'min')
    loo = LeaveOneOut("nimble.KNNClassifier", X, Y, cnt200)
    assert isinstance(loo, CrossValidator)
    assert not hasattr(loo, 'folds')
    loo.validate({'k': 1})
    loo.validate({'k': 3})
    assert loo._results == [49.5, 149.5]
    assert loo._arguments == [{'k': 1}, {'k': 3}]
    assert loo._deepResults == [list(range(100)), list(range(100, 200))]
    assert loo._best == (49.5, {'k': 1})
    assert cnt200.reachedMax()

    with assertCalled(nimble.core.tune.KFoldIterator, '__init__'):
        loo = LeaveOneOut("nimble.KNNClassifier", X, Y, cnt200)

def test_LeaveOneGroupOut(X, Y):
    cnt12 = CountingPerformance(12, 'max')
    logo = LeaveOneGroupOut("nimble.KNNClassifier", X, Y, cnt12, foldFeature=Y)
    assert isinstance(logo, CrossValidator)
    assert logo.foldFeature == "Matrix"
    logo.validate({'k': 7})
    logo.validate({'k': 1})
    logo.validate({'k': 3})
    assert logo._results == [1.5, 5.5, 9.5]
    assert logo._arguments == [{'k': 7}, {'k': 1}, {'k': 3}]
    assert logo._deepResults == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert logo._best == (9.5, {'k': 3})
    assert cnt12.reachedMax()

    cnt10 = CountingPerformance(10, 'min')
    X.features.transform(lambda ft: [1, 2, 3, 4, 5] * 20, features=0)
    logo = LeaveOneGroupOut("nimble.KNNClassifier", X, Y, cnt10, foldFeature=0)
    logo.validate({'k': 1})
    logo.validate({'k': 3})
    assert logo._results == [2, 7]
    assert logo._arguments == [{'k': 1}, {'k': 3}]
    assert logo._deepResults == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert logo._best == (2, {'k': 1})
    assert cnt10.reachedMax()

    with assertCalled(nimble.core.tune.GroupFoldIterator, '__init__'):
        logo = LeaveOneGroupOut("nimble.KNNClassifier", X, Y, cnt10,
                                foldFeature=0)

    with raises(InvalidArgumentValueCombination):
        logo = LeaveOneGroupOut("nimble.KNNClassifier", X, 0, cnt10,
                                foldFeature=0)

    with raises(InvalidArgumentValue):
        logo = LeaveOneGroupOut("nimble.KNNClassifier", X, Y, cnt10,
                                foldFeature=X[5:, 0])

def test_HoldoutData(X, Y):
    validateX = X[:20, :]
    validateY = Y[:20, :]

    cnt2 = CountingPerformance(2, 'min')
    hd = HoldoutData("nimble.KNNClassifier", X, Y, cnt2, validateX=validateX,
                     validateY=validateY)
    assert isinstance(hd, HoldoutValidator)
    assert hd.validateX == hd.validateY == "Matrix"
    hd.validate({'k': 1})
    hd.validate({'k': 3})
    assert not hasattr(hd, 'deepResults')
    assert hd._results == [0, 1]
    assert hd._arguments == [{'k': 1}, {'k': 3}]
    assert hd._best == (0, {'k': 1})
    assert cnt2.reachedMax()

    cnt2 = CountingPerformance(2, 'min')
    hd = HoldoutData("nimble.KNNClassifier", X, 0, cnt2, validateX=validateX,
                     validateY=0)
    assert hd.validateX == "Matrix"
    assert hd.validateY == 0
    hd.validate({'k': 1})
    hd.validate({'k': 3})
    assert hd._results == [0, 1]
    assert hd._arguments == [{'k': 1}, {'k': 3}]
    assert hd._best == (0, {'k': 1})
    assert cnt2.reachedMax()

    cnt2 = CountingPerformance(2, 'max')
    hd = HoldoutData("nimble.KNNClassifier", X, 0, cnt2,
                     validateX=validateX.copy("List"), validateY=None)
    assert hd.validateX == "List"
    assert hd.validateY == 0
    hd.validate({'k': 1})
    hd.validate({'k': 3})
    assert hd._results == [0, 1]
    assert hd._arguments == [{'k': 1}, {'k': 3}]
    assert hd._best == (1, {'k': 3})
    assert cnt2.reachedMax()

    with raises(InvalidArgumentValue):
        hd = HoldoutData("nimble.KNNClassifier", X, Y, cnt2, validateX=validateX,
                         validateY=None)

@logCountAssertionFactory(6) # 1 for each call to validate
def test_HoldoutProportion(X, Y):
    cnt3 = CountingPerformance(3, 'max')
    hp = HoldoutProportion("nimble.KNNClassifier", X, Y, cnt3, proportion=0.3)
    assert isinstance(hp, HoldoutValidator)
    assert hp.proportion == 0.3
    hp.validate({'k': 1})
    hp.validate({'k': 3})
    hp.validate({'k': 5})
    assert not hasattr(hp, 'deepResults')
    assert hp._results == [0, 1, 2]
    assert hp._arguments == [{'k': 1}, {'k': 3}, {'k': 5}]
    assert hp._best == (2, {'k': 5})
    assert cnt3.reachedMax()

    cnt3 = CountingPerformance(3, 'max')
    hp = HoldoutProportion("nimble.KNNClassifier", X, -1, cnt3)
    assert isinstance(hp, HoldoutValidator)
    assert hp.proportion == 0.2
    hp.validate({'k': 1})
    hp.validate({'k': 3})
    hp.validate({'k': 5})
    assert not hasattr(hp, 'deepResults')
    assert hp._results == [0, 1, 2]
    assert hp._arguments == [{'k': 1}, {'k': 3}, {'k': 5}]
    assert hp._best == (2, {'k': 5})
    assert cnt3.reachedMax()

    with raises(InvalidArgumentValue):
        hp = HoldoutProportion("nimble.KNNClassifier", X, Y, cnt3,
                               proportion=3)
