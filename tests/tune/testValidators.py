
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

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
    def _validate(self, arguments):
        return 1 if arguments['foo'] == 'high' else 0

def test_Validator(X, Y):
    # since _validate does not call train we do not need a real learner
    gv = GenericValidator("test.Learner", X, Y, fractionIncorrect, None, False)
    assert gv.learnerName == 'test.Learner'
    assert gv.X == X
    assert gv.Y == Y
    assert gv.performanceFunction == fractionIncorrect
    assert gv.randomSeed is not None
    assert gv.useLog is False
    gv.validate(foo='high')
    gv.validate(foo='low')
    assert gv._results == [1,  0]
    assert gv._arguments == [{'foo': 'high'}, {'foo': 'low'}]
    assert gv._best == (0, {'foo': 'low'})
    gv = GenericValidator("test.Learner", X, Y, fractionCorrect, None, False)
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

    gv = GenericValidator("test.Learner", X, 0, fractionCorrect, 23,
                          useLog=True, bar='baz')
    assert gv.Y != 0 and isinstance(gv.Y, nimble.core.data.Base)
    assert gv.randomSeed == 23
    assert gv.useLog is True
    string = str(gv)
    rep = repr(gv)
    assert string == rep
    assert rep == ('GenericValidator("test.Learner", '
                   + 'performanceFunction=fractionCorrect, randomSeed=23, '
                   + 'bar="baz")')

    with raises(InvalidArgumentValueCombination):
        gv = GenericValidator("test.Learner", X, Y.points[5:], fractionCorrect,
                              None, False)

class CountingPerformance:
    """
    Used to count the number of times that the performance function is
    used. This allows for checking that number of folds is as expected
    and provides performance values that are predefined.
    """
    def __init__(self, optimal, argCount, folds=None):
        self.optimal = optimal
        self.best = None
        self.predict = lambda tl, X, args: tl.apply(X, args, useLog=False)
        if folds is None:
            num = argCount
            self.div = 1
        else:
            num = argCount * folds + argCount
            self.div = folds + 1
        # add one because the performance function is called one additional
        # time to get the overall fold performance
        self.range = iter(range(num))
        self.__name__ = 'CountingPerformance'

    def __call__(self, known, predicted):
        # StopIteration -> count more than expected
        return next(self.range) // self.div

    def reachedMax(self):
        try:
            next(self.range)
            return False # count less than expected
        except StopIteration:
            return True

@logCountAssertionFactory(9) # 3 folds for each of 3 calls to validate
def test_KFold(X, Y):
    cnt9 = CountingPerformance('max', 3, 3)
    kf = KFold("nimble.KNNClassifier", X, Y, cnt9, folds=3)
    assert isinstance(kf, CrossValidator)
    kf.validate({'k': 7})
    kf.validate({'k': 5})
    kf.validate({'k': 1})

    assert kf._results == [0, 1, 2]
    assert kf._arguments == [{'k': 7}, {'k': 5}, {'k': 1}]
    assert kf._deepResults == [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    assert kf._best == (2, {'k': 1})
    assert cnt9.reachedMax()

    with assertCalled(nimble.core.tune.KFoldIterator, '__init__'):
        kf = KFold("nimble.KNNClassifier", X, Y, cnt9, folds=3)

def test_LeaveOneOut(X, Y):
    cnt200 = CountingPerformance('min', 2, 100)
    loo = LeaveOneOut("nimble.KNNClassifier", X, Y, cnt200)
    assert isinstance(loo, CrossValidator)
    assert not hasattr(loo, 'folds')
    loo.validate({'k': 1})
    loo.validate({'k': 3})
    assert loo._results == [0, 1]
    assert loo._arguments == [{'k': 1}, {'k': 3}]
    assert loo._deepResults == [[0] * 100, [1] * 100]
    assert loo._best == (0, {'k': 1})
    assert cnt200.reachedMax()

    with assertCalled(nimble.core.tune.KFoldIterator, '__init__'):
        loo = LeaveOneOut("nimble.KNNClassifier", X, Y, cnt200)

def test_LeaveOneGroupOut(X, Y):
    cnt12 = CountingPerformance('max', 3, 4)
    logo = LeaveOneGroupOut("nimble.KNNClassifier", X, Y, cnt12, foldFeature=Y)
    assert isinstance(logo, CrossValidator)
    logo.validate({'k': 7})
    logo.validate({'k': 1})
    logo.validate({'k': 3})
    assert logo._results == [0, 1, 2]
    assert logo._arguments == [{'k': 7}, {'k': 1}, {'k': 3}]
    assert logo._deepResults == [[0] * 4, [1] * 4, [2] * 4]
    assert logo._best == (2, {'k': 3})
    assert cnt12.reachedMax()

    cnt10 = CountingPerformance('min', 2, 5)
    X.features.transform(lambda ft: [1, 2, 3, 4, 5] * 20, features=0)
    logo = LeaveOneGroupOut("nimble.KNNClassifier", X, Y, cnt10, foldFeature=0)
    logo.validate({'k': 1})
    logo.validate({'k': 3})
    assert logo._results == [0, 1]
    assert logo._arguments == [{'k': 1}, {'k': 3}]
    assert logo._deepResults == [[0] * 5, [1] * 5]
    assert logo._best == (0, {'k': 1})
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

    cnt2 = CountingPerformance('min', 2)
    hd = HoldoutData("nimble.KNNClassifier", X, Y, cnt2, validateX=validateX,
                     validateY=validateY)
    assert isinstance(hd, HoldoutValidator)
    hd.validate({'k': 1})
    hd.validate({'k': 3})
    assert not hasattr(hd, 'deepResults')
    assert hd._results == [0, 1]
    assert hd._arguments == [{'k': 1}, {'k': 3}]
    assert hd._best == (0, {'k': 1})
    assert cnt2.reachedMax()

    cnt2 = CountingPerformance('min', 2)
    hd = HoldoutData("nimble.KNNClassifier", X, 0, cnt2, validateX=validateX,
                     validateY=0)
    hd.validate({'k': 1})
    hd.validate({'k': 3})
    assert hd._results == [0, 1]
    assert hd._arguments == [{'k': 1}, {'k': 3}]
    assert hd._best == (0, {'k': 1})
    assert cnt2.reachedMax()

    cnt2 = CountingPerformance('max', 2)
    hd = HoldoutData("nimble.KNNClassifier", X, 0, cnt2,
                     validateX=validateX.copy("List"), validateY=None)
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
    cnt3 = CountingPerformance('max', 3)
    hp = HoldoutProportion("nimble.KNNClassifier", X, Y, cnt3, proportion=0.3)
    assert isinstance(hp, HoldoutValidator)
    hp.validate({'k': 1})
    hp.validate({'k': 3})
    hp.validate({'k': 5})
    assert not hasattr(hp, 'deepResults')
    assert hp._results == [0, 1, 2]
    assert hp._arguments == [{'k': 1}, {'k': 3}, {'k': 5}]
    assert hp._best == (2, {'k': 5})
    assert cnt3.reachedMax()

    cnt3 = CountingPerformance('max', 3)
    hp = HoldoutProportion("nimble.KNNClassifier", X, -1, cnt3)
    assert isinstance(hp, HoldoutValidator)
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
