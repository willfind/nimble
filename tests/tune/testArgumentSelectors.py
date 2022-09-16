import operator
import time
from timeit import default_timer

import pytest
from pytest import fixture

import nimble
from nimble import Tune
from nimble.random import pythonRandom
from nimble.core.tune import ArgumentSelector
from nimble.core.tune import BruteForce, Consecutive, Bayesian, Iterative
from nimble.core.tune import StochasticRandomMutator
from nimble.calculate import fractionIncorrect
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from tests.helpers import noLogEntryExpected, raises
from nimble._utility import DeferredModuleImport

storm_tuner = DeferredModuleImport('storm_tuner')
hyperopt = DeferredModuleImport('hyperopt')

def wait(sec):
    def performance(args):
        time.sleep(sec)
        return 0
    return performance

class MockValidator:
    """
    Mocks a Validator object and uses attributes and/or methods to
    control the output of the calls to Validator.validate.
    """
    def __init__(self, optimal):
        if optimal == 'max':
            self._isBest = operator.gt
            self.optimal = 'max'

        else:
            self._isBest = operator.lt
            self.optimal = 'min'
        self._results = []
        self._arguments = []
        self._trainedLearner = None
        self._trainedLearnerBase = None
        self._incremental = None
        self._setIncremental = False
        self._defaultGenerateFunc = self._performanceFunction

    def reset(self, hasIncremental=False):
        self._results = []
        self._arguments = []
        self._incremental = None
        self._setIncremental = hasIncremental
        self._performanceFunction = self._defaultGenerateFunc

    def _updateTrainedLearner(self, trial):
        pass

    def _setNextPerformance(self, value):
        """
        Allows for predetermining the performance.
        """
        self._performanceFunction = lambda _: value

    def _performanceFunction(self, args):
        return 0

    def validate(self, arguments, record=True):
        self._incremental = self._setIncremental
        ret = self._performanceFunction(arguments)
        self._results.append(ret)
        self._arguments.append(arguments.copy())
        return ret

@fixture
def maxValidator():
    return MockValidator('max')

@fixture
def minValidator():
    return MockValidator('min')


class IncrementArgumentSelector(ArgumentSelector):
    name = "increment"

    def __init__(self, arguments, validator, iters=10):
        super().__init__(arguments, validator, iters=iters)
        self.iters = iters
        self._i = 0
        self._curr = self.arguments.copy()
        self._last = 0

    def __next__(self):
        if self._i >= self.iters:
            raise StopIteration
        if self._last > 0:
            inc = 1
        elif self._last < 0:
            inc = -1
        else:
            inc = 0
        for name in self._curr:
            self._curr[name] += inc
        performance = self.validator.validate(self._curr)
        self._last = performance
        self._i += 1
        return self._curr

@noLogEntryExpected
def test_ArgumentSelector(maxValidator):
    inc = IncrementArgumentSelector({'k': 5}, maxValidator, iters=5)
    maxValidator._setNextPerformance(1)
    assert next(inc) == {'k': 5}
    assert next(inc) == {'k': 6}
    maxValidator._setNextPerformance(-1)
    assert next(inc) == {'k': 7}
    maxValidator._setNextPerformance(0)
    assert next(inc) == {'k': 6}
    assert next(inc) == {'k': 6}

    with raises(StopIteration):
        next(inc)

##############
# BruteForce #
##############
@noLogEntryExpected
def test_BruteForce(maxValidator):
    # single argument, no tune
    bf = BruteForce({'k': 5}, maxValidator)
    assert next(bf) == {'k': 5}
    with raises(StopIteration):
        next(bf)

    # single argument, tune
    bf = BruteForce({'k': Tune([1, 3, 5])}, maxValidator)
    assert next(bf) == {'k': 1}
    assert next(bf) == {'k': 3}
    assert next(bf) == {'k': 5}
    with raises(StopIteration):
        next(bf)

    # multiple arguments, no tune
    bf = BruteForce({'k': 5, 'p': 2}, maxValidator)
    assert next(bf) == {'k': 5, 'p': 2}
    with raises(StopIteration):
        next(bf)

    # multiple arguments, tune
    bf = BruteForce({'k': Tune([3, 5, 7]), 'p': Tune([1, 2])},
                    maxValidator)
    assert next(bf) == {'k': 3, 'p': 1}
    assert next(bf) == {'k': 3, 'p': 2}
    assert next(bf) == {'k': 5, 'p': 1}
    assert next(bf) == {'k': 5, 'p': 2}
    assert next(bf) == {'k': 7, 'p': 1}
    assert next(bf) == {'k': 7, 'p': 2}
    with raises(StopIteration):
        next(bf)

###############
# Consecutive #
###############
@noLogEntryExpected
def test_Consecutive(maxValidator):
    # single argument, no tune
    con = Consecutive({'k': 5}, maxValidator)
    assert next(con) == {'k': 5}
    with raises(StopIteration):
        next(con)

    # single argument, tune
    con = Consecutive({'k': Tune([1, 3, 5])}, maxValidator)
    assert next(con) == {'k': 1}
    assert next(con) == {'k': 3}
    assert next(con) == {'k': 5}
    with raises(StopIteration):
        next(con)

    # multiple arguments, no tune
    con = Consecutive({'k': 5, 'p': 2}, maxValidator)
    assert next(con) == {'k': 5, 'p': 2}
    with raises(StopIteration):
        next(con)

    # multiple arguments, tune
    con = Consecutive({'k': Tune([3, 5, 7]), 'p': Tune([1, 2])},
                      maxValidator, order=['k', 'p'])
    maxValidator._setNextPerformance(1)
    assert next(con) == {'k': 3, 'p': 1}
    assert next(con) == {'k': 5, 'p': 1}
    maxValidator._setNextPerformance(2)
    assert next(con) == {'k': 7, 'p': 1}
    assert next(con) == {'k': 7, 'p': 2}
    with raises(StopIteration):
        next(con)

    con = Consecutive({'k': Tune([3, 5, 7]), 'p': Tune([1, 2])},
                      maxValidator, order=['p', 'k'])

    maxValidator._setNextPerformance(0)
    assert next(con) == {'k': 3, 'p': 1}
    maxValidator._setNextPerformance(1)
    assert next(con) == {'k': 3, 'p': 2}
    maxValidator._setNextPerformance(2)
    assert next(con) == {'k': 5, 'p': 2}
    maxValidator._setNextPerformance(3)
    assert next(con) == {'k': 7, 'p': 2}
    with raises(StopIteration):
        next(con)
    # since our performance function is always increasing, each new combo will
    # yield better results, however argument combos that have already been
    # tried will not be tried again. In the first loop the best will be
    # {'k': 7, 'p': 2}}, in the second {'k': 5, 'p': 2}
    # will be better because it is the last untried performance. The third
    # loop will not try anything as it will recognize that all combinations
    # have now been tried
    con = Consecutive({'k': Tune([3, 5, 7]), 'p': Tune([1, 2])},
                      maxValidator, loops=3, order=['k', 'p'])

    maxValidator._setNextPerformance(4)
    assert next(con) == {'k': 3, 'p': 1}
    maxValidator._setNextPerformance(5)
    assert next(con) == {'k': 5, 'p': 1}
    maxValidator._setNextPerformance(3)
    assert next(con) == {'k': 7, 'p': 1}
    maxValidator._setNextPerformance(6)
    assert next(con) == {'k': 5, 'p': 2}
    maxValidator._setNextPerformance(2)
    assert next(con) == {'k': 3, 'p': 2}
    maxValidator._setNextPerformance(1)
    assert next(con) == {'k': 7, 'p': 2}
    with raises(StopIteration):
        next(con)

@pytest.mark.skipif(storm_tuner=False, reason='Storm Tuner unavailable.')
@pytest.mark.skipif(hyperopt=False, reason='Hyperopt unavailable.')
@noLogEntryExpected
def test_Bayesian(minValidator, maxValidator):
    # requires min optimal performanceFunction
    with raises(InvalidArgumentValue):
        bay = Bayesian({'k': 5}, maxValidator, maxIterations=5)

    # single argument, no tune, no incremental training
    bay = Bayesian({'k': 5}, minValidator, maxIterations=5)
    assert next(bay) == {'k': 5}
    with raises(StopIteration):
        next(bay)

    # single argument, no tune, has incremental training
    minValidator.reset(hasIncremental=True)
    bay = Bayesian({'k': 5}, minValidator, maxIterations=5)
    for _ in range(5):
        assert next(bay) == {'k': 5}
    with raises(StopIteration):
        next(bay)

    # single argument, tune
    # make k=1 the optimal value
    minValidator._performanceFunction = lambda args: (args['k'] - 1) * 100
    bay = Bayesian({'k': Tune(range(1, 101))}, minValidator)
    guesses = []
    for i in range(bay.maxIterations):
        arg = next(bay)['k']
        guesses.append(arg)
    # 1 should be guessed most often (or at worst second most)
    counts = {}
    for item in guesses:
        count = counts.get(item, 0)
        counts[item] = count + 1
    oneBest = counts[1] == max(counts.values())
    # small chance for k=2 to be the winner
    del counts[2]
    oneSecondBest = counts[1] == max(counts.values())
    assert oneBest or oneSecondBest
    with raises(StopIteration):
        next(bay)

    # multiple arguments, no tune
    minValidator.reset()
    bay = Bayesian({'k': 5, 'p': 2}, minValidator, maxIterations=3)
    assert next(bay) == {'k': 5, 'p': 2}
    with raises(StopIteration):
        next(bay)

    minValidator.reset(hasIncremental=True)
    bay = Bayesian({'k': 5, 'p': 2}, minValidator, maxIterations=3)
    for _ in range(3):
        assert next(bay) == {'k': 5, 'p': 2}
    with raises(StopIteration):
        next(bay)

    # multiple arguments, tune, with timeout
    minValidator.reset()
    minValidator._performanceFunction = wait(1)
    bay = Bayesian({'k': Tune(start=3, end=101),
                    'p': Tune(range(1, 10))}, minValidator,
                   maxIterations=3, timeout=2)
    start = default_timer()
    for i in range(4):
        try:
            args = next(bay)
        except StopIteration:
            break
    duration = default_timer() - start
    assert 2 < duration < 2.1
    # with 1 second sleep, StopIteration should occur on 3rd iteration
    assert i == 2


    # multiple arguments, tune, with threshold
    # k = 3, p=1 is optimal
    minValidator.reset()
    minValidator._performanceFunction = lambda args: ((args['k'] * args['p']) - 3) * 10
    bay = Bayesian({'k': Tune(range(3, 11)), 'p': Tune([1, 2])}, minValidator,
                   threshold=2)
    for i in range(100):
        try:
            args = next(bay)
        except StopIteration:
            break

    assert args['k'] == 3
    assert args['p'] == 1

    assert i < 100

def test_Iterative(maxValidator):
    # single argument, no tune
    itr = Iterative({'k': 5}, maxValidator, maxIterations=5)
    assert next(itr) == {'k': 5}
    with raises(StopIteration):
        next(itr)

    # single argument, tune
    # 3 optimum, no incrementalTrain
    maxValidator._performanceFunction = lambda args: 10 if args['k'] == 3 else 1
    itr = Iterative({'k': Tune([1, 3, 5])}, maxValidator)
    assert next(itr) == {'k': 3}
    with raises(StopIteration):
        next(itr)

    # 1 optimum, no incrementalTrain
    maxValidator._performanceFunction = lambda args: 10 if args['k'] == 1 else 1
    itr = Iterative({'k': Tune([1, 3, 5])}, maxValidator)
    assert next(itr) == {'k': 3}
    assert next(itr) == {'k': 1}
    with raises(StopIteration):
        next(itr)

    # 5 optimum, no incrementalTrain
    maxValidator._performanceFunction = lambda args: 10 if args['k'] == 5 else 1
    itr = Iterative({'k': Tune([1, 3, 5])}, maxValidator)
    assert next(itr) == {'k': 3}
    assert next(itr) == {'k' : 5}
    with raises(StopIteration):
        next(itr)

    # 5 optimum, with incrementalTrain (always completes all iterations)
    maxValidator.reset(hasIncremental=True)
    maxValidator._performanceFunction = lambda args: 10 if args['k'] == 5 else 1
    itr = Iterative({'k': Tune([1, 3, 5])}, maxValidator, maxIterations=10)
    assert next(itr) == {'k': 3}
    for _ in range(7):
        assert next(itr) == {'k' : 5}
    maxValidator._performanceFunction = lambda args: 15 if args['k'] == 3 else 1
    assert next(itr) == {'k' : 3}
    maxValidator._performanceFunction = lambda args: 20 if args['k'] == 1 else 1
    assert next(itr) == {'k': 1}
    with raises(StopIteration):
        next(itr)

    # multiple arguments, no tune
    maxValidator.reset()
    itr = Iterative({'k': 5, 'p': 2}, maxValidator)
    assert next(itr) == {'k': 5, 'p': 2}
    with raises(StopIteration):
        next(itr)

    # multiple arguments, no tune, with incrementalTrain
    maxValidator.reset(hasIncremental=True)
    itr = Iterative({'k': 5, 'p': 2}, maxValidator, maxIterations=5)
    for _ in range(5):
        assert next(itr) == {'k': 5, 'p': 2}
    with raises(StopIteration):
        next(itr)

    # multiple arguments, tune, no incrementalTrain
    maxValidator.reset()
    maxValidator._performanceFunction = lambda args: args['k'] / args['p']
    itr = Iterative({'k': Tune(range(3, 8)), 'p': Tune(range(1, 6))}, maxValidator)
    assert next(itr) == {'k': 5, 'p': 3}
    assert next(itr) == {'k': 6, 'p': 2}
    assert next(itr) == {'k': 7, 'p': 1}
    with raises(StopIteration):
        next(itr)

    # multiple arguments, tune, with incrementalTrain
    maxValidator.reset(hasIncremental=True)
    maxValidator._performanceFunction = lambda args: args['k'] * args['p']
    itr = Iterative({'k': Tune(range(3, 8)), 'p': Tune(range(1, 6))},
                    maxValidator, maxIterations=8)
    assert next(itr) == {'k': 5, 'p': 3}
    assert next(itr) == {'k': 6, 'p': 4}
    assert next(itr) == {'k': 7, 'p': 5}
    assert next(itr) == {'k': 7, 'p': 5}
    # now make lower arguments generate higher performance
    maxValidator._performanceFunction = lambda args: 100 * (8 - args['k']) * (6 - args['p'])
    assert next(itr) == {'k': 6, 'p': 4}
    assert next(itr) == {'k': 5, 'p': 3}
    assert next(itr) == {'k': 4, 'p': 2}
    assert next(itr) == {'k': 3, 'p': 1}
    with raises(StopIteration):
        next(itr)

    # timeout
    maxValidator.reset(hasIncremental=True)
    maxValidator._performanceFunction = wait(.4)
    itr = Iterative({'k': Tune(range(3, 8)), 'p': Tune(range(1, 6))},
                    maxValidator, timeout=2)
    start = default_timer()
    for i in range(4):
        try:
            args = next(itr)
        except StopIteration:
            break
    duration = default_timer() - start
    # first next call gets only the performance of the middle values (.4 sec)
    # the second call tests the higher and lower values for each variable
    # (1.6 sec) and none will improve performance so expect just over 2 seconds
    assert 2 < duration < 2.1
    assert i == 2

    # threshold, k=7, p=2 will trigger threshold
    maxValidator.reset()
    maxValidator._performanceFunction = lambda args: args['k'] * args['p']
    itr = Iterative({'k': Tune(range(1, 8)), 'p': Tune([0, 1, 2])}, maxValidator,
                   threshold=13)
    assert next(itr) == {'k': 4, 'p': 1}
    assert next(itr) == {'k': 5, 'p': 2}
    assert next(itr) == {'k': 6, 'p': 2}
    assert next(itr) == {'k': 7, 'p': 2}
    with raises(StopIteration):
        next(itr)


@pytest.mark.skipif(storm_tuner=False, reason='Storm Tuner unavailable.')
def test_StochasticRandomMutator(minValidator):
    # single argument, no tune
    srm = StochasticRandomMutator({'k': 5}, minValidator, maxIterations=5)
    assert next(srm) == {'k': 5}
    with raises(StopIteration):
        next(srm)

    # single argument, tune
    # make k=1 the optimal value
    minValidator._performanceFunction = lambda args: (args['k'] - 1) * 100
    # NOTE: given this an an oversimplified example, using ordered parameters
    # is prone to triggering an infinite loop so use an unordered set
    srm = StochasticRandomMutator({'k': Tune([1, 3, 4, 9, 21])}, minValidator,
                                  initRandom=4, randomizeAxisFactor=0)
    # first guess 4 guesses are random and do not repeat
    best = next(srm)['k']
    for _ in range(3):
        arg = next(srm)['k']
        if arg < best:
            best = arg
    arg = next(srm)['k'] # last guess is the one that wasn't guessed randomly
    with raises(StopIteration):
        next(srm)

    # multiple arguments, no tune
    minValidator.reset()
    srm = StochasticRandomMutator({'k': 5, 'p': 2}, minValidator,
                                  maxIterations=3)
    assert next(srm) == {'k': 5, 'p': 2}
    with raises(StopIteration):
        next(srm)

    # multiple arguments, tune, with timeout
    minValidator.reset()
    minValidator._performanceFunction = wait(1)
    srm = StochasticRandomMutator({'k': Tune(start=3, end=101),
                                   'p': Tune(range(1, 10))}, minValidator,
                                   timeout=2)
    start = default_timer()
    for i in range(4):
        try:
            args = next(srm)
        except StopIteration:
            break
    duration = default_timer() - start
    assert 2 < duration < 2.1
    # with 1 second sleep, StopIteration should occur on 3rd iteration
    assert i == 2

    # multiple arguments, tune, with threshold
    # k = 3, p=1 is optimal
    minValidator.reset()
    minValidator._performanceFunction = lambda args: ((args['k'] * args['p']) - 3) * 10
    srm = StochasticRandomMutator({'k': Tune(range(3, 11)), 'p': Tune([1, 2])},
                                  minValidator, threshold=2)
    for i in range(100):
        try:
            args = next(srm)
        except StopIteration:
            break

    assert args['k'] == 3
    assert args['p'] == 1

    assert i < 100
