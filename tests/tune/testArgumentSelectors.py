
from pytest import fixture

import nimble
from nimble import Tune
from nimble.random import pythonRandom
from nimble.core.tune import ArgumentSelector
from nimble.core.tune import BruteForce, Consecutive, Bayesian, Iterative
from nimble.core.tune import HoldoutProportion
from nimble.calculate import fractionIncorrect
from tests.helpers import noLogEntryExpected, raises

class IncrementArgumentSelector(ArgumentSelector):
    name = "increment"

    def __init__(self, arguments=None, iters=10):
        super().__init__(arguments, iters=iters)
        self.iters = iters
        self._i = 0
        self._curr = self.arguments.copy()

    def __next__(self):
        if self._i >= self.iters:
            raise StopIteration
        self._i += 1
        return self._curr

    def update(self, performance):
        for arg in self._curr:
            if performance > 0:
                self._curr[arg] += 1
            else:
                self._curr[arg] -= 1

@noLogEntryExpected
def test_ArgumentSelector():
    # optimal not factored into update() so None is fine
    inc = IncrementArgumentSelector({'k': 5}, iters=5)
    assert next(inc) == {'k': 5}
    inc.update(1)
    assert next(inc) == {'k': 6}
    inc.update(1)
    assert next(inc) == {'k': 7}
    inc.update(0)
    assert next(inc) == {'k': 6}
    inc.update(0)
    assert next(inc) == {'k': 5}

    with raises(StopIteration):
        next(inc)

    string = str(inc)
    rep = repr(inc)
    assert string == rep
    assert rep == ("IncrementArgumentSelector({'k': 5}, iters=5)")

##############
# BruteForce #
##############
@noLogEntryExpected
def test_BruteForce():
    # single argument, no tune
    bf = BruteForce({'k': 5})
    assert next(bf) == {'k': 5}
    with raises(StopIteration):
        next(bf)

    # single argument, tune
    bf = BruteForce({'k': Tune([1, 3, 5])})
    assert next(bf) == {'k': 1}
    assert next(bf) == {'k': 3}
    assert next(bf) == {'k': 5}
    with raises(StopIteration):
        next(bf)

    # multiple arguments, no tune
    bf = BruteForce({'n_neighbors': 5, 'p': 2})
    assert next(bf) == {'n_neighbors': 5, 'p': 2}
    with raises(StopIteration):
        next(bf)

    # multiple arguments, tune
    bf = BruteForce({'n_neighbors': Tune([3, 5, 7]), 'p': Tune([1, 2])})
    assert next(bf) == {'n_neighbors': 3, 'p': 1}
    assert next(bf) == {'n_neighbors': 3, 'p': 2}
    assert next(bf) == {'n_neighbors': 5, 'p': 1}
    assert next(bf) == {'n_neighbors': 5, 'p': 2}
    assert next(bf) == {'n_neighbors': 7, 'p': 1}
    assert next(bf) == {'n_neighbors': 7, 'p': 2}
    with raises(StopIteration):
        next(bf)

###############
# Consecutive #
###############
@noLogEntryExpected
def test_Consecutive():
    # single argument, no tune
    con = Consecutive({'k': 5}, 'max')
    assert next(con) == {'k': 5}
    with raises(StopIteration):
        next(con)

    # single argument, tune
    con = Consecutive({'k': Tune([1, 3, 5])}, 'min')
    assert next(con) == {'k': 1}
    con.update(0)
    assert next(con) == {'k': 3}
    con.update(0)
    assert next(con) == {'k': 5}
    con.update(0)
    with raises(StopIteration):
        next(con)

    # multiple arguments, no tune
    con = Consecutive({'n_neighbors': 5, 'p': 2}, 'max')
    assert next(con) == {'n_neighbors': 5, 'p': 2}
    with raises(StopIteration):
        next(con)

    # multiple arguments, tune
    con = Consecutive({'n_neighbors': Tune([3, 5, 7]), 'p': Tune([1, 2])},
                      'min', order=['n_neighbors', 'p'])

    assert next(con) == {'n_neighbors': 3, 'p': 1}
    con.update(3)
    assert next(con) == {'n_neighbors': 5, 'p': 1}
    con.update(2)
    assert next(con) == {'n_neighbors': 7, 'p': 1}
    con.update(1)
    assert next(con) == {'n_neighbors': 7, 'p': 2}
    con.update(0)
    with raises(StopIteration):
        next(con)

    con = Consecutive({'n_neighbors': Tune([3, 5, 7]), 'p': Tune([1, 2])},
                      'max',  order=['p', 'n_neighbors'])

    assert next(con) == {'n_neighbors': 3, 'p': 1}
    con.update(0)
    assert next(con) == {'n_neighbors': 3, 'p': 2}
    con.update(1)
    assert next(con) == {'n_neighbors': 5, 'p': 2}
    con.update(2)
    assert next(con) == {'n_neighbors': 7, 'p': 2}
    con.update(3)
    with raises(StopIteration):
        next(con)
    # since our performance function is always increasing, each new combo will
    # yield better results, however argument combos that have already been
    # tried will not be tried again. In the first loop the best will be
    # {'n_neighbors': 7, 'p': 2}}, in the second {'n_neighbors': 5, 'p': 2}
    # will be better because it is the last untried performance. The third
    # loop will not try anything as it will recognize that all combinations
    # have now been tried
    con = Consecutive({'n_neighbors': Tune([3, 5, 7]), 'p': Tune([1, 2])},
                      'max', loops=3, order=['n_neighbors', 'p'])

    assert next(con) == {'n_neighbors': 3, 'p': 1}
    con.update(4)
    assert next(con) == {'n_neighbors': 5, 'p': 1}
    con.update(5)
    assert next(con) == {'n_neighbors': 7, 'p': 1}
    con.update(3)
    assert next(con) == {'n_neighbors': 5, 'p': 2}
    con.update(6)
    assert next(con) == {'n_neighbors': 3, 'p': 2}
    con.update(2)
    assert next(con) == {'n_neighbors': 7, 'p': 2}
    con.update(1)

# TODO: def test_Bayesian():
# TODO: def test_Iterative():
