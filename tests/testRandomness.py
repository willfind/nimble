"""
Tests for the nimble.random submodule
"""

import random

import numpy
import nose

import nimble
from nimble.random import _startAlternateControl, _endAlternateControl
from .assertionHelpers import logCountAssertionFactory


@nose.with_setup(_startAlternateControl, _endAlternateControl)
def testSetRandomSeedExplicit():
    """ Test nimble.random.setSeed yields Nimble accessible random objects with the correct random behavior """
    expPy = random.Random(1333)
    expNp = numpy.random.RandomState(1333)
    nimble.random.setSeed(1333)

    for i in range(50):
        assert nimble.random.pythonRandom.random() == expPy.random()
        assert nimble.random.numpyRandom.rand() == expNp.rand()


@nose.with_setup(_startAlternateControl, _endAlternateControl)
def testSetRandomSeedNone():
    """ Test nimble.random.setSeed operates as expected when passed None (-- use system time as seed) """
    nimble.random.setSeed(None)
    pyState = nimble.random.pythonRandom.getstate()
    npState = nimble.random.numpyRandom.get_state()

    origPy = random.Random()
    origPy.setstate(pyState)
    origNp = numpy.random.RandomState()
    origNp.set_state(npState)

    nimble.random.setSeed(None)

    assert origPy.random() != nimble.random.pythonRandom.random()
    assert origNp.rand() != nimble.random.numpyRandom.rand()


@nose.with_setup(_startAlternateControl, _endAlternateControl)
@logCountAssertionFactory(3)
def testSetRandomSeedPropagate():
    """ Test that nimble.random.setSeed will correctly control how randomized methods in nimble perform """
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [17, 18, 19], [2, 2, 2], [3, 3, 3], [4, 4, 4],
            [5, 5, 5]]
    featureNames = ['1', '2', '3']
    toTest1 = nimble.createData("List", data, featureNames=featureNames, useLog=False)
    toTest2 = toTest1.copy()
    toTest3 = toTest1.copy()

    nimble.random.setSeed(1337)
    toTest1.points.permute(useLog=False)

    nimble.random.setSeed(1336)
    toTest2.points.permute(useLog=False)

    nimble.random.setSeed(1337)
    toTest3.points.permute(useLog=False)

    assert toTest1 == toTest3
    assert toTest1 != toTest2
