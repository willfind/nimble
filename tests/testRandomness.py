"""
Tests for the nimble.randomness submodule

"""

from __future__ import absolute_import
import random

from six.moves import range
import numpy
import nose

import nimble
import nimble.randomness
from nimble.randomness import pythonRandom
from nimble.randomness import numpyRandom
from .assertionHelpers import logCountAssertionFactory


@nose.with_setup(nimble.randomness.startAlternateControl, nimble.randomness.endAlternateControl)
def testSetRandomSeedExplicit():
    """ Test nimble.setRandomSeed yields Nimble accessible random objects with the correct random behavior """
    expPy = random.Random(1333)
    expNp = numpy.random.RandomState(1333)
    nimble.setRandomSeed(1333)

    for i in range(50):
        assert pythonRandom.random() == expPy.random()
        assert numpyRandom.rand() == expNp.rand()


@nose.with_setup(nimble.randomness.startAlternateControl, nimble.randomness.endAlternateControl)
def testSetRandomSeedNone():
    """ Test nimble.setRandomSeed operates as expected when passed None (-- use system time as seed) """
    nimble.setRandomSeed(None)
    pyState = pythonRandom.getstate()
    npState = numpyRandom.get_state()

    origPy = random.Random()
    origPy.setstate(pyState)
    origNp = numpy.random.RandomState()
    origNp.set_state(npState)

    nimble.setRandomSeed(None)

    assert origPy.random() != pythonRandom.random()
    assert origNp.rand() != numpyRandom.rand()


@nose.with_setup(nimble.randomness.startAlternateControl, nimble.randomness.endAlternateControl)
@logCountAssertionFactory(3)
def testSetRandomSeedPropagate():
    """ Test that nimble.setRandomSeed will correctly control how randomized methods in nimble perform """
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [17, 18, 19], [2, 2, 2], [3, 3, 3], [4, 4, 4],
            [5, 5, 5]]
    featureNames = ['1', '2', '3']
    toTest1 = nimble.createData("List", data, featureNames=featureNames, useLog=False)
    toTest2 = toTest1.copy()
    toTest3 = toTest1.copy()

    nimble.setRandomSeed(1337)
    toTest1.points.shuffle(useLog=False)

    nimble.setRandomSeed(1336)
    toTest2.points.shuffle(useLog=False)

    nimble.setRandomSeed(1337)
    toTest3.points.shuffle(useLog=False)

    assert toTest1 == toTest3
    assert toTest1 != toTest2
