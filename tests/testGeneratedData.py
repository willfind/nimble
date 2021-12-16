"""
Unit tests for data creation functions other than nimble.data that
generate the values placed in the data object. Specifically tested are:
nimble.random.data, nimble.ones, nimble.zeros, nimble.identity
"""

import copy

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from tests.helpers import noLogEntryExpected
from tests.helpers import raises

returnTypes = copy.copy(nimble.core.data.available)

#######################
### shared backends ###
#######################

def back_constant_sizeChecking(toTest):
    with raises(InvalidArgumentValue):
        toTest(-1, 5)

    with raises(InvalidArgumentValue):
        toTest(4, -3)

    with raises(InvalidArgumentValueCombination):
        toTest(0, 0)


def back_constant_emptyCreation(toTest):
    fEmpty = np.array([[], []])
    pEmpty = fEmpty.T

    for t in returnTypes:
        retPEmpty = toTest(0, 2, returnType=t)
        retFEmpty = toTest(2, 0, returnType=t)

        expFEmpty = nimble.data(fEmpty, returnType=t)
        expPEmpty = nimble.data(pEmpty, returnType=t)

        assert retPEmpty == expPEmpty
        assert retFEmpty == expFEmpty


def back_constant_correctSizeAndContents(toTest, value):
    checkSizes = [(1, 1), (1, 5), (4, 1), (10, 10), (20, 5)]

    for t in returnTypes:
        for size in checkSizes:
            ret = toTest(size[0], size[1], returnType=t)
            assert t == ret.getTypeString()

            assert len(ret.points) == size[0]
            assert len(ret.features) == size[1]

            for p in range(size[0]):
                for f in range(size[1]):
                    assert ret[p, f] == value


def back_constant_correctNames(toTest):
    objName = "checkObjName"
    pnames = ["p1", "p2"]
    fnames = ["f1", "f2"]

    for t in returnTypes:
        ret = toTest(2, 2, pointNames=pnames, featureNames=fnames,
                     returnType=t, name=objName)

        assert ret.points.getNames() == pnames
        assert ret.features.getNames() == fnames
        assert ret.name == objName


def back_constant_conversionEqualityBetweenTypes(toTest):
    p, f = (10, 2)

    for makeT in returnTypes:
        ret = toTest(p, f, returnType=makeT)

        for matchT in returnTypes:
            convertedRet = ret.copy(to=matchT)
            toMatch = toTest(p, f, returnType=matchT)

            assert convertedRet == toMatch


def back_constant_logCount(toTest):

    @noLogEntryExpected
    def byType(rType):
        out = toTest(5, 5, returnType=rType)

    for t in returnTypes:
        byType(t)

def back_constant_hasInts(toTest):
    for t in returnTypes:
        out = toTest(5, 5, returnType=t)
        assert out.countElements(nimble.match.integer) == 25

############
### ones ###
############

# This function relies on nimble.data to actually instantiate our data, and
# never touches the pointNames, featureNames, or names arguments. The
# validity checking of those arguments is therefore not tested, since
# it is done exclusively in nimble.data. We only check for successful behaviour.

def test_ones_sizeChecking():
    back_constant_sizeChecking(nimble.ones)


def test_ones_emptyCreation():
    back_constant_emptyCreation(nimble.ones)


def test_ones_correctSizeAndContents():
    back_constant_correctSizeAndContents(nimble.ones, 1)


def test_ones_correctNames():
    back_constant_correctNames(nimble.ones)


def test_ones_conversionEqualityBetweenTypes():
    back_constant_conversionEqualityBetweenTypes(nimble.ones)


def test_ones_logCount():
    back_constant_logCount(nimble.ones)

def test_ones_hasInts():
    back_constant_hasInts(nimble.ones)

#############
### zeros ###
#############

# This function relies on nimble.data to actually instantiate our data, and
# never touches the pointNames, featureNames, or names arguments. The
# validity checking of those arguments is therefore not tested, since
# it is done exclusively in nimble.data. We only check for successful behaviour.

def test_zeros_sizeChecking():
    back_constant_sizeChecking(nimble.zeros)


def test_zeros_emptyCreation():
    back_constant_emptyCreation(nimble.zeros)


def test_zeros_correctSizeAndContents():
    back_constant_correctSizeAndContents(nimble.zeros, 0)


def test_zeros_correctNames():
    back_constant_correctNames(nimble.zeros)


def test_zeros_conversionEqualityBetweenTypes():
    back_constant_conversionEqualityBetweenTypes(nimble.zeros)

def test_zeros_logCount():
    back_constant_logCount(nimble.zeros)

def test_zeros_hasInts():
    back_constant_hasInts(nimble.zeros)


################
### identity ###
################

# This function relies on nimble.data to actually instantiate our data, and
# never touches the pointNames, featureNames, or names arguments. The
# validity checking of those arguments is therefore not tested, since
# it is done exclusively in nimble.data. We only check for successful behaviour.


def test_identity_sizeChecking():
    with raises(InvalidArgumentValue):
        nimble.identity(-1)

    with raises(InvalidArgumentValue):
        nimble.identity(0)


def test_identity_correctSizeAndContents():
    for t in returnTypes:
        for size in range(1, 5):
            toTest = nimble.identity(size, returnType=t)
            assert t == toTest.getTypeString()
            for p in range(size):
                for f in range(size):
                    if p == f:
                        assert toTest[p, f] == 1
                    else:
                        assert toTest[p, f] == 0


def test_identity_correctNames():
    objName = "checkObjName"
    pnames = ["p1", "p2"]
    fnames = ["f1", "f2"]

    for t in returnTypes:
        ret = nimble.identity(2, pointNames=pnames, featureNames=fnames,
                              returnType=t, name=objName)

        assert ret.points.getNames() == pnames
        assert ret.features.getNames() == fnames
        assert ret.name == objName


def test_identity_conversionEqualityBetweenTypes():
    size = 7

    for makeT in returnTypes:
        ret = nimble.identity(size, returnType=makeT)

        for matchT in returnTypes:
            convertedRet = ret.copy(to=matchT)
            toMatch = nimble.identity(size, returnType=matchT)

            assert convertedRet == toMatch

def test_identity_logCount():

    @noLogEntryExpected
    def byType(rType):
        toTest = nimble.identity(5, returnType=rType)

    for t in returnTypes:
        byType(t)

# EOF Marker
