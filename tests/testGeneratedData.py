"""
Unit tests for data creation functions other than nimble.data that
generate the values placed in the data object. Specifically tested are:
nimble.random.data, nimble.ones, nimble.zeros, nimble.identity
"""

import copy

import numpy as np
from nose.tools import *

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from tests.helpers import noLogEntryExpected

returnTypes = copy.copy(nimble.core.data.available)

#######################
### shared backends ###
#######################

def back_constant_sizeChecking(toTest):
    try:
        toTest("Matrix", -1, 5)
        assert False  # expected InvalidArgumentValue for negative numPoints
    except InvalidArgumentValue:
        pass

    try:
        toTest("Matrix", 4, -3)
        assert False  # expected InvalidArgumentValue for negative numFeatures
    except InvalidArgumentValue:
        pass

    try:
        toTest("Matrix", 0, 0)
        assert False  # expected InvalidArgumentValueCombination for 0 by 0 sized object
    except InvalidArgumentValueCombination:
        pass


def back_constant_emptyCreation(toTest):
    fEmpty = np.array([[], []])
    pEmpty = fEmpty.T

    for t in returnTypes:
        retPEmpty = toTest(t, 0, 2)
        retFEmpty = toTest(t, 2, 0)

        expFEmpty = nimble.data(t, fEmpty)
        expPEmpty = nimble.data(t, pEmpty)

        assert retPEmpty == expPEmpty
        assert retFEmpty == expFEmpty


def back_constant_correctSizeAndContents(toTest, value):
    checkSizes = [(1, 1), (1, 5), (4, 1), (10, 10), (20, 5)]

    for t in returnTypes:
        for size in checkSizes:
            ret = toTest(t, size[0], size[1])
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
        ret = toTest(t, 2, 2, pointNames=pnames, featureNames=fnames, name=objName)

        assert ret.points.getNames() == pnames
        assert ret.features.getNames() == fnames
        assert ret.name == objName


def back_constant_conversionEqualityBetweenTypes(toTest):
    p, f = (10, 2)

    for makeT in returnTypes:
        ret = toTest(makeT, p, f)

        for matchT in returnTypes:
            convertedRet = ret.copy(to=matchT)
            toMatch = toTest(matchT, p, f)

            assert convertedRet == toMatch


def back_constant_logCount(toTest):

    @noLogEntryExpected
    def byType(rType):
        out = toTest(rType, 5, 5)

    for t in returnTypes:
        byType(t)


############
### ones ###
############

#nimble.ones(returnType, numPoints, numFeatures, pointNames=None, featureNames=None, name=None)

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

#############
### zeros ###
#############

#nimble.zeros(returnType, numPoints, numFeatures, pointNames=None, featureNames=None, name=None)

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


################
### identity ###
################

#nimble.identity(returnType, size, pointNames=None, featureNames=None, name=None)

# This function relies on nimble.data to actually instantiate our data, and
# never touches the pointNames, featureNames, or names arguments. The
# validity checking of those arguments is therefore not tested, since
# it is done exclusively in nimble.data. We only check for successful behaviour.


def test_identity_sizeChecking():
    try:
        nimble.identity("Matrix", -1)
        assert False  # expected InvalidArgumentValue for negative size
    except InvalidArgumentValue:
        pass

    try:
        nimble.identity("Matrix", 0)
        assert False  # expected InvalidArgumentValue for 0 valued size
    except InvalidArgumentValue:
        pass


def test_identity_correctSizeAndContents():
    for t in returnTypes:
        for size in range(1, 5):
            toTest = nimble.identity(t, size)
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
        ret = nimble.identity(t, 2, pointNames=pnames, featureNames=fnames, name=objName)

        assert ret.points.getNames() == pnames
        assert ret.features.getNames() == fnames
        assert ret.name == objName


def test_identity_conversionEqualityBetweenTypes():
    size = 7

    for makeT in returnTypes:
        ret = nimble.identity(makeT, size)

        for matchT in returnTypes:
            convertedRet = ret.copy(to=matchT)
            toMatch = nimble.identity(matchT, size)

            assert convertedRet == toMatch

def test_identity_logCount():

    @noLogEntryExpected
    def byType(rType):
        toTest = nimble.identity(rType, 5)

    for t in returnTypes:
        byType(t)

# EOF Marker
