"""

Methods tested in this file:

In object NumericalDataSafe:
__mul__, __rmul__,  __add__, __radd__,  __sub__, __rsub__,
__truediv__, __rtruediv__,  __floordiv__, __rfloordiv__,
__mod__, __rmod__ ,  __pow__,  __pos__, __neg__, __abs__,
__matmul__, matrixMultiply, __rmatmul__, __imatmul__, matrixPower

In object NumericalModifying:
__imul__, __iadd__, __isub__,
__itruediv__, __ifloordiv__,  __imod__, __ipow__, __imatmul__, __and__,
__or__, __xor__

"""

import sys
import numpy
import os
import os.path
from unittest.mock import patch

from nose.tools import *

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction
from nimble.random import numpyRandom
from nimble.random import pythonRandom
from nimble.core.data._dataHelpers import DEFAULT_PREFIX

from .baseObject import DataTestObject
from tests.helpers import logCountAssertionFactory, noLogEntryExpected
from tests.helpers import assertNoNamesGenerated
from tests.helpers import CalledFunctionException, calledException


preserveName = "PreserveTestName"
preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
preserveRPath = os.path.relpath(preserveAPath)
preservePair = (preserveAPath, preserveRPath)


def calleeConstructor(data, constructor):
    if constructor is None:
        return pythonRandom.random()
    else:
        return constructor(data)


def back_unary_pfname_preservations(callerCon, op):
    """ Test that point / feature names are preserved when calling a unary op """
    data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    pnames = ['p1', 'p2', 'p3']
    fnames = ['f1', 'f2', 'f3']

    caller = callerCon(data, pnames, fnames)
    toCall = getattr(caller, op)
    ret = toCall()

    assert ret.points.getNames() == pnames
    assert ret.features.getNames() == fnames

    # changing the returned value, in case the caller is read-only.
    # We confirm the separation of the name recording either way.
    ret.points.setName('p1', 'p0')

    assert 'p1' in caller.points.getNames()
    assert 'p0' not in caller.points.getNames()
    assert 'p0' in ret.points.getNames()
    assert 'p1' not in ret.points.getNames()


def back_unary_NamePath_preservations(callerCon, op):
    """ Test that object names and pathes are preserved when calling a unary op """
    data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    caller = callerCon(data, name=preserveName, path=preservePair)

    toCall = getattr(caller, op)
    ret = toCall()

    assert ret.name != preserveName
    assert ret.nameIsDefault()
    assert caller.name == preserveName
    assert ret.absolutePath == preserveAPath
    assert caller.absolutePath == preserveAPath
    assert ret.path == preserveAPath
    assert caller.path == preserveAPath
    assert ret.relativePath == preserveRPath
    assert caller.relativePath == preserveRPath


def back_binaryscalar_pfname_preservations(callerCon, op, inplace):
    """ Test that p/f names are preserved when calling a binary scalar op """
    data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    pnames = ['p1', 'p2', 'p3']
    fnames = ['f1', 'f2', 'f3']

    for num in [-2, 0, 1, 4]:
        try:
            caller = callerCon(data, pnames, fnames)
            toCall = getattr(caller, op)
            ret = toCall(num)
        except ZeroDivisionError:
            continue

        assert ret.points.getNames() == pnames
        assert ret.features.getNames() == fnames

        try:
            caller.points.setName('p1', 'p0')
        except ImproperObjectAction: # Views
            continue

        if inplace:
            assert 'p0' in ret.points.getNames()
            assert 'p1' not in ret.points.getNames()
        else:
            assert 'p0' not in ret.points.getNames()
            assert 'p1' in ret.points.getNames()
        assert 'p0' in caller.points.getNames()
        assert 'p1' not in caller.points.getNames()

def back_binaryscalar_NamePath_preservations(callerCon, op):
    data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    caller = callerCon(data, name=preserveName, path=preservePair)

    toCall = getattr(caller, op)
    ret = toCall(1)

    if op.startswith('__i'):
        assert ret.name == preserveName
    else:
        assert ret.name != preserveName
        assert ret.nameIsDefault()
    assert ret.absolutePath == preserveAPath
    assert ret.path == preserveAPath
    assert ret.relativePath == preserveRPath

    assert caller.name == preserveName
    assert caller.absolutePath == preserveAPath
    assert caller.path == preserveAPath
    assert caller.relativePath == preserveRPath


def back_binaryelementwise_pfname_preservations(callerCon, op, inplace):
    """ Test that p/f names are preserved when calling a binary element wise op """
    data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    pnames = ['p1', 'p2', 'p3']
    fnames = ['f1', 'f2', 'f3']
    # use tuple so must be copied to list to modify
    defaultNames = tuple(callerCon(data).features.getNames())

    otherRaw = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    # both names same
    caller = callerCon(data, pnames, fnames)
    other = callerCon(otherRaw, pnames, fnames)
    toCall = getattr(caller, op)
    ret = toCall(other)

    assert ret.points.getNames() == pnames
    assert ret.features.getNames() == fnames

    if inplace:
        caller.points.setName('p1', 'p0')
        assert 'p0' in ret.points.getNames()
        assert 'p1' not in ret.points.getNames()
        assert 'p0' in caller.points.getNames()
        assert 'p1' not in caller.points.getNames()
    else:
        ret.points.setName('p1', 'p0')
        assert 'p0' not in caller.points.getNames()
        assert 'p1' in caller.points.getNames()
        assert 'p0' in ret.points.getNames()
        assert 'p1' not in ret.points.getNames()

    # both names different
    caller = callerCon(data, pnames, fnames)
    opnames = {'p0': 0, 'p1': 1, 'p2': 2}
    ofnames = {'f0': 0, 'f1': 1, 'f2': 2}
    other = callerCon(otherRaw, opnames, ofnames)
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

    # names interwoven
    other = callerCon(otherRaw, pnames, defaultNames)
    caller = callerCon(data, defaultNames, fnames)
    toCall = getattr(caller, op)
    ret = toCall(other)

    assert ret.points.getNames() == pnames
    assert ret.features.getNames() == fnames

    other = callerCon(otherRaw, defaultNames, fnames)
    caller = callerCon(data, pnames, defaultNames)
    toCall = getattr(caller, op)
    ret = toCall(other)
    assert ret.points.getNames() == pnames
    assert ret.features.getNames() == fnames

    # mixed defaults
    cpnames = list(defaultNames)
    cpnames[0] = 'p1'
    cfnames = list(defaultNames)
    cfnames[0] = 'f1'
    caller = callerCon(data, cpnames, cfnames)
    opnames = list(defaultNames)
    opnames[2] = 'p3'
    ofnames = list(defaultNames)
    ofnames[2] = 'f3'
    other = callerCon(otherRaw, opnames, ofnames)
    toCall = getattr(caller, op)
    ret = toCall(other)

    retPNames = ret.points.getNames()
    retFNames = ret.features.getNames()
    assert retPNames[0] == 'p1' and retPNames[2] == 'p3'
    assert retFNames[0] == 'f1' and retFNames[2] == 'f3'

    # pt names equal; ft names intersect
    caller = callerCon(data, pnames, fnames)
    opnames = pnames
    ofnames = {'f0': 0, 'f1': 1, 'f2': 2}
    other = callerCon(otherRaw, opnames, ofnames)
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert False
    except InvalidArgumentValue:
        pass

    # pt names equal; ft names disjoint
    caller = callerCon(data, pnames, fnames)
    opnames = pnames
    ofnames = {'1f': 0, '2f': 1, '3f': 2}
    other = callerCon(otherRaw, opnames, ofnames)
    # inplace requires feature names to match, otherwise not required
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert not ret.features._namesCreated()
    except InvalidArgumentValue:
        if inplace:
            pass
        else:
            raise

    # pt names equal; ft names + mixed ft names intersect
    caller = callerCon(data, pnames, fnames)
    ofnames = list(defaultNames)
    ofnames[0] = 'f3'
    other = callerCon(otherRaw, pnames, ofnames)
    assert other.features.getName(0) in caller.features.getNames()
    assert other.features.getName(1).startswith(DEFAULT_PREFIX)
    assert other.features.getName(2).startswith(DEFAULT_PREFIX)
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert False
    except InvalidArgumentValue:
        pass

    # pt names equal; ft names + mixed ft names disjoint
    caller = callerCon(data, pnames, fnames)
    ofnames = list(defaultNames)
    ofnames[0] = '3f'
    other = callerCon(otherRaw, pnames, ofnames)
    assert other.features.getName(0) not in caller.features.getNames()
    assert other.features.getName(1).startswith(DEFAULT_PREFIX)
    assert other.features.getName(2).startswith(DEFAULT_PREFIX)
    toCall = getattr(caller, op)
    # inplace requires feature names to match, otherwise not required
    try:
        ret = toCall(other)
        assert not ret.features._namesCreated()
    except InvalidArgumentValue:
        if inplace:
            pass
        else:
            raise

    # pt names equal; mixed ft names + mixed ft names intersect
    cfnames = list(defaultNames)
    cfnames[0] = 'f1'
    caller = callerCon(data, pnames, cfnames)
    ofnames = list(defaultNames)
    ofnames[2] = 'f1'
    other = callerCon(otherRaw, pnames, ofnames)
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert False
    except InvalidArgumentValue:
        pass

    # ft names equal; pt names intersect
    caller = callerCon(data, pnames, fnames)
    ofnames = fnames
    opnames = {'p0': 0, 'p1': 1, 'p2': 2}
    other = callerCon(otherRaw, opnames, ofnames)
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

    # ft names equal; pt names disjoint
    caller = callerCon(data, pnames, fnames)
    opnames = {'1p': 0, '2p': 1, '3p': 2}
    ofnames = fnames
    other = callerCon(otherRaw, opnames, ofnames)
    # inplace requires point names to match, otherwise not required
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert not ret.points._namesCreated()
    except InvalidArgumentValue:
        if inplace:
            pass
        else:
            raise

    # ft names equal; pt names + mixed pt names intersect
    caller = callerCon(data, pnames, fnames)
    opnames = list(defaultNames)
    opnames[0] = 'p3'
    other = callerCon(otherRaw, opnames, fnames)
    assert other.points.getName(0) in caller.points.getNames()
    assert other.points.getName(1).startswith(DEFAULT_PREFIX)
    assert other.points.getName(2).startswith(DEFAULT_PREFIX)
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

    # ft names equal; pt names + mixed pt names disjoint
    caller = callerCon(data, pnames, fnames)
    opnames = list(defaultNames)
    opnames[0] = '3p'
    other = callerCon(otherRaw, opnames, fnames)
    assert other.points.getName(0) not in caller.points.getNames()
    assert other.points.getName(1).startswith(DEFAULT_PREFIX)
    assert other.points.getName(2).startswith(DEFAULT_PREFIX)
    # inplace requires point names to match, otherwise not required
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert not ret.points._namesCreated()
    except InvalidArgumentValue:
        if inplace:
            pass
        else:
            raise

    # ft names equal; mixed pt names + mixed pt names intersect
    cpnames = list(defaultNames)
    cpnames[0] = 'p1'
    caller = callerCon(data, cpnames, fnames)
    opnames = list(defaultNames)
    opnames[2] = 'p1'
    other = callerCon(otherRaw, opnames, fnames)
    toCall = getattr(caller, op)
    try:
        ret = toCall(other)
        assert False
    except InvalidArgumentValue:
        pass

def back_binaryelementwise_NamePath_preservations(callerCon, attr1, inplace, attr2=None):
    data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    preserveNameOther = preserveName + "Other"
    preserveAPathOther = preserveAPath + "Other"
    preserveRPathOther = preserveRPath + "Other"
    preservePairOther = (preserveAPathOther, preserveRPathOther)

    ### emptry caller, full other ###
    caller = callerCon(data)
    other = callerCon(data, name=preserveNameOther, path=preservePairOther)

    assert caller.nameIsDefault()
    assert caller.absolutePath is None
    assert caller.relativePath is None

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    ret = toCall(other)

    assert not (ret is None and attr2 is None)

    # name should be default, path should be pulled from other
    if ret is not None and ret != NotImplemented:
        assert ret.name != preserveNameOther
        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPathOther
        assert ret.path == preserveAPathOther
        assert ret.relativePath == preserveRPathOther

    # if in place, ret == caller. if not, then values should be unchanged
    if not inplace:
        assert caller.nameIsDefault()
        assert caller.absolutePath is None
        assert caller.path is None
        assert caller.relativePath is None

    # confirm that other is unchanged
    assert other.name == preserveNameOther
    assert other.absolutePath == preserveAPathOther
    assert other.path == preserveAPathOther
    assert other.relativePath == preserveRPathOther

    ### full caller, empty other ###
    caller = callerCon(data, name=preserveName, path=preservePair)
    other = callerCon(data)

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    ret = toCall(other)

    assert not (ret is None and attr2 is None)

    # name should be default, path should be pulled from caller
    if ret is not None and ret != NotImplemented:
        # exception: if we are in place, then we do keep the name
        if inplace:
            assert ret.name == preserveName
        else:
            assert ret.name != preserveName
            assert ret.nameIsDefault()

        assert ret.absolutePath == preserveAPath
        assert ret.path == preserveAPath
        assert ret.relativePath == preserveRPath

    # if in place, ret == caller. if not, then values should be unchanged
    if not inplace:
        assert caller.name == preserveName
        assert caller.absolutePath == preserveAPath
        assert caller.path == preserveAPath
        assert caller.relativePath == preserveRPath

    # confirm that othe remains unchanged
    assert other.nameIsDefault()
    assert other.absolutePath is None
    assert other.path is None
    assert other.relativePath is None

    ### full caller, full other ###
    caller = callerCon(data, name=preserveName, path=preservePair)
    other = callerCon(data, name=preserveNameOther, path=preservePairOther)

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    ret = toCall(other)

    assert not (ret is None and attr2 is None)

    # name should be default, path should be pulled from caller
    if ret is not None and ret != NotImplemented:
        # exception: if we are in place, we keep the name from the caller
        # but the paths still obliterate each other
        if inplace:
            assert ret.name == preserveName
        #			assert ret.absolutePath == "TestAbsPathOther"
        #			assert ret.path == "TestAbsPathOther"
        #			assert ret.relativePath == "TestRelPathOther"
        else:
            assert ret.nameIsDefault()
        assert ret.absolutePath is None
        assert ret.path is None
        assert ret.relativePath is None

    # if in place, ret == caller. if not, then values should be unchanged
    if not inplace:
        assert caller.name == preserveName
        assert caller.absolutePath == preserveAPath
        assert caller.path == preserveAPath
        assert caller.relativePath == preserveRPath

    # confirm that other remains unchanged
    assert other.name == preserveNameOther
    assert other.absolutePath == preserveAPathOther
    assert other.path == preserveAPathOther
    assert other.relativePath == preserveRPathOther


def back_matrixmul_pfname_preservations(callerCon, attr1, inplace):
    """ Test that p/f names are preserved when calling a binary element wise op """
    data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    pnames = ['p1', 'p2', 'p3']
    fnames = ['f1', 'f2', 'f3']

    # [p x f1] times [f2 x p] where f1 != f2
    caller = callerCon(data, pnames, fnames)
    ofnames = {'f0': 0, 'f1': 1, 'f2': 2}
    other = callerCon(data, ofnames, pnames)
    # only rhs should pass
    try:
        toCall = getattr(caller, attr1)
        ret = toCall(other)
        assert ret.points.getNames() == other.points.getNames()
        assert ret.features.getNames() == caller.features.getNames()
    # lhs conflict between names; reverse operation eliminates conflict
    except InvalidArgumentValue:
        toCall = getattr(other, attr1)
        ret = toCall(caller)

    assert ret.points.getNames() == other.points.getNames()
    assert ret.features.getNames() == caller.features.getNames()

    # names interwoven
    interPnames = ['f1', 'f2', None]
    caller = callerCon(data, pnames, fnames)
    other = callerCon(data, interPnames, fnames)
    # only lhs should pass
    try:
        toCall = getattr(caller, attr1)
        ret = toCall(other)
    # rhs conflict between names; reverse operation eliminates conflict
    except InvalidArgumentValue:
        toCall = getattr(other, attr1)
        ret = toCall(caller)

    assert ret.points.getNames() == caller.points.getNames()
    assert ret.features.getNames() == other.features.getNames()

    # both names same
    caller = callerCon(data, pnames, pnames)
    other = callerCon(data, pnames, fnames)
    # only lhs should pass
    try:
        toCall = getattr(caller, attr1)
        ret = toCall(other)
    # rhs conflict between names; reverse operation eliminates conflict
    except InvalidArgumentValue:
        toCall = getattr(other, attr1)
        ret = toCall(caller)

    assert ret.points.getNames() == caller.points.getNames()
    assert ret.features.getNames() == other.features.getNames()

    # check name separation between caller and returned object
    ret.points.setName('p1', 'p0')
    if inplace:
        assert 'p0' in caller.points.getNames()
        assert 'p1' not in caller.points.getNames()
    else:
        assert 'p0' not in caller.points.getNames()
        assert 'p1' in caller.points.getNames()
    assert 'p0' in ret.points.getNames()
    assert 'p1' not in ret.points.getNames()


def back_otherObjectExceptions(callerCon, attr1, attr2=None):
    """ Test operation raises exception when param is not a nimble Base object """
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    caller = callerCon(data)
    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall({1: 1, 2: 2, 3: 'three'})


def back_selfNotNumericException(callerCon, calleeCon, attr1, attr2=None):
    """ Test operation raises exception if self has non numeric data """
    data1 = [['why', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
    data2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    caller = callerCon(data1)
    callee = calleeConstructor(data2, calleeCon)

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall(callee)


def back_otherNotNumericException(callerCon, calleeCon, attr1, attr2=None):
    """ Test operation raises exception if param object has non numeric data """
    data1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    data2 = [['one', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
    caller = callerCon(data1)
    callee = calleeConstructor(data2, nimble.core.data.List)  # need to use nimble.core.data.List for string valued element

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall(callee)


def back_pShapeException(callerCon, calleeCon, attr1, attr2=None):
    """ Test operation raises exception the shapes of the object don't fit correctly """
    data1 = [[1, 2, 6], [4, 5, 3], [7, 8, 6]]
    data2 = [[1, 2, 3], [4, 5, 6], ]
    caller = callerCon(data1)
    callee = calleeConstructor(data2, calleeCon)

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall(callee)


def back_fShapeException(callerCon, calleeCon, attr1, attr2=None):
    """ Test operation raises exception the shapes of the object don't fit correctly """
    data1 = [[1, 2], [4, 5], [7, 8]]
    data2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    caller = callerCon(data1)
    callee = calleeConstructor(data2, calleeCon)

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall(callee)


def back_pEmptyException(callerCon, calleeCon, attr1, attr2=None):
    """ Test operation raises exception for point empty data """
    data = numpy.zeros((0, 2))
    caller = callerCon(data)
    callee = calleeConstructor(data, calleeCon)

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall(callee)


def back_fEmptyException(callerCon, calleeCon, attr1, attr2=None):
    """ Test operation raises exception for feature empty data """
    data = [[], []]
    caller = callerCon(data)
    callee = calleeConstructor(data, calleeCon)

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall(callee)


def back_byZeroException(callerCon, calleeCon, opName):
    """ Test operation when other data contains zero """
    data1 = [[1, 2, 6], [4, 5, 3], [7, 8, 6]]
    data2 = [[1, 2, 3], [0, 0, 0], [6, 7, 8]]
    if opName.startswith('__r'):
        # put zeros in lhs
        data1, data2 = data2, data1
        callee = calleeConstructor(data2, calleeCon)
    elif calleeCon is None:
        callee = 0
    else:
        callee = calleeConstructor(data2, calleeCon)
    caller = callerCon(data1)

    toCall = getattr(caller, opName)
    toCall(callee)


def back_byInf(callerCon, calleeCon, opName):
    """ Test operation when other data contains zero """
    inf = numpy.inf
    data1 = [[1, 2, 6], [4, 5, 3], [7, 8, 6]]
    data2 = [[inf, inf, inf], [inf, inf, inf], [inf, inf, inf]]
    if opName.startswith('__r'):
        # put inf in lhs
        data1, data2 = data2, data1
        callee = calleeConstructor(data2, calleeCon)
    elif calleeCon is None:
        callee = inf
    else:
        callee = calleeConstructor(data2, calleeCon)
    caller = callerCon(data1)

    toCall = getattr(caller, opName)

    ret = toCall(callee)

    if 'div' in opName:
        exp = callerCon(numpy.zeros((3, 3)))
    elif opName == '__rmod__' and calleeCon is None:
        exp = callerCon([[callee] * 3] * 3)
    elif opName == '__rmod__':
        exp = callee.copy()
    else:
        exp = caller.copy()

    if opName.startswith('__i'):
        assert caller == exp
    else:
        assert ret == exp


def makeAllData(constructor, rhsCons, numPts, numFts, sparsity):
    randomlf = nimble.random.data('Matrix', numPts, numFts, sparsity, useLog=False)
    randomrf = nimble.random.data('Matrix', numPts, numFts, sparsity, useLog=False)
    lhsf = randomlf.copy(to="numpyarray")
    rhsf = randomrf.copy(to="numpyarray")
    lhsi = numpy.array(numpyRandom.random_integers(1, 10, (numPts, numFts)), dtype=float)
    rhsi = numpy.array(numpyRandom.random_integers(1, 10, (numPts, numFts)), dtype=float)

    lhsfObj = constructor(lhsf)
    lhsiObj = constructor(lhsi)
    rhsfObj = None
    rhsiObj = None
    if rhsCons is not None:
        rhsfObj = rhsCons(rhsf)
        rhsiObj = rhsCons(rhsi)

    return (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj)


def back_autoVsNumpyObjCallee(constructor, opName, nimbleinplace, sparsity):
    """ Test operation of automated data against numpy operations """
    trials = 5
    for t in range(trials):
        numPts = pythonRandom.randint(1, 15)
        # use square for matmul so shapes are compatible, otherwise randomize
        if 'matmul' in opName:
            numFts = numPts
        else:
            numFts = pythonRandom.randint(1, 15)

        datas = makeAllData(constructor, constructor, numPts, numFts, sparsity)
        if 'pow' in opName:
            # map to be positive and > 0 avoids complex numbers and 0 division
            datas = map(lambda d: abs(d) + 1, datas)
        (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj) = datas
        # numpy does not have __imatmul__ implemented yet, use __matmul__
        if opName == '__imatmul__':
            npOp = '__matmul__'
        else:
            npOp = opName

        resultf = getattr(lhsf, npOp)(rhsf)
        resulti = getattr(lhsi, npOp)(rhsi)
        resfObj = getattr(lhsfObj, opName)(rhsfObj)
        resiObj = getattr(lhsiObj, opName)(rhsiObj)

        expfObj = constructor(resultf)
        expiObj = constructor(resulti)

        if nimbleinplace:
            assert expfObj.isApproximatelyEqual(lhsfObj)
            assert expiObj.isIdentical(lhsiObj)
        else:
            assert expfObj.isApproximatelyEqual(resfObj)
            assert expiObj.isIdentical(resiObj)
        assertNoNamesGenerated(lhsfObj)
        assertNoNamesGenerated(lhsiObj)
        assertNoNamesGenerated(rhsfObj)
        assertNoNamesGenerated(rhsiObj)
        assertNoNamesGenerated(resfObj)
        assertNoNamesGenerated(resiObj)


def back_autoVsNumpyScalar(constructor, opName, nimbleinplace, sparsity):
    """ Test operation of automated data with a scalar argument, against numpy operations """
    trials = 5
    for t in range(trials):
        numPts = pythonRandom.randint(1, 10)
        numFts = pythonRandom.randint(1, 10)

        scalar = pythonRandom.randint(1, 4)

        datas = makeAllData(constructor, None, numPts, numFts, sparsity)
        (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj) = datas

        resultf = getattr(lhsf, opName)(scalar)
        resulti = getattr(lhsi, opName)(scalar)
        resfObj = getattr(lhsfObj, opName)(scalar)
        resiObj = getattr(lhsiObj, opName)(scalar)

        expfObj = constructor(resultf)
        expiObj = constructor(resulti)

        if nimbleinplace:
            assert expfObj.isApproximatelyEqual(lhsfObj)
            assert expiObj.isIdentical(lhsiObj)
        else:
            assert expfObj.isApproximatelyEqual(resfObj)
            assert expiObj.isIdentical(resiObj)
        assertNoNamesGenerated(lhsfObj)
        assertNoNamesGenerated(lhsiObj)
        assertNoNamesGenerated(resfObj)
        assertNoNamesGenerated(resiObj)


def back_autoVsNumpyObjCalleeDiffTypes(constructor, opName, nimbleinplace, sparsity):
    """ Test operation on handmade data with different types of data objects"""
    makers = [getattr(nimble.core.data, retType) for retType in nimble.core.data.available]

    for i in range(len(makers)):
        maker = makers[i]
        numPts = pythonRandom.randint(1, 10)
        # use square for matmul so shapes are compatible, otherwise randomize
        if 'matmul' in opName:
            numFts = numPts
        else:
            numFts = pythonRandom.randint(1, 15)

        datas = makeAllData(constructor, maker, numPts, numFts, sparsity)
        if 'pow' in opName:
            # map to be positive and > 0 avoids complex numbers and 0 division
            datas = map(lambda d: abs(d) + 1, datas)
        (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj) = datas
        # numpy does not have __imatmul__ implemented yet, use __matmul__
        if opName == '__imatmul__':
            npOp = '__matmul__'
        else:
            npOp = opName
        resultf = getattr(lhsf, npOp)(rhsf)
        resulti = getattr(lhsi, npOp)(rhsi)
        resfObj = getattr(lhsfObj, opName)(rhsfObj)
        resiObj = getattr(lhsiObj, opName)(rhsiObj)

        expfObj = constructor(resultf)
        expiObj = constructor(resulti)

        if nimbleinplace:
            assert expfObj.isApproximatelyEqual(lhsfObj)
            assert expiObj.isIdentical(lhsiObj)
        else:
            assert expfObj.isApproximatelyEqual(resfObj)
            assert expiObj.isIdentical(resiObj)

            if type(resfObj) != type(lhsfObj):
                assert isinstance(resfObj, nimble.core.data.Base)
            if type(resiObj) != type(lhsiObj):
                assert isinstance(resiObj, nimble.core.data.Base)

        assertNoNamesGenerated(lhsfObj)
        assertNoNamesGenerated(lhsiObj)
        assertNoNamesGenerated(rhsfObj)
        assertNoNamesGenerated(rhsiObj)
        assertNoNamesGenerated(resfObj)
        assertNoNamesGenerated(resiObj)

def wrapAndCall(toWrap, expected, *args):
    try:
        toWrap(*args)
        assert False # expected exception was not raised
    except expected:
        pass

def run_full_backendDivMod(constructor, opName, inplace, sparsity):
    wrapAndCall(back_byZeroException, ZeroDivisionError, *(constructor, constructor, opName))
    wrapAndCall(back_byZeroException, ZeroDivisionError, *(constructor, None, opName))
    back_byInf(constructor, constructor, opName)
    back_byInf(constructor, None, opName)
    run_full_backend(constructor, opName, inplace, sparsity)


def run_full_backend(constructor, opName, inplace, sparsity):
    wrapAndCall(back_otherObjectExceptions, InvalidArgumentType, *(constructor, opName))
    wrapAndCall(back_selfNotNumericException, ImproperObjectAction, *(constructor, constructor, opName))
    wrapAndCall(back_otherNotNumericException, InvalidArgumentValue, *(constructor, constructor, opName))
    wrapAndCall(back_pShapeException, InvalidArgumentValue, *(constructor, constructor, opName))
    wrapAndCall(back_fShapeException, InvalidArgumentValue, *(constructor, constructor, opName))
    wrapAndCall(back_pEmptyException, ImproperObjectAction, *(constructor, constructor, opName))
    wrapAndCall(back_fEmptyException, ImproperObjectAction, *(constructor, constructor, opName))

    back_autoVsNumpyObjCallee(constructor, opName, inplace, sparsity)
    back_autoVsNumpyScalar(constructor, opName, inplace, sparsity)
    back_autoVsNumpyObjCalleeDiffTypes(constructor, opName, inplace, sparsity)


@patch('nimble.core.data.Sparse._scalarZeroPreservingBinary_implementation', calledException)
def back_sparseScalarZeroPreserving(constructor, nimbleOp):
    data = [[1, 2, 3], [0, 0, 0]]
    toTest = constructor(data)
    rint = pythonRandom.randint(2, 5)
    try:
        ret = getattr(toTest, nimbleOp)(rint)
        if toTest.getTypeString() == "Sparse":
            assert False # did not use _scalarZeroPreservingBinary_implementation
    except CalledFunctionException:
        assert toTest.getTypeString() == "Sparse"
    except ZeroDivisionError:
        assert nimbleOp.startswith('__r')

@patch('nimble.core.data.Sparse._defaultBinaryOperations_implementation', calledException)
@patch('nimble.core.data.Sparse._scalarZeroPreservingBinary_implementation', calledException)
def back_sparseScalarOfOne(constructor, nimbleOp):
    """Test Sparse does not call helper functions for these scalar ops """
    data = [[1, 2, 3], [0, 0, 0]]
    toTest = constructor(data)
    rint = 1.0
    # Sparse should use a copy of toTest when exponent is 1, so neither
    # helper function should be called
    try:
        ret = getattr(toTest, nimbleOp)(rint)
    except CalledFunctionException:
        assert False # this function should not be used for this operation


class NumericalDataSafe(DataTestObject):

    ###############
    # matrixPower #
    ###############

    @raises(ImproperObjectAction)
    def test_matrixPower_notSquare(self):
        raw = [[1, 2, 3], [4, 5, 6]]
        obj = self.constructor(raw)

        ret = obj.matrixPower(2)

    @raises(InvalidArgumentType)
    def test_matrixPower_invalidPower(self):
        raw = [[1, 2], [3, 4]]
        obj = self.constructor(raw)

        ret = obj.matrixPower(2.3)

    @noLogEntryExpected
    def test_matrixPower_powerOfZero(self):
        raw = [[1, 2], [3, 4]]
        pNames = ['p1', 'p2']
        fNames = ['f1', 'f2']
        obj = self.constructor(raw, pointNames=pNames, featureNames=fNames)

        ret = obj.matrixPower(0)

        expRaw = [[1, 0], [0, 1]]
        exp = self.constructor(expRaw, pointNames=pNames, featureNames=fNames)

        assert ret == exp

    @noLogEntryExpected
    def test_matrixPower_positivePower(self):
        raw = [[1, 2], [3, 4]]
        pNames = ['p1', 'p2']
        fNames = ['f1', 'f2']
        obj = self.constructor(raw, pointNames=pNames, featureNames=fNames)

        ret = obj.matrixPower(3)

        expRaw = [[37,  54], [81, 118]]
        exp = self.constructor(expRaw, pointNames=pNames, featureNames=fNames)

        assert ret == exp

    @noLogEntryExpected
    def test_matrixPower_negativePower(self):
        raw = [[1, 2], [3, 4]]
        pNames = ['p1', 'p2']
        fNames = ['f1', 'f2']
        obj = self.constructor(raw, pointNames=pNames, featureNames=fNames)

        ret = obj.matrixPower(-3)

        expRaw = [[-14.75 , 6.75], [10.125, -4.625]]
        exp = self.constructor(expRaw, pointNames=pNames, featureNames=fNames)

        # can be small differences due to inverse calculation
        assert ret.isApproximatelyEqual(exp)

    @raises(CalledFunctionException)
    @patch('nimble.calculate.inverse', calledException)
    def test_matrixPower_callsNimbleCalculateInverse(self):
        raw = [[1, 2], [3, 4]]
        pNames = ['p1', 'p2']
        fNames = ['f1', 'f2']
        obj = self.constructor(raw, pointNames=pNames, featureNames=fNames)

        ret = obj.matrixPower(-3)

    @raises(InvalidArgumentValue)
    def test_matrixPower_notInvertable(self):
        raw = [[1, 1], [1, 1]]
        pNames = ['p1', 'p2']
        fNames = ['f1', 'f2']
        obj = self.constructor(raw, pointNames=pNames, featureNames=fNames)

        ret = obj.matrixPower(-3)

    ###############################
    # __matmul__ / matrixMultiply #
    ###############################

    @raises(CalledFunctionException)
    @patch('nimble.core.data.Base.__matmul__', calledException)
    def test_matrixMultiply_uses__matmul__backend(self):
        data1 = [[1, 2], [4, 5], [7, 8]]
        data2 = [[1, 2, 3], [4, 5, 6]]
        caller = self.constructor(data1)
        callee = self.constructor(data2)

        caller.matrixMultiply(callee)

    @raises(ImproperObjectAction)
    def test_matmul_selfNotNumericException(self):
        """ Test __matmul__ raises exception if self has non numeric data """
        back_selfNotNumericException(self.constructor, self.constructor, '__matmul__')

    @raises(InvalidArgumentValue)
    def test_matmul_otherNotNumericException(self):
        """ Test __matmul__ raises exception if param object has non numeric data """
        back_otherNotNumericException(self.constructor, self.constructor, '__matmul__')

    @raises(InvalidArgumentValue)
    def test_matmul_shapeException(self):
        """ Test __matmul__ raises exception the shapes of the object don't fit correctly """
        data1 = [[1, 2], [4, 5], [7, 8]]
        data2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        caller = self.constructor(data1)
        callee = self.constructor(data2)

        caller @ callee

    @raises(ImproperObjectAction)
    def test_matmul_pEmptyException(self):
        """ Test __matmul__ raises exception for point empty data """
        data = []
        fnames = ['one', 'two']
        caller = self.constructor(data, featureNames=fnames)
        callee = caller.copy()
        callee.transpose()

        caller @ callee

    @raises(ImproperObjectAction)
    def test_matmul_fEmptyException(self):
        """ Test __matmul__ raises exception for feature empty data """
        data = [[], []]
        pnames = ['one', 'two']
        caller = self.constructor(data, pointNames=pnames)
        callee = caller.copy()
        callee.transpose()

        caller @ callee

    @noLogEntryExpected
    def test_matmul_autoObjs(self):
        """ Test __matmul__ against automated data """
        back_autoVsNumpyObjCallee(self.constructor, '__matmul__', False, 0.2)

    def test_matmul_autoVsNumpyObjCalleeDiffTypes(self):
        """ Test __matmul__ against generated data with different nimble types of objects """
        back_autoVsNumpyObjCalleeDiffTypes(self.constructor, '__matmul__', False, 0.2)

    def test_matmul_matrixmul_pfname_preservations(self):
        """ Test p/f names are preserved when calling __matmul__ with obj arg"""
        back_matrixmul_pfname_preservations(self.constructor, '__matmul__', False)

    def test_matmul_matrixmul_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__matmul__', False)

    ###############
    # __rmatmul__ #
    ###############

    @raises(ImproperObjectAction)
    def test_rmatmul_selfNotNumericException(self):
        """ Test __rmatmul__ raises exception if self has non numeric data """
        back_selfNotNumericException(self.constructor, self.constructor, '__rmatmul__')

    @raises(InvalidArgumentValue)
    def test_rmatmul_otherNotNumericException(self):
        """ Test __rmatmul__ raises exception if param object has non numeric data """
        back_otherNotNumericException(self.constructor, self.constructor, '__rmatmul__')

    @raises(InvalidArgumentValue)
    def test_rmatmul_shapeException(self):
        """ Test __rmatmul__ raises exception the shapes of the object don't fit correctly """
        dataLHS = [[1, 2], [4, 5], [7, 8]]
        dataRHS = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        callee = self.constructor(dataLHS)
        caller = self.constructor(dataRHS)

        caller.__rmatmul__(callee)

    @raises(ImproperObjectAction)
    def test_rmatmul_pEmptyException(self):
        """ Test __rmatmul__ raises exception for point empty data """
        data = []
        fnames = ['one', 'two']
        caller = self.constructor(data, featureNames=fnames)
        callee = caller.copy()
        callee.transpose()

        caller.__rmatmul__(callee)

    @raises(ImproperObjectAction)
    def test_rmatmul_fEmptyException(self):
        """ Test __rmatmul__ raises exception for feature empty data """
        data = [[], []]
        pnames = ['one', 'two']
        caller = self.constructor(data, pointNames=pnames)
        callee = caller.copy()
        callee.transpose()

        caller.__rmatmul__(callee)

    @noLogEntryExpected
    def test_rmatmul_autoObjs(self):
        """ Test __rmatmul__ against automated data """
        back_autoVsNumpyObjCallee(self.constructor, '__rmatmul__', False, 0.2)

    def test_rmatmul_autoVsNumpyObjCalleeDiffTypes(self):
        """ Test __rmatmul__ against generated data with different nimble types of objects """
        back_autoVsNumpyObjCalleeDiffTypes(self.constructor, '__rmatmul__', False, 0.2)

    def test_rmatmul_matrixmul_pfname_preservations(self):
        """ Test p/f names are preserved when calling __rmatmul__ with obj arg"""
        back_matrixmul_pfname_preservations(self.constructor, '__rmatmul__', False)

    def test_rmatmul_matrixmul_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__rmatmul__', False)

    ############
    # __add__ #
    ############
    @noLogEntryExpected
    def test_add_fullSuite(self):
        """ __add__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__add__', False, 0.2)

    def test_add_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __add__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__add__', False)

    def test_add_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__add__')

    def test_add_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __add__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__add__', False)

    def test_add_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__add__', False)

    ############
    # __radd__ #
    ############
    @noLogEntryExpected
    def test_radd_fullSuite(self):
        """ __radd__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__radd__', False, 0.2)

    def test_radd_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __radd__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__radd__', False)

    def test_radd_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__radd__')

    def test_radd_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __radd__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__radd__', False)

    def test_radd_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__radd__', False)


    ############
    # __sub__ #
    ############
    @noLogEntryExpected
    def test_sub_fullSuite(self):
        """ __sub__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__sub__', False, 0.2)

    def test_sub_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __sub__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__sub__', False)

    def test_sub_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__sub__')

    def test_sub_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __sub__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__sub__', False)

    def test_sub_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__sub__', False)


    ############
    # __rsub__ #
    ############
    @noLogEntryExpected
    def test_rsub_fullSuite(self):
        """ __rsub__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__rsub__', False, 0.2)

    def test_rsub_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __rsub__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__rsub__', False)

    def test_rsub_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__rsub__')

    def test_rsub_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __rsub__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__rsub__', False)

    def test_rsub_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__rsub__', False)

    ############
    # __mul__ #
    ############
    @noLogEntryExpected
    def test_mul_fullSuite(self):
        """ __mul__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__mul__', False, 0.2)

    def test_mul_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __mul__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__mul__', False)

    def test_mul_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__mul__')

    def test_mul_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __mul__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__mul__', False)

    def test_mul_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__mul__', False)

    def test_mul_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__mul__')

    def test_mul_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__mul__')

    ############
    # __rmul__ #
    ############
    @noLogEntryExpected
    def test_rmul_fullSuite(self):
        """ __rmul__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__rmul__', False, 0.2)

    def test_rmul_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __rmul__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__rmul__', False)

    def test_rmul_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__rmul__')

    def test_rmul_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __rmul__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__rmul__', False)

    def test_rmul_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__rmul__', False)

    def test_rmul_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__rmul__')

    def test_rmul_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__rmul__')

    ###############
    # __truediv__ #
    ###############
    @noLogEntryExpected
    def test_truediv_fullSuite(self):
        """ __truediv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, '__truediv__', False, 0)

    def test_truediv_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __truediv__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__truediv__', False)

    def test_truediv_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__truediv__')

    def test_truediv_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __truediv__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__truediv__', False)

    def test_truediv_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__truediv__', False)

    def test_truediv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__truediv__')

    def test_truediv_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__truediv__')


    ################
    # __rtruediv__ #
    ################
    @noLogEntryExpected
    def test_rtruediv_fullSuite(self):
        """ __rtruediv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, '__rtruediv__', False, 0)

    def test_rtruediv_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __rtruediv__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__rtruediv__', False)

    def test_rtruediv_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__rtruediv__')

    def test_rtruediv_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __rtruediv__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__rtruediv__', False)

    def test_rtruediv_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__rtruediv__', False)

    def test_rtruediv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__rtruediv__')

    ###############
    # __floordiv__ #
    ###############
    @noLogEntryExpected
    def test_floordiv_fullSuite(self):
        """ __floordiv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, '__floordiv__', False, 0)

    def test_floordiv_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __floordiv__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__floordiv__', False)

    def test_floordiv_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__floordiv__')

    def test_floordiv_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __floordiv__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__floordiv__', False)

    def test_floordiv_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__floordiv__', False)

    def test_floordiv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__floordiv__')


    ################
    # __rfloordiv__ #
    ################
    @noLogEntryExpected
    def test_rfloordiv_fullSuite(self):
        """ __rfloordiv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, '__rfloordiv__', False, 0)

    def test_rfloordiv_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __rfloordiv__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__rfloordiv__', False)

    def test_rfloordiv_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__rfloordiv__')

    def test_rfloordiv_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __rfloordiv__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__rfloordiv__', False)

    def test_rfloordiv_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__rfloordiv__', False)

    def test_rfloordiv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__rfloordiv__')


    ###############
    # __mod__ #
    ###############
    @noLogEntryExpected
    def test_mod_fullSuite(self):
        """ __mod__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, '__mod__', False, 0)

    def test_mod_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __mod__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__mod__', False)

    def test_mod_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__mod__')

    def test_mod_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __mod__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__mod__', False)

    def test_mod_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__mod__', False)

    def test_mod_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__mod__')


    ################
    # __rmod__ #
    ################
    @noLogEntryExpected
    def test_rmod_fullSuite(self):
        """ __rmod__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, '__rmod__', False, 0)

    def test_rmod_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __rmod__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__rmod__', False)

    def test_rmod_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__rmod__')

    def test_rmod_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __rmod__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__rmod__', False)

    def test_rmod_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__rmod__', False)

    def test_rmod_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__rmod__')

    ###########
    # __pow__ #
    ###########
    @noLogEntryExpected
    def test_pow_fullSuite(self):
        """ __pow__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__pow__', False, 0)

    @raises(ZeroDivisionError)
    def test_pow_nimbleObj_zeroDivision_exception(self):
        lhs = [[1, 2, 0], [4, 5, 6], [7, 8, 9]]
        rhs = [[3, 2, -1], [2, 2, 2], [4, 5, 9]]
        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)
        lhsObj ** rhsObj

    @raises(ZeroDivisionError)
    def test_pow_scalar_zeroDivision_exception(self):
        lhs = [[1, 2, 0], [4, 5, 6], [7, 8, 9]]
        rhs = -1
        lhsObj = self.constructor(lhs)
        lhsObj ** rhs

    @raises(ImproperObjectAction)
    def test_pow_nimbleObj_complexNumber_exception(self):
        lhs = [[1, 2, -0.895], [4, 5, 6], [7, 8, 9]]
        rhs = [[3, 2, -0.895], [2, 2, 2], [4, 5, 9]]
        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)
        lhsObj ** rhsObj

    @raises(ImproperObjectAction)
    def test_pow_scalar_complexNumber_exception(self):
        lhs = [[1, 2, -0.895], [4, 5, 6], [7, 8, 9]]
        rhs = -0.895
        lhsObj = self.constructor(lhs)
        lhsObj ** rhs

    def test_pow_nimbleObj_inf(self):
        inf = numpy.inf
        lhs = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
        rhs = [[inf, inf, inf], [inf, inf, inf], [inf, inf, inf]]
        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)

        ret = lhsObj ** rhsObj
        exp = self.constructor(numpy.zeros((3, 3)))
        assert ret == exp

    def test_pow_scalar_inf(self):
        lhs = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
        rhs = numpy.inf
        lhsObj = self.constructor(lhs)

        ret = lhsObj ** rhs
        exp = self.constructor(numpy.zeros((3, 3)))
        assert ret == exp

    def test_pow_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __pow__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__pow__', False)

    def test_pow_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__pow__')

    def test_pow_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__pow__')

    def test_pow_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__pow__')

    ############
    # __rpow__ #
    ############

    @noLogEntryExpected
    def test_rpow_fullSuite(self):
        """ __rpow__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__rpow__', False, 0)

    @raises(ZeroDivisionError)
    def test_rpow_scalar_zeroDivision_exception(self):
        data = [[1, 2, -1], [4, 5, 6], [7, 8, 9]]
        num = 0
        obj = self.constructor(data)
        num ** obj

    @raises(ImproperObjectAction)
    def test_rpow_scalar_complexNumber_exception(self):
        data = [[1, 2, -0.895], [4, 5, 6], [7, 8, 9]]
        num = -0.895
        obj = self.constructor(data)
        num ** obj

    def test_rpow_scalar_inf(self):
        inf = numpy.inf
        lhs = [[inf, inf, inf], [inf, inf, inf], [inf, inf, inf]]
        num = 0.1
        lhsObj = self.constructor(lhs)

        ret = num ** lhsObj
        exp = self.constructor(numpy.zeros((3, 3)))
        assert ret == exp

    def test_rpow_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __rpow__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__rpow__', False)

    def test_rpow_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__rpow__')


    ###########
    # __pos__ #
    ###########
    @noLogEntryExpected
    def test_pos_DoesntCrash(self):
        """ Test that __pos__ does nothing and doesn't crash """
        data1 = [[1, 2], [4, 5], [7, 8]]
        caller = self.constructor(data1)

        ret1 = +caller
        ret2 = caller.__pos__()

        assert caller.isIdentical(ret1)
        assert caller.isIdentical(ret2)

    def test_pos_unary_pfname_preservations(self):
        """ Test that point / feature names are preserved when calling __pos__ """
        back_unary_pfname_preservations(self.constructor, '__pos__')


    def test_pos_unary_NamePath_preservatinos(self):
        back_unary_NamePath_preservations(self.constructor, '__pos__')

    ###########
    # __neg__ #
    ###########
    @noLogEntryExpected
    def test_neg_simple(self):
        """ Test that __neg__ works as expected on some simple data """
        data1 = [[1, 2], [-4, -5], [7, -8], [0, 0]]
        data2 = [[-1, -2], [4, 5], [-7, 8], [0, 0]]
        caller = self.constructor(data1)
        exp = self.constructor(data2)

        ret1 = -caller
        ret2 = caller.__neg__()

        assert exp.isIdentical(ret1)
        assert exp.isIdentical(ret2)

    def test_neg_unary_name_preservations(self):
        """ Test that point / feature names are preserved when calling __neg__ """
        back_unary_pfname_preservations(self.constructor, '__neg__')

    def test_neg_unary_NamePath_preservatinos(self):
        back_unary_NamePath_preservations(self.constructor, '__neg__')


    ###########
    # __abs__ #
    ###########
    @noLogEntryExpected
    def test_abs_simple(self):
        """ Test that __abs__ works as expected on some simple data """
        data1 = [[1, 2], [-4, -5], [7, -8], [0, 0]]
        data2 = [[1, 2], [4, 5], [7, 8], [0, 0]]
        caller = self.constructor(data1)
        exp = self.constructor(data2)

        ret1 = abs(caller)
        ret2 = caller.__abs__()

        assert exp.isIdentical(ret1)
        assert exp.isIdentical(ret2)

    def test_abs_unary_name_preservations(self):
        """ Test that point / feature names are preserved when calling __abs__ """
        back_unary_pfname_preservations(self.constructor, '__abs__')

    def test_abs_unary_NamePath_preservatinos(self):
        back_unary_NamePath_preservations(self.constructor, '__abs__')

    ####################
    # logical backends #
    ####################

    @raises(ImproperObjectAction)
    def back_logical_exception_selfInvalidValue(self, logicOp):
        lhs = [[True, 2], [True, False]]
        rhs = [[False, True], [False, False]]

        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)

        getattr(lhsObj, logicOp)(rhsObj)

    @raises(ImproperObjectAction)
    def back_logical_exception_otherInvalidValue(self, logicOp):
        lhs = [[True, True], [True, False]]
        rhs = [[False, True], [2, False]]

        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)

        getattr(lhsObj, logicOp)(rhsObj)

    @raises(InvalidArgumentValue)
    def back_logical_exception_shapeMismatch(self, logicOp):
        lhs = [[True, True, True], [True, False, False]]
        rhs = [[False, True], [False, False]]

        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)

        getattr(lhsObj, logicOp)(rhsObj)

    @raises(ImproperObjectAction)
    def back_logical_exception_pEmpty(self, logicOp):
        lhs = numpy.empty((0, 3))
        rhs = numpy.empty((0, 3))

        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)

        getattr(lhsObj, logicOp)(rhsObj)

    @raises(ImproperObjectAction)
    def back_logical_exception_fEmpty(self, logicOp):
        lhs = numpy.empty((3, 0))
        rhs = numpy.empty((3, 0))

        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)

        getattr(lhsObj, logicOp)(rhsObj)

    def getLogicalExpectedOutput(self, logicOp):
        if logicOp == '__and__':
            return [[False, True], [False, False]]
        if logicOp == '__or__':
            return [[True, True], [True, False]]
        if logicOp == '__xor__':
            return [[True, False], [True, False]]

    @noLogEntryExpected
    def back_logical_allCombinations(self, logicOp, inputType):
        if inputType == 'int':
            lhs = [[1, 1], [1, 0]]
            rhs = [[0, 1], [0, 0]]
        elif inputType == 'float':
            lhs = [[1.0, 1.0], [1.0, 0.0]]
            rhs = [[0.0, 1.0], [0.0, 0.0]]
        else:
            lhs = [[True, True], [True, False]]
            rhs = [[False, True], [False, False]]


        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)

        exp = self.getLogicalExpectedOutput(logicOp)
        expObj = self.constructor(exp)

        assert expObj == getattr(lhsObj, logicOp)(rhsObj)

    @noLogEntryExpected
    def back_logical_diffObjectTypes(self, logicOp):
        lhs = [[True, True], [True, False]]
        rhs = [[False, True], [False, False]]

        lhsObj = self.constructor(lhs)

        exp = self.getLogicalExpectedOutput(logicOp)
        expObj = self.constructor(exp)
        for rType in [t for t in nimble.core.data.available if t != lhsObj.getTypeString()]:
            rhsObj = nimble.data(rType, rhs, useLog=False)

            assert expObj == getattr(lhsObj, logicOp)(rhsObj)

    def run_logical_fullBackend(self, logicOp):
        self.back_logical_exception_selfInvalidValue(logicOp)
        self.back_logical_exception_otherInvalidValue(logicOp)
        self.back_logical_exception_shapeMismatch(logicOp)
        self.back_logical_exception_pEmpty(logicOp)
        self.back_logical_exception_fEmpty(logicOp)

        self.back_logical_allCombinations(logicOp, inputType='bool')
        self.back_logical_allCombinations(logicOp, inputType='int')
        self.back_logical_allCombinations(logicOp, inputType='float')
        self.back_logical_diffObjectTypes(logicOp)

    ###########
    # __and__ #
    ###########

    def test_logical_and_fullSuite(self):
        self.run_logical_fullBackend('__and__')

    def test_logical_and_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __and__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__and__', False)

    def test_logical_and_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__and__', False)


    ##########
    # __or__ #
    ##########

    def test_logical_or_fullSuite(self):
        self.run_logical_fullBackend('__or__')

    def test_logical_or_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __or__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__or__', False)

    def test_logical_or_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__or__', False)

    ##########
    # __xor__ #
    ##########

    def test_logical_xor_fullSuite(self):
        self.run_logical_fullBackend('__xor__')

    def test_logical_xor_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __xor__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__xor__', False)

    def test_logical_xor_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__xor__', False)

    ##############
    # __invert__ #
    ##############

    @raises(ImproperObjectAction)
    def test_invert_exception_InvalidValue(self):
        bools = [[True, 2], [False, 0]]
        boolsObj = self.constructor(bools)
        ~boolsObj

    def test_invert_returnsAllBools(self):
        bools = [[True, True], [False, False]]
        boolsObj = self.constructor(bools)
        boolsInvert = ~boolsObj

        assert boolsInvert[0, 0] is False or boolsInvert[0, 0] is numpy.bool_(False)
        assert boolsInvert[0, 1] is False or boolsInvert[0, 1] is numpy.bool_(False)
        assert boolsInvert[1, 0] is True or boolsInvert[1, 0] is numpy.bool_(True)
        assert boolsInvert[1, 1] is True or boolsInvert[1, 1] is numpy.bool_(True)

    @noLogEntryExpected
    def test_invert_TrueFalse(self):
        bools = [[True, True], [False, False]]
        boolsObj = self.constructor(bools)
        exp = [[False, False], [True, True]]
        expObj = self.constructor(exp)

        assert ~boolsObj == expObj

    @noLogEntryExpected
    def test_invert_int0sAnd1s(self):
        bools = [[1, 1], [0, 0]]
        boolsObj = self.constructor(bools)
        exp = [[False, False], [True, True]]
        expObj = self.constructor(exp)

        assert ~boolsObj == expObj

    @noLogEntryExpected
    def test_invert_float0sAnd1s(self):
        bools = [[1.0, 1.0], [0.0, 0.0]]
        boolsObj = self.constructor(bools)
        exp = [[False, False], [True, True]]
        expObj = self.constructor(exp)

        assert ~boolsObj == expObj

    def test_invert_pfname_preservation(self):
        bools = [[True, True], [False, False]]
        pnames = ['p1', 'p2']
        fnames = ['f1', 'f2']

        boolsObj = self.constructor(bools, pointNames=pnames, featureNames=fnames)

        boolsInvert = ~boolsObj

        assert boolsInvert.points.getNames() == pnames
        assert boolsInvert.features.getNames() == fnames

    def test_invert_namePath_preservation(self):
        bools = [[True, True], [False, False]]

        boolsObj = self.constructor(bools, name=preserveName, path=preservePair)

        boolsInvert = ~boolsObj

        assert boolsInvert.absolutePath == preserveAPath
        assert boolsInvert.relativePath == preserveRPath
        assert boolsInvert.name != preserveName
        assert boolsInvert.nameIsDefault()


class NumericalModifying(DataTestObject):

    ###############
    # __imatmul__ #
    ###############

    @raises(ImproperObjectAction)
    def test_imatmul_selfNotNumericException(self):
        """ Test __imatmul__ raises exception if self has non numeric data """
        back_selfNotNumericException(self.constructor, self.constructor, '__imatmul__')

    @raises(InvalidArgumentValue)
    def test_imatmul_otherNotNumericException(self):
        """ Test __imatmul__ raises exception if param object has non numeric data """
        back_otherNotNumericException(self.constructor, self.constructor, '__imatmul__')

    @raises(InvalidArgumentValue)
    def test_imatmul_shapeException(self):
        """ Test __imatmul__ raises exception the shapes of the object don't fit correctly """
        data1 = [[1, 2], [4, 5], [7, 8]]
        data2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        caller = self.constructor(data1)
        callee = self.constructor(data2)

        caller.__imatmul__(callee)

    @raises(ImproperObjectAction)
    def test_imatmul_pEmptyException(self):
        """ Test __imatmul__ raises exception for point empty data """
        data = []
        fnames = ['one', 'two']
        caller = self.constructor(data, featureNames=fnames)
        callee = caller.copy()
        callee.transpose()

        caller @= callee

    @raises(ImproperObjectAction)
    def test_imatmul_fEmptyException(self):
        """ Test __imatmul__ raises exception for feature empty data """
        data = [[], []]
        pnames = ['one', 'two']
        caller = self.constructor(data, pointNames=pnames)
        callee = caller.copy()
        callee.transpose()

        caller @= callee

    @noLogEntryExpected
    def test_imatmul_autoObjs(self):
        """ Test __imatmul__ against automated data """
        back_autoVsNumpyObjCallee(self.constructor, '__imatmul__', True, 0.2)

    def test_imatmul__autoVsNumpyObjCalleeDiffTypes(self):
        """ Test __imatmul__ against generated data with different nimble types of objects """
        back_autoVsNumpyObjCalleeDiffTypes(self.constructor, '__imatmul__', True, 0.2)

    def test_imatmul_matrixmul_pfname_preservations(self):
        """ Test p/f names are preserved when calling __imatmul__ with obj arg"""
        back_matrixmul_pfname_preservations(self.constructor, '__imatmul__', True)

    def test_imatmul_matrixmul_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__imatmul__', True)


    ############
    # __iadd__ #
    ############
    @noLogEntryExpected
    def test_iadd_fullSuite(self):
        """ __iadd__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__iadd__', True, 0.2)

    def test_iadd_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __iadd__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__iadd__', True)

    def test_iadd_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__iadd__')

    def test_iadd_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __iadd__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__iadd__', True)

    def test_iadd_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__iadd__', True)


    ############
    # __isub__ #
    ############
    @noLogEntryExpected
    def test_isub_fullSuite(self):
        """ __isub__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__isub__', True, 0.2)

    def test_isub_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __isub__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__isub__', True)

    def test_isub_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__isub__')

    def test_isub_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __isub__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__isub__', True)

    def test_isub_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__isub__', True)

    ############
    # __imul__ #
    ############
    @noLogEntryExpected
    def test_imul_fullSuite(self):
        """ __imul__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__imul__', True, 0.2)

    def test_imul_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __imul__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__imul__', True)

    def test_imul_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__imul__')

    def test_imul_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __imul__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__imul__', True)

    def test_imul_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__imul__', True)

    def test_imul_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__imul__')

    def test_imul_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__imul__')

    ################
    # __itruediv__ #
    ################
    @noLogEntryExpected
    def test_itruediv_fullSuite(self):
        """ __itruediv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, '__itruediv__', True, 0)

    def test_itruediv_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __itruediv__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__itruediv__', True)

    def test_itruediv_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__itruediv__')

    def test_itruediv_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __itruediv__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__itruediv__', True)

    def test_itruediv_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__itruediv__', True)

    def test_itruediv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__itruediv__')

    def test_itruediv_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__itruediv__')


    ################
    # __ifloordiv__ #
    ################
    @noLogEntryExpected
    def test_ifloordiv_fullSuite(self):
        """ __ifloordiv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, '__ifloordiv__', True, 0)

    def test_ifloordiv_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __ifloordiv__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__ifloordiv__', True)

    def test_ifloordiv_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__ifloordiv__')

    def test_ifloordiv_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __ifloordiv__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__ifloordiv__', True)

    def test_ifloordiv_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__ifloordiv__', True)

    def test_ifloordiv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__ifloordiv__')


    ################
    # __imod__ #
    ################
    @noLogEntryExpected
    def test_imod_fullSuite(self):
        """ __imod__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, '__imod__', True, 0)


    def test_imod_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __imod__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__imod__', True)

    def test_imod_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__imod__')

    def test_imod_binaryelementwise_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwise __imod__"""
        back_binaryelementwise_pfname_preservations(self.constructor, '__imod__', True)

    def test_imod_binaryelementwise_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__imod__', True)

    def test_imod_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__imod__')


    ###########
    # __ipow__ #
    ###########
    @noLogEntryExpected
    def test_ipow_fullSuite(self):
        """ __ipow__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, '__ipow__', True, 0)

    @raises(ZeroDivisionError)
    def test_ipow_nimbleObj_zeroDivision_exception(self):
        lhs = [[1, 2, 0], [4, 5, 6], [7, 8, 9]]
        rhs = [[3, 2, -1], [2, 2, 2], [4, 5, 9]]
        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)
        lhsObj **= rhsObj

    @raises(ZeroDivisionError)
    def test_ipow_scalar_zeroDivision_exception(self):
        lhs = [[1, 2, 0], [4, 5, 6], [7, 8, 9]]
        rhs = -1
        lhsObj = self.constructor(lhs)
        lhsObj **= rhs

    @raises(ImproperObjectAction)
    def test_ipow_nimbleObj_complexNumber_exception(self):
        lhs = [[1, 2, -0.895], [4, 5, 6], [7, 8, 9]]
        rhs = [[3, 2, -0.895], [2, 2, 2], [4, 5, 9]]
        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)
        lhsObj **= rhsObj

    @raises(ImproperObjectAction)
    def test_ipow_scalar_complexNumber_exception(self):
        lhs = [[1, 2, -0.895], [4, 5, 6], [7, 8, 9]]
        rhs = -0.895
        lhsObj = self.constructor(lhs)
        lhsObj **= rhs

    def test_ipow_nimbleObj_inf(self):
        inf = numpy.inf
        lhs = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
        rhs = [[inf, inf, inf], [inf, inf, inf], [inf, inf, inf]]
        lhsObj = self.constructor(lhs)
        rhsObj = self.constructor(rhs)

        lhsObj **= rhsObj
        exp = self.constructor(numpy.zeros((3, 3)))
        assert lhsObj == exp

    def test_ipow_scalar_inf(self):
        lhs = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
        rhs = numpy.inf
        lhsObj = self.constructor(lhs)

        lhsObj **= rhs
        exp = self.constructor(numpy.zeros((3, 3)))
        assert lhsObj == exp

    def test_ipow_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __ipow__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__ipow__', True)

    def test_ipow_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__ipow__')

    def test_ipow_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__ipow__')

    def test_ipow_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__ipow__')

class AllNumerical(NumericalDataSafe, NumericalModifying):
    pass
