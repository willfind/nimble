"""

Methods tested in this file:

In object NumericalDataSafe:
__mul__, __rmul__,  __add__, __radd__,  __sub__, __rsub__,
__truediv__, __rtruediv__,  __floordiv__, __rfloordiv__,
__mod__, __rmod__ ,  __pow__,  __pos__, __neg__, __abs__

In object NumericalModifying:
elements.power, elements.multiply, __imul__, __iadd__, __isub__,
__itruediv__, __ifloordiv__,  __imod__, __ipow__,

"""
from __future__ import absolute_import
import sys
import numpy
import os
import os.path
from unittest.mock import patch

from nose.tools import *
import six
from six.moves import range

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction
from nimble.randomness import numpyRandom
from nimble.randomness import pythonRandom

from .baseObject import DataTestObject
from ..assertionHelpers import logCountAssertionFactory, noLogEntryExpected
from ..assertionHelpers import assertNoNamesGenerated
from ..assertionHelpers import CalledFunctionException, calledException


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

            assert ret.points.getNames() == pnames
            assert ret.features.getNames() == fnames

            caller.points.setName('p1', 'p0')
            if inplace:
                assert 'p0' in ret.points.getNames()
                assert 'p1' not in ret.points.getNames()
            else:
                assert 'p0' not in ret.points.getNames()
                assert 'p1' in ret.points.getNames()
            assert 'p0' in caller.points.getNames()
            assert 'p1' not in caller.points.getNames()
        except AssertionError:
            einfo = sys.exc_info()
            six.reraise(*einfo)
        except:
            pass


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

    otherRaw = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    # names not the same
    caller = callerCon(data, pnames, fnames)
    opnames = pnames
    ofnames = {'f0': 0, 'f1': 1, 'f2': 2}
    other = callerCon(otherRaw, opnames, ofnames)
    try:
        toCall = getattr(caller, op)
        ret = toCall(other)
        if ret != NotImplemented:
            assert False
    except InvalidArgumentValue:
        pass
    # if it isn't the exception we expect, pass it on
    except:
        einfo = sys.exc_info()
        six.reraise(einfo[1], None, einfo[2])

    # names interwoven
    other = callerCon(otherRaw, pnames, False)
    caller = callerCon(data, False, fnames)
    toCall = getattr(caller, op)
    ret = toCall(other)

    if ret != NotImplemented:
        assert ret.points.getNames() == pnames
        assert ret.features.getNames() == fnames

    # both names same
    caller = callerCon(data, pnames, fnames)
    other = callerCon(otherRaw, pnames, fnames)
    toCall = getattr(caller, op)
    ret = toCall(other)

    if ret != NotImplemented:
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

    if ret is None and attr2 is None:
        raise ValueError("Unexpected None return")

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

    if ret is None and attr2 is None:
        raise ValueError("Unexpected None return")

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

    if ret is None and attr2 is None:
        raise ValueError("Unexpected None return")

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


def back_matrixmul_pfname_preservations(callerCon, attr1, inplace, attr2=None):
    """ Test that p/f names are preserved when calling a binary element wise op """
    data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    pnames = ['p1', 'p2', 'p3']
    fnames = ['f1', 'f2', 'f3']

    # [p x f1] times [f2 x p] where f1 != f2
    caller = callerCon(data, pnames, fnames)
    ofnames = {'f0': 0, 'f1': 1, 'f2': 2}
    other = callerCon(data, ofnames, pnames)
    try:
        toCall = getattr(caller, attr1)
        if attr2 is not None:
            toCall = getattr(toCall, attr2)
        ret = toCall(other)
        if ret != NotImplemented:
            assert False
    except InvalidArgumentValue:
        pass
    # if it isn't the exception we expect, pass it on
    except:
        einfo = sys.exc_info()
        six.reraise(einfo[1], None, einfo[2])

    # names interwoven
    interPnames = ['f1', 'f2', None]
    caller = callerCon(data, pnames, fnames)
    other = callerCon(data, interPnames, fnames)
    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    ret = toCall(other)

    if ret != NotImplemented:
        assert ret.points.getNames() == pnames
        assert ret.features.getNames() == fnames

    # both names same
    caller = callerCon(data, pnames, pnames)
    other = callerCon(data, pnames, fnames)
    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    ret = toCall(other)

    if ret != NotImplemented:
        assert ret.points.getNames() == pnames
        assert ret.features.getNames() == fnames

        # check name seperation between caller and returned object
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
    try:
        caller = callerCon(data1)
        callee = calleeConstructor(data2, calleeCon)
    except:
        raise ImproperObjectAction("Data type doesn't support non numeric data")
    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall(callee)


def back_otherNotNumericException(callerCon, calleeCon, attr1, attr2=None):
    """ Test operation raises exception if param object has non numeric data """
    data1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    data2 = [['one', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
    caller = callerCon(data1)
    callee = calleeConstructor(data2, nimble.data.List)  # need to use nimble.data.List for string valued element

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


def back_byZeroException(callerCon, calleeCon, attr1, attr2=None):
    """ Test operation when other data contains zero """
    data1 = [[1, 2, 6], [4, 5, 3], [7, 8, 6]]
    data2 = [[1, 2, 3], [0, 0, 0], [6, 7, 8]]
    if attr1.startswith('__r'):
        # put zeros in lhs
        data1, data2 = data2, data1
    caller = callerCon(data1)
    callee = calleeConstructor(data2, calleeCon)

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall(callee)


def back_byInfException(callerCon, calleeCon, attr1, attr2=None):
    """ Test operation when other data contains an infinity """
    data1 = [[1, 2, 6], [4, 5, 3], [7, 8, 6]]
    data2 = [[1, 2, 3], [5, numpy.Inf, 10], [6, 7, 8]]
    caller = callerCon(data1)
    callee = calleeConstructor(data2, calleeCon)

    toCall = getattr(caller, attr1)
    if attr2 is not None:
        toCall = getattr(toCall, attr2)
    toCall(callee)


def makeAllData(constructor, rhsCons, n, sparsity):
    randomlf = nimble.createRandomData('Matrix', n, n, sparsity, useLog=False)
    randomrf = nimble.createRandomData('Matrix', n, n, sparsity, useLog=False)
    lhsf = randomlf.copy(to="numpymatrix")
    rhsf = randomrf.copy(to="numpymatrix")
    lhsi = numpy.matrix(numpyRandom.random_integers(1, 10, (n, n)), dtype=float)
    rhsi = numpy.matrix(numpyRandom.random_integers(1, 10, (n, n)), dtype=float)

    lhsfObj = constructor(lhsf)
    lhsiObj = constructor(lhsi)
    rhsfObj = None
    rhsiObj = None
    if rhsCons is not None:
        rhsfObj = rhsCons(rhsf)
        rhsiObj = rhsCons(rhsi)

    return (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj)


def back_autoVsNumpyObjCallee(constructor, npOp, nimbleOp, nimbleinplace, sparsity):
    """ Test operation of automated data against numpy operations """
    trials = 1
    for t in range(trials):
        n = pythonRandom.randint(1, 15)

        datas = makeAllData(constructor, constructor, n, sparsity)
        (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj) = datas
        resultf = npOp(lhsf, rhsf)
        resulti = npOp(lhsi, rhsi)
        resfObj = getattr(lhsfObj, nimbleOp)(rhsfObj)
        resiObj = getattr(lhsiObj, nimbleOp)(rhsiObj)

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


def back_autoVsNumpyScalar(constructor, npOp, nimbleOp, nimbleinplace, sparsity):
    """ Test operation of automated data with a scalar argument, against numpy operations """
    lside = nimbleOp.startswith('__r')
    trials = 5
    for t in range(trials):
        n = pythonRandom.randint(1, 10)

        scalar = pythonRandom.randint(1, 4)

        datas = makeAllData(constructor, None, n, sparsity)
        (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj) = datas

        if lside:
            resultf = npOp(scalar, lhsf)
            resulti = npOp(scalar, lhsi)
            resfObj = getattr(lhsfObj, nimbleOp)(scalar)
            resiObj = getattr(lhsiObj, nimbleOp)(scalar)
        else:
            resultf = npOp(lhsf, scalar)
            resulti = npOp(lhsi, scalar)
            resfObj = getattr(lhsfObj, nimbleOp)(scalar)
            resiObj = getattr(lhsiObj, nimbleOp)(scalar)

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


def back_autoVsNumpyObjCalleeDiffTypes(constructor, npOp, nimbleOp, nimbleinplace, sparsity):
    """ Test operation on handmade data with different types of data objects"""
    makers = [getattr(nimble.data, retType) for retType in nimble.data.available]

    for i in range(len(makers)):
        maker = makers[i]
        n = pythonRandom.randint(1, 10)

        datas = makeAllData(constructor, maker, n, sparsity)
        (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj) = datas

        resultf = npOp(lhsf, rhsf)
        resulti = npOp(lhsi, rhsi)
        resfObj = getattr(lhsfObj, nimbleOp)(rhsfObj)
        resiObj = getattr(lhsiObj, nimbleOp)(rhsiObj)
        expfObj = constructor(resultf)
        expiObj = constructor(resulti)

        if nimbleinplace:
            assert expfObj.isApproximatelyEqual(lhsfObj)
            assert expiObj.isIdentical(lhsiObj)
        else:
            assert expfObj.isApproximatelyEqual(resfObj)
            assert expiObj.isIdentical(resiObj)

            if type(resfObj) != type(lhsfObj):
                assert isinstance(resfObj, nimble.data.Base)
            if type(resiObj) != type(lhsiObj):
                assert isinstance(resiObj, nimble.data.Base)

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
    except:
        raise


def run_full_backendDivMod(constructor, npEquiv, nimbleOp, inplace, sparsity):
    wrapAndCall(back_byZeroException, ZeroDivisionError, *(constructor, constructor, nimbleOp))
    wrapAndCall(back_byInfException, InvalidArgumentValue, *(constructor, constructor, nimbleOp))

    run_full_backend(constructor, npEquiv, nimbleOp, inplace, sparsity)


def run_full_backend(constructor, npEquiv, nimbleOp, inplace, sparsity):
    wrapAndCall(back_otherObjectExceptions, InvalidArgumentType, *(constructor, nimbleOp))

    wrapAndCall(back_selfNotNumericException, ImproperObjectAction, *(constructor, constructor, nimbleOp))

    wrapAndCall(back_otherNotNumericException, InvalidArgumentValue, *(constructor, constructor, nimbleOp))

    wrapAndCall(back_pShapeException, InvalidArgumentValue, *(constructor, constructor, nimbleOp))

    wrapAndCall(back_fShapeException, InvalidArgumentValue, *(constructor, constructor, nimbleOp))

    wrapAndCall(back_pEmptyException, ImproperObjectAction, *(constructor, constructor, nimbleOp))

    wrapAndCall(back_fEmptyException, ImproperObjectAction, *(constructor, constructor, nimbleOp))

    back_autoVsNumpyObjCallee(constructor, npEquiv, nimbleOp, inplace, sparsity)

    back_autoVsNumpyScalar(constructor, npEquiv, nimbleOp, inplace, sparsity)

    back_autoVsNumpyObjCalleeDiffTypes(constructor, npEquiv, nimbleOp, inplace, sparsity)


def run_full_backendDivMod_rop(constructor, npEquiv, nimbleOp, inplace, sparsity):
    wrapAndCall(back_byZeroException, ZeroDivisionError, *(constructor, constructor, nimbleOp))
    run_full_backend_rOp(constructor, npEquiv, nimbleOp, inplace, sparsity)


def run_full_backend_rOp(constructor, npEquiv, nimbleOp, inplace, sparsity):
    wrapAndCall(back_otherObjectExceptions, InvalidArgumentType, *(constructor, nimbleOp))

    wrapAndCall(back_selfNotNumericException, ImproperObjectAction, *(constructor, constructor, nimbleOp))

    wrapAndCall(back_pEmptyException, ImproperObjectAction, *(constructor, constructor, nimbleOp))

    wrapAndCall(back_fEmptyException, ImproperObjectAction, *(constructor, constructor, nimbleOp))

    back_autoVsNumpyScalar(constructor, npEquiv, nimbleOp, inplace, sparsity)

@patch('nimble.data.Sparse._scalarZeroPreservingBinary_implementation', calledException)
def back_sparseScalarZeroPreserving(constructor, nimbleOp):
    data = [[1, 2, 3], [0, 0, 0]]
    toTest = constructor(data)
    rint = pythonRandom.randint(1, 5)
    try:
        ret = getattr(toTest, nimbleOp)(rint)
        if toTest.getTypeString() == "Sparse":
            assert False # did not use _scalarZeroPreservingBinary_implementation
    except CalledFunctionException:
        assert toTest.getTypeString() == "Sparse"


class NumericalDataSafe(DataTestObject):

    ###########
    # __mul__ #
    ###########

    @raises(ImproperObjectAction)
    def test_mul_selfNotNumericException(self):
        """ Test __mul__ raises exception if self has non numeric data """
        back_selfNotNumericException(self.constructor, self.constructor, '__mul__')

    @raises(InvalidArgumentValue)
    def test_mul_otherNotNumericException(self):
        """ Test __mul__ raises exception if param object has non numeric data """
        back_otherNotNumericException(self.constructor, self.constructor, '__mul__')

    @raises(InvalidArgumentValue)
    def test_mul_shapeException(self):
        """ Test __mul__ raises exception the shapes of the object don't fit correctly """
        data1 = [[1, 2], [4, 5], [7, 8]]
        data2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        caller = self.constructor(data1)
        callee = self.constructor(data2)

        caller * callee

    @raises(ImproperObjectAction)
    def test_mul_pEmptyException(self):
        """ Test __mul__ raises exception for point empty data """
        data = []
        fnames = ['one', 'two']
        caller = self.constructor(data, featureNames=fnames)
        callee = caller.copy()
        callee.transpose()

        caller * callee

    @raises(ImproperObjectAction)
    def test_mul_fEmptyException(self):
        """ Test __mul__ raises exception for feature empty data """
        data = [[], []]
        pnames = ['one', 'two']
        caller = self.constructor(data, pointNames=pnames)
        callee = caller.copy()
        callee.transpose()

        caller * callee

    @noLogEntryExpected
    def test_mul_autoObjs(self):
        """ Test __mul__ against automated data """
        back_autoVsNumpyObjCallee(self.constructor, numpy.dot, '__mul__', False, 0.2)

    @noLogEntryExpected
    def test_mul_autoScalar(self):
        """ Test __mul__ of a scalar against automated data """
        back_autoVsNumpyScalar(self.constructor, numpy.dot, '__mul__', False, 0.2)

    def test_autoVsNumpyObjCalleeDiffTypes(self):
        """ Test __mul__ against generated data with different nimble types of objects """
        back_autoVsNumpyObjCalleeDiffTypes(self.constructor, numpy.dot, '__mul__', False, 0.2)

    def test_mul_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __mul__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__mul__', False)

    def test_mul_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__mul__')

    def test_mul_matrixmul_pfname_preservations(self):
        """ Test p/f names are preserved when calling __mul__ with obj arg"""
        back_matrixmul_pfname_preservations(self.constructor, '__mul__', False)

    def test_mul_matrixmul_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__mul__', False)


    ############
    # __rmul__ #
    ############
    @noLogEntryExpected
    def test_rmul_autoScalar(self):
        """ Test __rmul__ of a scalar against automated data """
        back_autoVsNumpyScalar(self.constructor, numpy.multiply, '__rmul__', False, 0.2)

    def test_rmul_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __rmul__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__rmul__', False)

    def test_rmul_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__rmul__')

    def test_rmul_matrixmul_pfname_preservations(self):
        """ Test p/f names are preserved when calling __rmul__ with obj arg"""
        back_matrixmul_pfname_preservations(self.constructor, '__rmul__', False)

    def test_rmul_matrixmul_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__rmul__', False)


    ############
    # __add__ #
    ############
    @noLogEntryExpected
    def test_add_fullSuite(self):
        """ __add__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, numpy.add, '__add__', False, 0.2)

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
        run_full_backend_rOp(self.constructor, numpy.add, '__radd__', False, 0.2)

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
        run_full_backend(self.constructor, numpy.subtract, '__sub__', False, 0.2)

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
        run_full_backend_rOp(self.constructor, numpy.subtract, '__rsub__', False, 0.2)

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


    ###############
    # __truediv__ #
    ###############
    @noLogEntryExpected
    def test_truediv_fullSuite(self):
        """ __truediv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, numpy.true_divide, '__truediv__', False, 0)

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


    ################
    # __rtruediv__ #
    ################
    @noLogEntryExpected
    def test_rtruediv_fullSuite(self):
        """ __rtruediv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod_rop(self.constructor, numpy.true_divide, '__rtruediv__', False, 0)

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
        run_full_backendDivMod(self.constructor, numpy.floor_divide, '__floordiv__', False, 0)

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
        run_full_backendDivMod_rop(self.constructor, numpy.floor_divide, '__rfloordiv__', False, 0)

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
        run_full_backendDivMod(self.constructor, numpy.mod, '__mod__', False, 0)

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
        run_full_backendDivMod_rop(self.constructor, numpy.mod, '__rmod__', False, 0)

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

    def test_pow_exceptions(self):
        """ __pow__ Run the full standardized suite of tests for a binary numeric op """
        constructor = self.constructor
        nimbleOp = '__pow__'
        inputs = (constructor, nimbleOp)
        wrapAndCall(back_otherObjectExceptions, InvalidArgumentType, *inputs)

        inputs = (constructor, int, nimbleOp)
        wrapAndCall(back_selfNotNumericException, ImproperObjectAction, *inputs)

        inputs = (constructor, constructor, nimbleOp)
        wrapAndCall(back_pEmptyException, ImproperObjectAction, *inputs)

        inputs = (constructor, constructor, nimbleOp)
        wrapAndCall(back_fEmptyException, ImproperObjectAction, *inputs)

    @noLogEntryExpected
    def test_pow_autoVsNumpyScalar(self):
        """ Test __pow__ with automated data and a scalar argument, against numpy operations """
        trials = 5
        for t in range(trials):
            n = pythonRandom.randint(1, 15)
            scalar = pythonRandom.randint(0, 5)

            datas = makeAllData(self.constructor, None, n, .02)
            (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj) = datas

            resultf = lhsf ** scalar
            resulti = lhsi ** scalar
            resfObj = lhsfObj ** scalar
            resiObj = lhsiObj ** scalar

            expfObj = self.constructor(resultf)
            expiObj = self.constructor(resulti)

            assert expfObj.isApproximatelyEqual(resfObj)
            assert expiObj.isIdentical(resiObj)

    def test_pow_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __pow__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__pow__', False)

    def test_pow_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__pow__')


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


class NumericalModifying(DataTestObject):
    ##################
    # elements.power #
    ##################

    @raises(InvalidArgumentType)
    def test_elements_power_otherObjectExceptions(self):
        """ Test elements.power raises exception when param is not a nimble Base object """
        back_otherObjectExceptions(self.constructor, 'elements', 'power')

    @raises(ImproperObjectAction)
    def test_elements_power_selfNotNumericException(self):
        """ Test elements.power raises exception if self has non numeric data """
        back_selfNotNumericException(self.constructor, self.constructor, 'elements', 'power')

    @raises(InvalidArgumentValue)
    def test_elements_power_otherNotNumericException(self):
        """ Test elements.power raises exception if param object has non numeric data """
        back_otherNotNumericException(self.constructor, self.constructor, 'elements', 'power')

    @raises(InvalidArgumentValue)
    def test_elements_power_pShapeException(self):
        """ Test elements.power raises exception the shapes of the object don't fit correctly """
        back_pShapeException(self.constructor, self.constructor, 'elements', 'power')

    @raises(InvalidArgumentValue)
    def test_elements_power_fShapeException(self):
        """ Test elements.power raises exception the shapes of the object don't fit correctly """
        back_fShapeException(self.constructor, self.constructor, 'elements', 'power')

    @raises(ImproperObjectAction)
    def test_elements_power_pEmptyException(self):
        """ Test elements.power raises exception for point empty data """
        back_pEmptyException(self.constructor, self.constructor, 'elements', 'power')

    @raises(ImproperObjectAction)
    def test_elements_power_fEmptyException(self):
        """ Test elements.power raises exception for feature empty data """
        back_fEmptyException(self.constructor, self.constructor, 'elements', 'power')

    @logCountAssertionFactory(len(nimble.data.available))
    def test_elements_power_handmade(self):
        """ Test elements.power on handmade data """
        data = [[1.0, 2], [4, 5], [7, 4]]
        exponents = [[0, -1], [-.5, 2], [2, .5]]
        exp1 = [[1, .5], [.5, 25], [49, 2]]
        callerpnames = ['1', '2', '3']

        calleepnames = ['I', 'dont', 'match']
        calleefnames = ['one', 'two']

        for retType in nimble.data.available:
            caller = self.constructor(data, pointNames=callerpnames)
            exponentsObj = nimble.createData(retType, exponents, pointNames=calleepnames,
                                             featureNames=calleefnames, useLog=False)
            caller.elements.power(exponentsObj)

            exp1Obj = self.constructor(exp1, pointNames=callerpnames)

            assert exp1Obj.isIdentical(caller)

    def test_elements_power_lazyNameGeneration(self):
        """ Test elements.power on handmade data """
        data = [[1.0, 2], [4, 5], [7, 4]]
        exponents = [[0, -1], [-.5, 2], [2, .5]]
        exp1 = [[1, .5], [.5, 25], [49, 2]]

        for retType in nimble.data.available:
            caller = self.constructor(data)
            exponentsObj = nimble.createData(retType, exponents)
            caller.elements.power(exponentsObj)

            assertNoNamesGenerated(caller)

    @logCountAssertionFactory(len([getattr(nimble.data, retType) for retType in nimble.data.available]))
    def test_elements_power_handmadeScalar(self):
        """ Test elements.power on handmade data with scalar parameter"""
        data = [[1.0, 2], [4, 5], [7, 4]]
        exponent = 2
        exp1 = [[1, 4], [16, 25], [49, 16]]
        callerpnames = ['1', '2', '3']

        makers = [getattr(nimble.data, retType) for retType in nimble.data.available]

        for maker in makers:
            caller = self.constructor(data, pointNames=callerpnames)
            caller.elements.power(exponent)

            exp1Obj = self.constructor(exp1, pointNames=callerpnames)

            assert exp1Obj.isIdentical(caller)

    def test_elements_power_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, 'elements', False, 'power')

    ######################
    # elements.multiply #
    #####################

    @raises(InvalidArgumentType)
    def test_elements_multiply_otherObjectExceptions(self):
        """ Test elements.multiply raises exception when param is not a nimble Base object """
        back_otherObjectExceptions(self.constructor, 'elements', 'multiply')

    @raises(ImproperObjectAction)
    def test_elements_multiply_selfNotNumericException(self):
        """ Test elements.multiply raises exception if self has non numeric data """
        back_selfNotNumericException(self.constructor, self.constructor, 'elements', 'multiply')

    @raises(InvalidArgumentValue)
    def test_elements_multiply_otherNotNumericException(self):
        """ Test elements.multiply raises exception if param object has non numeric data """
        back_otherNotNumericException(self.constructor, self.constructor, 'elements', 'multiply')

    @raises(InvalidArgumentValue)
    def test_elements_multiply_pShapeException(self):
        """ Test elements.multiply raises exception the shapes of the object don't fit correctly """
        back_pShapeException(self.constructor, self.constructor, 'elements', 'multiply')

    @raises(InvalidArgumentValue)
    def test_elements_multiply_fShapeException(self):
        """ Test elements.multiply raises exception the shapes of the object don't fit correctly """
        back_fShapeException(self.constructor, self.constructor, 'elements', 'multiply')

    @raises(ImproperObjectAction)
    def test_elements_multiply_pEmptyException(self):
        """ Test elements.multiply raises exception for point empty data """
        back_pEmptyException(self.constructor, self.constructor, 'elements', 'multiply')

    @raises(ImproperObjectAction)
    def test_elements_multiply_fEmptyException(self):
        """ Test elements.multiply raises exception for feature empty data """
        back_fEmptyException(self.constructor, self.constructor, 'elements', 'multiply')

    @logCountAssertionFactory(2)
    def test_elements_multiply_handmade(self):
        """ Test elements.multiply on handmade data """
        data = [[1, 2], [4, 5], [7, 8]]
        twos = [[2, 2], [2, 2], [2, 2]]
        exp1 = [[2, 4], [8, 10], [14, 16]]
        halves = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

        caller = self.constructor(data)
        twosObj = self.constructor(twos)
        caller.elements.multiply(twosObj)

        exp1Obj = self.constructor(exp1)

        assert exp1Obj.isIdentical(caller)
        assertNoNamesGenerated(caller)

        halvesObj = self.constructor(halves)
        caller.elements.multiply(halvesObj)

        exp2Obj = self.constructor(data)

        assert caller.isIdentical(exp2Obj)
        assertNoNamesGenerated(caller)

    @logCountAssertionFactory(len(nimble.data.available) * 2)
    def test_elements_multiply_handmadeDifInputs(self):
        """ Test elements.multiply on handmade data with different input object types"""
        data = [[1, 2], [4, 5], [7, 8]]
        twos = [[2, 2], [2, 2], [2, 2]]
        exp1 = [[2, 4], [8, 10], [14, 16]]
        halves = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

        for retType in nimble.data.available:
            caller = self.constructor(data)
            twosObj = nimble.createData(retType, twos, useLog=False)
            caller.elements.multiply(twosObj)

            exp1Obj = self.constructor(exp1)

            assert exp1Obj.isIdentical(caller)
            assertNoNamesGenerated(caller)

            halvesObj = nimble.createData(retType, halves, useLog=False)
            caller.elements.multiply(halvesObj)

            exp2Obj = self.constructor(data)

            assert caller.isIdentical(exp2Obj)
            assertNoNamesGenerated(caller)

    def test_elementwiseMultipy_auto(self):
        """ Test elements.multiply on generated data against the numpy op """
        makers = [getattr(nimble.data, retType) for retType in nimble.data.available]

        for i in range(len(makers)):
            maker = makers[i]
            n = pythonRandom.randint(1, 10)

            randomlf = nimble.createRandomData('Matrix', n, n, .2)
            randomrf = nimble.createRandomData('Matrix', n, n, .2)
            lhsf = randomlf.copy(to="numpymatrix")
            rhsf = randomrf.copy(to="numpymatrix")
            lhsi = numpy.matrix(numpy.ones((n, n)))
            rhsi = numpy.matrix(numpy.ones((n, n)))

            lhsfObj = self.constructor(lhsf)
            rhsfObj = maker(rhsf)
            lhsiObj = self.constructor(lhsi)
            rhsiObj = maker(rhsi)

            resultf = numpy.multiply(lhsf, rhsf)
            resulti = numpy.multiply(lhsi, rhsi)
            lhsfObj.elements.multiply(rhsfObj)
            lhsiObj.elements.multiply(rhsiObj)

            expfObj = self.constructor(resultf)
            expiObj = self.constructor(resulti)

            assert expfObj.isApproximatelyEqual(lhsfObj)
            assert expiObj.isIdentical(lhsiObj)

    def test_elementwiseMultipy_pfname_preservations(self):
        """ Test p/f names are preserved when calling elementwiseMultipy"""
        data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        pnames = ['p1', 'p2', 'p3']
        fnames = ['f1', 'f2', 'f3']

        otherRaw = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        # names not the same
        caller = self.constructor(data, pnames, fnames)
        opnames = pnames
        ofnames = {'f0': 0, 'f1': 1, 'f2': 2}
        other = self.constructor(otherRaw, opnames, ofnames)
        try:
            toCall = getattr(getattr(caller, 'elements'), 'multiply')
            ret = toCall(other)
            assert False
        except InvalidArgumentValue:
            pass
        # if it isn't the exception we expect, pass it on
        except:
            einfo = sys.exc_info()
            six.reraise(einfo[1], None, einfo[2])

        # names interwoven
        other = self.constructor(otherRaw, pnames, False)
        caller = self.constructor(data, False, fnames)
        toCall = getattr(getattr(caller, 'elements'), 'multiply')
        ret = toCall(other)

        assert ret is None
        assert caller.points.getNames() == pnames
        assert caller.features.getNames() == fnames

        # both names same
        caller = self.constructor(data, pnames, fnames)
        other = self.constructor(otherRaw, pnames, fnames)
        toCall = getattr(getattr(caller, 'elements'), 'multiply')
        ret = toCall(other)

        assert caller.points.getNames() == pnames
        assert caller.features.getNames() == fnames

    def test_elementwiseMultipy_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, 'elements', False, 'multiply')


    ############
    # __imul__ #
    ############

    @raises(ImproperObjectAction)
    def test_imul_selfNotNumericException(self):
        """ Test __imul__ raises exception if self has non numeric data """
        back_selfNotNumericException(self.constructor, self.constructor, '__imul__')

    @raises(InvalidArgumentValue)
    def test_imul_otherNotNumericException(self):
        """ Test __imul__ raises exception if param object has non numeric data """
        back_otherNotNumericException(self.constructor, self.constructor, '__imul__')

    @raises(InvalidArgumentValue)
    def test_imul_shapeException(self):
        """ Test __imul__ raises exception the shapes of the object don't fit correctly """
        data1 = [[1, 2], [4, 5], [7, 8]]
        data2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        caller = self.constructor(data1)
        callee = self.constructor(data2)

        caller.__imul__(callee)

    @raises(ImproperObjectAction)
    def test_imul_pEmptyException(self):
        """ Test __imul__ raises exception for point empty data """
        data = []
        fnames = ['one', 'two']
        caller = self.constructor(data, featureNames=fnames)
        callee = caller.copy()
        callee.transpose()

        caller *= callee

    @raises(ImproperObjectAction)
    def test_imul_fEmptyException(self):
        """ Test __imul__ raises exception for feature empty data """
        data = [[], []]
        pnames = ['one', 'two']
        caller = self.constructor(data, pointNames=pnames)
        callee = caller.copy()
        callee.transpose()

        caller *= callee

    @noLogEntryExpected
    def test_imul_autoObjs(self):
        """ Test __imul__ against automated data """
        back_autoVsNumpyObjCallee(self.constructor, numpy.dot, '__imul__', True, 0.2)

    @noLogEntryExpected
    def test_imul_autoScalar(self):
        """ Test __imul__ of a scalar against automated data """
        back_autoVsNumpyScalar(self.constructor, numpy.dot, '__imul__', True, 0.2)

    def test_imul__autoVsNumpyObjCalleeDiffTypes(self):
        """ Test __mul__ against generated data with different nimble types of objects """
        back_autoVsNumpyObjCalleeDiffTypes(self.constructor, numpy.dot, '__mul__', False, 0.2)

    def test_imul_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __imul__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__imul__', True)

    def test_imul_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__imul__')

    def test_imul_matrixmul_pfname_preservations(self):
        """ Test p/f names are preserved when calling __imul__ with obj arg"""
        back_matrixmul_pfname_preservations(self.constructor, '__imul__', True)

    def test_imul_matrixmul_NamePath_preservations(self):
        back_binaryelementwise_NamePath_preservations(self.constructor, '__imul__', True)


    ############
    # __iadd__ #
    ############
    @noLogEntryExpected
    def test_iadd_fullSuite(self):
        """ __iadd__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backend(self.constructor, numpy.add, '__iadd__', True, 0.2)

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
        run_full_backend(self.constructor, numpy.subtract, '__isub__', True, 0.2)

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


    ################
    # __itruediv__ #
    ################
    @noLogEntryExpected
    def test_itruediv_fullSuite(self):
        """ __itruediv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, numpy.true_divide, '__itruediv__', True, 0)

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


    ################
    # __ifloordiv__ #
    ################
    @noLogEntryExpected
    def test_ifloordiv_fullSuite(self):
        """ __ifloordiv__ Run the full standardized suite of tests for a binary numeric op """
        run_full_backendDivMod(self.constructor, numpy.floor_divide, '__ifloordiv__', True, 0)

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
        run_full_backendDivMod(self.constructor, numpy.mod, '__imod__', True, 0)


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
    def test_ipow_exceptions(self):
        """ __ipow__ Run the full standardized suite of tests for a binary numeric op """
        constructor = self.constructor
        nimbleOp = '__ipow__'
        inputs = (constructor, nimbleOp)
        wrapAndCall(back_otherObjectExceptions, InvalidArgumentType, *inputs)

        inputs = (constructor, int, nimbleOp)
        wrapAndCall(back_selfNotNumericException, ImproperObjectAction, *inputs)

        inputs = (constructor, constructor, nimbleOp)
        wrapAndCall(back_pEmptyException, ImproperObjectAction, *inputs)

        inputs = (constructor, constructor, nimbleOp)
        wrapAndCall(back_fEmptyException, ImproperObjectAction, *inputs)

    def test_ipow_autoVsNumpyScalar(self):
        """ Test __ipow__ with automated data and a scalar argument, against numpy operations """
        trials = 5
        for t in range(trials):
            n = pythonRandom.randint(1, 15)
            scalar = pythonRandom.randint(0, 5)

            datas = makeAllData(self.constructor, None, n, .02)
            (lhsf, rhsf, lhsi, rhsi, lhsfObj, rhsfObj, lhsiObj, rhsiObj) = datas

            resultf = lhsf ** scalar
            resulti = lhsi ** scalar
            resfObj = lhsfObj.__ipow__(scalar)
            resiObj = lhsiObj.__ipow__(scalar)

            expfObj = self.constructor(resultf)
            expiObj = self.constructor(resulti)

            assert expfObj.isApproximatelyEqual(resfObj)
            assert expiObj.isIdentical(resiObj)
            assert resfObj.isIdentical(lhsfObj)
            assert resiObj.isIdentical(lhsiObj)

    def test_ipow_binaryscalar_pfname_preservations(self):
        """ Test p/f names are preserved when calling __ipow__ with scalar arg"""
        back_binaryscalar_pfname_preservations(self.constructor, '__ipow__', True)

    def test_ipow_binaryscalar_NamePath_preservations(self):
        back_binaryscalar_NamePath_preservations(self.constructor, '__ipow__')


class AllNumerical(NumericalDataSafe, NumericalModifying):
    pass
