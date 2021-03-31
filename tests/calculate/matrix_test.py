"""
Test matrix operations
"""

from unittest.mock import patch

from nose.tools import raises

import nimble
from nimble.calculate import elementwiseMultiply
from nimble.calculate import elementwisePower
from nimble.exceptions import InvalidArgumentType
from tests.helpers import noLogEntryExpected
from tests.helpers import CalledFunctionException, calledException
from tests.helpers import getDataConstructors

@patch('nimble.core.data.Base.__mul__', calledException)
def test_elementwiseMultiply_callsObjElementsMultiply():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    for lCon in getDataConstructors():
        leftObj = lCon(left)
        for rCon in getDataConstructors():
            rightObj = rCon(right)
            try:
                mult = elementwiseMultiply(leftObj, rightObj)
                assert False # expected CalledFunctionException
            except CalledFunctionException:
                pass

def test_elementwiseMultiply():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    exp = [[6, 10, 12], [12, 10, 6]]
    for lCon in getDataConstructors():
        leftObj = lCon(left)
        origLeft = leftObj.copy()
        expObj = lCon(exp)
        for rCon in getDataConstructors():
            rightObj = rCon(right)
            origRight = rightObj.copy()
            mult = elementwiseMultiply(leftObj, rightObj)
            assert mult.isIdentical(expObj)
            assert leftObj.isIdentical(origLeft)
            assert rightObj.isIdentical(origRight)

@noLogEntryExpected
def test_elementwiseMultiply_logCount():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    leftObj = nimble.data('Matrix', left, useLog=False)
    rightObj = nimble.data('Matrix', right, useLog=False)
    mult = elementwiseMultiply(leftObj, rightObj)


@patch('nimble.core.data.Base.__pow__', calledException)
def test_elementwisePower_callsObjElementsMultiply():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    for lCon in getDataConstructors():
        leftObj = lCon(left)
        for rCon in getDataConstructors():
            rightObj = rCon(right)
            try:
                pow = elementwisePower(leftObj, rightObj)
                assert False # expected CalledFunctionException
            except CalledFunctionException:
                pass

def test_elementwisePower():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[1, 2, 3], [3, 2, 1]]
    exp = [[1, 4, 27], [64, 25, 6]]
    for lCon in getDataConstructors():
        leftObj = lCon(left)
        origLeft = leftObj.copy()
        expObj = lCon(exp)
        for rCon in getDataConstructors():
            rightObj = rCon(right)
            origRight = rightObj.copy()
            pow = elementwisePower(leftObj, rightObj)
            assert pow.isIdentical(expObj)
            assert leftObj.isIdentical(origLeft)
            assert rightObj.isIdentical(origRight)

@noLogEntryExpected
def test_elementwisePower_logCount():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    leftObj = nimble.data('Matrix', left, useLog=False)
    rightObj = nimble.data('Matrix', right, useLog=False)
    mult = elementwisePower(leftObj, rightObj)
