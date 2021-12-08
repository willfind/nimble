"""
Test matrix operations
"""

import nimble
from nimble.calculate import elementwiseMultiply
from nimble.calculate import elementwisePower
from tests.helpers import noLogEntryExpected
from tests.helpers import getDataConstructors
from tests.helpers import assertCalled

def test_elementwiseMultiply_callsObjElementsMultiply():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    for lCon in getDataConstructors():
        leftObj = lCon(left)
        for rCon in getDataConstructors():
            rightObj = rCon(right)
            with assertCalled(nimble.core.data.Base, '__mul__'):
                mult = elementwiseMultiply(leftObj, rightObj)

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
    leftObj = nimble.data(left, useLog=False)
    rightObj = nimble.data(right, useLog=False)
    mult = elementwiseMultiply(leftObj, rightObj)


def test_elementwisePower_callsObjElementsMultiply():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    for lCon in getDataConstructors():
        leftObj = lCon(left)
        for rCon in getDataConstructors():
            rightObj = rCon(right)
            with assertCalled(nimble.core.data.Base, '__pow__'):
                pow = elementwisePower(leftObj, rightObj)

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
    leftObj = nimble.data(left, useLog=False)
    rightObj = nimble.data(right, useLog=False)
    mult = elementwisePower(leftObj, rightObj)
