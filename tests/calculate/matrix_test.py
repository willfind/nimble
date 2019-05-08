"""
Test matrix operations
"""

from unittest.mock import patch

from nose.tools import raises

import UML as nimble
from UML.calculate import elementwiseMultiply
from UML.calculate import elementwisePower
from UML.exceptions import InvalidArgumentType
from ..assertionHelpers import noLogEntryExpected

class CalledFunctionException(Exception):
    pass

def functionCalled(*args, **kwargs):
    raise CalledFunctionException()

@patch('UML.data.Elements.multiply', side_effect=functionCalled)
def test_elementwiseMultiply_callsObjElementsMultiply(mockObj):
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    for leftRType in nimble.data.available:
        leftObj = nimble.createData(leftRType, left)
        for rightRType in nimble.data.available:
            rightObj = nimble.createData(rightRType, right)
            try:
                mult = elementwiseMultiply(leftObj, rightObj)
                assert False # expected CalledFunctionException
            except CalledFunctionException:
                pass

def test_elementwiseMultiply():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    exp = [[6, 10, 12], [12, 10, 6]]
    for leftRType in nimble.data.available:
        leftObj = nimble.createData(leftRType, left)
        origLeft = leftObj.copy()
        expObj = nimble.createData(leftRType, exp)
        for rightRType in nimble.data.available:
            rightObj = nimble.createData(rightRType, right)
            origRight = rightObj.copy()
            mult = elementwiseMultiply(leftObj, rightObj)
            assert mult.isIdentical(expObj)
            assert leftObj.isIdentical(origLeft)
            assert rightObj.isIdentical(origRight)

@noLogEntryExpected
def test_elementwiseMultiply_logCount():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    leftObj = nimble.createData('Matrix', left, useLog=False)
    rightObj = nimble.createData('Matrix', right, useLog=False)
    mult = elementwiseMultiply(leftObj, rightObj)


@patch('UML.data.Elements.power', side_effect=functionCalled)
def test_elementwisePower_callsObjElementsMultiply(mockObj):
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    for leftRType in nimble.data.available:
        leftObj = nimble.createData(leftRType, left)
        for rightRType in nimble.data.available:
            rightObj = nimble.createData(rightRType, right)
            try:
                pow = elementwisePower(leftObj, rightObj)
                assert False # expected CalledFunctionException
            except CalledFunctionException:
                pass

def test_elementwisePower():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[1, 2, 3], [3, 2, 1]]
    exp = [[1, 4, 27], [64, 25, 6]]
    for leftRType in nimble.data.available:
        leftObj = nimble.createData(leftRType, left)
        origLeft = leftObj.copy()
        expObj = nimble.createData(leftRType, exp)
        for rightRType in nimble.data.available:
            rightObj = nimble.createData(rightRType, right)
            origRight = rightObj.copy()
            pow = elementwisePower(leftObj, rightObj)
            assert pow.isIdentical(expObj)
            assert leftObj.isIdentical(origLeft)
            assert rightObj.isIdentical(origRight)

@noLogEntryExpected
def test_elementwisePower_logCount():
    left = [[1, 2, 3], [4, 5, 6]]
    right = [[6, 5, 4], [3, 2, 1]]
    leftObj = nimble.createData('Matrix', left, useLog=False)
    rightObj = nimble.createData('Matrix', right, useLog=False)
    mult = elementwisePower(leftObj, rightObj)
