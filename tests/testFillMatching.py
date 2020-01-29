"""
fillMatching tests.
"""

import nimble
from nimble import match
from nimble.exceptions import ImproperObjectAction, InvalidArgumentValue

from .assertionHelpers import logCountAssertionFactory

def test_fillMatching_exception_nansUnmatched():
    raw = [[1, 1, 1, 0], [1, 1, 1, None], [2, 2, 2, 0], [2, 2, 2, 3],
           [2, 2, 2, 4], [2, 2, 2, 4]]
    for t in nimble.data.available:
        data = nimble.createData(t, raw)
        try:
            nimble.fillMatching('Custom.KNNImputation', 1, data)
            assert False # expected ImproperObjectAction
        except ImproperObjectAction:
            pass

def test_fillMatching_trainXUnaffectedByFailure():
    raw = [[2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 0], [2, 2, 2, 3],
           [2, 2, 2, 4], [2, 2, 2, 4]]
    for t in nimble.data.available:
        data = nimble.createData(t, raw)
        dataCopy = data.copy()
        # trying to fill 2 will fail because the training data will be empty
        try:
            nimble.fillMatching('Custom.KNNImputation', 2, data)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            assert data == dataCopy

@logCountAssertionFactory(len(nimble.data.available))
def backend_fillMatching(matchingElements, raw, expRaw):
    for t in nimble.data.available:
        data = nimble.createData(t, raw, useLog=False)
        exp = nimble.createData(t, expRaw, useLog=False)
        nimble.fillMatching('Custom.KNNImputation', matchingElements, data, k=1)
        assert data == exp

def test_fillMatching_matchingElementsAsSingleValue():
    matchingElements = 0
    raw = [[1, 1, 1, 0], [1, 1, 1, 1], [2, 2, 2, 0], [2, 2, 2, 4]]
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)

def test_fillMatching_matchingElementsAsList():
    matchingElements = [-1, -2]
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)

def test_fillMatching_matchingElementsAsFunction():
    matchingElements = match.negative
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)

def test_fillMatching_matchingElementsAsBooleanMatrix():
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    obj = nimble.createData('Matrix', raw)
    matchingElements = obj.matchingElements(match.negative)
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)
