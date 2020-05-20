"""
fillMatching tests.
"""

import os

from nose.tools import raises

import nimble
from nimble import match
from nimble.exceptions import ImproperObjectAction, InvalidArgumentValue

from .assertionHelpers import logCountAssertionFactory
from .assertionHelpers import assertNoNamesGenerated

def test_fillMatching_exception_nansUnmatched():
    raw = [[1, 1, 1, 0], [1, 1, 1, None], [2, 2, 2, 0], [2, 2, 2, 3],
           [2, 2, 2, 4], [2, 2, 2, 4]]
    for t in nimble.data.available:
        data = nimble.createData(t, raw)
        try:
            nimble.fillMatching('nimble.KNNImputation', 1, data, mode='classification')
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
            nimble.fillMatching('nimble.KNNImputation', 2, data, mode='classification')
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            assert data == dataCopy

@logCountAssertionFactory(len(nimble.data.available))
def backend_fillMatching(matchingElements, raw, expRaw):
    for t in nimble.data.available:
        data = nimble.createData(t, raw, useLog=False)
        exp = nimble.createData(t, expRaw, useLog=False)
        nimble.fillMatching('nimble.KNNImputation', matchingElements, data,
                            mode='classification', k=1)
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

@raises(InvalidArgumentValue)
def test_fillMatching_matchingElementsAsBooleanMatrix_exception_wrongSize():
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    obj = nimble.createData('Matrix', raw)
    # reshaped to cause failure
    matchingElements = obj.matchingElements(match.negative)[:2, :2]
    data = nimble.createData('Matrix', raw, useLog=False)
    nimble.fillMatching('nimble.KNNImputation', matchingElements, data,
                        mode='classification', k=1)

@raises(InvalidArgumentValue)
def test_KNNImputation_exception_invalidMode():
    data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
    toTest = nimble.createData('Matrix', data)
    nimble.fillMatching('nimble.KNNImputation', match.nonNumeric, toTest,
                        k=3, mode='classify')


def test_fillMatching_pointsLimited():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    expData = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
        expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
        nimble.fillMatching('nimble.KNNImputation', match.missing, toTest,
                            points=[2, 3, 4], mode='classification', k=3)
        assert toTest == expTest

@raises(InvalidArgumentValue)
def test_fillMatching_sklDisallowedArgument():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, 0, 0], [1, 3, 6], [2, 1, 6], [1, 3, 7], [0, 3, 0]]
    toTest = nimble.createData('Matrix', data, pointNames=pNames, featureNames=fNames)
    nimble.fillMatching('skl.SimpleImputer', match.zero, toTest,
                        missing_values=0)
    assert toTest == expTest

def test_fillMatching_featuresLimited():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    expData = [[1, 3, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, None]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
        expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
        nimble.fillMatching('nimble.KNNImputation', match.missing, toTest,
                            features=[1,0], mode='classification', k=3)
        assert toTest == expTest

def test_fillMatching_pointsFeaturesLimited():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    expData = [[1, None, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
        expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
        nimble.fillMatching('nimble.KNNImputation', match.missing, toTest,
                            points=0, features=2, mode='classification', k=3)
        assert toTest == expTest

def test_fillMatching_lazyNameGeneration():
    data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
    expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data)
        expTest = nimble.createData(t, expData)
        nimble.fillMatching('nimble.KNNImputation', match.nonNumeric, toTest,
                            k=3, mode='classification')

        assert toTest == expTest
        assertNoNamesGenerated(toTest)

def test_fillMatching_NamePath_preservation():
    data = [[None, None, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data)

        toTest._name = "TestName"
        toTest._absPath = os.path.abspath("TestAbsPath")
        toTest._relPath = "testRelPath"

        nimble.fillMatching('nimble.KNNImputation', match.missing, toTest,
                            k=3, mode='regression')

        assert toTest.name == "TestName"
        assert toTest.absolutePath == os.path.abspath("TestAbsPath")
        assert toTest.relativePath == 'testRelPath'
