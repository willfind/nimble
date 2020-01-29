"""
KNNImputation tests.
"""

import os
from unittest import mock

from nose.tools import raises

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentValue

from ..assertionHelpers import logCountAssertionFactory
from ..assertionHelpers import assertNoNamesGenerated

oneLogPerObject = logCountAssertionFactory(len(nimble.data.available))

@raises(InvalidArgumentValue)
def test_KNNImputation_exception_invalidMode():
    data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
    toTest = nimble.createData('Matrix', data)
    nimble.fillMatching('Custom.KNNImputation', match.nonNumeric, toTest,
                        k=3, mode='classify')

@oneLogPerObject
def test_KNNImputation_regression_missing():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
    expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames, useLog=False)
        expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames, useLog=False)
        ret = nimble.fillMatching('Custom.KNNImputation', match.missing, toTest,
                                  mode='regression', k=3)

        assert toTest == expTest
        assert ret is None

def test_KNNImputation_regression_nonNumeric():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, 'na', 'x'], [1, 3, 9], [2, 1, 6], [3, 2, 3], ['na', 3, 'x']]
    expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
        expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
        nimble.fillMatching('Custom.KNNImputation', match.nonNumeric, toTest,
                            mode='regression', k=3)

        assert toTest == expTest

@raises(InvalidArgumentValue)
def test_KNNImputation_regression_exception_NoSKL():
    with mock.patch('nimble.customLearners.knn_imputation.sklearn', False):
        data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
        toTest = nimble.createData('Matrix', data)
        nimble.fillMatching('Custom.KNNImputation', match.nonNumeric, toTest,
                            k=3, mode='regression')

# TODO limit points/features for fillMatching
# @oneLogEntryExpected
# def test_KNNImputation_regression_pointsLimited():
#     fNames = ['a', 'b', 'c']
#     pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
#     data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
#     kwarguments = {'n_neighbors': 3}
#     toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
#     expData = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
#     expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
#     toTest.KNNImputation(match.missing, fill.regression, points=[2, 3, 4], **kwarguments)
#     assert toTest == expTest
#
# def test_KNNImputation_regression_featuresLimited():
#     fNames = ['a', 'b', 'c']
#     pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
#     data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
#     kwarguments = {'n_neighbors': 3}
#     toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
#     expData = [[1, 2, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, None]]
#     expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
#     toTest.KNNImputation(match.missing, fill.regression, features=[1,0], **kwarguments)
#     assert toTest == expTest
#
# def test_KNNImputation_regression_pointsFeaturesLimited():
#     fNames = ['a', 'b', 'c']
#     pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
#     data = data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
#     kwarguments = {'n_neighbors': 3}
#     toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
#     expData = [[1, None, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
#     expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
#     toTest.KNNImputation(match.missing, fill.regression, points=0, features=2, **kwarguments)
#     assert toTest == expTest
#
@oneLogPerObject
def test_KNNImputation_classification_missing():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames, useLog=False)
        expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames, useLog=False)
        ret = nimble.fillMatching('Custom.KNNImputation', match.missing, toTest,
                                  mode='classification', k=3)

        assert toTest == expTest
        assert ret is None

def test_KNNImputation_classification_nonNumeric():
    fNames = ['a', 'b', 'c']
    pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
    expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
        expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
        ret = nimble.fillMatching('Custom.KNNImputation', match.nonNumeric, toTest,
                                  mode='classification', k=3)

        assert toTest == expTest

# TODO limit points/features for fillMatching
# def test_KNNImputation_classification_pointsLimited():
#     fNames = ['a', 'b', 'c']
#     pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
#     data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
#     kwarguments = {'n_neighbors': 3}
#     toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
#     expData = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
#     expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
#     toTest.KNNImputation(match.missing, fill.classification, points=[2, 3, 4], **kwarguments)
#     assert toTest == expTest
#
# def test_KNNImputation_classification_featuresLimited():
#     fNames = ['a', 'b', 'c']
#     pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
#     data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
#     kwarguments = {'n_neighbors': 3}
#     toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
#     expData = [[1, 3, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, None]]
#     expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
#     toTest.KNNImputation(match.missing, fill.classification, features=[1,0], **kwarguments)
#     assert toTest == expTest
#
# def test_KNNImputation_classification_pointsFeaturesLimited():
#     fNames = ['a', 'b', 'c']
#     pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
#     data = data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
#     kwarguments = {'n_neighbors': 3}
#     toTest = nimble.createData(t, data, pointNames=pNames, featureNames=fNames)
#     expData = [[1, None, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
#     expTest = nimble.createData(t, expData, pointNames=pNames, featureNames=fNames)
#     toTest.KNNImputation(match.missing, fill.classification, points=0, features=2, **kwarguments)
#     assert toTest == expTest

def test_KNNImputation_lazyNameGeneration():
    data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
    expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data)
        expTest = nimble.createData(t, expData)
        nimble.fillMatching('Custom.KNNImputation', match.nonNumeric, toTest,
                            k=3, mode='classification')

        assert toTest == expTest
        assertNoNamesGenerated(toTest)

def test_KNNImputation_NamePath_preservation():
    data = [[None, None, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data)

        toTest._name = "TestName"
        toTest._absPath = os.path.abspath("TestAbsPath")
        toTest._relPath = "testRelPath"

        nimble.fillMatching('Custom.KNNImputation', match.missing, toTest,
                            k=3, mode='regression')

        assert toTest.name == "TestName"
        assert toTest.absolutePath == os.path.abspath("TestAbsPath")
        assert toTest.relativePath == 'testRelPath'
