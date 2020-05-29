"""
Tests for the top level function nimble.normalizeData
"""

import nimble
from nimble import CustomLearner
from tests.helpers import configSafetyWrapper
from tests.helpers import oneLogEntryExpected

# successful run no testX
def test_normalizeData_successTest_noTestX():
    data = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data("Matrix", data)
    orig = trainX.copy()

    nimble.normalizeData('scikitlearn.PCA', trainX, n_components=2)

    assert trainX != orig

# successful run trainX and testX
def test_normalizeData_successTest_BothDataSets():
    data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data("Matrix", data1)
    orig1 = trainX.copy()

    data2 = [[-1, 0, 5]]
    testX = nimble.data("Matrix", data2)
    orig2 = testX.copy()

    nimble.normalizeData('scikitlearn.PCA', trainX, testX=testX, n_components=2)

    assert trainX != orig1
    assert testX != orig2

# names changed
def test_normalizeData_namesChanged():
    data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data("Matrix", data1, name='trainX')

    data2 = [[-1, 0, 5]]
    testX = nimble.data("Matrix", data2, name='testX')

    nimble.normalizeData('scikitlearn.PCA', trainX, testX=testX, n_components=2)

    assert trainX.name == 'trainX PCA'
    assert testX.name == 'testX PCA'


# referenceData safety
@configSafetyWrapper
def test_mormalizeData_referenceDataSafety():
    class ListOutputer(CustomLearner):
        learnerType = 'unknown'

        def train(self, trainX, trainY):
            pass

        def apply(self, testX):
            return testX.copy(to='List')

    data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data("Matrix", data1, name='trainX')

    data2 = [[-1, 0, 5]]
    testX = nimble.data("Matrix", data2, name='testX')

    nimble.normalizeData(ListOutputer, trainX, testX=testX)

@oneLogEntryExpected
def test_normalizeData_logCount():
    data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data("Matrix", data1, useLog=False)
    data2 = [[-1, 0, 5]]
    testX = nimble.data("Matrix", data2, useLog=False)

    nimble.normalizeData('scikitlearn.PCA', trainX, testX=testX, n_components=2)

