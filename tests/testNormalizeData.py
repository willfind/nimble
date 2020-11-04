"""
Tests for the top level function nimble.normalizeData
"""

import nimble
from nimble import CustomLearner
from tests.helpers import configSafetyWrapper
from tests.helpers import logCountAssertionFactory

# successful run no testX
def test_normalizeData_successTest_noTestX():
    data = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data("Matrix", data)
    orig = trainX.copy()

    norm = nimble.normalizeData('scikitlearn.PCA', trainX, n_components=2)

    assert norm != trainX
    assert trainX == orig

# successful run trainX and testX
def test_normalizeData_successTest_BothDataSets():
    learners = ['scikitlearn.PCA', 'scikitlearn.StandardScaler']
    for learner, args in zip(learners, [{'n_components': 2}, {}]):
        data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
        ftNames = ['a', 'b', 'c']
        trainX = nimble.data("Matrix", data1, pointNames=['0', '1', '2'],
                             featureNames=ftNames)

        data2 = [[-1, 0, 5]]
        testX = nimble.data("Matrix", data2, pointNames=['4'],
                            featureNames=ftNames)

        norms = nimble.normalizeData(learner, trainX, testX=testX,
                                     arguments=args)
        assert isinstance(norms, tuple)

        normTrainX, normTestX = norms
        assert normTrainX != trainX
        assert normTestX != testX

        # pointNames should be preserved for both learners
        # featureNames not preserved when number of features changes (PCA)
        assert normTrainX.points.getNames() == trainX.points.getNames()
        if learner == 'scikitlearn.PCA':
            assert not normTrainX.features._namesCreated()
            assert len(normTrainX.features) == 2
        else:
            assert normTrainX.features.getNames() == trainX.features.getNames()

        assert normTestX.points.getNames() == testX.points.getNames()
        if learner == 'scikitlearn.PCA':
            assert not normTestX.features._namesCreated()
            assert len(normTestX.features) == 2
        else:
            assert normTestX.features.getNames() == testX.features.getNames()

# names changed
def test_normalizeData_namesChanged():
    data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data("Matrix", data1, name='trainX')

    data2 = [[-1, 0, 5]]
    testX = nimble.data("Matrix", data2, name='testX')

    norms = nimble.normalizeData('scikitlearn.PCA', trainX, testX=testX,
                                 n_components=2)

    assert norms[0].name == 'trainX PCA'
    assert norms[1].name == 'testX PCA'

@logCountAssertionFactory(2)
def test_normalizeData_logCount():
    data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    trainX = nimble.data("Matrix", data1, useLog=False)
    data2 = [[-1, 0, 5]]
    testX = nimble.data("Matrix", data2, useLog=False)

    _ = nimble.normalizeData('scikitlearn.StandardScaler', trainX, testX=testX)
    _ = nimble.normalizeData('scikitlearn.PCA', trainX, testX=testX,
                             n_components=2)
