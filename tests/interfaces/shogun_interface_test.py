"""
Unit tests for shogun_interface.py

"""

from __future__ import absolute_import
from six.moves import range

import numpy
from nose.tools import *
from nose.plugins.attrib import attr

import UML as nimble
from UML.randomness import numpyRandom
from UML.exceptions import InvalidArgumentValue
from UML.exceptions import InvalidArgumentValueCombination
from .skipTestDecorator import SkipMissing
from ..assertionHelpers import logCountAssertionFactory
from ..assertionHelpers import noLogEntryExpected, oneLogEntryExpected

scipy = nimble.importModule('scipy.sparse')

shogunSkipDec = SkipMissing('shogun')


@shogunSkipDec
@raises(InvalidArgumentValueCombination)
def testShogun_shapemismatchException():
    """ Test shogun raises exception when the shape of the train and test data don't match """
    variables = ["Y", "x1", "x2"]
    data = [[-1, 1, 0], [-1, 0, 1], [1, 3, 2]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables)

    data2 = [[3]]
    testObj = nimble.createData('Matrix', data2)

    args = {}
    ret = nimble.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


@shogunSkipDec
@raises(InvalidArgumentValue)
def testShogun_singleClassException():
    """ Test shogun raises exception when the training data only has a single label """
    variables = ["Y", "x1", "x2"]
    data = [[-1, 1, 0], [-1, 0, 1], [-1, 0, 0]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables)

    data2 = [[3, 3]]
    testObj = nimble.createData('Matrix', data2)

    args = {}
    ret = nimble.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


@shogunSkipDec
@raises(InvalidArgumentValue)
def testShogun_multiClassDataToBinaryAlg():
    """ Test shogun() raises InvalidArgumentValue when passing multiclass data to a binary classifier """
    variables = ["Y", "x1", "x2"]
    data = [[5, -11, -5], [1, 0, 1], [2, 3, 2]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables)

    data2 = [[5, 3], [-1, 0]]
    testObj = nimble.createData('Matrix', data2)

    args = {'kernel': 'GaussianKernel', 'width': 2, 'size': 10}
    # TODO -  is this failing because of kernel issues, or the thing we want to test?
    ret = nimble.trainAndApply("shogun.LibSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


@shogunSkipDec
@oneLogEntryExpected
def testShogunHandmadeBinaryClassification():
    """ Test shogun by calling a binary linear classifier """
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 0], [-0, 0, 1], [1, 3, 2]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables,
                                 useLog=False)

    data2 = [[3, 3], [-1, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {}
    ret = nimble.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y",
                            testX=testObj, output=None, arguments=args)

    assert ret is not None

    # shogun binary classifiers seem to return confidence values, not class ID
    assert ret.data[0, 0] > 0


@shogunSkipDec
@oneLogEntryExpected
def testShogunHandmadeBinaryClassificationWithKernel():
    """ Test shogun by calling a binary linear classifier with a kernel """

    variables = ["Y", "x1", "x2"]
    data = [[5, -11, -5], [1, 0, 1], [1, 3, 2]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[5, 3], [-1, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {'st': 1, 'kernel': 'GaussianKernel', 'w': 2, 'size': 10}
    ret = nimble.trainAndApply("shogun.LibSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

    assert ret is not None

    # shogun binary classifiers seem to return confidence values, not class ID
    assert ret.data[0, 0] > 0


@shogunSkipDec
@oneLogEntryExpected
def testShogunKMeans():
    """ Test shogun by calling the Kmeans classifier, a distance based machine """
    variables = ["Y", "x1", "x2"]
    data = [[0, 0, 0], [0, 0, 1], [1, 8, 1], [1, 7, 1], [2, 1, 9], [2, 1, 8]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[0, -10], [10, 1], [1, 10]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {'distance': 'ManhattanMetric'}

    ret = nimble.trainAndApply("shogun.KNN", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

    assert ret is not None

    assert ret.data[0, 0] == 0
    assert ret.data[1, 0] == 1
    assert ret.data[2, 0] == 2


@shogunSkipDec
@oneLogEntryExpected
def testShogunMulticlassSVM():
    """ Test shogun by calling a multilass classifier with a kernel """

    variables = ["Y", "x1", "x2"]
    data = [[0, 0, 0], [0, 0, 1], [1, -118, 1], [1, -117, 1], [2, 1, 191], [2, 1, 118], [3, -1000, -500]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[0, 0], [-101, 1], [1, 101], [1, 1]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {'C': .5, 'k': 'LinearKernel'}

    #	args = {'C':1}
    #	args = {}
    ret = nimble.trainAndApply("shogun.GMNPSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

    assert ret is not None

    assert ret.data[0, 0] == 0
    assert ret.data[1, 0] == 1
    assert ret.data[2, 0] == 2


@shogunSkipDec
@oneLogEntryExpected
def testShogunSparseRegression():
    """ Test shogun sparse data instantiation by calling on a sparse regression learner with a large, but highly sparse, matrix """
    if not scipy:
        return
    x = 100
    c = 10
    points = numpyRandom.randint(0, x, c)
    cols = numpyRandom.randint(0, x, c)
    data = numpyRandom.rand(c)
    A = scipy.sparse.coo_matrix((data, (points, cols)), shape=(x, x))
    obj = nimble.createData('Sparse', A, useLog=False)

    labelsData = numpyRandom.rand(x)
    labels = nimble.createData('Matrix', labelsData.reshape((x, 1)), useLog=False)

    ret = nimble.trainAndApply('shogun.MulticlassOCAS', trainX=obj, trainY=labels, testX=obj, max_train_time=10)

    assert ret is not None


@shogunSkipDec
@logCountAssertionFactory(4)
def testShogunRossData():
    """ Test shogun by calling classifers using the problematic data from Ross """

    p0 = [1, 0, 0, 0, 0.21, 0.12]
    p1 = [2, 0, 0.56, 0.77, 0, 0]
    p2 = [1, 0.24, 0, 0, 0.12, 0]
    p3 = [1, 0, 0, 0, 0, 0.33]
    p4 = [2, 0.55, 0, 0.67, 0.98, 0]
    p5 = [1, 0, 0, 0, 0.21, 0.12]
    p6 = [2, 0, 0.56, 0.77, 0, 0]
    p7 = [1, 0.24, 0, 0, 0.12, 0]

    data = [p0, p1, p2, p3, p4, p5, p6, p7]

    trainingObj = nimble.createData('Matrix', data, useLog=False)

    data2 = [[0, 0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {'C': 1.0}
    argsk = {'C': 1.0, 'k': "LinearKernel"}

    ret = nimble.trainAndApply("shogun.MulticlassLibSVM", trainingObj, trainY=0, testX=testObj, output=None,
                            arguments=argsk)
    assert ret is not None

    ret = nimble.trainAndApply("shogun.MulticlassLibLinear", trainingObj, trainY=0, testX=testObj, output=None,
                            arguments=args)
    assert ret is not None

    ret = nimble.trainAndApply("shogun.LaRank", trainingObj, trainY=0, testX=testObj, output=None, arguments=argsk)
    assert ret is not None

    ret = nimble.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY=0, testX=testObj, output=None, arguments=args)
    assert ret is not None


@shogunSkipDec
@attr('slow')
@oneLogEntryExpected
def testShogunEmbeddedRossData():
    """ Test shogun by MulticlassOCAS with the ross data embedded in random data """

    p0 = [3, 0, 0, 0, 0.21, 0.12]
    p1 = [2, 0, 0.56, 0.77, 0, 0]
    p2 = [3, 0.24, 0, 0, 0.12, 0]
    p3 = [3, 0, 0, 0, 0, 0.33]
    p4 = [2, 0.55, 0, 0.67, 0.98, 0]
    p5 = [3, 0, 0, 0, 0.21, 0.12]
    p6 = [2, 0, 0.56, 0.77, 0, 0]
    p7 = [3, 0.24, 0, 0, 0.12, 0]

    data = [p0, p1, p2, p3, p4, p5, p6, p7]

    numpyData = numpy.zeros((50, 10))

    for i in range(50):
        for j in range(10):
            if i < 8 and j < 6:
                numpyData[i, j] = data[i][j]
            else:
                if j == 0:
                    numpyData[i, j] = numpyRandom.randint(2, 3)
                else:
                    numpyData[i, j] = numpyRandom.rand()

    trainingObj = nimble.createData('Matrix', numpyData, useLog=False)

    data2 = [[0, 0, 0, 0, 0.33, 0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98, 0.55, 0, 0.67, 0.98, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {'C': 1.0}

    ret = nimble.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY=0, testX=testObj, output=None, arguments=args)
    assert ret is not None

    for value in ret.data:
        assert value == 2 or value == 3


@shogunSkipDec
@logCountAssertionFactory(3)
def testShogunScoreModeMulti():
    """ Test shogun returns the right dimensions when given different scoreMode flags, multi case"""
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 30, 20], [2, -300, 2]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    # default scoreMode is 'label'
    ret = nimble.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY="Y", testX=testObj, arguments={})
    assert len(ret.points) == 2
    assert len(ret.features) == 1

    ret = nimble.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY="Y", testX=testObj, arguments={},
                            scoreMode='bestScore')
    assert len(ret.points) == 2
    assert len(ret.features) == 2

    ret = nimble.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY="Y", testX=testObj, arguments={},
                            scoreMode='allScores')
    assert len(ret.points) == 2
    assert len(ret.features) == 3


@shogunSkipDec
@logCountAssertionFactory(3)
def testShogunScoreModeBinary():
    """ Test shogun returns the right dimensions when given different scoreMode flags, binary case"""
    variables = ["Y", "x1", "x2"]
    data = [[-1, 1, 1], [-1, 0, 1], [1, 30, 2], [1, 30, 3]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 1], [25, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    # default scoreMode is 'label'
    ret = nimble.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={})
    assert len(ret.points) == 2
    assert len(ret.features) == 1

    ret = nimble.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={},
                            scoreMode='bestScore')
    assert len(ret.points) == 2
    assert len(ret.features) == 2

    ret = nimble.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={},
                            scoreMode='allScores')
    assert len(ret.points) == 2
    assert len(ret.features) == 2

@shogunSkipDec
@logCountAssertionFactory(2)
def TODO_onlineLearneres():
    """ Test shogun can call online learners """
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [0, 3, 2], [1, -300, -25]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    ret = nimble.trainAndApply("shogun.OnlineLibLinear", trainingObj, trainY="Y", testX=testObj, arguments={})
    ret = nimble.trainAndApply("shogun.OnlineSVMSGD", trainingObj, trainY="Y", testX=testObj, arguments={})

@shogunSkipDec
@logCountAssertionFactory(1)
def TODO_ShogunMultiClassStrategyMultiDataBinaryAlg():
    """ Test shogun will correctly apply the provided strategies when given multiclass data and a binary learner """
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    ret = nimble.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={},
                            multiClassStrategy="OneVsOne")


@shogunSkipDec
@attr('slow')
@noLogEntryExpected
def testShogunListLearners():
    """ Test shogun's listShogunLearners() by checking the output for those learners we unit test """

    ret = nimble.listLearners('shogun')

    assert 'LibLinear' in ret
    assert 'KNN' in ret
    assert 'GMNPSVM' in ret
    assert 'LaRank' in ret
    assert 'MulticlassOCAS' in ret
    assert "SVMOcas" in ret
    assert 'LibSVM' in ret
    assert 'MulticlassLibSVM' in ret

    for name in ret:
        params = nimble.learnerParameters('shogun.' + name)
        assert params is not None
        defaults = nimble.learnerDefaultValues('shogun.' + name)
        for pSet in params:
            for dSet in defaults:
                for key in dSet.keys():
                    assert key in pSet
