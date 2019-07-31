"""
Unit tests for shogun_interface.py

"""

from __future__ import absolute_import
from six.moves import range

import distutils
import json
import os
import inspect
import time
import multiprocessing
import signal

import numpy
from nose.tools import *
from nose.plugins.attrib import attr
try:
    from shogun import RealFeatures
    from shogun import BinaryLabels, MulticlassLabels, RegressionLabels
    from shogun import LinearKernel, GaussianKernel, EuclideanDistance
except ImportError:
    pass

import nimble
from nimble.randomness import numpyRandom
from nimble.exceptions import InvalidArgumentValue
from nimble.interfaces.shogun_interface import Shogun
from nimble.interfaces.interface_helpers import PythonSearcher
from nimble.helpers import generateClassificationData
from nimble.helpers import generateRegressionData
from nimble.helpers import generateClusteredPoints
from nimble.interfaces.shogun_interface import raiseFailedProcess

from .skipTestDecorator import SkipMissing
from ..assertionHelpers import logCountAssertionFactory
from ..assertionHelpers import noLogEntryExpected, oneLogEntryExpected

scipy = nimble.importModule('scipy.sparse')

shogunSkipDec = SkipMissing('shogun')

OUTFILE_PREFIX = 'shogunParameterManifest_v'


@shogunSkipDec
def _DISABLED_testShogunObjectsInManifest():
    shogunInt = nimble.helpers.findBestInterface('shogun')
    fullVersion = shogunInt.version()
    print (fullVersion)
    version = distutils.version.LooseVersion(fullVersion.split('_')[0]).vstring[1:]

    fileName = OUTFILE_PREFIX + version
    filePath = os.path.join(nimble.nimblePath, 'interfaces', 'metadata', fileName)

    print (filePath)

    with open(filePath, 'r') as fp:
        manifest = json.load(fp)

    hasAll = hasattr(shogunInt.shogun, '__all__')
    contents = shogunInt.shogun.__all__ if hasAll else dir(shogunInt.shogun)
    depth = 1 if hasAll else 0

    # the function here is passed as the function to determine what kinds of objects
    # count as learners. For this application, we define all objects that are instantiable as
    # 'learners' so that we can easily get a list of instantiable objects in shogun to see if
    # they are represented int he manifest. We define an instantiable object as an
    # init callable object which is not a method, function, or builtin.
    def isInstantable(x):
        hasInit = hasattr(x, '__init__')
        hasCall = hasattr(x, '__call__')
        isMethod = inspect.ismethod(x)
        isFunction = inspect.isfunction(x)
        isBuiltiin = inspect.isbuiltin(x)
        if hasInit and hasCall and not isMethod and not isFunction and not isBuiltiin:
            return True
        return False

    allObjectsSearcher = PythonSearcher(shogunInt.shogun, contents, {}, isInstantable, depth)
    allObjects = allObjectsSearcher.allLearners()

    unseen = []
    for k in manifest.keys():
        if not(k in allObjects or k[1:] in allObjects):
            unseen.append(k)

#    print (unseen)
#    print ("")

#    for objName in allObjects:
#        assert objName in manifest or 'C' + objName in manifest
#        if not(objName in manifest or 'C' + objName in manifest):
#            print (objName)

#    assert False


@shogunSkipDec
@raises(InvalidArgumentValue)
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
    data = [[0, 1, 0], [-0, 0, 1], [5, 3, 2]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables,
                                 useLog=False)

    data2 = [[3, 3], [-1, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {}
    ret = nimble.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y",
                            testX=testObj, output=None, arguments=args)

    assert ret is not None
    assert ret[0, 0] == 5
    assert ret[1, 0] == 0


@shogunSkipDec
@oneLogEntryExpected
def testShogunHandmadeBinaryClassificationWithKernel():
    """ Test shogun by calling a binary linear classifier with a kernel """
    variables = ["Y", "x1", "x2"]
    data = [[5, 1, 18], [5, 1, 13], [5, 2, 9], [5, 3, 6], [-2, 3, 15], [-2, 6, 11],
            [-2, 6, 6], [5, 6, 3], [-2, 9, 5], [5, 9, 2], [-2, 10, 10], [-2, 11, 5],
            [-2, 12, 6], [5, 13, 1], [-2, 16, 3], [5, 18, 1]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[11, 11], [0, 0]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {'solver_type': 1, 'kernel': 'GaussianKernel', 'width': 2, 'cache_size': 10}
    ret = nimble.trainAndApply("shogun.LibSVM", trainingObj, trainY="Y",
                               testX=testObj, output=None, arguments=args)

    assert ret is not None
    assert ret[0, 0] == -2
    assert ret[1, 0] == 5


@shogunSkipDec
@oneLogEntryExpected
def testShogunKNN():
    """ Test shogun by calling the KNN classifier, a distance based machine """
    variables = ["Y", "x1", "x2"]
    data = [[0, 0, 0], [0, 0, 1], [1, 8, 1], [1, 7, 1], [2, 1, 9], [2, 1, 8]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[0, -10], [10, 1], [1, 10]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {'distance': 'ManhattanMetric'}

    ret = nimble.trainAndApply("shogun.KNN", trainingObj, trainY="Y",
                               testX=testObj, output=None, arguments=args)

    assert ret is not None
    assert ret.data[0, 0] == 0
    assert ret.data[1, 0] == 1
    assert ret.data[2, 0] == 2


@shogunSkipDec
@oneLogEntryExpected
def testShogunMulticlassSVM():
    """ Test shogun by calling a multiclass classifier with a kernel """

    variables = ["Y", "x1", "x2"]
    data = [[0, 0, 0], [0, 0, 1], [1, -118, 1], [1, -117, 1], [2, 1, 191], [2, 1, 118], [3, -1000, -500]]
    trainingObj = nimble.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[0, 0], [-101, 1], [1, 101], [1, 1]]
    testObj = nimble.createData('Matrix', data2, useLog=False)

    args = {'C': .5, 'kernel': 'LinearKernel'}

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
    argsk = {'C': 1.0, 'kernel': "LinearKernel"}

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


def toCall(learner):
    return "shogun." + learner

def getLearnersByType(lType=None, ignore=[]):
    learners = nimble.listLearners('shogun')
    typeMatch = []
    for learner in learners:
        if lType is not None:
            learnerType = nimble.learnerType(toCall(learner))
            if lType == learnerType and learner not in ignore:
                typeMatch.append(learner)
        elif learner not in ignore:
            typeMatch.append(learner)
    assert typeMatch # check not returning an empty list
    return typeMatch

def equalityAssertHelper(ret1, ret2, ret3=None):
    def identicalThenApprox(lhs, rhs):
        try:
            assert lhs.isIdentical(rhs)
        except AssertionError:
            assert lhs.isApproximatelyEqual(rhs)

    identicalThenApprox(ret1, ret2)
    if ret3 is not None:
        identicalThenApprox(ret1, ret3)
        identicalThenApprox(ret2, ret3)

def addDummyLabels(number):
    for i in range(number):
        nimble.log('pass', 'pass')

@shogunSkipDec
@attr('slow')
def testShogunClassificationLearners():
    binaryData = generateClassificationData(2, 10, 20)
    multiclassData = generateClassificationData(3, 10, 20)

    learners = getLearnersByType('classification')
    remove = ['Multitask', 'KMeans'] # TODO KMeans working always different result
    learners = [l for l in learners if not any(x in l for x in remove)]

    needKernel = ['GMNPSVM', 'GNPPSVM', 'GPBTSVM', 'LaRank', 'LibSVMOneClass',
                  'MKLClassification', 'MKLMulticlass', 'MKLOneClass',
                  'MPDSVM']
    needDistance = ['Hierarchical', 'KNN']
    expectedFailures = [
        'Autoencoder', 'BalancedConditionalProbabilityTree', 'CHAIDTree',
        'DeepAutoencoder', 'DomainAdaptationMulticlassLibLinear',
        'DomainAdaptationSVMLinear', 'FeatureBlockLogisticRegression',
        'KMeansMiniBatch', 'LDA', 'LibSVM', 'MKLClassification',
        'MKLMulticlass', 'MKLOneClass', 'MulticlassLibSVM',
        'MulticlassTreeGuidedLogisticRegression', 'NeuralNetwork',
        'PluginEstimate', 'RandomConditionalProbabilityTree', 'RandomForest',
        'RelaxedTree', 'ShareBoost', 'VowpalWabbit', 'WDSVMOcas'
        ]
    varyingPredictionsPossible = ['QDA']

    @logCountAssertionFactory(2)
    def compareOutputs(learner):
        sg = Shogun()
        sgObj = sg.findCallable(learner)
        shogunObj = sgObj()
        ptVal = shogunObj.get_machine_problem_type()
        if ptVal == sg._access('Classifier', 'PT_BINARY'):
            trainX = abs(binaryData[0][0])
            trainY = abs(binaryData[0][1])
            trainY.points.fill(0, -1, useLog=False)
            testX = abs(binaryData[1][0])
            Ytrain = trainY.copy('numpy array', outputAs1D=True)
            Ytrain = BinaryLabels(Ytrain)
            sgApply = shogunObj.apply_binary
        else:
            trainX = abs(multiclassData[0][0])
            trainY = abs(multiclassData[0][1])
            testX = abs(multiclassData[1][0])
            Ytrain = trainY.copy('numpy array', outputAs1D=True)
            Ytrain = MulticlassLabels(Ytrain)
            sgApply = shogunObj.apply_multiclass
        Xtrain = RealFeatures(trainX.copy('numpy array', rowsArePoints=False))
        Xtest = RealFeatures(testX.copy('numpy array', rowsArePoints=False))
        args = {}
        if learner in needKernel:
            kernel = GaussianKernel()
            kernel.init(Xtrain, Xtrain)
            shogunObj.set_kernel(kernel)
            args['kernel'] = 'GaussianKernel'
        if learner in needDistance:
            dist = EuclideanDistance()
            dist.init(Xtrain, Xtrain)
            shogunObj.set_distance(dist)
            args['distance'] = 'EuclideanDistance'
        try:
            raiseFailedProcess(shogunObj.set_labels, Ytrain)
            shogunObj.set_labels(Ytrain)
            raiseFailedProcess(shogunObj.train, Xtrain)
            shogunObj.train(Xtrain)
            raiseFailedProcess(sgApply, Xtest)
            predLabels = sgApply(Xtest)
            predArray = predLabels.get_labels().reshape(-1, 1)
            predSG = nimble.createData('Matrix', predArray, useLog=False)
            TL = nimble.train(toCall(learner), trainX, trainY, arguments=args)
            predNimble = TL.apply(testX)
            try:
                equalityAssertHelper(predSG, predNimble)
            except AssertionError:
                assert learner in varyingPredictionsPossible
        except SystemError:
            try:
                # shogun will fail so want to check that signal is caught
                # and error is raised
                TL = nimble.train(toCall(learner), trainX, trainY,
                                  arguments=args, useLog=False)
                predNimble = TL.apply(testX, useLog=False)
                assert False # expected SystemError
            except SystemError:
                assert learner in expectedFailures
                addDummyLabels(2) # so log assertion passes

    for learner in learners:
        compareOutputs(learner)

@shogunSkipDec
@attr('slow')
def testShogunRegressionLearners():
    data = generateRegressionData(3, 10, 20)
    trainX = abs(data[0][0])
    trainY = abs(data[0][1])
    testX = abs(data[1][0])
    Xtrain = RealFeatures(trainX.copy('numpy array', rowsArePoints=False))
    Ytrain = trainY.copy('numpy array', outputAs1D=True)
    Ytrain = RegressionLabels(Ytrain)
    Xtest = RealFeatures(testX.copy('numpy array', rowsArePoints=False))

    learners = getLearnersByType('regression')
    remove = ['Multitask', "LibLinearRegression"] # LibLinearRegression strange failure
    learners = [l for l in learners if not any(x in l for x in remove)]

    needKernel = ['KRRNystrom', 'KernelRidgeRegression', 'LibSVR', 'MKLRegression']
    needDistance = []
    expectedFailures = ['GaussianProcessRegression', 'MKLRegression']
    varyingPredictionsPossible = ['KRRNystrom']

    @logCountAssertionFactory(2)
    def compareOutputs(learner):
        sg = Shogun()
        sgObj = sg.findCallable(learner)
        shogunObj = sgObj()

        args = {}
        if learner in needKernel:
            kernel = GaussianKernel()
            kernel.init(Xtrain, Xtrain)
            shogunObj.set_kernel(kernel)
            args['kernel'] = 'GaussianKernel'
        if learner in needDistance:
            dist = EuclideanDistance()
            dist.init(Xtrain, Xtrain)
            shogunObj.set_distance(dist)
            args['distance'] = 'EuclideanDistance'
        try:
            raiseFailedProcess(shogunObj.set_labels, Ytrain)
            shogunObj.set_labels(Ytrain)
            raiseFailedProcess(shogunObj.train, Xtrain)
            shogunObj.train(Xtrain)
            raiseFailedProcess(shogunObj.apply_regression, Xtest)
            predLabels = shogunObj.apply_regression(Xtest)
            predArray = predLabels.get_labels().reshape(-1, 1)
            predSG = nimble.createData('Matrix', predArray, useLog=False)
            TL = nimble.train(toCall(learner), trainX, trainY, arguments=args)
            predNimble = TL.apply(testX)
            try:
                equalityAssertHelper(predSG, predNimble)
            except AssertionError:
                assert learner in varyingPredictionsPossible
        except SystemError as se:
            try:
                # shogun will fail so want to check that signal is caught
                # and error is raised
                TL = nimble.train(toCall(learner), trainX, trainY,
                                  arguments=args, useLog=False)
                predNimble = TL.apply(testX, useLog=False)
                assert False # expected SystemError
            except SystemError:
                assert learner in expectedFailures
                addDummyLabels(2) # so log assertion passes

    for learner in learners:
        compareOutputs(learner)


@shogunSkipDec
@attr('slow')
def testShogunClusterLearners():
    data = generateClusteredPoints(3, 60, 8)
    data = data[0]
    data.points.shuffle()
    trainX = data[:50,:]
    testX = data[50:,:]
    Xtrain = trainX.data
    Xtest = testX.data
    Xtrain = RealFeatures(trainX.copy('numpy array', rowsArePoints=False))
    Xtest = RealFeatures(testX.copy('numpy array', rowsArePoints=False))

    learners = ['Hierarchical', 'KMeans', 'KMeansMiniBatch']

    needDistance = ['Hierarchical', 'KMeans', 'KMeansMiniBatch']

    @logCountAssertionFactory(2)
    def compareOutputs(learner):
        sg = Shogun()
        sgObj = sg.findCallable(learner)
        shogunObj = sgObj()

        args = {}
        if hasattr(shogunObj, 'set_batch_size'):
            shogunObj.set_batch_size(10)
            shogunObj.set_mb_iter(1)
            args['batch_size'] = 10
            args['mb_iter'] = 1
        if learner in needDistance:
            dist = EuclideanDistance()
            dist.init(Xtrain, Xtrain)
            shogunObj.set_distance(dist)
            args['distance'] = 'EuclideanDistance'
        try:
            raiseFailedProcess(shogunObj.train, Xtrain)
            shogunObj.train(Xtrain)
            raiseFailedProcess(shogunObj.apply_multiclass, Xtest)
            predLabels = shogunObj.apply_multiclass(Xtest)
            predArray = predLabels.get_labels().reshape(-1, 1)
            predSG = nimble.createData('Matrix', predArray, useLog=False)
            TL = nimble.train(toCall(learner), trainX, arguments=args)
            predNimble = TL.apply(testX)
            try:
                equalityAssertHelper(predSG, predNimble)
            except AssertionError:
                pass
                # assert learner in varyingPredictionsPossible
        except SystemError as se:
            try:
                # shogun will fail so want to check that signal is caught
                # and error is raised
                TL = nimble.train(toCall(learner), trainX, arguments=args,
                                  useLog=False)
                predNimble = TL.apply(testX, useLog=False)
                assert False # expected SystemError
            except SystemError:
                assert learner in expectedFailures
                addDummyLabels(2) # so log assertion passes

    for learner in learners:
        print(learner)
        compareOutputs(learner)


def test_raiseFailedProcess_maxTime():
    def dontSleep():
        pass

    def sleepFiveSeconds():
        time.sleep(5)

    def exitSignal():
        os.kill(os.getpid(), signal.SIGSEGV)

    def failedProcessCheck(target):
        p = multiprocessing.Process(target=target)
        p.start()
        p.join(0.1) # limit to only enough time for signal in failed process
        exitcode = p.exitcode
        p.terminate()
        if exitcode:
            raise SystemError("shogun encountered an unidentifiable error")

    # successful process that takes less than 0.1s
    start = time.time()
    failedProcessCheck(dontSleep)
    end = time.time()
    assert end - start < 0.1

    # successful process takes over 0.1s
    start = time.time()
    failedProcessCheck(sleepFiveSeconds)
    end = time.time()
    assert 0.1 < end - start < 0.2 # join(0.1) will max out

    # failed process due to segfault signal
    start = time.time()
    try:
        failedProcessCheck(exitSignal)
        assert False # expected SystemError
    except SystemError:
        pass
    end = time.time()
    assert end - start < 0.1
