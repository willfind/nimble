"""
Unit tests for shogun_interface.py
"""

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
except ImportError:
    pass

import nimble
from nimble import Init
from nimble.random import numpyRandom
from nimble.random import _startAlternateControl, _endAlternateControl
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.core.interfaces.interface_helpers import PythonSearcher
from nimble.core._learnHelpers import generateClusteredPoints
from nimble.core.interfaces.shogun_interface import checkProcessFailure
from nimble.utility import scipy

from .skipTestDecorator import SkipMissing
from tests.helpers import logCountAssertionFactory
from tests.helpers import noLogEntryExpected, oneLogEntryExpected
from tests.helpers import generateClassificationData
from tests.helpers import generateRegressionData

shogunSkipDec = SkipMissing('shogun')

@shogunSkipDec
def test_Shogun_findCallable_nameAndDocPreservation():
    shogunInt = nimble.core._learnHelpers.findBestInterface('shogun')
    swigObj = shogunInt.findCallable('LibSVM')
    wrappedShogun = swigObj()
    assert 'WrappedShogun' in str(type(wrappedShogun))
    assert wrappedShogun.__name__ == 'LibSVM'
    assert wrappedShogun.__doc__


@shogunSkipDec
@raises(InvalidArgumentValueCombination)
def testShogun_shapemismatchException():
    """ Test shogun raises exception when the shape of the train and test data don't match """
    variables = ["Y", "x1", "x2"]
    data = [[-1, 1, 0], [-1, 0, 1], [1, 3, 2]]
    trainingObj = nimble.data('Matrix', data, featureNames=variables)

    data2 = [[3]]
    testObj = nimble.data('Matrix', data2)

    args = {}
    ret = nimble.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


@shogunSkipDec
@raises(InvalidArgumentValue)
def testShogun_singleClassException():
    """ Test shogun raises exception when the training data only has a single label """
    variables = ["Y", "x1", "x2"]
    data = [[-1, 1, 0], [-1, 0, 1], [-1, 0, 0]]
    trainingObj = nimble.data('Matrix', data, featureNames=variables)

    data2 = [[3, 3]]
    testObj = nimble.data('Matrix', data2)

    args = {}
    ret = nimble.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


@shogunSkipDec
@raises(InvalidArgumentValue)
def testShogun_multiClassDataToBinaryAlg():
    """ Test shogun() raises InvalidArgumentValue when passing multiclass data to a binary classifier """
    variables = ["Y", "x1", "x2"]
    data = [[5, -11, -5], [1, 0, 1], [2, 3, 2]]
    trainingObj = nimble.data('Matrix', data, featureNames=variables)

    data2 = [[5, 3], [-1, 0]]
    testObj = nimble.data('Matrix', data2)

    args = {'kernel': Init('GaussianKernel', width=2, size=10)}
    # TODO -  is this failing because of kernel issues, or the thing we want to test?
    ret = nimble.trainAndApply("shogun.LibSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


@shogunSkipDec
@logCountAssertionFactory(2)
def testShogunHandmadeBinaryClassification():
    """ Test shogun by calling a binary linear classifier """
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 0], [-0, 0, 1], [5, 3, 2]]
    trainingObj = nimble.data('Matrix', data, featureNames=variables,
                              useLog=False)

    data2 = [[3, 3], [-1, 0]]
    testObj = nimble.data('Matrix', data2, useLog=False)

    from shogun import LibLinear
    for value in ["shogun.LibLinear", LibLinear]:
        ret = nimble.trainAndApply(value, trainingObj, trainY="Y",
                                testX=testObj, output=None)

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
    trainingObj = nimble.data('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[11, 11], [0, 0]]
    testObj = nimble.data('Matrix', data2, useLog=False)

    args = {'solver_type': 1, 'kernel': Init('GaussianKernel', width=2, cache_size=10)}
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
    trainingObj = nimble.data('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[0, -10], [10, 1], [1, 10]]
    testObj = nimble.data('Matrix', data2, useLog=False)

    args = {'distance': Init('ManhattanMetric')}

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
    trainingObj = nimble.data('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[0, 0], [-101, 1], [1, 101], [1, 1]]
    testObj = nimble.data('Matrix', data2, useLog=False)

    args = {'C': .5, 'kernel': Init('LinearKernel')}

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
    x = 100
    c = 10
    points = numpyRandom.randint(0, x, c)
    cols = numpyRandom.randint(0, x, c)
    data = numpyRandom.rand(c)
    A = scipy.sparse.coo_matrix((data, (points, cols)), shape=(x, x))
    obj = nimble.data('Sparse', A, useLog=False)

    labelsData = numpyRandom.rand(x)
    labels = nimble.data('Matrix', labelsData.reshape((x, 1)), useLog=False)

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

    trainingObj = nimble.data('Matrix', data, useLog=False)

    data2 = [[0, 0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98, 0]]
    testObj = nimble.data('Matrix', data2, useLog=False)

    args = {'C': 1.0}
    argsk = {'C': 1.0, 'kernel': Init("LinearKernel")}

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

    trainingObj = nimble.data('Matrix', numpyData, useLog=False)

    data2 = [[0, 0, 0, 0, 0.33, 0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98, 0.55, 0, 0.67, 0.98, 0]]
    testObj = nimble.data('Matrix', data2, useLog=False)

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
    trainingObj = nimble.data('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = nimble.data('Matrix', data2, useLog=False)

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
    trainingObj = nimble.data('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 1], [25, 0]]
    testObj = nimble.data('Matrix', data2, useLog=False)

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
    trainingObj = nimble.data('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = nimble.data('Matrix', data2, useLog=False)

    ret = nimble.trainAndApply("shogun.OnlineLibLinear", trainingObj, trainY="Y", testX=testObj, arguments={})
    ret = nimble.trainAndApply("shogun.OnlineSVMSGD", trainingObj, trainY="Y", testX=testObj, arguments={})

@shogunSkipDec
@logCountAssertionFactory(1)
def TODO_ShogunMultiClassStrategyMultiDataBinaryAlg():
    """ Test shogun will correctly apply the provided strategies when given multiclass data and a binary learner """
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2]]
    trainingObj = nimble.data('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = nimble.data('Matrix', data2, useLog=False)

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
        for key in defaults.keys():
            assert key in params


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

def shogunTrainBackend(learner, data, toSet):
    Xtrain, Ytrain = data
    sg = nimble.core._learnHelpers.findBestInterface('shogun')
    sgObj = sg.findCallable(learner)
    shogunObj = sgObj()
    args = {}
    for arg, val in toSet.items():
        if isinstance(val, Init):
            allParams = sg._getParameterNames(val.name)[0]
            if 'labels' in allParams:
                val.kwargs['labels'] = Ytrain
            if 'features' in allParams:
                val.kwargs['features'] = Xtrain
            # val is an argument to instantiate
            val = sg._argumentInit(val)
        getattr(shogunObj, 'set_' + arg)(val)
        args[arg] = val
    if Ytrain is not None:
        checkProcessFailure('labels', shogunObj.set_labels, Ytrain)
        shogunObj.set_labels(Ytrain)
    checkProcessFailure('train', shogunObj.train, Xtrain)
    shogunObj.train(Xtrain)
    return shogunObj, args

def shogunApplyBackend(obj, toTest, applier):
    applyFunc = getattr(obj, applier)
    checkProcessFailure('apply', applyFunc, toTest)
    predLabels = applyFunc(toTest)
    predArray = predLabels.get_labels().reshape(-1, 1)
    predSG = nimble.data('Matrix', predArray, useLog=False)
    return predSG

@with_setup(_startAlternateControl, _endAlternateControl)
def trainAndApplyBackend(learner, data, applier, needKernel, needDistance,
                         extraTrainSetup):
    seed = nimble.random._generateSubsidiarySeed()
    nimble.random.setSeed(seed, useLog=False)
    trainX, trainY, testX = data[:3]
    shogunTraining = data[3:5]
    shogunTesting = data[5]
    toSet = {}
    if learner in needKernel:
        toSet['kernel'] = Init('GaussianKernel')
    if learner in needDistance:
        toSet['distance'] = Init('EuclideanDistance')
    trainShogun = data[3:5]
    if learner in extraTrainSetup:
        shogunObj, args = extraTrainSetup[learner](shogunTraining, toSet)
    else:
        shogunObj, args = shogunTrainBackend(learner, shogunTraining, toSet)
    testShogun = data[5]
    predSG = shogunApplyBackend(shogunObj, shogunTesting, applier)

    nimble.random.setSeed(seed, useLog=False)
    TL = nimble.train(toCall(learner), trainX, trainY, arguments=args)
    predNimble = TL.apply(testX)

    equalityAssertHelper(predSG, predNimble)

@shogunSkipDec
@attr('slow')
def testShogunClassificationLearners():
    binaryData = generateClassificationData(2, 20, 20)
    multiclassData = generateClassificationData(3, 20, 20)
    clusterData = generateClusteredPoints(3, 60, 8)[0]

    ignore = [ # learners that fail with provided parameters and data
        'Autoencoder', 'BalancedConditionalProbabilityTree', 'DeepAutoencoder',
        'DomainAdaptationMulticlassLibLinear', 'DomainAdaptationSVMLinear',
        'FeatureBlockLogisticRegression', 'MulticlassLibSVM',
        'MulticlassTreeGuidedLogisticRegression', 'NeuralNetwork',
        'PluginEstimate', 'RandomConditionalProbabilityTree', 'ShareBoost',
        'VowpalWabbit', 'WDSVMOcas',
        # tested elsewhere
        'GaussianProcessClassification']
    learners = getLearnersByType('classification', ignore)
    remove = ['Machine', 'Base']
    learners = [l for l in learners if not any(x in l for x in remove)]

    cluster = ['KMeans', 'KMeansMiniBatch']
    needKernel = ['GMNPSVM', 'GNPPSVM', 'GPBTSVM', 'LaRank', 'LibSVM',
                  'LibSVMOneClass', 'MPDSVM', 'RelaxedTree']
    needDistance = ['Hierarchical', 'KMeans', 'KMeansMiniBatch', 'KNN']
    extraTrainSetup = {'CHAIDTree': trainCHAIDTree,
                       'KMeansMiniBatch': trainKMeansMiniBatch,
                       'RandomForest': trainRandomForestClassifier,
                       'RelaxedTree': trainRelaxedTree}

    @logCountAssertionFactory(2)
    def compareOutputs(learner):
        sg = nimble.core._learnHelpers.findBestInterface('shogun')
        sgObj = sg.findCallable(learner)
        shogunObj = sgObj()
        ptVal = shogunObj.get_machine_problem_type()
        if learner in cluster:
            trainX = clusterData[:50,:]
            trainY = None
            testX = clusterData[50:,:]
            Ytrain = None
            sgApply = 'apply_multiclass'
        elif ptVal == sg._access('Classifier', 'PT_BINARY'):
            trainX = abs(binaryData[0][0])
            trainY = abs(binaryData[0][1])
            trainY.points.fillMatching(-1, 0, useLog=False)
            testX = abs(binaryData[1][0])
            Ytrain = trainY.copy('numpy array', outputAs1D=True)
            Ytrain = BinaryLabels(Ytrain)
            sgApply = 'apply_binary'
        else:
            trainX = abs(multiclassData[0][0])
            trainY = abs(multiclassData[0][1])
            testX = abs(multiclassData[1][0])
            Ytrain = trainY.copy('numpy array', outputAs1D=True)
            Ytrain = MulticlassLabels(Ytrain)
            sgApply = 'apply_multiclass'
        Xtrain = RealFeatures(trainX.copy('numpy array', rowsArePoints=False))
        Xtest = RealFeatures(testX.copy('numpy array', rowsArePoints=False))
        data = (trainX, trainY, testX, Xtrain, Ytrain, Xtest)

        trainAndApplyBackend(learner, data, sgApply, needKernel, needDistance,
                              extraTrainSetup)

    for learner in learners:
        compareOutputs(learner)

def trainCHAIDTree(data, toSet):
    ft = numpy.array([1] * 20, dtype='int32')
    toSet['feature_types'] = ft
    return shogunTrainBackend('CHAIDTree', data, toSet)

def trainRandomForestClassifier(data, toSet):
    toSet['combination_rule'] = Init('MajorityVote')
    toSet['num_bags'] = 5
    return shogunTrainBackend('RandomForest', data, toSet)

def trainKMeansMiniBatch(data, toSet):
    toSet['batch_size'] = 10
    toSet['mb_iter'] = 2
    return shogunTrainBackend('KMeansMiniBatch', data, toSet)

def trainRelaxedTree(data, toSet):
    toSet['machine_for_confusion_matrix'] = Init('MulticlassLibLinear')
    return shogunTrainBackend('RelaxedTree', data, toSet)


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

    ignore = ['LibLinearRegression'] # LibLinearRegression strange failure
    learners = getLearnersByType('regression', ignore)

    remove = ['Machine', 'Base']
    learners = [l for l in learners if not any(x in l for x in remove)]

    needKernel = ['KRRNystrom', 'KernelRidgeRegression', 'LibSVR']

    extraTrainSetup = {'GaussianProcessRegression': trainGaussianProcessRegression,}

    @logCountAssertionFactory(2)
    def compareOutputs(learner):
        data = (trainX, trainY, testX, Xtrain, Ytrain, Xtest)
        trainAndApplyBackend(learner, data, 'apply_regression', needKernel, [],
                              extraTrainSetup)

    for learner in learners:
        compareOutputs(learner)

def trainGaussianProcessRegression(data, toSet):
    # TODO can this be done without stepping outside of nimble?
    Xtrain, Ytrain = data
    kernel = Init('GaussianKernel')
    mean_function = Init('ZeroMean')
    gauss_likelihood = Init('GaussianLikelihood')
    eim = Init('ExactInferenceMethod', kernel=kernel, mean=mean_function,
               model=gauss_likelihood)
    toSet['inference_method'] = eim
    return shogunTrainBackend('GaussianProcessRegression', data, toSet)


@shogunSkipDec
def testShogunGaussianProcessClassification():
    # can be binary or multiclass classification

    data = generateClassificationData(2, 20, 20)
    trainX = data[0][0]
    trainY = data[0][1]
    trainY.points.fillMatching(-1, 0, useLog=False)
    testX = data[1][0]
    Ytrain = trainY.copy('numpy array', outputAs1D=True)
    Ytrain = BinaryLabels(Ytrain)
    Xtrain = RealFeatures(trainX.copy('numpy array', rowsArePoints=False))
    Xtest = RealFeatures(testX.copy('numpy array', rowsArePoints=False))

    extraTrainSetup = {'GaussianProcessClassification': trainGPCBinary}

    data = (trainX, trainY, testX, Xtrain, Ytrain, Xtest)
    trainAndApplyBackend('GaussianProcessClassification', data, 'apply_binary',
                         [], [],  extraTrainSetup)

    # Multiclass
    data = generateClassificationData(3, 20, 20)
    trainX = data[0][0]
    trainY = data[0][1]
    testX = data[1][0]
    Ytrain = trainY.copy('numpy array', outputAs1D=True)
    Ytrain = MulticlassLabels(Ytrain)
    Xtrain = RealFeatures(trainX.copy('numpy array', rowsArePoints=False))
    Xtest = RealFeatures(testX.copy('numpy array', rowsArePoints=False))

    extraTrainSetup = {'GaussianProcessClassification': trainGPCMulticlass}

    data = (trainX, trainY, testX, Xtrain, Ytrain, Xtest)
    trainAndApplyBackend('GaussianProcessClassification', data, 'apply_multiclass',
                         [], [],  extraTrainSetup)

def trainGPCBinary(data, toSet):
    Xtrain, Ytrain = data
    kernel = Init('GaussianKernel')
    mean_function = Init('ZeroMean')
    gauss_likelihood = Init('ProbitLikelihood')
    mlim = Init('SingleLaplaceInferenceMethod', kernel=kernel, mean=mean_function,
                model=gauss_likelihood)
    toSet['inference_method'] = mlim
    return shogunTrainBackend('GaussianProcessClassification', data, toSet)

def trainGPCMulticlass(data, toSet):
    Xtrain, Ytrain = data
    kernel = Init('GaussianKernel')
    mean_function = Init('ConstMean')
    gauss_likelihood = Init('SoftMaxLikelihood')
    mlim = Init('MultiLaplaceInferenceMethod', kernel=kernel, mean=mean_function,
                model=gauss_likelihood)
    toSet['inference_method'] = mlim
    return shogunTrainBackend('GaussianProcessClassification', data, toSet)


@shogunSkipDec
def TODOShogunMachineLearner():
    data = generateClassificationData(3, 20, 20)
    trainX = abs(data[0][0])
    trainY = abs(data[0][1])
    testX = abs(data[1][0])
    Ytrain = trainY.copy('numpy array', outputAs1D=True)
    Ytrain = MulticlassLabels(Ytrain)
    Xtrain = RealFeatures(trainX.copy('numpy array', rowsArePoints=False))
    Xtest = RealFeatures(testX.copy('numpy array', rowsArePoints=False))

    extraTrainSetup = {'LinearMulticlassMachine': trainLinearMulticlassMachine}

    data = (trainX, trainY, testX, Xtrain, Ytrain, Xtest)
    trainAndApplyBackend('LinearMulticlassMachine', data, 'apply_multiclass',
                         [], [],  extraTrainSetup, [])

def trainLinearMulticlassMachine(data, toSet):
    # TODO machine takes 2 parameters num and machine, this is problematic
    # because cannot provide two values for machine
    toSet['machine'] = 'LibLinear'
    toSet['rejection_strategy'] = 'MulticlassOneVsOneStrategy'
    return shogunTrainBackend('LinearMulticlassMachine', data, toSet)


@shogunSkipDec
def TODOShogunStructuredOutputLearner():
    # TODO these are primarily the learners which are classified as 'other'
    pass


@shogunSkipDec
def test_checkProcessFailure_maxTime():
    def dontSleep():
        pass

    def sleepFiveSeconds():
        time.sleep(5)

    def exitSignal():
        os.kill(os.getpid(), signal.SIGSEGV)

    def failedProcessCheck(target):
        p = multiprocessing.Process(target=target)
        p.start()
        p.join(timeout=0.1)
        exitcode = p.exitcode
        p.terminate()
        if exitcode:
            raise SystemError("exitcode")

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
