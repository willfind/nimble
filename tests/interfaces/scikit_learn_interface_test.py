"""
Unit tests for scikit_learn_interface.py

"""

from __future__ import absolute_import
import importlib
import inspect
import tempfile

import numpy.testing
from nose.plugins.attrib import attr
from nose.tools import raises

import UML
from UML import loadTrainedLearner
from UML.randomness import numpyRandom
from UML.randomness import generateSubsidiarySeed
from UML.exceptions import InvalidArgumentValue
from UML.helpers import generateClassificationData
from UML.helpers import generateRegressionData
from UML.helpers import generateClusteredPoints
from UML.helpers import inspectArguments
from UML.calculate.loss import rootMeanSquareError
from UML.interfaces.scikit_learn_interface import SciKitLearn
from UML.interfaces.universal_interface import UniversalInterface

from .test_helpers import checkLabelOrderingAndScoreAssociations
from .skipTestDecorator import SkipMissing
from ..assertionHelpers import logCountAssertionFactory
from ..assertionHelpers import noLogEntryExpected, oneLogEntryExpected

scipy = UML.importModule('scipy.sparse')
sklearn = UML.importExternalLibraries.importModule("sklearn")

packageName = 'sciKitLearn'

sklSkipDec = SkipMissing(packageName)

@sklSkipDec
@noLogEntryExpected
def test_SciKitLearn_version():
    interface = SciKitLearn()
    assert interface.version() == sklearn.__version__

def toCall(learner):
    return packageName + '.' + learner

@sklSkipDec
@logCountAssertionFactory(4)
def testScikitLearnAliases():
    """ Test availability of correct aliases for 'sciKitLearn' """
    variables = ["Y", "x1", "x2"]
    data = [[2, 1, 1], [3, 1, 2], [4, 2, 2], ]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[0, 1]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    # make a bundle of calls, don't care about the results, only
    # that they work.
    UML.trainAndApply("scikitlearn.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    UML.trainAndApply("SKL.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    UML.trainAndApply("skl.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    UML.trainAndApply("SciKitLearn.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})


@sklSkipDec
@oneLogEntryExpected
def testSciKitLearnHandmadeRegression():
    """ Test sciKitLearn() by calling on a regression learner with known output """
    variables = ["Y", "x1", "x2"]
    data = [[2, 1, 1], [3, 1, 2], [4, 2, 2], ]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[0, 1]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply(toCall("LinearRegression"), trainingObj, trainY="Y", testX=testObj,
                            output=None, arguments={})

    assert ret is not None

    expected = [[1.]]
    expectedObj = UML.createData('Matrix', expected, useLog=False)

    numpy.testing.assert_approx_equal(ret[0, 0], 1.)


@sklSkipDec
@oneLogEntryExpected
def testSciKitLearnSparseRegression():
    """ Test sciKitLearn() by calling on a sparse regression learner with an extremely large, but highly sparse, matrix """
    if not scipy:
        return

    x = 1000
    c = 10
    points = numpyRandom.randint(0, x, c)
    points2 = numpyRandom.randint(0, x, c)
    cols = numpyRandom.randint(0, x, c)
    cols2 = numpyRandom.randint(0, x, c)
    data = numpyRandom.rand(c)
    A = scipy.sparse.coo_matrix((data, (points, cols)), shape=(x, x))
    obj = UML.createData('Sparse', A, useLog=False)
    testObj = obj.copy()
    testObj.features.extract(cols[0], useLog=False)

    ret = UML.trainAndApply(toCall('SGDRegressor'), trainX=obj, trainY=cols[0], testX=testObj)

    assert ret is not None


@sklSkipDec
@oneLogEntryExpected
def testSciKitLearnHandmadeClustering():
    """ Test sciKitLearn() by calling a clustering classifier with known output """
    variables = ["x1", "x2"]
    data = [[1, 0], [3, 3], [5, 0], ]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[1, 0], [1, 1], [5, 1], [3, 4]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply(toCall("KMeans"), trainingObj, testX=testObj, output=None,
                            arguments={'n_clusters': 3})

    # clustering returns a row vector of indices, referring to the cluster centers,
    # we don't care about the exact numbers, this verifies that the appropriate
    # ones are assigned to the same clusters
    assert ret[0, 0] == ret[1, 0]
    assert ret[0, 0] != ret[2, 0]
    assert ret[0, 0] != ret[3, 0]
    assert ret[2, 0] != ret[3, 0]


@sklSkipDec
@oneLogEntryExpected
def testSciKitLearnHandmadeSparseClustering():
    """ Test sciKitLearn() by calling on a sparse clustering learner with known output """
    if not scipy:
        return
    trainData = scipy.sparse.lil_matrix((3, 3))
    trainData[0, :] = [2, 3, 1]
    trainData[1, :] = [2, 2, 1]
    trainData[2, :] = [0, 0, 0]
    trainData = UML.createData('Sparse', data=trainData, useLog=False)

    testData = scipy.sparse.lil_matrix((3, 2))
    testData[0, :] = [3, 3]
    testData[1, :] = [3, 2]
    testData[2, :] = [-1, 0]
    testData = UML.createData('Sparse', data=testData, useLog=False)

    ret = UML.trainAndApply(toCall('MiniBatchKMeans'), trainData, trainY=2, testX=testData, arguments={'n_clusters': 2})

    assert ret[0, 0] == ret[1, 0]
    assert ret[0, 0] != ret[2, 0]


@sklSkipDec
@logCountAssertionFactory(3)
def testSciKitLearnScoreMode():
    """ Test sciKitLearn() scoreMode flags"""
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    # default scoreMode is 'label'
    ret = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
    assert len(ret.points) == 2
    assert len(ret.features) == 1

    bestScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                   scoreMode='bestScore')
    assert len(bestScores.points) == 2
    assert len(bestScores.features) == 2

    allScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                  scoreMode='allScores')
    assert len(allScores.points) == 2
    assert len(allScores.features) == 3

    checkLabelOrderingAndScoreAssociations([0, 1, 2], bestScores, allScores)


@sklSkipDec
@logCountAssertionFactory(3)
def testSciKitLearnScoreModeBinary():
    """ Test sciKitLearn() scoreMode flags, binary case"""
    variables = ["Y", "x1", "x2"]
    data = [[1, 30, 2], [2, 1, 1], [2, 0, 1], [2, -1, -1], [1, 30, 3], [1, 34, 4]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 1], [25, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    # default scoreMode is 'label'
    ret = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
    assert len(ret.points) == 2
    assert len(ret.features) == 1

    bestScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                   scoreMode='bestScore')
    assert len(bestScores.points) == 2
    assert len(bestScores.features) == 2

    allScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                  scoreMode='allScores')
    assert len(allScores.points) == 2
    assert len(allScores.features) == 2

    checkLabelOrderingAndScoreAssociations([1, 2], bestScores, allScores)


@sklSkipDec
@logCountAssertionFactory(4)
def testSciKitLearnCrossDecomp():
    """ Test SKL on learners which take 2d Y data """
    variables = ["x1", "x2"]
    data = [[1, 0], [3, 3], [50, 0], [12, 3], [8, 228]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)
    dataY = [[0, 1], [0, 1], [2, 2], [1, 30], [5, 21]]
    trainingYObj = UML.createData('Matrix', dataY, useLog=False)

    data2 = [[1, 0], [1, 1], [5, 1], [34, 4]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    UML.trainAndApply(toCall("CCA"), trainingObj, testX=testObj, trainY=trainingYObj)
    UML.trainAndApply(toCall("PLSCanonical"), trainingObj, testX=testObj, trainY=trainingYObj)
    UML.trainAndApply(toCall("PLSRegression"), trainingObj, testX=testObj, trainY=trainingYObj)
    UML.trainAndApply(toCall("PLSSVD"), trainingObj, testX=testObj, trainY=trainingYObj)


@sklSkipDec
@noLogEntryExpected
def testSciKitLearnListLearners():
    """ Test scikit learn's listSciKitLearnLearners() by checking the output for those learners we unit test """

    ret = UML.listLearners(packageName)

    assert 'KMeans' in ret
    assert 'LinearRegression' in ret

    toExclude = []

    for name in ret:
        if name not in toExclude:
            params = UML.learnerParameters(toCall(name))
            assert params is not None
            defaults = UML.learnerDefaultValues(toCall(name))
            for pSet in params:
                for dSet in defaults:
                    for key in dSet.keys():
                        assert key in pSet

@sklSkipDec
@raises(InvalidArgumentValue)
def testSciKitLearnExcludedLearners():
    trainX = UML.createData('Matrix', [1,2,3])
    apply = UML.trainAndApply(toCall('KernelCenterer'), trainX)


def getLearnersByType(lType, ignore=[]):
    learners = UML.listLearners(packageName)
    typeMatch = []
    for learner in learners:
        learnerType = UML.learnerType(toCall(learner))
        if lType == learnerType and learner not in ignore:
            typeMatch.append(learner)
    return typeMatch


@sklSkipDec
@attr('slow')
def testSciKitLearnClassificationLearners():
    data = generateClassificationData(2, 20, 10)
    # some classification learners require non-negative data
    trainX = abs(data[0][0])
    trainY = abs(data[0][1])
    testX = abs(data[1][0])
    Xtrain = trainX.data
    Ytrain = trainY.data
    Xtest = testX.data

    learners = getLearnersByType('classification')

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)
        sciKitLearnObj.fit(Xtrain, Ytrain)
        predSKL = sciKitLearnObj.predict(Xtest)
        predSKL = UML.createData('Matrix', predSKL.reshape(-1,1), useLog=False)

        TL = UML.train(toCall(learner), trainX, trainY, arguments=arguments)
        predUML = TL.apply(testX)
        predSL = _apply_saveLoad(TL, testX)

        equalityAssertHelper(predSKL, predUML, predSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@attr('slow')
def testSciKitLearnRegressionLearners():
    data = generateRegressionData(2, 20, 10)
    trainX = data[0][0]
    trainY = data[0][1]
    testX = data[1][0]
    Xtrain = trainX.data
    Ytrain = trainY.data
    Xtest = testX.data

    ignore = ['MultiTaskElasticNet', 'MultiTaskElasticNetCV',   # special cases, tested elsewhere
              'MultiTaskLasso', 'MultiTaskLassoCV',]
    learners = getLearnersByType('regression', ignore)

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)
        sciKitLearnObj.fit(Xtrain, Ytrain)
        predSKL = sciKitLearnObj.predict(Xtest)
        predSKL = UML.createData('Matrix', predSKL.reshape(-1,1), useLog=False)

        TL = UML.train(toCall(learner), trainX, trainY, arguments=arguments)
        predUML = TL.apply(testX)
        predSL = _apply_saveLoad(TL, testX)

        equalityAssertHelper(predSKL, predUML, predSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@attr('slow')
def testSciKitLearnMultiTaskRegressionLearners():
    """ Test that predictions for from UML.trainAndApply match predictions from scikitlearn
    multitask learners with predict method"""

    skl = SciKitLearn()

    trainX = [[0,0], [1, 1], [2, 2]]
    trainY = [[0, 0], [1, 1], [2, 2]]
    testX = [[2,2], [0,0], [1,1]]

    trainXObj = UML.createData('Matrix', trainX, useLog=False)
    trainYObj = UML.createData('Matrix', trainY, useLog=False)
    testXObj = UML.createData('Matrix', testX, useLog=False)

    multiTaskLearners = ['MultiTaskElasticNet', 'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV']

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        sciKitLearnObj.fit(trainX, trainY)
        predictionSciKit = sciKitLearnObj.predict(testX)
        # convert to UML Base object for comparison
        predictionSciKit = UML.createData('Matrix', predictionSciKit, useLog=False)

        TL = UML.train(toCall(learner), trainXObj, trainYObj)
        predUML = TL.apply(testXObj)
        predSL = _apply_saveLoad(TL, testXObj)

        equalityAssertHelper(predictionSciKit, predUML, predSL)

    for learner in multiTaskLearners:
        compareOutputs(learner)


@sklSkipDec
@attr('slow')
def testSciKitLearnClusterLearners():
    data = generateClusteredPoints(3, 60, 8)
    data = data[0]
    data.points.shuffle()
    trainX = data[:50,:]
    testX = data[50:,:]
    Xtrain = trainX.data
    Xtest = testX.data

    learners = getLearnersByType('cluster')

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)
        try:
            sciKitLearnObj.fit(Xtrain)
            predSKL = sciKitLearnObj.predict(Xtest)
        except AttributeError:
            predSKL = sciKitLearnObj.fit_predict(Xtrain, Xtest)
        predSKL = UML.createData('Matrix', predSKL.reshape(-1,1), useLog=False)

        TL = UML.train(toCall(learner), trainX, arguments=arguments)
        predUML = TL.apply(testX)
        predSL = _apply_saveLoad(TL, testX)

        equalityAssertHelper(predSKL, predUML, predSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@attr('slow')
def testSciKitLearnOtherPredictLearners():
    data = generateClassificationData(2, 20, 10)
    trainX = abs(data[0][0])
    trainY = abs(data[0][1])
    testX = abs(data[1][0])
    Xtrain = trainX.data
    Ytrain = trainY.data
    Xtest = testX.data

    ignore = ['TSNE', 'MDS', 'SpectralEmbedding',] # special cases, tested elsewhere
    learners = getLearnersByType('other', ignore)

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)

        sciKitLearnObj.fit(Xtrain, Ytrain)
        predSKL = sciKitLearnObj.predict(Xtest)
        predSKL = UML.createData('Matrix', predSKL.reshape(-1,1), useLog=False)

        TL = UML.train(toCall(learner), trainX, trainY, arguments=arguments)
        predUML = TL.apply(testX)
        predSL = _apply_saveLoad(TL, testX)

        equalityAssertHelper(predSKL, predUML, predSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@attr('slow')
def testSciKitLearnTransformationLearners():

    data = generateClassificationData(2, 20, 10)
    trainX = abs(data[0][0])
    trainY = abs(data[0][1])
    Xtrain = trainX.data
    Ytrain = trainY.data

    ignore = ['GaussianRandomProjection', 'SparseRandomProjection',   # special cases, tested elsewhere
              'MiniBatchSparsePCA', 'SparsePCA', 'NMF', 'FastICA', 'Isomap', 'VarianceThreshold']
    learners = getLearnersByType('transformation', ignore)

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)
        sciKitLearnObj.fit(Xtrain, Ytrain)
        transSKL = sciKitLearnObj.transform(Xtrain)
        transSKL = UML.createData('Matrix', transSKL, useLog=False)

        TL = UML.train(toCall(learner), trainX, trainY, arguments=arguments)
        transSL = _apply_saveLoad(TL, trainX)
        transUML = TL.apply(trainX)


        equalityAssertHelper(transSKL, transUML, transSL)

    for learner in learners:
        compareOutputs(learner)

@sklSkipDec
@attr('slow')
def testSciKitLearnRandomProjectionTransformation():
    trainX = UML.createRandomData('Matrix', 10, 5000, 0.98)
    Xtrain = trainX.data

    learners = ['GaussianRandomProjection', 'SparseRandomProjection',]

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)

        transSKL = sciKitLearnObj.fit_transform(Xtrain)
        transSKL = UML.createData('Matrix', transSKL, useLog=False)

        TL = UML.train(toCall(learner), trainX, arguments=arguments)
        transUML = TL.apply(trainX)
        transSL = _apply_saveLoad(TL, trainX)

        equalityAssertHelper(transSKL, transUML, transSL)

    for learner in learners:
        compareOutputs(learner)

@sklSkipDec
@attr('slow')
def testSciKitLearnSparsePCATransformation():
    # do not accept sparse matrices
    trainX = UML.createRandomData('Matrix', 100, 10, sparsity=0.9)
    Xtrain = trainX.data

    learners = ['MiniBatchSparsePCA', 'SparsePCA',]

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        #TODO explore why ridge_alpha defaults to 'deprecated'
        arguments['ridge_alpha'] = 0.1
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)
        sciKitLearnObj.fit(Xtrain)
        transSKL = sciKitLearnObj.transform(Xtrain)
        transSKL = UML.createData('Matrix', transSKL, useLog=False)

        TL = UML.train(toCall(learner), trainX, arguments=arguments)
        transUML = TL.apply(trainX)
        transSL = _apply_saveLoad(TL, trainX)

        equalityAssertHelper(transSKL, transUML, transSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@attr('slow')
def testSciKitLearnEmbeddingLearners():
    data = generateClassificationData(2, 20, 10)
    trainX = abs(data[0][0])
    Xtrain = trainX.data

    learners = ['TSNE', 'MDS', 'SpectralEmbedding',]


    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)

        transSKL = sciKitLearnObj.fit_transform(Xtrain)
        transSKL = UML.createData('Matrix', transSKL, useLog=False)

        TL = UML.train(toCall(learner), trainX, arguments=arguments)
        transUML = TL.apply(trainX)
        transSL = _apply_saveLoad(TL, trainX)

        equalityAssertHelper(transSKL, transUML, transSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
def testSciKitLearnTransformationDataInputIssues():
    # must be non-negative matrix
    Xtrain = [[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]]
    trainX = UML.createData('Matrix', Xtrain)

    learners = ['NMF', 'FastICA', 'Isomap', 'VarianceThreshold',]

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)
        sciKitLearnObj.fit(Xtrain)
        transSKL = sciKitLearnObj.transform(Xtrain)
        transSKL = UML.createData('Matrix', transSKL, useLog=False)

        TL = UML.train(toCall(learner), trainX, arguments=arguments)
        transUML = TL.apply(trainX)
        transSL = _apply_saveLoad(TL, trainX)

        equalityAssertHelper(transSKL, transUML, transSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@logCountAssertionFactory(4)
def testCustomRidgeRegressionCompare():
    """ Sanity check for custom RidgeRegression, compare results to SKL's Ridge """
    data = [[0, 1, 2], [13, 12, 4], [345, 233, 76]]
    trainObj = UML.createData('Matrix', data, useLog=False)

    data2 = [[122, 34], [76, -3]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    name = 'Custom.RidgeRegression'
    TL = UML.train(name, trainX=trainObj, trainY=0, arguments={'lamb': 1})
    ret1 = TL.apply(testObj)
    ret2 = UML.trainAndApply("Scikitlearn.Ridge", trainX=trainObj, trainY=0, testX=testObj,
                             arguments={'alpha': 1, 'fit_intercept': False})
    ret3 = _apply_saveLoad(TL, testObj)

    equalityAssertHelper(ret1, ret2, ret3)


@sklSkipDec
@logCountAssertionFactory(4)
def testCustomRidgeRegressionCompareRandomized():
    """ Sanity check for custom RidgeRegression, compare results to SKL's Ridge on random data"""
    trainObj = UML.createRandomData("Matrix", 1000, 60, .1, useLog=False)
    testObj = UML.createRandomData("Matrix", 100, 59, .1, useLog=False)

    name = 'Custom.RidgeRegression'
    TL = UML.train(name, trainX=trainObj, trainY=0, arguments={'lamb': 1})
    ret1 = TL.apply(testObj)
    ret2 = UML.trainAndApply("Scikitlearn.Ridge", trainX=trainObj, trainY=0, testX=testObj,
                             arguments={'alpha': 1, 'fit_intercept': False})
    ret3 = _apply_saveLoad(TL, testObj)

    equalityAssertHelper(ret1, ret2, ret3)


@sklSkipDec
@attr('slow')
@logCountAssertionFactory(4)
def testCustomKNNClassficationCompareRandomized():
    """ Sanity check on custom KNNClassifier, compare to SKL's KNeighborsClassifier on random data"""
    trainX, ignore, trainY = generateClusteredPoints(5, 50, 5, addFeatureNoise=True, addLabelNoise=False,
                                                     addLabelColumn=False)
    testX, ignore, testY = generateClusteredPoints(5, 5, 5, addFeatureNoise=True, addLabelNoise=False,
                                                   addLabelColumn=False)

    cusname = 'Custom.KNNClassifier'
    sklname = "Scikitlearn.KNeighborsClassifier"
    TL = UML.train(cusname, trainX, trainY=trainY, k=5)
    ret1 = TL.apply(testX)
    ret2 = UML.trainAndApply(sklname, trainX, trainY=trainY, testX=testX, n_neighbors=5, algorithm='brute')
    ret3 = _apply_saveLoad(TL, testX)

    equalityAssertHelper(ret1, ret2, ret3)


@sklSkipDec
@attr('slow')
def testGetAttributesCallable():
    """ Demonstrate getAttribtues will work for each learner (with default params) """
    cData = generateClassificationData(2, 10, 5)
    ((cTrainX, cTrainY), (cTestX, cTestY)) = cData
    rData = generateRegressionData(2, 10, 5)
    ((rTrainX, rTrainY), (rTestX, rTestY)) = rData
    printExceptions = False

    allLearners = UML.listLearners('scikitlearn')
    toTest = allLearners

    for learner in toTest:
        fullName = 'scikitlearn.' + learner
        lType = UML.learnerType(fullName)
        if lType in ['classification', 'transformation', 'cluster', 'other']:
            X = cTrainX
            Y = cTrainY
        if lType == 'regression':
            X = rTrainX
            Y = rTrainY

        try:
            tl = UML.train(fullName, X, Y)
        # this is meant to safely bypass those learners that have required
        # arguments or require unique data
        except InvalidArgumentValue as iav:
            if printExceptions:
                print (learner + " : " + lType)
                print(iav)
        tl.getAttributes()


@sklSkipDec
@logCountAssertionFactory(2)
def testConvertYTrainDType():
    """ test trainY dtype is converted to float when learner requires Y to be numeric"""
    train = [['a', 1, -1, -3, -3, -1],
              ['b', 2, 0.4, -0.8, 0.2, -0.3],
              ['c', 3, 2, 1, 2, 4]]
    test = [['a', 1, -2, -1, -3, -2],
            ['c', 3, 1, 2, 3, 1]]
    # object will have 'object' dtype because of strings in data
    trainObj = UML.createData('Matrix', train, useLog=False)
    trainObj.features.retain([1, 2, 3, 4, 5], useLog=False)
    testObj = UML.createData('Matrix', test, useLog=False)
    testObj.features.retain([2,3,4,5], useLog=False)

    # case1 trainY passed as integer
    assert trainObj[:,0].data.dtype == numpy.object_
    pred = UML.trainAndApply('SciKitLearn.LogisticRegression', trainObj, 0, testObj)

    #case2 trainY passed as UML object
    trainY = trainObj.features.extract(0, useLog=False)
    assert trainY.data.dtype == numpy.object_
    pred = UML.trainAndApply('SciKitLearn.LogisticRegression', trainObj, trainY, testObj)

@sklSkipDec
@logCountAssertionFactory(3)
def test_applier_acceptsNewArguments():
    """ Test an skl function that accepts new arguments for transform """
    data = [[-1., -1.],
            [-1., -1.],
            [ 1.,  1.],
            [ 1.,  1.]]

    dataObj = UML.createData('Matrix', data, useLog=False)

    # StandardScaler.transform takes a 'copy' argument. Default is None.
    tl = UML.train('SciKitLearn.StandardScaler', dataObj)
    assert tl.transformedArguments['copy'] is None
    # using arguments parameter
    transformed = tl.apply(dataObj, arguments={'copy':True})

    # using kwarguments
    transformed = tl.apply(dataObj, copy=True)

@sklSkipDec
def test_applier_exception():
    """ Test an skl function with invalid arguments for transform """
    data = [[-1., -1.],
            [-1., -1.],
            [ 1.,  1.],
            [ 1.,  1.]]

    dataObj = UML.createData('Matrix', data)

    # StandardScaler.transform does not takes a 'foo' argument
    tl = UML.train('SciKitLearn.StandardScaler', dataObj)
    assert 'foo' not in tl.transformedArguments
    try:
        # using arguments parameter
        transformed = tl.apply(dataObj, arguments={'foo': True})
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    try:
        # using kwarguments
        transformed = tl.apply(dataObj, foo=True)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

@sklSkipDec
@logCountAssertionFactory(3)
def test_getScores_acceptsNewArguments():
    """ Test an skl function that accepts new arguments for predict_proba """
    train = [[1, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1],
             [1, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1]]
    testX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Need to set elementType b/c conversion will not be done when check_input=False
    trainObj = UML.createData('Matrix', train, elementType=numpy.float32, useLog=False)
    testObj = UML.createData('Matrix', testX, elementType=numpy.float32, useLog=False)

    # DecisionTreeClassifier.predict_proba takes a 'check_input' argument. Default is True.
    tl = UML.train('SciKitLearn.DecisionTreeClassifier', trainObj, 0)
    assert tl.transformedArguments['check_input'] is True
    # using arguments parameter
    transformed = tl.apply(testObj, arguments={'check_input':False})

    # using kwarguments
    transformed = tl.apply(testObj, check_input=False)

@sklSkipDec
def test_getScores_exception():
    """ Test an skl function with invalid arguments for predict_proba """
    train = [[1, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1],
             [1, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1]]
    testX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    trainObj = UML.createData('Matrix', train, elementType=numpy.float32)
    testObj = UML.createData('Matrix', testX, elementType=numpy.float32)

    # DecisionTreeClassifier.predict_proba does not take a 'foo' argument.
    tl = UML.train('SciKitLearn.DecisionTreeClassifier', trainObj, 0)
    assert 'foo' not in tl.transformedArguments
    try:
        # using arguments parameter
        transformed = tl.getScores(testObj, arguments={'foo': True})
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    try:
        # using kwarguments
        transformed = tl.getScores(testObj, foo=True)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

def _apply_saveLoad(trainerLearnerObj, givenTestX):
    """
    Given a TrainedLearner object, return the results of apply after having
    saved then loaded the learner from a file.
    """
    with tempfile.NamedTemporaryFile(suffix=".umlm") as tmpFile:
        trainerLearnerObj.save(tmpFile.name)
        trainer_ret_l = loadTrainedLearner(tmpFile.name)
        return trainer_ret_l.apply(givenTestX, useLog=False)

@sklSkipDec
@oneLogEntryExpected
def test_saveLoadTrainedLearner_logCount():
    train = [[1, -1, -3, -3, -1],
              [2, 0.4, -0.8, 0.2, -0.3],
              [3, 2, 1, 2, 4]]
    trainObj = UML.createData('Matrix', train, useLog=False)

    tl = UML.train('SciKitLearn.LogisticRegression', trainObj, 0, useLog=False)
    with tempfile.NamedTemporaryFile(suffix=".umlm") as tmpFile:
        tl.save(tmpFile.name)
        load = loadTrainedLearner(tmpFile.name)

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
