
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Unit tests for scikit_learn_interface.py
"""

import numpy as np
import pytest

import nimble
from nimble import loadTrainedLearner
from nimble.random import numpyRandom
from nimble.random import generateSubsidiarySeed
from nimble.exceptions import InvalidArgumentValue
from nimble.core._learnHelpers import generateClusteredPoints
from nimble._utility import scipy
from tests.helpers import raises
from tests.helpers import logCountAssertionFactory
from tests.helpers import noLogEntryExpected, oneLogEntryExpected
from tests.helpers import generateClassificationData
from tests.helpers import generateRegressionData
from tests.helpers import skipMissingPackage
from tests.helpers import PortableNamedTempFileContext
from .test_helpers import checkLabelOrderingAndScoreAssociations

packageName = 'sciKitLearn'

sklSkipDec = skipMissingPackage(packageName)

@sklSkipDec
@noLogEntryExpected
def test_SciKitLearn_version():
    import sklearn
    interface = nimble.core._learnHelpers.findBestInterface('scikitlearn')
    assert interface.version() == sklearn.__version__

def toCall(learner):
    return packageName + '.' + learner

@sklSkipDec
@logCountAssertionFactory(6)
def testScikitLearnAliases():
    """ Test availability of correct aliases for 'sciKitLearn' """
    variables = ["Y", "x1", "x2"]
    data = [[2, 1, 1], [3, 1, 2], [4, 2, 2], ]
    trainingObj = nimble.data(data, featureNames=variables, useLog=False)

    data2 = [[0, 1]]
    testObj = nimble.data(data2, useLog=False)

    # make a bundle of calls, don't care about the results, only
    # that they work.
    nimble.trainAndApply("scikitlearn.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    nimble.trainAndApply("SKL.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    nimble.trainAndApply("skl.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    nimble.trainAndApply("SciKitLearn.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    nimble.trainAndApply("sklearn.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    from sklearn.linear_model import LinearRegression
    nimble.trainAndApply(LinearRegression, trainingObj, trainY="Y", testX=testObj, arguments={})


@sklSkipDec
@oneLogEntryExpected
def testSciKitLearnHandmadeRegression():
    """ Test sciKitLearn() by calling on a regression learner with known output """
    variables = ["Y", "x1", "x2"]
    data = [[2, 1, 1], [3, 1, 2], [4, 2, 2], ]
    trainingObj = nimble.data(data, featureNames=variables, useLog=False)

    data2 = [[0, 1]]
    testObj = nimble.data(data2, useLog=False)

    ret = nimble.trainAndApply(toCall("LinearRegression"), trainingObj,
                               trainY="Y", testX=testObj, arguments={})

    assert ret is not None

    expected = [[1.]]
    expectedObj = nimble.data(expected, useLog=False)

    np.testing.assert_approx_equal(ret[0, 0], 1.)


@sklSkipDec
@oneLogEntryExpected
def testSciKitLearnSparseRegression():
    """ Test sciKitLearn() by calling on a sparse regression learner with an extremely large, but highly sparse, matrix """
    x = 1000
    c = 10
    points = numpyRandom.randint(0, x, c)
    points2 = numpyRandom.randint(0, x, c)
    cols = numpyRandom.randint(0, x, c)
    cols2 = numpyRandom.randint(0, x, c)
    data = numpyRandom.rand(c)
    A = scipy.sparse.coo_matrix((data, (points, cols)), shape=(x, x))
    obj = nimble.data(A, useLog=False)
    testObj = obj.copy()
    testObj.features.extract(cols[0], useLog=False)

    ret = nimble.trainAndApply(toCall('SGDRegressor'), trainX=obj, trainY=cols[0], testX=testObj)

    assert ret is not None


@sklSkipDec
@oneLogEntryExpected
def testSciKitLearnHandmadeClustering():
    """ Test sciKitLearn() by calling a clustering classifier with known output """
    variables = ["x1", "x2"]
    data = [[1, 0], [3, 3], [5, 0], ]
    trainingObj = nimble.data(data, featureNames=variables, useLog=False)

    data2 = [[1, 0], [1, 1], [5, 1], [3, 4]]
    testObj = nimble.data(data2, useLog=False)

    ret = nimble.trainAndApply(toCall("KMeans"), trainingObj, testX=testObj,
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
    trainData = scipy.sparse.lil_matrix((3, 3))
    trainData[0, :] = [2, 3, 1]
    trainData[1, :] = [2, 2, 1]
    trainData[2, :] = [0, 0, 0]
    trainData = nimble.data(source=trainData, useLog=False)

    testData = scipy.sparse.lil_matrix((3, 2))
    testData[0, :] = [3, 3]
    testData[1, :] = [3, 2]
    testData[2, :] = [-1, 0]
    testData = nimble.data(source=testData, useLog=False)

    ret = nimble.trainAndApply(toCall('MiniBatchKMeans'), trainData, trainY=2, testX=testData, arguments={'n_clusters': 2})

    assert ret[0, 0] == ret[1, 0]
    assert ret[0, 0] != ret[2, 0]


@sklSkipDec
@logCountAssertionFactory(3)
def testSciKitLearnScoreMode():
    """ Test sciKitLearn() scoreMode flags"""
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2]]
    trainingObj = nimble.data(data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = nimble.data(data2, useLog=False)

    # default scoredMode is None
    ret = nimble.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
    assert len(ret.points) == 2
    assert len(ret.features) == 1

    bestScores = nimble.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                   scoreMode='bestScore')
    assert len(bestScores.points) == 2
    assert len(bestScores.features) == 2

    allScores = nimble.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
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
    trainingObj = nimble.data(data, featureNames=variables, useLog=False)

    data2 = [[2, 1], [25, 0]]
    testObj = nimble.data(data2, useLog=False)

    # default scoredMode is None
    ret = nimble.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
    assert len(ret.points) == 2
    assert len(ret.features) == 1

    bestScores = nimble.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                   scoreMode='bestScore')
    assert len(bestScores.points) == 2
    assert len(bestScores.features) == 2

    allScores = nimble.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                  scoreMode='allScores')
    assert len(allScores.points) == 2
    assert len(allScores.features) == 2

    checkLabelOrderingAndScoreAssociations([1, 2], bestScores, allScores)


@sklSkipDec
@logCountAssertionFactory(9)
def testSciKitLearnCrossDecomp():
    """ Test SKL on learners which take 2d Y data """
    variables = ["x1", "x2"]
    data = [[1, 0], [3, 3], [50, 0], [12, 3], [8, 228]]
    trainingObj = nimble.data(data, featureNames=variables, useLog=False)
    dataY = [[0, 1], [0, 1], [2, 2], [1, 30], [5, 21]]
    trainingYObj = nimble.data(dataY, useLog=False)

    data2 = [[1, 0], [1, 1], [5, 1], [34, 4]]
    testObj = nimble.data(data2, useLog=False)

    learners = ["PLSCanonical", "PLSRegression", "CCA"]

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        arguments = setupSKLArguments(sciKitLearnObj)
        sciKitLearnObj.fit(data, dataY)
        predSKL = sciKitLearnObj.predict(data2)
        predSKL = nimble.data(predSKL, useLog=False)

        seed = adjustRandomParamForNimble(arguments)
        TL = nimble.train(toCall(learner), trainingObj, trainingYObj, arguments=arguments,
                          randomSeed=seed)
        assert TL.learnerType == 'regression'
        predNimble = TL.apply(testObj)
        predSL = _apply_saveLoad(TL, testObj)

        equalityAssertHelper(predSKL, predNimble, predSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@noLogEntryExpected
def testSciKitLearnListLearners():
    """ Test scikit learn's listSciKitLearnLearners() by checking the output for those learners we unit test """

    ret = nimble.learnerNames(packageName)

    assert 'KMeans' in ret
    assert 'LinearRegression' in ret

    for name in ret:
        params = nimble.learnerParameters(toCall(name))
        assert params is not None
        defaults = nimble.learnerParameterDefaults(toCall(name))
        for key in defaults.keys():
            assert key in params

@sklSkipDec
@raises(InvalidArgumentValue)
def testSciKitLearnExcludedLearners():
    trainX = nimble.data([1,2,3])
    apply = nimble.trainAndApply(toCall('KernelCenterer'), trainX)


def getLearnersByType(lType=None, ignore=[], learnersRequired=True):
    learners = nimble.learnerNames(packageName)
    typeMatch = []
    for learner in learners:
        if lType is not None:
            learnerType = nimble.learnerType(toCall(learner))
            if lType == learnerType and learner not in ignore:
                typeMatch.append(learner)
        elif learner not in ignore:
            typeMatch.append(learner)
    if learnersRequired:
        assert typeMatch # check not returning an empty list
    return typeMatch


@sklSkipDec
@pytest.mark.slow
def testSciKitLearnClassificationLearners():
    data = generateClassificationData(2, 20, 10)
    # some classification learners require non-negative data
    trainX = abs(data[0][0])
    trainY = abs(data[0][1])
    testX = abs(data[1][0])
    Xtrain = trainX.copy('numpy array')
    Ytrain = trainY.copy('numpy array', outputAs1D=True)
    Xtest = testX.copy('numpy array')

    learners = getLearnersByType('classification')

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        arguments = setupSKLArguments(sciKitLearnObj)
        sciKitLearnObj.fit(Xtrain, Ytrain)
        predSKL = sciKitLearnObj.predict(Xtest)
        predSKL = nimble.data(predSKL.reshape(-1,1), useLog=False)

        seed = adjustRandomParamForNimble(arguments)
        TL = nimble.train(toCall(learner), trainX, trainY, arguments=arguments,
                          randomSeed=seed)
        assert TL.learnerType == 'classification'
        predNimble = TL.apply(testX)
        predSL = _apply_saveLoad(TL, testX)

        equalityAssertHelper(predSKL, predNimble, predSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@pytest.mark.slow
def testSciKitLearnRegressionLearners():
    data = generateRegressionData(2, 20, 10)
    trainX = data[0][0]
    trainY = abs(data[0][1]) # all positive required for some learners
    testX = data[1][0]
    Xtrain = trainX._data
    Ytrain = trainY._data
    Xtest = testX._data

    # requires 2D Y data, tested in cross decomp
    ignore = ["PLSCanonical", "PLSRegression", "CCA"]

    regressors = getLearnersByType('regression', ignore=ignore)
    learners = [r for r in regressors if 'MultiTask' not in r]
    assert learners

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
        sklObj = skl.findCallable(learner)

        extraArgs = {}
        if learner == 'QuantileRegressor':
            extraArgs = {'solver':"highs"}

        sciKitLearnObj = sklObj()
        arguments = setupSKLArguments(sciKitLearnObj, extraArgs)
        sciKitLearnObj.fit(Xtrain, Ytrain)

        predSKL = sciKitLearnObj.predict(Xtest)
        predSKL = nimble.data(predSKL.reshape(-1,1), useLog=False)

        seed = adjustRandomParamForNimble(arguments)
        TL = nimble.train(toCall(learner), trainX, trainY, arguments=arguments,
                          randomSeed=seed)
        assert TL.learnerType == 'regression'
        predNimble = TL.apply(testX)
        predSL = _apply_saveLoad(TL, testX)

        equalityAssertHelper(predSKL, predNimble, predSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@pytest.mark.slow
def testSciKitLearnMultiTaskRegressionLearners():
    """ Test that predictions for from nimble.trainAndApply match predictions from scikitlearn
    multitask learners with predict method"""

    skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')

    trainX = [[0., 0.], [1., 1.], [2., 2.], [0., 0.], [1., 1.], [2., 2.]]
    trainY = [[0, 0], [1, 1], [2, 2], [0, 0], [1, 1], [2, 2]]
    testX = [[2., 2.], [0., 0.], [1., 1.]]

    trainXObj = nimble.data(trainX, useLog=False)
    trainYObj = nimble.data(trainY, useLog=False)
    testXObj = nimble.data(testX, useLog=False)

    regressors = getLearnersByType('regression')
    multiTaskLearners = [r for r in regressors if 'MultiTask' in r]
    assert multiTaskLearners

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        arguments = setupSKLArguments(sciKitLearnObj)
        sciKitLearnObj.fit(trainX, trainY)
        predictionSciKit = sciKitLearnObj.predict(testX)
        # convert to nimble Base object for comparison
        predictionSciKit = nimble.data(predictionSciKit,
                                       useLog=False)

        seed = adjustRandomParamForNimble(arguments)
        TL = nimble.train(toCall(learner), trainXObj, trainYObj,
                          randomSeed=seed)
        assert TL.learnerType == 'regression'
        predNimble = TL.apply(testXObj)
        predSL = _apply_saveLoad(TL, testXObj)

        equalityAssertHelper(predictionSciKit, predNimble, predSL)

    for learner in multiTaskLearners:
        compareOutputs(learner)


@sklSkipDec
@pytest.mark.slow
def testSciKitLearnClusterLearners():
    data = generateClusteredPoints(3, 60, 8)
    data = data[0]
    trainX = data[:50,:]
    testX = data[50:,:]
    Xtrain = trainX._data
    Xtest = testX._data

    learners = getLearnersByType('cluster')

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        arguments = setupSKLArguments(sciKitLearnObj)
        try:
            sciKitLearnObj.fit(Xtrain)
            predSKL = sciKitLearnObj.predict(Xtest)
        except AttributeError:
            predSKL = sciKitLearnObj.fit_predict(Xtrain, Xtest)
        predSKL = nimble.data(predSKL.reshape(-1,1), useLog=False)

        seed = adjustRandomParamForNimble(arguments)
        TL = nimble.train(toCall(learner), trainX, arguments=arguments,
                          randomSeed=seed)
        assert TL.learnerType == 'cluster'
        predNimble = TL.apply(testX)
        predSL = _apply_saveLoad(TL, testX)

        equalityAssertHelper(predSKL, predNimble, predSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@pytest.mark.slow
def testSciKitLearnOtherPredictLearners():
    data = generateClassificationData(2, 20, 10)
    trainX = abs(data[0][0])
    trainY = abs(data[0][1])
    testX = abs(data[1][0])
    Xtrain = trainX._data
    Ytrain = trainY._data
    Xtest = testX._data

    skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
    predictors = getLearnersByType('UNKNOWN', learnersRequired=False)
    learners = [p for p in predictors if hasattr(skl.findCallable(p), 'predict')]

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        arguments = setupSKLArguments(sciKitLearnObj)
        sciKitLearnObj.fit(Xtrain, Ytrain)
        predSKL = sciKitLearnObj.predict(Xtest)
        predSKL = nimble.data(predSKL.reshape(-1,1), useLog=False)

        seed = adjustRandomParamForNimble(arguments)
        TL = nimble.train(toCall(learner), trainX, trainY, arguments=arguments,
                          randomSeed=seed)
        predNimble = TL.apply(testX)
        predSL = _apply_saveLoad(TL, testX)

        equalityAssertHelper(predSKL, predNimble, predSL)

    for learner in learners:
        compareOutputs(learner)


@logCountAssertionFactory(3)
def _compareDualInputOutputs(learner, XObj, XNP, YObj, YNP):
    skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
    sklObj = skl.findCallable(learner)
    sciKitLearnObj = sklObj()
    arguments = setupSKLArguments(sciKitLearnObj)
    if hasattr(sciKitLearnObj, 'transform'):
        sciKitLearnObj.fit(XNP, YNP)
        transSKL = sciKitLearnObj.transform(XNP)
    else:
        transSKL = sciKitLearnObj.fit_transform(XNP, YNP)
    transSKL = nimble.data(transSKL, useLog=False)

    seed = adjustRandomParamForNimble(arguments)
    TL = nimble.train(toCall(learner), XObj, YObj, arguments=arguments,
                        randomSeed=seed)
    assert TL.learnerType == 'transformation'
    transSL = _apply_saveLoad(TL, XObj)
    transNimble = TL.apply(XObj)
    equalityAssertHelper(transSKL, transNimble, transSL)

@logCountAssertionFactory(3)
def _compareSingleInputOutputs(learner, XObj, XNP):
    skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
    sklObj = skl.findCallable(learner)
    sciKitLearnObj = sklObj()
    arguments = setupSKLArguments(sciKitLearnObj)

    if hasattr(sciKitLearnObj, 'transform'):
        sciKitLearnObj.fit(XNP)
        transSKL = sciKitLearnObj.transform(XNP)
    else:
        transSKL = sciKitLearnObj.fit_transform(XNP)
    transSKL = nimble.data(transSKL, useLog=False)

    seed = adjustRandomParamForNimble(arguments)
    TL = nimble.train(toCall(learner), XObj, arguments=arguments,
                        randomSeed=seed)
    transNimble = TL.apply(XObj)
    transSL = _apply_saveLoad(TL, XObj)

    equalityAssertHelper(transSKL, transNimble, transSL)

@sklSkipDec
@pytest.mark.slow
def testSciKitLearnTransformationLearners():
    ignore = ['MiniBatchSparsePCA', 'SparsePCA', 'CountVectorizer',
              'TfidfVectorizer', 'PatchExtractor'] # tested elsewhere
    learners = getLearnersByType('transformation', ignore)

    data = generateClassificationData(2, 20, 10)
    for learner in learners:
        trainX = abs(data[0][0])
        trainY = abs(data[0][1])

        if learner in ["PLSSVD"]:
            # Operates on multiple targets
            trainY.features.append(trainY)
        if learner in ["GaussianRandomProjection", "SparseRandomProjection"]:
            # These learners (with default args and this number of samples)
            # requires inputs with a large number of feature dimensions
            trainX = nimble.random.data(40, 5000, 0.98)

        # Check all transformers as both supervised and unsupervised;
        # some are supervised only.
        _compareDualInputOutputs(learner, trainX, trainX.copy("numpyarray"),
                                 trainY, trainY.copy("numpyArray"))

        try:
            _compareSingleInputOutputs(learner, trainX, trainX.copy("numpyarray"))
        except TypeError as TE:
            # at issue could be the the parameter 'y' or 'Y' - .lower covers both
            # cases. Sometimes our target string is prefaced with the object
            # in question, so we use in instead of ==
            target = "fit() missing 1 required positional argument: 'y'"
            assert target in str(TE).lower()
        except ValueError as VE:
            # Similar to above, yet the param is provided as None
            target = "estimator requires y to be passed, but the target y is None"
            assert target in str(VE)


@sklSkipDec
@pytest.mark.slow
def testSciKitLearnSparsePCATransformation():
    # do not accept sparse matrices
    trainX = nimble.random.data(100, 10, sparsity=0.9)
    Xtrain = trainX._data

    learners = ['MiniBatchSparsePCA', 'SparsePCA',]

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
        sklObj = skl.findCallable(learner)
        #TODO explore why ridge_alpha defaults to 'deprecated'
        sciKitLearnObj = sklObj(ridge_alpha=0.1)
        arguments = setupSKLArguments(sciKitLearnObj)
        sciKitLearnObj.fit(Xtrain)
        transSKL = sciKitLearnObj.transform(Xtrain)
        transSKL = nimble.data(transSKL, useLog=False)

        arguments['ridge_alpha'] = 0.1
        seed = adjustRandomParamForNimble(arguments)
        TL = nimble.train(toCall(learner), trainX, arguments=arguments,
                          randomSeed=seed)
        transNimble = TL.apply(trainX)
        transSL = _apply_saveLoad(TL, trainX)

        equalityAssertHelper(transSKL, transNimble, transSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
@pytest.mark.slow
def testSciKitLearnOtherFitTransformLearners():
    data = generateClassificationData(2, 20, 10)
    trainX = abs(data[0][0])
    Xtrain = trainX._data


    skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
    transform = getLearnersByType('UNKNOWN', learnersRequired=False)
    learners = [t for t in transform if hasattr(skl.findCallable(t), 'fit_transform')]

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        arguments = setupSKLArguments(sciKitLearnObj)

        transSKL = sciKitLearnObj.fit_transform(Xtrain)
        transSKL = nimble.data(transSKL, useLog=False)

        seed = adjustRandomParamForNimble(arguments)
        TL = nimble.train(toCall(learner), trainX, arguments=arguments,
                          randomSeed=seed)
        transNimble = TL.apply(trainX)
        transSL = _apply_saveLoad(TL, trainX)

        equalityAssertHelper(transSKL, transNimble, transSL)

    for learner in learners:
        compareOutputs(learner)


@sklSkipDec
def testSciKitLearnTextVectorizers():
    data = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
    trainX = nimble.data(data)
    Xtrain = data


    skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')
    learners = ['CountVectorizer', 'TfidfVectorizer']

    @logCountAssertionFactory(3)
    def compareOutputs(learner):
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()

        transSKL = sciKitLearnObj.fit_transform(Xtrain)
        transSKL = nimble.data(transSKL, useLog=False)

        TL = nimble.train(toCall(learner), trainX)
        transNimble = TL.apply(trainX)
        transSL = _apply_saveLoad(TL, trainX)

        equalityAssertHelper(transSKL, transNimble, transSL)

    for learner in learners:
        compareOutputs(learner)

@sklSkipDec
@logCountAssertionFactory(3)
def testSciKitLearnPatchExtractor():
    data = np.array([[[ 2, 19, 13], [ 3, 18, 13], [ 7, 20, 13], [ 8, 21, 14]],
                     [[ 1, 18, 12], [ 3, 18, 13], [ 7, 20, 13], [ 8, 21, 14]],
                     [[ 2, 17, 12], [ 6, 19, 12], [ 7, 20, 13], [ 7, 20, 13]],
                     [[ 3, 18, 13], [ 7, 20, 13], [ 7, 20, 13], [ 5, 20, 13]]])
    trainX = nimble.data(data, useLog=False)
    Xtrain = data

    skl = nimble.core._learnHelpers.findBestInterface('scikitlearn')


    sklObj = skl.findCallable('PatchExtractor')
    sciKitLearnObj = sklObj(patch_size=(2, 2))
    sciKitLearnObj.fit(Xtrain)
    transSKL = sciKitLearnObj.transform(Xtrain)
    transSKL = nimble.data(transSKL, useLog=False)

    TL = nimble.train(toCall('PatchExtractor'), trainX, patch_size=(2, 2))
    transNimble = TL.apply(trainX)
    transSL = _apply_saveLoad(TL, trainX)

    equalityAssertHelper(transSKL, transNimble, transSL)

@sklSkipDec
@logCountAssertionFactory(4)
def testCustomRidgeRegressionCompare():
    """ Sanity check for custom RidgeRegression, compare results to SKL's Ridge """
    data = [[0, 1, 2], [13, 12, 4], [345, 233, 76]]
    trainObj = nimble.data(data, useLog=False)

    data2 = [[122, 34], [76, -3]]
    testObj = nimble.data(data2, useLog=False)

    name = 'nimble.RidgeRegression'
    TL = nimble.train(name, trainX=trainObj, trainY=0, arguments={'lamb': 1})
    ret1 = TL.apply(testObj)
    ret2 = nimble.trainAndApply("Scikitlearn.Ridge", trainX=trainObj, trainY=0, testX=testObj,
                             arguments={'alpha': 1, 'fit_intercept': False})
    ret3 = _apply_saveLoad(TL, testObj)

    equalityAssertHelper(ret1, ret2, ret3)


@sklSkipDec
@logCountAssertionFactory(4)
def testCustomRidgeRegressionCompareRandomized():
    """ Sanity check for custom RidgeRegression, compare results to SKL's Ridge on random data"""
    trainObj = nimble.random.data(1000, 60, .1, useLog=False)
    testObj = nimble.random.data(100, 59, .1, useLog=False)

    name = 'nimble.RidgeRegression'
    TL = nimble.train(name, trainX=trainObj, trainY=0, arguments={'lamb': 1})
    ret1 = TL.apply(testObj)
    ret2 = nimble.trainAndApply("Scikitlearn.Ridge", trainX=trainObj, trainY=0, testX=testObj,
                             arguments={'alpha': 1, 'fit_intercept': False})
    ret3 = _apply_saveLoad(TL, testObj)

    equalityAssertHelper(ret1, ret2, ret3)


@sklSkipDec
@pytest.mark.slow
@logCountAssertionFactory(4)
def testCustomKNNClassficationCompareRandomized():
    """ Sanity check on custom KNNClassifier, compare to SKL's KNeighborsClassifier on random data"""
    trainX, ignore, trainY = generateClusteredPoints(5, 50, 5, addFeatureNoise=True, addLabelNoise=False,
                                                     addLabelColumn=False)
    testX, ignore, testY = generateClusteredPoints(5, 5, 5, addFeatureNoise=True, addLabelNoise=False,
                                                   addLabelColumn=False)

    cusname = 'nimble.KNNClassifier'
    sklname = "Scikitlearn.KNeighborsClassifier"
    TL = nimble.train(cusname, trainX, trainY=trainY, k=5)
    ret1 = TL.apply(testX)
    ret2 = nimble.trainAndApply(sklname, trainX, trainY=trainY, testX=testX, n_neighbors=5, algorithm='brute')
    ret3 = _apply_saveLoad(TL, testX)

    equalityAssertHelper(ret1, ret2, ret3)


@sklSkipDec
@pytest.mark.slow
def testGetAttributesCallable():
    """ Demonstrate getAttributes will work for each learner (with default params) """
    cData = generateClassificationData(2, 10, 5)
    ((cTrainX, cTrainY), (cTestX, cTestY)) = cData
    rData = generateRegressionData(2, 10, 5)
    ((rTrainX, rTrainY), (rTestX, rTestY)) = rData
    printExceptions = False

    text = ['CountVectorizer', 'TfidfVectorizer']
    allLearners = getLearnersByType(ignore=text)
    toTest = allLearners

    for learner in toTest:
        fullName = 'scikitlearn.' + learner
        lType = nimble.learnerType(fullName)
        if lType in ['classification', 'transformation', 'cluster', 'UNKNOWN']:
            X = cTrainX
            Y = cTrainY
        elif lType == 'regression':
            X = rTrainX
            Y = rTrainY
        else:
            raise ValueError('unexpected learnerType')

        try:
            tl = nimble.train(fullName, X, Y)
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
    trainObj = nimble.data(train, useLog=False)
    trainObj.features.retain([1, 2, 3, 4, 5], useLog=False)
    testObj = nimble.data(test, useLog=False)
    testObj.features.retain([2,3,4,5], useLog=False)

    # case1 trainY passed as integer
    assert trainObj[:,0].getTypeString() == "DataFrame"
    pred = nimble.trainAndApply('SciKitLearn.LogisticRegression', trainObj, 0,
                                testObj)

    #case2 trainY passed as nimble object
    trainY = trainObj.features.extract(0, useLog=False)
    assert trainY.getTypeString() == "DataFrame"
    pred = nimble.trainAndApply('SciKitLearn.LogisticRegression', trainObj,
                                trainY, testObj)

@sklSkipDec
@logCountAssertionFactory(3)
def test_applier_acceptsNewArguments():
    """ Test an skl function that accepts new arguments for transform """
    data = [[-1., -1.],
            [-1., -1.],
            [ 1.,  1.],
            [ 1.,  1.]]

    dataObj = nimble.data(data, useLog=False)

    # StandardScaler.transform takes a 'copy' argument. Default is None.
    tl = nimble.train('SciKitLearn.StandardScaler', dataObj)
    assert 'copy' not in tl._transformedArguments
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

    dataObj = nimble.data(data)

    # StandardScaler.transform does not takes a 'foo' argument
    tl = nimble.train('SciKitLearn.StandardScaler', dataObj)
    assert 'foo' not in tl._transformedArguments
    with raises(InvalidArgumentValue):
        # using arguments parameter
        transformed = tl.apply(dataObj, arguments={'foo': True})
    with raises(InvalidArgumentValue):
        # using kwarguments
        transformed = tl.apply(dataObj, foo=True)

@sklSkipDec
@logCountAssertionFactory(3)
def test_getScores_acceptsNewArguments():
    """ Test an skl function that accepts new arguments for predict_proba """
    train = [[1, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1],
             [1, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1]]
    testX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Need to set convertToType b/c conversion will not be done when check_input=False
    trainObj = nimble.data(train, convertToType=np.float32, useLog=False)
    testObj = nimble.data(testX, convertToType=np.float32, useLog=False)

    # DecisionTreeClassifier.predict_proba takes a 'check_input' argument. Default is True.
    tl = nimble.train('SciKitLearn.DecisionTreeClassifier', trainObj, 0)
    assert 'check_input' not in tl._transformedArguments
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

    trainObj = nimble.data(train, convertToType=np.float32)
    testObj = nimble.data(testX, convertToType=np.float32)

    # DecisionTreeClassifier.predict_proba does not take a 'foo' argument.
    tl = nimble.train('SciKitLearn.DecisionTreeClassifier', trainObj, 0)
    assert 'foo' not in tl._transformedArguments
    with raises(InvalidArgumentValue):
        # using arguments parameter
        transformed = tl.getScores(testObj, arguments={'foo': True})
    with raises(InvalidArgumentValue):
        # using kwarguments
        transformed = tl.getScores(testObj, foo=True)

def _apply_saveLoad(trainerLearnerObj, givenTestX):
    """
    Given a TrainedLearner object, return the results of apply after having
    saved then loaded the learner from a file.
    """
    with PortableNamedTempFileContext(suffix=".pickle") as tmpFile:
        trainerLearnerObj.save(tmpFile.name)
        trainer_ret_l = loadTrainedLearner(tmpFile.name)
        return trainer_ret_l.apply(givenTestX, useLog=False)

@sklSkipDec
@oneLogEntryExpected
def test_saveLoadTrainedLearner_logCount():
    train = [[1, -1, -3, -3, -1],
              [2, 0.4, -0.8, 0.2, -0.3],
              [3, 2, 1, 2, 4]]
    trainObj = nimble.data(train, useLog=False)

    tl = nimble.train('SciKitLearn.LogisticRegression', trainObj, 0, useLog=False)
    with PortableNamedTempFileContext(suffix=".pickle") as tmpFile:
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

def setupSKLArguments(sciKitLearnObj, extraArgs=None):
    arguments = {} if extraArgs is None else extraArgs
    if 'random_state' in sciKitLearnObj.get_params():
        arguments['random_state'] = generateSubsidiarySeed()
    sciKitLearnObj.set_params(**arguments)

    return arguments

def adjustRandomParamForNimble(arguments):
    # nimble uses randomSeed parameter
    if 'random_state' in arguments:
        seed = arguments['random_state']
        del arguments['random_state']
    else:
        seed = None
    return seed

@sklSkipDec
def testLearnerTypes():
    learners = ['skl.' + l for l in nimble.learnerNames('skl')]
    allowed = ['classification', 'regression', 'transformation', 'cluster',
               'UNKNOWN'] # TODO outlier classifiers are currently UNKNOWN
    assert all(lt in allowed for lt in nimble.learnerType(learners))
