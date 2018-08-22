"""
Unit tests for scikit_learn_interface.py

"""

from __future__ import absolute_import
import numpy.testing
from nose.plugins.attrib import attr
import importlib
import inspect

import UML

from UML.interfaces.tests.test_helpers import checkLabelOrderingAndScoreAssociations

from UML.helpers import generateClusteredPoints

from UML.randomness import numpyRandom
from UML.randomness import generateSubsidiarySeed
from UML.exceptions import ArgumentException
from UML.helpers import generateClassificationData
from UML.helpers import generateRegressionData
from UML.calculate.loss import rootMeanSquareError
from UML.interfaces.scikit_learn_interface import SciKitLearn

from sklearn.metrics import mean_squared_error

scipy = UML.importModule('scipy.sparse')

packageName = 'sciKitLearn'


def toCall(learner):
    return packageName + '.' + learner


def testScikitLearnAliases():
    """ Test availability of correct aliases for 'sciKitLearn' """
    variables = ["Y", "x1", "x2"]
    data = [[2, 1, 1], [3, 1, 2], [4, 2, 2], ]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)

    data2 = [[0, 1]]
    testObj = UML.createData('Matrix', data2)

    # make a bundle of calls, don't care about the results, only
    # that they work.
    UML.trainAndApply("scikitlearn.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    UML.trainAndApply("SKL.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    UML.trainAndApply("skl.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})
    UML.trainAndApply("SciKitLearn.LinearRegression", trainingObj, trainY="Y", testX=testObj, arguments={})


def testSciKitLearnHandmadeRegression():
    """ Test sciKitLearn() by calling on a regression learner with known output """
    variables = ["Y", "x1", "x2"]
    data = [[2, 1, 1], [3, 1, 2], [4, 2, 2], ]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)

    data2 = [[0, 1]]
    testObj = UML.createData('Matrix', data2)

    ret = UML.trainAndApply(toCall("LinearRegression"), trainingObj, trainY="Y", testX=testObj, output=None,
                            arguments={})

    assert ret is not None

    expected = [[1.]]
    expectedObj = UML.createData('Matrix', expected)

    numpy.testing.assert_approx_equal(ret[0, 0], 1.)


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
    obj = UML.createData('Sparse', A)
    testObj = obj.copy()
    testObj.extractFeatures(cols[0])

    ret = UML.trainAndApply(toCall('SGDRegressor'), trainX=obj, trainY=cols[0], testX=testObj)

    assert ret is not None


def testSciKitLearnHandmadeClustering():
    """ Test sciKitLearn() by calling a clustering classifier with known output """
    variables = ["x1", "x2"]
    data = [[1, 0], [3, 3], [5, 0], ]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)

    data2 = [[1, 0], [1, 1], [5, 1], [3, 4]]
    testObj = UML.createData('Matrix', data2)

    ret = UML.trainAndApply(toCall("KMeans"), trainingObj, testX=testObj, output=None, arguments={'n_clusters': 3})

    # clustering returns a row vector of indices, referring to the cluster centers,
    # we don't care about the exact numbers, this verifies that the appropriate
    # ones are assigned to the same clusters
    assert ret[0, 0] == ret[1, 0]
    assert ret[0, 0] != ret[2, 0]
    assert ret[0, 0] != ret[3, 0]
    assert ret[2, 0] != ret[3, 0]


def testSciKitLearnHandmadeSparseClustering():
    """ Test sciKitLearn() by calling on a sparse clustering learner with known output """
    if not scipy:
        return
    trainData = scipy.sparse.lil_matrix((3, 3))
    trainData[0, :] = [2, 3, 1]
    trainData[1, :] = [2, 2, 1]
    trainData[2, :] = [0, 0, 0]
    trainData = UML.createData('Sparse', data=trainData)

    testData = scipy.sparse.lil_matrix((3, 2))
    testData[0, :] = [3, 3]
    testData[1, :] = [3, 2]
    testData[2, :] = [-1, 0]
    testData = UML.createData('Sparse', data=testData)

    ret = UML.trainAndApply(toCall('MiniBatchKMeans'), trainData, trainY=2, testX=testData, arguments={'n_clusters': 2})

    assert ret[0, 0] == ret[1, 0]
    assert ret[0, 0] != ret[2, 0]


def testSciKitLearnScoreMode():
    """ Test sciKitLearn() scoreMode flags"""
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2)

    # default scoreMode is 'label'
    ret = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
    assert ret.points == 2
    assert ret.features == 1

    bestScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                   scoreMode='bestScore')
    assert bestScores.points == 2
    assert bestScores.features == 2

    allScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                  scoreMode='allScores')
    assert allScores.points == 2
    assert allScores.features == 3

    checkLabelOrderingAndScoreAssociations([0, 1, 2], bestScores, allScores)


def testSciKitLearnScoreModeBinary():
    """ Test sciKitLearn() scoreMode flags, binary case"""
    variables = ["Y", "x1", "x2"]
    data = [[1, 30, 2], [2, 1, 1], [2, 0, 1], [2, -1, -1], [1, 30, 3], [1, 34, 4]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)

    data2 = [[2, 1], [25, 0]]
    testObj = UML.createData('Matrix', data2)

    # default scoreMode is 'label'
    ret = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
    assert ret.points == 2
    assert ret.features == 1

    bestScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                   scoreMode='bestScore')
    assert bestScores.points == 2
    assert bestScores.features == 2

    allScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={},
                                  scoreMode='allScores')
    assert allScores.points == 2
    assert allScores.features == 2

    checkLabelOrderingAndScoreAssociations([1, 2], bestScores, allScores)


def testSciKitLearnCrossDecomp():
    """ Test SKL on learners which take 2d Y data """
    variables = ["x1", "x2"]
    data = [[1, 0], [3, 3], [50, 0], [12, 3], [8, 228]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)
    dataY = [[0, 1], [0, 1], [2, 2], [1, 30], [5, 21]]
    trainingYObj = UML.createData('Matrix', dataY)

    data2 = [[1, 0], [1, 1], [5, 1], [34, 4]]
    testObj = UML.createData('Matrix', data2)

    UML.trainAndApply(toCall("CCA"), trainingObj, testX=testObj, trainY=trainingYObj)
    UML.trainAndApply(toCall("PLSCanonical"), trainingObj, testX=testObj, trainY=trainingYObj)
    UML.trainAndApply(toCall("PLSRegression"), trainingObj, testX=testObj, trainY=trainingYObj)
    UML.trainAndApply(toCall("PLSSVD"), trainingObj, testX=testObj, trainY=trainingYObj)


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


def getLearnersByType(lType, ignore=[]):
    learners = UML.listLearners(packageName)
    typeMatch = []
    for learner in learners:
        learnerType = UML.learnerType(toCall(learner))
        if lType == learnerType and learner not in ignore:
            typeMatch.append(learner)
    return typeMatch


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

    for learner in learners:
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
        predSKL = UML.createData('Matrix', predSKL.reshape(-1,1))
        predUML = UML.trainAndApply(toCall(learner), trainX, trainY, testX, arguments=arguments)

        assert predUML.isApproximatelyEqual(predSKL)


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

    for learner in learners:
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
        predSKL = UML.createData('Matrix', predSKL.reshape(-1,1))
        predUML = UML.trainAndApply(toCall(learner), trainX, trainY, testX, arguments=arguments)

        assert predUML.isApproximatelyEqual(predSKL)


@attr('slow')
def testSciKitLearnMultiTaskRegressionLearners():
    """ Test that predictions for from UML.trainAndApply match predictions from scikitlearn
    multitask learners with predict method"""

    skl = SciKitLearn()

    trainX = [[0,0], [1, 1], [2, 2]]
    trainY = [[0, 0], [1, 1], [2, 2]]
    testX = [[2,2], [0,0], [1,1]]

    trainXObj = UML.createData('Matrix', trainX)
    trainYObj = UML.createData('Matrix', trainY)
    testXObj = UML.createData('Matrix', testX)

    multiTaskLearners = ['MultiTaskElasticNet', 'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV']

    for learner in multiTaskLearners:
        predictionUML = UML.trainAndApply(toCall(learner),trainX=trainXObj, trainY=trainYObj, testX=testXObj)
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        sciKitLearnObj.fit(trainX, trainY)
        predictionSciKit = sciKitLearnObj.predict(testX)
        # convert to UML data object for comparison
        predictionSciKit = UML.createData('Matrix', predictionSciKit)

        assert predictionUML.isIdentical(predictionSciKit)


@attr('slow')
def testSciKitLearnClusterLearners():
    data = generateClusteredPoints(3, 60, 8)
    data = data[0]
    data.shufflePoints()
    trainX = data[:50,:]
    testX = data[50:,:]
    Xtrain = trainX.data
    Xtest = testX.data

    learners = getLearnersByType('cluster')

    for learner in learners:
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
        predSKL = UML.createData('Matrix', predSKL.reshape(-1,1))
        predUML = UML.trainAndApply(toCall(learner), trainX, testX=testX, arguments=arguments)

        assert predUML.isIdentical(predSKL)


@attr('slow')
def testSciKitLearnOtherPredictLearners():
    data = generateClassificationData(2, 20, 10)
    trainX = abs(data[0][0])
    trainY = abs(data[0][1])
    testX = abs(data[1][0])
    Xtrain = trainX.data
    Ytrain = trainY.data
    Xtest = testX.data

    ignore = ['TSNE', 'MDS', 'SpectralEmbedding', 'RandomTreesEmbedding',] # special cases, tested elsewhere
    learners = getLearnersByType('other', ignore)

    for learner in learners:
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
        predSKL = UML.createData('Matrix', predSKL.reshape(-1,1))
        predUML = UML.trainAndApply(toCall(learner), trainX, trainY, testX, arguments=arguments)

        assert predUML.isApproximatelyEqual(predSKL)


@attr('slow')
def testSciKitLearnTransformationLearners():

    data = generateClassificationData(2, 20, 10)
    trainX = abs(data[0][0])
    trainY = abs(data[0][1])
    Xtrain = trainX.data
    Ytrain = trainY.data

    ignore = ['GaussianRandomProjection', 'SparseRandomProjection',   # special cases, tested elsewhere
              'MiniBatchSparsePCA', 'SparsePCA', 'NMF']
    learners = getLearnersByType('transformation', ignore)

    for learner in learners:
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
        transSKL = UML.createData('Matrix', transSKL)
        transUML = UML.trainAndApply(toCall(learner), trainX, trainY, arguments=arguments)

        assert transUML.isApproximatelyEqual(transSKL)


@attr('slow')
def testSciKitLearnRandomProjectionTransformation():
    trainX = UML.createRandomData('Matrix', 10, 10000, 0.98)
    Xtrain = trainX.data

    learners = ['GaussianRandomProjection', 'SparseRandomProjection',]

    for learner in learners:
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)

        transSKL = sciKitLearnObj.fit_transform(Xtrain)
        transSKL = UML.createData('Matrix', transSKL)
        transUML = UML.trainAndApply(toCall(learner), trainX, arguments=arguments)

        assert transUML.isApproximatelyEqual(transSKL)

@attr('slow')
def testSciKitLearnSparsePCATransformation():
    # do not accept sparse matrices
    trainX = UML.createRandomData('Matrix', 100, 10, sparsity=0.9)
    Xtrain = trainX.data

    learners = ['MiniBatchSparsePCA', 'SparsePCA',]

    for learner in learners:
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
        transSKL = UML.createData('Matrix', transSKL)
        transUML = UML.trainAndApply(toCall(learner), trainX, arguments=arguments)

        assert transUML.isApproximatelyEqual(transSKL)

@attr('slow')
def testSciKitLearnEmbeddingLearners():
    data = generateClassificationData(2, 20, 10)
    trainX = abs(data[0][0])
    Xtrain = trainX.data

    learners = ['TSNE', 'MDS', 'SpectralEmbedding', 'RandomTreesEmbedding',]

    for learner in learners:
        skl = SciKitLearn()
        sklObj = skl.findCallable(learner)
        sciKitLearnObj = sklObj()
        seed = UML.randomness.generateSubsidiarySeed()
        arguments = {}
        if 'random_state' in sciKitLearnObj.get_params():
            arguments['random_state'] = seed
            sciKitLearnObj.set_params(**arguments)

        transSKL = sciKitLearnObj.fit_transform(Xtrain)
        transSKL = UML.createData('Matrix', transSKL)
        transUML = UML.trainAndApply(toCall(learner), trainX, arguments=arguments)

        assert transUML.isApproximatelyEqual(transSKL)


def testSciKitLearnNMF():
    # must be non-negative matrix
    Xtrain = [[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]]
    trainX = UML.createData('Matrix', Xtrain)

    skl = SciKitLearn()
    sklObj = skl.findCallable('NMF')
    sciKitLearnObj = sklObj()
    seed = UML.randomness.generateSubsidiarySeed()
    arguments = {}
    arguments['random_state'] = seed
    sciKitLearnObj.set_params(**arguments)
    sciKitLearnObj.fit(Xtrain)
    transSKL = sciKitLearnObj.transform(Xtrain)
    transSKL = UML.createData('Matrix', transSKL)
    transUML = UML.trainAndApply(toCall('NMF'), trainX, arguments=arguments)

    assert transUML.isApproximatelyEqual(transSKL)


def testCustomRidgeRegressionCompare():
    """ Sanity check for custom RidgeRegression, compare results to SKL's Ridge """
    data = [[0, 1, 2], [13, 12, 4], [345, 233, 76]]
    trainObj = UML.createData('Matrix', data)

    data2 = [[122, 34], [76, -3]]
    testObj = UML.createData('Matrix', data2)

    name = 'Custom.RidgeRegression'
    ret1 = UML.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj, arguments={'lamb': 1})
    ret2 = UML.trainAndApply("Scikitlearn.Ridge", trainX=trainObj, trainY=0, testX=testObj,
                             arguments={'alpha': 1, 'fit_intercept': False})

    assert ret1.isApproximatelyEqual(ret2)


def testCustomRidgeRegressionCompareRandomized():
    """ Sanity check for custom RidgeRegression, compare results to SKL's Ridge on random data"""
    trainObj = UML.createRandomData("Matrix", 1000, 60, .1)
    testObj = UML.createRandomData("Matrix", 100, 59, .1)

    name = 'Custom.RidgeRegression'
    ret1 = UML.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj, arguments={'lamb': 1})
    ret2 = UML.trainAndApply("Scikitlearn.Ridge", trainX=trainObj, trainY=0, testX=testObj,
                             arguments={'alpha': 1, 'fit_intercept': False})

    assert ret1.isApproximatelyEqual(ret2)


@attr('slow')
def testCustomKNNClassficationCompareRandomized():
    """ Sanity check on custom KNNClassifier, compare to SKL's KNeighborsClassifier on random data"""
    trainX, ignore, trainY = generateClusteredPoints(5, 50, 5, addFeatureNoise=True, addLabelNoise=False,
                                                     addLabelColumn=False)
    testX, ignore, testY = generateClusteredPoints(5, 5, 5, addFeatureNoise=True, addLabelNoise=False,
                                                   addLabelColumn=False)

    cusname = 'Custom.KNNClassifier'
    sklname = "Scikitlearn.KNeighborsClassifier"
    ret1 = UML.trainAndApply(cusname, trainX, trainY=trainY, testX=testX, k=5)
    ret2 = UML.trainAndApply(sklname, trainX, trainY=trainY, testX=testX, n_neighbors=5, algorithm='brute')

    assert ret1.isApproximatelyEqual(ret2)


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
        except ArgumentException as ae:
            if printExceptions:
                print (learner + " : " + lType)
                print(ae)
        tl.getAttributes()
