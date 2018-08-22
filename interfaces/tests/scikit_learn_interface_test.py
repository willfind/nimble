"""
Unit tests for scikit_learn_interface.py

129/153 learners tested

17/153 untested due to various errors
    # sklearn.feature_extraction
    ['DictVectorizer', 'PatchExtractor', 'CountVectorizer', 'TfidfTransformer',
     'TfidfVectorizer', 'FeatureHasher']

    # sklearn.preprocessing
    ['LabelBinarizer', 'LabelEncoder', 'StandardScaler',
     'KernelCenterer', 'MultiLabelBinarizer',]

    # sklearn.decomposition
    ['RandomizedPCA', 'SparsePCA', 'MiniBatchSparsePCA',]

    # sklearn.dummy
    ['DummyClassifier']

7/153 untest due to not having a predict or transform method
    # sklearn.manifold
    ['MDS', 'SpectralEmbedding', 'TSNE']
    # sklearn.cluster
    ['AgglomerativeClustering', 'DBSCAN', 'FeatureAgglomeration', 'SpectralClustering']
    # sklearn.neighbors
    ['LocalOutlierFactor']

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


def testSciKitLearnUnsupervisedProblemLearners():
    """ Test sciKitLearn() by calling some unsupervised learners problematic in previous implementations """
    variables = ["x1", "x2"]
    data = [[1, 0], [3, 3], [50, 0]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)

    data2 = [[1, 0], [1, 1], [5, 1], [34, 4]]
    testObj = UML.createData('Matrix', data2)

    UML.trainAndApply(toCall("GMM"), trainingObj, testX=testObj)
    UML.trainAndApply(toCall("DPGMM"), trainingObj, testX=testObj)
    UML.trainAndApply(toCall("VBGMM"), trainingObj, testX=testObj)


#def testSciKitLearnObsAsArgumentName():
#	""" Test scikitLearn() by calling learners with 'obs' instead of 'X' as a fit/predict argument """
#	data = [[1,3,], [2,3], [0,1], [3,0], [3,2]]
#	trainingObj = UML.createData('Matrix', data)

#	data2 = [[2,3],[1,2],[0,1], [1,3]]
#	testObj = UML.createData('Matrix', data2)

#	ret = UML.trainAndApply(toCall("GMMHMM"), trainingObj, testX=testObj, arguments={'n_components':3})
#	ret = UML.trainAndApply(toCall("GaussianHMM"), trainingObj, testX=testObj)
#	ret = UML.trainAndApply(toCall("MultinomialHMM"), trainingObj, testX=testObj)

def testSciKitLearnArgspecFailures():
    """ Test scikitLearn() on those learners that cannot be passed to inspect.getargspec """
    variables = ["x1", "x2"]
    data = [[1, 0], [3, 3], [50, 0]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)

    dataY = [[0], [1], [2]]
    trainingYObj = UML.createData('Matrix', dataY)

    data2 = [[1, 0], [1, 1], [5, 1], [34, 4]]
    testObj = UML.createData('Matrix', data2)

    UML.trainAndApply(toCall("GaussianNB"), trainingObj, testX=testObj, trainY=trainingYObj)
    # data dependent?
    UML.trainAndApply(toCall("MultinomialNB"), trainingObj, testX=testObj, trainY=trainingYObj)


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


@attr('slow')
def testSciKitLearnPredictAndTransformLearners():
    """ Test that output from UML.trainAndApply match output from scikitlearn
    learners directly"""

    skl = SciKitLearn()

    ((rTrainX, rTrainY) , (rTestX, rTestY)) = generateRegressionData(2, 20, 10)
    ((cTrainX, cTrainY) , (cTestX, cTestY)) = generateClassificationData(2, 20, 10)
    # some classification learners cannot handle negative data
    cTrainX, cTrainY, cTestX, cTestY = abs(cTrainX), abs(cTrainY), abs(cTestX), abs(cTestY)

    learners = UML.listLearners('scikitlearn')

    exclude = ['MultiTaskElasticNet', 'MultiTaskElasticNetCV', 'MultiTaskLasso',
               'MultiTaskLassoCV', 'MiniBatchSparsePCA', 'RandomizedPCA',
               'SparsePCA', 'PatchExtractor', 'FeatureHasher', 'KernelCenterer',
               'StandardScaler', 'LabelBinarizer', 'LabelEncoder',
               'DummyClassifier', 'DictVectorizer', 'CountVectorizer',
               'TfidfVectorizer', 'MultiLabelBinarizer', 'SparseRandomProjection',
               'GaussianRandomProjection', 'LinearDiscriminantAnalysis',
               'Normalizer', 'ZeroEstimator', 'ScaledLogOddsEstimator']

    for learner in learners:
        if learner not in exclude:
            sklObj = skl.findCallable(learner)
            fullName = 'scikitlearn.' + learner
            lType = UML.learnerType(fullName)
            if lType == 'regression':
                trainXObj = rTrainX
                trainYObj = rTrainY
                testXObj = rTestX
                trainX = rTrainX.data
                trainY = rTrainY.data
                testX = rTestX.data
            else:
                trainXObj = cTrainX
                trainYObj = cTrainY
                testXObj = cTestX
                trainX = cTrainX.data
                trainY = cTrainY.data
                testX = cTestX.data

            try:
                args, _, _, _  = inspect.getargspec(sklObj)
            except TypeError:
                args, _, _, _ = inspect.getargspec(sklObj.__init__)
            uml_kwds = {}
            uml_kwds['trainX'] = trainXObj
            uml_kwds['trainY'] = trainYObj
            if 'random_state' in args:
                seed = UML.randomness.generateSubsidiarySeed()
                sciKitLearnObj = sklObj(random_state=seed)
                uml_kwds['arguments'] = {'random_state':seed}
            else:
                sciKitLearnObj = sklObj()
            sciKitLearnObj.fit(trainX, trainY)

            if hasattr(sklObj, 'predict'):
                uml_kwds['testX'] = testXObj
                predictionUML = UML.trainAndApply(toCall(learner), **uml_kwds)
                predictionSciKit = sciKitLearnObj.predict(testX)
                predictionSciKit = UML.createData('Matrix', predictionSciKit.reshape(-1,1))

                assert predictionUML is not None
                try:    
                    assert predictionUML.isIdentical(predictionSciKit)
                except AssertionError:
                    assert predictionUML.isApproximatelyEqual(predictionSciKit)

            elif hasattr(sklObj, 'transform'):
                transArgs, _, _, _ = inspect.getargspec(sklObj.transform)
                skl_kwds = {}
                if 'X' in transArgs:
                    skl_kwds['X'] = trainX
                if 'y' in transArgs:
                    skl_kwds['y'] = trainY

                transformUML = UML.trainAndApply(toCall(learner), **uml_kwds)
                transformSciKit = sciKitLearnObj.transform(**skl_kwds)
                transformSciKit = UML.createData('Matrix', transformSciKit)

                assert transformUML is not None
                try:
                    assert transformUML.isIdentical(transformSciKit)
                except AssertionError:
                    assert transformUML.isApproximatelyEqual(transformSciKit)


def testSciKitLearnMultiTaskLearners():
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

    allLearners = UML.listLearners('scikitlearn')
    toTest = allLearners

    for learner in toTest:
    #		print learner
        fullName = 'scikitlearn.' + learner
        lType = UML.learnerType(fullName)
        if lType == 'classification':
            try:
                tl = UML.train(fullName, cTrainX, cTrainY)
            # this is meant to safely bypass those learners that have required arguments
            except ArgumentException as ae:
                pass
            # this is generally how shogun explodes
            except SystemError as se:
                pass
            tl.getAttributes()
        if lType == 'regression':
            try:
                tl = UML.train(fullName, rTrainX, rTrainY)
            # this is meant to safely bypass those learners that have required arguments
            except ArgumentException as ae:
                pass
            except SystemError as se:
                pass
            tl.getAttributes()

def testConvertYTrainDType():
    """ test trainY dtype is converted to float when learner requires Y to be numeric"""
    train = [['a', 1, -1, -3, -3, -1],
              ['b', 2, 0.4, -0.8, 0.2, -0.3],
              ['c', 3, 2, 1, 2, 4]]
    test = [['a', 1, -2, -1, -3, -2],
            ['c', 3, 1, 2, 3, 1]]
    # object will have 'object' dtype because of strings in data
    trainObj = UML.createData('Matrix', train)
    trainObj.retainFeatures([1, 2, 3, 4, 5])
    testObj = UML.createData('Matrix', test)
    testObj.retainFeatures([2,3,4,5])

    # case1 trainY passed as integer
    assert trainObj[:,0].data.dtype == numpy.object_
    pred = UML.trainAndApply('SciKitLearn.LogisticRegression', trainObj, 0, testObj)

    #case2 trainY passed as UML object
    trainY = trainObj.extractFeatures(0)
    assert trainY.data.dtype == numpy.object_
    pred = UML.trainAndApply('SciKitLearn.LogisticRegression', trainObj, trainY, testObj)
