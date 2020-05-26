import math
from math import fabs

import numpy
from nose.tools import *
from nose.plugins.attrib import attr

import nimble
from nimble import learnerType
from nimble import createData
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction
from nimble.exceptions import PackageException
from nimble.helpers import extractWinningPredictionLabel
from nimble.helpers import generateAllPairs
from nimble.helpers import findBestInterface
from nimble.helpers import FoldIterator
from nimble.helpers import sumAbsoluteDifference
from nimble.helpers import generateClusteredPoints
from nimble.helpers import _mergeArguments
from nimble.helpers import computeMetrics
from nimble.helpers import inspectArguments
from nimble.utility import numpy2DArray, is2DArray
from nimble.calculate import rootMeanSquareError
from nimble.calculate import meanAbsoluteError
from nimble.calculate import fractionIncorrect
from nimble.randomness import pythonRandom

##########
# TESTER #
##########

class FoldIteratorTester(object):
    def __init__(self, constructor):
        self.constructor = constructor

    @raises(InvalidArgumentValueCombination)
    def test_makeFoldIterator_exceptionPEmpty(self):
        """ Test makeFoldIterator() for InvalidArgumentValueCombination when object is point empty """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        FoldIterator([toTest], 2)

    #	@raises(ImproperActionException)
    #	def test_makeFoldIterator_exceptionFEmpty(self):
    #		""" Test makeFoldIterator() for ImproperActionException when object is feature empty """
    #		data = [[],[]]
    #		data = numpy.array(data)
    #		toTest = self.constructor(data)
    #		makeFoldIterator([toTest],2)


    @raises(InvalidArgumentValue)
    def test_makeFoldIterator_exceptionTooManyFolds(self):
        """ Test makeFoldIterator() for exception when given too many folds """
        data = [[1], [2], [3], [4], [5]]
        names = ['col']
        toTest = self.constructor(data, names)
        FoldIterator([toTest, toTest], 6)


    def test_makeFoldIterator_verifyPartitions(self):
        """ Test makeFoldIterator() yields the correct number folds and partitions the data """
        data = [[1], [2], [3], [4], [5]]
        names = ['col']
        toTest = self.constructor(data, names)
        folds = FoldIterator([toTest], 2)

        [(fold1Train, fold1Test)] = next(folds)
        [(fold2Train, fold2Test)] = next(folds)

        try:
            next(folds)
            assert False
        except StopIteration:
            pass

        assert len(fold1Train.points) + len(fold1Test.points) == 5
        assert len(fold2Train.points) + len(fold2Test.points) == 5

        fold1Train.points.append(fold1Test)
        fold2Train.points.append(fold2Test)

    def test_makeFoldIterator_verifyPartitions_Unsupervised(self):
        """ Test makeFoldIterator() yields the correct number folds and partitions the data, with a None data """
        data = [[1], [2], [3], [4], [5]]
        names = ['col']
        toTest = self.constructor(data, names)
        folds = FoldIterator([toTest, None], 2)

        [(fold1Train, fold1Test), (fold1NoneTrain, fold1NoneTest)] = next(folds)
        [(fold2Train, fold2Test), (fold2NoneTrain, fold2NoneTest)] = next(folds)

        try:
            next(folds)
            assert False
        except StopIteration:
            pass

        assert len(fold1Train.points) + len(fold1Test.points) == 5
        assert len(fold2Train.points) + len(fold2Test.points) == 5

        fold1Train.points.append(fold1Test)
        fold2Train.points.append(fold2Test)

        assert fold1NoneTrain is None
        assert fold1NoneTest is None
        assert fold2NoneTrain is None
        assert fold2NoneTest is None


    def test_makeFoldIterator_verifyMatchups(self):
        """ Test makeFoldIterator() maintains the correct pairings when given multiple data objects """
        data0 = [[1], [2], [3], [4], [5], [6], [7]]
        toTest0 = self.constructor(data0)

        data1 = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]
        toTest1 = self.constructor(data1)

        data2 = [[-1], [-2], [-3], [-4], [-5], [-6], [-7]]
        toTest2 = self.constructor(data2)

        folds = FoldIterator([toTest0, toTest1, toTest2], 2)

        fold0 = next(folds)
        fold1 = next(folds)
        [(fold0Train0, fold0Test0), (fold0Train1, fold0Test1), (fold0Train2, fold0Test2)] = fold0
        [(fold1Train0, fold1Test0), (fold1Train1, fold1Test1), (fold1Train2, fold1Test2)] = fold1

        try:
            next(folds)
            assert False
        except StopIteration:
            pass

        # check that the partitions are the right size (ie, no overlap in training and testing)
        assert len(fold0Train0.points) + len(fold0Test0.points) == 7
        assert len(fold1Train0.points) + len(fold1Test0.points) == 7

        assert len(fold0Train1.points) + len(fold0Test1.points) == 7
        assert len(fold1Train1.points) + len(fold1Test1.points) == 7

        assert len(fold0Train2.points) + len(fold0Test2.points) == 7
        assert len(fold1Train2.points) + len(fold1Test2.points) == 7

        # check that the data is in the same order accross objects, within
        # the training or testing sets of a single fold
        for fold in [fold0, fold1]:
            trainList = []
            testList = []
            for (train, test) in fold:
                trainList.append(train)
                testList.append(test)

            for train in trainList:
                assert len(train.points) == len(trainList[0].points)
                for index in range(len(train.points)):
                    assert fabs(train[index, 0]) == fabs(trainList[0][index, 0])

            for test in testList:
                assert len(test.points) == len(testList[0].points)
                for index in range(len(test.points)):
                    assert fabs(test[index, 0]) == fabs(testList[0][index, 0])


class TestList(FoldIteratorTester):
    def __init__(self):
        def maker(data=None, featureNames=False):
            return nimble.createData("List", data=data, featureNames=featureNames)

        super(TestList, self).__init__(maker)


class TestMatrix(FoldIteratorTester):
    def __init__(self):
        def maker(data, featureNames=False):
            return nimble.createData("Matrix", data=data, featureNames=featureNames)

        super(TestMatrix, self).__init__(maker)


class TestSparse(FoldIteratorTester):
    def __init__(self):
        def maker(data, featureNames=False):
            return nimble.createData("Sparse", data=data, featureNames=featureNames)

        super(TestSparse, self).__init__(maker)


class TestRand(FoldIteratorTester):
    def __init__(self):
        def maker(data, featureNames=False):
            possible = ['List', 'Matrix', 'Sparse']
            returnType = possible[pythonRandom.randint(0, 2)]
            return nimble.createData(returnType=returnType, data=data, featureNames=featureNames)

        super(TestRand, self).__init__(maker)


@attr('slow')
def testClassifyAlgorithms(printResultsDontThrow=False):
    """tries the algorithm names (which are keys in knownAlgorithmToTypeHash) with learnerType().
    Next, compares the result to the algorithm's assocaited value in knownAlgorithmToTypeHash.
    If the algorithm types don't match, an AssertionError is thrown."""

    knownAlgorithmToTypeHash = {'nimble.KNNClassifier': 'classification',
                                'nimble.RidgeRegression': 'regression',
    }
    try:
        findBestInterface('sciKitLearn')
        knownAlgorithmToTypeHash['sciKitLearn.RadiusNeighborsClassifier'] = 'classification'
        knownAlgorithmToTypeHash['sciKitLearn.RadiusNeighborsRegressor'] = 'regression'
    except PackageException:
        pass
    try:
        findBestInterface('mlpy')
        knownAlgorithmToTypeHash['mlpy.LDAC'] = 'classification'
        knownAlgorithmToTypeHash['mlpy.Ridge'] = 'regression'
    except PackageException:
        pass
    try:
        findBestInterface('shogun')
        knownAlgorithmToTypeHash['shogun.MulticlassOCAS'] = 'classification'
        knownAlgorithmToTypeHash['shogun.LibSVR'] = 'regression'
    except PackageException:
        pass

    for curAlgorithm in knownAlgorithmToTypeHash.keys():
        actualType = knownAlgorithmToTypeHash[curAlgorithm]
        predictedType = nimble.learnerType(curAlgorithm)
        try:
            assert (actualType in predictedType)
        except AssertionError:
            errorString = 'Classification failure. Classified ' + curAlgorithm + ' as ' + predictedType + ', when it really is a ' + actualType
            if printResultsDontThrow:
                print(errorString)
            else:
                raise AssertionError(errorString)
        else:
            if printResultsDontThrow:
                print('Passed test for ' + curAlgorithm)


def testGenerateClusteredPoints():
    """tests that the shape of data produced by generateClusteredPoints() is predictable and that the noisiness of the data
    matches that requested via the addFeatureNoise and addLabelNoise flags"""
    clusterCount = 3
    pointsPer = 10
    featuresPer = 5

    dataset, labelsObj, noiselessLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer,
                                                                  addFeatureNoise=True, addLabelNoise=True,
                                                                  addLabelColumn=True)
    pts, feats = len(noiselessLabels.points), len(noiselessLabels.features)
    for i in range(pts):
        for j in range(feats):
            #assert that the labels don't have noise in noiselessLabels
            assert (noiselessLabels[i, j] % 1 == 0.0)

    pts, feats = len(dataset.points), len(dataset.features)
    for i in range(pts):
        for j in range(feats):
            #assert dataset has noise for all entries
            assert (dataset[i, j] % 1 != 0.0)

    dataset, labelsObj, noiselessLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer,
                                                                  addFeatureNoise=False, addLabelNoise=False,
                                                                  addLabelColumn=True)
    pts, feats = len(noiselessLabels.points), len(noiselessLabels.features)
    for i in range(pts):
        for j in range(feats):
            #assert that the labels don't have noise in noiselessLabels
            assert (noiselessLabels[i, j] % 1 == 0.0)

    pts, feats = len(dataset.points), len(dataset.features)
    for i in range(pts):
        for j in range(feats):
            #assert dataset has no noise for all entries
            assert (dataset[i, j] % 1 == 0.0)

    #test that addLabelColumn flag works
    dataset, labelsObj, noiselessLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer,
                                                                  addFeatureNoise=False, addLabelNoise=False,
                                                                  addLabelColumn=False)
    labelColumnlessRows, labelColumnlessCols = len(dataset.points), len(dataset.features)
    #columnLess should have one less column in the DATASET, rows should be the same
    assert (labelColumnlessCols - feats == -1)
    assert (labelColumnlessRows - pts == 0)


    #test that generated points have plausible values expected from the nature of generateClusteredPoints
    allNoiseDataset, labsObj, noiselessLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer,
                                                                        addFeatureNoise=True, addLabelNoise=True,
                                                                        addLabelColumn=True)
    pts, feats = len(allNoiseDataset.points), len(allNoiseDataset.features)
    for curRow in range(pts):
        for curCol in range(feats):
            #assert dataset has no noise for all entries
            assert (allNoiseDataset[curRow, curCol] % 1 > 0.0000000001)

            currentClusterNumber = math.floor(curRow / pointsPer)
            expectedNoiselessValue = currentClusterNumber
            absoluteDifference = abs(allNoiseDataset[curRow, curCol] - expectedNoiselessValue)
            #assert that the noise is reasonable:
            assert (absoluteDifference < 0.01)


def testSumDifferenceFunction():
    """ Function verifies that for different shaped matricies, generated via createData, sumAbsoluteDifference() throws an InvalidArgumentValueCombination."""

    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    data2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 0]]
    matrix1 = createData('Matrix', data1)
    matrix2 = createData('Matrix', data2)
    failedShape = False
    try:
        result = sumAbsoluteDifference(matrix1, matrix2)
    except InvalidArgumentValueCombination:
        failedShape = True
    assert (failedShape)

    data1 = [[0, 0, 1], [1, 0, 2], [0, 1, 3], [0, 0, 1], [1, 0, 2], [0, 1, 3], [0, 0, 1], [1, 0, 2], [0, 1, 3],
             [0, 0, 1], [1, 0, 2], [0, 1, 3], [0, 0, 1], [1, 0, 2], [0, 1, 3], [0, 0, 3], [1, 0, 1], [0, 1, 2]]
    data2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    matrix1 = createData('Matrix', data1)
    matrix2 = createData('Matrix', data2)
    failedShape = False
    try:
        result = sumAbsoluteDifference(matrix1, matrix2)
    except InvalidArgumentValueCombination:
        failedShape = True
    assert (failedShape)

    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    data2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    matrix1 = createData('Matrix', data1)
    matrix2 = createData('Matrix', data2)
    failedShape = False
    try:
        result = sumAbsoluteDifference(matrix1, matrix2)
    except InvalidArgumentValueCombination:
        failedShape = True
    assert (failedShape is False)

    #asserts differece function gets absolute difference correct (18 * 0.1 * 2)
    #18 rows
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    data2 = [[1 - 0.1, 0, 0, 1.1], [0 - 0.1, 1, 0, 2.1], [0 - 0.1, 0, 1, 3.1], [1 - 0.1, 0, 0, 1.1],
             [0 - 0.1, 1, 0, 2.1], [0 - 0.1, 0, 1, 3.1], [1 - 0.1, 0, 0, 1.1], [0 - 0.1, 1, 0, 2.1],
             [0 - 0.1, 0, 1, 3.1], [1 - 0.1, 0, 0, 1.1], [0 - 0.1, 1, 0, 2.1], [0 - 0.1, 0, 1, 3.1],
             [1 - 0.1, 0, 0, 1.1], [0 - 0.1, 1, 0, 2.1], [0 - 0.1, 0, 1, 3.1], [1 - 0.1, 0, 0, 3.1],
             [0 - 0.1, 1, 0, 1.1], [0 - 0.1, 0, 1, 2.1]]
    matrix1 = createData('Matrix', data1)
    matrix2 = createData('Matrix', data2)
    diffResult = sumAbsoluteDifference(matrix1, matrix2)
    shouldBe = 18 * 0.1 * 2
    # 18 entries, discrepencies of 0.1 in the first column, and the last column
    discrepencyEffectivelyZero = (diffResult - shouldBe) < .000000001 and (diffResult - shouldBe) > -.0000000001
    if not discrepencyEffectivelyZero:
        raise AssertionError("difference result should be " + str(18 * 0.1 * 2) + ' but it is ' + str(diffResult))
    # assert(diffResult == 18 * 0.1 * 2)

@raises(InvalidArgumentValueCombination)
def testMergeArgumentsException():
    """ Test helpers._mergeArguments will throw the exception it should """
    args = {1: 'a', 2: 'b', 3: 'd'}
    kwargs = {1: 1, 2: 'b'}

    _mergeArguments(args, kwargs)


def testMergeArgumentsHand():
    """ Test helpers._mergeArguments is correct on hand construsted data """
    args = {1: 'a', 2: 'b', 3: 'd'}
    kwargs = {1: 'a', 4: 'b'}

    ret = _mergeArguments(args, kwargs)

    assert ret == {1: 'a', 2: 'b', 3: 'd', 4: 'b'}

def testMergeArgumentsMakesCopy():
    """Test helpers._mergeArguments does not modify or return either original dict"""
    emptyA = {}
    emptyB = {}
    dictA = {'foo': 'bar'}
    dictB = {'bar': 'foo'}

    merged0 = _mergeArguments(None, emptyB)
    assert emptyB == merged0
    assert id(emptyB) != id(merged0)
    assert emptyB == {}

    merged1 = _mergeArguments(emptyA, emptyB)
    assert emptyA == merged1
    assert emptyB == merged1
    assert id(emptyA) != id(merged1)
    assert id(emptyB) != id(merged1)
    assert emptyA == {}
    assert emptyB == {}

    merged2 = _mergeArguments(dictA, emptyB)
    assert dictA == merged2
    assert id(dictA) != id(merged2)
    assert dictA == {'foo': 'bar'}
    assert emptyB == {}

    merged3 = _mergeArguments(emptyA, dictB)
    assert dictB == merged3
    assert id(dictB) != id(merged3)
    assert emptyA == {}
    assert dictB == {'bar': 'foo'}

    merged4 = _mergeArguments(dictA, dictB)
    assert dictA != merged4
    assert dictB != merged4
    assert dictA == {'foo': 'bar'}
    assert dictB == {'bar': 'foo'}


def test_computeMetrics_1d_2arg():
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    metricFunctions = rootMeanSquareError
    result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
    assert result == 0.0

    knownLabels = numpy.array([[1.5], [2.5], [3.5]])
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    metricFunctions = meanAbsoluteError
    result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
    assert result > 0.49
    assert result < 0.51


def test_computeMetrics_1d_labelsInData():
    training = numpy.array([[1.0, 5], [2.0, 27], [3.0, 42]])
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])

    trainingObj = createData('Matrix', data=training)
    predictedObj = createData('Matrix', data=predictedLabels)

    metricFunctions = rootMeanSquareError
    result = computeMetrics(0, trainingObj, predictedObj, metricFunctions)
    assert result == 0.0

    result = computeMetrics([0], trainingObj, predictedObj, metricFunctions)
    assert result == 0.0


# multi val, two arg metric
def test_computeMetrics_2d_2arg():
    knownLabels = numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    predictedLabels = numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    metricFunctions = nimble.calculate.meanFeaturewiseRootMeanSquareError
    result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
    assert isinstance(result, float)
    assert result == 0.0


def test_computeMetrics_2d_labelsInData():
    training = numpy.array([[1.0, 5, 1.0], [2.0, 27, 2.0], [3.0, 42, 3.0]])
    predictedLabels = numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    trainingObj = createData('Matrix', data=training)
    predictedObj = createData('Matrix', data=predictedLabels)

    metricFunctions = nimble.calculate.meanFeaturewiseRootMeanSquareError
    result = computeMetrics([0, 2], trainingObj, predictedObj, metricFunctions)
    assert result == 0.0


# TODO multi val, three arg metric. do we have one of those?

# single val, symetric two arg metric (similarity)
def test_computeMetrics_1d_2d_symmetric():
    origData = numpy.array([[1.0], [2.0], [3.0]])
    outputData = numpy.array([[1.0], [2.0], [3.0]])

    origObj = createData('Matrix', data=origData)
    outObj = createData('Matrix', data=outputData)

    metricFunctions = nimble.calculate.cosineSimilarity
    result = computeMetrics(origObj, None, outObj, metricFunctions)
    assert result == 1.0

# multi metrics should not be allowed
@raises(TypeError)
def test_computeMetrics_multiple_metrics_disallowed():
    origData = numpy.array([[1.0], [2.0], [3.0]])
    outputData = numpy.array([[1.0], [2.0], [3.0]])

    origObj = createData('Matrix', data=origData)
    outObj = createData('Matrix', data=outputData)

    metricFunctions = [nimble.calculate.cosineSimilarity, rootMeanSquareError]
    computeMetrics(origObj, None, outObj, metricFunctions)


def testExtractWinningPredictionLabel():
    """
    Unit test for extractWinningPrediction function in runner.py
    """
    predictionData = [[1, 3, 3, 2, 3, 2], [2, 3, 3, 2, 2, 2], [1, 1, 1, 1, 1, 1], [4, 4, 4, 3, 3, 3]]
    BaseObj = createData('Matrix', predictionData)
    BaseObj.transpose()
    predictions = BaseObj.features.calculate(extractWinningPredictionLabel)
    listPredictions = predictions.copy(to="python list")

    assert listPredictions[0][0] - 3 == 0.0
    assert listPredictions[0][1] - 2 == 0.0
    assert listPredictions[0][2] - 1 == 0.0
    assert (listPredictions[0][3] - 4 == 0.0) or (listPredictions[0][3] - 3 == 0.0)


def testGenerateAllPairs():
    """
    Unit test function for testGenerateAllPairs
    """
    testList1 = [1, 2, 3, 4]
    testPairs = generateAllPairs(testList1)
    print(testPairs)

    assert len(testPairs) == 6
    assert ((1, 2) in testPairs) or ((2, 1) in testPairs)
    assert not (((1, 2) in testPairs) and ((2, 1) in testPairs))
    assert ((1, 3) in testPairs) or ((3, 1) in testPairs)
    assert not (((1, 3) in testPairs) and ((3, 1) in testPairs))
    assert ((1, 4) in testPairs) or ((4, 1) in testPairs)
    assert not (((1, 4) in testPairs) and ((4, 1) in testPairs))
    assert ((2, 3) in testPairs) or ((3, 2) in testPairs)
    assert not (((2, 3) in testPairs) and ((3, 2) in testPairs))
    assert ((2, 4) in testPairs) or ((4, 2) in testPairs)
    assert not (((2, 4) in testPairs) and ((4, 2) in testPairs))
    assert ((3, 4) in testPairs) or ((4, 3) in testPairs)
    assert not (((3, 4) in testPairs) and ((4, 3) in testPairs))

    testList2 = []
    testPairs2 = generateAllPairs(testList2)
    assert testPairs2 is None

def test_inspectArguments():

    def checkSignature(a, b, c, d=False, e=True, f=None, *sigArgs, **sigKwargs):
        pass

    a, v, k, d = inspectArguments(checkSignature)

    assert a == ['a', 'b', 'c', 'd', 'e', 'f']
    assert v == 'sigArgs'
    assert k == 'sigKwargs'
    assert d == (False, True, None)

def test_numpy2DArray_converts1D():
    raw = [1, 2, 3, 4]
    ret = numpy2DArray(raw)
    fromNumpy = numpy.array(raw)
    assert len(ret.shape) == 2
    assert not numpy.array_equal(ret,fromNumpy)

@raises(InvalidArgumentValue)
def test_numpy2DArray_dimensionException():
    raw = [[[1, 2], [3, 4]]]
    ret = numpy2DArray(raw)

def test_is2DArray():
    raw1D = [1, 2, 3]
    arr1D = numpy.array(raw1D)
    mat1D = numpy.matrix(raw1D)
    assert not is2DArray(arr1D)
    assert is2DArray(mat1D)
    raw2D = [[1, 2, 3]]
    arr2D = numpy.array(raw2D)
    mat2D = numpy.matrix(raw2D)
    assert is2DArray(arr2D)
    assert is2DArray(mat2D)
    raw3D = [[[1, 2, 3]]]
    arr3D = numpy.array(raw3D)
    assert not is2DArray(arr3D)
