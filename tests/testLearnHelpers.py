import math
from math import fabs

import numpy
from nose.tools import *
from nose.plugins.attrib import attr

import nimble
from nimble import learnerType
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction
from nimble.exceptions import PackageException
from nimble.core._learnHelpers import findBestInterface
from nimble.core._learnHelpers import FoldIterator
from nimble.core._learnHelpers import sumAbsoluteDifference
from nimble.core._learnHelpers import generateClusteredPoints
from nimble.core._learnHelpers import computeMetrics
from nimble.calculate import rootMeanSquareError
from nimble.calculate import meanAbsoluteError
from nimble.calculate import fractionIncorrect
from nimble.random import pythonRandom

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
            return nimble.data("List", source=data, featureNames=featureNames)

        super(TestList, self).__init__(maker)


class TestMatrix(FoldIteratorTester):
    def __init__(self):
        def maker(data, featureNames=False):
            return nimble.data("Matrix", source=data, featureNames=featureNames)

        super(TestMatrix, self).__init__(maker)


class TestSparse(FoldIteratorTester):
    def __init__(self):
        def maker(data, featureNames=False):
            return nimble.data("Sparse", source=data, featureNames=featureNames)

        super(TestSparse, self).__init__(maker)


class TestRand(FoldIteratorTester):
    def __init__(self):
        def maker(data, featureNames=False):
            possible = ['List', 'Matrix', 'Sparse']
            returnType = possible[pythonRandom.randint(0, 2)]
            return nimble.data(returnType=returnType, source=data, featureNames=featureNames)

        super(TestRand, self).__init__(maker)


@attr('slow')
def testClassifyAlgorithms(printResultsDontThrow=False):
    """tries the algorithm names (which are keys in knownAlgorithmToTypeHash) with learnerType().
    Next, compares the result to the algorithm's associated value in knownAlgorithmToTypeHash.
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

    dataset, labelsObj, noiselessLabels = generateClusteredPoints(
        clusterCount, pointsPer, featuresPer, addFeatureNoise=True,
        addLabelNoise=True, addLabelColumn=True)
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

    dataset, labelsObj, noiselessLabels = generateClusteredPoints(
        clusterCount, pointsPer, featuresPer, addFeatureNoise=False,
        addLabelNoise=False, addLabelColumn=True)
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
    dataset, labelsObj, noiselessLabels = generateClusteredPoints(
        clusterCount, pointsPer, featuresPer, addFeatureNoise=False,
        addLabelNoise=False, addLabelColumn=False)
    labelColumnlessRows, labelColumnlessCols = len(dataset.points), len(dataset.features)
    #columnLess should have one less column in the DATASET, rows should be the same
    assert (labelColumnlessCols - feats == -1)
    assert (labelColumnlessRows - pts == 0)


    #test that generated points have plausible values expected from the nature of generateClusteredPoints
    allNoiseDataset, labsObj, noiselessLabels = generateClusteredPoints(
        clusterCount, pointsPer, featuresPer, addFeatureNoise=True,
        addLabelNoise=True, addLabelColumn=True)
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
    """ Function verifies that for different shaped matricies, generated via nimble.data, sumAbsoluteDifference() throws an InvalidArgumentValueCombination."""

    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    data2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 0]]
    matrix1 = nimble.data('Matrix', data1)
    matrix2 = nimble.data('Matrix', data2)
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
    matrix1 = nimble.data('Matrix', data1)
    matrix2 = nimble.data('Matrix', data2)
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
    matrix1 = nimble.data('Matrix', data1)
    matrix2 = nimble.data('Matrix', data2)
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
    matrix1 = nimble.data('Matrix', data1)
    matrix2 = nimble.data('Matrix', data2)
    diffResult = sumAbsoluteDifference(matrix1, matrix2)
    shouldBe = 18 * 0.1 * 2
    # 18 entries, discrepencies of 0.1 in the first column, and the last column
    discrepencyEffectivelyZero = (diffResult - shouldBe) < .000000001 and (diffResult - shouldBe) > -.0000000001
    if not discrepencyEffectivelyZero:
        raise AssertionError("difference result should be " + str(18 * 0.1 * 2) + ' but it is ' + str(diffResult))
    # assert(diffResult == 18 * 0.1 * 2)


def test_computeMetrics_1d_2arg():
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    metricFunctions = rootMeanSquareError
    result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
    assert result == 0.0

    knownLabels = numpy.array([[1.5], [2.5], [3.5]])
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    metricFunctions = meanAbsoluteError
    result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
    assert result > 0.49
    assert result < 0.51


def test_computeMetrics_1d_labelsInData():
    training = numpy.array([[1.0, 5], [2.0, 27], [3.0, 42]])
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])

    trainingObj = nimble.data('Matrix', source=training)
    predictedObj = nimble.data('Matrix', source=predictedLabels)

    metricFunctions = rootMeanSquareError
    result = computeMetrics(0, trainingObj, predictedObj, metricFunctions)
    assert result == 0.0

    result = computeMetrics([0], trainingObj, predictedObj, metricFunctions)
    assert result == 0.0


# multi val, two arg metric
def test_computeMetrics_2d_2arg():
    knownLabels = numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    predictedLabels = numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    knownLabelsMatrix = nimble.data('Matrix', source=knownLabels)
    predictedLabelsMatrix = nimble.data('Matrix', source=predictedLabels)

    metricFunctions = nimble.calculate.meanFeaturewiseRootMeanSquareError
    result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
    assert isinstance(result, float)
    assert result == 0.0


def test_computeMetrics_2d_labelsInData():
    training = numpy.array([[1.0, 5, 1.0], [2.0, 27, 2.0], [3.0, 42, 3.0]])
    predictedLabels = numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    trainingObj = nimble.data('Matrix', source=training)
    predictedObj = nimble.data('Matrix', source=predictedLabels)

    metricFunctions = nimble.calculate.meanFeaturewiseRootMeanSquareError
    result = computeMetrics([0, 2], trainingObj, predictedObj, metricFunctions)
    assert result == 0.0


# TODO multi val, three arg metric. do we have one of those?

# single val, symetric two arg metric (similarity)
def test_computeMetrics_1d_2d_symmetric():
    origData = numpy.array([[1.0], [2.0], [3.0]])
    outputData = numpy.array([[1.0], [2.0], [3.0]])

    origObj = nimble.data('Matrix', source=origData)
    outObj = nimble.data('Matrix', source=outputData)

    metricFunctions = nimble.calculate.cosineSimilarity
    result = computeMetrics(origObj, None, outObj, metricFunctions)
    assert result == 1.0

# multi metrics should not be allowed
@raises(TypeError)
def test_computeMetrics_multiple_metrics_disallowed():
    origData = numpy.array([[1.0], [2.0], [3.0]])
    outputData = numpy.array([[1.0], [2.0], [3.0]])

    origObj = nimble.data('Matrix', source=origData)
    outObj = nimble.data('Matrix', source=outputData)

    metricFunctions = [nimble.calculate.cosineSimilarity, rootMeanSquareError]
    computeMetrics(origObj, None, outObj, metricFunctions)