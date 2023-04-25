import math
import builtins
from io import StringIO

import numpy as np
import pytest

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import PackageException
from nimble.core._learnHelpers import findBestInterface
from nimble.core._learnHelpers import sumAbsoluteDifference
from nimble.core._learnHelpers import generateClusteredPoints
from nimble.core._learnHelpers import computeMetrics
from nimble.calculate import rootMeanSquareError
from nimble.calculate import meanAbsoluteError
from nimble.random import pythonRandom
from tests.helpers import raises, patch

@pytest.mark.slow
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


def testGenerateClusteredPoints_handmade():
    clusterCount = 2
    pointsPer = 1
    featuresPer = 2

    dataset, labelsObj, _ = generateClusteredPoints(
        clusterCount, pointsPer, featuresPer, addFeatureNoise=False,
        addLabelNoise=False, addLabelColumn=False)

    dataExp = [[0, 0], [-1,-1]]
    labelsExp = [[0],[1]]

    assert dataset == nimble.data(dataExp)
    assert labelsObj == nimble.data(labelsExp)

    clusterCount = 5
    pointsPer = 1
    featuresPer = 6

    dataset, labelsObj, _ = generateClusteredPoints(
        clusterCount, pointsPer, featuresPer, addFeatureNoise=False,
        addLabelNoise=False, addLabelColumn=False)

    dataExp = [[0,0,0,0,0,0],[-1,-1,0,0,0,0],[0,0,2,2,0,0],[0,0,-3,-3,0,0],[0,0,0,0,4,4]]
    labelsExp = [[0],[1],[2],[3],[4]]

    assert dataset == nimble.data(dataExp)
    assert labelsObj == nimble.data(labelsExp)

def testGenerateClusteredPoints():
    """tests that the shape of data produced by generateClusteredPoints() is predictable and that the noisiness of the data
    matches that requested via the addFeatureNoise and addLabelNoise flags"""
    clusterCount = 3
    pointsPer = 10
    featuresPer = 5

    dataset, _, noiselessLabels = generateClusteredPoints(
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

    dataset, _, noiselessLabels = generateClusteredPoints(
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
    dataset, _, noiselessLabels = generateClusteredPoints(
        clusterCount, pointsPer, featuresPer, addFeatureNoise=False,
        addLabelNoise=False, addLabelColumn=False)
    labelColumnlessRows, labelColumnlessCols = len(dataset.points), len(dataset.features)
    #columnLess should have one less column in the DATASET, rows should be the same
    assert (labelColumnlessCols - feats == -1)
    assert (labelColumnlessRows - pts == 0)

    #test that generated points have plausible values expected from the nature of generateClusteredPoints
    allNoiseDataset, _, noiselessLabels = generateClusteredPoints(
        clusterCount, pointsPer, featuresPer, addFeatureNoise=True,
        addLabelNoise=True, addLabelColumn=True)

    # map cluster numbers to indices where they will be non-zero in generated
    # points.
    nonZeroIds = {0:[0,1,2], 1:[0,1,2], 2:[3,4]}

    pts, feats = len(allNoiseDataset.points), len(allNoiseDataset.features)
    for curRow in range(pts):
        currentClusterNumber = math.floor(curRow / pointsPer)
        for curCol in range(feats):
            #assert dataset has no noise for all entries
            assert (allNoiseDataset[curRow, curCol] % 1 > 0.0000000001)

            # last column is a noisy label
            if curCol != feats - 1:
                check = curCol in nonZeroIds[currentClusterNumber]
                expectedNoiselessValue = currentClusterNumber if check else 0
                if currentClusterNumber % 2 == 1:
                    expectedNoiselessValue *= -1
            else:
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
    matrix1 = nimble.data(data1)
    matrix2 = nimble.data(data2)

    with raises(InvalidArgumentValueCombination):
        result = sumAbsoluteDifference(matrix1, matrix2)

    data1 = [[0, 0, 1], [1, 0, 2], [0, 1, 3], [0, 0, 1], [1, 0, 2], [0, 1, 3], [0, 0, 1], [1, 0, 2], [0, 1, 3],
             [0, 0, 1], [1, 0, 2], [0, 1, 3], [0, 0, 1], [1, 0, 2], [0, 1, 3], [0, 0, 3], [1, 0, 1], [0, 1, 2]]
    data2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    matrix1 = nimble.data(data1)
    matrix2 = nimble.data(data2)

    with raises(InvalidArgumentValueCombination):
        result = sumAbsoluteDifference(matrix1, matrix2)

    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    data2 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    matrix1 = nimble.data(data1)
    matrix2 = nimble.data(data2)

    # this should work
    result = sumAbsoluteDifference(matrix1, matrix2)

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
    matrix1 = nimble.data(data1)
    matrix2 = nimble.data(data2)
    diffResult = sumAbsoluteDifference(matrix1, matrix2)
    shouldBe = 18 * 0.1 * 2
    # 18 entries, discrepencies of 0.1 in the first column, and the last column
    discrepencyEffectivelyZero = (diffResult - shouldBe) < .000000001 and (diffResult - shouldBe) > -.0000000001
    if not discrepencyEffectivelyZero:
        raise AssertionError("difference result should be " + str(18 * 0.1 * 2) + ' but it is ' + str(diffResult))
    # assert(diffResult == 18 * 0.1 * 2)


def test_computeMetrics_1d_2arg():
    knownLabels = np.array([[1.0], [2.0], [3.0]])
    predictedLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels)
    predictedLabelsMatrix = nimble.data(source=predictedLabels)

    metricFunctions = rootMeanSquareError
    result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
    assert result == 0.0

    knownLabels = np.array([[1.5], [2.5], [3.5]])
    predictedLabels = np.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels)
    predictedLabelsMatrix = nimble.data(source=predictedLabels)

    metricFunctions = meanAbsoluteError
    result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
    assert result > 0.49
    assert result < 0.51


def test_computeMetrics_1d_labelsInData():
    training = np.array([[1.0, 5], [2.0, 27], [3.0, 42]])
    predictedLabels = np.array([[1.0], [2.0], [3.0]])

    trainingObj = nimble.data(source=training)
    predictedObj = nimble.data(source=predictedLabels)

    metricFunctions = rootMeanSquareError
    result = computeMetrics(0, trainingObj, predictedObj, metricFunctions)
    assert result == 0.0

    result = computeMetrics([0], trainingObj, predictedObj, metricFunctions)
    assert result == 0.0


# multi val, two arg metric
def test_computeMetrics_2d_2arg():
    #import pdb; pdb.set_trace()
    knownLabels = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    predictedLabels = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    knownLabelsMatrix = nimble.data(source=knownLabels)
    predictedLabelsMatrix = nimble.data(source=predictedLabels)

    metricFunctions = nimble.calculate.meanFeaturewiseRootMeanSquareError
    result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
    assert isinstance(result, float)
    assert result == 0.0


def test_computeMetrics_2d_labelsInData():
    training = np.array([[1.0, 5, 1.0], [2.0, 27, 2.0], [3.0, 42, 3.0]])
    predictedLabels = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    trainingObj = nimble.data(source=training)
    predictedObj = nimble.data(source=predictedLabels)

    metricFunctions = nimble.calculate.meanFeaturewiseRootMeanSquareError
    result = computeMetrics([0, 2], trainingObj, predictedObj, metricFunctions)
    assert result == 0.0


# TODO multi val, three arg metric. do we have one of those?

# single val, symetric two arg metric (similarity)
def test_computeMetrics_1d_2d_symmetric():
    origData = np.array([[1.0], [2.0], [3.0]])
    outputData = np.array([[1.0], [2.0], [3.0]])

    origObj = nimble.data(source=origData)
    outObj = nimble.data(source=outputData)

    metricFunctions = nimble.calculate.cosineSimilarity
    result = computeMetrics(origObj, None, outObj, metricFunctions)
    assert result == 1.0

# multi metrics should not be allowed
@raises(TypeError)
def test_computeMetrics_multiple_metrics_disallowed():
    origData = np.array([[1.0], [2.0], [3.0]])
    outputData = np.array([[1.0], [2.0], [3.0]])

    origObj = nimble.data(source=origData)
    outObj = nimble.data(source=outputData)

    metricFunctions = [nimble.calculate.cosineSimilarity, rootMeanSquareError]
    computeMetrics(origObj, None, outObj, metricFunctions)

def back_show(func, *args):
    output = StringIO()
    printer = print

    def mockPrint(*args, **kwargs):
        kwargs['file'] = output
        printer(*args, **kwargs)

    with patch(builtins, 'print', mockPrint):
        func(*args)

    output.seek(0)
    ret = output.readlines()
    assert ret

    return ret

def test_showLearnerNames():
    contents = back_show(nimble.showLearnerNames)
    for line in contents:
        assert '.' in line

    contents = back_show(nimble.showLearnerNames, 'nimble')
    for line in contents:
        assert '.' not in line

def test_showLearnerParameters():
    contents = back_show(nimble.showLearnerParameters, 'nimble.KNNClassifier')
    assert 'k\n' in contents

    contents = back_show(nimble.showLearnerParameters, 'nimble.RidgeRegression')
    assert 'lamb\n' in contents

def test_showLearnerParameterDefaults():
    contents = back_show(nimble.showLearnerParameterDefaults, 'nimble.KNNClassifier')
    splits = [content.split() for content in contents]
    assert ['k', '5'] in splits

    contents = back_show(nimble.showLearnerParameterDefaults, 'nimble.RidgeRegression')
    splits = [content.split() for content in contents]
    assert ['lamb', '0'] in splits
