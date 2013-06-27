from UML import data
from UML import runAndTestOneVsOne
from UML import runOneVsOne
from UML import runOneVsAll
from UML.umlHelpers import extractWinningPredictionLabel
from UML.performance.metric_functions import classificationError


def testRunAndTestOneVsOne():
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    data2 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [0,1,1,4], [0,1,1,4], [0,1,1,4], [0,1,1,4], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    trainObj1 = data('DenseMatrixData', data1, variables)
    trainObj2 = data('DenseMatrixData', data2, variables)

    testData1 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3]]
    testData2 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3], [0, 1, 1, 2]]
    testObj1 = data('DenseMatrixData', testData1)
    testObj2 = data('DenseMatrixData', testData2)

    metricFuncs = []
    metricFuncs.append(classificationError)

    results1 = runAndTestOneVsOne('sciKitLearn.SVC', trainObj1, testObj1, trainDependentVar=3,  arguments={}, performanceMetricFuncs=metricFuncs)
    results2 = runAndTestOneVsOne('sciKitLearn.SVC', trainObj2, testObj2, trainDependentVar=3,  arguments={}, performanceMetricFuncs=metricFuncs)

    assert results1['classificationError'] == 0.0
    assert results2['classificationError'] == 0.25

def testRunOneVsAll():
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    data2 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [0,1,1,4], [0,1,1,4], [0,1,1,4], [0,1,1,4], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    trainObj1 = data('CooSparseData', data1, variables)
    trainObj2 = data('CooSparseData', data2, variables)

    testData1 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3]]
    testData2 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3], [0, 1, 1, 2]]
    testObj1 = data('CooSparseData', testData1)
    testObj2 = data('CooSparseData', testData2)

    metricFuncs = []
    metricFuncs.append(classificationError)

    results1 = runOneVsAll('sciKitLearn.LogisticRegression', trainObj1, testObj1, trainDependentVar=3,  arguments={}, scoreMode='label')
    results2 = runOneVsAll('sciKitLearn.LinearRegression', trainObj1.duplicate(), testObj1.duplicate(), trainDependentVar=3,  arguments={}, scoreMode='bestScore')
    results3 = runOneVsAll('sciKitLearn.LinearRegression', trainObj1.duplicate(), testObj1.duplicate(), trainDependentVar=3,  arguments={}, scoreMode='allScores')

    print "Results 1 output: " + str(results1.data)
    print "Results 2 output: " + str(results2.data)
    print "Results 3 output: " + str(results3.data)

    assert results1.toListOfLists()[0][0] >= 0.0
    assert results1.toListOfLists()[0][0] <= 3.0

    assert results2.toListOfLists()[0][0] 

def testRunOneVsOne():
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    trainObj1 = data('DenseMatrixData', data1, variables)

    testData1 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3]]
    testObj1 = data('DenseMatrixData', testData1)

    metricFuncs = []
    metricFuncs.append(classificationError)

    results1 = runOneVsOne('sciKitLearn.SVC', trainObj1.duplicate(), testObj1.duplicate(), trainDependentVar=3,  arguments={}, scoreMode='label')
    results2 = runOneVsOne('sciKitLearn.SVC', trainObj1.duplicate(), testObj1.duplicate(), trainDependentVar=3,  arguments={}, scoreMode='bestScore')
    results3 = runOneVsOne('sciKitLearn.SVC', trainObj1.duplicate(), testObj1.duplicate(), trainDependentVar=3,  arguments={}, scoreMode='allScores')

    assert results1.data[0][0] == 1.0
    assert results1.data[1][0] == 2.0
    assert results1.data[2][0] == 3.0
    assert len(results1.data) == 3

    assert results2.data[0][0] == 1.0
    assert results2.data[0][1] == 2
    assert results2.data[1][0] == 2.0
    assert results2.data[1][1] == 2
    assert results2.data[2][0] == 3.0
    assert results2.data[2][1] == 2

    results3FeatureMap = results3.featureNamesInverse
    for i in range(len(results3.data)):
        row = results3.data[i]
        for j in range(len(row)):
            score = row[j]
            if i == 0:
                if score == 2:
                    assert results3FeatureMap[j] == str(1)
            elif i == 1:
                if score == 2:
                    assert results3FeatureMap[j] == str(2)
            else:
                if score == 2:
                    assert results3FeatureMap[j] == str(3)


def testExtractWinningPredictionLabel():
    """
    Unit test for extractWinningPrediction function in runner.py
    """
    predictionData = [[1, 3, 3, 2, 3, 2], [2, 3, 3, 2, 2, 2], [1, 1, 1, 1, 1, 1], [4, 4, 4, 3, 3, 3]]
    baseDataObj = data('DenseMatrixData', predictionData)
    baseDataObj.transpose()
    predictions = baseDataObj.applyFunctionToEachFeature(extractWinningPredictionLabel)
    listPredictions = predictions.toListOfLists()
    
    assert listPredictions[0][0] - 3 == 0.0
    assert listPredictions[0][1] - 2 == 0.0
    assert listPredictions[0][2] - 1 == 0.0
    assert (listPredictions[0][3] - 4 == 0.0) or (listPredictions[0][3] - 3 == 0.0)




