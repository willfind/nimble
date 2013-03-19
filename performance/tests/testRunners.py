from UML import data
from UML.performance.runner import runAndTestOneVsOne
from UML.performance.runner import extractWinningPredictionLabel
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




