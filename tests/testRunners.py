from UML import createData
from UML.runners import runAndTestOneVsOne
from UML.runners import runOneVsOne
from UML.runners import runOneVsAll
from UML.umlHelpers import extractWinningPredictionLabel
from UML.umlHelpers import generateAllPairs
from UML.metrics import fractionIncorrect


def testRunAndTestOneVsOne():
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    data2 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [0,1,1,4], [0,1,1,4], [0,1,1,4], [0,1,1,4], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    trainObj1 = createData('Matrix', data1, variables)
    trainObj2 = createData('Matrix', data2, variables)

    testData1 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3]]
    testData2 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3], [0, 1, 1, 2]]
    testObj1 = createData('Matrix', testData1)
    testObj2 = createData('Matrix', testData2)

    metricFuncs = []
    metricFuncs.append(fractionIncorrect)

    results1 = runAndTestOneVsOne('sciKitLearn.SVC', trainObj1, testObj1, trainDependentVar=3,  arguments={}, performanceMetricFuncs=metricFuncs)
    results2 = runAndTestOneVsOne('sciKitLearn.SVC', trainObj2, testObj2, trainDependentVar=3,  arguments={}, performanceMetricFuncs=metricFuncs)

    assert results1['fractionIncorrect'] == 0.0
    assert results2['fractionIncorrect'] == 0.25

def testRunOneVsAll():
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    data2 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [0,1,1,4], [0,1,1,4], [0,1,1,4], [0,1,1,4], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
    trainObj1 = createData('Sparse', data1, variables)
    trainObj2 = createData('Sparse', data2, variables)

    testData1 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3]]
    testData2 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3], [0, 1, 1, 2]]
    testObj1 = createData('Sparse', testData1)
    testObj2 = createData('Sparse', testData2)

    metricFuncs = []
    metricFuncs.append(fractionIncorrect)

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
    trainObj1 = createData('Matrix', data1, variables)

    testData1 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3]]
    testObj1 = createData('Matrix', testData1)

    metricFuncs = []
    metricFuncs.append(fractionIncorrect)

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
    BaseObj = createData('Matrix', predictionData)
    BaseObj.transpose()
    predictions = BaseObj.applyFunctionToEachFeature(extractWinningPredictionLabel)
    listPredictions = predictions.toListOfLists()
    
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
    print testPairs

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



