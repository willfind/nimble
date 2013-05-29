"""
Script that uses 50K points of job posts to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
    from UML import crossValidateReturnBest
    from UML import functionCombinations
    from UML.combinations.Combinations import executeCode
    from UML import runAndTest
    from UML import data
    from UML import loadTrainingAndTesting
    from UML.performance.metric_functions import classificationError
    from UML.performance.metric_functions import bottomProportionPercentNegative10
    from UML.performance.metric_functions import proportionPercentNegative50
    from UML.performance.metric_functions import proportionPercentNegative90

    pathIn = "UML/applications/example_data/tfIdfApproval50K.mtx"
    trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID=0, fractionForTestSet=.2, loadType="CooSparseData", fileType="mtx")
    print "Finished loading data"
    print "trainX shape: " + str(trainX.data.shape)
    print "trainY shape: " + str(trainY.data.shape)

    # sparse types aren't playing nice with the error metrics currently, so convert
    trainY = trainY.toDenseMatrixData()
    testY = testY.toDenseMatrixData()

    trainYList = []
    
    for i in range(len(trainY.data)):
        label = trainY.data[i][0]
        trainYList.append([int(label)])

    testYList = []
    for i in range(len(testY.data)):
        label = testY.data[i][0]
        testYList.append([int(label)])

    trainY = data('dense', trainYList)
    testY = data('dense', testYList)

    print "Finished converting labels to ints"


    # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    scores = runAndTest("shogun.MulticlassLibLinear", trainX, testX, trainY, testY, {"C":0.75}, [proportionPercentNegative90, proportionPercentNegative50], scoreMode="allScores", negativeLabel="2", sendToLog=False)

    print "top 90 proportion Rejected: " + str(scores)
    