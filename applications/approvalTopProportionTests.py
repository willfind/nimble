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
    toRun = 'runAndTest("shogun.MulticlassOCAS", trainX, testX, trainY, testY, {"C":<0.1|0.5|0.75|1.0|5.0>}, <[proportionPercentNegative90]|[proportionPercentNegative50]>, scoreMode="allScores", negativeLabel="2", sendToLog=False)'
    runs = functionCombinations(toRun)
    extraParams = {'runAndTest':runAndTest, 'proportionPercentNegative90':proportionPercentNegative90, 'proportionPercentNegative50':proportionPercentNegative50}
    run, results = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)

    # for run in runs:
    dataHash={"trainX": trainX.duplicate(), 
              "testX":testX.duplicate(), 
              "trainY":trainY.duplicate(), 
              "testY":testY.duplicate(), 
              'runAndTest':runAndTest, 
              'proportionPercentNegative90':proportionPercentNegative90,
              'proportionPercentNegative50':proportionPercentNegative50}
    #   print "Run call: "+repr(run)
    print "Best run code: " + str(run)
    print "Best Run confirmation: "+repr(executeCode(run, dataHash))

