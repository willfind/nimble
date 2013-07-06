"""
Script that uses Shogun's KNN implementation to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
    from UML import crossValidateReturnBest
    from UML import functionCombinations
    from UML.umlHelpers import executeCode
    from UML import runAndTest
    from UML import create
    from UML import loadTrainingAndTesting
    from UML.metrics import classificationError
    from UML.metrics import bottomProportionPercentNegative10
    from UML.metrics import proportionPercentNegative50
    from UML.metrics import proportionPercentNegative90

    pathIn = "UML/datasets/tfIdfApproval50K.mtx"
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

    trainY = create('dense', trainYList)
    testY = create('dense', testYList)

    print "Finished converting labels to ints"


    # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    toRun = 'runAndTest("shogun.KNN", trainX, testX, trainY, testY, {"k":<1|3|5|10|15|20|25>}, [classificationError],sendToLog=False)'
    runs = functionCombinations(toRun)
    extraParams = {'runAndTest':runAndTest, 'classificationError':classificationError}
    run, results = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)

    run = run.replace('sendToLog=False', 'sendToLog=True')
    dataHash={"trainX": trainX,
              "testX":testX, 
              "trainY":trainY, 
              "testY":testY, 
              'runAndTest':runAndTest, 
              'classificationError':classificationError}
    #   print "Run call: "+repr(run)
    print "Best run code: " + str(run)
    print "Best Run confirmation: "+repr(executeCode(run, dataHash))


    # # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    # toRun = 'runAndTest("shogun.KNN", trainX, testX, trainY, testY, {"k":<1|3|5|10|15|20|25>}, [proportionPercentNegative50], scoreMode="allScores", negativeLabel="2", sendToLog=False)'
    # runs = functionCombinations(toRun)
    # extraParams = {'runAndTest':runAndTest, 'proportionPercentNegative50':proportionPercentNegative50}
    # run, results = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)

    # run = run.replace('sendToLog=False', 'sendToLog=True')
    # dataHash={"trainX": trainX,
    #           "testX":testX, 
    #           "trainY":trainY, 
    #           "testY":testY, 
    #           'runAndTest':runAndTest, 
    #           'proportionPercentNegative50':proportionPercentNegative50}
    # #   print "Run call: "+repr(run)
    # print "Best run code: " + str(run)
    # print "Best Run confirmation: "+repr(executeCode(run, dataHash))