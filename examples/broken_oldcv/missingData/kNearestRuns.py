"""
Script that uses Shogun's KNN implementation to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
    import os.path 
    import UML
    from UML import crossValidateReturnBest
    from UML import functionCombinations
    from UML.umlHelpers import executeCode
    from UML import runAndTest
    from UML import createData
    from UML import splitData
    from UML.metrics import fractionIncorrect

    pathIn = os.path.join(UML.UMLPath, "datasets/tfIdfApproval50K.mtx")
    allData = createData("Sparse", pathIn, fileType="mtx")
    trainX, trainY, testX, testY = splitData(allData, labelID=0, fractionForTestSet=.2)
    print "Finished loading data"
    print "trainX shape: " + str(trainX.data.shape)
    print "trainY shape: " + str(trainY.data.shape)

    # sparse types aren't playing nice with the error metrics currently, so convert
    trainY = trainY.copyAs(format="Matrix")
    testY = testY.copyAs(format="Matrix")

    trainYList = []
    
    for i in range(len(trainY.data)):
        label = trainY.data[i][0]
        trainYList.append([int(label)])

    testYList = []
    for i in range(len(testY.data)):
        label = testY.data[i][0]
        testYList.append([int(label)])

    trainY = createData('Matrix', trainYList)
    testY = createData('Matrix', testYList)

    print "Finished converting labels to ints"


    # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    toRun = 'runAndTest("shogun.KNN", trainX, trainY, testX, testY, {"k":<1|3|5|10|15|20|25>}, [fractionIncorrect],sendToLog=False)'
    runs = functionCombinations(toRun)
    extraParams = {'runAndTest':runAndTest, 'fractionIncorrect':fractionIncorrect}
    run, results = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)

    run = run.replace('sendToLog=False', 'sendToLog=True')
    dataHash={"trainX": trainX,
              "testX":testX, 
              "trainY":trainY, 
              "testY":testY, 
              'runAndTest':runAndTest, 
              'fractionIncorrect':fractionIncorrect}
    #   print "Run call: "+repr(run)
    print "Best run code: " + str(run)
    print "Best Run confirmation: "+repr(executeCode(run, dataHash))


    # # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    # toRun = 'runAndTest("shogun.KNN", trainX, trainY, testX, testY, {"k":<1|3|5|10|15|20|25>}, [fractionTrueNegativeTop50], scoreMode="allScores", negativeLabel="2", sendToLog=False)'
    # runs = functionCombinations(toRun)
    # extraParams = {'runAndTest':runAndTest, 'fractionTrueNegativeTop50':fractionTrueNegativeTop50}
    # run, results = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)

    # run = run.replace('sendToLog=False', 'sendToLog=True')
    # dataHash={"trainX": trainX,
    #           "testX":testX, 
    #           "trainY":trainY, 
    #           "testY":testY, 
    #           'runAndTest':runAndTest, 
    #           'fractionTrueNegativeTop50':fractionTrueNegativeTop50}
    # #   print "Run call: "+repr(run)
    # print "Best run code: " + str(run)
    # print "Best Run confirmation: "+repr(executeCode(run, dataHash))
