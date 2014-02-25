"""
Script that uses 50K points of job posts to try to predict approved/rejected status
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
    from UML.metrics import fractionTrueNegativeTop50
    from UML.metrics import fractionTrueNegativeTop90

    pathIn = os.path.join(UML.UMLPath, "datasets/tfIdfApproval50K.mtx")
    allData = createData("Sparse", pathIn, fileType="mtx")
    trainX, trainY, testX, testY = splitData(allData, labelID=0, fractionForTestSet=.2)
    print "Finished loading data"
    print "trainX shape: " + str(trainX.data.shape)
    print "trainY shape: " + str(trainY.data.shape)

    # sparse types aren't playing nice with the error metrics currently, so convert
    trainY = trainY.copy(asType="Matrix")
    testY = testY.copy(asType="Matrix")

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
    toRun = 'runAndTest("shogun.MulticlassOCAS", trainX, trainY, testX, testY, {"C":<0.1|0.5|0.75|1.0|5.0>}, <[fractionTrueNegativeTop90]|[fractionTrueNegativeTop50]>, scoreMode="allScores", negativeLabel="2", sendToLog=False)'
    runs = functionCombinations(toRun)
    extraParams = {'runAndTest':runAndTest, 'fractionTrueNegativeTop90':fractionTrueNegativeTop90, 'fractionTrueNegativeTop50':fractionTrueNegativeTop50}
    run, results = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)

    # for run in runs:
    dataHash={"trainX": trainX.copy(), 
              "testX":testX.copy(), 
              "trainY":trainY.copy(), 
              "testY":testY.copy(), 
              'runAndTest':runAndTest, 
              'fractionTrueNegativeTop90':fractionTrueNegativeTop90,
              'fractionTrueNegativeTop50':fractionTrueNegativeTop50}
    #   print "Run call: "+repr(run)
    print "Best run code: " + str(run)
    print "Best Run confirmation: "+repr(executeCode(run, dataHash))

