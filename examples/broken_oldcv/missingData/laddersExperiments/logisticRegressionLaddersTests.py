"""
Script that uses 50K points of job posts to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
    import UML
    from UML import crossValidateReturnBest
    from UML import functionCombinations
    from UML.umlHelpers import executeCode
    from UML import runAndTest
    from UML import createData
    from UML.metrics import fractionTrueNegativeTop90
    import os.path

    bigdataPath = os.path.join(os.path.dirname(UML.UMLPath), "bigdata")

    pathIn = os.path.join(bigdataPath, "DocVectors.mtx")
    trainX = createData('Sparse', pathIn, fileType='mtx')
    testX = trainX.extractPoints(start=0, end=trainX.pointCount, number=int(round(0.2*trainX.pointCount)), randomize=True)
    trainY = trainX.extractFeatures(0)
    testY = testX.extractFeatures(0)
    print "Finished loading data"
    print "trainX shape: " + str(trainX.data.shape)
    print "trainY shape: " + str(trainY.data.shape)


    # sparse types aren't playing nice with the error metrics currently, so convert
    #trainX = trainX.copyAs(format="Matrix")
    #testX = testX.copyAs(format="Matrix")

    trainY = trainY.copyAs(format="Matrix")
    testY = testY.copyAs(format="Matrix")

    trainYList = []
    trainRemoveList = []
    for i in range(len(trainY.data)):
        label = trainY.data[i][0]
        if int(label) == 1 or int(label) == 2:
            trainYList.append([int(label)])
        else:
            #trainYList.append([1])
            trainRemoveList.append(i)
            print "found null label: " + str(i)
            print "label: " + str(label)


    testYList = []
    testRemoveList = []
    for i in range(len(testY.data)):
        label = testY.data[i][0]
        if int(label) == 1 or int(label) == 2:
            testYList.append([int(label)])
        else:
            #testYList.append([1])
            testRemoveList.append(i)
            print "found null label: " + str(i)
            print "label: " + str(label)

    trainX.extractPoints(trainRemoveList)
    testX.extractPoints(testRemoveList)

    trainY = createData('Matrix', trainYList)
    testY = createData('Matrix', testYList)

    print "Finished converting labels to ints"


    # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    toRun = 'runAndTest("sciKitLearn.LogisticRegression", trainX, trainY, testX, testY, {"C":<0.000001|0.00001|0.0001|0.001|0.01|0.1|1.0|10.0|50.0>}, [fractionTrueNegativeTop90], scoreMode="allScores", negativeLabel="2", sendToLog=False)'
    runs = functionCombinations(toRun)
    extraParams = {'runAndTest':runAndTest, 'fractionTrueNegativeTop90':fractionTrueNegativeTop90}
    run, results = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)

    run = run.replace('sendToLog=False', 'sendToLog=True')
    dataHash={"trainX": trainX,
              "testX":testX,
              "trainY":trainY,
              "testY":testY,
              'runAndTest':runAndTest,
              'fractionTrueNegativeTop90':fractionTrueNegativeTop90}
    #   print "Run call: "+repr(run)
    print "Best run code: " + str(run)
    print "Best Run confirmation: "+repr(executeCode(run, dataHash))
