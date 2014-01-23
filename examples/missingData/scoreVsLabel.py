"""
Script that uses 50K points of job posts to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
    import os.path
    import UML
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
    scores = runAndTest("shogun.MulticlassLibLinear", trainX, trainY, testX, testY, {"C":0.75}, [fractionTrueNegativeTop90, fractionTrueNegativeTop50], scoreMode="allScores", negativeLabel="2", sendToLog=False)

    print "top 90 proportion Rejected: " + str(scores)
