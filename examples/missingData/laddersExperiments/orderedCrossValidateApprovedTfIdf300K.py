"""
Script that uses 300K points of job posts to try to predict approved/rejected status,
using orderedCrossValidate - ordered over timestamp data
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
    import UML
    from UML import orderedCrossValidate
    from UML import functionCombinations
    from UML import trainAndTest
    from UML import createData
    from UML.metrics import fractionTrueNegativeTop90
    import os.path

    bigdataPath = os.path.join(os.path.dirname(UML.UMLPath), "bigdata")

    pathIn = os.path.join(bigdataPath, "umlApprovalEntryDate300K.mtx")
    trainX = createData('Sparse', pathIn, fileType='mtx')
    docId = trainX.extractFeatures(0)
    trainY = trainX.extractFeatures(0)
    print "Finished loading data"
    print "trainX shape: " + str(trainX.data.shape)
    print "trainY shape: " + str(trainY.data.shape)


    # sparse types aren't playing nice with the error metrics currently, so convert
    #trainX = trainX.copyAs(format="Matrix")
    #testX = testX.copyAs(format="Matrix")

    trainY = trainY.copyAs(format="Matrix")

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


    trainX.extractPoints(trainRemoveList)

    trainY = createData('Sparse', trainYList)

    print "Finished converting labels to ints"


    # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    toRun = 'trainAndTest("shogun.MulticlassLibLinear", trainX, trainY, testX, testY, {"C":<0.001|0.01|0.1|1.0|5.0>}, [fractionTrueNegativeTop90], scoreMode="allScores", negativeLabel="2", sendToLog=True)'
    runs = functionCombinations(toRun)
    extraParams = {'trainAndTest':trainAndTest, 'fractionTrueNegativeTop90':fractionTrueNegativeTop90}
    stepDelta = timedelta(days=30)
    results = orderedCrossValidate(trainX, trainY, runs, orderedFeature=1, minTrainSize=24000, maxTrainSize=75000, minTestSize=9000, maxTestSize=20000, stepSize=stepDelta, gap=0, extraParams=extraParams)

    for function, errorMetric in results.iteritems():
        print "Function: " + str(function)
        print "Proportion of Approved in Top 90%: " + str(errorMetric)




