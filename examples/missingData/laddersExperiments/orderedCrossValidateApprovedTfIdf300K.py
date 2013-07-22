"""
Script that uses 300K points of job posts to try to predict approved/rejected status,
using orderedCrossValidate - ordered over timestamp data
"""

import sys
from datetime import timedelta
# add UML parent directory to sys.path
sys.path.append(sys.path[0].rsplit('/',2)[0])
import UML
import UML.examples
#import UML.examples.laddersExperiments
__package__ = "UML.examples.laddersExperiments"

if __name__ == "__main__":
    from UML import orderedCrossValidate
    from UML import functionCombinations
    from UML import runAndTest
    from UML import createData
    from UML.metrics import fractionTrueNegativeTop90

    pathIn = "/home/ross/library/LaddersData/umlApprovalEntryDate300K.mtx"
    trainX = createData('Sparse', pathIn, fileType='mtx')
    docId = trainX.extractFeatures(0)
    trainY = trainX.extractFeatures(0)
    print "Finished loading data"
    print "trainX shape: " + str(trainX.data.shape)
    print "trainY shape: " + str(trainY.data.shape)


    # sparse types aren't playing nice with the error metrics currently, so convert
    #trainX = trainX.toMatrix()
    #testX = testX.toMatrix()

    trainY = trainY.toMatrix()

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
    toRun = 'runAndTest("shogun.MulticlassLibLinear", trainX, trainY, testX, testY, {"C":<0.001|0.01|0.1|1.0|5.0>}, [fractionTrueNegativeTop90], scoreMode="allScores", negativeLabel="2", sendToLog=True)'
    runs = functionCombinations(toRun)
    extraParams = {'runAndTest':runAndTest, 'fractionTrueNegativeTop90':fractionTrueNegativeTop90}
    stepDelta = timedelta(days=30)
    results = orderedCrossValidate(trainX, trainY, runs, orderedFeature=1, minTrainSize=24000, maxTrainSize=75000, minTestSize=9000, maxTestSize=20000, stepSize=stepDelta, gap=0, extraParams=extraParams)

    for function, errorMetric in results.iteritems():
        print "Function: " + str(function)
        print "Proportion of Approved in Top 90%: " + str(errorMetric)




