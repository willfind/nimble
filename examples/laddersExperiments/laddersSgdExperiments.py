"""
Run tests w/Ladders data: try to predict Approved/Rejected using scikit-learn's implementation of 
SGD.
"""

"""
Script that uses 50K points of job posts to try to predict approved/rejected status
"""
import sys
# add UML parent directory to sys.path
sys.path.append(sys.path[0].rsplit('/',2)[0])
import UML
import UML.examples
import UML.examples.laddersExperiments
__package__ = "UML.examples.laddersExperiments"

if __name__ == "__main__":
    from UML import crossValidateReturnBest
    from UML import functionCombinations
    from UML.umlHelpers import executeCode
    from UML import runAndTest
    from UML import data
    from UML.metrics import proportionPercentNegative90

    pathIn = "UML/datasets/tfIdfApproval50K.mtx"
    trainX = data('coo', pathIn, fileType='mtx')
    testX = trainX.extractPoints(start=0, end=trainX.points(), number=int(round(0.2*trainX.points())), randomize=True)
    trainY = trainX.extractFeatures(0)
    testY = testX.extractFeatures(0)
    print "Finished loading data"
    print "trainX shape: " + str(trainX.data.shape)
    print "trainY shape: " + str(trainY.data.shape)

    # sparse types aren't playing nice with the error metrics currently, so convert
    trainY = trainY.toDenseMatrixData()
    testY = testY.toDenseMatrixData()

    trainYList = []
    trainRemoveList = []
    for i in range(len(trainY.data)):
        label = trainY.data[i][0]
        if int(label) != 0:
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
        if int(label) != 0:
            testYList.append([int(label)])
        else:
            #testYList.append([1])
            testRemoveList.append(i)
            print "found null label: " + str(i)
            print "label: " + str(label)

    trainX.extractPoints(trainRemoveList)
    testX.extractPoints(testRemoveList)

    trainY = data('dense', trainYList)
    testY = data('dense', testYList)

    print "Finished converting labels to ints"


    # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    toRun = 'runAndTest("sciKitLearn.SGDClassifier", trainX, testX, trainY, testY, {"class_weight":{1:1.0, 2:<0.5|1.0|1.25|1.5|1.75|2.0|5.0|10.0>, "alpha":0.00001}, [proportionPercentNegative90], scoreMode="allScores", negativeLabel="2", sendToLog=False)'
    runs = functionCombinations(toRun)
    extraParams = {'runAndTest':runAndTest, 'proportionPercentNegative90':proportionPercentNegative90}
    run, results = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)

    run = run.replace('sendToLog=False', 'sendToLog=True')
    # runWithL1Penalty = re.sub(r'(?P<args>\{\"alpha[^}]*)', '\g<args>, "penalty":"l1"', run)
    dataHash={"trainX": trainX,
              "testX":testX,
              "trainY":trainY,
              "testY":testY,
              'runAndTest':runAndTest,
              'proportionPercentNegative90':proportionPercentNegative90}
    print "Best run code: " + str(run)
    print "Best Run confirmation: "+repr(executeCode(run, dataHash))

    # print "Best run code, with L1 penalty: " + str(run)
    # print "Best Run confirmation, with L1 penalty: "+repr(executeCode(runWithL1Penalty, dataHash))

    # runNItersCombinations = re.sub(r'(?P<args>\{\"alpha[^}]*)', '\g<args>, "n_iter":<1|5|10|15|20|25>', run)
    # runNItersCombinations = 'runAndTest("sciKitLearn.SGDClassifier", trainX, testX, trainY, testY, {"alpha":<0.0000001|0.000001|0.00001|0.0001|0.001|0.01|0.1|1.0>}, [proportionPercentNegative90], scoreMode="allScores", negativeLabel="2", sendToLog=True)'
    # nItersRuns = functionCombinations(runNItersCombinations)
    # nItersResults = [None] * len(nItersRuns)
    # for run in nItersRuns:
    #     result = executeCode(run, dataHash)
    #     print "Run call: " + str(run)
    #     print "Result: " + str(result)
