"""
Select the best questions to keep in a survey from a larger set of questions
"""
import numpy
import sys

from allowImports import boilerplate

boilerplate()

import os.path
import UML
from UML import trainAndTest
from UML import trainAndApply
from UML import train
from UML import createData
from UML.calculate import fractionIncorrect
from UML import calculate


def rSquared(knownValues, predictedValues):
    diffObject = predictedValues - knownValues
    rawDiff = diffObject.copyAs("numpy array")
    rawKnowns = knownValues.copyAs("numpy array")
    return 1.0 - numpy.var(rawDiff) / float(numpy.var(rawKnowns))


def buildTrainingAndTestingSetsForPredictions(data, fractionOfDataForTesting, featuresToPredict,
                                              functionsToExcludePoints):
    """creates  the training and testing sets for each label we're going to predict"""
    trainXs = []
    trainYs = []
    testXs = []
    testYs = []

    #make our training and testing sets
    #for each of the different labels we want to predict
    for labelNumber in xrange(len(featuresToPredict)):
        #create the features for the current labelNum, and exclude irrelevant poins
        currentFeatures = data.copy()
        currentFeatures.extractPoints(
            functionsToExcludePoints[labelNumber]) #get rid of points that aren't relevant to this label

        #remove the labels from the features
        labelsMatrix = currentFeatures.extractFeatures(featuresToPredict)

        #get the labels we'll be predicting
        featureToPredict = featuresToPredict[labelNumber]
        currentLabels = labelsMatrix.copyFeatures(featureToPredict)

        #add just those labels we'll be predicting to the features to form a combined set
        featuresWithLabels = currentFeatures.copy()
        featuresWithLabels.appendFeatures(currentLabels)
        labelFeatureNum = featuresWithLabels.featureCount - 1

        #get the training and testing data for this label
        trainX, trainY, testX, testY = featuresWithLabels.trainAndTestSets(testFraction=fractionOfDataForTesting,
                                                                           labels=labelFeatureNum)
        trainXs.append(trainX)
        trainYs.append(trainY)
        testXs.append(testX)
        testYs.append(testY)

    #confirm the training X sets all have the same number of features (even though they may not have the same number of points)
    for trainX, trainY in zip(trainXs, trainYs):
        assert trainX.featureCount == trainXs[0].featureCount
        assert trainY.pointCount == trainX.pointCount
        assert trainY.featureCount == 1

    return trainXs, trainYs, testXs, testYs


def testBuildTrainingAndTestingSetsForPredictions():
    data = [["x1", "x2", "x3", "y1", "x4", "y2"], [1, 5, 2, 3, 7, 1], [2, 2, 3.2, 5, 9.1, -7], [3, 5, 2, 1, 3, 9],
            [4, 9.2, 3, 5, 5, 1], [5, -4, 2, 1, 1, 0], [6, -2, -3, -1, -2, -3]]
    data = createData("Matrix", data, featureNames=True)
    fractionOfDataForTesting = 1.0 / 3.0
    featuresToPredict = ["y1", "y2"]
    functionsToExcludePoints = [lambda r: r["x2"] < 3, lambda r: False]
    trainXs, trainYs, testXs, testYs = buildTrainingAndTestingSetsForPredictions(data, fractionOfDataForTesting,
                                                                                 featuresToPredict,
                                                                                 functionsToExcludePoints)
    assert (len(trainXs)) == 2
    assert trainXs[0].featureCount == 4
    assert trainXs[1].featureCount == 4
    assert trainXs[0].getFeatureNames() == ["x1", "x2", "x3", "x4"]
    assert trainXs[1].getFeatureNames() == ["x1", "x2", "x3", "x4"]
    assert trainXs[0].pointCount == 2
    assert testXs[0].pointCount == 1
    assert trainXs[1].pointCount == 4
    assert testXs[1].pointCount == 2
    jointXs0 = trainXs[0].copy()
    jointXs0.appendPoints(testXs[0])
    jointXs0.sortPoints("x1")
    print "jointXs0\n", jointXs0
    assert jointXs0.isApproximatelyEqual(createData("Matrix", [[1, 5, 2, 7], [3, 5, 2, 3], [4, 9.2, 3, 5]]))

    jointYs0 = trainYs[0].copy()
    jointYs0.appendPoints(testYs[0])
    jointYs0.sortPoints(0)

    print "jointYs0\n", jointYs0
    assert jointYs0.isApproximatelyEqual(createData("Matrix", [[1], [3], [5]]))

    jointXs1 = trainXs[1].copy()
    jointXs1.appendPoints(testXs[1])
    jointXs1.sortPoints("x1")
    assert jointXs1.isApproximatelyEqual(createData("Matrix",
                                                    [[1, 5, 2, 7], [2, 2, 3.2, 9.1], [3, 5, 2, 3], [4, 9.2, 3, 5],
                                                     [5, -4, 2, 1], [6, -2, -3, -2]]))

    jointYs1 = trainYs[1].copy()
    jointYs1.appendPoints(testYs[1])
    jointYs1.sortPoints(0)

    print "jointYs1\n", jointYs1
    assert jointYs1.isApproximatelyEqual(createData("Matrix", [[-7], [-3], [0], [1], [1], [9]]))


def reduceDataToBestFeatures(trainXs, trainYs, testXs, testYs, numFeaturesToKeep, predictionAlgorithms,
                             featuresToPredict):
    "tries dropping one feature at a time from all datasets, and kill off the feature that is most useless until we have the right number"""
    assert isinstance(trainXs, list)
    assert isinstance(trainYs, list)
    assert isinstance(testXs, list)
    assert isinstance(testYs, list)
    assert len(trainXs) == len(trainYs) and len(trainXs) == len(testXs) and len(trainXs) == len(testYs)

    if numFeaturesToKeep > trainXs[0].featureCount: raise Exception(
        "Cannot keep " + str(numFeaturesToKeep) + " features since the data has only " + str(
            trainXs[0].featureCount) + " features.")

    while trainXs[0].featureCount > numFeaturesToKeep:
        print str(trainXs[0].featureCount) + " features left"

        accuraciesForEachFeatureDropped = []

        #try dropping each feature one by one
        for featureNumToDrop in xrange(trainXs[0].featureCount):
            sys.stdout.write(" " + str(trainXs[0].getFeatureNames()[featureNumToDrop]))
            accuraciesForThisFeatureDrop = []
            #for each label we're predicting
            for labelNum, trainX, trainY in zip(range(len(trainXs)), trainXs, trainYs):
                #print "trainX", trainX
                #build a feature set to train on that doesn't include the feature we're dropping
                trainXWithoutFeature = trainX.copy()
                trainXWithoutFeature.extractFeatures(featureNumToDrop)

                algorithmName = predictionAlgorithms[labelNum]
                if "Logistic" in algorithmName:
                #C = tuple([10.0**k for k in xrange(-6, 6)])
                    C = 1000000
                    error = trainAndTest(algorithmName, trainXWithoutFeature, trainY, testX=trainXWithoutFeature,
                                         testY=trainY, performanceFunction=fractionIncorrect, C=C)
                    accuracy = 1.0 - error
                elif "Ridge" in algorithmName:
                    #alpha = tuple([10.0**k for k in xrange(-6, 6)])
                    alpha = 0
                    accuracy = trainAndTest(algorithmName, trainXWithoutFeature, trainY, testX=trainXWithoutFeature,
                                            testY=trainY, performanceFunction=rSquared, alpha=alpha)
                else:
                    raise Exception("Don't know how to set parameters for algorithm: " + str(algorithmName))

                accuraciesForThisFeatureDrop.append(accuracy)

            combinedAccuracyForFeatureDrop = numpy.mean(accuraciesForThisFeatureDrop)
            accuraciesForEachFeatureDropped.append((combinedAccuracyForFeatureDrop, featureNumToDrop))

        accuraciesForEachFeatureDropped.sort(lambda x, y: cmp(x[0], y[
            0])) #sort asscending by accuracy so that the last element corresponds to the most useless feature
        mostUselessFeatureNum = accuraciesForEachFeatureDropped[-1][1]
        print "\nRemoving feature " + str(mostUselessFeatureNum) + " with combined accuracy " + str(
            round(accuraciesForEachFeatureDropped[-1][0], 3))
        for trainX, testX in zip(trainXs, testXs):
            trainX.extractFeatures(mostUselessFeatureNum)
            testX.extractFeatures(mostUselessFeatureNum)
        #print "viableFeaturesLeft", trainXs[0].featureCount

    return trainXs, trainYs, testXs, testYs


def getPredictionAccuracies(trainXs, trainYs, testXs, testYs, predictionAlgorithms, featuresToPredict):
    accuraciesHash = {}
    parametersHash = {}
    firstTrainX = trainXs[0]
    #now test the models out of sample on our final feature sets!
    for labelNum, trainX, trainY, testX, testY in zip(range(len(trainXs)), trainXs, trainYs, testXs, testYs):
        algorithmName = predictionAlgorithms[labelNum]
        featureToPredict = featuresToPredict[labelNum]
        if "Logistic" in algorithmName:
        #C = tuple([10.0**k for k in xrange(-6, 6)])
            C = 1000000
            #error = trainAndTest(algorithmName, trainX, trainY, testX=testX, testY=testY, performanceFunction=fractionIncorrect, C=C)
            learner = UML.train(algorithmName, trainX, trainY, C=C)
            error = learner.test(testX=testX, testY=testY, performanceFunction=fractionIncorrect)
            accuracy = 1.0 - error
            backend = learner.backend
            parametersHash[featureToPredict] = {"intercept": backend.intercept_, "coefs": backend.coef_}
        elif "Ridge" in algorithmName:
            #alpha = tuple([10.0**k for k in xrange(-6, 6)])
            alpha = 0
            #accuracy = trainAndTest(algorithmName, trainX, trainY, testX=testX, testY=testY, performanceFunction=rSquared, alpha=alpha)
            learner = UML.train(algorithmName, trainX, trainY, alpha=alpha)
            accuracy = learner.test(testX=testX, testY=testY, performanceFunction=rSquared)
            backend = learner.backend
            parametersHash[featureToPredict] = {"intercept": backend.intercept_, "coefs": backend.coef_}
        else:
            raise Exception("Don't know how to set parameters for algorithm: " + str(algorithmName))
        accuraciesHash[featureToPredict] = accuracy #record the accuracy

    bestFeatures = firstTrainX.getFeatureNames()
    return bestFeatures, accuraciesHash, parametersHash


def getBestFeaturesAndAccuracies(data, numFeaturesToKeep, predictionAlgorithms, fractionOfDataForTesting,
                                 featuresToPredict, functionsToExcludePoints):
    trainXs, trainYs, testXs, testYs = buildTrainingAndTestingSetsForPredictions(data,
                                                                                 fractionOfDataForTesting=fractionOfDataForTesting,
                                                                                 featuresToPredict=featuresToPredict,
                                                                                 functionsToExcludePoints=functionsToExcludePoints)
    trainXs, trainYs, testXs, testYs = reduceDataToBestFeatures(trainXs, trainYs, testXs, testYs,
                                                                numFeaturesToKeep=numFeaturesToKeep,
                                                                predictionAlgorithms=predictionAlgorithms,
                                                                featuresToPredict=featuresToPredict)
    bestFeatures, accuraciesHash, parametersHash = getPredictionAccuracies(trainXs, trainYs, testXs, testYs,
                                                                           predictionAlgorithms=predictionAlgorithms,
                                                                           featuresToPredict=featuresToPredict)

    return bestFeatures, accuraciesHash, parametersHash


def testGetBestFeaturesAndAccuracies():
    data = [["x0", "x1", "x2", "x3", "p1", "p2", "x6", "x7"], [1, 5, 2, 3, 7, 1, 3, 9], [2, 2, 3.2, 5, 9.1, -7, 2, 7],
            [3, 5, 2, 1, 3, 9, 4, 8], [4, 9.2, 3, 5, 5, 1, 8, 1], [5, -4, 2, 1, 1, 0, 6, 3],
            [6, -2, -3, -1, -2, -3, -3, -2], [7, 1, 4, 18, 5, 6, 1, 4], [8, 2, 3, 4, 5, 6, 3, 2],
            [9, 1, 12, 10, 5, 2, 8, 5], [10, 8, 3, 2, 1, 5, 4, 1], [11, 5, 2, 10, 4, 2, 8, 1],
            [12, 6, 2, 14, 4, 5.3, 2, -1], [13, 2, 2, 12, 4, 5.4, 3, -1], [14, 6, 2, 15, 4, 5.3, 3, -1],
            [15, 2, 1, 13, 2, 0.3, -3, 4], [16, 6, 3, 4, 4, -1, 3, 0], [17, 1, 4, 20, 3, -6, -2, 1],
            [18, 4, 3, 2, 5, 8, 9, 3]]

    #setup values for the labels
    firstRow = True
    noiseLevel = 0
    for row in data:
        if firstRow:
            firstRow = False
            continue

        #set p1
        row[4] = 10 * row[0] + row[1] + 100 * row[3] + noiseLevel * numpy.random.uniform(0, 1)

        #set p2
        #if row[0] > row[3] + noiseLevel*numpy.random.uniform(0,1): row[5] = 1
        if row[0] > 9:
            row[5] = 1
        else:
            row[5] = 0

    data = createData("Matrix", data, featureNames=True)
    fractionOfDataForTesting = 1.0 / 3.0
    featuresToPredict = ["p1", "p2"]

    print "data\n", data

    functionsToExcludePoints = [lambda r: False, lambda r: False]
    predictionAlgorithms = ["SciKitLearn.Ridge", "SciKitLearn.LogisticRegression"]

    numFeaturesToKeep = 3
    bestFeatures, accuraciesHash, parametersHash = getBestFeaturesAndAccuracies(data, numFeaturesToKeep,
                                                                                predictionAlgorithms,
                                                                                fractionOfDataForTesting,
                                                                                featuresToPredict,
                                                                                functionsToExcludePoints)
    assert bestFeatures == ["x0", "x1", "x3"]

    print "accuraciesHash:", accuraciesHash
    print "parametersHash", parametersHash

    assert abs(accuraciesHash["p1"] - 1) < 1E-5
    assert abs(accuraciesHash["p2"] - 1) < 1E-5

    numFeaturesToKeep = 2
    bestFeatures, accuraciesHash, parametersHash = getBestFeaturesAndAccuracies(data, numFeaturesToKeep,
                                                                                predictionAlgorithms,
                                                                                fractionOfDataForTesting,
                                                                                featuresToPredict,
                                                                                functionsToExcludePoints)
    assert bestFeatures == ["x0", "x3"]

    numFeaturesToKeep = 1
    bestFeatures, accuraciesHash, parametersHash = getBestFeaturesAndAccuracies(data, numFeaturesToKeep,
                                                                                predictionAlgorithms,
                                                                                fractionOfDataForTesting,
                                                                                featuresToPredict,
                                                                                functionsToExcludePoints)
    assert bestFeatures == ["x3"]


if __name__ == "__main__":
    #fileName = "spencerTest.csv"
    fileName = "Academic_Cons_Survey_445_points.csv"
    numFeaturesToKeep = 3
    fractionOfDataForTesting = 0.25
    featuresToPredict = ["TotalAcademicScore", "inLyingGroup"] #the features we'll be predicting
    #these functions determine what to exclude for each feature prediction
    functionsToExcludePoints = [lambda x: x["inLyingGroup"] == 1, lambda x: False]
    predictionAlgorithms = ["SciKitLearn.Ridge", "SciKitLearn.LogisticRegression"]
    featuresToRemoveCompletely = ["User Number", "attention1", "attention2", "AttentionChecksPassed", "AcademicAtt1",
                                  "AcademicAtt2", "AcademicAttChksPassed", "AcademicScore"]

    #load the data
    pathIn = os.path.join(UML.UMLPath, "datasets/", fileName)
    allFeatures = createData("Matrix", pathIn, featureNames=True)
    allFeatures.extractFeatures(featuresToRemoveCompletely)

    bestFeatures, accuraciesHash, parametersHash = getBestFeaturesAndAccuracies(allFeatures,
                                                                                numFeaturesToKeep=numFeaturesToKeep,
                                                                                predictionAlgorithms=predictionAlgorithms,
                                                                                fractionOfDataForTesting=fractionOfDataForTesting,
                                                                                featuresToPredict=featuresToPredict,
                                                                                functionsToExcludePoints=functionsToExcludePoints)

    print "Best features: " + str(bestFeatures)
    print ""
    print "Accuracies: " + str(accuraciesHash)
    print ""
    print "Paramaters: " + str(parametersHash)


	
				

