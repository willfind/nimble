"""
Select the best questions to keep in a survey from a larger set of questions
"""
import numpy

from allowImports import boilerplate

boilerplate()

import os.path
import UML
from UML import trainAndTest
from UML import trainAndApply
from UML import createData
from UML.calculate import fractionIncorrect
from UML import calculate


def rSquared(knownValues, predictedValues):
    diffObject = predictedValues - knownValues
    rawDiff = diffObject.copyAs("numpy array")
    rawKnowns = knownValues.copyAs("numpy array")
    return 1.0 - numpy.var(rawDiff) / float(numpy.var(rawKnowns))


if __name__ == "__main__":
    #fileName = "spencerTest.csv"
    fileName = "Academic_Cons_Survey_445_points.csv"
    finalFeaturesToKeep = 3
    fractionOfDataForValidation = 0.4
    featuresToPredict = ["TotalAcademicScore", "inLyingGroup"] #the features we'll be predicting
    #these functions determine what to exclude for each feature prediction
    functionsToExcludePoints = [lambda x: x["inLyingGroup"] == 1, lambda x: False]
    predictionAlgorithms = ["SciKitLearn.Ridge", "SciKitLearn.LogisticRegression"]

    #load the data
    pathIn = os.path.join(UML.UMLPath, "datasets/", fileName)
    allFeatures = createData("Matrix", pathIn, featureNames=True)

    #print "allFeatures\n", allFeatures

    #these will store the training and testing data for each label we're going to predict
    trainXs = []
    trainYs = []
    testXs = []
    testYs = []

    #for each of the different labels we want to predict, make a distinct training set
    for labelNumber in xrange(len(featuresToPredict)):
        featuresForLabel = allFeatures.copy()
        functionToExcludePoints = functionsToExcludePoints[labelNumber]
        featuresForLabel.extractPoints(functionToExcludePoints) #get rid of points that aren't relevant to this label

        #take all the labels we'll be predicting out of the features
        labelsMatrix = featuresForLabel.extractFeatures(featuresToPredict)

        currentLabels = labelsMatrix.copyFeatures(labelNumber)
        featuresWithLabels = featuresForLabel.copy()

        #print "currentLabels\n", currentLabels
        #print "featuresWithLabels\n", featuresWithLabels

        featuresWithLabels.appendFeatures(currentLabels)
        labelFeatureNum = featuresWithLabels.featureCount - 1

        #get the training and testing data for this label
        trainX, trainY, testX, testY = featuresWithLabels.trainAndTestSets(testFraction=.2, labels=labelFeatureNum)
        trainXs.append(trainX)
        trainYs.append(trainY)
        testXs.append(testXs)
        testYs.append(testYs)


    #try dropping one feature at a time, and kill off the feature that is most useless until we have the right number
    while allFeatures.featureCount > finalFeaturesToKeep:

        #try dropping each feature one by one
        for featureToDrop in xrange(allFeatures.featureCount):

            #for each label we're predicting
            labelNum = -1
            for trainX, trainY in zip(trainXs, trainYs):
                labelNum += 1

                #build a feature set to train on that doesn't include the feature we're dropping
                trainXWithoutFeature = trainX.copy()
                trainXWithoutFeature.extractFeatures(featureToDrop)

                #args = {"alphas":[2**k for k in xrange(-4,4)]}
                #results = trainAndTest("SciKitLearn.RidgeCV", trainXWithoutFeature, trainY, testX=trainXWithoutFeature, testY=trainY, performanceFunction=rSquared, arguments=args)
                #args = {"penalty":"l2", "C":[2**k for k in xrange(-4,4)]}
                #sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0)
                print "trainY", trainY

                algorithmName = predictionAlgorithms[labelNum]
                if "Logistic" in algorithmName:
                    results = trainAndTest(algorithmName, trainXWithoutFeature, trainY, testX=trainXWithoutFeature,
                                           testY=trainY, performanceFunction=rSquared,
                                           C=tuple([10.0 ** k for k in xrange(-6, 6)]))
                elif "Ridge" in algorithmName:
                    results = trainAndTest(algorithmName, trainXWithoutFeature, trainY, testX=trainXWithoutFeature,
                                           testY=trainY, performanceFunction=rSquared,
                                           alpha=tuple([10.0 ** k for k in xrange(-6, 6)]))
                else:
                    raise Exception("Don't know how to set parameters for algorithm: " + str(algorithmName))

                print "results", results
            ### This part isn't done. This is where we need code store the error made without featureToDrop
            ### and combine this error across all labels we're predictin
            ### and then we'll use that to decide which feature is most useless
            ### and then continue to the next iteration
				


