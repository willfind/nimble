from __future__ import absolute_import
from __future__ import print_function
import sys
import numpy
import os.path
import functools
from functools import partial
import inspect

from allowImports import boilerplate
from six.moves import map
from six.moves import range
boilerplate()

import UML

from UML.calculate import fractionCorrect
from UML.examples.working.gender.gender_prediction import LogisticRegressionNoTraining



#ORIG_HELPER_PATH = "/home/tpburns/Dropbox/ML_intern_tpb/python_workspace/"
#sys.path.append(ORIG_HELPER_PATH)


def loadPredictionData(inPath):
    allData = UML.createData("Matrix", inPath, featureNames=True, pointNames=True)

#    allData[:, "participant_id"].show("", maxHeight=None)

    nonQFeatures = ['femaleAs1MaleAs0', 'participant_id']
#    nonQFeatures = ['femaleAs1MaleAs0']

    qs_indices = []
    for i,n in enumerate(allData.getFeatureNames()):
        if n in nonQFeatures or (n[:2] == 'q_' and n != 'q_score'):
            qs_indices.append(i)

    responseData = allData.copyFeatures(qs_indices)

    return responseData

def checkResponseData(toCheck):
    def inRange(val):
        assert val >= -3 and val <=3

    toCheck.calculateForEachElement(inRange)


def discardToParityGender(toParity):
    counts = toParity.countUniqueFeatureValues("femaleAs1MaleAs0")

    totalMen = counts[0]
    totalWomen = counts[1]

    if totalMen < totalWomen:
        extractValue = 1
    elif totalWomen < totalMen:
        extractValue = 0
    else:  # if already equal, we don't need to do anything.
        return

    numerous = toParity.extractPoints(lambda x: x["femaleAs1MaleAs0"] == extractValue)
    equalCount = numerous.extractPoints(number=totalMen, randomize=True)

    assert equalCount.points == toParity.points
    assert len(equalCount.countUniqueFeatureValues("femaleAs1MaleAs0")) == 1

    toParity.appendPoints(equalCount)


def addSquaredFeatures(toExpand):
    nonQ = toExpand.extractFeatures(lambda x: x.getFeatureName(0)[:2] != 'q_')

    squares = toExpand.copy()
    squares.elementwisePower(3)
    squares.setFeatureNames([name + "^3" for name in toExpand.getFeatureNames()])

#    print(squares[0,:])

    toExpand.appendFeatures(squares)

    assert toExpand.points == squares.points
    assert toExpand.features == (squares.features * 2)

    offset = squares.features
    def checkValidity(toCheckP):
        for i in range(int(toCheckP.features / 2)):
            assert toCheckP[i] == 0 or toCheckP[i] == (toCheckP[offset+i] / toCheckP[i])

#    toExpand.calculateForEachPoint(checkValidity)

    toExpand.appendFeatures(nonQ)

#    for n in toExpand.getFeatureNames():
#        print (n)


def predict_logReg_L2(trainX, trainY, testX, testY, pred_coefs_path, predictions_path):
    train_pIDs = trainX.extractFeatures("participant_id")
    if "participant_id" in testX.getFeatureNames():
        test_pIDs = testX.extractFeatures("participant_id")
    else:
        test_pIDs = train_pIDs
    assert "participant_id" not in testX.getFeatureNames()

#    print(trainX.getPointNames())

#    for n in trainY.getFeatureNames():
#        print(n)

    C = [3**(i-10) for i in range(0,20)]
    print(C)
    C = tuple(C)

    trainY = UML.createData("Matrix", trainY.data, elementType=int, featureNames=["femaleAs1MaleAs0"])

    learner = UML.train("SKL.LogisticRegression", trainX, trainY, C=C, performanceFunction=fractionCorrect)

    print (learner.test(testX, testY, performanceFunction=fractionCorrect))
    print ("C={}".format(learner.getAttributes()['C']))

    predictions = learner.apply(testX)
    predictions.setFeatureNames(["predicted-femaleAs1MaleAs0"])
    test_pIDs.appendFeatures(predictions)
    test_pIDs.appendFeatures(testY)

    raw = testX.copyAs("numpyarray")
    probabilities = learner.backend.predict_proba(raw)

    probabilitiesObj = UML.createData("Matrix", probabilities, featureNames=["prob_Male", "prob_Female"])
    test_pIDs.appendFeatures(probabilitiesObj)
    test_pIDs.sortPoints(sortHelper=(lambda x: int(x.getPointName(0))))

    test_pIDs.writeFile(predictions_path, 'csv')

    rawCoef = learner.getAttributes()['coef_']
    coefObj = UML.createData("Matrix", rawCoef)

    raw_intercept = learner.getAttributes()['intercept_'][0]

    independentModel = UML.train("custom.LogisticRegressionNoTraining", trainX, trainY, coefs=coefObj, intercept=raw_intercept)
    independentPredictions = independentModel.apply(testX)
    independentPredictions.setFeatureNames(["predicted-femaleAs1MaleAs0"])

    assert predictions.isIdentical(independentPredictions)
    assert learner.test(testX, testY, fractionCorrect) == independentModel.test(testX, testY, fractionCorrect)

    coefObj.setFeatureNames([n+" coef" for n in testX.getFeatureNames()])

    interceptObj = UML.createData("Matrix", [raw_intercept])
    interceptObj.setFeatureNames(["intercept"])
    interceptObj.appendFeatures(coefObj)
    interceptObj.transpose()

    interceptObj.writeFile(pred_coefs_path, 'csv')



def splitDataForTrial(data, numTest, seed):
    # Split gender / response data for testing
    if numTest != 0:
        testFraction = float(numTest) / data.points
        UML.setRandomSeed(seed)
        responseTrain, genderTrain, responseTest, genderTest = data.trainAndTestSets(testFraction, "femaleAs1MaleAs0", randomOrder=True)
    else:
        genderTrain = data.extractFeatures("femaleAs1MaleAs0")
        responseTrain = data
        genderTest = genderTrain
        responseTest = responseTrain

    return responseTrain, genderTrain, responseTest, genderTest


if __name__ == "__main__":
    import time
    print(time.asctime(time.localtime()))

    UML.registerCustomLearner("custom", LogisticRegressionNoTraining)

    # Constants controlling how the data is split in train and test sets
    TEST_NUMBER = 0
    SPLIT_SEED = 42

    PARITY_GENDER_TRAINING = True
    ADD_SQUARED_FEATURES = False

    # Source Data
    sourceDir = sys.argv[1]
    path_responses = os.path.join(sourceDir, "inData", "gender_personality_final_training_data_999_people.csv")

    # Output location for transformed data
    outpath_pred_coefs = os.path.join(sourceDir, "outData", "coefs.csv")
    outpath_pred_class_and_prob = os.path.join(sourceDir, "outData", "predictions.csv")

    # Output files for visualizations
    outDir_plots = os.path.join(sourceDir, "plots")

    # load response data and gender prediction variable
    predictionData = loadPredictionData(path_responses)

    if ADD_SQUARED_FEATURES:
        addSquaredFeatures(predictionData)

    splits = splitDataForTrial(predictionData, TEST_NUMBER, SPLIT_SEED)
    (responseTrain, genderTrain, responseTest, genderTest) = splits

    print(genderTrain.countUniqueFeatureValues("femaleAs1MaleAs0"))
    print (genderTest.countUniqueFeatureValues("femaleAs1MaleAs0"))

    if PARITY_GENDER_TRAINING:
        if TEST_NUMBER == 0:
            responseTest = responseTrain.copy()
            genderTest = genderTrain.copy()
        responseTrain.appendFeatures(genderTrain)
        discardToParityGender(responseTrain)
        genderAdjusted = responseTrain.extractFeatures("femaleAs1MaleAs0")
        genderTrain.referenceDataFrom(genderAdjusted)

    print(genderTrain.countUniqueFeatureValues("femaleAs1MaleAs0"))
    print (genderTest.countUniqueFeatureValues("femaleAs1MaleAs0"))

    predict_logReg_L2(responseTrain, genderTrain, responseTest, genderTest, outpath_pred_coefs, outpath_pred_class_and_prob)


    print(time.asctime(time.localtime()))





    pass  # EOF marker
