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

import PIL

import UML

from UML.calculate import fractionCorrect
from UML.examples.working.gender.gender_prediction import LogisticRegressionNoTraining

scipy = UML.importModule("scipy")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#ORIG_HELPER_PATH = "/home/tpburns/Dropbox/ML_intern_tpb/python_workspace/"
#sys.path.append(ORIG_HELPER_PATH)


colorM_raw ='#66a6ff'
colorF_raw ='#ff668c'
cc = matplotlib.colors.ColorConverter()
colorM = cc.to_rgb(colorM_raw)
colorF = cc.to_rgb(colorF_raw)
blendColor = ((colorM[0] + colorF[0])/2., (colorM[1] + colorF[1])/2., (colorM[2] + colorF[2])/2.)


def loadMetaData(inPath):
    meta = UML.createData("List", inPath, featureNames=True)
    categories = {}
    qParity = {}
    for point in meta.pointIterator():
        categories[point["Category"]] = (point["Q1"], point["Q2"])
        qParity[point['Q1']] = point['Q1 parity to category']
        qParity[point['Q2']] = point['Q2 parity to category']

    return categories, qParity

def loadPredictionData(inPath):
    allData = UML.createData("Matrix", inPath, featureNames=True, pointNames=True)

    nonQFeatures = ['femaleAs1MaleAs0', 'participant_id']
#    nonQFeatures = ['femaleAs1MaleAs0']

    qs_indices = []
    for i,n in enumerate(allData.getFeatureNames()):
        if n in nonQFeatures or (n[:2] == 'q_' and n != 'q_score'):
            qs_indices.append(i)

    responseData = allData.copyFeatures(qs_indices)

    return responseData

def checkResponseData(toCheck):
    """
    Assert that the scores in toCheck are in the range we expect
    """
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


def qResponsesToCatResponses(qResponses, categories, qParity):
    """
    Return a data set containing the category scores for each person
    """
    catList = [c for c in categories.keys()]

    def makeCats(point):
        ret = numpy.empty(len(catList))
        for i,c in enumerate(catList):
            q1 = categories[c][0]
            q1p = qParity[q1]
            q2 = categories[c][1]
            q2p = qParity[q2]
            ret[i] = (q1p * point[q1]) + (q2p * point[q2])
        return ret

    catResRet = qResponses.calculateForEachPoint(makeCats)
    catResRet.setFeatureNames(catList)

    return catResRet


def responsesToQuantity(responses):
    """
    Control function determining whether we count absolute or relative quantities
    of responses
    """
    return responsesToScorePercents(responses)

def responsesToScorePercents(responses):
    """
    Get score percents from score values.

    :param responses: feature vector category scores for a single category
    :return: a list mapping range(-6,7) to percent of responses for that value
    """
    total = float(responses.points)
    uDict = responses.countUniqueFeatureValues(0)
    responseValues = [i for i in range(-6,7)]
    for num in responseValues:
        if num not in uDict:
            uDict[num] = 0

    responseQuantity = [uDict[i]/total for i in responseValues]

    return responseQuantity


def plotData(gender, qData, cats, qParity, outPath):
    """
    Control function determining which style of bar plots we use: side by side or stacked
    """
    catData = qResponsesToCatResponses(qData, cats, qParity)

    for i in range(catData.features):
        f = catData.copyFeatures(i)
        cat = f.getFeatureName(0)
        currPath = os.path.join(outPath, cat)

        W = 960.0
        H = 432.0
        DPI = 96.0
        DPISAVE = 144.0
        fig = plt.figure(facecolor='white', figsize=(W / DPI, H / DPI), tight_layout=True)

#        xVals = plot_bar_sidebyside(gender, f)
        xVals = plot_bar_stacked(gender, f)

        tickNames = ["Least {0}".format(cat)] + ["Most {0}".format(cat)]
#        tickNames = ["Least {0}".format(cat)] + [""] * 11 + ["Most {0}".format(cat)]
#        tickNames = ["< Least {0}".format(cat)] + [""]*11 + ["Most {0} >".format(cat)]
        loc, labels = plt.xticks([xVals[0], xVals[-1]], tickNames)
#        loc, labels = plt.xticks(xVals, tickNames)
        labels[0].set_horizontalalignment('left')
        labels[-1].set_horizontalalignment('right')

        plt.ylim(0, .3)
        plt.yticks([0,.05,.1,.15,.2,.25, .3], ["0%", "5%", "10%", "15%", "20%", "25%", "30%"])
        plt.ylabel("Percent of respondents (within gender)")

        plt.savefig(currPath, dpi=DPISAVE)
        plt.close()

        # use PIL to cut vertical whitespace
#        toCrop = PIL.Image.open(currPath + ".png")
#        box = (0, 25, 1440, 626)
#        tight = toCrop.crop(box)
#        tight.save(currPath + ".png", "png")


def plot_bar_stacked(gender, toPlot):
    responseValues = numpy.array([i for i in range(-6, 7)])
    width = .9

    mData = toPlot.extractPoints(lambda p: gender[p.getPointName(0)] == 0)
    fData = toPlot
    mQuant = numpy.array(responsesToQuantity(mData))
    fQuant = numpy.array(responsesToQuantity(fData))
    overlap = numpy.min(numpy.vstack((mQuant,fQuant)), axis=0)

    plt.bar(responseValues, mQuant, width, color=colorM)
    plt.bar(responseValues, fQuant, width, color=colorF)
    plt.bar(responseValues, overlap, width, color=blendColor)

    M_patch = mpatches.Patch(color=colorM)
    F_patch = mpatches.Patch(color=colorF)
    mix_patch = mpatches.Patch(color=blendColor)
    handles = [M_patch, mix_patch, F_patch]
    labels = ["Male", "Overlap", "Female"]
    plt.legend(handles, labels, loc=2, frameon=False)

    return responseValues

def plot_bar_sidebyside(gender, toPlot):
    responseValues = numpy.array([i for i in range(-6, 7)])
    width = .45

    mData = toPlot.extractPoints(lambda p: gender[p.getPointName(0)] == 0)
    fData = toPlot
    mQuant = responsesToQuantity(mData)
    fQuant = responsesToQuantity(fData)

    plt.bar(responseValues, mQuant, width, color=colorM)
    plt.bar(responseValues+width, fQuant, width, color=colorF)

    one_patch = mpatches.Patch(color=colorM)
    two_patch = mpatches.Patch(color=colorF)
    handles = [one_patch, two_patch]
    labels = ["Male", "Female"]
    plt.legend(handles, labels, loc=2, frameon=False)

    return (responseValues + width / 2)


if __name__ == "__main__":
    import time
    print(time.asctime(time.localtime()))

    UML.registerCustomLearner("custom", LogisticRegressionNoTraining)

    # Constants controlling how the data is split in train and test sets
    TEST_NUMBER = 0
    SPLIT_SEED = 42

    PLOT_RESPONSES = True
    PARITY_GENDER_TRAINING = False
    ADD_SQUARED_FEATURES = False

    # Source Data
    sourceDir = sys.argv[1]
    path_responses = os.path.join(sourceDir, "inData", "gender_personality_final_training_data_999_people.csv")
    path_metadata = os.path.join(sourceDir, "inData", "gender_Q_Cat_meta.csv")

    # Output location for transformed data
    outpath_pred_coefs = os.path.join(sourceDir, "outData", "coefs.csv")
    outpath_pred_class_and_prob = os.path.join(sourceDir, "outData", "predictions.csv")

    # Output files for visualizations
    outDir_plots = os.path.join(sourceDir, "plots")

    # load response data and gender prediction variable
    predictionData = loadPredictionData(path_responses)
    categories, qParity = loadMetaData(path_metadata)

    if ADD_SQUARED_FEATURES:
        addSquaredFeatures(predictionData)

    if PLOT_RESPONSES:
        safeResponses = predictionData.copy()
        safeResponses.deleteFeatures("participant_id")
        safeGender = safeResponses.extractFeatures("femaleAs1MaleAs0")
        plotData(safeGender, safeResponses, categories, qParity, outDir_plots)
        sys.exit(0)

    splits = splitDataForTrial(predictionData, TEST_NUMBER, SPLIT_SEED)
    (responseTrain, genderTrain, responseTest, genderTest) = splits


    if PARITY_GENDER_TRAINING:
        if TEST_NUMBER == 0:
            responseTest = responseTrain.copy()
            genderTest = genderTrain.copy()
        responseTrain.appendFeatures(genderTrain)
        discardToParityGender(responseTrain)
        genderAdjusted = responseTrain.extractFeatures("femaleAs1MaleAs0")
        genderTrain.referenceDataFrom(genderAdjusted)

#    print(genderTrain.countUniqueFeatureValues("femaleAs1MaleAs0"))
#    print (genderTest.countUniqueFeatureValues("femaleAs1MaleAs0"))

#    predict_logReg_L2(responseTrain, genderTrain, responseTest, genderTest, outpath_pred_coefs, outpath_pred_class_and_prob)


    print(time.asctime(time.localtime()))





    pass  # EOF marker
