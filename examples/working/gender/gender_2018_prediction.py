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
import math
import UML

from UML.calculate import fractionCorrect
from UML.examples.working.gender.gender_prediction import LogisticRegressionNoTraining

scipy = UML.importModule("scipy")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties


#ORIG_HELPER_PATH = "/home/tpburns/Dropbox/ML_intern_tpb/python_workspace/"
#sys.path.append(ORIG_HELPER_PATH)


colorM_raw ='#66a6ff'
colorF_raw ='#ff668c'
meanColorM = '#3d6399'
meanColorF = '#993d54'

cc = matplotlib.colors.ColorConverter()
colorM = cc.to_rgb(colorM_raw)
colorF = cc.to_rgb(colorF_raw)
blendColor = ((colorM[0] + colorF[0])/2., (colorM[1] + colorF[1])/2., (colorM[2] + colorF[2])/2.)

# Use if you want all text uniformly larger
#matplotlib.rc('font', size=12)

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


def plotDataOLD(gender, qData, cats, qParity, outPath):
    """
    Control function determining which style of bar plots we use: side by side or stacked
    """
    catData = qResponsesToCatResponses(qData, cats, qParity)

    for i in range(catData.features):
        f = catData.copyFeatures(i)
        cat = f.getFeatureName(0)
        print(cat)
        currPath = os.path.join(outPath, cat)

        W = 960.0
        H = 432.0
        DPI = 96.0
        DPISAVE = 144.0
        fig = plt.figure(facecolor='white', figsize=(W / DPI, H / DPI), tight_layout=True)

#        xVals = plot_bar_sidebyside(gender, f)
#        xVals = plot_bar_stacked(gender, f)
        xVals = plot_dotsWithLines(gender, f)

#        tickNames = ["Least\n{0}".format(cat), "Most\n{0}".format(cat)]
#        tickNames = ["Least {0}".format(cat)] + [""] * 11 + ["Most {0}".format(cat)]
#        tickNames = ["< Least {0}".format(cat)] + [""]*11 + ["Most {0} >".format(cat)]
        tickNames = range(-6,7)
#        loc, labels = plt.xticks([xVals[0], xVals[-1]], tickNames)
        loc, labels = plt.xticks(xVals, tickNames)
#        labels[0].set_horizontalalignment('left')
#        labels[-1].set_horizontalalignment('right')

#        xLab = "Least {0}".format(cat) + (" " * 60) + "Most {0}".format(cat)
#        xLab = "1        0" * 10
#        plt.xlabel(xLab, horizontalalignment='center')

        plt.ylim(0)
#        plt.yticks([0,.05,.1,.15,.2,.25, .3], ["0%", "5%", "10%", "15%", "20%", "25%", "30%"])
#        ytickVals = plt.yticks()[0]
#        ytickStr = list(map((lambda n: str(int(n*100))+"%"), ytickVals))

#        plt.yticks(ytickVals, ytickStr)
        plt.ylabel("% of respondents", fontweight='bold')

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


def plotData(gender, qData, cats, qParity, outPath, makeXAxisLabels):
    """
    Control function determining which style of bar plots we use: side by side or stacked
    """
    catData = qResponsesToCatResponses(qData, cats, qParity)

    # control vars shared by each plot
    W = 960.0
    H = 432.0
    DPI = 96.0
    DPISAVE = 144.0

    for i in range(catData.features):
        fig = plt.figure(facecolor='white', figsize=(W / DPI, H / DPI), tight_layout=True)

        # data setup
        f = catData.copyFeatures(i)
        cat = f.getFeatureName(0)
        currPath = os.path.join(outPath, cat)
        xVals = numpy.array([i for i in range(-6, 7)])
        mData = f.extractPoints(lambda p: gender[p.getPointName(0)] == 0)
        fData = f
        mQuant = responsesToQuantity(mData)
        fQuant = responsesToQuantity(fData)

        # plotting the actual lines
        mLine = plt.plot(xVals, mQuant, marker='.', color=colorM, linestyle='-', markersize=10, zorder=1)
        fLine = plt.plot(xVals, fQuant, marker='.', color=colorF, linestyle='-', markersize=10, zorder=1)

        # legend setup
        one_patch = mpatches.Patch(color=colorM)
        two_patch = mpatches.Patch(color=colorF)
        handles = [one_patch, two_patch]
        labels = ["Males", "Females"]
        leg = plt.legend(handles, labels, loc=2, frameon=False)

        # try to set legend text to same color as box
#        for line,text in zip(leg.get_lines(), leg.texts):
#            text.set_color(line.get_color())

        # X axis ticks and label
#        tickNames = range(-6, 7)
        tickNames = [-3, "", -2, "", -1, "", 0, "", 1, "", 2, "", 3]
        loc, labels = plt.xticks(xVals, tickNames)

        # Y axis limit, ticks and label
        plt.ylim(0)
        ytickVals = plt.yticks()[0]
        ytickStr = list(map((lambda n: str(int(n*100))+"%"), ytickVals))
        plt.yticks(ytickVals, ytickStr)
        plt.ylabel("% of respondents", fontweight='bold', fontsize=12)

        if makeXAxisLabels:
            spaces = {}
            # For default text size
#            spaces['Aesthetic'] = 138
#            spaces['Amicable'] = 140
#            spaces['At Ease'] = 144
#            spaces['Compassionate'] = 120
#            spaces['Complexity Seeking'] = 106
#            spaces['Emotionally Aware'] = 110
#            spaces['Forgiving'] = 140
#            spaces['Honest'] = 148
#            spaces['Improvisational'] = 120
#            spaces['Peaceful'] = 142
#            spaces['Risk Taking'] = 134
#            spaces['Self Defending'] = 122
#            spaces['Self Valuing'] = 131
#            spaces['Sex Focused'] = 131
#            spaces['Thick Skinned'] = 124
#            spaces['Unselfish'] = 140
#            spaces['Unusual'] = 144
#            spaces['Warm'] = 150
            spaces['Aesthetic'] = 106
            spaces['Amicable'] = 107
            spaces['At Ease'] = 112
            spaces['Compassionate'] = 86
            spaces['Complexity Seeking'] = 72
            spaces['Emotionally Aware'] = 75
            spaces['Forgiving'] = 105
            spaces['Honest'] = 112
            spaces['Improvisational'] = 85
            spaces['Peaceful'] = 108
            spaces['Risk Taking'] = 98
            spaces['Self Defending'] = 88
            spaces['Self Valuing'] = 92
            spaces['Sex Focused'] = 95
            spaces['Thick Skinned'] = 90
            spaces['Unselfish'] = 106
            spaces['Unusual'] = 110
            spaces['Warm'] = 117


            xLab = " "*15 + "Less {0}".format(cat) + (" " * spaces[cat]) + "More {0}".format(cat) + " "*5
            plt.xlabel(xLab, horizontalalignment='center', fontweight='bold', fontsize=12)
        else:
            # fill in dummy label
            plt.xlabel("TEMP", horizontalalignment='center', fontweight='bold', fontsize=12)

        # mean helpers
        def makeQMap(quantities):
            ret = {}
            for i,val in enumerate(quantities):
                ret[i-6] = val
            return ret

        def getMeanHeight(mean, qMap):
            leftSideIndex = int(math.floor(mean))
            leftSideHeight = qMap[leftSideIndex]
            rightSideIndex = int(math.ceil(mean))
            rightSideHeight = qMap[rightSideIndex]
            slope = rightSideHeight - leftSideHeight
            height = leftSideHeight + (slope * abs(mean - leftSideIndex))
            assert height >= min(leftSideHeight, rightSideHeight)
            assert height <= max(leftSideHeight, rightSideHeight)
            return height

        # mean positions
        meanMX = sum(mData) / float(mData.points)
        meanMY = getMeanHeight(meanMX, makeQMap(mQuant))
        meanFX = sum(fData) / float(fData.points)
        meanFY = getMeanHeight(meanFX, makeQMap(fQuant))

        # possible: 'line', 'full line', 'bubble_legend', 'bubble_inside'
        meanStyle = 'bubble_inside'

        if meanStyle == 'full line':
            # plot a full height dotted line at mean
            pHeight = plt.ylim()[1]
            bottom = 0
            top = (37/40.0)
            plt.axvline(meanMX, bottom, top, linestyle='--', linewidth=3, color=meanColorM)
            plt.axvline(meanFX, bottom, top, linestyle='--', linewidth=3, color=meanColorF)

            # annotate mean
            meanM = str(round(meanMX/2.0, 1))
            meanF = str(round(meanFX/2.0, 1))
            meanFont = FontProperties()
            meanFont.set_weight("bold")
            drop = pHeight - (pHeight/20.0)
            shiftUp = pHeight / 11.0
            shiftDown = pHeight / 9.0
            plt.annotate(meanM, xy=(meanMX, drop), color=meanColorM,
                     horizontalalignment='center', fontproperties=meanFont)
            plt.annotate(meanF, xy=(meanFX, drop), color=meanColorF,
                     horizontalalignment='center', fontproperties=meanFont)

        if meanStyle == 'line':
            # plot a line at mean
            pHeight = plt.ylim()[1]
            relMY = meanMY / pHeight
            relMYB = relMY - .05
            relMYT = relMY + .05
            relFY = meanFY / pHeight
            relFYB = relFY - .05
            relFYT = relFY + .05
            plt.axvline(meanMX, relMYB, relMYT, linewidth=3, color=meanColorM)
            plt.axvline(meanFX, relFYB, relFYT, linewidth=3, color=meanColorF)

            # annotate mean
            meanM = str(round(meanMX/2.0, 1))
            meanF = str(round(meanFX/2.0, 1))
            meanFont = FontProperties()
            meanFont.set_weight("bold")
            drop = pHeight - (pHeight/20.0)
            shiftUp = pHeight / 11.0
            shiftDown = pHeight / 9.0
            plt.annotate(meanM, xy=(meanMX, drop), color=meanColorM,
                     horizontalalignment='center', fontproperties=meanFont)
            plt.annotate(meanF, xy=(meanFX, drop), color=meanColorF,
                     horizontalalignment='center', fontproperties=meanFont)

        if meanStyle == 'bubble_legend':
            mscatter = plt.scatter(meanMX, meanMY, s=250, c=meanColorM, zorder=2)
            fscatter = plt.scatter(meanFX, meanFY, s=250, c=meanColorF, zorder=2)

            mT = str(round(meanMX/2.0, 1))
            fT = str(round(meanFX/2.0, 1))
            shared = {}
            shared['fontsize'] = 8
            shared['horizontalalignment'] = 'center'
            shared['verticalalignment'] = 'center'
            shared['color'] = 'white'
            plt.text(meanMX, meanMY, mT, **shared)
            plt.text(meanFX, meanFY, fT, **shared)

            # legend resetup
            one_patch = mpatches.Patch(color=colorM)
            mm_patch = mpatches.Circle((), color=meanColorM)
            two_patch = mpatches.Patch(color=colorF)
            mf_patch = mpatches.Circle((), color=meanColorF)
            handles = [fscatter, two_patch, mscatter, one_patch]
            labels = ["Female Average", "Females", "Male Average", "Males"]
            leg = plt.legend(handles, labels, loc=2, frameon=False)

        if meanStyle == 'bubble_inside':
            mscatter = plt.scatter(meanMX, meanMY, s=465, c=meanColorM, zorder=2)
            fscatter = plt.scatter(meanFX, meanFY, s=465, c=meanColorF, zorder=2)

            mTVal = round(meanMX/2.0, 1)
            mTVal = abs(mTVal) if mTVal == -0 else mTVal
            fTVal = round(meanFX/2.0, 1)
            fTVal = abs(fTVal) if fTVal == -0 else fTVal
            mT = str(mTVal) + '\navg'
            fT = str(fTVal) + '\navg'
            shared = {}
            shared['fontsize'] = 8
            shared['fontweight'] = 'bold'
            shared['horizontalalignment'] = 'center'
            shared['verticalalignment'] = 'center'
            shared['color'] = 'white'
            plt.text(meanMX, meanMY, mT, **shared)
            plt.text(meanFX, meanFY, fT, **shared)

        # save with size adjusting dpi
        plt.savefig(currPath, dpi=DPISAVE)
        plt.close()

        if makeXAxisLabels:
            toCrop = PIL.Image.open(currPath + ".png")
            box = (0, 600, 1440, 626)
            tight = toCrop.crop(box)
            tight.save(outPath + "/labels/" + cat +".png", "png")
        else:  # glue labels to bottom
            toLabel = toCrop = PIL.Image.open(currPath + ".png")
            label = PIL.Image.open(outPath + "/labels/" + cat +".png")
            toLabelBox = (0, 600, 1440, 626)
            toLabel.paste(label, toLabelBox)
            toLabel.save(currPath + ".png", "png")

        # use PIL to cut vertical whitespace
#        toCrop = PIL.Image.open(currPath + ".png")
#        box = (0, 25, 1440, 626)
#        tight = toCrop.crop(box)
#        tight.save(currPath + ".png", "png")



if __name__ == "__main__":
    import time
    print(time.asctime(time.localtime()))

    UML.registerCustomLearner("custom", LogisticRegressionNoTraining)

    # Constants controlling input data
    USETRAINGDATA = False
    TRAINGDATA_QUANTITY_ASSERT = 999
    TESTDATA_QUANTITY_ASSERT = 1306
    INDATA_QUANTITY_ASSERT = TRAINGDATA_QUANTITY_ASSERT if USETRAINGDATA else TESTDATA_QUANTITY_ASSERT

    # Constants controlling how the data is split in train and test sets
    TEST_NUMBER = 0
    SPLIT_SEED = 42

    PLOT_RESPONSES = True
    PLOT_MAKE_LABELS =True
    PARITY_GENDER_TRAINING = False
    ADD_SQUARED_FEATURES = False

    # Source Data
    sourceDir = sys.argv[1]
    if USETRAINGDATA:
        path_responses = os.path.join(sourceDir, "inData", "gender_personality_final_training_data_999_people.csv")
    else:
        path_responses = os.path.join(sourceDir, "inData", "gender_personality_final_combined_data_1306_people.csv")
    path_metadata = os.path.join(sourceDir, "inData", "gender_Q_Cat_meta.csv")

    # Output location for transformed data
    outpath_pred_coefs = os.path.join(sourceDir, "outData", "coefs.csv")
    outpath_pred_class_and_prob = os.path.join(sourceDir, "outData", "predictions.csv")

    # Output files for visualizations
    outDir_plots = os.path.join(sourceDir, "plots")

    # load response data and gender prediction variable
    predictionData = loadPredictionData(path_responses)
    assert predictionData.points == INDATA_QUANTITY_ASSERT
    categories, qParity = loadMetaData(path_metadata)


    if ADD_SQUARED_FEATURES:
        addSquaredFeatures(predictionData)

    if PLOT_RESPONSES:
        safeResponses = predictionData.copy()
        safeResponses.deleteFeatures("participant_id")
        safeGender = safeResponses.extractFeatures("femaleAs1MaleAs0")
        if PLOT_MAKE_LABELS:
            safeResponses2 = safeResponses.copy()
            safeGender2 = safeGender.copy()
            plotData(safeGender2, safeResponses2, categories, qParity, outDir_plots, PLOT_MAKE_LABELS)
        plotData(safeGender, safeResponses, categories, qParity, outDir_plots, False)
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
