import sys
import numpy
import os.path

import PIL
import math
import scipy

import UML

from UML.calculate import fractionCorrect
from UML.calculate import residuals

from .gender_prediction import LogisticRegressionNoTraining

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties


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
    for point in iter(meta.points):
        categories[point["Category"]] = (point["Q1"], point["Q2"])
        qParity[point['Q1']] = point['Q1 parity to category']
        qParity[point['Q2']] = point['Q2 parity to category']

    return categories, qParity

def loadPredictionData(inPath):
    allData = UML.createData("Matrix", inPath, featureNames=True, pointNames=True)

    nonQFeatures = ['femaleAs1MaleAs0', 'participant_id']
#    nonQFeatures = ['femaleAs1MaleAs0']

    qs_indices = []
    for i,n in enumerate(allData.features.getNames()):
        if n in nonQFeatures or (n[:2] == 'q_' and n != 'q_score'):
            qs_indices.append(i)

    responseData = allData.features.copy(qs_indices)

    return responseData

def checkResponseData(toCheck):
    """
    Assert that the scores in toCheck are in the range we expect
    """
    def inRange(val):
        assert val >= -3 and val <=3

    toCheck.elements.calculate(inRange)


def discardToParityGender(toParity):
    counts = toParity.elements.countUnique(features="femaleAs1MaleAs0")

    totalMen = counts[0]
    totalWomen = counts[1]

    if totalMen < totalWomen:
        extractValue = 1
    elif totalWomen < totalMen:
        extractValue = 0
    else:  # if already equal, we don't need to do anything.
        return

    parityCount = min(totalMen, totalWomen)

    numerous = toParity.points.extract(lambda x: x["femaleAs1MaleAs0"] == extractValue)
    equalCount = numerous.points.extract(number=parityCount, randomize=True)

    assert len(equalCount.points) == len(toParity.points)
    assert len(equalCount.elements.countUnique(features="femaleAs1MaleAs0")) == 1

    toParity.points.add(equalCount)


def addNonLinearFeatures(toExpand):
    nonQ = toExpand.features.extract(lambda x: x.getFeatureName(0)[:2] != 'q_')

    squares = toExpand.copy()
    squares.elements.power(2)
    squares.features.setName([name + "^2" for name in toExpand.getFeatureNames()])

    cubes = toExpand.copy()
    cubes.elements.power(3)
    cubes.features.setName([name + "^3" for name in toExpand.getFeatureNames()])

#    print(squares[0,:])

    toExpand.features.add(squares)
    toExpand.features.add(cubes)

#    assert len(toExpand.points) == len(squares.points)
#    assert len(toExpand.points) == len(cubes.points)
#    assert len(toExpand.features) == (len(squares.features) * 2) + len(cubes.features)

    offset = len(squares.features)
    def checkValidity(toCheckP):
        for i in range(int(len(toCheckP.features) / 2)):
            assert toCheckP[i] == 0 or toCheckP[i] == (toCheckP[offset+i] / toCheckP[i])

#    toExpand.points.calculate(checkValidity)

    toExpand.features.add(nonQ)

#    for n in toExpand.features.getNames():
#        print (n)



def predict_decisionTree(trainX, trainY, testX, testY, ):

    train_pIDs = trainX.features.extract("participant_id")
    if "participant_id" in testX.features.getNames():
        test_pIDs = testX.features.extract("participant_id")
    else:
        test_pIDs = train_pIDs
    assert "participant_id" not in testX.features.getNames()

    cvargs = {}
#    cvargs['splitter'] = ("best", 'random')
    cvargs['max_depth'] = tuple([math.ceil(2**(x*.5)) for x in range(10)])
#    cvargs['min_samples_split'] = tuple([.66**x for x in range(2,14)])
#    cvargs['min_samples_leaf'] = tuple([.66**x for x in range(2, 14)])

#    print(cvargs['max_depth'])
#    print (cvargs['min_samples_split'])
#    print (cvargs['min_samples_leaf'])


    learner = UML.train("SKL.DecisionTreeClassifier", trainX, trainY,
                        performanceFunction=fractionCorrect,random_state=422, **cvargs)
    print (learner.test(testX, testY, performanceFunction=fractionCorrect))
    print ("\n")
    for k in cvargs.keys():
        print("{0}=".format(k) + str(learner.getAttributes()[k]))


def predict_knnClassifier(trainX, trainY, testX, testY, ):

    train_pIDs = trainX.features.extract("participant_id")
    if "participant_id" in testX.features.getNames():
        test_pIDs = testX.features.extract("participant_id")
    else:
        test_pIDs = train_pIDs
    assert "participant_id" not in testX.features.getNames()

    cvargs = {}
    cvargs['n_neighbors'] = (1,2,3,5,7,11,17,26,39,59)

#    for n_neighbors in (1,2,3,5,7,11,17,26,39,59):
#    learner = UML.train("SKL.KNeighborsClassifier", trainX, trainY,
#                        performanceFunction=fractionCorrect, n_neighbors=n_neighbors)  # , **cvargs)

    learner = UML.train("SKL.KNeighborsClassifier", trainX, trainY,
                        performanceFunction=fractionCorrect, **cvargs)
    print (learner.test(testX, testY, performanceFunction=fractionCorrect))
    print ("\n")
    for k in cvargs.keys():
        print("{0}=".format(k) + str(learner.getAttributes()[k]))



def predict_logReg_L2(trainX, trainY, testX, testY, pred_coefs_path, predictions_path):
    train_pIDs = trainX.features.extract("participant_id")

    if "participant_id" in testX.features.getNames():
        test_pIDs = testX.features.extract("participant_id")
    else:
        test_pIDs = train_pIDs
    assert "participant_id" not in testX.features.getNames()

#    print(trainX.points.getNames())
#    for n in trainY.features.getNames():
#        print(n)

    C = [3**(i-10) for i in range(0,20)]
#    print(C)
    C = tuple(C)

    trainY = UML.createData("Matrix", trainY.data, elementType=int, featureNames=["femaleAs1MaleAs0"])

    learner = UML.train("SKL.LogisticRegression", trainX, trainY, C=C, performanceFunction=fractionCorrect)

    print (learner.test(testX, testY, performanceFunction=fractionCorrect))
    print ("C={}".format(learner.getAttributes()['C']))

    predictions = learner.apply(testX)
    predictions.features.setNames(["predicted-femaleAs1MaleAs0"])
    predictions.points.setNames(testX.points.getNames())
    test_pIDs.features.add(predictions)
    test_pIDs.features.add(testY)

    raw = testX.copyAs("numpyarray")
    probabilities = learner.backend.predict_proba(raw)

    probabilitiesObj = UML.createData("Matrix", probabilities, featureNames=["prob_Male", "prob_Female"])
    probabilitiesObj.points.setNames(testX.points.getNames())
    test_pIDs.features.add(probabilitiesObj)
    test_pIDs.points.sort(sortHelper=(lambda x: int(x.getPointName(0))))

    test_pIDs.writeFile(predictions_path, 'csv')

#    print(test_pIDs.features.getNames())

    confObj = test_pIDs.copy()

    def makeConf(p):
        return max(p['prob_Male'], p['prob_Female'])

    confFeat = confObj.points.calculate(makeConf)
    confFeat.features.setNames(['confidence'])
    confObj.featurse.add(confFeat)
    confObj.points.sort('confidence')

    def isCorrect(p):
        return p["predicted-femaleAs1MaleAs0"] == p["femaleAs1MaleAs0"]

    isC = confObj.points.calculate(isCorrect)
    isC.features.setNames(['isCorrect'])
    confObj.features.add(isC)

    boundary = .05
    lowerIndex = 0
    upperIndex = 0
    perPointCorrectnessNeighborhood = numpy.empty((len(confObj.points),1))

    for i, val in enumerate(confObj[:, 'confidence']):
        lower = val - boundary
        upper = val + boundary

        # go until you're inside the lower range - inclusive index
        while confObj[lowerIndex, "confidence"] < lower:
            lowerIndex += 1
        # go until you're outside the upper range - exclusive index
        while upperIndex < len(confObj.points) and confObj[upperIndex, "confidence"] < upper:
            upperIndex += 1

        # but we want inclusive indices
        if upperIndex > i:
            upperIndex -= 1

#        print("l={0} i={1} u={2}".format(lowerIndex, i, upperIndex))

        correctRato = sum(confObj[lowerIndex:upperIndex, 'isCorrect'])/ ((upperIndex - lowerIndex) + 1)
        perPointCorrectnessNeighborhood[i] = correctRato

    cNObj = UML.createData("Matrix", perPointCorrectnessNeighborhood)
    cNObj.points.setNames(confObj.points.getNames())
    cNObj.features.setNames(['rollingAverageCorrectness'])
    confObj.features.add(cNObj)

    confObj.plotFeatureAgainstFeature("confidence", 'rollingAverageCorrectness', xMin=.4,xMax=1,yMin=.4,yMax=1)

    rawCoef = learner.getAttributes()['coef_']
    coefObj = UML.createData("Matrix", rawCoef)

    raw_intercept = learner.getAttributes()['intercept_'][0]

    independentModel = UML.train("custom.LogisticRegressionNoTraining", trainX, trainY, coefs=coefObj, intercept=raw_intercept)
    independentPredictions = independentModel.apply(testX)
    independentPredictions.features.setNames(["predicted-femaleAs1MaleAs0"])
    independentPredictions.points.setNames(testX.points.getNames())

    assert predictions.isIdentical(independentPredictions)
    assert learner.test(testX, testY, fractionCorrect) == independentModel.test(testX, testY, fractionCorrect)

    coefObj.features.setNames([n+" coef" for n in testX.features.getNames()])

    interceptObj = UML.createData("Matrix", [raw_intercept])
    interceptObj.features.setNames(["intercept"])
    interceptObj.features.add(coefObj)
    interceptObj.transpose()

    interceptObj.writeFile(pred_coefs_path, 'csv')



def splitDataForTrial(data, numTest, seed):
    # Split gender / response data for testing
    if numTest != 0:
        testFraction = float(numTest) / len(data.points)
        UML.setRandomSeed(seed)
        responseTrain, genderTrain, responseTest, genderTest = data.trainAndTestSets(testFraction, "femaleAs1MaleAs0", randomOrder=True)
    else:
        genderTrain = data.features.extract("femaleAs1MaleAs0")
        responseTrain = data
        genderTest = genderTrain
        responseTest = responseTrain

    return responseTrain, genderTrain, responseTest, genderTest


def qResponsesToCatResponses(qResponses, categories, qParity):
    """
    Return a data set containing the category scores for each person

    qResponses: object where each point is a person's list of of responses
    to each question

    categories: dict mapping the name of a category to a double of the questions
    in that category

    qParity: dict mapping the name of a question to the sign modifier needed to
    convert the question score into the scale of the category
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

    catResRet = qResponses.points.calculate(makeCats)
    catResRet.features.setNames(catList)

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
    total = float(len(responses.points))
    uDict = responses.elements.countUnique(features=0)
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

    for i in range(len(catData.features)):
        f = catData.features.copy(i)
        cat = f.features.getName(0)
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

    mData = toPlot.points.extract(lambda p: gender[p.points.getName(0)] == 0)
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

    mData = toPlot.points.extract(lambda p: gender[p.points.getName(0)] == 0)
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
#    DPISAVE = 144.0
    DPISAVE = 288.0
    RATIO = DPISAVE / DPI

    for i in range(catData.features):
        fig = plt.figure(facecolor='white', figsize=(W / DPI, H / DPI), tight_layout=True)

        # data setup
        f = catData.features.copy(i)
        cat = f.features.getName(0)
        currPath = os.path.join(outPath, cat)
        xVals = numpy.array([i for i in range(-6, 7)])
        mData = f.points.extract(lambda p: gender[p.points.getName(0)] == 0)
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
            spaces['Compassionate'] = 87
            spaces['Complexity Seeking'] = 72
            spaces['Emotionally Aware'] = 76
            spaces['Forgiving'] = 106
            spaces['Honest'] = 112
            spaces['Improvisational'] = 85
            spaces['Peaceful'] = 108
            spaces['Risk Taking'] = 98
            spaces['Self Defending'] = 88
            spaces['Self Valuing'] = 96


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
        meanMX = sum(mData) / float(len(mData.points))
        meanMY = getMeanHeight(meanMX, makeQMap(mQuant))
        meanFX = sum(fData) / float(len(fData.points))
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
            mscatter = plt.scatter(meanMX, meanMY, s=550, c=meanColorM, zorder=2)
            fscatter = plt.scatter(meanFX, meanFY, s=550, c=meanColorF, zorder=2)

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

        box = (0, 400, 960, 417)
        def adjust(val):
            return int(math.floor(val * RATIO))
        box = tuple(map(adjust, box))
#        box = (0, 600, 1440, 626)
        if makeXAxisLabels:
            toCrop = PIL.Image.open(currPath + ".png")
            tight = toCrop.crop(box)
            tight.save(outPath + "/labels/" + cat +".png", "png")
        else:  # glue labels to bottom
            toLabel = toCrop = PIL.Image.open(currPath + ".png")
            label = PIL.Image.open(outPath + "/labels/" + cat +".png")
            toLabel.paste(label, box)
            toLabel.save(currPath + ".png", "png")

        # use PIL to cut vertical whitespace
#        toCrop = PIL.Image.open(currPath + ".png")
#        box = (0, 25, 1440, 626)
#        tight = toCrop.crop(box)
#        tight.save(currPath + ".png", "png")


def plotCompanionCorrelationMatrix(qData, gender, categories, qParity, outFile):
    catData = qResponsesToCatResponses(qData, categories, qParity)

    residuals_catData = residuals(catData, gender)
    corrs = residuals_catData.feature.similarities('correlation')

#    corrs = catData.feature.similarities('correlation')

    corrs.points.sort(sortHelper=lambda x: x.points.getName(0))
    corrs.points.sort(sortHelper=lambda x: x.features.getName(0))
    corrs.writeFile(outFile)


if __name__ == "__main__":
    import time
    print(time.asctime(time.localtime()))

    UML.registerCustomLearner("custom", LogisticRegressionNoTraining)

    # Constants controlling the source of the input data
    USETRAINGDATA = False

    # Constants controlling how the data is split in train and test sets
    ALREADYSPLIT = False
    SPLITTEST_QUANTITY = 307
#    TEST_NUMBER = SPLITTEST_QUANTITY if ALREADYSPLIT else INDATA_QUANTITY_ASSERT * .2
    TEST_NUMBER = 0
    SPLIT_SEED = 42

    # Constants controlling validation of the input given the previous variables
    TRAINGDATA_QUANTITY_ASSERT = 999
    FULLDATA_QUANTITY_ASSERT = 1306
    if USETRAINGDATA:
        INDATA_QUANTITY_ASSERT = TRAINGDATA_QUANTITY_ASSERT
    else:
        if ALREADYSPLIT:
            INDATA_QUANTITY_ASSERT = TRAINGDATA_QUANTITY_ASSERT
        else:
            INDATA_QUANTITY_ASSERT = FULLDATA_QUANTITY_ASSERT

    # Constants controlling which tasks we do when running this script
    # NOTE: some of these may be mutually exclusive.
    PLOT_RESPONSES = False
    PLOT_MAKE_LABELS = False
    PARITY_GENDER_TRAINING = True
    ADD_SQUARED_FEATURES = False
    OUTPUT_PLOT_COMPANION_CORRELATION_MATRIX = False

    # Source Data
    sourceDir = sys.argv[1]
    if USETRAINGDATA:
        trainingDataFileName = "gender_personality_final_training_data_999_people.csv"
    else:
        trainingDataFileName = "gender_personality_final_combined_data_1306_people.csv"
    path_responses = os.path.join(sourceDir, "inData", trainingDataFileName)
    path_metadata = os.path.join(sourceDir, "inData", "gender_Q_Cat_meta.csv")
    path_FinalTest_responses = os.path.join(sourceDir, "inData", "GENDER STUDY VALIDATION DATA 308 SUBJECTS.csv")

    # Output location for transformed data
    outpath_pred_coefs = os.path.join(sourceDir, "outData", "coefs.csv")
    outpath_pred_class_and_prob = os.path.join(sourceDir, "outData", "predictions.csv")

    # Output files for visualizations
    outDir_plots = os.path.join(sourceDir, "plots")

    # Output data for analysis data
    outpath_plotCompanion_correlation = os.path.join(sourceDir, "metaData", "category_correlations.csv")

    # load response data and gender prediction variable
    predictionData = loadPredictionData(path_responses)
    assert len(predictionData.points) == INDATA_QUANTITY_ASSERT
    categories, qParity = loadMetaData(path_metadata)

    if PLOT_RESPONSES:
        safeResponses = predictionData.copy()
        safeResponses.features.delete("participant_id")
        safeGender = safeResponses.features.extract("femaleAs1MaleAs0")
        if PLOT_MAKE_LABELS:
            safeResponses2 = safeResponses.copy()
            safeGender2 = safeGender.copy()
            plotData(safeGender2, safeResponses2, categories, qParity, outDir_plots, PLOT_MAKE_LABELS)
        plotData(safeGender, safeResponses, categories, qParity, outDir_plots, False)
        sys.exit(0)

    if OUTPUT_PLOT_COMPANION_CORRELATION_MATRIX:
        safeResponses = predictionData.copy()
        safeResponses.features.delete("participant_id")
        safeGender = safeResponses.features.extract("femaleAs1MaleAs0")
        plotCompanionCorrelationMatrix(safeResponses, safeGender, categories, qParity,outpath_plotCompanion_correlation)

    if ADD_SQUARED_FEATURES:
        addNonLinearFeatures(predictionData)

    if ALREADYSPLIT:
        assert USETRAINGDATA  # were we using the full data, the data could not have been already split
        responseTest = loadPredictionData(path_FinalTest_responses)
        genderTest = responseTest.features.extract("femaleAs1MaleAs0")
        (responseTrain, genderTrain, _, _) = splitDataForTrial(predictionData, 0, SPLIT_SEED)
    else:
        splits = splitDataForTrial(predictionData, TEST_NUMBER, SPLIT_SEED)
        (responseTrain, genderTrain, responseTest, genderTest) = splits

    print ("train: {0} | test: {1}".format(len(responseTrain.points), len(responseTest.points)))

    if PARITY_GENDER_TRAINING:
        if TEST_NUMBER == 0:
            responseTest = responseTrain.copy()
            genderTest = genderTrain.copy()
        responseTrain.features.add(genderTrain)
        discardToParityGender(responseTrain)
        genderAdjusted = responseTrain.features.extract("femaleAs1MaleAs0")
        genderTrain.referenceDataFrom(genderAdjusted)
        print("parity size={0}".format(str(len(genderTrain.points) / 2)))

    print(genderTrain.elements.countUnique(features="femaleAs1MaleAs0"))
    print(genderTest.elements.countUnique(features="femaleAs1MaleAs0"))

    print("\nlogReg_L2")
    predict_logReg_L2(responseTrain.copy(), genderTrain.copy(), responseTest.copy(), genderTest.copy(), outpath_pred_coefs, outpath_pred_class_and_prob)

#    print("\ndecision tree")
#    predict_decisionTree(responseTrain.copy(), genderTrain.copy(), responseTest.copy(), genderTest.copy(),)

#    print("\nknnclassifier")
#    predict_knnClassifier(responseTrain.copy(), genderTrain.copy(), responseTest.copy(), genderTest.copy(),)

    print(time.asctime(time.localtime()))





s    pass  # EOF marker
