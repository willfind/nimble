"""
Tests for the user facing functions for cross validation and
the backend helpers they rely on.
"""

from __future__ import absolute_import
from __future__ import print_function
import math
import sys
import numpy

import nose
from nose.tools import *
from nose.plugins.attrib import attr

import UML

from UML import crossValidate
from UML import createData

from UML.exceptions import InvalidArgumentValueCombination
from UML.calculate import *
from UML.randomness import pythonRandom
from UML.helpers import computeMetrics
from UML.helpers import generateClassificationData
from UML.customLearners import CustomLearner
from UML.configuration import configSafetyWrapper
from six.moves import range
from .assertionHelpers import oneLogEntryExpected


def _randomLabeledDataSet(dataType='Matrix', numPoints=50, numFeatures=5, numLabels=3):
    """returns a tuple of two data objects of type dataType
    the first object in the tuple contains the feature information ('X' in UML language)
    the second object in the tuple contains the labels for each feature ('Y' in UML language)
    """
    if numLabels is None:
        labelsRaw = [[pythonRandom.random()] for _x in range(numPoints)]
    else:  # labels data set
        labelsRaw = [[int(pythonRandom.random() * numLabels)] for _x in range(numPoints)]

    rawFeatures = [[pythonRandom.random() for _x in range(numFeatures)] for _y in range(numPoints)]

    return (createData(dataType, rawFeatures, useLog=False), createData(dataType, labelsRaw, useLog=False))


def test_crossValidate_XY_unchanged():
    """assert that after running cross validate on datasets passed to
    X and Y, the original data is unchanged

    """
    classifierAlgo = 'Custom.KNNClassifier'
    X, Y = _randomLabeledDataSet(numLabels=5)
    copyX = X.copy()
    copyY = Y.copy()
    resultsObj = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
    assert X == copyX
    assert Y == copyY


def test_crossValidate_callable():
    """tests that crossValidate is callable on different learners with
    different UML data types

    """
    #just scrap data to make sure it doesn't crash
    numLabels = 3
    numPoints = 10

    for dType in UML.data.available:
        X, Y = _randomLabeledDataSet(numPoints=numPoints, numLabels=numLabels, dataType=dType)

        classifierAlgos = ['Custom.KNNClassifier']
        for curAlgo in classifierAlgos:
            resultsObj = crossValidate(curAlgo, X, Y, fractionIncorrect, {}, numFolds=3)
            assert isinstance(resultsObj.bestScore, float)

            #With regression dataset (no repeated labels)
            X, Y = _randomLabeledDataSet(numPoints=numPoints, numLabels=None, dataType=dType)
            classifierAlgos = ['Custom.RidgeRegression']
            for curAlgo in classifierAlgos:
                resultsObj = crossValidate(curAlgo, X, Y, meanAbsoluteError, {}, numFolds=3)
                assert isinstance(resultsObj.bestScore, float)


def _assertClassifierErrorOnRandomDataPlausible(actualError, numLabels, tolerance=.1):
    """assert the actual error on a labeled data set (for a classifier)
    is plausible, given the number of (evenly distributed) labels in hte data set
    """
    idealFractionIncorrect = 1.0 - 1.0 / numLabels
    error = abs(actualError - idealFractionIncorrect)
    assert error <= tolerance


@attr('slow')
def test_crossValidate_reasonable_results():
    """Assert that crossValidate returns reasonable errors for known algorithms
    on cooked data sets:
    crossValidate should do the following:
    classifiers:
        have no error when there is only one label in the data set
        classify random data at roughly the accuracy of 1/numLabels
    regressors:
        LinearRegression - have no error when the dataset all lies on one plane
    """
    classifierAlgo = 'Custom.KNNClassifier'
    #assert that when whole dataset has the same label, crossValidated score
    #reflects 100% accruacy (with a classifier)
    X, Y = _randomLabeledDataSet(numLabels=1)
    resultsObj = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
    assert resultsObj.bestScore < 0.000001  # 0 incorrect ever

    #assert that a random dataset will have accuracy roughly equal to 1/numLabels
    numLabelsList = [2, 3, 5]
    for curNumLabels in numLabelsList:
        X, Y = _randomLabeledDataSet(numPoints=50, numLabels=curNumLabels)
        resultsObj = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
        _assertClassifierErrorOnRandomDataPlausible(
            resultsObj.bestScore, curNumLabels, tolerance=(1.0 / curNumLabels))

    #assert that for an easy dataset (no noise, overdetermined linear hyperplane!),
    #crossValidated error is perfect
    #for all folds, with simple LinearRegression
    regressionAlgo = 'Custom.RidgeRegression'

    #make random data set where all points lie on a linear hyperplane
    numFeats = 3
    numPoints = 50
    points = [[pythonRandom.gauss(0, 1) for _x in range(numFeats)] for _y in range(numPoints)]
    labels = [[sum(featVector)] for featVector in points]
    X = createData('Matrix', points)
    Y = createData('Matrix', labels)

    #run in crossValidate
    resultsObj = crossValidate(regressionAlgo, X, Y, meanAbsoluteError, {}, numFolds=5)
    #assert error essentially zero since there's no noise
    assert resultsObj.bestScore < .001

    index = len(X.features)
    X.features.add(Y)
    resultsObj = crossValidate(regressionAlgo, X, index, meanAbsoluteError, {}, numFolds=5)
    #assert error essentially zero since there's no noise
    assert resultsObj.bestScore < .001

    # ensures nonmodification of X data object when getting Y data
    assert len(X.features) == 4
    X.features.setNames(['X1', 'X2', 'X3', 'Y'])
    resultsObj = crossValidate(regressionAlgo, X, 'Y', meanAbsoluteError, {}, numFolds=5)
    assert len(X.features) == 4
    #assert error essentially zero since there's no noise
    assert resultsObj.bestScore < .001


def test_crossValidate_2d_api_check():
    """Check that crossValidate is callable with 2d data given to the Y argument

    """
    # using an easy dataset (no noise, overdetermined linear hyperplane!),
    # check that crossValidated error is perfect for all folds, with simple
    # LinearRegression
    regressionAlgo = 'Custom.RidgeRegression'

    #make random data set where all points lie on a linear hyperplane
    numFeats = 3
    numPoints = 50
    points = [[pythonRandom.gauss(0, 1) for _x in range(numFeats)] for _y in range(numPoints)]
    labels = [[sum(featVector), sum(featVector)] for featVector in points]
    X = createData('Matrix', points)
    Y = createData('Matrix', labels)

    # crossValidate.bestScore
    metric = meanFeaturewiseRootMeanSquareError
    resultsObj = crossValidate(regressionAlgo, X, Y, metric, {}, numFolds=5)
    #assert error essentially zero since there's no noise
    assert isinstance(resultsObj.bestScore, float)
    assert resultsObj.bestScore < .001

    index = len(X.features)
    combined = X.copy()
    combined.features.add(Y)
    combined.features.setNames(['X1', 'X2', 'X3', 'Y1', 'Y2'])
    resultsObj = crossValidate(regressionAlgo, combined, [index, 'Y2'], metric, {}, numFolds=5)
    #assert error essentially zero since there's no noise
    assert isinstance(resultsObj.bestScore, float)
    assert resultsObj.bestScore < .001

    # repeat for crossValidate.results
    resultsObj = crossValidate(regressionAlgo, X, Y, metric, {}, numFolds=5)
    assert len(resultsObj.results) == 1
    assert len(resultsObj.results[0]) == 2
    assert isinstance(resultsObj.results[0][0], dict) and resultsObj.results[0][0] == {}
    assert isinstance(resultsObj.results[0][1], float) and resultsObj.results[0][1] < .001

    resultsObj = crossValidate(regressionAlgo, combined, [index, 'Y2'], metric, {}, numFolds=5)
    assert len(resultsObj.results) == 1
    assert len(resultsObj.results[0]) == 2
    assert isinstance(resultsObj.results[0][0], dict) and resultsObj.results[0][0] == {}
    assert isinstance(resultsObj.results[0][1], float) and resultsObj.results[0][1] < .001

    # repeat for crossValidate.bestArgument
    resultsObj = crossValidate(regressionAlgo, X, Y, metric, {}, numFolds=5)
    assert isinstance(resultsObj.bestArguments, dict) and resultsObj.bestArguments == {}


def test_crossValidate_2d_Non_label_scoremodes_disallowed():
    """
    Cross validation methods on 2d label data disallow non-default scoreModes

    """
    #assert that for an easy dataset (no noise, overdetermined linear hyperplane!),
    #crossValidated error is perfect
    #for all folds, with simple LinearRegression
    regressionAlgo = 'Custom.RidgeRegression'

    #make random data set where all points lie on a linear hyperplane
    numFeats = 3
    numPoints = 50
    points = [[pythonRandom.gauss(0, 1) for _x in range(numFeats)] for _y in range(numPoints)]
    labels = [[sum(featVector), sum(featVector)] for featVector in points]
    X = createData('Matrix', points)
    Y = createData('Matrix', labels)

    #run in crossValidate
    metric = meanFeaturewiseRootMeanSquareError
    try:
        crossValidate(regressionAlgo, X, Y, metric, {}, numFolds=5, scoreMode='bestScore')
        assert False
    except InvalidArgumentValueCombination:
        pass

    try:
        crossValidate(regressionAlgo, X, Y, metric, {}, numFolds=5, scoreMode='allScores')
        assert False
    except InvalidArgumentValueCombination:
        pass


@attr('slow')
@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def test_crossValidate_foldingRandomness():
    """Assert that for a dataset, the same algorithm will generate the same model
    (and have the same accuracy) when presented with identical random state (and
    therefore identical folds).
    Assert that the model is different when the random state is different

    """
    numTrials = 5
    for _ in range(numTrials):
        X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
        seed = UML.randomness.pythonRandom.randint(0, 2**32 - 1)
        UML.setRandomSeed(seed)
        resultOne = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, {}, numFolds=3)
        UML.setRandomSeed(seed)
        resultTwo = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, {}, numFolds=3)
        assert resultOne.bestScore == resultTwo.bestScore

@attr('slow')
@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def test_crossValidateResults():
    """Check basic properties of crossValidateReturnAll

    assert that default arguments will be filled in by the function
    assert that having the same function arguments yields the same results.
    assert that return all gives a cross validated performance for all of its
    parameter permutations
    """
    X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
    #try with no extra arguments at all; yet we know an argument exists (k):
    resultsObj = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect)
    assert resultsObj.results
    assert 1 == len(resultsObj.results)
    assert resultsObj.results[0][0] == {}
    #try with some extra elements, including the default
    resultsObj = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, k=UML.CV([1, 2, 3]))
    assert resultsObj.results
    assert 3 == len(resultsObj.results)

    # since the same seed is used, and these calls are effectively building the
    # same arguments, the scores in results list should be the same, though
    # ordered differently
    seed = UML.randomness.pythonRandom.randint(0, 2**32 - 1)
    UML.setRandomSeed(seed)
    result1 = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, k=UML.CV([1, 2, 3, 4, 5]))
    UML.setRandomSeed(seed)
    result2 = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, k=UML.CV([1, 5, 4, 3, 2]))
    #assert the the resulting SCORES are identical
    #uncertain about the order
    resultOneScores = [curEntry[1] for curEntry in result1.results]
    resultTwoScores = [curEntry[1] for curEntry in result2.results]
    resultsOneSet = set(resultOneScores)
    resultsTwoSet = set(resultTwoScores)
    assert resultsOneSet == resultsTwoSet

    #assert results have the expected data structure:
    #a list of tuples where the first entry is the argument dict
    #and second entry is the score (float)
    assert isinstance(result1.results, list)
    for curResult in result1.results:
        assert isinstance(curResult, tuple)
        assert isinstance(curResult[0], dict)
        assert isinstance(curResult[1], float)


@attr('slow')
@configSafetyWrapper
@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def test_crossValidateReturnBest():
    """Check that the best / fittest argument set is returned.

    """
    # needs to be binary: FlipWrapper only works on binary classification
    # data
    ((X, Y), (testX, testY)) = generateClassificationData(2, 20, 5)

    # need to setup a situation where we guarantee certain returns
    # from the performanceMetric fractionIncorrect. Thus, we generate
    # obvious data, that custom.KNNClassifer will predict with 100%
    # accuracy, and FlipWrapper messes up a specified percentage
    # of the returns
    class FlipWrapper(CustomLearner):
        learnerType = "classification"

        def train(self, trainX, trainY, wrapped, flip, **args):
            self.trained = UML.train(wrapped, trainX, trainY, **args)
            self.flip = flip

        def apply(self, testX):
            num = int(math.floor(len(testX.points) * self.flip))
            ret = self.trained.apply(testX).copy(to='pythonList')
            for i in range(num):
                if ret[i][0] == 0:
                    ret[i][0] = 1
                else:
                    ret[i][0] = 0
            return ret

    UML.registerCustomLearner('custom', FlipWrapper)

    # want to have a predictable random state in order to control folding
    seed = UML.randomness.pythonRandom.randint(0, 2**32 - 1)

    def trial(metric, maximize):
        # get a baseline result
        UML.setRandomSeed(seed)
        resultsObj = crossValidate('custom.FlipWrapper', X, Y,
                                   metric, flip=UML.CV([0, .5, .9]),
                                   wrapped="custom.KNNClassifier")
        resultTuple = (resultsObj.bestArguments, resultsObj.bestScore)
        assert resultTuple

        # Confirm that the best result is also returned in the 'returnAll' results
        UML.setRandomSeed(seed)
        resultsObj = crossValidate('custom.FlipWrapper', X, Y,
                                   metric, flip=UML.CV([0, .5, .9]),
                                   wrapped="custom.KNNClassifier")
        allResultsList = resultsObj.results
        #since same args were used, the best tuple should be in allResultsList
        allArguments = [curResult[0] for curResult in allResultsList]
        allScores = [curResult[1] for curResult in allResultsList]
        assert resultTuple[0] in allArguments
        assert resultTuple[1] in allScores

        # confirm that we have actually tested something: ie, that there is a difference
        # in the results and the ordering therefore matters
        for i in range(len(allScores)):
            for j in range(i + 1, len(allScores)):
                assert allScores[i] != allScores[j]

        # verify that resultTuple was in fact the best in allResultsList
        for curError in allScores:
            #assert that the error is not 'better' than our best error:
            if maximize:
                assert curError <= resultTuple[1]
            else:
                assert curError >= resultTuple[1]


    trial(fractionIncorrect, False)
    trial(fractionCorrect, True)

    UML.deregisterCustomLearner('custom', 'FlipWrapper')


def test_crossValidateReturnEtc_withDefaultArgs():
    """Assert that return best and return all work with default arguments as predicted
    ie generating scores for '{}' as the arguments
    """
    X, Y = _randomLabeledDataSet(numPoints=20, numFeatures=5, numLabels=5)
    #run with default arguments
    resultsObj = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect)
    bestTuple = (resultsObj.bestArguments, resultsObj.bestScore)
    assert bestTuple
    assert isinstance(bestTuple, tuple)
    assert bestTuple[0] == {}
    #run return all with default arguments
    resultsObj = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect)
    allResultsList = resultsObj.results
    assert allResultsList
    assert 1 == len(allResultsList)
    assert allResultsList[0][0] == {}


@attr('slow')
def test_crossValidate_sameResults_avgfold_vs_allcollected():
    # When whole dataset has the same label, crossValidated score
    #reflects 100% accruacy (with a classifier)
    classifierAlgo = 'Custom.KNNClassifier'
    X, Y = _randomLabeledDataSet(numLabels=1)

    def copiedPerfFunc(knowns, predicted):
        return fractionIncorrect(knowns, predicted)

    copiedPerfFunc.optimal = fractionIncorrect.optimal

    copiedPerfFunc.avgFolds = False
    resultsObj = crossValidate(classifierAlgo, X, Y, copiedPerfFunc, {}, numFolds=5)
    nonAvgResult = resultsObj.bestScore

    copiedPerfFunc.avgFolds = True
    resultsObj = crossValidate(classifierAlgo, X, Y, copiedPerfFunc, {}, numFolds=5)
    avgResult = resultsObj.bestScore

    # 0 incorrect ever
    assert nonAvgResult < 0.000001
    # For this metric, the result should be the same either way; helps confirm that
    # both methods are correctly implemented if they agree
    assert nonAvgResult == avgResult

    #For an easy dataset (no noise, overdetermined linear hyperplane!),
    #crossValidated error is perfect
    regressionAlgo = 'Custom.RidgeRegression'

    #make random data set where all points lie on a linear hyperplane
    numFeats = 3
    numPoints = 50
    points = [[pythonRandom.gauss(0, 1) for _x in range(numFeats)] for _y in range(numPoints)]
    labels = [[sum(featVector)] for featVector in points]
    X = createData('Matrix', points)
    Y = createData('Matrix', labels)

    def copiedPerfFunc(knowns, predicted):
        return meanAbsoluteError(knowns, predicted)

    copiedPerfFunc.optimal = fractionIncorrect.optimal

    copiedPerfFunc.avgFolds = False
    resultsObj = crossValidate(regressionAlgo, X, Y, copiedPerfFunc, {}, numFolds=5)
    nonAvgResult = resultsObj.bestScore

    copiedPerfFunc.avgFolds = True
    resultsObj = crossValidate(regressionAlgo, X, Y, copiedPerfFunc, {}, numFolds=5)
    avgResult = resultsObj.bestScore

    #assert error essentially zero since there's no noise
    assert nonAvgResult < .0000001
    # For this metric, the result should be the same either way; helps confirm that
    # both methods are correctly implemented if they agree
    assert abs(nonAvgResult - avgResult) < .0000001


@configSafetyWrapper
def test_crossValidate_sameResults_avgfold_vs_allcollected_orderReliant():
    # To dispel concerns relating to correctly collecting the Y data with
    # respect to the ordering of the X data in the non-average CV code.

    class UnitPredictor(CustomLearner):
        learnerType = "classification"

        def train(self, trainX, trainY, bozoArg):
            return

        def apply(self, testX):
            return testX.copy()

    def copiedPerfFunc(knowns, predicted):
        return fractionIncorrect(knowns, predicted)

    copiedPerfFunc.optimal = fractionIncorrect.optimal

    UML.registerCustomLearner('custom', UnitPredictor)

    data = [1, 3, 5, 6, 8, 4, 10, -12, -2, 22]
    X = UML.createData("Matrix", data)
    X.transpose()
    Y = UML.createData("Matrix", data)
    Y.transpose()

    copiedPerfFunc.avgFolds = False
    resultsObj = crossValidate('custom.UnitPredictor', X, Y, copiedPerfFunc, {'bozoArg': (1, 2)}, numFolds=5)
    nonAvgResult = resultsObj.bestScore

    copiedPerfFunc.avgFolds = True
    resultsObj = crossValidate('custom.UnitPredictor', X, Y, copiedPerfFunc, {'bozoArg': (1, 2)}, numFolds=5)
    avgResult = resultsObj.bestScore

    # should have 100 percent accuracy, so these results should be the same
    assert nonAvgResult == avgResult

@oneLogEntryExpected
def back_crossValidate_logCount(toCall):
    classifierAlgo = 'Custom.KNNClassifier'
    X, Y = _randomLabeledDataSet(numLabels=5)
    copyX = X.copy()
    copyY = Y.copy()
    result = toCall(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)

@oneLogEntryExpected
def test_crossValidate_logCount():
    back_crossValidate_logCount(crossValidate)
