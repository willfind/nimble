"""
Tests for the user facing functions for cross validation and
the backend helpers they rely on.
"""

import math
import sys
from unittest import mock

import numpy
import nose
from nose.tools import *
from nose.plugins.attrib import attr

import nimble
from nimble import crossValidate
from nimble import createData
from nimble import CV
from nimble import CustomLearner
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction
from nimble.calculate import *
from nimble.random import pythonRandom
from nimble.learners import KNNClassifier
from nimble.core.helpers import computeMetrics
from nimble.core.helpers import generateClassificationData
from nimble.core.helpers import KFoldCrossValidator
from .assertionHelpers import configSafetyWrapper
from .assertionHelpers import oneLogEntryExpected


def _randomLabeledDataSet(dataType='Matrix', numPoints=50, numFeatures=5, numLabels=3):
    """returns a tuple of two data objects of type dataType
    the first object in the tuple contains the feature information ('X' in nimble language)
    the second object in the tuple contains the labels for each feature ('Y' in nimble language)
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
    for classifierAlgo in ['nimble.KNNClassifier', KNNClassifier]:
        X, Y = _randomLabeledDataSet(numLabels=5)
        copyX = X.copy()
        copyY = Y.copy()
        crossValidator = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, folds=5)
        assert X == copyX
        assert Y == copyY


def test_crossValidate_callable():
    """tests that crossValidate is callable on different learners with
    different nimble data types

    """
    #just scrap data to make sure it doesn't crash
    numLabels = 3
    numPoints = 10

    for dType in nimble.core.data.available:
        X, Y = _randomLabeledDataSet(numPoints=numPoints, numLabels=numLabels, dataType=dType)

        classifierAlgos = ['nimble.KNNClassifier']
        for curAlgo in classifierAlgos:
            crossValidator = crossValidate(curAlgo, X, Y, fractionIncorrect, {}, folds=3)
            assert isinstance(crossValidator.bestResult, float)

            #With regression dataset (no repeated labels)
            X, Y = _randomLabeledDataSet(numPoints=numPoints, numLabels=None, dataType=dType)
            classifierAlgos = ['nimble.RidgeRegression']
            for curAlgo in classifierAlgos:
                crossValidator = crossValidate(curAlgo, X, Y, meanAbsoluteError, {}, folds=3)
                assert isinstance(crossValidator.bestResult, float)


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
    classifierAlgo = 'nimble.KNNClassifier'
    #assert that when whole dataset has the same label, crossValidated score
    #reflects 100% accruacy (with a classifier)
    X, Y = _randomLabeledDataSet(numLabels=1)
    crossValidator = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, folds=5)
    assert crossValidator.bestResult < 0.000001  # 0 incorrect ever

    #assert that a random dataset will have accuracy roughly equal to 1/numLabels
    numLabelsList = [2, 3, 5]
    for curNumLabels in numLabelsList:
        X, Y = _randomLabeledDataSet(numPoints=50, numLabels=curNumLabels)
        crossValidator = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, folds=5)
        _assertClassifierErrorOnRandomDataPlausible(
            crossValidator.bestResult, curNumLabels, tolerance=(1.0 / curNumLabels))

    #assert that for an easy dataset (no noise, overdetermined linear hyperplane!),
    #crossValidated error is perfect
    #for all folds, with simple LinearRegression
    regressionAlgo = 'nimble.RidgeRegression'

    #make random data set where all points lie on a linear hyperplane
    numFeats = 3
    numPoints = 50
    points = [[pythonRandom.gauss(0, 1) for _x in range(numFeats)] for _y in range(numPoints)]
    labels = [[sum(featVector)] for featVector in points]
    X = createData('Matrix', points)
    Y = createData('Matrix', labels)

    #run in crossValidate
    crossValidator = crossValidate(regressionAlgo, X, Y, meanAbsoluteError, {}, folds=5)
    #assert error essentially zero since there's no noise
    assert crossValidator.bestResult < .001

    index = len(X.features)
    X.features.append(Y)
    crossValidator = crossValidate(regressionAlgo, X, index, meanAbsoluteError, {}, folds=5)
    #assert error essentially zero since there's no noise
    assert crossValidator.bestResult < .001

    # ensures nonmodification of X data object when getting Y data
    assert len(X.features) == 4
    X.features.setNames(['X1', 'X2', 'X3', 'Y'])
    crossValidator = crossValidate(regressionAlgo, X, 'Y', meanAbsoluteError, {}, folds=5)
    assert len(X.features) == 4
    #assert error essentially zero since there's no noise
    assert crossValidator.bestResult < .001


def test_crossValidate_2d_api_check():
    """Check that crossValidate is callable with 2d data given to the Y argument

    """
    # using an easy dataset (no noise, overdetermined linear hyperplane!),
    # check that crossValidated error is perfect for all folds, with simple
    # LinearRegression
    regressionAlgo = 'nimble.RidgeRegression'

    #make random data set where all points lie on a linear hyperplane
    numFeats = 3
    numPoints = 50
    points = [[pythonRandom.gauss(0, 1) for _x in range(numFeats)] for _y in range(numPoints)]
    labels = [[sum(featVector), sum(featVector)] for featVector in points]
    X = createData('Matrix', points)
    Y = createData('Matrix', labels)

    # crossValidate.bestResult
    metric = meanFeaturewiseRootMeanSquareError
    crossValidator = crossValidate(regressionAlgo, X, Y, metric, {}, folds=5)
    #assert error essentially zero since there's no noise
    assert isinstance(crossValidator.bestResult, float)
    assert crossValidator.bestResult < .001

    index = len(X.features)
    combined = X.copy()
    combined.features.append(Y)
    combined.features.setNames(['X1', 'X2', 'X3', 'Y1', 'Y2'])
    crossValidator = crossValidate(regressionAlgo, combined, [index, 'Y2'], metric, {}, folds=5)
    #assert error essentially zero since there's no noise
    assert isinstance(crossValidator.bestResult, float)
    assert crossValidator.bestResult < .001

    # repeat for crossValidate.allResults
    crossValidator = crossValidate(regressionAlgo, X, Y, metric, {}, folds=5)
    resultsList = crossValidator.allResults
    assert len(resultsList) == 1
    assert isinstance(resultsList[0], dict) and len(resultsList[0]) == 1
    assert metric.__name__ in resultsList[0]
    performance = resultsList[0][metric.__name__]
    assert isinstance(performance, float) and performance < .001

    crossValidator = crossValidate(regressionAlgo, combined, [index, 'Y2'], metric, {}, folds=5)
    resultsList = crossValidator.allResults
    assert len(resultsList) == 1
    assert isinstance(resultsList[0], dict) and len(resultsList[0]) == 1
    assert metric.__name__ in resultsList[0]
    performance = resultsList[0][metric.__name__]
    assert isinstance(performance, float) and performance < .001

    # repeat for crossValidate.bestArgument
    crossValidator = crossValidate(regressionAlgo, X, Y, metric, {}, folds=5)
    assert isinstance(crossValidator.bestArguments, dict) and crossValidator.bestArguments == {}


def test_crossValidate_2d_Non_label_scoremodes_disallowed():
    """
    Cross validation methods on 2d label data disallow non-default scoreModes

    """
    #assert that for an easy dataset (no noise, overdetermined linear hyperplane!),
    #crossValidated error is perfect
    #for all folds, with simple LinearRegression
    regressionAlgo = 'nimble.RidgeRegression'

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
        crossValidate(regressionAlgo, X, Y, metric, {}, folds=5, scoreMode='bestScore')
        assert False
    except InvalidArgumentValueCombination:
        pass

    try:
        crossValidate(regressionAlgo, X, Y, metric, {}, folds=5, scoreMode='allScores')
        assert False
    except InvalidArgumentValueCombination:
        pass


@attr('slow')
@nose.with_setup(nimble.random._startAlternateControl, nimble.random._endAlternateControl)
def test_crossValidate_foldingRandomness():
    """Assert that for a dataset, the same algorithm will generate the same model
    (and have the same accuracy) when presented with identical random state (and
    therefore identical folds).
    Assert that the model is different when the random state is different

    """
    numTrials = 5
    for _ in range(numTrials):
        X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
        seed = nimble.random.pythonRandom.randint(0, 2**32 - 1)
        nimble.random.setSeed(seed)
        resultOne = crossValidate('nimble.KNNClassifier', X, Y, fractionIncorrect, {}, folds=3)
        nimble.random.setSeed(seed)
        resultTwo = crossValidate('nimble.KNNClassifier', X, Y, fractionIncorrect, {}, folds=3)
        assert resultOne.bestResult == resultTwo.bestResult

@attr('slow')
@nose.with_setup(nimble.random._startAlternateControl, nimble.random._endAlternateControl)
def test_crossValidateResults():
    """Check basic properties of crossValidate.allResults

    assert that default arguments will be filled in by the function
    assert that having the same function arguments yields the same results.
    assert that return all gives a cross validated performance for all of its
    parameter permutations
    """
    X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
    #try with no extra arguments at all; yet we know an argument exists (k):
    crossValidator = crossValidate('nimble.KNNClassifier', X, Y, fractionIncorrect)
    resultsList = crossValidator.allResults
    assert resultsList
    assert 1 == len(resultsList)
    assert len(resultsList[0]) == 1 and 'fractionIncorrect' in resultsList[0]
    #try with some extra elements, including the default
    crossValidator = crossValidate('nimble.KNNClassifier', X, Y, fractionIncorrect, k=nimble.CV([1, 2, 3]))
    resultsList = crossValidator.allResults
    assert resultsList
    assert 3 == len(resultsList)

    # since the same seed is used, and these calls are effectively building the
    # same arguments, the scores in results list should be the same, though
    # ordered differently
    seed = nimble.random.pythonRandom.randint(0, 2**32 - 1)
    nimble.random.setSeed(seed)
    result1 = crossValidate('nimble.KNNClassifier', X, Y, fractionIncorrect, k=nimble.CV([1, 2, 3, 4, 5]))
    nimble.random.setSeed(seed)
    result2 = crossValidate('nimble.KNNClassifier', X, Y, fractionIncorrect, k=nimble.CV([1, 5, 4, 3, 2]))
    #assert the the resulting SCORES are identical
    #uncertain about the order
    resultOneScores = [curEntry['fractionIncorrect'] for curEntry in result1.allResults]
    resultTwoScores = [curEntry['fractionIncorrect'] for curEntry in result2.allResults]
    resultsOneSet = set(resultOneScores)
    resultsTwoSet = set(resultTwoScores)
    assert resultsOneSet == resultsTwoSet

    #assert results have the expected data structure:
    #a list of tuples where the first entry is the argument dict
    #and second entry is the score (float)
    assert isinstance(result1.allResults, list)
    for curResult in result1.allResults:
        assert isinstance(curResult, dict)
        assert isinstance(curResult['fractionIncorrect'], float)


@attr('slow')
@configSafetyWrapper
@nose.with_setup(nimble.random._startAlternateControl, nimble.random._endAlternateControl)
def test_crossValidateBestArguments():
    """Check that the best / fittest argument set is returned.

    """
    # needs to be binary: FlipWrapper only works on binary classification
    # data
    ((X, Y), (testX, testY)) = generateClassificationData(2, 20, 5)

    # need to setup a situation where we guarantee certain returns
    # from the performanceMetric fractionIncorrect. Thus, we generate
    # obvious data, that custom.KNNClassifier will predict with 100%
    # accuracy, and FlipWrapper messes up a specified percentage
    # of the returns
    class FlipWrapper(CustomLearner):
        learnerType = "classification"

        def train(self, trainX, trainY, wrapped, flip, **args):
            self.trained = nimble.train(wrapped, trainX, trainY, **args)
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

    # want to have a predictable random state in order to control folding
    seed = nimble.random.pythonRandom.randint(0, 2**32 - 1)

    def trial(metric, maximize):
        # get a baseline result
        nimble.random.setSeed(seed)
        crossValidator = crossValidate(FlipWrapper, X, Y,
                                   metric, flip=nimble.CV([0, .5, .9]),
                                   wrapped="nimble.KNNClassifier")
        resultTuple = (crossValidator.bestArguments, crossValidator.bestResult)
        assert resultTuple

        # Confirm that the best result is also returned in the 'returnAll' results
        nimble.random.setSeed(seed)
        crossValidator = crossValidate(FlipWrapper, X, Y,
                                   metric, flip=nimble.CV([0, .5, .9]),
                                   wrapped="nimble.KNNClassifier")
        allResultsList = crossValidator.allResults
        #since same args were used, the best tuple should be in allResultsList
        allArguments = []
        allScores = []
        for result in allResultsList:
            score = result.pop(metric.__name__)
            allArguments.append(result)
            allScores.append(score)
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


def test_crossValidate_attributes_withDefaultArgs():
    """Assert that return best and return all work with default arguments as predicted
    ie generating scores for '{}' as the arguments
    """
    X, Y = _randomLabeledDataSet(numPoints=20, numFeatures=5, numLabels=5)
    #run with default arguments
    crossValidator = crossValidate('nimble.KNNClassifier', X, Y, fractionIncorrect)
    bestTuple = (crossValidator.bestArguments, crossValidator.bestResult)
    assert bestTuple
    assert isinstance(bestTuple, tuple)
    assert bestTuple[0] == {}
    #run return all with default arguments
    crossValidator = crossValidate('nimble.KNNClassifier', X, Y, fractionIncorrect)
    allResultsList = crossValidator.allResults
    assert allResultsList
    assert 1 == len(allResultsList)
    assert len(allResultsList[0]) == 1 and 'fractionIncorrect' in allResultsList[0]


@attr('slow')
def test_crossValidate_sameResults_avgfold_vs_allcollected():
    # When whole dataset has the same label, crossValidated score
    #reflects 100% accruacy (with a classifier)
    classifierAlgo = 'nimble.KNNClassifier'
    X, Y = _randomLabeledDataSet(numLabels=1)

    def copiedPerfFunc(knowns, predicted):
        return fractionIncorrect(knowns, predicted)

    copiedPerfFunc.optimal = fractionIncorrect.optimal

    copiedPerfFunc.avgFolds = False
    crossValidator = crossValidate(classifierAlgo, X, Y, copiedPerfFunc, {}, folds=5)
    nonAvgResult = crossValidator.bestResult

    copiedPerfFunc.avgFolds = True
    crossValidator = crossValidate(classifierAlgo, X, Y, copiedPerfFunc, {}, folds=5)
    avgResult = crossValidator.bestResult

    # 0 incorrect ever
    assert nonAvgResult < 0.000001
    # For this metric, the result should be the same either way; helps confirm that
    # both methods are correctly implemented if they agree
    assert nonAvgResult == avgResult

    #For an easy dataset (no noise, overdetermined linear hyperplane!),
    #crossValidated error is perfect
    regressionAlgo = 'nimble.RidgeRegression'

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
    crossValidator = crossValidate(regressionAlgo, X, Y, copiedPerfFunc, {}, folds=5)
    nonAvgResult = crossValidator.bestResult

    copiedPerfFunc.avgFolds = True
    crossValidator = crossValidate(regressionAlgo, X, Y, copiedPerfFunc, {}, folds=5)
    avgResult = crossValidator.bestResult

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

    data = [1, 3, 5, 6, 8, 4, 10, -12, -2, 22]
    X = nimble.createData("Matrix", data)
    X.transpose()
    Y = nimble.createData("Matrix", data)
    Y.transpose()

    copiedPerfFunc.avgFolds = False
    crossValidator = crossValidate(UnitPredictor, X, Y, copiedPerfFunc, {'bozoArg': (1, 2)}, folds=5)
    nonAvgResult = crossValidator.bestResult

    copiedPerfFunc.avgFolds = True
    crossValidator = crossValidate(UnitPredictor, X, Y, copiedPerfFunc, {'bozoArg': (1, 2)}, folds=5)
    avgResult = crossValidator.bestResult

    # should have 100 percent accuracy, so these results should be the same
    assert nonAvgResult == avgResult

class MockedKFoldCrossValidator(KFoldCrossValidator):
    def _crossValidate(self, X, Y, useLog):
        pass

@mock.patch('tests.testCrossValidate.KFoldCrossValidator', MockedKFoldCrossValidator)
def test_KFoldCrossValidator():
    # we have mocked the object so it will not actually perform crossValidation
    # instead we will pass our own dummy data to the backend so we can check
    # the outputs match our expectations
    learnerName = 'Dummy.learner'
    X, Y = None, None # not actually used when not crossValidating
    performanceFunction = rootMeanSquareError
    aVals = nimble.CV([1, 2])
    arguments = {'a': aVals}
    bVals = nimble.CV([1, 2]) # will be passed as kwarguments
    folds = 3
    scoreMode = 'label'

    crossValidator = KFoldCrossValidator(
        learnerName, X, Y, performanceFunction, arguments, folds, scoreMode,
        useLog=False, b=bVals)
    assert crossValidator.learnerName == learnerName
    assert crossValidator.folds == folds
    assert crossValidator.performanceFunction == performanceFunction
    assert crossValidator.arguments == {'a': aVals, 'b': bVals}
    assert crossValidator.scoreMode == 'label'

    # add dummy results
    results = [({'a': 1, 'b': 1}, .25), ({'a': 1, 'b': 2}, .5),
               ({'a': 2, 'b': 1}, .75), ({'a': 2, 'b': 2}, 1.0)]
    crossValidator._allResults = results
    assert crossValidator.allResults != results
    retResults = [{'a': 1, 'b': 1, 'rootMeanSquareError': .25},
                  {'a': 1, 'b': 2, 'rootMeanSquareError': .5},
                  {'a': 2, 'b': 1, 'rootMeanSquareError': .75},
                  {'a': 2, 'b': 2, 'rootMeanSquareError': 1.0}]
    assert crossValidator.allResults == retResults
    # best arguments/score should not be calculated until requested
    assert crossValidator._bestArguments is None
    assert crossValidator._bestResult is None
    assert crossValidator.bestArguments == {'a': 1, 'b': 1}
    # accessing bestArguments property will also set bestScore
    assert crossValidator._bestResult is not None
    assert crossValidator.bestResult == .25
    assert crossValidator.getResult(crossValidator.bestArguments) == .25
    assert crossValidator.getResult(a=2, b=1) == .75
    assert crossValidator.getResult({'a': 1}, b=2) == .5

    # add some dummy fold results
    fold_results = ([({'a': 1, 'b': 1}, .25), ({'a': 1, 'b': 2}, .5),
                     ({'a': 2, 'b': 1}, .75), ({'a': 2, 'b': 2}, 1.0)] * folds)
    crossValidator._resultsByFold = fold_results
    assert crossValidator.getFoldResults({'a': 1, 'b': 1}) == [.25] * folds
    assert crossValidator.getFoldResults(a=2, b=2) == [1.0] * folds
    assert crossValidator.getFoldResults({'a': 1}, b=2) == [.5] * folds

    # test allResults for lambda or unnamed function
    retResults = [{'a': 1, 'b': 1, 'performance': .25},
                  {'a': 1, 'b': 2, 'performance': .5},
                  {'a': 2, 'b': 1, 'performance': .75},
                  {'a': 2, 'b': 2, 'performance': 1.0}]
    # these performanceFunction are invalid, but since we have already
    # instantiated this object this is acceptable because allResults only
    # attempts to access the __name__ attribute of a performanceFunction
    crossValidator.performanceFunction = None
    assert crossValidator.allResults == retResults
    crossValidator.performanceFunction = lambda y, yPred: None
    assert crossValidator.allResults == retResults

@raises(InvalidArgumentValue)
def test_KFoldCrossValidator_invalidPerformanceFunction():
    def noOptimal(yActual, yPred):
        return 100

    xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 1], [1, 1, 0], [0, 1, 1]]
    yRaw = [[1], [2], [3],
            [1], [2], [3],
            [1], [2], [3]]
    X = nimble.createData('Matrix', xRaw)
    Y = nimble.createData('Matrix', yRaw)
    crossValidator = KFoldCrossValidator(
        'nimble.KNNClassifier', X, Y, arguments={'k': 3},
        performanceFunction=noOptimal, folds=3)

@raises(InvalidArgumentValue)
def test_KFoldCrossValidator_zeroFolds():
    xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 0, 1], [1, 1, 0], [0, 1, 1]]
    yRaw = [[1], [2], [3],
            [1], [2], [3],
            [1], [2], [3]]
    X = nimble.createData('Matrix', xRaw)
    Y = nimble.createData('Matrix', yRaw)
    crossValidator = KFoldCrossValidator(
        'nimble.KNNClassifier', X, Y, arguments={'k': 3},
        performanceFunction=nimble.calculate.fractionIncorrect, folds=0)

def test_CV():
    crossVal = CV([1, 2, 3])
    assert len(crossVal) == 3
    assert crossVal[1] == 2
    assert str(crossVal) == "(1, 2, 3)"
    assert repr(crossVal) == "CV([1, 2, 3])"

@raises(ImproperObjectAction)
def test_CV_immutable():
    crossVal = CV([1, 2, 3])
    # can get
    assert crossVal[1] == 2
    # cannot set
    crossVal[1] = 0

@oneLogEntryExpected
def back_crossValidate_logCount(toCall):
    classifierAlgo = 'nimble.KNNClassifier'
    X, Y = _randomLabeledDataSet(numLabels=5)
    copyX = X.copy()
    copyY = Y.copy()
    result = toCall(classifierAlgo, X, Y, fractionIncorrect, {}, folds=5)

@oneLogEntryExpected
def test_crossValidate_logCount():
    back_crossValidate_logCount(crossValidate)
