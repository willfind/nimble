"""
A group of tests which passes data to train, trainAndApply, and trainAndTest,
checking that after the call, the input data remains unmodified. It makes
use of listLearners and learnerType to try this operation with as many learners
as possible

"""
from unittest import mock

from nose.plugins.attrib import attr
from nose.tools import raises

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.helpers import generateClassificationData
from nimble.helpers import generateRegressionData
from nimble.randomness import pythonRandom
from .assertionHelpers import calledException, CalledFunctionException


def assertUnchanged4Obj(learnerName, passed, trainX, trainY, testX, testY):
    """
    When labels are separate nimble objects, assert that all 4 objects
    passed down into a function are identical to copies made before the
    call.
    """
    ((pTrainX, pTrainY), (pTestX, pTestY)) = passed

    if not pTrainX.isIdentical(trainX):
        raise ValueError(learnerName + " modified its trainX data")
    if not pTrainY.isIdentical(trainY):
        raise ValueError(learnerName + " modified its trainY data")
    if not pTestX.isIdentical(testX):
        raise ValueError(learnerName + " modified its testX data")
    if not pTestY.isIdentical(testY):
        raise ValueError(learnerName + " modified its testY data")

def assertUnchanged2Obj(learnerName, passed, train, test):
    """
    When labels are included in the data object, assert that the 2
    objects passed down into a function are identical to copies made
    before the call.
    """
    (pTrain, pTest) = passed

    if not pTrain.isIdentical(train):
        raise ValueError(learnerName + " modified its trainX data")
    if not pTest.isIdentical(test):
        raise ValueError(learnerName + " modified its testX data")

def handleApplyTestLabels(testX, testY):
    # testX cannot include labels for apply functions, remove labels if test Y is an ID.
    if isinstance(testY, (str, int)):
        testX = testX.copy()
        testX.features.delete(testY)
    return testX

def wrappedTrain(learnerName, trainX, trainY, testX, testY):
    return nimble.train(learnerName, trainX, trainY)


def wrappedTrainAndApply(learnerName, trainX, trainY, testX, testY):
    testX = handleApplyTestLabels(testX, testY)
    return nimble.trainAndApply(learnerName, trainX, trainY, testX)


def wrappedTLApply(learnerName, trainX, trainY, testX, testY):
    testX = handleApplyTestLabels(testX, testY)
    tl = nimble.train(learnerName, trainX, trainY)
    return tl.apply(testX)


def wrappedTrainAndApplyOvO(learnerName, trainX, trainY, testX, testY):
    testX = handleApplyTestLabels(testX, testY)
    return nimble.trainAndApply(learnerName, trainX, trainY, testX,
                                multiClassStrategy='OneVsOne')

def wrappedTrainAndApplyOvA(learnerName, trainX, trainY, testX, testY):
    testX = handleApplyTestLabels(testX, testY)
    return nimble.trainAndApply(learnerName, trainX, trainY, testX,
                                multiClassStrategy='OneVsAll')


def wrappedTrainAndTest(learnerName, trainX, trainY, testX, testY):
    # our performance function doesn't actually matter, we're just checking the data
    return nimble.trainAndTest(learnerName, trainX, trainY, testX, testY,
                               performanceFunction=nimble.calculate.fractionIncorrect)


def wrappedTLTest(learnerName, trainX, trainY, testX, testY):
    # our performance function doesn't actually matter, we're just checking the data
    tl = nimble.train(learnerName, trainX, trainY)
    return tl.test(testX, testY, performanceFunction=nimble.calculate.fractionIncorrect)


def wrappedTrainAndTestOvO(learnerName, trainX, trainY, testX, testY):
    return nimble.trainAndTest(learnerName, trainX, trainY, testX, testY,
                               performanceFunction=nimble.calculate.fractionIncorrect,
                               multiClassStrategy='OneVsOne')


def wrappedTrainAndTestOvA(learnerName, trainX, trainY, testX, testY):
    return nimble.trainAndTest(learnerName, trainX, trainY, testX, testY,
                               performanceFunction=nimble.calculate.fractionIncorrect,
                               multiClassStrategy='OneVsAll')


def wrappedCrossValidate(learnerName, trainX, trainY, testX, testY):
    return nimble.crossValidate(learnerName, trainX, trainY, performanceFunction=nimble.calculate.fractionIncorrect)


def setupAndCallIncrementalTrain(learnerName, trainX, trainY, testX, testY):
    tl = nimble.train(learnerName, trainX, trainY)
    tl.incrementalTrain(trainX.points.copy([0]), trainY.points.copy([0]))


def setupAndCallRetrain(learnerName, trainX, trainY, testX, testY):
    tl = nimble.train(learnerName, trainX, trainY)
    tl.retrain(trainX, trainY)


def setupAndCallGetScores(learnerName, trainX, trainY, testX, testY):
    tl = nimble.train(learnerName, trainX, trainY)
    testX = handleApplyTestLabels(testX, testY)
    tl.getScores(testX)


def backend(toCall, portionToTest, allowRegression=True, allowNotImplemented=False):
    cData = generateClassificationData(2, 10, 5)
    ((cTrainX, cTrainY), (cTestX, cTestY)) = cData
    # data and labels in one object; labels at idx 0
    cTrainCombined = cTrainY.copy()
    cTrainCombined.features.append(cTrainX)
    cTestCombined = cTestY.copy()
    cTestCombined.features.append(cTestX)

    backCTrainX = cTrainX.copy()
    backCTrainY = cTrainY.copy()
    backCTestX = cTestX.copy()
    backCTestY = cTestY.copy()
    backCTrainCombined = cTrainCombined.copy()
    backCTestCombined = cTestCombined.copy()

    rData = generateRegressionData(2, 10, 5)
    ((rTrainX, rTrainY), (rTestX, rTestY)) = rData
    # data and labels in one object; labels at idx 0
    rTrainCombined = rTrainY.copy()
    rTrainCombined.features.append(rTrainX)
    rTestCombined = rTestY.copy()
    rTestCombined.features.append(rTestX)

    backRTrainX = rTrainX.copy()
    backRTrainY = rTrainY.copy()
    backRTestX = rTestX.copy()
    backRTestY = rTestY.copy()
    backRTrainCombined = rTrainCombined.copy()
    backRTestCombined = rTestCombined.copy()

    allLearners = nimble.listLearners()
    numSamples = int(len(allLearners) * portionToTest)
    toTest = pythonRandom.sample(allLearners, numSamples)

#    toTest = filter(lambda x: x[:6] == 'shogun', allLearners)
    for learner in toTest:
        package = learner.split('.', 1)[0].lower()
        lType = nimble.learnerType(learner)
        if lType == 'classification':
            try:
                toCall(learner, cTrainX, cTrainY, cTestX, cTestY)
                toCall(learner, cTrainCombined, 0, cTestCombined, 0)
            # this is meant to safely bypass those learners that have required arguments
            except InvalidArgumentValue as iav:
                continue
            # this is generally how shogun explodes
            except SystemError as se:
                continue
            except NotImplementedError as nie:
                if not allowNotImplemented:
                    raise nie
                continue
            assertUnchanged4Obj(learner, cData, backCTrainX, backCTrainY,
                                backCTestX, backCTestY)
            assertUnchanged2Obj(learner, (cTrainCombined, cTestCombined),
                                backCTrainCombined, backCTestCombined)
        if lType == 'regression' and allowRegression:
            try:
                toCall(learner, rTrainX, rTrainY, rTestX, rTestY)
                toCall(learner, rTrainCombined, 0, rTestCombined, 0)
            # this is meant to safely bypass those learners that have required arguments
            except InvalidArgumentValue as iav:
                continue
            except SystemError as se:
                continue
            except NotImplementedError as nie:
                if not allowNotImplemented:
                    raise nie
                continue
            assertUnchanged4Obj(learner, rData, backRTrainX, backRTrainY,
                                backRTestX, backRTestY)
            assertUnchanged2Obj(learner, (rTrainCombined, rTestCombined),
                                backRTrainCombined, backRTestCombined)

@attr('slow')
def testDataIntegrityTrain():
    backend(wrappedTrain, 1)


@attr('slow')
def testDataIntegrityTrainAndApply():
    backend(wrappedTrainAndApply, 1)


@attr('slow')
def testDataIntegrityTLApply():
    backend(wrappedTLApply, 1)

# we can test smaller portions here because the backends are all being tested by
# the previous tests. We only care about the trainAndApply One vs One and One vs
# all code.
@attr('slow')
def testDataIntegrityTrainAndApplyMulticlassStrategies():
    backend(wrappedTrainAndApplyOvO, .1, False)
    backend(wrappedTrainAndApplyOvA, .1, False)


@attr('slow')
def testDataIntegrityTrainAndTest():
    backend(wrappedTrainAndTest, 1)


@attr('slow')
def testDataIntegrityTLTest():
    backend(wrappedTLTest, 1)

# we can test smaller portions here because the backends are all being tested by
# the previous tests. We only care about the trainAndTest One vs One and One vs
# all code.
@attr('slow')
def testDataIntegrityTrainAndTestMulticlassStrategies():
    backend(wrappedTrainAndTestOvO, .1, False)
    backend(wrappedTrainAndTestOvA, .1, False)

# test crossValidate x3
@attr('slow')
def testDataIntegrityCrossValidate():
    backend(wrappedCrossValidate, 1)

# test TrainedLearner methods
# only those that the top level trainers, appliers, and testers are not reliant on.
# Exclusions for above reason: apply(), test()
@attr('slow')
def testDataIntegrityTrainedLearner():
#	backend(setupAndCallIncrementalTrain, 1) TODO
    backend(setupAndCallRetrain, 1)
    backend(setupAndCallGetScores, 1, False, True)


#######################
# Arguments Integrity #
#######################

# _mergeArguments calls indicate that the user arguments parameter will be not
# be modified because it returns a new dictionary (verified in testHelpers)

def patch_mergeArguments(source):
    return mock.patch(source + '._mergeArguments', calledException)

@raises(CalledFunctionException)
@patch_mergeArguments('nimble.core')
def testArgumentIntegrityTrain():
    arguments = {'k': 1}
    train = nimble.createData('Matrix', [[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    tl = nimble.train('nimble.KNNClassifier', train, 2, arguments=arguments)

@raises(CalledFunctionException)
@patch_mergeArguments('nimble.core')
def testArgumentIntegrityTrainAndApply():
    arguments = {'k': 1}
    train = nimble.createData('Matrix', [[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    test = nimble.createData('Matrix', [[0, 1], [1, 0]])
    pred = nimble.trainAndApply('nimble.KNNClassifier', train, 2, test, arguments=arguments)

@raises(CalledFunctionException)
@patch_mergeArguments('nimble.core')
def testArgumentIntegrityTrainAndTest():
    arguments = {'k': 1}
    train = nimble.createData('Matrix', [[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    test = nimble.createData('Matrix', [[0, 1, 1], [1, 0, 2]])
    perf = nimble.trainAndTest('nimble.KNNClassifier', train, 2, test, 2,
                               performanceFunction=nimble.calculate.fractionIncorrect,
                               arguments=arguments)

@patch_mergeArguments('nimble.core')
def testArgumentIntegrityTrainAndTestOnTrainingData():
    arguments = {'k': 1}
    train = nimble.createData('Matrix', [[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    try:
        perf = nimble.trainAndTestOnTrainingData(
            'nimble.KNNClassifier', train, 2, arguments=arguments,
            performanceFunction=nimble.calculate.fractionIncorrect)
        assert False # expected CalledFunctionException
    except CalledFunctionException:
        pass
    try:
        perf = nimble.trainAndTestOnTrainingData(
            'nimble.KNNClassifier', train, 2, folds=2, arguments=arguments,
            crossValidationError=True,
            performanceFunction=nimble.calculate.fractionIncorrect)
        assert False # expected CalledFunctionException
    except CalledFunctionException:
        pass

@raises(CalledFunctionException)
@patch_mergeArguments('nimble.interfaces.universal_interface')
def testArgumentIntegrityTLApply():
    arguments = {'k': 1}
    train = nimble.createData('Matrix', [[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    test = nimble.createData('Matrix', [[0, 1], [1, 0]])
    tl = nimble.train('nimble.KNNClassifier', train, 2, arguments=arguments)
    pred = tl.apply(test)

@raises(CalledFunctionException)
@patch_mergeArguments('nimble.interfaces.universal_interface')
def testArgumentIntegrityTLTest():
    arguments = {'k': 1}
    train = nimble.createData('Matrix', [[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    test = nimble.createData('Matrix', [[0, 1, 1], [1, 0, 2]])
    tl = nimble.train('nimble.KNNClassifier', train, 2, arguments=arguments)
    perf = tl.test(test, 2, performanceFunction=nimble.calculate.fractionIncorrect)
