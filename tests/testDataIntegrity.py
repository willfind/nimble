"""
A group of tests which passes data to train, trainAndApply, and trainAndTest, 
checking that after the call, the input data remains unmodified. It makes
use of listLearners and learnerType to try this operation with as many learners
as possible

"""

from __future__ import absolute_import
from nose.plugins.attrib import attr

import UML

from UML.exceptions import ArgumentException

from UML.helpers import generateClassificationData
from UML.helpers import generateRegressionData
from UML.randomness import pythonRandom


def assertUnchanged(learnerName, passed, trainX, trainY, testX, testY):
    """
    Helper to assert that those objects passed down into a function are
    identical to copies made before the call
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


def wrappedTrain(learnerName, trainX, trainY, testX, testY):
    return UML.train(learnerName, trainX, trainY)


def wrappedTrainAndApply(learnerName, trainX, trainY, testX, testY):
    return UML.trainAndApply(learnerName, trainX, trainY, testX)


def wrappedTrainAndApplyOvO(learnerName, trainX, trainY, testX, testY):
    return UML.helpers.trainAndApplyOneVsOne(learnerName, trainX, trainY, testX)


def wrappedTrainAndApplyOvA(learnerName, trainX, trainY, testX, testY):
    return UML.helpers.trainAndApplyOneVsAll(learnerName, trainX, trainY, testX)


def wrappedTrainAndTest(learnerName, trainX, trainY, testX, testY):
    # our performance function doesn't actually matter, we're just checking the data
    return UML.trainAndTest(learnerName, trainX, trainY, testX, testY,
                            performanceFunction=UML.calculate.fractionIncorrect)


def wrappedTrainAndTestOvO(learnerName, trainX, trainY, testX, testY):
    return UML.helpers.trainAndTestOneVsOne(learnerName, trainX, trainY, testX, testY,
                                            performanceFunction=UML.calculate.fractionIncorrect)


def wrappedTrainAndTestOvA(learnerName, trainX, trainY, testX, testY):
    return UML.helpers.trainAndTestOneVsAll(learnerName, trainX, trainY, testX, testY,
                                            performanceFunction=UML.calculate.fractionIncorrect)


def wrappedCrossValidate(learnerName, trainX, trainY, testX, testY):
    return UML.crossValidate(learnerName, trainX, trainY, performanceFunction=UML.calculate.fractionIncorrect)


def wrappedCrossValidateReturnBest(learnerName, trainX, trainY, testX, testY):
    return UML.crossValidateReturnBest(learnerName, trainX, trainY, performanceFunction=UML.calculate.fractionIncorrect)


def wrappedCrossValidateReturnAll(learnerName, trainX, trainY, testX, testY):
    return UML.crossValidateReturnAll(learnerName, trainX, trainY, performanceFunction=UML.calculate.fractionIncorrect)


def setupAndCallIncrementalTrain(learnerName, trainX, trainY, testX, testY):
    tl = UML.train(learnerName, trainX, trainY)
    tl.incrementalTrain(trainX.copyPoints([0]), trainY.copyPoints([0]))


def setupAndCallRetrain(learnerName, trainX, trainY, testX, testY):
    tl = UML.train(learnerName, trainX, trainY)
    tl.retrain(trainX, trainY)


def setupAndCallGetScores(learnerName, trainX, trainY, testX, testY):
    tl = UML.train(learnerName, trainX, trainY)
    tl.getScores(testX)


def backend(toCall, portionToTest, allowRegression=True):
    cData = generateClassificationData(2, 10, 5)
    ((cTrainX, cTrainY), (cTestX, cTestY)) = cData
    backCTrainX = cTrainX.copy()
    backCTrainY = cTrainY.copy()
    backCTestX = cTestX.copy()
    backCTestY = cTestY.copy()
    rData = generateRegressionData(2, 10, 5)
    ((rTrainX, rTrainY), (rTestX, rTestY)) = rData
    backRTrainX = rTrainX.copy()
    backRTrainY = rTrainY.copy()
    backRTestX = rTestX.copy()
    backRTestY = rTestY.copy()

    allLearners = UML.listLearners()
    numSamples = int(len(allLearners) * portionToTest)
    toTest = pythonRandom.sample(allLearners, numSamples)

    for learner in toTest:
        package = learner.split('.', 1)[0].lower()
        #		if package != 'mlpy' and package != 'scikitlearn':
        #			continue
        #		if package == 'shogun':
        #			print learner
        #		else:
        #			continue
        lType = UML.learnerType(learner)
        if lType == 'classification':
            try:
                toCall(learner, cTrainX, cTrainY, cTestX, cTestY)
            # this is meant to safely bypass those learners that have required arguments
            except ArgumentException as ae:
                pass
            #print ae
            # this is generally how shogun explodes
            except SystemError as se:
                pass
            #print se
            assertUnchanged(learner, cData, backCTrainX, backCTrainY, backCTestX, backCTestY)
        if lType == 'regression' and allowRegression:
            try:
                toCall(learner, rTrainX, rTrainY, rTestX, rTestY)
            # this is meant to safely bypass those learners that have required arguments
            except ArgumentException as ae:
                pass
            #print ae
            except SystemError as se:
                pass
            #print se
            assertUnchanged(learner, rData, backRTrainX, backRTrainY, backRTestX, backRTestY)


@attr('slow')
def testDataIntegrityTrain():
    backend(wrappedTrain, 1)


@attr('slow')
def testDataIntegrityTrainAndApply():
    backend(wrappedTrainAndApply, 1)

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
    backend(wrappedCrossValidateReturnAll, 1)
    backend(wrappedCrossValidateReturnBest, 1)

# test TrainedLearner methods
# only those that the top level trainers, appliers, and testers are not reliant on.
# Exclusions for above reason: apply(), test()
@attr('slow')
def testDataIntegrityTrainedLearner():
#	backend(setupAndCallIncrementalTrain, 1) TODO
    backend(setupAndCallRetrain, 1)
    backend(setupAndCallGetScores, 1, False)
