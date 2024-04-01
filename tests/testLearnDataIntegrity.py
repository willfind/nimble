
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
A group of tests which passes data to train, trainAndApply, and
trainAndTest, checking that after the call, the input data remains
unmodified. It makes use of learnerNames and learnerType to try this
operation with as many learners as possible.
"""
import pytest

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.calculate import fractionIncorrect
from nimble.random import pythonRandom
from tests.helpers import assertCalled
from tests.helpers import generateClassificationData
from tests.helpers import generateRegressionData


def assertUnchanged4Obj(learnerName, passed, trainX, trainY, testX, testY):
    """
    When labels are separate nimble objects, assert that all 4 objects
    passed down into a function are identical to copies made before the
    call.
    """
    ((pTrainX, pTrainY), (pTestX, pTestY)) = passed

    if not pTrainX.isIdentical(trainX):
        raise AssertionError(learnerName + " modified its trainX data")
    if not pTrainY.isIdentical(trainY):
        raise AssertionError(learnerName + " modified its trainY data")
    if not pTestX.isIdentical(testX):
        raise AssertionError(learnerName + " modified its testX data")
    if not pTestY.isIdentical(testY):
        raise AssertionError(learnerName + " modified its testY data")

def assertUnchanged2Obj(learnerName, passed, train, test):
    """
    When labels are included in the data object, assert that the 2
    objects passed down into a function are identical to copies made
    before the call.
    """
    (pTrain, pTest) = passed

    if not pTrain.isIdentical(train):
        raise AssertionError(learnerName + " modified its trainX data")
    if not pTest.isIdentical(test):
        raise AssertionError(learnerName + " modified its testX data")

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
    return nimble.trainAndTest(learnerName, fractionIncorrect, trainX, trainY,
                               testX, testY)


def wrappedTLTest(learnerName, trainX, trainY, testX, testY):
    # our performance function doesn't actually matter, we're just checking the data
    tl = nimble.train(learnerName, trainX, trainY)
    return tl.test(fractionIncorrect, testX, testY)


def wrappedTrainAndTestOvO(learnerName, trainX, trainY, testX, testY):
    return nimble.trainAndTest(learnerName, fractionIncorrect, trainX, trainY,
                               testX, testY, multiClassStrategy='OneVsOne')


def wrappedTrainAndTestOvA(learnerName, trainX, trainY, testX, testY):
    return nimble.trainAndTest(learnerName, fractionIncorrect, trainX, trainY,
                               testX, testY, multiClassStrategy='OneVsAll')


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

    allLearners = nimble.learnerNames()
    numSamples = int(len(allLearners) * portionToTest)
    toTest = pythonRandom.sample(allLearners, numSamples)

    for learner in toTest:
        package = learner.split('.', 1)[0].lower()
        lType = nimble.learnerType(learner)
        if lType == 'classification':
            try:
                toCall(learner, cTrainX, cTrainY, cTestX, cTestY)
                assertUnchanged4Obj(learner, cData, backCTrainX, backCTrainY,
                                    backCTestX, backCTestY)
                toCall(learner, cTrainCombined, 0, cTestCombined, 0)
                assertUnchanged2Obj(learner, (cTrainCombined, cTestCombined),
                                backCTrainCombined, backCTestCombined)
            except IndexError:
                # If a learner transforms our randomly generated data to
                # integers, it's possible a feature of all 0s can be created.
                # This is problematic if the learner requires multiple
                # categories, so we use round to avoid this possibility.
                def rounded(ft):
                    return [round(v) for v in ft]

                cTrainXInt = cTrainX.features.calculate(rounded)
                cTestXInt = cTestX.features.calculate(rounded)
                cDataInt = ((cTrainXInt, cTrainY), (cTestXInt, cTestY))
                backCTrainXInt = cTrainXInt.copy()
                backCTestXInt = cTestXInt.copy()
                toCall(learner, cTrainXInt, cTrainY, cTestXInt, cTestY)
                assertUnchanged4Obj(learner, cDataInt, backCTrainXInt,
                                    backCTrainY, backCTestXInt, backCTestY)
                cTrainComInt = cTrainCombined.features.calculate(rounded)
                cTestComInt = cTestCombined.features.calculate(rounded)
                backCTrainComInt = cTrainComInt.copy()
                backCTestComInt = cTestComInt.copy()
                toCall(learner, cTrainComInt, 0, cTestComInt, 0)
                assertUnchanged2Obj(learner, (cTrainComInt, cTestComInt),
                                    backCTrainComInt, backCTestComInt)
            # this is meant to safely bypass those learners that have required arguments
            except InvalidArgumentValue as iav:
                continue
            except NotImplementedError as nie:
                if not allowNotImplemented:
                    raise nie
                continue
        elif lType == 'regression' and allowRegression:
            try:
                toCall(learner, rTrainX, rTrainY, rTestX, rTestY)
                assertUnchanged4Obj(learner, rData, backRTrainX, backRTrainY,
                                    backRTestX, backRTestY)
                toCall(learner, rTrainCombined, 0, rTestCombined, 0)
                assertUnchanged2Obj(learner, (rTrainCombined, rTestCombined),
                                    backRTrainCombined, backRTestCombined)
            # this is meant to safely bypass those learners that have required arguments
            except InvalidArgumentValue as iav:
                continue
            except SystemError as se:
                continue
            except NotImplementedError as nie:
                if not allowNotImplemented:
                    raise nie
                continue


@pytest.mark.slow
def testDataIntegrityTrain():
    backend(wrappedTrain, 1)


@pytest.mark.slow
def testDataIntegrityTrainAndApply():
    backend(wrappedTrainAndApply, 1)


@pytest.mark.slow
def testDataIntegrityTLApply():
    backend(wrappedTLApply, 1)

# we can test smaller portions here because the backends are all being tested by
# the previous tests. We only care about the trainAndApply One vs One and One vs
# all code.
@pytest.mark.slow
def testDataIntegrityTrainAndApplyMulticlassStrategies():
    backend(wrappedTrainAndApplyOvO, .1, False)
    backend(wrappedTrainAndApplyOvA, .1, False)


@pytest.mark.slow
def testDataIntegrityTrainAndTest():
    backend(wrappedTrainAndTest, 1)


@pytest.mark.slow
def testDataIntegrityTLTest():
    backend(wrappedTLTest, 1)

# we can test smaller portions here because the backends are all being tested by
# the previous tests. We only care about the trainAndTest One vs One and One vs
# all code.
@pytest.mark.slow
def testDataIntegrityTrainAndTestMulticlassStrategies():
    backend(wrappedTrainAndTestOvO, .1, False)
    backend(wrappedTrainAndTestOvA, .1, False)

# test TrainedLearner methods
# only those that the top level trainers, appliers, and testers are not reliant on.
# Exclusions for above reason: apply(), test()
@pytest.mark.slow
def testDataIntegrityTrainedLearner():
#	backend(setupAndCallIncrementalTrain, 1) TODO
    backend(setupAndCallRetrain, 1)
    backend(setupAndCallGetScores, 1, False, True)


#######################
# Arguments Integrity #
#######################

# mergeArguments calls indicate that the user arguments parameter will be not
# be modified because it returns a new dictionary (verified in testHelpers)

@assertCalled(nimble.core.learn, 'mergeArguments')
def testArgumentIntegrityTrain():
    arguments = {'k': 1}
    train = nimble.data([[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    tl = nimble.train('nimble.KNNClassifier', train, 2, arguments=arguments)

@assertCalled(nimble.core.learn, 'mergeArguments')
def testArgumentIntegrityTrainAndApply():
    arguments = {'k': 1}
    train = nimble.data([[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    test = nimble.data([[0, 1], [1, 0]])
    pred = nimble.trainAndApply('nimble.KNNClassifier', train, 2, test, arguments=arguments)

@assertCalled(nimble.core.learn, 'mergeArguments')
def testArgumentIntegrityTrainAndTest():
    arguments = {'k': 1}
    train = nimble.data([[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    test = nimble.data([[0, 1, 1], [1, 0, 2]])
    perf = nimble.trainAndTest('nimble.KNNClassifier', fractionIncorrect,
                               train, 2, test, 2, arguments=arguments)

def testArgumentIntegrityTrainAndTestOnTrainingData():
    arguments = {'k': 1}
    train = nimble.data([[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    mergeArgumentsCalled = assertCalled(nimble.core.learn, 'mergeArguments')
    with mergeArgumentsCalled:
        perf = nimble.trainAndTestOnTrainingData(
            'nimble.KNNClassifier', fractionIncorrect, train, 2,
            arguments=arguments)

    with mergeArgumentsCalled:
        perf = nimble.trainAndTestOnTrainingData(
            'nimble.KNNClassifier', fractionIncorrect, train, 2, folds=2,
            arguments=arguments, crossValidationError=True)


@assertCalled(nimble.core.interfaces.universal_interface, 'mergeArguments')
def testArgumentIntegrityTLApply():
    arguments = {'k': 1}
    train = nimble.data([[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]],
                        pointNames=['0', '1', '2', '3'])
    test = nimble.data([[0, 1], [1, 0]], pointNames=['4', '5'])
    tl = nimble.train('nimble.KNNClassifier', train, 2, arguments=arguments)
    pred = tl.apply(test)

@assertCalled(nimble.core.interfaces.universal_interface, 'mergeArguments')
def testArgumentIntegrityTLTest():
    arguments = {'k': 1}
    train = nimble.data([[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 1, 3]])
    test = nimble.data([[0, 1, 1], [1, 0, 2]])
    tl = nimble.train('nimble.KNNClassifier', train, 2, arguments=arguments)
    perf = tl.test(fractionIncorrect, test, 2)
