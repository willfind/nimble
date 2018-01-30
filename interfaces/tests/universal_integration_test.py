"""
Integration tests to demonstrate consistency between output of different methods
of a single interface. All tests are general, testing knowledge guaranteed by
the UniversalInterface api.

"""

from __future__ import absolute_import
from __future__ import print_function
import nose
from nose.tools import raises
from nose.plugins.attrib import attr
#@attr('slow')

import UML

from UML.exceptions import ArgumentException
from UML.interfaces.universal_interface import UniversalInterface
from UML.helpers import generateClusteredPoints
from UML.helpers import generateClassificationData
from UML.helpers import generateRegressionData


def checkFormat(scores, numLabels):
    """
    Check that the provided UML data typed scores structurally match either a one vs
    one or a one vs all formatting scheme.

    """
    if scores.features != numLabels and scores.features != (numLabels * (numLabels - 1)) / 2:
        raise RuntimeError("_getScores() must return scores that are either One vs One or One vs All formatted")


def checkFormatRaw(scores, numLabels):
    """
    Check that the provided numpy typed scores structurally match either a one vs
    one or a one vs all formatting scheme.

    """
    if scores.shape[1] != numLabels and scores.shape[1] != (numLabels * (numLabels - 1)) / 2:
        raise RuntimeError("_getScores() must return scores that are either One vs One or One vs All formatted")


@attr('slow')
def test__getScoresFormat():
    """
    Automatically checks the _getScores() format for as many classifiers we can identify in each
    interface.
    """
    data2 = generateClassificationData(2, 4, 2)
    ((trainX2, trainY2), (testX2, testY2)) = data2
    data4 = generateClassificationData(4, 4, 2)
    ((trainX4, trainY4), (testX4, testY4)) = data4
    for interface in UML.interfaces.available:
        interfaceName = interface.getCanonicalName()

        learners = interface.listLearners()
        for lName in learners:
            fullName = interfaceName + '.' + lName
            if UML.learnerType(fullName) == 'classifier':
                try:
                    tl2 = UML.train(fullName, trainX2, trainY2)
                except ArgumentException:
                    # this is to catch learners that have required arguments.
                    # we have to skip them in that case
                    continue
                (ign1, ign2, transTestX2, ign3) = interface._inputTransformation(lName, None, None, testX2, {},
                                                                                 tl2.customDict)
                try:
                    scores2 = interface._getScores(tl2.backend, transTestX2, {}, tl2.customDict)
                except ArgumentException:
                    # this is to catch learners that cannot output scores
                    continue
                checkFormatRaw(scores2, 2)

                try:
                    tl4 = UML.train(fullName, trainX4, trainY4)
                except:
                    # some classifiers are binary only
                    continue
                (ign1, ign2, transTestX4, ign3) = interface._inputTransformation(lName, None, None, testX4, {},
                                                                                 tl4.customDict)
                scores4 = interface._getScores(tl4.backend, transTestX4, {}, tl4.customDict)
                checkFormatRaw(scores4, 4)


@attr('slow')
def testGetScoresFormat():
    """
    Automatically checks the TrainedLearner getScores() format for as many classifiers we
    can identify in each interface

    """
    data2 = generateClassificationData(2, 4, 2)
    ((trainX2, trainY2), (testX2, testY2)) = data2
    data4 = generateClassificationData(4, 4, 2)
    ((trainX4, trainY4), (testX4, testY4)) = data4
    for interface in UML.interfaces.available:
        interfaceName = interface.getCanonicalName()

        learners = interface.listLearners()
        for lName in learners:
            if interfaceName == 'shogun':
                print(lName)

            fullName = interfaceName + '.' + lName
            if UML.learnerType(fullName) == 'classifier':
                try:
                    tl2 = UML.train(fullName, trainX2, trainY2)
                except ArgumentException:
                    # this is to catch learners that have required arguments.
                    # we have to skip them in that case
                    continue
                try:
                    scores2 = tl2.getScores(testX2)
                except ArgumentException:
                    # this is to catch learners that cannot output scores
                    continue
                checkFormat(scores2, 2)

                try:
                    tl4 = UML.train(fullName, trainX4, trainY4)
                except:
                    # some classifiers are binary only
                    continue
                scores4 = tl4.getScores(testX4)
                checkFormat(scores4, 4)


@attr('slow')
@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def testRandomnessControl():
    """ Test that UML takes over the control of randomness of each interface """

    #	assert 'RanomizedLogisticRegression' in UML.listLearners('sciKitLearn')

    for interface in UML.interfaces.available:
        interfaceName = interface.getCanonicalName()
        #		if interfaceName != 'shogun':
        #			continue

        listOf = UML.listLearners(interfaceName)

        for learner in listOf:
            if interfaceName == 'shogun':
                print(learner)
            currType = UML.learnerType(interfaceName + '.' + learner)
            if currType == 'regression':
                ((trainData, trainLabels), (testData, testLabels)) = generateRegressionData(5, 10, 5)
            elif currType == 'classification':
                ((trainData, trainLabels), (testData, testLabels)) = generateClassificationData(2, 10, 5)
            else:
                continue

            result1 = None
            try:
                UML.setRandomSeed(50)
                result1 = UML.trainAndApply(interfaceName + '.' + learner, trainData, trainLabels, testData)

                UML.setRandomSeed(50)
                result2 = UML.trainAndApply(interfaceName + '.' + learner, trainData, trainLabels, testData)

                UML.setRandomSeed(None)
                result3 = UML.trainAndApply(interfaceName + '.' + learner, trainData, trainLabels, testData)

                UML.setRandomSeed(13)
                result4 = UML.trainAndApply(interfaceName + '.' + learner, trainData, trainLabels, testData)

            #				print interfaceName + '.' + learner
            #				if interfaceName == 'sciKitLearn':
            #					args = UML.learnerParameters(interfaceName + '.' + learner)
            #					if 'random_state' in args[0]:
            #						print "   ^^^^"
            except Exception as e:
                print(interfaceName + '.' + learner + ' BANG: ' + str(e))
                continue

            if result1 is not None:
                assert result1 == result2
                if result1 != result3:
                    assert result1 != result4
                    assert result3 != result4


#	assert False



# TODO
#def testGetParamsOverListLearners():
#def testGetParamDefaultsOverListLearners():


# comparison between UML.learnerType and interface learnerType
