"""
Integration tests to demonstrate consistency between output of different methods
of a single interface. All tests are general, testing knowledge guaranteed by
the UniversalInterface api.

"""

from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import importlib
import copy
from unittest import mock
import tempfile

import nose
from nose.tools import raises
from nose.plugins.attrib import attr
#@attr('slow')

import UML
from UML.configuration import configSafetyWrapper
from UML.exceptions import InvalidArgumentValue, PackageException
from UML.interfaces.universal_interface import UniversalInterface
from UML.helpers import generateClusteredPoints
from UML.helpers import generateClassificationData
from UML.helpers import generateRegressionData


def checkFormat(scores, numLabels):
    """
    Check that the provided UML data typed scores structurally match either a one vs
    one or a one vs all formatting scheme.

    """
    if len(scores.features) != numLabels and len(scores.features) != (numLabels * (numLabels - 1)) / 2:
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
                except InvalidArgumentValue:
                    # this is to catch learners that have required arguments.
                    # we have to skip them in that case
                    continue
                (ign1, ign2, transTestX2, ign3) = interface._inputTransformation(lName, None, None, testX2, {},
                                                                                 tl2.customDict)
                try:
                    scores2 = interface._getScores(tl2.backend, transTestX2, {}, tl2.customDict)
                except InvalidArgumentValue:
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
                except InvalidArgumentValue:
                    # this is to catch learners that have required arguments.
                    # we have to skip them in that case
                    continue
                try:
                    scores2 = tl2.getScores(testX2)
                except InvalidArgumentValue:
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

            except Exception as e:
                print(interfaceName + '.' + learner + ' BANG: ' + str(e))
                continue

            if result1 is not None:
                assert result1 == result2
                if result1 != result3:
                    assert result1 != result4
                    assert result3 != result4


def getCanonicalNameAndPossibleAliases(interface):
    if interface.__name__ == 'SciKitLearn':
        canonicalName = 'sciKitLearn'
        aliases = ['scikitlearn', 'skl', 'sklearn', 'sKLeARN', 'SKL']
    elif interface.__name__ == 'Mlpy':
        canonicalName = 'mlpy'
        aliases = ['mlpy', 'MLPY', 'mLpY']
    elif interface.__name__ == 'Keras':
        canonicalName = 'keras'
        aliases = ['keras', 'KERAS', 'KeRaS']
    elif interface.__name__ == 'Shogun':
        canonicalName = 'shogun'
        aliases = ['shogun', 'SHOGUN', 'ShOGUn']
    else:
        msg = "the canonical name and aliases are not defined for this interface"
        raise ValueError(msg)
    return canonicalName, aliases

def test_classmethods():
    for interface in UML.interfaces.builtin:
        canonicalName, aliases = getCanonicalNameAndPossibleAliases(interface)
        assert interface.getCanonicalName() == canonicalName
        for alias in aliases:
            assert interface.isAlias(alias)
        assert not interface.isAlias('foo')


def test_failedInit():
    availableBackup = UML.interfaces.available
    builtinBackup = UML.interfaces.builtin
    try:
        for interface in UML.interfaces.builtin:
            class MockInterface(interface):
                pass

            # override interfaces.available so findBestInterface will check builtin
            # and use mock object as the builtin so we can raise different errors
            UML.interfaces.available = []
            UML.interfaces.builtin = [MockInterface]

            # the learner must start with package and the data must be UML objects but
            # the learner name and data values are irrelevant since a PackageException
            # should occur
            learner = interface.__name__ + '.Foo'
            X = UML.createData('Matrix', [])
            y = X.copy()
            def raiseError(error):
                raise error
            try:
                MockInterface.__init__ = lambda self: raiseError(ImportError)
                UML.train(learner, X, y)
                assert False # expected PackageException
            except PackageException as e:
                assert "ImportError" in str(e)
                # path message included for ImportErrors
                assert "If package installed" in str(e)
                # interface may have install instructions
                if interface._installInstructions():
                    assert "To install" in str(e)
                assert "Exception information" in str(e)

            try:
                MockInterface.__init__ = lambda self: raiseError(ValueError)
                UML.train(learner, X, y)
                assert False # expected PackageException
            except PackageException as e:
                assert "ValueError" in str(e)
                # path message not included for other error types
                assert not "If package installed" in str(e)
                assert "Exception information" in str(e)
    finally:
        UML.interfaces.available = availableBackup
        UML.interfaces.builtin = builtinBackup


## Helpers for test_loadModulesFromConfigLocation ##
class ImportedMockModule(Exception):
    pass

mockedInit = """
from tests.interfaces.universal_integration_test import ImportedMockModule
raise ImportedMockModule
"""

importlib.import_module_ = importlib.import_module
# need to reload module, since import_module will may return the
# module loaded on init
def reload(module):
    """
    One of the two importlib calls will access the modified sys.path
    and throw an ImportedMockModuleException
    """
    # if the module did not load successfully on init, this will
    # access the new sys.path and should raise exception here
    mod = importlib.import_module_(module)
    # if the module loaded successfully on init, need to reload so
    # it checks the new sys.path and should raise exception here
    mod = importlib.reload(mod)
    return mod

@configSafetyWrapper
def test_loadModulesFromConfigLocation():
    for interface in UML.interfaces.builtin:
        sysPathBackup = sys.path.copy()
        canonicalName, _ = getCanonicalNameAndPossibleAliases(interface)
        packageName = canonicalName
        if packageName == 'sciKitLearn':
            packageName = 'sklearn'
        with tempfile.TemporaryDirectory() as mockDirectory:
            packagePath = os.path.join(mockDirectory, packageName)
            os.mkdir(packagePath)
            initPath = os.path.join(packagePath, '__init__.py')
            with open(initPath, 'w+') as initFile:
                initFile.write(mockedInit)

            UML.settings.set(canonicalName, 'location', mockDirectory)
            try:
                # patch import_module so it reloads from settings location
                with mock.patch('importlib.import_module', reload):
                    interface()
                assert False # expected ImportedMockModule
            except ImportedMockModule:
                pass
            finally:
                sys.path = sysPathBackup


# TODO
#def testGetParamsOverListLearners():
#def testGetParamDefaultsOverListLearners():


# comparison between UML.learnerType and interface learnerType
