"""
Integration tests to demonstrate consistency between output of different methods
of a single interface. All tests are general, testing knowledge guaranteed by
the UniversalInterface api.
"""

import os
import sys
import importlib
import tempfile

import pytest

import nimble
from nimble.exceptions import InvalidArgumentValue, ImproperObjectAction
from nimble.exceptions import PackageException
from tests.helpers import generateClassificationData
from tests.helpers import generateRegressionData
from tests.helpers import patch, raises


def checkFormat(scores, numLabels):
    """
    Check that the provided nimble data typed scores structurally match either a one vs
    one or a one vs all formatting scheme.
    """
    assert (len(scores.features) == numLabels
            or len(scores.features) == (numLabels * (numLabels - 1)) / 2)


def checkFormatRaw(scores, numLabels):
    """
    Check that the provided numpy typed scores structurally match either a one vs
    one or a one vs all formatting scheme.
    """
    assert (scores.shape[1] == numLabels
            or scores.shape[1] == (numLabels * (numLabels - 1)) / 2)

# # TODO outliers as 'classification' see test__getScoresFormat and testGetScoresFormat
# OUTLIER_LEARNERS = ['EllipticEnvelope', 'IsolationForest', 'OneClassSVM',
#                     'SGDOneClassSVM']
#
# def checkOutlierFormat(scores, testX):
#     """
#     Outliers give a single score per point
#     """
#     assert scores.shape[0] == testX.shape[0]
#     assert scores.shape[1] == 1

@pytest.mark.slow
def test__getScoresFormat():
    """
    Automatically checks the _getScores() format for as many classifiers
    we can identify in each interface.
    """
    for i in [2, 4]:
        data = generateClassificationData(i, 4, 3)
        ((trainX, trainY), (testX, _)) = data
        for interface in nimble.core.interfaces.available.values():
            interfaceName = interface.getCanonicalName()

            learners = interface.learnerNames()
            for lName in learners:
                fullName = interfaceName + '.' + lName
                if nimble.learnerType(fullName) == 'classification':
                    try:
                        tl = nimble.train(fullName, trainX, trainY)
                    except (InvalidArgumentValue, SystemError):
                        # skip learners that have required arguments
                        continue
                    except ValueError as VE:
                        # this will catch strictly binary classification
                        # learners, and skip them when i == 4
                        if i == 4:
                            continue
                        raise VE
                    (transTrainX, _, transTestX, _) = interface._inputTransformation(
                        lName, trainX, None, testX, 42, {}, tl._customDict)
                    try:
                        scores = interface._getScores(
                            lName, tl._backend, transTestX, {},
                            tl._transformedArguments, tl._customDict)
                    except IndexError:
                        # categorical data needed; use train data
                        scores = interface._getScores(
                            lName, tl._backend, transTrainX, {},
                            tl._transformedArguments, tl._customDict)
                    except (NotImplementedError, SystemError):
                        # this is to catch learners that cannot output scores
                        continue
                    checkFormatRaw(scores, i)
                    # # TODO handle outlier classifiers correctly
                    # if lName in OUTLIER_LEARNERS:
                    #     checkOutlierFormat(scores, testX)
                    # else:
                    #     checkFormatRaw(scores, i)

@pytest.mark.slow
def testGetScoresFormat():
    """
    Automatically checks the TrainedLearner getScores() format for as many classifiers we
    can identify in each interface

    """
    for i in [2, 4]:
        data = generateClassificationData(i, 4, 2)
        ((trainX, trainY), (testX, _)) = data
        for interface in nimble.core.interfaces.available.values():
            interfaceName = interface.getCanonicalName()
            learners = interface.learnerNames()
            for lName in learners:
                fullName = interfaceName + '.' + lName
                if nimble.learnerType(fullName) == 'classification':
                    try:
                        tl = nimble.train(fullName, trainX, trainY)
                    except (InvalidArgumentValue, SystemError):
                        # skip learners that have required arguments
                        continue
                    except ValueError as VE:
                        # this will catch strictly binary classification
                        # learners, and skip them when i == 4
                        if i == 4:
                            continue
                        raise VE

                    try:
                        scores = tl.getScores(testX)
                    except IndexError:
                        # categorical data needed; use train data
                        scores = tl.getScores(trainX)
                    except (NotImplementedError, SystemError):
                        # this is to catch learners that cannot output scores
                        continue
                    checkFormatRaw(scores, i)
                    # # TODO handle outlier classifiers correctly
                    # if lName in OUTLIER_LEARNERS:
                    #     checkOutlierFormat(scores, testX)
                    # else:
                    #     checkFormatRaw(scores, i)


@pytest.mark.slow
def testApplyFeatureNames():
    """ Check train label feature name is set during apply for regression and classification """
    regressionData = generateRegressionData(5, 10, 5)
    classificationData = generateClassificationData(2, 10, 5)
    success = 0
    for interface in nimble.core.interfaces.available.values():
        interfaceName = interface.getCanonicalName()
        for learner in interface.learnerNames():
            currType = interface.learnerType(learner)
            if currType == 'regression':
                data = regressionData
            elif currType == 'classification':
                data = classificationData
            else:
                continue
            ((trainData, trainLabels), (testData, testLabels)) = data
            trainLabels.features.setNames(['label'])
            trainData.features.transform(lambda ft: abs(ft))
            testData.features.transform(lambda ft: abs(ft))
            for mode in [None, 'allScores', 'bestScore']:
                strResult = None
                try:
                    result = nimble.trainAndApply(interfaceName + '.' + learner,
                                                  trainData, trainLabels, testData,
                                                  scoreMode=mode)
                    if currType == 'classification':
                        toString = {0: 'a', 1: 'b'}
                        strLabels = data[0][1].calculateOnElements(lambda e: toString[e])
                        strResult = nimble.trainAndApply(
                            interfaceName + '.' + learner, trainData, strLabels,
                            testData, scoreMode=mode)
                except InvalidArgumentValue:
                    # try multioutput learner; only label mode is allowed
                    if mode is not None:
                        continue
                    multiLabels = trainLabels.copy()
                    labels2 = trainLabels.copy()
                    labels2.features.setNames(['label2'])
                    multiLabels.features.append(labels2)
                    try:
                        result = nimble.trainAndApply(interfaceName + '.' + learner,
                                                      trainData, multiLabels, testData,
                                                      scoreMode=mode)
                    except (InvalidArgumentValue, ImproperObjectAction):
                        continue # incompatible data for this operation
                except Exception:
                    continue
                if mode is None:
                    assert result.features.getName(0) == 'label'
                elif mode == 'bestScore':
                    assert result.features.getNames() == ['label', 'bestScore']
                else:
                    assert result.features.getNames() == ['0.0', '1.0']
                    if strResult is not None:
                        assert strResult.features.getNames() == ['a', 'b']
                success += 1
    # ensure not passing because except Exception is catching all cases
    assert success

@pytest.mark.slow
def testRandomnessControl():
    """ Test that nimble takes over the control of randomness of each interface """

    #	assert 'RanomizedLogisticRegression' in nimble.learnerNames('sciKitLearn')

    for interface in nimble.core.interfaces.available.values():
        interfaceName = interface.getCanonicalName()
        listOf = nimble.learnerNames(interfaceName)

        for learner in listOf:
            currType = nimble.learnerType(interfaceName + '.' + learner)
            if currType == 'regression':
                ((trainData, trainLabels), (testData, _)) = generateRegressionData(5, 10, 9)
            elif currType == 'classification':
                ((trainData, trainLabels), (testData, _)) = generateClassificationData(2, 10, 9)
            else:
                continue

            fullName = interfaceName + '.' + learner
            try:
                # using global random seed
                nimble.random.setSeed(50)
                result1 = nimble.trainAndApply(fullName, trainData, trainLabels, testData)

                nimble.random.setSeed(50)
                result2 = nimble.trainAndApply(fullName, trainData, trainLabels, testData)

                nimble.random.setSeed(None)
                result3 = nimble.trainAndApply(fullName, trainData, trainLabels, testData)

                nimble.random.setSeed(13)
                result4 = nimble.trainAndApply(fullName, trainData, trainLabels, testData)

                # using parameter level seed
                nimble.random.setSeed(12)
                result5 = nimble.trainAndApply(fullName, trainData, trainLabels,
                                               testData, randomSeed=50)

                nimble.random.setSeed(1)
                result6 = nimble.trainAndApply(fullName, trainData, trainLabels,
                                               testData, randomSeed=50)

                nimble.random.setSeed(12)
                result7 = nimble.trainAndApply(fullName, trainData, trainLabels,
                                               testData, randomSeed=None)

                nimble.random.setSeed(12)
                result8 = nimble.trainAndApply(fullName, trainData, trainLabels,
                                               testData, randomSeed=15)

            except Exception as e:
                print(interfaceName + '.' + learner + ' BANG: ' + str(e))
                continue

            assert result1 == result2
            # different random state still generates same results occassionally
            # so check against two different objects.
            if result1 != result3 or result1 != result4:
                # if appears that randomness is involved, check that we can
                # replicate result4.
                nimble.random.setSeed(13)
                assert result4 == nimble.trainAndApply(fullName, trainData,
                                                       trainLabels, testData)
            assert result5 == result6 # different global, same param
            if result5 != result7 or result5 != result8:
                # if appears that randomness is involved, check that we can
                # replicate result8.
                nimble.random.setSeed(13)
                assert result8 == nimble.trainAndApply(fullName, trainData, trainLabels,
                                                       testData, randomSeed=15)

def getCanonicalNameAndPossibleAliases(interface):
    if interface.__name__ == 'SciKitLearn':
        canonicalName = 'sklearn'
        aliases = ['scikitlearn', 'skl', 'SciKitLearn', 'sKLeARN', 'SKL']
    elif interface.__name__ == 'Keras':
        canonicalName = 'keras'
        aliases = ['keras', 'KERAS', 'KeRaS', 'tensorflow.keras', 'tf.keras']
    elif interface.__name__ == 'Autoimpute':
        canonicalName = 'autoimpute'
        aliases = ['autoimpute', 'AUTOIMPUTE', 'AuToImpUTe']
    else:
        msg = "the canonical name and aliases are not defined for this interface"
        raise ValueError(msg)
    return canonicalName, aliases

def test_classmethods():
    for interface in nimble.core.interfaces.predefined:
        canonicalName, aliases = getCanonicalNameAndPossibleAliases(interface)
        assert interface.getCanonicalName() == canonicalName
        for alias in aliases:
            assert interface.isAlias(alias)
        assert not interface.isAlias('foo')


def test_failedInit():
    availableBackup = nimble.core.interfaces.available
    predefinedBackup = nimble.core.interfaces.predefined
    try:
        for interface in nimble.core.interfaces.predefined:
            class MockInterface(interface):
                pass

            # override interfaces.available so findBestInterface will check
            # predefined and use mock object as the predefined so we can raise
            # different errors
            nimble.core.interfaces.available = {}
            nimble.core.interfaces.predefined = [MockInterface]

            # the learner must start with package and the data must be nimble
            # objects but the learner name and data values are irrelevant
            # since a PackageException should occur
            learner = interface.__name__ + '.Foo'
            X = nimble.data([])
            y = X.copy()
            def raiseError(error):
                raise error
            try:
                MockInterface.__init__ = lambda self: raiseError(ImportError)
                nimble.train(learner, X, y)
                assert False # expected PackageException
            except PackageException as e:
                # install and path message included for ImportError
                assert "To install" in str(e)
                assert "If {0} installed".format(interface.getCanonicalName()) in str(e)

            try:
                MockInterface.__init__ = lambda self: raiseError(ValueError)
                nimble.train(learner, X, y)
                assert False # expected PackageException
            except PackageException as e:
                # install and path message not included for other error types
                assert not "To install" in str(e)
                assert not "If {0} installed".format(interface.getCanonicalName()) in str(e)
    finally:
        nimble.core.interfaces.available = availableBackup
        nimble.core.interfaces.predefined = predefinedBackup


## Helpers for test_loadModulesFromConfigLocation ##
class FirstImportedModule(Exception):
    pass

class SecondImportedModule(Exception):
    pass

mockedInit1 = """
from tests.interfaces.universal_integration_test import FirstImportedModule
raise FirstImportedModule
"""

mockedInit2 = """
from tests.interfaces.universal_integration_test import SecondImportedModule
raise SecondImportedModule
"""

importlib.import_module_ = importlib.import_module
# need to reload module, since import_module will may return the
# module loaded on init
def reload(module):
    """
    Reload the module so that the returned module is the one from the
    first available location on sys.path.
    """
    # reloading a module like tensorflow.keras requires reloading
    # tensorflow first, then reloading tensorflow.keras
    split = module.split('.')
    modules = ['.'.join(split[:i + 1]) for i in range(len(split))]
    for module in modules:
        # if the module did not load successfully on init, this will
        # access the new sys.path and should raise exception here
        mod = importlib.import_module_(module)
        # if the module loaded successfully on init, need to reload so
        # it checks the new sys.path and should raise exception here
        try:
            mod = importlib.reload(mod)
        except ImportError:
            # the reload failed because our previous reloads of higher
            # level packages affected this submodule loading. So instead of
            # reloading we must delete the module and import again.
            modulesBackup = sys.modules
            try:
                del mod
                del sys.modules[module]
                mod = importlib.import_module_(module)
            finally:
                sys.modules = modulesBackup

    return mod

def makeMockedModule(mockDirectory, packageName, mockedMsg):
    packagePath = mockDirectory
    packageNames = packageName.split('.')
    for i, packageName in enumerate(packageNames):
        packagePath = os.path.join(packagePath, packageName)
        os.mkdir(packagePath)
        if i == len(packageNames) - 1:
            toWrite = mockedMsg # raise expected exception
        else:
            toWrite = '' # make an importable python package
        initPath = os.path.join(packagePath, '__init__.py')
        with open(initPath, 'w+') as initFile:
            initFile.write(toWrite)

def test_loadModulesFromConfigLocation():
    for interface in nimble.core.interfaces.predefined:
        sysPathBackup = sys.path.copy()
        try:
            canonicalName, _ = getCanonicalNameAndPossibleAliases(interface)
            packageName1 = packageName2 = canonicalName

            with tempfile.TemporaryDirectory() as mockDirectory1:
                # first directory containing the same package
                makeMockedModule(mockDirectory1, packageName1, mockedInit1)
                with tempfile.TemporaryDirectory() as mockDirectory2:
                    # second directory containing the same package
                    makeMockedModule(mockDirectory2, packageName2, mockedInit2)
                    # manually insert first directory to sys.path
                    # first, check that it loads from the path of the first
                    # directory which raises FirstImportedModule
                    sys.path.insert(0, mockDirectory1)
                    with raises(FirstImportedModule):
                        # patch import_module to reload from settings location
                        with patch(importlib, 'import_module', reload):
                            interface()
                    # next, set a location in settings to check that this
                    # takes priority over mockDirectory1. Also check that
                    # the change to the path is temporary
                    nimble.settings.set(canonicalName, 'location',
                                        mockDirectory2)
                    with raises(SecondImportedModule):
                        # patch import_module to reload from settings location
                        with patch(importlib, 'import_module', reload):
                            interface()
                    # check that the modifications made to the path during
                    # __init__ are no longer present
                    assert not mockDirectory2 in sys.path
        finally:
            sys.path = sysPathBackup

def test_getParametersAndDefaultsReturnTypes():
    for interface in nimble.core.interfaces.available.values():
        interfaceName = interface.getCanonicalName()
        for learner in nimble.learnerNames(interfaceName):
            params = interface._getParameterNames(learner)
            defaults = interface._getDefaultValues(learner)
            learnerParams = interface.getLearnerParameterNames(learner)
            defaultParams = interface.getLearnerDefaultValues(learner)
            fullName = interfaceName + '.' + learner
            paramsFromTop = nimble.learnerParameters(fullName)
            defaultsFromTop = nimble.learnerParameterDefaults(fullName)

            if params is not None:
                assert isinstance(params, list)
                assert all(isinstance(option, list) for option in params)
                assert isinstance(defaults, list)
                assert all(isinstance(option, dict) for option in defaults)

            assert isinstance(learnerParams, list)
            assert all(isinstance(option, list) for option in learnerParams)
            assert isinstance(defaultParams, list)
            assert all(isinstance(option, dict) for option in defaultParams)

            # top-level will not be nested in list if only one item in list
            assert isinstance(paramsFromTop, list)
            if len(learnerParams) > 1:
                assert all(isinstance(option, list) for option in paramsFromTop)

            if isinstance(defaultsFromTop, list):
                assert all(isinstance(option, dict) for option in defaultsFromTop)
            else:
                assert isinstance(defaultsFromTop, dict)

def test_getParametersAndDefaultsException():
    for interface in nimble.core.interfaces.available.keys():
        fullName = interface + '.wrong'
        expMsg = 'is not the name of a learner exposed by this interface'
        with raises(InvalidArgumentValue, match=expMsg):
            nimble.learnerParameters(fullName)
        with raises(InvalidArgumentValue, match=expMsg):
            nimble.learnerParameterDefaults(fullName)
# TODO
#def testGetParamsOverListLearners():
#def testGetParamDefaultsOverListLearners():


# comparison between nimble.learnerType and interface learnerType
