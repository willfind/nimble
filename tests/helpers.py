"""
Common assertions helpers to be used in multiple test files.

Custom assertion types can be helpful if the assertion can be added to
existing tests which are also testing other functionality.
"""
import tempfile
from functools import wraps
import os
import copy

import nimble
from nimble.core.data import BaseView
from nimble.core._learnHelpers import generateClusteredPoints
from nimble.core.configuration import SessionConfiguration

def configSafetyWrapper(toWrap):
    """
    Decorator which ensures the safety of nimble.settings, the
    configuration file, and associated global nimble state. To be used
    to wrap unit tests which intersect with configuration functionality.
    """
    @wraps(toWrap)
    def wrapped(*args, **kwargs):
        configCopy = tempfile.NamedTemporaryFile('w+', delete=False)
        configFilePath = os.path.join(nimble.nimblePath, 'configuration.ini')
        configurationFile = open(configFilePath, 'r')
        configCopy.write(configurationFile.read())
        configurationFile.close()
        configCopy.seek(0)

        def loadSettingsFromTempFile():
            return SessionConfiguration(configCopy.name)

        copyChanges = copy.copy(nimble.settings.changes)
        copyHooks = copy.copy(nimble.settings.hooks)
        backupSettings = nimble.settings
        backupLoadSettings = nimble.core.configuration.loadSettings
        backupLogger = nimble.core.logger.active
        availableBackup = copy.copy(nimble.core.interfaces.available)
        # CustomLearnerInterface attributes are not copied above
        clInterface = nimble.core.interfaces.available['custom']
        clReg = copy.copy(clInterface.registeredLearners)
        clOpt = copy.copy(clInterface._configurableOptionNamesAvailable)

        nimble.settings = loadSettingsFromTempFile()
        nimble.settings.changes = copyChanges
        nimble.settings.hooks = copyHooks
        nimble.core.configuration.loadSettings = loadSettingsFromTempFile

        try:
            toWrap(*args, **kwargs)
        finally:
            nimble.settings = backupSettings
            nimble.core.logger.active = backupLogger
            nimble.core.configuration.loadSettings = backupLoadSettings
            nimble.core.interfaces.available = availableBackup
            clInterface.registeredLearners = clReg
            clInterface._configurableOptionNamesAvailable = clOpt

    return wrapped

class LogCountAssertionError(AssertionError):
    pass

def logCountAssertionFactory(count):
    """
    Generate a wrapper to assert the log increased by a certain count.
    """
    def logCountAssertion(function):
        @wraps(function)
        @configSafetyWrapper
        def wrapped(*args, **kwargs):
            nimble.settings.set('logger', 'enabledByDefault', 'True')
            nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'True')
            logger = nimble.core.logger.active
            countQuery = "SELECT COUNT(entry) FROM logger"
            startCount = logger.extractFromLog(countQuery)[0][0]
            ret = function(*args, **kwargs)
            endCount = logger.extractFromLog(countQuery)[0][0]
            if startCount + count != endCount:
                nimble.showLog(mostSessionsAgo=1)
                msg = "Expected an additional {0} logs, but got {1}"
                msg = msg.format(count, endCount - startCount)
                raise LogCountAssertionError(msg)
            return ret

        return wrapped
    return logCountAssertion

noLogEntryExpected = logCountAssertionFactory(0)
oneLogEntryExpected = logCountAssertionFactory(1)


class LazyNameGenerationAssertionError(AssertionError):
    pass

def assertNoNamesGenerated(obj):
    if obj.points._namesCreated() or obj.features._namesCreated():
        raise LazyNameGenerationAssertionError


class CalledFunctionException(Exception):
    pass

def calledException(*args, **kwargs):
    raise CalledFunctionException()


# TODO: polish and relocate to random module
def generateClassificationData(labels, pointsPer, featuresPer):
    """
    Randomly generate sensible data for a classification problem.
    Returns a tuple of tuples, where the first value is a tuple
    containing (trainX, trainY) and the second value is a tuple
    containing (testX ,testY).
    """
    #add noise to the features only
    trainData, _, noiselessTrainLabels = generateClusteredPoints(
        labels, pointsPer, featuresPer, addFeatureNoise=True,
        addLabelNoise=False, addLabelColumn=False)
    testData, _, noiselessTestLabels = generateClusteredPoints(
        labels, 1, featuresPer, addFeatureNoise=True, addLabelNoise=False,
        addLabelColumn=False)

    return ((trainData, noiselessTrainLabels), (testData, noiselessTestLabels))


# TODO: polish and relocate to random module
def generateRegressionData(labels, pointsPer, featuresPer):
    """
    Randomly generate sensible data for a regression problem. Returns a
    tuple of tuples, where the first value is a tuple containing
    (trainX, trainY) and the second value is a tuple containing
    (testX ,testY).
    """
    #add noise to both the features and the labels
    regressorTrainData, trainLabels, _ = generateClusteredPoints(
        labels, pointsPer, featuresPer, addFeatureNoise=True,
        addLabelNoise=True, addLabelColumn=False)
    regressorTestData, testLabels, _ = generateClusteredPoints(
        labels, 1, featuresPer, addFeatureNoise=True, addLabelNoise=True,
        addLabelColumn=False)

    return ((regressorTrainData, trainLabels), (regressorTestData, testLabels))
