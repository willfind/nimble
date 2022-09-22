"""
Common assertions helpers to be used in multiple test files.

Custom assertion exc_types can be helpful if the assertion can be added to
existing tests which are also testing other functionality.
"""
from functools import wraps, partial

import pytest

import nimble
from nimble.exceptions import PackageException
from nimble.core._learnHelpers import generateClusteredPoints

class LogCountAssertionError(AssertionError):
    pass

def logCountAssertionFactory(count):
    """
    Generate a wrapper to assert the log increased by a certain count.
    """
    def logCountAssertion(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            nimble.settings.set('logger', 'enabledByDefault', 'True')
            nimble.settings.set('logger', 'enableDeepLogging', 'True')
            logger = nimble.core.logger.active
            countQuery = "SELECT COUNT(entry) FROM logger"
            startCount = logger.extractFromLog(countQuery)[0][0]
            ret = function(*args, **kwargs)
            endCount = logger.extractFromLog(countQuery)[0][0]
            if startCount + count != endCount:
                nimble.showLog(mostSessionsAgo=1, levelOfDetail=3)
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
def generateClassificationData(labels, pointsPer, featuresPer, testSizePerLabel=1):
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
        labels, testSizePerLabel, featuresPer, addFeatureNoise=True, addLabelNoise=False,
        addLabelColumn=False)

    return ((trainData, noiselessTrainLabels), (testData, noiselessTestLabels))


# TODO: polish and relocate to random module
def generateRegressionData(labels, pointsPer, featuresPer, testSizePerLabel=1):
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
        labels, testSizePerLabel, featuresPer, addFeatureNoise=True, addLabelNoise=True,
        addLabelColumn=False)

    return ((regressorTrainData, trainLabels), (regressorTestData, testLabels))

def _getViewFunc(returnType):
    """
    Return function creating a view of the given returnType.
    Helper for dataConstructors.
    """
    def getView(*args, **kwargs):
        obj = nimble.data(*args, returnType=returnType, **kwargs)
        return obj.view()
    # mirror attributes of functools.partial
    getView.func = nimble.data
    getView.args = tuple()
    getView.keywords = {'returnType': returnType}
    return getView

def getDataConstructors(includeViews=True):
    """
    Create data object constructors for tests iterating through each
    concrete data exc_type. By default includes constructors for views.
    """
    constructors = []
    for returnType in nimble.core.data.available:
        constructors.append(partial(nimble.data, returnType=returnType))
        if includeViews:
            constructors.append(_getViewFunc(returnType))
    return constructors

class raises:
    """
    Ensure that a given operation raises the expected exception.

    Can be used as a decorator or a context manager. The context manager
    returns a pytest.ExceptionInfo object for further exception analysis.

    Parameters
    ----------
    exception
        The exception object expected to be raised.
    match
        An optional string or regex object to further test that the string
        representation of the exception is as expected.
    """
    def __init__(self, exception, *args, **kwargs):
        self.raiser = pytest.raises(exception, *args, **kwargs)

    # as decorator
    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with self.raiser:
                return func(*args, **kwargs)
        return wrapped

    # as context manager
    def __enter__(self):
        return self.raiser.__enter__()

    def __exit__(self, exc_type, value, traceback):
        return self.raiser.__exit__(exc_type, value, traceback)

class patch:
    """
    Patch an object to do something different than it was designed to do.

    Can be used as a decorator or a context manager.

    Parameters
    ----------
    obj
        The object containing the attribute that will be patched.
    name : str
        The name of the attribute to apply the patch to.
    value
        The object that will replace the attribute.
    """
    def __init__(self, obj, name, value):
        self.obj = obj
        self.name = name
        self.value = value
        self.patch = pytest.MonkeyPatch()

    # as decorator
    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            self.patch.setattr(self.obj, self.name, self.value)
            try:
                return func(*args, **kwargs)
            finally:
                self.patch.undo()
        return wrapped

    # as context manager
    def __enter__(self):
        self.patch.setattr(self.obj, self.name, self.value)

    def __exit__(self, exc_type, value, traceback):
        self.patch.undo()

def assertNotCalled(obj, name):
    """
    Patch the object to raise a CalledFunctionException.

    Can be used as a decorator or a context manager. This is primarily useful
    when not expecting the mocked object to be used.
    """
    return patch(obj, name, calledException)

class assertCalled:
    """
    For testing that a given object is utilized during the test.

    Can be used as a decorator or a context manager. The object will be patched
    to return a CalledFunctionException and the test passes if that exception
    occurs. The return of the context manager is a pytest.ExceptionInfo object.
    """
    def __init__(self, obj, name):
        self.patch = assertNotCalled(obj, name)
        self.raises = raises(CalledFunctionException)

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            return self.raises(self.patch(func))(*args, **kwargs)
        return wrapped

    def __enter__(self):
        self.patch.__enter__()
        return self.raises.__enter__()

    def __exit__(self, exc_type, value, traceback):
        self.patch.__exit__(None, None, None)
        return self.raises.__exit__(exc_type, value, traceback)

def skipMissingPackage(package):
    """
    Return a decorator to skip a test if the package required is missing.
    """
    try:
        nimble.core._learnHelpers.findBestInterface(package)
        missing = False
    except PackageException:
        missing = True
    reason = package + ' package is not available'
    return pytest.mark.skipif(missing, reason=reason)
