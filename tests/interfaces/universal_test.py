"""
Unit tests for the universal interface object.
"""

import sys
import os
import warnings

from nose.tools import raises

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.core.interfaces.universal_interface import UniversalInterface
from tests.helpers import noLogEntryExpected, oneLogEntryExpected
from tests.helpers import generateClassificationData


class Initable(object):
    def __init__(self, C=1, thresh=0.5, sub=None):
        self.C = C
        self.thresh = thresh
        self.sub = sub

    def __eq__(self, other):
        if self.C != other.C:
            return False
        if self.thresh != other.thresh:
            return False
        if self.sub != other.sub:
            return False
        return True


class SubObj(object):
    def __init__(self, dup):
        self.dup = dup

    def __eq__(self, other):
        return self.dup == other.dup


class TestInterface(UniversalInterface):
    def __init__(self):
        super(TestInterface, self).__init__()

    def accessible(self):
        return True

    def _listLearnersBackend(self):
        return ['l0', 'l1', 'l2', 'exposeTest', 'foo']

    def _getLearnerParameterNamesBackend(self, name):
        return self._getParameterNames(name)

    def _getParameterNamesBackend(self, name):
        if name == 'l0':
            return [['l0a0', 'l0a1']]
        elif name == 'l1':
            return [['l1a0']]
        elif name == 'l2':
            return [['dup', 'sub']]
        elif name == 'l1a0':
            return [['l1a0a0']]
        elif name == 'subObj':
            return [['dup']]
        elif name == 'foo':
            return [['estimator']]
        elif name == 'initable':
            return [['C', 'thresh', 'sub']]
        else:
            return [[]]

    def _getLearnerDefaultValuesBackend(self, name):
        return self._getDefaultValues(name)

    def _getDefaultValuesBackend(self, name):
        if name == 'foo':
            return [{'estimator': Initable()}]
        if name == 'initable':
            return [{'C': 1, 'thresh': 0.5, 'sub': None}]
        return [{}]

    def isAlias(self, name):
        if name.lower() in ['test']:
            return True
        else:
            return False

    def getCanonicalName(self):
        return "Test"

    def _inputTransformation(self, learnerName, trainX, trainY, testX, arguments, customDict):
        return (trainX, trainY, testX, arguments)

    def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputType, outputFormat, customDict):
        return outputValue

    def _configurableOptionNames(self):
        return ['option']

    def _optionDefaults(self, option):
        return None

    def _exposedFunctions(self):
        return [self.exposedOne, self.exposedTwo, self.exposedThree]

    def _trainer(self, learnerName, trainX, trainY, arguments, randomSeed,
                 customDict):
        return (learnerName, trainX, trainY, arguments)

    def _applier(self, learnerName, learner, testX, arguments, customDict):
        return testX

    def _findCallableBackend(self, name):
        if name == 'initable':
            return Initable
        if name == 'subObj':
            return SubObj
        available = ['l0', 'l1', 'l2', 'l1a0', 'subObj', 'foo', 'exposeTest']
        if name in available:
            return name
        else:
            return None

    def _getScores(self, learner, trainX, trainY, arguments):
        pass

    def _getScoresOrder(self, learner):
        pass

    def learnerType(self, name):
        pass

    def _getAttributes(self, learnerBackend):
        pass

    def _incrementalTrainer(self, learnerName, learner, trainX, trainY,
                            arguments, customDict):
        pass

    def version(self):
        return "0.0.0"

    def exposedOne(self):
        return 1

    def exposedTwo(self):
        return 2

    def exposedThree(self, trainedLearner):
        assert trainedLearner.learnerName == 'exposeTest'
        return 3


TestObject = TestInterface()

###################
### nimble.Init ###
###################

def test_Init_str_repr_output():
    init = nimble.Init('foo', bar=2)
    exp = "Init('foo', bar=2)"
    assert str(init) == exp
    assert repr(init) == exp

@raises(TypeError)
def test_Init_args():
    init = nimble.Init('foo', 1)

########################
### _argumentInit() ###
#######################

@raises(InvalidArgumentValue)
def test_argumentInit_MissingArgumentForObjectInstantiation():
    toInit = nimble.Init('subObj')
    TestObject._argumentInit(toInit)

@raises(InvalidArgumentValue)
def test_argumentInit_ExtraArgumentForObjectInstantiation():
    toInit = nimble.Init('subObj', dup=2, bar=5)
    TestObject._argumentInit(toInit)

def test_argumentInit_ObjectInstantiatedDefault():
    toInit = nimble.Init('initable')
    ret = TestObject._argumentInit(toInit)
    assert ret == Initable()

def test_argumentInit_ObjectInstantiated_UninstantiatedParams():
    toInit = nimble.Init('initable', C=0.1, thresh=0.05)
    ret = TestObject._argumentInit(toInit)
    assert ret == Initable(C=0.1, thresh=0.05)

def test_argumentInit_ObjectInstantiated_InstantiatedParam():
    sub = nimble.Init('subObj', dup=2)
    toInit = nimble.Init('initable', C=1, thresh=0.05, sub=sub)
    ret = TestObject._argumentInit(toInit)
    assert ret == Initable(C=1, thresh=0.05, sub=SubObj(2))

@noLogEntryExpected
def test_accessible():
    assert TestObject.accessible()

@noLogEntryExpected
def test_getCanonicalName():
    assert TestObject.getCanonicalName() == 'Test'

@noLogEntryExpected
def test_learnerType():
    assert TestObject.learnerType('l0') is None

@noLogEntryExpected
def test_isAlias():
    assert TestObject.isAlias('test')

@noLogEntryExpected
def test_version():
    assert TestObject.version() == '0.0.0'

@noLogEntryExpected
def test_setOptionGetOption():
    assert TestObject.getOption('option') is None
    TestObject.setOption('option', 'set')
    assert TestObject.getOption('option') == 'set'

#########################
### exposed functions ###
#########################

def test_eachExposedPresent():
    dummyNimble = nimble.data('Matrix', [[1, 1], [2, 2]])
    tl = TestObject.train('exposeTest', dummyNimble, dummyNimble)
    assert hasattr(tl, 'exposedOne')
    assert tl.exposedOne() == 1
    assert hasattr(tl, 'exposedTwo')
    assert tl.exposedTwo() == 2
    assert hasattr(tl, 'exposedThree')
    assert tl.exposedThree() == 3


#####################
### OutputCapture ###
#####################

class AlwaysWarnInterface(UniversalInterface):
    def __init__(self):
        super(AlwaysWarnInterface, self).__init__()

    def issueWarnings(self):
        # Warnings that should not be ignored
        warnings.warn('Warning is NOT ignored', Warning)
        warnings.warn('UserWarning is NOT ignored', UserWarning)
        warnings.warn('SyntaxWarning is NOT ignored', SyntaxWarning)
        warnings.warn('RuntimeWarning is NOT ignored', RuntimeWarning)
        warnings.warn('UnicodeWarning is NOT ignored', UnicodeWarning)
        warnings.warn('BytesWarning is NOT ignored', BytesWarning)
        warnings.warn('ResourceWarning is NOT ignored', ResourceWarning)
        # warnings we ignore
        warnings.warn('DeprecationWarning is ignored', DeprecationWarning)
        warnings.warn('FutureWarning is ignored', FutureWarning)
        warnings.warn('ImportWarning is ignored', ImportWarning)
        warnings.warn('PendingDeprecationWarning is ignored',
                      PendingDeprecationWarning)

    def accessible(self):
        return True

    def _listLearnersBackend(self):
        self.issueWarnings()
        return ['foo']

    def _findCallableBackend(self, name):
        self.issueWarnings()
        return 'fooCallableBackend'

    def _getParameterNamesBackend(self, name):
        self.issueWarnings()
        return [[]]

    def _getLearnerParameterNamesBackend(self, name):
        return self._getParameterNames(name)

    def _getDefaultValuesBackend(self, name):
        self.issueWarnings()
        return [{}]

    def _getLearnerDefaultValuesBackend(self, name):
        return self._getDefaultValues(name)

    def learnerType(self, name):
        pass

    def _getScores(self, learnerName, learner, testX, newArguments,
                   storedArguments, customDict):
        self.issueWarnings()
        num = len(testX.points)
        raw = [0] * num
        return nimble.data("Matrix", raw, useLog=False)

    def _getScoresOrder(self, learner):
        self.issueWarnings()
        return [0, 1]

    def isAlias(self, name):
        if name.lower() in ['alwayswarn']:
            return True
        else:
            return False

    def getCanonicalName(self):
        return "AlwaysWarn"

    def _inputTransformation(self, learnerName, trainX, trainY, testX, arguments, customDict):
        self.issueWarnings()
        return (trainX, trainY, testX, arguments)

    def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputType, outputFormat, customDict):
        self.issueWarnings()
        return outputValue

    def _trainer(self, learnerName, trainX, trainY, arguments, randomSeed,
                 customDict):
        self.issueWarnings()
        return (learnerName, trainX, trainY, arguments)

    def _incrementalTrainer(self, learnerName, learner, trainX, trainY,
                            arguments, customDict):
        self.issueWarnings()
        pass

    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        self.issueWarnings()
        return testX

    def _getAttributes(self, learnerBackend):
        self.issueWarnings()
        pass

    def _optionDefaults(self, option):
        return []

    def _configurableOptionNames(self):
        return []

    def _exposedFunctions(self):
        return []

    def version(self):
        return "0.0.0"


#def test_warningscapture_init():
#	AWObject = AlwaysWarnInterface()

def backend_warningscapture(toCall, prepCall=None):
    """
    Only 7 out of the 11 warnings issued should be captured, the rest
    are ignored.
    """
    if prepCall is not None:
        AWObject = AlwaysWarnInterface()
        arg = prepCall(AWObject)
    else:
        arg = AlwaysWarnInterface()

    with warnings.catch_warnings(record=True) as warnCall:
        toCall(arg)

    ignored = [DeprecationWarning, PendingDeprecationWarning, FutureWarning,
               ImportWarning]

    # some warnings should have been captured
    assert len(warnCall) > 0
    assert not any(c.category in ignored for c in warnCall)
    # some calls capture multiple times but each time 7 should be caught
    assert len(warnCall) % 7 == 0


def test_warningscapture_train():
    cData = generateClassificationData(2, 10, 5)
    ((trainX, trainY), (testX, testY)) = cData

    @noLogEntryExpected
    def wrapped(AWObject):
        AWObject.train('foo', trainX)

    backend_warningscapture(wrapped)


def test_warningscapture_TL_test():
    cData = generateClassificationData(2, 10, 5)
    ((trainX, trainY), (testX, testY)) = cData

    def metric(x, y):
        return 0

    def prep(AWObject):
        return AWObject.train('foo', trainX, trainY)

    @oneLogEntryExpected
    def wrapped(arg):
        arg.test(testX, testY, metric)

    backend_warningscapture(wrapped, prep)


def test_warningscapture_TL_apply():
    cData = generateClassificationData(2, 10, 5)
    ((trainX, trainY), (testX, testY)) = cData

    def prep(AWObject):
        return AWObject.train('foo', trainX, trainY)

    @oneLogEntryExpected
    def wrapped(arg):
        arg.apply(testX)

    backend_warningscapture(wrapped, prep)


def test_warningscapture_TL_retrain():
    cData = generateClassificationData(2, 10, 5)
    ((trainX, trainY), (testX, testY)) = cData

    def prep(AWObject):
        return AWObject.train('foo', trainX, trainY)

    @oneLogEntryExpected
    def wrapped(arg):
        arg.retrain(testX, testY)

    backend_warningscapture(wrapped, prep)


def test_warningscapture_TL_incrementalTrain():
    cData = generateClassificationData(2, 10, 5)
    ((trainX, trainY), (testX, testY)) = cData

    def prep(AWObject):
        return AWObject.train('foo', trainX, trainY)

    @oneLogEntryExpected
    def wrapped(arg):
        arg.incrementalTrain(testX, testY)

    backend_warningscapture(wrapped, prep)


def test_warningscapture_TL_getAttributes():
    cData = generateClassificationData(2, 10, 5)
    ((trainX, trainY), (testX, testY)) = cData

    def prep(AWObject):
        return AWObject.train('foo', trainX, trainY)

    @noLogEntryExpected
    def wrapped(arg):
        arg.getAttributes()

    backend_warningscapture(wrapped, prep)


def test_warningscapture_TL_getScores():
    cData = generateClassificationData(2, 10, 5)
    ((trainX, trainY), (testX, testY)) = cData

    def prep(AWObject):
        return AWObject.train('foo', trainX, trainY)

    @noLogEntryExpected
    def wrapped(arg):
        arg.getScores(testX)

    backend_warningscapture(wrapped, prep)


def test_warningscapture_TL_exceptions_featureMismatch():
    cData = generateClassificationData(2, 10, 5)
    ((trainX, trainY), (testX, testY)) = cData
    # add an extra testX feature to raise exception
    testX.features.append(testX[:, -1])

    def prep(AWObject):
        return AWObject.train('foo', trainX, trainY)

    try:
        def wrapped(tl):
            tl.apply(testX)
        backend_warningscapture(wrapped, prep)
        assert False # expected InvalidArgumentValueCombination
    except InvalidArgumentValueCombination:
        pass

    try:
        def metric(x, y):
            pass
        def wrapped(tl):
            tl.test(testX, testY, metric)
        backend_warningscapture(wrapped, prep)
        assert False # expected InvalidArgumentValueCombination
    except InvalidArgumentValueCombination:
        pass

    try:
        def wrapped(tl):
            tl.getScores(testX)
        backend_warningscapture(wrapped, prep)
        assert False # expected InvalidArgumentValueCombination
    except InvalidArgumentValueCombination:
        pass


def test_warningscapture_listLearners():
    @noLogEntryExpected
    def wrapped(AWObject):
        AWObject.listLearners()

    backend_warningscapture(wrapped)


def test_warningscapture_findCallable():
    @noLogEntryExpected
    def wrapped(AWObject):
        AWObject.findCallable('foo')

    backend_warningscapture(wrapped)


def test_warningscapture_getLearnerParameterNames():
    @noLogEntryExpected
    def wrapped(AWObject):
        AWObject.getLearnerParameterNames('foo')

    backend_warningscapture(wrapped)


def test_warningscapture_getLearnerDefaultValues():
    @noLogEntryExpected
    def wrapped(AWObject):
        AWObject.getLearnerDefaultValues('foo')

    backend_warningscapture(wrapped)

# import each test suite


# one test for each interface, recalls all the tests in the suite?
