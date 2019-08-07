"""
Unit tests for the universal interface object.

"""

from __future__ import absolute_import
import sys
import tempfile
import os

from nose.tools import raises

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.interfaces.universal_interface import UniversalInterface
from nimble.helpers import generateClassificationData
from ..assertionHelpers import noLogEntryExpected, oneLogEntryExpected


class Initable(object):
    def __init__(self, C=1, thresh=0.5):
        self.C = C
        self.thresh = thresh

    def __eq__(self, other):
        if self.C == other.C and self.thresh == other.thresh:
            return True
        return False


class TestInterface(UniversalInterface):
    def __init__(self):
        super(TestInterface, self).__init__()

    def accessible(self):
        return True

    def _listLearnersBackend(self):
        return ['l0', 'l1', 'l2', 'exposeTest']

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
        elif name == 'subFunc':
            return [['dup']]
        elif name == 'foo':
            return [['estimator']]
        elif name == 'initable':
            return [['C', 'thresh']]
        elif name == 'initable2':
            return [['C', 'thresh']]
        else:
            return [[]]

    def _getLearnerDefaultValuesBackend(self, name):
        return self._getDefaultValues(name)

    def _getDefaultValuesBackend(self, name):
        if name == 'bar':
            return [{'estimator': 'initable'}, {'estimater2': 'initable'}]
        if name == 'foo':
            return [{'estimator': Initable()}]
        if name == 'initable':
            return [{'C': 1, 'thresh': 0.5}]
        if name == 'initable2':
            return [{'C': 1, 'thresh': 0.5}]
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

    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        return (learnerName, trainX, trainY, arguments)

    def _applier(self, learnerName, learner, testX, arguments, customDict):
        return testX

    def _findCallableBackend(self, name):
        if name == 'initable':
            return Initable
        if name == 'initable2':
            return Initable
        available = ['l0', 'l1', 'l2', 'l1a0', 'subFunc', 'foo', 'bar', 'exposeTest']
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

    def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
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


#######################################
### _validateArgumentDistribution() ###
#######################################

#def test__validateArgumentDistribution

# TODO tests involving default arguments

@raises(InvalidArgumentValue)
def test__validateArgumentDistributionMissingArgument():
    learner = 'l0'
    arguments = {'l0a0': 1}
    TestObject._validateArgumentDistribution(learner, arguments)


@raises(InvalidArgumentValue)
def test__validateArgumentDistributionOverlappingArgumentsFlat():
    learner = 'l2'
    arguments = {'dup': 1, 'sub': 'subFunc'}
    TestObject._validateArgumentDistribution(learner, arguments)


@raises(InvalidArgumentValue)
def test__validateArgumentDistributionExtraArgument():
    learner = 'l1'
    arguments = {'l1a0': 1, 'l5a100': 11}
    TestObject._validateArgumentDistribution(learner, arguments)


def test__validateArgumentDistributionWorking():
    learner = 'l1'
    arguments = {'l1a0': 1}
    ret = TestObject._validateArgumentDistribution(learner, arguments)
    assert ret == {'l1a0': 1}


def test__validateArgumentDistributionOverlappingArgumentsNested():
    learner = 'l2'
    arguments = {'dup': 1, 'sub': 'subFunc', 'subFunc': {'dup': 2}}
    ret = TestObject._validateArgumentDistribution(learner, arguments)
    assert ret == {'dup': 1, 'sub': 'subFunc', 'subFunc': {'dup': 2}}


def test__validateArgumentDistributionInstantiableDefaultValue():
    learner = 'foo'
    arguments = {}
    ret = TestObject._validateArgumentDistribution(learner, arguments)
    assert ret == {'estimator': Initable()}

###############
# ignoring this test for now, since the code now works under the assumption that an interaces'
# default values will not be set up in a way to take advantage of our argument instantiation
# conventions. If a default value is a string, it should STAY a string, even if that string
# corresponds to an instantiable object
#def test__validateArgumentDistributionInstantiableDelayedAllocationOfSubArgsSeparate():
#	learner = 'foo'
#	arguments = {'initable':{'C':11}}
#	ret = TestObject._validateArgumentDistribution(learner, arguments)
#	assert ret == {'estimator':'initable', 'initable':{'C':11, 'thresh':0.5}}
##############

def test__validateArgumentDistributionInstantiableDelayedAllocationOfSubArgsInFlat():
    learner = 'foo'
    arguments = {'estimator': 'initable2', 'C': 11}
    ret = TestObject._validateArgumentDistribution(learner, arguments)
    assert ret == {'estimator': 'initable2', 'initable2': {'C': 11, 'thresh': 0.5}}


def test__validateArgumentDistributionInstantiableArgWithDefaultValue():
    learner = 'foo'
    arguments = {'estimator': 'initable', 'C': .11, 'thresh': 15}
    ret = TestObject._validateArgumentDistribution(learner, arguments)
    assert ret == {'estimator': 'initable', 'initable': {'C': .11, 'thresh': 15}}


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
    dummyNimble = nimble.createData('Matrix', [[1, 1], [2, 2]])
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
        self.writeWarningToStdErr()
        sys.stderr.flush()
        super(AlwaysWarnInterface, self).__init__()

    def writeWarningToStdErr(self):
        try: #py2
            sys.stderr.write('WARN TEST\n')
        except TypeError: #py3
            sys.stderr.write(b'WARN TEST\n')

    def accessible(self):
        return True

    def _listLearnersBackend(self):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        return ['foo']

    def _findCallableBackend(self, name):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        return 'fooCallableBackend'

    def _getParameterNamesBackend(self, name):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        return [[]]

    def _getLearnerParameterNamesBackend(self, name):
        return self._getParameterNames(name)

    def _getDefaultValuesBackend(self, name):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        return [{}]

    def _getLearnerDefaultValuesBackend(self, name):
        return self._getDefaultValues(name)

    def learnerType(self, name):
        pass

    def _getScores(self, learnerName, learner, testX, newArguments,
                   storedArguments, customDict):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        num = len(testX.points)
        raw = [0] * num
        return nimble.createData("Matrix", raw, useLog=False)

    def _getScoresOrder(self, learner):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        return [0, 1]

    def isAlias(self, name):
        if name.lower() in ['alwayswarn']:
            return True
        else:
            return False

    def getCanonicalName(self):
        return "AlwaysWarn"

    def _inputTransformation(self, learnerName, trainX, trainY, testX, arguments, customDict):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        return (trainX, trainY, testX, arguments)

    def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputType, outputFormat, customDict):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        return outputValue

    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        return (learnerName, trainX, trainY, arguments)

    def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        pass

    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        self.writeWarningToStdErr()
        sys.stderr.flush()
        return testX

    def _getAttributes(self, learnerBackend):
        self.writeWarningToStdErr()
        sys.stderr.flush()
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
    tempErr = tempfile.NamedTemporaryFile()
    backup = sys.stderr
    sys.stderr = tempErr

    try:
        if prepCall is not None:
            AWObject = AlwaysWarnInterface()
            arg = prepCall(AWObject)
        else:
            arg = AlwaysWarnInterface()

        startSizeErr = os.path.getsize(tempErr.name)
        startSizeCap = os.path.getsize(nimble.capturedErr.name)

        toCall(arg)

        endSizeErr = os.path.getsize(tempErr.name)
        endSizeCap = os.path.getsize(nimble.capturedErr.name)

        assert startSizeErr == endSizeErr
        assert startSizeCap != endSizeCap
    finally:
        sys.stderr = backup


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
    testX.features.add(testX[:, -1])

    def prep(AWObject):
        return AWObject.train('foo', trainX, trainY)

    try:
        def wrapped(tl):
            tl.apply(testX)
        backend_warningscapture(wrapped, prep)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

    try:
        def metric(x, y):
            pass
        def wrapped(tl):
            tl.test(testX, testY, metric)
        backend_warningscapture(wrapped, prep)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

    try:
        def wrapped(tl):
            tl.getScores(testX)
        backend_warningscapture(wrapped, prep)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
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
