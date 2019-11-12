"""
Test all user-facing functions add the expected number of log entries.

A list of all user-facing functions is created and compared to the list
of functions that we have confirmed are tested. If a functions are added
or removed this tests will fail indicating an update is necessary.

Tests for most functions will usually be located in the same file as
other tests for that function. Otherwise, some tests for basic
user-facing functions might be included here as well.  All tests for log
count should make use of the wrappers in tests/assertionHelpers.py
"""
import sys
import tempfile

import nimble
import nimble.calculate as calculate
import nimble.fill as fill
import nimble.match as match
from nimble.data import Base
from nimble.data import Axis
from nimble.data import Points
from nimble.data import Features
from nimble.data import Elements
from nimble.interfaces.universal_interface import UniversalInterface
from nimble.interfaces.universal_interface import TrainedLearner
from ..assertionHelpers import noLogEntryExpected, oneLogEntryExpected

ALL_USER_FACING = []
modules = [nimble, calculate, fill, match]
classes = [Base, Axis, Points, Features, Elements, UniversalInterface,
           TrainedLearner]
modulesAndClasses = modules + classes
for call in modulesAndClasses:
    for attribute in dir(call):
        if not attribute.startswith('_') and callable(getattr(call, attribute)):
            ALL_USER_FACING.append(call.__name__ + '.' + attribute)

def prefixAdder(prefix):
    def addPrefix(value):
        return prefix + '.' + value
    return addPrefix


#  Untested functions: register/deregisterCustomLearnerAsDefault, importModule
nimble_logged = [
    'createData', 'createRandomData', 'crossValidate', 'log', 'loadData',
    'loadTrainedLearner', 'normalizeData', 'setRandomSeed', 'train',
    'trainAndApply', 'trainAndTest', 'trainAndTestOnTrainingData',
    ]
nimble_notLogged = [
    'CV', 'Init', 'deregisterCustomLearner', 'deregisterCustomLearnerAsDefault',
    'identity', 'importModule', 'listLearners', 'learnerParameters',
    'learnerDefaultValues', 'learnerType', 'ones', 'registerCustomLearner',
    'registerCustomLearnerAsDefault', 'showLog', 'zeros',
    ]
nimble_funcs = nimble_logged + nimble_notLogged
nimble_tested = list(map(prefixAdder('nimble'), nimble_funcs))

# no calculate functions should be logged.
calculate_funcs = [
    'confidenceIntervalHelper', 'correlation', 'cosineSimilarity',
    'covariance', 'detectBestResult', 'elementwiseMultiply',
    'elementwisePower', 'fractionCorrect', 'fractionIncorrect', 'inverse',
    'leastSquaresSolution', 'maximum', 'mean', 'meanAbsoluteError',
    'meanFeaturewiseRootMeanSquareError', 'median', 'minimum', 'mode',
    'proportionMissing', 'proportionZero', 'pseudoInverse', 'quartiles',
    'residuals', 'rootMeanSquareError', 'rSquared', 'solve',
    'standardDeviation', 'uniqueCount', 'varianceFractionRemaining',
    ]
calculate_tested = list(map(prefixAdder('nimble.calculate'), calculate_funcs))

# no fill functions should be logged.
fill_funcs = [
    'backwardFill', 'constant', 'factory', 'forwardFill', 'interpolate',
    'kNeighborsClassifier', 'kNeighborsRegressor', 'mean', 'median', 'mode',
    ]
fill_tested = list(map(prefixAdder('nimble.fill'), fill_funcs))

# no match functions should not be logged.
match_funcs = [
    'allBoolean', 'allFalse', 'allInfinity', 'allMissing', 'allNegative',
    'allNonNumeric', 'allNonZero', 'allNumeric', 'allPositive', 'allTrue',
    'allValues', 'allZero', 'anyBoolean', 'anyFalse', 'anyInfinity',
    'anyMissing', 'anyNegative', 'anyNonNumeric', 'anyNonZero', 'anyNumeric',
    'anyPositive', 'anyTrue', 'anyValues', 'anyZero', 'boolean',
    'convertMatchToFunction', 'false', 'infinity', 'missing', 'negative',
    'nonNumeric', 'nonZero', 'numeric', 'positive', 'true', 'zero'
    ]
match_tested = list(map(prefixAdder('nimble.match'), match_funcs))

# NOTES:
#  The functionality of these functions is untested, but a test of their
#  expected log count can be found in this script:
#      copy, featureReport, summaryReport, getTypeString, groupByFeature,
#      hashCode, nameIsDefault, show, validate
base_logged = [
    'fillUsingAllData', 'featureReport', 'fillWith', 'flattenToOneFeature',
    'flattenToOnePoint', 'groupByFeature', 'matrixMultiply', 'merge',
    'replaceFeatureWithBinaryFeatures', 'summaryReport', 'trainAndTestSets',
    'transformFeatureToIntegers', 'transpose', 'unflattenFromOneFeature',
    'unflattenFromOnePoint',
    ]
base_notLogged = [
    'containsZero', 'copy', 'featureView', 'getTypeString', 'hashCode',
    'inverse', 'isApproximatelyEqual', 'isIdentical', 'nameIsDefault', 'plot',
    'plotFeatureAgainstFeature', 'plotFeatureAgainstFeatureRollingAverage',
    'plotFeatureDistribution', 'pointView', 'referenceDataFrom', 'save',
    'show', 'solveLinearSystem', 'toString', 'validate', 'view', 'writeFile',
    ]
base_funcs = base_logged + base_notLogged
base_tested = list(map(prefixAdder('Base'), base_funcs))

features_logged = [
    'add', 'calculate', 'copy', 'delete', 'extract', 'fill', 'mapReduce',
    'normalize', 'retain', 'setName', 'setNames', 'shuffle', 'sort',
    'transform', 'splitByParsing', 'matching',
    ]
features_notLogged = [
    'count', 'repeat', 'getIndex', 'getIndices', 'getName', 'getNames',
    'hasName', 'nonZeroIterator', 'similarities', 'statistics', 'unique',
    ]
features_funcs = features_logged + features_notLogged
features_tested = list(map(prefixAdder('Features'), features_funcs))

points_logged = [
    'add', 'calculate', 'copy', 'delete', 'extract', 'fill', 'mapReduce',
    'normalize', 'retain', 'setName', 'setNames', 'shuffle', 'sort',
    'transform', 'combineByExpandingFeatures', 'splitByCollapsingFeatures',
    'matching',
    ]
points_notLogged = [
    'count', 'repeat', 'getIndex', 'getIndices', 'getName', 'getNames',
    'hasName', 'nonZeroIterator', 'similarities', 'statistics', 'unique',
    ]
points_funcs = points_logged + points_notLogged
points_tested = list(map(prefixAdder('Points'), points_funcs))

elements_logged = [
    'calculate', 'transform', 'multiply', 'power', 'matching',
    ]
elements_notLogged = [
    'count', 'countUnique',
    ]
elements_funcs = elements_logged + elements_notLogged
elements_tested = list(map(prefixAdder('Elements'), elements_funcs))

ui_logged = [
    'train',
    ]
ui_notLogged = [
    'accessible', 'findCallable', 'getCanonicalName',
    'getLearnerDefaultValues', 'getLearnerParameterNames', 'getOption',
    'isAlias', 'learnerType', 'listLearners', 'setOption', 'version',
]
ui_funcs = ui_logged + ui_notLogged
ui_tested = list(map(prefixAdder('UniversalInterface'), ui_funcs))

tl_logged = [
    'apply', 'incrementalTrain', 'retrain', 'test',
    ]
tl_notLogged = [
    'getAttributes', 'getScores', 'save',
    ]
tl_funcs = tl_logged + tl_notLogged
tl_tested = list(map(prefixAdder('TrainedLearner'), tl_funcs))

USER_FACING_TESTED = (nimble_tested + calculate_tested + fill_tested
                      + match_tested + base_tested + features_tested
                      + points_tested + elements_tested + ui_tested
                      + tl_tested)

##############
# All tested #
##############

def testAllUserFacingLoggingTested():
    """Ensure that all user facing functions are tested for logging"""
    if not sorted(USER_FACING_TESTED) == sorted(ALL_USER_FACING):
        missing = [f for f in ALL_USER_FACING if f not in USER_FACING_TESTED]
        removed = [f for f in USER_FACING_TESTED if f not in ALL_USER_FACING]
        if missing:
            msg = 'The log count is not tested for {0} '.format(len(missing))
            msg += 'functions:'
            print(msg)
            print(missing)
        if removed:
            msg = 'The following {0} functions are '.format(len(removed))
            msg += 'no longer user-facing:'
            print(msg)
            print(removed)
        assert False

##########
# nimble #
##########
@oneLogEntryExpected
def test_log_logCount():
    customString = "enter this string into the log"
    nimble.log("customString", customString)

@noLogEntryExpected
def test_showLog_logCount():
    def wrapped(obj):
        return nimble.showLog()
    captureOutput(wrapped)

@noLogEntryExpected
def test_importModule_logCount():
    pd = nimble.importModule('pandas')

@noLogEntryExpected
def test_CV_logCount():
    k = nimble.CV([1, 3, 5])

@noLogEntryExpected
def test_Init_logCount():
    i = nimble.Init('foo', bar=1)

########
# Base #
########

@noLogEntryExpected
def test_copy_logCount():
    for rType in nimble.data.available:
        obj = nimble.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
        copy = obj.copy()

def test_featureReport_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.featureReport()
    captureOutput(wrapped)

def test_summaryReport_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.summaryReport()
    captureOutput(wrapped)

def test_groupByFeature_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.groupByFeature(0)
    for rType in nimble.data.available:
        obj = nimble.createData(rType, [[1,2,3],[1,4,5],[2,2,3],[2,4,5]],
                             useLog=False)
        grouped = wrapped(obj)

@noLogEntryExpected
def test_getTypeString_logCount():
    for rType in nimble.data.available:
        obj = nimble.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
        ts = obj.getTypeString()

@noLogEntryExpected
def test_hashCode_logCount():
    for rType in nimble.data.available:
        obj = nimble.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
        hash = obj.hashCode()

@noLogEntryExpected
def test_nameIsDefault_logCount():
    for rType in nimble.data.available:
        obj = nimble.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
        isDefault = obj.nameIsDefault()

@noLogEntryExpected
def test_show_logCount():
    def wrapped(obj):
        return obj.show(None)
    captureOutput(wrapped)

@noLogEntryExpected
def test_toString_logCount():
    def wrapped(obj):
        return obj.toString()
    captureOutput(wrapped)

@noLogEntryExpected
def test_validate_logCount():
    for rType in nimble.data.available:
        obj = nimble.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
        isDefault = obj.validate()

############################
# Points/Features/Elements #
############################

def test_points_shuffle_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.points.shuffle()
    for rType in nimble.data.available:
        obj = nimble.createData(rType, [[1,2,3],[1,4,5],[2,2,3],[2,4,5]],
                             useLog=False)
        grouped = wrapped(obj)

def test_features_shuffle_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.features.shuffle()
    for rType in nimble.data.available:
        obj = nimble.createData(rType, [[1,2,3],[1,4,5],[2,2,3],[2,4,5]],
                             useLog=False)
        grouped = wrapped(obj)

###############################
# dunder functions in classes #
###############################

ALL_DUNDER = []
for call in classes:
    objectDir = dir(object)
    ignore = ["__weakref__", "__module__", "__dict__", '__abstractmethods__']
    for attribute in dir(call):
        if (attribute.startswith('__')
                and attribute not in objectDir
                and attribute not in ignore):
            ALL_DUNDER.append(call.__name__ + '.' + attribute)

baseDunder_tested = list(map(prefixAdder('Base'),
    ['__abs__', '__add__', '__and__', '__bool__', '__copy__', '__deepcopy__',
     '__getitem__', '__floordiv__', '__iadd__', '__ifloordiv__', '__imod__',
     '__imatmul__', '__imul__', '__invert__', '__ipow__', '__isub__',
     '__itruediv__', '__len__', '__matmul__', '__mod__', '__mul__', '__neg__',
     '__or__', '__pos__', '__pow__', '__radd__', '__rfloordiv__',
     '__rmatmul__', '__rmod__', '__rmul__', '__rpow__', '__rsub__',
     '__rtruediv__', '__sub__', '__truediv__', '__xor__',
    ]))
axisDunder_tested = ['Axis.__bool__', 'Axis.__iter__', 'Axis.__len__',
                     'Axis.__getitem__']
pointsDunder_tested = []
featuresDunder_tested = []
elementsDunder_tested = ['Elements.__bool__', 'Elements.__iter__']
uiDunder_tested = []
tlDunder_tested = []

ALL_DUNDER_TESTED = (baseDunder_tested + axisDunder_tested
                     + pointsDunder_tested + featuresDunder_tested
                     + elementsDunder_tested
                     + uiDunder_tested + tlDunder_tested)

def testAllClassesDunderFunctions():
    assert sorted(ALL_DUNDER_TESTED) == sorted(ALL_DUNDER)

###########
# Helpers #
###########

def captureOutput(toCall):
    tmpFile = tempfile.TemporaryFile(mode='w')
    backupOut = sys.stdout
    sys.stdout = tmpFile
    try:
        for rType in nimble.data.available:
            obj = nimble.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
            ret = toCall(obj)
    finally:
        sys.stdout = backupOut
        tmpFile.close()
