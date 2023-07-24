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
import nimble.exceptions as exceptions
import nimble.fill as fill
import nimble.match as match
import nimble.random as random
from nimble.core.data import Base
from nimble.core.data import Axis
from nimble.core.data import Points
from nimble.core.data import Features
from nimble.core.interfaces.universal_interface import TrainedLearner
from tests.helpers import noLogEntryExpected, oneLogEntryExpected
from tests.helpers import getDataConstructors

ALL_USER_FACING = []
modules = [nimble, calculate, exceptions, fill, match, random]
classes = [Base, Axis, Points, Features, TrainedLearner]
modulesAndClasses = modules + classes
for call in modulesAndClasses:
    for attribute in dir(call):
        if not attribute.startswith('_') and callable(getattr(call, attribute)):
            ALL_USER_FACING.append(call.__name__ + '.' + attribute)

def prefixAdder(prefix):
    def addPrefix(value):
        return prefix + '.' + value
    return addPrefix


nimble_logged = [
    'data', 'fillMatching', 'log', 'loadTrainedLearner', 'normalizeData',
    'train', 'trainAndApply', 'trainAndTest', 'trainAndTestOnTrainingData',
    ]
nimble_notLogged = [
    'Tuning', 'CustomLearner', 'Tune', 'Init', 'fetchFiles',
    'identity', 'learnerNames', 'learnerParameters',
    'learnerParameterDefaults', 'learnerType', 'ones', 'showAvailablePackages',
    'showLearnerParameterDefaults', 'showLearnerNames',
    'showLearnerParameters', 'showLog', 'zeros',
    ]
nimble_funcs = nimble_logged + nimble_notLogged
nimble_tested = list(map(prefixAdder('nimble'), nimble_funcs))

# no calculate functions should be logged.
calculate_funcs = [
    'balancedAccuracy', 'confusionMatrix', 'correlation', 'cosineSimilarity',
    'count', 'covariance', 'detectBestResult', 'elementwiseMultiply',
    'elementwisePower', 'f1Score', 'falseNegative', 'falsePositive',
    'fractionCorrect', 'fractionIncorrect',
    'fractionIncorrectConfidenceInterval', 'inverse', 'leastSquaresSolution',
    'maximum', 'mean', 'meanAbsoluteError',
    'meanAbsoluteErrorConfidenceInterval',
    'meanFeaturewiseRootMeanSquareError', 'meanNormalize',
    'meanStandardDeviationNormalize', 'median', 'medianAbsoluteDeviation',
    'minimum', 'mode', 'percentileNormalize', 'precision',
    'performanceFunction', 'proportionMissing', 'proportionZero',
    'pseudoInverse', 'quartiles', 'range0to1Normalize', 'recall', 'residuals',
    'rootMeanSquareError', 'rootMeanSquareErrorConfidenceInterval', 'rSquared',
    'solve', 'specificity', 'standardDeviation', 'sum', 'trueNegative',
    'truePositive', 'uniqueCount', 'variance', 'varianceFractionRemaining',
    ]
calculate_tested = list(map(prefixAdder('nimble.calculate'), calculate_funcs))

# no exceptions should be logged.
exceptions_funcs = [
    'NimbleException', 'InvalidArgumentType', 'InvalidArgumentValue',
    'InvalidArgumentTypeCombination', 'InvalidArgumentValueCombination',
    'ImproperObjectAction', 'PackageException', 'FileFormatException',
    ]
exceptions_tested = list(map(prefixAdder('nimble.exceptions'),
                             exceptions_funcs))

# no fill functions should be logged.
fill_funcs = [
    'backwardFill', 'constant', 'forwardFill', 'interpolate', 'mean', 'median',
    'mode',
    ]
fill_tested = list(map(prefixAdder('nimble.fill'), fill_funcs))

# no match functions should be logged.

match_funcs = [
    'QueryString', 'allBoolean', 'allFalse', 'allFloating', 'allInfinity',
    'allInteger', 'allMissing', 'allNegative', 'allNonMissing',
    'allNonNumeric', 'allNonZero', 'allNumeric', 'allPositive', 'allTrue',
    'allValues', 'allZero', 'anyBoolean', 'anyFalse', 'anyFloating',
    'anyInfinity', 'anyInteger', 'anyMissing', 'anyNegative', 'anyNonMissing',
    'anyNonNumeric', 'anyNonZero', 'anyNumeric', 'anyPositive', 'anyTrue',
    'anyValues', 'anyZero', 'boolean', 'false', 'floating', 'infinity',
    'integer', 'missing', 'negative', 'nonMissing', 'nonNumeric', 'nonZero',
    'numeric', 'positive', 'true', 'zero',
    ]
match_tested = list(map(prefixAdder('nimble.match'), match_funcs))

random_logged = [
    'data', 'setSeed', 'alternateControl', 'generateSubsidiarySeed'
    ]
random_funcs = random_logged
random_tested = list(map(prefixAdder('nimble.random'), random_funcs))

# NOTES:
#  The functionality of these functions is untested, but a test of their
#  expected log count can be found in this script:
#      copy, report, getTypeString, groupByFeature, hashCode, show,
#      checkInvariants
base_logged = [
    'calculateOnElements', 'flatten', 'groupByFeature', 'matchingElements',
    'merge', 'replaceFeatureWithBinaryFeatures', 'replaceRectangle', 'report',
    'trainAndTestSets', 'transformElements', 'transformFeatureToIntegers',
    'transpose', 'unflatten',
    ]
base_notLogged = [
    'containsZero', 'copy', 'countElements', 'countUniqueElements',
    'featureView', 'getTypeString', 'hashCode', 'inverse',
    'isApproximatelyEqual', 'isIdentical', 'iterateElements', 'matrixMultiply',
    'matrixPower', 'plotHeatMap', 'plotFeatureAgainstFeature',
    'plotFeatureAgainstFeatureRollingAverage', 'plotFeatureDistribution',
    'plotFeatureGroupMeans', 'plotFeatureGroupStatistics', 'pointView',
    'save', 'show', 'solveLinearSystem', 'toString', 'checkInvariants', 'view',
    ]
base_funcs = base_logged + base_notLogged
base_tested = list(map(prefixAdder('Base'), base_funcs))

features_logged = [
    'append', 'calculate', 'copy', 'delete', 'extract', 'fillMatching',
    'insert', 'matching', 'normalize', 'permute', 'repeat','replace', 'report',
    'retain', 'setNames', 'sort', 'transform', 'splitByParsing',
    ]
features_notLogged = [
    'count', 'getIndex', 'getIndices', 'getName', 'getNames', 'hasName',
    'plot', 'plotMeans', 'plotStatistics', 'similarities', 'statistics',
    'unique',
    ]
features_funcs = features_logged + features_notLogged
features_tested = list(map(prefixAdder('Features'), features_funcs))

points_logged = [
    'append', 'calculate', 'copy', 'delete', 'extract', 'fillMatching',
    'insert', 'mapReduce', 'matching', 'permute', 'repeat', 'replace',
    'retain', 'setNames', 'sort', 'transform',
    'combineByExpandingFeatures', 'splitByCollapsingFeatures',
    ]
points_notLogged = [
    'count', 'getIndex', 'getIndices', 'getName', 'getNames', 'hasName',
    'plot', 'plotMeans', 'plotStatistics', 'similarities', 'statistics',
    'unique',
    ]
points_funcs = points_logged + points_notLogged
points_tested = list(map(prefixAdder('Points'), points_funcs))

tl_logged = [
    'apply', 'incrementalTrain', 'retrain', 'test',
    ]
tl_notLogged = [
    'getAttributes', 'getScores', 'save',
    ]
tl_funcs = tl_logged + tl_notLogged
tl_tested = list(map(prefixAdder('TrainedLearner'), tl_funcs))

USER_FACING_TESTED = (nimble_tested + calculate_tested + exceptions_tested
                      + fill_tested + match_tested + random_tested
                      + base_tested + features_tested + points_tested
                      + tl_tested)

##############
# All tested #
##############

def testAllUserFacingLoggingTested():
    """
    Ensure that all user facing functions are tested for logging.

    Checks consistency by a hand maintained list of methods the developers
    have confirmed are tested for the presence of absence of logging side
    effects against a list of user accessible methods that has been
    programatically generated.
    """
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
def test_Tune_logCount():
    k = nimble.Tune([1, 3, 5])

@noLogEntryExpected
def test_Init_logCount():
    i = nimble.Init('foo', bar=1)

@noLogEntryExpected
def test_showAvailablePackages_logCount():
    nimble.showAvailablePackages()

########
# Base #
########

@noLogEntryExpected
def test_copy_logCount():
    for constructor in getDataConstructors():
        obj = constructor([[1,2,3],[4,5,6]], useLog=False)
        copy = obj.copy()

def test_features_report_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.features.report()
    captureOutput(wrapped)

def test_report_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.report()
    captureOutput(wrapped)

def test_groupByFeature_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.groupByFeature(0)
    for constructor in getDataConstructors():
        obj = constructor([[1,2,3],[1,4,5],[2,2,3],[2,4,5]],
                          useLog=False)
        grouped = wrapped(obj)

@noLogEntryExpected
def test_getTypeString_logCount():
    for constructor in getDataConstructors():
        obj = constructor([[1,2,3],[4,5,6]], useLog=False)
        ts = obj.getTypeString()

@noLogEntryExpected
def test_hashCode_logCount():
    for constructor in getDataConstructors():
        obj = constructor([[1,2,3],[4,5,6]], useLog=False)
        hash = obj.hashCode()

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
def test_checkInvariants_logCount():
    for constructor in getDataConstructors():
        obj = constructor([[1,2,3],[4,5,6]], useLog=False)
        isDefault = obj.checkInvariants()

############################
# Points/Features/Elements #
############################

def test_points_permute_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.points.permute()
    for constructor in getDataConstructors(includeViews=False):
        obj = constructor([[1,2,3],[1,4,5],[2,2,3],[2,4,5]],
                          useLog=False)
        grouped = wrapped(obj)

def test_features_permute_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.features.permute()
    for constructor in getDataConstructors(includeViews=False):
        obj = constructor([[1,2,3],[1,4,5],[2,2,3],[2,4,5]],
                             useLog=False)
        grouped = wrapped(obj)

###############################
# dunder functions in classes #
###############################

ALL_DUNDER = []
for call in classes:
    objectDir = dir(object)
    ignore = ["__weakref__", "__module__", "__dict__", '__abstractmethods__',
              "__slots__"]
    for attribute in dir(call):
        if (attribute.startswith('__')
                and attribute not in objectDir
                and attribute not in ignore):
            ALL_DUNDER.append(call.__name__ + '.' + attribute)

baseDunder_tested = list(map(prefixAdder('Base'),
    ['__abs__', '__add__', '__and__', '__bool__', '__copy__', '__deepcopy__',
     '__getitem__', '__floordiv__', '__iadd__', '__ifloordiv__', '__imod__',
     '__imatmul__', '__imul__', '__invert__', '__ipow__', '__isub__',
     '__iter__', '__itruediv__', '__len__', '__matmul__', '__mod__', '__mul__',
     '__neg__', '__or__', '__pos__', '__pow__', '__radd__', '__rfloordiv__',
     '__rmatmul__', '__rmod__', '__rmul__', '__rpow__', '__rsub__',
     '__rtruediv__', '__sub__', '__truediv__', '__xor__',
    ]))
axisDunder_tested = ['Axis.__bool__', 'Axis.__len__']
pointsDunder_tested = ['Points.__iter__', 'Points.__getitem__']
featuresDunder_tested = ['Features.__iter__', 'Features.__getitem__']
uiDunder_tested = []
tlDunder_tested = []

ALL_DUNDER_TESTED = (baseDunder_tested + axisDunder_tested
                     + pointsDunder_tested + featuresDunder_tested
                     + uiDunder_tested + tlDunder_tested)

def testAllClassesDunderFunctions():
    assert sorted(ALL_DUNDER_TESTED) == sorted(ALL_DUNDER)

###########
# Helpers #
###########

def captureOutput(toCall):
    with tempfile.TemporaryFile(mode='w') as tmpFile:
        backupOut = sys.stdout
        sys.stdout = tmpFile
        try:
            for constructor in getDataConstructors():
                obj = constructor([[1,2,3],[4,5,6]], useLog=False)
                ret = toCall(obj)
        finally:
            sys.stdout = backupOut
