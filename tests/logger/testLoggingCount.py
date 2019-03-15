"""
Test all user-facing functions add the expected number of log entries.

A list of all user-facing functions is created and compared to the list
of functions that we have confirmed are tested. If a functions are added
or removed this tests will fail indicating an update is necessary.

Tests for most functions will usually be located in the same file as
other tests for that function. Otherwise, some tests for basic
user-facing functions might be included here as well.  All tests for log
count should make use of the wrappers in tests/logHelpers.py
"""
import sys
import tempfile

import UML
import UML.calculate as calculate
import UML.fill as fill
import UML.match as match
from UML.data import Base
from UML.data import Points
from UML.data import Features
from UML.data import Elements
from UML.interfaces.universal_interface import UniversalInterface
from UML.interfaces.universal_interface import TrainedLearner
from ..logHelpers import noLogEntryExpected, oneLogEntryExpected

calculate.__name__ = 'calculate'
fill.__name__ = 'fill'
match.__name__ = 'match'

ALL_USER_FACING = []
modulesAndClasses = [UML, calculate, fill, match, Base, Points, Features,
                     Elements, UniversalInterface, TrainedLearner]
for call in modulesAndClasses:
    for attribute in dir(call):
        if not attribute.startswith('_') and callable(getattr(call, attribute)):
            ALL_USER_FACING.append(call.__name__ + '.' + attribute)

def prefixAdder(prefix):
    def addPrefix(value):
        return prefix + '.' + value
    return addPrefix

# NOTES:
#  setRandomSeed logged?
#  Untested functions: register/deregisterCustomLearnerAsDefault, importModule
#  LogCount Testing Below: log, showLog, importModule

uml_logged = [
    'createData', 'createRandomData', 'train', 'trainAndApply', 'trainAndTest',
    'trainAndTestOnTrainingData', 'crossValidate', 'crossValidateReturnAll',
    'crossValidateReturnBest', 'normalizeData', 'log', 'loadData',
    'loadTrainedLearner',
    ]
uml_notLogged = [
    'ones', 'zeros', 'identity', 'setRandomSeed', 'showLog',
    'registerCustomLearner', 'deregisterCustomLearner', 'listLearners',
    'learnerParameters', 'learnerDefaultValues', 'learnerType', 'importModule',
    ]
uml_funcs = uml_logged + uml_notLogged
uml_tested = list(map(prefixAdder('UML'), uml_funcs))

# NOTES:
#  untested functions: elementWiseMultiply, elementWisePower, fractionCorrect,
#                      fractionIncorrect, rSquared, varianceFractionRemaining
# TODO in lin_alg.py inverse, leastSquaresSolution, pseudoInverse, solve
# no calculate functions should be logged.
calculate_funcs = [
    'confidenceIntervalHelper', 'correlation', 'cosineSimilarity',
    'covariance', 'detectBestResult', 'maximum', 'mean', 'meanAbsoluteError',
    'meanFeaturewiseRootMeanSquareError', 'median', 'minimum', 'mode',
    'proportionMissing', 'proportionZero', 'quartiles', 'residuals',
    'rootMeanSquareError', 'standardDeviation', 'uniqueCount',
    ]
calculate_tested = list(map(prefixAdder('calculate'), calculate_funcs))

# no fill functions should be logged.
fill_funcs = [
    'backwardFill', 'constant', 'factory', 'forwardFill', 'interpolate',
    'kNeighborsClassifier', 'kNeighborsRegressor', 'mean', 'median', 'mode',
    ]
fill_tested = list(map(prefixAdder('fill'), fill_funcs))

# no match functions should not be logged.
match_funcs = [
    'allMissing', 'allNegative', 'allNonNumeric', 'allNonZero', 'allNumeric',
    'allPositive', 'allValues', 'allZero', 'anyMissing', 'anyNegative',
    'anyNonNumeric', 'anyNonZero', 'anyNumeric', 'anyPositive', 'anyValues',
    'anyZero', 'convertMatchToFunction', 'missing', 'negative', 'nonNumeric',
    'nonZero', 'numeric', 'positive', 'zero',
    ]
match_tested = list(map(prefixAdder('match'), match_funcs))
# NOTES:
#  The functionality of these functions is untested, but a test of their
#  expected log count can be found in this script:
#      copy, featureReport, summaryReport, getTypeString, groupByFeature,
#      hashCode, nameIsDefault, show, validate

base_logged = [
    'copy', 'fillUsingAllData', 'featureReport', 'fillWith',
    'flattenToOneFeature', 'flattenToOnePoint', 'groupByFeature', 'merge',
    'replaceFeatureWithBinaryFeatures', 'summaryReport', 'trainAndTestSets',
    'transformFeatureToIntegers', 'transpose', 'unflattenFromOneFeature',
    'unflattenFromOnePoint',
    ]
base_notLogged = [
    'containsZero', 'copyAs', 'featureView', 'save', 'getTypeString',
    'hashCode', 'inverse', 'isApproximatelyEqual', 'isIdentical',
    'nameIsDefault', 'plot', 'plotFeatureAgainstFeature',
    'plotFeatureAgainstFeatureRollingAverage', 'plotFeatureDistribution',
    'pointView', 'referenceDataFrom', 'save', 'show', 'solveLinearSystem',
    'toString', 'validate', 'view', 'writeFile',
    ]
base_funcs = base_logged + base_notLogged
base_tested = list(map(prefixAdder('Base'), base_funcs))

features_logged = [
    'add', 'calculate', 'copy', 'delete', 'extract', 'fill', 'mapReduce',
    'normalize', 'retain', 'setName', 'setNames', 'shuffle', 'sort',
    'transform', 'splitByParsing',
    ]
features_notLogged = [
    'count', 'getIndex', 'getIndices', 'getName', 'getNames', 'hasName',
    'nonZeroIterator', 'similarities', 'statistics', 'unique',
    ]
features_funcs = features_logged + features_notLogged
features_tested = list(map(prefixAdder('Features'), features_funcs))

points_logged = [
    'add', 'calculate', 'copy', 'delete', 'extract', 'fill', 'mapReduce',
    'normalize', 'retain', 'setName', 'setNames', 'shuffle', 'sort',
    'transform', 'combineByExpandingFeatures', 'splitByCollapsingFeatures',
    ]
points_notLogged = [
    'count', 'getIndex', 'getIndices', 'getName', 'getNames', 'hasName',
    'nonZeroIterator', 'similarities', 'statistics', 'unique',
    ]
points_funcs = points_logged + points_notLogged
points_tested = list(map(prefixAdder('Points'), points_funcs))

elements_logged = [
    'calculate', 'transform', 'multiply', 'power',
    ]
elements_notLogged = [
    'count', 'countUnique', 'next',
    ]
elements_funcs = elements_logged + elements_notLogged
elements_tested = list(map(prefixAdder('Elements'), elements_funcs))

ui_logged = [

    ]
ui_notLogged = [
    'accessible', 'findCallable', 'getCanonicalName',
    'getLearnerDefaultValues', 'getLearnerParameterNames', 'isAlias',
    'learnerType', 'listLearners', 'version',
]
ui_funcs = ui_logged + ui_notLogged
ui_tested = list(map(prefixAdder('UniversalInterface'), ui_funcs))

tl_logged = [
    'apply', 'incrementalTrain', 'retrain', 'test',
    ]
tl_notLogged = [
    'save', 'getAttributes', 'getScores',
    ]
tl_funcs = tl_logged + tl_notLogged
tl_tested = list(map(prefixAdder('TrainedLearner'), tl_funcs))

TESTED = (uml_tested + calculate_tested + fill_tested + match_tested
          + base_tested + features_tested + points_tested + elements_tested
          + ui_tested + tl_tested)

##############
# All tested #
##############

def testAllUserFacingLoggingTested():
    """Ensure that all user facing functions are tested for logging"""
    if not sorted(TESTED) == sorted(ALL_USER_FACING):
        missing = [f for f in ALL_USER_FACING if f not in TESTED]
        removed = [f for f in TESTED if f not in ALL_USER_FACING]
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

#######
# UML #
#######
@oneLogEntryExpected
def test_log_logCount():
    customString = "enter this string into the log"
    UML.log("customString", customString)

@noLogEntryExpected
def test_showLog_logCount():
    def wrapped(obj):
        return UML.showLog()
    captureOutput(wrapped)

@noLogEntryExpected
def test_importModule_logCount():
    pd = UML.importModule('pandas')

########
# Base #
########

@noLogEntryExpected
def test_copy_logCount():
    for rType in UML.data.available:
        obj = UML.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
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
    for rType in UML.data.available:
        obj = UML.createData(rType, [[1,2,3],[1,4,5],[2,2,3],[2,4,5]],
                             useLog=False)
        grouped = wrapped(obj)


@noLogEntryExpected
def test_getTypeString_logCount():
    for rType in UML.data.available:
        obj = UML.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
        ts = obj.getTypeString()

@noLogEntryExpected
def test_hashCode_logCount():
    for rType in UML.data.available:
        obj = UML.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
        hash = obj.hashCode()

@noLogEntryExpected
def test_nameIsDefault_logCount():
    for rType in UML.data.available:
        obj = UML.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
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
    for rType in UML.data.available:
        obj = UML.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
        isDefault = obj.validate()

############################
# Points/Features/Elements #
############################

def test_points_shuffle_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.points.shuffle()
    for rType in UML.data.available:
        obj = UML.createData(rType, [[1,2,3],[1,4,5],[2,2,3],[2,4,5]],
                             useLog=False)
        grouped = wrapped(obj)

def test_features_shuffle_logCount():
    @oneLogEntryExpected
    def wrapped(obj):
        return obj.features.shuffle()
    for rType in UML.data.available:
        obj = UML.createData(rType, [[1,2,3],[1,4,5],[2,2,3],[2,4,5]],
                             useLog=False)
        grouped = wrapped(obj)

# TODO dunder functions in classes

###########
# Helpers #
###########

def captureOutput(toCall):
    tmpFile = tempfile.TemporaryFile(mode='w')
    backupOut = sys.stdout
    sys.stdout = tmpFile
    try:
        for rType in UML.data.available:
            obj = UML.createData(rType, [[1,2,3],[4,5,6]], useLog=False)
            ret = toCall(obj)
    finally:
        sys.stdout = backupOut
        tmpFile.close()
