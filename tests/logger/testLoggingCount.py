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
from UML.interfaces.universal_interface import TrainedLearner
from UML.configuration import configSafetyWrapper
from UML import importModule
from ..logHelpers import noLogEntryExpected, oneLogEntryExpected

calculate.__name__ = 'calculate'
fill.__name__ = 'fill'
match.__name__ = 'match'

ALL_USER_FACING = []
for call in [UML, calculate, fill, match, Base, Points, Features, Elements, TrainedLearner]:
    for attribute in dir(call):
        if not attribute.startswith('_') and callable(getattr(call,attribute)):
            ALL_USER_FACING.append(call.__name__ + '.' + attribute)

# NOTES:
#  UML.setRandomSeed logged?
#  Untested functions: register/deregisterCustomLearnerAsDefault, importModule
#  LogCount Testing Below: log, showLog, importModule

UML_logged = ['UML.createData', 'UML.createRandomData', 'UML.train',
              'UML.trainAndApply', 'UML.trainAndTest',
              'UML.trainAndTestOnTrainingData', 'UML.crossValidate',
              'UML.crossValidateReturnAll', 'UML.crossValidateReturnBest',
              'UML.normalizeData', 'UML.log', 'UML.loadData',
              'UML.loadTrainedLearner',
              ]
UML_notLogged = ['UML.ones', 'UML.zeros', 'UML.identity', 'UML.setRandomSeed',
                 'UML.showLog', 'UML.registerCustomLearner',
                 'UML.deregisterCustomLearner', 'UML.listLearners',
                 'UML.learnerParameters', 'UML.learnerDefaultValues',
                 'UML.learnerType', 'UML.importModule',
                 ]
UML_tested = UML_logged + UML_notLogged

calculate_logged = []
calculate_notLogged = []
calculate_tested = calculate_logged + calculate_notLogged

# all fill functions should not be logged.
fill_tested = ['fill.backwardFill', 'fill.constant', 'fill.factory',
               'fill.forwardFill', 'fill.interpolate',
               'fill.kNeighborsClassifier', 'fill.kNeighborsRegressor',
               'fill.mean', 'fill.median', 'fill.mode'
               ]

# all match functions should not be logged.
match_tested = ['match.allMissing', 'match.allNegative', 'match.allNonNumeric',
                'match.allNonZero', 'match.allNumeric', 'match.allPositive',
                'match.allValues', 'match.allZero', 'match.anyMissing',
                'match.anyNegative', 'match.anyNonNumeric', 'match.anyNonZero',
                'match.anyNumeric', 'match.anyPositive', 'match.anyValues',
                'match.anyZero', 'match.convertMatchToFunction',
                'match.missing', 'match.negative', 'match.nonNumeric',
                'match.nonZero', 'match.numeric', 'match.positive',
                'match.zero'
                ]

# NOTES:
#  The functionality of these functions is untested, but a test of their
#  expected log count can be found in this script:
#      copy, featureReport, summaryReport, getTypeString, groupByFeature,
#      hashCode, nameIsDefault, show, validate

Base_logged = ['Base.copy', 'Base.fillUsingAllData', 'Base.featureReport',
               'Base.fillWith', 'Base.flattenToOneFeature',
               'Base.flattenToOnePoint', 'Base.groupByFeature',
               'Base.merge', 'Base.replaceFeatureWithBinaryFeatures',
               'Base.summaryReport', 'Base.trainAndTestSets',
               'Base.transformFeatureToIntegers', 'Base.transpose',
               'Base.unflattenFromOneFeature', 'Base.unflattenFromOnePoint',
               ]
Base_notLogged = ['Base.containsZero', 'Base.copyAs', 'Base.featureView',
                  'Base.save', 'Base.getTypeString', 'Base.hashCode',
                  'Base.inverse', 'Base.isApproximatelyEqual',
                  'Base.isIdentical', 'Base.nameIsDefault', 'Base.plot',
                  'Base.plotFeatureAgainstFeature',
                  'Base.plotFeatureAgainstFeatureRollingAverage',
                  'Base.plotFeatureDistribution', 'Base.pointView',
                  'Base.referenceDataFrom', 'Base.save', 'Base.show',
                  'Base.solveLinearSystem', 'Base.toString', 'Base.validate',
                  'Base.view', 'Base.writeFile'
                  ]
Base_tested = Base_logged + Base_notLogged

Features_logged = []
Features_notLogged = []
Features_tested = Features_logged + Features_notLogged

Points_logged = []
Points_notLogged = []
Points_tested = Points_logged + Points_notLogged

Elements_logged = []
Elements_notLogged = []
Elements_tested = Elements_logged + Elements_notLogged

TrainedLearner_logged = []
TrainedLearner_notLogged = ['TrainedLearner.save']
TrainedLearner_tested = TrainedLearner_logged + TrainedLearner_notLogged


TESTED = (UML_tested + calculate_tested + fill_tested + match_tested
          + Base_tested + Features_tested + Points_tested + Elements_tested)

##############
# All tested #
##############

def testAllUserFacingLoggingTested():
    """This is to ensure that new user facing functions are tested for logging"""

    assert sorted(TESTED) == sorted(ALL_USER_FACING)

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
