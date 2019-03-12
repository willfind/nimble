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
from ..logHelpers import noLogEntryExpected

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
#  (de)registerCustomLearnerAsDefault functions are completely untested

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

Base_logged = []
Base_notLogged = ['save']
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
TrainedLearner_notLogged = ['save']
TrainedLearner_tested = TrainedLearner_logged + TrainedLearner_notLogged


TESTED = (UML_tested + calculate_tested + fill_tested + match_tested
          + Base_tested + Features_tested + Points_tested + Elements_tested)

def testAllUserFacingLoggingTested():
    """This is to ensure that new user facing functions are tested for logging"""
    print([f for f in ALL_USER_FACING if f.startswith('match')])
    assert sorted(TESTED) == sorted(ALL_USER_FACING)

@noLogEntryExpected
def test_importModule_logCount():
    pd = UML.importModule('pandas')
