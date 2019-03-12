"""
Test all user-facing functions add the expected number of log entries.

A list of all user-facing functions is created and compared to the list
of functions that we have confirmed are tested. If a functions are added
or removed this tests will fail indicating an update is necessary.

Tests for most functions will usually be located in the same file as
other tests for that function, and the wrappers created here can be
imported to test for log count. Otherwise, some tests for basic
user-facing functions might be included here as well.
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

fill_logged = []
fill_notLogged = []
fill_tested = fill_logged + fill_notLogged

match_logged = []
match_notLogged = []
match_tested = match_logged + match_notLogged

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
TrainedLearner_notLogged = []
TrainedLearner_tested = TrainedLearner_logged + TrainedLearner_notLogged


TESTED = (UML_tested + calculate_tested + fill_tested + match_tested
          + Base_tested + Features_tested + Points_tested + Elements_tested)

def testAllUserFacingLoggingTested():
    """This is to ensure that new user facing functions are tested for logging"""
    assert sorted(TESTED) == sorted(ALL_USER_FACING)

def logCountAssertionFactory(count):
    """
    Generate a wrapper to assert the log increased by a certain count.
    """
    def logCountAssertion(function):
        @configSafetyWrapper
        def wrapped(*args, **kwargs):
            UML.settings.set('logger', 'enabledByDefault', 'True')
            UML.settings.set('logger', 'enableCrossValidationDeepLogging', 'True')
            logger = UML.logger.active
            countQuery = "SELECT COUNT(entry) FROM logger"
            startCount = logger.extractFromLog(countQuery)[0][0]
            ret = function(*args, **kwargs)
            endCount = logger.extractFromLog(countQuery)[0][0]
            assert startCount + count == endCount
            return ret
        wrapped.__name__ = function.__name__
        wrapped.__doc__ = function.__doc__
        return wrapped
    return logCountAssertion

noLogEntryExpected = logCountAssertionFactory(0)
oneLogEntryExpected = logCountAssertionFactory(1)
twoLogEntriesExpected = logCountAssertionFactory(2)

@noLogEntryExpected
def test_importModule_logCount():
    pd = UML.importModule('pandas')
