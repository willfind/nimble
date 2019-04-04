"""
Common assertions helpers to be used in multiple test files.

Custom assertion types can be helpful if the assertion can be added to
existing tests which are also testing other functionality.
"""


import UML
from UML.configuration import configSafetyWrapper
from UML.data import BaseView


class LogCountAssertionError(AssertionError):
    pass

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
            if startCount + count != endCount:
                raise LogCountAssertionError
            return ret
        wrapped.__name__ = function.__name__
        wrapped.__doc__ = function.__doc__
        return wrapped
    return logCountAssertion

noLogEntryExpected = logCountAssertionFactory(0)
oneLogEntryExpected = logCountAssertionFactory(1)

class LazyNameGenerationAssertionError(AssertionError):
    pass

def assertNoNamesGenerated(obj):
    if obj.points._namesCreated() or obj.features._namesCreated():
        raise LazyNameGenerationAssertionError
