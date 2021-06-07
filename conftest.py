"""
Settings for running test suite.
"""

import os
import tempfile
import configparser
import copy
import inspect
import sys

import nimble
from nimble.core.configuration import SessionConfiguration
from nimble.core._learnHelpers import initAvailablePredefinedInterfaces

currPath = os.path.abspath(inspect.getfile(inspect.currentframe()))
nimblePath = os.path.dirname(currPath)
sys.path.append(os.path.dirname(nimblePath))

tempdir = tempfile.TemporaryDirectory()
configuration = {
    'logger': {'location': tempdir.name,
               'name': "tmpLogs",
               'enabledByDefault': "False",
               'enableCrossValidationDeepLogging': "False"},
    'fetch': {'location': tempdir.name}
    }
# Predefined interfaces were previously loaded on nimble import but
# are now loaded as requested. Some tests operate under the assumption
# that all these interfaces have already been loaded, but since that
# is no longer the case we need to load them now to ensure that those
# tests continue to test all interfaces.
initAvailablePredefinedInterfaces()
interfaces = nimble.core.interfaces.available
# loading interfaces adds changes (interface options) to settings
changes = nimble.settings.changes

class DictSessionConfig(SessionConfiguration):
    """
    Use a dictionary instead of configuration file for config settings.
    """
    def __init__(self, dictionary):
        self.parser = configparser.ConfigParser()
        # Needs to be set if you want option names to be case sensitive
        self.parser.optionxform = str
        self.parser.read_dict(dictionary)
        self.dictionary = dictionary

        # dict of section name to dict of option name to value
        self.changes = {}
        self.hooks = {}

    def saveChanges(self, section=None, option=None):
        if section is not None:
            sectionInChanges = section in self.changes
            if sectionInChanges and section not in self.dictionary:
                self.dictionary[section] = {}
                self.parser.add_section(section)
            if sectionInChanges and option is not None:
                newOption = self.changes[section][option]
                del self.changes[section][option]
                self.dictionary[section][option] = newOption
                self.parser.set(section, option, newOption)
            elif sectionInChanges:
                self.dictionary[section].update(self.changes[section])
                for opt, value in self.changes[section].items():
                    self.parser.set(section, opt, value)
                del self.changes[section]
        else:
            self.dictionary.update(self.changes)
            for sect in self.changes:
                if not self.parser.has_section(sect):
                    self.parser.add_section(sect)
                for opt, value in self.changes[sect].items():
                    self.parser.set(sect, opt, value)
            self.changes = {}

def overrideSettings():
    """
    Replace nimble.settings with DictSessionConfig using default testing
    settings. This avoids any need to interact with the config file.
    """
    currSettings = copy.deepcopy(configuration)

    def loadSavedSettings():
        return DictSessionConfig(currSettings)

    nimble.settings = loadSavedSettings()
    # include changes made during call to begin()
    nimble.settings.changes = copy.deepcopy(changes)
    # for tests that load settings during the test
    nimble.core.configuration.loadSettings = loadSavedSettings
    # setup logger to use new settings and set logging hooks in settings
    nimble.core.logger.initLoggerAndLogConfig()

def pytest_sessionstart():
    """
    Setup prior to running any tests.
    """
    overrideSettings()

def pytest_runtest_setup():
    """
    Ensure nimble is in the same state at the start of each test.
    """
    # copying interfaces does not copy registeredLearners attribute and
    # deepcopy cannot be used so we reset it to have no custom learners
    interfaces['custom'].registeredLearners = {}
    nimble.core.interfaces.available = copy.copy(interfaces)

    overrideSettings()

def pytest_sessionfinish():
    """
    Cleanup after all tests have completed.
    """
    tempdir.cleanup()

    deleteFiles = ['simpleData.csv']

    for fileName in deleteFiles:
        fullPath = os.path.join(nimblePath, fileName)
        if os.path.exists(fullPath):
            os.remove(fullPath)
