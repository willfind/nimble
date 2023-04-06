"""
Settings for running test suite.
"""

import os
import tempfile
import configparser
import copy
import importlib
import inspect
import shutil

import pytest

import nimble
from nimble.core.configuration import SessionConfiguration
from nimble.core._learnHelpers import initAvailablePredefinedInterfaces


TEMPDIRLOG = tempfile.TemporaryDirectory()
DUMMYCONFIG = {
    'logger': {'location': TEMPDIRLOG.name,
               'name': "tmpLogs",
               'enabledByDefault': "False",
               'enableDeepLogging': "False"},
    'fetch': {'location': TEMPDIRLOG.name}
    }


@pytest.fixture(autouse=True)
def addNimble(doctest_namespace):
    """
    Adds nimble to doctest namespace so import is not required in the file.
    """
    doctest_namespace["nimble"] = nimble

@pytest.fixture
def tmpDataToFileFixture():
    """
    Used for doctests which write out a file in the cwd. This will change
    the cwd temporarily so that this doctest will not polute the filesystem
    """
    backupCWD = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdirForCSV:
        try:
            os.chdir(tmpdirForCSV)
            yield
        finally:
            os.chdir(backupCWD)


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

    def reset(self, backingDict, changes):
        """ Reuse the same object for each test by just changing the state"""
        self.parser.read_dict(backingDict)
        self.dictionary = backingDict
        self.changes = changes


class OverrideSettings:
    """
    Replace nimble.settings with DictSessionConfig using default testing
    settings. This avoids any need to interact with the config file
    which is a benefit to both safety and speed.

    This is done via the override method of a stateful object so that
    the sessionstart function can set the original backup values that will be
    applied every test.
    """
    def __init__(self):
        self.backupChanges = {}
        self.backupInterfaces = {}
        self.backupConfig = None

    def override(self):
        """ Execute the per test settings override """
        # copying interfaces does not copy registeredLearners attribute and
        # deepcopy cannot be used so we reset it to have no custom learners
        nimble.core.interfaces.available = copy.copy(self.backupInterfaces)
        self.backupInterfaces['custom'].registeredLearners = {}

        def loadSavedSettings():
            baseDict = copy.deepcopy(DUMMYCONFIG)
            self.backupConfig.reset(baseDict, self.backupChanges)
            return self.backupConfig

        nimble.settings = loadSavedSettings()
        # include changes made during call to begin()
        nimble.settings.changes = copy.deepcopy(self.backupChanges)
        # for tests that load settings during the test
        nimble.core.configuration.loadSettings = loadSavedSettings
        # setup logger to use new settings and set logging hooks in settings
        nimble.core.logger.initLoggerAndLogConfig()

overrideObj = OverrideSettings()

def pytest_sessionstart():
    """
    Setup prior to running any tests.
    """
    # Only proceed if we are NOT using an installed version of nimble,
    # which we take to mean that it either cannot be found, or isn't
    # found in a standard install location (in a 'site-packages' subfolder)
    nimbleSpec = importlib.util.find_spec("nimble")
    if nimbleSpec is None or 'site-packages' not in nimbleSpec.origin:
        currPath = os.path.abspath(inspect.getfile(inspect.currentframe()))
        projectPath = os.path.dirname(currPath)
        target = os.path.join(projectPath, "pyproject.toml")
        destination = os.path.join(projectPath, "nimble", "pyproject.toml")
        shutil.copy(target, destination)

    # Predefined interfaces were previously loaded on nimble import but
    # are now loaded as requested. Some tests operate under the assumption
    # that all these interfaces have already been loaded, but since that
    # is no longer the case we need to load them now to ensure that those
    # tests continue to test all interfaces.
    initAvailablePredefinedInterfaces()

    # For use to standardize interface state per test
    overrideObj.backupInterfaces = nimble.core.interfaces.available
    # loading interfaces adds changes (interface options) to settings
    overrideObj.backupChanges = nimble.settings.changes

    # setup our initial non-file backed settings object.
    # This allows override to simply reset this object
    overrideObj.backupConfig = DictSessionConfig(copy.deepcopy(DUMMYCONFIG))

def pytest_runtest_setup():
    """
    Ensure nimble is in the same state at the start of each test.
    """
    overrideObj.override()

def pytest_sessionfinish():
    """
    Cleanup after all tests have completed.
    """
    TEMPDIRLOG.cleanup()

    currPath = os.path.abspath(inspect.getfile(inspect.currentframe()))
    projectPath = os.path.dirname(currPath)
    destination = os.path.join(projectPath, "nimble", "pyproject.toml")
    if os.path.exists(destination):
        os.remove(destination)
