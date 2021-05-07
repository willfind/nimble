#!/usr/bin/python

"""
Script acting as the canonical means to run the test suite for the entirety of
nimble. Run as main to execute.
"""

import warnings
import inspect
import os
import sys
import tempfile
from io import StringIO#python 3
import logging
import configparser
import copy

import nose
from nose.plugins.base import Plugin
import nose.pyversion
from nose.util import ln

import nimble
from nimble.core.configuration import SessionConfiguration
from nimble.core._learnHelpers import initAvailablePredefinedInterfaces

currPath = os.path.abspath(inspect.getfile(inspect.currentframe()))
nimblePath = os.path.dirname(currPath)
sys.path.append(os.path.dirname(nimblePath))

# Prevent logging of these modules cluttering nose output
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class ExtensionPlugin(Plugin):
    name = "ExtensionPlugin"

    def options(self, parser, env):
        Plugin.options(self, parser, env)

    def configure(self, options, config):
        Plugin.configure(self, options, config)
        self.enabled = True

    # Controls which files are checked for tests. In this case, we check every
    # file we discover, as long as it ends with '.py'
    def wantFile(self, file):
        # TODO fix selection of files in interfaces/tests?
        if not file.endswith('.py'):
            return False

        return True

    def wantDirectory(self, directory):
        if os.path.basename(directory) in ['broken', 'documentation']:
            return False
        return True

    def wantModule(self, file):
        return True


class CaptureError(Plugin):
    """
    Error output capture plugin. This plugin captures stderr during test
    execution, appending any output captured to the error or failure output,
    should the test fail or raise an error.

    Modified from nose's builtin 'Capture' plugin for capturing stdout
    """
    enabled = True
    env_opt = 'NOSE_NOCAPTUREERROR'
    name = 'CaptureError'
    score = 1600

    def __init__(self):
        self.stderr = []
        self._buf = None
        super(CaptureError, self).__init__()

    def options(self, parser, env):
        """Register commandline options"""
        parser.add_option(
            "--nocaptureerror", action="store_false",
            default=not env.get(self.env_opt), dest="captureerror",
            help="Don't capture stderr (any stderr output "
                 "will be printed immediately) [NOSE_NOCAPTURE]")

    def configure(self, options, conf):
        """Configure plugin. Plugin is enabled by default."""
        self.conf = conf
        if not options.captureerror:
            self.enabled = False

    def afterTest(self, test):
        """Clear capture buffer. """
        self.end()
        self._buf = None

    def begin(self):
        """Replace sys.stderr with capture buffer."""
        self.start() # get an early handle on sys.stderr

    def beforeTest(self, test):
        """Flush capture buffer."""
        self.start()

    def formatError(self, test, err):
        """Add captured output to error report."""
        test.capturedOutput = output = self.buffer
        self._buf = None
        if not output:
            # Don't return None as that will prevent other
            # formatters from formatting and remove earlier formatters
            # formats, instead return the err we got
            return err
        ec, ev, tb = err
        return (ec, self.addCaptureToErr(ev, output), tb)

    def formatFailure(self, test, err):
        """Add captured output to failure report."""
        return self.formatError(test, err)

    def addCaptureToErr(self, ev, output):
        return u'\n'.join([str(ev), ln(u'>> begin captured stderr <<'),
                           str(output), ln(u'>> end captured stderr <<')])

    def start(self):
        self.stderr.append(sys.stderr)
        self._buf = StringIO()
        sys.stderr = self._buf

    def end(self):
        if self.stderr:
            sys.stderr = self.stderr.pop()

    def finalize(self, result):
        """Restore stderr."""
        while self.stderr:
            self.end()

    def _get_buffer(self):
        if self._buf is not None:
            return self._buf.getvalue()

    buffer = property(_get_buffer, None, None,
                      """Captured stderr output.""")


class FileRemover(Plugin):
    """
    Remove specified files generated during testing.
    """
    name = "FileRemover"

    def __init__(self):
        super(FileRemover, self).__init__()
        # nimblePath is working directory
        self.deleteFiles = ['simpleData.csv']

    def options(self, parser, env):
        Plugin.options(self, parser, env)

    def configure(self, options, config):
        Plugin.configure(self, options, config)
        self.enabled = True

    def finalize(self, result):
        for fileName in self.deleteFiles:
            fullPath = os.path.join(nimblePath, fileName)
            if os.path.exists(fullPath):
                os.remove(fullPath)

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

class ConfigPlugin(Plugin):
    """
    Setup configuration settings for testing and restore settings before
    each test.
    """
    def options(self, parser, env):
        super().options(parser, env)

    def configure(self, options, config):
        super().configure(options, config)
        self.enabled = True

    def overrideSettings(self):
        """
        Replace nimble.settings with DictSessionConfig using default testing
        settings. This avoids any need to interact with the config file.
        """
        currSettings = copy.deepcopy(self.configuration)

        def loadSavedSettings():
            return DictSessionConfig(currSettings)

        nimble.settings = loadSavedSettings()
        # include changes made during call to begin()
        nimble.settings.changes = copy.deepcopy(self.changes)
        # for tests that load settings during the test
        nimble.core.configuration.loadSettings = loadSavedSettings
        # setup logger to use new settings and set logging hooks in settings
        nimble.core.logger.initLoggerAndLogConfig()

    def begin(self):
        """
        Setup so each test can start in the same state.
        """
        self.tempdir = tempfile.TemporaryDirectory()
        self.configuration = {
            'logger': {'location': self.tempdir.name,
                       'name': "tmpLogs",
                       'enabledByDefault': "False",
                       'enableCrossValidationDeepLogging': "False"},
            'fetch': {'location': self.tempdir.name}
            }
        # Predefined interfaces were previously loaded on nimble import but
        # are now loaded as requested. Some tests operate under the assumption
        # that all these interfaces have already been loaded, but since that
        # is no longer the case we need to load them now to ensure that those
        # tests continue to test all interfaces.
        initAvailablePredefinedInterfaces()
        self.interfaces = nimble.core.interfaces.available
        # loading interfaces adds changes (interface options) to settings
        self.changes = nimble.settings.changes

        self.overrideSettings()

    def beforeTest(self, test):
        """
        Ensure nimble is in the same state at the start of each test.
        """
        # copying interfaces does not copy registeredLearners attribute and
        # deepcopy cannot be used so we reset it to have no custom learners
        self.interfaces['custom'].registeredLearners = {}
        nimble.core.interfaces.available = copy.copy(self.interfaces)

        self.overrideSettings()

    def finalize(self, result):
        self.tempdir.cleanup()

if __name__ == '__main__':
    # any args passed to this script will be passed down into nose
    args = sys.argv

    # TODO: check for -w and override???

    # setup so that this only tests the nimble dir, regardless of where it has
    # been called and doctest is always run
    workingDirDef = ["-w", nimblePath, '--with-doctest']
    args.extend(workingDirDef)

    plugins = [ExtensionPlugin(), CaptureError(), FileRemover(),
               ConfigPlugin()]
    # suppress all warnings -- nosetests only captures std out, not stderr,
    # and there are some tests that call learners in unfortunate ways,
    # causing ALOT of annoying warnings.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # need to turn on warnings for tests/interfaces/universal_test
        warnings.filterwarnings('always', module=r'.*universal_test')
        warnings.filterwarnings('always', module=r'.*keras_interface')
        nose.run(addplugins=plugins, argv=args)
    exit(0)
