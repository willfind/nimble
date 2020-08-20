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

import nose
from nose.plugins.base import Plugin
import nose.pyversion
from nose.util import ln

import nimble

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
        if os.path.basename(directory) == 'broken':
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


class LoggerControl(object):
    """
    Set logging configuration state while testing.

    Restores the original state upon completion.
    """
    def __init__(self):
        self._backupLoc = nimble.settings.get('logger', 'location')
        self._backupName = nimble.settings.get('logger', 'name')
        self._backupEnabled = nimble.settings.get('logger', 'enabledByDefault')
        self._crossValBackupEnabled = nimble.settings.get(
            'logger', 'enableCrossValidationDeepLogging')
        self.logDir = tempfile.TemporaryDirectory()

    def __enter__(self):
        # change name of log file (settings hook will init new log
        # files after .set())
        nimble.settings.set('logger', 'location', self.logDir.name)
        nimble.settings.set("logger", 'name', "tmpLogs")
        nimble.settings.set("logger", "enabledByDefault", "False")
        nimble.settings.set("logger", "enableCrossValidationDeepLogging",
                            "False")
        nimble.settings.saveChanges("logger")

    def __exit__(self, type, value, traceback):
        nimble.settings.set("logger", 'location', self._backupLoc)
        nimble.settings.set("logger", 'name', self._backupName)
        nimble.settings.set("logger", "enabledByDefault", self._backupEnabled)
        nimble.settings.set("logger", "enableCrossValidationDeepLogging",
                            self._crossValBackupEnabled)
        nimble.settings.saveChanges("logger")
        self.logDir.cleanup()

if __name__ == '__main__':
    # any args passed to this script will be passed down into nose
    args = sys.argv

    # TODO: check for -w and override???

    # setup so that this only tests the nimble dir, regardless of where it has
    # been called and doctest is always run
    workingDirDef = ["-w", nimblePath, '--with-doctest']
    args.extend(workingDirDef)

    plugins = [ExtensionPlugin(), CaptureError(), FileRemover()]
    with LoggerControl():
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
