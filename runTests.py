#!/usr/bin/python

"""
Script acting as the canonical means to run the test suite for the entirety of
UML. Run as main to execute.

"""

from __future__ import absolute_import
import warnings
import inspect
import os
import os.path
import nose
import sys
from nose.plugins.base import Plugin

import nose.pyversion
#import pdb
#pdb.set_trace()
#print dir(nose.pyversion)
from nose.util import ln
try:
    from StringIO import StringIO#python 2
except:
    from six import StringIO#python 3

UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(UMLPath))
import UML

available = UML.interfaces.available


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
        # TODO fix selection of files in interfaces/tests
        if not file.endswith('.py'):
            return False

        if file == '__init__.py' and sys.version_info.major > 2:#in python3, don't check __init__.py
            return False

        dname = os.path.dirname(file)
        if dname == os.path.join(UMLPath, 'interfaces', 'tests'):
            fname = os.path.basename(file)

            #need to confirm that fname is associated with an interface
            associated = False
            for intName in os.listdir(dname):
                if not intName.endswith('.py'):
                    continue
                intName = intName.split('.')[0]
                if intName in fname:
                    associated = True
                    break
                # if it is associated with an interface, we only want to run it
            # if it is associated with an available interface.
            if associated:
                associated = False
                for interface in UML.interfaces.available:
                    if interface.__module__.rsplit('.', 1)[1] in fname:
                        associated = True
                        break
                if not associated:
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


class LoggerControl(object):
    def __enter__(self):
        self._backupLoc = UML.settings.get('logger', 'location')
        self._backupName = UML.settings.get('logger', 'name')

        # delete previous testing logs:
        location = os.path.join(UMLPath, 'logs-UML')
        hrPath = os.path.join(location, 'log-UML-unitTests.txt')
        mrPath = os.path.join(location, 'log-UML-unitTests.mr')
        if os.path.exists(hrPath):
            os.remove(hrPath)
        if os.path.exists(mrPath):
            os.remove(mrPath)

        # change name of log file (settings hook will init new log
        # files after .set())
        UML.settings.set('logger', 'location', location)
        UML.settings.set("logger", 'name', 'log-UML-unitTests')
        UML.settings.saveChanges("logger")

    def __exit__(self, type, value, traceback):
        UML.settings.set("logger", 'location', self._backupLoc)
        UML.settings.saveChanges("logger", 'location')
        UML.settings.set("logger", 'name', self._backupName)
        UML.settings.saveChanges("logger", 'name')


if __name__ == '__main__':
    # any args passed to this script will be passed down into nose
    args = sys.argv

    # TODO: check for -w and override???

    # setup so that this only tests the UML dir, regardless of where it has been called
    workingDirDef = ["-w", UMLPath]
    args.extend(workingDirDef)

    # Set options so that the logger outputs to a different file than
    # during the course of normal operations
    with LoggerControl():
        # suppress all warnings -- nosetests only captures std out, not stderr,
        # and there are some tests that call learners in unfortunate ways, causing
        # ALOT of annoying warnings.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # nose.run(addplugins=[ExtensionPlugin(), CaptureError()], argv=args, defaultTest='interfaces/tests/mlpy_interface_test.py')
            nose.run(addplugins=[ExtensionPlugin(), CaptureError()], argv=args)
    exit(0)
