#!/usr/bin/python

"""
Script acting as the canonical means to run the test suite for the entirety of
UML. Run as main to execute.

"""

import inspect
import os
import nose
import sys
from nose.plugins.base import Plugin

class ExtensionPlugin(Plugin):

    name = "ExtensionPlugin"

    def options(self, parser, env):
        Plugin.options(self,parser,env)

    def configure(self, options, config):
        Plugin.configure(self, options, config)
        self.enabled = True

    # Controls which files are checked for tests. In this case, we check every
    # file we discover, as long as it ends with '.py'
    def wantFile(self, file):
        return file.endswith('.py')

    def wantDirectory(self,directory):
        return True

    def wantModule(self,file):
        return True


if __name__ == '__main__':
    # any args passed to this script will be passed down into nose
    args = sys.argv

    # TODO: check for -w and override???

    # setup so that this only tests the UML dir, regardless of where it has been called
    UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    workingDirDef = ["-w", UMLPath]
    args.extend(workingDirDef)

    nose.run(addplugins=[ExtensionPlugin()], argv=args)
    exit(0)

