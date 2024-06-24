
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Build a nimble distribution or install nimble.

build_ext - build extensions (requires Cython)
sdist - Build source distribution
bdist_wheel - binary distribution (if extensions are available builds
                                   platform wheel otherwise pure python)
install - install nimble (not recommended as install method)
clean - remove setup generated directories and files. By default, clean
        only removes the "temporary" files in the build/ directory.
        Using the --all flag will fully remove the build/ and
        nimble.egg-info/ directores as well as any C extension files.
"""

import os
import os.path
import glob

from setuptools import setup, Distribution
from setuptools.extension import Extension

# If available, import
try:
    from Cython.Build import cythonize
except ModuleNotFoundError:
    cyythonize = None

def getCFiles():
    """
    return list of all C files within the nimble module
    """
    return glob.glob(os.path.join('nimble', '**', '*.c'), recursive=True)

def getExtensions():
    """
    After C files have been generated, make list of Extension classes for each
    """
    extensions = []
    for extension in getCFiles():
        # name convention dir.subdir.filename
        name = ".".join(extension[:-2].split(os.path.sep))
        extensions.append(Extension(name, [extension], optional=True))
    return extensions

def cythonizeFiles():
    """
    Use cython to convert to C sources
    """
    if not cythonize:
        return

    toCythonize = glob.glob(os.path.join('nimble', '**', '*.py'), recursive=True)
    alreadyDone = glob.glob(os.path.join('nimble', '**', '*.c'), recursive=True)

    # depending on where/how this was build, the files may already be present.
    # if so, add them to a skip list
    exclude = []
    for cFile in alreadyDone:
        raw = os.path.splitext(cFile)
        exclude.append(raw[0] + ".py")

    cythonize(toCythonize, exclude=exclude, force=True,
              compiler_directives={'always_allow_keywords': True,
                                   'language_level': 3,
                                   'binding': True},
              exclude_failures=True,)


def commandUsesExtensions(commands):
    """
    Identify any command that should try to generate the C extensions.
    """
    for cmd in commands:
        if 'build' in cmd or 'dist' in cmd or 'install' in cmd:
            return True
    return False

class NimbleDistribution(Distribution):
    """
    Extend the Distribution class to attempt to use Cython to generate
    the C extension files for certain command line commands.
    """
    def __init__(self, attrs=None):
        super().__init__(attrs)
        self.parse_command_line()
        if commandUsesExtensions(self.commands):
            cythonizeFiles()
            extensions = getExtensions()
            attrs['ext_modules'] = extensions
            # re-init with ext_modules
            super().__init__(attrs)

class EmptyConfigFile:
    """
    Provide empty configuration.ini file for the distribution.

    Including an empty configuration.ini file ensures that uninstalling
    the package will remove this file as well. If a configuration file
    already exists, the contents will be stored upon entering and
    rewritten back to the file on exit.
    """

    def __init__(self):
        self.target = os.path.join('nimble', 'configuration.ini')
        self.exists = os.path.exists(self.target)
        self.configLines = []

    def __enter__(self):
        if self.exists:
            with open(self.target, 'r') as config:
                self.configLines = config.readlines()
        with open(self.target, 'w'):
            pass # empty file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.exists:
            with open(self.target, 'w') as config:
                for line in self.configLines:
                    config.write(line)

if __name__ == '__main__':
    with EmptyConfigFile():
        setupKwargs = {}
        setupKwargs['distclass'] = NimbleDistribution
        setup(**setupKwargs)
