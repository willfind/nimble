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
import glob
import textwrap

from setuptools import setup, find_packages, Distribution
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils import log
from distutils.command.clean import clean
from distutils.dir_util import remove_tree

from Cython.Build import cythonize

def getCFiles():
    return glob.glob(os.path.join('nimble', '**', '*.c'), recursive=True)

def getExtensions():
    extensions = []
    for extension in getCFiles():
        # name convention dir.subdir.filename
        name = ".".join(extension[:-2].split(os.path.sep))
        extensions.append(Extension(name, [extension]))
    return extensions

def cythonizeFiles():
    corePath = os.path.join('nimble', 'core')
    to_cythonize = [os.path.join('nimble', 'calculate', '*.py'),
                    os.path.join(corePath, 'data', '*.py'),
                    os.path.join(corePath, 'interfaces', '*.py'),
                    os.path.join(corePath, 'logger', '*.py'),
                    os.path.join(corePath, '*.py'),
                    os.path.join('nimble', 'fill', '*.py'),
                    os.path.join('nimble', 'learners', '*.py'),
                    os.path.join('nimble', 'match', '*.py'),
                    os.path.join('nimble', 'random', '*.py'),
                    os.path.join('nimble', '*.py')
                   ]
    exclude = []
    cythonize(to_cythonize, exclude=exclude, force=True,
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


class _build_ext(build_ext):
    """
    Allows C extension building to fail but python build to continue.
    """
    def run(self):
        build_ext.run(self)

    def build_extension(self, ext):
        build_ext.build_extension(self, ext)


class CleanCommand(clean):
    """
    Custom clean command to tidy up the project root.
    """
    def run(self):
        super().run()
        if self.all:
            for directory in ['build', 'nimble.egg-info']:
                if os.path.exists(directory):
                    remove_tree(directory, dry_run=self.dry_run)
            for f in getCFiles():
                if not self.dry_run:
                    os.remove(f)
                log.info('removing %s', os.path.relpath(f))


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


def run_setup():
    setupKwargs = {}

    setupKwargs['cmdclass'] = {'clean': CleanCommand,
                               'build_ext': _build_ext}
    # our Distribution class that builds and C includes extensions
    setupKwargs['distclass'] = NimbleDistribution
    dist = setup(**setupKwargs)


if __name__ == '__main__':
    with EmptyConfigFile():
        run_setup()
