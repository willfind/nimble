"""
Build a nimble distribution or install nimble.

Build source distribution with C extensions
    command line: python setup.py sdist
Build binary platform-dependent distribution with C extensions
    command line: python setup.py bdist_wheel
Build binary universal distribution which does not include C extensions
    command line: python setup.py bdist_wheel --universal

Install via setup.py:
    command line: python setup.py install

    If cython is installed, it will install with C extensions. If cython
    is not present or compiling the files fails, a pure python version
    will be installed.
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import glob
import warnings

from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist

from distutils.errors import (CCompilerError, DistutilsExecError,
                              DistutilsPlatformError)

def getExtensions():
    nimbleC = glob.glob(os.path.join('nimble', '*.c'))
    dataC = glob.glob(os.path.join('nimble', '*', '*.c'))
    allExtensions = nimbleC + dataC
    for extension in allExtensions:
        # name convention dir.subdir.filename
        name = ".".join(extension[:-2].split(os.path.sep))
        extensions.append(Extension(name, [extension]))
    return extensions

extensions = []
# universal build does not include extensions
if '--universal' not in sys.argv:
    extensions = getExtensions()
    # Make sure the compiled Cython files in the distribution are up-to-date
    try:
        from Cython.Build import cythonize, build_ext
        to_cythonize = [os.path.join('nimble', 'core.py'),
                        os.path.join('nimble', 'helpers.py'),
                        os.path.join('nimble', 'data', '*.py'),
                        os.path.join('nimble', 'calculate', '*.py'),
                        os.path.join('nimble', 'match', '*.py'),
                        os.path.join('nimble', 'fill', '*.py')]
        exclude = [os.path.join('nimble', '*', '__init__.py')]
        cythonize(to_cythonize, exclude=exclude,
                  compiler_directives={'always_allow_keywords': True,
                                       'language_level': 3,
                                       'binding': True},
                  exclude_failures=True,)
        extensions = getExtensions()
    except ImportError:
        pass

ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)

class BuildFailed(Exception):
    pass

class _build_ext(build_ext):
    """
    Allows C extension building to fail but python build to continue.
    """

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors:
            raise BuildFailed()

def run_setup(extensions=None):
    setupKwargs = {}
    setupKwargs['name'] = 'nimble'
    setupKwargs['version'] = '0.0.0.dev1'
    setupKwargs['author'] = "Spark Wave"
    setupKwargs['author_email'] = "willfind@gmail.com"
    setupKwargs['description'] = "Interfaces and tools for data science."
    setupKwargs['url'] = "https://nimbledata.org"
    setupKwargs['packages'] = find_packages(exclude=('tests', 'tests.*'))
    setupKwargs['classifiers'] = (
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        )
    setupKwargs['include_package_data'] = True
    setupKwargs['install_requires'] = ['six>=1.5.1', 'numpy>=1.10.4',]
    # extras
    pandas = 'pandas>=0.20'
    scipy = 'scipy>=1.0'
    matplotlib = 'matplotlib>=3.0'
    cloudpickle = 'cloudpickle>=1.0'
    cython = 'cython>=0.29'
    mlpy = 'machine-learning-py>=3.5'
    scikitlearn = 'scikit-learn>=0.19'
    keras = 'keras>=2.0'
    nose = 'nose>=1.3'
    requests = 'requests>2.12'
    interfaces = [mlpy, scikitlearn, keras]
    all = [pandas, scipy, matplotlib, cloudpickle, cython] + interfaces

    setupKwargs['extras_require'] = {
        'all': all, 'interfaces': interfaces, 'pandas': pandas, 'scipy': scipy,
        'matplotlib': matplotlib, 'cloudpickle': cloudpickle, 'cython': cython,
        'mlpy': mlpy, 'scikit-learn': scikitlearn, 'keras': keras,
        'nose': nose, 'requests': requests,
        }
    if extensions is not None:
        setupKwargs['ext_modules'] = extensions
        cmdclass = {'build_ext': _build_ext}
        setupKwargs['cmdclass'] = cmdclass

    setup(**setupKwargs)


def plainPythonMessage():
    print('*' * 66)
    print("* WARNING: failed to build nimble with C extensions.             *")
    print("* This does NOT affect the functionality of the build, but it    *")
    print("* will not benefit from the speed increases of the C extensions. *")
    print('*' * 66)
    print("* nimble build in plain Python successful.                       *")
    print('*' * 66)


if extensions:
    try:
        run_setup(extensions)
        print('*' * 46)
        print("* nimble build with C extensions successful. *")
        print('*' * 46)
    except BuildFailed:
        run_setup()
        plainPythonMessage()
else:
    run_setup()
    plainPythonMessage()

# TODO
# determine best version requiremnts for install_requires, extras_require
# make any changes to setup metadata (author, description, classifiers, etc.)
# additional setup metadata (see below)
    # with open("README.md", "r") as fh:
    #     long_description = fh.read()

    # long_description=long_description,
    # long_description_content_type="text/markdown",
