"""
Build a UML distribution or install UML.

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

from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError

def getExtensions():
    umlC = glob.glob(os.path.join('UML', 'helpers.c'))
    dataC = glob.glob(os.path.join('UML', 'data', '*.c'))
    allExtensions = umlC + dataC
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
        to_cythonize = [os.path.join('UML', 'helpers.py'),
                        os.path.join('UML', 'data', '*.py'),]
        exclude = [os.path.join('UML', 'data', '__init__.py'),]
        cythonize(to_cythonize, exclude=exclude,
                  compiler_directives={'always_allow_keywords': True,
                                       'language_level': 3,
                                       'binding': True},
                  exclude_failures=True,)
        extensions = getExtensions()
    except ImportError:
        pass

# modified from Bob Ippolito's simplejson project
if sys.platform == 'win32' and sys.version_info < (2, 7):
   # 2.6's distutils.msvc9compiler can raise an IOError when failing to
   # find the compiler
   # It can also raise ValueError https://bugs.python.org/issue7511
   ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError,
                 IOError, ValueError)
else:
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
    setupKwargs['name'] = 'UML'
    setupKwargs['version'] = '0.0.0.dev1'
    setupKwargs['author'] = "Spark Wave"
    setupKwargs['author_email'] = "willfind@gmail.com"
    setupKwargs['description'] = "Universal Machine Learning"
    setupKwargs['url'] = "https://willfind.github.io/UML/"
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
    setupKwargs['install_requires'] = ['six>=1.5.1', 'numpy>=1.10.4']
    if extensions is not None:
        setupKwargs['ext_modules'] = extensions
        cmdclass = {'build_ext': _build_ext}
        setupKwargs['cmdclass'] = cmdclass

    setup(**setupKwargs)

if extensions:
    try:
        run_setup(extensions)
        print('*' * 79)
        print("Successfully built with C extensions.")
        print('*' * 79)
    except BuildFailed:
        run_setup()
        print('*' * 79)
        print("WARNING: Failed to compile C extensions. This does NOT affect ")
        print("the functionality of the build, but this build will not ")
        print("benefit from the speed increases of the C extensions.")
        print("Plain-Python build of UML succeeded.")
        print('*' * 79)
else:
    run_setup()
    print('*' * 79)
    print("WARNING: This build does not include the C extensions. ")
    print("This does NOT affect the functionality of the build, but it will ")
    print("not benefit from the speed increases of the C extensions.")
    print("Plain-Python build of UML succeeded.")
    print('*' * 79)

# TODO
    # determine which packages to exclude in distribution
    # determine correct versions for install_requires
    # make any changes to setup metadata (author, description, classifiers, etc.)
    # additional setup metadata (see below)
        # with open("README.md", "r") as fh:
        #     long_description = fh.read()

        # long_description=long_description,
        # long_description_content_type="text/markdown",
