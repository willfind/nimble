from __future__ import absolute_import

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

# command line: python setup.py sdist
    # builds source distribution with C extensions
# command line: python setup.py bdist_wheel
    # builds binary platform-dependent distribution with C extensions
# command line: python setup.py bdist_wheel --universal
    # builds binary universal distribution without C extensions

def getExtensions():
    umlC = glob.glob(os.path.join('UML', 'helpers.c'))
    dataC = glob.glob(os.path.join('UML', 'data', '*.c'))
    allExtensions = umlC + dataC
    for extension in allExtensions:
        # name convention dir.subdir.filename
        name = ".".join(extension[:-2].split(os.path.sep))
        extensions.append(Extension(name, [extension]))
    return extensions

cmdclass = {}
extensions = []
# universal build does not include extensions
if '--universal' not in sys.argv:
    extensions = getExtensions()
# Make sure the compiled Cython files in the distribution are up-to-date
if 'sdist' in sys.argv:
    # inspect module does not work with cython
    # TODO remove one instance of inspect in UML/data/base.py
    to_cythonize = [os.path.join('UML', 'helpers.py'),
                    os.path.join('UML', 'data', '*.py'),]
    exclude = [os.path.join('UML', 'data', '__init__.py'),
               os.path.join('UML', 'data', 'dataHelpers.py'), # bug in python2
               ]
    from Cython.Build import cythonize, build_ext
    cythonize(to_cythonize, exclude=exclude,
              compiler_directives={'always_allow_keywords': True},
              exclude_failures=True,)
    extensions = getExtensions()

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
    # This class allows C extension building to fail.

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

def run_setup(with_extensions):
    if with_extensions:
        ext_modules = extensions
        cmdclass['build_ext']= _build_ext

    setup(
        name='UML',
        version='0.0.0.dev1',
        packages=find_packages(exclude=['*.tests',
                                        '*.examples*']),
        install_requires=['six>=1.5.1',
                          'future>=0.11.4',
                          'numpy>=1.10.4',
                         ],
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        )

try:
    run_setup(True)
except BuildFailed:

    msg = "WARNING: UML failed to compile C extensions. This does NOT affect "
    msg += "the functionality of UML, but this install will not benefit from "
    msg += "the speed increases of the C extensions."

    print('*' * 75)
    print(msg)
    print("Failure information, if any, is above.")
    print("Retrying the build without the C extensions now.")
    print('*' * 75)

    run_setup(False)

    print('*' * 75)
    print(msg)
    print("Plain-Python install of UML succeeded.")
    print('*' * 75)


## TODO
    # determine which packages to exclude in distribution
    # determine correct versions for install_requires
    # additional metadata (see below)

# with open("README.md", "r") as fh:
#     long_description = fh.read()

# author="Example Author",
# author_email="author@example.com",
# description="A small example package",
# long_description=long_description,
# long_description_content_type="text/markdown",
# url="https://github.com/pypa/example-project",
# classifiers=(
#     'Development Status :: 3 - Alpha',
#     'Programming Language :: Python :: 2',
#     'Programming Language :: Python :: 2.7',
#     'Programming Language :: Python :: 3',
#     'Programming Language :: Python :: 3.4',
#     'Programming Language :: Python :: 3.5',
#     'Programming Language :: Python :: 3.6',
#     'Operating System :: OS Independent'
