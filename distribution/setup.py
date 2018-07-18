from __future__ import absolute_import
from setuptools import setup, find_packages, Extension
from distutils.command.sdist import sdist as _sdist
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import sys
import os
import glob
import warnings

def getCExtensions():
    umlC = glob.glob(os.path.join('UML', 'helpers.c'))
    dataC = glob.glob(os.path.join('UML', 'data', '*.c'))
    allExtensions = umlC + dataC
    for extension in allExtensions:
        # name convention dir.subdir.filename
        name = ".".join(extension[:-2].split(os.path.sep))
        extensions.append(Extension(name, [extension], optional=True))
    return extensions

extensions = []
# --universal will build binary universal distribution w/ no C extensions
# command line: python setup.py bdist_wheel --universal
if '--universal' not in sys.argv:
    extensions.extend(getCExtensions())

# build source distribution with C extensions
# command line: python setup.py sdist
cmdclass = {}
# inspect module does not work with cython
# TODO remove one instance of inspect in UML/data/base.py
to_cythonize = [os.path.join('UML', 'helpers.py'),
                os.path.join('UML', 'data', '*.py'),]
exclude = [os.path.join('UML', 'data', '__init__.py'),
           os.path.join('UML', 'data', 'dataHelpers.py'), # bug in python2
           ]
# Make sure the compiled Cython files in the distribution are up-to-date
class sdist(_sdist):
    def run(self):
        from Cython.Build import cythonize, build_ext
        cythonize(to_cythonize, exclude=exclude,
                  compiler_directives={'always_allow_keywords': True},
                  exclude_failures=True,)
        extensions.extend(getCExtensions())
        _sdist.run(self)
cmdclass['sdist'] = sdist

# setup with C extensions
try:
    setup(
        name='UML',
        version='0.0.0.dev1',
        packages=find_packages(exclude=['*.tests',
                                        '*.examples*']),
        install_requires=['six>=1.5.1',
                          'future>=0.11.4',
                          'numpy>=1.10.4',
                         ],
        ext_modules = extensions,
        cmdclass = cmdclass,
        )

# setup without C extensions
except Exception:
    msg = "UML is being installed without C extensions. This does NOT affect "
    msg += "the functionality of UML, but this install will not benefit from "
    msg += "the speed increases of the C extensions."
    warnings.warn(msg)
    setup(
        name='UML',
        version='0.0.0.dev1',
        packages=find_packages(exclude=['*.tests',
                                        '*.examples*']),
        install_requires=['six>=1.5.1',
                          'future>=0.11.4',
                          'numpy>=1.10.4',
                         ],
        )

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
