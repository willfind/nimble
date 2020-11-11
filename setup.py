"""
Build a nimble distribution or install nimble.

build_ext - build extensions (requires Cython)
sdist - Build source distribution
bdist_wheel - binary distribution (if extensions are available builds
                                   platform wheel otherwise pure python)
bdist_conda - binary conda distribution (requires conda-build)
install - install nimble (not recommended as install method)
clean - remove setup generated directories and files. By default, clean
        only removes the "temporary" files in the build/ directory.
        Using the --all flag will fully remove the build/ and
        nimble.egg-info/ directores as well as any C extension files.

Note: the bdist_conda and bdist_wheel commands cannot be run together.
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

try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

# conda
try:
    import distutils.command.bdist_conda
    distclass = distutils.command.bdist_conda.CondaDistribution
except ImportError:
    distclass = Distribution

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
    to_cythonize = [os.path.join(corePath, 'create.py'),
                    os.path.join(corePath, '_createHelpers.py'),
                    os.path.join(corePath, 'learn.py'),
                    os.path.join(corePath, '_learnHelpers.py'),
                    os.path.join(corePath, 'data', '*.py'),
                    os.path.join('nimble', 'calculate', '*.py'),
                    os.path.join('nimble', 'random', '*.py'),
                    os.path.join('nimble', 'match', '*.py'),
                    os.path.join('nimble', 'fill', '*.py')]
    exclude = [os.path.join('nimble', '*', '__init__.py'),
               os.path.join(corePath, '*', '__init__.py')]
    cythonize(to_cythonize, exclude=exclude, force=True,
              compiler_directives={'always_allow_keywords': True,
                                   'language_level': 3,
                                   'binding': True},
              exclude_failures=True,)

class ExtensionsFailed(Exception):
    """
    Raised when any process related to the C extensions is unsuccessful.
    """


def commandUsesExtensions(commands):
    """
    Identify any command that should try to generate the C extensions.
    """
    for cmd in commands:
        if 'build' in cmd or 'dist' in cmd or 'install' in cmd:
            return True
    return False

class NimbleDistribution(distclass):
    """
    Extend the Distribution class to attempt to use Cython to generate
    the C extension files for certain command line commands.
    """
    def __init__(self, attrs=None):
        super().__init__(attrs)
        try:
            self.parse_command_line()
            if commandUsesExtensions(self.commands):
                if CYTHON_AVAILABLE:
                    cythonizeFiles()
                extensions = getExtensions()
                attrs['ext_modules'] = extensions
                # re-init with ext_modules
                super().__init__(attrs)
        except Exception:
            raise ExtensionsFailed()

class _build_ext(build_ext):
    """
    Allows C extension building to fail but python build to continue.
    """
    def run(self):
        try:
            build_ext.run(self)
        except Exception:
            raise ExtensionsFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except Exception:
            raise ExtensionsFailed()

class CleanCommand(clean):
    """
    Custom clean command to tidy up the project root.
    """
    def run(self):
        super().run()
        if self.all:
            if os.path.exists('nimble.egg-info'):
                remove_tree('nimble.egg-info', dry_run=self.dry_run)
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
    setupKwargs['name'] = 'nimble'
    setupKwargs['version'] = '0.0.0.dev1'
    setupKwargs['author'] = "Spark Wave"
    setupKwargs['author_email'] = "willfind@gmail.com"
    setupKwargs['description'] = "Interfaces and tools for data science."
    setupKwargs['url'] = "https://nimbledata.org"
    setupKwargs['packages'] = find_packages(exclude=('tests', 'tests.*'))
    setupKwargs['python_requires'] = '>=3.4'
    setupKwargs['classifiers'] = [
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
        ]

    setupKwargs['include_package_data'] = True
    setupKwargs['convert_2to3_doctests'] = []
    setupKwargs['install_requires'] = ['numpy>=1.10.4']
    # extras
    pandas = 'pandas>=0.20'
    scipy = 'scipy>=1.0'
    matplotlib = 'matplotlib>=3.0'
    cloudpickle = 'cloudpickle>=1.0'
    cython = 'cython>=0.29'
    mlpy = 'machine-learning-py>=3.5;python_version<"3.7"'
    scikitlearn = 'scikit-learn>=0.19'
    keras = 'keras>=2.0'
    autoimpute = 'autoimpute>=0.12'
    requests = 'requests>2.12'
    h5py = 'h5py>=2.10'
    nose = 'nose>=1.3'
    data = [pandas, scipy]
    interfaces = [mlpy, scikitlearn, keras, autoimpute]
    userAll = [matplotlib, cloudpickle, cython, requests, h5py]
    userAll.extend(data)
    userAll.extend(interfaces)
    setupKwargs['extras_require'] = {
        'all': userAll, 'data': data, 'interfaces': interfaces,
        'pandas': pandas, 'scipy': scipy, 'matplotlib': matplotlib,
        'cloudpickle': cloudpickle, 'cython': cython, 'requests': requests,
        'h5py': h5py, 'mlpy': mlpy,  'scikit-learn': scikitlearn,
        'keras': keras, 'autoimpute': autoimpute, 'nose': nose,
        }

    # TODO
    # determine best version requirements for install_requires, extras_require
    # make any changes to setup metadata (author, description, classifiers, etc.)
    # additional setup metadata (see below)
        # with open("README.md", "r") as fh:
        #     long_description = fh.read()

        # long_description=long_description,
        # long_description_content_type="text/markdown",
    # discuss versioning style (semantic versioning recommended)

    setupKwargs['cmdclass'] = {'clean': CleanCommand,
                               'build_ext': _build_ext}
    try:
        # our Distribution class that builds and C includes extensions
        setupKwargs['distclass'] = NimbleDistribution
        dist = setup(**setupKwargs)
    except ExtensionsFailed:
        # The default Distribution without extensions added
        setupKwargs['distclass'] = distclass
        dist = setup(**setupKwargs)

    # print if able to build nimble with extensions
    if commandUsesExtensions(dist.commands):
        if dist.has_ext_modules():
            msg = 'Nimble built successfully with C extensions.'
        else:
            msg = 'Nimble built successfully in pure python. The '
            msg += "functionality is the same but will not benefit speed "
            msg += 'increases of the C extensions. '
            if CYTHON_AVAILABLE:
                msg += 'Cython was available to generate the extension files '
                msg += 'but there was an error when building the extensions.'
            else:
                msg += 'If wanting to generate the extension files, install '
                msg += 'Cython.'

        lineFormat = '* {:74s} *\n'
        out = '*' * 78 + '\n'
        for text in textwrap.wrap(msg, 74):
            out += lineFormat.format(text)
        out += '*' * 78
        print(out)

if __name__ == '__main__':
    with EmptyConfigFile():
        run_setup()
