
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
Store dependency versions requirements.

Also includes helpers for validating optional dependency versions at
runtime.
"""
import importlib.metadata
import tomli
import os
import inspect

from packaging.requirements import Requirement
from packaging.version import Version

from nimble.exceptions import PackageException

def _getNimbleMetadata():
    checker = importlib.metadata.requires
    try:
        infoFull = checker("nimble")
        # each entry in list is in the format of "package req,req; extra"
        # we only care about the package and requirement part.
        info = map(lambda x: x.split(";")[0], infoFull)
    except ModuleNotFoundError:
        # If standard metadata grabbing for installed packages fails,
        # we fall back on the local copy of pyproject.toml; which is
        # either bundled into the install as part of the build process
        # or moved into the nimble directory if doing local testing
        currPath = os.path.abspath(inspect.getfile(inspect.currentframe()))
        nimbleDir = os.path.dirname(currPath)
        pyprojPath = os.path.join(nimbleDir, "pyproject.toml")

        # If that doesn't work, we're likely in a local interactive session
        # instead of a testing session, so the file will be in it's standard
        # location in the git repo.
        if not os.path.exists(pyprojPath):
            nimParentDir = os.path.dirname(nimbleDir)
            pyprojPath = os.path.join(nimParentDir, "pyproject.toml")

        with open(pyprojPath, "rb") as f:
            tomlDict = tomli.load(f)
            infoFull = tomlDict["project"]

        # each entry in the dict maps to a section name in pyproject.
        # We need to pull from two sub entries: 'optional-dependencies' is
        # a dict with maps the extra name to a list of package requirements,
        # and 'dependencies' which is a list of requirements.
        optional = infoFull['optional-dependencies']['all']
        always = infoFull['dependencies']
        info = always + optional

    return info

def _getRequirementText(name):
    """
    Get version requirements of the given package's name.

    Backend depends on the execution environment and may differ
    depending on the version of python and whether nimble is installed
    or was found via a relative path (aka the case of local testing).
    """
    info = _getNimbleMetadata()
    relevant = [entry for entry in info if entry.startswith(name)]
    return relevant[0]

class Dependency:
    """
    Store details of a Nimble dependency.

    Parameters
    ----------
    name : str
        The name of the package as it is imported.
    section : str
        The category for setup.py to classify this dependency.
    description : str, None
        A description of the package's function within Nimble.
    reqName : str, None
        The name as given in requirements recording; assumed to be the
        same as name if None
    """
    # each dependency must be categorized into a section for setup.py
    # each section maps to a boolean value for whether it requires a
    # descriptions to be used with nimble.showAvailablePackages
    _sections = {'required': False, 'data': True, 'operation': True,
                'interfaces': True, 'development': False}

    def __init__(self, name, section, description=None, reqName=None):
        self.name = name
        self.reqName = reqName if reqName else name
        if section not in Dependency._sections:
            raise ValueError('Invalid section name')
        self.section = section
        if description is None and Dependency._sections[section]:
            msg = 'description is required for packages in this section'
            raise ValueError(msg)
        self.description = description

    @property
    def requires(self):
        """The text denoting the version requirements for this Dependeny"""
        return _getRequirementText(self.reqName)

# All dependencies, required and optional, must be included here
# and for optional dependencies, a description must be included
_DEPENDENCIES = [
    Dependency('numpy', 'required'),
    Dependency('packaging', 'required'),
    Dependency('pandas', 'data', "Nimble's DataFrame object"),
    Dependency('scipy', 'data',
               "Nimble's Sparse object and scientific calculations"),
    Dependency('matplotlib', 'operation', 'Plotting'),
    Dependency('cloudpickle', 'operation',
               'Saving and loading Nimble data objects'),
    Dependency('requests', 'operation',
               'Retrieving data from the web'),
    Dependency('h5py', 'operation', 'Loading hdf5 files'),
    Dependency('dateutil', 'operation',
               'Parsing strings to datetime objects', 'python-dateutil'),
    Dependency('sklearn', 'interfaces', 'Machine Learning', 'scikit-learn'),
    Dependency('hyperopt', 'operation',
               'Bayesian approach for hyperparameter tuning'),
    Dependency('storm_tuner', 'operation',
               'Stochastic Random Mutator for hyperparameter tuning'),
    Dependency('tensorflow', 'interfaces',
               'Neural Networks'),
    Dependency('keras', 'interfaces', 'Neural Networks'),
    Dependency('autoimpute', 'interfaces',
               'Imputation & machine learning with missing data'),
    ]

DEPENDENCIES = {dep.name: dep for dep in _DEPENDENCIES}

def checkVersion(package):
    """
    Compare the package requirements to the available version.
    """

    if hasattr(package, '__version__'):
        version = package.__version__
        vers = Version(version)
        
        name = package.__name__
        requirement = DEPENDENCIES[name].requires
        req = Requirement(requirement)
        for specifier in req.specifier:
            if not specifier.contains(vers):
                msg = f'The installed version of {req.name} ({vers}) does not '
                msg += f'meet the version requirements: {requirement}'
                raise PackageException(msg)
