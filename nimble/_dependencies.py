"""
Store dependency versions requirements.

Also includes helpers for validating optional dependency versions at
runtime.
"""
from packaging.requirements import Requirement
from packaging.version import Version, LegacyVersion, InvalidVersion
from packaging.specifiers import LegacySpecifier

from nimble.exceptions import PackageException

class Dependency:
    """
    Store details of a Nimble dependency.

    Parameters
    ----------
    name : str
        The name of the package.
    requires : str
        The version requirements for the package.
    section : str
        The category for setup.py to classify this dependency.
    description : str, None
        A description of the package's function within Nimble.
    """
    # each dependency must be categorized into a section for setup.py
    # each section maps to a boolean value for whether it requires a
    # descriptions to be used with nimble.showAvailablePackages
    _sections = {'required': False, 'data': True, 'operation': True,
                'interfaces': True, 'development': False}

    def __init__(self, name, requires, section, description=None):
        self.name = name
        self.requires = requires
        if section not in Dependency._sections:
            raise ValueError('Invalid section name')
        self.section = section
        if description is None and Dependency._sections[section]:
            msg = 'description is required for packages in this section'
            raise ValueError(msg)
        self.description = description

# All dependencies, required and optional, must be included here
_DEPENDENCIES = [
    Dependency('numpy', 'numpy>=1.14, <1.24', 'required'),
    Dependency('packaging', 'packaging>=20.0,<=21.3', 'required'),
    Dependency('pandas', 'pandas>=0.24', 'data', "Nimble's DataFrame object"),
    Dependency('scipy', 'scipy>=1.1,<1.9', 'data',
               "Nimble's Sparse object and scientific calculations"),
    Dependency('matplotlib', 'matplotlib>=3.1', 'operation', 'Plotting'),
    Dependency('cloudpickle', 'cloudpickle>=1.0', 'operation',
               'Saving and loading Nimble data objects'),
    Dependency('requests', 'requests>2.12', 'operation',
               'Retrieving data from the web'),
    Dependency('h5py', 'h5py>=2.10', 'operation', 'Loading hdf5 files'),
    Dependency('dateutil', 'python-dateutil>=2.6', 'operation',
               'Parsing strings to datetime objects'),
    Dependency('sklearn', 'scikit-learn>=1.0', 'interfaces',
               'Machine Learning'),
    Dependency('hyperopt', 'hyperopt>=0.2', 'operation',
               'Bayesian approach for hyperparameter tuning'),
    Dependency('storm_tuner', 'storm_tuner', 'operation', # no __version__
               'Stochastic Random Mutator for hyperparameter tuning'),
    Dependency('tensorflow', 'tensorflow>=1.14', 'interfaces',
               'Neural Networks'),
    Dependency('keras', 'keras>=2.0', 'interfaces', 'Neural Networks'),
    Dependency('autoimpute', 'autoimpute>=0.12', 'interfaces',
               'Imputation & machine learning with missing data'),
    Dependency('pytest', 'pytest>=6.2', 'development'),
    Dependency('pylint', 'pylint>=2.7.4', 'development'),
    Dependency('cython', 'cython>=0.29', 'development'),
    Dependency('sphinx', 'sphinx>=3.3', 'development'),
    ]

DEPENDENCIES = {dep.name: dep for dep in _DEPENDENCIES}

def checkVersion(package):
    """
    Compare the package requirements to the available version.
    """

    if hasattr(package, '__version__'):
        version = package.__version__
        try:
            vers = Version(version)
            legacy = False
        except InvalidVersion:
            vers = LegacyVersion(version)
            legacy = True
        name = package.__name__
        requirement = DEPENDENCIES[name].requires
        req = Requirement(requirement)
        for specifier in req.specifier:
            # need specifiers to be LegacySpecifier for LegacyVersion
            if legacy and not isinstance(specifier, LegacySpecifier):
                specifier = LegacySpecifier(str(specifier))
            if not specifier.contains(vers):
                msg = f'The installed version of {req.name} ({vers}) does not '
                msg += f'meet the version requirements: {requirement}'
                raise PackageException(msg)
