"""
Store dependency versions requirements.

Also includes helpers for validating optional dependency versions at
runtime.
"""
from packaging.requirements import Requirement
from packaging.version import Version, LegacyVersion, InvalidVersion
from packaging.specifiers import LegacySpecifier

from nimble.exceptions import PackageException

DEPENDENCIES = {
    'required': {
        'numpy': 'numpy>=1.14',
        'packaging': 'packaging>=20.0'
    },
    'data': {
        'pandas': 'pandas>=0.24',
        'scipy': 'scipy>=1.1'
    },
    'operation': {
        'matplotlib': 'matplotlib>=3.1',
        'cloudpickle': 'cloudpickle>=1.0',
        'requests': 'requests>2.12',
        'h5py': 'h5py>=2.10',
        'dateutil': 'python-dateutil>=2.6'
    },
    'interfaces': {
        'sklearn': 'scikit-learn>=0.19',
        'tensorflow': 'tensorflow>=1.14',
        'keras': 'keras>=2.0',
        'autoimpute': 'autoimpute>=0.12',
        'shogun': 'shogun>=3',
        'mlpy': 'machine-learning-py>=3.5;python_version<"3.7"'
    },
    'development': {
        'pytest': 'pytest>=2.7.4',
        'pylint': 'pylint>=6.2',
        'cython': 'cython>=0.29',
        'sphinx': 'sphinx>=3.3'
    }
}

_LOCATIONS = {}
for key1, value in DEPENDENCIES.items():
    for key2 in value:
        _LOCATIONS[key2] = key1

def checkVersion(package):
    """
    Compare the package requirements to the available version.

    Generally, the package name and version and can be extracted from
    the package, but when that is not the case or the extracted value
    does not align with DEPENDENCIES, then alternate values can be
    provided.
    """
    version = package.__version__
    try:
        vers = Version(version)
        legacy = False
    except InvalidVersion:
        vers = LegacyVersion(version)
        legacy = True
    name = package.__name__
    location = _LOCATIONS[name]
    requirement = DEPENDENCIES[location][name]
    req = Requirement(requirement)
    for specifier in req.specifier:
        # when vers is a LegacyVersion, need specifiers to be LegacySpecifier
        if legacy and not isinstance(specifier, LegacySpecifier):
            specifier = LegacySpecifier(str(specifier))
        if not specifier.contains(vers):
            msg = 'The installed version of {0} ({1}) does not meet the '
            msg += 'version requirements: {2}'
            raise PackageException(msg.format(req.name, vers, requirement))
