"""
Nimble

Nimble offers interfaces into other machine learning packages and
tools for data representation and processing. Available at
the top level in this package are the functions necessary to
create data objects, call machine learning algorithms on that
data, and do package level configuration and information querying.
"""

from __future__ import absolute_import
import os
import inspect
import tempfile
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy
numpy.seterr(divide='raise')

from . import configuration
from .configuration import nimblePath
# load settings from configuration file
settings = configuration.loadSettings()

# Import those submodules that need setup or we want to be
# accessible to the user
from .importExternalLibraries import importModule
from . import interfaces
from . import calculate
from . import randomness
from . import logger

# Import those functions that we want to be accessible in the
# top level
from .randomness import setRandomSeed
from .randomness import pythonRandom
from .randomness import numpyRandom
from .core import train
from .core import trainAndApply
from .core import trainAndTest
from .core import trainAndTestOnTrainingData
from .core import createData
from .core import createRandomData
from .core import ones
from .core import zeros
from .core import identity
from .core import normalizeData
from .core import registerCustomLearner
from .core import registerCustomLearnerAsDefault
from .core import deregisterCustomLearner
from .core import deregisterCustomLearnerAsDefault
from .core import listLearners
from .core import learnerParameters
from .core import learnerDefaultValues
from .core import crossValidate
from .core import log
from .core import showLog
from .core import learnerType
from .core import loadData
from .core import loadTrainedLearner
from .helpers import CV

capturedErr = tempfile.NamedTemporaryFile()

# now finish out with any other configuration that needs to be done

# These learners are required for unit testing, so we ensure they will
# be automatically registered by making surey they have entries in
# nimble.settings.
settings.set("RegisteredLearners", "Custom.RidgeRegression",
             'nimble.customLearners.RidgeRegression')
settings.set("RegisteredLearners", "Custom.KNNClassifier",
             'nimble.customLearners.KNNClassifier')
settings.set("RegisteredLearners", "Custom.MeanConstant",
             'nimble.customLearners.MeanConstant')
settings.set("RegisteredLearners", "Custom.MultiOutputRidgeRegression",
             'nimble.customLearners.MultiOutputRidgeRegression')
settings.saveChanges("RegisteredLearners")

# register those custom learners listed in nimble.settings
configuration.autoRegisterFromSettings()

# Now that we have loaded everything else, sync up the the settings object
# as needed.
configuration.setAndSaveAvailableInterfaceOptions()

# initialize the logging file
logger.active = logger.initLoggerAndLogConfig()

__all__ = ['createData', 'createRandomData', 'crossValidate',
           'deregisterCustomLearner', 'deregisterCustomLearnerAsDefault',
           'identity', 'learnerDefaultValues', 'learnerParameters',
           'learnerType', 'listLearners', 'loadData', 'loadTrainedLearner',
           'log', 'normalizeData', 'ones', 'registerCustomLearner',
           'registerCustomLearnerAsDefault', 'setRandomSeed', 'settings',
           'showLog', 'train', 'trainAndApply', 'trainAndTest',
           'trainAndTestOnTrainingData', 'nimblePath', 'zeros']
