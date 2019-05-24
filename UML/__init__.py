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

from . import configuration
from .configuration import nimblePath

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
from .uml import train
from .uml import trainAndApply
from .uml import trainAndTest
from .uml import trainAndTestOnTrainingData
from .uml import createData
from .uml import createRandomData
from .uml import ones
from .uml import zeros
from .uml import identity
from .uml import normalizeData
from .uml import registerCustomLearner
from .uml import registerCustomLearnerAsDefault
from .uml import deregisterCustomLearner
from .uml import deregisterCustomLearnerAsDefault
from .uml import listLearners
from .uml import learnerParameters
from .uml import learnerDefaultValues
from .uml import crossValidate
from .uml import log
from .uml import showLog
from .uml import learnerType
from .uml import loadData
from .uml import loadTrainedLearner
from .helpers import CV

capturedErr = tempfile.NamedTemporaryFile()

# load settings from configuration file
settings = configuration.loadSettings()

# now finish out with any other configuration that needs to be done

# These learners are required for unit testing, so we ensure they will
# be automatically registered by making surey they have entries in
# nimble.settings.
settings.set("RegisteredLearners", "Custom.RidgeRegression",
             'UML.customLearners.RidgeRegression')
settings.set("RegisteredLearners", "Custom.KNNClassifier",
             'UML.customLearners.KNNClassifier')
settings.set("RegisteredLearners", "Custom.MeanConstant",
             'UML.customLearners.MeanConstant')
settings.set("RegisteredLearners", "Custom.MultiOutputRidgeRegression",
             'UML.customLearners.MultiOutputRidgeRegression')
settings.saveChanges("RegisteredLearners")

# register those custom learners listed in nimble.settings
configuration.autoRegisterFromSettings()

# Now that we have loaded everything else, sync up the the settings object
# as needed.
for interface in interfaces.available:
    configuration.setInterfaceOptions(settings, interface, save=True)

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
