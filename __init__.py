"""
Universal Machine Learning

UML offers interfaces into other machine learning packages and
tools for data representation and processing. Available at
the top level in this package are the functions necessary to
create data objects, call machine learning algorithms on that
data, and do package level configuration and information querying.

"""

from __future__ import absolute_import
import os
import inspect

UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import tempfile

capturedErr = tempfile.NamedTemporaryFile()

# load settings from configuration file
import UML.configuration

settings = UML.configuration.loadSettings()

# Import those submodules that need setup or we want to be
# accessible to the user
from UML.importExternalLibraries import importModule
import UML.interfaces
import UML.calculate
import UML.randomness
import UML.logger

# Import those functions that we want to be accessible in the
# top level
from UML.randomness import setRandomSeed

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
from .uml import crossValidateReturnAll
from .uml import crossValidateReturnBest

from .uml import log
from .uml import showLog

from .uml import learnerType

# now finish out with any other configuration that needs to be done

# These learners are required for unit testing, so we ensure they will
# be automatically registered by making surey they have entries in
# UML.settings.
UML.settings.set("RegisteredLearners", "Custom.RidgeRegression", 'UML.customLearners.RidgeRegression')
UML.settings.set("RegisteredLearners", "Custom.KNNClassifier", 'UML.customLearners.KNNClassifier')
UML.settings.set("RegisteredLearners", "Custom.MeanConstant", 'UML.customLearners.MeanConstant')
UML.settings.set("RegisteredLearners", "Custom.MultiOutputRidgeRegression",
                 'UML.customLearners.MultiOutputRidgeRegression')
UML.settings.saveChanges("RegisteredLearners")

# register those custom learners listed in UML.settings
UML.helpers.autoRegisterFromSettings()

# Now that we have loaded everything else, sync up the the settings object
# as needed.
UML.configuration.syncWithInterfaces(UML.settings)

# initialize the logging file
UML.logger.uml_logger.initLoggerAndLogConfig()

__all__ = ['createData', 'createRandomData', 'crossValidate',
           'crossValidateReturnAll', 'crossValidateReturnBest',
           'deregisterCustomLearner', 'deregisterCustomLearnerAsDefault',
           'identity', 'learnerDefaultValues', 'learnerParameters',
           'learnerType', 'listDataFunctions', 'listLearners',
           'listUMLFunctions', 'log', 'normalizeData', 'ones',
           'registerCustomLearner', 'registerCustomLearnerAsDefault',
           'setRandomSeed', 'settings', 'showLog', 'train', 'trainAndApply',
           'trainAndTest', 'trainAndTestOnTrainingData', 'UMLPath', 'zeros']
