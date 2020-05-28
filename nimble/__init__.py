"""
Nimble offers interfaces into other machine learning packages and
tools for data representation and processing. Available at
the top level in this package are the functions necessary to
create data objects, call machine learning algorithms on that
data, and do package level configuration and information querying.
"""

# Import those functions that we want to be accessible in the
# top level
from nimble.configuration import nimblePath
from nimble.core.core import train
from nimble.core.core import trainAndApply
from nimble.core.core import trainAndTest
from nimble.core.core import trainAndTestOnTrainingData
from nimble.core.core import createData
from nimble.core.core import createRandomData
from nimble.core.core import ones
from nimble.core.core import zeros
from nimble.core.core import identity
from nimble.core.core import normalizeData
from nimble.core.core import fillMatching
from nimble.core.core import listLearners
from nimble.core.core import learnerParameters
from nimble.core.core import learnerDefaultValues
from nimble.core.core import crossValidate
from nimble.core.core import log
from nimble.core.core import showLog
from nimble.core.core import learnerType
from nimble.core.core import loadData
from nimble.core.core import loadTrainedLearner
from nimble.core.core import CV
from nimble.core.core import Init
from nimble.core.custom_learner import CustomLearner

# Import those submodules that need setup or we want to be
# accessible to the user
from nimble import configuration
from nimble import core
from nimble import learners
from nimble import calculate
from nimble import random
from nimble import match
from nimble import fill

# load settings from configuration file
settings = configuration.loadSettings()

# initialize the interfaces
core.interfaces.initInterfaceSetup()

# initialize the logging file
core.logger.initLoggerAndLogConfig()

__all__ = ['calculate', 'createData', 'createRandomData', 'crossValidate',
           'CustomLearner', 'CV', 'fill', 'identity', 'Init',
           'learnerDefaultValues', 'learnerParameters', 'learners',
           'learnerType', 'listLearners', 'loadData', 'loadTrainedLearner',
           'log', 'match', 'normalizeData', 'ones', 'setRandomSeed',
           'settings', 'showLog', 'randomness', 'train', 'trainAndApply',
           'trainAndTest', 'trainAndTestOnTrainingData', 'nimblePath', 'zeros']
