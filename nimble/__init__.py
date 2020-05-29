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
from nimble.core.create import createData
from nimble.core.create import createRandomData
from nimble.core.create import ones
from nimble.core.create import zeros
from nimble.core.create import identity
from nimble.core.create import loadData
from nimble.core.create import loadTrainedLearner
from nimble.core.learn import learnerType
from nimble.core.learn import listLearners
from nimble.core.learn import learnerParameters
from nimble.core.learn import learnerDefaultValues
from nimble.core.learn import train
from nimble.core.learn import trainAndApply
from nimble.core.learn import trainAndTest
from nimble.core.learn import trainAndTestOnTrainingData
from nimble.core.learn import normalizeData
from nimble.core.learn import fillMatching
from nimble.core.learn import crossValidate
from nimble.core.learn import CV
from nimble.core.learn import Init
from nimble.core.logger import log
from nimble.core.logger import showLog
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
