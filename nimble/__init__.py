"""
Nimble offers interfaces into other machine learning packages and
tools for data representation and processing. Available at
the top level in this package are the functions necessary to
create data objects, call machine learning algorithms on that
data, and do package level configuration and information querying.
"""

import os
import inspect

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
from .core import fillMatching
from .core import listLearners
from .core import learnerParameters
from .core import learnerDefaultValues
from .core import crossValidate
from .core import log
from .core import showLog
from .core import learnerType
from .core import loadData
from .core import loadTrainedLearner
from .core import CV
from .core import Init
from .custom_learner import CustomLearner
from ._configuration import nimblePath

# Import those submodules that need setup or we want to be
# accessible to the user
from . import learners
from . import calculate
from . import randomness
from . import match
from . import fill
from . import interfaces
from . import logger
from . import _configuration

# load settings from configuration file
settings = _configuration.loadSettings()

# initialize the logging file
logger.active = logger.initLoggerAndLogConfig()

__all__ = ['calculate', 'createData', 'createRandomData', 'crossValidate',
           'CustomLearner', 'CV', 'fill', 'identity', 'Init',
           'learnerDefaultValues', 'learnerParameters', 'learners',
           'learnerType', 'listLearners', 'loadData', 'loadTrainedLearner',
           'log', 'match', 'normalizeData', 'ones', 'setRandomSeed',
           'settings', 'showLog', 'randomness', 'train', 'trainAndApply',
           'trainAndTest', 'trainAndTestOnTrainingData', 'nimblePath', 'zeros']
