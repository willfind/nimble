"""
Nimble offers interfaces into other machine learning packages and
tools for data representation and processing. Available at
the top level in this package are the functions necessary to
create data objects, call machine learning algorithms on that
data, and do package level configuration and information querying.
"""
# pylint: disable=cyclic-import

# Import those functions that we want to be accessible in the
# top level
from nimble.core.configuration import nimblePath
from nimble.core.configuration import showAvailablePackages
from nimble.core.create import data
from nimble.core.create import ones
from nimble.core.create import zeros
from nimble.core.create import identity
from nimble.core.create import loadTrainedLearner
from nimble.core.create import fetchFile
from nimble.core.create import fetchFiles
from nimble.core.learn import learnerType
from nimble.core.learn import learnerNames
from nimble.core.learn import showLearnerNames
from nimble.core.learn import learnerParameters
from nimble.core.learn import showLearnerParameters
from nimble.core.learn import learnerParameterDefaults
from nimble.core.learn import showLearnerParameterDefaults
from nimble.core.learn import train
from nimble.core.learn import trainAndApply
from nimble.core.learn import trainAndTest
from nimble.core.learn import trainAndTestOnTrainingData
from nimble.core.learn import normalizeData
from nimble.core.learn import fillMatching
from nimble.core.learn import Init
from nimble.core.tune import Tune
from nimble.core.tune import Tuning
from nimble.core.logger import log
from nimble.core.logger import showLog
from nimble.core.interfaces import CustomLearner
from nimble._utility import _setAll

# import core (not in __all__)
from nimble import core

# import submodules accessible to the user (in __all__)
from nimble import learners
from nimble import calculate
from nimble import random
from nimble import match
from nimble import fill
from nimble import exceptions
# load settings from configuration file (comments below for Sphinx docstring)
#: User control over configurable options.
#:
#: Use nimble.settings.get() to see all sections and options.
#:
#: See Also
#: --------
#: nimble.core.configuration.SessionConfiguration
#:
#: Keywords
#: --------
#: configure, configuration, options
settings = core.configuration.loadSettings()

# initialize the interfaces
core.interfaces.initInterfaceSetup()

# initialize the logging file
core.logger.initLoggerAndLogConfig()

__all__ = _setAll(vars(), includeModules=True, ignore=['core'])

__version__ = "0.3.0"
