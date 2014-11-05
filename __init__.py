"""
Universal Machine Learning

UML offers interfaces into other machine learning packages,
tools for data representation and processing, 

"""

import os
import inspect
UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# load settings from configuration file
import UML.configuration
settings = UML.configuration.loadSettings()

import UML.interfaces
import UML.metrics
import UML.randomness

from UML.randomness import setRandomSeed

from uml import train
from uml import trainAndApply
from uml import trainAndTest
from uml import createData
from uml import createRandomData
from uml import normalizeData
from uml import splitData

from uml import registerCustomLearner
from uml import deregisterCustomLearner
from uml import listDataFunctions
from uml import listUMLFunctions
from uml import listLearners
from uml import learnerParameters
from uml import learnerDefaultValues

from uml import crossValidate
from uml import crossValidateReturnAll
from uml import crossValidateReturnBest

from uml import learnerType

# These learners are required for unit testing, so we ensure they will
# be automatically registered by making surey they have entries in
# UML.settings.
UML.settings.set("RegisteredLearners", "Custom.RidgeRegression", 'UML.customLearners.RidgeRegression')
UML.settings.set("RegisteredLearners", "Custom.KNNClassifier", 'UML.customLearners.KNNClassifier')

# register those custom learners listed in UML.settings
UML.umlHelpers.autoRegisterFromSettings()

# Now that we have loaded everything else, sync up the the settings object
# as needed.
UML.configuration.syncWithInterfaces(UML.settings)
