"""
Universal Machine Learning

UML offers interfaces into other machine learning packages,
tools for data representation and processing, 

"""

import os
import inspect
UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import UML.interfaces
import UML.metrics

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

# automatic registration of custom learners to be used in unit testing 
UML.registerCustomLearner('Custom', UML.customLearners.RidgeRegression)
UML.registerCustomLearner('Custom', UML.customLearners.KNNClassifier)
