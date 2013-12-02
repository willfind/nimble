#from uml import UMLPath
import os
import inspect
UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

from runners import run
from runners import runAndTest

from uml import createData
from uml import createRandomizedData
from uml import normalizeData
from uml import splitData


from uml import listDataFunctions
from uml import listUMLFunctions
from uml import listLearners
from uml import learnerParameters
from uml import learnerDefaultValues

from uml import crossValidate
from uml import crossValidateReturnAll
from uml import crossValidateReturnBest

