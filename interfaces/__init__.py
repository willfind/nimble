import os
import UML
import _collect_completed

available = _collect_completed.collect(os.path.join(UML.UMLPath, 'interfaces'))
from custom_learner_interface import CustomLearnerInterface 
from custom_learner import CustomLearner
