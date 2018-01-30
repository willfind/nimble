from __future__ import absolute_import
import os
import UML
from . import _collect_completed

available = _collect_completed.collect(os.path.join(UML.UMLPath, 'interfaces'))
from .custom_learner_interface import CustomLearnerInterface 
