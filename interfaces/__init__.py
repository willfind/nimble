"""
Make available any interfaces accessible to the user.
"""

from __future__ import absolute_import
import os

from UML import UMLPath
from . import _collect_completed
from .custom_learner_interface import CustomLearnerInterface

interfacesPath = os.path.join(UMLPath, 'interfaces')
available = _collect_completed.collect(interfacesPath)
