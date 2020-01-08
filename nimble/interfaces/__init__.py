"""
Make available any interfaces accessible to the user.
"""

import os

from nimble import nimblePath
from . import _collect_completed
from .custom_learner_interface import CustomLearnerInterface

interfacesPath = os.path.join(nimblePath, 'interfaces')
predefined = _collect_completed.collectPredefinedInterfaces(interfacesPath)
available = _collect_completed.collect(interfacesPath)
