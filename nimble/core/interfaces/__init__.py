"""
Make available any interfaces accessible to the user.
"""

from .universal_interface import TrainedLearner
from .custom_learner import CustomLearner
from ._collect_interfaces import initInterfaceSetup

predefined = []
available = {}
