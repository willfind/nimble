"""
Make available any interfaces accessible to the user.
"""

from .universal_interface import TrainedLearner
from .custom_learner import CustomLearner
from ._collect_interfaces import initInterfaceSetup

predefined = []
available = {}

def customMlGetattr(self, name):
    base = f"Attribute {name} does not exist for Nimble learner objects. "
    if name == 'fit':
        msg = "Try .train() instead."
        raise AttributeError(base + msg)
    elif name == 'fit_transform':
        msg = "Try nimble.fillMatching/nimble.Normalize() instead."
        raise AttributeError(base + msg)
    elif name == 'transform':  
        msg = "Try nimble.fillMatching/nimble.Normalize() instead."
        raise AttributeError(base + msg)
    elif name == 'predict':
        msg = "Try .apply() instead."
        raise AttributeError(base + msg)
    elif name == 'score':
        msg = "Try .getScores() instead."
        raise AttributeError(base + msg)
    elif name == 'get_params':
        msg = "Try .getLearnerParameterNames() instead."
        raise AttributeError(base + msg)
    else:
        return self.__getattribute__(name)

TrainedLearner.__getattr__ = customMlGetattr
CustomLearner.__getattr__ = customMlGetattr
