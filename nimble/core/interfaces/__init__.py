"""
Make available any interfaces accessible to the user.
"""

from .universal_interface import TrainedLearner
from .custom_learner import CustomLearner
from ._collect_interfaces import initInterfaceSetup

predefined = []
available = {}

#[fit, fit_transform, transform,predict, predict_proba, score, get_params]

def custom_getattr(obj, name):
    if name == 'predict':
        raise AttributeError("Attribute predict does not exist for Nimble model objects. Try .apply() instead.")
        #return lambda: print(f"{obj.__name__} doesn't have '{name}()' function. Consider using '{obj.__name__}.linspace()' instead.")
    elif name == 'fit':
        raise AttributeError("Attribute fit does not exist for Nimble model objects. Try .train() instead.")
        #return lambda: print(f"{obj.__name__} doesn't have '{name}()' function. Consider using '{obj.__name__}.rand()' or '{obj.__name__}.randn()' instead.")
    else:
        raise AttributeError(f"'{obj.__name__}' object has no attribute '{name}'")

TrainedLearner.__getattr__ = custom_getattr
CustomLearner.__getattr__ = custom_getattr
