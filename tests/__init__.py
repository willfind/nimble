import logging

from nimble.core._learnHelpers import initAvailablePredefinedInterfaces

def setup():
    """
    Predefined interfaces were previously loaded on nimble import but
    are now loaded as requested. Some tests operate under the assumption
    that all these interfaces have already been loaded, but since that
    is no longer the case we need to load them now to ensure that those
    tests continue to test all interfaces.
    """
    initAvailablePredefinedInterfaces()
