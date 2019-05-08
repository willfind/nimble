"""
Helper function for importing external packages into nimble.
"""

def importModule(name):
    """
    Attempt to import packages and return None if import fails.

    Parameters
    ----------
    name : str
        The package name.

    Examples
    --------
    mlpy = importModule('mlpy')
    """
    try:
        return __import__(name)
    except ImportError:
        return None
