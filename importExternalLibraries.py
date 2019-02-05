"""
Helper function for importing external packages into UML.
"""

def importModule(name):
    """
    Attempt to import packages and return None if import fails.

    Parameters
    ----------
    name : str
        The package name.
    """
    try:
        return __import__(name)
    except ImportError:
        return None
