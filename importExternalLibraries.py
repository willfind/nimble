def importModule(name):
    """
    call
    """
    try:
        return __import__(name)
    except ImportError:
        return