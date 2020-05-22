from nimble.interfaces import predefined
from nimble.helpers import findBestInterface
from nimble.exceptions import PackageException

def setup():
    """
    Predefined interfaces were previously loaded on nimble import but
    are now loaded as needed. Some tests operate under the assumption
    that all these interfaces have already been loaded, but since that
    is no longer the case we need to load them now to ensure that those
    tests continue to test all interfaces.
    """
    for interface in predefined:
        try:
            findBestInterface(interface.getCanonicalName())
        except PackageException:
            pass
