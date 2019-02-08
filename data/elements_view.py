"""
Defines a subclass of the Elements object, which serves as the primary
base class for read only elements views of data objects.
"""
from .elements import Elements
from .dataHelpers import readOnlyException

class ElementsView(Elements):
    """
    Class defining read only view objects, which have the same api as a
    normal Elements object, but disallow all methods which could change
    the data.

    Parameters
    ----------
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def transform(self, toTransform, points=None, features=None,
                  preserveZeros=False, skipNoneReturnValues=False):
        readOnlyException("transform")

    def multiply(self, other):
        readOnlyException("multiply")

    def power(self, other):
        readOnlyException("power")
