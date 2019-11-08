"""
Defines a subclass of the Elements object, which serves as the primary
base class for read only elements views of data objects.
"""

from __future__ import absolute_import

import nimble
from nimble.utility import inheritDocstringsFactory
from .elements import Elements
from .dataHelpers import readOnlyException
from .dataHelpers import exceptionDocstringFactory

exceptionDocstring = exceptionDocstringFactory(Elements)

@inheritDocstringsFactory(Elements)
class ElementsView(Elements):
    """
    Class limiting the Elements class to read-only by disallowing
    methods which could change the data.

    Parameters
    ----------
    base : BaseView
        The BaseView instance that will be queried.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    @exceptionDocstring
    def transform(self, toTransform, points=None, features=None,
                  preserveZeros=False, skipNoneReturnValues=False,
                  useLog=None):
        readOnlyException("transform")
