"""
TODO
"""
from __future__ import absolute_import

from .elements import Elements

class ListElements(Elements):
    """
    TODO
    """
    def __init__(self, source):
        self.source = source
        kwds = {}
        kwds['source'] = self.source
        super(ListElements, self).__init__(**kwds)

    ################################
    # Higher Order implementations #
    ################################

    def _calculate_implementation(self, function, points, features,
                                  preserveZeros, outputType):
        return self._calculateForEachElementGenericVectorized(
               function, points, features, outputType)
