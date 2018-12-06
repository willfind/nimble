"""
TODO
"""
from __future__ import absolute_import

from .elements import Elements

class MatrixElements(Elements):
    """
    TODO
    """
    def __init__(self, source):
        self.source = source
        kwds = {}
        kwds['source'] = self.source
        super(MatrixElements, self).__init__(**kwds)

    ################################
    # Higher Order implementations #
    ################################

    def _calculate_implementation(self, function, points, features,
                                  preserveZeros, outputType):
        return self._calculateForEachElementGenericVectorized(
               function, points, features, outputType)
