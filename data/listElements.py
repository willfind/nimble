"""

"""
from __future__ import absolute_import

from .elements import Elements

class ListElements(Elements):
    """

    """
    def __init__(self, source):
        self.source = source
        super(ListElements, self).__init__()

    ################################
    # Higher Order implementations #
    ################################

    def _calculate_implementation(self, function, points, features,
                                  preserveZeros, outputType):
        return self._calculateForEachElementGenericVectorized(
            function, points, features, outputType)
