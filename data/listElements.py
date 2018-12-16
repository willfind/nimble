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

    def _multiply_implementation(self, other):
        """
        Perform element wise multiplication of this UML data object
        against the provided other UML data object. Both objects must
        contain only numeric data. The pointCount and featureCount of
        both objects must be equal. The types of the two objects may be
        different, but the returned object will be the inplace
        modification of the calling object.
        """
        for pNum in range(len(self.source.points)):
            for fNum in range(len(self.source.features)):
                # Divided by 1 to make it raise if it involves non-numeric
                # types ('str')
                self.source.data[pNum][fNum] *= other[pNum, fNum] / 1