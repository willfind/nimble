"""

"""
from __future__ import absolute_import

import numpy as np

import UML
from .elements import Elements

pd = UML.importModule('pandas')
if pd:
    import pandas as pd

class DataFrameElements(Elements):
    """

    """
    def __init__(self, source, **kwds):
        self.source = source
        kwds['source'] = source
        super(DataFrameElements, self).__init__(**kwds)

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
        if isinstance(other, UML.data.Sparse):
            result = other.data.multiply(self.source.data.values)
            if hasattr(result, 'todense'):
                result = result.todense()
            self.source.data = pd.DataFrame(result)
        else:
            self.source.data = pd.DataFrame(
                np.multiply(self.source.data.values, other.data))
