"""
Method implementations and helpers acting specifically on each element
List object.
"""
from __future__ import absolute_import
import itertools

import numpy as np

import UML
from .elements import Elements

pd = UML.importModule('pandas')
if pd:
    import pandas as pd

class DataFrameElements(Elements):
    """
    DataFrame method implementations performed on each element.

    Parameters
    ----------
    source : UML data object
        The object containing features data.
    """
    def __init__(self, source, **kwds):
        self.source = source
        kwds['source'] = source
        super(DataFrameElements, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

    def _transform_implementation(self, toTransform, points, features,
                                  preserveZeros, skipNoneReturnValues):
        oneArg = False
        try:
            toTransform(0, 0, 0)
        except TypeError:
            if isinstance(toTransform, dict):
                oneArg = None
            else:
                oneArg = True

        IDs = itertools.product(range(len(self.source.points)),
                                range(len(self.source.features)))
        for (i, j) in IDs:
            currVal = self.source.data.iloc[i, j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue
            if preserveZeros and currVal == 0:
                continue

            if oneArg is None:
                if currVal in toTransform:
                    currRet = toTransform[currVal]
                else:
                    continue
            elif oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            if skipNoneReturnValues and currRet is None:
                continue

            self.source.data.iloc[i, j] = currRet

    ################################
    # Higher Order implementations #
    ################################

    def _calculate_implementation(self, function, points, features,
                                  preserveZeros, outputType):
        return self._calculateForEachElementGenericVectorized(
            function, points, features, outputType)

    #############################
    # Numerical implementations #
    #############################

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
