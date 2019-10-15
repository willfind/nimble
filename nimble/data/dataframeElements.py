"""
Method implementations and helpers acting specifically on each element
List object.
"""

from __future__ import absolute_import
import itertools

import numpy as np

import nimble
from nimble.utility import cooMatrixToArray
from .elements import Elements
from .elements_view import ElementsView
from .dataHelpers import denseCountUnique

pd = nimble.importModule('pandas')
if pd:
    import pandas as pd

class DataFrameElements(Elements):
    """
    DataFrame method implementations performed on each element.

    Parameters
    ----------
    source : nimble data object
        The object containing point and feature data.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _transform_implementation(self, toTransform, points, features):
        IDs = itertools.product(range(len(self._source.points)),
                                range(len(self._source.features)))
        for i, j in IDs:
            currVal = self._source.data.values[i, j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            self._source.data.iloc[i, j] = currRet

    ################################
    # Higher Order implementations #
    ################################

    def _calculate_implementation(self, function, points, features,
                                  preserveZeros, outputType):
        return self._calculate_genericVectorized(
            function, points, features, outputType)


    #########################
    # Query implementations #
    #########################

    def _countUnique_implementation(self, points, features):
        return denseCountUnique(self._source, points, features)

    #############################
    # Numerical implementations #
    #############################

    def _multiply_implementation(self, other):
        """
        Perform element wise multiplication of this nimble Base object
        against the provided other nimble Base object. Both objects must
        contain only numeric data. The pointCount and featureCount of
        both objects must be equal. The types of the two objects may be
        different, but the returned object will be the inplace
        modification of the calling object.
        """
        if isinstance(other, nimble.data.Sparse):
            result = other.data.multiply(self._source.data.values)
            if hasattr(result, 'toarray'):
                result = cooMatrixToArray(result)
            self._source.data = pd.DataFrame(result)
        else:
            self._source.data = pd.DataFrame(
                np.multiply(self._source.data.values, other.data))

class DataFrameElementsView(ElementsView, DataFrameElements):
    """
    Limit functionality of DataFrameElements to read-only
    """
    pass
