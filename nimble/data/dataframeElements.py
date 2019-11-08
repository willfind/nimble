"""
Method implementations and helpers acting specifically on each element
List object.
"""

from __future__ import absolute_import
import itertools

import numpy as np

import nimble
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
    base : DataFrame
        The DataFrame instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _transform_implementation(self, toTransform, points, features):
        IDs = itertools.product(range(len(self._base.points)),
                                range(len(self._base.features)))
        for i, j in IDs:
            currVal = self._base.data.values[i, j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            self._base.data.iloc[i, j] = currRet

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
        return denseCountUnique(self._base, points, features)


class DataFrameElementsView(ElementsView, DataFrameElements):
    """
    Limit functionality of DataFrameElements to read-only.

    Parameters
    ----------
    base : DataFrameView
        The DataFrameView instance that will be queried.
    """
    pass
