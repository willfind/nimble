"""
Method implementations and helpers acting specifically on each element
List object.
"""

from __future__ import absolute_import
import itertools

from .elements import Elements
from .elements_view import ElementsView
from .dataHelpers import denseCountUnique

class ListElements(Elements):
    """
    List method implementations performed on each element.

    Parameters
    ----------
    base : List
        The List instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _transform_implementation(self, toTransform, points, features):
        IDs = itertools.product(range(len(self._base.points)),
                                range(len(self._base.features)))
        for i, j in IDs:
            currVal = self._base.data[i][j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            self._base.data[i][j] = currRet

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


class ListElementsView(ElementsView, ListElements):
    """
    Limit functionality of ListElements to read-only.

    Parameters
    ----------
    base : ListView
        The ListView instance that will be queried.
    """
    pass
