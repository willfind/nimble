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
    source : nimble data object
        The object containing point and feature data.
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
        for pNum in range(len(self._base.points)):
            for fNum in range(len(self._base.features)):
                # Divided by 1 to make it raise if it involves non-numeric
                # types ('str')
                self._base.data[pNum][fNum] *= other[pNum, fNum] / 1

class ListElementsView(ElementsView, ListElements):
    """
    Limit functionality of ListElements to read-only
    """
    pass
