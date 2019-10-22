"""
Method implementations and helpers acting specifically on each element
Matrix object.
"""

from __future__ import absolute_import
import itertools

import numpy

import nimble
from nimble.utility import numpy2DArray
from .elements import Elements
from .elements_view import ElementsView
from .dataHelpers import denseCountUnique

class MatrixElements(Elements):
    """
    Matrix method implementations performed on each element.

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
            currVal = self._base.data[i, j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            self._base.data[i, j] = currRet

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
        if isinstance(other, nimble.data.Sparse):
            result = other.data.multiply(self._base.data)
            if hasattr(result, 'todense'):
                result = result.todense()
        else:
            result = numpy.multiply(self._base.data, other.data)
        self._base.data = numpy2DArray(result)


class MatrixElementsView(ElementsView, MatrixElements):
    """
    Limit functionality of MatrixElements to read-only
    """
    pass
