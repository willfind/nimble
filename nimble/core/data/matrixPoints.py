"""
Method implementations and helpers acting specifically on points in a
Matrix object.
"""

import numpy

from nimble._utility import numpy2DArray
from .axis_view import AxisView
from .matrixAxis import MatrixAxis
from .points import Points
from .points_view import PointsView
from .dataHelpers import fillArrayWithCollapsedFeatures
from .dataHelpers import fillArrayWithExpandedFeatures

class MatrixPoints(MatrixAxis, Points):
    """
    Matrix method implementations performed on the points axis.

    Parameters
    ----------
    base : Matrix
        The Matrix instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _insert_implementation(self, insertBefore, toInsert):
        """
        Insert the points from the toInsert object below the provided
        index in this object, the remaining points from this object will
        continue below the inserted points.
        """
        startData = self._base.data[:insertBefore, :]
        endData = self._base.data[insertBefore:, :]
        self._base.data = numpy.concatenate(
            (startData, toInsert.data, endData), 0)

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)
            retArray = numpy.array(currRet, dtype=function.convertType)
            self._convertBaseDtype(retArray.dtype)
            self._base.data[i, :] = retArray
        # if transformations applied to all data and function.convertType is
        # not object we can convert base with object dtype to a numeric dtype.
        if (self._base.data.dtype == numpy.object_ and limitTo is None
                and function.convertType is not object):
            self._base.data = self._base.data.astype(function.convertType)

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = self._base.data[:, collapseIndices]
        retainData = self._base.data[:, retainIndices]

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, collapseData, currNumPoints,
            currFtNames, numRetPoints, numRetFeatures)

        self._base.data = numpy2DArray(tmpData)

    def _combineByExpandingFeatures_implementation(self, uniqueDict, namesIdx,
                                                   uniqueNames, numRetFeatures,
                                                   numExpanded):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures,
                                                numExpanded)

        self._base.data = numpy2DArray(tmpData)


class MatrixPointsView(PointsView, AxisView, MatrixPoints):
    """
    Limit functionality of MatrixPoints to read-only.

    Parameters
    ----------
    base : MatrixView
        The MatrixView instance that will be queried.
    """
    pass
