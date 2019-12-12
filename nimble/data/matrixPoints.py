"""
Method implementations and helpers acting specifically on points in a
Matrix object.
"""

from __future__ import absolute_import

import numpy

from nimble.exceptions import InvalidArgumentValue
from nimble.utility import numpy2DArray
from .axis_view import AxisView
from .matrixAxis import MatrixAxis
from .points import Points
from .points_view import PointsView
from .dataHelpers import fillArrayWithCollapsedFeatures
from .dataHelpers import fillArrayWithExpandedFeatures
from .dataHelpers import allDataIdentical

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
        dtypes = []
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)
            if len(currRet) != len(self._base.features):
                msg = "function must return an iterable with as many elements "
                msg += "as features in this object"
                raise InvalidArgumentValue(msg)

            retArray = numpy.array(currRet)
            if not numpy.issubdtype(retArray.dtype, numpy.number):
                retArray = numpy.array(currRet, dtype=numpy.object_)
            dtypes.append(retArray.dtype)
            if self._base.data.dtype == numpy.object_:
                self._base.data[i, :] = retArray
            elif self._base.data.dtype == numpy.int:
                if retArray.dtype == numpy.float:
                    self._base.data = self._base.data.astype(numpy.float)
                elif retArray.dtype != numpy.int:
                    self._base.data = self._base.data.astype(numpy.object_)
                self._base.data[i, :] = retArray
            else:
                if retArray.dtype not in [numpy.float, numpy.int]:
                    self._base.data = self._base.data.astype(numpy.object_)
                self._base.data[i, :] = retArray
        # if transformations to an object dtype returned numeric dtypes and
        # applied to all data we will convert to a float dtype.
        if (self._base.data.dtype == numpy.object_ and limitTo is None
                and all(numpy.issubdtype(dt, numpy.number) for dt in dtypes)):
            self._base.data = self._base.data.astype(numpy.float)

    # def _flattenToOne_implementation(self):
    #     numElements = len(self._base.points) * len(self._base.features)
    #     self._base.data = self._base.data.reshape((1, numElements),
    #                                                 order='C')
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     numPoints = divideInto
    #     numFeatures = len(self._base.features) // numPoints
    #     self._base.data = self._base.data.reshape((numPoints,
    #                                                    numFeatures),
    #                                                 order='C')

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

    def _combineByExpandingFeatures_implementation(
            self, uniqueDict, namesIdx, uniqueNames, numRetFeatures):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures)

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
