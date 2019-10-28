"""
Method implementations and helpers acting specifically on points in a
Sparse object.
"""

from __future__ import absolute_import

import numpy

import nimble
from nimble.utility import OptionalPackage
from .axis_view import AxisView
from .sparseAxis import SparseAxis
from .points import Points
from .points_view import PointsView

scipy = OptionalPackage('scipy')

class SparsePoints(SparseAxis, Points):
    """
    Sparse method implementations performed on the points axis.

    Parameters
    ----------
    source : nimble data object
        The object containing point and feature data.
    """

    ##############################
    # Structural implementations #
    ##############################

    # def _flattenToOne_implementation(self):
    #     self._source._sortInternal('point')
    #     pLen = len(self._source.features)
    #     numElem = len(self._source.points) * len(self._source.features)
    #     data = self._source.data.data
    #     row = self._source.data.row
    #     col = self._source.data.col
    #     for i in range(len(data)):
    #         if row[i] > 0:
    #             col[i] += (row[i] * pLen)
    #             row[i] = 0
    #
    #     self._source.data = coo_matrix((data, (row, col)), (1, numElem))
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     # only one feature, so both sorts are the same order
    #     if self._source._sorted is None:
    #         self._source._sortInternal('point')
    #
    #     numPoints = divideInto
    #     numFeatures = len(self._source.features) // numPoints
    #     newShape = (numPoints, numFeatures)
    #     data = self._source.data.data
    #     row = self._source.data.row
    #     col = self._source.data.col
    #     for i in range(len(data)):
    #         # must change the row entry before modifying the col entry
    #         row[i] = col[i] / numFeatures
    #         col[i] = col[i] % numFeatures
    #
    #     self._source.data = coo_matrix((data, (row, col)), newShape)
    #     self._source._sorted = 'point'

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        if self._source._sorted is None:
            self._source._sortInternal('point')
        data = self._source.data.data
        row = self._source.data.row
        col = self._source.data.col
        tmpData = []
        tmpRow = []
        tmpCol = []
        collapseNames = [self._source.features.getName(idx)
                         for idx in collapseIndices]
        for ptIdx in range(len(self)):
            inRetain = [val in retainIndices for val in col]
            inCollapse = [val in collapseIndices for val in col]
            retainData = data[(row == ptIdx) & (inRetain)]
            retainCol = col[(row == ptIdx) & (inRetain)]
            collapseData = data[(row == ptIdx) & (inCollapse)]
            sort = numpy.argsort(collapseIndices)
            collapseData = collapseData[sort]
            for idx, value in enumerate(collapseData):
                tmpData.extend(retainData)
                tmpRow.extend([ptIdx * len(featuresToCollapse) + idx]
                              * len(retainData))
                tmpCol.extend([i for i in range(len(retainCol))])
                tmpData.append(collapseNames[idx])
                tmpRow.append(ptIdx * len(featuresToCollapse) + idx)
                tmpCol.append(numRetFeatures - 2)
                tmpData.append(value)
                tmpRow.append(ptIdx * len(featuresToCollapse) + idx)
                tmpCol.append(numRetFeatures - 1)

        tmpData = numpy.array(tmpData, dtype=numpy.object_)
        self._source.data = scipy.sparse.coo_matrix(
            (tmpData, (tmpRow, tmpCol)), shape=(numRetPoints, numRetFeatures))
        self._source._sorted = None

    def _combineByExpandingFeatures_implementation(
            self, uniqueDict, namesIdx, uniqueNames, numRetFeatures):
        tmpData = []
        tmpRow = []
        tmpCol = []
        for idx, point in enumerate(uniqueDict):
            tmpPoint = list(point[:namesIdx])
            for name in uniqueNames:
                if name in uniqueDict[point]:
                    tmpPoint.append(uniqueDict[point][name])
                else:
                    tmpPoint.append(numpy.nan)
            tmpPoint.extend(point[namesIdx:])
            tmpData.extend(tmpPoint)
            tmpRow.extend([idx for _ in range(len(point) + len(uniqueNames))])
            tmpCol.extend([i for i in range(numRetFeatures)])

        tmpData = numpy.array(tmpData, dtype=numpy.object_)
        shape = (len(uniqueDict), numRetFeatures)
        self._source.data = scipy.sparse.coo_matrix(
            (tmpData, (tmpRow, tmpCol)), shape=shape)
        self._source._sorted = None

class SparsePointsView(PointsView, AxisView, SparsePoints):
    """
    Limit functionality of SparsePoints to read-only
    """

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

    def _unique_implementation(self):
        unique = self._source.copy(to='Sparse')
        return unique.points._unique_implementation()

    def _repeat_implementation(self, totalCopies, copyValueByValue):
        copy = self._source.copy(to='Sparse')
        return copy.points._repeat_implementation(totalCopies,
                                                  copyValueByValue)

class nzIt(object):
    """
    Non-zero iterator to return when iterating through each feature.
    """
    # IDEA: check if sorted in the way you want.
    # if yes, iterate through
    # if no, use numpy argsort? this gives you indices that
    # would sort it, iterate through those indices to do access?
    #
    # safety: somehow check that your sorting setup hasn't changed
    def __init__(self, source):
        self._sourceIter = iter(source.points)
        self._currGroup = None
        self._index = 0

    def __iter__(self):
        return self

    def next(self):
        """
        Get next non zero value.
        """
        while True:
            try:
                value = self._currGroup[self._index]
                self._index += 1

                if value != 0:
                    return value
            except Exception:
                self._currGroup = next(self._sourceIter)
                self._index = 0

    def __next__(self):
        return self.next()
