"""
Method implementations and helpers acting specifically on points in a
Sparse object.
"""

import numpy

import nimble
from nimble._utility import scipy
from .axis_view import AxisView
from .sparseAxis import SparseAxis
from .points import Points
from .points_view import PointsView

class SparsePoints(SparseAxis, Points):
    """
    Sparse method implementations performed on the points axis.

    Parameters
    ----------
    base : Sparse
        The Sparse instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    # def _flattenToOne_implementation(self):
    #     self._base._sortInternal('point')
    #     pLen = len(self._base.features)
    #     numElem = len(self._base.points) * len(self._base.features)
    #     data = self._base.data.data
    #     row = self._base.data.row
    #     col = self._base.data.col
    #     for i in range(len(data)):
    #         if row[i] > 0:
    #             col[i] += (row[i] * pLen)
    #             row[i] = 0
    #
    #     self._base.data = coo_matrix((data, (row, col)), (1, numElem))
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     # only one feature, so both sorts are the same order
    #     if self._base._sorted is None:
    #         self._base._sortInternal('point')
    #
    #     numPoints = divideInto
    #     numFeatures = len(self._base.features) // numPoints
    #     newShape = (numPoints, numFeatures)
    #     data = self._base.data.data
    #     row = self._base.data.row
    #     col = self._base.data.col
    #     for i in range(len(data)):
    #         # must change the row entry before modifying the col entry
    #         row[i] = col[i] / numFeatures
    #         col[i] = col[i] % numFeatures
    #
    #     self._base.data = coo_matrix((data, (row, col)), newShape)
    #     self._base._sorted = 'point'

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        if self._base._sorted is None:
            self._base._sortInternal('point')
        data = self._base.data.data
        row = self._base.data.row
        col = self._base.data.col
        tmpData = []
        tmpRow = []
        tmpCol = []
        collapseNames = [self._base.features.getName(idx)
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
        self._base.data = scipy.sparse.coo_matrix(
            (tmpData, (tmpRow, tmpCol)), shape=(numRetPoints, numRetFeatures))
        self._base._sorted = None

    def _combineByExpandingFeatures_implementation( self, uniqueDict, namesIdx,
                                                   uniqueNames, numRetFeatures,
                                                   numExpanded):
        tmpData = []
        tmpRow = []
        tmpCol = []
        numNewFts = len(uniqueNames) * numExpanded
        for idx, point in enumerate(uniqueDict):
            tmpPoint = list(point[:namesIdx])
            for name in uniqueNames:
                if name in uniqueDict[point]:
                    tmpPoint.extend(uniqueDict[point][name])
                else:
                    tmpPoint.extend([numpy.nan] * numExpanded)
            tmpPoint.extend(point[namesIdx:])
            tmpData.extend(tmpPoint)
            tmpRow.extend([idx for _ in range(len(point) + numNewFts)])
            tmpCol.extend([i for i in range(numRetFeatures)])

        tmpData = numpy.array(tmpData, dtype=numpy.object_)
        shape = (len(uniqueDict), numRetFeatures)
        self._base.data = scipy.sparse.coo_matrix((tmpData, (tmpRow, tmpCol)),
                                                  shape=shape)
        self._base._sorted = None

class SparsePointsView(PointsView, AxisView, SparsePoints):
    """
    Limit functionality of SparsePoints to read-only.

    Parameters
    ----------
    base : SparseView
        The SparseView instance that will be queried.
    """

    #########################
    # Query implementations #
    #########################

    def _unique_implementation(self):
        unique = self._base.copy(to='Sparse')
        return unique.points._unique_implementation()

    def _repeat_implementation(self, totalCopies, copyValueByValue):
        copy = self._base.copy(to='Sparse')
        return copy.points._repeat_implementation(totalCopies,
                                                  copyValueByValue)
