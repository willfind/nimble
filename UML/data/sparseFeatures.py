"""
Method implementations and helpers acting specifically on features in a
Sparse object.
"""

from __future__ import absolute_import

import numpy

import UML as nimble
from .axis_view import AxisView
from .sparseAxis import SparseAxis
from .features import Features
from .features_view import FeaturesView

scipy = nimble.importModule('scipy')
if scipy is not None:
    from scipy.sparse import coo_matrix

class SparseFeatures(SparseAxis, Features):
    """
    Sparse method implementations performed on the feature axis.

    Parameters
    ----------
    source : nimble data object
        The object containing point and feature data.
    """

    ##############################
    # Structural implementations #
    ##############################

    # def _flattenToOne_implementation(self):
    #     self._source._sortInternal('feature')
    #     fLen = len(self._source.points)
    #     numElem = len(self._source.points) * len(self._source.features)
    #     data = self._source.data.data
    #     row = self._source.data.row
    #     col = self._source.data.col
    #     for i in range(len(data)):
    #         if col[i] > 0:
    #             row[i] += (col[i] * fLen)
    #             col[i] = 0
    #
    #     self._source.data = coo_matrix((data, (row, col)), (numElem, 1))
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     # only one feature, so both sorts are the same order
    #     if self._source._sorted is None:
    #         self._source._sortInternal('feature')
    #
    #     numFeatures = divideInto
    #     numPoints = len(self._source.points) // numFeatures
    #     newShape = (numPoints, numFeatures)
    #     data = self._source.data.data
    #     row = self._source.data.row
    #     col = self._source.data.col
    #     for i in range(len(data)):
    #         # must change the col entry before modifying the row entry
    #         col[i] = row[i] / numPoints
    #         row[i] = row[i] % numPoints
    #
    #     self._source.data = coo_matrix((data, (row, col)), newShape)
    #     self._source._sorted = 'feature'

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        keep = self._source.data.col != featureIndex
        tmpData = self._source.data.data[keep]
        tmpRow = self._source.data.row[keep]
        tmpCol = self._source.data.col[keep]

        shift = tmpCol > featureIndex
        tmpCol[shift] = tmpCol[shift] + numResultingFts - 1

        for idx in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[idx])
            tmpData = numpy.concatenate((tmpData, newFeat))
            newRows = [i for i in range(len(self._source.points))]
            tmpRow = numpy.concatenate((tmpRow, newRows))
            newCols = [featureIndex + idx for _
                       in range(len(self._source.points))]
            tmpCol = numpy.concatenate((tmpCol, newCols))

        tmpData = numpy.array(tmpData, dtype=numpy.object_)
        shape = (len(self._source.points), numRetFeatures)
        self._source.data = coo_matrix((tmpData, (tmpRow, tmpCol)),
                                       shape=shape)
        self._source._sorted = None

class SparseFeaturesView(FeaturesView, AxisView, SparseFeatures):
    """
    Limit functionality of SparseFeatures to read-only
    """

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

    def _unique_implementation(self):
        unique = self._source.copy(to='Sparse')
        return unique.features._unique_implementation()

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
        self._sourceIter = iter(source.features)
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
