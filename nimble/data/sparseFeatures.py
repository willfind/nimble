"""
Method implementations and helpers acting specifically on features in a
Sparse object.
"""

from __future__ import absolute_import

import numpy

import nimble
from nimble.utility import ImportModule
from .axis_view import AxisView
from .sparseAxis import SparseAxis
from .features import Features
from .features_view import FeaturesView

scipy = ImportModule('scipy')

class SparseFeatures(SparseAxis, Features):
    """
    Sparse method implementations performed on the feature axis.

    Parameters
    ----------
    base : Sparse
        The Sparse instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    # def _flattenToOne_implementation(self):
    #     self._base._sortInternal('feature')
    #     fLen = len(self._base.points)
    #     numElem = len(self._base.points) * len(self._base.features)
    #     data = self._base.data.data
    #     row = self._base.data.row
    #     col = self._base.data.col
    #     for i in range(len(data)):
    #         if col[i] > 0:
    #             row[i] += (col[i] * fLen)
    #             col[i] = 0
    #
    #     self._base.data = coo_matrix((data, (row, col)), (numElem, 1))
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     # only one feature, so both sorts are the same order
    #     if self._base._sorted is None:
    #         self._base._sortInternal('feature')
    #
    #     numFeatures = divideInto
    #     numPoints = len(self._base.points) // numFeatures
    #     newShape = (numPoints, numFeatures)
    #     data = self._base.data.data
    #     row = self._base.data.row
    #     col = self._base.data.col
    #     for i in range(len(data)):
    #         # must change the col entry before modifying the row entry
    #         col[i] = row[i] / numPoints
    #         row[i] = row[i] % numPoints
    #
    #     self._base.data = coo_matrix((data, (row, col)), newShape)
    #     self._base._sorted = 'feature'

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        keep = self._base.data.col != featureIndex
        tmpData = self._base.data.data[keep]
        tmpRow = self._base.data.row[keep]
        tmpCol = self._base.data.col[keep]

        shift = tmpCol > featureIndex
        tmpCol[shift] = tmpCol[shift] + numResultingFts - 1

        for idx in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[idx])
            tmpData = numpy.concatenate((tmpData, newFeat))
            newRows = [i for i in range(len(self._base.points))]
            tmpRow = numpy.concatenate((tmpRow, newRows))
            newCols = [featureIndex + idx for _
                       in range(len(self._base.points))]
            tmpCol = numpy.concatenate((tmpCol, newCols))

        tmpData = numpy.array(tmpData, dtype=numpy.object_)
        shape = (len(self._base.points), numRetFeatures)
        self._base.data = scipy.sparse.coo_matrix((tmpData, (tmpRow, tmpCol)),
                                                  shape=shape)
        self._base._sorted = None

class SparseFeaturesView(FeaturesView, AxisView, SparseFeatures):
    """
    Limit functionality of SparseFeatures to read-only.

    Parameters
    ----------
    base : SparseView
        The SparseView instance that will be queried.
    """

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._base)

    def _unique_implementation(self):
        unique = self._base.copy(to='Sparse')
        return unique.features._unique_implementation()

    def _repeat_implementation(self, totalCopies, copyValueByValue):
        copy = self._base.copy(to='Sparse')
        return copy.features._repeat_implementation(totalCopies,
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
            except (TypeError, IndexError):
                self._currGroup = next(self._sourceIter)
                self._index = 0

    def __next__(self):
        return self.next()
