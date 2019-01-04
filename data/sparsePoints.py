"""
Method implementations and helpers acting specifically on points in a
Sparse object.
"""
from __future__ import absolute_import

import numpy

import UML
from .axis import Axis
from .axis_view import AxisView
from .sparseAxis import SparseAxis
from .points import Points

scipy = UML.importModule('scipy')
if scipy is not None:
    from scipy.sparse import coo_matrix

class SparsePoints(SparseAxis, Axis, Points):
    """
    Sparse method implementations performed on the points axis.

    Parameters
    ----------
    source : UML data object
        The object containing the points data.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, source, **kwds):
        self._source = source
        self._axis = 'point'
        kwds['axis'] = self._axis
        kwds['source'] = self._source
        super(SparsePoints, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the points from the toAdd object below the provided index
        in this object, the remaining points from this object will
        continue below the inserted points.
        """
        self._source._sortInternal('point')
        newData = []
        newRow = []
        newCol = []
        # add original data until insert location
        for i, row in enumerate(self._source.data.row):
            if row < insertBefore:
                newRow.append(row)
                newCol.append(self._source.data.col[i])
                newData.append(self._source.data.data[i])
            else:
                break
        splitLength = len(newRow)
        # add inserted data with adjusted row
        for i, row in enumerate(toAdd.data.row):
            newRow.append(row + insertBefore)
            newCol.append(toAdd.data.col[i])
            newData.append(toAdd.data.data[i])
        # add remaining original data with adjusted row
        for i, row in enumerate(self._source.data.row[splitLength:]):
            newRow.append(row + len(toAdd.points))
            newCol.append(self._source.data.col[splitLength:][i])
            newData.append(self._source.data.data[splitLength:][i])
        # handle conflicts between original dtype and inserted data
        try:
            newData = numpy.array(newData, dtype=self._source.data.dtype)
        except ValueError:
            newData = numpy.array(newData, dtype=numpy.object_)
        numNewRows = len(self._source.points) + len(toAdd.points)
        shape = (numNewRows, len(self._source.features))
        self._source.data = coo_matrix((newData, (newRow, newCol)),
                                       shape=shape)
        self._source._sorted = None

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

class SparsePointsView(AxisView, SparsePoints, SparseAxis, Axis, Points):
    """
    Limit functionality of SparsePoints to read-only
    """
    def __init__(self, source, **kwds):
        kwds['source'] = source
        kwds['axis'] = 'point'
        super(SparsePointsView, self).__init__(**kwds)

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

    def _unique_implementation(self):
        unique = self._source.copyAs('Sparse')
        return unique.points._unique_implementation()

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
