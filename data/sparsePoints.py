"""

"""
from __future__ import absolute_import

import numpy

import UML
from .axis import Axis
from .sparseAxis import SparseAxis
from .points import Points

scipy = UML.importModule('scipy')
if scipy is not None:
    from scipy.sparse import coo_matrix

class SparsePoints(SparseAxis, Axis, Points):
    """

    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'point'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(SparsePoints, self).__init__(**kwds)

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the points from the toAdd object below the provided index
        in this object, the remaining points from this object will
        continue below the inserted points.
        """
        self.source._sortInternal('point')
        newData = []
        newRow = []
        newCol = []
        # add original data until insert location
        for i, row in enumerate(self.source.data.row):
            if row < insertBefore:
                newRow.append(row)
                newCol.append(self.source.data.col[i])
                newData.append(self.source.data.data[i])
            else:
                break
        splitLength = len(newRow)
        # add inserted data with adjusted row
        for i, row in enumerate(toAdd.data.row):
            newRow.append(row + insertBefore)
            newCol.append(toAdd.data.col[i])
            newData.append(toAdd.data.data[i])
        # add remaining original data with adjusted row
        for i, row in enumerate(self.source.data.row[splitLength:]):
            newRow.append(row + len(toAdd.points))
            newCol.append(self.source.data.col[splitLength:][i])
            newData.append(self.source.data.data[splitLength:][i])
        # handle conflicts between original dtype and inserted data
        try:
            newData = numpy.array(newData, dtype=self.source.data.dtype)
        except ValueError:
            newData = numpy.array(newData, dtype=numpy.object_)
        numNewRows = len(self.source.points) + len(toAdd.points)
        shape = (numNewRows, len(self.source.features))
        self.source.data = coo_matrix((newData, (newRow, newCol)),
                                      shape=shape)
        self.source._sorted = None
