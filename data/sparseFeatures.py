"""
Method implementations and helpers acting specifically on features in a
Sparse object.
"""
from __future__ import absolute_import

import numpy

import UML
from .axis import Axis
from .sparseAxis import SparseAxis
from .features import Features

scipy = UML.importModule('scipy')
if scipy is not None:
    from scipy.sparse import coo_matrix

class SparseFeatures(SparseAxis, Axis, Features):
    """
    Sparse method implementations performed on the feature axis.

    Parameters
    ----------
    source : UML data object
        The object containing features data.
    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'feature'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(SparseFeatures, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        self.source._sortInternal('feature')
        newData = []
        newRow = []
        newCol = []
        # add original data until insert location
        for i, col in enumerate(self.source.data.col):
            if col < insertBefore:
                newRow.append(self.source.data.row[i])
                newCol.append(col)
                newData.append(self.source.data.data[i])
            else:
                break
        # add inserted data with adjusted col
        splitLength = len(newCol)
        for i, col in enumerate(toAdd.data.col):
            newRow.append(toAdd.data.row[i])
            newCol.append(col + insertBefore)
            newData.append(toAdd.data.data[i])
        # add remaining original data with adjusted col
        for i, col in enumerate(self.source.data.col[splitLength:]):
            newRow.append(self.source.data.row[splitLength:][i])
            newCol.append(col + len(toAdd.features))
            newData.append(self.source.data.data[splitLength:][i])
        # handle conflicts between original dtype and inserted data
        try:
            newData = numpy.array(newData, dtype=self.source.data.dtype)
        except ValueError:
            newData = numpy.array(newData, dtype=numpy.object_)

        numNewCols = len(self.source.features) + len(toAdd.features)
        shape = (len(self.source.points), numNewCols)
        self.source.data = coo_matrix((newData, (newRow, newCol)),
                                      shape=shape)
        self.source._sorted = None
