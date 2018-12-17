"""

"""
from __future__ import absolute_import

import numpy

from .axis import Axis
from .matrixAxis import MatrixAxis
from .points import Points

class MatrixPoints(MatrixAxis, Axis, Points):
    """

    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'point'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(MatrixPoints, self).__init__(**kwds)

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the points from the toAdd object below the provided index
        in this object, the remaining points from this object will
        continue below the inserted points.
        """
        startData = self.source.data[:insertBefore, :]
        endData = self.source.data[insertBefore:, :]
        self.source.data = numpy.concatenate((startData, toAdd.data, endData),
                                             0)
