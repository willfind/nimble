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
    def __init__(self, source):
        self.source = source
        self.axis = 'point'
        super(MatrixPoints, self).__init__()

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
