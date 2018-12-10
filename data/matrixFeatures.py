"""

"""
from __future__ import absolute_import

import numpy

from .axis import Axis
from .matrixAxis import MatrixAxis

class MatrixFeatures(MatrixAxis, Axis):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'feature'
        super(MatrixFeatures, self).__init__()

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        startData = self.source.data[:, :insertBefore]
        endData = self.source.data[:, insertBefore:]
        self.source.data = numpy.concatenate((startData, toAdd.data, endData),
                                             1)
