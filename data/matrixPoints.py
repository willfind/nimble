"""

"""
from __future__ import absolute_import

from .axis import Axis
from .matrixAxis import MatrixAxis

class MatrixPoints(MatrixAxis, Axis):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'point'
        super(MatrixPoints, self).__init__()
