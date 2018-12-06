"""
TODO
"""
from __future__ import absolute_import

from .axis import Axis
from .matrixAxis import MatrixAxis

class MatrixFeatures(MatrixAxis, Axis):
    """
    TODO
    """
    def __init__(self, source):
        self.source = source
        self.axis = 'feature'
        kwds = {}
        kwds['source'] = self.source
        kwds['axis'] = self.axis
        super(MatrixFeatures, self).__init__(**kwds)
