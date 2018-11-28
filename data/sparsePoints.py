"""
TODO
"""
from __future__ import absolute_import

from .axis import Axis
from .sparseAxis import SparseAxis

class SparsePoints(SparseAxis, Axis):
    """
    TODO
    """
    def __init__(self):
        self.source = source
        self.axis = 'point'
        super(SparsePoints, self).__init__()
