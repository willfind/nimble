"""
TODO
"""
from __future__ import absolute_import

from .axis import Axis
from .sparseAxis import SparseAxis

class SparseFeatures(SparseAxis, Axis):
    """
    TODO
    """
    def __init__(self):
        self.source = source
        self.axis = 'feature'
        super(SparseFeatures, self).__init__()
