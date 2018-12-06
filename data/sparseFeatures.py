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
    def __init__(self, source):
        self.source = source
        self.axis = 'feature'
        kwds = {}
        kwds['source'] = self.source
        kwds['axis'] = self.axis
        super(SparseFeatures, self).__init__(**kwds)
