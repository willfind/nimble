"""

"""
from __future__ import absolute_import

from .axis import Axis
from .sparseAxis import SparseAxis

class SparseFeatures(SparseAxis, Axis):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'feature'
        super(SparseFeatures, self).__init__()
