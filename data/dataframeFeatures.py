"""
TODO
"""
from __future__ import absolute_import

from .axis import Axis
from .dataframeAxis import DataFrameAxis

class DataFrameFeatures(DataFrameAxis, Axis):
    """
    TODO
    """
    def __init__(self, source):
        self.source = source
        self.axis = 'feature'
        kwds = {}
        kwds['source'] = self.source
        kwds['axis'] = self.axis
        super(DataFrameFeatures, self).__init__(**kwds)
