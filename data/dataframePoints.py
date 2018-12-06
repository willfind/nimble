"""
TODO
"""
from __future__ import absolute_import

from .axis import Axis
from .dataframeAxis import DataFrameAxis

class DataFramePoints(DataFrameAxis, Axis):
    """
    TODO
    """
    def __init__(self, source):
        self.source = source
        self.axis = 'point'
        kwds = {}
        kwds['source'] = self.source
        kwds['axis'] = self.axis
        super(DataFramePoints, self).__init__(**kwds)
