"""
TODO
"""
from __future__ import absolute_import

from .axis import Axis
from .dataFrameAxis import DataFrameAxis

class DataFramePoints(DataFrameAxis, Axis):
    """
    TODO
    """
    def __init__(self):
        self.source = source
        self.axis = 'point'
        super(DataFramePoints, self).__init__()
        