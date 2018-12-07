"""

"""
from __future__ import absolute_import

from .axis import Axis
from .dataframeAxis import DataFrameAxis

class DataFramePoints(DataFrameAxis, Axis):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'point'
        super(DataFramePoints, self).__init__()
