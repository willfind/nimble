"""

"""
from __future__ import absolute_import

from .axis import Axis
from .dataframeAxis import DataFrameAxis

class DataFrameFeatures(DataFrameAxis, Axis):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'feature'
        super(DataFrameFeatures, self).__init__()
