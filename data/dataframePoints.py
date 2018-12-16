"""

"""
from __future__ import absolute_import

import UML
from .axis import Axis
from .dataframeAxis import DataFrameAxis
from .points import Points

pd = UML.importModule('pandas')
if pd:
    import pandas as pd

class DataFramePoints(DataFrameAxis, Axis, Points):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'point'
        super(DataFramePoints, self).__init__()

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the points from the toAdd object below the provided index
        in this object, the remaining points from this object will
        continue below the inserted points.
        """
        startData = self.source.data.iloc[:insertBefore, :]
        endData = self.source.data.iloc[insertBefore:, :]
        self.source.data = pd.concat((startData, toAdd.data, endData), axis=0)
        self.source._updateName(axis='point')
