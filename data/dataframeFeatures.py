"""

"""
from __future__ import absolute_import

import UML
from .axis import Axis
from .dataframeAxis import DataFrameAxis
from .features import Features

pd = UML.importModule('pandas')
if pd:
    import pandas as pd

class DataFrameFeatures(DataFrameAxis, Axis, Features):
    """

    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'feature'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(DataFrameFeatures, self).__init__(**kwds)

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        startData = self.source.data.iloc[:, :insertBefore]
        endData = self.source.data.iloc[:, insertBefore:]
        self.source.data = pd.concat((startData, toAdd.data, endData), axis=1)
        self.source._updateName(axis='feature')
