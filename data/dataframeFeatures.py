"""

"""
from __future__ import absolute_import

import UML
from .axis import Axis
from .dataframeAxis import DataFrameAxis

pd = UML.importModule('pandas')
if pd:
    import pandas as pd

class DataFrameFeatures(DataFrameAxis, Axis):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'feature'
        super(DataFrameFeatures, self).__init__()

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
