"""
Method implementations and helpers acting specifically on points in a
DataFrame object.
"""
from __future__ import absolute_import

import UML
from UML.exceptions import ArgumentException
from .axis import Axis
from .dataframeAxis import DataFrameAxis
from .points import Points

pd = UML.importModule('pandas')
if pd:
    import pandas as pd

class DataFramePoints(DataFrameAxis, Axis, Points):
    """
    DataFrame method implementations performed on the points axis.

    Parameters
    ----------
    source : UML data object
        The object containing the points data.
    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'point'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(DataFramePoints, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

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

    def _transform_implementation(self, function, included):
        for i, p in enumerate(self):
            if included is not None and i not in included:
                continue
            currRet = function(p)
            # currRet might return an ArgumentException with a message which needs to be
            # formatted with the axis and current index before being raised
            if isinstance(currRet, ArgumentException):
                currRet.value = currRet.value.format('point', i)
                raise currRet
            if len(currRet) != len(self.source.features):
                msg = "function must return an iterable with as many elements "
                msg += "as features in this object"
                raise ArgumentException(msg)

            self.source.data.iloc[i, :] = currRet
