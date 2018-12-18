"""
Method implementations and helpers acting specifically on features in a
DataFrame object.
"""
from __future__ import absolute_import
from __future__ import division

import UML
from UML.exceptions import ArgumentException
from .axis import Axis
from .dataframeAxis import DataFrameAxis
from .features import Features

pd = UML.importModule('pandas')
if pd:
    import pandas as pd

class DataFrameFeatures(DataFrameAxis, Axis, Features):
    """
    DataFrame method implementations performed on the feature axis.

    Parameters
    ----------
    source : UML data object
        The object containing features data.
    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'feature'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(DataFrameFeatures, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

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

    def _transform_implementation(self, function, included):
        for j, f in enumerate(self):
            if included is not None and j not in included:
                continue
            currRet = function(f)
            # currRet might return an ArgumentException with a message which needs to be
            # formatted with the axis and current index before being raised
            if isinstance(currRet, ArgumentException):
                currRet.value = currRet.value.format('feature', j)
                raise currRet
            if len(currRet) != len(self.source.points):
                msg = "function must return an iterable with as many elements "
                msg += "as points in this object"
                raise ArgumentException(msg)

            self.source.data.iloc[:, j] = currRet

    def _flattenToOne_implementation(self):
        numElements = len(self.source.points) * len(self.source.features)
        self.source.data = pd.DataFrame(
            self.source.data.values.reshape((numElements, 1), order='F'))

    def _unflattenFromOne_implementation(self, divideInto):
        numFeatures = divideInto
        numPoints = len(self.source.points) // numFeatures
        self.source.data = pd.DataFrame(
            self.source.data.values.reshape((numPoints, numFeatures),
                                            order='F'))
