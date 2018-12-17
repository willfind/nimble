"""
Method implementations and helpers acting specifically on features in a
List object.
"""
from __future__ import absolute_import

from UML.exceptions import ArgumentException
from .axis import Axis
from .listAxis import ListAxis
from .features import Features

class ListFeatures(ListAxis, Axis, Features):
    """
    List method implementations performed on the feature axis.

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
        super(ListFeatures, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        for i in range(len(self.source.points)):
            startData = self.source.data[i][:insertBefore]
            endData = self.source.data[i][insertBefore:]
            allPointData = startData + list(toAdd.data[i]) + endData
            self.source.data[i] = allPointData
        self.source._numFeatures += len(toAdd.features)

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

            for i in range(len(self.source.points)):
                self.source.data[i][j] = currRet[i]
