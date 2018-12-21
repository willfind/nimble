"""
Method implementations and helpers acting specifically on features in a
List object.
"""
from __future__ import absolute_import

from UML.exceptions import ArgumentException
from .axis import Axis
from .axis_view import AxisView
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

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
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

    def _flattenToOne_implementation(self):
        result = []
        for i in range(len(self.source.features)):
            for p in self.source.data:
                result.append([p[i]])

        self.source.data = result
        self.source._numFeatures = 1

    def _unflattenFromOne_implementation(self, divideInto):
        result = []
        numFeatures = divideInto
        numPoints = len(self.source.points) // numFeatures
        # reconstruct the shape we want, point by point. We access the
        # singleton values from the current data in an out of order iteration
        for i in range(numPoints):
            temp = []
            for j in range(i, len(self.source.points), numPoints):
                temp += self.source.data[j]
            result.append(temp)

        self.source.data = result
        self.source._numFeatures = numFeatures

class ListFeaturesView(AxisView, ListFeatures, ListAxis, Axis, Features):
    def __init__(self, source, **kwds):
        kwds['source'] = source
        kwds['axis'] = 'feature'
        super(ListFeaturesView, self).__init__(**kwds)
