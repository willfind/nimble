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
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, source, **kwds):
        self._source = source
        self._axis = 'feature'
        kwds['axis'] = self._axis
        kwds['source'] = self._source
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
        for i in range(len(self._source.points)):
            startData = self._source.data[i][:insertBefore]
            endData = self._source.data[i][insertBefore:]
            allPointData = startData + list(toAdd.data[i]) + endData
            self._source.data[i] = allPointData
        self._source._numFeatures += len(toAdd.features)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self._source.features):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)
            # currRet might return an ArgumentException with a message which
            # needs to be formatted with the axis and current index before
            # being raised
            if isinstance(currRet, ArgumentException):
                currRet.value = currRet.value.format('feature', j)
                raise currRet
            if len(currRet) != len(self._source.points):
                msg = "function must return an iterable with as many elements "
                msg += "as points in this object"
                raise ArgumentException(msg)

            for i in range(len(self._source.points)):
                self._source.data[i][j] = currRet[i]

    # def _flattenToOne_implementation(self):
    #     result = []
    #     for i in range(len(self._source.features)):
    #         for p in self._source.data:
    #             result.append([p[i]])
    #
    #     self._source.data = result
    #     self._source._numFeatures = 1
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     result = []
    #     numFeatures = divideInto
    #     numPoints = len(self._source.points) // numFeatures
    #     # reconstruct the shape we want, point by point. We access the
    #     # singleton values from the current data in an out of order iteration
    #     for i in range(numPoints):
    #         temp = []
    #         for j in range(i, len(self._source.points), numPoints):
    #             temp += self._source.data[j]
    #         result.append(temp)
    #
    #     self._source.data = result
    #     self._source._numFeatures = numFeatures

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

class ListFeaturesView(AxisView, ListFeatures, ListAxis, Axis, Features):
    """
    Limit functionality of ListFeatures to read-only
    """
    def __init__(self, source, **kwds):
        kwds['source'] = source
        kwds['axis'] = 'feature'
        super(ListFeaturesView, self).__init__(**kwds)

class nzIt(object):
    """
    Non-zero iterator to return when iterating through each feature.
    """
    def __init__(self, source):
        self._source = source
        self._pIndex = 0
        self._pStop = len(source.points)
        self._fIndex = 0
        self._fStop = len(source.features)

    def __iter__(self):
        return self

    def next(self):
        """
        Get next non zero value.
        """
        while self._fIndex < self._fStop:
            value = self._source.data[self._pIndex][self._fIndex]

            self._pIndex += 1
            if self._pIndex >= self._pStop:
                self._pIndex = 0
                self._fIndex += 1

            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
