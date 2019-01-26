"""
Method implementations and helpers acting specifically on points in a
List object.
"""
from __future__ import absolute_import

from UML.exceptions import InvalidArgumentValue
from .axis import Axis
from .axis_view import AxisView
from .listAxis import ListAxis
from .points import Points

class ListPoints(ListAxis, Axis, Points):
    """
    List method implementations performed on the points axis.

    Parameters
    ----------
    source : UML data object
        The object containing the points data.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, source, **kwds):
        self._source = source
        self._axis = 'point'
        kwds['axis'] = self._axis
        kwds['source'] = self._source
        super(ListPoints, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the points from the toAdd object below the provided index
        in this object, the remaining points from this object will
        continue below the inserted points.
        """
        insertedLength = len(self._source.points) + len(toAdd.points)
        insertRange = range(insertBefore, insertBefore + len(toAdd.points))
        insertIndex = 0
        selfIndex = 0
        allData = []
        for pointIndex in range(insertedLength):
            if pointIndex in insertRange:
                allData.append(toAdd.data[insertIndex])
                insertIndex += 1
            else:
                allData.append(self._source.data[selfIndex])
                selfIndex += 1
        self._source.data = allData

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)
            # currRet might return an InvalidArgumentValue with a message which
            # needs to be formatted with the axis and current index before
            # being raised
            if isinstance(currRet, InvalidArgumentValue):
                currRet.value = currRet.value.format('point', i)
                raise currRet
            if len(currRet) != len(self._source.features):
                msg = "function must return an iterable with as many elements "
                msg += "as features in this object"
                raise InvalidArgumentValue(msg)

            self._source.data[i] = currRet

    # def _flattenToOne_implementation(self):
    #     onto = self._source.data[0]
    #     for _ in range(1, len(self._source.points)):
    #         onto += self._source.data[1]
    #         del self._source.data[1]
    #
    #     self._source._numFeatures = len(onto)
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     result = []
    #     numPoints = divideInto
    #     numFeatures = len(self._source.features) // numPoints
    #     for i in range(numPoints):
    #         temp = self._source.data[0][(i*numFeatures):((i+1)*numFeatures)]
    #         result.append(temp)
    #
    #     self._source.data = result
    #     self._source._numFeatures = numFeatures

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

class ListPointsView(AxisView, ListPoints, ListAxis, Axis, Points):
    """
    Limit functionality of ListPoints to read-only
    """
    def __init__(self, source, **kwds):
        kwds['source'] = source
        kwds['axis'] = 'point'
        super(ListPointsView, self).__init__(**kwds)

class nzIt(object):
    """
    Non-zero iterator to return when iterating through each point.
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
        while self._pIndex < self._pStop:
            value = self._source.data[self._pIndex][self._fIndex]

            self._fIndex += 1
            if self._fIndex >= self._fStop:
                self._fIndex = 0
                self._pIndex += 1

            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
