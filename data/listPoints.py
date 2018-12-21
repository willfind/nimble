"""
Method implementations and helpers acting specifically on points in a
List object.
"""
from __future__ import absolute_import

from UML.exceptions import ArgumentException
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
    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'point'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
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
        insertedLength = len(self.source.points) + len(toAdd.points)
        insertRange = range(insertBefore, insertBefore + len(toAdd.points))
        insertIndex = 0
        selfIndex = 0
        allData = []
        for pointIndex in range(insertedLength):
            if pointIndex in insertRange:
                allData.append(toAdd.data[insertIndex])
                insertIndex += 1
            else:
                allData.append(self.source.data[selfIndex])
                selfIndex += 1
        self.source.data = allData

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
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

            self.source.data[i] = currRet

    def _flattenToOne_implementation(self):
        onto = self.source.data[0]
        for _ in range(1, len(self.source.points)):
            onto += self.source.data[1]
            del self.source.data[1]

        self.source._numFeatures = len(onto)

    def _unflattenFromOne_implementation(self, divideInto):
        result = []
        numPoints = divideInto
        numFeatures = len(self.source.features) // numPoints
        for i in range(numPoints):
            temp = self.source.data[0][(i*numFeatures):((i+1)*numFeatures)]
            result.append(temp)

        self.source.data = result
        self.source._numFeatures = numFeatures

class ListPointsView(AxisView, ListPoints, ListAxis, Axis, Points):
    def __init__(self, source, **kwds):
        kwds['source'] = source
        kwds['axis'] = 'point'
        super(ListPointsView, self).__init__(**kwds)
