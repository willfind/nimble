"""

"""
from __future__ import absolute_import

from .axis import Axis
from .listAxis import ListAxis
from .points import Points

class ListPoints(ListAxis, Axis, Points):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'point'
        super(ListPoints, self).__init__()

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
