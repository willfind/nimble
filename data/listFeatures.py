"""

"""
from __future__ import absolute_import

from .axis import Axis
from .listAxis import ListAxis
from .features import Features

class ListFeatures(ListAxis, Axis, Features):
    """

    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'feature'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(ListFeatures, self).__init__(**kwds)

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
