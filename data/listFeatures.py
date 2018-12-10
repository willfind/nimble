"""

"""
from __future__ import absolute_import

from .axis import Axis
from .listAxis import ListAxis

class ListFeatures(ListAxis, Axis):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'feature'
        super(ListFeatures, self).__init__()

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        for i in range(self.source.pts):
            startData = self.source.data[i][:insertBefore]
            endData = self.source.data[i][insertBefore:]
            allPointData = startData + list(toAdd.data[i]) + endData
            self.source.data[i] = allPointData
        self.source._numFeatures = self.source._numFeatures + toAdd.fts
