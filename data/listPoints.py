"""

"""
from __future__ import absolute_import

from .axis import Axis
from .listAxis import ListAxis

class ListPoints(ListAxis, Axis):
    """

    """
    def __init__(self, source):
        self.source = source
        self.axis = 'point'
        super(ListPoints, self).__init__()
