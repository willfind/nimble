"""
TODO
"""
from __future__ import absolute_import

from .axis import Axis
from .listAxis import ListAxis

class ListFeatures(ListAxis, Axis):
    """
    TODO
    """
    def __init__(self):
        self.source = source
        self.axis = 'feature'
        super(ListFeatures, self).__init__()
