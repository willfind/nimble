"""
TODO
"""
from __future__ import absolute_import

class Axis(object):
    """
    TODO
    """
    def __init__(self):
        super(Axis, self).__init__()

    def __getitem__(self, key):
        if self.axis == 'point':
            key = (key, slice(None, None, None))
        else:
            key = (slice(None, None, None), key)
        return self.__class__(self.source.__getitem__(key))
