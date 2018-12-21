"""

"""
from .axis import Axis

class AxisView(Axis):
    """

    """
    def __init__(self, source, axis, **kwds):
        kwds['source'] = source
        kwds['axis'] = axis
        super(AxisView, self).__init__(**kwds)

    def _getNames(self):
        if self.axis == 'point':
            start = self.source._pStart
            end = self.source._pEnd
            names = self.source._source.pointNamesInverse
        else:
            start = self.source._fStart
            end = self.source._fEnd
            names = self.source._source.featureNamesInverse

        if names is None:
            self.source._source._setAllDefault(self.axis)

        return names[start:end]

    def _getName(self, index):
        return self._getNames()[index]
