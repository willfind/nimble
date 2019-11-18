"""
Defines a subclass of the Axis object, which serves as the primary
base class for read only axis views of data objects.
"""

from __future__ import absolute_import

from nimble.utility import inheritDocstringsFactory
from .axis import Axis
from .points import Points

@inheritDocstringsFactory(Axis)
class AxisView(Axis):
    """
    Class limiting the Axis class to read-only by disallowing methods
    which could change the data.

    Parameters
    ----------
    base : BaseView
        The BaseView instance that will be queried.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def _getNames(self):
        # _base is always a view object
        if isinstance(self, Points):
            start = self._base._pStart
            end = self._base._pEnd
            if not self._namesCreated():
                self._base._source.points._setAllDefault()
            namesList = self._base._source.pointNamesInverse
        else:
            start = self._base._fStart
            end = self._base._fEnd
            if not self._namesCreated():
                self._base._source.features._setAllDefault()
            namesList = self._base._source.featureNamesInverse

        return namesList[start:end]

    def _getName(self, index):
        return self._getNames()[index]

    def _getIndices(self, names):
        return [self._getIndex(n) for n in names]

    def _getIndexByName(self, name):
        # _base is always a view object
        if isinstance(self, Points):
            start = self._base._pStart
            end = self._base._pEnd
            possible = self._base._source.points.getIndex(name)
        else:
            start = self._base._fStart
            end = self._base._fEnd
            possible = self._base._source.features.getIndex(name)
        if start <= possible < end:
            return possible - start
        else:
            raise KeyError(name)

    def _namesCreated(self):
        # _base is always a view object
        if isinstance(self, Points):
            return not self._base._source.pointNames is None
        else:
            return not self._base._source.featureNames is None
