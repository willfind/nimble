"""
Defines a subclass of the Axis object, which serves as the primary
base class for read only axis views of data objects.
"""

from __future__ import absolute_import

from UML.docHelpers import inheritDocstringsFactory
from .axis import Axis
from .points import Points

@inheritDocstringsFactory(Axis)
class AxisView(Axis):
    """
    Class defining read only view objects, which have the same api as a
    normal Axis object, but disallow all methods which could change the
    data.

    Parameters
    ----------
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def _getNames(self):
        if isinstance(self, Points):
            start = self._source._pStart
            end = self._source._pEnd
            if not self._namesCreated():
                self._source._source.points._setAllDefault()
            namesList = self._source._source.pointNamesInverse
        else:
            start = self._source._fStart
            end = self._source._fEnd
            if not self._namesCreated():
                self._source._source.features._setAllDefault()
            namesList = self._source._source.featureNamesInverse

        return namesList[start:end]

    def _getName(self, index):
        return self._getNames()[index]

    def _getIndices(self, names):
        return [self._getIndex(n) for n in names]

    def _getIndexByName(self, name):
        if isinstance(self, Points):
            start = self._source._pStart
            end = self._source._pEnd
            possible = self._source._source.points.getIndex(name)
        else:
            start = self._source._fStart
            end = self._source._fEnd
            possible = self._source._source.features.getIndex(name)
        if start <= possible < end:
            return possible - start
        else:
            raise KeyError(name)

    def _namesCreated(self):
        if isinstance(self, Points):
            return not self._source._source.pointNames is None
        else:
            return not self._source._source.featureNames is None

    def _getIndices(self, names):
        return [self._getIndex(n) for n in names]
