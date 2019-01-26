"""
Defines a subclass of the Axis object, which serves as the primary
base class for read only axis views of data objects.
"""
from __future__ import absolute_import

from .axis import Axis
from .points import Points
from .base_view import readOnlyException

# TODO inherit docstrings

class AxisView(Axis):
    """
    Class defining read only view objects, which have the same api as a
    normal Axis object, but disallow all methods which could change the
    data.

    Parameters
    ----------
    source : UML data object
        The UML object that this is a view into.
    axis : str
        Either 'point' or 'feature'.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, source, axis, **kwds):
        kwds['source'] = source
        kwds['axis'] = axis
        super(AxisView, self).__init__(**kwds)

    def _getNames(self):
        if isinstance(self, Points):
            start = self._source._pStart
            end = self._source._pEnd
        else:
            start = self._source._fStart
            end = self._source._fEnd

        if not self._namesCreated():
            self._source._source._setAllDefault(self._axis)
        if isinstance(self, Points):
            namesList = self._source._source.pointNamesInverse
        else:
            namesList = self._source._source.featureNamesInverse

        return namesList[start:end]

    def _getName(self, index):
        return self._getNames()[index]

    def _getIndex(self, name):
        if isinstance(self, Points):
            start = self._source._pStart
            end = self._source._pEnd
            possible = self._source._source.points.getIndex(name)
        else:
            start = self._source._fStart
            end = self._source._fEnd
            possible = self._source._source.features.getIndex(name)
        if possible >= start and possible < end:
            return possible - start
        else:
            raise KeyError(name)

    def _getIndices(self, names):
        return [self._getIndex(n) for n in names]

    ####################################
    # Low Level Operations, Disallowed #
    ####################################

    def setName(self, oldIdentifier, newName):
        readOnlyException('setName')

    def setNames(self, assignments=None):
        readOnlyException('setNames')

    ##############################################################
    #   Subclass implemented structural manipulation functions   #
    ##############################################################

    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False):
        readOnlyException('extract')

    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False):
        readOnlyException('delete')

    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False):
        readOnlyException('retain')

    def add(self, toAdd, insertBefore=None):
        readOnlyException('add')

    def shuffle(self):
        readOnlyException('shuffle')

    def sort(self, sortBy=None, sortHelper=None):
        readOnlyException('sort')

    def transform(self, function, limitTo=None):
        readOnlyException('transform')

    ##############################################################
    #   Subclass implemented high level manipulation functions   #
    ##############################################################

    def fill(self, match, fill, arguments=None, limitTo=None,
             returnModified=False):
        readOnlyException('fill')

    def normalize(self, subtract=None, divide=None, applyResultTo=None):
        readOnlyException('normalize')
