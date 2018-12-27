"""

"""
from .axis import Axis
from .base_view import readOnlyException

class AxisView(Axis):
    """

    """
    def __init__(self, source, axis, **kwds):
        kwds['source'] = source
        kwds['axis'] = axis
        super(AxisView, self).__init__(**kwds)

    def _getNames(self):
        if self._axis == 'point':
            start = self._source._pStart
            end = self._source._pEnd
        else:
            start = self._source._fStart
            end = self._source._fEnd

        if not getattr(self._source._source, self._axis + 'NamesInverse'):
            self._source._source._setAllDefault(self._axis)
        names = getattr(self._source._source, self._axis + 'NamesInverse')

        return names[start:end]

    def _getName(self, index):
        return self._getNames()[index]

    def _getIndex(self, name):
        if self._axis == 'point':
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
            raise KeyError()

    def _getIndices(self, names):
        return [self._getIndex(n) for n in names]

    ####################################
    # Low Level Operations, Disallowed #
    ####################################

    def setName(self, oldIdentifier, newName):
        readOnlyException('setName')

    def setNames(self, oldIdentifier, newName):
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

