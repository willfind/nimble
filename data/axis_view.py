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
        if self.axis == 'point':
            start = self.source._pStart
            end = self.source._pEnd
        else:
            start = self.source._fStart
            end = self.source._fEnd

        if not getattr(self.source._source, self.axis + 'NamesInverse'):
            self.source._source._setAllDefault(self.axis)
        names = getattr(self.source._source, self.axis + 'NamesInverse')

        return names[start:end]

    def _getName(self, index):
        return self._getNames()[index]

    def _getIndex(self, name):
        if self.axis == 'point':
            start = self.source._pStart
            end = self.source._pEnd
            possible = self.source._source.points.getIndex(name)
        else:
            start = self.source._fStart
            end = self.source._fEnd
            possible = self.source._source.features.getIndex(name)
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

