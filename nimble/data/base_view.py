"""
Defines a subclass of the base data object, which serves as the primary
base class for read only views of data objects.
"""

from __future__ import division
from __future__ import absolute_import
import copy

import nimble
from nimble.utility import inheritDocstringsFactory
from .base import Base
from .dataHelpers import readOnlyException
from .dataHelpers import exceptionDocstringFactory

exceptionDocstring = exceptionDocstringFactory(Base)

@inheritDocstringsFactory(Base)
class BaseView(Base):
    """
    Class limiting the Base class to read-only by disallowing methods
    which could change the data.

    Parameters
    ----------
    source : nimble Base object
        The nimble object that this is a view into.
    pointStart : int
        The inclusive index of the first point this view will have
        access to.
    pointEnd : int
        The EXCLUSIVE index defining the last point this view will
        have access to. This internal representation cannot match
        the style of the factory method (in which both start and end
        are inclusive) because we must be able to define empty
        ranges by having start = end
    featureStart : int
        The inclusive index of the first feature this view will have
        access to.
    featureEnd : int
        The EXCLUSIVE index defining the last feature this view will
        have access to. This internal representation cannot match
        the style of the factory method (in which both start and end
        are inclusive) because we must be able to define empty
        ranges by having start = end
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """

    def __init__(self, source, pointStart, pointEnd, featureStart, featureEnd,
                 **kwds):
        self._source = source
        self._pStart = pointStart
        self._pEnd = pointEnd
        self._fStart = featureStart
        self._fEnd = featureEnd
        #		kwds['name'] = self._source.name
        super(BaseView, self).__init__(**kwds)

    # redefinition from Base, except without the setter, using source
    # object's attributes
    def _getObjName(self):
        return self._name

    @property
    def name(self):
        """
        A name to be displayed when printing or logging this object.
        """
        return self._getObjName()

    # redefinition from Base, using source object's attributes
    def _getAbsPath(self):
        return self._source._absPath

    @property
    def absolutePath(self):
        """
        The path to the file this data originated from, in absolute
        form.
        """
        return self._getAbsPath()

    # redefinition from Base, using source object's attributes
    def _getRelPath(self):
        return self._source._relPath

    @property
    def relativePath(self):
        """
        The path to the file this data originated from, in relative
        form.
        """
        return self._getRelPath()

    def _pointNamesCreated(self):
        if self._source.pointNamesInverse is None:
            return False
        else:
            return True

    def _featureNamesCreated(self):
        if self._source.featureNamesInverse is None:
            return False
        else:
            return True

    def _getData(self):
        return self._source.data

    # TODO: retType

    ############################
    # Reimplemented Operations #
    ############################

    def view(self, pointStart=None, pointEnd=None, featureStart=None,
             featureEnd=None):

        # -1 because _pEnd and _fEnd are exclusive indices,
        # but view takes inclusive

        if pointStart is None:
            psAdj = None if len(self._source.points) == 0 else self._pStart
        else:
            psIndex = self._source.points.getIndex(pointStart)
            psAdj = psIndex + self._pStart

        if pointEnd is None:
            peAdj = None if len(self._source.points) == 0 else self._pEnd - 1
        else:
            peIndex = self._source.points.getIndex(pointEnd)
            peAdj = peIndex + self._pStart

        if featureStart is None:
            fsAdj = None if len(self._source.features) == 0 else self._fStart
        else:
            fsIndex = self._source.features.getIndex(featureStart)
            fsAdj = fsIndex + self._fStart

        if featureEnd is None:
            feAdj = None if len(self._source.features) == 0 else self._fEnd - 1
        else:
            feIndex = self._source.features.getIndex(featureEnd)
            feAdj = feIndex + self._fStart

        return self._source.view(psAdj, peAdj, fsAdj, feAdj)

    ###########################
    # Higher Order Operations #
    ###########################

    @exceptionDocstring
    def fillUsingAllData(self, match, fill, points=None, features=None,
                         returnModified=False, useLog=None, **kwarguments):
        readOnlyException("fillUsingAllData")

    @exceptionDocstring
    def replaceFeatureWithBinaryFeatures(self, featureToReplace, useLog=None):
        readOnlyException("replaceFeatureWithBinaryFeatures")

    @exceptionDocstring
    def transformFeatureToIntegers(self, featureToConvert, useLog=None):
        readOnlyException("transformFeatureToIntegers")

    ########################################
    ########################################
    ###   Functions related to logging   ###
    ########################################
    ########################################


    ###############################################################
    ###############################################################
    ###   Subclass implemented information querying functions   ###
    ###############################################################
    ###############################################################


    ##################################################################
    ##################################################################
    ###   Subclass implemented structural manipulation functions   ###
    ##################################################################
    ##################################################################

    @exceptionDocstring
    def transformElements(self, toTransform, points=None, features=None,
                          preserveZeros=False, skipNoneReturnValues=False,
                          useLog=None):
        readOnlyException("transform")

    @exceptionDocstring
    def transpose(self, useLog=None):
        readOnlyException("transpose")

    @exceptionDocstring
    def referenceDataFrom(self, other, useLog=None):
        readOnlyException("referenceDataFrom")

    @exceptionDocstring
    def fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd,
                 useLog=None):
        readOnlyException("fillWith")

    @exceptionDocstring
    def fillUsingAllData(self, match, fill, points=None, features=None,
                         returnModified=False, useLog=None, **kwarguments):
        readOnlyException("fillUsingAllData")

    @exceptionDocstring
    def flattenToOnePoint(self, useLog=None):
        readOnlyException("flattenToOnePoint")

    @exceptionDocstring
    def flattenToOneFeature(self, useLog=None):
        readOnlyException("flattenToOneFeature")

    @exceptionDocstring
    def unflattenFromOnePoint(self, numPoints, useLog=None):
        readOnlyException("unflattenFromOnePoint")

    @exceptionDocstring
    def unflattenFromOneFeature(self, numFeatures, useLog=None):
        readOnlyException("unflattenFromOneFeature")

    @exceptionDocstring
    def merge(self, other, point='strict', feature='union', onFeature=None,
              useLog=None):
        readOnlyException('merge')

    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################

    @exceptionDocstring
    def __imatmul__(self, other):
        readOnlyException("__imatmul__")

    @exceptionDocstring
    def __imul__(self, other):
        readOnlyException("__imul__")

    @exceptionDocstring
    def __iadd__(self, other):
        readOnlyException("__iadd__")

    @exceptionDocstring
    def __isub__(self, other):
        readOnlyException("__isub__")

    @exceptionDocstring
    def __idiv__(self, other):
        readOnlyException("__idiv__")

    @exceptionDocstring
    def __itruediv__(self, other):
        readOnlyException("__itruediv__")

    @exceptionDocstring
    def __ifloordiv__(self, other):
        readOnlyException("__ifloordiv__")

    @exceptionDocstring
    def __imod__(self, other):
        readOnlyException("__imod__")

    @exceptionDocstring
    def __ipow__(self, other):
        readOnlyException("__ipow__")
