"""
Defines a subclass of the base data object, which serves as the primary
base class for read only views of data objects.

"""

from __future__ import division
from __future__ import absolute_import
from .base import Base
from UML.exceptions import ImproperActionException

import copy

from inspect import getmembers, isfunction

# wrapper to inherit docstrings from Base if no docstring is present,
# if a docstring is available in BaseView it will override the Base docstring
def inherit_docstrings(cls):
    for name, func in getmembers(cls, isfunction):
        if func.__doc__: continue
        for parent in cls.__mro__[1:]:
            if hasattr(parent, name):
                func.__doc__ = getattr(parent, name).__doc__
    return cls

@inherit_docstrings
class BaseView(Base):
    """
    Class defining read only view objects, which have the same api as a
    normal data object, but disallow all methods which could change the
    data.

    """

    def __init__(self, source, pointStart, pointEnd, featureStart, featureEnd,
                 **kwds):
        """
        Initializes the object which overides all of the funcitonality in
        UML.data.Base to either handle the provided access limits or throw
        exceptions for inappropriate operations.

        source: the UML object that this is a view into.

        pointStart: the inclusive index of the first point this view will have
        access to.

        pointEnd: the EXCLUSIVE index defining the last point this view will
        have access to. This internal representation cannot match the style
        of the factory method (in which both start and end are inclusive)
        because we must be able to define empty ranges by having start = end

        featureStart: the inclusive index of the first feature this view will
        have access to.

        featureEnd: the EXCLUSIVE index defining the last feature this view
        will have access to. This internal representation cannot match the
        style of the factory method (in which both start and end are inclusive)
        because we must be able to define empty ranges by having start = end

        kwds: included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.

        """
        self._source = source
        self._pStart = pointStart
        self._pEnd = pointEnd
        self._fStart = featureStart
        self._fEnd = featureEnd
        #		kwds['name'] = self._source.name
        super(BaseView, self).__init__(**kwds)

    # redifinition from Base, except without the setter, using source
    # object's attributes
    def _getObjName(self):
        return self._name

    name = property(_getObjName, doc="A name to be displayed when printing or logging this object")

    # redifinition from Base, using source object's attributes
    def _getAbsPath(self):
        return self._source._absPath

    absolutePath = property(_getAbsPath, doc="The path to the file this data originated from, in absolute form")

    # redifinition from Base, using source object's attributes
    def _getRelPath(self):
        return self._source._relPath

    relativePath = property(_getRelPath, doc="The path to the file this data originated from, in relative form")

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

    def getPointNames(self):
        ret = self._source.getPointNames()
        ret = ret[self._pStart:self._pEnd]

        return ret

    def getFeatureNames(self):
        ret = self._source.getFeatureNames()
        ret = ret[self._fStart:self._fEnd]

        return ret

    def getPointName(self, index):
        corrected = index + self._pStart
        return self._source.getPointName(corrected)

    def getPointIndex(self, name):
        possible = self._source.getPointIndex(name)
        if possible >= self._pStart and possible < self._pEnd:
            return possible - self._pStart
        else:
            raise KeyError()

    def getFeatureName(self, index):
        corrected = index + self._fStart
        return self._source.getFeatureName(corrected)

    def getFeatureIndex(self, name):
        possible = self._source.getFeatureIndex(name)
        if possible >= self._fStart and possible < self._fEnd:
            return possible - self._fStart
        else:
            raise KeyError()

    def _copyNames(self, CopyObj):

        if self._pointNamesCreated():
            CopyObj.pointNamesInverse = self.getPointNames()
            CopyObj.pointNames = copy.copy(self._source.pointNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.index = self.getPointNames()
        else:
            CopyObj.pointNamesInverse = None
            CopyObj.pointNames = None

        if self._featureNamesCreated():
            CopyObj.featureNamesInverse = self.getFeatureNames()
            CopyObj.featureNames = copy.copy(self._source.featureNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.columns = self.getFeatureNames()
        else:
            CopyObj.featureNamesInverse = None
            CopyObj.featureNames = None

        CopyObj._nextDefaultValueFeature = self._source._nextDefaultValueFeature
        CopyObj._nextDefaultValuePoint = self._source._nextDefaultValuePoint

        if self.points != self._source.points:
            if self._pStart != 0:
                CopyObj.pointNames = {}
                for idx, name in enumerate(CopyObj.pointNamesInverse):
                    CopyObj.pointNames[name] = idx
            else:
                for name in self._source.pointNamesInverse[self._pEnd:self._source.points + 1]:
                    del CopyObj.pointNames[name]

        if self.features != self._source.features:
            if self._fStart != 0:
                CopyObj.featureNames = {}
                for idx, name in enumerate(CopyObj.featureNamesInverse):
                    CopyObj.featureNames[name] = idx
            else:
                for name in self._source.featureNamesInverse[self._fEnd:self._source.features + 1]:
                    del CopyObj.featureNames[name]


    def view(self, pointStart=None, pointEnd=None, featureStart=None,
             featureEnd=None):

        # -1 because _pEnd and _fEnd are exclusive indices, but view takes inclusive

        if pointStart is None:
            psAdj = None if self._source.points == 0 else self._pStart
        else:
            psIndex = self._source._getIndex(pointStart, 'point')
            psAdj = psIndex + self._pStart

        if pointEnd is None:
            peAdj = None if self._source.points == 0 else self._pEnd - 1
        else:
            peIndex = self._source._getIndex(pointEnd, 'point')
            peAdj = peIndex + self._pStart

        if featureStart is None:
            fsAdj = None if self._source.features == 0 else self._fStart
        else:
            fsIndex = self._source._getIndex(featureStart, 'feature')
            fsAdj = fsIndex + self._fStart

        if featureEnd is None:
            feAdj = None if self._source.features == 0 else self._fEnd - 1
        else:
            feIndex = self._source._getIndex(featureEnd, 'feature')
            feAdj = feIndex + self._fStart

        return self._source.view(psAdj, peAdj, fsAdj, feAdj)


    ####################################
    # Low Level Operations, Disallowed #
    ####################################

    def setPointName(self, oldIdentifier, newName):
        self._readOnlyException("setPointName")

    def setFeatureName(self, oldIdentifier, newName):
        self._readOnlyException("setFeatureName")


    def setPointNames(self, assignments=None):
        self._readOnlyException("setPointNames")

    def setFeatureNames(self, assignments=None):
        self._readOnlyException("setFeatureNames")


    ###########################
    # Higher Order Operations #
    ###########################

    def dropFeaturesContainingType(self, typeToDrop):
        self._readOnlyException("dropFeaturesContainingType")

    def replaceFeatureWithBinaryFeatures(self, featureToReplace):
        self._readOnlyException("replaceFeatureWithBinaryFeatures")

    def transformFeatureToIntegers(self, featureToConvert):
        self._readOnlyException("transformFeatureToIntegers")

    def extractPointsByCoinToss(self, extractionProbability):
        self._readOnlyException("extractPointsByCoinToss")

    def shufflePoints(self):
        self._readOnlyException("shufflePoints")

    def shuffleFeatures(self):
        self._readOnlyException("shuffleFeatures")

    def normalizePoints(self, subtract=None, divide=None, applyResultTo=None):
        self._readOnlyException("normalizePoints")

    def normalizeFeatures(self, subtract=None, divide=None, applyResultTo=None):
        self._readOnlyException("normalizeFeatures")


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

    def transpose(self):
        self._readOnlyException("transpose")

    def appendPoints(self, toAppend):
        self._readOnlyException("appendPoints")

    def appendFeatures(self, toAppend):
        self._readOnlyException("appendFeatures")

    def sortPoints(self, sortBy=None, sortHelper=None):
        self._readOnlyException("sortPoints")

    def sortFeatures(self, sortBy=None, sortHelper=None):
        self._readOnlyException("sortFeatures")

    def extractPoints(self, toExtract=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("extractPoints")

    def extractFeatures(self, toExtract=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("extractFeatures")

    def deletePoints(self, toDelete=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("deletePoints")

    def deleteFeatures(self, toDelete=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("deleteFeatures")

    def retainPoints(self, toRetain=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("retainPoints")

    def retainFeatures(self, toRetain=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("retainFeatures")

    def referenceDataFrom(self, other):
        self._readOnlyException("referenceDataFrom")

    def transformEachPoint(self, function, points=None):
        self._readOnlyException("transformEachPoint")


    def transformEachFeature(self, function, features=None):
        self._readOnlyException("transformEachFeature")

    def transformEachElement(self, function, points=None, features=None, preserveZeros=False,
                             skipNoneReturnValues=False):
        self._readOnlyException("transformEachElement")

    def fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd):
        self._readOnlyException("fillWith")

    def handleMissingValues(self, method='remove points', features=None, arguments=None, alsoTreatAsMissing=[], markMissing=False):
        self._readOnlyException("handleMissingValues")

    def flattenToOnePoint(self):
        self._readOnlyException("flattenToOnePoint")

    def flattenToOneFeature(self):
        self._readOnlyException("flattenToOneFeature")

    def unflattenFromOnePoint(self, numPoints):
        self._readOnlyException("unflattenFromOnePoint")

    def unflattenFromOneFeature(self, numFeatures):
        self._readOnlyException("unflattenFromOneFeature")


    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################

    def elementwiseMultiply(self, other):
        self._readOnlyException("elementwiseMultiply")

    def elementwisePower(self, other):
        self._readOnlyException("elementwisePower")

    def __imul__(self, other):
        self._readOnlyException("__imul__")

    def __iadd__(self, other):
        self._readOnlyException("__iadd__")

    def __isub__(self, other):
        self._readOnlyException("__isub__")

    def __idiv__(self, other):
        self._readOnlyException("__idiv__")

    def __itruediv__(self, other):
        self._readOnlyException("__itruediv__")

    def __ifloordiv__(self, other):
        self._readOnlyException("__ifloordiv__")

    def __imod__(self, other):
        self._readOnlyException("__imod__")

    def __ipow__(self, other):
        self._readOnlyException("__ipow__")


    ####################
    ####################
    ###   Helpers    ###
    ####################
    ####################

    def _readOnlyException(self, name):
        msg = "The " + name + " method is disallowed for View objects. View "
        msg += "objects are read only, yet this method modifies the object"
        raise ImproperActionException(msg)
