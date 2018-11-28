"""
Defines a subclass of the base data object, which serves as the primary
base class for read only views of data objects.

"""

from __future__ import division
from __future__ import absolute_import
import copy

from .base import Base
from .dataHelpers import inheritDocstringsFactory
from UML.exceptions import ImproperActionException

# prepend a message that view objects will raise an exception to Base docstring
def exception_docstring(func):
    name = func.__name__
    baseDoc = getattr(Base, name).__doc__
    if baseDoc is not None:
        viewMsg = "The {0} method is object modifying and ".format(name)
        viewMsg += "will always raise an exception for view objects.\n\n"
        viewMsg += "For reference, the docstring for this method "
        viewMsg += "when objects can be modified is below:\n"
        func.__doc__ = viewMsg + baseDoc
    return func


@inheritDocstringsFactory(Base)
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

    # redefinition from Base, except without the setter, using source
    # object's attributes
    def _getObjName(self):
        return self._name

    name = property(_getObjName, doc="A name to be displayed when printing or logging this object")

    # redefinition from Base, using source object's attributes
    def _getAbsPath(self):
        return self._source._absPath

    absolutePath = property(_getAbsPath, doc="The path to the file this data originated from, in absolute form")

    # redefinition from Base, using source object's attributes
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

        if self.pts != self._source.pts and self._source._pointNamesCreated():
            if self._pStart != 0:
                CopyObj.pointNames = {}
                for idx, name in enumerate(CopyObj.pointNamesInverse):
                    CopyObj.pointNames[name] = idx
            else:
                for name in self._source.pointNamesInverse[self._pEnd:self._source.pts + 1]:
                    del CopyObj.pointNames[name]

        if self.fts != self._source.fts and self._source._featureNamesCreated():
            if self._fStart != 0:
                CopyObj.featureNames = {}
                for idx, name in enumerate(CopyObj.featureNamesInverse):
                    CopyObj.featureNames[name] = idx
            else:
                for name in self._source.featureNamesInverse[self._fEnd:self._source.fts + 1]:
                    del CopyObj.featureNames[name]

    def view(self, pointStart=None, pointEnd=None, featureStart=None,
             featureEnd=None):

        # -1 because _pEnd and _fEnd are exclusive indices, but view takes inclusive

        if pointStart is None:
            psAdj = None if self._source.pts == 0 else self._pStart
        else:
            psIndex = self._source._getIndex(pointStart, 'point')
            psAdj = psIndex + self._pStart

        if pointEnd is None:
            peAdj = None if self._source.pts == 0 else self._pEnd - 1
        else:
            peIndex = self._source._getIndex(pointEnd, 'point')
            peAdj = peIndex + self._pStart

        if featureStart is None:
            fsAdj = None if self._source.fts == 0 else self._fStart
        else:
            fsIndex = self._source._getIndex(featureStart, 'feature')
            fsAdj = fsIndex + self._fStart

        if featureEnd is None:
            feAdj = None if self._source.fts == 0 else self._fEnd - 1
        else:
            feIndex = self._source._getIndex(featureEnd, 'feature')
            feAdj = feIndex + self._fStart

        return self._source.view(psAdj, peAdj, fsAdj, feAdj)


    ####################################
    # Low Level Operations, Disallowed #
    ####################################

    @exception_docstring
    def setPointName(self, oldIdentifier, newName):
        self._readOnlyException("setPointName")

    @exception_docstring
    def setFeatureName(self, oldIdentifier, newName):
        self._readOnlyException("setFeatureName")

    @exception_docstring
    def setPointNames(self, assignments=None):
        self._readOnlyException("setPointNames")

    @exception_docstring
    def setFeatureNames(self, assignments=None):
        self._readOnlyException("setFeatureNames")


    ###########################
    # Higher Order Operations #
    ###########################

    @exception_docstring
    def dropFeaturesContainingType(self, typeToDrop):
        self._readOnlyException("dropFeaturesContainingType")

    @exception_docstring
    def replaceFeatureWithBinaryFeatures(self, featureToReplace):
        self._readOnlyException("replaceFeatureWithBinaryFeatures")

    @exception_docstring
    def transformFeatureToIntegers(self, featureToConvert):
        self._readOnlyException("transformFeatureToIntegers")

    @exception_docstring
    def extractPointsByCoinToss(self, extractionProbability):
        self._readOnlyException("extractPointsByCoinToss")

    @exception_docstring
    def shufflePoints(self):
        self._readOnlyException("shufflePoints")

    @exception_docstring
    def shuffleFeatures(self):
        self._readOnlyException("shuffleFeatures")

    @exception_docstring
    def normalizePoints(self, subtract=None, divide=None, applyResultTo=None):
        self._readOnlyException("normalizePoints")

    @exception_docstring
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

    @exception_docstring
    def transpose(self):
        self._readOnlyException("transpose")

    @exception_docstring
    def addPoints(self, toAdd, insertBefore=None):
        self._readOnlyException("addPoints")

    @exception_docstring
    def addFeatures(self, toAdd, insertBefore=None):
        self._readOnlyException("addFeatures")

    @exception_docstring
    def sortPoints(self, sortBy=None, sortHelper=None):
        self._readOnlyException("sortPoints")

    @exception_docstring
    def sortFeatures(self, sortBy=None, sortHelper=None):
        self._readOnlyException("sortFeatures")

    @exception_docstring
    def extractPoints(self, toExtract=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("extractPoints")

    @exception_docstring
    def extractFeatures(self, toExtract=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("extractFeatures")

    @exception_docstring
    def deletePoints(self, toDelete=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("deletePoints")

    @exception_docstring
    def deleteFeatures(self, toDelete=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("deleteFeatures")

    @exception_docstring
    def retainPoints(self, toRetain=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("retainPoints")

    @exception_docstring
    def retainFeatures(self, toRetain=None, start=None, end=None, number=None, randomize=False):
        self._readOnlyException("retainFeatures")

    @exception_docstring
    def referenceDataFrom(self, other):
        self._readOnlyException("referenceDataFrom")

    @exception_docstring
    def transformEachPoint(self, function, points=None):
        self._readOnlyException("transformEachPoint")

    @exception_docstring
    def transformEachFeature(self, function, features=None):
        self._readOnlyException("transformEachFeature")

    @exception_docstring
    def transformEachElement(self, function, points=None, features=None, preserveZeros=False,
                             skipNoneReturnValues=False):
        self._readOnlyException("transformEachElement")

    @exception_docstring
    def fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd):
        self._readOnlyException("fillWith")

    @exception_docstring
    def handleMissingValues(self, method='remove points', features=None, arguments=None, alsoTreatAsMissing=[], markMissing=False):
        self._readOnlyException("handleMissingValues")

    @exception_docstring
    def flattenToOnePoint(self):
        self._readOnlyException("flattenToOnePoint")

    @exception_docstring
    def flattenToOneFeature(self):
        self._readOnlyException("flattenToOneFeature")

    @exception_docstring
    def unflattenFromOnePoint(self, numPoints):
        self._readOnlyException("unflattenFromOnePoint")

    @exception_docstring
    def unflattenFromOneFeature(self, numFeatures):
        self._readOnlyException("unflattenFromOneFeature")


    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################

    @exception_docstring
    def elementwiseMultiply(self, other):
        self._readOnlyException("elementwiseMultiply")

    @exception_docstring
    def elementwisePower(self, other):
        self._readOnlyException("elementwisePower")

    @exception_docstring
    def __imul__(self, other):
        self._readOnlyException("__imul__")

    @exception_docstring
    def __iadd__(self, other):
        self._readOnlyException("__iadd__")

    @exception_docstring
    def __isub__(self, other):
        self._readOnlyException("__isub__")

    @exception_docstring
    def __idiv__(self, other):
        self._readOnlyException("__idiv__")

    @exception_docstring
    def __itruediv__(self, other):
        self._readOnlyException("__itruediv__")

    @exception_docstring
    def __ifloordiv__(self, other):
        self._readOnlyException("__ifloordiv__")

    @exception_docstring
    def __imod__(self, other):
        self._readOnlyException("__imod__")

    @exception_docstring
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
