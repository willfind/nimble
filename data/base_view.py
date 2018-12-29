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

    Parameters
    ----------
    source : UML data object
        The UML object that this is a view into.
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

    def _copyNames(self, CopyObj):
        CopyObj.pointNamesInverse = self._getPointNames()
        CopyObj.pointNames = copy.copy(self._source.pointNames)

        if self._pointNamesCreated():
            CopyObj.pointNamesInverse = self.points.getNames()
            CopyObj.pointNames = copy.copy(self._source.pointNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.index = self.points.getNames()
        else:
            CopyObj.pointNamesInverse = None
            CopyObj.pointNames = None

        if self._featureNamesCreated():
            CopyObj.featureNamesInverse = self.features.getNames()
            CopyObj.featureNames = copy.copy(self._source.featureNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.columns = self.features.getNames()
        else:
            CopyObj.featureNamesInverse = None
            CopyObj.featureNames = None

        CopyObj._nextDefaultValueFeature = self._source._nextDefaultValueFeature
        CopyObj._nextDefaultValuePoint = self._source._nextDefaultValuePoint

        if len(self.points) != len(self._source.points) and self._source._pointNamesCreated():
            if self._pStart != 0:
                CopyObj.pointNames = {}
                for idx, name in enumerate(CopyObj.pointNamesInverse):
                    CopyObj.pointNames[name] = idx
            else:
                for name in self._source.pointNamesInverse[self._pEnd:len(self._source.points) + 1]:
                    del CopyObj.pointNames[name]

        if len(self.features) != len(self._source.features) and self._source._featureNamesCreated():
            if self._fStart != 0:
                CopyObj.featureNames = {}
                for idx, name in enumerate(CopyObj.featureNamesInverse):
                    CopyObj.featureNames[name] = idx
            else:
                for name in self._source.featureNamesInverse[self._fEnd:len(self._source.features) + 1]:
                    del CopyObj.featureNames[name]

    def view(self, pointStart=None, pointEnd=None, featureStart=None,
             featureEnd=None):

        # -1 because _pEnd and _fEnd are exclusive indices, but view takes inclusive

        if pointStart is None:
            psAdj = None if len(self._source.points) == 0 else self._pStart
        else:
            psIndex = self._source._getIndex(pointStart, 'point')
            psAdj = psIndex + self._pStart

        if pointEnd is None:
            peAdj = None if len(self._source.points) == 0 else self._pEnd - 1
        else:
            peIndex = self._source._getIndex(pointEnd, 'point')
            peAdj = peIndex + self._pStart

        if featureStart is None:
            fsAdj = None if len(self._source.features) == 0 else self._fStart
        else:
            fsIndex = self._source._getIndex(featureStart, 'feature')
            fsAdj = fsIndex + self._fStart

        if featureEnd is None:
            feAdj = None if len(self._source.features) == 0 else self._fEnd - 1
        else:
            feIndex = self._source._getIndex(featureEnd, 'feature')
            feAdj = feIndex + self._fStart

        return self._source.view(psAdj, peAdj, fsAdj, feAdj)

    ###########################
    # Higher Order Operations #
    ###########################

    @exception_docstring
    def fillUsingAllData(self, match, fill, arguments=None, points=None,
                          features=None, returnModified=False):
        readOnlyException("fillUsingAllData")

    @exception_docstring
    def replaceFeatureWithBinaryFeatures(self, featureToReplace):
        readOnlyException("replaceFeatureWithBinaryFeatures")

    @exception_docstring
    def transformFeatureToIntegers(self, featureToConvert):
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

    @exception_docstring
    def transpose(self):
        readOnlyException("transpose")

    @exception_docstring
    def referenceDataFrom(self, other):
        readOnlyException("referenceDataFrom")

    @exception_docstring
    def fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd):
        readOnlyException("fillWith")

    @exception_docstring
    def flattenToOnePoint(self):
        readOnlyException("flattenToOnePoint")

    @exception_docstring
    def flattenToOneFeature(self):
        readOnlyException("flattenToOneFeature")

    @exception_docstring
    def unflattenFromOnePoint(self, numPoints):
        readOnlyException("unflattenFromOnePoint")

    @exception_docstring
    def unflattenFromOneFeature(self, numFeatures):
        readOnlyException("unflattenFromOneFeature")

    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################

    @exception_docstring
    def __imul__(self, other):
        readOnlyException("__imul__")

    @exception_docstring
    def __iadd__(self, other):
        readOnlyException("__iadd__")

    @exception_docstring
    def __isub__(self, other):
        readOnlyException("__isub__")

    @exception_docstring
    def __idiv__(self, other):
        readOnlyException("__idiv__")

    @exception_docstring
    def __itruediv__(self, other):
        readOnlyException("__itruediv__")

    @exception_docstring
    def __ifloordiv__(self, other):
        readOnlyException("__ifloordiv__")

    @exception_docstring
    def __imod__(self, other):
        readOnlyException("__imod__")

    @exception_docstring
    def __ipow__(self, other):
        readOnlyException("__ipow__")

    ####################
    ####################
    ###   Helpers    ###
    ####################
    ####################

def readOnlyException(self, name):
    msg = "The " + name + " method is disallowed for View objects. View "
    msg += "objects are read only, yet this method modifies the object"
    raise ImproperActionException(msg)
