"""
Implementations and helpers specific to performing axis-generic
operations on a nimble Matrix object.
"""

from abc import abstractmethod

import numpy

import nimble
from .axis import Axis
from .points import Points
from .dataHelpers import sortIndexPosition
from .dataHelpers import nonSparseAxisUniqueArray, uniqueNameGetter

class MatrixAxis(Axis):
    """
    Differentiate how Matrix methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    base : Matrix
        The Matrix instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _structuralBackend_implementation(self, structure, targetList):
        """
        Backend for points/features.extract points/features.delete,
        points/features.retain, and points/features.copy. Returns a new
        object containing only the points or features in targetList and
        performs some modifications to the original object if necessary.
        This function does not perform all of the modification or
        process how each function handles the returned value, these are
        managed separately by each frontend function.
        """
        pointNames, featureNames = self._getStructuralNames(targetList)
        if isinstance(self, Points):
            axisVal = 0
            ret = self._base.data[targetList]
        else:
            axisVal = 1
            ret = self._base.data[:, targetList]

        if structure != 'copy':
            self._base.data = numpy.delete(self._base.data,
                                             targetList, axisVal)

        return nimble.data.Matrix(ret, pointNames=pointNames,
                                  featureNames=featureNames, reuseData=True)

    def _sort_implementation(self, indexPosition):
        # use numpy indexing to change the ordering
        if isinstance(self, Points):
            self._base.data = self._base.data[indexPosition, :]
        else:
            self._base.data = self._base.data[:, indexPosition]


    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = nonSparseAxisUniqueArray(self._base,
                                                             self._axis)
        if numpy.array_equal(self._base.data, uniqueData):
            return self._base.copy()

        axisNames, offAxisNames = uniqueNameGetter(self._base, self._axis,
                                                   uniqueIndices)
        if isinstance(self, Points):
            return nimble.createData('Matrix', uniqueData, pointNames=axisNames,
                                     featureNames=offAxisNames, useLog=False)
        else:
            return nimble.createData('Matrix', uniqueData,
                                     pointNames=offAxisNames,
                                     featureNames=axisNames, useLog=False)

    def _repeat_implementation(self, totalCopies, copyValueByValue):
        if isinstance(self, Points):
            axis = 0
            ptDim = totalCopies
            ftDim = 1
        else:
            axis = 1
            ptDim = 1
            ftDim = totalCopies
        if copyValueByValue:
            repeated = numpy.repeat(self._base.data, totalCopies, axis)
        else:
            repeated = numpy.tile(self._base.data, (ptDim, ftDim))
        return repeated

    ###########
    # Helpers #
    ###########

    def _convertBaseDtype(self, retDtype):
        """
        Convert the dtype of the Base object if necessary to replace the
        current values with the transformed values.
        """
        baseDtype = self._base.data.dtype
        if baseDtype != numpy.object_ and retDtype == numpy.object_:
            self._base.data = self._base.data.astype(numpy.object_)
        elif baseDtype == numpy.int and retDtype == numpy.float:
            self._base.data = self._base.data.astype(numpy.float)
        elif baseDtype == numpy.bool_ and retDtype != numpy.bool_:
            self._base.data = self._base.data.astype(retDtype)

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _insert_implementation(self, insertBefore, toInsert):
        pass

    # @abstractmethod
    # def _flattenToOne_implementation(self):
    #     pass
    #
    # @abstractmethod
    # def _unflattenFromOne_implementation(self, divideInto):
    #     pass

    @abstractmethod
    def _transform_implementation(self, function, limitTo):
        pass
