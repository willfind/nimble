"""
Implementations and helpers specific to performing axis-generic
operations on a nimble Matrix object.
"""

from __future__ import absolute_import
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
    source : nimble data object
        The object containing point and feature data.
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
            ret = self._source.data[targetList]
        else:
            axisVal = 1
            ret = self._source.data[:, targetList]

        if structure != 'copy':
            self._source.data = numpy.delete(self._source.data,
                                             targetList, axisVal)

        return nimble.data.Matrix(ret, pointNames=pointNames,
                                  featureNames=featureNames, reuseData=True)

    def _sort_implementation(self, indexPosition):
        # use numpy indexing to change the ordering
        if isinstance(self, Points):
            self._source.data = self._source.data[indexPosition, :]
        else:
            self._source.data = self._source.data[:, indexPosition]


    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = nonSparseAxisUniqueArray(self._source,
                                                             self._axis)
        if numpy.array_equal(self._source.data, uniqueData):
            return self._source.copy()

        axisNames, offAxisNames = uniqueNameGetter(self._source, self._axis,
                                                   uniqueIndices)
        if isinstance(self, Points):
            return nimble.createData('Matrix', uniqueData, pointNames=axisNames,
                                     featureNames=offAxisNames, useLog=False)
        else:
            return nimble.createData('Matrix', uniqueData,
                                     pointNames=offAxisNames,
                                     featureNames=axisNames, useLog=False)

    def _duplicate_implementation(self, totalCopies):
        if isinstance(self, Points):
            ptDim = totalCopies
            ftDim = 1
        else:
            ptDim = 1
            ftDim = totalCopies
        duplicated = numpy.tile(self._source.data, (ptDim, ftDim))
        return duplicated

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _add_implementation(self, toAdd, insertBefore):
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
