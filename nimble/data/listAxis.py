"""
Implementations and helpers specific to performing axis-generic
operations on a nimble List object.
"""

from __future__ import absolute_import
import copy
from abc import abstractmethod

import numpy

import nimble
from .axis import Axis
from .points import Points

from .dataHelpers import sortIndexPosition
from .dataHelpers import nonSparseAxisUniqueArray, uniqueNameGetter

class ListAxis(Axis):
    """
    Differentiate how List methods act dependent on the axis.

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
            satisfying = [self._source.data[pt] for pt in targetList]
            if structure != 'copy':
                keepList = [i for i in range(len(self)) if i not in targetList]
                self._source.data = [self._source.data[pt] for pt in keepList]
            if satisfying == []:
                return nimble.data.List(satisfying, pointNames=pointNames,
                                        featureNames=featureNames,
                                        shape=((0, self._source.shape[1])),
                                        checkAll=False, reuseData=True)

        else:
            if self._source.data == []:
                # create empty matrix with correct shape
                shape = (len(self._source.points), len(targetList))
                empty = numpy.empty(shape)
                satisfying = numpy.matrix(empty, dtype=numpy.object_)
            else:
                satisfying = [[self._source.data[pt][ft] for ft in targetList]
                              for pt in range(len(self._source.points))]
            if structure != 'copy':
                keepList = [i for i in range(len(self)) if i not in targetList]
                self._source.data = [[self._source.data[pt][ft] for ft in keepList]
                                     for pt in range(len(self._source.points))]
                remainingFts = self._source._numFeatures - len(targetList)
                self._source._numFeatures = remainingFts

        return nimble.data.List(satisfying, pointNames=pointNames,
                                featureNames=featureNames,
                                checkAll=False, reuseData=True)

    def _sort_implementation(self, indexPosition):
        # run through target axis and change indices
        if isinstance(self, Points):
            source = copy.copy(self._source.data)
            for i in range(len(self._source.data)):
                self._source.data[i] = source[indexPosition[i]]
        else:
            for i in range(len(self._source.data)):
                currPoint = self._source.data[i]
                temp = copy.copy(currPoint)
                for j in range(len(indexPosition)):
                    currPoint[j] = temp[indexPosition[j]]

    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = nonSparseAxisUniqueArray(self._source,
                                                             self._axis)
        uniqueData = uniqueData.tolist()
        if self._source.data == uniqueData:
            return self._source.copy()

        axisNames, offAxisNames = uniqueNameGetter(self._source, self._axis,
                                                   uniqueIndices)
        if isinstance(self, Points):
            return nimble.createData('List', uniqueData, pointNames=axisNames,
                                     featureNames=offAxisNames, useLog=False)
        else:
            return nimble.createData('List', uniqueData, pointNames=offAxisNames,
                                      featureNames=axisNames, useLog=False)

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
