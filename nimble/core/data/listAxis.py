"""
Implementations and helpers specific to performing axis-generic
operations on a nimble List object.
"""

import copy
from abc import abstractmethod

import numpy

import nimble
from .axis import Axis
from ._dataHelpers import denseAxisUniqueArray, uniqueNameGetter

class ListAxis(Axis):
    """
    Differentiate how List methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    base : List
        The List instance that will be queried and modified.
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
        if self._isPoint:
            satisfying = [self._base.data[pt] for pt in targetList]
            if structure != 'copy':
                keepList = [i for i in range(len(self)) if i not in targetList]
                self._base.data = [self._base.data[pt] for pt in keepList]
            if satisfying == []:
                return nimble.core.data.List(satisfying, pointNames=pointNames,
                                             featureNames=featureNames,
                                             shape=((0, self._base.shape[1])),
                                             checkAll=False, reuseData=True)

        else:
            if self._base.data == []:
                # create empty matrix with correct shape
                shape = (len(self._base.points), len(targetList))
                satisfying = numpy.empty(shape, dtype=numpy.object_)
            else:
                satisfying = [[self._base.data[pt][ft] for ft in targetList]
                              for pt in range(len(self._base.points))]
            if structure != 'copy':
                keepList = [i for i in range(len(self)) if i not in targetList]
                self._base.data = [[self._base.data[pt][ft] for ft in keepList]
                                   for pt in range(len(self._base.points))]
                remainingFts = self._base._numFeatures - len(targetList)
                self._base._numFeatures = remainingFts

        return nimble.core.data.List(satisfying, pointNames=pointNames,
                                     featureNames=featureNames,
                                     checkAll=False, reuseData=True)

    def _sort_implementation(self, indexPosition):
        # run through target axis and change indices
        if self._isPoint:
            source = copy.copy(self._base.data)
            for i in range(len(self._base.data)):
                self._base.data[i] = source[indexPosition[i]]
        else:
            for i in range(len(self._base.data)):
                currPoint = self._base.data[i]
                temp = copy.copy(currPoint)
                for j in range(len(indexPosition)):
                    currPoint[j] = temp[indexPosition[j]]

    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = denseAxisUniqueArray(self._base,
                                                         self._axis)
        uniqueData = uniqueData.tolist()
        if self._base.data == uniqueData:
            return self._base.copy()

        axisNames, offAxisNames = uniqueNameGetter(self._base, self._axis,
                                                   uniqueIndices)
        if self._isPoint:
            return nimble.data('List', uniqueData, pointNames=axisNames,
                               featureNames=offAxisNames, useLog=False)
        else:
            return nimble.data('List', uniqueData, pointNames=offAxisNames,
                               featureNames=axisNames, useLog=False)

    def _repeat_implementation(self, totalCopies, copyValueByValue):
        if self._isPoint:
            if copyValueByValue:
                repeated = [list(lst) for lst in self._base.data
                            for _ in range(totalCopies)]
            else:
                repeated = [list(lst) for _ in range(totalCopies)
                            for lst in self._base.data]
        else:
            repeated = []
            for lst in self._base.data:
                if not isinstance(lst, list): # FeatureViewer
                    lst = list(lst)
                if copyValueByValue:
                    extended = []
                    for v in lst:
                        extended.extend([v] * totalCopies)
                else:
                    extended = lst * totalCopies
                repeated.append(extended)
        return repeated

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _insert_implementation(self, insertBefore, toInsert):
        pass

    @abstractmethod
    def _transform_implementation(self, function, limitTo):
        pass
