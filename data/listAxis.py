"""
Implementations and helpers specific to performing axis-generic
operations on a UML List object.
"""
from __future__ import absolute_import
import copy
from abc import abstractmethod

import numpy

import UML
from UML.exceptions import ArgumentException
from .axis import Axis
from .points import Points
from .dataHelpers import sortIndexPosition

class ListAxis(Axis):
    """
    Differentiate how List methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, **kwds):
        super(ListAxis, self).__init__(**kwds)

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
        pnames = []
        fnames = []
        data = numpy.matrix(self._source.data, dtype=object)

        if isinstance(self, Points):
            keepList = []
            for idx in range(len(self)):
                if idx not in targetList:
                    keepList.append(idx)
            satisfying = data[targetList, :]
            if structure != 'copy':
                keep = data[keepList, :]
                self._source.data = keep.tolist()

            for index in targetList:
                pnames.append(self._getName(index))
            fnames = self._source.features.getNames()

        else:
            if self._source.data == []:
                # create empty matrix with correct shape
                shape = (len(self._source.points), len(self))
                empty = numpy.empty(shape)
                data = numpy.matrix(empty, dtype=numpy.object_)

            keepList = []
            for idx in range(len(self)):
                if idx not in targetList:
                    keepList.append(idx)
            satisfying = data[:, targetList]
            if structure != 'copy':
                keep = data[:, keepList]
                self._source.data = keep.tolist()

            for index in targetList:
                fnames.append(self._getName(index))
            pnames = self._source.points.getNames()

            if structure != 'copy':
                remainingFts = self._source._numFeatures - len(targetList)
                self._source._numFeatures = remainingFts

        return UML.data.List(satisfying, pointNames=pnames,
                             featureNames=fnames, reuseData=True)

    def _sort_implementation(self, sortBy, sortHelper):
        if isinstance(sortHelper, list):
            sortData = numpy.array(self._source.data, dtype=numpy.object_)
            if isinstance(self, Points):
                sortData = sortData[sortHelper, :]
            else:
                sortData = sortData[:, sortHelper]
            self._source.data = sortData.tolist()
            names = self._getNames()
            newNameOrder = [names[idx] for idx in sortHelper]
            return newNameOrder

        indexPosition = sortIndexPosition(self, sortBy, sortHelper)
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

        # convert indices of their previous location into their feature names
        newNameOrder = []
        for i in range(len(indexPosition)):
            oldIndex = indexPosition[i]
            newName = self._getName(oldIndex)
            newNameOrder.append(newName)
        return newNameOrder

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
