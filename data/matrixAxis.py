"""
Implementations and helpers specific to performing axis-generic
operations on a UML Matrix object.
"""
from __future__ import absolute_import
from abc import abstractmethod

import numpy

import UML
from UML.exceptions import ArgumentException
from .axis import Axis
from .points import Points
from .dataHelpers import sortIndexPosition

class MatrixAxis(Axis):
    """
    Differentiate how Matrix methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    source : UML data object
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
        nameList = []
        if isinstance(self, Points):
            axisVal = 0
            ret = self._source.data[targetList]
            pointNames = nameList
            featureNames = self._source.features.getNames()
        else:
            axisVal = 1
            ret = self._source.data[:, targetList]
            featureNames = nameList
            pointNames = self._source.points.getNames()

        if structure != 'copy':
            self._source.data = numpy.delete(self._source.data,
                                             targetList, axisVal)

        # construct nameList
        for index in targetList:
            nameList.append(self._getName(index))

        return UML.data.Matrix(ret, pointNames=pointNames,
                               featureNames=featureNames)

    def _sort_implementation(self, sortBy, sortHelper):
        if isinstance(sortHelper, list):
            if isinstance(self, Points):
                self._source.data = self._source.data[sortHelper, :]
            else:
                self._source.data = self._source.data[:, sortHelper]
            names = self._getNames()
            newNameOrder = [names[idx] for idx in sortHelper]
            return newNameOrder

        indexPosition = sortIndexPosition(self, sortBy, sortHelper)
        # use numpy indexing to change the ordering
        if isinstance(self, Points):
            self._source.data = self._source.data[indexPosition, :]
        else:
            self._source.data = self._source.data[:, indexPosition]

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
