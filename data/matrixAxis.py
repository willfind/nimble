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
from .base import cmp_to_key

class MatrixAxis(Axis):
    """
    Differentiate how Matrix methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    axis : str
        The axis ('point' or 'feature') which the function will be
        applied to.
    source : UML data object
        The object containing point and feature data.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, axis, source, **kwds):
        self._axis = axis
        self._source = source
        kwds['axis'] = self._axis
        kwds['source'] = self._source
        super(MatrixAxis, self).__init__(**kwds)

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
        if self._axis == 'point':
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
        if self._axis == 'point':
            test = self._source.pointView(0)
            viewIter = self._source.points
        else:
            test = self._source.featureView(0)
            viewIter = self._source.features
        names = self._getNames()

        if isinstance(sortHelper, list):
            if self._axis == 'point':
                self._source.data = self._source.data[sortHelper, :]
            else:
                self._source.data = self._source.data[:, sortHelper]
            newNameOrder = [names[idx] for idx in sortHelper]
            return newNameOrder

        scorer = None
        comparator = None
        try:
            sortHelper(test)
            scorer = sortHelper
        except TypeError:
            pass
        try:
            sortHelper(test, test)
            comparator = sortHelper
        except TypeError:
            pass

        if sortHelper is not None and scorer is None and comparator is None:
            msg = "sortHelper is neither a scorer or a comparator"
            raise ArgumentException(msg)

        if comparator is not None:
            # make array of views
            viewArray = []
            for v in viewIter:
                viewArray.append(v)

            viewArray.sort(key=cmp_to_key(comparator))
            indexPosition = []
            for i in range(len(viewArray)):
                viewAxis = getattr(viewArray[i], self._axis + 's')
                index = self._getIndex(getattr(viewAxis, 'getName')(0))
                indexPosition.append(index)
            indexPosition = numpy.array(indexPosition)
        elif hasattr(scorer, 'permuter'):
            scoreArray = scorer.indices
            indexPosition = numpy.argsort(scoreArray)
        else:
            # make array of views
            viewArray = []
            for v in viewIter:
                viewArray.append(v)

            scoreArray = viewArray
            if scorer is not None:
                # use scoring function to turn views into values
                for i in range(len(viewArray)):
                    scoreArray[i] = scorer(viewArray[i])
            else:
                for i in range(len(viewArray)):
                    scoreArray[i] = viewArray[i][sortBy]

            # use numpy.argsort to make desired index array
            # this results in an array whose ith entry contains the the
            # index into the data of the value that should be in the ith
            # position.
            indexPosition = numpy.argsort(scoreArray)

        # use numpy indexing to change the ordering
        if self._axis == 'point':
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

    @abstractmethod
    def _flattenToOne_implementation(self):
        pass

    @abstractmethod
    def _unflattenFromOne_implementation(self, divideInto):
        pass

    @abstractmethod
    def _transform_implementation(self, function, limitTo):
        pass
