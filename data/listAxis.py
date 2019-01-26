"""
Implementations and helpers specific to performing axis-generic
operations on a UML List object.
"""
from __future__ import absolute_import
import copy
from abc import abstractmethod

import numpy

import UML
from UML.exceptions import InvalidArgumentType
from .axis import Axis
from .points import Points
from .base import cmp_to_key

class ListAxis(Axis):
    """
    Differentiate how List methods act dependent on the axis.

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
        if isinstance(self, Points):
            test = self._source.pointView(0)
            viewIter = self._source.points
        else:
            test = self._source.featureView(0)
            viewIter = self._source.features
        names = self._getNames()

        if isinstance(sortHelper, list):
            sortData = numpy.array(self._source.data, dtype=numpy.object_)
            if isinstance(self, Points):
                sortData = sortData[sortHelper, :]
            else:
                sortData = sortData[:, sortHelper]
            self._source.data = sortData.tolist()
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
            raise InvalidArgumentType(msg)

        # make array of views
        viewArray = []
        for v in viewIter:
            viewArray.append(v)

        if comparator is not None:
            # try:
            #     viewArray.sort(cmp=comparator)#python2
            # except:
            viewArray.sort(key=cmp_to_key(comparator))#python2 and 3
            indexPosition = []
            for i in range(len(viewArray)):
                viewAxis = getattr(viewArray[i], self._axis + 's')
                index = self._getIndex(getattr(viewAxis, 'getName')(0))
                indexPosition.append(index)
        else:
            #scoreArray = viewArray
            scoreArray = []
            if scorer is not None:
                # use scoring function to turn views into values
                for i in range(len(viewArray)):
                    scoreArray.append(scorer(viewArray[i]))
            else:
                for i in range(len(viewArray)):
                    scoreArray.append(viewArray[i][sortBy])

            # use numpy.argsort to make desired index array
            # this results in an array whole ith index contains the the
            # index into the data of the value that should be in the ith
            # position
            indexPosition = numpy.argsort(scoreArray)

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
