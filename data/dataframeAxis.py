"""
Implementations and helpers specific to performing axis-generic
operations on a UML DataFrame object.
"""
from __future__ import absolute_import
from abc import abstractmethod

import numpy

import UML
from .axis import Axis
from .dataHelpers import sortIndexPosition
from .dataHelpers import nonSparseAxisUniqueArray, uniqueNameGetter
from .points import Points

pd = UML.importModule('pandas')

class DataFrameAxis(Axis):
    """
    Differentiate how DataFrame methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    source : UML data object
        The object containing point and feature data.
    """
    def _setName_implementation(self, oldIdentifier, newName):
        super(DataFrameAxis, self)._setName_implementation(oldIdentifier,
                                                           newName)
        #update the index or columns in self.data
        self._updateName()

    def _setNamesFromList(self, assignments, count):
        super(DataFrameAxis, self)._setNamesFromList(assignments, count)
        self._updateName()

    def _setNamesFromDict(self, assignments, count):
        super(DataFrameAxis, self)._setNamesFromDict(assignments, count)
        self._updateName()

    def _updateName(self):
        """
        update self.data.index or self.data.columns
        """
        if isinstance(self, Points):
            self._source.data.index = range(len(self._source.data.index))
        else:
            self._source.data.columns = range(len(self._source.data.columns))

    ##############################
    # Structural implementations #
    ##############################

    def _structuralBackend_implementation(self, structure, targetList):
        """
        Backend for points/features.extract points/features.delete,
        points/features.retain, and points/features.copy. Returns a new
        object containing only the members in targetList and performs
        some modifications to the original object if necessary. This
        function does not perform all of the modification or process how
        each function handles the returned value, these are managed
        separately by each frontend function.
        """
        df = self._source.data

        pointNames, featureNames = self._getStructuralNames(targetList)
        if isinstance(self, Points):
            ret = df.iloc[targetList, :]
            axis = 0
        else:
            ret = df.iloc[:, targetList]
            axis = 1

        if structure.lower() != "copy":
            df.drop(targetList, axis=axis, inplace=True)

        if axis == 0:
            df.index = numpy.arange(len(df.index), dtype=df.index.dtype)
        else:
            df.columns = numpy.arange(len(df.columns), dtype=df.columns.dtype)

        return UML.data.DataFrame(ret, pointNames=pointNames,
                                  featureNames=featureNames)

    def _sort_implementation(self, sortBy, sortHelper):
        if isinstance(sortHelper, list):
            if isinstance(self, Points):
                self._source.data = self._source.data.iloc[sortHelper, :]
            else:
                self._source.data = self._source.data.iloc[:, sortHelper]
            names = self._getNames()
            newNameOrder = [names[idx] for idx in sortHelper]
            return newNameOrder

        axisAttr = 'points' if isinstance(self, Points) else 'features'
        indexPosition = sortIndexPosition(self, sortBy, sortHelper, axisAttr)

        # use numpy indexing to change the ordering
        if isinstance(self, Points):
            self._source.data = self._source.data.iloc[indexPosition, :]
        else:
            self._source.data = self._source.data.iloc[:, indexPosition]

        # convert indices of their previous location into their feature names
        newNameOrder = []
        for i in range(len(indexPosition)):
            oldIndex = indexPosition[i]
            newName = self._getName(oldIndex)
            newNameOrder.append(newName)
        return newNameOrder

    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = nonSparseAxisUniqueArray(self._source,
                                                             self._axis)
        uniqueData = pd.DataFrame(uniqueData)
        if numpy.array_equal(self._source.data.values, uniqueData):
            return self._source.copy()
        axisNames, offAxisNames = uniqueNameGetter(self._source, self._axis,
                                                   uniqueIndices)

        if isinstance(self, Points):
            return UML.createData('DataFrame', uniqueData,
                                  pointNames=axisNames,
                                  featureNames=offAxisNames, useLog=False)
        else:
            return UML.createData('DataFrame', uniqueData,
                                  pointNames=offAxisNames,
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
