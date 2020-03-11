"""
Implementations and helpers specific to performing axis-generic
operations on a nimble DataFrame object.
"""

from abc import abstractmethod

import numpy

import nimble
from nimble.utility import ImportModule
from .axis import Axis
from .dataHelpers import sortIndexPosition
from .dataHelpers import nonSparseAxisUniqueArray, uniqueNameGetter
from .points import Points

pd = ImportModule('pandas')

class DataFrameAxis(Axis):
    """
    Differentiate how DataFrame methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    base : DataFrame
        The DataFrame instance that will be queried and modified.
    """

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
        df = self._base.data

        pointNames, featureNames = self._getStructuralNames(targetList)
        if self._isPoint:
            ret = df.values[targetList, :]
            axis = 0
        else:
            ret = df.values[:, targetList]
            axis = 1

        if structure.lower() != "copy":
            df.drop(targetList, axis=axis, inplace=True)

        if axis == 0:
            df.index = pd.RangeIndex(len(df.index))
        else:
            df.columns = pd.RangeIndex(len(df.columns))

        return nimble.data.DataFrame(pd.DataFrame(ret), pointNames=pointNames,
                                     featureNames=featureNames, reuseData=True)

    def _sort_implementation(self, indexPosition):
        # use numpy indexing to change the ordering
        if self._isPoint:
            self._base.data = self._base.data.iloc[indexPosition, :]
            self._base.data.index = pd.RangeIndex(len(self))
        else:
            self._base.data = self._base.data.iloc[:, indexPosition]
            self._base.data.columns = pd.RangeIndex(len(self))


    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = nonSparseAxisUniqueArray(self._base,
                                                             self._axis)
        uniqueData = pd.DataFrame(uniqueData)
        if numpy.array_equal(self._base.data.values, uniqueData):
            return self._base.copy()
        axisNames, offAxisNames = uniqueNameGetter(self._base, self._axis,
                                                   uniqueIndices)

        if self._isPoint:
            return nimble.createData('DataFrame', uniqueData,
                                     pointNames=axisNames,
                                     featureNames=offAxisNames, useLog=False)
        else:
            return nimble.createData('DataFrame', uniqueData,
                                     pointNames=offAxisNames,
                                     featureNames=axisNames, useLog=False)

    def _repeat_implementation(self, totalCopies, copyValueByValue):
        if self._isPoint:
            axis = 0
            ptDim = totalCopies
            ftDim = 1
        else:
            axis = 1
            ptDim = 1
            ftDim = totalCopies
        if copyValueByValue:
            repeated = numpy.repeat(self._base.data.values, totalCopies,
                                    axis)
        else:
            repeated = numpy.tile(self._base.data.values, (ptDim, ftDim))
        return pd.DataFrame(repeated)

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
