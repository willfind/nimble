"""
Implementations and helpers specific to performing axis-generic
operations on a nimble DataFrame object.
"""

from abc import abstractmethod

import numpy

import nimble
from nimble._utility import pd
from .axis import Axis
from .views import AxisView
from .points import Points
from .views import PointsView
from .features import Features
from .views import FeaturesView
from ._dataHelpers import denseAxisUniqueArray, uniqueNameGetter
from ._dataHelpers import fillArrayWithCollapsedFeatures
from ._dataHelpers import fillArrayWithExpandedFeatures


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

        return nimble.core.data.DataFrame(
            pd.DataFrame(ret), pointNames=pointNames,
            featureNames=featureNames, reuseData=True)

    def _permute_implementation(self, indexPosition):
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
        uniqueData, uniqueIndices = denseAxisUniqueArray(self._base,
                                                         self._axis)
        uniqueData = pd.DataFrame(uniqueData)
        if numpy.array_equal(self._base.data.values, uniqueData):
            return self._base.copy()
        axisNames, offAxisNames = uniqueNameGetter(self._base, self._axis,
                                                   uniqueIndices)

        if self._isPoint:
            return nimble.data('DataFrame', uniqueData, pointNames=axisNames,
                               featureNames=offAxisNames, useLog=False)
        else:
            return nimble.data('DataFrame', uniqueData, featureNames=axisNames,
                               pointNames=offAxisNames, useLog=False)

    def _repeat_implementation(self, totalCopies, copyVectorByVector):
        if self._isPoint:
            axis = 0
            ptDim = totalCopies
            ftDim = 1
        else:
            axis = 1
            ptDim = 1
            ftDim = totalCopies
        if copyVectorByVector:
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

    @abstractmethod
    def _transform_implementation(self, function, limitTo):
        pass


class DataFramePoints(DataFrameAxis, Points):
    """
    DataFrame method implementations performed on the points axis.

    Parameters
    ----------
    base : DataFrame
        The DataFrame instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _insert_implementation(self, insertBefore, toInsert):
        """
        Insert the points from the toInsert object below the provided
        index in this object, the remaining points from this object will
        continue below the inserted points.
        """
        startData = self._base.data.iloc[:insertBefore, :]
        endData = self._base.data.iloc[insertBefore:, :]
        self._base.data = pd.concat((startData, toInsert.data, endData),
                                    axis=0, ignore_index=True)

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)

            self._base.data.iloc[i, :] = currRet

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = self._base.data.values[:, collapseIndices]
        retainData = self._base.data.values[:, retainIndices]

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, numpy.array(collapseData),
            currNumPoints, currFtNames, numRetPoints, numRetFeatures)

        self._base.data = pd.DataFrame(tmpData)

    def _combineByExpandingFeatures_implementation(self, uniqueDict, namesIdx,
                                                   uniqueNames, numRetFeatures,
                                                   numExpanded):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures,
                                                numExpanded)

        self._base.data = pd.DataFrame(tmpData)


class DataFramePointsView(PointsView, AxisView, DataFramePoints):
    """
    Limit functionality of DataFramePoints to read-only.

    Parameters
    ----------
    base : DataFrameView
        The DataFrameView instance that will be queried.
    """
    pass


class DataFrameFeatures(DataFrameAxis, Features):
    """
    DataFrame method implementations performed on the feature axis.

    Parameters
    ----------
    base : DataFrame
        The DataFrame instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _insert_implementation(self, insertBefore, toInsert):
        """
        Insert the features from the toInsert object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        startData = self._base.data.iloc[:, :insertBefore]
        endData = self._base.data.iloc[:, insertBefore:]
        self._base.data = pd.concat((startData, toInsert.data, endData),
                                    axis=1, ignore_index=True)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)

            self._base.data.iloc[:, j] = currRet


    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = numpy.empty(shape=(len(self._base.points), numRetFeatures),
                              dtype=numpy.object_)

        tmpData[:, :featureIndex] = self._base.data.values[:, :featureIndex]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = self._base.data.values[:, featureIndex + 1:]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._base.data = pd.DataFrame(tmpData)


class DataFrameFeaturesView(FeaturesView, AxisView, DataFrameFeatures):
    """
    Limit functionality of DataFrameFeatures to read-only.

    Parameters
    ----------
    base : DataFrameView
        The DataFrameView instance that will be queried.
    """
    pass
