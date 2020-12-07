"""
Implementations and helpers specific to performing axis-generic
operations on a nimble DataFrame object.
"""

from abc import ABCMeta, abstractmethod

import numpy

import nimble
from nimble._utility import pd
from .axis import Axis
from .base import Base
from .views import AxisView
from .points import Points
from .views import PointsView
from .features import Features
from .views import FeaturesView
from ._dataHelpers import denseAxisUniqueArray, uniqueNameGetter
from ._dataHelpers import fillArrayWithCollapsedFeatures
from ._dataHelpers import fillArrayWithExpandedFeatures


class DataFrameAxis(Axis, metaclass=ABCMeta):
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
        dataframe = self._base._data

        pointNames, featureNames = self._getStructuralNames(targetList)
        if self._isPoint:
            ret = dataframe.values[targetList, :]
            axis = 0
        else:
            ret = dataframe.values[:, targetList]
            axis = 1

        if structure.lower() != "copy":
            dataframe.drop(targetList, axis=axis, inplace=True)

        if axis == 0:
            dataframe.index = pd.RangeIndex(len(dataframe.index))
        else:
            dataframe.columns = pd.RangeIndex(len(dataframe.columns))

        return nimble.core.data.DataFrame(
            pd.DataFrame(ret), pointNames=pointNames,
            featureNames=featureNames, reuseData=True)

    def _permute_implementation(self, indexPosition):
        # use numpy indexing to change the ordering
        if self._isPoint:
            self._base._data = self._base._data.iloc[indexPosition, :]
            self._base._data.index = pd.RangeIndex(len(self))
        else:
            self._base._data = self._base._data.iloc[:, indexPosition]
            self._base._data.columns = pd.RangeIndex(len(self))


    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = denseAxisUniqueArray(self._base,
                                                         self._axis)
        uniqueData = pd.DataFrame(uniqueData)
        if numpy.array_equal(self._base._data.values, uniqueData):
            return self._base.copy()
        axisNames, offAxisNames = uniqueNameGetter(self._base, self._axis,
                                                   uniqueIndices)

        if self._isPoint:
            return nimble.data('DataFrame', uniqueData, pointNames=axisNames,
                               featureNames=offAxisNames, useLog=False)

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
            repeated = numpy.repeat(self._base._data.values, totalCopies,
                                    axis)
        else:
            repeated = numpy.tile(self._base._data.values, (ptDim, ftDim))
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
        startData = self._base._data.iloc[:insertBefore, :]
        endData = self._base._data.iloc[insertBefore:, :]
        self._base._data = pd.concat((startData, toInsert._data, endData),
                                     axis=0, ignore_index=True)

    def _transform_implementation(self, function, limitTo):
        for i, pt in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(pt)

            # Some versions of pandas require 1-d inputs, and fail when
            # given nimble objects or 2d arrays
            if isinstance(currRet, Base):
                currRet = currRet.copy("numpyarray", outputAs1D=True)

            self._base._data.iloc[i, :] = currRet

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = self._base._data.values[:, collapseIndices]
        retainData = self._base._data.values[:, retainIndices]

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, numpy.array(collapseData),
            currNumPoints, currFtNames, numRetPoints, numRetFeatures)

        self._base._data = pd.DataFrame(tmpData)

    def _combineByExpandingFeatures_implementation(self, uniqueDict, namesIdx,
                                                   uniqueNames, numRetFeatures,
                                                   numExpanded):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures,
                                                numExpanded)

        self._base._data = pd.DataFrame(tmpData)


class DataFramePointsView(PointsView, AxisView, DataFramePoints):
    """
    Limit functionality of DataFramePoints to read-only.

    Parameters
    ----------
    base : DataFrameView
        The DataFrameView instance that will be queried.
    """


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
        startData = self._base._data.iloc[:, :insertBefore]
        endData = self._base._data.iloc[:, insertBefore:]
        self._base._data = pd.concat((startData, toInsert._data, endData),
                                     axis=1, ignore_index=True)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)

            self._base._data.iloc[:, j] = currRet


    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = numpy.empty(shape=(len(self._base.points), numRetFeatures),
                              dtype=numpy.object_)

        tmpData[:, :featureIndex] = self._base._data.values[:, :featureIndex]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = self._base._data.values[:, featureIndex + 1:]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._base._data = pd.DataFrame(tmpData)


class DataFrameFeaturesView(FeaturesView, AxisView, DataFrameFeatures):
    """
    Limit functionality of DataFrameFeatures to read-only.

    Parameters
    ----------
    base : DataFrameView
        The DataFrameView instance that will be queried.
    """
