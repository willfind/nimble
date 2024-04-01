
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Implementations and helpers specific to performing axis-generic
operations on a nimble DataFrame object.
"""

from abc import ABCMeta, abstractmethod
import numbers

import numpy as np

import nimble
from nimble._utility import pd
from .axis import Axis
from .base import Base
from .points import Points
from .views import PointsView
from .features import Features
from .views import FeaturesView
from ._dataHelpers import denseAxisUniqueArray
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
        array = self._base._asNumpyArray()

        pointNames, featureNames = self._getStructuralNames(targetList)
        if self._isPoint:
            ret = array[targetList, :]
            axis = 0
            dtypes = tuple(dataframe.dtypes)
        else:
            ret = array[:, targetList]
            axis = 1
            dtypes = tuple(dataframe.dtypes[i] for i in targetList)

        if structure.lower() != "copy":
            dataframe.drop(targetList, axis=axis, inplace=True)

        if axis == 0:
            dataframe.index = pd.RangeIndex(len(dataframe.index))
        else:
            dataframe.columns = pd.RangeIndex(len(dataframe.columns))

        ret = nimble.core.data.DataFrame(
            pd.DataFrame(ret), pointNames=pointNames,
            featureNames=featureNames, reuseData=True)
        ret._setDtypes(dtypes)

        return ret

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
        if np.array_equal(self._base._asNumpyArray(), uniqueData):
            return self._base.copy()
        axisNames, offAxisNames = self._uniqueNameGetter(uniqueIndices)

        if self._isPoint:
            return nimble.data(uniqueData, pointNames=axisNames,
                               featureNames=offAxisNames,
                               returnType='DataFrame', useLog=False)

        return nimble.data(uniqueData, featureNames=axisNames,
                           pointNames=offAxisNames,
                           returnType='DataFrame', useLog=False)

    def _repeat_implementation(self, totalCopies, copyVectorByVector):
        repeated = {}
        if self._isPoint:
            for i, series in self._base._data.items():
                if copyVectorByVector:
                    repeated[i] = np.repeat(series, totalCopies)
                else:
                    repeated[i] = np.tile(series, totalCopies)
            ret = pd.DataFrame(repeated).reset_index(drop=True)
        else:
            for i, series in self._base._data.items():
                if copyVectorByVector:
                    for idx in range(totalCopies):
                        repeated[i * totalCopies + idx] = series
                else:
                    numFts = len(self._base.features)
                    for idx in range(totalCopies):
                        repeated[i + numFts * idx] = series
            ret = pd.DataFrame(repeated).sort_index(axis=1)

        return ret

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

            datetimeCols = [i for i, dt in enumerate(self._base._data.dtypes)
                            if dt.type == np.datetime64]
            for col in datetimeCols:
                if isinstance(currRet[col], (numbers.Number, str)):
                    self._base._data[col]  = self._base._data[col].astype(object)

            self._base._data.iloc[i, :] = currRet

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = self._base._asNumpyArray()[:, collapseIndices]
        retainData = self._base._asNumpyArray()[:, retainIndices]

        collapseDtypes = (np.dtype(object), max(self._base._data.dtypes))
        dtypes = []
        for i in range(len(self._base.features)):
            if i in retainIndices:
                dtypes.append(self._base._data.dtypes.iloc[i])
            elif i == collapseIndices[0]:
                dtypes.extend(collapseDtypes)

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, collapseData, currNumPoints,
            currFtNames, numRetPoints, numRetFeatures)

        self._base._data = pd.DataFrame(tmpData)
        self._base._setDtypes(dtypes)

    def _combineByExpandingFeatures_implementation(
        self, uniqueDict, namesIdx, valuesIdx, uniqueNames, numRetFeatures):
        uncombined = [dtype for i, dtype in self._base._data.dtypes.items()
                      if i not in valuesIdx + [namesIdx]]
        combined = list(self._base._data.dtypes.iloc[valuesIdx])
        dtypes = (uncombined[:namesIdx]
                  + combined * len(uniqueNames)
                  + uncombined[namesIdx:])

        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures,
                                                len(valuesIdx))

        self._base._data = pd.DataFrame(tmpData)
        if tuple(self._base._data.dtypes) != tuple(dtypes):
            for (i, col), dtype in zip(self._base._data.items(), dtypes):
                if col.dtype !=  dtype:
                    self._base._data[i] = col.astype(dtype)


class DataFramePointsView(PointsView, DataFramePoints):
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

            self._base._data[j] = currRet


    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        before = self._base._data.iloc[:, :featureIndex]
        new = []
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])

            new.append(pd.Series(newFeat, name=featureIndex + i))

        after = self._base._data.iloc[:, featureIndex + 1:]
        after.columns = [i + numResultingFts - 1 for i in after.columns]

        self._base._data = pd.concat((before, *new, after), axis=1)


class DataFrameFeaturesView(FeaturesView, DataFrameFeatures):
    """
    Limit functionality of DataFrameFeatures to read-only.

    Parameters
    ----------
    base : DataFrameView
        The DataFrameView instance that will be queried.
    """
