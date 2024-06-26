
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
operations on a nimble Matrix object.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

import nimble
from nimble._utility import numpy2DArray
from .axis import Axis
from .points import Points
from .views import PointsView
from .features import Features
from .views import FeaturesView
from ._dataHelpers import denseAxisUniqueArray
from ._dataHelpers import fillArrayWithCollapsedFeatures
from ._dataHelpers import fillArrayWithExpandedFeatures

class MatrixAxis(Axis, metaclass=ABCMeta):
    """
    Differentiate how Matrix methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    base : Matrix
        The Matrix instance that will be queried and modified.
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
            axisVal = 0
            ret = self._base._data[targetList]
        else:
            axisVal = 1
            ret = self._base._data[:, targetList]

        if structure != 'copy':
            self._base._data = np.delete(self._base._data, targetList, axisVal)

        return nimble.core.data.Matrix(ret, pointNames=pointNames,
                                       featureNames=featureNames,
                                       reuseData=True)

    def _permute_implementation(self, indexPosition):
        # use numpy indexing to change the ordering
        if self._isPoint:
            self._base._data = self._base._data[indexPosition, :]
        else:
            self._base._data = self._base._data[:, indexPosition]


    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = denseAxisUniqueArray(self._base,
                                                         self._axis)
        if np.array_equal(self._base._data, uniqueData):
            return self._base.copy()

        axisNames, offAxisNames = self._uniqueNameGetter(uniqueIndices)
        if self._isPoint:
            return nimble.data(uniqueData, pointNames=axisNames,
                               featureNames=offAxisNames,
                               returnType=self._base.getTypeString(),
                               useLog=False)

        return nimble.data(uniqueData, pointNames=offAxisNames,
                           featureNames=axisNames,
                           returnType=self._base.getTypeString(),
                           useLog=False)

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
            repeated = np.repeat(self._base._data, totalCopies, axis)
        else:
            repeated = np.tile(self._base._data, (ptDim, ftDim))
        return repeated

    ###########
    # Helpers #
    ###########

    def _convertBaseDtype(self, retDtype):
        """
        Convert the dtype of the Base object if necessary to replace the
        current values with the transformed values.
        """
        baseDtype = self._base._data.dtype
        if baseDtype != np.object_ and retDtype == np.object_:
            self._base._data = self._base._data.astype(np.object_)
        elif baseDtype == np.int_ and retDtype == np.float64:
            self._base._data = self._base._data.astype(np.float64)
        elif baseDtype == np.bool_ and retDtype != np.bool_:
            self._base._data = self._base._data.astype(retDtype)

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _insert_implementation(self, insertBefore, toInsert):
        pass

    @abstractmethod
    def _transform_implementation(self, function, limitTo):
        pass


class MatrixPoints(MatrixAxis, Points):
    """
    Matrix method implementations performed on the points axis.

    Parameters
    ----------
    base : Matrix
        The Matrix instance that will be queried and modified.
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
        startData = self._base._data[:insertBefore, :]
        endData = self._base._data[insertBefore:, :]
        self._base._data = np.concatenate(
            (startData, toInsert._data, endData), 0)

    def _transform_implementation(self, function, limitTo):
        for i, pt in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(pt)
            retArray = np.array(currRet, dtype=function.convertType)
            self._convertBaseDtype(retArray.dtype)
            self._base._data[i, :] = retArray
        # if transformations applied to all data and function.convertType is
        # not object we can convert base with object dtype to a numeric dtype.
        if (self._base._data.dtype == np.object_ and limitTo is None
                and function.convertType is not object):
            self._base._data = self._base._data.astype(function.convertType)

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = self._base._data[:, collapseIndices]
        retainData = self._base._data[:, retainIndices]

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, collapseData, currNumPoints,
            currFtNames, numRetPoints, numRetFeatures)

        self._base._data = numpy2DArray(tmpData)

    def _combineByExpandingFeatures_implementation(
        self, uniqueDict, namesIdx, valuesIdx, uniqueNames, numRetFeatures):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures,
                                                len(valuesIdx))

        self._base._data = numpy2DArray(tmpData)


class MatrixPointsView(PointsView, MatrixPoints):
    """
    Limit functionality of MatrixPoints to read-only.

    Parameters
    ----------
    base : MatrixView
        The MatrixView instance that will be queried.
    """


class MatrixFeatures(MatrixAxis, Features):
    """
    Matrix method implementations performed on the feature axis.

    Parameters
    ----------
    base : Matrix
        The Matrix instance that will be queried and modified.
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
        startData = self._base._data[:, :insertBefore]
        endData = self._base._data[:, insertBefore:]
        self._base._data = np.concatenate(
            (startData, toInsert._data, endData), 1)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)
            retArray = np.array(currRet, dtype=function.convertType)
            self._convertBaseDtype(retArray.dtype)
            self._base._data[:, j] = retArray
        # if transformations applied to all data and function.convertType is
        # not object we can convert base object dtype to a numeric dtype.
        if (self._base._data.dtype == np.object_ and limitTo is None
                and function.convertType is not object):
            self._base._data = self._base._data.astype(function.convertType)

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = np.empty(shape=(len(self._base.points), numRetFeatures),
                           dtype=np.object_)

        tmpData[:, :featureIndex] = self._base._data[:, :featureIndex]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = self._base._data[:, featureIndex + 1:]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._base._data = numpy2DArray(tmpData)


class MatrixFeaturesView(FeaturesView, MatrixFeatures):
    """
    Limit functionality of MatrixFeatures to read-only.

    Parameters
    ----------
    base : MatrixView
        The MatrixView instance that will be queried.
    """
