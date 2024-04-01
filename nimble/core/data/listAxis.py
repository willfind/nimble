
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
operations on a nimble List object.
"""

import copy
from abc import ABCMeta, abstractmethod

import numpy as np

import nimble
from .axis import Axis
from .points import Points
from .views import PointsView
from .features import Features
from .views import FeaturesView
from ._dataHelpers import denseAxisUniqueArray
from ._dataHelpers import fillArrayWithCollapsedFeatures
from ._dataHelpers import fillArrayWithExpandedFeatures

class ListAxis(Axis, metaclass=ABCMeta):
    """
    Differentiate how List methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    base : List
        The List instance that will be queried and modified.
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
            satisfying = [copy.copy(self._base._data[pt]) for pt in targetList]
            if structure != 'copy':
                keepList = [i for i in range(len(self)) if i not in targetList]
                self._base._data = [self._base._data[pt] for pt in keepList]
            if satisfying == []:
                return nimble.core.data.List(satisfying, pointNames=pointNames,
                                             featureNames=featureNames,
                                             shape=((0, self._base.shape[1])),
                                             checkAll=False, reuseData=True)

        else:
            if self._base._data == []:
                # create empty matrix with correct shape
                shape = (len(self._base.points), len(targetList))
                satisfying = np.empty(shape, dtype=np.object_)
            else:
                satisfying = [[self._base._data[pt][ft] for ft in targetList]
                              for pt in range(len(self._base.points))]
            if structure != 'copy':
                keepList = [i for i in range(len(self)) if i not in targetList]
                self._base._data = [
                    [self._base._data[pt][ft] for ft in keepList]
                    for pt in range(len(self._base.points))
                    ]
                remainingFts = self._base._numFeatures - len(targetList)
                self._base._numFeatures = remainingFts

        return nimble.core.data.List(satisfying, pointNames=pointNames,
                                     featureNames=featureNames,
                                     checkAll=False, reuseData=True)

    def _permute_implementation(self, indexPosition):
        # run through target axis and change indices
        if self._isPoint:
            source = copy.copy(self._base._data)
            for i, _ in enumerate(self._base._data):
                self._base._data[i] = source[indexPosition[i]]
        else:
            for i, _ in enumerate(self._base._data):
                currPoint = self._base._data[i]
                temp = copy.copy(currPoint)
                for j, idxPos in enumerate(indexPosition):
                    currPoint[j] = temp[idxPos]

    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = denseAxisUniqueArray(self._base,
                                                         self._axis)
        uniqueData = uniqueData.tolist()
        if self._base._data == uniqueData:
            return self._base.copy()

        axisNames, offAxisNames = self._uniqueNameGetter(uniqueIndices)
        if self._isPoint:
            return nimble.data(uniqueData, pointNames=axisNames,
                               featureNames=offAxisNames,
                               returnType='List', useLog=False)

        return nimble.data(uniqueData, pointNames=offAxisNames,
                           featureNames=axisNames,
                           returnType='List', useLog=False)

    def _repeat_implementation(self, totalCopies, copyVectorByVector):
        if self._isPoint:
            if copyVectorByVector:
                repeated = [list(lst) for lst in self._base._data
                            for _ in range(totalCopies)]
            else:
                repeated = [list(lst) for _ in range(totalCopies)
                            for lst in self._base._data]
        else:
            repeated = []
            for lst in self._base._data:
                if not isinstance(lst, list): # FeatureViewer
                    lst = list(lst)
                if copyVectorByVector:
                    extended = []
                    for v in lst:
                        extended.extend([v] * totalCopies)
                else:
                    extended = lst * totalCopies
                repeated.append(extended)
        return repeated

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _insert_implementation(self, insertBefore, toInsert):
        pass

    @abstractmethod
    def _transform_implementation(self, function, limitTo):
        pass


class ListPoints(ListAxis, Points):
    """
    List method implementations performed on the points axis.

    Parameters
    ----------
    base : List
        The List instance that will be queried and modified.
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
        insert = toInsert.copy('List')._data
        if insertBefore != 0 and insertBefore != len(self):
            start = self._base._data[:insertBefore]
            end = self._base._data[insertBefore:]
            allData = start + insert + end
        elif insertBefore == 0:
            allData = insert + self._base._data
        else:
            allData = self._base._data + insert

        self._base._data = allData

    def _transform_implementation(self, function, limitTo):
        for i, pt in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(pt)

            self._base._data[i] = list(currRet)

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = []
        retainData = []
        for pt in self._base._data:
            collapseFeatures = []
            retainFeatures = []
            for idx in collapseIndices:
                collapseFeatures.append(pt[idx])
            for idx in retainIndices:
                retainFeatures.append(pt[idx])
            collapseData.append(collapseFeatures)
            retainData.append(retainFeatures)

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, np.array(collapseData),
            currNumPoints, currFtNames, numRetPoints, numRetFeatures)

        self._base._data = tmpData.tolist()
        self._base._numFeatures = numRetFeatures

    def _combineByExpandingFeatures_implementation(
        self, uniqueDict, namesIdx, valuesIdx, uniqueNames, numRetFeatures):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures,
                                                len(valuesIdx))

        self._base._data = tmpData.tolist()
        self._base._numFeatures = numRetFeatures


class ListPointsView(PointsView, ListPoints):
    """
    Limit functionality of ListPoints to read-only.

    Parameters
    ----------
    base : ListView
        The ListView instance that will be queried.
    """


class ListFeatures(ListAxis, Features):
    """
    List method implementations performed on the feature axis.

    Parameters
    ----------
    base : List
        The List instance that will be queried and modified.
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
        insert = toInsert.copy('pythonlist')
        if insertBefore != 0 and insertBefore != len(self):
            breakIdx = insertBefore - 1
            restartIdx = insertBefore
            start = self._base.view(featureEnd=breakIdx).copy('pythonlist')
            end = self._base.view(featureStart=restartIdx).copy('pythonlist')
            zipChunks = zip(start, insert, end)
            allData = list(map(lambda pt: pt[0] + pt[1] + pt[2], zipChunks))
        elif insertBefore == 0:
            end = self._base.copy('pythonlist')
            allData = list(map(lambda pt: pt[0] + pt[1], zip(insert, end)))
        else:
            start = self._base.copy('pythonlist')
            allData = list(map(lambda pt: pt[0] + pt[1], zip(start, insert)))

        self._base._data = allData
        self._base._numFeatures += len(toInsert.features)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)

            for i in range(len(self._base.points)):
                self._base._data[i][j] = currRet[i]

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = np.empty(shape=(len(self._base.points), numRetFeatures),
                           dtype=np.object_)

        tmpData[:, :featureIndex] = [ft[:featureIndex] for ft
                                     in self._base._data]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = [ft[featureIndex + 1:] for ft in self._base._data]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._base._data = tmpData.tolist()
        self._base._numFeatures = numRetFeatures


class ListFeaturesView(FeaturesView, ListFeatures):
    """
    Limit functionality of ListFeatures to read-only.

    Parameters
    ----------
    base : ListView
        The ListView instance that will be queried.
    """
