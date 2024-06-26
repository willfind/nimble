
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
operations on a nimble Sparse object.
"""
from abc import ABCMeta

import numpy as np

import nimble
from nimble._utility import scipy
from .axis import Axis
from .points import Points
from .views import PointsView
from .features import Features
from .views import FeaturesView

class SparseAxis(Axis, metaclass=ABCMeta):
    """
    Differentiate how Sparse methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    base : Sparse
        The Sparse instance that will be queried and modified.
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
        # SparseView or object dtype
        if (isinstance(self._base, nimble.core.data.BaseView)
                or self._base._data.dtype == np.object_):
            return self._structuralIterative_implementation(
                structure, targetList, pointNames, featureNames)
        # nonview numeric objects
        return self._structuralVectorized_implementation(
            structure, targetList, pointNames, featureNames)

    def _permute_implementation(self, indexPosition):
        # since we want to access with positions in the original
        # data, we reverse the 'map'
        reverseIdxPosition = np.empty(len(indexPosition))
        for i, idxPos in enumerate(indexPosition):
            reverseIdxPosition[idxPos] = i

        if self._isPoint:
            self._base._data.row[:] = reverseIdxPosition[self._base._data.row]
        else:
            self._base._data.col[:] = reverseIdxPosition[self._base._data.col]
        self._base._resetSorted()

    def _transform_implementation(self, function, limitTo):
        modData = []
        modRow = []
        modCol = []

        if self._isPoint:
            modTarget = modRow
            modOther = modCol
        else:
            modTarget = modCol
            modOther = modRow

        for viewID, view in enumerate(self):
            if limitTo is not None and viewID not in limitTo:
                currOut = list(view)
            else:
                currOut = function(view)

            for i, retVal in enumerate(currOut):
                if retVal != 0:
                    modData.append(retVal)
                    modTarget.append(viewID)
                    modOther.append(i)

        baseDtype = self._base._data.dtype
        retDtype = function.convertType
        # if applying transformation to a subset, need to be sure dtype is
        # still compatible with the data that was not transformed
        if (limitTo is not None and
                (baseDtype == np.object_ or
                 (baseDtype == np.float64 and retDtype is not object) or
                 (baseDtype == np.int_ and retDtype not in (float, object))
                 )):
            retDtype = baseDtype

        modData = np.array(modData, dtype=retDtype)
        shape = (len(self._base.points), len(self._base.features))
        self._base._data = scipy.sparse.coo_matrix(
            (modData, (modRow, modCol)), shape=shape)
        self._base._resetSorted()

    def _insert_implementation(self, insertBefore, toInsert):
        """
        Insert the points/features from the toInsert object below the
        provided index in this object, the remaining points/features
        from this object will continue below the inserted
        points/features.
        """
        selfData = self._base._data.data
        addData = toInsert._data.data
        newData = np.concatenate((selfData, addData))
        if self._isPoint:
            selfAxis = self._base._data.row.copy()
            selfOffAxis = self._base._data.col
            addAxis = toInsert._data.row.copy()
            addOffAxis = toInsert._data.col
            addLength = len(toInsert.points)
            shape = (len(self) + addLength, len(self._base.features))
        else:
            selfAxis = self._base._data.col.copy()
            selfOffAxis = self._base._data.row
            addAxis = toInsert._data.col.copy()
            addOffAxis = toInsert._data.row
            addLength = len(toInsert.features)
            shape = (len(self._base.points), len(self) + addLength)

        selfAxis[selfAxis >= insertBefore] += addLength
        addAxis += insertBefore

        newAxis = np.concatenate((selfAxis, addAxis))
        newOffAxis = np.concatenate((selfOffAxis, addOffAxis))

        if self._isPoint:
            rowColTuple = (newAxis, newOffAxis)
        else:
            rowColTuple = (newOffAxis, newAxis)

        self._base._data = scipy.sparse.coo_matrix((newData, rowColTuple),
                                                   shape=shape)
        self._base._resetSorted()

    def _repeat_implementation(self, totalCopies, copyVectorByVector):
        if copyVectorByVector:
            numpyFunc = np.repeat
        else:
            numpyFunc = np.tile
        repData = numpyFunc(self._base._data.data, totalCopies)
        fillDup = np.empty_like(repData, dtype=np.int_)
        if self._isPoint:
            repCol = numpyFunc(self._base._data.col, totalCopies)
            repRow = fillDup
            toRepeat = self._base._data.row
            numRepeatd = len(self)
            startIdx = 0
            shape = ((len(self) * totalCopies), len(self._base.features))
        else:
            repRow = numpyFunc(self._base._data.row, totalCopies)
            repCol = fillDup
            toRepeat = self._base._data.col
            numRepeatd = len(self)
            shape = (len(self._base.points), (len(self) * totalCopies))

        startIdx = 0
        if copyVectorByVector:
            for idx in toRepeat:
                endIdx = startIdx + totalCopies
                indexRange = np.array(range(totalCopies))
                adjustedIndices = indexRange + (totalCopies * idx)
                fillDup[startIdx:endIdx] = adjustedIndices
                startIdx = endIdx
        else:
            for i in range(totalCopies):
                endIdx = len(toRepeat) * (i + 1)
                fillDup[startIdx:endIdx] = toRepeat + (numRepeatd * i)
                startIdx = endIdx

        repeated = scipy.sparse.coo_matrix((repData, (repRow, repCol)),
                                           shape=shape)

        return repeated

    ######################
    # Structural Helpers #
    ######################

    def _structuralVectorized_implementation(self, structure, targetList,
                                             pointNames, featureNames):
        """
        Use scipy csr or csc matrices for indexing targeted values
        """
        if structure != 'copy':
            notTarget = []
            for idx in range(len(self)):
                if idx not in targetList:
                    notTarget.append(idx)

        if self._isPoint:
            data = self._base._data.tocsr()
            targeted = data[targetList, :]
            if structure != 'copy':
                notTargeted = data[notTarget, :]
        else:
            data = self._base._data.tocsc()
            targeted = data[:, targetList]
            if structure != 'copy':
                notTargeted = data[:, notTarget]

        if structure != 'copy':
            self._base._data = notTargeted.tocoo()
            self._base._resetSorted()

        ret = targeted.tocoo()

        return nimble.core.data.Sparse(ret, pointNames=pointNames,
                                       featureNames=featureNames,
                                       reuseData=True)

    def _structuralIterative_implementation(self, structure, targetList,
                                            pointNames, featureNames):
        """
        Iterate through each member to index targeted values
        """
        selfData = self._base._getSparseData()
        dtype = selfData.data.dtype

        targetLength = len(targetList)
        targetData = []
        targetRows = []
        targetCols = []
        targetIdx = 0

        if self._isPoint:
            targetAxis = targetRows
            targetOffAxis = targetCols
            targetAxisData = selfData.row
            offAxisData = selfData.col
        else:
            targetAxis = targetCols
            targetOffAxis = targetRows
            targetAxisData = selfData.col
            offAxisData = selfData.row

        for i in targetList:
            locs = targetAxisData == i
            targetData.extend(selfData.data[locs])
            targetAxis.extend([targetIdx] * sum(locs))
            targetIdx += 1
            targetOffAxis.extend(offAxisData[locs])


        # instantiate return data
        selfShape, targetShape = _calcShapes(self._base._data._shape,
                                             targetLength, self._axis)
        if structure != 'copy':
            keepData = []
            keepRows = []
            keepCols = []
            keepIdx = 0
            if self._isPoint:
                keepAxis = keepRows
                keepOffAxis = keepCols
            else:
                keepAxis = keepCols
                keepOffAxis = keepRows
            for i in range(len(self)):
                if i not in targetList:
                    locs = targetAxisData == i
                    keepData.extend(self._base._data.data[locs])
                    keepAxis.extend([keepIdx] * sum(locs))
                    keepIdx += 1
                    keepOffAxis.extend(offAxisData[locs])
            keepArr = np.array(keepData, dtype=dtype)
            self._base._data = scipy.sparse.coo_matrix(
                (keepArr, (keepRows, keepCols)), shape=selfShape)
            self._base._resetSorted()

        # need to manually set dtype or coo_matrix will force to simplest dtype
        targetArr = np.array(targetData)
        if not np.issubdtype(targetArr.dtype, np.number):
            targetArr = np.array(targetData, dtype=np.object_)
        ret = scipy.sparse.coo_matrix((targetArr, (targetRows, targetCols)),
                                      shape=targetShape)
        return nimble.core.data.Sparse(ret, pointNames=pointNames,
                                       featureNames=featureNames,
                                       reuseData=True)

    def _unique_implementation(self):
        self._base._sortInternal(self._axis)
        count = len(self)
        hasAxisNames = self._namesCreated()
        getAxisName = self._getName
        getAxisNames = self._getNames
        data = self._base._data.data
        row = self._base._data.row
        col = self._base._data.col
        if self._isPoint:
            axisLocator = row
            offAxisLocator = col
            hasOffAxisNames = self._base.features._namesCreated()
            getOffAxisNames = self._base.features.getNames
        else:
            axisLocator = col
            offAxisLocator = row
            hasOffAxisNames = self._base.points._namesCreated()
            getOffAxisNames = self._base.points.getNames

        unique = set()
        uniqueData = []
        uniqueAxis = []
        uniqueOffAxis = []
        keepNames = []
        axisCount = 0
        for i in range(count):
            axisLoc = axisLocator == i
            # data values can look the same but have zeros in different places;
            # zip with offAxis to ensure the locations are the same as well
            key = tuple(zip(data[axisLoc], offAxisLocator[axisLoc]))
            if key not in unique:
                unique.add(key)
                uniqueData.extend(data[axisLoc])
                uniqueAxis.extend([axisCount for _ in range(sum(axisLoc))])
                uniqueOffAxis.extend(offAxisLocator[axisLoc])
                if hasAxisNames:
                    keepNames.append(getAxisName(i))
                axisCount += 1

        if hasAxisNames and keepNames == getAxisNames():
            return self._base.copy()

        axisNames = False
        offAxisNames = False
        if len(keepNames) > 0:
            axisNames = keepNames
        if hasOffAxisNames:
            offAxisNames = getOffAxisNames()

        uniqueData = np.array(uniqueData, dtype=np.object_)
        if self._isPoint:
            shape = (axisCount, len(self._base.features))
            uniqueCoo = scipy.sparse.coo_matrix(
                (uniqueData, (uniqueAxis, uniqueOffAxis)), shape=shape)
            return nimble.data(uniqueCoo, pointNames=axisNames,
                               featureNames=offAxisNames,
                               returnType='Sparse', useLog=False)

        shape = (len(self._base.points), axisCount)
        uniqueCoo = scipy.sparse.coo_matrix(
            (uniqueData, (uniqueOffAxis, uniqueAxis)), shape=shape)
        return nimble.data(uniqueCoo, pointNames=offAxisNames,
                           featureNames=axisNames,
                           returnType='Sparse', useLog=False)


class SparsePoints(SparseAxis, Points):
    """
    Sparse method implementations performed on the points axis.

    Parameters
    ----------
    base : Sparse
        The Sparse instance that will be queried and modified.
    """

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        self._base._sortInternal(self._axis)
        data = self._base._data.data
        row = self._base._data.row
        col = self._base._data.col
        tmpData = []
        tmpRow = []
        tmpCol = []
        for ptIdx in range(len(self)):
            inRetain = [val in retainIndices for val in col]
            inCollapse = [val in collapseIndices for val in col]
            retainData = data[(row == ptIdx) & (inRetain)]
            retainCol = col[(row == ptIdx) & (inRetain)]
            collapseData = data[(row == ptIdx) & (inCollapse)]
            sort = np.argsort(collapseIndices)
            collapseData = collapseData[sort]
            for idx, value in enumerate(collapseData):
                tmpData.extend(retainData)
                tmpRow.extend([ptIdx * len(featuresToCollapse) + idx]
                              * len(retainData))
                tmpCol.extend(list(range(len(retainCol))))
                tmpData.append(currFtNames[idx])
                tmpRow.append(ptIdx * len(featuresToCollapse) + idx)
                tmpCol.append(numRetFeatures - 2)
                tmpData.append(value)
                tmpRow.append(ptIdx * len(featuresToCollapse) + idx)
                tmpCol.append(numRetFeatures - 1)

        tmpData = np.array(tmpData, dtype=np.object_)
        self._base._data = scipy.sparse.coo_matrix(
            (tmpData, (tmpRow, tmpCol)), shape=(numRetPoints, numRetFeatures))
        self._base._resetSorted()

    def _combineByExpandingFeatures_implementation(
        self, uniqueDict, namesIdx, valuesIdx, uniqueNames, numRetFeatures):
        tmpData = []
        tmpRow = []
        tmpCol = []
        numExpanded = len(valuesIdx)
        numNewFts = len(uniqueNames) * numExpanded
        for idx, point in enumerate(uniqueDict):
            tmpPoint = list(point[:namesIdx])
            for name in uniqueNames:
                if name in uniqueDict[point]:
                    tmpPoint.extend(uniqueDict[point][name])
                else:
                    tmpPoint.extend([np.nan] * numExpanded)
            tmpPoint.extend(point[namesIdx:])
            tmpData.extend(tmpPoint)
            tmpRow.extend([idx for _ in range(len(point) + numNewFts)])
            tmpCol.extend(list(range(numRetFeatures)))

        tmpData = np.array(tmpData, dtype=np.object_)
        shape = (len(uniqueDict), numRetFeatures)
        self._base._data = scipy.sparse.coo_matrix((tmpData, (tmpRow, tmpCol)),
                                                   shape=shape)
        self._base._resetSorted()


class SparsePointsView(PointsView, SparsePoints):
    """
    Limit functionality of SparsePoints to read-only.

    Parameters
    ----------
    base : SparseView
        The SparseView instance that will be queried.
    """

    #########################
    # Query implementations #
    #########################

    def _unique_implementation(self):
        unique = self._base.copy(to='Sparse')
        return unique.points._unique_implementation()

    def _repeat_implementation(self, totalCopies, copyVectorByVector):
        copy = self._base.copy(to='Sparse')
        return copy.points._repeat_implementation(totalCopies,
                                                  copyVectorByVector)


class SparseFeatures(SparseAxis, Features):
    """
    Sparse method implementations performed on the feature axis.

    Parameters
    ----------
    base : Sparse
        The Sparse instance that will be queried and modified.
    """

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        keep = self._base._data.col != featureIndex
        tmpData = self._base._data.data[keep]
        tmpRow = self._base._data.row[keep]
        tmpCol = self._base._data.col[keep]

        shift = tmpCol > featureIndex
        tmpCol[shift] = tmpCol[shift] + numResultingFts - 1

        for idx in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[idx])
            tmpData = np.concatenate((tmpData, newFeat))
            newRows = list(range(len(self._base.points)))
            tmpRow = np.concatenate((tmpRow, newRows))
            newCols = [featureIndex + idx for _
                       in range(len(self._base.points))]
            tmpCol = np.concatenate((tmpCol, newCols))

        tmpData = np.array(tmpData, dtype=np.object_)
        shape = (len(self._base.points), numRetFeatures)
        self._base._data = scipy.sparse.coo_matrix((tmpData, (tmpRow, tmpCol)),
                                                   shape=shape)
        self._base._resetSorted()


class SparseFeaturesView(FeaturesView, SparseFeatures):
    """
    Limit functionality of SparseFeatures to read-only.

    Parameters
    ----------
    base : SparseView
        The SparseView instance that will be queried.
    """

    #########################
    # Query implementations #
    #########################

    def _unique_implementation(self):
        unique = self._base.copy(to='Sparse')
        return unique.features._unique_implementation()

    def _repeat_implementation(self, totalCopies, copyVectorByVector):
        copy = self._base.copy(to='Sparse')
        return copy.features._repeat_implementation(totalCopies,
                                                    copyVectorByVector)

###################
# Generic Helpers #
###################

def _calcShapes(currShape, numExtracted, axisType):
    if axisType == "feature":
        (rowShape, colShape) = currShape
        selfRowShape = rowShape
        selfColShape = colShape - numExtracted
        extRowShape = rowShape
        extColShape = numExtracted
    elif axisType == "point":
        rowShape = currShape[0]
        # flattened call shape since can be more than 2D
        colShape = int(np.prod(currShape[1:]))
        selfRowShape = rowShape - numExtracted
        selfColShape = colShape
        extRowShape = numExtracted
        extColShape = colShape

    return ((selfRowShape, selfColShape), (extRowShape, extColShape))
