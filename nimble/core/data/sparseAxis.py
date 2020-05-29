"""
Implementations and helpers specific to performing axis-generic
operations on a nimble Sparse object.
"""

from abc import abstractmethod

import numpy

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.utility import scipy
from .axis import Axis
from .points import Points
from .dataHelpers import sortIndexPosition

class SparseAxis(Axis):
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
                or self._base.data.dtype == numpy.object_):
            return self._structuralIterative_implementation(
                structure, targetList, pointNames, featureNames)
        # nonview numeric objects
        return self._structuralVectorized_implementation(
            structure, targetList, pointNames, featureNames)

    def _sort_implementation(self, indexPosition):
        # since we want to access with with positions in the original
        # data, we reverse the 'map'
        reverseIdxPosition = numpy.empty(len(indexPosition))
        for i, idxPos in enumerate(indexPosition):
            reverseIdxPosition[idxPos] = i

        if self._isPoint:
            self._base.data.row[:] = reverseIdxPosition[self._base.data.row]
        else:
            self._base.data.col[:] = reverseIdxPosition[self._base.data.col]
        self._base._sorted = None

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

        baseDtype = self._base.data.dtype
        retDtype = function.convertType
        # if applying transformation to a subset, need to be sure dtype is
        # still compatible with the data that was not transformed
        if (limitTo is not None and
                (baseDtype == numpy.object_ or
                 (baseDtype == numpy.float and retDtype is not object) or
                 (baseDtype == numpy.int and retDtype not in (float, object))
                 )):
            retDtype = baseDtype

        modData = numpy.array(modData, dtype=retDtype)
        shape = (len(self._base.points), len(self._base.features))
        self._base.data = scipy.sparse.coo_matrix(
            (modData, (modRow, modCol)), shape=shape)
        self._base._sorted = None

    def _insert_implementation(self, insertBefore, toInsert):
        """
        Insert the points/features from the toInsert object below the
        provided index in this object, the remaining points/features
        from this object will continue below the inserted
        points/features.
        """
        selfData = self._base.data.data
        addData = toInsert.data.data
        newData = numpy.concatenate((selfData, addData))
        if self._isPoint:
            selfAxis = self._base.data.row.copy()
            selfOffAxis = self._base.data.col
            addAxis = toInsert.data.row.copy()
            addOffAxis = toInsert.data.col
            addLength = len(toInsert.points)
            shape = (len(self) + addLength, len(self._base.features))
        else:
            selfAxis = self._base.data.col.copy()
            selfOffAxis = self._base.data.row
            addAxis = toInsert.data.col.copy()
            addOffAxis = toInsert.data.row
            addLength = len(toInsert.features)
            shape = (len(self._base.points), len(self) + addLength)

        selfAxis[selfAxis >= insertBefore] += addLength
        addAxis += insertBefore

        newAxis = numpy.concatenate((selfAxis, addAxis))
        newOffAxis = numpy.concatenate((selfOffAxis, addOffAxis))

        if self._isPoint:
            rowColTuple = (newAxis, newOffAxis)
        else:
            rowColTuple = (newOffAxis, newAxis)

        self._base.data = scipy.sparse.coo_matrix((newData, rowColTuple),
                                                  shape=shape)
        self._base._sorted = None

    def _repeat_implementation(self, totalCopies, copyValueByValue):
        if copyValueByValue:
            numpyFunc = numpy.repeat
        else:
            numpyFunc = numpy.tile
        repData = numpyFunc(self._base.data.data, totalCopies)
        fillDup = numpy.empty_like(repData, dtype=numpy.int)
        if self._isPoint:
            repCol = numpyFunc(self._base.data.col, totalCopies)
            repRow = fillDup
            toRepeat = self._base.data.row
            numRepeatd = len(self)
            startIdx = 0
            shape = ((len(self) * totalCopies), len(self._base.features))
        else:
            repRow = numpyFunc(self._base.data.row, totalCopies)
            repCol = fillDup
            toRepeat = self._base.data.col
            numRepeatd = len(self)
            shape = (len(self._base.points), (len(self) * totalCopies))

        startIdx = 0
        if copyValueByValue:
            for idx in toRepeat:
                endIdx = startIdx + totalCopies
                indexRange = numpy.array(range(totalCopies))
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
        self._base._sorted = None

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
            data = self._base.data.tocsr()
            targeted = data[targetList, :]
            if structure != 'copy':
                notTargeted = data[notTarget, :]
        else:
            data = self._base.data.tocsc()
            targeted = data[:, targetList]
            if structure != 'copy':
                notTargeted = data[:, notTarget]

        if structure != 'copy':
            self._base.data = notTargeted.tocoo()
            self._base._sorted = None

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
        selfShape, targetShape = _calcShapes(self._base.data._shape,
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
                    keepData.extend(self._base.data.data[locs])
                    keepAxis.extend([keepIdx] * sum(locs))
                    keepIdx += 1
                    keepOffAxis.extend(offAxisData[locs])
            keepArr = numpy.array(keepData, dtype=dtype)
            self._base.data = scipy.sparse.coo_matrix(
                (keepArr, (keepRows, keepCols)), shape=selfShape)
            self._base._sorted = None

        # need to manually set dtype or coo_matrix will force to simplest dtype
        targetArr = numpy.array(targetData)
        if not numpy.issubdtype(targetArr.dtype, numpy.number):
            targetArr = numpy.array(targetData, dtype=numpy.object_)
        ret = scipy.sparse.coo_matrix((targetArr, (targetRows, targetCols)),
                                      shape=targetShape)
        return nimble.core.data.Sparse(ret, pointNames=pointNames,
                                  featureNames=featureNames, reuseData=True)

    def _unique_implementation(self):
        if self._base._sorted is None:
            self._base._sortInternal("feature")
        count = len(self)
        hasAxisNames = self._namesCreated()
        getAxisName = self._getName
        getAxisNames = self._getNames
        data = self._base.data.data
        row = self._base.data.row
        col = self._base.data.col
        if self._isPoint:
            axisLocator = row
            offAxisLocator = col
            hasOffAxisNames = self._base._featureNamesCreated()
            getOffAxisNames = self._base.features.getNames
        else:
            axisLocator = col
            offAxisLocator = row
            hasOffAxisNames = self._base._pointNamesCreated()
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
        self._base._sorted = None

        uniqueData = numpy.array(uniqueData, dtype=numpy.object_)
        if self._isPoint:
            shape = (axisCount, len(self._base.features))
            uniqueCoo = scipy.sparse.coo_matrix(
                (uniqueData, (uniqueAxis, uniqueOffAxis)), shape=shape)
            return nimble.data('Sparse', uniqueCoo, pointNames=axisNames,
                               featureNames=offAxisNames, useLog=False)
        else:
            shape = (len(self._base.points), axisCount)
            uniqueCoo = scipy.sparse.coo_matrix(
                (uniqueData, (uniqueOffAxis, uniqueAxis)), shape=shape)
            return nimble.data('Sparse', uniqueCoo, pointNames=offAxisNames,
                               featureNames=axisNames, useLog=False)

    ####################
    # Abstract Methods #
    ####################

    # @abstractmethod
    # def _flattenToOne_implementation(self):
    #     pass
    #
    # @abstractmethod
    # def _unflattenFromOne_implementation(self, divideInto):
    #     pass

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
        colShape = int(numpy.prod(currShape[1:]))
        selfRowShape = rowShape - numExtracted
        selfColShape = colShape
        extRowShape = numExtracted
        extColShape = colShape

    return ((selfRowShape, selfColShape), (extRowShape, extColShape))
