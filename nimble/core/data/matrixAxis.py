"""
Implementations and helpers specific to performing axis-generic
operations on a nimble Matrix object.
"""

from abc import abstractmethod

import numpy

import nimble
from nimble._utility import numpy2DArray
from .axis import Axis
from .views import AxisView
from .points import Points
from .views import PointsView
from .features import Features
from .views import FeaturesView
from ._dataHelpers import denseAxisUniqueArray, uniqueNameGetter
from ._dataHelpers import fillArrayWithCollapsedFeatures
from ._dataHelpers import fillArrayWithExpandedFeatures

class MatrixAxis(Axis):
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
            ret = self._base.data[targetList]
        else:
            axisVal = 1
            ret = self._base.data[:, targetList]

        if structure != 'copy':
            if len(targetList) == 1:
                targetList = targetList[0]
            # numpy.delete can be faster by passing an integer when possible
            self._base.data = numpy.delete(self._base.data,
                                           targetList, axisVal)

        return nimble.core.data.Matrix(ret, pointNames=pointNames,
                                       featureNames=featureNames,
                                       reuseData=True)

    def _permute_implementation(self, indexPosition):
        # use numpy indexing to change the ordering
        if self._isPoint:
            self._base.data = self._base.data[indexPosition, :]
        else:
            self._base.data = self._base.data[:, indexPosition]


    ##############################
    # High Level implementations #
    ##############################

    def _unique_implementation(self):
        uniqueData, uniqueIndices = denseAxisUniqueArray(self._base,
                                                         self._axis)
        if numpy.array_equal(self._base.data, uniqueData):
            return self._base.copy()

        axisNames, offAxisNames = uniqueNameGetter(self._base, self._axis,
                                                   uniqueIndices)
        if self._isPoint:
            return nimble.data('Matrix', uniqueData, pointNames=axisNames,
                               featureNames=offAxisNames, useLog=False)
        else:
            return nimble.data('Matrix', uniqueData, pointNames=offAxisNames,
                               featureNames=axisNames, useLog=False)

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
            repeated = numpy.repeat(self._base.data, totalCopies, axis)
        else:
            repeated = numpy.tile(self._base.data, (ptDim, ftDim))
        return repeated

    ###########
    # Helpers #
    ###########

    def _convertBaseDtype(self, retDtype):
        """
        Convert the dtype of the Base object if necessary to replace the
        current values with the transformed values.
        """
        baseDtype = self._base.data.dtype
        if baseDtype != numpy.object_ and retDtype == numpy.object_:
            self._base.data = self._base.data.astype(numpy.object_)
        elif baseDtype == numpy.int and retDtype == numpy.float:
            self._base.data = self._base.data.astype(numpy.float)
        elif baseDtype == numpy.bool_ and retDtype != numpy.bool_:
            self._base.data = self._base.data.astype(retDtype)

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
        startData = self._base.data[:insertBefore, :]
        endData = self._base.data[insertBefore:, :]
        self._base.data = numpy.concatenate(
            (startData, toInsert.data, endData), 0)

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)
            retArray = numpy.array(currRet, dtype=function.convertType)
            self._convertBaseDtype(retArray.dtype)
            self._base.data[i, :] = retArray
        # if transformations applied to all data and function.convertType is
        # not object we can convert base with object dtype to a numeric dtype.
        if (self._base.data.dtype == numpy.object_ and limitTo is None
                and function.convertType is not object):
            self._base.data = self._base.data.astype(function.convertType)

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = self._base.data[:, collapseIndices]
        retainData = self._base.data[:, retainIndices]

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, collapseData, currNumPoints,
            currFtNames, numRetPoints, numRetFeatures)

        self._base.data = numpy2DArray(tmpData)

    def _combineByExpandingFeatures_implementation(self, uniqueDict, namesIdx,
                                                   uniqueNames, numRetFeatures,
                                                   numExpanded):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures,
                                                numExpanded)

        self._base.data = numpy2DArray(tmpData)


class MatrixPointsView(PointsView, AxisView, MatrixPoints):
    """
    Limit functionality of MatrixPoints to read-only.

    Parameters
    ----------
    base : MatrixView
        The MatrixView instance that will be queried.
    """
    pass


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
        startData = self._base.data[:, :insertBefore]
        endData = self._base.data[:, insertBefore:]
        self._base.data = numpy.concatenate(
            (startData, toInsert.data, endData), 1)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)
            retArray = numpy.array(currRet, dtype=function.convertType)
            self._convertBaseDtype(retArray.dtype)
            self._base.data[:, j] = retArray
        # if transformations applied to all data and function.convertType is
        # not object we can convert base object dtype to a numeric dtype.
        if (self._base.data.dtype == numpy.object_ and limitTo is None
                and function.convertType is not object):
            self._base.data = self._base.data.astype(function.convertType)

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = numpy.empty(shape=(len(self._base.points), numRetFeatures),
                              dtype=numpy.object_)

        tmpData[:, :featureIndex] = self._base.data[:, :featureIndex]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = self._base.data[:, featureIndex + 1:]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._base.data = numpy2DArray(tmpData)


class MatrixFeaturesView(FeaturesView, AxisView, MatrixFeatures):
    """
    Limit functionality of MatrixFeatures to read-only.

    Parameters
    ----------
    base : MatrixView
        The MatrixView instance that will be queried.
    """
    pass
