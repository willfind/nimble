"""
Implementations and helpers specific to performing axis-generic
operations on a UML Sparse object.
"""
from __future__ import absolute_import
from abc import abstractmethod

import numpy

import UML
from UML.exceptions import ArgumentException
from .axis import Axis
from .base import cmp_to_key

scipy = UML.importModule('scipy')
if scipy is not None:
    from scipy.sparse import coo_matrix

class SparseAxis(Axis):
    """
    Differentiate how Sparse methods act dependent on the axis.

    Also provides abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    axis : str
        The axis ('point' or 'feature') which the function will be
        applied to.
    source : UML data object
        The object containing point and feature data.
    """
    def __init__(self, axis, source, **kwds):
        self.axis = axis
        self.source = source
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(SparseAxis, self).__init__(**kwds)

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
        # SparseView or object dtype
        if (self.source.data.data is None
                or self.source.data.data.dtype == numpy.object_):
            return self._structuralIterative_implementation(
                structure, targetList)
        # nonview numeric objects
        return self._structuralVectorized_implementation(
            structure, targetList)

    def _sort_implementation(self, sortBy, sortHelper):
        scorer = None
        comparator = None
        if self.axis == 'point':
            viewMaker = self.source.pointView
            getViewIter = self.source.pointIterator
            indexGetter = self.source.getPointIndex
            nameGetter = self.source.getPointName
            nameGetterStr = 'getPointName'
            names = self.source.getPointNames()
        else:
            viewMaker = self.source.featureView
            getViewIter = self.source.featureIterator
            indexGetter = self.source.getFeatureIndex
            nameGetter = self.source.getFeatureName
            nameGetterStr = 'getFeatureName'
            names = self.source.getFeatureNames()

        if isinstance(sortHelper, list):
            sortedData = []
            idxDict = {val: idx for idx, val in enumerate(sortHelper)}
            if self.axis == 'point':
                sortedData = [idxDict[val] for val in self.source.data.row]
                self.source.data.row = numpy.array(sortedData)
            else:
                sortedData = [idxDict[val] for val in self.source.data.col]
                self.source.data.col = numpy.array(sortedData)
            newNameOrder = [names[idx] for idx in sortHelper]
            self.source._sorted = None
            return newNameOrder

        test = viewMaker(0)
        try:
            sortHelper(test)
            scorer = sortHelper
        except TypeError:
            pass
        try:
            sortHelper(test, test)
            comparator = sortHelper
        except TypeError:
            pass

        if sortHelper is not None and scorer is None and comparator is None:
            msg = "sortHelper is neither a scorer or a comparator"
            raise ArgumentException(msg)

        if comparator is not None:
            # make array of views
            viewArray = []
            viewIter = getViewIter()
            for v in viewIter:
                viewArray.append(v)

            viewArray.sort(key=cmp_to_key(comparator))
            indexPosition = []
            for i in range(len(viewArray)):
                index = indexGetter(getattr(viewArray[i], nameGetterStr)(0))
                indexPosition.append(index)
            indexPosition = numpy.array(indexPosition)
        elif hasattr(scorer, 'permuter'):
            scoreArray = scorer.indices
            indexPosition = numpy.argsort(scoreArray)
        else:
            # make array of views
            viewArray = []
            viewIter = getViewIter()
            for v in viewIter:
                viewArray.append(v)

            scoreArray = viewArray
            if scorer is not None:
                # use scoring function to turn views into values
                for i in range(len(viewArray)):
                    scoreArray[i] = scorer(viewArray[i])
            else:
                for i in range(len(viewArray)):
                    scoreArray[i] = viewArray[i][sortBy]

            # use numpy.argsort to make desired index array
            # this results in an array whose ith entry contains the the
            # index into the data of the value that should be in the ith
            # position.
            indexPosition = numpy.argsort(scoreArray)

        # since we want to access with with positions in the original
        # data, we reverse the 'map'
        reverseIdxPosition = numpy.empty(indexPosition.shape[0])
        for i in range(indexPosition.shape[0]):
            reverseIdxPosition[indexPosition[i]] = i

        if self.axis == 'point':
            self.source.data.row[:] = reverseIdxPosition[self.source.data.row]
        else:
            self.source.data.col[:] = reverseIdxPosition[self.source.data.col]

        # we need to return an array of the feature names in their new order.
        # convert indices of their previous location into their names
        newNameOrder = []
        for i in range(len(indexPosition)):
            oldIndex = indexPosition[i]
            newName = nameGetter(oldIndex)
            newNameOrder.append(newName)

        self.source._sorted = None
        return newNameOrder

    def _transform_implementation(self, function, limitTo):
        modData = []
        modRow = []
        modCol = []

        if self.axis == 'point':
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
                # currRet might return an ArgumentException with a message which needs to be
                # formatted with the axis and current index before being raised
                if isinstance(currOut, ArgumentException):
                    currOut.value = currOut.value.format(self.axis, viewID)
                    raise currOut

            # easy way to reuse code if we have a singular return
            if not hasattr(currOut, '__iter__'):
                currOut = [currOut]

            # if there are multiple values, they must be random accessible
            if not hasattr(currOut, '__getitem__'):
                msg = "function must return random accessible data "
                msg += "(ie has a __getitem__ attribute)"
                raise ArgumentException(msg)

            for i, retVal in enumerate(currOut):
                if retVal != 0:
                    modData.append(retVal)
                    modTarget.append(viewID)
                    modOther.append(i)

        if len(modData) != 0:
            try:
                modData = numpy.array(modData, dtype=numpy.float)
            except Exception:
                modData = numpy.array(modData, dtype=numpy.object_)
            shape = (len(self.source.points), len(self.source.features))
            self.source.data = coo_matrix((modData, (modRow, modCol)),
                                          shape=shape)
            self.source._sorted = None

        ret = None
        return ret

    ######################
    # Structural Helpers #
    ######################

    def _structuralVectorized_implementation(self, structure, targetList):
        """
        Use scipy csr or csc matrices for indexing targeted values
        """
        axisNames = []
        if self.axis == 'point':
            getAxisName = self.source.getPointName
            getOtherNames = self.source.getFeatureNames
            data = self.source.data.tocsr()
            targeted = data[targetList, :]
            if structure != 'copy':
                notTarget = []
                for idx in range(len(self.source.points)):
                    if idx not in targetList:
                        notTarget.append(idx)
                notTargeted = data[notTarget, :]
        else:
            getAxisName = self.source.getFeatureName
            getOtherNames = self.source.getPointNames
            data = self.source.data.tocsc()
            targeted = data[:, targetList]
            if structure != 'copy':
                notTarget = []
                for idx in range(len(self.source.features)):
                    if idx not in targetList:
                        notTarget.append(idx)
                notTargeted = data[:, notTarget]

        self.source._validateAxis(self.axis)

        for index in targetList:
            axisNames.append(getAxisName(index))
        otherNames = getOtherNames()

        if structure != 'copy':
            self.source.data = notTargeted.tocoo()
            self.source._sortInternal(self.axis)

        ret = targeted.tocoo()

        if self.axis == 'point':
            return UML.data.Sparse(ret, pointNames=axisNames,
                                   featureNames=otherNames,
                                   reuseData=True)

        return UML.data.Sparse(ret, pointNames=otherNames,
                               featureNames=axisNames,
                               reuseData=True)

    def _structuralIterative_implementation(self, structure, targetList):
        """
        Iterate through each member to index targeted values
        """
        dtype = numpy.object_
        if self.axis == 'point':
            viewIterator = self.source.pointIterator
        else:
            viewIterator = self.source.featureIterator

        targetLength = len(targetList)
        targetData = []
        targetRows = []
        targetCols = []
        keepData = []
        keepRows = []
        keepCols = []
        keepIndex = 0

        # iterate through self.axis data
        for targetID, view in enumerate(viewIterator()):
            # coo_matrix data for return object
            if targetID in targetList:
                for otherID, value in enumerate(view.data.data):
                    targetData.append(value)
                    if self.axis == 'point':
                        targetRows.append(targetList.index(targetID))
                        targetCols.append(view.data.col[otherID])
                    else:
                        targetRows.append(view.data.row[otherID])
                        targetCols.append(targetList.index(targetID))
            # coo_matrix data for modified self.source
            elif structure != 'copy':
                for otherID, value in enumerate(view.data.data):
                    keepData.append(value)
                    if self.axis == 'point':
                        keepRows.append(keepIndex)
                        keepCols.append(view.data.col[otherID])
                    else:
                        keepRows.append(view.data.row[otherID])
                        keepCols.append(keepIndex)
                keepIndex += 1

        # instantiate return data
        selfShape, targetShape = _calcShapes(self.source.data.shape,
                                             targetLength, self.axis)
        if structure != 'copy':
            keepData = numpy.array(keepData, dtype=dtype)
            self.source.data = coo_matrix((keepData, (keepRows, keepCols)),
                                          shape=selfShape)
        # need to manually set dtype or coo_matrix will force to simplest dtype
        targetData = numpy.array(targetData, dtype=dtype)
        ret = coo_matrix((targetData, (targetRows, targetCols)),
                         shape=targetShape)

        # get names for return obj
        pnames = []
        fnames = []
        if self.axis == 'point':
            for index in targetList:
                pnames.append(self.source.getPointName(index))
            fnames = self.source.getFeatureNames()
        else:
            pnames = self.source.getPointNames()
            for index in targetList:
                fnames.append(self.source.getFeatureName(index))

        return UML.data.Sparse(ret, pointNames=pnames, featureNames=fnames,
                               reuseData=True)

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _add_implementation(self, toAdd, insertBefore):
        pass

    @abstractmethod
    def _flattenToOne_implementation(self):
        pass

    @abstractmethod
    def _unflattenFromOne_implementation(self, divideInto):
        pass

###################
# Generic Helpers #
###################

def _calcShapes(currShape, numExtracted, axisType):
    (rowShape, colShape) = currShape
    if axisType == "feature":
        selfRowShape = rowShape
        selfColShape = colShape - numExtracted
        extRowShape = rowShape
        extColShape = numExtracted
    elif axisType == "point":
        selfRowShape = rowShape - numExtracted
        selfColShape = colShape
        extRowShape = numExtracted
        extColShape = colShape

    return ((selfRowShape, selfColShape), (extRowShape, extColShape))
