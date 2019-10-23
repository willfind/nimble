"""
Method implementations and helpers acting specifically on features in a
List object.
"""

from __future__ import absolute_import

import numpy

from nimble.exceptions import InvalidArgumentValue

from .axis_view import AxisView
from .listAxis import ListAxis
from .features import Features
from .features_view import FeaturesView

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

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        insert = toAdd.copy('pythonlist')
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

        self._base.data = allData
        self._base._numFeatures += len(toAdd.features)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)
            if len(currRet) != len(self._base.points):
                msg = "function must return an iterable with as many elements "
                msg += "as points in this object"
                raise InvalidArgumentValue(msg)

            for i in range(len(self._base.points)):
                self._base.data[i][j] = currRet[i]

    # def _flattenToOne_implementation(self):
    #     result = []
    #     for i in range(len(self._base.features)):
    #         for p in self._base.data:
    #             result.append([p[i]])
    #
    #     self._base.data = result
    #     self._base._numFeatures = 1
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     result = []
    #     numFeatures = divideInto
    #     numPoints = len(self._base.points) // numFeatures
    #     # reconstruct the shape we want, point by point. We access the
    #     # singleton values from the current data in an out of order iteration
    #     for i in range(numPoints):
    #         temp = []
    #         for j in range(i, len(self._base.points), numPoints):
    #             temp += self._base.data[j]
    #         result.append(temp)
    #
    #     self._base.data = result
    #     self._base._numFeatures = numFeatures

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = numpy.empty(shape=(len(self._base.points), numRetFeatures),
                              dtype=numpy.object_)

        tmpData[:, :featureIndex] = [ft[:featureIndex] for ft
                                     in self._base.data]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = [ft[featureIndex + 1:] for ft in self._base.data]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._base.data = tmpData.tolist()
        self._base._numFeatures = numRetFeatures

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._base)

class ListFeaturesView(FeaturesView, AxisView, ListFeatures):
    """
    Limit functionality of ListFeatures to read-only.

    Parameters
    ----------
    base : ListView
        The ListView instance that will be queried.
    """
    pass

class nzIt(object):
    """
    Non-zero iterator to return when iterating through each feature.
    """
    def __init__(self, source):
        self._source = source
        self._pIndex = 0
        self._pStop = len(source.points)
        self._fIndex = 0
        self._fStop = len(source.features)

    def __iter__(self):
        return self

    def next(self):
        """
        Get next non zero value.
        """
        while self._fIndex < self._fStop:
            value = self._source.data[self._pIndex][self._fIndex]

            self._pIndex += 1
            if self._pIndex >= self._pStop:
                self._pIndex = 0
                self._fIndex += 1

            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
