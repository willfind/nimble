"""
Method implementations and helpers acting specifically on features in a
Matrix object.
"""
from __future__ import absolute_import

import numpy

from UML.exceptions import ArgumentException
from .axis_view import AxisView
from .matrixAxis import MatrixAxis
from .features import Features
from .features_view import FeaturesView

class MatrixFeatures(MatrixAxis, Features):
    """
    Matrix method implementations performed on the feature axis.

    Parameters
    ----------
    source : UML data object
        The object containing point and feature data.
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
        startData = self._source.data[:, :insertBefore]
        endData = self._source.data[:, insertBefore:]
        self._source.data = numpy.concatenate((startData, toAdd.data, endData),
                                              1)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)
            # currRet might return an ArgumentException with a message which
            # needs to be formatted with the axis and current index before
            # being raised
            if isinstance(currRet, ArgumentException):
                currRet.value = currRet.value.format('feature', j)
                raise currRet
            if len(currRet) != len(self._source.points):
                msg = "function must return an iterable with as many elements "
                msg += "as points in this object"
                raise ArgumentException(msg)
            try:
                currRet = numpy.array(currRet, dtype=numpy.float)
            except ValueError:
                currRet = numpy.array(currRet, dtype=numpy.object_)
                # need self.data to be object dtype if inserting object dtype
                if numpy.issubdtype(self._source.data.dtype, numpy.number):
                    self._source.data = self._source.data.astype(numpy.object_)
            reshape = (len(self._source.points), 1)
            self._source.data[:, j] = numpy.array(currRet).reshape(reshape)

    # def _flattenToOne_implementation(self):
    #     numElements = len(self._source.points) * len(self._source.features)
    #     self._source.data = self._source.data.reshape((numElements, 1),
    #                                                 order='F')
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     numFeatures = divideInto
    #     numPoints = len(self._source.points) // numFeatures
    #     self._source.data = self._source.data.reshape((numPoints, numFeatures),
    #                                                 order='F')

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = numpy.empty(shape=(len(self._source.points), numRetFeatures),
                              dtype=numpy.object_)

        tmpData[:, :featureIndex] = self._source.data[:, :featureIndex]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = self._source.data[:, featureIndex + 1:]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._source.data = numpy.matrix(tmpData)

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

class MatrixFeaturesView(FeaturesView, AxisView, MatrixFeatures):
    """
    Limit functionality of MatrixFeatures to read-only
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
            value = self._source.data[self._pIndex, self._fIndex]

            self._pIndex += 1
            if self._pIndex >= self._pStop:
                self._pIndex = 0
                self._fIndex += 1

            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
