"""
Method implementations and helpers acting specifically on points in a
Matrix object.
"""
from __future__ import absolute_import

import numpy

from UML.exceptions import ArgumentException
from .axis import Axis
from .axis_view import AxisView
from .matrixAxis import MatrixAxis
from .points import Points

class MatrixPoints(MatrixAxis, Axis, Points):
    """
    Matrix method implementations performed on the points axis.

    Parameters
    ----------
    source : UML data object
        The object containing the points data.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, source, **kwds):
        self._source = source
        self._axis = 'point'
        kwds['axis'] = self._axis
        kwds['source'] = self._source
        super(MatrixPoints, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the points from the toAdd object below the provided index
        in this object, the remaining points from this object will
        continue below the inserted points.
        """
        startData = self._source.data[:insertBefore, :]
        endData = self._source.data[insertBefore:, :]
        self._source.data = numpy.concatenate((startData, toAdd.data, endData),
                                              0)

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)
            # currRet might return an ArgumentException with a message which
            # needs to be formatted with the axis and current index before
            # being raised
            if isinstance(currRet, ArgumentException):
                currRet.value = currRet.value.format('point', i)
                raise currRet
            if len(currRet) != len(self._source.features):
                msg = "function must return an iterable with as many elements "
                msg += "as features in this object"
                raise ArgumentException(msg)
            try:
                currRet = numpy.array(currRet, dtype=numpy.float)
            except ValueError:
                currRet = numpy.array(currRet, dtype=numpy.object_)
                # need self.data to be object dtype if inserting object dtype
                if numpy.issubdtype(self._source.data.dtype, numpy.number):
                    self._source.data = self._source.data.astype(numpy.object_)
            reshape = (1, len(self._source.features))
            self._source.data[i, :] = numpy.array(currRet).reshape(reshape)

    # def _flattenToOne_implementation(self):
    #     numElements = len(self._source.points) * len(self._source.features)
    #     self._source.data = self._source.data.reshape((1, numElements),
    #                                                 order='C')
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     numPoints = divideInto
    #     numFeatures = len(self._source.features) // numPoints
    #     self._source.data = self._source.data.reshape((numPoints, numFeatures),
    #                                                 order='C')

    def _nonZeroIterator_implementation(self):
        class nzIt(object):
            def __init__(self, source):
                self._source = source
                self._pIndex = 0
                self._pStop = len(source.points)
                self._fIndex = 0
                self._fStop = len(source.features)

            def __iter__(self):
                return self

            def next(self):
                while (self._pIndex < self._pStop):
                    value = self._source.data[self._pIndex, self._fIndex]

                    self._fIndex += 1
                    if self._fIndex >= self._fStop:
                        self._fIndex = 0
                        self._pIndex += 1

                    if value != 0:
                        return value

                raise StopIteration

            def __next__(self):
                return self.next()

        return nzIt(self._source)

class MatrixPointsView(AxisView, MatrixPoints, MatrixAxis, Axis, Points):
    def __init__(self, source, **kwds):
        kwds['source'] = source
        kwds['axis'] = 'point'
        super(MatrixPointsView, self).__init__(**kwds)
