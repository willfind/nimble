"""
Method implementations and helpers acting specifically on points in a
Matrix object.
"""
from __future__ import absolute_import

import numpy

from UML.exceptions import ArgumentException
from .axis import Axis
from .matrixAxis import MatrixAxis
from .points import Points

class MatrixPoints(MatrixAxis, Axis, Points):
    """
    Matrix method implementations performed on the points axis.

    Parameters
    ----------
    source : UML data object
        The object containing the points data.
    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'point'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
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
        startData = self.source.data[:insertBefore, :]
        endData = self.source.data[insertBefore:, :]
        self.source.data = numpy.concatenate((startData, toAdd.data, endData),
                                             0)

    def _transform_implementation(self, function, included):
        for i, p in enumerate(self):
            if included is not None and i not in included:
                continue
            currRet = function(p)
            # currRet might return an ArgumentException with a message which needs to be
            # formatted with the axis and current index before being raised
            if isinstance(currRet, ArgumentException):
                currRet.value = currRet.value.format('point', i)
                raise currRet
            if len(currRet) != len(self.source.features):
                msg = "function must return an iterable with as many elements "
                msg += "as features in this object"
                raise ArgumentException(msg)
            try:
                currRet = numpy.array(currRet, dtype=numpy.float)
            except ValueError:
                currRet = numpy.array(currRet, dtype=numpy.object_)
                # need self.data to be object dtype if inserting object dtype
                if numpy.issubdtype(self.source.data.dtype, numpy.number):
                    self.source.data = self.source.data.astype(numpy.object_)
            reshape = (1, len(self.source.features))
            self.source.data[i, :] = numpy.array(currRet).reshape(reshape)
