"""
Method implementations and helpers acting specifically on features in a
Matrix object.
"""
from __future__ import absolute_import

import numpy

from UML.exceptions import ArgumentException
from .axis import Axis
from .matrixAxis import MatrixAxis
from .features import Features

class MatrixFeatures(MatrixAxis, Axis, Features):
    """
    Matrix method implementations performed on the feature axis.

    Parameters
    ----------
    source : UML data object
        The object containing features data.
    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'feature'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(MatrixFeatures, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        startData = self.source.data[:, :insertBefore]
        endData = self.source.data[:, insertBefore:]
        self.source.data = numpy.concatenate((startData, toAdd.data, endData),
                                             1)

    def _transform_implementation(self, function, included):
        for j, f in enumerate(self):
            if included is not None and j not in included:
                continue
            currRet = function(f)
            # currRet might return an ArgumentException with a message which needs to be
            # formatted with the axis and current index before being raised
            if isinstance(currRet, ArgumentException):
                currRet.value = currRet.value.format('feature', j)
                raise currRet
            if len(currRet) != len(self.source.points):
                msg = "function must return an iterable with as many elements "
                msg += "as points in this object"
                raise ArgumentException(msg)
            try:
                currRet = numpy.array(currRet, dtype=numpy.float)
            except ValueError:
                currRet = numpy.array(currRet, dtype=numpy.object_)
                # need self.data to be object dtype if inserting object dtype
                if numpy.issubdtype(self.source.data.dtype, numpy.number):
                    self.source.data = self.source.data.astype(numpy.object_)
            reshape = (len(self.source.points), 1)
            self.source.data[:, j] = numpy.array(currRet).reshape(reshape)
