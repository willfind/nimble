"""
Method implementations and helpers acting specifically on features in a
DataFrame object.
"""
from __future__ import absolute_import
from __future__ import division

import UML
from UML.exceptions import InvalidArgumentValue
from .axis import Axis
from .axis_view import AxisView
from .dataframeAxis import DataFrameAxis
from .features import Features

pd = UML.importModule('pandas')
if pd:
    import pandas as pd

class DataFrameFeatures(DataFrameAxis, Axis, Features):
    """
    DataFrame method implementations performed on the feature axis.

    Parameters
    ----------
    source : UML data object
        The object containing features data.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, source, **kwds):
        self._source = source
        self._axis = 'feature'
        kwds['axis'] = self._axis
        kwds['source'] = self._source
        super(DataFrameFeatures, self).__init__(**kwds)

    ##############################
    # Structural implementations #
    ##############################

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        startData = self._source.data.iloc[:, :insertBefore]
        endData = self._source.data.iloc[:, insertBefore:]
        self._source.data = pd.concat((startData, toAdd.data, endData), axis=1)
        self._source._updateName(axis='feature')

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)
            # currRet might return an InvalidArgumentValue with a message which
            # needs to be formatted with the axis and current index before
            # being raised
            if isinstance(currRet, InvalidArgumentValue):
                currRet.value = currRet.value.format('feature', j)
                raise currRet
            if len(currRet) != len(self._source.points):
                msg = "function must return an iterable with as many elements "
                msg += "as points in this object"
                raise InvalidArgumentValue(msg)

            self._source.data.iloc[:, j] = currRet

    # def _flattenToOne_implementation(self):
    #     numElements = len(self._source.points) * len(self._source.features)
    #     self._source.data = pd.DataFrame(
    #         self._source.data.values.reshape((numElements, 1), order='F'))
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     numFeatures = divideInto
    #     numPoints = len(self._source.points) // numFeatures
    #     self._source.data = pd.DataFrame(
    #         self._source.data.values.reshape((numPoints, numFeatures),
    #                                         order='F'))

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

class DataFrameFeaturesView(AxisView, DataFrameFeatures, DataFrameAxis, Axis,
                            Features):
    """
    Limit functionality of DataFrameFeatures to read-only
    """
    def __init__(self, source, **kwds):
        kwds['source'] = source
        kwds['axis'] = 'feature'
        super(DataFrameFeaturesView, self).__init__(**kwds)

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
            value = self._source.data.iloc[self._pIndex, self._fIndex]

            self._pIndex += 1
            if self._pIndex >= self._pStop:
                self._pIndex = 0
                self._fIndex += 1

            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
