"""
Method implementations and helpers acting specifically on features in a
Matrix object.
"""

import numpy

from nimble.exceptions import InvalidArgumentValue
from nimble.utility import numpy2DArray
from .axis_view import AxisView
from .matrixAxis import MatrixAxis
from .features import Features
from .features_view import FeaturesView
from .dataHelpers import allDataIdentical

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

    # def _flattenToOne_implementation(self):
    #     numElements = len(self._base.points) * len(self._base.features)
    #     self._base.data = self._base.data.reshape((numElements, 1),
    #                                                 order='F')
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     numFeatures = divideInto
    #     numPoints = len(self._base.points) // numFeatures
    #     self._base.data = self._base.data.reshape((numPoints,
    #                                                    numFeatures),
    #                                                 order='F')

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
