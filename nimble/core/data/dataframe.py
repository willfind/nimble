
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Class extending Base, using a pandas DataFrame to store data.
"""

import itertools
import numbers

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble._utility import allowedNumpyDType
from nimble._utility import inheritDocstringsFactory
from nimble._utility import isAllowedSingleElement
from nimble._utility import pandasDataFrameToList
from nimble._utility import scipy, pd
from .base import Base
from .views import BaseView
from .dataframeAxis import DataFramePoints, DataFramePointsView
from .dataframeAxis import DataFrameFeatures, DataFrameFeaturesView
from ._dataHelpers import allDataIdentical
from ._dataHelpers import createDataNoValidation
from ._dataHelpers import denseCountUnique
from ._dataHelpers import NimbleElementIterator
from ._dataHelpers import convertToNumpyOrder, isValid2DObject
from ._dataHelpers import getFeatureDtypes

@inheritDocstringsFactory(Base)
class DataFrame(Base):
    """
    Class providing implementations of data manipulation operations on
    data stored in a pandas DataFrame.

    Parameters
    ----------
    data : object
        pandas DataFrame or two-dimensional numpy array.
    reuseData : bool
        Only used when data is a pandas DataFrame.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """

    def __init__(self, data, reuseData=False, **kwds):
        if not pd.nimbleAccessible():
            msg = 'To use class DataFrame, pandas must be installed.'
            raise PackageException(msg)

        if not (isinstance(data, pd.DataFrame) or isValid2DObject(data)):
            msg = "the input data can only be a pandas DataFrame or a two-"
            msg += "dimensional list or numpy array."
            raise InvalidArgumentType(msg)

        if isinstance(data, pd.DataFrame):
            if reuseData:
                self._data = data
            else:
                self._data = data.copy()
        else:
            self._data = pd.DataFrame(data, copy=True)

        shape = kwds.get('shape', None)
        if shape is None:
            kwds['shape'] = self._data.shape
        super().__init__(**kwds)

    def _getPoints(self, names):
        return DataFramePoints(self, names)

    def _getFeatures(self, names):
        return DataFrameFeatures(self, names)

    def _transform_implementation(self, toTransform, points, features):
        ids = itertools.product(range(len(self.points)),
                                range(len(self.features)))
        for i, j in ids:
            currVal = self._data.iat[i, j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)
            # convert datetime columns so pandas can't convert to datetimes
            if (self._data.dtypes.iloc[j].type == np.datetime64
                    and isinstance(currRet, (numbers.Number, str))):
                self._data[j] = self._data[j].astype(object)
            self._data.iloc[i, j] = currRet

    # pylint: disable=unused-argument
    def _calculate_implementation(self, function, points, features,
                                  preserveZeros):
        if points is not None or features is not None:
            if points is None:
                points = slice(None)
            if features is None:
                features = slice(None)
            toCalculate = self._data.iloc[points, features]
        else:
            toCalculate = self._data

        ret = toCalculate.applymap(function)

        ret.index = pd.RangeIndex(ret.shape[0])
        ret.columns = pd.RangeIndex(ret.shape[1])

        return ret

    def _countUnique_implementation(self, points, features):
        return denseCountUnique(self, points, features)

    def _transpose_implementation(self):
        """
        Function to transpose the data, ie invert the feature and point
        indices of the data.

        This is not an in place operation, a new pandas DataFrame is
        constructed.
        """
        self._data = self._data.T

    def _getTypeString_implementation(self):
        return 'DataFrame'

    def _isIdentical_implementation(self, other):
        if not isinstance(other, DataFrame):
            return False

        return allDataIdentical(self._asNumpyArray(), other._asNumpyArray())

    def _saveCSV_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        """
        Function to write the data in this object to a CSV file at the
        designated path.
        """
        with open(outPath, 'w', encoding='utf-8') as outFile:
            if includeFeatureNames:
                self._data.columns = self.features.getNames()
                if includePointNames:
                    outFile.write('pointNames')

            if includePointNames:
                self._data.index = self.points.getNames()

        self._data.to_csv(outPath, mode='a', index=includePointNames,
                          header=includeFeatureNames)

        if includePointNames:
            self._data.index = pd.RangeIndex(len(self._data.index))
        if includeFeatureNames:
            self._data.columns = pd.RangeIndex(len(self._data.columns))

    def _saveMTX_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        """
        Function to write the data in this object to a matrix market
        file at the designated path.
        """
        if not scipy.nimbleAccessible():
            msg = "scipy is not available"
            raise PackageException(msg)

        comment = '#'
        if includePointNames:
            comment += ','.join(self.points.getNames())
        if includeFeatureNames:
            comment += '\n#' + ','.join(self.features.getNames())
        scipy.io.mmwrite(outPath, self._data.astype(np.float64),
                         comment=comment)

    def _copy_implementation(self, to):
        """
        Copy the current DataFrame object to one in a specified format.

        to : string
            Sparse, List, Matrix, pythonlist, numpyarray, numpymatrix,
            scipycsc, scipycsr.
        """
        if to in nimble.core.data.available:
            ptNames = self.points._getNamesNoGeneration()
            ftNames = self.features._getNamesNoGeneration()
            if to == 'DataFrame':
                data = self._data.copy()
            elif self._data.empty:
                data = self._asNumpyArray()
            else:
                # convert to list because it preserves data types
                data = pandasDataFrameToList(self._data)
            # reuseData=True since we already made copies here
            return createDataNoValidation(to, data, ptNames, ftNames,
                                          reuseData=True)

        needsReshape = len(self._dims) > 2
        if to in ['pythonlist', 'numpyarray']:
            # convert pandas Timestamp type if necessary
            timestamp = [d.type == np.datetime64 for d in self._data.dtypes]
            arr = self._asNumpyArray()
            if any(timestamp):
                attr = 'to_pydatetime' if to == 'pythonlist' else 'to_numpy'
                convTimestamp = np.vectorize(lambda v: getattr(v, attr)())
                arr[:, timestamp] = convTimestamp(arr[:, timestamp])

            if needsReshape:
                arr = arr.reshape(self._dims)
            if to == 'pythonlist':
                return arr.tolist()
            if needsReshape:
                return arr
            return arr.copy()

        if needsReshape:
            data = np.empty(self._dims[:2], dtype=np.object_)
            for i in range(self.shape[0]):
                data[i] = self.points[i].copy('pythonlist')
        elif to == 'pandasdataframe':
            data = self._data.copy()
        else:
            data = self._data
        if to == 'numpymatrix':
            return np.matrix(data)
        if 'scipy' in to:
            if not scipy.nimbleAccessible():
                msg = "scipy is not available"
                raise PackageException(msg)
            if to == 'scipycoo':
                return scipy.sparse.coo_matrix(data)
            try:
                ret = self._asNumpyArray(numericRequired=True)
            except ValueError as e:
                msg = f'Must create scipy {to[-3:]} matrix from numeric data'
                raise ValueError(msg) from e
            if to == 'scipycsc':
                return scipy.sparse.csc_matrix(ret)
            if to == 'scipycsr':
                return scipy.sparse.csr_matrix(ret)
        # pandasdataframe
        pnames = self.points._getNamesNoGeneration()
        fnames = self.features._getNamesNoGeneration()
        dataframe = pd.DataFrame(data)
        if pnames is not None:
            dataframe.index = pnames
        if fnames is not None:
            dataframe.columns = fnames
        return dataframe

    def _replaceRectangle_implementation(self, replaceWith, pointStart,
                                         featureStart, pointEnd, featureEnd):
        dtypes = self._data.dtypes
        ftRange = range(featureStart, featureEnd + 1)
        if isinstance(replaceWith, DataFrame):
            replaceDtypes = replaceWith._data.dtypes
        else:
            replaceDtypes = (np.dtype(type(replaceWith)),) * len(ftRange)
        for i, rdt in zip(ftRange, replaceDtypes):
            dtypes.iloc[i] = max(dtypes.iloc[i], rdt)
        if not isinstance(replaceWith, Base):
            values = replaceWith * np.ones((pointEnd - pointStart + 1,
                                            featureEnd - featureStart + 1))
        else:
            #convert values to be array or matrix, instead of pandas DataFrame
            values = replaceWith._asNumpyArray()

        # pandas is exclusive
        pointEnd += 1
        featureEnd += 1
        self._data.iloc[pointStart:pointEnd, featureStart:featureEnd] = values

        self._setDtypes(dtypes)

    def _flatten_implementation(self, order):
        numElements = len(self.points) * len(self.features)
        dtypes = self._data.dtypes
        if order == 'point':
            newDtypes = tuple(np.tile(dtypes, len(self.points)))
        else:
            newDtypes = tuple(np.repeat(dtypes, len(self.points)))
        order = convertToNumpyOrder(order)
        array = self._asNumpyArray()
        values = array.reshape((1, numElements), order=order)

        self._data = pd.DataFrame(values)
        self._setDtypes(newDtypes)

    def _unflatten_implementation(self, reshape, order):
        dtypes = tuple(self._data.dtypes)
        if len(dtypes) == 1 and reshape[1] > 1: # feature shaped
            dtypes = (dtypes[0],) * reshape[0] * reshape[1]

        if order == 'point':
            jumps = range(reshape[1])
            newDtypes = tuple(max(dtypes[i::reshape[1]]) for i in jumps)
        else:
            jumps = range(0, np.prod(reshape), reshape[0])
            newDtypes = tuple(max(dtypes[i:i+reshape[1]]) for i in jumps)
        order = convertToNumpyOrder(order)
        array = self._asNumpyArray()
        values = array.reshape(reshape, order=order)

        self._data = pd.DataFrame(values)
        self._setDtypes(newDtypes)

    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):
        if point == 'union':
            point = 'outer'
        elif point == 'intersection':
            point = 'inner'
        if self.features._namesCreated():
            self._data.columns = self.features.getNames()
        tmpDfR = other._data.copy()
        if other.features._namesCreated():
            tmpDfR.columns = other.features.getNames()

        if feature == 'intersection':
            self._data = self._data.iloc[:, matchingFtIdx[0]]
            tmpDfR = tmpDfR.iloc[:, matchingFtIdx[1]]
            matchingFtIdx[0] = list(range(self._data.shape[1]))
            matchingFtIdx[1] = list(range(tmpDfR.shape[1]))
        elif feature == "left":
            tmpDfR = tmpDfR.iloc[:, matchingFtIdx[1]]
            matchingFtIdx[1] = list(range(tmpDfR.shape[1]))

        numColsL = len(self._data.columns)
        if onFeature is None:
            if (self.points._namesCreated()
                    and other.points._namesCreated()):
                # differentiate default names between objects using integers
                self._data.index = [i if n is None else n for i, n
                                    in enumerate(self.points.getNames())]
                maxL = len(self.points)
                tmpDfR.index = [i + maxL if n is None else n for i, n
                                in enumerate(other.points.getNames())]
            else:
                # left guaranteed equiv to pd.RangeIndex(self._data.shape[0])
                idxRange = range(self.shape[0], self.shape[0] + other.shape[0])
                tmpDfR.index = list(idxRange)

            self._data = self._data.merge(tmpDfR, how=point, left_index=True,
                                          right_index=True)
        else:
            onIdxL = self._data.columns.get_loc(onFeature)
            onIdxR = tmpDfR.columns.get_loc(onFeature) + numColsL
            self._data = self._data.merge(tmpDfR, how=point, on=onFeature)

        # return labels to default after we've executed the merge
        self._data.index = pd.RangeIndex(self._data.shape[0])
        self._data.columns = pd.RangeIndex(self._data.shape[1])

        toDrop = []
        for left, right in zip(matchingFtIdx[0], matchingFtIdx[1]):
            if onFeature is not None and left == onIdxL:
                # onFeature column has already been merged
                continue
            right += numColsL
            if onFeature is not None and right > onIdxR:
                # one less to account for merged onFeature
                right -= 1

            matches = self._data.iloc[:, left] == self._data.iloc[:, right]
            nansL = np.array([x != x for x in self._data.iloc[:, left]])
            nansR = np.array([x != x for x in self._data.iloc[:, right]])
            acceptableValues = matches | nansL | nansR
            if not all(acceptableValues):
                msg = "The objects contain different values for the same "
                msg += "feature"
                raise InvalidArgumentValue(msg)
            if nansL.any():
                leftNansLocInRight = self._data.iloc[:, right][nansL]
                self._data.iloc[:, left][nansL] = leftNansLocInRight
            toDrop.append(right)

        if toDrop:
            self._data.drop(toDrop, axis=1, inplace=True)
            self._data.columns = pd.RangeIndex(self._data.shape[1])

        numFeatures = (numColsL + len(tmpDfR.columns) - len(matchingFtIdx[1]))
        numPoints = len(self._data.index)

        self._dims = [numPoints, numFeatures]

    def _replaceFeatureWithBinaryFeatures_implementation(self, uniqueIdx):
        toFill = np.zeros((len(self.points), len(uniqueIdx)))
        for ptIdx, val in self._data.iterrows():
            ftIdx = uniqueIdx[val[0]]
            toFill[ptIdx, ftIdx] = 1
        return DataFrame(pd.DataFrame(toFill))

    def _getitem_implementation(self, x, y):
        # .iat should be used for accessing scalar values in pandas
        return self._data.iat[x, y]

    def _setitem_implementation(self, x, y, value):
       # need to see if there is an issue with scalar vs slice
      self._data.iat[x, y] = value



    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd, dropDimension):
        kwds = {}
        kwds['data'] = self._data.iloc[pointStart:pointEnd,
                                       featureStart:featureEnd]
        kwds['source'] = self
        pRange = pointEnd - pointStart
        fRange = featureEnd - featureStart
        if len(self._dims) > 2:
            if dropDimension:
                shape = self._dims[1:]
                source = self._createNestedObject(pointStart)
                kwds['source'] = source
                kwds['data'] = source._data
                pointStart, pointEnd = 0, source.shape[0]
                featureStart, featureEnd = 0, source.shape[1]
                pRange = source.shape[0]
                fRange = source.shape[1]
            else:
                shape = self._dims.copy()
                shape[0] = pRange
            kwds['shape'] = shape
        kwds['pointStart'] = pointStart
        kwds['pointEnd'] = pointEnd
        kwds['featureStart'] = featureStart
        kwds['featureEnd'] = featureEnd

        ret = DataFrameView(**kwds)

        # Reassign labels as to match the positions in the view object,
        # not the positions in the source object.
        ret._data.index = pd.RangeIndex(pRange)
        ret._data.columns = pd.RangeIndex(fRange)

        return ret

    def _createNestedObject(self, pointIndex):
        """
        Create an object of one less dimension.
        """
        reshape = (self._dims[1], int(np.prod(self._dims[2:])))
        data = self._asNumpyArray()[pointIndex].reshape(reshape)
        return DataFrame(data, shape=self._dims[1:], reuseData=True)

    def _checkInvariants_implementation(self, level):
        dfShape = self._data.shape
        assert dfShape[0] == self.shape[0]
        assert dfShape[1] == self.shape[1]
        assert all(self._data.index == pd.RangeIndex(self.shape[0]))
        assert all(self._data.columns == pd.RangeIndex(self.shape[1]))

        assert all(allowedNumpyDType(d) for d in self._data.dtypes)
        if level > 1 and any(d == object for d in self._data.dtypes):
            for pt, ft in itertools.product(range(dfShape[0]),
                                            range(dfShape[1])):
                assert isAllowedSingleElement(self._data.at[pt,ft])

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0
        contained in this object. False otherwise.
        """
        return 0 in self._asNumpyArray()


    def _binaryOperations_implementation(self, opName, other):
        """
        Attempt to perform operation with data as is, preserving sparse
        representations if possible. Otherwise, uses the generic
        implementation.
        """
        initialDtypes = tuple(self._data.dtypes)
        if isinstance(other, Base):
            otherDtypes = getFeatureDtypes(other)
            if len(self.features) != len(other.features): # for stretch
                if len(initialDtypes) == 1:
                    initialDtypes = (initialDtypes[0],) * len(other.features)
                if len(otherDtypes) == 1:
                    otherDtypes = (otherDtypes[0],) * len(self.features)
        else:
            dtype = np.dtype(type(other))
            if not np.can_cast(dtype, float):
                dtype = np.object_
            otherDtypes = tuple(dtype for _ in initialDtypes)

        dtypes = []
        # truediv will return floats given two ints
        # pow for negative ints will be floats but the data has already
        # been converted to floats in that case
        alwaysFloat = 'truediv' in opName
        for dtype1, dtype2 in zip(initialDtypes, otherDtypes):
            useType = max(dtype1, dtype2)
            if alwaysFloat and np.can_cast(useType, float):
                dtypes.append(np.dtype(float))
            else:
                dtypes.append(useType)

        # pow operations for DataFrame do not raise expected ZeroDivisionErrors
        # and rhs Sparse may return array of sparse matrices, so use default
        if not ((isinstance(other, DataFrame) and 'pow' in opName)
                or (isinstance(other, nimble.core.data.Sparse)
                    and (opName.startswith('__r')))):
            try:
                array = self._asNumpyArray(numericRequired=True)
                values = getattr(array, opName)(other._data)
                ret = DataFrame(values)
                ret._setDtypes(dtypes)
                return ret
            except (AttributeError, InvalidArgumentType, ValueError):
                pass

        ret = self._defaultBinaryOperations_implementation(opName, other)
        ret._setDtypes(dtypes)

        return ret

    def _matmul__implementation(self, other):
        """
        Matrix multiply this nimble Base object against the provided
        other nimble Base object. Both object must contain only numeric
        data. The featureCount of the calling object must equal the
        pointCount of the other object. The types of the two objects may
        be different, and the return is guaranteed to be the same type
        as at least one out of the two, to be automatically determined
        according to efficiency constraints.
        """
        # need the max dtype of this object since multiplying rows of this
        # object by features of other object
        rowDtype = max(self._data.dtypes)
        otherDtypes = getFeatureDtypes(other)
        dtypes = tuple(map(lambda dtype: max(dtype, rowDtype), otherDtypes))

        if isinstance(other, nimble.core.data.Sparse):
            # scipy performs mat mul with * operator
            array = self._asNumpyArray(numericRequired=True)
            values = array * other._getSparseData()
        else:
            values = np.matmul(self._asNumpyArray(numericRequired=True),
                               other.copy('numpyarray'))
        ret = DataFrame(values)
        ret._setDtypes(dtypes)

        return ret

    def _convertToNumericTypes_implementation(self, usableTypes):
        if not all(dtype in usableTypes for dtype in self._data.dtypes):
            self._data = self._data.astype(float)

    def _iterateElements_implementation(self, order, only):
        return NimbleElementIterator(self._asNumpyArray(), order, only)

    def _setDtypes(self, dtypes):
        """
        Set each feature to the dtype from a list of dtypes.
        """
        if tuple(self._data.dtypes) != tuple(dtypes):
            if len(self._data.dtypes) != len(tuple(dtypes)):
                msg = 'A dtype must be specified for each feature'
                raise InvalidArgumentValue(msg)
            for (i, col), dtype in zip(self._data.items(), dtypes):
                if col.dtype !=  dtype:
                    self._data[i] = col.astype(dtype)

    def _asNumpyArray(self, numericRequired=False):
        """
        Convert the dataframe to a numpy array with appropriate dtype.

        Object dtype is used whenever more than one dtype is present.
        If numericRequired is True, the data will be all be converted to
        floats if the dtypes are not already all the same numeric dtype.
        """
        dtypes = self._data.dtypes
        allowed = set((np.dtype(bool), np.dtype(int), np.dtype(float)))
        if not numericRequired:
            allowed.add(np.dtype(np.object_))

        if len(dtypes) > 0:
            if all(d in allowed and np.can_cast(d, float) for d in dtypes):
                return self._data.to_numpy()
        if numericRequired:
            return self._data.to_numpy(float)

        return self._data.astype(np.object_).to_numpy()


class DataFrameView(BaseView, DataFrame):
    """
    Read only access to a DataFrame object.
    """

    def _getPoints(self, names):
        return DataFramePointsView(self, names)

    def _getFeatures(self, names):
        return DataFrameFeaturesView(self, names)
