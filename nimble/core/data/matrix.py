
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
Class extending Base, using a numpy dense matrix to store data.
"""

import itertools

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble._utility import allowedNumpyDType
from nimble._utility import inheritDocstringsFactory
from nimble._utility import isAllowedSingleElement
from nimble._utility import numpy2DArray
from nimble._utility import scipy, pd
from .base import Base
from .views import BaseView
from .matrixAxis import MatrixPoints, MatrixPointsView
from .matrixAxis import MatrixFeatures, MatrixFeaturesView
from ._dataHelpers import allDataIdentical
from ._dataHelpers import createDataNoValidation
from ._dataHelpers import csvCommaFormat
from ._dataHelpers import denseCountUnique
from ._dataHelpers import NimbleElementIterator
from ._dataHelpers import convertToNumpyOrder, modifyNumpyArrayValue
from ._dataHelpers import isValid2DObject

@inheritDocstringsFactory(Base)
class Matrix(Base):
    """
    Class providing implementations of data manipulation operations on
    data stored in a numpy dense matrix.

    Parameters
    ----------
    data : object
        Must be a two-dimensional numpy array.
    reuseData : bool
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """

    def __init__(self, data, reuseData=False, **kwds):
        if not isValid2DObject(data):
            msg = "the input data can only be a two-dimensional numpy array "
            msg += "or python list."
            raise InvalidArgumentType(msg)

        if isinstance(data, (np.matrix, list)):
            data = numpy2DArray(data)

        if reuseData:
            self._data = data
        else:
            self._data = data.copy()

        shape = kwds.get('shape', None)
        if shape is None:
            kwds['shape'] = self._data.shape
        super().__init__(**kwds)

    def _getPoints(self, names):
        return MatrixPoints(self, names)

    def _getFeatures(self, names):
        return MatrixFeatures(self, names)

    def _transform_implementation(self, toTransform, points, features):
        ids = itertools.product(range(len(self.points)),
                                range(len(self.features)))
        for i, j in ids:
            currVal = self._data[i, j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            self._data = modifyNumpyArrayValue(self._data, (i, j), currRet)

    # pylint: disable=unused-argument
    def _calculate_implementation(self, function, points, features,
                                  preserveZeros):
        return self._calculate_genericVectorized(function, points, features)

    def _countUnique_implementation(self, points, features):
        return denseCountUnique(self, points, features)

    def _transpose_implementation(self):
        """
        Function to transpose the data, ie invert the feature and point
        indices of the data.

        This is not an in place operation, a new list of lists is
        constructed.
        """
        self._data = self._data.transpose()

    def _getTypeString_implementation(self):
        return 'Matrix'

    def _isIdentical_implementation(self, other):
        if not isinstance(other, Matrix):
            return False
        return allDataIdentical(self._data, other._data)


    def _saveCSV_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        """
        Function to write the data in this object to a CSV file at the
        designated path.
        """
        with open(outPath, 'w', encoding='utf-8') as outFile:
            if includeFeatureNames:
                self._writeFeatureNamesToCSV(outFile, includePointNames)
            if not np.issubdtype(self._data.dtype, np.number):
                vectorizeCommas = np.vectorize(csvCommaFormat, otypes=[object])
                viewData = vectorizeCommas(self._data).view()
            else:
                viewData = self._data.view()
            if includePointNames:
                pnames = list(map(csvCommaFormat, self.points.getNames()))
                pnames = numpy2DArray(pnames).transpose()
                viewData = np.concatenate((pnames, viewData), 1)

            np.savetxt(outFile, viewData, delimiter=',', fmt='%s')

    def _saveMTX_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        if not scipy.nimbleAccessible():
            msg = "scipy is not available"
            raise PackageException(msg)

        def makeNameString(count, namesGetter):
            nameString = "#"
            for i in range(count):
                nameString += namesGetter(i)
                if not i == count - 1:
                    nameString += ','
            return nameString

        header = ''
        if includePointNames:
            header = makeNameString(len(self.points), self.points.getName)
            header += '\n'
        else:
            header += '#\n'
        if includeFeatureNames:
            header += makeNameString(len(self.features), self.features.getName)
        else:
            header += '#\n'

        scipy.io.mmwrite(target=outPath, a=self._data.astype(np.float64),
                         comment=header)

    def _copy_implementation(self, to):
        if to in nimble.core.data.available:
            ptNames = self.points._getNamesNoGeneration()
            ftNames = self.features._getNamesNoGeneration()
            # reuseData=False since using original data
            return createDataNoValidation(to, self._data, ptNames, ftNames)
        if to == 'pythonlist':
            return self._data.tolist()
        needsReshape = len(self._dims) > 2
        if to == 'numpyarray':
            if needsReshape:
                return self._data.reshape(self._dims)
            return self._data.copy()
        if needsReshape:
            data = np.empty(self._dims[:2], dtype=np.object_)
            for i in range(self.shape[0]):
                data[i] = self.points[i].copy('pythonlist')
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
                ret = data.astype(np.float64)
            except ValueError as e:
                msg = f'Must create scipy {to[-3:]} matrix from numeric data'
                raise ValueError(msg) from e
            if to == 'scipycsc':
                return scipy.sparse.csc_matrix(ret)
            if to == 'scipycsr':
                return scipy.sparse.csr_matrix(ret)
        # pandasdataframe
        if not pd.nimbleAccessible():
            msg = "pandas is not available"
            raise PackageException(msg)
        pnames = self.points._getNamesNoGeneration()
        fnames = self.features._getNamesNoGeneration()
        return pd.DataFrame(data.copy(), index=pnames, columns=fnames)

    def _replaceRectangle_implementation(self, replaceWith, pointStart,
                                         featureStart, pointEnd, featureEnd):
        if not isinstance(replaceWith, Base):
            values = replaceWith * np.ones((pointEnd - pointStart + 1,
                                            featureEnd - featureStart + 1))
        else:
            values = replaceWith._data

        # numpy is exclusive
        pointEnd += 1
        featureEnd += 1
        self._data[pointStart:pointEnd, featureStart:featureEnd] = values

    def _flatten_implementation(self, order):
        numElements = len(self.points) * len(self.features)
        order = convertToNumpyOrder(order)
        self._data = self._data.reshape((1, numElements), order=order)

    def _unflatten_implementation(self, reshape, order):
        order = convertToNumpyOrder(order)
        self._data = self._data.reshape(reshape, order=order)

    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):
        self._data = np.array(self._data, dtype=np.object_)
        otherArr = np.array(other._data, dtype=np.object_)
        if onFeature is not None:
            if feature in ["intersection", "left"]:
                onFeatureIdx = self.features.getIndex(onFeature)
                onIdxLoc = matchingFtIdx[0].index(onFeatureIdx)
                onIdxL = onIdxLoc
                onIdxR = onIdxLoc
                right = otherArr[:, matchingFtIdx[1]]
                # matching indices in right were sorted when slicing above
                matchingFtIdx[1] = list(range(right.shape[1]))
                if feature == "intersection":
                    self._data = self._data[:, matchingFtIdx[0]]
                    # matching indices in left were sorted when slicing above
                    matchingFtIdx[0] = list(range(self._data.shape[1]))
            else:
                onIdxL = self.features.getIndex(onFeature)
                onIdxR = other.features.getIndex(onFeature)
                right = otherArr
        else:
            # using pointNames, prepend pointNames to left and right arrays
            onIdxL = 0
            onIdxR = 0
            if not self.points._anyDefaultNames():
                ptsL = np.array(self.points.getNames(), dtype=np.object_)
                ptsL = ptsL.reshape(-1, 1)
            elif self.points._namesCreated():
                # differentiate default names between objects
                namesL = [i if n is None else n for i, n
                          in enumerate(self.points.getNames())]
                ptsL = np.array(namesL, dtype=np.object_)
                ptsL = ptsL.reshape(-1, 1)
            else:
                ptsL = np.array(range(len(self.points)), dtype=np.object_)
                ptsL = ptsL.reshape(-1, 1)
            if not other.points._anyDefaultNames():
                ptsR = np.array(other.points.getNames(), dtype=np.object_)
                ptsR = ptsR.reshape(-1, 1)
            elif other.points._namesCreated():
                maxL = len(self.points)
                # differentiate default names between objects
                namesR = [i + maxL if n is None else n for i, n in
                          enumerate(other.points.getNames())]
                ptsR = np.array(namesR, dtype=np.object_)
                ptsR = ptsR.reshape(-1, 1)
            else:
                ptsR = np.array(range(len(other.points)), dtype=np.object_)
                ptsR = ptsR.reshape(-1, 1)
            if feature == "intersection":
                concatL = (ptsL, self._data[:, matchingFtIdx[0]])
                self._data = np.concatenate(concatL, axis=1)
                concatR = (ptsR, otherArr[:, matchingFtIdx[1]])
                right = np.concatenate(concatR, axis=1)
                # matching indices were sorted when slicing above
                # this also accounts for prepended column
                matchingFtIdx[0] = list(range(self._data.shape[1]))
                matchingFtIdx[1] = matchingFtIdx[0]
            elif feature == "left":
                self._data = np.concatenate((ptsL, self._data), axis=1)
                concatR = (ptsR, otherArr[:, matchingFtIdx[1]])
                right = np.concatenate(concatR, axis=1)
                # account for new column in matchingFtIdx
                matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
                matchingFtIdx[0].insert(0, 0)
                # matching indices were sorted when slicing above
                # this also accounts for prepended column
                matchingFtIdx[1] = list(range(right.shape[1]))
            else:
                self._data = np.concatenate((ptsL, self._data), axis=1)
                right = np.concatenate((ptsR, otherArr), axis=1)
                # account for new column in matchingFtIdx
                matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
                matchingFtIdx[0].insert(0, 0)
                matchingFtIdx[1] = list(map(lambda x: x + 1, matchingFtIdx[1]))
                matchingFtIdx[1].insert(0, 0)
        left = self._data

        matched = []
        merged = []
        unmatchedPtCountR = right.shape[1] - len(matchingFtIdx[1])
        matchMapper = {}
        for pt in left:
            matches = right[right[:, onIdxR] == pt[onIdxL]]
            if len(matches) > 0:
                matchMapper[pt[onIdxL]] = matches
        for ptL in left:
            target = ptL[onIdxL]
            if target in matchMapper:
                matchesR = matchMapper[target]
                for ptR in matchesR:
                    # check for conflicts between matching features
                    matches = ptL[matchingFtIdx[0]] == ptR[matchingFtIdx[1]]
                    nansL = np.array([x != x for x
                                         in ptL[matchingFtIdx[0]]])
                    nansR = np.array([x != x for x
                                         in ptR[matchingFtIdx[1]]])
                    acceptableValues = matches + nansL + nansR
                    if not all(acceptableValues):
                        msg = "The objects contain different values for the "
                        msg += "same feature"
                        raise InvalidArgumentValue(msg)
                    if nansL.any():
                        # fill any nan values in left with the corresponding
                        # right value
                        for i, value in enumerate(ptL[matchingFtIdx[0]]):
                            if value != value:
                                fill = ptR[matchingFtIdx[1]][i]
                                ptL[matchingFtIdx[0]][i] = fill
                    ptR = np.delete(ptR, matchingFtIdx[1])
                    pt = np.concatenate((ptL, ptR)).flatten()
                    merged.append(pt)
                matched.append(target)
            elif point in ['union', 'left']:
                ptL = ptL.reshape(1, -1)
                ptR = np.ones((1, unmatchedPtCountR)) * np.nan
                pt = np.append(ptL, ptR)
                merged.append(pt)

        if point == 'union':
            notMatchingR = [i for i in range(right.shape[1])
                            if i not in matchingFtIdx[1]]
            for row in right:
                target = row[onIdxR]
                if target not in matched:
                    ones = np.ones((left.shape[1] + unmatchedPtCountR,),
                                   dtype=np.object_)
                    pt = ones * np.nan
                    pt[matchingFtIdx[0]] = row[matchingFtIdx[1]]
                    pt[left.shape[1]:] = row[notMatchingR]
                    merged.append(pt)

        self._dims = [len(merged), left.shape[1] + unmatchedPtCountR]
        if len(merged) == 0 and onFeature is None:
            merged = np.empty((0, left.shape[1] + unmatchedPtCountR - 1))
            self._dims[1] -= 1
        elif len(merged) == 0:
            merged = np.empty((0, left.shape[1] + unmatchedPtCountR))
        elif onFeature is None:
            # remove point names feature
            merged = [row[1:] for row in merged]
            self._dims[1] -= 1

        self._data = numpy2DArray(merged, dtype=np.object_)

    def _replaceFeatureWithBinaryFeatures_implementation(self, uniqueIdx):
        toFill = np.zeros((len(self.points), len(uniqueIdx)))
        for ptIdx, val in enumerate(self._data):
            ftIdx = uniqueIdx[val.item()]
            toFill[ptIdx, ftIdx] = 1
        return Matrix(toFill)

    def _getitem_implementation(self, x, y):
        return self._data[x, y]
    
    def _setitem_implementation(self, x, y, value):
        self._data[x, y] = value

    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd, dropDimension):
        kwds = {}
        kwds['data'] = self._data[pointStart:pointEnd, featureStart:featureEnd]
        kwds['source'] = self
        if len(self._dims) > 2:
            if dropDimension:
                shape = self._dims[1:]
                source = self._createNestedObject(pointStart)
                kwds['source'] = source
                kwds['data'] = source._data
                pointStart, pointEnd = 0, source.shape[0]
                featureStart, featureEnd = 0, source.shape[1]
            else:
                shape = self._dims.copy()
                shape[0] = pointEnd - pointStart
            kwds['shape'] = shape
        kwds['pointStart'] = pointStart
        kwds['pointEnd'] = pointEnd
        kwds['featureStart'] = featureStart
        kwds['featureEnd'] = featureEnd

        return MatrixView(**kwds)

    def _createNestedObject(self, pointIndex):
        """
        Create an object of one less dimension
        """
        reshape = (self._dims[1], int(np.prod(self._dims[2:])))
        data = self._data[pointIndex].reshape(reshape)
        return Matrix(data, shape=self._dims[1:], reuseData=True)

    def _checkInvariants_implementation(self, level):
        mShape = np.shape(self._data)
        assert mShape[0] == self.shape[0]
        assert mShape[1] == self.shape[1]
        assert len(mShape) == 2
        assert allowedNumpyDType(self._data.dtype)

        if level > 1 and self._data.dtype == object:
            allowed = np.vectorize(isAllowedSingleElement, otypes=[bool])
            assert allowed(self._data).all()

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0
        contained in this object. False otherwise
        """
        return 0 in self._data


    def _binaryOperations_implementation(self, opName, other):
        """
        Attempt to perform operation with data as is, preserving sparse
        representations if possible. Otherwise, uses the generic
        implementation.
        """
        if (isinstance(other, nimble.core.data.Sparse)
                and opName.startswith('__r')):
            # rhs may return array of sparse matrices so use default
            return self._defaultBinaryOperations_implementation(opName, other)
        try:
            ret = getattr(self._data, opName)(other._data)
            return Matrix(ret)
        except (AttributeError, InvalidArgumentType, ValueError):
            return self._defaultBinaryOperations_implementation(opName, other)

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
        if isinstance(other, Matrix):
            return Matrix(np.matmul(self._data, other._data))
        if isinstance(other, nimble.core.data.Sparse):
            # '*' is matrix multiplication in scipy
            return Matrix(self._data * other._getSparseData())
        return Matrix(np.matmul(self._data, other.copy(to="numpyarray")))

    def _convertToNumericTypes_implementation(self, usableTypes):
        if self._data.dtype not in usableTypes:
            self._data = self._data.astype(float)

    def _iterateElements_implementation(self, order, only):
        return NimbleElementIterator(self._data, order, only)

    def _isBooleanData(self):
        return self._data.dtype in [bool, np.bool_]

class MatrixView(BaseView, Matrix):
    """
    Read only access to a Matrix object.
    """

    def _getPoints(self, names):
        return MatrixPointsView(self, names)

    def _getFeatures(self, names):
        return MatrixFeaturesView(self, names)
