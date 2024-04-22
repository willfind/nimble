
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
Class extending Base, using a list of lists to store data.
"""

import copy
import itertools

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble._utility import inheritDocstringsFactory, numpy2DArray, is2DArray
from nimble._utility import isAllowedSingleElement, allowedNumpyDType
from nimble._utility import scipy, pd
from .base import Base
from .views import BaseView
from .listAxis import ListPoints, ListPointsView
from .listAxis import ListFeatures, ListFeaturesView
from ._dataHelpers import createDataNoValidation
from ._dataHelpers import csvCommaFormat
from ._dataHelpers import denseCountUnique
from ._dataHelpers import NimbleElementIterator

@inheritDocstringsFactory(Base)
class List(Base):
    """
    Class providing implementations of data manipulation operations on
    data stored in a list of lists implementation, where the outer list
    is a list of points of data, and each inner list is a list of values
    for each feature.

    Parameters
    ----------
    data : object
        A list, two-dimensional numpy array, or a ListPassThrough.
    reuseData : bool
        Only works when input data is a list.
    shape : tuple
        The number of points and features in the object in the format
        (points, features).
    checkAll : bool
        Perform a validity check for all elements.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """

    def __init__(self, data, featureNames=None, reuseData=False, shape=None,
                 checkAll=True, **kwds):
        if not (isinstance(data, (list, ListPassThrough)) or is2DArray(data)):
            msg = "the input data can only be a list, a two-dimensional numpy "
            msg += "array, or ListPassThrough."
            raise InvalidArgumentType(msg)

        if isinstance(data, list):
            #case1: data=[]. self._data will be [], shape will be (0, shape[1])
            # or (0, len(featureNames)) or (0, 0)
            if len(data) == 0:
                if shape and len(shape) == 2:
                    shape = (0, shape[1])
                elif shape is None:
                    shape = (0, len(featureNames) if featureNames else 0)
            elif isAllowedSingleElement(data[0]):
            #case2: data=['a', 'b', 'c'] or [1,2,3]. self._data will be
            # [[1,2,3]], shape will be (1, 3)
                if checkAll:#check all items
                    for i in data:
                        if not isAllowedSingleElement(i):
                            msg = 'invalid input data format.'
                            raise InvalidArgumentValue(msg)
                if shape is None:
                    shape = (1, len(data))
                data = [data]
            elif isinstance(data[0], (list, FeatureViewer)):
            #case3: data=[[1,2,3], ['a', 'b', 'c']] or [[]] or [[], []].
            # self._data will be data, shape will be (len(data), len(data[0]))
            #case4: data=[FeatureViewer]
                numFeatures = len(data[0])
                if checkAll:#check all items
                    for i in data:
                        if len(i) != numFeatures:
                            msg = 'invalid input data format.'
                            raise InvalidArgumentValue(msg)
                        for j in i:
                            if not isAllowedSingleElement(j):
                                msg = f'{j} is invalid input data format.'
                                raise InvalidArgumentValue(msg)
                if shape is None:
                    shape = (len(data), numFeatures)

            if not reuseData:
                #this is to convert a list x=[[1,2,3]]*2 to a
                # list y=[[1,2,3], [1,2,3]]
                # the difference is that x[0] is x[1], but y[0] is not y[1]
                # Both list and FeatureViewer have a copy method.
                data = [pt.copy() for pt in data]

        if is2DArray(data):
            #case5: data is a numpy array. shape is already in np array
            if shape is None:
                shape = data.shape
            data = data.tolist()

        if len(data) == 0:
            #case6: data is a ListPassThrough associated with empty list
            data = []

        self._numFeatures = int(np.prod(shape[1:]))
        self._data = data

        kwds['featureNames'] = featureNames
        kwds['shape'] = shape
        super().__init__(**kwds)

    def _getPoints(self, names):
        return ListPoints(self, names)

    def _getFeatures(self, names):
        return ListFeatures(self, names)

    def _transform_implementation(self, toTransform, points, features):
        ids = itertools.product(range(len(self.points)),
                                range(len(self.features)))
        for i, j in ids:
            currVal = self._data[i][j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            self._data[i][j] = currRet

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
        tempFeatures = len(self._data)
        transposed = []
        #load the new data with an empty point for each feature in the original
        for i in range(len(self.features)):
            transposed.append([])
        for point in self._data:
            for i, val in enumerate(point):
                transposed[i].append(val)

        self._data = transposed
        self._numFeatures = tempFeatures

    def _getTypeString_implementation(self):
        return 'List'

    def _isIdentical_implementation(self, other):
        if not isinstance(other, List):
            return False

        for index in range(len(self.points)):
            sPoint = self._data[index]
            oPoint = other._data[index]
            if sPoint != oPoint:
                for sVal, oVal in zip(sPoint, oPoint):

                    if sVal != oVal and (sVal == sVal or oVal == oVal):
                        return False
        return True

    def _saveCSV_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        """
        Function to write the data in this object to a CSV file at the
        designated path.
        """
        with open(outPath, 'w', encoding='utf-8') as outFile:
            if includeFeatureNames:
                self._writeFeatureNamesToCSV(outFile, includePointNames)

            for point in self.points:
                first = True
                if includePointNames:
                    currPname = csvCommaFormat(point.points.getName(0))
                    outFile.write(currPname)
                    first = False

                for value in point:
                    if not first:
                        outFile.write(',')
                    outFile.write(str(csvCommaFormat(value)))
                    first = False
                outFile.write('\n')

    def _saveMTX_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        """
        Function to write the data in this object to a matrix market
        file at the designated path.
        """
        with open(outPath, 'w', encoding='utf-8') as outFile:
            outFile.write("%%MatrixMarket matrix array real general\n")

            def writeNames(nameList):
                for i, n in enumerate(nameList):
                    if i == 0:
                        outFile.write('%#')
                    else:
                        outFile.write(',')
                    outFile.write(n)
                outFile.write('\n')

            if includePointNames:
                writeNames(self.points.getNames())
            else:
                outFile.write('%#\n')
            if includeFeatureNames:
                writeNames(self.features.getNames())
            else:
                outFile.write('%#\n')

            outFile.write(f"{len(self.points)} {len(self.features)}\n")

            for j in range(len(self.features)):
                for i in range(len(self.points)):
                    value = self._data[i][j]
                    outFile.write(str(value) + '\n')

    def _referenceFrom_implementation(self, other, kwargs):
        kwargs.setdefault('checkAll', False)
        super()._referenceFrom_implementation(other, kwargs)

    def _copy_implementation(self, to):
        isEmpty = False
        if len(self.points) == 0 or len(self.features) == 0:
            isEmpty = True
            emptyData = np.empty(shape=self.shape)

        if to == 'pythonlist':
            return [pt.copy() for pt in self._data]

        if to in nimble.core.data.available:
            ptNames = self.points._getNamesNoGeneration()
            ftNames = self.features._getNamesNoGeneration()
            if isEmpty:
                data = numpy2DArray(emptyData)
            elif to == 'List':
                data = [pt.copy() for pt in self._data]
            else:
                data = _convertList(numpy2DArray, self._data)
            # reuseData=True since we already made copies here
            return createDataNoValidation(to, data, ptNames, ftNames,
                                          reuseData=True)

        needsReshape = len(self._dims) > 2
        if to == 'numpyarray':
            if isEmpty:
                ret = emptyData
            else:
                ret = _convertList(numpy2DArray, self._data)
            if needsReshape:
                return ret.reshape(self._dims)
            return ret
        if needsReshape:
            data = np.empty(self._dims[:2], dtype=np.object_)
            for i in range(self.shape[0]):
                data[i] = self.points[i].copy('pythonlist')
            if isEmpty:
                emptyData = data
        else:
            data = _convertList(numpy2DArray, self._data)
        if to == 'numpymatrix':
            if isEmpty:
                return np.matrix(emptyData)
            return np.matrix(data)
        if 'scipy' in to:
            if not scipy.nimbleAccessible():
                msg = "scipy is not available"
                raise PackageException(msg)
            if to == 'scipycsc':
                if isEmpty:
                    return scipy.sparse.csc_matrix(emptyData)
                return scipy.sparse.csc_matrix(data)
            if to == 'scipycsr':
                if isEmpty:
                    return scipy.sparse.csr_matrix(emptyData)
                return scipy.sparse.csr_matrix(data)
            if to == 'scipycoo':
                if isEmpty:
                    return scipy.sparse.coo_matrix(emptyData)
                return scipy.sparse.coo_matrix(data)
        # pandasdataframe
        if not pd.nimbleAccessible():
            msg = "pandas is not available"
            raise PackageException(msg)
        if isEmpty:
            return pd.DataFrame(emptyData)
        pnames = self.points._getNamesNoGeneration()
        fnames = self.features._getNamesNoGeneration()
        return pd.DataFrame(data, index=pnames, columns=fnames)

    def _replaceRectangle_implementation(self, replaceWith, pointStart,
                                         featureStart, pointEnd, featureEnd):
        if not isinstance(replaceWith, Base):
            values = [replaceWith] * (featureEnd - featureStart + 1)
            for pIdx in range(pointStart, pointEnd + 1):
                self._data[pIdx][featureStart:featureEnd + 1] = values
        else:
            for pIdx in range(pointStart, pointEnd + 1):
                fill = replaceWith._data[pIdx - pointStart]
                self._data[pIdx][featureStart:featureEnd + 1] = fill


    def _flatten_implementation(self, order):
        if order == 'point':
            onto = self._data[0]
            for _ in range(1, len(self.points)):
                onto += self._data[1]
                del self._data[1]
        else:
            result = [[]]
            for i in range(len(self.features)):
                result[0].extend(p[i] for p in self._data)
            self._data = result
        self._numFeatures = self.shape[0] * self.shape[1]

    def _unflatten_implementation(self, reshape, order):
        result = []
        numPoints = reshape[0]
        numFeatures = np.prod(reshape[1:])
        data = self.copy('pythonlist', outputAs1D=True)
        if order == 'point':
            for i in range(numPoints):
                temp = data[(i*numFeatures):((i+1)*numFeatures)]
                result.append(temp)
        else:
            for i in range(numPoints):
                temp = data[i::numPoints]
                result.append(temp)

        self._data = result
        self._numFeatures = numFeatures

    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):
        if onFeature is not None:
            if feature in ["intersection", "left"]:
                onFeatureIdx = self.features.getIndex(onFeature)
                onIdxLoc = matchingFtIdx[0].index(onFeatureIdx)
                onIdxL = onIdxLoc
                onIdxR = onIdxLoc
                right = [[row[i] for i in matchingFtIdx[1]]
                         for row in other._data]
                # matching indices in right were sorted when slicing above
                if len(right) > 0:
                    matchingFtIdx[1] = list(range(len(right[0])))
                else:
                    matchingFtIdx[1] = []
                if feature == "intersection":
                    self._data = [[row[i] for i in matchingFtIdx[0]]
                                  for row in self._data]
                    # matching indices in left were sorted when slicing above
                    if len(self._data) > 0:
                        matchingFtIdx[0] = list(range(len(self._data[0])))
                    else:
                        matchingFtIdx[0] = []
            else:
                onIdxL = self.features.getIndex(onFeature)
                onIdxR = other.features.getIndex(onFeature)
                right = copy.copy(other._data)
        else:
            # using pointNames, prepend pointNames to left and right lists
            onIdxL = 0
            onIdxR = 0
            left = []
            right = []

            def ptNameGetter(obj, idx, add=0):
                if obj.points._namesCreated():
                    name = obj.points.getName(idx)
                    if name is not None:
                        return name
                return idx + add

            if feature == "intersection":
                for i, pt in enumerate(self._data):
                    ptL = [ptNameGetter(self, i)]
                    intersect = [val for idx, val in enumerate(pt)
                                 if idx in matchingFtIdx[0]]
                    self._data[i] = ptL + intersect
                for i, pt in enumerate(other._data):
                    maxL = len(self.points)
                    ptR = [ptNameGetter(other, i, maxL)]
                    ptR.extend([pt[i] for i in matchingFtIdx[1]])
                    right.append(ptR)
                # matching indices were sorted above
                # this also accounts for prepended column
                if len(self._data) > 0:
                    matchingFtIdx[0] = list(range(len(self._data[0])))
                else:
                    matchingFtIdx[0] = []
                matchingFtIdx[1] = matchingFtIdx[0]
            elif feature == "left":
                for i, pt in enumerate(self._data):
                    ptL = [ptNameGetter(self, i)]
                    self._data[i] = ptL + pt
                for i, pt in enumerate(other._data):
                    maxL = len(self.points)
                    ptR = [ptNameGetter(other, i, maxL)]
                    ptR.extend([pt[i] for i in matchingFtIdx[1]])
                    right.append(ptR)
                # account for new column in matchingFtIdx
                matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
                matchingFtIdx[0].insert(0, 0)
                # matching indices were sorted when slicing above
                # this also accounts for prepended column
                matchingFtIdx[1] = list(range(len(right[0])))
            else:
                for i, pt in enumerate(self._data):
                    ptL = [ptNameGetter(self, i)]
                    self._data[i] = ptL + pt
                for i, pt in enumerate(other._data):
                    maxL = len(self.points)
                    ptR = [ptNameGetter(other, i, maxL)]
                    ptR.extend(pt)
                    right.append(ptR)
                matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
                matchingFtIdx[0].insert(0, 0)
                matchingFtIdx[1] = list(map(lambda x: x + 1, matchingFtIdx[1]))
                matchingFtIdx[1].insert(0, 0)
        left = self._data

        matched = []
        merged = []
        unmatchedFtCountR = len(right[0]) - len(matchingFtIdx[1])
        matchMapper = {}
        for pt in left:
            match = [right[i] for i in range(len(right))
                     if right[i][onIdxR] == pt[onIdxL]]
            if len(match) > 0:
                matchMapper[pt[onIdxL]] = match

        for ptL in left:
            target = ptL[onIdxL]
            if target in matchMapper:
                matchesR = matchMapper[target]
                for ptR in matchesR:
                    # check for conflicts between matching features
                    matches = [ptL[i] == ptR[j] for i, j
                               in zip(matchingFtIdx[0], matchingFtIdx[1])]
                    nansL = [ptL[i] != ptL[i] for i in matchingFtIdx[0]]
                    nansR = [ptR[j] != ptR[j] for j in matchingFtIdx[1]]
                    acceptableValues = [m + nL + nR for m, nL, nR
                                        in zip(matches, nansL, nansR)]
                    if not all(acceptableValues):
                        msg = "The objects contain different values for the "
                        msg += "same feature"
                        raise InvalidArgumentValue(msg)
                    if sum(nansL) > 0:
                        # fill any nan values in left with the corresponding
                        # right value
                        for i, value in enumerate(ptL):

                            if value != value and i in matchingFtIdx[0]:
                                lIdx = matchingFtIdx[0].index(i)
                                ptL[i] = ptR[matchingFtIdx[1][lIdx]]
                    ptR = [ptR[i] for i in range(len(ptR))
                           if i not in matchingFtIdx[1]]
                    pt = ptL + ptR
                    merged.append(pt)
                matched.append(target)
            elif point in ['union', 'left']:
                ptR = [np.nan] * (len(right[0]) - len(matchingFtIdx[1]))
                pt = ptL + ptR
                merged.append(pt)

        if point == 'union':
            for row in right:
                target = row[onIdxR]
                if target not in matched:
                    pt = [np.nan] * (len(left[0]) + unmatchedFtCountR)
                    for i, j in zip(matchingFtIdx[0], matchingFtIdx[1]):
                        pt[i] = row[j]
                    pt[len(left[0]):] = [row[i] for i in range(len(right[0]))
                                         if i not in matchingFtIdx[1]]
                    merged.append(pt)

        self._dims = [len(merged), len(left[0]) + unmatchedFtCountR]
        if onFeature is None:
            # remove point names feature
            merged = [row[1:] for row in merged]
            self._dims[1] -= 1
        self._numFeatures = self._dims[1]

        self._data = merged

    def _replaceFeatureWithBinaryFeatures_implementation(self, uniqueIdx):
        toFill = np.zeros((len(self.points), len(uniqueIdx)))
        for ptIdx, val in enumerate(self._data):
            ftIdx = uniqueIdx[val[0]]
            toFill[ptIdx, ftIdx] = 1
        return List(toFill.tolist())

    def _getitem_implementation(self, x, y):
        return self._data[x][y]
    
    def _setitem_implementation(self, x, y, value):
        self._data[x][y] = value

    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd, dropDimension):
        kwds = {}
        kwds['data'] = ListPassThrough(self, pointStart, pointEnd,
                                       featureStart, featureEnd)
        kwds['shape'] = (pointEnd - pointStart, featureEnd - featureStart)
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

        return ListView(**kwds)

    def _createNestedObject(self, pointIndex):
        """
        Create an object of one less dimension
        """
        reshape = (self._dims[1], int(np.prod(self._dims[2:])))
        data = []
        point = self._data[pointIndex]
        for i in range(reshape[0]):
            start = i * reshape[1]
            end = start + reshape[1]
            data.append(point[start:end])

        return List(data, shape=self._dims[1:], reuseData=True)

    def _checkInvariants_implementation(self, level):
        assert len(self._data) == self.shape[0]
        assert self._numFeatures == self.shape[1]

        if level > 0:
            if len(self._data) > 0:
                expectedLength = len(self._data[0])
                assert expectedLength == self._numFeatures
            for point in self._data:
                if not isinstance(self, BaseView):
                    assert isinstance(point, list)
                assert len(point) == expectedLength
        if level > 1:
            for ithList in self._data:
                for jval in ithList:
                    assert isAllowedSingleElement(jval)

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0
        contained in this object. False otherwise
        """
        for point in self.points:
            if len(point._dims) == 2 and point._dims[0] == 1:
                for val in point:
                    if val == 0:
                        return True
            else:
                return point.containsZero()
        return False


    def _binaryOperations_implementation(self, opName, other):
        """
        Directs operations to use generic (numpy) operations, given that
        certain operations are implemented differently or not possible
        for lists.
        """
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
        ret = []
        for sPoint in self.points:
            retP = []
            for oFeature in other.features:
                runningTotal = 0
                for index in range(len(other.points)):
                    runningTotal += sPoint[index] * oFeature[index]
                retP.append(runningTotal)
            ret.append(retP)
        return List(ret)

    def _convertToNumericTypes_implementation(self, usableTypes):
        def needConversion(val):
            return type(val) not in usableTypes

        def convertType(val):
            if type(val) in usableTypes:
                return val
            return float(val)

        if any(any(needConversion(v) for v in pt) for pt in self._data):
            self._data = [list(map(convertType, pt)) for pt in self._data]

    def _iterateElements_implementation(self, order, only):
        array = np.array(self._data, dtype=np.object_)
        return NimbleElementIterator(array, order, only)


class ListView(BaseView, List):
    """
    Read only access to a List object.
    """

    def _getPoints(self, names):
        return ListPointsView(self, names)

    def _getFeatures(self, names):
        return ListFeaturesView(self, names)

    def _copy_implementation(self, to):
        # we only want to change how List and pythonlist copying is
        # done we also temporarily convert self._data to a python list
        # for copy
        if ((len(self.points) == 0 or len(self.features) == 0)
                and to != 'List'):
            emptyStandin = np.empty(self._dims)
            intermediate = nimble.data(emptyStandin, useLog=False)
            return intermediate.copy(to=to)

        # fastest way to generate list of view data
        listForm = [self._source._data[i][self._fStart:self._fEnd]
                    for i in range(self._pStart, self._pEnd)]

        if to not in ['List', 'pythonlist']:
            origData = self._data
            self._data = listForm
            res = super()._copy_implementation(to)
            self._data = origData
            return res

        if to == 'List':
            ptNames = self.points._getNamesNoGeneration()
            ftNames = self.features._getNamesNoGeneration()
            return List(listForm, pointNames=ptNames,
                        featureNames=ftNames, shape=self.shape)

        return listForm

    def _convertToNumericTypes_implementation(self, usableTypes):
        self._source._convertToNumericTypes_implementation(usableTypes)

class FeatureViewer(object):
    """
    View by feature axis for list.
    """
    def __init__(self, source, fStart, fEnd, pIndex):
        self.source = source
        self.fStart = fStart
        self.fEnd = fEnd
        self.fRange = fEnd - fStart
        self.limit = pIndex

    def __getitem__(self, key):
        if key < 0 or key >= self.fRange:
            msg = "The given index " + str(key) + " is outside of the "
            msg += "range  of possible indices in the feature axis (0 "
            msg += "to " + str(self.fRange - 1) + ")."
            raise IndexError(msg)

        return self.source._data[self.limit][key + self.fStart]

    def __len__(self):
        return self.fRange

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for sVal, oVal in zip(self, other):
            # check element equality - which is only relevant if one
            # of the elements is non-NaN

            if sVal != oVal and (sVal == sVal or oVal == oVal):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        return self.source._data[self.limit][self.fStart:self.fEnd]

class ListPassThrough(object):
    """
    Pass through to support View.
    """
    def __init__(self, source, pStart, pEnd, fStart, fEnd):
        self.source = source
        self.pStart = pStart
        self.pEnd = pEnd
        self.pRange = pEnd - pStart
        self.fStart = fStart
        self.fEnd = fEnd
        self.fviewer = None

    def __getitem__(self, key):
        if key < 0 or key >= self.pRange:
            msg = "The given index " + str(key) + " is outside of the "
            msg += "range  of possible indices in the point axis (0 "
            msg += "to " + str(self.pRange - 1) + ")."
            raise IndexError(msg)

        self.fviewer = FeatureViewer(self.source, self.fStart,
                                     self.fEnd, key + self.pStart)
        return self.fviewer

    def __len__(self):
        return self.pRange

    def __array__(self, dtype=None):
        tmpArray = np.array(self.source._data, dtype=dtype)
        return tmpArray[self.pStart:self.pEnd, self.fStart:self.fEnd]

###########
# Helpers #
###########

def _convertList(constructor, data):
    convert = constructor(data)
    if not allowedNumpyDType(convert.dtype):
        convert = constructor(data, dtype=object)
    return convert
