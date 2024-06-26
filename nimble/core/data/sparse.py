
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
Class extending Base, defining an object to hold and manipulate a scipy
coo_matrix.
"""

import warnings
import numbers

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble._utility import inheritDocstringsFactory
from nimble._utility import isAllowedSingleElement
from nimble._utility import scipy, pd
from nimble._utility import sparseMatrixToArray, removeDuplicatesNative
from nimble._utility import numpy2DArray
from .base import Base
from .views import BaseView
from .sparseAxis import SparsePoints, SparsePointsView
from .sparseAxis import SparseFeatures, SparseFeaturesView
from .stretch import StretchSparse
from ._dataHelpers import allDataIdentical
from ._dataHelpers import createDataNoValidation
from ._dataHelpers import csvCommaFormat
from ._dataHelpers import denseCountUnique
from ._dataHelpers import NimbleElementIterator
from ._dataHelpers import convertToNumpyOrder, modifyNumpyArrayValue
from ._dataHelpers import isValid2DObject
from ._dataHelpers import validateAxis

@inheritDocstringsFactory(Base)
class Sparse(Base):
    """
    Class providing implementations of data manipulation operations on
    data stored in a scipy coo matrix.

    Parameters
    ----------
    data : object
        A scipy sparse matrix or two-dimensional numpy array.
    reuseData : bool
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    _cooMatrixSkipCheck = None

    def __init__(self, data, reuseData=False, **kwds):
        if not scipy.nimbleAccessible():
            msg = 'To use class Sparse, scipy must be installed.'
            raise PackageException(msg)

        if not (isValid2DObject(data) or scipy.sparse.isspmatrix(data)):
            msg = "the input data can only be a scipy sparse matrix or a "
            msg += "two-dimensional numpy array or python list"
            raise InvalidArgumentType(msg)

        if scipy.sparse.isspmatrix_coo(data):
            if reuseData:
                self._data = data
            else:
                self._data = data.copy()
        elif scipy.sparse.isspmatrix(data):
            # data is a spmatrix in other format instead of coo
            self._data = data.tocoo()
        else: # data is np.array or python list
            if isinstance(data, list):
                data = numpy2DArray(data)
            # Sparse will convert None to 0 so we need to use np.nan instead
            # pylint: disable=singleton-comparison
            if data[data == None].size:
                if data.dtype not in [float, np.floating, object]:
                    data = data.astype(float)
                data[data == None] = np.nan

            self._data = scipy.sparse.coo_matrix(data)

        # class attribute prevents repeated object creation for subsequent
        # instances but set here to avoid deferred scipy import issues
        if Sparse._cooMatrixSkipCheck is None:

            class coo_matrix_skipcheck(scipy.sparse.coo_matrix):
                """
                coo_matrix constructor that ignores _check method.

                Used for increased efficiency when the data for this coo
                matrix was derived directly from another coo matrix so
                we are confident _check is unnecessary.
                """
                def __init__(self, *args, **kwargs):
                    backup = scipy.sparse.coo_matrix._check
                    try:
                        self._check = self._check_override
                        super().__init__(*args, **kwargs)
                    finally:
                        scipy.sparse.coo_matrix._check = backup

                def _check_override(self):
                    pass

            Sparse._cooMatrixSkipCheck = coo_matrix_skipcheck

        self._sorted = {'axis': None, 'indices': None}
        shape = kwds.get('shape', None)
        if shape is None:
            kwds['shape'] = self._data.shape
        super().__init__(**kwds)

    def _getPoints(self, names):
        return SparsePoints(self, names)

    def _getFeatures(self, names):
        return SparseFeatures(self, names)

    @property
    def stretch(self):
        return StretchSparse(self)

    def _plot(self, includeColorbar, outPath, show, title, xAxisLabel,
              yAxisLabel, **kwargs):
        toPlot = self.copy(to="Matrix")
        return toPlot._plot(includeColorbar, outPath, show, title, xAxisLabel,
                            yAxisLabel, **kwargs)

    def _transform_implementation(self, toTransform, points, features):
        if toTransform.preserveZeros:
            self._transformEachElement_zeroPreserve_implementation(
                toTransform, points, features)
        else:
            self._transformEachElement_noPreserve_implementation(
                toTransform, points, features)

    def _transformEachElement_noPreserve_implementation(self, toTransform,
                                                        points, features):
        # returns None if outside of the specified points and feature so that
        # when calculateForEach is called we are given a full data object
        # with only certain values modified.
        def wrapper(value, pID, fID):
            if points is not None and pID not in points:
                return None
            if features is not None and fID not in features:
                return None

            if toTransform.oneArg:
                return toTransform(value)

            return toTransform(value, pID, fID)

        # perserveZeros is always False in this helper, skipNoneReturnValues
        # is being hijacked by the wrapper: even if it was False, Sparse can't
        # contain None values.
        ret = self.calculateOnElements(wrapper, None, None,
                                       preserveZeros=False,
                                       skipNoneReturnValues=True, useLog=False)

        pnames = self.points._getNamesNoGeneration()
        fnames = self.features._getNamesNoGeneration()
        self._referenceFrom(ret, pointNames=pnames, featureNames=fnames)

    def _transformEachElement_zeroPreserve_implementation(
            self, toTransform, points, features):
        for index, val in enumerate(self._data.data):
            pID = self._data.row[index]
            fID = self._data.col[index]
            if points is not None and pID not in points:
                continue
            if features is not None and fID not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(val)
            else:
                currRet = toTransform(val, pID, fID)

            self._data.data = modifyNumpyArrayValue(self._data.data, index,
                                                    currRet)

        if all(issubclass(type(d), numbers.Number) for d in self._data.data):
            self._data.data = np.array(list(self._data.data))
        self._data.eliminate_zeros()

    def _calculate_implementation(self, function, points, features,
                                  preserveZeros):
        if not isinstance(self, nimble.core.data.BaseView):
            data = self._data.data
            row = self._data.row
            col = self._data.col
        else:
            # initiate generic implementation for view types
            preserveZeros = False
        # all data
        if preserveZeros and points is None and features is None:
            try:
                data = function(data)
            except Exception: # pylint: disable=broad-except
                function.otypes = [np.object_]
                data = function(data)
            shape = self._data.shape
            values = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
            # note: even if function transforms nonzero values into zeros
            # our init methods will filter them out from the data attribute
            return values
        # subset of data
        if preserveZeros:
            dataSubset = []
            rowSubset = []
            colSubset = []
            for idx, val in enumerate(data):
                if row[idx] in points and col[idx] in features:
                    rowSubset.append(row[idx])
                    colSubset.append(col[idx])
                    dataSubset.append(val)
            dataSubset = function(dataSubset)
            values = scipy.sparse.coo_matrix((dataSubset,
                                              (rowSubset, colSubset)))
            # note: even if function transforms nonzero values into zeros
            # our init methods will filter them out from the data attribute
            return values
        # zeros not preserved
        return self._calculate_genericVectorized(function, points, features)

    def _countUnique_implementation(self, points, features):
        uniqueCount = {}
        isView = isinstance(self, nimble.core.data.BaseView)
        if points is None and features is None and not isView:
            source = self
        else:
            pWanted = points if points is not None else slice(None)
            fWanted = features if features is not None else slice(None)
            source = self[pWanted, fWanted]
        uniqueCount = denseCountUnique(source._data.data)
        totalValues = (len(source.points) * len(source.features))
        numZeros = totalValues - len(source._data.data)
        if numZeros > 0:
            uniqueCount[0] = numZeros
        return uniqueCount

    def _transpose_implementation(self):
        self._data = self._data.transpose()
        self._resetSorted()

    def _isIdentical_implementation(self, other):
        if not isinstance(other, Sparse):
            return False
        # for nonempty matrices, we use a shape mismatch to indicate
        # non-equality
        if self._data.shape != other._data.shape:
            return False

        if isinstance(other, SparseView):
            return other._isIdentical_implementation(self)
        # not equal if number of non zero values differs
        if self._data.nnz != other._data.nnz:
            return False

        selfAxis = self._sorted['axis']
        otherAxis = other._sorted['axis']
        # make sure sorted internally the same way then compare
        if selfAxis != otherAxis or selfAxis is None:
            if selfAxis is None:
                self._sortInternal('feature')
                selfAxis = 'feature'
            if otherAxis != selfAxis:
                other._sortInternal(selfAxis)

        return (allDataIdentical(self._data.data, other._data.data)
                and allDataIdentical(self._data.row, other._data.row)
                and allDataIdentical(self._data.col, other._data.col))

    def _getTypeString_implementation(self):
        return 'Sparse'

    def _saveCSV_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        """
        Function to write the data in this object to a CSV file at the
        designated path.
        """
        with open(outPath, 'w', encoding='utf-8') as outFile:
            if includeFeatureNames:
                self._writeFeatureNamesToCSV(outFile, includePointNames)

            self._sortInternal('point')

            pointer = 0
            pmax = len(self._data.data)
            # write zero to file as same type as this data
            zero = self._data.data.dtype.type(0)
            for i in range(len(self.points)):
                if includePointNames:
                    currPname = csvCommaFormat(self.points.getName(i))
                    outFile.write(currPname)
                    outFile.write(',')
                for j in range(len(self.features)):
                    if (pointer < pmax and i == self._data.row[pointer]
                            and j == self._data.col[pointer]):
                        value = csvCommaFormat(self._data.data[pointer])
                        pointer = pointer + 1
                    else:
                        value = zero

                    if j != 0:
                        outFile.write(',')
                    outFile.write(str(value))
                outFile.write('\n')

    def _saveMTX_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        def makeNameString(count, namesItoN):
            nameString = "#"
            for i in range(count):
                nameString += namesItoN[i]
                if not i == count - 1:
                    nameString += ','
            return nameString

        header = ''
        if includePointNames:
            pNames = self.points.getNames()
            header = makeNameString(len(self.points), pNames)
            header += '\n'
        else:
            header += '#\n'
        if includeFeatureNames:
            fNames = self.features.getNames()
            header += makeNameString(len(self.features), fNames)
            header += '\n'
        else:
            header += '#\n'

        scipy.io.mmwrite(target=outPath, a=self._data.astype(np.float64),
                         comment=header)

    def _referenceFrom_implementation(self, other, kwargs):
        super()._referenceFrom_implementation(other, kwargs)
        self._sorted = other._sorted

    def _copy_implementation(self, to):
        if to in nimble.core.data.available:
            ptNames = self.points._getNamesNoGeneration()
            ftNames = self.features._getNamesNoGeneration()
            if to == 'Sparse':
                data = self._data.copy()
            else:
                data = sparseMatrixToArray(self._data)
            # reuseData=True since we already made copies here
            return createDataNoValidation(to, data, ptNames, ftNames,
                                          reuseData=True)
        if to == 'pythonlist':
            return sparseMatrixToArray(self._data).tolist()
        needsReshape = len(self._dims) > 2
        if to == 'numpyarray':
            ret = sparseMatrixToArray(self._data)
            if needsReshape:
                return ret.reshape(self._dims)
            return ret
        if needsReshape:
            data = np.empty(self._dims[:2], dtype=np.object_)
            for i in range(self.shape[0]):
                data[i] = self.points[i].copy('pythonlist')
        elif 'scipy' in to:
            data = self._data
        else:
            data = sparseMatrixToArray(self._data)
        if to == 'numpymatrix':
            return np.matrix(data)
        if 'scipy' in to:
            if to == 'scipycoo':
                if needsReshape:
                    return scipy.sparse.coo_matrix(data)
                return data.copy()
            try:
                ret = data.astype(np.float64)
            except ValueError as e:
                msg = f'Must create scipy {to[-3:]} matrix from numeric data'
                raise ValueError(msg) from e
            if to == 'scipycsc':
                return ret.tocsc()
            if to == 'scipycsr':
                return ret.tocsr()
        # pandasdataframe
        if not pd.nimbleAccessible():
            msg = "pandas is not available"
            raise PackageException(msg)
        pnames = self.points._getNamesNoGeneration()
        fnames = self.features._getNamesNoGeneration()
        return pd.DataFrame(data, index=pnames, columns=fnames)

    def _replaceRectangle_implementation(self, replaceWith, pointStart,
                                         featureStart, pointEnd, featureEnd):
        # sort values or call helper as needed
        constant = not isinstance(replaceWith, Base)
        if constant:
            if replaceWith == 0:
                self._replaceRectangle_zeros_implementation(
                    pointStart, featureStart, pointEnd, featureEnd)
                return
        else:
            replaceWith._sortInternal('point')

        # this has to be after the possible call to
        # _replaceRectangle_zeros_implementation; unnecessary for that helper
        self._sortInternal('point', setIndices=True)

        selfIdx = 0
        valsIdx = 0
        copyIndex = 0
        toAddData = []
        toAddRow = []
        toAddCol = []
        selfEnd = self._sorted['indices'][pointEnd + 1]
        if constant:
            valsEnd = ((pointEnd - pointStart + 1)
                       * (featureEnd - featureStart + 1))
        else:
            valsEnd = len(replaceWith._data.data)

        # Adjust selfIdx so that it begins at the values that might need to be
        # replaced, or, if no such values exist, set selfIdx such that the main
        # loop will ignore the contents of self.
        if len(self._data.data) > 0:
            selfIdx = self._sorted['indices'][pointStart]

            pcheck = self._data.row[selfIdx]
            fcheck = self._data.col[selfIdx]
            # the condition in the while loop is a natural break, if it isn't
            # satisfied then selfIdx will be exactly where we want it
            while fcheck < featureStart or fcheck > featureEnd:
                # this condition is an unatural break, when it is satisfied,
                # that means no value of selfIdx will point into the desired
                # values
                if pcheck > pointEnd or selfIdx == len(self._data.data) - 1:
                    selfIdx = selfEnd
                    break

                selfIdx += 1
                pcheck = self._data.row[selfIdx]
                fcheck = self._data.col[selfIdx]

            copyIndex = selfIdx

        # Walk full contents of both, modifying, shifing, or setting aside
        # values as needed. We will only ever increment one of selfIdx or
        # valsIdx at a time, meaning if there are matching entries, we will
        # encounter them. Due to the sorted precondition, if the location in
        # one object is less than the location in the other object, the lower
        # one CANNOT have a match.
        while selfIdx < selfEnd or valsIdx < valsEnd:
            if selfIdx < selfEnd:
                locationSP = self._data.row[selfIdx]
                locationSF = self._data.col[selfIdx]
            else:
                # we want to use unreachable values as sentials, so we + 1
                # since we're using inclusive endpoints
                locationSP = pointEnd + 1
                locationSF = featureEnd + 1

            # adjust 'values' locations into the scale of the calling object
            locationVP = pointStart
            locationVF = featureStart
            if constant:
                vData = replaceWith
                # uses truncation of int division
                locationVP += valsIdx / (featureEnd - featureStart + 1)
                locationVF += valsIdx % (featureEnd - featureStart + 1)
            elif valsIdx >= valsEnd:
                locationVP += pointEnd + 1
                locationVF += featureEnd + 1
            else:
                vData = replaceWith._data.data[valsIdx]
                locationVP += replaceWith._data.row[valsIdx]
                locationVF += replaceWith._data.col[valsIdx]

            pCmp = locationSP - locationVP
            fCmp = locationSF - locationVF
            trueCmp = pCmp if pCmp != 0 else fCmp

            # Case: location at index into self is higher than location
            # at index into values. No matching entry in self;
            # copy if space, or record to be added at end.
            if trueCmp > 0:
                # can only copy into self if there is open space
                if copyIndex < selfIdx:
                    self._data.data[copyIndex] = vData
                    self._data.row[copyIndex] = locationVP
                    self._data.col[copyIndex] = locationVF
                    copyIndex += 1
                else:
                    toAddData.append(vData)
                    toAddRow.append(locationVP)
                    toAddCol.append(locationVF)

                #increment valsIdx
                valsIdx += 1
            # Case: location at index into other is higher than location at
            # index into self. no matching entry in values - fill this entry
            # in self with zero (by shifting past it)
            elif trueCmp < 0:
                # need to do cleanup if we're outside of the relevant bounds
                if locationSF < featureStart or locationSF > featureEnd:
                    self._data.data[copyIndex] = self._data.data[selfIdx]
                    self._data.row[copyIndex] = self._data.row[selfIdx]
                    self._data.col[copyIndex] = self._data.col[selfIdx]
                    copyIndex += 1
                selfIdx += 1
            # Case: indices point to equal locations.
            else:
                self._data.data[copyIndex] = vData
                self._data.row[copyIndex] = locationVP
                self._data.col[copyIndex] = locationVF
                copyIndex += 1

                # increment both??? or just one?
                selfIdx += 1
                valsIdx += 1

        # Now we have to walk through the rest of self, finishing the copying
        # shift if necessary
        if copyIndex != selfIdx:
            while selfIdx < len(self._data.data):
                self._data.data[copyIndex] = self._data.data[selfIdx]
                self._data.row[copyIndex] = self._data.row[selfIdx]
                self._data.col[copyIndex] = self._data.col[selfIdx]
                selfIdx += 1
                copyIndex += 1
        else:
            copyIndex = len(self._data.data)

        newData = np.empty(copyIndex + len(toAddData),
                           dtype=self._data.data.dtype)
        newData[:copyIndex] = self._data.data[:copyIndex]
        newData[copyIndex:] = toAddData
        newRow = np.empty(copyIndex + len(toAddRow))
        newRow[:copyIndex] = self._data.row[:copyIndex]
        newRow[copyIndex:] = toAddRow
        newCol = np.empty(copyIndex + len(toAddCol))
        newCol[:copyIndex] = self._data.col[:copyIndex]
        newCol[copyIndex:] = toAddCol
        shape = (len(self.points), len(self.features))
        self._data = scipy.sparse.coo_matrix((newData, (newRow, newCol)),
                                             shape)

        self._resetSorted()

    def _flatten_implementation(self, order):
        numElem = len(self.points) * len(self.features)
        order = convertToNumpyOrder(order)

        self._data = self._data.reshape((1, numElem), order=order)
        self._resetSorted()

    def _unflatten_implementation(self, reshape, order):
        order = convertToNumpyOrder(order)
        self._data = self._data.reshape(reshape, order=order)
        self._resetSorted()

    def _mergeIntoNewData(self, copyIndex, toAddData, toAddRow, toAddCol):
        #instead of always copying, use reshape or resize to sometimes cut
        # array down to size???
        pass

    def _replaceRectangle_zeros_implementation(self, pointStart, featureStart,
                                               pointEnd, featureEnd):
        # walk through col listing and partition all data: extract, and kept,
        # reusing the sparse matrix underlying structure to save space
        copyIndex = 0

        for lookIndex, lookData in enumerate(self._data.data):
            currP = self._data.row[lookIndex]
            currF = self._data.col[lookIndex]
            # if it is in range we want to obliterate the entry by just passing
            # it by and copying over it later
            if (currP >= pointStart
                    and currP <= pointEnd
                    and currF >= featureStart
                    and currF <= featureEnd):
                pass
            else:
                self._data.data[copyIndex] = lookData
                self._data.row[copyIndex] = self._data.row[lookIndex]
                self._data.col[copyIndex] = self._data.col[lookIndex]
                copyIndex += 1

        # reinstantiate self
        # (cannot reshape coo matrices, so cannot do this in place)
        newData = self._data.data[0:copyIndex]
        newRow = self._data.row[0:copyIndex]
        newCol = self._data.col[0:copyIndex]
        shape = (len(self.points), len(self.features))
        self._data = scipy.sparse.coo_matrix((newData, (newRow, newCol)),
                                             shape)

    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):
        self._sortInternal('feature')
        other._sortInternal('feature')
        tempFtsL = len(self.features)
        leftFtCount = tempFtsL
        tempFtsR = len(other.features)
        rightFtCount = tempFtsR - len(matchingFtIdx[0])
        if onFeature is not None:
            onIdxL = self.features.getIndex(onFeature)
            onIdxR = other.features.getIndex(onFeature)
            leftData = self._data.data
            leftRow = self._data.row
            leftCol = self._data.col
            rightData = other._data.data.copy()
            rightRow = other._data.row.copy()
            rightCol = other._data.col.copy()
        else:
            onIdxL = 0
            onIdxR = 0
            tempFtsL += 1
            tempFtsR += 1
            leftData = self._data.data.astype(np.object_)
            if not self.points._anyDefaultNames():
                leftData = np.append([self.points.getNames()], leftData)
            elif self.points._namesCreated():
                # differentiate default names between objects
                leftNames = [i if n is None else n for i, n
                             in enumerate(self.points.getNames())]
                leftData = np.append([leftNames], leftData)
            else:
                leftNames = list(range(len(self.points)))
                leftData = np.append(leftNames, leftData)
            leftRow = np.append(list(range(len(self.points))), self._data.row)
            leftCol = np.append([0 for _ in range(len(self.points))],
                                self._data.col + 1)
            rightData = other._data.data.copy().astype(np.object_)
            if not other.points._anyDefaultNames():
                rightData = np.append([other.points.getNames()], rightData)
            elif other.points._namesCreated():
                maxL = len(self.points)
                # differentiate default names between objects
                rightNames = [i + maxL if n is None else n for i, n in
                              enumerate(other.points.getNames())]
                rightData = np.append([rightNames], rightData)
            else:
                rtRange = range(self.shape[0], self.shape[0] + other.shape[0])
                rightNames = list(rtRange)
                rightData = np.append(rightNames, rightData)
            rightRow = np.append(list(range(len(other.points))),
                                 other._data.row.copy())
            rightCol = np.append([0 for i in range(len(other.points))],
                                 other._data.col.copy() + 1)
            matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
            matchingFtIdx[0].insert(0, 0)
            matchingFtIdx[1] = list(map(lambda x: x + 1, matchingFtIdx[1]))
            matchingFtIdx[1].insert(0, 0)

        mergedData = np.empty((0, 0))
        mergedRow = []
        mergedCol = []
        matched = []
        nextPt = 0
        numPts = 0

        for ptIdxL, target in enumerate(leftData[leftCol == onIdxL]):
            rowIdxR = np.where(rightData[rightCol == onIdxR] == target)[0]
            if len(rowIdxR) > 0:
                for ptIdxR in rowIdxR:
                    ptL = np.zeros((tempFtsL,), dtype=np.object_)
                    inRowL = leftRow == ptIdxL
                    ptL[leftCol[inRowL]] = leftData[inRowL]
                    ptR = np.zeros((tempFtsR,), dtype=np.object_)
                    inRowR = rightRow == ptIdxR
                    ptR[rightCol[inRowR]] = rightData[inRowR]
                    matches = ptL[matchingFtIdx[0]] == ptR[matchingFtIdx[1]]
                    nansL = ptL[matchingFtIdx[0]] != ptL[matchingFtIdx[0]]
                    nansR = ptR[matchingFtIdx[1]] != ptR[matchingFtIdx[1]]
                    acceptableValues = matches + nansL + nansR
                    if not all(acceptableValues):
                        msg = "The objects contain different values for the "
                        msg += "same feature"
                        raise InvalidArgumentValue(msg)
                    if len(nansL) > 0:
                        # fill any nan values in left with the corresponding
                        # right value
                        for i, value in enumerate(ptL[matchingFtIdx[0]]):
                            if value != value:
                                fill = ptR[matchingFtIdx[1]][i]
                                ptL[matchingFtIdx[0]][i] = fill
                    ptR = ptR[[i for i in range(len(ptR))
                               if i not in matchingFtIdx[1]]]
                    pt = np.append(ptL, ptR)
                    if feature == "intersection":
                        pt = pt[matchingFtIdx[0]]
                        leftFtCount = len(matchingFtIdx[0])
                    elif onFeature is not None and feature == "left":
                        pt = pt[:len(self.features)]
                    elif feature == "left":
                        pt = pt[:len(self.features) + 1]
                    if onFeature is None:
                        pt = pt[1:]
                    matched.append(target)
                    mergedData = np.append(mergedData, pt)
                    mergedRow.extend([nextPt] * len(pt))
                    mergedCol.extend(list(range(len(pt))))
                    nextPt += 1
                    numPts += 1
            elif point in ["union", "left"]:
                ptL = np.zeros((tempFtsL,), dtype=np.object_)
                inRowL = leftRow == ptIdxL
                ptL[leftCol[inRowL]] = leftData[inRowL]
                ptR = [np.nan] * (tempFtsR - len(matchingFtIdx[1]))
                pt = np.append(ptL, ptR)
                if feature == "intersection":
                    pt = pt[matchingFtIdx[0]]
                elif feature == "left":
                    pt = pt[:tempFtsL]
                if onFeature is None:
                    pt = pt[1:]
                mergedData = np.append(mergedData, pt)
                mergedRow.extend([nextPt] * len(pt))
                mergedCol.extend(list(range(len(pt))))
                nextPt += 1
                numPts += 1

        if point == 'union':
            for ptIdxR, target in enumerate(rightData[rightCol == onIdxR]):
                if target not in matched:
                    nanList = [np.nan] * tempFtsL
                    ptL = np.array(nanList, dtype=np.object_)
                    ptR = np.zeros((tempFtsR,), dtype=np.object_)
                    inRowR = rightRow == ptIdxR
                    ptR[rightCol[inRowR]] = rightData[inRowR]
                    fill = ptR[matchingFtIdx[1]]
                    ptL[matchingFtIdx[0]] = fill
                    # only unmatched points from right
                    notMatchedR = np.in1d(rightCol, matchingFtIdx[1],
                                          invert=True)
                    ptR = rightData[(rightRow == ptIdxR) & (notMatchedR)]
                    pt = np.append(ptL, ptR)
                    if feature == "intersection":
                        pt = pt[matchingFtIdx[0]]
                    elif feature == "left":
                        pt = pt[:tempFtsL]
                    if onFeature is None:
                        # remove pointNames column added
                        pt = pt[1:]
                    mergedData = np.append(mergedData, pt)
                    mergedRow.extend([nextPt] * len(pt))
                    mergedCol.extend(list(range(len(pt))))
                    nextPt += 1
                    numPts += 1

        numFts = leftFtCount + rightFtCount
        if onFeature is not None and feature == "intersection":
            numFts = len(matchingFtIdx[0])
        elif feature == "intersection":
            numFts = len(matchingFtIdx[0]) - 1
        elif feature == "left":
            numFts = len(self.features)
        if len(mergedData) == 0:
            mergedData = []

        mergedData = mergedData.astype(np.float64)
        self._dims = [numPts, numFts]
        self._data = scipy.sparse.coo_matrix(
            (mergedData, (mergedRow, mergedCol)), shape=(numPts, numFts))

        self._data.eliminate_zeros()
        self._resetSorted()

    def _replaceFeatureWithBinaryFeatures_implementation(self, uniqueIdx):
        binaryRow = []
        binaryCol = []
        binaryData = []
        for ptIdx, val in zip(self._data.row, self._data.data):
            ftIdx = uniqueIdx[val]
            binaryRow.append(ptIdx)
            binaryCol.append(ftIdx)
            binaryData.append(1)
        shape = (len(self.points), len(uniqueIdx))
        binaryCoo = scipy.sparse.coo_matrix(
            (binaryData, (binaryRow, binaryCol)), shape=shape)
        self._resetSorted()
        return Sparse(binaryCoo)

    def _getitem_implementation(self, x, y):
        """
        currently, we sort the data first and then do binary search
        """
        sortedAxis = self._sorted['axis']
        sort = sortedAxis if sortedAxis is not None else 'point'
        self._sortInternal(sort, setIndices=True)

        if self._sorted['axis'] == 'point':
            offAxis = self._data.col
            axisVal = x
            offAxisVal = y
        else:
            offAxis = self._data.row
            axisVal = y
            offAxisVal = x

        # binary search
        # if value is zero, return should be the same type as the nonzero data
        start, end = self._sorted['indices'][axisVal:axisVal + 2]
        if start == end: # axisVal is not in self._data.row
            return self._data.data.dtype.type(0)
        k = np.searchsorted(offAxis[start:end], offAxisVal) + start
        if k < end and offAxis[k] == offAxisVal:
            return self._data.data[k]
        return self._data.data.dtype.type(0) 
    
    def _setitem_implementation(self, x, y, value):
        """ 
        currently, we sort the data first and then do binary search
        """
        sortedAxis = self._sorted['axis']
        sort = sortedAxis if sortedAxis is not None else 'point'
        self._sortInternal(sort, setIndices=True)
        
        if self._sorted['axis'] == 'point':
            offAxis = self._data.col
            axisVal = x
            offAxisVal = y
        else:
            offAxis = self._data.row
            axisVal = y
            offAxisVal = x
            
        # binary search
        start, end = self._sorted['indices'][axisVal:axisVal + 2]
        if start != end:
            k = np.searchsorted(offAxis[start:end], offAxisVal) + start
            if k < end and offAxis[k] == offAxisVal:
                if value != 0:
                    self._data.data[k] = value  # Update existing entry
                else:
                    # Remove entry if value is zero (maintain sparsity)
                    self._data.data = np.delete(self._data.data, k)
                    self._data.row = np.delete(self._data.row, k)
                    self._data.col = np.delete(self._data.col, k)
                    # Re-sort and re-index
                    self._sortInternal(self._sorted['axis'], setIndices=True, forceSort=True)
                return
        if value != 0:
            numPts = len(list(self.points))
            numFts = len(list(self.features))
            # Insert new entry if value is not zero
            self._data.data = np.append(self._data.data, value)
            self._data.row = np.append(self._data.row, x)
            self._data.col = np.append(self._data.col, y)
            self._data = scipy.sparse.coo_matrix(
                (self._data.data, (self._data.row, self._data.col)),
                 shape=(numPts,numFts)
            )
            # Re-sort and re-index
            self._sortInternal(self._sorted['axis'], setIndices=True, forceSort=True)
            
    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd, dropDimension):
        """
        The Sparse object specific implementation necessarly to complete
        the Base object's view method. pointStart and feature start are
        inclusive indices, pointEnd and featureEnd are exclusive
        indices.
        """
        kwds = {}
        kwds['source'] = self
        kwds['pointStart'] = pointStart
        kwds['pointEnd'] = pointEnd
        kwds['featureStart'] = featureStart
        kwds['featureEnd'] = featureEnd

        allPoints = pointStart == 0 and pointEnd == len(self.points)
        singlePoint = pointEnd - pointStart == 1
        allFeats = featureStart == 0 and featureEnd == len(self.features)
        singleFeat = featureEnd - featureStart == 1
        # singleFeat = singlePoint = False
        if singleFeat or singlePoint:
            pshape = pointEnd - pointStart
            fshape = featureEnd - featureStart

            axis = 'point' if singlePoint else 'feature'
            self._sortInternal(axis, setIndices=True)

            if singlePoint:
                primaryStart = pointStart
                allOtherAxis = allFeats
                sortedSecondary = self._data.col
                secondaryStart = featureStart
                secondaryEnd = featureEnd
            else:
                primaryStart = featureStart
                allOtherAxis = allPoints
                sortedSecondary = self._data.row
                secondaryStart = pointStart
                secondaryEnd = pointEnd

            # start and end values have already been stored during sorting
            sortedAxisIndices = self._sorted['indices']
            start, end = sortedAxisIndices[primaryStart:primaryStart + 2]
            if not allOtherAxis:
                secondaryLimited = sortedSecondary[start:end]
                targetSecondary = [secondaryStart, secondaryEnd]
                innerStart, innerEnd = np.searchsorted(secondaryLimited,
                                                       targetSecondary)
                outerStart = start
                start = start + innerStart
                end = outerStart + innerEnd
            # high dimension data only allowed for points
            if singlePoint and len(self._dims) > 2:
                if dropDimension:
                    firstIdx = 1
                    pshape = self._dims[firstIdx]
                    fshape = int(np.prod(self._dims[firstIdx + 1:]))
                    kwds['pointStart'] = 0
                    kwds['pointEnd'] = self._dims[1]
                    kwds['featureStart'] = 0
                    kwds['featureEnd'] = int(np.prod(self._dims[2:]))
                else:
                    firstIdx = 0
                primary = sortedSecondary[start:end] // fshape
                secondary = sortedSecondary[start:end] % fshape
                kwds['shape'] = [pshape] + self._dims[firstIdx + 1:]
            else:
                primary = np.zeros((end - start,), dtype=int)
                secondary = sortedSecondary[start:end] - secondaryStart

            if singlePoint:
                row = primary
                col = secondary
            else:
                col = primary
                row = secondary

            data = self._data.data[start:end]

            newInternal = Sparse._cooMatrixSkipCheck(
                (data, (row, col)), shape=(pshape, fshape), copy=False)
            kwds['data'] = newInternal
            if singlePoint and len(self._dims) > 2 and dropDimension:
                kwds['source'] = Sparse(newInternal, shape=kwds['shape'],
                                        reuseData=True)

            return SparseVectorView(**kwds)

        # window shaped View
        # the data should be dummy data, but data.shape must
        # be = (pointEnd - pointStart, featureEnd - featureStart)
        newInternal = scipy.sparse.coo_matrix([])
        newInternal._shape = (pointEnd - pointStart,
                              featureEnd - featureStart)
        newInternal._data = None
        kwds['data'] = newInternal
        if len(self._dims) > 2:
            shape = self._dims.copy()
            shape[0] = pointEnd - pointStart
            kwds['shape'] = shape

        return SparseView(**kwds)

    def _checkInvariants_implementation(self, level):
        assert self._data.shape[0] == self.shape[0]
        assert self._data.shape[1] == self.shape[1]
        assert scipy.sparse.isspmatrix_coo(self._data)

        if level > 0:
            try:
                noZerosInData = all(self._data.data != 0)
                #numpy may say: elementwise comparison failed; returning
                # scalar instead, but in the future will perform
                # elementwise comparison
            except ValueError:
                noZerosInData = all(i != 0 for i in self._data.data)
            assert noZerosInData

            assert self._data.dtype.type is not np.string_

            sortedAxis = self._sorted['axis']
            sortedIndices = self._sorted['indices']
            if sortedAxis is not None:
                rowBefore = self._data.row.copy()
                colBefore = self._data.col.copy()
                self._resetSorted()
                if sortedIndices is not None:
                    self._sortInternal(sortedAxis, setIndices=True)
                    # _sortInternal indices sort incorrect?
                    assert all(self._sorted['indices'][:] == sortedIndices[:])
                else:
                    self._sortInternal(sortedAxis)
                # _sortInternal axis sort incorrect?
                assert all(self._data.row[:] == rowBefore[:])
                assert all(self._data.col[:] == colBefore[:])

            withoutReplicasCoo = removeDuplicatesNative(self._data)
            assert len(self._data.data) == len(withoutReplicasCoo.data)

            with warnings.catch_warnings():
                warnings.simplefilter('error')
                # call the coo_matrix structure consistency checker
                self._data._check()

        if level > 1 and self._data.data.dtype == object:
            allowed = np.vectorize(isAllowedSingleElement, otypes=[bool])
            assert allowed(self._data.data).all()

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0
        contained in this object. False otherwise
        """
        return (self.shape[0] * self.shape[1]) > self._data.nnz


    def _binaryOperations_implementation(self, opName, other):
        """
        Directs the operation to the best implementation available,
        preserving the sparse representation whenever possible.
        """
        # scipy mul and pow operators are not elementwise
        if 'mul' in opName:
            return self._genericMul_implementation(opName, other)
        if 'pow' in opName:
            return self._genericPow_implementation(opName, other)
        try:
            if isinstance(other, Base):
                selfData = self._getSparseData()
                if isinstance(other, SparseView):
                    other = other.copy(to='Sparse')
                if isinstance(other, Sparse):
                    otherData = other._getSparseData()
                else:
                    otherData = other.copy('Matrix')._data
                ret = getattr(selfData, opName)(otherData)
            else:
                return self._scalarBinary_implementation(opName, other)

            if ret is NotImplemented:
                # most NotImplemented are inplace operations
                if opName.startswith('__i'):
                    return self._inplaceBinary_implementation(opName, other)
                if opName == '__rsub__':
                    return self._rsub__implementation(other)
                return self._defaultBinaryOperations_implementation(opName,
                                                                    other)

            return Sparse(ret)

        except AttributeError:
            if opName.startswith('__i'):
                return self._inplaceBinary_implementation(opName, other)
            if 'floordiv' in opName:
                return self._genericFloordiv_implementation(opName, other)
            if 'mod' in opName:
                return self._genericMod_implementation(opName, other)
            return self._defaultBinaryOperations_implementation(opName, other)


    def _scalarBinary_implementation(self, opName, other):
        oneSafe = ['mul', '__truediv__', '__itruediv__', '__pow__', '__ipow__']
        if any(name in opName for name in oneSafe) and other == 1:
            selfData = self._getSparseData()
            return Sparse(selfData)
        zeroSafe = ['mul', 'div', 'mod']
        zeroPreserved = any(name in opName for name in zeroSafe)
        if opName in ['__pow__', '__ipow__'] and other > 0:
            zeroPreserved = True
        if zeroPreserved:
            return self._scalarZeroPreservingBinary_implementation(
                opName, other)

        # scalar operations apply to all elements; use dense
        return self._defaultBinaryOperations_implementation(opName,
                                                            other)

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
        if isinstance(other, BaseView):
            retData = other.copy(to='scipycsr')
            retData = self._data * retData
        else:
            # for other._data as any dense or sparse matrix
            retData = self._data * other._data

        return nimble.data(retData, returnType='Sparse', useLog=False)

    def _inplaceBinary_implementation(self, opName, other):
        notInplace = '__' + opName[3:]
        ret = self._binaryOperations_implementation(notInplace, other)
        self._referenceFrom(ret, paths=(self._absPath, self._relPath))
        return self

    def _rsub__implementation(self, other):
        return (self * -1)._binaryOperations_implementation('__add__', other)

    def _genericMul_implementation(self, opName, other):
        if not isinstance(other, Base):
            return self._scalarBinary_implementation(opName, other)
        if 'i' in opName:
            target = self
        else:
            target = self.copy()

        # CHOICE OF OUTPUT WILL BE DETERMINED BY SCIPY!!!!!!!!!!!!
        # for other._data as any dense or sparse matrix
        if isinstance(other, Sparse):
            toMul = other._getSparseData()
        elif isinstance(other, nimble.core.data.Matrix):
            toMul = scipy.sparse.coo_matrix(other._data)
        else:
            toMul = scipy.sparse.coo_matrix(other.copy(to='numpyarray'))
        raw = target._data.multiply(toMul)
        if scipy.sparse.isspmatrix(raw):
            raw = raw.tocoo()
        else:
            raw = scipy.sparse.coo_matrix(raw, shape=self._data.shape)
        target._data = raw
        return target

    def _genericPow_implementation(self, opName, other):
        if not isinstance(other, Base):
            return self._scalarBinary_implementation(opName, other)
        if 'i' in opName:
            caller = self
            callee = other
        elif 'r' in opName:
            caller = other.copy('Sparse')
            callee = self
        else:
            caller = self.copy()
            callee = other

        def powFromRight(val, pnum, fnum):
            try:
                return val ** callee[pnum, fnum]
            except Exception:
                self._numericValidation()
                other._numericValidation(right=True)
                raise

        return caller.calculateOnElements(powFromRight, useLog=False)

    def _genericFloordiv_implementation(self, opName, other):
        """
        Perform floordiv by modifying the results of truediv.

        There is no need for additional conversion when an inplace
        operation is called because _binaryOperations_implementation will
        return the self object in those cases, so the changes below are
        reflected inplace.
        """
        opSplit = opName.split('floordiv')
        trueDiv = opSplit[0] + 'truediv__'
        # ret is self for inplace operation
        ret = self._binaryOperations_implementation(trueDiv, other)
        ret._data.data = np.floor(ret._data.data)
        ret._data.eliminate_zeros()
        return ret

    def _genericMod_implementation(self, opName, other):
        """
        Uses np.mod on the nonzero values in the coo matrix.

        Since 0 % any value is 0, the zero values can be ignored for
        this operation.
        """
        sortedAxis = self._sorted['axis']
        sort = sortedAxis if sortedAxis is not None else 'point'
        self._sortInternal(sort)
        selfData = self._getSparseData()
        if not isinstance(other, Sparse):
            other = other.copy('Sparse')
        other._sortInternal(sort)
        otherData = other._getSparseData()
        if opName == '__rmod__':
            ret = np.mod(otherData.data, selfData.data)
        else:
            ret = np.mod(selfData.data, otherData.data)
        coo = scipy.sparse.coo_matrix((ret, (selfData.row, selfData.col)),
                                      shape=self.shape)
        coo.eliminate_zeros() # remove any zeros introduced into data

        return Sparse(coo)

    def _scalarZeroPreservingBinary_implementation(self, opName, other):
        """
        Helper applying operation directly to data attribute.

        Zeros are preserved for these operations, so we only need to
        apply the operation to the data attribute, the row and col
        attributes will remain unchanged.
        """
        selfData = self._getSparseData()
        ret = getattr(selfData.data, opName)(other)
        coo = scipy.sparse.coo_matrix((ret, (selfData.row, selfData.col)),
                                      shape=self.shape)
        coo.eliminate_zeros() # remove any zeros introduced into data
        if opName.startswith('__i'):
            self._data = coo
            return self
        return Sparse(coo)


    ###########
    # Helpers #
    ###########

    def _sortInternal(self, axis, setIndices=False, forceSort=False):
        if self._sorted['axis'] == axis and not forceSort:
            # axis sorted; can return now if do not need to set indices
            if not setIndices or self._sorted['indices'] is not None:
                return
        else:
            validateAxis(axis)
            if axis == "point":
                sortPrime = self._data.row
                sortOff = self._data.col
            else:
                sortPrime = self._data.col
                sortOff = self._data.row
            # sort least significant axis first
            sortKeys = np.lexsort((sortOff, sortPrime))

            self._data.data = self._data.data[sortKeys]
            self._data.row = self._data.row[sortKeys]
            self._data.col = self._data.col[sortKeys]

            self._sorted['axis'] = axis
            self._sorted['indices'] = None

        if setIndices:
            if axis == "point":
                sortedAxis = self._data.row
                sortedLength = len(self.points)
            else:
                sortedAxis = self._data.col
                sortedLength = len(self.features)

            indices = np.searchsorted(sortedAxis, range(sortedLength + 1))
            self._sorted['indices'] = indices

    def _getSparseData(self):
        """
        Get the backend coo_matrix data for this object.

        Since Views set self._data.data to None, we need to copy the view
        to gain access to the coo_matrix data.
        """
        if isinstance(self, BaseView):
            selfData = self.copy()._data
        else:
            selfData = self._data
        return selfData

    def _convertToNumericTypes_implementation(self, usableTypes):
        if self._data.dtype not in usableTypes:
            self._data = self._data.astype(float)
            self._resetSorted()

    def _iterateElements_implementation(self, order, only):
        if only is not None and not only(0): # we can ignore zeros
            self._sortInternal(order)
            array = self._data.data
        else:
            array = sparseMatrixToArray(self._data)
        return NimbleElementIterator(array, order, only)

    def _isBooleanData(self):
        return self._data.dtype in [bool, np.bool_]

    def _resetSorted(self):
        self._sorted['axis'] = None
        self._sorted['indices'] = None

###################
# Generic Helpers #
###################

class SparseVectorView(BaseView, Sparse):
    """
    A view of a Sparse data object limited to a full point or full
    feature.
    """

    def _getPoints(self, names):
        return SparsePointsView(self, names)

    def _getFeatures(self, names):
        return SparseFeaturesView(self, names)

class SparseView(BaseView, Sparse):
    """
    Read only access to a Sparse object.
    """

    def _getPoints(self, names):
        return SparsePointsView(self, names)

    def _getFeatures(self, names):
        return SparseFeaturesView(self, names)

    def _checkInvariants_implementation(self, level):
        self._source.checkInvariants(level)

    def _getitem_implementation(self, x, y):
        adjX = x + self._pStart
        adjY = y + self._fStart
        return self._source._getitem_implementation(adjX, adjY)

    def _copy_implementation(self, to):
        if to == "Sparse":
            sourceData = self._source._data.data.copy()
            sourceRow = self._source._data.row.copy()
            sourceCol = self._source._data.col.copy()

            keep = ((sourceRow >= self._pStart)
                    & (sourceRow < self._pEnd)
                    & (sourceCol >= self._fStart)
                    &(sourceCol < self._fEnd))
            keepData = sourceData[keep]
            keepRow = sourceRow[keep]
            keepCol = sourceCol[keep]
            if self._pStart > 0:
                keepRow = list(map(lambda x: x - self._pStart, keepRow))
            if self._fStart > 0:
                keepCol = list(map(lambda x: x - self._fStart, keepCol))
            shape = (len(self.points), len(self.features))
            coo = scipy.sparse.coo_matrix((keepData, (keepRow, keepCol)),
                                          shape=shape)
            pNames = None
            fNames = None
            if self.points._namesCreated():
                pNames = self.points.getNames()
            if self.features._namesCreated():
                fNames = self.features.getNames()
            return Sparse(coo, pointNames=pNames, featureNames=fNames)

        if len(self.points) == 0 or len(self.features) == 0:
            emptyStandin = np.empty(self._dims)
            intermediate = nimble.data(emptyStandin, useLog=False)
            return intermediate.copy(to=to)

        if to == 'numpyarray':
            pStart, pEnd = self._pStart, self._pEnd
            fStart, fEnd = self._fStart, self._fEnd
            asArray = sparseMatrixToArray(self._source._data)
            limited = asArray[pStart:pEnd, fStart:fEnd]
            if len(self._dims) > 2:
                return limited.reshape(self._dims)
            return limited.copy()

        limited = self._source.points.copy(start=self._pStart,
                                           end=self._pEnd - 1, useLog=False)
        if self._fEnd - self._fStart < len(self._source.features):
            limited = limited.features.copy(start=self._fStart,
                                            end=self._fEnd - 1, useLog=False)

        return limited._copy_implementation(to)

    def _isIdentical_implementation(self, other):
        if not isinstance(other, Sparse):
            return False
        # for nonempty matrices, we use a shape mismatch to indicate
        # non-equality
        if self._data.shape != other._data.shape:
            return False

        # empty object means no values. Since shapes match they're equal
        if self._data.shape[0] == 0 or self._data.shape[1] == 0:
            return True

        sIt = self.points
        oIt = other.points
        for sPoint, oPoint in zip(sIt, oIt):

            if sPoint != oPoint:
                return False
            if sPoint != sPoint and oPoint == oPoint:
                return False
            if sPoint == sPoint and oPoint != oPoint:
                return False

        return True

    def _containsZero_implementation(self):
        for sPoint in self.points:
            if sPoint.containsZero():
                return True
        return False

    def _binaryOperations_implementation(self, opName, other):
        selfConv = self.copy(to="Sparse")
        if isinstance(other, BaseView):
            other = other.copy(to=other.getTypeString())

        return selfConv._binaryOperations_implementation(opName, other)

    def __abs__(self):
        """ Perform element wise absolute value on this object """
        ret = self.copy(to="Sparse")
        np.absolute(ret._data.data, out=ret._data.data)
        ret._name = None

        return ret

    def _matmul__implementation(self, other):
        selfConv = self.copy(to="Sparse")
        if isinstance(other, BaseView):
            other = other.copy(to=other.getTypeString())
        return selfConv._matmul__implementation(other)

    def _saveCSV_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        selfConv = self.copy(to="Sparse")
        selfConv._saveCSV_implementation(outPath, includePointNames,
                                         includeFeatureNames)

    def _saveMTX_implementation(self, outPath, includePointNames,
                                includeFeatureNames):
        selfConv = self.copy(to="Sparse")
        selfConv._saveMTX_implementation(outPath, includePointNames,
                                         includeFeatureNames)

    def _convertToNumericTypes_implementation(self, usableTypes):
        self._source._convertToNumericTypes_implementation(usableTypes)

    def _iterateElements_implementation(self, order, only):
        selfConv = self.copy(to="Sparse")
        return selfConv._iterateElements_implementation(order, only)

    def _isBooleanData(self):
        return self._source._data.dtype in [bool, np.bool_]
