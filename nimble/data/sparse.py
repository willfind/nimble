"""
Class extending Base, defining an object to hold and manipulate a scipy
coo_matrix.
"""

from __future__ import division
from __future__ import absolute_import
from functools import reduce

import numpy
from six.moves import range
from six.moves import zip

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException, ImproperObjectAction
from nimble.utility import inheritDocstringsFactory, numpy2DArray, is2DArray
from . import dataHelpers
from .base import Base
from .base_view import BaseView
from .sparsePoints import SparsePoints, SparsePointsView
from .sparseFeatures import SparseFeatures, SparseFeaturesView
from .sparseElements import SparseElements, SparseElementsView
from .dataHelpers import DEFAULT_PREFIX
from .dataHelpers import allDataIdentical
from .dataHelpers import createDataNoValidation

scipy = nimble.importModule('scipy')
if scipy is not None:
    from scipy.sparse import coo_matrix
    from scipy.io import mmwrite

pd = nimble.importModule('pandas')

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
    elementType : type
        The scipy or numpy dtype of the data.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, data, reuseData=False, elementType=None, **kwds):
        if not scipy:
            msg = 'To use class Sparse, scipy must be installed.'
            raise PackageException(msg)

        if not is2DArray(data) and not scipy.sparse.isspmatrix(data):
            msg = "the input data can only be a scipy sparse matrix or a "
            msg += "two-dimensional numpy array"
            raise InvalidArgumentType(msg)

        if scipy.sparse.isspmatrix_coo(data):
            if reuseData:
                self.data = data
            else:
                self.data = data.copy()
        elif scipy.sparse.isspmatrix(data):
            #data is a spmatrix in other format instead of coo
            self.data = data.tocoo()
        else:#data is numpy.array
            self.data = scipy.sparse.coo_matrix(data)

        self._sorted = None
        kwds['shape'] = self.data.shape
        super(Sparse, self).__init__(**kwds)

    def _getPoints(self):
        return SparsePoints(self)

    def _getFeatures(self):
        return SparseFeatures(self)

    def _getElements(self):
        return SparseElements(self)

    def plot(self, outPath=None, includeColorbar=False):
        toPlot = self.copy(to="Matrix")
        toPlot.plot(outPath, includeColorbar)

    def _plot(self, outPath=None, includeColorbar=False):
        toPlot = self.copy(to="Matrix")
        return toPlot._plot(outPath, includeColorbar)

    def _transpose_implementation(self):
        self.data = self.data.transpose()
        self._sorted = None
        #_resync(self.data)

    #		if self._sorted == 'point':
    #			self._sorted = 'feature'
    #		elif self._sorted == 'feature':
    #			self._sorted = 'point'

    def _isIdentical_implementation(self, other):
        if not isinstance(other, Sparse):
            return False
        # for nonempty matrices, we use a shape mismatch to indicate
        # non-equality
        if self.data.shape != other.data.shape:
            return False

        if isinstance(other, SparseView):
            return other._isIdentical_implementation(self)
        else:
            #let's do internal sort first then compare
            self._sortInternal('feature')
            other._sortInternal('feature')

            return (allDataIdentical(self.data.data, other.data.data)
                    and allDataIdentical(self.data.row, other.data.row)
                    and allDataIdentical(self.data.col, other.data.col))

    def _getTypeString_implementation(self):
        return 'Sparse'

    def _writeFileCSV_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
        """
        Function to write the data in this object to a CSV file at the
        designated path.
        """
        outFile = open(outPath, 'w')

        if includeFeatureNames:
            def combine(a, b):
                return a + ',' + b

            fnames = self.features.getNames()
            fnamesLine = reduce(combine, fnames)
            fnamesLine += '\n'
            if includePointNames:
                outFile.write('pointNames,')

            outFile.write(fnamesLine)

        # sort by rows first, then columns
        placement = numpy.lexsort((self.data.col, self.data.row))
        self.data.data[placement]
        self.data.row[placement]
        self.data.col[placement]

        pointer = 0
        pmax = len(self.data.data)
        for i in range(len(self.points)):
            if includePointNames:
                currPname = self.points.getName(i)
                outFile.write(currPname)
                outFile.write(',')
            for j in range(len(self.features)):
                if (pointer < pmax and i == self.data.row[pointer]
                        and j == self.data.col[pointer]):
                    value = self.data.data[pointer]
                    pointer = pointer + 1
                else:
                    value = 0

                if j != 0:
                    outFile.write(',')
                outFile.write(str(value))
            outFile.write('\n')

        outFile.close()

    def _writeFile_implementation(self, outPath, fileFormat, includePointNames,
                                  includeFeatureNames):
        """
        Function to write the data in this object to a file using the
        specified format. outPath is the location (including file name
        and extension) where we want to write the output file.
        ``includeNames`` is boolean argument indicating whether the file
        should start with comment lines designating pointNames and
        featureNames.
        """
        # if format not in ['csv', 'mtx']:
        #     msg = "Unrecognized file format. Accepted types are 'csv' and "
        #     msg += "'mtx'. They may either be input as the format parameter, "
        #     msg += "or as the extension in the outPath"
        #     raise InvalidArgumentValue(msg)

        if fileFormat == 'csv':
            return self._writeFileCSV_implementation(
                outPath, includePointNames, includeFeatureNames)
        if fileFormat == 'mtx':
            return self._writeFileMTX_implementation(
                outPath, includePointNames, includeFeatureNames)

    def _writeFileMTX_implementation(self, outPath, includePointNames,
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

        if header != '':
            mmwrite(target=outPath, a=self.data, comment=header)
        else:
            mmwrite(target=outPath, a=self.data)

    def _referenceDataFrom_implementation(self, other):
        if not isinstance(other, Sparse):
            msg = "Other must be the same type as this object"
            raise InvalidArgumentType(msg)

        self.data = other.data
        self._sorted = other._sorted

    def _copy_implementation(self, to):
        if to in nimble.data.available:
            ptNames = self.points._getNamesNoGeneration()
            ftNames = self.features._getNamesNoGeneration()
            if to == 'Sparse':
                try:
                    data = self.data.copy().astype(numpy.float)
                except ValueError:
                    data = self.data.copy()
            else:
                data = self.data.todense()
            # reuseData=True since we already made copies here
            return createDataNoValidation(to, data, ptNames, ftNames,
                                          reuseData=True)
        if to == 'pythonlist':
            return self.data.todense().tolist()
        if to == 'numpyarray':
            return self.data.todense()
        if to == 'numpymatrix':
            return numpy.matrix(self.data.todense())
        if 'scipy' in to:
            if to == 'scipycsc':
                return self.data.tocsc()
            if to == 'scipycsr':
                return self.data.tocsr()
            if to == 'scipycoo':
                return self.data.copy()
        if to == 'pandasdataframe':
            if not pd:
                msg = "pandas is not available"
                raise PackageException(msg)
            return pd.DataFrame(self.data.todense())

    def _fillWith_implementation(self, values, pointStart, featureStart,
                                 pointEnd, featureEnd):
        # sort values or call helper as needed
        constant = not isinstance(values, Base)
        if constant:
            if values == 0:
                self._fillWith_zeros_implementation(pointStart, featureStart,
                                                    pointEnd, featureEnd)
                return
        else:
            values._sortInternal('point')

        # this has to be after the possible call to
        # _fillWith_zeros_implementation; it is uncessary for that helper
        self._sortInternal('point')

        self_i = 0
        vals_i = 0
        copyIndex = 0
        toAddData = []
        toAddRow = []
        toAddCol = []
        selfEnd = numpy.searchsorted(self.data.row, pointEnd, 'right')
        if constant:
            valsEnd = ((pointEnd - pointStart + 1)
                       * (featureEnd - featureStart + 1))
        else:
            valsEnd = len(values.data.data)

        # Adjust self_i so that it begins at the values that might need to be
        # replaced, or, if no such values exist, set self_i such that the main
        # loop will ignore the contents of self.
        if len(self.data.data) > 0:
            self_i = numpy.searchsorted(self.data.row, pointStart, 'left')

            pcheck = self.data.row[self_i]
            fcheck = self.data.col[self_i]
            # the condition in the while loop is a natural break, if it isn't
            # satisfied then self_i will be exactly where we want it
            while fcheck < featureStart or fcheck > featureEnd:
                # this condition is an unatural break, when it is satisfied,
                # that means no value of self_i will point into the desired
                # values
                if pcheck > pointEnd or self_i == len(self.data.data) - 1:
                    self_i = selfEnd
                    break

                self_i += 1
                pcheck = self.data.row[self_i]
                fcheck = self.data.col[self_i]

            copyIndex = self_i

        # Walk full contents of both, modifying, shifing, or setting aside
        # values as needed. We will only ever increment one of self_i or vals_i
        # at a time, meaning if there are matching entries, we will encounter
        # them. Due to the sorted precondition, if the location in one object
        # is less than the location in the other object, the lower one CANNOT
        # have a match.
        while self_i < selfEnd or vals_i < valsEnd:
            if self_i < selfEnd:
                locationSP = self.data.row[self_i]
                locationSF = self.data.col[self_i]
            else:
                # we want to use unreachable values as sentials, so we + 1
                # since we're using inclusive endpoints
                locationSP = pointEnd + 1
                locationSF = featureEnd + 1

            # adjust 'values' locations into the scale of the calling object
            locationVP = pointStart
            locationVF = featureStart
            if constant:
                vData = values
                # uses truncation of int division
                locationVP += vals_i / (featureEnd - featureStart + 1)
                locationVF += vals_i % (featureEnd - featureStart + 1)
            elif vals_i >= valsEnd:
                locationVP += pointEnd + 1
                locationVF += featureEnd + 1
            else:
                vData = values.data.data[vals_i]
                locationVP += values.data.row[vals_i]
                locationVF += values.data.col[vals_i]

            pCmp = locationSP - locationVP
            fCmp = locationSF - locationVF
            trueCmp = pCmp if pCmp != 0 else fCmp

            # Case: location at index into self is higher than location
            # at index into values. No matching entry in self;
            # copy if space, or record to be added at end.
            if trueCmp > 0:
                # can only copy into self if there is open space
                if copyIndex < self_i:
                    self.data.data[copyIndex] = vData
                    self.data.row[copyIndex] = locationVP
                    self.data.col[copyIndex] = locationVF
                    copyIndex += 1
                else:
                    toAddData.append(vData)
                    toAddRow.append(locationVP)
                    toAddCol.append(locationVF)

                #increment vals_i
                vals_i += 1
            # Case: location at index into other is higher than location at
            # index into self. no matching entry in values - fill this entry
            # in self with zero (by shifting past it)
            elif trueCmp < 0:
                # need to do cleanup if we're outside of the relevant bounds
                if locationSF < featureStart or locationSF > featureEnd:
                    self.data.data[copyIndex] = self.data.data[self_i]
                    self.data.row[copyIndex] = self.data.row[self_i]
                    self.data.col[copyIndex] = self.data.col[self_i]
                    copyIndex += 1
                self_i += 1
            # Case: indices point to equal locations.
            else:
                self.data.data[copyIndex] = vData
                self.data.row[copyIndex] = locationVP
                self.data.col[copyIndex] = locationVF
                copyIndex += 1

                # increment both??? or just one?
                self_i += 1
                vals_i += 1

        # Now we have to walk through the rest of self, finishing the copying
        # shift if necessary
        if copyIndex != self_i:
            while self_i < len(self.data.data):
                self.data.data[copyIndex] = self.data.data[self_i]
                self.data.row[copyIndex] = self.data.row[self_i]
                self.data.col[copyIndex] = self.data.col[self_i]
                self_i += 1
                copyIndex += 1
        else:
            copyIndex = len(self.data.data)

        newData = numpy.empty(copyIndex + len(toAddData),
                              dtype=self.data.data.dtype)
        newData[:copyIndex] = self.data.data[:copyIndex]
        newData[copyIndex:] = toAddData
        newRow = numpy.empty(copyIndex + len(toAddRow))
        newRow[:copyIndex] = self.data.row[:copyIndex]
        newRow[copyIndex:] = toAddRow
        newCol = numpy.empty(copyIndex + len(toAddCol))
        newCol[:copyIndex] = self.data.col[:copyIndex]
        newCol[copyIndex:] = toAddCol
        shape = (len(self.points), len(self.features))
        self.data = scipy.sparse.coo_matrix((newData, (newRow, newCol)), shape)

        self._sorted = None

    def _flattenToOnePoint_implementation(self):
        self._sortInternal('point')
        pLen = len(self.features)
        numElem = len(self.points) * len(self.features)
        for i in range(len(self.data.data)):
            if self.data.row[i] > 0:
                self.data.col[i] += (self.data.row[i] * pLen)
                self.data.row[i] = 0

        data = self.data.data
        row = self.data.row
        col = self.data.col
        self.data = coo_matrix((data, (row, col)), (1, numElem))

    def _flattenToOneFeature_implementation(self):
        self._sortInternal('feature')
        fLen = len(self.points)
        numElem = len(self.points) * len(self.features)
        for i in range(len(self.data.data)):
            if self.data.col[i] > 0:
                self.data.row[i] += (self.data.col[i] * fLen)
                self.data.col[i] = 0

        data = self.data.data
        row = self.data.row
        col = self.data.col
        self.data = coo_matrix((data, (row, col)), (numElem, 1))

    def _unflattenFromOnePoint_implementation(self, numPoints):
        # only one feature, so both sorts are the same order
        if self._sorted is None:
            self._sortInternal('point')

        numFeatures = len(self.features) // numPoints
        newShape = (numPoints, numFeatures)

        for i in range(len(self.data.data)):
            # must change the row entry before modifying the col entry
            self.data.row[i] = self.data.col[i] / numFeatures
            self.data.col[i] = self.data.col[i] % numFeatures

        data = self.data.data
        row = self.data.row
        col = self.data.col
        self.data = coo_matrix((data, (row, col)), newShape)

    def _unflattenFromOneFeature_implementation(self, numFeatures):
        # only one feature, so both sorts are the same order
        if self._sorted is None:
            self._sortInternal('feature')

        numPoints = len(self.points) // numFeatures
        newShape = (numPoints, numFeatures)

        for i in range(len(self.data.data)):
            # must change the col entry before modifying the row entry
            self.data.col[i] = self.data.row[i] / numPoints
            self.data.row[i] = self.data.row[i] % numPoints

        data = self.data.data
        row = self.data.row
        col = self.data.col
        self.data = coo_matrix((data, (row, col)), newShape)

    def _mergeIntoNewData(self, copyIndex, toAddData, toAddRow, toAddCol):
        #instead of always copying, use reshape or resize to sometimes cut
        # array down to size???
        pass

    def _fillWith_zeros_implementation(self, pointStart, featureStart,
                                       pointEnd, featureEnd):
        # walk through col listing and partition all data: extract, and kept,
        # reusing the sparse matrix underlying structure to save space
        copyIndex = 0

        for lookIndex in range(len(self.data.data)):
            currP = self.data.row[lookIndex]
            currF = self.data.col[lookIndex]
            # if it is in range we want to obliterate the entry by just passing
            # it by and copying over it later
            if (currP >= pointStart
                    and currP <= pointEnd
                    and currF >= featureStart
                    and currF <= featureEnd):
                pass
            else:
                self.data.data[copyIndex] = self.data.data[lookIndex]
                self.data.row[copyIndex] = self.data.row[lookIndex]
                self.data.col[copyIndex] = self.data.col[lookIndex]
                copyIndex += 1

        # reinstantiate self
        # (cannot reshape coo matrices, so cannot do this in place)
        newData = self.data.data[0:copyIndex]
        newRow = self.data.row[0:copyIndex]
        newCol = self.data.col[0:copyIndex]
        shape = (len(self.points), len(self.features))
        self.data = scipy.sparse.coo_matrix((newData, (newRow, newCol)), shape)

    def _binarySearch(self, x, y):
        if self._sorted == 'point':
            axis = self.data.row
            offAxis = self.data.col
            axisVal = x
            offAxisVal = y
        elif self._sorted == 'feature':
            axis = self.data.col
            offAxis = self.data.row
            axisVal = y
            offAxisVal = x
        else:
            msg = 'self._sorted is not either point nor feature.'
            raise ImproperObjectAction(msg)
        #binary search
        start, end = numpy.searchsorted(axis, [axisVal, axisVal+1])
        if start == end: # axisVal is not in self.data.row
            if numpy.issubdtype(self.data.dtype, numpy.bool_):
                return False
            return 0
        k = numpy.searchsorted(offAxis[start:end], offAxisVal) + start
        if k < end and offAxis[k] == offAxisVal:
            return self.data.data[k]
        if numpy.issubdtype(self.data.dtype, numpy.bool_):
            return False
        return 0

    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):
        if self._sorted != 'feature':
            self._sortInternal('feature')
        if other._sorted != 'feature':
            other._sortInternal('feature')
        leftFtCount = len(self.features)
        rightFtCount = len(other.features) - len(matchingFtIdx[0])
        if onFeature:
            onIdxL = self.features.getIndex(onFeature)
            onIdxR = other.features.getIndex(onFeature)
            leftData = self.data.data
            leftRow = self.data.row
            leftCol = self.data.col
            rightData = other.data.data.copy()
            rightRow = other.data.row.copy()
            rightCol = other.data.col.copy()
        else:
            onIdxL = 0
            onIdxR = 0
            leftData = self.data.data.astype(numpy.object_)
            if not self._anyDefaultPointNames():
                leftData = numpy.append([self.points.getNames()], leftData)
            elif self._pointNamesCreated():
                # differentiate default names between objects;
                # note still start with DEFAULT_PREFIX
                leftNames = [n + '_l' if n.startswith(DEFAULT_PREFIX) else n
                             for n in self.points.getNames()]
                leftData = numpy.append([leftNames], leftData)
            else:
                leftNames = [DEFAULT_PREFIX + str(i) for i
                             in range(len(self.points))]
                leftData = numpy.append(leftNames, leftData)
            leftRow = numpy.append([i for i in range(len(self.points))],
                                   self.data.row)
            leftCol = numpy.append([0 for _ in range(len(self.points))],
                                   self.data.col + 1)
            rightData = other.data.data.copy().astype(numpy.object_)
            if not other._anyDefaultPointNames():
                rightData = numpy.append([other.points.getNames()], rightData)
            elif other._pointNamesCreated():
                # differentiate default names between objects;
                # note still start with DEFAULT_PREFIX
                rightNames = [n + '_r' if n.startswith(DEFAULT_PREFIX) else n
                              for n in other.points.getNames()]
                rightData = numpy.append([rightNames], rightData)
            else:
                rtRange = range(self.shape[0], self.shape[0] + other.shape[0])
                rightNames = [DEFAULT_PREFIX + str(i) for i in rtRange]
                rightData = numpy.append(rightNames, rightData)
            rightRow = numpy.append([i for i in range(len(other.points))],
                                    other.data.row.copy())
            rightCol = numpy.append([0 for i in range(len(other.points))],
                                    other.data.col.copy() + 1)
            matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
            matchingFtIdx[0].insert(0, 0)
            matchingFtIdx[1] = list(map(lambda x: x + 1, matchingFtIdx[1]))
            matchingFtIdx[1].insert(0, 0)

        mergedData = numpy.empty((0, 0), dtype=numpy.object_)
        mergedRow = []
        mergedCol = []
        matched = []
        nextPt = 0
        numPts = 0

        for ptIdxL, target in enumerate(leftData[leftCol == onIdxL]):
            rowIdxR = numpy.where(rightData[rightCol == onIdxR] == target)[0]
            if len(rowIdxR) > 0:
                for ptIdxR in rowIdxR:
                    ptL = leftData[leftRow == ptIdxL]
                    ptR = rightData[rightRow == ptIdxR]
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
                    pt = numpy.append(ptL, ptR)
                    if feature == "intersection":
                        pt = pt[matchingFtIdx[0]]
                        leftFtCount = len(matchingFtIdx[0])
                    elif onFeature and feature == "left":
                        pt = pt[:len(self.features)]
                    elif feature == "left":
                        pt = pt[:len(self.features) + 1]
                    if onFeature is None:
                        pt = pt[1:]
                    matched.append(target)
                    mergedData = numpy.append(mergedData, pt)
                    mergedRow.extend([nextPt] * len(pt))
                    mergedCol.extend([i for i in range(len(pt))])
                    nextPt += 1
                    numPts += 1
            elif point in ["union", "left"]:
                ptL = leftData[leftRow == ptIdxL]
                if onFeature:
                    numNaN = len(other.features) - len(matchingFtIdx[1])
                    ptR = [numpy.nan] * numNaN
                else:
                    numNaN = len(other.features) - len(matchingFtIdx[1]) + 1
                    ptR = [numpy.nan] * numNaN
                pt = numpy.append(ptL, ptR)
                if feature == "intersection":
                    pt = pt[matchingFtIdx[0]]
                elif onFeature and feature == "left":
                    pt = pt[:len(self.features)]
                elif feature == "left":
                    pt = pt[:len(self.features) + 1]
                if onFeature is None:
                    pt = pt[1:]
                mergedData = numpy.append(mergedData, pt)
                mergedRow.extend([nextPt] * len(pt))
                mergedCol.extend([i for i in range(len(pt))])
                nextPt += 1
                numPts += 1

        if point == 'union':
            for ptIdxR, target in enumerate(rightData[rightCol == onIdxR]):
                if target not in matched:
                    if onFeature:
                        nanList = [numpy.nan] * len(self.features)
                        ptL = numpy.array(nanList, dtype=numpy.object_)
                    else:
                        nanList = [numpy.nan] * (len(self.features) + 1)
                        ptL = numpy.array(nanList, dtype=numpy.object_)
                    fill = rightData[(rightRow == ptIdxR)][matchingFtIdx[1]]
                    ptL[matchingFtIdx[0]] = fill
                    # only unmatched points from right
                    notMatchedR = numpy.in1d(rightCol, matchingFtIdx[1],
                                             invert=True)
                    ptR = rightData[(rightRow == ptIdxR) & (notMatchedR)]
                    pt = numpy.append(ptL, ptR)
                    if feature == "intersection":
                        pt = pt[matchingFtIdx[0]]
                    elif onFeature and feature == "left":
                        pt = pt[:len(self.features)]
                    elif feature == "left":
                        pt = pt[:len(self.features) + 1]
                    if onFeature is None:
                        # remove pointNames column added
                        pt = pt[1:]
                    mergedData = numpy.append(mergedData, pt)
                    mergedRow.extend([nextPt] * len(pt))
                    mergedCol.extend([i for i in range(len(pt))])
                    nextPt += 1
                    numPts += 1

        numFts = leftFtCount + rightFtCount
        if onFeature and feature == "intersection":
            numFts = len(matchingFtIdx[0])
        elif feature == "intersection":
            numFts = len(matchingFtIdx[0]) - 1
        elif feature == "left":
            numFts = len(self.features)
        if len(mergedData) == 0:
            mergedData = []

        self._featureCount = numFts
        self._pointCount = numPts

        self.data = coo_matrix((mergedData, (mergedRow, mergedCol)),
                               shape=(numPts, numFts))

        self._sorted = None

    def _replaceFeatureWithBinaryFeatures_implementation(self, uniqueVals):
        if self._sorted is None:
            self._sortInternal('feature')
        binaryRow = []
        binaryCol = []
        binaryData = []
        for ptIdx, val in zip(self.data.row, self.data.data):
            ftIdx = uniqueVals.index(val)
            binaryRow.append(ptIdx)
            binaryCol.append(ftIdx)
            binaryData.append(1)
        binaryCoo = coo_matrix((binaryData, (binaryRow, binaryCol)),
                               shape=(len(self.points), len(uniqueVals)))
        self._sorted = None
        return Sparse(binaryCoo)

    def _getitem_implementation(self, x, y):
        """
        currently, we sort the data first and then do binary search
        """

        if self._sorted is None:
            self._sortInternal('point')

        return self._binarySearch(x, y)

    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd):
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
        kwds['reuseData'] = True

        allPoints = pointStart == 0 and pointEnd == len(self.points)
        singlePoint = pointEnd - pointStart == 1
        allFeats = featureStart == 0 and featureEnd == len(self.features)
        singleFeat = featureEnd - featureStart == 1
        # singleFeat = singlePoint = False
        if singleFeat or singlePoint:
            if singlePoint:
                if self._sorted is None or self._sorted == 'feature':
                    self._sortInternal('point')
                sortedIndices = self.data.row

                start = numpy.searchsorted(sortedIndices, pointStart, 'left')
                end = numpy.searchsorted(sortedIndices, pointEnd - 1, 'right')

                if not allFeats:
                    sortedIndices = self.data.col[start:end]
                    innerStart = numpy.searchsorted(sortedIndices,
                                                    featureStart, 'left')
                    innerEnd = numpy.searchsorted(sortedIndices,
                                                  featureEnd - 1, 'right')
                    outerStart = start
                    start = start + innerStart
                    end = outerStart + innerEnd

                row = numpy.tile([0], end - start)
                col = self.data.col[start:end] - featureStart

            else:  # case single feature
                if self._sorted is None or self._sorted == 'point':
                    self._sortInternal('feature')
                sortedIndices = self.data.col

                start = numpy.searchsorted(sortedIndices, featureStart, 'left')
                end = numpy.searchsorted(sortedIndices,
                                         featureEnd - 1, 'right')

                if not allPoints:
                    sortedIndices = self.data.row[start:end]
                    innerStart = numpy.searchsorted(sortedIndices,
                                                    pointStart, 'left')
                    innerEnd = numpy.searchsorted(sortedIndices,
                                                  pointEnd - 1, 'right')
                    outerStart = start
                    start = start + innerStart
                    end = outerStart + innerEnd

                row = self.data.row[start:end] - pointStart
                col = numpy.tile([0], end - start)

            data = self.data.data[start:end]
            pshape = pointEnd - pointStart
            fshape = featureEnd - featureStart

            newInternal = scipy.sparse.coo_matrix((data, (row, col)),
                                                  shape=(pshape, fshape))
            kwds['data'] = newInternal

            return SparseVectorView(**kwds)

        else:  # window shaped View
            # the data should be dummy data, but data.shape must
            # be = (pointEnd - pointStart, featureEnd - featureStart)
            newInternal = scipy.sparse.coo_matrix([])
            newInternal._shape = (pointEnd - pointStart,
                                  featureEnd - featureStart)
            newInternal.data = None
            kwds['data'] = newInternal

            return SparseView(**kwds)

    def _validate_implementation(self, level):
        assert self.data.shape[0] == len(self.points)
        assert self.data.shape[1] == len(self.features)
        assert scipy.sparse.isspmatrix_coo(self.data)

        if level > 0:
            try:
                noZerosInData = all(self.data.data != 0)
                #numpy may say: elementwise comparison failed; returning
                # scalar instead, but in the future will perform
                # elementwise comparison
            except ValueError:
                noZerosInData = all(i != 0 for i in self.data.data)
            assert noZerosInData

            assert self.data.dtype.type is not numpy.string_

            row = self.data.row
            col = self.data.col
            if self._sorted == 'point' or self._sorted == 'feature':
                sortedAxis = self._sorted
                self._sorted = None
                self._sortInternal(sortedAxis)
            assert all(self.data.row[:] == row[:]) # _sortInternal incorrect
            assert all(self.data.col[:] == col[:]) # _sortInternal incorrect

            without_replicas_coo = removeDuplicatesNative(self.data)
            assert len(self.data.data) == len(without_replicas_coo.data)

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0
        contained in this object. False otherwise
        """
        return (self.data.shape[0] * self.data.shape[1]) > self.data.nnz


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
                if isinstance(other, Sparse):
                    otherData = other._getSparseData()
                else:
                    otherConv = other.copy('Matrix')
                    otherData = otherConv.data
                ret = getattr(selfData, opName)(otherData)
            else:
                return self._scalarBinary_implementation(opName, other)

            if ret is NotImplemented:
                # most NotImplemented are inplace operations
                if opName.startswith('__i'):
                    return self._inplaceBinary_implementation(opName, other)
                elif opName == '__rsub__':
                    return self._rsub__implementation(other)
                return self._defaultBinaryOperations_implementation(opName,
                                                                     other)

            return Sparse(ret)

        except AttributeError as ae:
            if opName.startswith('__i'):
                return self._inplaceBinary_implementation(opName, other)
            if 'floordiv' in opName:
                return self._genericFloordiv_implementation(opName, other)
            if 'mod' in opName:
                return self._genericMod_implementation(opName, other)
            return self._defaultBinaryOperations_implementation(opName, other)


    def _scalarBinary_implementation(self, opName, other):
        oneSafe = ['__truediv__', '__itruediv__', 'mul', '__pow__', '__ipow__']
        if any(name in opName for name in oneSafe) and other == 1:
            selfData = self._getSparseData()
            return Sparse(selfData)
        zeroSafe = ['mul', 'truediv', 'floordiv', 'mod']
        zeroPreserved = any(name in opName for name in zeroSafe)
        if 'pow' in opName and opName != '__rpow__' and other != 0:
            zeroPreserved = True
        if zeroPreserved:
            return self._scalarZeroPreservingBinary_implementation(
                opName, other)
        else:
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
            retData = self.data * retData
        else:
            # for other.data as any dense or sparse matrix
            retData = self.data * other.data

        return nimble.createData('Sparse', retData, useLog=False)

    def _inplaceBinary_implementation(self, opName, other):
        notInplace = '__' + opName[3:]
        ret = self._binaryOperations_implementation(notInplace, other)
        absPath, relPath = self._absPath, self._relPath
        self.referenceDataFrom(ret, useLog=False)
        self._absPath, self._relPath = absPath, relPath
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

        target.elements.multiply(other, useLog=False)

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

        caller.elements.power(callee, useLog=False)
        return caller

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
        ret.data.data = numpy.floor(ret.data.data)
        ret.data.eliminate_zeros()
        return ret

    def _genericMod_implementation(self, opName, other):
        """
        Uses numpy.mod on the nonzero values in the coo matrix.

        Since 0 % any value is 0, the zero values can be ignored for
        this operation.
        """
        selfData = self._getSparseData()
        if isinstance(other, Sparse):
            otherData = other._getSparseData().data
        else: # another Base object type
            otherData = other.copy('Sparse').data.data

        if opName == '__rmod__':
            ret = numpy.mod(otherData, selfData.data)
        else:
            ret = numpy.mod(selfData.data, otherData)
        coo = coo_matrix((ret, (selfData.row, selfData.col)),
                         shape=self.shape)
        coo.eliminate_zeros() # remove any zeros introduced into data
        if opName.startswith('__i'):
            self.data = coo
            return self
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
        coo = coo_matrix((ret, (selfData.row, selfData.col)),
                         shape=self.shape)
        coo.eliminate_zeros() # remove any zeros introduced into data
        if opName.startswith('__i'):
            self.data = coo
            return self
        return Sparse(coo)


    ###########
    # Helpers #
    ###########

    def _sortInternal(self, axis):
        if axis != 'point' and axis != 'feature':
            raise InvalidArgumentValue("invalid axis type")

        if (self._sorted == axis
                or len(self.points) == 0
                or len(self.features) == 0):
            return

        sortAsParam = 'row-major' if axis == 'point' else 'col-major'
        _sortInternal_coo_matrix(self.data, sortAsParam)

        # flag that we are internally sorted
        self._sorted = axis

    def _getSparseData(self):
        """
        Get the backend coo_matrix data for this object.

        Since Views set self.data.data to None, we need to copy the view
        to gain access to the coo_matrix data.
        """
        if isinstance(self, BaseView):
            selfData = self.copy().data
        else:
            selfData = self.data
        return selfData

###################
# Generic Helpers #
###################

def _sortInternal_coo_matrix(obj, sortAs):
    if sortAs != 'row-major' and sortAs != 'col-major':
        raise InvalidArgumentValue("invalid axis type")

    # sort least significant axis first
    if sortAs == "row-major":
        sortPrime = obj.row
        sortOff = obj.col
    else:
        sortPrime = obj.col
        sortOff = obj.row

    sortKeys = numpy.lexsort((sortOff, sortPrime))

    obj.data = obj.data[sortKeys]
    obj.row = obj.row[sortKeys]
    obj.col = obj.col[sortKeys]

    # newData = obj.data[sortKeys]
    # newRow = obj.row[sortKeys]
    # newCol = obj.col[sortKeys]
    #
    # n = len(newData)
    # obj.data[:n] = newData
    # obj.row[:n] = newRow
    # obj.col[:n] = newCol

def _numLessThan(value, toCheck): # TODO caching
    ltCount = 0
    for i in range(len(toCheck)):
        if toCheck[i] < value:
            ltCount += 1

    return ltCount

def _resync(obj):
    if 0 in obj.shape:
        obj.nnz = 0
        obj.data = numpy.array([])
        obj.row = numpy.array([])
        obj.col = numpy.array([])
        obj.shape = obj.shape
    else:
        obj.nnz = obj.nnz
        obj.data = obj.data
        obj.row = obj.row
        obj.col = obj.col
        obj.shape = obj.shape

def removeDuplicatesNative(coo_obj):
    """
    Creates a new coo_matrix, using summation for numeric data to remove
    duplicates. If there are duplicate entires involving non-numeric
    data, an exception is raised.

    coo_obj : the coo_matrix from which the data of the return object
    originates from. It will not be modified by the function.

    Returns : a new coo_matrix with the same data as in the input
    matrix, except with duplicate numerical entries summed to a single
    value. This operation is NOT stable - the row / col attributes are
    not guaranteed to have an ordering related to those from the input
    object. This operation is guaranteed to not introduce any 0 values
    to the data attribute.
    """
    if coo_obj.data is None:
        #When coo_obj data is not iterable: Empty
        #It will throw TypeError: zip argument #3 must support iteration.
        #Decided just to do this quick check instead of duck typing.
        return coo_obj

    seen = {}
    for i, j, v in zip(coo_obj.row, coo_obj.col, coo_obj.data):
        if (i, j) not in seen:
            # all types are allows to be present once
            seen[(i, j)] = v
        else:
            try:
                seen[(i, j)] += float(v)
            except ValueError:
                msg = 'Unable to represent this configuration of data in '
                msg += 'Sparse object. At least one removeDuplicatesNativeof '
                msg += 'the duplicate entries is a non-numerical type'
                raise ValueError(msg)

    rows = []
    cols = []
    data = []

    for (i, j) in seen:
        if seen[(i, j)] != 0:
            rows.append(i)
            cols.append(j)
            data.append(seen[(i, j)])

    dataNP = numpy.array(data)
    # if there are mixed strings and numeric values numpy will automatically
    # turn everything into strings. This will check to see if that has
    # happened and use the object dtype instead.
    if len(dataNP) > 0 and isinstance(dataNP[0], numpy.flexible):
        dataNP = numpy.array(data, dtype='O')
    new_coo = coo_matrix((dataNP, (rows, cols)), shape=coo_obj.shape)

    return new_coo

def removeDuplicatesByConversion(coo_obj):
    try:
        return coo_obj.tocsr().tocoo()
        # return coo_obj.tocsc().tocoo()
    except TypeError:
        msg = 'Unable to represent this configuration of data in a '
        msg += 'Sparse object.'
        raise TypeError(msg)

class SparseVectorView(BaseView, Sparse):
    """
    A view of a Sparse data object limited to a full point or full
    feature.
    """

    def __init__(self, **kwds):
        super(SparseVectorView, self).__init__(**kwds)

    def _getPoints(self):
        return SparsePointsView(self)

    def _getFeatures(self):
        return SparseFeaturesView(self)

    def _getElements(self):
        return SparseElementsView(self)

class SparseView(BaseView, Sparse):
    """
    Read only access to a Sparse object.
    """
    def __init__(self, **kwds):
        super(SparseView, self).__init__(**kwds)

    def _getPoints(self):
        return SparsePointsView(self)

    def _getFeatures(self):
        return SparseFeaturesView(self)

    def _getElements(self):
        return SparseElementsView(self)

    def _validate_implementation(self, level):
        self._source.validate(level)

    def _getitem_implementation(self, x, y):
        adjX = x + self._pStart
        adjY = y + self._fStart
        return self._source[adjX, adjY]

    def _copy_implementation(self, to):
        if to == "Sparse":
            sourceData = self._source.data.data.copy()
            sourceRow = self._source.data.row.copy()
            sourceCol = self._source.data.col.copy()

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

            coo = coo_matrix((keepData, (keepRow, keepCol)),
                             shape=(len(self.points), len(self.features)))
            pNames = None
            fNames = None
            if self._pointNamesCreated():
                pNames = self.points.getNames()
            if self._featureNamesCreated():
                fNames = self.features.getNames()
            return Sparse(coo, pointNames=pNames, featureNames=fNames)

        if len(self.points) == 0 or len(self.features) == 0:
            emptyStandin = numpy.empty((len(self.points), len(self.features)))
            intermediate = nimble.createData('Matrix', emptyStandin,
                                             useLog=False)
            return intermediate.copy(to=to)

        if to == 'numpyarray':
            pStart, pEnd = self._pStart, self._pEnd
            fStart, fEnd = self._fStart, self._fEnd
            limited = self._source.data.todense()[pStart:pEnd, fStart:fEnd]
            return numpy.array(limited)

        limited = self._source.points.copy(start=self._pStart,
                                           end=self._pEnd - 1, useLog=False)
        limited = limited.features.copy(start=self._fStart,
                                        end=self._fEnd - 1, useLog=False)

        return limited._copy_implementation(to)

    def _isIdentical_implementation(self, other):
        if not isinstance(other, Sparse):
            return False
        # for nonempty matrices, we use a shape mismatch to indicate
        # non-equality
        if self.data.shape != other.data.shape:
            return False

        # empty object means no values. Since shapes match they're equal
        if self.data.shape[0] == 0 or self.data.shape[1] == 0:
            return True

        sIt = self.points
        oIt = other.points
        for sPoint, oPoint in zip(sIt, oIt):
            for i, sVal in enumerate(sPoint):
                oVal = oPoint[i]
                # check element equality - which is only relevant if one of
                # the elements is non-NaN
                if sVal != oVal and (sVal == sVal or oVal == oVal):
                    return False
        return True

    def _containsZero_implementation(self):
        for sPoint in self.points:
            if sPoint.containsZero():
                return True

        return False

    def __abs__(self):
        """ Perform element wise absolute value on this object """
        ret = self.copy(to="Sparse")
        numpy.absolute(ret.data.data, out=ret.data.data)
        ret._name = dataHelpers.nextDefaultObjectName()

        return ret

    def _matmul__implementation(self, other):
        selfConv = self.copy(to="Sparse")
        if isinstance(other, BaseView):
            other = other.copy(to=other.getTypeString())
        return selfConv._matmul__implementation(other)
