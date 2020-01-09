"""
Class extending Base, using a list of lists to store data.
"""

import copy
from functools import reduce

import numpy

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble.utility import inheritDocstringsFactory, numpy2DArray, is2DArray
from nimble.utility import ImportModule
from .base import Base
from .base_view import BaseView
from .listPoints import ListPoints, ListPointsView
from .listFeatures import ListFeatures, ListFeaturesView
from .listElements import ListElements, ListElementsView
from .dataHelpers import DEFAULT_PREFIX
from .dataHelpers import isAllowedSingleElement
from .dataHelpers import createDataNoValidation
from .dataHelpers import csvCommaFormat

scipy = ImportModule('scipy')
pd = ImportModule('pandas')

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
                 checkAll=True, elementType=None, **kwds):
        if (not (isinstance(data, list) or is2DArray(data))
                and 'PassThrough' not in str(type(data))):
            msg = "the input data can only be a list, a two-dimensional numpy "
            msg += "array, or ListPassThrough."
            raise InvalidArgumentType(msg)

        if isinstance(data, list):
            #case1: data=[]. self.data will be [], shape will be (0, shape[1])
            # or (0, len(featureNames)) or (0, 0)
            if len(data) == 0:
                if shape:
                    shape = (0, shape[1])
                else:
                    shape = (0, len(featureNames) if featureNames else 0)
            elif isAllowedSingleElement(data[0]):
            #case2: data=['a', 'b', 'c'] or [1,2,3]. self.data will be
            # [[1,2,3]], shape will be (1, 3)
                if checkAll:#check all items
                    for i in data:
                        if not isAllowedSingleElement(i):
                            msg = 'invalid input data format.'
                            raise InvalidArgumentValue(msg)
                shape = (1, len(data))
                data = [data]
            elif isinstance(data[0], list) or hasattr(data[0], 'setLimit'):
            #case3: data=[[1,2,3], ['a', 'b', 'c']] or [[]] or [[], []].
            # self.data will be = data, shape will be (len(data), len(data[0]))
            #case4: data=[<nimble.data.list.FeatureViewer object at 0x43fd410>]
                numFeatures = len(data[0])
                if checkAll:#check all items
                    for i in data:
                        if len(i) != numFeatures:
                            msg = 'invalid input data format.'
                            raise InvalidArgumentValue(msg)
                        for j in i:
                            if not isAllowedSingleElement(j):
                                msg = '%s is invalid input data format.'%j
                                raise InvalidArgumentValue(msg)
                shape = (len(data), numFeatures)

            if reuseData:
                data = data
            else:
                #this is to convert a list x=[[1,2,3]]*2 to a
                # list y=[[1,2,3], [1,2,3]]
                # the difference is that x[0] is x[1], but y[0] is not y[1]
                # Both list and FeatureViewer have a copy method.
                data = [pt.copy() for pt in data]

        if is2DArray(data):
            #case5: data is a numpy array. shape is already in np array
            shape = data.shape
            data = data.tolist()

        if len(data) == 0:
            #case6: data is a ListPassThrough associated with empty list
            data = []

        self._numFeatures = shape[1]
        self.data = data
        self._elementType = elementType

        kwds['featureNames'] = featureNames
        kwds['shape'] = shape
        super(List, self).__init__(**kwds)

    def _getPoints(self):
        return ListPoints(self)

    def _getFeatures(self):
        return ListFeatures(self)

    def _getElements(self):
        return ListElements(self)

    def _transpose_implementation(self):
        """
        Function to transpose the data, ie invert the feature and point
        indices of the data.

        This is not an in place operation, a new list of lists is
        constructed.
        """
        tempFeatures = len(self.data)
        transposed = []
        #load the new data with an empty point for each feature in the original
        for i in range(len(self.features)):
            transposed.append([])
        for point in self.data:
            for i in range(len(point)):
                transposed[i].append(point[i])

        self.data = transposed
        self._numFeatures = tempFeatures

    def _getTypeString_implementation(self):
        return 'List'

    def _isIdentical_implementation(self, other):
        if not isinstance(other, List):
            return False
        if len(self.points) != len(other.points):
            return False
        if len(self.features) != len(other.features):
            return False
        for index in range(len(self.points)):
            sPoint = self.data[index]
            oPoint = other.data[index]
            if sPoint != oPoint:
                return False
        return True

    def _writeFileCSV_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
        """
        Function to write the data in this object to a CSV file at the
        designated path.
        """
        with open(outPath, 'w') as outFile:
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

    def _writeFileMTX_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
        """
        Function to write the data in this object to a matrix market
        file at the designated path.
        """
        with open(outPath, 'w') as outFile:
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

            outFile.write("{0} {1}\n".format(len(self.points), len(self.features)))

            for j in range(len(self.features)):
                for i in range(len(self.points)):
                    value = self.data[i][j]
                    outFile.write(str(value) + '\n')

    def _referenceDataFrom_implementation(self, other):
        if not isinstance(other, List):
            msg = "Other must be the same type as this object"
            raise InvalidArgumentType(msg)

        self.data = other.data
        self._numFeatures = other._numFeatures

    def _copy_implementation(self, to):
        isEmpty = False
        if len(self.points) == 0 or len(self.features) == 0:
            isEmpty = True
            emptyData = numpy.empty(shape=(len(self.points),
                                           len(self.features)))
        elementType = self._elementType
        if elementType is None:
            arr = numpy.array(self.data)
            if issubclass(arr.dtype.type, numpy.number):
                elementType = arr.dtype
            else:
                elementType = numpy.object_
        if to in nimble.data.available:
            ptNames = self.points._getNamesNoGeneration()
            ftNames = self.features._getNamesNoGeneration()
            if isEmpty:
                data = numpy2DArray(emptyData)
            elif to == 'List':
                data = [pt.copy() for pt in self.data]
            else:
                data = numpy2DArray(self.data, dtype=elementType)
            # reuseData=True since we already made copies here
            return createDataNoValidation(to, data, ptNames, ftNames,
                                          reuseData=True)
        if to == 'pythonlist':
            return [pt.copy() for pt in self.data]
        if to == 'numpyarray':
            if isEmpty:
                return emptyData
            return numpy2DArray(self.data, dtype=elementType)
        if to == 'numpymatrix':
            if isEmpty:
                return numpy.matrix(emptyData)
            return numpy.matrix(self.data, dtype=elementType)
        if 'scipy' in to:
            if not scipy:
                msg = "scipy is not available"
                raise PackageException(msg)
            asArray = numpy.array(self.data, dtype=elementType)
            if to == 'scipycsc':
                if isEmpty:
                    return scipy.sparse.csc_matrix(emptyData)
                return scipy.sparse.csc_matrix(asArray)
            if to == 'scipycsr':
                if isEmpty:
                    return scipy.sparse.csr_matrix(emptyData)
                return scipy.sparse.csr_matrix(asArray)
            if to == 'scipycoo':
                if isEmpty:
                    return scipy.sparse.coo_matrix(emptyData)
                return scipy.sparse.coo_matrix(asArray)
        if to == 'pandasdataframe':
            if not pd:
                msg = "pandas is not available"
                raise PackageException(msg)
            if isEmpty:
                return pd.DataFrame(emptyData)
            return pd.DataFrame(self.data, dtype=elementType)

    def _fillWith_implementation(self, values, pointStart, featureStart,
                                 pointEnd, featureEnd):
        if not isinstance(values, Base):
            values = [values] * (featureEnd - featureStart + 1)
            for p in range(pointStart, pointEnd + 1):
                self.data[p][featureStart:featureEnd + 1] = values
        else:
            for p in range(pointStart, pointEnd + 1):
                fill = values.data[p - pointStart]
                self.data[p][featureStart:featureEnd + 1] = fill

    def _flattenToOnePoint_implementation(self):
        onto = self.data[0]
        for _ in range(1, len(self.points)):
            onto += self.data[1]
            del self.data[1]

        self._numFeatures = len(onto)

    def _flattenToOneFeature_implementation(self):
        result = []
        for i in range(len(self.features)):
            for p in self.data:
                result.append([p[i]])

        self.data = result
        self._numFeatures = 1

    def _unflattenFromOnePoint_implementation(self, numPoints):
        result = []
        numFeatures = len(self.features) // numPoints
        for i in range(numPoints):
            temp = self.data[0][(i*numFeatures):((i+1)*numFeatures)]
            result.append(temp)

        self.data = result
        self._numFeatures = numFeatures

    def _unflattenFromOneFeature_implementation(self, numFeatures):
        result = []
        numPoints = len(self.points) // numFeatures
        # reconstruct the shape we want, point by point. We access the
        # singleton values from the current data in an out of order iteration
        for i in range(numPoints):
            temp = []
            for j in range(i, len(self.points), numPoints):
                temp += self.data[j]
            result.append(temp)

        self.data = result
        self._numFeatures = numFeatures

    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):
        if onFeature:
            if feature in ["intersection", "left"]:
                onFeatureIdx = self.features.getIndex(onFeature)
                onIdxLoc = matchingFtIdx[0].index(onFeatureIdx)
                onIdxL = onIdxLoc
                onIdxR = onIdxLoc
                right = [[row[i] for i in matchingFtIdx[1]]
                         for row in other.data]
                # matching indices in right were sorted when slicing above
                if len(right) > 0:
                    matchingFtIdx[1] = list(range(len(right[0])))
                else:
                    matchingFtIdx[1] = []
                if feature == "intersection":
                    self.data = [[row[i] for i in matchingFtIdx[0]]
                                 for row in self.data]
                    # matching indices in left were sorted when slicing above
                    if len(self.data) > 0:
                        matchingFtIdx[0] = list(range(len(self.data[0])))
                    else:
                        matchingFtIdx[0] = []
            else:
                onIdxL = self.features.getIndex(onFeature)
                onIdxR = other.features.getIndex(onFeature)
                right = copy.copy(other.data)
        else:
            # using pointNames, prepend pointNames to left and right lists
            onIdxL = 0
            onIdxR = 0
            left = []
            right = []

            def ptNameGetter(obj, idx, suffix):
                if obj._pointNamesCreated():
                    name = obj.points.getName(idx)
                    if not name.startswith(DEFAULT_PREFIX):
                        return name
                    else:
                        # differentiate default names between objects;
                        # note still start with DEFAULT_PREFIX
                        return name + suffix
                else:
                    return DEFAULT_PREFIX + str(idx) + suffix

            if feature == "intersection":
                for i, pt in enumerate(self.data):
                    ptL = [ptNameGetter(self, i, '_l')]
                    intersect = [val for idx, val in enumerate(pt)
                                 if idx in matchingFtIdx[0]]
                    self.data[i] = ptL + intersect
                for i, pt in enumerate(other.data):
                    ptR = [ptNameGetter(other, i, '_r')]
                    ptR.extend([pt[i] for i in matchingFtIdx[1]])
                    right.append(ptR)
                # matching indices were sorted above
                # this also accounts for prepended column
                if len(self.data) > 0:
                    matchingFtIdx[0] = list(range(len(self.data[0])))
                else:
                    matchingFtIdx[0] = []
                matchingFtIdx[1] = matchingFtIdx[0]
            elif feature == "left":
                for i, pt in enumerate(self.data):
                    ptL = [ptNameGetter(self, i, '_l')]
                    self.data[i] = ptL + pt
                for i, pt in enumerate(other.data):
                    ptR = [ptNameGetter(other, i, '_r')]
                    ptR.extend([pt[i] for i in matchingFtIdx[1]])
                    right.append(ptR)
                # account for new column in matchingFtIdx
                matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
                matchingFtIdx[0].insert(0, 0)
                # matching indices were sorted when slicing above
                # this also accounts for prepended column
                matchingFtIdx[1] = list(range(len(right[0])))
            else:
                for i, pt in enumerate(self.data):
                    ptL = [ptNameGetter(self, i, '_l')]
                    self.data[i] = ptL + pt
                for i, pt in enumerate(other.data):
                    ptR = [ptNameGetter(other, i, '_r')]
                    ptR.extend(pt)
                    right.append(ptR)
                matchingFtIdx[0] = list(map(lambda x: x + 1, matchingFtIdx[0]))
                matchingFtIdx[0].insert(0, 0)
                matchingFtIdx[1] = list(map(lambda x: x + 1, matchingFtIdx[1]))
                matchingFtIdx[1].insert(0, 0)
        left = self.data

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
                ptR = [numpy.nan] * (len(right[0]) - len(matchingFtIdx[1]))
                pt = ptL + ptR
                merged.append(pt)

        if point == 'union':
            for row in right:
                target = row[onIdxR]
                if target not in matched:
                    pt = [numpy.nan] * (len(left[0]) + unmatchedFtCountR)
                    for i, j in zip(matchingFtIdx[0], matchingFtIdx[1]):
                        pt[i] = row[j]
                    pt[len(left[0]):] = [row[i] for i in range(len(right[0]))
                                         if i not in matchingFtIdx[1]]
                    merged.append(pt)

        self._featureCount = len(left[0]) + unmatchedFtCountR
        self._pointCount = len(merged)
        if onFeature is None:
            # remove point names feature
            merged = [row[1:] for row in merged]
            self._featureCount -= 1
        self._numFeatures = self._featureCount

        self.data = merged

    def _replaceFeatureWithBinaryFeatures_implementation(self, uniqueVals):
        toFill = numpy.zeros((len(self.points), len(uniqueVals)))
        for ptIdx, val in enumerate(self.data):
            ftIdx = uniqueVals.index(val[0])
            toFill[ptIdx, ftIdx] = 1
        return List(toFill.tolist())

    def _getitem_implementation(self, x, y):
        return self.data[x][y]

    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd):
        kwds = {}
        kwds['data'] = ListPassThrough(self, pointStart, pointEnd,
                                       featureStart, featureEnd)
        kwds['source'] = self
        kwds['pointStart'] = pointStart
        kwds['pointEnd'] = pointEnd
        kwds['featureStart'] = featureStart
        kwds['featureEnd'] = featureEnd
        kwds['reuseData'] = True
        kwds['shape'] = (pointEnd - pointStart, featureEnd - featureStart)
        return ListView(**kwds)

    def _validate_implementation(self, level):
        assert len(self.data) == len(self.points)
        assert self._numFeatures == len(self.features)

        if level > 0:
            if len(self.data) > 0:
                expectedLength = len(self.data[0])
            for point in self.data:
            #				assert isinstance(point, list)
                assert len(point) == expectedLength

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0
        contained in this object. False otherwise
        """
        for point in self.points:
            for i in range(len(point)):
                if point[i] == 0:
                    return True
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

class ListView(BaseView, List):
    """
    Read only access to a List object.
    """
    def __init__(self, **kwds):
        super(ListView, self).__init__(**kwds)

    def _getPoints(self):
        return ListPointsView(self)

    def _getFeatures(self):
        return ListFeaturesView(self)

    def _getElements(self):
        return ListElementsView(self)

    def _copy_implementation(self, to):
        # we only want to change how List and pythonlist copying is
        # done we also temporarily convert self.data to a python list
        # for copy
        if ((len(self.points) == 0 or len(self.features) == 0)
                and to != 'List'):
            emptyStandin = numpy.empty((len(self.points),
                                        len(self.features)))
            intermediate = nimble.createData('Matrix', emptyStandin,
                                             useLog=False)
            return intermediate.copy(to=to)

        listForm = [list(pt) for pt in self.points]

        if to not in ['List', 'pythonlist']:
            origData = self.data
            self.data = listForm
            res = super(ListView, self)._copy_implementation(
                to)
            self.data = origData
            return res

        if to == 'List':
            ptNames = self.points._getNamesNoGeneration()
            ftNames = self.features._getNamesNoGeneration()
            return List(listForm, pointNames=ptNames,
                        featureNames=ftNames, shape=self.shape)
        else:
            return listForm

class FeatureViewer(object):
    """
    View by feature axis for list.
    """
    def __init__(self, source, fStart, fEnd):
        self.source = source
        self.fStart = fStart
        self.fRange = fEnd - fStart
        self.limit = None

    def setLimit(self, pIndex):
        """
        Limit to a given point in the feature.
        """
        self.limit = pIndex

    def __getitem__(self, key):
        if key < 0 or key >= self.fRange:
            msg = "The given index " + str(key) + " is outside of the "
            msg += "range  of possible indices in the feature axis (0 "
            msg += "to " + str(self.fRange - 1) + ")."
            raise IndexError(msg)

        return self.source.data[self.limit][key + self.fStart]

    def __len__(self):
        return self.fRange

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for i, sVal in enumerate(self):
            oVal = other[i]
            # check element equality - which is only relevant if one
            # of the elements is non-NaN
            if sVal != oVal and (sVal == sVal or oVal == oVal):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

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

    def __getitem__(self, key):
        self.fviewer = FeatureViewer(self.source, self.fStart,
                                     self.fEnd)
        if key < 0 or key >= self.pRange:
            msg = "The given index " + str(key) + " is outside of the "
            msg += "range  of possible indices in the point axis (0 "
            msg += "to " + str(self.pRange - 1) + ")."
            raise IndexError(msg)

        self.fviewer.setLimit(key + self.pStart)
        return self.fviewer

    def __len__(self):
        return self.pRange

    def __array__(self, dtype=None):
        tmpArray = numpy.array(self.source.data, dtype=dtype)
        return tmpArray[self.pStart:self.pEnd, self.fStart:self.fEnd]
