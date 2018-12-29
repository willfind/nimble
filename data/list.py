"""
Class extending Base, using a list of lists to store data.

"""

from __future__ import division
from __future__ import absolute_import
import copy
import numbers
from functools import reduce

import numpy
import six
from six.moves import range

import UML
from UML.exceptions import ArgumentException, PackageException
from .base import Base
from .base_view import BaseView
from .listPoints import ListPoints, ListPointsView
from .listFeatures import ListFeatures, ListFeaturesView
from .listElements import ListElements, ListElementsView
from .dataHelpers import inheritDocstringsFactory

scipy = UML.importModule('scipy.io')
pd = UML.importModule('pandas')

allowedItemType = (numbers.Number, six.string_types)
def isAllowedSingleElement(x):
    """
    This function is to determine if an element is an allowed single
    element
    """
    if isinstance(x, allowedItemType):
        return True

    if hasattr(x, '__len__'):#not a single element
        return False

    if x is None or x != x:#None and np.NaN are allowed
        return True

    return

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
        A list, numpy matrix, or a ListPassThrough.
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

    def __init__(self, data, reuseData=False, shape=None, checkAll=True,
                 elementType=None, **kwds):
        if ((not isinstance(data, (list, numpy.matrix)))
                and 'PassThrough' not in str(type(data))):
            msg = "the input data can only be a list or a numpy matrix "
            msg += "or ListPassThrough."
            raise ArgumentException(msg)

        if 'featureNames' not in kwds:
            kwds['featureNames'] = None
        featureNames = kwds['featureNames']

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
                            raise ArgumentException(msg)
                shape = (1, len(data))
                data = [data]
            elif isinstance(data[0], list) or hasattr(data[0], 'setLimit'):
            #case3: data=[[1,2,3], ['a', 'b', 'c']] or [[]] or [[], []].
            # self.data will be = data, shape will be (len(data), len(data[0]))
            #case4: data=[<UML.data.list.FeatureViewer object at 0x43fd410>]
                numFeatures = len(data[0])
                if checkAll:#check all items
                    for i in data:
                        if len(i) != numFeatures:
                            msg = 'invalid input data format.'
                            raise ArgumentException(msg)
                        for j in i:
                            if not isAllowedSingleElement(j):
                                msg = '%s is invalid input data format.'%j
                                raise ArgumentException(msg)
                shape = (len(data), numFeatures)

            if reuseData:
                data = data
            else:
                data = [copy.deepcopy(i) for i in data]#copy.deepcopy(data)
                #this is to convert a list x=[[1,2,3]]*2 to a
                # list y=[[1,2,3], [1,2,3]]
                # the difference is that x[0] is x[1], but y[0] is not y[1]

        if isinstance(data, numpy.matrix):
            #case5: data is a numpy matrix. shape is already in np matrix
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

    def _writeFile_implementation(self, outPath, format, includePointNames,
                                  includeFeatureNames):
        """
        Function to write the data in this object to a file using the
        specified format. outPath is the location (including file name
        and extension) where we want to write the output file.
        ``includeNames`` is boolean argument indicating whether the file
        should start with comment lines designating pointNames and
        featureNames.
        """
        if format not in ['csv', 'mtx']:
            msg = "Unrecognized file format. Accepted types are 'csv' and "
            msg += "'mtx'. They may either be input as the format parameter, "
            msg += "or as the extension in the outPath"
            raise ArgumentException(msg)

        if format == 'csv':
            return self._writeFileCSV_implementation(
                outPath, includePointNames, includeFeatureNames)
        if format == 'mtx':
            return self._writeFileMTX_implementation(
                outPath, includePointNames, includeFeatureNames)

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
                outFile.write('point_names,')

            outFile.write(fnamesLine)

        for point in self.points:
            currPname = point.points.getName(0)
            first = True
            if includePointNames:
                outFile.write(currPname)
                first = False

            for value in point:
                if not first:
                    outFile.write(',')
                outFile.write(str(value))
                first = False
            outFile.write('\n')
        outFile.close()

    def _writeFileMTX_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
        """
        Function to write the data in this object to a matrix market
        file at the designated path.
        """
        outFile = open(outPath, 'w')
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
        outFile.close()

    def _referenceDataFrom_implementation(self, other):
        if not isinstance(other, List):
            msg = "Other must be the same type as this object"
            raise ArgumentException(msg)

        self.data = other.data
        self._numFeatures = other._numFeatures

    def _copyAs_implementation(self, format):

        if format == 'Sparse':
            if len(self.points) == 0 or len(self.features) == 0:
                emptyData = numpy.empty(shape=(len(self.points),
                                               len(self.features)))
                return UML.createData('Sparse', emptyData)
            return UML.createData('Sparse', self.data)

        if format is None or format == 'List':
            if len(self.points) == 0 or len(self.features) == 0:
                emptyData = numpy.empty(shape=(len(self.points),
                                               len(self.features)))
                return UML.createData('List', emptyData)
            else:
                return UML.createData('List', self.data)
        if format == 'Matrix':
            if len(self.points) == 0 or len(self.features) == 0:
                emptyData = numpy.empty(shape=(len(self.points),
                                               len(self.features)))
                return UML.createData('Matrix', emptyData)
            else:
                return UML.createData('Matrix', self.data)
        if format == 'DataFrame':
            if len(self.points) == 0 or len(self.features) == 0:
                emptyData = numpy.empty(shape=(len(self.points),
                                               len(self.features)))
                return UML.createData('DataFrame', emptyData)
            else:
                return UML.createData('DataFrame', self.data)
        if format == 'pythonlist':
            return copy.deepcopy(self.data)
        if format == 'numpyarray':
            if len(self.points) == 0 or len(self.features) == 0:
                return numpy.empty(shape=(len(self.points),
                                          len(self.features)))
            return numpy.array(self.data, dtype=self._elementType)
        if format == 'numpymatrix':
            if len(self.points) == 0 or len(self.features) == 0:
                return numpy.matrix(numpy.empty(shape=(len(self.points),
                                                       len(self.features))))
            return numpy.matrix(self.data)
        if format == 'scipycsc':
            if not scipy:
                msg = "scipy is not available"
                raise PackageException(msg)
            return scipy.sparse.csc_matrix(numpy.array(self.data))
        if format == 'scipycsr':
            if not scipy:
                msg = "scipy is not available"
                raise PackageException(msg)
            return scipy.sparse.csr_matrix(numpy.array(self.data))

    def _fillWith_implementation(self, values, pointStart, featureStart,
                                 pointEnd, featureEnd):
        if not isinstance(values, UML.data.Base):
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

    def _getitem_implementation(self, x, y):
        return self.data[x][y]

    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd):
        class ListView(BaseView, List):
            def __init__(self, **kwds):
                super(ListView, self).__init__(**kwds)

            def _getPoints(self):
                return ListPointsView(self)

            def _getFeatures(self):
                return ListFeaturesView(self)

            def _getElements(self):
                return ListElementsView(self)

            def _copyAs_implementation(self, format):
                # we only want to change how List and pythonlist copying is
                # done we also temporarily convert self.data to a python list
                # for copyAs
                if self._pointNamesCreated():
                    pNames = self.points.getNames()
                else:
                    pNames = False
                if self._featureNamesCreated():
                    fNames = self.features.getNames()
                else:
                    fNames = False

                if ((len(self.points) == 0 or len(self.features) == 0)
                        and format != 'List'):
                    emptyStandin = numpy.empty((len(self.points),
                                                len(self.features)))
                    intermediate = UML.createData('Matrix', emptyStandin)
                    return intermediate.copyAs(format)

                listForm = [[self._source.data[pID][fID] for fID
                             in range(self._fStart, self._fEnd)]
                            for pID in range(self._pStart, self._pEnd)]
                if format is None:
                    format = 'List'
                if format != 'List' and format != 'pythonlist':
                    origData = self.data
                    self.data = listForm
                    res = super(ListView, self)._copyAs_implementation(
                        format)
                    self.data = origData
                    return res

                if format == 'List':
                    return List(listForm, pointNames=self.points.getNames(),
                                featureNames=self.features.getNames())
                else:
                    return listForm

        class FeatureViewer(object):
            def __init__(self, source, fStart, fEnd):
                self.source = source
                self.fStart = fStart
                self.fRange = fEnd - fStart

            def setLimit(self, pIndex):
                self.limit = pIndex

            def __getitem__(self, key):
                if key < 0 or key >= self.fRange:
                    raise IndexError("")

                return self.source.data[self.limit][key + self.fStart]

            def __len__(self):
                return self.fRange

            def __eq__(self, other):
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
            def __init__(self, source, pStart, pEnd, fStart, fEnd):
                self.source = source
                self.pStart = pStart
                self.pEnd = pEnd
                self.pRange = pEnd - pStart
                self.fStart = fStart
                self.fEnd = fEnd
                self.fviewer = FeatureViewer(self.source, fStart, fEnd)

            def __getitem__(self, key):
                if key < 0 or key >= self.pRange:
                    raise IndexError("")

                self.fviewer.setLimit(key + self.pStart)
                return self.fviewer

            def __len__(self):
                return self.pRange

            def __array__(self, dtype=None):
                tmpArray = numpy.array(self.source.data, dtype=dtype)
                return tmpArray[self.pStart:self.pEnd, self.fStart:self.fEnd]

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

    def _mul__implementation(self, other):
        if isinstance(other, UML.data.Base):
            return self._matrixMultiply_implementation(other)
        else:
            ret = self.copy()
            ret._scalarMultiply_implementation(other)
            return ret

    def _matrixMultiply_implementation(self, other):
        """
        Matrix multiply this UML data object against the provided other
        UML data object. Both object must contain only numeric data. The
        featureCount of the calling object must equal the pointCount of
        the other object. The types of the two objects may be different,
        and the return is guaranteed to be the same type as at least one
        out of the two, to be automatically determined according to
        efficiency constraints.
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

    def _scalarMultiply_implementation(self, scalar):
        """
        Multiply every element of this UML data object by the provided
        scalar. This object must contain only numeric data. The 'scalar'
        parameter must be a numeric data type. The returned object will
        be the inplace modification of the calling object.
        """
        for point in self.data:
            for i in range(len(point)):
                point[i] *= scalar

    def outputMatrixData(self):
        """
        convert self.data to a numpy matrix
        """
        if len(self.data) == 0:# in case, self.data is []
            return numpy.matrix(numpy.empty([len(self.points.getNames()),
                                             len(self.features.getNames())]))

        return numpy.matrix(self.data)
