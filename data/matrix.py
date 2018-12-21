"""
Class extending Base, using a numpy dense matrix to store data.

"""

from __future__ import division
from __future__ import absolute_import
import sys
import itertools
import copy
from functools import reduce

import numpy
from six.moves import range
from six.moves import zip

from .base import Base, cmp_to_key
from .base_view import BaseView
from .matrixPoints import MatrixPoints, MatrixPointsView
from .matrixFeatures import MatrixFeatures, MatrixFeaturesView
from .matrixElements import MatrixElements, MatrixElementsView
from .dataHelpers import inheritDocstringsFactory
import UML
from UML.exceptions import ArgumentException, PackageException
from UML.randomness import pythonRandom
from UML.randomness import numpyRandom

scipy = UML.importModule('scipy.io')

@inheritDocstringsFactory(Base)
class Matrix(Base):
    """
    Class providing implementations of data manipulation operations on data stored
    in a numpy dense matrix.

    """

    def __init__(self, data, featureNames=None, reuseData=False, elementType=None, **kwds):
        """
        data can only be a numpy matrix
        """
        if (not isinstance(data, (numpy.matrix, numpy.ndarray))):# and 'PassThrough' not in str(type(data)):
            msg = "the input data can only be a numpy matrix or ListPassThrough."
            raise ArgumentException(msg)

        if isinstance(data, numpy.matrix):
            if reuseData:
                self.data = data
            else:
                self.data = copy.copy(data)#copy.deepcopy may give messed data
        else:
            #when data is a np matrix, its dtype has been adjusted in extractNamesAndConvertData
            #but when data is a ListPassThrough, we need to do dtype adjustment here
            if elementType:
                self.data = numpy.matrix(data, dtype=elementType)
            else:
                try:
                    self.data = numpy.matrix(data, dtype=numpy.float)
                except ValueError:
                    self.data = numpy.matrix(data, dtype=object)

        kwds['featureNames'] = featureNames
        kwds['shape'] = self.data.shape
        super(Matrix, self).__init__(**kwds)

    def _getPoints(self):
        return MatrixPoints(self)

    def _getFeatures(self):
        return MatrixFeatures(self)

    def _getElements(self):
        return MatrixElements(self)

    def _transpose_implementation(self):
        """
        Function to transpose the data, ie invert the feature and point indices of the data.

        This is not an in place operation, a new list of lists is constructed.
        """
        self.data = self.data.getT()

    def _getTypeString_implementation(self):
        return 'Matrix'

    def _isIdentical_implementation(self, other):
        if not isinstance(other, Matrix):
            return False
        if len(self.points) != len(other.points):
            return False
        if len(self.features) != len(other.features):
            return False

        try:
            numpy.testing.assert_equal(self.data, other.data)
        except AssertionError:
            return False
        return True

    def _writeFile_implementation(self, outPath, format, includePointNames, includeFeatureNames):
        """
        Function to write the data in this object to a file using the specified
        format. outPath is the location (including file name and extension) where
        we want to write the output file. includeNames is boolean argument
        indicating whether the file should start with comment lines designating
        pointNames and featureNames.

        """
        if format not in ['csv', 'mtx']:
            msg = "Unrecognized file format. Accepted types are 'csv' and 'mtx'. They may "
            msg += "either be input as the format parameter, or as the extension in the "
            msg += "outPath"
            raise ArgumentException(msg)

        if format == 'csv':
            return self._writeFileCSV_implementation(outPath, includePointNames, includeFeatureNames)
        if format == 'mtx':
            return self._writeFileMTX_implementation(outPath, includePointNames, includeFeatureNames)

    def _writeFileCSV_implementation(self, outPath, includePointNames, includeFeatureNames):
        """
        Function to write the data in this object to a CSV file at the designated
        path.

        """
        with open(outPath, 'w') as outFile:

            if includeFeatureNames:
                def combine(a, b):
                    return a + ',' + b

                fnames = self.getFeatureNames()
                fnamesLine = reduce(combine, fnames)
                fnamesLine += '\n'
                if includePointNames:
                    outFile.write('point_names,')

                outFile.write(fnamesLine)

        with open(outPath, 'ab') as outFile:#python3 need this.
            if includePointNames:
                pnames = numpy.matrix(self.getPointNames())
                pnames = pnames.transpose()

                viewData = self.data.view()
                toWrite = numpy.concatenate((pnames, viewData), 1)

                numpy.savetxt(outFile, toWrite, delimiter=',', fmt='%s')
            else:
                numpy.savetxt(outFile, self.data, delimiter=',')

    def _writeFileMTX_implementation(self, outPath, includePointNames, includeFeatureNames):
        if not scipy:
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
            header = makeNameString(len(self.points), self.getPointName)
            header += '\n'
        else:
            header += '#\n'
        if includeFeatureNames:
            header += makeNameString(len(self.features), self.getFeatureName)
        else:
            header += '#\n'

        if header != '':
            scipy.io.mmwrite(target=outPath, a=self.data, comment=header)
        else:
            scipy.io.mmwrite(target=outPath, a=self.data)

    def _referenceDataFrom_implementation(self, other):
        if not isinstance(other, Matrix):
            raise ArgumentException("Other must be the same type as this object")

        self.data = other.data

    def _copyAs_implementation(self, format):

        if format is None or format == 'Matrix':
            return UML.createData('Matrix', self.data)
        if format == 'Sparse':
            return UML.createData('Sparse', self.data)
        if format == 'List':
            return UML.createData('List', self.data)
        if format == 'DataFrame':
            return UML.createData('DataFrame', self.data)
        if format == 'pythonlist':
            return self.data.tolist()
        if format == 'numpyarray':
            return numpy.array(self.data)
        if format == 'numpymatrix':
            return numpy.matrix(self.data)
        if format == 'scipycsc':
            if not scipy:
                msg = "scipy is not available"
                raise PackageException(msg)
            return scipy.sparse.csc_matrix(self.data)
        if format == 'scipycsr':
            if not scipy:
                msg = "scipy is not available"
                raise PackageException(msg)
            return scipy.sparse.csr_matrix(self.data)

        return UML.createData('Matrix', self.data)

    def _fillWith_implementation(self, values, pointStart, featureStart, pointEnd, featureEnd):
        if not isinstance(values, UML.data.Base):
            values = values * numpy.ones((pointEnd - pointStart + 1, featureEnd - featureStart + 1))
        else:
            values = values.data

        self.data[pointStart:pointEnd + 1, featureStart:featureEnd + 1] = values

    def _flattenToOnePoint_implementation(self):
        numElements = len(self.points) * len(self.features)
        self.data = self.data.reshape((1, numElements), order='C')

    def _flattenToOneFeature_implementation(self):
        numElements = len(self.points) * len(self.features)
        self.data = self.data.reshape((numElements,1), order='F')

    def _unflattenFromOnePoint_implementation(self, numPoints):
        numFeatures = len(self.features) // numPoints
        self.data = self.data.reshape((numPoints, numFeatures), order='C')

    def _unflattenFromOneFeature_implementation(self, numFeatures):
        numPoints = len(self.points) // numFeatures
        self.data = self.data.reshape((numPoints, numFeatures), order='F')

    def _getitem_implementation(self, x, y):
        return self.data[x, y]

    def _view_implementation(self, pointStart, pointEnd, featureStart, featureEnd):
        class MatrixView(BaseView, Matrix):
            def __init__(self, **kwds):
                super(MatrixView, self).__init__(**kwds)

            def _getPoints(self):
                return MatrixPointsView(self)

            def _getFeatures(self):
                return MatrixFeaturesView(self)

            def _getElements(self):
                return MatrixElementsView(self)

        kwds = {}
        kwds['data'] = self.data[pointStart:pointEnd, featureStart:featureEnd]
        kwds['source'] = self
        kwds['pointStart'] = pointStart
        kwds['pointEnd'] = pointEnd
        kwds['featureStart'] = featureStart
        kwds['featureEnd'] = featureEnd
        kwds['reuseData'] = True

        return MatrixView(**kwds)

    def _validate_implementation(self, level):
        shape = numpy.shape(self.data)
        assert shape[0] == len(self.points)
        assert shape[1] == len(self.features)

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0 contained
        in this object. False otherwise

        """
        return 0 in self.data

    def _nonZeroIteratorPointGrouped_implementation(self):
        class nzIt(object):
            def __init__(self, source):
                self._source = source
                self._pIndex = 0
                self._pStop = len(source.points)
                self._fIndex = 0
                self._fStop = len(source.features)

            def __iter__(self):
                return self

            def next(self):
                while (self._pIndex < self._pStop):
                    value = self._source.data[self._pIndex, self._fIndex]

                    self._fIndex += 1
                    if self._fIndex >= self._fStop:
                        self._fIndex = 0
                        self._pIndex += 1

                    if value != 0:
                        return value

                raise StopIteration

            def __next__(self):
                return self.next()

        return nzIt(self)

    def _nonZeroIteratorFeatureGrouped_implementation(self):
        class nzIt(object):
            def __init__(self, source):
                self._source = source
                self._pIndex = 0
                self._pStop = len(source.points)
                self._fIndex = 0
                self._fStop = len(source.features)

            def __iter__(self):
                return self

            def next(self):
                while (self._fIndex < self._fStop):
                    value = self._source.data[self._pIndex, self._fIndex]

                    self._pIndex += 1
                    if self._pIndex >= self._pStop:
                        self._pIndex = 0
                        self._fIndex += 1

                    if value != 0:
                        return value

                raise StopIteration

            def __next__(self):
                return self.next()

        return nzIt(self)

    def _matrixMultiply_implementation(self, other):
        """
        Matrix multiply this UML data object against the provided other UML data
        object. Both object must contain only numeric data. The featureCount of
        the calling object must equal the pointCount of the other object. The
        types of the two objects may be different, and the return is guaranteed
        to be the same type as at least one out of the two, to be automatically
        determined according to efficiency constraints.

        """
        if isinstance(other, Matrix) or isinstance(other, UML.data.Sparse):
            return Matrix(self.data * other.data)
        else:
            return Matrix(self.data * other.copyAs("numpyarray"))

    def _elementwiseMultiply_implementation(self, other):
        """
        Perform element wise multiplication of this UML data object against the
        provided other UML data object. Both objects must contain only numeric
        data. The pointCount and featureCount of both objects must be equal. The
        types of the two objects may be different, but the returned object will
        be the inplace modification of the calling object.

        """
        if isinstance(other, UML.data.Sparse):
            result = other.data.multiply(self.data)
            if hasattr(result, 'todense'):
                result = result.todense()
            self.data = numpy.matrix(result)
        else:
            self.data = numpy.multiply(self.data, other.data)

    def _scalarMultiply_implementation(self, scalar):
        """
        Multiply every element of this UML data object by the provided scalar.
        This object must contain only numeric data. The 'scalar' parameter must
        be a numeric data type. The returned object will be the inplace modification
        of the calling object.

        """
        self.data = self.data * scalar

    def _mul__implementation(self, other):
        if isinstance(other, UML.data.Base):
            return self._matrixMultiply_implementation(other)
        else:
            ret = self.copy()
            ret._scalarMultiply_implementation(other)
            return ret

    def _add__implementation(self, other):
        if isinstance(other, UML.data.Base):
            ret = self.data + other.data
        else:
            ret = self.data + other
        return Matrix(ret, reuseData=True)

    def _radd__implementation(self, other):
        ret = other + self.data
        return Matrix(ret, reuseData=True)

    def _iadd__implementation(self, other):
        if isinstance(other, UML.data.Base):
            ret = self.data + other.data
        else:
            ret = self.data + other
        self.data = ret
        return self

    def _sub__implementation(self, other):
        if isinstance(other, UML.data.Base):
            ret = self.data - other.data
        else:
            ret = self.data - other
        return Matrix(ret, reuseData=True)

    def _rsub__implementation(self, other):
        ret = other - self.data
        return Matrix(ret, reuseData=True)

    def _isub__implementation(self, other):
        if isinstance(other, UML.data.Base):
            ret = self.data - other.data
        else:
            ret = self.data - other
        self.data = ret
        return self

    def _div__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data / other.data.todense()
            else:
                ret = self.data / other.data
        else:
            ret = self.data / other
        return Matrix(ret, reuseData=True)

    def _rdiv__implementation(self, other):
        ret = other / self.data
        return Matrix(ret, reuseData=True)

    def _idiv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data / other.data.todense()
            else:
                ret = self.data / other.data
        else:
            ret = self.data / other
        self.data = ret
        return self

    def _truediv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data.__truediv__(other.data.todense())
            else:
                ret = self.data.__truediv__(other.data)
        else:
            ret = self.data.__itruediv__(other)
        return Matrix(ret, reuseData=True)

    def _rtruediv__implementation(self, other):
        ret = self.data.__rtruediv__(other)
        return Matrix(ret, reuseData=True)

    def _itruediv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data.__itruediv__(other.data.todense())
            else:
                ret = self.data.__itruediv__(other.data)
        else:
            ret = self.data.__itruediv__(other)
        self.data = ret
        return self

    def _floordiv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data // other.data.todense()
            else:
                ret = self.data // other.data
        else:
            ret = self.data // other
        return Matrix(ret, reuseData=True)


    def _rfloordiv__implementation(self, other):
        ret = other // self.data
        return Matrix(ret, reuseData=True)

    def _ifloordiv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data // other.data.todense()
            else:
                ret = self.data // other.data
        else:
            ret = self.data // other
        self.data = ret
        return self

    def _mod__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data % other.data.todense()
            else:
                ret = self.data % other.data
        else:
            ret = self.data % other
        return Matrix(ret, reuseData=True)


    def _rmod__implementation(self, other):
        ret = other % self.data
        return Matrix(ret, reuseData=True)


    def _imod__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data % other.data.todense()
            else:
                ret = self.data % other.data
        else:
            ret = self.data % other
        self.data = ret
        return self


def viewBasedApplyAlongAxis(function, axis, outerObject):
    """ applies the given function to each view along the given axis, returning the results
    of the function in numpy array """
    if axis == "point":
        maxVal = outerObject.data.shape[0]
        viewMaker = outerObject.pointView
    else:
        if axis != "feature":
            raise ArgumentException("axis must be 'point' or 'feature'")
        maxVal = outerObject.data.shape[1]
        viewMaker = outerObject.featureView
    ret = numpy.zeros(maxVal, dtype=numpy.float)
    for i in range(0, maxVal):
        funcOut = function(viewMaker(i))
        ret[i] = funcOut

    return ret

def matrixBasedApplyAlongAxis(function, axis, outerObject):
    """
    applies the given function to the underlying numpy matrix along the given axis,
    returning the results of the function in numpy array
    """
    #make sure the 3 attributes are in the function object
    if not (hasattr(function, 'nameOfFeatureOrPoint') \
                and hasattr(function, 'valueOfFeatureOrPoint') and hasattr(function, 'optr')):
        msg = "some important attribute is missing in the input function"
        raise ArgumentException(msg)
    if axis == "point":
        #convert name of feature to index of feature
        indexOfFeature = outerObject.getFeatureIndex(function.nameOfFeatureOrPoint)
        #extract the feature from the underlying matrix
        queryData = outerObject.data[:, indexOfFeature]
    else:
        if axis != "feature":
            raise ArgumentException("axis must be 'point' or 'feature'")
        #convert name of point to index of point
        indexOfPoint = outerObject.getPointIndex(function.nameOfFeatureOrPoint)
        #extract the point from the underlying matrix
        queryData = outerObject.data[indexOfPoint, :]
    ret = function.optr(queryData, function.valueOfFeatureOrPoint)
    #convert the result from matrix to numpy array
    ret = ret.astype(numpy.float).A1

    return ret
