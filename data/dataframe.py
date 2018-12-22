"""
Class extending Base, using a pandas DataFrame to store data.
"""

from __future__ import division
from __future__ import absolute_import
import itertools
import copy
import re

import numpy
import numpy as np
from six.moves import range

from .base import Base, cmp_to_key
from .base_view import BaseView
from .dataframePoints import DataFramePoints, DataFramePointsView
from .dataframeFeatures import DataFrameFeatures, DataFrameFeaturesView
from .dataframeElements import DataFrameElements, DataFrameElementsView
from .dataHelpers import DEFAULT_PREFIX
from .dataHelpers import inheritDocstringsFactory
import UML
from UML.exceptions import ArgumentException, PackageException

pd = UML.importModule('pandas')
scipy = UML.importModule('scipy.sparse')

@inheritDocstringsFactory(Base)
class DataFrame(Base):
    """
    Class providing implementations of data manipulation operations on data stored
    in a pandas DataFrame.
    """

    def __init__(self, data, reuseData=False, elementType=None, **kwds):
        """
        The initializer.
        Inputs:
            data: pandas DataFrame, or numpy matrix.
            reuseData: boolean. only used when data is a pandas DataFrame.
        """
        if not pd:
            msg = 'To use class DataFrame, pandas must be installed.'
            raise PackageException(msg)

        if (not isinstance(data, (pd.DataFrame, np.matrix))):
            msg = "the input data can only be a pandas DataFrame or a numpy matrix or ListPassThrough."
            raise ArgumentException(msg)

        if isinstance(data, pd.DataFrame):
            if reuseData:
                self.data = data
            else:
                self.data = data.copy()
        else:
            self.data = pd.DataFrame(data)

        kwds['shape'] = self.data.shape
        super(DataFrame, self).__init__(**kwds)

    def _getPoints(self):
        return DataFramePoints(self)

    def _getFeatures(self):
        return DataFrameFeatures(self)

    def _getElements(self):
        return DataFrameElements(self)

    def _transpose_implementation(self):
        """
        Function to transpose the data, ie invert the feature and point indices of the data.

        This is not an in place operation, a new pandas DataFrame is constructed.
        """
        self.data = self.data.T

    def _getTypeString_implementation(self):
        return 'DataFrame'

    def _isIdentical_implementation(self, other):
        if not isinstance(other, DataFrame):
            return False
        if len(self.points) != len(other.points):
            return False
        if len(self.features) != len(other.features):
            return False

        try:
            tmp1 = self.data.values
            tmp2 = other.data.values
            np.testing.assert_equal(tmp1, tmp2)
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
                self.data.columns = self.features.getNames()
                if includePointNames:
                    outFile.write('point_names')

            if includePointNames:
                    self.data.index = self.points.getNames()

        self.data.to_csv(outPath, mode='a', index=includePointNames, header=includeFeatureNames)

        if includePointNames:
            self._updateName('point')
        if includeFeatureNames:
            self._updateName('feature')

    def _writeFileMTX_implementation(self, outPath, includePointNames, includeFeatureNames):
        """
        Function to write the data in this object to a matrix market file at the designated
        path.
        """
        if not scipy:
            msg = "scipy is not available"
            raise PackageException(msg)

        from scipy.io import mmwrite

        comment = '#'
        if includePointNames:
            comment += ','.join(self.points.getNames())
        if includeFeatureNames:
            comment += '\n#' + ','.join(self.features.getNames())
        mmwrite(outPath, self.data, comment=comment)

    def _referenceDataFrom_implementation(self, other):
        if not isinstance(other, DataFrame):
            raise ArgumentException("Other must be the same type as this object")

        self.data = other.data

    def _copyAs_implementation(self, format):
        """
        Copy the current DataFrame object to another one in the format.
        Input:
            format: string. Sparse, List, Matrix, pythonlist, numpyarray, numpymatrix, scipycsc, scipycsr or None
                    if format is None, a new DataFrame will be created.
        """
        dataArray = self.data.values.copy()
        if format is None or format == 'DataFrame':
            return UML.createData('DataFrame', dataArray)
        if format == 'Sparse':
            return UML.createData('Sparse', dataArray)
        if format == 'List':
            return UML.createData('List', dataArray)
        if format == 'Matrix':
            return UML.createData('Matrix', dataArray)
        if format == 'pythonlist':
            return dataArray.tolist()
        if format == 'numpyarray':
            return dataArray
        if format == 'numpymatrix':
            return np.matrix(dataArray)
        if format == 'scipycsc':
            if not scipy:
                msg = "scipy is not available"
                raise PackageException(msg)
            return scipy.sparse.csc_matrix(dataArray)
        if format == 'scipycsr':
            if not scipy:
                msg = "scipy is not available"
                raise PackageException(msg)
            return scipy.sparse.csr_matrix(dataArray)

        return UML.createData('DataFrame', dataArray)

    def _fillWith_implementation(self, values, pointStart, featureStart, pointEnd, featureEnd):
        """
        """
        if not isinstance(values, UML.data.Base):
            values = values * np.ones((pointEnd - pointStart + 1, featureEnd - featureStart + 1))
        else:
            #convert values to be array or matrix, instead of pandas DataFrame
            values = values.data.values

        self.data.iloc[pointStart:pointEnd + 1, featureStart:featureEnd + 1] = values

    def _flattenToOnePoint_implementation(self):
        numElements = len(self.points) * len(self.features)
        self.data = pd.DataFrame(self.data.values.reshape((1, numElements), order='C'))

    def _flattenToOneFeature_implementation(self):
        numElements = len(self.points) * len(self.features)
        self.data = pd.DataFrame(self.data.values.reshape((numElements,1), order='F'))

    def _unflattenFromOnePoint_implementation(self, numPoints):
        numFeatures = len(self.features) // numPoints
        self.data = pd.DataFrame(self.data.values.reshape((numPoints, numFeatures), order='C'))

    def _unflattenFromOneFeature_implementation(self, numFeatures):
        numPoints = len(self.points) // numFeatures
        self.data = pd.DataFrame(self.data.values.reshape((numPoints, numFeatures), order='F'))

    def _getitem_implementation(self, x, y):
        # return self.data.ix[x, y]
        #use self.data.values is much faster
        return self.data.values[x, y]

    def _view_implementation(self, pointStart, pointEnd, featureStart, featureEnd):
        """

        """

        class DataFrameView(BaseView, DataFrame):
            def __init__(self, **kwds):
                super(DataFrameView, self).__init__(**kwds)

            def _getPoints(self):
                return DataFramePointsView(self)

            def _getFeatures(self):
                return DataFrameFeaturesView(self)

            def _getElements(self):
                return DataFrameElementsView(self)

            def _setAllDefault(self, axis):
                super(DataFrameView, self)._setAllDefault(axis)

        kwds = {}
        kwds['data'] = self.data.iloc[pointStart:pointEnd, featureStart:featureEnd]
        kwds['source'] = self
        kwds['pointStart'] = pointStart
        kwds['pointEnd'] = pointEnd
        kwds['featureStart'] = featureStart
        kwds['featureEnd'] = featureEnd
        kwds['reuseData'] = True

        ret = DataFrameView(**kwds)
        ret._updateName('point')
        ret._updateName('feature')

        return ret

    def _validate_implementation(self, level):
        shape = self.data.shape
        assert shape[0] == len(self.points)
        assert shape[1] == len(self.features)

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0 contained
        in this object. False otherwise

        """
        return 0 in self.data.values

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
                    value = self._source.data.iloc[self._pIndex, self._fIndex]

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

        leftData = np.matrix(self.data)
        rightData = other.data if isinstance(other, UML.data.Sparse) else np.matrix(other.data)
        return DataFrame(leftData * rightData)

    def _elementwiseMultiply_implementation(self, other):
        """
        Perform element wise multiplication of this UML data object against the
        provided other UML data object. Both objects must contain only numeric
        data. The pointCount and featureCount of both objects must be equal. The
        types of the two objects may be different, but the returned object will
        be the inplace modification of the calling object.

        """
        if isinstance(other, UML.data.Sparse):
            result = other.data.multiply(self.data.values)
            if hasattr(result, 'todense'):
                result = result.todense()
            self.data = pd.DataFrame(result)
        else:
            self.data = pd.DataFrame(np.multiply(self.data.values, other.data))

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
        """
        """
        leftData = np.matrix(self.data)
        if isinstance(other, UML.data.Base):
            rightData = other.data if isinstance(other, UML.data.Sparse) else np.matrix(other.data)
        else:
            rightData = other
        leftData += rightData
        return DataFrame(leftData, reuseData=True)

    def _radd__implementation(self, other):
        ret = other + self.data
        return DataFrame(ret, reuseData=True)

    def _iadd__implementation(self, other):
        if isinstance(other, UML.data.Base):
            ret = np.matrix(self.data) + (other.data if isinstance(other, UML.data.Sparse) else np.matrix(other.data))
        else:
            ret = np.matrix(self.data) + np.matrix(other)
        self.data = pd.DataFrame(ret)
        return self

    def _sub__implementation(self, other):
        leftData = np.matrix(self.data)
        if isinstance(other, UML.data.Base):
            rightData = other.data if isinstance(other, UML.data.Sparse) else np.matrix(other.data)
        else:
            rightData = other
        leftData -= rightData

        return DataFrame(leftData, reuseData=True)

    def _rsub__implementation(self, other):
        ret = other - self.data.values
        return UML.createData('DataFrame', ret, pointNames=self.points.getNames(), featureNames=self.features.getNames(), reuseData=True)

    def _isub__implementation(self, other):
        if isinstance(other, UML.data.Base):
            ret = np.matrix(self.data) - (other.data if isinstance(other, UML.data.Sparse) else np.matrix(other.data))
        else:
            ret = np.matrix(self.data) - np.matrix(other)
        self.data = pd.DataFrame(ret)
        return self

    def _div__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data / other.data.todense()
            elif isinstance(other.data, list):
                ret = self.data / np.array(other.data)
            else:
                ret = self.data / other.data
        else:
            ret = self.data / other
        return DataFrame(ret, reuseData=True)


    def _rdiv__implementation(self, other):
        ret = np.asmatrix(other / self.data.values)
        return DataFrame(ret, reuseData=True)

    def _idiv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                self.data /= other.data.todense()
            else:
                self.data /= np.matrix(other.data)
        else:
            tmp_mat = np.matrix(other)
            self.data /= (other if tmp_mat.shape == (1, 1) else tmp_mat)
        return self

    def _truediv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data.values.__truediv__(other.data.todense())
            else:
                ret = self.data.values.__truediv__(other.data)
        else:
            ret = self.data.values.__itruediv__(other)
        return DataFrame(np.asmatrix(ret), reuseData=True)

    def _rtruediv__implementation(self, other):
        ret = self.data.values.__rtruediv__(other)
        return DataFrame(np.asmatrix(ret), reuseData=True)

    def _itruediv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = np.matrix(self.data).__itruediv__(other.data.todense())
            else:
                ret = np.matrix(self.data).__itruediv__(np.matrix(other.data))
        else:
            ret = np.matrix(self.data).__itruediv__(np.matrix(other))
        self.data = pd.DataFrame(ret)
        return self

    def _floordiv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data.values // other.data.todense()
            else:
                ret = self.data.values // other.data
        else:
            ret = self.data.values // other
        return DataFrame(np.asmatrix(ret), reuseData=True)


    def _rfloordiv__implementation(self, other):
        ret = other // self.data.values
        return DataFrame(np.asmatrix(ret), reuseData=True)

    def _ifloordiv__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = np.matrix(self.data) // other.data.todense()
            else:
                ret = np.matrix(self.data) // np.matrix(other.data)
        else:
            ret = np.matrix(self.data) // np.matrix(other)
        self.data = pd.DataFrame(ret)
        return self

    def _mod__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                ret = self.data.values % other.data.todense()
            else:
                ret = self.data.values % other.data
        else:
            ret = self.data.values % other
        return DataFrame(np.asmatrix(ret), reuseData=True)


    def _rmod__implementation(self, other):
        ret = other % self.data.values
        return DataFrame(np.asmatrix(ret), reuseData=True)


    def _imod__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if scipy and scipy.sparse.isspmatrix(other.data):
                self.data %= other.data.todense()
            else:
                self.data %= np.matrix(other.data)
        else:
            tmp_mat = np.matrix(other)
            self.data %= (other if tmp_mat.shape == (1, 1) else tmp_mat)

        return self

    def _updateName(self, axis):
        """
        update self.data.index or self.data.columns
        """
        if axis == 'point':
            self.data.index = list(range(len(self.data.index)))
        else:
            # self.data.columns = self.features.getNames()
            self.data.columns = list(range(len(self.data.columns)))
        #-----------------------------------------------------------------------
