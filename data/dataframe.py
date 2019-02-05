"""
Class extending Base, using a pandas DataFrame to store data.
"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
from six.moves import range
from six.moves import zip

import UML
from UML.exceptions import ArgumentException, PackageException
from .base import Base
from .base_view import BaseView
from .dataframePoints import DataFramePoints, DataFramePointsView
from .dataframeFeatures import DataFrameFeatures, DataFrameFeaturesView
from .dataframeElements import DataFrameElements, DataFrameElementsView
from .dataHelpers import inheritDocstringsFactory
from .dataHelpers import allDataIdentical
from .dataHelpers import DEFAULT_PREFIX

pd = UML.importModule('pandas')
scipy = UML.importModule('scipy.sparse')

@inheritDocstringsFactory(Base)
class DataFrame(Base):
    """
    Class providing implementations of data manipulation operations on
    data stored in a pandas DataFrame.

    Parameters
    ----------
    data : object
        pandas DataFrame or numpy matrix.
    reuseData : bool
        Only used when data is a pandas DataFrame.
    elementType : type
        The pandas dtype of the data.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """

    def __init__(self, data, reuseData=False, elementType=None, **kwds):
        """
        The initializer.
        Inputs:
            data: pandas DataFrame, or numpy matrix.
            reuseData: boolean. only used when data is a pandas
            DataFrame.
        """
        if not pd:
            msg = 'To use class DataFrame, pandas must be installed.'
            raise PackageException(msg)

        if not isinstance(data, (pd.DataFrame, np.matrix)):
            msg = "the input data can only be a pandas DataFrame or a numpy "
            msg += "matrix or ListPassThrough."
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
        Function to transpose the data, ie invert the feature and point
        indices of the data.

        This is not an in place operation, a new pandas DataFrame is
        constructed.
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

        return allDataIdentical(self.data.values, other.data.values)

    def _writeFile_implementation(self, outPath, format, includePointNames,
                                  includeFeatureNames):
        """
        Function to write the data in this object to a file using the
        specified format. ``outPath`` is the location (including file
        name and extension) where we want to write the output file.
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
        with open(outPath, 'w') as outFile:
            if includeFeatureNames:
                self.data.columns = self.features.getNames()
                if includePointNames:
                    outFile.write('point_names')

            if includePointNames:
                self.data.index = self.points.getNames()

        self.data.to_csv(outPath, mode='a', index=includePointNames,
                         header=includeFeatureNames)

        if includePointNames:
            self._updateName('point')
        if includeFeatureNames:
            self._updateName('feature')

    def _writeFileMTX_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
        """
        Function to write the data in this object to a matrix market
        file at the designated path.
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
            msg = "Other must be the same type as this object"
            raise ArgumentException(msg)

        self.data = other.data

    def _copyAs_implementation(self, format):
        """
        Copy the current DataFrame object to another one in the format.

        format: string.
            Sparse, List, Matrix, pythonlist, numpyarray, numpymatrix,
            scipycsc, scipycsr or None. If format is None, a new
            DataFrame will be created.
        """
        dataArray = self.data.values.copy()
        if format is None or format == 'DataFrame':
            return UML.createData('DataFrame', dataArray, useLog=False)
        if format == 'Sparse':
            return UML.createData('Sparse', dataArray, useLog=False)
        if format == 'List':
            return UML.createData('List', dataArray, useLog=False)
        if format == 'Matrix':
            return UML.createData('Matrix', dataArray, useLog=False)
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

        return UML.createData('DataFrame', dataArray, useLog=False)

    def _fillWith_implementation(self, values, pointStart, featureStart,
                                 pointEnd, featureEnd):
        """
        """
        if not isinstance(values, UML.data.Base):
            values = values * np.ones((pointEnd - pointStart + 1,
                                       featureEnd - featureStart + 1))
        else:
            #convert values to be array or matrix, instead of pandas DataFrame
            values = values.data.values

        # pandas is exclusive
        pointEnd += 1
        featureEnd += 1
        self.data.iloc[pointStart:pointEnd, featureStart:featureEnd] = values

    def _flattenToOnePoint_implementation(self):
        numElements = len(self.points) * len(self.features)
        self.data = pd.DataFrame(
            self.data.values.reshape((1, numElements), order='C'))

    def _flattenToOneFeature_implementation(self):
        numElements = len(self.points) * len(self.features)
        self.data = pd.DataFrame(
            self.data.values.reshape((numElements, 1), order='F'))

    def _unflattenFromOnePoint_implementation(self, numPoints):
        numFeatures = len(self.features) // numPoints
        self.data = pd.DataFrame(
            self.data.values.reshape((numPoints, numFeatures), order='C'))

    def _unflattenFromOneFeature_implementation(self, numFeatures):
        numPoints = len(self.points) // numFeatures
        self.data = pd.DataFrame(
            self.data.values.reshape((numPoints, numFeatures), order='F'))

    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):

        if point == 'union':
            point = 'outer'
        elif point == 'intersection':
            point = 'inner'
        if self._featureNamesCreated():
            self.data.columns = self.features.getNames()
        tmpDfR = other.data.copy()
        if other._featureNamesCreated():
            tmpDfR.columns = other.features.getNames()

        if feature == 'intersection':
            self.data = self.data.iloc[:, matchingFtIdx[0]]
            tmpDfR = tmpDfR.iloc[:, matchingFtIdx[1]]
            matchingFtIdx[0] = list(range(self.data.shape[1]))
            matchingFtIdx[1] = list(range(tmpDfR.shape[1]))
        elif feature == "left":
            tmpDfR = tmpDfR.iloc[:, matchingFtIdx[1]]
            matchingFtIdx[1] = list(range(tmpDfR.shape[1]))

        numColsL = len(self.data.columns)
        if onFeature is None:
            if self._pointNamesCreated() and other._pointNamesCreated():
                # differentiate default names between objects
                self.data.index = [n + '_l' if n.startswith(DEFAULT_PREFIX)
                                   else n for n in self.points.getNames()]
                tmpDfR.index = [n + '_r' if n.startswith(DEFAULT_PREFIX)
                                else n for n in other.points.getNames()]
            elif self._pointNamesCreated() or other._pointNamesCreated():
                # there will be no matches, need left points ordered first
                self.data.index = [i for i in range(len(self.points))]
                idxRange = range(self.shape[0], self.shape[0] + other.shape[0])
                tmpDfR.index = [i for i in idxRange]
            else:
                # left already has index set to range(len(self.points))
                idxRange = range(self.shape[0], self.shape[0] + other.shape[0])
                tmpDfR.index = [i for i in idxRange]

            self.data = self.data.merge(tmpDfR, how=point, left_index=True,
                                        right_index=True)
            self.data.reset_index(drop=True, inplace=True)
            self.data.columns = range(self.data.shape[1])
        else:
            onIdxL = self.data.columns.get_loc(onFeature)
            self.data = self.data.merge(tmpDfR, how=point, on=onFeature)
            self.data.reset_index()
            self.data.columns = range(self.data.shape[1])

        toDrop = []
        for l, r in zip(matchingFtIdx[0], matchingFtIdx[1]):
            if onFeature and l == onIdxL:
                # onFeature column has already been merged
                continue
            elif onFeature and l > onIdxL:
                # one less to account for merged onFeature
                r = r + numColsL - 1
            else:
                r = r + numColsL
            matches = self.data.iloc[:, l] == self.data.iloc[:, r]
            nansL = np.array([x != x for x in self.data.iloc[:, l]])
            nansR = np.array([x != x for x in self.data.iloc[:, r]])
            acceptableValues = matches + nansL + nansR
            if not all(acceptableValues):
                msg = "The objects contain different values for the same "
                msg += "feature"
                raise ArgumentException(msg)
            if nansL.any():
                self.data.iloc[:, l][nansL] = self.data.iloc[:, r][nansL]
            toDrop.append(r)
        self.data.drop(toDrop, axis=1, inplace=True)

        self._featureCount = (numColsL + len(tmpDfR.columns)
                              - len(matchingFtIdx[1]))
        self._pointCount = len(self.data.index)

    def _getitem_implementation(self, x, y):
        # return self.data.ix[x, y]
        #use self.data.values is much faster
        return self.data.values[x, y]

    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd):
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
        kwds['data'] = self.data.iloc[pointStart:pointEnd,
                                      featureStart:featureEnd]
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
        Returns True if there is a value that is equal to integer 0
        contained in this object. False otherwise.
        """
        return 0 in self.data.values

    def _matrixMultiply_implementation(self, other):
        """
        Matrix multiply this UML Base object against the provided other
        UML Base object. Both object must contain only numeric data.
        The featureCount of the calling object must equal the pointCount
        of the other object. The types of the two objects may be
        different, and the return is guaranteed to be the same type as
        at least one out of the two, to be automatically determined
        according to efficiency constraints.
        """

        leftData = np.matrix(self.data)
        if isinstance(other, UML.data.Sparse):
            rightData = other.data
        else:
            rightData = np.matrix(other.data)

        return DataFrame(leftData * rightData)

    def _scalarMultiply_implementation(self, scalar):
        """
        Multiply every element of this UML Base object by the provided
        scalar. This object must contain only numeric data. The 'scalar'
        parameter must be a numeric data type. The returned object will
        be the inplace modification of the calling object.
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
            if isinstance(other, UML.data.Sparse):
                rightData = other.data
            else:
                rightData = np.matrix(other.data)
        else:
            rightData = other
        leftData += rightData
        return DataFrame(leftData, reuseData=True)

    def _radd__implementation(self, other):
        ret = other + self.data
        return DataFrame(ret, reuseData=True)

    def _iadd__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if isinstance(other, UML.data.Sparse):
                rightData = other.data
            else:
                rightData = np.matrix(other.data)
            ret = np.matrix(self.data) + rightData
        else:
            ret = np.matrix(self.data) + np.matrix(other)
        self.data = pd.DataFrame(ret)
        return self

    def _sub__implementation(self, other):
        leftData = np.matrix(self.data)
        if isinstance(other, UML.data.Base):
            if isinstance(other, UML.data.Sparse):
                rightData = other.data
            else:
                rightData = np.matrix(other.data)
        else:
            rightData = other
        leftData -= rightData

        return DataFrame(leftData, reuseData=True)

    def _rsub__implementation(self, other):
        ret = other - self.data.values
        pNames = self.points.getNames()
        fNames = self.features.getNames()
        return UML.createData('DataFrame', ret, pointNames=pNames,
                              featureNames=fNames, reuseData=True, useLog=False)

    def _isub__implementation(self, other):
        if isinstance(other, UML.data.Base):
            if isinstance(other, UML.data.Sparse):
                rightData = other.data
            else:
                rightData = np.matrix(other.data)
            ret = np.matrix(self.data) - rightData
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
        #----------------------------------------------------------------------
