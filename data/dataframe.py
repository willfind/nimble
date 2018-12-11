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
from .points import Points
from .features import Features
from .dataframePoints import DataFramePoints
from .dataframeFeatures import DataFrameFeatures
from .dataframeElements import DataFrameElements
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
        return Points(DataFramePoints, self)

    points = property(_getPoints, doc="TODO")

    def _getFeatures(self):
        return Features(DataFrameFeatures, self)

    features = property(_getFeatures, doc="TODO")

    def _getElements(self):
        return DataFrameElements(self)

    elements = property(_getElements, doc="TODO")

    def _transpose_implementation(self):
        """
        Function to transpose the data, ie invert the feature and point indices of the data.

        This is not an in place operation, a new pandas DataFrame is constructed.
        """
        self.data = self.data.T

    def _addPoints_implementation(self, toAdd, insertBefore):
        """
        Insert the points from the toAdd object below the provided index in
        this object, the remaining points from this object will continue below
        the inserted points

        """
        startData = self.data.iloc[:insertBefore, :]
        endData = self.data.iloc[insertBefore:, :]
        self.data = pd.concat((startData, toAdd.data, endData), axis=0)
        self._updateName(axis='point')

    def _addFeatures_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this object
        will continue to the right of the inserted points

        """
        startData = self.data.iloc[:, :insertBefore]
        endData = self.data.iloc[:, insertBefore:]
        self.data = pd.concat((startData, toAdd.data, endData), axis=1)
        self._updateName(axis='feature')

    def _sortPoints_implementation(self, sortBy, sortHelper):
        """
        Modify this object so that the points are sorted using the built in python
        sort on point views. The input arguments are passed to that function unaltered
        This function returns a list of pointNames indicating the new order of the data.

        """
        return self._sort_implementation(sortBy, sortHelper, 'point')

    def _sortFeatures_implementation(self, sortBy, sortHelper):
        """
        Modify this object so that the features are sorted using the built in python
        sort on feature views. The input arguments are passed to that function unaltered
        This function returns a list of featureNames indicating the new order of the data.

        """
        return self._sort_implementation(sortBy, sortHelper, 'feature')

    def _sort_implementation(self, sortBy, sortHelper, axis):
        if axis == 'point':
            test = self.pointView(0)
            viewIter = self.pointIterator()
            indexGetter = self.getPointIndex
            nameGetter = self.getPointName
            nameGetterStr = 'getPointName'
            names = self.getPointNames()
        else:
            test = self.featureView(0)
            viewIter = self.featureIterator()
            indexGetter = self.getFeatureIndex
            nameGetter = self.getFeatureName
            nameGetterStr = 'getFeatureName'
            names = self.getFeatureNames()

        if isinstance(sortHelper, list):
            if axis == 'point':
                self.data = self.data.iloc[sortHelper, :]
            else:
                self.data = self.data.iloc[:, sortHelper]
            newNameOrder = [names[idx] for idx in sortHelper]
            return newNameOrder

        scorer = None
        comparator = None
        try:
            sortHelper(test)
            scorer = sortHelper
        except TypeError:
            pass
        try:
            sortHelper(test, test)
            comparator = sortHelper
        except TypeError:
            pass

        if sortHelper is not None and scorer is None and comparator is None:
            raise ArgumentException("sortHelper is neither a scorer or a comparator")

        if comparator is not None:
            # make array of views
            viewArray = []
            for v in viewIter:
                viewArray.append(v)

            viewArray.sort(key=cmp_to_key(comparator))
            indexPosition = []
            for i in range(len(viewArray)):
                index = indexGetter(getattr(viewArray[i], nameGetterStr)(0))
                indexPosition.append(index)
            indexPosition = np.array(indexPosition)
        elif hasattr(scorer, 'permuter'):
            scoreArray = scorer.indices
            indexPosition = np.argsort(scoreArray)
        else:
            # make array of views
            viewArray = []
            for v in viewIter:
                viewArray.append(v)

            scoreArray = viewArray
            if scorer is not None:
                # use scoring function to turn views into values
                for i in range(len(viewArray)):
                    scoreArray[i] = scorer(viewArray[i])
            else:
                for i in range(len(viewArray)):
                    scoreArray[i] = viewArray[i][sortBy]

            # use numpy.argsort to make desired index array
            # this results in an array whose ith entry contains the the
            # index into the data of the value that should be in the ith
            # position.
            indexPosition = np.argsort(scoreArray)

        # use numpy indexing to change the ordering
        if axis == 'point':
            self.data = self.data.iloc[indexPosition, :]
        else:
            self.data = self.data.iloc[:, indexPosition]

        # we convert the indices of the their previous location into their feature names
        newNameOrder = []
        for i in range(len(indexPosition)):
            oldIndex = indexPosition[i]
            newName = nameGetter(oldIndex)
            newNameOrder.append(newName)
        return newNameOrder

    def _structuralBackend_implementation(self, structure, axis, targetList):
        """
        Backend for extractPoints/Features, deletePoints/Features, retainPoints/Features, and
        copyPoints/Features. Returns a new object containing only the points in targetList and
        performs some modifications to the original object if necessary. This function does not
        perform all of the modification or process how each function handles the returned value,
        these are managed separately by each frontend function.
        """
        if structure == 'copy':
            return self.pointsOrFeaturesVectorized(targetList, axis, 'copy', True)
        else:
            return self.pointsOrFeaturesVectorized(targetList, axis, 'extract', True)


    def _mapReducePoints_implementation(self, mapper, reducer):
        # apply_along_axis() expects a scalar or array of scalars as output,
        # but our mappers output a list of tuples (ie a sequence type)
        # which is not allowed. This packs key value pairs into an array
        def mapperWrapper(point):
            pairs = mapper(point)
            ret = []
            for (k, v) in pairs:
                ret.append(k)
                ret.append(v)
            return np.array(ret)

        mapResultsMatrix = np.apply_along_axis(mapperWrapper, 1, self.data.values)
        mapResults = {}
        for pairsArray in mapResultsMatrix:
            for i in range(len(pairsArray) / 2):
                # pairsArray has key value pairs packed back to back
                k = pairsArray[i * 2]
                v = pairsArray[(i * 2) + 1]
                # if key is new, we must add an empty list
                if k not in mapResults:
                    mapResults[k] = []
                # append this value to the list of values associated with the key
                mapResults[k].append(v)

        # apply the reducer to the list of values associated with each key
        ret = []
        for mapKey in mapResults.keys():
            mapValues = mapResults[mapKey]
            # the reducer will return a tuple of a key to a value
            redRet = reducer(mapKey, mapValues)
            if redRet is not None:
                (redKey, redValue) = redRet
                ret.append([redKey, redValue])
        return UML.createData('DataFrame', ret)

    def _getTypeString_implementation(self):
        return 'DataFrame'


    def _isIdentical_implementation(self, other):
        if not isinstance(other, DataFrame):
            return False
        if self.pts != other.pts:
            return False
        if self.fts != other.fts:
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
                self.data.columns = self.getFeatureNames()
                if includePointNames:
                    outFile.write('point_names')

            if includePointNames:
                    self.data.index = self.getPointNames()

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
            comment += ','.join(self.getPointNames())
        if includeFeatureNames:
            comment += '\n#' + ','.join(self.getFeatureNames())
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


    def _calculateForEachElement_implementation(self, function, points, features,
                                                preserveZeros, outputType):
        return self._calculateForEachElementGenericVectorized(
               function, points, features, outputType)


    def _transformEachPoint_implementation(self, function, points):
        """

        """
        for i, p in enumerate(self.pointIterator()):
            if points is not None and i not in points:
                continue
            currRet = function(p)
            # currRet might return an ArgumentException with a message which needs to be
            # formatted with the axis and current index before being raised
            if isinstance(currRet, ArgumentException):
                currRet.value = currRet.value.format('point', i)
                raise currRet
            if len(currRet) != self.fts:
                msg = "function must return an iterable with as many elements as features in this object"
                raise ArgumentException(msg)

            self.data.iloc[i, :] = currRet

    def _transformEachFeature_implementation(self, function, features):
        for j, f in enumerate(self.featureIterator()):
            if features is not None and j not in features:
                continue
            currRet = function(f)
            # currRet might return an ArgumentException with a message which needs to be
            # formatted with the axis and current index before being raised
            if isinstance(currRet, ArgumentException):
                currRet.value = currRet.value.format('feature', j)
                raise currRet
            if len(currRet) != self.pts:
                msg = "function must return an iterable with as many elements as points in this object"
                raise ArgumentException(msg)

            self.data.iloc[:, j] = currRet

    def _transformEachElement_implementation(self, toTransform, points, features, preserveZeros, skipNoneReturnValues):
        oneArg = False
        try:
            toTransform(0, 0, 0)
        except TypeError:
            if isinstance(toTransform, dict):
                oneArg = None
            else:
                oneArg = True

        IDs = itertools.product(range(self.pts), range(self.fts))
        for (i, j) in IDs:
            currVal = self.data.iloc[i, j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue
            if preserveZeros and currVal == 0:
                continue

            if oneArg is None:
                if currVal in toTransform.keys():
                    currRet = toTransform[currVal]
                else:
                    continue
            elif oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            if skipNoneReturnValues and currRet is None:
                continue

            self.data.iloc[i, j] = currRet

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
        numElements = self.pts * self.fts
        self.data = pd.DataFrame(self.data.values.reshape((1, numElements), order='C'))

    def _flattenToOneFeature_implementation(self):
        numElements = self.pts * self.fts
        self.data = pd.DataFrame(self.data.values.reshape((numElements,1), order='F'))


    def _unflattenFromOnePoint_implementation(self, numPoints):
        numFeatures = self.fts // numPoints
        self.data = pd.DataFrame(self.data.values.reshape((numPoints, numFeatures), order='C'))

    def _unflattenFromOneFeature_implementation(self, numFeatures):
        numPoints = self.pts // numFeatures
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
        assert shape[0] == self.pts
        assert shape[1] == self.fts

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
                self._pStop = source.pts
                self._fIndex = 0
                self._fStop = source.fts

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
                self._pStop = source.pts
                self._fIndex = 0
                self._fStop = source.fts

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
        return UML.createData('DataFrame', ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

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

    def _setName_implementation(self, oldIdentifier, newName, axis, allowDefaults=False):
        super(DataFrame, self)._setName_implementation(oldIdentifier, newName, axis, allowDefaults)
        #update the index or columns in self.data
        self._updateName(axis)

    def _setNamesFromList(self, assignments, count, axis):
        super(DataFrame, self)._setNamesFromList(assignments, count, axis)
        self._updateName(axis)

    def _setNamesFromDict(self, assignments, count, axis):
        super(DataFrame, self)._setNamesFromDict(assignments, count, axis)
        self._updateName(axis)

    def _updateName(self, axis):
        """
        update self.data.index or self.data.columns
        """
        if axis == 'point':
            self.data.index = list(range(len(self.data.index)))
        else:
            # self.data.columns = self.getFeatureNames()
            self.data.columns = list(range(len(self.data.columns)))
        #-----------------------------------------------------------------------


    def pointsOrFeaturesVectorized(self, indexList, axis, funcType, inplace=True):
        """

        """
        df = self.data

        if axis == 0 or axis == 'point':
            ret = df.iloc[indexList, :]
            axis = 0
            name = 'pointNames'
            nameList = [self.getPointName(i) for i in indexList]
            otherName = 'featureNames'
            otherNameList = self.getFeatureNames()
        elif axis == 1 or axis == 'feature':
            ret = df.iloc[:, indexList]
            axis = 1
            name = 'featureNames'
            nameList = [self.getFeatureName(i) for i in indexList]
            otherName = 'pointNames'
            otherNameList = self.getPointNames()
        else:
            msg = 'axis can only be 0,1 or point, feature'
            raise ArgumentException(msg)

        if funcType.lower() == "extract":
            df.drop(indexList, axis=axis, inplace=inplace)

        if axis == 0:
            df.index = numpy.arange(len(df.index), dtype=df.index.dtype)
        else:
            df.columns = numpy.arange(len(df.columns), dtype=df.columns.dtype)

        return UML.createData('DataFrame', ret, **{name: nameList, otherName: otherNameList})
