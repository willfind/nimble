"""
Class extending Base, using a pandas DataFrame to store data.
"""

import itertools

import numpy

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble._utility import inheritDocstringsFactory, is2DArray
from nimble._utility import scipy, pd
from .base import Base
from .views import BaseView
from .dataframeAxis import DataFramePoints, DataFramePointsView
from .dataframeAxis import DataFrameFeatures, DataFrameFeaturesView
from ._dataHelpers import allDataIdentical
from ._dataHelpers import DEFAULT_PREFIX
from ._dataHelpers import createDataNoValidation
from ._dataHelpers import denseCountUnique
from ._dataHelpers import NimbleElementIterator
from ._dataHelpers import convertToNumpyOrder

@inheritDocstringsFactory(Base)
class DataFrame(Base):
    """
    Class providing implementations of data manipulation operations on
    data stored in a pandas DataFrame.

    Parameters
    ----------
    data : object
        pandas DataFrame or two-dimensional numpy array.
    reuseData : bool
        Only used when data is a pandas DataFrame.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """

    def __init__(self, data, reuseData=False, **kwds):
        if not pd.nimbleAccessible():
            msg = 'To use class DataFrame, pandas must be installed.'
            raise PackageException(msg)

        if not isinstance(data, pd.DataFrame) and not is2DArray(data):
            msg = "the input data can only be a pandas DataFrame or a two-"
            msg += "dimensional numpy array."
            raise InvalidArgumentType(msg)

        if isinstance(data, pd.DataFrame):
            if reuseData:
                self.data = data
            else:
                self.data = data.copy()
        else:
            self.data = pd.DataFrame(data, copy=True)

        shape = kwds.get('shape', None)
        if shape is None:
            kwds['shape'] = self.data.shape
        super(DataFrame, self).__init__(**kwds)

    def _getPoints(self):
        return DataFramePoints(self)

    def _getFeatures(self):
        return DataFrameFeatures(self)

    def _transform_implementation(self, toTransform, points, features):
        IDs = itertools.product(range(len(self.points)),
                                range(len(self.features)))
        for i, j in IDs:
            currVal = self.data.values[i, j]

            if points is not None and i not in points:
                continue
            if features is not None and j not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(currVal)
            else:
                currRet = toTransform(currVal, i, j)

            self.data.iloc[i, j] = currRet

    def _calculate_implementation(self, function, points, features,
                                  preserveZeros, outputType):
        return self._calculate_genericVectorized(
            function, points, features, outputType)

    def _countUnique_implementation(self, points, features):
        return denseCountUnique(self, points, features)

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

        return allDataIdentical(self.data.values, other.data.values)

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
                    outFile.write('pointNames')

            if includePointNames:
                self.data.index = self.points.getNames()

        self.data.to_csv(outPath, mode='a', index=includePointNames,
                         header=includeFeatureNames)

        if includePointNames:
            self.data.index = pd.RangeIndex(len(self.data.index))
        if includeFeatureNames:
            self.data.columns = pd.RangeIndex(len(self.data.columns))

    def _writeFileMTX_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
        """
        Function to write the data in this object to a matrix market
        file at the designated path.
        """
        if not scipy.nimbleAccessible():
            msg = "scipy is not available"
            raise PackageException(msg)

        from scipy.io import mmwrite

        comment = '#'
        if includePointNames:
            comment += ','.join(self.points.getNames())
        if includeFeatureNames:
            comment += '\n#' + ','.join(self.features.getNames())
        mmwrite(outPath, self.data.astype(numpy.float), comment=comment)

    def _referenceDataFrom_implementation(self, other):
        if not isinstance(other, DataFrame):
            msg = "Other must be the same type as this object"
            raise InvalidArgumentType(msg)

        self.data = other.data

    def _copy_implementation(self, to):
        """
        Copy the current DataFrame object to one in a specified format.

        to : string
            Sparse, List, Matrix, pythonlist, numpyarray, numpymatrix,
            scipycsc, scipycsr.
        """
        if to in nimble.core.data.available:
            ptNames = self.points._getNamesNoGeneration()
            ftNames = self.features._getNamesNoGeneration()
            if to == 'DataFrame':
                data = self.data.copy()
            else:
                data = self.data.values.copy()
            # reuseData=True since we already made copies here
            return createDataNoValidation(to, data, ptNames, ftNames,
                                          reuseData=True)
        if to == 'pythonlist':
            return self.data.values.tolist()
        needsReshape = len(self._shape) > 2
        if to == 'numpyarray':
            if needsReshape:
                return self.data.values.reshape(self._shape)
            return self.data.values.copy()
        if needsReshape:
            data = numpy.empty(self._shape[:2], dtype=numpy.object_)
            for i in range(self.shape[0]):
                data[i] = self.points[i].copy('pythonlist')
        elif to == 'pandasdataframe':
            data = self.data.copy()
        else:
            data = self.data
        if to == 'numpymatrix':
            return numpy.matrix(data)
        if 'scipy' in to:
            if not scipy.nimbleAccessible():
                msg = "scipy is not available"
                raise PackageException(msg)
            if to == 'scipycoo':
                return scipy.sparse.coo_matrix(data)
            try:
                ret = self.data.values.astype(numpy.float)
            except ValueError:
                msg = 'Can only create scipy {0} matrix from numeric data'
                raise ValueError(msg.format(to[-3:]))
            if to == 'scipycsc':
                return scipy.sparse.csc_matrix(ret)
            if to == 'scipycsr':
                return scipy.sparse.csr_matrix(ret)
        if to == 'pandasdataframe':
            pnames = self.points._getNamesNoGeneration()
            fnames = self.features._getNamesNoGeneration()
            df = pd.DataFrame(data)
            if pnames is not None:
                df.index = pnames
            if fnames is not None:
                df.columns = fnames
            return df

    def _replaceRectangle_implementation(self, replaceWith, pointStart,
                                         featureStart, pointEnd, featureEnd):
        """
        """
        if not isinstance(replaceWith, Base):
            values = replaceWith * numpy.ones((pointEnd - pointStart + 1,
                                               featureEnd - featureStart + 1))
        else:
            #convert values to be array or matrix, instead of pandas DataFrame
            values = replaceWith.data.values

        # pandas is exclusive
        pointEnd += 1
        featureEnd += 1
        self.data.iloc[pointStart:pointEnd, featureStart:featureEnd] = values

    def _flatten_implementation(self, order):
        numElements = len(self.points) * len(self.features)
        order = convertToNumpyOrder(order)
        self.data = pd.DataFrame(self.data.values.reshape((1, numElements),
                                                          order=order))

    def _unflatten_implementation(self, reshape, order):
        order = convertToNumpyOrder(order)
        self.data = pd.DataFrame(self.data.values.reshape(reshape,
                                                          order=order))

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
        else:
            onIdxL = self.data.columns.get_loc(onFeature)
            self.data = self.data.merge(tmpDfR, how=point, on=onFeature)

        # return labels to default after we've executed the merge
        self.data.index = pd.RangeIndex(self.data.shape[0])
        self.data.columns = pd.RangeIndex(self.data.shape[1])

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
            nansL = numpy.array([x != x for x in self.data.iloc[:, l]])
            nansR = numpy.array([x != x for x in self.data.iloc[:, r]])
            acceptableValues = matches + nansL + nansR
            if not all(acceptableValues):
                msg = "The objects contain different values for the same "
                msg += "feature"
                raise InvalidArgumentValue(msg)
            if nansL.any():
                self.data.iloc[:, l][nansL] = self.data.iloc[:, r][nansL]
            toDrop.append(r)

        if toDrop:
            self.data.drop(toDrop, axis=1, inplace=True)
            self.data.columns = pd.RangeIndex(self.data.shape[1])

        self._featureCount = (numColsL + len(tmpDfR.columns)
                              - len(matchingFtIdx[1]))
        self._pointCount = len(self.data.index)

    def _replaceFeatureWithBinaryFeatures_implementation(self, uniqueVals):
        toFill = numpy.zeros((len(self.points), len(uniqueVals)))
        for ptIdx, val in enumerate(self.data.values):
            ftIdx = uniqueVals.index(val)
            toFill[ptIdx, ftIdx] = 1
        return DataFrame(pd.DataFrame(toFill))

    def _getitem_implementation(self, x, y):
        #use self.data.values is much faster
        return self.data.values[x, y]

    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd, dropDimension):
        kwds = {}
        kwds['data'] = self.data.iloc[pointStart:pointEnd,
                                      featureStart:featureEnd]
        kwds['source'] = self
        pRange = pointEnd - pointStart
        fRange = featureEnd - featureStart
        if len(self._shape) > 2:
            if dropDimension:
                shape = self._shape[1:]
                source = self._createNestedObject(pointStart)
                kwds['source'] = source
                kwds['data'] = source.data
                pointStart, pointEnd = 0, source.shape[0]
                featureStart, featureEnd = 0, source.shape[1]
                pRange = source.shape[0]
                fRange = source.shape[1]
            else:
                shape = self._shape.copy()
                shape[0] = pRange
            kwds['shape'] = shape
        kwds['pointStart'] = pointStart
        kwds['pointEnd'] = pointEnd
        kwds['featureStart'] = featureStart
        kwds['featureEnd'] = featureEnd
        kwds['reuseData'] = True

        ret = DataFrameView(**kwds)

        # Reassign labels as to match the positions in the view object,
        # not the positions in the source object.
        ret.data.index = pd.RangeIndex(pRange)
        ret.data.columns = pd.RangeIndex(fRange)

        return ret

    def _createNestedObject(self, pointIndex):
        """
        Create an object of one less dimension
        """
        reshape = (self._shape[1], int(numpy.prod(self._shape[2:])))
        data = self.data.values[pointIndex].reshape(reshape)
        return DataFrame(data, shape=self._shape[1:], reuseData=True)

    def _validate_implementation(self, level):
        shape = self.data.shape
        assert shape[0] == len(self.points)
        assert shape[1] == len(self.features)
        assert all(self.data.index == pd.RangeIndex(len(self.points)))
        assert all(self.data.columns == pd.RangeIndex(len(self.features)))

    def _containsZero_implementation(self):
        """
        Returns True if there is a value that is equal to integer 0
        contained in this object. False otherwise.
        """
        return 0 in self.data.values


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
            ret = getattr(self.data.values, opName)(other.data)
            return DataFrame(ret)
        except (AttributeError, InvalidArgumentType):
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
        if isinstance(other, nimble.core.data.Sparse):
            ret = self.data.values * other.data
        else:
            ret = numpy.matmul(self.data.values, other.copy('numpyarray'))
        return DataFrame(ret)

    def _convertUnusableTypes_implementation(self, convertTo, usableTypes):
        if not all(dtype in usableTypes for dtype in self.data.dtypes):
            return self.data.astype(convertTo)
        return self.data

    def _iterateElements_implementation(self, order, only):
        return NimbleElementIterator(self.data.values, order, only)

class DataFrameView(BaseView, DataFrame):
    """
    Read only access to a DataFrame object.
    """
    def __init__(self, **kwds):
        super(DataFrameView, self).__init__(**kwds)

    def _getPoints(self):
        return DataFramePointsView(self)

    def _getFeatures(self):
        return DataFrameFeaturesView(self)
