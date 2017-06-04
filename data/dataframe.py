"""
Class extending Base, using a pandas DataFrame to store data.
"""
import UML
from UML.exceptions import ArgumentException, PackageException
from base import Base
import pandas as pd
import numpy as np
try:
	import scipy.sparse
	from scipy.sparse import isspmatrix
	scipyImported = True
except ImportError:
	scipyImported = False
import itertools
from base_view import BaseView

class DataFrame(Base):
	"""
	Class providing implementations of data manipulation operations on data stored
	in a pandas DataFrame.
	"""

	def __init__(self, data, reuseData=False, **kwds):
		"""
		The initializer.
		Inputs:
			data: pandas DataFrame, list, list of list, scipy sparse, numpy array or numpy matrix.
			reuseData: boolean. only used when data is a pandas DataFrame.
		"""
		if isinstance(data, pd.DataFrame):
			if reuseData:
				self.data = data
			else:
				self.data = data.copy()
		elif type(data) in [list, np.ndarray, np.matrix, tuple, pd.Series]:
			if np.array(data).ndim > 2:
				msg = "the input data can be list such as [1,2,3], list of list such as [[1,2,3],[3,2,1]], or 2D data in \
					  numpy array or matrix. the current input has ndim > 2."
				raise ArgumentException(msg)
			#np.matrix can convert 1D data to 2D, such as [1,2,3] to matrix([[1,2,3]])
			self.data = pd.DataFrame(np.matrix(data))
		elif isspmatrix(data):
			self.data = pd.DataFrame(data.todense())
		else:
			msg = "the input data can only be pandas DataFrame, list, list of list, scipy sparse, numpy array or numpy matrix"
			raise ArgumentException(msg)

		kwds['shape'] = self.data.shape
		super(DataFrame, self).__init__(**kwds)
		#it is very import to set up self.data's index and columns, other wise int index or column name will be set
		#if so, pandas DataFrame ix sliding is label based, its behaviour is not what we want
		self.data.index = self.getPointNames()
		self.data.columns = self.getFeatureNames()

	def _transpose_implementation(self):
		"""
		Function to transpose the data, ie invert the feature and point indices of the data.

		This is not an in place operation, a new pandas DataFrame is constructed.
		"""
		self.data = self.data.T

	def appendPoints(self, toAppend):
		super(DataFrame, self).appendPoints(toAppend)
		self._updateName(axis='point')

	def _appendPoints_implementation(self, toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object

		"""
		self.data = pd.concat((self.data, toAppend.data), axis=0)

	def appendFeatures(self, toAppend):
		super(DataFrame, self).appendFeatures(toAppend)
		self._updateName(axis='feature')

	def _appendFeatures_implementation(self, toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		"""
		self.data = pd.concat((self.data, toAppend.data), axis=1)

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
		else:
			test = self.featureView(0)
			viewIter = self.featureIterator()
			indexGetter = self.getFeatureIndex
			nameGetter = self.getFeatureName
			nameGetterStr = 'getFeatureName'
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

			viewArray.sort(cmp=comparator)
			indexPosition = []
			for i in xrange(len(viewArray)):
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
				for i in xrange(len(viewArray)):
					scoreArray[i] = scorer(viewArray[i])
			else:
				for i in xrange(len(viewArray)):
					scoreArray[i] = viewArray[i][sortBy]

			# use numpy.argsort to make desired index array
			# this results in an array whose ith entry contains the the
			# index into the data of the value that should be in the ith
			# position.
			indexPosition = np.argsort(scoreArray)

		# use numpy indexing to change the ordering
		if axis == 'point':
			self.data = self.data.ix[indexPosition, :]
		else:
			self.data = self.data.ix[:, indexPosition]

		# we convert the indices of the their previous location into their feature names
		newNameOrder = []
		for i in xrange(len(indexPosition)):
			oldIndex = indexPosition[i]
			newName = nameGetter(oldIndex)
			newNameOrder.append(newName)
		return newNameOrder

	def _extractPoints_implementation(self, toExtract, start, end, number, randomize):
		"""
		Function to extract points according to the parameters, and return an object containing
		the removed points with default names. The actual work is done by further helper
		functions, this determines which helper to call, and modifies the input to accomodate
		the number and randomize parameters, where number indicates how many of the possibilities
		should be extracted, and randomize indicates whether the choice of who to extract should
		be by order or uniform random.

		"""
		# list of identifiers
		if isinstance(toExtract, list):
			assert number == len(toExtract)
			assert not randomize
			return self._extractPointsByList_implementation(toExtract)
		# boolean function
		elif hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError  # TODO randomize in the extractPointByFunction case
			else:
				return self._extractPointsByFunction_implementation(toExtract, number)
		# by range
		elif start is not None or end is not None:
			return self._extractPointsByRange_implementation(start, end)
		else:
			msg = "Malformed or missing inputs"
			raise ArgumentException(msg)

	def _extractPointsByList_implementation(self, toExtract):
		"""
		Modify this object to have only the points that are not given in the input,
		returning an object containing those points that are.

		"""
		indexList = self.data.index[toExtract]
		return self.extractPointsOrFeaturesVectorized(indexList, 'point', True)

	def _extractPointsByFunction_implementation(self, toExtract, number):
		"""
		Modify this object to have only the points that do not satisfy the given function,
		returning an object containing those points that do.

		"""
		if hasattr(toExtract, 'vectorized') and toExtract.vectorized:
			indexList = self.data.index[toExtract(self.data)]
			return self.extractPointsOrFeaturesVectorized(indexList, 'point', True)
		else:
			results = UML.data.matrix.viewBasedApplyAlongAxis(toExtract, 'point', self)
			results = results.astype(np.int)

			# need to convert our 1/0 array to to list of points to be removed
			# can do this by just getting the non-zero indices
			toRemove = np.flatnonzero(results)

			return self._extractPointsByList_implementation(toRemove)


	def _extractPointsByRange_implementation(self, start, end):
		"""
		Modify this object to have only those points that are not within the given range,
		inclusive; returning an object containing those points that are.

		"""
		# +1 on end in ranges, because our ranges are inclusive
		indexList = self.data.index[start:end+1]
		return self.extractPointsOrFeaturesVectorized(indexList, 'point', True)

	def _extractFeatures_implementation(self, toExtract, start, end, number, randomize):
		"""
		Function to extract features according to the parameters, and return an object containing
		the removed features with their featureName names from this object. The actual work is done by
		further helper functions, this determines which helper to call, and modifies the input
		to accomodate the number and randomize parameters, where number indicates how many of the
		possibilities should be extracted, and randomize indicates whether the choice of who to
		extract should be by order or uniform random.

		"""
		# list of identifiers
		if isinstance(toExtract, list):
			assert number == len(toExtract)
			assert not randomize
			return self._extractFeaturesByList_implementation(toExtract)
		# boolean function
		elif hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError  # TODO
			else:
				return self._extractFeaturesByFunction_implementation(toExtract, number)
		# by range
		elif start is not None or end is not None:
			return self._extractFeaturesByRange_implementation(start, end)
		else:
			raise ArgumentException("Malformed or missing inputs")

	def _extractFeaturesByList_implementation(self, toExtract):
		"""
		Modify this object to have only the features that are not given in the input,
		returning an object containing those features that are.

		"""
		featureList = self.data.columns[toExtract]
		return self.extractPointsOrFeaturesVectorized( featureList, 'feature', True)

	def _extractFeaturesByFunction_implementation(self, toExtract, number):
		"""
		Modify this object to have only the features whose views do not satisfy the given
		function, returning an object containing those features whose views do.

		"""
		if hasattr(toExtract, 'vectorized') and toExtract.vectorized:
			featureList = self.data.columns[toExtract(self.data.loc)]
			return self.extractPointsOrFeaturesVectorized(featureList, 'feature', True)
		else:
			#have to use view based method.
			results = UML.data.matrix.viewBasedApplyAlongAxis(toExtract, 'feature', self)
			results = results.astype(np.int)

			# need to convert our 1/0 array to to list of points to be removed
			# can do this by just getting the non-zero indices
			toRemove = np.flatnonzero(results)

			return self._extractFeaturesByList_implementation(toRemove)


	def _extractFeaturesByRange_implementation(self, start, end):
		"""
		Modify this object to have only those features that are not within the given range,
		inclusive; returning an object containing those features that are.

		start and end must not be null, must be within the range of possible features,
		and start must not be greater than end

		"""
		# +1 on end in ranges, because our ranges are inclusive
		featureList = self.data.columns[start:end+1]
		return self.extractPointsOrFeaturesVectorized(featureList, 'feature', True)

	def _mapReducePoints_implementation(self, mapper, reducer):
			# apply_along_axis() expects a scalar or array of scalars as output,
			# but our mappers output a list of tuples (ie a sequence type)
			# which is not allowed. This packs key value pairs into an array
			def mapperWrapper(point):
				pairs = mapper(point)
				ret = []
				for (k,v) in pairs:
					ret.append(k)
					ret.append(v)
				return np.array(ret)

			mapResultsMatrix = np.apply_along_axis(mapperWrapper, 1, self.data.values)
			mapResults = {}
			for pairsArray in mapResultsMatrix:
				for i in xrange(len(pairsArray)/2):
					# pairsArray has key value pairs packed back to back
					k = pairsArray[i * 2]
					v = pairsArray[(i * 2) +1]
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
			return DataFrame(ret)

	def _getTypeString_implementation(self):
		return 'DataFrame'


	def _isIdentical_implementation(self, other):
		if not isinstance(other, DataFrame):
			return False
		if self.pointCount != other.pointCount:
			return False
		if self.featureCount != other.featureCount:
			return False

		try:
			np.testing.assert_equal(self.data.values, other.data.values)
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
				outFile.write('\n\n')
				if includePointNames:
					outFile.write('point_names')
		self.data.to_csv(outPath, mode='a', index=includePointNames, header=includeFeatureNames)

	def _writeFileMTX_implementation(self, outPath, includePointNames, includeFeatureNames):
		"""
		Function to write the data in this object to a matrix market file at the designated
		path.
		"""
		if not scipyImported:
			msg = "scipy is not available"
			raise PackageException(msg)

		from scipy.io import mmwrite
		comment = '#'
		if includePointNames:
			comment += ','.join(self.data.index)
		if includeFeatureNames:
			comment += '\n#'+','.join(self.data.columns)
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
		if format == 'Sparse':
			return UML.data.Sparse(dataArray, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
		if format == 'List':
			return UML.data.List(dataArray, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
		if format == 'Matrix':
			return UML.data.Matrix(dataArray, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
		if format == 'pythonlist':
			return dataArray.tolist()
		if format == 'numpyarray':
			return dataArray
		if format == 'numpymatrix':
			return np.matrix(dataArray)
		if format == 'scipycsc':
			if not scipyImported:
				msg = "scipy is not available"
				raise PackageException(msg)
			return scipy.sparse.csc_matrix(dataArray)
		if format == 'scipycsr':
			if not scipyImported:
				msg = "scipy is not available"
				raise PackageException(msg)
			return scipy.sparse.csr_matrix(dataArray)

		return DataFrame(dataArray, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())

	def _copyPoints_implementation(self, points, start, end):
		if points is not None:
			#indexList = self.data.index[points]
			#ret = self.data.ix[indexList, :]
			ret = self.data.ix[points, :]
		else:
			ret = self.data.ix[start:end+1, :]

		return DataFrame(ret)

	def _copyFeatures_implementation(self, indices, start, end):
		if indices is not None:
			ret = self.data.ix[:, indices]
		else:
			ret = self.data.ix[:, start:end+1]

		return DataFrame(ret)

	def _transformEachPoint_implementation(self, function, points):
		"""

		"""
		for i, p in enumerate(self.pointIterator()):
			if points is not None and i not in points:
				continue
			currRet = function(p)
			if len(currRet) != self.featureCount:
				msg = "function must return an iterable with as many elements as features in this object"
				raise ArgumentException(msg)

			self.data.ix[i, :] = currRet

	def _transformEachFeature_implementation(self, function, features):
		for j, f in enumerate(self.featureIterator()):
			if features is not None and j not in features:
				continue
			currRet = function(f)
			if len(currRet) != self.pointCount:
				msg = "function must return an iterable with as many elements as points in this object"
				raise ArgumentException(msg)

			self.data.ix[:, j] = currRet

	def _transformEachElement_implementation(self, function, points, features, preserveZeros, skipNoneReturnValues):
		oneArg = False
		try:
			function(0, 0, 0)
		except TypeError:
			oneArg = True

		IDs = itertools.product(xrange(self.pointCount), xrange(self.featureCount))
		for (i, j) in IDs:
			currVal = self.data.ix[i, j]

			if points is not None and i not in points:
				continue
			if features is not None and j not in features:
				continue
			if preserveZeros and currVal == 0:
				continue

			if oneArg:
				currRet = function(currVal)
			else:
				currRet = function(currVal, i, j)

			if skipNoneReturnValues and currRet is None:
				continue

			self.data.ix[i, j] = currRet

	def _fillWith_implementation(self, values, pointStart, featureStart, pointEnd, featureEnd):
		"""
		"""
		if not isinstance(values, UML.data.Base):
			values = values * np.ones((pointEnd-pointStart+1, featureEnd-featureStart+1))
		else:
			#convert values to be array or matrix, instead of pandas DataFrame
			values = values.data.values

		self.data.ix[pointStart:pointEnd+1, featureStart:featureEnd+1] = values

	def _getitem_implementation(self, x, y):
		return self.data.ix[x, y]

	def _view_implementation(self, pointStart, pointEnd, featureStart, featureEnd):
		"""

		"""
		class DataFrameView(BaseView, DataFrame):
			def __init__(self, **kwds):
				super(DataFrameView, self).__init__(**kwds)

		kwds = {}
		kwds['data'] = self.data.ix[pointStart:pointEnd, featureStart:featureEnd]
		kwds['source'] = self
		kwds['pointStart'] = pointStart
		kwds['pointEnd'] = pointEnd
		kwds['featureStart'] = featureStart
		kwds['featureEnd'] = featureEnd
		kwds['reuseData'] = True

		return DataFrameView(**kwds)

	def _validate_implementation(self, level):
		shape = self.data.shape
		assert shape[0] == self.pointCount
		assert shape[1] == self.featureCount


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
				self._pStop = source.pointCount
				self._fIndex = 0
				self._fStop = source.featureCount

			def __iter__(self):
				return self

			def next(self):
				while (self._pIndex < self._pStop):
					value = self._source.data.ix[self._pIndex, self._fIndex]

					self._fIndex += 1
					if self._fIndex >= self._fStop:
						self._fIndex = 0
						self._pIndex += 1

					if value != 0:
						return value

				raise StopIteration

		return nzIt(self)

	def _nonZeroIteratorFeatureGrouped_implementation(self):
		class nzIt(object):
			def __init__(self, source):
				self._source = source
				self._pIndex = 0
				self._pStop = source.pointCount
				self._fIndex = 0
				self._fStop = source.featureCount

			def __iter__(self):
				return self

			def next(self):
				while (self._fIndex < self._fStop):
					value = self._source.data.ix[self._pIndex, self._fIndex]

					self._pIndex += 1
					if self._pIndex >= self._pStop:
						self._pIndex = 0
						self._fIndex += 1

					if value != 0:
						return value

				raise StopIteration

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
		rightData = other.data.todense() if isinstance(other, UML.data.Sparse) else np.matrix(other.data)
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
			self.data = pd.DataFrame(other.data.multiply(self.data.values))
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
		rightData = other.data.todense() if isinstance(other, UML.data.Sparse) else np.matrix(other.data)
		ret = leftData + rightData

		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

	def _radd__implementation(self, other):
		ret = other + self.data.values
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

	def _iadd__implementation(self, other):
		if isinstance(other, UML.data.Base):
			ret = self.data + other.data.todense() if isinstance(other, UML.data.Sparse) else np.matrix(other.data)
		else:
			ret = self.data + np.matrix(other)
		self.data = ret
		return self

	def _sub__implementation(self, other):
		leftData = np.matrix(self.data)
		rightData = other.data.todense() if isinstance(other, UML.data.Sparse) else np.matrix(other.data)
		ret = leftData - rightData

		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

	def _rsub__implementation(self, other):
		ret = other - self.data.values
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

	def _isub__implementation(self, other):
		if isinstance(other, UML.data.Base):
			ret = self.data - other.data.todense() if isinstance(other, UML.data.Sparse) else np.matrix(other.data)
		else:
			ret = self.data - np.matrix(other)
		self.data = ret
		return self

	def _div__implementation(self, other):
		if isinstance(other, UML.data.Base):
			if scipyImported and scipy.sparse.isspmatrix(other.data):
				ret = self.data.values / other.data.todense()
			else:
				ret = self.data.values / other.data
		else:
			ret = self.data.values / other
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)


	def _rdiv__implementation(self, other):
		ret = other / self.data.values
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

	def _idiv__implementation(self, other):
		if isinstance(other, UML.data.Base):
			if scipyImported and scipy.sparse.isspmatrix(other.data):
				ret = self.data / other.data.todense()
			else:
				ret = self.data / np.matrix(other.data)
		else:
			ret = self.data / np.matrix(other)
		self.data = ret
		return self

	def _truediv__implementation(self, other):
		if isinstance(other, UML.data.Base):
			if scipyImported and scipy.sparse.isspmatrix(other.data):
				ret = self.data.values.__truediv__(other.data.todense())
			else:
				ret = self.data.values.__truediv__(other.data)
		else:
			ret = self.data.values.__itruediv__(other)
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

	def _rtruediv__implementation(self, other):
		ret = self.data.values.__rtruediv__(other)
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

	def _itruediv__implementation(self, other):
		if isinstance(other, UML.data.Base):
			if scipyImported and scipy.sparse.isspmatrix(other.data):
				ret = self.data.__itruediv__(other.data.todense())
			else:
				ret = self.data.__itruediv__(np.matrix(other.data))
		else:
			ret = self.data.__itruediv__(np.matrix(other))
		self.data = ret
		return self

	def _floordiv__implementation(self, other):
		if isinstance(other, UML.data.Base):
			if scipyImported and scipy.sparse.isspmatrix(other.data):
				ret = self.data.values // other.data.todense()
			else:
				ret = self.data.values // other.data
		else:
			ret = self.data.values // other
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)


	def _rfloordiv__implementation(self, other):
		ret = other // self.data.values
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

	def _ifloordiv__implementation(self, other):
		if isinstance(other, UML.data.Base):
			if scipyImported and scipy.sparse.isspmatrix(other.data):
				ret = self.data // other.data.todense()
			else:
				ret = self.data // np.matrix(other.data)
		else:
			ret = self.data // np.matrix(other)
		self.data = ret
		return self

	def _mod__implementation(self, other):
		if isinstance(other, UML.data.Base):
			if scipyImported and scipy.sparse.isspmatrix(other.data):
				ret = self.data.values % other.data.todense()
			else:
				ret = self.data.values % other.data
		else:
			ret = self.data.values % other
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)


	def _rmod__implementation(self, other):
		ret = other % self.data.values
		return DataFrame(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)


	def _imod__implementation(self, other):
		if isinstance(other, UML.data.Base):
			if scipyImported and scipy.sparse.isspmatrix(other.data):
				ret = self.data % other.data.todense()
			else:
				ret = self.data % np.matrix(other.data)
		else:
			ret = self.data % np.matrix(other)
		self.data = ret
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
			self.data.index = self.getPointNames()
		else:
			self.data.columns = self.getFeatureNames()
#-----------------------------------------------------------------------


	def extractPointsOrFeaturesVectorized(self, nameList, axis, inplace=True):
		"""

		"""
		df = self.data
		nameList = list(nameList)
		if axis == 0 or axis == 'point':
			ret = df.ix[nameList, :]
			name = 'pointNames'
			axis = 0
			otherName = 'featureNames'
			otherNameList = self.getFeatureNames()
		elif axis == 1 or axis == 'feature':
			ret = df.ix[:, nameList]
			name = 'featureNames'
			axis = 1
			otherName = 'pointNames'
			otherNameList = self.getPointNames()
		else:
			msg = 'axis can only be 0,1 or point, feature'
			raise ArgumentException(msg)

		df.drop(nameList, axis=axis, inplace=inplace)

		return DataFrame(ret, **{name:nameList, otherName:otherNameList})
	