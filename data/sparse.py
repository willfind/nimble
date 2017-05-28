"""
Class extending Base, defining an object to hold and manipulate a scipy coo_matrix.

"""


import numpy
import scipy

from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import copy

import UML.data
import dataHelpers
from base import Base
from base_view import BaseView
from dataHelpers import View
from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException
from UML.randomness import pythonRandom
try:
	import pandas as pd
	pdImported = True
except ImportError:
	pdImported = False


class Sparse(Base):

	def __init__(self, data, pointNames=None, featureNames=None,
				reuseData=False, **kwds):
		#convert tuple, pandas Series and numpy ndarray to numpy matrix
		if isinstance(data, tuple) or (pdImported and isinstance(data, pd.Series)) or isinstance(data, numpy.ndarray):
			data = numpy.matrix(data)

		self._sorted = None
		if hasattr(data,'shape') and (data.shape[0] == 0 or data.shape[1] == 0):
			if isinstance(data, CooWithEmpty):
				data = data.internal
			else:
				rowShape = 0
				colShape = 0
				if hasattr(data,'shape'):
					rowShape = data.shape[0] 
					colShape = data.shape[1]
				if featureNames is not None and colShape == 0:
					colShape = len(featureNames)
				if pointNames is not None and rowShape == 0:
					rowShape = len(pointNames)
				
				data = numpy.empty(shape=(rowShape,colShape))
			
			data = numpy.matrix(data, dtype=numpy.float)
			self._data = CooWithEmpty(data)
		elif isinstance(data, list) and data == []:
			rowShape = 0
			colShape = 0

			if featureNames is not None and colShape == 0:
				colShape = len(featureNames)
			if pointNames is not None and rowShape == 0:
				rowShape = len(pointNames)

			data = numpy.empty(shape=(rowShape,colShape))
			data = numpy.matrix(data, dtype=numpy.float)
			self._data = CooWithEmpty(data)
		elif isinstance(data, CooWrapper):
			if reuseData:
				self._data = data
			else:
				self._data = CooWithEmpty(data)
		else:
			if scipy.sparse.isspmatrix(data):
				if reuseData:
					self._data = CooWithEmpty(data, reuseData=True)
				else:
					self._data = CooWithEmpty(data.copy())
			else:
				self._data = CooWithEmpty(data)

		kwds['shape'] = scipy.shape(self._data)
		kwds['pointNames'] = pointNames
		kwds['featureNames'] = featureNames
		super(Sparse, self).__init__(**kwds)


	def getdata(self):
		return self._data.internal
	data = property(getdata)


	def _features_implementation(self):
		(points, cols) = scipy.shape(self._data)
		return cols

	def _points_implementation(self):
		(points, cols) = scipy.shape(self._data)
		return points


	def pointIterator(self):
		self._sortInternal('point')

		class pointIt():
			def __init__(self, outer):
				self._outer = outer
				self._nextID = 0
				self._stillSorted = True
				self._sortedPosition = 0
			def __iter__(self):
				return self
			def next(self):
				if self._nextID >= self._outer.pointCount:
					raise StopIteration
				if self._outer._sorted != "point" or not self._stillSorted:
#					print "actually called"
					self._stillSorted = False
					value = self._outer.pointView(self._nextID)	
				else:
					end = self._sortedPosition
					#this ensures end is always in range, and always exclusive
					while (end < len(self._outer._data.data) and self._outer._data.row[end] == self._nextID):
						end += 1
					value = self._outer.pointView(self._nextID)
					self._sortedPosition = end
				self._nextID += 1
				return value

		return pointIt(self)


	def featureIterator(self):
		self._sortInternal('feature')

		class featureIt():
			def __init__(self, outer):
				self._outer = outer
				self._nextID = 0
				self._stillSorted = True
				self._sortedPosition = 0
			def __iter__(self):
				return self
			def next(self):
				if self._nextID >= self._outer.featureCount:
					raise StopIteration
				if self._outer._sorted != "feature" or not self._stillSorted:
#					print "actually called"

					self._stillSorted = False
					value = self._outer.featureView(self._nextID)	
				else:
					end = self._sortedPosition
					#this ensures end is always in range, and always exclusive
					while (end < len(self._outer._data.data) and self._outer._data.col[end] == self._nextID):
						end += 1
					value = self._outer.featureView(self._nextID)
					self._sortedPosition = end
				self._nextID += 1
				return value
		return featureIt(self)

	def plot(self, outPath=None, includeColorbar=False):
		toPlot = self.copyAs("Matrix")
		toPlot.plot(outPath, includeColorbar)

	def _plot(self, outPath=None, includeColorbar=False):
		toPlot = self.copyAs("Matrix")
		return toPlot._plot(outPath, includeColorbar)

	def _appendPoints_implementation(self, toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object
		
		"""
		newData = numpy.append(self._data.data, toAppend._data.data)
		newRow = numpy.append(self._data.row, toAppend._data.row)
		newCol = numpy.append(self._data.col, toAppend._data.col)

		# correct the row entries
		offset = self.pointCount
		toAdd = numpy.ones(len(newData) - len(self._data.data), dtype=newRow.dtype) * offset
		newRow[len(self._data.data):] += toAdd
		
		numNewRows = self.pointCount + toAppend.pointCount
		self._data = CooWithEmpty((newData,(newRow,newCol)),shape=(numNewRows,self.featureCount))
		if self._sorted == 'feature':
			self._sorted = None


	def _appendFeatures_implementation(self,toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		"""
		newData = numpy.append(self._data.data, toAppend._data.data)
		newRow = numpy.append(self._data.row, toAppend._data.row)
		newCol = numpy.append(self._data.col, toAppend._data.col)

		# correct the col entries
		offset = self.featureCount
		toAdd = numpy.ones(len(newData) - len(self._data.data), dtype=newCol.dtype) * offset
		newCol[len(self._data.data):] += toAdd
		
		numNewCols = self.featureCount + toAppend.featureCount
		self._data = CooWithEmpty((newData,(newRow,newCol)),shape=(self.pointCount,numNewCols))
		if self._sorted == 'point':
			self._sorted = None


	def _sortPoints_implementation(self, sortBy, sortHelper):
		indices = self.sort_general_implementation(sortBy, sortHelper, 'point')
		self._sorted = None
		return indices


	def _sortFeatures_implementation(self, sortBy, sortHelper):
		indices = self.sort_general_implementation(sortBy, sortHelper, 'feature')
		self._sorted = None
		return indices


	def sort_general_implementation(self, sortBy, sortHelper, axisType):
		scorer = None
		comparator = None
		if axisType == 'point':
			viewMaker = self.pointView
			getViewIter = self.pointIterator
			targetAxis = self._data.row
			indexGetter = self.getPointIndex
			nameGetter = self.getPointName
			nameGetterStr = 'getPointName'
		else:
			viewMaker = self.featureView
			targetAxis = self._data.col
			getViewIter = self.featureIterator
			indexGetter = self.getFeatureIndex
			nameGetter = self.getFeatureName
			nameGetterStr = 'getFeatureName'

		test = viewMaker(0)
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
			viewIter = getViewIter()
			for v in viewIter:
				viewArray.append(v)

			viewArray.sort(cmp=comparator)
			indexPosition = []
			for i in xrange(len(viewArray)):
				index = indexGetter(getattr(viewArray[i], nameGetterStr)(0))
				indexPosition.append(index)
			indexPosition = numpy.array(indexPosition)
		elif hasattr(scorer, 'permuter'):
			scoreArray = scorer.indices
			indexPosition = numpy.argsort(scoreArray)
		else:
			# make array of views
			viewArray = []
			viewIter = getViewIter()
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
			indexPosition = numpy.argsort(scoreArray)

		# since we want to access with with positions in the original
		# data, we reverse the 'map'
		reverseIndexPosition = numpy.empty(indexPosition.shape[0])
		for i in xrange(indexPosition.shape[0]):
			reverseIndexPosition[indexPosition[i]] = i

		if axisType == 'point':
			self._data.row[:] = reverseIndexPosition[self._data.row]
		else:
			self._data.col[:] = reverseIndexPosition[self._data.col]

		# we need to return an array of the feature names in their new order.
		# we convert the indices of the their previous location into their names
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
			return self._extractByList_implementation(toExtract, 'point')
		# boolean function
		elif hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError  # TODO
			else:
				return self._extractByFunction_implementation(toExtract, number, 'point')
		# by range
		elif start is not None or end is not None:
			return self._extractByRange_implementation(start, end, 'point')
		else:
			raise ArgumentException("Malformed or missing inputs")


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
			return self._extractByList_implementation(toExtract, 'feature')
		# boolean function
		elif hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError  # TODO
			else:
				return self._extractByFunction_implementation(toExtract, number, 'feature')
		# by range
		elif start is not None or end is not None:
			return self._extractByRange_implementation(start, end, 'feature')
		else:
			raise ArgumentException("Malformed or missing inputs")


	def _extractByList_implementation(self, toExtract, axisType):
		extractLength = len(toExtract)
		extractData = []
		extractRows = []
		extractCols = []

		self._sortInternal(axisType)
		if axisType == "feature":
			targetAxis = self._data.col
			otherAxis = self._data.row
			extractTarget = extractCols
			extractOther = extractRows
		else:
			targetAxis = self._data.row
			otherAxis = self._data.col
			extractTarget = extractRows
			extractOther = extractCols

		#List of rows or columns to extract must be sorted in ascending order
		toExtractSorted = copy.copy(toExtract)
		toExtractSorted.sort()

		# need mapping from values in sorted list to index in nonsorted list
		positionMap = {}
		for i in xrange(len(toExtract)):
			positionMap[toExtract[i]] = i
		
		#walk through col listing and partition all data: extract, and kept, reusing the sparse matrix
		# underlying structure to save space
		copyIndex = 0
		extractIndex = 0
		for i in xrange(len(self._data.data)):
			value = targetAxis[i]
			# Move extractIndex forward until we get to an entry that might
			# match the current (or a future) value
			while extractIndex < extractLength and value > toExtractSorted[extractIndex]:
				extractIndex = extractIndex + 1
			
			# Check if the current value matches one we want extracted
			if extractIndex < extractLength and value == toExtractSorted[extractIndex]:
				extractData.append(self._data.data[i])
				extractOther.append(otherAxis[i])
				extractTarget.append(positionMap[value])	
			# Not extracted. Copy / pack to front of arrays, adjusting for
			# those already extracted
			else:
				self._data.data[copyIndex] = self._data.data[i]				
				otherAxis[copyIndex] = otherAxis[i]
				targetAxis[copyIndex] = targetAxis[i] - extractIndex
				copyIndex = copyIndex + 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		(selfShape, extShape) = _calcShapes(self._data.shape, extractLength, axisType)
		self._data = CooWithEmpty( (self._data.data[0:copyIndex],(self._data.row[0:copyIndex],self._data.col[0:copyIndex])), selfShape)

		# instantiate return data
		ret = CooWithEmpty((extractData,(extractRows,extractCols)),shape=extShape)
		
		# get names for return obj
		pnames = []
		fnames = []
		if axisType == 'point':
			for index in toExtract:
				pnames.append(self.getPointName(index))
			fnames = self.getFeatureNames()
		else:
			pnames = self.getPointNames()
			for index in toExtract:
				fnames.append(self.getFeatureName(index))


		return Sparse(ret, pointNames=pnames, featureNames=fnames, reuseData=True) 


	def _extractByFunction_implementation(self, toExtract, number, axisType):
		extractData = []
		extractRows = []
		extractCols = []

		self._sortInternal(axisType)
		if axisType == "feature":
			targetAxis = self._data.col
			otherAxis = self._data.row
			extractTarget = extractCols
			extractOther = extractRows
			viewMaker = self.featureView
		else:
			targetAxis = self._data.row
			otherAxis = self._data.col
			extractTarget = extractRows
			extractOther = extractCols
			viewMaker = self.pointView

		copyIndex = 0
		vectorStartIndex = 0
		extractedIDs = []
		# consume zeroed vectors, up to the first nonzero value
		if self._data.nnz > 0 and self._data.data[0] != 0:
			for i in xrange(0,targetAxis[0]):
				if toExtract(viewMaker(i)):
					extractedIDs.append(i)
		#walk through each coordinate entry
		for i in xrange(len(self._data.data)):
			#collect values into points
			targetValue = targetAxis[i]
			#if this is the end of a point
			if i == len(self._data.data)-1 or targetAxis[i+1] != targetValue:
				#evaluate whether curr point is to be extracted or not
				# and perform the appropriate copies
				if toExtract(viewMaker(targetValue)):
					for j in xrange(vectorStartIndex, i + 1):
						extractData.append(self._data.data[j])
						extractOther.append(otherAxis[j])
						extractTarget.append(len(extractedIDs))
					extractedIDs.append(targetValue)
				else:
					for j in xrange(vectorStartIndex, i + 1):
						self._data.data[copyIndex] = self._data.data[j]				
						otherAxis[copyIndex] = otherAxis[j]
						targetAxis[copyIndex] = targetAxis[j] - len(extractedIDs)
						copyIndex = copyIndex + 1
				# process zeroed vectors up to the ID of the new vector
				if i < len(self._data.data) - 1:
					nextValue = targetAxis[i+1]
				else:
					if axisType == 'point':
						nextValue = self.pointCount
					else:
						nextValue = self.featureCount
				for j in xrange(targetValue+1, nextValue):
					if toExtract(viewMaker(j)):
						extractedIDs.append(j)

				#reset the vector starting index
				vectorStartIndex = i + 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		(selfShape, extShape) = _calcShapes(self._data.shape, len(extractedIDs), axisType)
		self._data = CooWithEmpty( (self._data.data[0:copyIndex],(self._data.row[0:copyIndex],self._data.col[0:copyIndex])), selfShape)

		# instantiate return data
		ret = CooWithEmpty((extractData,(extractRows,extractCols)),shape=extShape)
		
		# get names for return obj
		pnames = []
		fnames = []
		if axisType == 'point':
			for index in extractedIDs:
				pnames.append(self.getPointName(index))
			fnames = self.getFeatureNames()
		else:
			pnames = self.getPointNames()
			for index in extractedIDs:
				fnames.append(self.getFeatureName(index))

		return Sparse(ret, pointNames=pnames, featureNames=fnames, reuseData=True) 


	def _extractByRange_implementation(self, start, end, axisType):
		rangeLength = end - start + 1
		extractData = []
		extractRows = []
		extractCols = []

		if axisType == "feature":
			targetAxis = self._data.col
			otherAxis = self._data.row
			extractTarget = extractCols
			extractOther = extractRows
		else:
			targetAxis = self._data.row
			otherAxis = self._data.col
			extractTarget = extractRows
			extractOther = extractCols

		#walk through col listing and partition all data: extract, and kept, reusing the sparse matrix
		# underlying structure to save space
		copyIndex = 0

		for i in xrange(len(self._data.data)):
			value = targetAxis[i]
			if value >= start and value <= end:
				extractData.append(self._data.data[i])
				extractOther.append(otherAxis[i])
				extractTarget.append(value - start)
			else:
				self._data.data[copyIndex] = self._data.data[i]
				otherAxis[copyIndex] = otherAxis[i]
				if targetAxis[i] < start:
					targetAxis[copyIndex] = targetAxis[i]
				else:
					# end is inclusive, so we subtract end + 1
					targetAxis[copyIndex] = targetAxis[i] - rangeLength
				copyIndex = copyIndex + 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		(selfShape, extShape) = _calcShapes(self._data.shape, rangeLength, axisType)
		self._data = CooWithEmpty( (self._data.data[0:copyIndex],(self._data.row[0:copyIndex],self._data.col[0:copyIndex])), selfShape)

		# instantiate return data
		ret = CooWithEmpty((extractData,(extractRows,extractCols)),shape=extShape)
		
		# get names for return obj
		pnames = []
		fnames = []
		if axisType == 'point':
			for i in xrange(start,end+1):
				pnames.append(self.getPointName(i))
			fnames = self.getFeatureNames()
		else:
			pnames = self.getPointNames()
			for i in xrange(start,end+1):
				fnames.append(self.getFeatureName(i))

		return Sparse(ret, pointNames=pnames, featureNames=fnames, reuseData=True) 

	def _transpose_implementation(self):
		self._data = self._data.transpose()
		self._sorted = None
		_resync(self._data)
#		if self._sorted == 'point':
#			self._sorted = 'feature'
#		elif self._sorted == 'feature':
#			self._sorted = 'point'


	def _mapReducePoints_implementation(self, mapper, reducer):
		self._sortInternal("point")
		mapperResults = {}
		maxVal = self.featureCount

		# consume zeroed vectors, up to the first nonzero value
		if self._data.data[0] != 0:
			for i in xrange(0, self._data.row[0]):
				currResults = mapper(self.pointView(i))
				for (k,v) in currResults:
					if not k in mapperResults:
						mapperResults[k] = []
					mapperResults[k].append(v)

		vectorStartIndex = 0
		#walk through each coordinate entry
		for i in xrange(len(self._data.data)):
			#collect values into points
			targetValue = self._data.row[i]
			#if this is the end of a point
			if i == len(self._data.data)-1 or self._data.row[i+1] != targetValue:
				#evaluate whether curr point is to be extracted or not
				# and perform the appropriate copies
				currResults = mapper(self.pointView(targetValue))
				for (k,v) in currResults:
					if not k in mapperResults:
						mapperResults[k] = []
					mapperResults[k].append(v)

				# process zeroed vectors up to the ID of the new vector
				if i < len(self._data.data) - 1:
					nextValue = self._data.row[i+1]
				else:
					nextValue = self.pointCount
				for j in xrange(targetValue+1, nextValue):
					currResults = mapper(self.pointView(j))
					for (k,v) in currResults:
						if not k in mapperResults:
							mapperResults[k] = []
						mapperResults[k].append(v)

				#reset the vector starting index
				vectorStartIndex = i + 1

		# apply the reducer to the list of values associated with each key
		ret = []
		for mapKey in mapperResults.keys():
			mapValues = mapperResults[mapKey]
			# the reducer will return a tuple of a key to a value
			redRet = reducer(mapKey, mapValues)
			if redRet is not None:
				(redKey,redValue) = redRet
				ret.append([redKey, redValue])
		return Sparse(numpy.matrix(ret))

	def _isIdentical_implementation(self, other):
		if not isinstance(other, Sparse):
			return False
		# for nonempty matrices, we use a shape mismatch to indicate non-equality
		if self._data.shape != other._data.shape:
			return False

		if isinstance(other, SparseView):
			return other._isIdentical_implementation(self)
		else:
			return self._data == other._data

	def _getTypeString_implementation(self):
		return 'Sparse'

	def _writeFileCSV_implementation(self, outPath, includePointNames, includeFeatureNames):
		"""
		Function to write the data in this object to a CSV file at the designated
		path.

		"""
		outFile = open(outPath, 'w')
	
		if includeFeatureNames:
			# to signal that the first line contains feature Names
			outFile.write('\n\n')

			def combine(a, b):
				return a + ',' + b

			fnames = self.getFeatureNames()
			fnamesLine = reduce(combine, fnames)
			fnamesLine += '\n'
			if includePointNames:
				outFile.write('point_names,')

			outFile.write(fnamesLine)

		# sort by rows first, then columns
		placement = numpy.lexsort((self._data.col, self._data.row))
		self._data.data[placement]
		self._data.row[placement]
		self._data.col[placement]

		pointer = 0
		pmax = len(self._data.data)
		for i in xrange(self.pointCount):
			currPname = self.getPointName(i)
			if includePointNames:
				outFile.write(currPname)
				outFile.write(',')
			for j in xrange(self.featureCount):
				if pointer < pmax and i == self._data.row[pointer] and j == self._data.col[pointer]:
					value = self._data.data[pointer]
					pointer = pointer + 1
				else:
					value = 0

				if j != 0:
					outFile.write(',')	
				outFile.write(str(value))
			outFile.write('\n')

		outFile.close()

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

	def _writeFileMTX_implementation(self, outPath, includePointNames, includeFeatureNames):
		def makeNameString(count, namesItoN):
				nameString = "#"
				for i in xrange(count):
					nameString += namesItoN[i]
					if not i == count - 1:
						nameString += ','
				return nameString

		header = ''
		if includePointNames:
			header = makeNameString(self.pointCount, self.getPointNames())
			header += '\n'
		else:
			header += '#\n'
		if includeFeatureNames:
			header += makeNameString(self.featureCount, self.getFeatureNames())
			header += '\n'
		else:
			header += '#\n'

		if header != '':
			mmwrite(target=outPath, a=self.data, comment=header)		
		else:
			mmwrite(target=outPath, a=self._data.internal)


	def _referenceDataFrom_implementation(self, other):
		if not isinstance(other, Sparse):
			raise ArgumentException("Other must be the same type as this object")

		self._data = other._data
		self._sorted = other._sorted

	def _copyAs_implementation(self, format):
		if format is None or format == 'Sparse':
			ret = Sparse(self._data.internal, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
			ret._sorted = self._sorted
			return ret
		if format == 'List':
			return UML.data.List(self._data.internal, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
		if format == 'Matrix':
			return UML.data.Matrix(self._data.internal, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
		if format == 'DataFrame':
			return UML.data.DataFrame(self._data.internal, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
		if format == 'pythonlist':
			return self._data.todense().tolist()
		if format == 'numpyarray':
			return numpy.array(self._data.todense())
		if format == 'numpymatrix':
			return self._data.todense()
		if format == 'scipycsc':
			return self._data.internal.tocsc()
		if format == 'scipycsr':
			return self._data.internal.tocsr()

	def _copyPoints_implementation(self, points, start, end):
		retData = []
		retRow = []
		retCol = []
		if points is not None:
			for i in xrange(len(self._data.data)):
				if self._data.row[i] in points:
					retData.append(self._data.data[i])
					retRow.append(points.index(self._data.row[i]))
					retCol.append(self._data.col[i])

			newShape = (len(points), numpy.shape(self._data)[1])
		else:
			for i in xrange(len(self._data.data)):
				if self._data.row[i] >= start and self._data.row[i] <= end:
					retData.append(self._data.data[i])
					retRow.append(self._data.row[i] - start)
					retCol.append(self._data.col[i])

			newShape = (end - start + 1, numpy.shape(self._data)[1])

		retData = CooWithEmpty((retData,(retRow,retCol)),shape=newShape)
		return Sparse(retData, reuseData=True)


	def _copyFeatures_implementation(self, features, start, end):
		retData = []
		retRow = []
		retCol = []
		if features is not None:
			for i in xrange(len(self._data.data)):
				if self._data.col[i] in features:
					retData.append(self._data.data[i])
					retRow.append(self._data.row[i])
					retCol.append(features.index(self._data.col[i]))

			newShape = (numpy.shape(self._data)[0], len(features))
		else:
			for i in xrange(len(self._data.data)):
				if self._data.col[i] >= start and self._data.col[i] <= end:
					retData.append(self._data.data[i])
					retRow.append(self._data.row[i])
					retCol.append(self._data.col[i] - start)

			newShape = (numpy.shape(self._data)[0], end - start + 1)

		retData = CooWithEmpty((retData,(retRow,retCol)),shape=newShape)
		return Sparse(retData, reuseData=True)


	def _transformEachPoint_implementation(self, function, points):
		self._transformEach_implementation(function, points, 'point')


	def _transformEachFeature_implementation(self, function, features):
		self._transformEach_implementation(function, features, 'feature')


	def _transformEach_implementation(self, function, included, axis):
		modData = []
		modRow = []
		modCol = []

		if axis == 'point':
			viewIterator = self.pointIterator()
			modTarget = modRow
			modOther = modCol
		else:
			viewIterator = self.featureIterator()
			modTarget = modCol
			modOther = modRow

		for viewID, view in enumerate(viewIterator):
			if included is not None and viewID not in included:
				currOut = list(view)
			else:
				currOut = function(view)

			# easy way to reuse code if we have a singular return
			if not hasattr(currOut, '__iter__'):
				currOut = [currOut]

			# if there are multiple values, they must be random accessible
			if not hasattr(currOut, '__getitem__'):
				raise ArgumentException("function must return random accessible data (ie has a __getitem__ attribute)")

			for i, retVal in enumerate(currOut):
				if retVal != 0:
					modData.append(retVal)
					modTarget.append(viewID)
					modOther.append(i)

		if len(modData) != 0:
			self._data = CooWithEmpty((modData,(modRow,modCol)),shape=(self.pointCount,self.featureCount))
			self._sorted = None

		ret = None
		return ret


	def _transformEachElement_implementation(self, function, points, features, preserveZeros, skipNoneReturnValues):
		oneArg = False
		try:
			function(0,0,0)
		except TypeError:
			oneArg = True

		if oneArg and function(0) == 0:
			preserveZeros = True

		if preserveZeros:
			self._transformEachElement_zeroPreserve_implementation(function, points, features, skipNoneReturnValues, oneArg)
		else:
			self._transformEachElement_noPreserve_implementation(function, points, features, skipNoneReturnValues, oneArg)

	def _transformEachElement_noPreserve_implementation(self, function, points, features, skipNoneReturnValues, oneArg):
		# returns None if outside of the specified points and feature so that
		# when calculateForEach is called we are given a full data object
		# with only certain values modified.
		def wrapper(value, pID, fID):
			if points is not None and pID not in points:
				return None
			if features is not None and fID not in features:
				return None

			if oneArg:
				return function(value)
			else:
				return function(value, pID, fID)

		# perserveZeros is always False in this helper, skipNoneReturnValues
		# is being hijacked by the wrapper: even if it was False, Sparse can't
		# contain None values.
		ret = self.calculateForEachElement(wrapper, None, None, preserveZeros=False, skipNoneReturnValues=True)

		pnames = self.getPointNames()
		fnames = self.getFeatureNames()
		self.referenceDataFrom(ret)
		self.setPointNames(pnames)
		self.setFeatureNames(fnames)


	def _transformEachElement_zeroPreserve_implementation(self, function, points, features, skipNoneReturnValues, oneArg):
		for index, val in enumerate(self.data.data):
			pID = self.data.row[index]
			fID = self.data.col[index]
			if points is not None and pID not in points:
				continue
			if features is not None and fID not in features:
				continue

			if oneArg:
				currRet = function(val)
			else:
				currRet = function(val, pID, fID)

			if skipNoneReturnValues and currRet is None:
				continue

			self.data.data[index] = currRet


	def _fillWith_implementation(self, values, pointStart, featureStart, pointEnd, featureEnd):
		# sort values or call helper as needed
		constant = not isinstance(values, UML.data.Base)
		if constant:
			if values == 0:
				self._fillWith_zeros_implementation(pointStart, featureStart, pointEnd, featureEnd)
				return
		else:
			values._sortInternal('point')

		# this has to be after the possible call to _fillWith_zeros_implementation;
		# it is uncessary for that helper
		self._sortInternal('point')

		self_i = 0	
		vals_i = 0
		copyIndex = 0
		toAddData = []
		toAddRow = []
		toAddCol = []
		selfEnd = numpy.searchsorted(self._data.row, pointEnd, 'right')
		if constant:
			valsEnd = (pointEnd - pointStart + 1) * (featureEnd - featureStart + 1)
		else:
			valsEnd = len(values._data.data)

		# Adjust self_i so that it begins at the values that might need to be
		# replaced, or, if no such values exist, set self_i such that the main loop
		# will ignore the contents of self.		
		if len(self._data.data) > 0:
			self_i = numpy.searchsorted(self._data.row, pointStart, 'left')

			pcheck = self._data.row[self_i]
			fcheck = self._data.col[self_i]
			# the condition in the while loop is a natural break, if it isn't
			# satisfied then self_i will be exactly where we want it
			while fcheck < featureStart or fcheck > featureEnd:
				# this condition is an unatural break, when it is satisfied,
				# that means no value of self_i will point into the desired
				# values
				if pcheck > pointEnd or self_i == len(self._data.data)-1:
					self_i = selfEnd
					break

				self_i += 1
				pcheck = self._data.row[self_i]
				fcheck = self._data.col[self_i]

			copyIndex = self_i

		# Walk full contents of both, modifying, shifing, or setting aside values as needed.
		# We will only ever increment one of self_i or vals_i at a time, meaning if there are
		# matching entries, we will encounter them. Due to the sorted precondition, if
		# the location in one object is less than the location in the other object, the
		# lower one CANNOT have a match.	
		while self_i < selfEnd or vals_i < valsEnd:
			if self_i < selfEnd:
				locationSP = self._data.row[self_i]
				locationSF = self._data.col[self_i]		 
			else:
				# we want to use unreachable values as sentials, so we + 1 since we're
				# using inclusive endpoints
				locationSP = pointEnd + 1
				locationSF = featureEnd + 1
			
			# we adjust the 'values' locations into the scale of the calling object
			locationVP = pointStart
			locationVF = featureStart
			if constant:
				vData = values
				locationVP += vals_i / (pointEnd - pointStart + 1)  # uses truncation of int division
				locationVF += vals_i % (pointEnd - pointStart + 1)
			elif vals_i >= valsEnd:
				locationVP += pointEnd + 1
				locationVF += featureEnd + 1
			else:
				vData = values._data.data[vals_i]
				locationVP += values._data.row[vals_i]
				locationVF += values._data.col[vals_i]

			pCmp = locationSP - locationVP
			fCmp = locationSF - locationVF
			trueCmp = pCmp if pCmp != 0 else fCmp

			# Case: location at index into self is higher than location at index into values.
			# No matching entry in self; copy if space, or record to be added at end.
			if trueCmp > 0:			
				# can only copy into self if there is open space
				if copyIndex < self_i:
					self._data.data[copyIndex] = vData
					self._data.row[copyIndex] = locationVP
					self._data.col[copyIndex] = locationVF
					copyIndex += 1
				else:
					toAddData.append(vData)
					toAddRow.append(locationVP)
					toAddCol.append(locationVF)

				#increment vals_i
				vals_i += 1
			# Case: location at index into other is higher than location at index into self.
			# no matching entry in values - fill this entry in self with zero
			# (by shifting past it)
			elif trueCmp < 0:
				# need to do cleanup if we're outside of the relevant bounds
				if locationSF < featureStart or locationSF > featureEnd:
					self._data.data[copyIndex] = self._data.data[self_i]
					self._data.row[copyIndex] = self._data.row[self_i]
					self._data.col[copyIndex] = self._data.col[self_i]
					copyIndex += 1
				self_i += 1
			# Case: indices point to equal locations.
			else:
				self._data.data[copyIndex] = vData
				self._data.row[copyIndex] = locationVP
				self._data.col[copyIndex] = locationVF
				copyIndex += 1

				# increment both??? or just one?
				self_i += 1
				vals_i += 1

		# Now we have to walk through the rest of self, finishing the copying shift
		# if necessary
		if copyIndex != self_i:
			while self_i < len(self._data.data):
				self._data.data[copyIndex] = self._data.data[self_i]
				self._data.row[copyIndex] = self._data.row[self_i]
				self._data.col[copyIndex] = self._data.col[self_i]
				self_i += 1
				copyIndex += 1
		else:
			copyIndex = len(self._data.data)

		newData = numpy.empty(copyIndex + len(toAddData))
		newData[:copyIndex] = self._data.data[:copyIndex]
		newData[copyIndex:] = toAddData
		newRow = numpy.empty(copyIndex + len(toAddRow))
		newRow[:copyIndex] = self._data.row[:copyIndex]
		newRow[copyIndex:] = toAddRow
		newCol = numpy.empty(copyIndex + len(toAddCol))
		newCol[:copyIndex] = self._data.col[:copyIndex]
		newCol[copyIndex:] = toAddCol
		self._data = CooWithEmpty((newData,(newRow,newCol)), (self.pointCount, self.featureCount))

		if len(toAddData) != 0:
			self._sorted = None


	def _mergeIntoNewData(self, copyIndex, toAddData, toAddRow, toAddCol):
		#instead of always copying, use reshape or resize to sometimes cut array down
		# to size???
		pass

	def _fillWith_zeros_implementation(self, pointStart, featureStart, pointEnd, featureEnd):
		#walk through col listing and partition all data: extract, and kept, reusing the sparse matrix
		# underlying structure to save space
		copyIndex = 0

		for lookIndex in xrange(len(self._data.data)):
			currP = self._data.row[lookIndex]
			currF = self._data.col[lookIndex]
			# if it is in range we want to obliterate the entry by just passing it by
			# and copying over it later
			if currP >= pointStart and currP <= pointEnd and currF >= featureStart and currF <= featureEnd:
				pass
			else:
				self._data.data[copyIndex] = self._data.data[lookIndex]
				self._data.row[copyIndex] = self._data.row[lookIndex]
				self._data.col[copyIndex] = self._data.col[lookIndex]
				copyIndex += 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		newData = (self._data.data[0:copyIndex],(self._data.row[0:copyIndex],self._data.col[0:copyIndex]))
		self._data = CooWithEmpty(newData, (self.pointCount, self.featureCount))


	def _getitem_implementation(self, x, y):
		for i in xrange(len(self._data.row)):
			rowVal = self._data.row[i]
			if rowVal == x and self._data.col[i] == y:
				return self._data.data[i]

		return 0

	def _view_implementation(self, pointStart, pointEnd, featureStart, featureEnd):
		"""
		The Sparse object specific implementation necessarly to complete the Base
		object's view method. pointStart and feature start are inclusive indices,
		pointEnd and featureEnd are exclusive indices.

		"""
		kwds = {}
		kwds['source'] = self
		kwds['pointStart'] = pointStart
		kwds['pointEnd'] = pointEnd
		kwds['featureStart'] = featureStart
		kwds['featureEnd'] = featureEnd
		kwds['reuseData'] = True

		allPoints = pointStart == 0 and pointEnd == self.pointCount
		singlePoint = pointEnd - pointStart == 1
		allFeats = featureStart == 0 and featureEnd == self.featureCount
		singleFeat = featureEnd - featureStart == 1

		if singleFeat or singlePoint:
			if singlePoint:
				if self._sorted is None or self._sorted == 'feature':
					self._sortInternal('point')
				sortedIndices = self._data.row

				start = numpy.searchsorted(sortedIndices, pointStart, 'left')
				end = numpy.searchsorted(sortedIndices, pointEnd-1, 'right')

				if not allFeats:
					sortedIndices = self._data.col[start:end]
					innerStart = numpy.searchsorted(sortedIndices, featureStart, 'left')
					innerEnd = numpy.searchsorted(sortedIndices, featureEnd-1, 'right')
					outerStart = start
					start = start + innerStart
					end = outerStart + innerEnd

				row = numpy.tile([0], end-start)
				col = self._data.col[start:end] - featureStart

			else:  # case single feature
				if self._sorted is None or self._sorted == 'point':
					self._sortInternal('feature')
				sortedIndices = self._data.col

				start = numpy.searchsorted(sortedIndices, featureStart, 'left')
				end = numpy.searchsorted(sortedIndices, featureEnd-1, 'right')

				if not allPoints:
					sortedIndices = self._data.row[start:end]
					innerStart = numpy.searchsorted(sortedIndices, pointStart, 'left')
					innerEnd = numpy.searchsorted(sortedIndices, pointEnd-1, 'right')
					outerStart = start
					start = start + innerStart
					end = outerStart + innerEnd

				row = self._data.row[start:end] - pointStart
				col = numpy.tile([0], end-start)

			data = self._data.data[start:end]
			pshape = pointEnd - pointStart
			fshape = featureEnd - featureStart

			newInternal = scipy.sparse.coo_matrix((data,(row,col)), shape=(pshape,fshape))
			kwds['data'] = newInternal

			return SparseVectorView(**kwds)

		else:  # window shaped View
			class CooDummy(CooWrapper):
				def __init__(self, shape, **kwargs):
					self.shape = shape
					self.internal = None
					super(CooDummy, self).__init__(**kwargs)

			newInternal = CooDummy((pointEnd-pointStart, featureEnd-featureStart))
			kwds['data'] = newInternal

			return SparseView(**kwds)


	def _validate_implementation(self, level):
		assert self._data.shape[0] == self.pointCount
		assert self._data.shape[1] == self.featureCount
		assert isinstance(self._data, CooWrapper)

		if level > 0:
			for value in self._data.data:
				assert value != 0
			if scipy.sparse.isspmatrix(self._data.internal):
				assert numpy.array_equal(self._data.data, self._data.internal.data)
				assert numpy.array_equal(self._data.row, self._data.internal.row)
				assert numpy.array_equal(self._data.col, self._data.internal.col)

			if self._sorted == 'point':
				for i in xrange(len(self._data.row)):
					if i != len(self._data.row) - 1:
						assert self._data.row[i] <= self._data.row[i+1]
			if self._sorted == 'feature':
				for i in xrange(len(self._data.col)):
					if i != len(self._data.col) - 1:
						assert self._data.col[i] <= self._data.col[i+1]

	def _containsZero_implementation(self):
		"""
		Returns True if there is a value that is equal to integer 0 contained
		in this object. False otherwise

		"""
		return (self.data.shape[0] * self.data.shape[1]) > self.data.nnz


	def _nonZeroIteratorPointGrouped_implementation(self):
		self._sortInternal('point')
		return self._nonZeroIterator_general_implementation()

	def _nonZeroIteratorFeatureGrouped_implementation(self):
		self._sortInternal('feature')
		return self._nonZeroIterator_general_implementation()

	def _nonZeroIterator_general_implementation(self):
		# Assumption: underlying data already correctly sorted by
		# per axis helper; will not be modified during iteration
		class nzIt(object):
			def __init__(self, source):
				self._source = source
				self._index = 0

			def __iter__(self):
				return self

			def next(self):
				while (self._index < len(self._source.data.data)):
					value = self._source.data.data[self._index]

					self._index += 1
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
		if isinstance(other, BaseView):
			retData = other.copyAs('scipycsr')
			retData = self._data.internal * retData
		else:
			# for other.data as any dense or sparse matrix
			retData = self._data.internal * other.data

		return Sparse(retData)


	def _elementwiseMultiply_implementation(self, other):
		"""
		Perform element wise multiplication of this UML data object against the
		provided other UML data object. Both objects must contain only numeric
		data. The pointCount and featureCount of both objects must be equal. The
		types of the two objects may be different, but the returned object will
		be the inplace modification of the calling object.

		"""
		# CHOICE OF OUTPUT WILL BE DETERMINED BY SCIPY!!!!!!!!!!!!
		# for other.data as any dense or sparse matrix
		toMul = None
		directMul = isinstance(other, Sparse) or isinstance(other, UML.data.Matrix)
		notView = not isinstance(other, BaseView)
		if directMul and notView:
			toMul = other.data
		else:
			toMul = other.copyAs('numpyarray')

		raw = self._data.internal.multiply(toMul)
		reuse = False
		if scipy.sparse.isspmatrix(raw):
			raw = raw.tocoo()
			reuse = True
		self._data = CooWithEmpty(raw, shape=self._data.shape, reuseData=reuse)
		

	def _scalarMultiply_implementation(self, scalar):
		"""
		Multiply every element of this UML data object by the provided scalar.
		This object must contain only numeric data. The 'scalar' parameter must
		be a numeric data type. The returned object will be the inplace modification
		of the calling object.
		
		"""
		if scalar != 0:
			scaled = self._data.data * scalar
			self._data.data = scaled
			self._data.internal.data = scaled
		else:
			self._data = CooWithEmpty(([],([],[])),shape=(self.pointCount,self.featureCount))

	def _mul__implementation(self, other):
		if isinstance(other, UML.data.Base):
			return self._matrixMultiply_implementation(other)
		else:
			ret = self.copy()
			ret._scalarMultiply_implementation(other)
			return ret

#	def _div__implementation(self, other):
#		if isinstance(other, UML.data.Base):
#			ret = self.data.tocsr() / other.copyAs("scipycsr")
#			ret = ret.tocoo()
#		else:
#			retData = self._data.data / other
#			retRow = numpy.array(self.data.row)
#			retCol = numpy.array(self.data.col)
#			ret = scipy.sparse.coo_matrix((retData,(retRow, retCol)))
#		return Sparse(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)


#	def _rdiv__implementation(self, other):
#		retData = other / self.data
#		retRow = numpy.array(self.data.row)
#		retCol = numpy.array(self.data.col)
#		ret = scipy.sparse.coo_matrix((retData,(retRow, retCol)))
#		return Sparse(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

#	def _idiv__implementation(self, other):
#		if isinstance(other, UML.data.Base):
#			ret = self.data.tocsr() / other.copyAs("scipycsr")
#			ret = ret.tocoo()
#		else:
#			ret = self.data.data / other
#		self.data = ret
#		return self

#	def _truediv__implementation(self, other):
#		if isinstance(other, UML.data.Base):
#			ret = self.data.tocsr().__truediv__(other.copyAs("scipycsr"))
#			ret = ret.tocoo()
#		else:
#			retData = self.data.data.__truediv__(other)
#			retRow = numpy.array(self.data.row)
#			retCol = numpy.array(self.data.col)
#			ret = scipy.sparse.coo_matrix((retData,(retRow, retCol)))
#		return Sparse(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

#	def _rtruediv__implementation(self, other):
#		retData = self.data.data.__rtruediv__(other)
#		retRow = numpy.array(self.data.row)
#		retCol = numpy.array(self.data.col)
#		ret = scipy.sparse.coo_matrix((retData,(retRow, retCol)))

#		return Sparse(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

#	def _itruediv__implementation(self, other):
#		if isinstance(other, UML.data.Base):
#			ret = self.data.tocsr().__itruediv__(other.copyAs("scipycsr"))
#			ret = ret.tocoo()
#		else:
#			retData = self.data.data.__itruediv__(other)
#			retRow = numpy.array(self.data.row)
#			retCol = numpy.array(self.data.col)
#			ret = scipy.sparse.coo_matrix((retData,(retRow, retCol)))
#		self.data = ret
#		return self

#	def _floordiv__implementation(self, other):
#		if isinstance(other, UML.data.Base):
#			ret = self.data.tocsr() // other.copyAs("scipycsr")
#			ret = ret.tocoo()
#		else:
#			retData = self._data.data // other
#			nzIDs = numpy.nonzero(retData)
#			retData = retData[nzIDs]
#			retRow = self.data.row[nzIDs]
#			retCol = self.data.col[nzIDs]
#			ret = scipy.sparse.coo_matrix((retData,(retRow, retCol)))
#		return Sparse(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)


#	def _rfloordiv__implementation(self, other):
#		retData = other // self._data.data
#		nzIDs = numpy.nonzero(retData)
#		retData = retData[nzIDs]
#		retRow = self.data.row[nzIDs]
#		retCol = self.data.col[nzIDs]
#		ret = scipy.sparse.coo_matrix((retData,(retRow, retCol)))
#		return Sparse(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)

#	def _ifloordiv__implementation(self, other):
#		if isinstance(other, UML.data.Base):
#			ret = self.data.tocsr() // other.copyAs("scipycsr")
#			ret = ret.tocoo()
#		else:
#			ret = self.data // other
#			nzIDs = numpy.nonzero(ret)
#			ret = ret[nzIDs]

#		self.data = ret
#		return self

	def _mod__implementation(self, other):
		if isinstance(other, UML.data.Base):
			return super(Sparse, self).__mod__(other)
		else:
			retData = self.data.data % other
			retRow = numpy.array(self.data.row)
			retCol = numpy.array(self.data.col)
			ret = scipy.sparse.coo_matrix((retData,(retRow, retCol)))
		return Sparse(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)


	def _rmod__implementation(self, other):
		retData = other % self.data.data
		retRow = numpy.array(self.data.row)
		retCol = numpy.array(self.data.col)
		ret = scipy.sparse.coo_matrix((retData,(retRow, retCol)))

		return Sparse(ret, pointNames=self.getPointNames(), featureNames=self.getFeatureNames(), reuseData=True)


	def _imod__implementation(self, other):
		if isinstance(other, UML.data.Base):
			return super(Sparse, self).__imod__(other)
		else:
			ret = self._data.data % other
		self._data.data = ret
		return self




	###########
	# Helpers #
	###########

	def _sortInternal(self, axis):
		if axis != 'point' and axis != 'feature':
			raise ArgumentException("invalid axis type")

		if self._sorted == axis or self.pointCount == 0 or self.featureCount == 0:
			return

		# sort least significant axis first
		if axis == "point":
			sortPrime = self._data.row
			sortOff = self._data.col
		else:
			sortPrime = self._data.col
			sortOff = self._data.row

		sortKeys = numpy.lexsort((sortOff, sortPrime))

		newData = self.data.data[sortKeys]
		newRow = self.data.row[sortKeys]
		newCol = self.data.col[sortKeys]

		n = len(newData)
		self._data.data[:n] = newData
		self._data.row[:n] = newRow
		self._data.col[:n] = newCol

		# flag that we are internally sorted
		self._sorted = axis


###################
# Generic Helpers #
###################

def _numLessThan(value, toCheck): # TODO caching
	ltCount = 0
	for i in xrange(len(toCheck)):
		if toCheck[i] < value:
			ltCount += 1

	return ltCount

def _calcShapes(currShape, numExtracted, axisType):
	(rowShape, colShape) = currShape
	if axisType == "feature":
		selfRowShape = rowShape
		selfColShape = colShape - numExtracted
		extRowShape = rowShape
		extColShape = numExtracted
	elif axisType == "point":
		selfRowShape = rowShape - numExtracted
		selfColShape = colShape
		extRowShape = numExtracted
		extColShape = colShape
	else:
		raise ArgumentException("invalid axis type")

	return ((selfRowShape,selfColShape),(extRowShape,extColShape))

def _resync(obj):
	if 0 in obj.internal.shape:
		obj.nnz = 0
		obj.data = numpy.array([])
		obj.row = numpy.array([])
		obj.col = numpy.array([])
		obj.shape = obj.internal.shape
	else:
		obj.nnz = obj.internal.nnz
		obj.data = obj.internal.data
		obj.row = obj.internal.row
		obj.col = obj.internal.col
		obj.shape = obj.internal.shape

class CooWrapper(object):
	def __init__(self, **kwargs):
		super(CooWrapper, self).__init__(**kwargs)

class CooWithEmpty(CooWrapper):

	def __init__(self, arg1, shape=None, dtype=None, copy=False, reuseData=False, **kwds):
		self.ndim = 2
		if isinstance(arg1, CooWithEmpty):
			arg1 = arg1.internal

		if isinstance(arg1, numpy.matrix):
			shape = arg1.shape

		if shape is not None and (shape[0] == 0 or shape[1] == 0):
			if isinstance(arg1, tuple):
				if shape is None:
					self.shape = (0,0)
				else:
					self.shape = shape
			else:
				converted = numpy.array(arg1)
				self.shape = converted.shape
			self.nnz = 0
			self.data = numpy.array([])
			self.row = numpy.array([])
			self.col = numpy.array([])
			self.internal = numpy.empty(self.shape)
		else:
			if reuseData:
				internal = arg1
			else:
				internal = coo_matrix(arg1, shape, dtype, copy)
			self.nnz = internal.nnz
			
			self.data = internal.data
			self.row = internal.row
			self.col = internal.col
			self.shape = internal.shape
			self.internal = internal

		super(CooWithEmpty, self).__init__(**kwds)


	def transpose(self):
		self.internal = self.internal.transpose()
		self.shape = self.internal.shape
		return self

	def todense(self):
		if scipy.sparse.isspmatrix(self.internal):
			return self.internal.todense()
		else:
			return numpy.empty(self.shape)

	def __eq__(self, other):
		wrappedRight = isinstance(other, CooWithEmpty)
		if not wrappedRight:
			return False

		if self.shape != other.shape:
			return False

		leftSparse = scipy.sparse.isspmatrix(self.internal)
		rightSparse = scipy.sparse.isspmatrix(other.internal)
		if leftSparse:
			if rightSparse:
				# == for scipy sparse types is inconsistent. This is testing how many are
				# nonzero after subtracting one from the other.
				ret = abs(self.internal - other.internal).nnz == 0
				return ret
			else:
				return False
		else:
			if rightSparse:
				return False
			else:
				return True





class SparseVectorView(BaseView, Sparse):
	"""
	A view of a Sparse data object limited to a full point or full feature

	"""
	def __init__(self, **kwds):
		super(SparseVectorView, self).__init__(**kwds)



class SparseView(BaseView, Sparse):
	def __init__(self, **kwds):
		super(SparseView, self).__init__(**kwds)

	def _validate_implementation(self, level):
		self._source.validate(level)

	def _getitem_implementation(self, x, y):
		adjX = x + self._pStart
		adjY = y + self._fStart
		return self._source[adjX, adjY]

	def pointIterator(self):
		return self._generic_iterator("point")

	def featureIterator(self):
		return self._generic_iterator("feature")

	def _generic_iterator(self, axis):
		source = self._source
		if axis == 'point':
			positionLimit = self._pStart + self.pointCount
			sourceStart = self._pStart
			# Needs to be None if we're dealing with a fully empty point
			fixedStart = self._fStart if self._fStart != 0 else None
			# self._fEnd is exclusive, but view takes inclusive indices
			fixedEnd = (self._fEnd - 1) if self._fEnd != 0 else None
		else:
			positionLimit = self._fStart + self.featureCount
			sourceStart = self._fStart
			# Needs to be None if we're dealing with a fully empty feature
			fixedStart = self._pStart if self._pStart != 0 else None
			# self._pEnd is exclusive, but view takes inclusive indices
			fixedEnd = (self._pEnd - 1) if self._pEnd != 0 else None

		class GenericIt(object):
			def __init__(self):
				self._position = sourceStart

			def __iter__(self):
				return self

			def next(self):
				if self._position < positionLimit:
					if axis == 'point':
						value = source.view(self._position, self._position, fixedStart, fixedEnd)
					else:
						value = source.view(fixedStart, fixedEnd, self._position, self._position)
					self._position += 1
					return value

				raise StopIteration

		return GenericIt()

	def _copyAs_implementation(self, format):
		if self.pointCount == 0 or self.featureCount == 0:
			emptyStandin = numpy.empty((self.pointCount, self.featureCount))
			intermediate = UML.data.Matrix(emptyStandin, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
			return intermediate.copyAs(format)

		limited = self._source.copyPoints(start=self._pStart, end=self._pEnd-1)
		limited = limited.copyFeatures(start=self._fStart, end=self._fEnd-1)

		if format is None or format == 'Sparse':
			return limited
		else:
			return limited._copyAs_implementation(format)

	def _isIdentical_implementation(self, other):
		if not isinstance(other, Sparse):
			return False
		# for nonempty matrices, we use a shape mismatch to indicate non-equality
		if self._data.shape != other._data.shape:
			return False

		# empty object means no values. Since shapes match they're equal
		if self._data.shape[0] == 0 or self._data.shape[1] == 0:
			return True


		sIt = self.pointIterator()
		oIt = other.pointIterator()
		for sPoint in sIt:
			oPoint = oIt.next()

			for i, val in enumerate(sPoint):
				if val != oPoint[i]:
					return False

		return True

	def _containsZero_implementation(self):
		for sPoint in self.pointIterator():
			if sPoint.containsZero():
				return True

		return False

	def __abs__(self):
		""" Perform element wise absolute value on this object """
		ret = self.copyAs("Sparse")
		numpy.absolute(ret._data.internal.data, out=ret._data.internal.data)
		ret._name = dataHelpers.nextDefaultObjectName()

		return ret

	def _mul__implementation(self, other):
		selfConv = self.copyAs("Sparse")
		if isinstance(other, BaseView):
			other = other.copyAs(other.getTypeString())
		return selfConv._mul__implementation(other)

	def _genericNumericBinary_implementation(self, opName, other):
		if isinstance(other, BaseView):
			other = other.copyAs(other.getTypeString())

		implName = opName[1:] + 'implementation'
		opType = opName[-5:-2]

		if opType in ['add', 'sub', 'div', 'truediv', 'floordiv']:
			selfConv = self.copyAs("Matrix")
			toCall = getattr(selfConv, implName)
			ret = toCall(other)
			ret = UML.createData("Sparse", ret.data)
			return ret

		selfConv = self.copyAs("Sparse")

		toCall = getattr(selfConv, implName)
		ret = toCall(other)
		ret = UML.createData(self.getTypeString(), ret.data)

		return ret

	def _copyPoints_implementation(self, points, start, end):
		retData = []
		retRow = []
		retCol = []
		piNN = points is not None

		for pID, pView in enumerate(self.pointIterator()):
			if (piNN and pID in points) or (pID >= start and pID <= end):
				for i,val in enumerate(pView.data.data):
					retData.append(val)
					if piNN:
						retRow.append(points.index(pID))
					else:
						retRow.append(pID - start)
					retCol.append(pView.data.col[i])

		if points is not None:
			newShape = (len(points), numpy.shape(self._data)[1])
		else:
			newShape = (end - start + 1, numpy.shape(self._data)[1])

		retData = CooWithEmpty((retData,(retRow,retCol)),shape=newShape)
		return Sparse(retData, reuseData=True)


	def _copyFeatures_implementation(self, features, start, end):
		retData = []
		retRow = []
		retCol = []
		fiNN = features is not None

		for fID, fView in enumerate(self.featureIterator()):
			if (fiNN and fID in features) or (fID >= start and fID <= end):
				for i,val in enumerate(fView.data.data):
					retData.append(val)
					retRow.append(fView.data.row[i])
					if fiNN:
						retCol.append(features.index(fID))
					else:
						retCol.append(fID - start)

		if features is not None:
			newShape = (numpy.shape(self._data)[0], len(features))
		else:
			newShape = (numpy.shape(self._data)[0], end - start + 1)

		retData = CooWithEmpty((retData,(retRow,retCol)),shape=newShape)
		return Sparse(retData, reuseData=True)



	def _nonZeroIteratorPointGrouped_implementation(self):
		return self._nonZeroIterator_general_implementation(self.pointIterator())

	def _nonZeroIteratorFeatureGrouped_implementation(self):
		return self._nonZeroIterator_general_implementation(self.featureIterator())

	def _nonZeroIterator_general_implementation(self, sourceIter):
		# IDEA: check if sorted in the way you want.
		# if yes, iterate through
		# if no, use numpy argsort? this gives you indices that
		# would sort it, iterate through those indices to do access?
		#
		# safety: somehow check that your sorting setup hasn't changed

		class nzIt(object):
			def __init__(self):
				self._sourceIter = sourceIter
				self._currGroup = None
				self._index = 0

			def __iter__(self):
				return self

			def next(self):
				while True:
					try:
						value = self._currGroup[self._index]
						self._index += 1

						if value != 0:
							return value
					except:
						self._currGroup = self._sourceIter.next()
						self._index = 0

		return nzIt()
