"""
Class extending SparseBaseData, defining an object to hold and manipulate a scipy coo_matrix.

"""

import numpy
import scipy
import random
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
from copy import deepcopy

from base_data import *
from sparse_data import *
from ..utility.custom_exceptions import ArgumentException



class CooSparseData(SparseData):

	def __init__(self, data=None, featureNames=None, name=None, path=None):
		if data == [] or data == numpy.array([]):
			raise ArgumentException("Data must not be shapeless (in other words, empty)")
		else:
			self.data = coo_matrix(data)
		super(CooSparseData, self).__init__(self.data, featureNames, name, path)


	def _appendPoints_implementation(self,toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object
		
		"""
		newData = numpy.append(self.data.data, toAppend.data.data)
		newRow = numpy.append(self.data.row, toAppend.data.row)
		newCol = numpy.append(self.data.col, toAppend.data.col)

		# correct the row entries
		offset = self.points()
		for index in xrange(len(self.data.data), len(newData)):
			newRow[index] = newRow[index] + offset
		
		numNewRows = self.points() + toAppend.points()
		self.data = coo_matrix((newData,(newRow,newCol)),shape=(numNewRows,self.features()))


	def _appendFeatures_implementation(self,toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		"""
		newData = numpy.append(self.data.data, toAppend.data.data)
		newRow = numpy.append(self.data.row, toAppend.data.row)
		newCol = numpy.append(self.data.col, toAppend.data.col)

		# correct the col entries
		offset = self.features()
		for index in xrange(len(self.data.data), len(newData)):
			newCol[index] = newCol[index] + offset
		
		numNewCols = self.features() + toAppend.features()
		self.data = coo_matrix((newData,(newRow,newCol)),shape=(self.points(),numNewCols))


	def _sortPoints_implementation(self, sortBy, sortHelper):
		sort_general_implementation(sortBy, scoreFcn, comparator, 'point')


	def _sortFeatures_implementation(self, sortBy, sortHelper):
		raise NotImplementedError
		sort_general_implementation(sortBy, scoreFcn, comparator, 'feature')


	def sort_general_implementation(self, sortBy, sortHelper, axisType):
		raise NotImplementedError



	def _extractPoints_implementation(self, toExtract, start, end, number, randomize):
		"""
		Function to extract points according to the parameters, and return an object containing
		the removed points with default names. The actual work is done by further helper
		functions, this determines which helper to call, and modifies the input to accomodate
		the number and randomize parameters, where number indicates how many of the possibilities
		should be extracted, and randomize indicates whether the choice of who to extract should
		be by order or uniform random.

		"""
		# single identifier
		if isinstance(toExtract, int):
			toExtract = [toExtract]	
		# list of identifiers
		if isinstance(toExtract, list):
			if number is None:
				number = len(toExtract)
			# if randomize, use random sample
			if randomize:
				raise NotImplementedError # TODO implement using sample(), but without losing the extraction order
			# else take the first number members of toExtract
			else:
				toExtract = toExtract[:number]
			return self._extractByList_implementation(toExtract, 'point')
		# boolean function
		if hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError # TODO randomize in the extractPointByFunction case
			else:
				if number is None:
					number = self.points()		
				return self._extractByFunction_implementation(toExtract, number, "point")
		# by range
		if start is not None or end is not None:
			if number is None:
				number = end - start
			if randomize:
				toExtract = random.sample(xrange(start,end),number)
				toExtract.sort()
				return self._extractByList_implementation(toExtract, 'point')
			else:
				return self._extractByRange_implementation(start, end, 'point')


	def _extractFeatures_implementation(self, toExtract, start, end, number, randomize):
		"""
		Function to extract features according to the parameters, and return an object containing
		the removed features with their featureName names from this object. The actual work is done by
		further helper functions, this determines which helper to call, and modifies the input
		to accomodate the number and randomize parameters, where number indicates how many of the
		possibilities should be extracted, and randomize indicates whether the choice of who to
		extract should be by order or uniform random.

		"""
		# single identifier
		if isinstance(toExtract, int) or isinstance(toExtract, basestring):
			toExtract = [toExtract]	
		# list of identifiers
		if isinstance(toExtract, list):
			if number is None:
				number = len(toExtract)
			# if randomize, use random sample
			if randomize:
				raise NotImplementedError # TODO implement using sample(), but without losing the extraction order
			# else take the first number members of toExtract
			else:
				toExtract = toExtract[:number]
			# convert IDs if necessary
			toExtractIndices = []
			for value in toExtract:
				toExtractIndices.append(self._getIndex(value))
			return self._extractByList_implementation(toExtractIndices, "feature")
		# boolean function
		if hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError # TODO randomize in the extractFeatureByFunction case
			else:
				if number is None:
					number = self.points()		
				return self._extractByFunction_implementation(toExtract, number, "feature")
		# by range
		if start is not None or end is not None:
			if number is None:
				number = end - start
			if randomize:
				toExtract = random.sample(xrange(start,end),number)
				toExtract.sort()
				return self._extractByList_implementation(toExtract, 'feature')
			else:
				return self._extractByRange_implementation(start, end, 'feature')


	def _extractByList_implementation(self, toExtract, axisType):
		"""
		

		"""
		extractLength = len(toExtract)
		extractData = []
		extractRows = []
		extractCols = []

		self._sortInternal(axisType)
		if axisType == "feature":
			targetAxis = self.data.col
			otherAxis = self.data.row
			extractTarget = extractCols
			extractOther = extractRows
		else:
			targetAxis = self.data.row
			otherAxis = self.data.col
			extractTarget = extractRows
			extractOther = extractCols

		#List of rows or columns to extract must be sorted in ascending order
		toExtract.sort()
		
		#walk through col listing and partition all data: extract, and kept, reusing the sparse matrix
		# underlying structure to save space
		copy = 0
		extractIndex = 0
		for i in xrange(len(self.data.data)):
			value = targetAxis[i]
			if extractIndex < extractLength and value > toExtract[extractIndex]:
				extractIndex = extractIndex + 1
			if extractIndex < extractLength and value == toExtract[extractIndex]:
				extractData.append(self.data.data[i])
				extractOther.append(otherAxis[i])
				extractTarget.append(extractIndex)	
			else:
				self.data.data[copy] = self.data.data[i]				
				otherAxis[copy] = otherAxis[i]
				targetAxis[copy] = targetAxis[i] - extractIndex
				copy = copy + 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		(selfShape, extShape) = _calcShapes(self.data.shape, len(toExtract), axisType)
		self.data = coo_matrix( (self.data.data[0:copy],(self.data.row[0:copy],self.data.col[0:copy])), selfShape)

		# instantiate return data
		ret = coo_matrix((extractData,(extractRows,extractCols)),shape=extShape)
		
		# get featureNames for return obj
		featureNames = []
		if axisType == "feature":
			for index in toExtract:
				featureNames.append(self.featureNamesInverse[index])
		else:
			featureNames = self.featureNames

		return CooSparseData(ret, featureNames) 


	def _extractByFunction_implementation(self, toExtract, number, axisType):
		extractData = []
		extractRows = []
		extractCols = []

		self._sortInternal(axisType)
		if axisType == "feature":
			targetAxis = self.data.col
			otherAxis = self.data.row
			extractTarget = extractCols
			extractOther = extractRows
			maxVal = self.points()		
		else:
			targetAxis = self.data.row
			otherAxis = self.data.col
			extractTarget = extractRows
			extractOther = extractCols
			maxVal = self.features()

		copy = 0
		vectorStartIndex = 0
		extractedIDs = []
		# consume zeroed vectors, up to the first nonzero value
		if self.data.data[0] != 0:
			for i in xrange(0,targetAxis[0]):
				if toExtract(VectorView(self, None, None, {}, maxVal, i, axisType)):
					extractedIDs.append(i)
		#walk through each coordinate entry
		for i in xrange(len(self.data.data)):
			#collect values into points
			targetValue = targetAxis[i]
			#if this is the end of a point
			if i == len(self.data.data)-1 or targetAxis[i+1] != targetValue:
				#evaluate whether curr point is to be extracted or not
				# and perform the appropriate copies
				if toExtract(VectorView(self, vectorStartIndex, i, None, maxVal, targetValue, axisType)):
					for j in xrange(vectorStartIndex, i + 1):
						extractData.append(self.data.data[j])
						extractOther.append(otherAxis[j])
						extractTarget.append(len(extractedIDs))
					extractedIDs.append(targetValue)
				else:
					for j in xrange(vectorStartIndex, i + 1):
						self.data.data[copy] = self.data.data[j]				
						otherAxis[copy] = otherAxis[j]
						targetAxis[copy] = targetAxis[j] - len(extractedIDs)
						copy = copy + 1
				# process zeroed vectors up to the ID of the new vector
				if i < len(self.data.data) - 1:
					nextValue = targetAxis[i+1]
				else:
					if axisType == 'point':
						nextValue = self.points()
					else:
						nextValue = self.features()
				for j in xrange(targetValue+1, nextValue):
					if toExtract(VectorView(self, None, None, {}, maxVal, j, axisType)):
						extractedIDs.append(j)

				#reset the vector starting index
				vectorStartIndex = i + 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		(selfShape, extShape) = _calcShapes(self.data.shape, len(extractedIDs), axisType)
		self.data = coo_matrix( (self.data.data[0:copy],(self.data.row[0:copy],self.data.col[0:copy])), selfShape)

		# instantiate return data
		ret = coo_matrix((extractData,(extractRows,extractCols)),shape=extShape)
		
		# get featureNames for return obj
		featureNames = []
		if axisType == "feature":
			for index in extractedIDs:
				featureNames.append(self.featureNamesInverse[index])
		else:
			featureNames = self.featureNames

		return CooSparseData(ret, featureNames) 





	def _extractByRange_implementation(self, start, end, axisType):
		"""
		

		"""
		rangeLength = end - start + 1
		extractData = []
		extractRows = []
		extractCols = []

		if axisType == "feature":
			targetAxis = self.data.col
			otherAxis = self.data.row
			extractTarget = extractCols
			extractOther = extractRows
		else:
			targetAxis = self.data.row
			otherAxis = self.data.col
			extractTarget = extractRows
			extractOther = extractCols

		#walk through col listing and partition all data: extract, and kept, reusing the sparse matrix
		# underlying structure to save space
		copy = 0

		for i in xrange(len(self.data.data)):
			value = targetAxis[i]
			if value >= start and value <= end:
				extractData.append(self.data.data[i])
				extractOther.append(otherAxis[i])
				extractTarget.append(value - start)
			else:
				self.data.data[copy] = self.data.data[i]
				otherAxis[copy] = otherAxis[i]
				if targetAxis[i] < start:
					targetAxis[copy] = targetAxis[i]
				else:
					# end is inclusive, so we subtract end + 1
					targetAxis[copy] = targetAxis[i] - (end + 1)
				copy = copy + 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		(selfShape, extShape) = _calcShapes(self.data.shape, rangeLength, axisType)
		self.data = coo_matrix( (self.data.data[0:copy],(self.data.row[0:copy],self.data.col[0:copy])), selfShape)

		# instantiate return data
		ret = coo_matrix((extractData,(extractRows,extractCols)),shape=extShape)
		
		# get featureNames for return obj
		featureNames = []
		if axisType == "feature":
			for i in xrange(start,end+1):
				featureNames.append(self.featureNamesInverse[i])
		else:
			featureNames = self.featureNames

		return CooSparseData(ret, featureNames) 

	def _applyFunctionToEachPoint_implementation(self,function):
		return self._applyFunctionAlongAxis(function, 'point')

	def _applyFunctionToEachFeature_implementation(self,function):
		return self._applyFunctionAlongAxis(function, 'feature')

	def _applyFunctionAlongAxis(self, function, axisType):
		retData = []
		retRows = []
		retCols = []

		self._sortInternal(axisType)
		if axisType == "feature":
			targetAxis = self.data.col
			otherAxis = self.data.row
			retTarget = retCols
			retOther = retRows
			maxVal = self.points()		
		else:
			targetAxis = self.data.row
			otherAxis = self.data.col
			retTarget = retRows
			retOther = retCols
			maxVal = self.features()

		# consume zeroed vectors, up to the first nonzero value
		if self.data.data[0] != 0:
			for i in xrange(0,targetAxis[0]):
				funcRet = function(VectorView(self, None, None, {}, maxVal, i, axisType))
				retData.append(funcRet)
				retTarget.append(i)
				retOther.append(0)

		vectorStartIndex = 0
		#walk through each coordinate entry
		for i in xrange(len(self.data.data)):
			#collect values into points
			targetValue = targetAxis[i]
			#if this is the end of a point
			if i == len(self.data.data)-1 or targetAxis[i+1] != targetValue:
				#evaluate whether curr point is to be extracted or not
				# and perform the appropriate copies
				funcRet = function(VectorView(self, vectorStartIndex, i, None, maxVal, targetValue, axisType))
				retData.append(funcRet)
				retTarget.append(targetValue)
				retOther.append(0)

				# process zeroed vectors up to the ID of the new vector
				if i < len(self.data.data) - 1:
					nextValue = targetAxis[i+1]
				else:
					if axisType == 'point':
						nextValue = self.points()
					else:
						nextValue = self.features()
				for j in xrange(targetValue+1, nextValue):
					funcRet = function(VectorView(self, None, None, {}, maxVal, j, axisType))
					retData.append(funcRet)
					retTarget.append(targetValue)
					retOther.append(0)

				#reset the vector starting index
				vectorStartIndex = i + 1

		# calculate return data shape
		if axisType == 'point':
			extShape = (self.points(), 1)
		else:
			extShape = (1, self.features())

		# instantiate return data
		ret = coo_matrix((retData,(retRows,retCols)),shape=extShape)
		return CooSparseData(ret) 


	def _transpose_implementation(self):
		"""

		"""
		self.data = self.data.transpose()


	def _mapReduceOnPoints_implementation(self, mapper, reducer):
		self._sortInternal("point")
		mapperResults = {}
		maxVal = self.features()
#		for i in xrange(len(self.data.data)):
#			rowValue = self.data.row[i]
#			if rowValue != currIndex:
#				currResults = mapper(VectorView(currPoint, self.features()))
#				for (k,v) in currResults:
#					if not k in mapperResults:
#						mapperResults[k] = []
#					mapperResults[k].append(v)
#				currPoint = {}
#				currIndex = rowValue
#				nonZeroPoints = nonZeroPoints + 1
#			currPoint[self.data.col[i]] = self.data.data[i]
#
		# run the mapper on the last collected point outside the loop
#		currResults = mapper(VectorView(currPoint, self.features()))
#		for (k,v) in currResults:
#			if k not in mapperResults:
#				mapperResults[k] = []
#			mapperResults[k].append(v)
#		nonZeroPoints = nonZeroPoints + 1

		# consume zeroed vectors, up to the first nonzero value
		if self.data.data[0] != 0:
			for i in xrange(0, self.data.row[0]):
				currResults = mapper(VectorView(self, None, None, {}, maxVal, i, 'point'))
				for (k,v) in currResults:
					if not k in mapperResults:
						mapperResults[k] = []
					mapperResults[k].append(v)

		vectorStartIndex = 0
		#walk through each coordinate entry
		for i in xrange(len(self.data.data)):
			#collect values into points
			targetValue = self.data.row[i]
			#if this is the end of a point
			if i == len(self.data.data)-1 or self.data.row[i+1] != targetValue:
				#evaluate whether curr point is to be extracted or not
				# and perform the appropriate copies
				currResults = mapper(VectorView(self, vectorStartIndex, i, None, maxVal, targetValue, 'point'))
				for (k,v) in currResults:
					if not k in mapperResults:
						mapperResults[k] = []
					mapperResults[k].append(v)

				# process zeroed vectors up to the ID of the new vector
				if i < len(self.data.data) - 1:
					nextValue = self.data.row[i+1]
				else:
					nextValue = self.points()
				for j in xrange(targetValue+1, nextValue):
					currResults = mapper(VectorView(self, None, None, {}, maxVal, j, 'point'))
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
		return CooSparseData(numpy.matrix(ret))


	def _equals_implementation(self,other):
		if not isinstance(other, CooSparseData):
			return False
		if scipy.shape(self.data) != scipy.shape(other.data):
			return False
		# == for scipy sparse types is inconsistent. This is testing how many are
		# nonzero after substracting one from the other.
		return abs(self.data - other.data).nnz == 0

	def _features_implementation(self):
		return self.data.shape[1]

	def _points_implementation(self):
		return self.data.shape[0]

	def _getType_implementation(self):
		return 'CooSparseData'

	def _toRowListData_implementation(self):
		"""	Returns a RowListData object with the same data and featureNames as this one """
		from row_list_data import RowListData
		return RowListData(self.data.todense(), self.featureNames)


	def _toDenseMatrixData_implementation(self):
		""" Returns a DenseMatrixData object with the same data and featureNames as this object """
		from dense_matrix_data import DenseMatrixData
		return DenseMatrixData(self.data.todense(), self.featureNames)


	def _writeFileCSV_implementation(self, outPath, includeFeatureNames):
		"""
		Function to write the data in this object to a CSV file at the designated
		path.

		"""
		outFile = open(outPath, 'w')
	
		if includeFeatureNames and self.featureNames != None:
			pairs = self.featureNames.items()
			# sort according to the value, not the key. ie sort by feature number
			pairs = sorted(pairs,lambda (a,x),(b,y): x-y)
			for (a,x) in pairs:
				if pairs.index((a,x)) == 0:
					outFile.write('#')
				else:
					outFile.write(',')
				outFile.write(str(a))
			outFile.write('\n')

		# sort by rows first, then columns
		placement = numpy.lexsort((self.data.col, self.data.row))
		self.data.data[placement]
		self.data.row[placement]
		self.data.col[placement]

		pointer = 0
		pmax = len(self.data.data)
		for i in xrange(self.points()):
			for j in xrange(self.features()):
				if pointer < pmax and i == self.data.row[pointer] and j == self.data.col[pointer]:
					value = self.data.data[pointer]
					pointer = pointer + 1
				else:
					value = 0

				if j != 0:
					outFile.write(',')	
				outFile.write(str(value))
			outFile.write('\n')

		outFile.close()



	def _writeFileMTX_implementation(self, outPath, includeFeatureNames):
		if includeFeatureNames:
			featureNameString = "#"
			for i in xrange(self.features()):
				featureNameString += self.featureNamesInverse[i]
				if not i == self.features() - 1:
					featureNameString += ','
			
			mmwrite(target=outPath, a=self.data, comment=featureNameString)		
		else:
			mmwrite(target=outPath, a=self.data)


	def _copyReferences_implementation(self, other):
		if not isinstance(other, CooSparseData):
			raise ArgumentException("Other must be the same type as this object")

		self.data = other.data

	def _duplicate_implementation(self):
		return CooSparseData(self.data.copy(), deepcopy(self.featureNames))


	def _copyPoints_implementation(self, points, start, end):
		retData = []
		retRow = []
		retCol = []
		if points is not None:
			for i in xrange(len(self.data.data)):
				if self.data.row[i] in points:
					retData.append(self.data.data[i])
					retRow.append(_numLessThan(self.data.row[i], points))
					retCol.append(self.data.col[i])

			newShape = (len(points), numpy.shape(self.data)[1])
		else:
			for i in xrange(len(self.data.data)):
				if self.data.row[i] >= start and self.data.row[i] <= end:
					retData.append(self.data.data[i])
					retRow.append(self.data.row[i] - start)
					retCol.append(self.data.col[i])

			newShape = (end - start + 1, numpy.shape(self.data)[1])

		return CooSparseData(coo_matrix((retData,(retRow,retCol)),shape=newShape), self.featureNames)


	def _copyFeatures_implementation(self, features, start, end):
		retData = []
		retRow = []
		retCol = []
		if features is not None:
			for i in xrange(len(self.data.data)):
				if self.data.col[i] in features:
					retData.append(self.data.data[i])
					retRow.append(self.data.row[i])
					retCol.append(_numLessThan(self.data.col[i], features))

			newShape = (numpy.shape(self.data)[0], len(features))
			newNames = {}
			for i in xrange(len(features)):
				value = self.featureNamesInverse[features[i]]
				newNames[value] = i
		else:
			for i in xrange(len(self.data.data)):
				if self.data.col[i] >= start and self.data.col[i] <= end:
					retData.append(self.data.data[i])
					retRow.append(self.data.row[i])
					retCol.append(self.data.col[i] - start)

			newShape = (numpy.shape(self.data)[0], end - start + 1)
			newNames = {}
			for i in xrange(start,end+1):
				value = self.featureNamesInverse[i]
				newNames[value] = i - start

		return CooSparseData(coo_matrix((retData,(retRow,retCol)),shape=newShape), newNames)
	

	def _getitem_implementation(self, x, y):
		for i in xrange(len(self.data.row)):
			rowVal = self.data.row[i]
			if rowVal == x and self.data.col[i] == y:
				return self.data.data[i]

		return 0

	def _getPointView_implementation(self, ID):
		nzMap = {}
		#check each value in the matrix
		for i in xrange(len(self.data.data)):
			rowIndex = self.data.row[i]
			if rowIndex == ID:
				nzMap[self.data.col[i]] = i

		return VectorView(self,None,None,nzMap,self.features(),ID,'point')

	def _getFeatureView_implementation(self, ID):
		nzMap = {}
		#check each value in the matrix
		for i in xrange(len(self.data.data)):
			colIndex = self.data.col[i]
			if colIndex == ID:
				nzMap[self.data.row[i]] = i

		return VectorView(self,None,None,nzMap,self.points(),ID,'feature')

	###########
	# Helpers #
	###########

	def _sortInternal(self, axis):
		if axis != 'point' and axis != 'feature':
			raise ArgumentException("invalid axis type")
		# sort least significant axis first
		if axis == "point":
			sortKeys = numpy.argsort(self.data.col)
		else:
			sortKeys = numpy.argsort(self.data.row)
		self.data.data = self.data.data[sortKeys]
		self.data.row = self.data.row[sortKeys]
		self.data.col = self.data.col[sortKeys]

		# then sort by the significant axis
		if axis == "point":
			sortKeys = numpy.argsort(self.data.row)
		else:
			sortKeys = numpy.argsort(self.data.col)
		self.data.data = self.data.data[sortKeys]
		self.data.row = self.data.row[sortKeys]
		self.data.col = self.data.col[sortKeys]

###################
# Generic Helpers #
###################

class VectorView(View):
	"""
	Class to simulate direct random access of a either a single point or feature

	"""
	def __init__(self, CooObject, startInData, endInData, nzMap, maxVal, index, axis):
		if startInData is None and endInData is None and nzMap is None:
			raise ArgumentException("No vector data given as input")
		if nzMap is None and (startInData is None or endInData is None):
			raise ArgumentException("Must provide both start and end in the data")
		self._outer = CooObject
		self._start = startInData
		self._end = endInData
		self._nzMap = nzMap
		self._max = maxVal
		self._index = index
		self._axis = axis
		if axis == "feature":
			self._name = CooObject.featureNamesInverse[index]
		else:
			self._name = None
	def __getitem__(self, key):
		if self._nzMap is None:
			self._makeMap()

		if isinstance(key, slice):
			start = key.start
			stop = key.stop
			if key.start is None:
				start = 0
			if key.stop is None:
				stop = self._max
			retMap = {}
			for mapKey in self._nzMap:
				if mapKey >= start and mapKey <= stop: 
					if key.step is None or (mapKey - start)/key.step == 0:
						retMap[mapKey - start] = self._nzMap[mapKey]
			return VectorView(self._outer, None, None, retMap, self._max-start, self._index, self._axis)
		if isinstance(key, int):
			if key in self._nzMap:
				return self._outer.data.data[self._nzMap[key]]
			elif key >= self._max:
				raise IndexError()
			else:
				return 0
		else:
			raise TypeError()
	def __setitem__(self, key, value):
		if self._nzMap is None:
			self._makeMap()

		if key in self._nzMap:
			self._outer.data.data[self._nzMap[key]] = value
		else:
			self._outer.data.data.append(value)
			if self._axis == 'point':
				self._outer.data.row.append(self._index)
				self._outer.data.col.append(key)
			else:
				self._outer.data.row.append(key)
				self._outer.data.col.append(self._index)
	def nonZeroIterator(self):
		if self._nzMap is not None:
			return nzItMap(self._outer, self._nzMap)
		else:
			return nzItRange(self._outer, self._start, self._end)
	def __len__(self):
		return self._max
	def index(self):
		return self._index
	def name(self):
		return self._name
	def _makeMap(self):
		self._nzMap = {}
		for i in xrange(self._start, self._end+1):
			if self._axis == 'point':
				mapKey = self._outer.data.col[i]
			else:
				mapKey = self._outer.data.row[i]
			self._nzMap[mapKey] = i

class nzItMap():
	def __init__(self, outer, nzMap):
		self._outer = outer
		self._nzMap = nzMap
		self._indices = nzMap.keys()
		self._indices.sort()
		self._position = 0
	def __iter__(self):
		return self
	def next(self):
		while (self._position < len(self._indices)):
			index = self._nzMap[self._indices[self._position]]
			value = self._outer.data.data[index]
			self._position += 1
			if value != 0:
				return value
		raise StopIteration

class nzItRange():
	def __init__(self, outer, start, end):
		self._outer = outer
		self._end = end
		self._position = start
	def __iter__(self):
		return self
	def next(self):
		if self._position > self._end:
			raise StopIteration
		ret = self._outer.data.data[self._position]
		self._position = self._position + 1
		return ret

def _numLessThan(value, toCheck): # TODO caching
	i = 0
	while i < len(toCheck):
		if toCheck[i] < value:
			i = i + 1
		else:
			break

	return i

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

