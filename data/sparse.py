"""
Class extending Base, defining an object to hold and manipulate a scipy coo_matrix.

"""


import numpy
import scipy
import random
from scipy.sparse import coo_matrix
from scipy.io import mmwrite
import copy

import UML
from base import Base
from dataHelpers import View
from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException


class Sparse(Base):

	def __init__(self, data=None, featureNames=None, name=None, path=None, reuseData=False):
		self._sorted = None
		if data == [] or (hasattr(data,'shape') and (data.shape[0] == 0 or data.shape[1] == 0)):
			rowShape = 0
			colShape = 0
			if hasattr(data,'shape'):
				rowShape = data.shape[0] 
				colShape = data.shape[1]
			if featureNames is not None and colShape == 0:
				colShape = len(featureNames)
				data = numpy.empty(shape=(rowShape,colShape))
			if isinstance(data, CooWithEmpty):
				data = data.internal
			data = numpy.matrix(data, dtype=numpy.float)
			self._data = CooWithEmpty(data)
		else:
			if scipy.sparse.isspmatrix(data):
				if reuseData:
					self._data = CooWithEmpty(data, reuseData=True)
				else:
					self._data = CooWithEmpty(data.copy())
			else:
				self._data = CooWithEmpty(data)
		
		shape = scipy.shape(self._data)
		super(Sparse, self).__init__(shape,featureNames, name, path)


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
		if self.featureCount == 0:
			raise ImproperActionException("We do not allow iteration over points if there are 0 features")
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
					value = VectorView(self._outer, self._sortedPosition, end, None, self._outer.featureCount, self._nextID, 'point')
					self._sortedPosition = end
				self._nextID += 1
				return value

		return pointIt(self)


	def featureIterator(self):
		if self.pointCount == 0:
			raise ImproperActionException("We do not allow iteration over features if there are 0 points")

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
					value = VectorView(self._outer, self._sortedPosition, end, None, self._outer.pointCount, self._nextID, 'feature')
					self._sortedPosition = end
				self._nextID += 1
				return value
		return featureIt(self)

	def _applyTo_implementation(self, function, included, inPlace, axis):
		retData = []
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

		for view in viewIterator:
			viewID = view.index()
			if included is not None and viewID not in included:
				continue
			currOut = function(view)
			# first we branch on whether the output has multiple values or is singular.
			if hasattr(currOut, '__iter__'):
				# if there are multiple values, they must be random accessible
				if not hasattr(currOut, '__getitem__'):
					raise ArgumentException("function must return random accessible data (ie has a __getitem__ attribute)")
				if inPlace:
					for i in xrange(len(currOut)):
						if view[i] != 0:
							view[i] = currOut[i]
						else:
							modData.append(currOut[i])
							modTarget.append(viewID)
							modOther.append(i)
				else:
					toCopyInto = []
					for value in currOut:
						toCopyInto.append(value)
					retData.append(toCopyInto)
			# singular return
			else:
				if inPlace:
					if view[0] != 0:
						view[0] = currOut
					else:
							modData.append(currOut[i])
							modTarget.append(viewID)
							modOther.append(i)
				else:
					retData.append([currOut])

		if inPlace and len(modData) != 0:
			newData = numpy.append(self._data.data, modData)
			newRow = numpy.append(self._data.row, modRow)
			newCol = numpy.append(self._data.col, modCol)
			self._data = CooWithEmpty((newData,(newRow,newCol)),shape=(self.pointCount,self.featureCount))
			self._sorted = None

		if inPlace:
			ret = self
		else:
			ret = UML.createData(self.getTypeString(), retData)
			if axis != 'point':
				ret.transpose()

		return ret


	def _appendPoints_implementation(self, toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object
		
		"""
		newData = numpy.append(self._data.data, toAppend._data.data)
		newRow = numpy.append(self._data.row, toAppend._data.row)
		newCol = numpy.append(self._data.col, toAppend._data.col)

		# correct the row entries
		offset = self.pointCount
		for index in xrange(len(self._data.data), len(newData)):
			newRow[index] = newRow[index] + offset
		
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
		for index in xrange(len(self._data.data), len(newData)):
			newCol[index] = newCol[index] + offset
		
		numNewCols = self.featureCount + toAppend.featureCount
		self._data = CooWithEmpty((newData,(newRow,newCol)),shape=(self.pointCount,numNewCols))
		if self._sorted == 'point':
			self._sorted = None


	def _sortPoints_implementation(self, sortBy, sortHelper):
		self.sort_general_implementation(sortBy, sortHelper, 'point')
		self._sorted = None


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
		else:
			viewMaker = self.featureView
			targetAxis = self._data.col
			getViewIter = self.featureIterator

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

		# make array of views
		viewArray = []
		viewIter = getViewIter()
		for v in viewIter:
			viewArray.append(v)

		if comparator is not None:
			viewArray.sort(cmp=comparator)
			indexPosition = []
			for i in xrange(len(viewArray)):
				indexPosition.append(viewArray[i].index())
		else:
			scoreArray = viewArray
			if scorer is not None:
				# use scoring function to turn views into values
				for i in xrange(len(viewArray)):
					scoreArray[i] = scorer(viewArray[i])
			else:
				for i in xrange(len(viewArray)):
					scoreArray[i] = viewArray[i][sortBy]

			# use numpy.argsort to make desired index array
			# this results in an array whole ith index contains the the
			# index into the data of the value that should be in the ith
			# position
			indexPosition = numpy.argsort(scoreArray)

		# run through array making curr index to new index map
		indexMap = {}
		for i in xrange(len(indexPosition)):
			indexMap[indexPosition[i]] = i
		# run through target axis and change indices
		for i in xrange(len(targetAxis)):
			targetAxis[i] = indexMap[targetAxis[i]]

		# if we are sorting features we need to return an array of the feature names
		# in their new order
		if axisType == 'feature':
			# we convert the indices of the their previous location into their feature names
			newFeatureNameOrder = []
			for i in xrange(len(indexPosition)):
				oldIndex = indexPosition[i]
				newName = self.featureNamesInverse[oldIndex]
				newFeatureNameOrder.append(newName)
			return newFeatureNameOrder




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
					number = self.pointCount		
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
				toExtractIndices.append(self._getFeatureIndex(value))
			return self._extractByList_implementation(toExtractIndices, "feature")
		# boolean function
		if hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError # TODO randomize in the extractFeatureByFunction case
			else:
				if number is None:
					number = self.pointCount		
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
			if extractIndex < extractLength and value > toExtractSorted[extractIndex]:
				extractIndex = extractIndex + 1
			if extractIndex < extractLength and value == toExtractSorted[extractIndex]:
				extractData.append(self._data.data[i])
				extractOther.append(otherAxis[i])
				extractTarget.append(positionMap[value])	
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
		
		# get featureNames for return obj
		featureNames = []
		if axisType == "feature":
			for index in toExtract:
				featureNames.append(self.featureNamesInverse[index])
		else:
			featureNames = self.featureNames

		return Sparse(ret, featureNames, reuseData=True) 


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
			maxVal = self.pointCount		
		else:
			targetAxis = self._data.row
			otherAxis = self._data.col
			extractTarget = extractRows
			extractOther = extractCols
			maxVal = self.featureCount

		copyIndex = 0
		vectorStartIndex = 0
		extractedIDs = []
		# consume zeroed vectors, up to the first nonzero value
		if self._data.nnz > 0 and self._data.data[0] != 0:
			for i in xrange(0,targetAxis[0]):
				if toExtract(VectorView(self, None, None, {}, maxVal, i, axisType)):
					extractedIDs.append(i)
		#walk through each coordinate entry
		for i in xrange(len(self._data.data)):
			#collect values into points
			targetValue = targetAxis[i]
			#if this is the end of a point
			if i == len(self._data.data)-1 or targetAxis[i+1] != targetValue:
				#evaluate whether curr point is to be extracted or not
				# and perform the appropriate copies
				if toExtract(VectorView(self, vectorStartIndex, i, None, maxVal, targetValue, axisType)):
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
					if toExtract(VectorView(self, None, None, {}, maxVal, j, axisType)):
						extractedIDs.append(j)

				#reset the vector starting index
				vectorStartIndex = i + 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		(selfShape, extShape) = _calcShapes(self._data.shape, len(extractedIDs), axisType)
		self._data = CooWithEmpty( (self._data.data[0:copyIndex],(self._data.row[0:copyIndex],self._data.col[0:copyIndex])), selfShape)

		# instantiate return data
		ret = CooWithEmpty((extractData,(extractRows,extractCols)),shape=extShape)
		
		# get featureNames for return obj
		featureNames = []
		if axisType == "feature":
			for index in extractedIDs:
				featureNames.append(self.featureNamesInverse[index])
		else:
			featureNames = self.featureNames

		return Sparse(ret, featureNames, reuseData=True) 





	def _extractByRange_implementation(self, start, end, axisType):
		"""
		

		"""
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
					targetAxis[copyIndex] = targetAxis[i] - (end + 1)
				copyIndex = copyIndex + 1

		# reinstantiate self
		# (cannot reshape coo matrices, so cannot do this in place)
		(selfShape, extShape) = _calcShapes(self._data.shape, rangeLength, axisType)
		self._data = CooWithEmpty( (self._data.data[0:copyIndex],(self._data.row[0:copyIndex],self._data.col[0:copyIndex])), selfShape)

		# instantiate return data
		ret = CooWithEmpty((extractData,(extractRows,extractCols)),shape=extShape)
		
		# get featureNames for return obj
		featureNames = []
		if axisType == "feature":
			for i in xrange(start,end+1):
				featureNames.append(self.featureNamesInverse[i])
		else:
			featureNames = self.featureNames

		return Sparse(ret, featureNames, reuseData=True) 

	def _transpose_implementation(self):
		"""

		"""
#		import pdb
#		pdb.set_trace()
		self._data = self._data.transpose()
		self._sorted = None
#		if self._sorted == 'point':
#			self._sorted = 'feature'
#		elif self._sorted == 'feature':
#			self._sorted = 'point'


	def _mapReducePoints_implementation(self, mapper, reducer):
		self._sortInternal("point")
		mapperResults = {}
		maxVal = self.featureCount
#		for i in xrange(len(self._data.data)):
#			rowValue = self._data.row[i]
#			if rowValue != currIndex:
#				currResults = mapper(VectorView(currPoint, self.featureCount))
#				for (k,v) in currResults:
#					if not k in mapperResults:
#						mapperResults[k] = []
#					mapperResults[k].append(v)
#				currPoint = {}
#				currIndex = rowValue
#				nonZeroPoints = nonZeroPoints + 1
#			currPoint[self._data.col[i]] = self._data.data[i]
#
		# run the mapper on the last collected point outside the loop
#		currResults = mapper(VectorView(currPoint, self.featureCount))
#		for (k,v) in currResults:
#			if k not in mapperResults:
#				mapperResults[k] = []
#			mapperResults[k].append(v)
#		nonZeroPoints = nonZeroPoints + 1

		# consume zeroed vectors, up to the first nonzero value
		if self._data.data[0] != 0:
			for i in xrange(0, self._data.row[0]):
				currResults = mapper(VectorView(self, None, None, {}, maxVal, i, 'point'))
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
				currResults = mapper(VectorView(self, vectorStartIndex, i, None, maxVal, targetValue, 'point'))
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
		return Sparse(numpy.matrix(ret))

	def _isIdentical_implementation(self, other):
		if not isinstance(other, Sparse):
			return False
		# for nonempty matrices, we use a shape mismatch to indicate non-equality
		if self._data.shape != other._data.shape:
			return False

		return _wrapperEquals(self._data, other._data)

	def _getTypeString_implementation(self):
		return 'Sparse'

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
		placement = numpy.lexsort((self._data.col, self._data.row))
		self._data.data[placement]
		self._data.row[placement]
		self._data.col[placement]

		pointer = 0
		pmax = len(self._data.data)
		for i in xrange(self.pointCount):
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



	def _writeFileMTX_implementation(self, outPath, includeFeatureNames):
		if includeFeatureNames:
			featureNameString = "#"
			for i in xrange(self.featureCount):
				featureNameString += self.featureNamesInverse[i]
				if not i == self.featureCount - 1:
					featureNameString += ','
			
			mmwrite(target=outPath, a=self._data.internal, comment=featureNameString)		
		else:
			mmwrite(target=outPath, a=self._data.internal)


	def _referenceDataFrom_implementation(self, other):
		if not isinstance(other, Sparse):
			raise ArgumentException("Other must be the same type as this object")

		self._data = other._data
		self._sorted = None

	def _copyAs_implementation(self, format, rowsArePoints, outputAs1D):
		if format is None or format == 'Sparse':
			return Sparse(self._data.internal, self.featureNames)
		if format == 'List':
			return UML.data.List(self._data.internal, self.featureNames)
		if format == 'Matrix':
			return UML.data.Matrix(self._data.internal, self.featureNames)
		if format == 'pythonlist':
			return self._data.todense().tolist()
		if format == 'numpyarray':
			return numpy.array(self._data.todense())
		if format == 'numpymatrix':
			return self._data.todense()

	def _copyPoints_implementation(self, points, start, end):
		retData = []
		retRow = []
		retCol = []
		if points is not None:
			for i in xrange(len(self._data.data)):
				if self._data.row[i] in points:
					retData.append(self._data.data[i])
					retRow.append(_numLessThan(self._data.row[i], points))
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
		return Sparse(retData, self.featureNames, reuseData=True)


	def _copyFeatures_implementation(self, features, start, end):
		retData = []
		retRow = []
		retCol = []
		if features is not None:
			for i in xrange(len(self._data.data)):
				if self._data.col[i] in features:
					retData.append(self._data.data[i])
					retRow.append(self._data.row[i])
					retCol.append(_numLessThan(self._data.col[i], features))

			newShape = (numpy.shape(self._data)[0], len(features))
			newNames = {}
			for i in xrange(len(features)):
				value = self.featureNamesInverse[features[i]]
				newNames[value] = i
		else:
			for i in xrange(len(self._data.data)):
				if self._data.col[i] >= start and self._data.col[i] <= end:
					retData.append(self._data.data[i])
					retRow.append(self._data.row[i])
					retCol.append(self._data.col[i] - start)

			newShape = (numpy.shape(self._data)[0], end - start + 1)
			newNames = {}
			for i in xrange(start,end+1):
				value = self.featureNamesInverse[i]
				newNames[value] = i - start

		retData = CooWithEmpty((retData,(retRow,retCol)),shape=newShape)
		return Sparse(retData, newNames, reuseData=True)
	

	def _getitem_implementation(self, x, y):
		for i in xrange(len(self._data.row)):
			rowVal = self._data.row[i]
			if rowVal == x and self._data.col[i] == y:
				return self._data.data[i]

		return 0

	def _pointView_implementation(self, ID):
		nzMap = {}
		#check each value in the matrix
		for i in xrange(len(self._data.data)):
			rowIndex = self._data.row[i]
			if rowIndex == ID:
				nzMap[self._data.col[i]] = i

		return VectorView(self,None,None,nzMap,self.featureCount,ID,'point')

	def _featureView_implementation(self, ID):
		nzMap = {}
		#check each value in the matrix
		for i in xrange(len(self._data.data)):
			colIndex = self._data.col[i]
			if colIndex == ID:
				nzMap[self._data.row[i]] = i

		return VectorView(self,None,None,nzMap,self.pointCount,ID,'feature')


	def _validate_implementation(self, level):
		assert self._data.shape[0] == self.pointCount
		assert self._data.shape[1] == self.featureCount
		assert isinstance(self._data, CooWithEmpty)

		if level > 0:
			for value in self._data.data:
				assert value != 0

			if self._sorted == 'point':
				for i in xrange(len(self._data.row)):
					if i != len(self._data.row) - 1:
						assert self._data.row[i] <= self._data.row[i+1]
			if self._sorted == 'feature':
				for i in xrange(len(self._data.col)):
					if i != len(self._data.col) - 1:
						assert self._data.col[i] <= self._data.col[i+1]



	###########
	# Helpers #
	###########

	def _sortInternal(self, axis):
		if axis != 'point' and axis != 'feature':
			raise ArgumentException("invalid axis type")
		# sort least significant axis first
		if axis == "point":
			sortKeys = numpy.argsort(self._data.col)
		else:
			sortKeys = numpy.argsort(self._data.row)
		newData = self._data.data[sortKeys]
		newRow = self._data.row[sortKeys]
		newCol = self._data.col[sortKeys]

		# then sort by the significant axis
		if axis == "point":
			sortKeys = numpy.argsort(newRow)
		else:
			sortKeys = numpy.argsort(newCol)
		newData = newData[sortKeys]
		newRow = newRow[sortKeys]
		newCol = newCol[sortKeys]

		for i in xrange(len(newData)):
			self._data.data[i] = newData[i]
		for i in xrange(len(newRow)):
			self._data.row[i] = newRow[i]
		for i in xrange(len(newCol)):
			self._data.col[i] = newCol[i]

		# flag that we are internally sorted
		self._sorted = axis

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
		self._start = startInData # inclusive
		self._end = endInData # exclusive
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
		elif isinstance(key, int):
			if key in self._nzMap:
				return self._outer._data.data[self._nzMap[key]]
			elif key >= self._max:
				raise IndexError('key is greater than the max possible value')
			else:
				return 0
		elif isinstance(key, basestring) and self._axis =='point':
			index = self._outer.featureNames[key]
			if index in self._nzMap:
				return self._outer._data.data[self._nzMap[index]]
			else:
				return 0
		else:
			raise TypeError('key is not a recognized type')
	def __setitem__(self, key, value):
		if self._nzMap is None:
			self._makeMap()

		if isinstance(key, basestring):
			if self._axis =='point':
				key = self._outer.featureNames[key]
			else:
				raise TypeError('key is not a recognized type')

		if key in self._nzMap:
			self._outer._data.data[self._nzMap[key]] = value
		else:
			raise ArgumentException("Sparse objects do not support element level modification of zero valued entries")
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
		if self._start >= len(self._outer._data.data):
			return
		for i in xrange(self._start, self._end):
			if self._axis == 'point':
				mapKey = self._outer._data.col[i]
			else:
				mapKey = self._outer._data.row[i]
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
			value = self._outer._data.data[index]
			self._position += 1
			if value != 0:
				return value
		raise StopIteration

class nzItRange():
	def __init__(self, outer, start, end):
		self._outer = outer
		self._end = end # exclusive
		self._position = start
	def __iter__(self):
		return self
	def next(self):
		if self._position >= self._end:
			raise StopIteration
		ret = self._outer._data.data[self._position]
		self._position = self._position + 1
		if ret != 0:
			return ret
		else:
			return self.next()

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


def _wrapperEquals(left, right):
		wrappedLeft = isinstance(left, CooWithEmpty)
		wrappedRight = isinstance(right, CooWithEmpty)
		if not wrappedLeft and wrappedRight:
			return False

		if left.shape != right.shape:
			return False

		leftSparse = scipy.sparse.isspmatrix(left.internal)
		rightSparse = scipy.sparse.isspmatrix(right.internal)
		if leftSparse:
			if rightSparse:
				# == for scipy sparse types is inconsistent. This is testing how many are
				# nonzero after substracting one from the other.
				return abs(left.internal - right.internal).nnz == 0
			else:
				return False
		else:
			if rightSparse:
				return False
			else:
				return True

class CooWithEmpty(object):

	def __init__(self, arg1, shape=None, dtype=None, copy=False, reuseData=False):
		self.ndim = 2
		try:
			if isinstance(arg1, CooWithEmpty):
				arg1 = arg1.internal
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

		except ValueError as ve:
			if str(ve) == 'invalid shape':
				if isinstance(arg1, tuple):
					if shape is None:
						self.shape = (0,0)
					else:
						self.shape = shape
				else:
					self.shape = arg1.shape
				self.nnz = 0
				self.data = numpy.array([])
				self.row = numpy.array([])
				self.col = numpy.array([])
				self.internal = numpy.empty(self.shape)


	def transpose(self):
		self.internal = self.internal.transpose()
		self.shape = self.internal.shape
		return self

	def todense(self):
		if scipy.sparse.isspmatrix(self.internal):
			return self.internal.todense()
		else:
			return numpy.empty(self.shape)


