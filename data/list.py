"""
Class extending Base, using a list of lists to store data.

"""

import copy
import numpy
import scipy
import itertools
from scipy.sparse import isspmatrix

import UML
from base import Base
from base_view import BaseView
from dataHelpers import View
from dataHelpers import reorderToMatchExtractionList
from UML.exceptions import ArgumentException
from UML.randomness import pythonRandom



class List(Base):
	"""
	Class providing implementations of data manipulation operations on data stored
	in a list of lists implementation, where the outer list is a list of
	points of data, and each inner list is a list of values for each feature.

	"""
	def __init__(self, data,  featureNames=None, reuseData=False, shape=None, **kwds):
		self._numFeatures = shape[1] if shape is not None else None
		# Format / copy the data if necessary
		# if input as a list, copy it
		if isinstance(data, list):
			if len(data) > 0 and hasattr(data[0], "__len__") and len(data[0]) > 0:
				if isinstance(data[0][0], list):
					msg = "python lists are not allowed as elements for our "
					msg += "List datatype"
					raise ArgumentException(msg)
			if reuseData:
				data = data
			else:
				data = copy.deepcopy(data)
		# if sparse, make dense
		if isspmatrix(data):
			data = data.todense()
		# if its a numpy construct, convert it to a python list
		try:
			temp = data
			data = data.tolist()
			# if the above was successful, then we might have to exctact emptiness info
			self._numFeatures = temp.shape[1]
		except AttributeError:
			pass

		# assign attributes
		if data is None or len(data) == 0:
			if self._numFeatures is None:
				if featureNames is not None:
					self._numFeatures = len(featureNames) 
				else:
					self._numFeatures = 0
			self.data = []
			shape = (0,self._numFeatures)
			
			kwds['featureNames'] = featureNames
			kwds['shape'] = shape
			super(List, self).__init__(**kwds)
		else:
			self._numFeatures = len(data[0])
			for point in data:
				if len(point) != self._numFeatures:
					raise ArgumentException("Points must be of equal size")
#				if not isinstance(point, list):
#					raise ArgumentException("If a python list is given as input, each entry must also be a list")
			self.data = data
			shape = (len(self.data), self._numFeatures)

			kwds['featureNames'] = featureNames
			kwds['shape'] = shape
			super(List, self).__init__(**kwds)


	def _transpose_implementation(self):
		"""
		Function to transpose the data, ie invert the feature and point indices of the data.
		
		This is not an in place operation, a new list of lists is constructed.
		"""
		tempFeatures = len(self.data)
		transposed = []
		#load the new data with an empty point for each feature in the original
		for i in xrange(self.featureCount):
			transposed.append([])
		for point in self.data:
			for i in xrange(len(point)):
				transposed[i].append(point[i])
		
		self.data = transposed
		self._numFeatures = tempFeatures

	def _appendPoints_implementation(self,toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object
		
		"""
		for pointIndex in xrange(toAppend.pointCount):
			self.data.append(toAppend.data[pointIndex])
	
	def _appendFeatures_implementation(self,toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		"""
		featureLimit = toAppend.featureCount
		for i in xrange(self.pointCount):
			for j in xrange(featureLimit):
				value = toAppend.data[i][j]
				self.data[i].append(value)
		self._numFeatures = self._numFeatures + toAppend.featureCount


	def _sortPoints_implementation(self, sortBy, sortHelper):
		return self._sort_generic_implementation(sortBy, sortHelper, 'point')

	def _sortFeatures_implementation(self, sortBy, sortHelper):
		return self._sort_generic_implementation(sortBy, sortHelper, 'feature')

	def _sort_generic_implementation(self, sortBy, sortHelper, axis):
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

		# make array of views
		viewArray = []
		for v in viewIter:
			viewArray.append(v)

		if comparator is not None:
			viewArray.sort(cmp=comparator)
			indexPosition = []
			for i in xrange(len(viewArray)):
				index = indexGetter(getattr(viewArray[i], nameGetterStr)(0))
				indexPosition.append(index)
		else:
			#scoreArray = viewArray
			scoreArray = []
			if scorer is not None:
				# use scoring function to turn views into values
				for i in xrange(len(viewArray)):
					scoreArray.append(scorer(viewArray[i]))
			else:
				for i in xrange(len(viewArray)):
					scoreArray.append(viewArray[i][sortBy])

			# use numpy.argsort to make desired index array
			# this results in an array whole ith index contains the the
			# index into the data of the value that should be in the ith
			# position
			indexPosition = numpy.argsort(scoreArray)

		# run through target axis and change indices
		if axis == 'point':
			source = copy.copy(self.data)
			for i in xrange(len(self.data)):
				self.data[i] = source[indexPosition[i]]
		else:
			for i in xrange(len(self.data)):
				currPoint = self.data[i]
				temp = copy.copy(currPoint)
				for j in xrange(len(indexPosition)):
					currPoint[j] = temp[indexPosition[j]]

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
		the removed points with default feature names. The actual work is done by further helper
		functions, this determines which helper to call, and modifies the input to accomodate
		the number and randomize parameters, where number indicates how many of the possibilities
		should be extracted, and randomize indicates whether the choice of who to extract should
		be by order or uniform random.

		"""
		# list of identifiers
		if isinstance(toExtract, list):
			if number is None or len(toExtract) < number:
				number = len(toExtract)
			# if randomize, use random sample
			if randomize:
				indices = []
				for i in xrange(len(toExtract)):
					indices.append(i)
				randomIndices = pythonRandom.sample(indices, number)
				randomIndices.sort()
				temp = []
				for index in randomIndices:
					temp.append(toExtract[index])
				toExtract = temp
			# else take the first number members of toExtract
			else:
				toExtract = toExtract[:number]
			return self._extractPointsByList_implementation(toExtract)
		# boolean function
		if hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError  # TODO randomize in the extractPointByFunction case
			else:
				if number is None:
					number = self.pointCount		
				return self._extractPointsByFunction_implementation(toExtract, number)
		# by range
		if start is not None or end is not None:
			if number is None:
				number = end - start
			if randomize:
				# +1 because range is endpoint exclusive
				toExtract = pythonRandom.sample(xrange(start,end+1),number)
				toExtract.sort()
				return self._extractPointsByList_implementation(toExtract)
			else:
				return self._extractPointsByRange_implementation(start, end)

	def _extractPointsByList_implementation(self, toExtract):
		"""
		Modify this object to have only the points that are not listed in toExtract,
		returning an object containing those points that are.

		"""
		toWrite = 0
		satisfying = []
		for i in xrange(self.pointCount):
			if i not in toExtract:
				self.data[toWrite] = self.data[i]
				toWrite += 1
			else:
				satisfying.append(self.data[i])

		# blank out the elements beyond our last copy, ie our last wanted point.
		for index in xrange(toWrite,len(self.data)):
			self.data.pop()

		# construct pointName list
		nameList = []
		for index in toExtract:
			nameList.append(self.getPointName(index))

		extracted = List(satisfying, reuseData=True)
		reorderToMatchExtractionList(extracted, toExtract, 'point')
		extracted.setPointNames(nameList)

		return extracted

	def _extractPointsByFunction_implementation(self, toExtract, number):
		"""
		Modify this object to have only the points that do not satisfy the given function,
		returning an object containing those points that do.

		"""
		toWrite = 0
		satisfying = []
		names = []
		# walk through each point, copying the wanted points back to the toWrite index
		# toWrite is only incremented when we see a wanted point; unwanted points are copied
		# over
		for index in xrange(len(self.data)):
			point = self.data[index]
			if number > 0 and toExtract(self.pointView(index)):
				satisfying.append(point)
				number = number - 1
				names.append(self.getPointName(index))
			else:
				self.data[toWrite] = point
				toWrite += 1

		# blank out the elements beyond our last copy, ie our last wanted point.
		for index in xrange(toWrite,len(self.data)):
			self.data.pop()

		if len(satisfying) == 0:
			rawTrans = []
			for i in xrange(self.featureCount):
				rawTrans.append([])
			ret = List(rawTrans)
			ret.transpose()
			return ret
		else:
			return List(satisfying, pointNames=names, reuseData=True)

	def _extractPointsByRange_implementation(self, start, end):
		"""
		Modify this object to have only those points that are not within the given range,
		inclusive; returning an object containing those points that are.

		"""
		toWrite = start
		inRange = []
		for i in xrange(start,self.pointCount):
			if i <= end:
				inRange.append(self.data[i])		
			else:
				self.data[toWrite] = self.data[i]
				toWrite += 1

		# blank out the elements beyond our last copy, ie our last wanted point.
		for index in xrange(toWrite,len(self.data)):
			self.data.pop()

		# construct featureName list
		nameList = []
		for index in xrange(start,end+1):
			nameList.append(self.getPointName(index))

		return List(inRange, pointNames=nameList, reuseData=True)


	def _extractFeatures_implementation(self, toExtract, start, end, number, randomize):
		"""
		Function to extract features according to the parameters, and return an object containing
		the removed features with their featureNames from this object. The actual work is done by
		further helper functions, this determines which helper to call, and modifies the input
		to accomodate the number and randomize parameters, where number indicates how many of the
		possibilities should be extracted, and randomize indicates whether the choice of who to
		extract should be by order or uniform random.

		"""
		# list of identifiers
		if isinstance(toExtract, list):
			if number is None or len(toExtract) < number:
				number = len(toExtract)
			# if randomize, use random sample
			if randomize:
				indices = []
				for i in xrange(len(toExtract)):
					indices.append(i)
				randomIndices = pythonRandom.sample(indices, number)
				randomIndices.sort()
				temp = []
				for index in randomIndices:
					temp.append(toExtract[index])
				toExtract = temp
			# else take the first number members of toExtract
			else:
				toExtract = toExtract[:number]
			return self._extractFeaturesByList_implementation(toExtract)	
		# boolean function
		if hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError # TODO randomize in the extractFeaturesByFunction case
			else:
				if number is None:
					number = self.pointCount		
				return self._extractFeaturesByFunction_implementation(toExtract, number)
		# by range
		if start is not None or end is not None:
			if start is None:
				start = 0
			if end is None:
				end = self.pointCount
			if number is None:
				number = end - start
			if randomize:
				toExtract = pythonRandom.sample(xrange(start,end),number)
				toExtract.sort()
				return self._extractFeaturesByList_implementation(toExtract)
			else:
				return self._extractFeaturesByRange_implementation(start, end)


	def _extractFeaturesByList_implementation(self, toExtract):
		"""
		Modify this object to have only the features that are not given in the input,
		returning an object containing those features that are, with the same featureNames
		they had previously. It does not modify the featureNames for the calling object.

		"""
		targetPos = {}
		for index in xrange(len(toExtract)):
			targetPos[toExtract[index]] = index

		# we want to extract values from a list from the end
		# for efficiency. So we sort and reverse the removal indices
		toExtractSortRev = copy.copy(toExtract)
		toExtractSortRev.sort()
		toExtractSortRev.reverse()
		extractedData = []
		for point in self.data:
			extractedPoint = [None] * len(toExtract)
			for fID in toExtractSortRev:
				extractedPoint[targetPos[fID]] = point.pop(fID)
			extractedData.append(extractedPoint)

		self._numFeatures = self._numFeatures - len(toExtract)

		# construct featureName list from the original unsorted toExtract
		featureNameList = []
		for index in toExtract:
			featureNameList.append(self.getFeatureName(index))

		return List(extractedData, featureNames=featureNameList, reuseData=True)


	def _extractFeaturesByFunction_implementation(self, function, number):
		"""
		Modify this object to have only the features whose views do not satisfy the given
		function, returning an object containing those features whose views do, with the
		same featureNames	they had previously. It does not modify the featureNames for the calling object.

		"""
		# all we're doing is making a list and calling extractFeaturesBy list, no need
		# deal with featureNames or the number of features.
		toExtract = []
		for i, ithView in enumerate(self.featureIterator()):
			if function(ithView):
				toExtract.append(i)
		return self._extractFeaturesByList_implementation(toExtract)


	def _extractFeaturesByRange_implementation(self, start, end):
		"""
		Modify this object to have only those features that are not within the given range,
		inclusive; returning an object containing those features that are, with the same featureNames
		they had previously. It does not modify the featureNames for the calling object.
		"""
		extractedData = []
		for point in self.data:
			extractedPoint = []
			#end + 1 because our ranges are inclusive, xrange's are not
			for index in reversed(xrange(start,end+1)):
				extractedPoint.append(point.pop(index))
			extractedPoint.reverse()
			extractedData.append(extractedPoint)

		self._numFeatures = self._numFeatures - len(extractedPoint)

		# construct featureName list
		featureNameList = []
		for index in xrange(start,end+1):
			featureNameList.append(self.getFeatureName(index))
	
		return List(extractedData, featureNames=featureNameList, reuseData=True)

	def _mapReducePoints_implementation(self, mapper, reducer):
		mapResults = {}
		# apply the mapper to each point in the data
		for i in xrange(self.pointCount):
			currResults = mapper(self.pointView(i))
			# the mapper will return a list of key value pairs
			for (k,v) in currResults:
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
				(redKey,redValue) = redRet
				ret.append([redKey,redValue])
		return List(ret, reuseData=True)

	def _getTypeString_implementation(self):
		return 'List'

	def _isIdentical_implementation(self,other):
		if not isinstance(other,List):
			return False
		if self.pointCount != other.pointCount:
			return False
		if self.featureCount != other.featureCount:
			return False
		for index in xrange(self.pointCount):
			if self.data[index] != other.data[index]:
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

		for point in self.pointIterator():
			currPname = point.getPointName(0)
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

	def _writeFileMTX_implementation(self, outPath, includePointNames, includeFeatureNames):
		"""
		Function to write the data in this object to a matrix market file at the designated
		path.

		"""
		outFile = open(outPath, 'w')
		outFile.write("%%MatrixMarket matrix array real general\n")
		
		def writeNames(nameList):
			for i,n in enumerate(nameList):
				if i == 0:
					outFile.write('%#')
				else:
					outFile.write(',')
				outFile.write(n)
			outFile.write('\n')

		if includePointNames:
			writeNames(self.getPointNames())
		else:
			outFile.write('%#\n')
		if includeFeatureNames:
			writeNames(self.getFeatureNames())
		else:
			outFile.write('%#\n')

		outFile.write(str(self.pointCount) + " " + str(self.featureCount) + "\n")

		for j in xrange(self.featureCount):
			for i in xrange(self.pointCount):
				value = self.data[i][j]
				outFile.write(str(value) + '\n')
		outFile.close()

	def _referenceDataFrom_implementation(self, other):
		if not isinstance(other, List):
			raise ArgumentException("Other must be the same type as this object")

		self.data = other.data

	def _copyAs_implementation(self, format):
		if format == 'Sparse':
			if self.pointCount == 0 or self.featureCount == 0:
				emptyData = numpy.empty(shape=(self.pointCount, self.featureCount))
				return UML.data.Sparse(emptyData, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
			return UML.data.Sparse(self.data, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
		if format is None or format == 'List':
			return UML.data.List(self.data, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
		if format == 'Matrix':
			return UML.data.Matrix(self.data, pointNames=self.getPointNames(), featureNames=self.getFeatureNames())
		if format == 'pythonlist':
			return copy.deepcopy(self.data)
		if format == 'numpyarray':
			if self.pointCount == 0 or self.featureCount == 0:
				return numpy.empty(shape=(self.pointCount, self.featureCount))
			return numpy.array(self.data)
		if format == 'numpymatrix':
			if self.pointCount == 0 or self.featureCount == 0:
				return numpy.matrix(numpy.empty(shape=(self.pointCount, self.featureCount)))
			return numpy.matrix(self.data)
		if format == 'scipycsc':
			return scipy.sparse.csc_matrix(numpy.array(self.data))
		if format == 'scipycsr':
			return scipy.sparse.csr_matrix(numpy.array(self.data))

	def _copyPoints_implementation(self, points, start, end):
		retData = []
		if points is not None:
			for index in points:
				retData.append(copy.copy(self.data[index]))
		else:
			for i in range(start,end+1):
				retData.append(copy.copy(self.data[i]))

		return List(retData, reuseData=True)

	def _copyFeatures_implementation(self, indices, start, end):
		if self.pointCount == 0:
			ret = []
			count = len(indices) if indices is not None else (end + 1 - start)
			for i in range(count):
				ret.append([])
			retObj = List(ret)
			retObj.transpose()
			return retObj

		ret = []
		for point in self.data:
			retPoint = []
			if indices is not None:
				for i in indices:
					retPoint.append(point[i])
			else:
				for i in range(start,end+1):
					retPoint.append(point[i])
			ret.append(retPoint)

		return List(ret, reuseData=True)

	def _transformEachPoint_implementation(self, function, points):
		for i, p in enumerate(self.pointIterator()):
			if points is not None and i not in points:
				continue
			currRet = function(p)
			if len(currRet) != self.featureCount:
				msg = "function must return an iterable with as many elements as features in this object"
				raise ArgumentException(msg)

			self.data[i] = currRet

	def _transformEachFeature_implementation(self, function, features):
		for j, f in enumerate(self.featureIterator()):
			if features is not None and j not in features:
				continue
			currRet = function(f)
			if len(currRet) != self.pointCount:
				msg = "function must return an iterable with as many elements as points in this object"
				raise ArgumentException(msg)

			for i in xrange(self.pointCount):
				self.data[i][j] = currRet[i]

	def _transformEachElement_implementation(self, function, points, features, preserveZeros, skipNoneReturnValues):
		oneArg = False
		try:
			function(0,0,0)
		except TypeError:
			oneArg = True

		IDs = itertools.product(xrange(self.pointCount), xrange(self.featureCount))
		for (i,j) in IDs:
			currVal = self.data[i][j]

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

			self.data[i][j] = currRet



	def _getitem_implementation(self, x, y):
		return self.data[x][y]

	def _pointView_implementation(self, ID):
		return PointView(self, self.data[ID], ID)

	def _featureView_implementation(self, ID):
		return FeatureView(self, ID, self.getFeatureName(ID))


	def _view_implementation(self, pointStart, pointEnd, featureStart, featureEnd):
		class ListView(BaseView, List):
			def __init__(self, **kwds):
				super(ListView, self).__init__(**kwds)

			def _copyAs_implementation(self, format):
				# we only want to change how List and pythonlist copying is done
				if format is None:
					format = 'List'
				if format != 'List' and format != 'pythonlist':
					return super(ListView, self)._copyAs_implementation(format)

				listForm = []
				for pID in xrange(self._pStart, self._pEnd):
					curr = []
					for fID in xrange(self._fStart, self._fEnd):
						curr.append(self._source.data[pID][fID])
					listForm.append(curr)

				if format == 'List':
					return UML.data.List(listForm, pointNames=self.getPointNames(),
						featureNames=self.getFeatureNames())
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

				return self.source.data[self.limit][key+self.fStart]

			def __len__(self):
				return self.fRange

			def __eq__(self, other):
				for i, val in enumerate(self):
					if val != other[i]:
						return False
				return True

			def __ne__(self, other):
				return not self.__eq__(other)

		class ListPassThrough(object):
			def __init__(self, source, pStart, pEnd, fStart, fEnd):
				self.source = source
				self.pStart = pStart
				self.pRange = pEnd - pStart
				self.fviewer = FeatureViewer(self.source, fStart, fEnd)

			def __getitem__(self, key):
				if key < 0 or key >= self.pRange:
					raise IndexError("")

				self.fviewer.setLimit(key + self.pStart)
				return self.fviewer

			def __len__(self):
				return self.pRange

		kwds = {}
		kwds['data'] = ListPassThrough(self, pointStart, pointEnd, featureStart, featureEnd)
		kwds['source'] = self
		kwds['pointStart'] = pointStart
		kwds['pointEnd'] = pointEnd
		kwds['featureStart'] = featureStart
		kwds['featureEnd'] = featureEnd
		kwds['reuseData'] = True
		kwds['shape'] = (pointEnd - pointStart, featureEnd - featureStart)

		return ListView(**kwds)

	def _validate_implementation(self, level):
		assert len(self.data) == self.pointCount
		assert self._numFeatures == self.featureCount

		if level > 0:
			if len(self.data) > 0:
				expectedLength = len(self.data[0])
			for point in self.data:
#				assert isinstance(point, list)
				assert len(point) == expectedLength

	def _containsZero_implementation(self):
		"""
		Returns True if there is a value that is equal to integer 0 contained
		in this object. False otherwise

		"""
		for point in self.pointIterator():
			for i in range(len(point)):
				if point[i] == 0:
					return True
		return False

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
					value = self._source.data[self._pIndex][self._fIndex]

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
					value = self._source.data[self._pIndex][self._fIndex]

					self._pIndex += 1
					if self._pIndex >= self._pStop:
						self._pIndex = 0
						self._fIndex += 1

					if value != 0:
						return value

				raise StopIteration

		return nzIt(self)


	def _mul__implementation(self, other):
		if isinstance(other, UML.data.Base):
			return self._matrixMultiply_implementation(other)
		else:
			ret = self.copy()
			ret._scalarMultiply_implementation(other)
			return ret

	def _matrixMultiply_implementation(self, other):
		"""
		Matrix multiply this UML data object against the provided other UML data
		object. Both object must contain only numeric data. The featureCount of
		the calling object must equal the pointCount of the other object. The
		types of the two objects may be different, and the return is guaranteed
		to be the same type as at least one out of the two, to be automatically
		determined according to efficiency constraints. 

		"""
		ret = []
		for sPoint in self.pointIterator():
			retP = []
			for oFeature in other.featureIterator():
				runningTotal = 0
				for index in xrange(other.pointCount):
					runningTotal += sPoint[index] * oFeature[index]
				retP.append(runningTotal)
			ret.append(retP)
		return List(ret)


	def _elementwiseMultiply_implementation(self, other):
		"""
		Perform element wise multiplication of this UML data object against the
		provided other UML data object. Both objects must contain only numeric
		data. The pointCount and featureCount of both objects must be equal. The
		types of the two objects may be different, but the returned object will
		be the inplace modification of the calling object.

		"""
		for pNum in xrange(self.pointCount):
			for fNum in xrange(self.featureCount):
				self.data[pNum][fNum] *= other[pNum,fNum]

	def _scalarMultiply_implementation(self, scalar):
		"""
		Multiply every element of this UML data object by the provided scalar.
		This object must contain only numeric data. The 'scalar' parameter must
		be a numeric data type. The returned object will be the inplace modification
		of the calling object.
		
		"""
		for point in self.data:
			for i in xrange(len(point)):
				point[i] *= scalar






###########
# Helpers #
###########

class FeatureView(View):
	"""
	Class to simulate direct random access of a feature, along with other helpers.

	"""
	def __init__(self, outer, colNum, colName):
		self._outer = outer
		self._data = outer.data
		self._colNum = colNum
		self._colName = colName
	def __getitem__(self, index):
		if isinstance(index, basestring):
			index = self._outer.getPointIndex(index)
		point = self._data[index]
		value = point[self._colNum]
		return value	
	def __setitem__(self, key, value):
		if isinstance(key, basestring):
			key = self._outer.getPointIndex(key)
		point = self._data[key]
		point[self._colNum] = value
	def nonZeroIterator(self):
		return nzIt(self)
	def __len__(self):
		return len(self._data)
	def index(self):
		return self._colNum
	def name(self):
		return self._colName
	def getPointName(self, index):
		return self._outer.getPointName(index)
	def getFeatureName(self, index):
		return self._outer.getFeatureName(self._colNum)

class PointView(View):
	"""
	Class to wrap direct random access of a point, along with other helpers.

	"""
	def __init__(self, outer, point, index):
		self._outer = outer
		self._point = point
		self._index = index
	def __getitem__(self, key):
		if isinstance(key, basestring):
			key = self._outer.getFeatureIndex(key)
		return self._point[key]	
	def __setitem__(self, key, value):
		if isinstance(key, basestring):
			key = self._outer.getFeatureIndex(key)
		self._point[key] = value
	def nonZeroIterator(self):
		return nzIt(self)
	def __len__(self):
		return len(self._point)
	def index(self):
		return self._index
	def name(self):
		return self._outer.getPointName(self._index)
	def getPointName(self, index):
		return self._outer.getPointName(self._index)
	def getFeatureName(self, index):
		return self._outer.getFeatureName(index)

class nzIt():
	def __init__(self, indexable):
		self._indexable = indexable
		self._position = 0
	def __iter__(self):
		return self
	def next(self):
		while (self._position < len(self._indexable)):
			value = self._indexable[self._position]
			self._position += 1
			if value != 0:
				return value
		raise StopIteration




