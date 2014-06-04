"""
Anchors the hierarchy of data representation types, providing stubs and common functions.

"""

# TODO conversions
# TODO who sorts inputs to derived implementations?

import random
import math
import itertools
import copy
import numpy

import UML
from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException
from UML.logger import produceFeaturewiseReport
from UML.logger import produceAggregateReport

import dataHelpers
# a default seed for testing and predictible trials
from dataHelpers import DEFAULT_SEED

# the prefix for default featureNames
from dataHelpers import DEFAULT_PREFIX

from dataHelpers import DEFAULT_NAME_PREFIX


class Base(object):
	"""
	Class defining important data manipulation operations and giving functionality
	for the naming the features of that data. A mapping from feature names to feature
	indices is given by the featureNames attribute, the inverse of that mapping is
	given by featureNamesInverse.

	"""

	def __init__(self, shape, pointNames=None, featureNames=None, name=None, path=None):
		"""
		Instantiates the featureName book-keeping structures that are defined by this representation.
		
		featureNames may be None if the object is to have default names, or a list or dict defining the
		featureName mapping.

		"""
		self._pointCount = shape[0]
		self._featureCount = shape[1]

		if pointNames is not None and len(pointNames) != shape[0]:
			raise ArgumentException("The length of the pointNames must match the points given in shape")
		if featureNames is not None and len(featureNames) != shape[1]:
			raise ArgumentException("The length of the featureNames must match the features given in shape")

		self._nextDefaultValuePoint = 0
		self._setAllDefault('point')
		if isinstance(pointNames, list) or pointNames is None:
			self.setPointNamesFromList(pointNames)
		elif isinstance(pointNames, dict):
			self.setPointNamesFromDict(pointNames)
		else:
			raise ArgumentException("pointNames may only be a list or a dict, defining a mapping between integers and pointNames")
		if pointNames is not None and len(pointNames) != self.pointCount:
			raise ArgumentException("Cannot have different number of pointNames and points, len(pointNames): " + str(len(pointNames)) + ", self.pointCount: " + str(self.pointCount))

		self._nextDefaultValueFeature = 0
		self._setAllDefault('feature')
		if isinstance(featureNames, list) or featureNames is None:
			self.setFeatureNamesFromList(featureNames)
		elif isinstance(featureNames, dict):
			self.setFeatureNamesFromDict(featureNames)
		else:
			raise ArgumentException("featureNames may only be a list or a dict, defining a mapping between integers and featureNames")
		if featureNames is not None and len(featureNames) != self.featureCount:
			raise ArgumentException("Cannot have different number of featureNames and features, len(featureNames): " + str(len(featureNames)) + ", self.featureCount: " + str(self.featureCount))
		
		if name is None:
			self.name = DEFAULT_NAME_PREFIX + str(dataHelpers.defaultObjectNumber)
			dataHelpers.defaultObjectNumber += 1
		else:
			self.name = name
		self.path = path


	#######################
	# Property Attributes #
	#######################

	def _getpointCount(self):
		return self._pointCount
	pointCount = property(_getpointCount)

	def _getfeatureCount(self):
		return self._featureCount
	featureCount = property(_getfeatureCount)

	########################
	# Low Level Operations #
	########################

	def setPointName(self, oldIdentifier, newName):
		"""
		Changes the pointName specified by previous to the supplied input name.
		
		oldIdentifier must be a non None string or integer, specifying either a current pointName
		or the index of a current pointName. newName may be either a string not currently
		in the pointName set, or None for an default pointName. newName cannot begin with the
		default prefix.

		"""
		if len(self.pointNames) == 0:
			raise ArgumentException("Cannot set any point names; this object has no points ")
		self._setName_implementation(oldIdentifier, newName, 'point', False)
		return self

	def setFeatureName(self, oldIdentifier, newName):
		"""
		Changes the featureName specified by previous to the supplied input name.
		
		oldIdentifier must be a non None string or integer, specifying either a current featureName
		or the index of a current featureName. newName may be either a string not currently
		in the featureName set, or None for an default featureName. newName cannot begin with the
		default prefix.

		"""
		if len(self.featureNames) == 0:
			raise ArgumentException("Cannot set any feature names; this object has no features ")
		self._setName_implementation(oldIdentifier, newName, 'feature', False)
		return self

	def setPointNamesFromList(self, assignments=None):
		"""
		Rename all of the point names of this object according to the assignments in a list.
		We use the mapping between names and array indices to create a new dictionary, which
		will eventually be assigned as self.pointNames. assignments may be None to set all
		pointNames to new default values. If assignment is an unexpected type, the names
		are not strings, or the names are not unique, then an ArgumentException will be raised.

		"""
		return self._setNamesFromList(assignments, self.pointCount, 'point')


	def setFeatureNamesFromList(self, assignments=None):
		"""
		Rename all of the feature names of this object according to the assignments in a list.
		We use the mapping between names and array indices to create a new dictionary, which
		will eventually be assigned as self.featureNames. assignments may be None to set all
		featureNames to new default values. If assignment is an unexpected type, the names
		are not strings, or the names are not unique, then an ArgumentException will be raised.

		"""
		return self._setNamesFromList(assignments, self.featureCount, 'feature')

	def setPointNamesFromDict(self, assignments=None):
		"""
		Rename all of the point names of this object according to the mapping in a dict.
		We will use a copy of the input dictionary to be assigned as self.pointNames.
		assignments may be None to set all pointNames to new default values. If assignment
		is an unexpected type, if the names are not strings, the names are not unique,
		or the point indices are not integers then an ArgumentException will be raised.

		"""
		return self._setNamesFromDict(assignments, self.pointCount, 'point')

	def setFeatureNamesFromDict(self, assignments=None):
		"""
		Rename all of the feature names of this object according to the mapping in a dict.
		We will use a copy of the input dictionary to be assigned as self.featureNames.
		assignments may be None to set all featureNames to new default values. If assignment
		is an unexpected type, if the names are not strings, the names are not unique,
		or the feature indices are not integers then an ArgumentException will be raised.

		"""
		return self._setNamesFromDict(assignments, self.featureCount, 'feature')


	def nameData(self, name):
		"""
		Copy over the name attribute of this object with the input name

		"""
		if name is not None and not isinstance(name, basestring):
			raise ArgumentException("name must be a string")
		self.name = name
		return self

	###########################
	# Higher Order Operations #
	###########################
	
	def dropFeaturesContainingType(self, typeToDrop):
		"""
		Modify this object so that it no longer contains features which have the specified
		type as values

		"""
		if not isinstance(typeToDrop, list):
			if not isinstance(typeToDrop, type):
				raise ArgumentException("The only allowed inputs are a list of types or a single type, yet the input is neither a list or a type")
			typeToDrop = [typeToDrop]
		else:
			for value in typeToDrop:
				if not isinstance(value, type):
					raise ArgumentException("When giving a list as input, every contained value must be a type")

		if self.pointCount == 0 or self.featureCount == 0:
			return self

		def hasType(feature):
			for value in feature:
				for typeValue in typeToDrop:
					if isinstance(value, typeValue):
						return True
			return False
	
		removed = self.extractFeatures(hasType)
		return self


	def replaceFeatureWithBinaryFeatures(self, featureToReplace):
		"""
		Modify this object so that the choosen feature is removed, and binary valued
		features are added, one for each possible value seen in the original feature.

		"""		
		if self.pointCount == 0:
			raise ImproperActionException("This action is impossible, the object has 0 points")

		index = self._getFeatureIndex(featureToReplace)
		# extract col.
		toConvert = self.extractFeatures([index])

		# MR to get list of values
		def getValue(point):
			return [(point[0],1)]
		def simpleReducer(identifier, valuesList):
			return (identifier,0)

		values = toConvert.mapReducePoints(getValue, simpleReducer)
		values.setFeatureName(0,'values')
		values = values.extractFeatures([0])

		# Convert to List, so we can have easy access
		values = values.copyAs(format="List")

		# for each value run applyToEach to produce a category point for each value
		def makeFunc(value):
			def equalTo(point):
				if point[0] == value:
					return 1
				return 0
			return equalTo

		varName = toConvert.featureNamesInverse[0]

		for point in values.data:
			value = point[0]
			ret = toConvert.applyToPoints(makeFunc(value), inPlace=False)
			ret.setFeatureName(0, varName + "=" + str(value).strip())
			toConvert.appendFeatures(ret)

		# remove the original feature, and combine with self
		toConvert.extractFeatures([varName])
		self.appendFeatures(toConvert)
		return self


	def transformFeatureToIntegerFeature(self, featureToConvert):
		"""
		Modify this object so that the chosen feature in removed, and a new integer
		valued feature is added with values 0 to n-1, one for each of n values present
		in the original feature. 

		"""
		if self.pointCount == 0:
			raise ImproperActionException("This action is impossible, the object has 0 points")

		index = self._getFeatureIndex(featureToConvert)

		# extract col.
		toConvert = self.extractFeatures([index])

		# MR to get list of values
		def getValue(point):
			return [(point[0],1)]
		def simpleReducer(identifier, valuesList):
			return (identifier,0)

		values = toConvert.mapReducePoints(getValue, simpleReducer)
		values.setFeatureName(0,'values')
		values = values.extractFeatures([0])

		# Convert to List, so we can have easy access
		values = values.copyAs(format="List")

		mapping = {}
		index = 0
		for point in values.data:
			if point[0] not in mapping:
				mapping[point[0]] = index
				index = index + 1

		# use apply to each to make new feature with the int mappings
		def lookup(point):
			return mapping[point[0]]

		converted = toConvert.applyToPoints(lookup, inPlace=False)
		converted.setFeatureName(0,toConvert.featureNamesInverse[0])		

		self.appendFeatures(converted)
		return self

	def extractPointsByCoinToss(self, extractionProbability, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of points
		from this object, where a random experiment is performed for each
		point, with the chance of selection equal to the extractionProbabilty
		parameter. Those selected values are also removed from this object.

		"""
#		if self.pointCount == 0:
#			raise ImproperActionException("Cannot extract points from an object with 0 points")

		random.seed(seed)
		if extractionProbability is None:
			raise ArgumentException("Must provide a extractionProbability")
		if extractionProbability <= 0:
			raise ArgumentException("extractionProbability must be greater than zero")
		if extractionProbability >= 1:
			raise ArgumentException("extractionProbability must be less than one")

		def experiment(point):
			return bool(random.random() < extractionProbability)

		ret = self.extractPoints(experiment)

		return ret


	def foldIterator(self, numFolds, seed=DEFAULT_SEED):
		"""
		Returns an iterator object that iterates through folds of this object

		"""

		if self.pointCount == 0:
			raise ArgumentException("This object has 0 points, therefore, we cannot specify a valid number of folds")

		if self.featureCount == 0:
			raise ImproperActionException("We do not allow this operation on objects with 0 features")


		# note: we want truncation here
		numInFold = int(self.pointCount / numFolds)
		if numInFold == 0:
			raise ArgumentException("Must specify few enough folds so there is a point in each")

		# randomly select the folded portions
		indices = range(self.pointCount)
		random.seed(seed)
		random.shuffle(indices)
		foldList = []
		for fold in xrange(numFolds):
			start = fold * numInFold
			if fold == numFolds - 1:
				end = self.pointCount
			else:
				end = (fold + 1) * numInFold
			foldList.append(indices[start:end])

		# return that lists iterator as the fold iterator 	
		return self._foldIteratorClass(foldList, self)


	def applyToPoints(self, function, points=None, inPlace=True):
		"""
		Applies the given function to each point in this object, copying the
		output into this object. Or, if the inPlace flag is False, collecting
		the output values into a new object that is returned upon completion.

		function must not be none and accept a point as an argument

		points may be None to indicate application to all points, a single point
		ID or a list of point ID's to limit application only to those specified

		"""
		if self.pointCount == 0:
			raise ImproperActionException("We disallow this function when there are 0 points")
		if self.featureCount == 0:
			raise ImproperActionException("We disallow this function when there are 0 features")
		if function is None:
			raise ArgumentException("function must not be None")

		if points is not None and not isinstance(points, list):
			if not isinstance(points, int):
				raise ArgumentException("Only allowable inputs to 'points' parameter is an int ID, a list of int ID's, or None")
			points = [points]

		self.validate()

		return self._applyTo_implementation(function, points, inPlace, 'point')


	def applyToFeatures(self, function, features=None, inPlace=True):
		"""
		Applies the given function to each feature in this object, copying the
		output into this object. Or, if the inPlace flag is False, collecting
		the output values into a new object that is returned upon completion.

		function must not be none and accept a feature as an argument

		features may be None to indicate application to all features, a single feature
		ID or a list of features ID's to limit application only to those specified

		"""
		if self.pointCount == 0:
			raise ImproperActionException("We disallow this function when there are 0 points")
		if self.featureCount == 0:
			raise ImproperActionException("We disallow this function when there are 0 features")
		if function is None:
			raise ArgumentException("function must not be None")

		if features is not None and not isinstance(features, list):
			if not (isinstance(features, int) or isinstance(features, basestring)):
				raise ArgumentException("Only allowable inputs to 'features' parameter is an ID, a list of int ID's, or None")
			features = [features]

		if features is not None:
			for i in xrange(len(features)):
				features[i] = self._getFeatureIndex(features[i])

		self.validate()

		return self._applyTo_implementation(function, features, inPlace, 'feature')


	def _applyTo_implementation(self, function, included, inPlace, axis):
		if axis == 'point':
			viewIterator = self.pointIterator()
		else:
			viewIterator = self.featureIterator()

		retData = []
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
						view[i] = currOut[i]
				else:
					toCopyInto = []
					for value in currOut:
						toCopyInto.append(value)
					retData.append(toCopyInto)
			# singular return
			else:
				if inPlace:
					view[0] = currOut
				else:
					retData.append([currOut])
		
		if inPlace:
			ret = self
		else:
			ret = UML.createData(self.getTypeString(), retData)
			if axis != 'point':
				ret.transpose()

		return ret


	def mapReducePoints(self, mapper, reducer):
		if self.pointCount == 0:
			return UML.createData(self.getTypeString(), numpy.empty(shape=(0,0)))
		if self.featureCount == 0:
			raise ImproperActionException("We do not allow operations over points if there are 0 features")

		if mapper is None or reducer is None:
			raise ArgumentException("The arguments must not be none")
		if not hasattr(mapper, '__call__'):
			raise ArgumentException("The mapper must be callable")
		if not hasattr(reducer, '__call__'):
			raise ArgumentException("The reducer must be callable")

		self.validate()

		mapResults = {}
		# apply the mapper to each point in the data
		for point in self.pointIterator():
			currResults = mapper(point)
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
		return UML.createData(self.getTypeString(), ret)

	def pointIterator(self):
		if self.featureCount == 0:
			raise ImproperActionException("We do not allow iteration over points if there are 0 features")

		class pointIt():
			def __init__(self, outer):
				self._outer = outer
				self._position = 0
			def __iter__(self):
				return self
			def next(self):
				while (self._position < self._outer.pointCount):
					value = self._outer.pointView(self._position)
					self._position += 1
					return value
				raise StopIteration
		return pointIt(self)

	def featureIterator(self):
		if self.pointCount == 0:
			raise ImproperActionException("We do not allow iteration over features if there are 0 points")

		class featureIt():
			def __init__(self, outer):
				self._outer = outer
				self._position = 0
			def __iter__(self):
				return self
			def next(self):
				while (self._position < self._outer.featureCount):
					value = self._outer.featureView(self._position)
					self._position += 1
					return value
				raise StopIteration
		return featureIt(self)

	def applyToElements(self, function, points=None, features=None, inPlace=True, preserveZeros=False, skipNoneReturnValues=False):
		"""
		Applies the function(elementValue) or function(elementValue, pointNum,
		featureNum) to each element, and returns an object (of the same type as the
		calling object) containing the resulting values. If preserveZeros=True it does not
		apply function to elements in the dataMatrix that are 0 and a 0 is placed in
		it's place in the output. If skipNoneReturnValues=True, any time function()
		returns None, the value that was input to the function will be put in the output
		in place of None. The result of both of the flags is to modify the output, yet
		ensure it still has the same dimensions as the calling object.

		"""
		oneArg = False
		try:
			function(0,0,0)
		except TypeError:
			oneArg = True

		if points is not None and not isinstance(points, list):
			if not isinstance(points, int):
				raise ArgumentException("Only allowable inputs to 'points' parameter is an int ID, a list of int ID's, or None")
			points = [points]

		if features is not None and not isinstance(features, list):
			if not isinstance(features, int):
				raise ArgumentException("Only allowable inputs to 'features' parameter is an ID, a list of int ID's, or None")
			features = [features]

		if features is not None:
			for i in xrange(len(features)):
				features[i] = self._getFeatureIndex(features[i])

		self.validate()

		valueList = []
		for currPoint in self.pointIterator():
			currPointID = currPoint.index()
			if points is not None and currPointID not in points:
				continue
			tempList = []
			for j in xrange(len(currPoint)):
				if features is not None and j not in features:
					continue
				value = currPoint[j]
				if preserveZeros and value == 0:
					if not inPlace:
						tempList.append(0)
					continue
				if oneArg:
					currRet = function(value)
				else:
					currRet = function(value, currPointID, j)
				if currRet is None and skipNoneReturnValues:
					if not inPlace:
						tempList.append(value)
				else:
					if not inPlace:
						tempList.append(currRet)
					else:
						currPoint[j] = currRet
			if not inPlace:
				valueList.append(tempList)

		if not inPlace:
			return UML.createData(self.getTypeString(), valueList)
		else:
			return self

	def hashCode(self):
		"""returns a hash for this matrix, which is a number x in the range 0<= x < 1 billion
		that should almost always change when the values of the matrix are changed by a substantive amount"""
		if self.pointCount == 0 or self.featureCount == 0:
			return 0
		valueObj = self.applyToElements(lambda elementValue, pointNum, featureNum: ((math.sin(pointNum) + math.cos(featureNum))/2.0) * elementValue, inPlace=False, preserveZeros=True)
		valueList = valueObj.copyAs(format="python list")
		avg = sum(itertools.chain.from_iterable(valueList))/float(self.pointCount*self.featureCount)
		bigNum = 1000000000
		#this should return an integer x in the range 0<= x < 1 billion
		return int(int(round(bigNum*avg)) % bigNum)	


	def isApproximatelyEqual(self, other):
		"""If it returns False, this DataMatrix and otherDataMatrix definitely don't store equivalent data. 
		If it returns True, they probably do but you can't be absolutely sure.
		Note that only the actual data stored is considered, it doesn't matter whether the data matrix objects 
		passed are of the same type (Matrix, Sparse, etc.)"""
		self.validate()
		#first check to make sure they have the same number of rows and columns
		if self.pointCount != other.pointCount: return False
		if self.featureCount != other.featureCount: return False
		#now check if the hashes of each matrix are the same
		if self.hashCode() != other.hashCode(): return False
		return True


	def shufflePoints(self, indices=None, seed=DEFAULT_SEED):
		"""
		Permute the indexing of the points so they are in a random order. Note: this relies on
		python's random.shuffle() so may not be sufficiently random for large number of points.
		See shuffle()'s documentation.

		"""
		if indices is None:
			indices = range(0, self.pointCount)
			random.shuffle(indices)
		else:
			if len(indices) != self.pointCount:
				raise ArgumentException("If indices are supplied, it must be a list with all and only valid point indices")
			for value in indices:
				if value < 0 or value > self.pointCount:
					raise ArgumentException("A value in indices is out of bounds of the valid range of points")

		def permuter(pointView):
			return indices[pointView.index()]
		self.sortPoints(sortHelper=permuter)
		return self


	def shuffleFeatures(self, indices=None, seed=DEFAULT_SEED):
		"""
		Permute the indexing of the features so they are in a random order. Note: this relies on
		python's random.shuffle() so may not be sufficiently random for large number of features.
		See shuffle()'s documentation.

		"""
		if indices is None:
			indices = range(0, self.featureCount)
			random.shuffle(indices)
		else:
			if len(indices) != self.featureCount:
				raise ArgumentException("If indices are supplied, it must be a list with all and only valid features indices")
			for value in indices:
				if value < 0 or value > self.featureCount:
					raise ArgumentException("A value in indices is out of bounds of the valid range of features")
		def permuter(featureView):
			return indices[featureView.index()]
		self.sortFeatures(sortHelper=permuter)
		return self

	def copy(self):
		"""
		Return a new object which has the same data (and featureNames, depending on
		the return type) and in the same UML format as this object.

		"""
		return self.copyAs(self.getTypeString())


	#################################
	# Functions related to logging  #
	#################################
	def featureReport(self, displayDigits=2):
		"""
		Produce a report, in a string formatted as a table, containing summary and statistical
		information about each feature in the data set, up to 50 features.  If there are more
		than 50 features, only information about 50 of those features will be reported.
		"""
		return produceFeaturewiseReport(self, displayDigits=displayDigits)

	def summaryReport(self, displayDigits=2):
		"""
		Produce a report, in a string formatted as a table, containing summary 
		information about the data set contained in this object.  Includes 
		proportion of missing values, proportion of zero values, total # of points,
		and number of features.
		"""
		return produceAggregateReport(self, displayDigits=displayDigits)

	
	#################################
	# Functions for derived classes #
	#################################

	def transpose(self):
		"""
		Function to transpose the data, ie invert the feature and point indices of the data.
	
		Features are then given default featureNames.

		"""
		self._transpose_implementation()

		temp = self._pointCount
		self._pointCount = self._featureCount
		self._featureCount = temp

		self.setFeatureNamesFromDict(None)		
		return self

	def appendPoints(self, toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object.

		toAppend cannot be None, and must be a kind of data representation object with the same
		number of features as the calling object.
		
		"""
		if toAppend is None:
			raise ArgumentException("toAppend must not be None")
		if not isinstance(toAppend,Base):
			raise ArgumentException("toAppend must be a kind of data representation object")
		if not self.featureCount == toAppend.featureCount:
			raise ArgumentException("toAppend must have the same number of features as this object")
		if not self._equalFeatureNames(toAppend):
			raise ArgumentException("The featureNames of the two objects must match")
		
		self.validate()

		self._appendPoints_implementation(toAppend)
		self._pointCount += toAppend.pointCount
		return self
		
	def appendFeatures(self, toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		toAppend cannot be None, must be a kind of data representation object with the same
		number of points as the calling object, and must not share any feature names with the calling
		object.
		
		"""	
		if toAppend is None:
			raise ArgumentException("toAppend must not be None")
		if not isinstance(toAppend,Base):
			raise ArgumentException("toAppend must be a kind of data representation object")
		if not self.pointCount == toAppend.pointCount:
			raise ArgumentException("toAppend must have the same number of points as this object")
		intersection = self._featureNameIntersection(toAppend)
		if intersection:
			for name in intersection:
				if not name.startswith(DEFAULT_PREFIX):
					raise ArgumentException("toAppend must not share any featureNames with this object")
		
		self.validate()

		self._appendFeatures_implementation(toAppend)
		self._featureCount += toAppend.featureCount

		for i in xrange(toAppend.featureCount):
			currName = toAppend.featureNamesInverse[i]
			if currName.startswith(DEFAULT_PREFIX):
				currName += '_' + toAppend.name
			self._addFeatureName(currName)
		
		return self

	def sortPoints(self, sortBy=None, sortHelper=None):
		""" 
		Modify this object so that the points are sorted in place, where sortBy may
		indicate the feature to sort by or None if the entire point is to be taken as a key,
		sortHelper may either be comparator, a scoring function, or None to indicate the natural
		ordering.
		"""
		# its already sorted in these cases
		if self.featureCount == 0 or self.pointCount == 0:
			return

		sortByIndex = sortBy
		if sortBy is not None:
			sortByIndex = self._getFeatureIndex(sortBy)
			if sortHelper is not None:
				raise ArgumentException("Cannot specify a feature to sort by and a helper function")
		else:
			if sortHelper is None:
				raise ArgumentException("Either sortBy or sortHelper must not be None")

		self._sortPoints_implementation(sortByIndex, sortHelper)
		return self

	def sortFeatures(self, sortBy=None, sortHelper=None):
		""" 
		Modify this object so that the features are sorted in place, where sortBy may
		indicate the feature to sort by or None if the entire point is to be taken as a key,
		sortHelper may either be comparator, a scoring function, or None to indicate the natural
		ordering.

		"""
		# its already sorted in these cases
		if self.featureCount == 0 or self.pointCount == 0:
			return
		if sortBy is not None and sortHelper is not None:
			raise ArgumentException("Cannot specify a feature to sort by and a helper function")
		if sortBy is None and sortHelper is None:
			raise ArgumentException("Either sortBy or sortHelper must not be None")

		if sortBy is not None and isinstance(sortBy, basestring):
			sortBy = self._getFeatureIndex(sortBy)

		newFeatureNameOrder = self._sortFeatures_implementation(sortBy, sortHelper)
		self.setFeatureNamesFromList(newFeatureNameOrder)
		return self


	def extractPoints(self, toExtract=None, start=None, end=None, number=None, randomize=False):
		"""
		Modify this object, removing those points that are specified by the input, and returning
		an object containing those removed points.

		toExtract may be a single identifier, a list of identifiers, or a function that when
		given a point will return True if it is to be removed. number is the quantity of points that
		we are to be extracted, the default None means unlimited extraction. start and end are
		parameters indicating range based extraction: if range based extraction is employed,
		toExtract must be None, and vice versa. If only one of start and end are non-None, the
		other defaults to 0 and self.pointCount respectably. randomize indicates whether random
		sampling is to be used in conjunction with the number parameter, if randomize is False,
		the chosen points are determined by point order, otherwise it is uniform random across the
		space of possible removals.

		"""
#		if self.pointCount == 0:
#			raise ImproperActionException("Cannot extract points from an object with 0 points")

		self.validate()

		if toExtract is not None:
			if start is not None or end is not None:
				raise ArgumentException("Range removal is exclusive, to use it, toExtract must be None")
		else:
			if start is None:
				start = 0
			if end is None:
				end = self.pointCount - 1
			if start < 0 or start > self.pointCount:
				raise ArgumentException("start must be a valid index, in the range of possible points")
			if end < 0 or end > self.pointCount:
				raise ArgumentException("end must be a valid index, in the range of possible points")
			if start > end:
				raise ArgumentException("start cannot be an index greater than end")
			if not randomize and number is not None:
				#then we can do the windowing calculation here
				possibleEnd = start + number -1
				if possibleEnd < end:
					end = possibleEnd

		ret = self._extractPoints_implementation(toExtract, start, end, number, randomize)
		self._pointCount -= ret.pointCount
		if ret.pointCount != 0:
			ret.setFeatureNamesFromDict(self.featureNames)
		return ret

	def extractFeatures(self, toExtract=None, start=None, end=None, number=None, randomize=False):
		"""
		Modify this object, removing those features that are specified by the input, and returning
		an object containing those removed features. This particular function only does argument
		checking and modifying the featureNames for this object. It is the job of helper functions in
		the derived class to perform the removal and assign featureNames for the returned object.

		toExtract may be a single identifier, a list of identifiers, or a function that when
		given a feature will return True if it is to be removed. number is the quantity of features that
		are to be extracted, the default None means unlimited extraction. start and end are
		parameters indicating range based extraction: if range based extraction is employed,
		toExtract must be None, and vice versa. If only one of start and end are non-None, the
		other defaults to 0 and self.featureCount respectably. randomize indicates whether random
		sampling is to be used in conjunction with the number parameter, if randomize is False,
		the chosen features are determined by feature order, otherwise it is uniform random across the
		space of possible removals.

		"""
#		if self.featureCount == 0:
#			raise ImproperActionException("Cannot extract features from an object with 0 features")

		self.validate()

		if toExtract is not None:
			if start is not None or end is not None:
				raise ArgumentException("Range removal is exclusive, to use it, toExtract must be None")
		elif start is not None or end is not None:
			if start is None:
				start = 0
			if end is None:
				end = self.featureCount - 1
			if start < 0 or start > self.featureCount:
				raise ArgumentException("start must be a valid index, in the range of possible features")
			if end < 0 or end > self.featureCount:
				raise ArgumentException("end must be a valid index, in the range of possible features")
			if start > end:
				raise ArgumentException("start cannot be an index greater than end")
			if not randomize and number is not None:
				#then we can do the windowing calculation here
				possibleEnd = start + number -1
				if possibleEnd < end:
					end = possibleEnd

		ret = self._extractFeatures_implementation(toExtract, start, end, number, randomize)
		self._featureCount -= ret.featureCount
		for key in ret.featureNames.keys():
			self._removeFeatureNameAndShift(key)
		return ret

	def isIdentical(self, other):
		if not self._equalFeatureNames(other):
			return False

		return self._isIdentical_implementation(other)

	def writeFile(self, outPath, format=None, includeFeatureNames=True):
		"""
		Function to write the data in this object to a file using the specified
		format. outPath is the location (including file name and extension) where
		we want to write the output file. includeFeatureNames is boolean argument
		indicating whether the file should start with a comment line designating featureNames.

		"""
		if self.pointCount == 0 or self.featureCount == 0:
			raise ImproperActionException("We do not allow writing to file when an object has 0 points or features")

		self.validate()

		# if format is not specified, we fall back on the extension in outPath
		if format is None:
			split = outPath.rsplit('.', 1)
			format = None
			if len(split) > 1:
				format = split[1].lower()

		if format.lower() == "csv":
			return self._writeFileCSV_implementation(outPath, includeFeatureNames)
		elif format.lower() == "mtx":
			return self._writeFileMTX_implementation(outPath, includeFeatureNames)
		else:
			msg = "Unrecognized file format. Accepted types are 'csv' and 'mtx'. They may "
			msg += "either be input as the format parameter, or as the extension in the "
			msg += "outPath"
			raise ArgumentException()

	def referenceDataFrom(self, other):
		"""
		Modifies the internal data of this object to refer to the same data as other. In other
		words, the data wrapped by both the self and other objects resides in the
		same place in memory. Other must be an object of the same type as
		the calling object. Also, the shape of other should be consistent with the set
		of featureNames currently in this object

		"""
		# this is called first because it checks the data type
		self._referenceDataFrom_implementation(other)
		self.featureNames = other.featureNames
		self.featureNamesInverse = other.featureNamesInverse
		return self


	def copyAs(self, format, rowsArePoints=True, outputAs1D=False):
		"""
		Return a new object which has the same data (and featureNames, depending on
		the return type) as this object. To return a specific kind of UML data
		object, one may specify the format parameter to be 'List', 'Matrix', or
		'Sparse'. To specify a raw return type (which will not include feature names),
		one may specify 'python list', 'numpy array', or 'numpy matrix', 'scipy csr'
		or 'scypy csc'.

		"""
		#make lower case, strip out all white space and periods, except if format
		# is one of the accepted UML data types
		if format not in ['List', 'Matrix', 'Sparse']:
			format = format.lower()
			format = format.strip()
			tokens = format.split(' ')
			format = ''.join(tokens)
			tokens = format.split('.')
			format = ''.join(tokens)
			if format not in ['pythonlist', 'numpyarray', 'numpymatrix', 'scipycsr', 'scipycsc']:
				msg = "The only accepted asTypes are: 'List', 'Matrix', 'Sparse'"
				msg +=", 'python list', 'numpy array', 'numpy matrix', 'scipy csr', and 'scipy csc'"
				raise ArgumentException(msg)
		else:
			if outputAs1D:
				raise ArgumentException('Cannot output a UML format as 1D')

		# we enforce very specific shapes in the case of emptiness along one
		# or both axes
		if format == 'pythonlist':
			if self.pointCount == 0:
				return []
			if self.featureCount == 0:
				ret = []
				for i in xrange(self.pointCount):
					ret.append([])
				return ret
		if format.startswith('scipy'):
			if self.pointCount == 0 or self.featureCount == 0:
				raise ArgumentException('Cannot output a point empty object in a scipy format')

		return self._copyAs_implementation(format, rowsArePoints, outputAs1D)

	def copyPoints(self, points=None, start=None, end=None):
		"""
		Return a new object which consists only of those specified points, without mutating
		the calling object object.
		
		"""
		if isinstance(points, int):
			points = [points]
		if self.pointCount == 0:
			raise ArgumentException("Object contains 0 points, there is no valid possible input")
		if points is None:
			if start is not None or end is not None:
				if start is None:
					start = 0
				if end is None:
					end = self.pointCount - 1
				if start < 0 or start > self.pointCount:
					raise ArgumentException("start must be a valid index, in the range of possible features")
				if end < 0 or end > self.pointCount:
					raise ArgumentException("end must be a valid index, in the range of possible features")
				if start > end:
					raise ArgumentException("start cannot be an index greater than end")
			else:
				raise ArgumentException("must specify something to copy")
		else:
			if start is not None or end is not None:
				raise ArgumentException("Cannot specify both IDs and a range")
			#verify everything in list is a valid index TODO
			for index in points:
				if index < 0 or index >= self.pointCount:
					raise ArgumentException("input must contain only valid indices")

		retObj = self._copyPoints_implementation(points, start, end)
		retObj.setFeatureNamesFromDict(self.featureNames)
		return retObj
	
	def copyFeatures(self, features=None, start=None, end=None):
		"""
		Return a new object which consists only of those specified features, without mutating
		this object.
		
		"""
		if isinstance(features, basestring) or isinstance(features, int):
			features = [features]
		if self.featureCount == 0:
			raise ArgumentException("Object contains 0 features, there is no valid possible input")
		indices = None
		if features is None:
			if start is not None or end is not None:
				if start is None:
					start = 0
				if end is None:
					end = self.featureCount - 1
				if start < 0 or start > self.featureCount:
					raise ArgumentException("start must be a valid index, in the range of possible features")
				if end < 0 or end > self.featureCount:
					raise ArgumentException("end must be a valid index, in the range of possible features")
				if start > end:
					raise ArgumentException("start cannot be an index greater than end")
			else:
				raise ArgumentException("must specify something to copy")
		else:
			if start is not None or end is not None:
				raise ArgumentException("Cannot specify both IDs and a range")
			indices = []
			for identifier in features:
				indices.append(self._getFeatureIndex(identifier))

		return self._copyFeatures_implementation(indices, start, end)

	def getTypeString(self):
		"""
			Return a string representing the non-abstract type of this object (e.g. Matrix,
			Sparse, etc.) that can be passed to createData() function to create a new object
			of the same type.
		"""
		return self._getTypeString_implementation()

	def __getitem__(self, key):
		try:
			(x,y) = key
		except TypeError:
			raise ArgumentException("Must include a point and feature index")
		if not isinstance(x,int) or x < 0 or x >= self.pointCount:
			raise ArgumentException(str(x) + " is not a valid point ID")

		if isinstance(y,basestring):
			y = self._getFeatureIndex(y)

		if not isinstance(y,int) or y < 0 or y >= self.featureCount:
			raise ArgumentException(str(y) + " is not a valid feature ID")

		return self._getitem_implementation(x,y)

	def pointView(self, ID):
		"""
		Returns a View object into the data of the point with the given ID. See View object
		comments for its capabilities. This View is only valid until the next modification
		to the shape or ordering of the internal data. After such a modification, there is
		no guarantee to the validity of the results.
		"""
		if self.pointCount == 0:
			raise ImproperActionException("ID is invalid, This object contains no points")
		if not isinstance(ID,int):
			raise ArgumentException("Point IDs must be integers")
		return self._pointView_implementation(ID)

	def featureView(self, ID):
		"""
		Returns a View object into the data of the point with the given ID. See View object
		comments for its capabilities. This View is only valid until the next modification
		to the shape or ordering of the internal data. After such a modification, there is
		no guarantee to the validity of the results.
		"""
		if self.featureCount == 0:
			raise ArgumentException("ID is invalid, This object contains no features")

		index = self._getFeatureIndex(ID)
		return self._featureView_implementation(index)
	

	def validate(self, level=1):
		"""
		Checks the integrety of the data with respect to the limitations and invariants
		that our objects enforce.

		"""
		assert self.featureCount == len(self.featureNames)
		assert len(self.featureNames) == len(self.featureNamesInverse)
		if level > 0:
			for key in self.featureNames.keys():
				assert self.featureNamesInverse[self.featureNames[key]] == key

		self._validate_implementation(level)

	####################
	# Helper functions #
	####################

	def _pointNameDifference(self, other):
		"""
		Returns a set containing those pointNames in this object that are not also in the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, Base):
			raise ArgumentException("Must provide another representation type to determine pointName difference")
		
		return self.pointNames.viewkeys() - other.pointNames.viewkeys() 

	def _featureNameDifference(self, other):
		"""
		Returns a set containing those featureNames in this object that are not also in the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, Base):
			raise ArgumentException("Must provide another representation type to determine featureName difference")
		
		return self.featureNames.viewkeys() - other.featureNames.viewkeys() 

	def _pointNameIntersection(self, other):
		"""
		Returns a set containing only those pointNames that are shared by this object and the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, Base):
			raise ArgumentException("Must provide another representation type to determine pointName intersection")
		
		return self.pointNames.viewkeys() & other.pointNames.viewkeys() 

	def _featureNameIntersection(self, other):
		"""
		Returns a set containing only those featureNames that are shared by this object and the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, Base):
			raise ArgumentException("Must provide another representation type to determine featureName intersection")
		
		return self.featureNames.viewkeys() & other.featureNames.viewkeys() 


	def _pointNameSymmetricDifference(self, other):
		"""
		Returns a set containing only those pointNames not shared between this object and the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, Base):
			raise ArgumentException("Must provide another representation type to determine pointName difference")
		
		return self.pointNames.viewkeys() ^ other.pointNames.viewkeys() 

	def _featureNameSymmetricDifference(self, other):
		"""
		Returns a set containing only those featureNames not shared between this object and the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, Base):
			raise ArgumentException("Must provide another representation type to determine featureName difference")
		
		return self.featureNames.viewkeys() ^ other.featureNames.viewkeys() 

	def _pointNameUnion(self, other):
		"""
		Returns a set containing all pointNames in either this object or the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, Base):
			raise ArgumentException("Must provide another representation type to determine pointNames union")
		
		return self.pointNames.viewkeys() | other.pointNames.viewkeys() 

	def _featureNameUnion(self, other):
		"""
		Returns a set containing all featureNames in either this object or the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, Base):
			raise ArgumentException("Must provide another representation type to determine featureName union")
		
		return self.featureNames.viewkeys() | other.featureNames.viewkeys() 


	def _equalPointNames(self, other):
		if other is None or not isinstance(other, Base):
			return False
		return self._equalNames(self.pointNames, self.pointNamesInverse, other.pointNames, other.pointNamesInverse)

	def _equalFeatureNames(self, other):
		if other is None or not isinstance(other, Base):
			return False
		return self._equalNames(self.featureNames, self.featureNamesInverse, other.featureNames, other.featureNamesInverse)

	def _equalNames(self, selfNames, selfNamesInv, otherNames, otherInvNames):
		"""
		Private function to determine equality of either pointNames or featureNames.
		It ignores equality of default values, though if default values are present,
		the number of names and their indices must match up.

		"""
		if len(selfNames) != len(otherNames):
			return False
		if len(selfNamesInv) != len(otherInvNames):
			return False
		# check both featureName directions
		for featureName in selfNames.keys():
			if not featureName.startswith(DEFAULT_PREFIX) and featureName not in otherNames:
				return False
			if not featureName.startswith(DEFAULT_PREFIX) and selfNames[featureName] != otherNames[featureName]:
				return False
		for index in selfNamesInv.keys():
			if index not in otherInvNames:
				return False
			if not selfNamesInv[index].startswith(DEFAULT_PREFIX):
				if selfNamesInv[index] != selfNamesInv[index]:
					return False
		for featureName in otherNames.keys():
			if not featureName.startswith(DEFAULT_PREFIX) and featureName not in selfNames:
				return False
			if not featureName.startswith(DEFAULT_PREFIX) and otherNames[featureName] != selfNames[featureName]:
				return False
		for index in otherInvNames.keys():
			if index not in selfNamesInv:
				return False
			if not otherInvNames[index].startswith(DEFAULT_PREFIX):
				if otherInvNames[index] != otherInvNames[index]:
					return False
		return True


	def _getPointIndex(self, identifier):
		return self._getIndex(identifier, self.pointNames, self.pointNamesInverse)

	def _getFeatureIndex(self, identifier):
		return self._getIndex(identifier, self.featureNames, self.featureNamesInverse)

	def _getIndex(self, identifier, names, namesInv):
		toReturn = identifier
		if len(names) == 0:
			raise ArgumentException("There are no valid feature identifiers; this object has 0 features")
		if identifier is None:
			raise ArgumentException("An identifier cannot be None")
		if (not isinstance(identifier,basestring)) and (not isinstance(identifier,int)):
			raise ArgumentException("The indentifier must be either a string or integer index")
		if isinstance(identifier,int):
			if identifier < 0 or identifier >= len(namesInv):
				raise ArgumentException("The index " + str(identifier) +" is outside of the range of possible values")
		if isinstance(identifier,basestring):
			if identifier not in names:
				raise ArgumentException("The name '" + identifier + "' cannot be found")
			# set as index for return
			toReturn = names[identifier]
		return toReturn

	def _nextDefaultName(self, axis):
		self._validateAxis(axis)
		if axis == 'point':
			ret = DEFAULT_PREFIX + str(self._nextDefaultValuePoint)
			self._nextDefaultValuePoint += 1
		else:
			ret = DEFAULT_PREFIX + str(self._nextDefaultValueFeature)
			self._nextDefaultValueFeature += 1
		return ret

	def _setAllDefault(self, axis):
		self._validateAxis(axis)
		if axis == 'point':
			self.pointNames = {}
			self.pointNamesInverse = {}
			names = self.pointNames
			invNames = self.pointNamesInverse
			count = self._pointCount
		else:
			self.featureNames = {}
			self.featureNamesInverse = {}
			names = self.featureNames
			invNames = self.featureNamesInverse
			count = self._featureCount
		for i in xrange(count):
			defaultName = self._nextDefaultName(axis)
			invNames[i] = defaultName
			names[defaultName] = i

	def _addPointName(self, pointName):
		return self._addName(pointName, self.pointNames, self.pointNamesInverse, 'point')

	def _addFeatureName(self, featureName):
		return self._addName(featureName, self.featureNames, self.featureNamesInverse, 'feature')

	def _addName(self, name, selfNames, selfNamesInv, axis):
		"""
		Name the next vector outside of the current possible range on the given axis using the
		provided name.

		name may be either a string, or None if you want a default name. If the name is
		not a string, or already being used as another name on this axis, an
		ArgumentException will be raised.

		"""
		if name is not None and not isinstance(name, basestring):
			raise ArgumentException("The name must be a string")
		if name in selfNames:
			raise ArgumentException("This name is already in use")
		
		if name is None:
			name = self._nextDefaultName(axis)

		self._incrementDefaultIfNeeded(name, axis)	

		numInAxis = len(selfNamesInv)
		selfNamesInv[numInAxis] = name
		selfNames[name] = numInAxis

	def _removePointNameAndShift(self, toRemove):
		"""
		Removes the specified name from pointNames, changing the indices
		of other pointNames to fill in the missing index.

		toRemove must be a non None string or integer, specifying either a current pointName
		or the index of a current pointName in the given axis.
		
		"""
		return self._removeNameAndShift(toRemove, 'point', self.pointNames, self.pointNamesInverse)

	def _removeFeatureNameAndShift(self, toRemove):
		"""
		Removes the specified name from featureNames, changing the indices
		of other featureNames to fill in the missing index.

		toRemove must be a non None string or integer, specifying either a current featureNames
		or the index of a current featureNames in the given axis.
		
		"""
		return self._removeNameAndShift(toRemove, 'feature', self.featureNames, self.featureNamesInverse)

	def _removeNameAndShift(self, toRemove, axis, selfNames, selfNamesInv):
		"""
		Removes the specified name from the name set for the given axis, changing the indices
		of other names to fill in the missing index.

		toRemove must be a non None string or integer, specifying either a current name
		or the index of a current name in the given axis.

		axis must be either 'point' or 'feature'
		
		selfNames must be the names dict associated with the provided axis in this object

		selfNamesInv must be the indices to names dict associated with the provided axis
		in this object

		"""
		#this will throw the appropriate exceptions, if need be
		index = self._getIndex(toRemove, selfNames,selfNamesInv)
		name = selfNamesInv[index]

		del selfNames[name]

		numInAxis = len(selfNamesInv)
		# remapping each index starting with the one we removed
		for i in xrange(index, numInAxis-1):
			nextName = selfNamesInv[i+1]
			if name is not None:
				selfNames[nextName] = i
			selfNamesInv[i] = nextName
		#delete the last mapping, that name was shifted in the for loop
		del selfNamesInv[numInAxis-1]

	def _setName_implementation(self, oldIdentifier, newName, axis, allowDefaults=False):
		"""
		Changes the featureName specified by previous to the supplied input featureName.
		
		oldIdentifier must be a non None string or integer, specifying either a current featureName
		or the index of a current featureName. newFeatureName may be either a string not currently
		in the featureName set, or None for an default featureName. newFeatureName may begin with the
		default prefix

		"""
		self._validateAxis(axis)
		if axis == 'point':
			names = self.pointNames
			invNames = self.pointNamesInverse
			index = self._getPointIndex(oldIdentifier)
		else:
			names = self.featureNames
			invNames = self.featureNamesInverse
			index = self._getFeatureIndex(oldIdentifier)

		if newName is not None: 
			if not isinstance(newName, basestring):
				raise ArgumentException("The new name must be either None or a string")
#			if not allowDefaults and newFeatureName.startswith(DEFAULT_PREFIX):
#				raise ArgumentException("Cannot manually add a featureName with the default prefix")
		if newName in names:
			if invNames[index] == newName:
				return
			raise ArgumentException("This name is already in use")
		
		if newName is None:
			newName = self._nextDefaultName(axis)

		#remove the current featureName
		oldName = invNames[index]
		del names[oldName]		

		# setup the new featureName
		invNames[index] = newName
		names[newName] = index

		self._incrementDefaultIfNeeded(newName, axis)

	def _setNamesFromList(self, assignments, count, axis):
		self._validateAxis(axis)
		if assignments is None:
			self._setAllDefault(axis)
			return self
		if not isinstance(assignments, list):
			raise ArgumentException("assignments may only be a list, with as many entries as this axis is long")
		if count == 0:
			if len(assignments) > 0:
				raise ArgumentException("assignments is too large; this axis is empty ")
			return self
		if len(assignments) != count:
			raise ArgumentException("assignments may only be a list, with as many entries as this axis is long")

		#convert to dict so we only write the checking code once
		temp = {}
		for index in xrange(len(assignments)):
			name = assignments[index]
			if name in temp:
				raise ArgumentException("Cannot input duplicate names: " + str(name))
			temp[name] = index
		assignments = temp

		self._setNamesFromDict(assignments, count, axis)
		return self

	def _setNamesFromDict(self, assignments, count, axis):
		self._validateAxis(axis)
		if assignments is None:
			self._setAllDefault(axis)
			return self
		if not isinstance(assignments, dict):
			raise ArgumentException("assignments may only be a dict, with as many entries as this axis is long")
		if count == 0:
			if len(assignments) > 0:
				raise ArgumentException("assignments is too large; this axis is empty ")
			return self
		if len(assignments) != count:
			raise ArgumentException("assignments may only be a dict, with as many entries as this axis is long")

		# at this point, the input must be a dict
		#check input before performing any action
		for name in assignments.keys():
			if not isinstance(name, basestring):
				raise ArgumentException("Names must be strings")
			if not isinstance(assignments[name], int):
				raise ArgumentException("Indices must be integers")
			if assignments[name] < 0 or assignments[name] >= count:
				countName = 'pointCount' if axis == 'point' else 'featureCount'
				raise ArgumentException("Indices must be within 0 to self." + countName + " - 1")

		reverseDict = {}
		for name in assignments.keys():
			self._incrementDefaultIfNeeded(name, axis)
			reverseDict[assignments[name]] = name

		# have to copy the input, could be from another object
		if axis == 'point':
			self.pointNames = copy.copy(assignments)
			self.pointNamesInverse = reverseDict
		else:
			self.featureNames = copy.copy(assignments)
			self.featureNamesInverse = reverseDict
		return self


	def _validateAxis(self, axis):
		if axis != 'point' and axis != 'feature':
			raise ArgumentException('axis parameter may only be "point" or "feature"')

	def _incrementDefaultIfNeeded(self, name, axis):
		self._validateAxis(axis)
		if name.startswith(DEFAULT_PREFIX):
			intString = name[len(DEFAULT_PREFIX):]
			if '_' in intString:
				firstUnderscore = intString.index('_')
				intString = intString[:firstUnderscore]
			if '=' in intString:
				firstEquals = intString.index('=')
				intString = intString[:firstEquals]
			nameNum = int(intString)
			if axis == 'point':
				if nameNum >= self._nextDefaultValuePoint:
					self._nextDefaultValuePoint = nameNum + 1
			else:
				if nameNum >= self._nextDefaultValueFeature:
					self._nextDefaultValueFeature = nameNum + 1

	class _foldIteratorClass():
		def __init__(self, foldList, outerReference):
			self.foldList= foldList
			self.index = 0
			self.outerReference = outerReference

		def __iter__(self):
			return self

		def next(self):
			if self.index >= len(self.foldList):
				raise StopIteration
			copied = self.outerReference.copy()
			dataY = copied.extractPoints(self.foldList[self.index])
			dataX = copied
#			dataX.shufflePoints()
			self.index = self.index +1
			return dataX, dataY


