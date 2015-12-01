"""
Anchors the hierarchy of data representation types, providing stubs and common functions.

"""

# TODO conversions
# TODO who sorts inputs to derived implementations?

try:
	import matplotlib
	mplError = None
except ImportError as mplError:
	pass

import math
import itertools
import copy
import numpy
import scipy
import sys
import os.path
import inspect
from multiprocessing import Process

import UML
from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException
from UML.logger import produceFeaturewiseReport
from UML.logger import produceAggregateReport
from UML.randomness import pythonRandom

import dataHelpers

# the prefix for default point and feature names
from dataHelpers import DEFAULT_PREFIX

from dataHelpers import DEFAULT_NAME_PREFIX

from dataHelpers import formatIfNeeded

from dataHelpers import makeConsistentFNamesAndData

class Base(object):
	"""
	Class defining important data manipulation operations and giving functionality
	for the naming the features of that data. A mapping from feature names to feature
	indices is given by the featureNames attribute, the inverse of that mapping is
	given by featureNamesInverse.

	"""

	def __init__(self, shape, pointNames=None, featureNames=None, name=None,
			paths=(None,None), **kwds):
		"""
		Instantiates the book-keeping structures that are taken to be common
		across all data types. Specifically, this includes point and feature
		names, an object name, and originating pathes for the data in this
		object. Note: this method (as should all other __init__ methods in
		this hierarchy) makes use of super()
		
		pointNames: may be a list or dict mapping names to indices. None is
		given if default names are desired.

		featureNames: may be a list or dict mapping names to indices. None is
		given if default names are desired.

		name: the name to be associated with this object.

		pathes: a tuple, where the first entry is taken to be the string
		representing the absolute path to the source file of the data and
		the second entry is taken to be the relative path. Both may be
		None if these values are to be unspecified.

		**kwds: potentially full of arguments further up the class hierarchy,
		as following best practices for use of super(). Note however, that
		this class is the root of the object hierarchy as statically defined. 

		"""
		self._pointCount = shape[0]
		self._featureCount = shape[1]

		if pointNames is not None and len(pointNames) != shape[0]:
			msg = "The length of the pointNames (" + str(len(pointNames))
			msg += ") must match the points given in shape (" + str(shape[0])
			msg += ")"
			raise ArgumentException(msg)
		if featureNames is not None and len(featureNames) != shape[1]:
			msg = "The length of the featureNames (" + str(len(featureNames))
			msg += ") must match the features given in shape ("
			msg += str(shape[1]) + ")"
			raise ArgumentException(msg)

		# Set up point names
		self._nextDefaultValuePoint = 0
		self._setAllDefault('point')
		if isinstance(pointNames, list):
			self.setPointNames(pointNames)
		elif isinstance(pointNames, dict):
			self.setPointNames(pointNames)
		# could still be an ordered container, pass it on to the list helper
		elif hasattr(pointNames, '__len__') and hasattr(pointNames, '__getitem__'):
			self.setPointNames(pointNames)
		elif pointNames is None:
			pass
		else:
			raise ArgumentException("pointNames may only be a list, an ordered container, or a dict, defining a mapping between integers and pointNames")

		# Set up feature names
		self._nextDefaultValueFeature = 0
		self._setAllDefault('feature')
		if isinstance(featureNames, list):
			self.setFeatureNames(featureNames)
		elif isinstance(featureNames, dict):
			self.setFeatureNames(featureNames)
		# could still be an ordered container, pass it on to the list helper
		elif hasattr(featureNames, '__len__') and hasattr(featureNames, '__getitem__'):
			self.setFeatureNames(featureNames)
		elif featureNames is None:
			pass
		else:
			raise ArgumentException("featureNames may only be a list, an ordered container, or a dict, defining a mapping between integers and featureNames")
		
		# Set up object name
		if name is None:
			self._name = dataHelpers.nextDefaultObjectName()
		else:
			self._name = name

		# Set up paths
		if paths[0] is not None and not isinstance(paths[0], basestring):
			raise ArgumentException("paths[0] must be None or an absolute path to the file from which the data originates")
		if paths[0] is not None and not os.path.isabs(paths[0]):
			raise ArgumentException("paths[0] must be an absolute path")
		self._absPath = paths[0]

		if paths[1] is not None and not isinstance(paths[1], basestring):
			raise ArgumentException("paths[1] must be None or a relative path to the file from which the data originates")
		self._relPath = paths[1]

		# call for safety
		super(Base,self).__init__(**kwds)


	#######################
	# Property Attributes #
	#######################

	def _getpointCount(self):
		return self._pointCount

	pointCount = property(_getpointCount, doc="The number of points in this object")

	def _getfeatureCount(self):
		return self._featureCount
	featureCount = property(_getfeatureCount, doc="The number of features in this object")

	def _getObjName(self):
		return self._name

	def _setObjName(self, value):
		if value is None:
			self._name = dataHelpers.nextDefaultObjectName()
		else:
			if not isinstance(value,basestring):
				msg = "The name of an object may only be a string, or the value None"
				raise ValueError(msg)
			self._name = value
	name = property(_getObjName, _setObjName, doc="A name to be displayed when printing or logging this object")

	def _getAbsPath(self):
		return self._absPath
	absolutePath = property(_getAbsPath, doc="The path to the file this data originated from, in absolute form")

	def _getRelPath(self):
		return self._relPath
	relativePath = property(_getRelPath, doc="The path to the file this data originated from, in relative form")

	def _getPath(self):
		return self.absolutePath
	path = property(_getPath, doc="The path to the file this data originated from")

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

		None is always returned.

		"""
		if self.pointCount == 0:
			raise ArgumentException("Cannot set any point names; this object has no points ")
		self._setName_implementation(oldIdentifier, newName, 'point', False)

	def setFeatureName(self, oldIdentifier, newName):
		"""
		Changes the featureName specified by previous to the supplied input name.
		
		oldIdentifier must be a non None string or integer, specifying either a current featureName
		or the index of a current featureName. newName may be either a string not currently
		in the featureName set, or None for an default featureName. newName cannot begin with the
		default prefix.

		None is always returned.

		"""
		if self.featureCount == 0:
			raise ArgumentException("Cannot set any feature names; this object has no features ")
		self._setName_implementation(oldIdentifier, newName, 'feature', False)


	def setPointNames(self, assignments=None): 
		"""
		Rename all of the point names of this object according to the values
		specified by the assignments parameter. If given a list, then we use
		the mapping between names and array indices to define the point
		names. If given a dict, then that mapping will be used to define the
		point names. If assignments is None, then all point names will be
		given new default values. If assignment is an unexpected type, the names
		are not strings, the names are not unique, or point indices are missing,
		then an ArgumentException will be raised. None is always returned.

		"""
		if assignments is None or isinstance(assignments, list):
			self._setNamesFromList(assignments, self.pointCount, 'point')
		elif isinstance(assignments, dict):
			self._setNamesFromDict(assignments, self.pointCount, 'point')
		else:
			msg = "'assignments' parameter may only be a list, a dict, or None, "
			msg += "yet a value of type " + str(type(assignments)) + " was given"
			raise ArgumentException(msg)

	def setFeatureNames(self, assignments=None): 
		"""
		Rename all of the feature names of this object according to the values
		specified by the assignments parameter. If given a list, then we use
		the mapping between names and array indices to define the feature
		names. If given a dict, then that mapping will be used to define the
		feature names. If assignments is None, then all feature names will be
		given new default values. If assignment is an unexpected type, the names
		are not strings, the names are not unique, or feature indices are missing,
		then an ArgumentException will be raised. None is always returned.

		"""
		if assignments is None or isinstance(assignments, list):
			self._setNamesFromList(assignments, self.featureCount, 'feature')
		elif isinstance(assignments, dict):
			self._setNamesFromDict(assignments, self.featureCount, 'feature')
		else:
			msg = "'assignments' parameter may only be a list, a dict, or None, "
			msg += "yet a value of type " + str(type(assignments)) + " was given"
			raise ArgumentException(msg)

	def nameIsDefault(self):
		"""Returns True if self.name has a default value"""
		return self.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)

	def getPointNames(self):
		"""Returns a list containing all point names, where their index
		in the list is the same as the index of the point they correspond
		to.

		"""
		ret = []
		for i in xrange(len(self.pointNamesInverse)):
			ret.append(self.getPointName(i))

		return ret

	def getFeatureNames(self):
		"""Returns a list containing all feature names, where their index
		in the list is the same as the index of the feature they
		correspond to.

		"""
		ret = []
		for i in xrange(len(self.featureNamesInverse)):
			ret.append(self.getFeatureName(i))

		return ret

	def getPointName(self, index):
		return self.pointNamesInverse[index]

	def getPointIndex(self, name):
		return self.pointNames[name]

	def getFeatureName(self, index):
		return self.featureNamesInverse[index]

	def getFeatureIndex(self, name):
		return self.featureNames[name]

	###########################
	# Higher Order Operations #
	###########################
	
	def dropFeaturesContainingType(self, typeToDrop):
		"""
		Modify this object so that it no longer contains features which have the specified
		type as values. None is always returned.

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
			return

		def hasType(feature):
			for value in feature:
				for typeValue in typeToDrop:
					if isinstance(value, typeValue):
						return True
			return False
	
		removed = self.extractFeatures(hasType)
		return


	def replaceFeatureWithBinaryFeatures(self, featureToReplace):
		"""
		Modify this object so that the chosen feature is removed, and binary valued
		features are added, one for each possible value seen in the original feature.
		None is always returned.

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

		varName = toConvert.getFeatureName(0)

		for point in values.data:
			value = point[0]
			ret = toConvert.applyToPoints(makeFunc(value), inPlace=False)
			ret.setFeatureName(0, varName + "=" + str(value).strip())
			toConvert.appendFeatures(ret)

		# remove the original feature, and combine with self
		toConvert.extractFeatures([varName])
		self.appendFeatures(toConvert)


	def transformFeatureToIntegers(self, featureToConvert):
		"""
		Modify this object so that the chosen feature in removed, and a new integer
		valued feature is added with values 0 to n-1, one for each of n values present
		in the original feature. None is always returned.

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
		converted.setPointNames(toConvert.getPointNames())
		converted.setFeatureName(0, toConvert.getFeatureName(0))

		self.appendFeatures(converted)

	def extractPointsByCoinToss(self, extractionProbability):
		"""
		Return a new object containing a randomly selected sample of points
		from this object, where a random experiment is performed for each
		point, with the chance of selection equal to the extractionProbabilty
		parameter. Those selected values are also removed from this object.

		"""
#		if self.pointCount == 0:
#			raise ImproperActionException("Cannot extract points from an object with 0 points")

		if extractionProbability is None:
			raise ArgumentException("Must provide a extractionProbability")
		if extractionProbability <= 0:
			raise ArgumentException("extractionProbability must be greater than zero")
		if extractionProbability >= 1:
			raise ArgumentException("extractionProbability must be less than one")

		def experiment(point):
			return bool(pythonRandom.random() < extractionProbability)

		ret = self.extractPoints(experiment)

		return ret


	def applyToPoints(self, function, points=None, inPlace=True):
		"""
		Applies the given function to each point in this object, copying the
		output into this object and returning None. Alternatively, if the inPlace
		flag is False, output values are collected into a new object that is
		returned upon completion.

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

		if points is not None:
			for i in xrange(len(points)):
				points[i] = self._getPointIndex(points[i])

		self.validate()

		ret = self._applyTo_implementation(function, points, inPlace, 'point')

		if ret is not None:
			ret._absPath = self.absolutePath
			ret._relPath = self.relativePath

		return ret


	def applyToFeatures(self, function, features=None, inPlace=True):
		"""
		Applies the given function to each feature in this object, copying the
		output into this object and returning None. Alternatively, if the inPlace
		flag is False, output values are collected into a new object that is
		returned upon completion.

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

		ret = self._applyTo_implementation(function, features, inPlace, 'feature')
		if ret is not None:
			ret._absPath = self.absolutePath
			ret._relPath = self.relativePath
		return ret


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
			ret = None
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
		ret = UML.createData(self.getTypeString(), ret)

		ret._absPath = self.absolutePath
		ret._relPath = self.relativePath

		return ret

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
		featureNum) to each element. If inPlace == False, returns an object (of
		the same type as the calling object) containing the resulting values, or
		None if inPlace == True. If preserveZeros=True it does not
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
			if not isinstance(points, (int, basestring)):
				raise ArgumentException("Only allowable inputs to 'points' parameter is an int ID, a list of int ID's, or None")
			points = [points]

		if features is not None and not isinstance(features, list):
			if not isinstance(features, (int, basestring)):
				raise ArgumentException("Only allowable inputs to 'features' parameter is an ID, a list of int ID's, or None")
			features = [features]

		if points is not None:
			for i in xrange(len(points)):
				points[i] = self._getPointIndex(points[i])

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
			ret = UML.createData(self.getTypeString(), valueList)

			ret._absPath = self.absolutePath
			ret._relPath = self.relativePath

			return ret
		else:
			return None

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


	def shufflePoints(self, indices=None):
		"""
		Permute the indexing of the points so they are in a random order. Note: this relies on
		python's random.shuffle() so may not be sufficiently random for large number of points.
		See shuffle()'s documentation. None is always returned.

		"""
		if indices is None:
			indices = range(0, self.pointCount)
			pythonRandom.shuffle(indices)
		else:
			if len(indices) != self.pointCount:
				raise ArgumentException("If indices are supplied, it must be a list with all and only valid point indices")
			for value in indices:
				if value < 0 or value > self.pointCount:
					raise ArgumentException("A value in indices is out of bounds of the valid range of points")

		def permuter(pointView):
			return indices[pointView.index()]
		self.sortPoints(sortHelper=permuter)
		


	def shuffleFeatures(self, indices=None):
		"""
		Permute the indexing of the features so they are in a random order. Note: this relies on
		python's random.shuffle() so may not be sufficiently random for large number of features.
		See shuffle()'s documentation. None is always returned.

		"""
		if indices is None:
			indices = range(0, self.featureCount)
			pythonRandom.shuffle(indices)
		else:
			if len(indices) != self.featureCount:
				raise ArgumentException("If indices are supplied, it must be a list with all and only valid features indices")
			for value in indices:
				if value < 0 or value > self.featureCount:
					raise ArgumentException("A value in indices is out of bounds of the valid range of features")
		def permuter(featureView):
			return indices[featureView.index()]
		self.sortFeatures(sortHelper=permuter)
		

	def copy(self):
		"""
		Return a new object which has the same data (and featureNames, depending on
		the return type) and in the same UML format as this object.

		"""
		return self.copyAs(self.getTypeString())

	def trainAndTestSets(self, testFraction, labels=None, randomOrder=True):
		"""Partitions this object into training / testing, data / labels
		sets, returning a new object for each as needed.

		testFraction: the fraction of the data to be placed in the testing
		sets. If randomOrder is False, then the points are taken from the
		end of this object. 

		labels: may be None, a single feature ID, or a list of feature
		IDs depending on whether one is dealing with data for unsupervised
		learning, single variable supervised learning, or multi-output
		supervised learning. Consequently, if labels is None, then only two
		data sets (training and testing) are returned, otherwise, four sets
		are returned (data and labels each for both training and testing).

		randomOrder: controls whether the order of the points in the returns
		sets matches that of the original object, or if their order is
		randomized.

		"""
		toSplit = self.copy()
		if randomOrder:
			toSplit.shufflePoints()

		testXSize = int(round(testFraction * self.pointCount))
		startIndex = self.pointCount - testXSize

		#pull out a testing set
		if testXSize == 0:
			testX = toSplit.extractPoints([])
		else:
			testX = toSplit.extractPoints(start=startIndex)

		if labels is None:
			toSplit.name = self.name + " trainX"
			testX.name = self.name + " testX"
			return toSplit, testX

		# safety for empty objects
		toExtract = labels
		if testXSize == 0:
			toExtract = []

		trainY = toSplit.extractFeatures(toExtract)
		testY = testX.extractFeatures(toExtract)
	
		toSplit.name = self.name + " trainX"
		trainY.name = self.name + " trainY"
		testX.name = self.name + " testX"
		testY.name = self.name + " testY"

		return toSplit, trainY, testX, testY


	########################################
	########################################
	###   Functions related to logging   ###
	########################################
	########################################


	def featureReport(self, maxFeaturesToCover=50, displayDigits=2):
		"""
		Produce a report, in a string formatted as a table, containing summary and statistical
		information about each feature in the data set, up to 50 features.  If there are more
		than 50 features, only information about 50 of those features will be reported.
		"""
		return produceFeaturewiseReport(self, maxFeaturesToCover=maxFeaturesToCover, displayDigits=displayDigits)

	def summaryReport(self, displayDigits=2):
		"""
		Produce a report, in a string formatted as a table, containing summary 
		information about the data set contained in this object.  Includes 
		proportion of missing values, proportion of zero values, total # of points,
		and number of features.
		"""
		return produceAggregateReport(self, displayDigits=displayDigits)


	###############################################################
	###############################################################
	###   Subclass implemented information querying functions   ###
	###############################################################
	###############################################################


	def isIdentical(self, other):
		if not self._equalFeatureNames(other):
			return False
		if not self._equalPointNames(other):
			return False

		return self._isIdentical_implementation(other)


	def writeFile(self, outPath, format=None, includeNames=True):
		"""
		Function to write the data in this object to a file using the specified
		format. outPath is the location (including file name and extension) where
		we want to write the output file. includeNames is boolean argument
		indicating whether the file should start with comment lines designating
		pointNames and featureNames.

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

		if format not in ['csv', 'mtx']:
			msg = "Unrecognized file format. Accepted types are 'csv' and 'mtx'. They may "
			msg += "either be input as the format parameter, or as the extension in the "
			msg += "outPath"
			raise ArgumentException(msg)

		includePointNames = includeNames
		if includePointNames:
			seen = False
			for name in self.getPointNames():
				if not name.startswith(DEFAULT_PREFIX):
					seen = True
			if not seen:
				includePointNames = False

		includeFeatureNames = includeNames
		if includeFeatureNames:
			seen = False
			for name in self.getFeatureNames():
				if not name.startswith(DEFAULT_PREFIX):
					seen = True
			if not seen:
				includeFeatureNames = False

		try:
			self._writeFile_implementation(outPath, format, includePointNames, includeFeatureNames)
		except Exception:
			if format.lower() == "csv":
				toOut = self.copyAs("Matrix")
				toOut._writeFile_implementation(outPath, format, includePointNames, includeFeatureNames)
				return
			if format.lower() == "mtx":
				toOut = self.copyAs('Sparse')
				toOut._writeFile_implementation(outPath, format, includePointNames, includeFeatureNames)
				return	


	def getTypeString(self):
		"""
			Return a string representing the non-abstract type of this object (e.g. Matrix,
			Sparse, etc.) that can be passed to createData() function to create a new object
			of the same type.
		"""
		return self._getTypeString_implementation()

	def __getitem__(self, key):
		if isinstance(key, float):
			intVal = int(key)
			if intVal != key:
				msg = "A float valued key of value x is only accepted if x == "
				msg += "int(x). The given value was " + str(key) + " yet int("
				msg += str(key) + ") = " + str(intVal)
				raise ArgumentException(msg)
			key = intVal

		if isinstance(key, (int, basestring)):
			if self.pointCount == 1:
				x = 0
				y = key
			elif self.featureCount == 1:
				x = key
				y = 0
			else:
				msg = "Must include a point and feature index; or, "
				msg += "since this is vector chaped, a single index "
				msg += "into the axis whose length > 1"
				raise ArgumentException(msg)
		else:
			try:
				(x,y) = key
				if isinstance(x, float):
					intValX = int(x)
					if intValX != x:
						msg = "A key with a float valued index of value x is only "
						msg += "accepted if x == int(x). The point ID (left hand "
						msg += "side of the tuple) was given as " + str(x) + " yet"
						msg += " int(" + str(x) + ") = " + str(intValX)
						raise ArgumentException(msg) 
					x = intValX
				if isinstance(y, float):
					intValY = int(y)
					if intValY != y:
						msg = "A key with a float valued index of value y is only "
						msg += "accepted if y == int(y). The feature ID (right hand "
						msg += "side of the tuple) was given as " + str(y) + " yet"
						msg += " int(" + str(y) + ") = " + str(intValY)
						raise ArgumentException(msg) 
					y = intValY
			except TypeError:
				msg = "Must include a point and feature index; or, "
				msg += "if this has only a single point or feature, "
				msg += "you may specify a single index "
				msg += "into the axis whose length > 1"
				raise ArgumentException(msg)

		x = self._getPointIndex(x)
		if not isinstance(x,int) or x < 0 or x >= self.pointCount:
			raise ArgumentException(str(x) + " is not a valid point ID")

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
		
		index = self._getPointIndex(ID)
		return self._pointView_implementation(index)

	def featureView(self, ID):
		"""
		Returns a View object into the data of the point with the given ID. See View object
		comments for its capabilities. This View is only valid until the next modification
		to the shape or ordering of the internal data. After such a modification, there is
		no guarantee to the validity of the results.
		"""
		if self.featureCount == 0:
			raise ImproperActionException("ID is invalid, This object contains no features")

		index = self._getFeatureIndex(ID)
		return self._featureView_implementation(index)
	
	def view(self, pointStart=None, pointEnd=None, featureStart=None,
			featureEnd=None):
		
		if pointStart is None:
			pointStart = 0
		if pointEnd is None:
			pointEnd = self.pointCount
		if featureStart is None:
			featureStart = 0
		if featureEnd is None:
			featureEnd = self.featureCount

		return self._view_implementation(pointStart, pointEnd, featureStart,
				featureEnd)

	def validate(self, level=1):
		"""
		Checks the integrity of the data with respect to the limitations and invariants
		that our objects enforce.

		"""
		assert self.featureCount == len(self.getFeatureNames())
		assert self.pointCount == len(self.getPointNames())

		if level > 0:
			for key in self.getPointNames():
				assert self.getPointName(self.getPointIndex(key)) == key
			for key in self.getFeatureNames():
				assert self.getFeatureName(self.getFeatureIndex(key)) == key

		self._validate_implementation(level)


	def containsZero(self):
		"""
		Returns True if there is a value that is equal to integer 0 contained
		in this object. False otherwise

		"""
		# trivially False.
		if self.pointCount == 0 or self.featureCount == 0:
			return False
		return self._containsZero_implementation()

	def __eq__(self, other):
		return self.isIdentical(other)

	def __ne__(self, other):
		return not self.__eq__(other)


	def toString(self, includeNames=True, maxWidth=80, maxHeight=30,
				sigDigits=3, nameLength=11):

		if self.pointCount == 0 or self.featureCount == 0:
			return ""

		# setup a bundle of fixed constants
		colSep = ' '
		colHold = '--'
		rowHold = '|'
		pnameSep = ' '
		nameHolder = '...'
		holderOrientation = 'center'
		dataOrientation = 'center'
		pNameOrientation = 'rjust'
		fNameOrientation = 'center'

		#setup a bundle of default values
		maxHeight = self.pointCount + 2 if maxHeight is None else maxHeight
		maxWidth = float('inf') if maxWidth is None else maxWidth
		maxRows = min(maxHeight, self.pointCount)
		maxDataRows = maxRows
		includePNames = False
		includeFNames = False

		if includeNames:
			includePNames = dataHelpers.hasNonDefault(self, 'point')
			includeFNames = dataHelpers.hasNonDefault(self, 'feature')
			if includeFNames:
				# plus or minus 2 because we will be dealing with both
				# feature names and a gap row
				maxRows = min(maxHeight, self.pointCount + 2)
				maxDataRows = maxRows - 2

		# Set up point Names and determine how much space they take up
		pnames = None
		pnamesWidth = None
		maxDataWidth = maxWidth
		if includePNames:
			pnames, pnamesWidth = self._arrangePointNames(maxDataRows, nameLength,
					rowHold, nameHolder)
			# The available space for the data is reduced by the width of the
			# pnames, a column separator, the pnames seperator, and another
			# column seperator 
			maxDataWidth = maxWidth - (pnamesWidth + 2 * len(colSep) + len(pnameSep))	

		# Set up data values to fit in the available space
		dataTable, colWidths = self._arrangeDataWithLimits(maxDataWidth, maxDataRows,
				sigDigits, colSep, colHold, rowHold)

		# set up feature names list, record widths
		fnames = None
		if includeFNames:
			fnames = self._arrangeFeatureNames(maxWidth, nameLength,
					colSep, colHold, nameHolder)

			# adjust data or fnames according to the more restrictive set
			# of col widths
			makeConsistentFNamesAndData(fnames, dataTable, colWidths, colHold)

		# combine names into finalized table
		finalTable, finalWidths = self._arrangeFinalTable(pnames, pnamesWidth,
				dataTable, colWidths, fnames, pnameSep)

		# set up output string
		out = ""
		for r in xrange(len(finalTable)):
			row = finalTable[r]
			for c in xrange(len(row)):
				val = row[c]
				if c == 0 and includePNames:
					padded = getattr(val, pNameOrientation)(finalWidths[c])
				elif r == 0 and includeFNames:
					padded = getattr(val, fNameOrientation)(finalWidths[c])
				else:
					padded = getattr(val, dataOrientation)(finalWidths[c])
				row[c] = padded
			line = colSep.join(finalTable[r]) + "\n"
			out += line

		return out



	def __repr__(self):
		indent = '    '
		maxW = 80
		maxH = 40

		# setup type call
		ret = self.getTypeString() + "(\n"
		
		# setup data
		dataStr = self.toString(includeNames=False, maxWidth=maxW, maxHeight=maxH)
		byLine = dataStr.split('\n')
		# toString ends with a \n, so we get rid of the empty line produced by
		# the split
		byLine = byLine[:(len(byLine)-1)]
		for i in range(len(byLine)):
			line = byLine[i]
			newLine = indent + '[' if i == 0 else indent + ' '
			newLine += '[' + line + ']'
			if i == (len(byLine)-1):
				newLine += ']'
			ret += newLine + '\n'

		numRows = min(self.pointCount, maxW)
		# if exists non default point names, print all (truncated) point names
		ret += dataHelpers.makeNamesLines(indent, maxW, numRows, self.pointCount,
				self.getPointNames(), 'pointNames')
		# if exists non default feature names, print all (truncated) feature names
		splited = byLine[0].split(' ')
		numCols = 0
		for val in splited:
			if val != '' and val != '...':
				numCols += 1
		# because of how dataHelers.indicesSplit works, we need this to be plus one
		# in some cases this means one extra feature name is displayed. But that's
		# acceptable
		if numCols <= self.featureCount:
			numCols += 1
		ret += dataHelpers.makeNamesLines(indent, maxW, numCols, self.featureCount,
				self.getFeatureNames(), 'featureNames')

		# if name not None, print
		if not self.name.startswith(DEFAULT_NAME_PREFIX):
			prep = indent + 'name="'
			toUse = self.name
			nonNameLen = len(prep) + 1
			if nonNameLen + len(toUse) > 80:
				toUse = toUse[:(80 - nonNameLen - 3)]
				toUse += '...'

			ret += prep + toUse + '"\n'

		# if path not None, print
		if self.path is not None:
			prep = indent + 'path="'
			toUse = self.path
			nonPathLen = len(prep) + 1
			if nonPathLen + len(toUse) > 80:
				toUse = toUse[:(80 - nonPathLen - 3)]
				toUse += '...'

			ret += prep + toUse + '"\n'

		ret += indent +')'

		return ret

	def __str__(self):
		return self.toString()

	def show(self, description, includeObjectName=True, includeAxisNames=True, maxWidth=80,
			maxHeight=30, sigDigits=3, nameLength=11):
		"""Method to simplify printing a representation of this data object,
		with some context. The backend is the toString() method, and this
		method includes control over all of the same functionality via
		arguments. Prior to the names and data, it additionally prints a
		description provided by the user, (optionally) this object's name
		attribute, and the number of points and features that are in the
		data.

		description: Unless None, this is printed as-is before the rest of
		the output.

		includeObjectName: if True, the object's name attribute will be
		printed.

		includeAxisNames: if True, the point and feature names will be
		printed.

		maxWidth: a bound on the maximum number of characters printed on
		each line of the output.

		maxHeight: a bound on the maximum number of lines printed in the
		outout.

		sigDigits: the number of decimal places to show when printing
		float valued data.

		nameLength: a bound on the maximum number of characters we allow
		for each point or feature name.

		"""
		if description is not None:
			print description

		if includeObjectName:
			context = self.name + " : "
		else:
			context = ""
		context += str(self.pointCount) + "pt x "
		context += str(self.featureCount) + "ft"
		print context
		print self.toString(includeAxisNames, maxWidth, maxHeight, sigDigits, nameLength)


	def plot(self, outPath=None, includeColorbar=False):
		self._plot(outPath, includeColorbar)


	def _setupOutFormatForPlotting(self, outPath):
		outFormat = None
		if isinstance(outPath, basestring):
			(path, ext) = os.path.splitext(outPath)
			if len(ext) == 0:
				outFormat = 'png'
		return outFormat

	def _plot(self, outPath=None, includeColorbar=False):
		self._validateMatPlotLibImport(mplError, 'plot')
		outFormat = self._setupOutFormatForPlotting(outPath)

		def plotter(d):
			import matplotlib.pyplot as plt
			plt.matshow(d, cmap=matplotlib.cm.gray)

			if includeColorbar:
				plt.colorbar()

			if not self.name.startswith(DEFAULT_NAME_PREFIX):
				#plt.title("Heatmap of " + self.name)
				plt.title(self.name)
			plt.xlabel("Feature Values", labelpad=10)
			plt.ylabel("Point Values")

			if outPath is None:
				plt.show()
			else:
				plt.savefig(outPath, format=outFormat)

		toPlot = self.copyAs('numpyarray')
		p = Process(target=plotter, kwargs={'d':self.data})
		p.start()
		return p


	def plotPointDistribution(self, point, outPath=None, xMin=None, xMax=None):
		"""Plot a histogram of the distribution of values in the specified
		point. Along the x axis of the plot will be the values seen in
		the point, grouped into bins; along the y axis will be the number
		of values in each bin. Bin width is calculated using
		Freedman-Diaconis' rule. Control over the width of the x axis
		is also given, with the warning that user specified values
		can obscure data that would otherwise be plotted given default
		inputs.

		point: the identifier (index of name) of the point to show

		xMin: the least value shown on the x axis of the resultant plot.

		xMax: the largest value shown on the x axis of teh resultant plot
		
		"""
		self._plotPointDistribution(point, outPath, xMin, xMax)

	def _plotPointDistribution(self, point, outPath, xMin=None, xMax=None):
		self._validateMatPlotLibImport(mplError, 'plotPointDistribution')
		return self._plotDistribution('point', point, outPath, xMin, xMax)

	def plotFeatureDistribution(self, feature, outPath=None, xMin=None, xMax=None):
		"""Plot a histogram of the distribution of values in the specified
		Feature. Along the x axis of the plot will be the values seen in
		the feature, grouped into bins; along the y axis will be the number
		of values in each bin. Bin width is calculated using
		Freedman-Diaconis' rule. Control over the width of the x axis
		is also given, with the warning that user specified values
		can obscure data that would otherwise be plotted given default
		inputs.

		feature: the identifier (index of name) of the feature to show

		xMin: the least value shown on the x axis of the resultant plot.

		xMax: the largest value shown on the x axis of teh resultant plot
		
		"""
		self._plotFeatureDistribution(feature, outPath, xMin, xMax)

	def _plotFeatureDistribution(self, feature, outPath=None, xMin=None, xMax=None):
		self._validateMatPlotLibImport(mplError, 'plotFeatureDistribution')
		return self._plotDistribution('feature', feature, outPath, xMin, xMax)

	def _plotDistribution(self, axis, identifier, outPath, xMin, xMax):
		outFormat = self._setupOutFormatForPlotting(outPath)
		index = self._getIndex(identifier, axis)
		if axis == 'point':
			getter = self.pointView
			name = self.getPointName(index)
		else:
			getter = self.featureView
			name = self.getFeatureName(index)

#		getter = self.pointView if axis == 'point' else self.featureView
		toPlot = getter(index)

		quartiles = UML.calculate.quartiles(toPlot)

		IQR = quartiles[2] - quartiles[0]
		binWidth = (2 * IQR) / (len(toPlot) ** (1./3))
		# TODO: replace with calculate points after it subsumes
		# pointStatistics?
		valMax = max(toPlot)
		valMin = min(toPlot)
		if binWidth == 0:
			binCount = 1
		else:
			binCount = math.ceil((valMax - valMin) / binWidth)

		def plotter(d, xLim):
			import matplotlib.pyplot as plt
			plt.hist(d, binCount)
			
			if name.startswith(DEFAULT_PREFIX):
				titlemsg = '#' + str(index)
			else:
				titlemsg = "named: " + name
			plt.title("Distribution of " + axis + " " + titlemsg)
			plt.xlabel("Values")
			plt.ylabel("Number of values")

			plt.xlim(xLim)
			
			if outPath is None:
				plt.show()
			else:
				plt.savefig(outPath, format=outFormat)

		p = Process(target=plotter, kwargs={'d':toPlot, 'xLim':(xMin,xMax)})
		p.start()
		return p


	def plotPointAgainstPoint(self, x, y, outPath=None, xMin=None, xMax=None, yMin=None,
				yMax=None):
		"""Plot a scatter plot of the two input points using the pairwise
		combination of their values as coordinates. Control over the width
		of the both axes is given, with the warning that user specified
		values can obscure data that would otherwise be plotted given default
		inputs.

		x: the identifier (index of name) of the point from which we
		draw x-axis coordinates

		y: the identifier (index of name) of the point from which we
		draw y-axis coordinates

		xMin: the least value shown on the x axis of the resultant plot.

		xMax: the largest value shown on the x axis of teh resultant plot

		yMin: the least value shown on the y axis of the resultant plot.

		yMax: the largest value shown on the y axis of teh resultant plot
		
		"""
		self._plotPointAgainstPoint(x, y, outPath, xMin, xMax, yMin, yMax)

	def _plotPointAgainstPoint(self, x, y, outPath=None, xMin=None, xMax=None, yMin=None,
				yMax=None):
		self._validateMatPlotLibImport(mplError, 'plotPointComparison')
		return self._plotCross(x, 'point', y, 'point', outPath, xMin, xMax, yMin, yMax)

	def plotFeatureAgainstFeature(self, x, y, outPath=None, xMin=None, xMax=None, yMin=None,
				yMax=None):
		"""Plot a scatter plot of the two input features using the pairwise
		combination of their values as coordinates. Control over the width
		of the both axes is given, with the warning that user specified
		values can obscure data that would otherwise be plotted given default
		inputs.

		x: the identifier (index of name) of the feature from which we
		draw x-axis coordinates

		y: the identifier (index of name) of the feature from which we
		draw y-axis coordinates

		xMin: the least value shown on the x axis of the resultant plot.

		xMax: the largest value shown on the x axis of the resultant plot

		yMin: the least value shown on the y axis of the resultant plot.

		yMax: the largest value shown on the y axis of the resultant plot
		
		"""
		self._plotFeatureAgainstFeature(x, y, outPath, xMin, xMax, yMin, yMax)

	def _plotFeatureAgainstFeature(self, x, y, outPath=None, xMin=None, xMax=None, yMin=None,
				yMax=None):
		self._validateMatPlotLibImport(mplError, 'plotFeatureComparison')
		return self._plotCross(x, 'feature', y, 'feature', outPath, xMin, xMax, yMin, yMax)

	def _plotCross(self, x, xAxis, y, yAxis, outPath, xMin, xMax, yMin, yMax):
		outFormat = self._setupOutFormatForPlotting(outPath)
		xIndex = self._getIndex(x, xAxis)
		yIndex = self._getIndex(y, yAxis)

		def customGetter(index, axis):
			copyied = self.copyPoints(index) if axis == 'point' else self.copyFeatures(index)
			return copyied.copyAs('numpyarray', outputAs1D=True)

		def pGetter(index):
			return customGetter(index, 'point')
		def fGetter(index):
			return customGetter(index, 'feature')

		if xAxis == 'point':
			xGetter = pGetter
			xName = self.getPointName(xIndex)
		else:
			xGetter = fGetter
			xName = self.getFeatureName(xIndex)

		if yAxis == 'point':
			yGetter = pGetter
			yName = self.getPointName(yIndex)
		else:
			yGetter = fGetter
			yName = self.getFeatureName(yIndex)

		xToPlot = xGetter(xIndex)
		yToPlot = yGetter(yIndex)

		def plotter(inX, inY, xLim, yLim):
			import matplotlib.pyplot as plt
			#plt.scatter(inX, inY)
			plt.scatter(inX, inY, marker='.')
			
			if not self.name.startswith(DEFAULT_PREFIX):
				plt.title("Comparison of data in object " + self.name)
			
			if xName.startswith(DEFAULT_PREFIX):
				plt.xlabel(xAxis + ' #' + str(xIndex))
			else:
				plt.xlabel(xName)
			if yName.startswith(DEFAULT_PREFIX):
				plt.ylabel(yAxis + ' #' + str(yIndex))
			else:
				plt.ylabel(yName)

			plt.xlim(xLim)
			plt.ylim(yLim)

			if outPath is None:
				plt.show()
			else:
				plt.savefig(outPath, format=outFormat)

		p = Process(target=plotter, kwargs={'inX':xToPlot, 'inY':yToPlot, 'xLim':(xMin,xMax), 'yLim':(yMin,yMax)})
		p.start()
		return p

	##################################################################
	##################################################################
	###   Subclass implemented structural manipulation functions   ###
	##################################################################
	##################################################################


	def transpose(self):
		"""
		Function to transpose the data, ie invert the feature and point indices of the data.
		The point and feature names are also swapped. None is always returned.

		"""
		self._transpose_implementation()

		temp = self._pointCount
		self._pointCount = self._featureCount
		self._featureCount = temp

		temp = self.featureNames
		self.setFeatureNames(self.pointNames)
		self.setPointNames(temp)

		self.validate()

	def appendPoints(self, toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object.

		toAppend cannot be None, and must be a kind of data representation object with the same
		number of features as the calling object. None is always returned.
		
		"""
		self._validateValueIsNotNone("toAppend", toAppend)
		self._validateValueIsUMLDataObject("toAppend", toAppend)
		self._validateObjHasSameNumberOfFeatures("toAppend", toAppend)
		self._validateEqualNames('feature', 'feature', 'toAppend', toAppend)
		self._validateEmptyNamesIntersection("point", "toAppend", toAppend)

		self._appendPoints_implementation(toAppend)
		self._pointCount += toAppend.pointCount

		for i in xrange(toAppend.pointCount):
			currName = toAppend.getPointName(i)
			if currName.startswith(DEFAULT_PREFIX):
				currName += '_' + toAppend.name
			self._addPointName(currName)

		self.validate()

	def appendFeatures(self, toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		toAppend cannot be None, must be a kind of data representation object with the same
		number of points as the calling object, and must not share any feature names with the calling
		object. None is always returned.
		
		"""	
		self._validateValueIsNotNone("toAppend", toAppend)
		self._validateValueIsUMLDataObject("toAppend", toAppend)
		self._validateObjHasSameNumberOfPoints("toAppend", toAppend)
		self._validateEqualNames('point', 'point', 'toAppend', toAppend)
		self._validateEmptyNamesIntersection('feature', "toAppend", toAppend)

		self._appendFeatures_implementation(toAppend)
		self._featureCount += toAppend.featureCount

		for i in xrange(toAppend.featureCount):
			currName = toAppend.getFeatureName(i)
			if currName.startswith(DEFAULT_PREFIX):
				currName += '_' + toAppend.name
			self._addFeatureName(currName)

		self.validate()

	def sortPoints(self, sortBy=None, sortHelper=None):
		""" 
		Modify this object so that the points are sorted in place, where sortBy may
		indicate the feature to sort by or None if the entire point is to be taken as a key,
		sortHelper may either be comparator, a scoring function, or None to indicate the natural
		ordering. None is always returned.
		"""
		# its already sorted in these cases
		if self.featureCount == 0 or self.pointCount == 0 or self.pointCount == 1:
			return
		if sortBy is not None and sortHelper is not None:
			raise ArgumentException("Cannot specify a feature to sort by and a helper function")
		if sortBy is None and sortHelper is None:
			raise ArgumentException("Either sortBy or sortHelper must not be None")

		if sortBy is not None and isinstance(sortBy, basestring):
			sortBy = self._getFeatureIndex(sortBy)

		newPointNameOrder = self._sortPoints_implementation(sortBy, sortHelper)
		self.setPointNames(newPointNameOrder)

		self.validate()

	def sortFeatures(self, sortBy=None, sortHelper=None):
		""" 
		Modify this object so that the features are sorted in place, where sortBy may
		indicate the feature to sort by or None if the entire point is to be taken as a key,
		sortHelper may either be comparator, a scoring function, or None to indicate the natural
		ordering.  None is always returned.

		"""
		# its already sorted in these cases
		if self.featureCount == 0 or self.pointCount == 0 or self.featureCount == 1:
			return
		if sortBy is not None and sortHelper is not None:
			raise ArgumentException("Cannot specify a feature to sort by and a helper function")
		if sortBy is None and sortHelper is None:
			raise ArgumentException("Either sortBy or sortHelper must not be None")

		if sortBy is not None and isinstance(sortBy, basestring):
			sortBy = self._getPointIndex(sortBy)

		newFeatureNameOrder = self._sortFeatures_implementation(sortBy, sortHelper)
		self.setFeatureNames(newFeatureNameOrder)

		self.validate()


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

		if toExtract is not None:
			if start is not None or end is not None:
				raise ArgumentException("Range removal is exclusive, to use it, toExtract must be None")
			if isinstance(toExtract, basestring) or isinstance(toExtract, int):
				toExtract = [toExtract]
			if isinstance(toExtract, list):
				#verify everything in list is a valid index and convert names into indices
				indices = []
				for identifier in toExtract:
					indices.append(self._getPointIndex(identifier))
				toExtract = indices
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
			if number is not None:
				#then we can do the windowing calculation here
				possibleEnd = start + number -1
				if possibleEnd < end:
					if not randomize:
						end = possibleEnd
				else:
					number = (end - start) +1

		ret = self._extractPoints_implementation(toExtract, start, end, number, randomize)
		self._pointCount -= ret.pointCount
		if ret.pointCount != 0:
			ret.setFeatureNames(self.getFeatureNames())
		for key in ret.getPointNames():
			self._removePointNameAndShift(key)

		ret._relPath = self.relativePath
		ret._absPath = self.absolutePath

		self.validate()
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

		if toExtract is not None:
			if start is not None or end is not None:
				raise ArgumentException("Range removal is exclusive, to use it, toExtract must be None")
			if isinstance(toExtract, basestring) or isinstance(toExtract, int):
				toExtract = [toExtract]
			if isinstance(toExtract, list):
				#verify everything in list is a valid index and convert names into indices
				indices = []
				for identifier in toExtract:
					indices.append(self._getFeatureIndex(identifier))
				toExtract = indices
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
			if number is not None:
				#then we can do the windowing calculation here
				possibleEnd = start + number - 1
				if possibleEnd < end:
					if not randomize:
						end = possibleEnd
				else:
					number = (end - start) + 1

		ret = self._extractFeatures_implementation(toExtract, start, end, number, randomize)
		self._featureCount -= ret.featureCount
		if ret.featureCount != 0:
			ret.setPointNames(self.getPointNames())
		for key in ret.getFeatureNames():
			self._removeFeatureNameAndShift(key)

		ret._relPath = self.relativePath
		ret._absPath = self.absolutePath

		self.validate()
		return ret


	def referenceDataFrom(self, other):
		"""
		Modifies the internal data of this object to refer to the same data as other. In other
		words, the data wrapped by both the self and other objects resides in the
		same place in memory. Other must be an object of the same type as
		the calling object. Also, the shape of other should be consistent with the set
		of featureNames currently in this object. None is always returned.

		"""
		# this is called first because it checks the data type
		self._referenceDataFrom_implementation(other)
		self.pointNames = other.pointNames
		self.pointNamesInverse = other.pointNamesInverse
		self.featureNames = other.featureNames
		self.featureNamesInverse = other.featureNamesInverse

		self._pointCount = other._pointCount
		self._featureCount = other._featureCount

		self._absPath = other.absolutePath
		self._relPath = other.relativePath

		self.validate()


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
		
		if outputAs1D:
			if format != 'numpyarray' and format != 'pythonlist':
				raise ArgumentException("Cannot output as 1D if format != 'numpy array' or 'python list'")
			if self.pointCount != 1 and self.featureCount != 1:
				raise ArgumentException("To output as 1D there may either be only one point or one feature")
			if self.pointCount == 0 or self.featureCount == 0:
				if format == 'numpyarray':
					return numpy.array([])
				if format == 'pythonlist':
					return []
			raw = self._copyAs_implementation('numpyarray').flatten()
			if format != 'numpyarray':
				raw = raw.tolist()
			return raw

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
				raise ArgumentException('Cannot output a point or feature empty object in a scipy format')

		ret = self._copyAs_implementation(format)

		if isinstance(ret, UML.data.Base):
			ret._name = self.name
			ret._relPath = self.relativePath
			ret._absPath = self.absolutePath

		if not rowsArePoints:
			if format in ['List', 'Matrix', 'Sparse']:
				ret.transpose()
			elif format != 'pythonlist':
				ret = ret.transpose()
			else:
				ret = numpy.transpose(ret).tolist()

		return ret

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
			#verify everything in list is a valid index and convert names into indices
			indices = []
			for identifier in points:
				indices.append(self._getPointIndex(identifier))
			points = indices

		retObj = self._copyPoints_implementation(points, start, end)

		# construct featureName list
		pointNameList = []
		if points is not None:
			for i in points:
				pointNameList.append(self.getPointName(i))
		else:
			for i in range(start,end+1):
				pointNameList.append(self.getPointName(i))

		retObj.setPointNames(pointNameList)
		retObj.setFeatureNames(self.getFeatureNames())

		retObj._absPath = self.absolutePath
		retObj._relPath = self.relativePath

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
				raise ArgumentException("must specify something to copy; 'features', 'start', and 'end' were all None")
		else:
			if start is not None or end is not None:
				raise ArgumentException("Cannot specify both IDs and a range")
			indices = []
			for identifier in features:
				indices.append(self._getFeatureIndex(identifier))

		ret = self._copyFeatures_implementation(indices, start, end)

		# construct featureName list
		featureNameList = []
		if indices is not None:
			for i in indices:
				featureNameList.append(self.getFeatureName(i))
		else:
			for i in range(start,end+1):
				featureNameList.append(self.getFeatureName(i))

		ret.setPointNames(self.getPointNames())
		ret.setFeatureNames(featureNameList)

		ret._absPath = self.absolutePath
		ret._relPath = self.relativePath

		return ret


	###############################################################
	###############################################################
	###   Subclass implemented numerical operation functions    ###
	###############################################################
	###############################################################

	def elementwiseMultiply(self, other):
		"""
		Perform element wise multiplication of this UML data object against the
		provided other UML data object, with the result being stored in-place in
		the calling object. Both objects must contain only numeric data. The
		pointCount and featureCount of both objects must be equal. The types of
		the two objects may be different. None is always returned.

		"""
		if not isinstance(other, UML.data.Base):
			raise ArgumentException("'other' must be an instance of a UML data object")
		# Test element type self
		if self.pointCount > 0:
			for val in self.pointView(0):
				if not dataHelpers._looksNumeric(val):
					raise ArgumentException("This data object contains non numeric data, cannot do this operation")

		# test element type other
		if other.pointCount > 0:
			for val in other.pointView(0):
				if not dataHelpers._looksNumeric(val):
					raise ArgumentException("This data object contains non numeric data, cannot do this operation")

		if self.pointCount != other.pointCount:
			raise ArgumentException("The number of points in each object must be equal.")
		if self.featureCount != other.featureCount:
			raise ArgumentException("The number of features in each object must be equal.")
		
		if self.pointCount == 0 or self.featureCount == 0:
			raise ImproperActionException("Cannot do elementwiseMultiply when points or features is emtpy")

		self._validateEqualNames('point', 'point', 'elementwiseMultiply', other)
		self._validateEqualNames('feature', 'feature', 'elementwiseMultiply', other)

		self._elementwiseMultiply_implementation(other)

		(retPNames, retFNames) = dataHelpers.mergeNonDefaultNames(self, other)
		self.setPointNames(retPNames)
		self.setFeatureNames(retFNames)
		self.validate()

	def elementwisePower(self, other):
		# other is UML or single numerical value
		singleValue = dataHelpers._looksNumeric(other)
		if not singleValue and not isinstance(other, UML.data.Base):
			raise ArgumentException("'other' must be an instance of a UML data object or a single numeric value")

		# Test element type self
		if self.pointCount > 0:
			for val in self.pointView(0):
				if not dataHelpers._looksNumeric(val):
					raise ArgumentException("This data object contains non numeric data, cannot do this operation")

		# test element type other
		if isinstance(other, UML.data.Base):
			if other.pointCount > 0:
				for val in other.pointView(0):
					if not dataHelpers._looksNumeric(val):
						raise ArgumentException("This data object contains non numeric data, cannot do this operation")

			# same shape
			if self.pointCount != other.pointCount:
				raise ArgumentException("The number of points in each object must be equal.")
			if self.featureCount != other.featureCount:
				raise ArgumentException("The number of features in each object must be equal.")
		
		if self.pointCount == 0 or self.featureCount == 0:
			raise ImproperActionException("Cannot do elementwiseMultiply when points or features is emtpy")

		if isinstance(other, UML.data.Base):
			def powFromRight(val, pnum, fnum):
				return val ** other[pnum,fnum]
			self.applyToElements(powFromRight)
		else: 
			def powFromRight(val, pnum, fnum):
				return val ** other
			self.applyToElements(powFromRight)

		self.validate()



	def __mul__(self, other):
		"""
		Perform matrix multiplication or scalar multiplication on this object depending on
		the input 'other'

		"""
		if not isinstance(other, UML.data.Base) and not dataHelpers._looksNumeric(other):
			return NotImplemented
		
		if self.pointCount == 0 or self.featureCount == 0:
			raise ImproperActionException("Cannot do a multiplication when points or features is empty")

		# Test element type self
		if self.pointCount > 0:
			for val in self.pointView(0):
				if not dataHelpers._looksNumeric(val):
					raise ArgumentException("This data object contains non numeric data, cannot do this operation")

		# test element type other
		if isinstance(other, UML.data.Base):
			if other.pointCount == 0 or other.featureCount == 0:
				raise ImproperActionException("Cannot do a multiplication when points or features is empty")

			if other.pointCount > 0:
				for val in other.pointView(0):
					if not dataHelpers._looksNumeric(val):
						raise ArgumentException("This data object contains non numeric data, cannot do this operation")

			if self.featureCount != other.pointCount:
				raise ArgumentException("The number of features in the calling object must "
						+ "match the point in the callee object.")
		
			self._validateEqualNames('feature', 'point', '__mul__', other)

		ret = self._mul__implementation(other)

		if isinstance(other, UML.data.Base):
			ret.setPointNames(self.getPointNames())
			ret.setFeatureNames(other.getFeatureNames())

		pathSource = 'merge' if isinstance(other, UML.data.Base) else 'self'

		dataHelpers.binaryOpNamePathMerge(self, other, ret, None, pathSource)

		return ret
	
	def __rmul__(self, other):
		"""	Perform scalar multiplication with this object on the right """
		if dataHelpers._looksNumeric(other):
			return self.__mul__(other)
		else:
			return NotImplemented

	def __imul__(self, other):
		"""
		Perform in place matrix multiplication or scalar multiplication, depending in the
		input 'other'

		"""
		ret = self.__mul__(other)
		if ret is not NotImplemented:
			self.referenceDataFrom(ret)
			ret = self

		return ret

	def __add__(self, other):
		"""
		Perform addition on this object, element wise if 'other' is a UML data
		object, or element wise with a scalar if other is some kind of numeric
		value.

		"""
		return self._genericNumericBinary('__add__', other)

	def __radd__(self, other):
		""" Perform scalar addition with this object on the right """
		return self._genericNumericBinary('__radd__', other)

	def __iadd__(self, other):
		"""
		Perform in-place addition on this object, element wise if 'other' is a UML data
		object, or element wise with a scalar if other is some kind of numeric
		value.

		"""
		return self._genericNumericBinary('__iadd__', other)

	def __sub__(self, other):
		"""
		Subtract from this object, element wise if 'other' is a UML data
		object, or element wise by a scalar if other is some kind of numeric
		value.

		"""
		return self._genericNumericBinary('__sub__', other)

	def __rsub__(self, other):
		"""
		Subtract each element of this object from the given scalar

		"""
		return self._genericNumericBinary('__rsub__', other)

	def __isub__(self, other):
		"""
		Subtract (in place) from this object, element wise if 'other' is a UML data
		object, or element wise with a scalar if other is some kind of numeric
		value. 

		"""
		return self._genericNumericBinary('__isub__', other)

	def __div__(self, other):
		"""
		Perform division using this object as the numerator, element wise if 'other'
		is a UML data object, or element wise by a scalar if other is some kind of
		numeric value.

		"""
		return self._genericNumericBinary('__div__', other)

	def __rdiv__(self, other):
		"""
		Perform element wise division using this object as the denominator, and the
		given scalar value as the numerator

		"""
		return self._genericNumericBinary('__rdiv__', other)

	def __idiv__(self, other):
		"""
		Perform division (in place) using this object as the numerator, element
		wise if 'other' is a UML data object, or element wise by a scalar if other
		is some kind of numeric value.

		"""
		return self._genericNumericBinary('__idiv__', other)

	def __truediv__(self, other):
		"""
		Perform true division using this object as the numerator, element wise
		if 'other' is a UML data object, or element wise by a scalar if other is
		some kind of numeric value.

		"""
		return self._genericNumericBinary('__truediv__', other)

	def __rtruediv__(self, other):
		"""
		Perform element wise true division using this object as the denominator,
		and the given scalar value as the numerator

		"""
		return self._genericNumericBinary('__rtruediv__', other)

	def __itruediv__(self, other):
		"""
		Perform true division (in place) using this object as the numerator, element
		wise if 'other' is a UML data object, or element wise by a scalar if other
		is some kind of numeric value.

		"""
		return self._genericNumericBinary('__itruediv__', other)

	def __floordiv__(self, other):
		"""
		Perform floor division using this object as the numerator, element wise
		if 'other' is a UML data object, or element wise by a scalar if other is
		some kind of numeric value.

		"""
		return self._genericNumericBinary('__floordiv__', other)

	def __rfloordiv__(self, other):
		"""
		Perform element wise floor division using this object as the denominator,
		and the given scalar value as the numerator

		"""
		return self._genericNumericBinary('__rfloordiv__', other)

	def __ifloordiv__(self, other):
		"""
		Perform floor division (in place) using this object as the numerator, element
		wise if 'other' is a UML data object, or element wise by a scalar if other
		is some kind of numeric value.

		"""
		return self._genericNumericBinary('__ifloordiv__', other)

	def __mod__(self, other):
		"""
		Perform mod using the elements of this object as the dividends, element wise
		if 'other' is a UML data object, or element wise by a scalar if other is
		some kind of numeric value.

		"""
		return self._genericNumericBinary('__mod__', other)

	def __rmod__(self, other):
		"""
		Perform mod using the elements of this object as the divisors, and the
		given scalar value as the dividend

		"""
		return self._genericNumericBinary('__rmod__', other)

	def __imod__(self, other):
		"""
		Perform mod (in place) using the elements of this object as the dividends,
		element wise if 'other' is a UML data object, or element wise by a scalar
		if other is some kind of numeric value.

		"""
		return self._genericNumericBinary('__imod__', other)

	def __pow__(self, other):
		"""
		Perform exponentiation (iterated __mul__) using the elements of this object
		as the bases, element wise if 'other' is a UML data object, or element wise
		by a scalar if other is some kind of numeric value.

		"""
		if self.pointCount == 0 or self.featureCount == 0:
			raise ImproperActionException("Cannot do ** when points or features is empty")
		if not dataHelpers._looksNumeric(other):
			raise ArgumentException("'other' must be an instance of a scalar")
		if other != int(other):
			raise ArgumentException("other may only be an integer type")
		if other < 0:
			raise ArgumentException("other must be greater than zero")
	
		retPNames = self.getPointNames()
		retFNames = self.getFeatureNames()

		if other == 1:
			ret = self.copy()
			ret._name = dataHelpers.nextDefaultObjectName()
			return ret

		# exact conditions in which we need to instantiate this object
		if other == 0 or other % 2 == 0:
			identity = UML.createData(self.getTypeString(), numpy.eye(self.pointCount),
				pointNames=retPNames, featureNames=retFNames)
		if other == 0:
			return identity

		# this means that we don't start with a multiplication at the ones place,
		# so we need to reserve the identity as the in progress return value
		if other % 2 == 0:
			ret = identity
		else:
			ret = self.copy()

		# by setting up ret, we've taken care of the original ones place
		curr = other >> 1
		# the running binary exponent we've calculated. We've done the ones
		# place, so this is just a copy
		running = self.copy()

		while curr != 0:
			running = running._matrixMultiply_implementation(running)
			if (curr % 2) == 1:
				ret = ret._matrixMultiply_implementation(running)

			# shift right to put the next digit in the ones place
			curr = curr >> 1

		ret.setPointNames(retPNames)
		ret.setFeatureNames(retFNames)

		ret._name = dataHelpers.nextDefaultObjectName()

		return ret

	def __ipow__(self, other):
		"""
		Perform in-place exponentiation (iterated __mul__) using the elements
		of this object as the bases, element wise if 'other' is a UML data
		object, or element wise by a scalar if other is some kind of numeric
		value.

		"""
		ret = self.__pow__(other)
		self.referenceDataFrom(ret)
		return self

	def __pos__(self):
		""" Return this object. """
		ret = self.copy()
		ret._name = dataHelpers.nextDefaultObjectName()

		return ret

	def __neg__(self):
		""" Return this object where every element has been multiplied by -1 """
		ret = self.copy()
		ret *= -1
		ret._name = dataHelpers.nextDefaultObjectName()

		return ret

	def __abs__(self):
		""" Perform element wise absolute value on this object """
		ret = self.applyToElements(abs, inPlace=False)
		ret.setPointNames(self.getPointNames())
		ret.setFeatureNames(self.getFeatureNames())
		
		ret._name = dataHelpers.nextDefaultObjectName()
		ret._absPath = self.absolutePath
		ret._relPath = self.relativePath
		return ret

	def _genericNumericBinary(self, opName, other):
		nameSource = 'self' if opName.startswith('__i') else None
		pathSource = 'self'

		isUML = isinstance(other, UML.data.Base)
		if not isUML and not dataHelpers._looksNumeric(other):
			raise ArgumentException("'other' must be an instance of a UML data object or a scalar")
		# Test element type self
		if self.pointCount > 0:
			for val in self.pointView(0):
				if not dataHelpers._looksNumeric(val):
					raise ArgumentException("This data object contains non numeric data, cannot do this operation")

		# test element type other
		if isUML:
			if opName.startswith('__r'):
				return NotImplemented
			if other.pointCount > 0:
				for val in other.pointView(0):
					if not dataHelpers._looksNumeric(val):
						raise ArgumentException("This data object contains non numeric data, cannot do this operation")

			if self.pointCount != other.pointCount:
				msg = "The number of points in each object must be equal. "
				msg += "(self=" + str(self.pointCount) + " vs other="
				msg += str(other.pointCount) + ")"
				raise ArgumentException(msg)
			if self.featureCount != other.featureCount:
				raise ArgumentException("The number of features in each object must be equal.")
			pathSource = 'merge'

		if self.pointCount == 0 or self.featureCount == 0:
			raise ImproperActionException("Cannot do " + opName + " when points or features is empty")

		# check name restrictions
		if isUML:
			self._validateEqualNames('point', 'point', opName, other)
			self._validateEqualNames('feature', 'feature', opName, other)

		divNames = ['__div__','__rdiv__','__idiv__','__truediv__','__rtruediv__',
					'__itruediv__','__floordiv__','__rfloordiv__','__ifloordiv__',
					'__mod__','__rmod__','__imod__',]
		if isUML and opName in divNames:
			if other.containsZero():
				raise ZeroDivisionError("Cannot perform " + opName + " when the second argument"
						+ "contains any zeros")
			if isinstance(other, UML.data.Matrix):
				if False in numpy.isfinite(other.data):
					raise ArgumentException("Cannot perform " + opName + " when the second argument"
						+ "contains any NaNs or Infs")
		if not isUML and opName in divNames:
			if other == 0:
				msg = "Cannot perform " + opName + " when the second argument"
				msg += + "is zero"
				raise ZeroDivisionError(msg)

		# figure out return obj's point / feature names
		# if unary:
		if opName in ['__pos__', '__neg__', '__abs__'] or isinstance(other, int):
			retPNames = self.getPointNames()
			retFNames = self.getFeatureNames()
		# else (everything else that uses this helper is a binary scalar op)
		else:
			(retPNames, retFNames) = dataHelpers.mergeNonDefaultNames(self, other)

		startType = self.getTypeString()
		implName = opName[1:] + 'implementation'
		if startType == 'Matrix':
			toCall = getattr(self, implName)
			ret = toCall(other)
		else:
			selfConv = self.copyAs("Matrix")
			toCall = getattr(selfConv, implName)
			ret = toCall(other)
			if opName.startswith('__i'):
				ret = ret.copyAs(startType)
				self.referenceDataFrom(ret)
				ret = self
			else:
				ret = UML.createData(startType, ret.data)

		ret.setPointNames(retPNames)
		ret.setFeatureNames(retFNames)

		dataHelpers.binaryOpNamePathMerge(self, other, ret, nameSource, pathSource)

		return ret

	#################################
	#################################
	###   Statistical functions   ###
	#################################
	#################################


	def pointSimilarities(self, similarityFunction):
		""" """
		return self._axisSimilaritiesBackend(similarityFunction, 'point')

	def featureSimilarities(self, similarityFunction):
		""" """
		return self._axisSimilaritiesBackend(similarityFunction, 'feature')

	def _axisSimilaritiesBackend(self, similarityFunction, axis):
		acceptedPretty = [
			'correlation', 'covariance', 'dot product', 'sample covariance',
			'population covariance'
		]
		accepted = map(dataHelpers.cleanKeywordInput, acceptedPretty)

		msg = "The similarityFunction must be equivaltent to one of the "
		msg += "following: "
		msg += str(acceptedPretty) + ", but '" + str(similarityFunction)
		msg += "' was given instead. Note: casing and whitespace is "
		msg += "ignored when checking the input."

		if not isinstance(similarityFunction, basestring):
			raise ArgumentException(msg)

		cleanFuncName = dataHelpers.cleanKeywordInput(similarityFunction)

		if cleanFuncName not in accepted:
			raise ArgumentException(msg)

		if cleanFuncName == 'correlation':
			toCall = UML.calculate.correlation
		elif cleanFuncName == 'covariance' or cleanFuncName == 'samplecovariance':
			toCall = UML.calculate.covariance
		elif cleanFuncName == 'populationcovariance':
			def populationCovariance(X, X_T):
				return UML.calculate.covariance(X, X_T, False)
			toCall = populationCovariance
		elif cleanFuncName == 'dotproduct':
			def dotProd(X, X_T):
				return X * X_T
			toCall = dotProd

		transposed = self.copy()
		transposed.transpose()

		if axis == 'point':
			ret = toCall(self, transposed)
		else:
			ret = toCall(transposed, self)

		# TODO validation or result.

		ret._absPath = self.absolutePath
		ret._relPath = self.relativePath

		return ret

	def pointStatistics(self, statisticsFunction):
		""" """
		return self._axisStatisticsBackend(statisticsFunction, 'point')

	def featureStatistics(self, statisticsFunction):
		""" """
		return self._axisStatisticsBackend(statisticsFunction, 'feature')

	def _axisStatisticsBackend(self, statisticsFunction, axis):
		acceptedPretty = [
			'max', 'mean', 'median', 'min', 'unique count', 'proportion missing',
			'proportion zero', 'standard deviation', 'std', 'population std',
			'population standard deviation', 'sample std', 
			'sample standard deviation'
			]
		accepted = map(dataHelpers.cleanKeywordInput, acceptedPretty)

		msg = "The statisticsFunction must be equivaltent to one of the "
		msg += "following: "
		msg += str(acceptedPretty) + ", but '" + str(statisticsFunction)
		msg += "' was given instead. Note: casing and whitespace is "
		msg += "ignored when checking the input."

		if not isinstance(statisticsFunction, basestring):
			raise ArgumentException(msg)

		cleanFuncName = dataHelpers.cleanKeywordInput(statisticsFunction)

		if cleanFuncName not in accepted:
			raise ArgumentException(msg)

		if cleanFuncName == 'max':
			toCall = UML.calculate.maximum
		elif cleanFuncName == 'mean':
			toCall = UML.calculate.mean
		elif cleanFuncName == 'median':
			toCall = UML.calculate.median
		elif cleanFuncName == 'min':
			toCall = UML.calculate.minimum
		elif cleanFuncName == 'uniquecount':
			toCall = UML.calculate.uniqueCount
		elif cleanFuncName == 'proportionmissing':
			toCall = UML.calculate.proportionMissing
		elif cleanFuncName == 'proportionzero':
			toCall = UML.calculate.proportionZero
		elif cleanFuncName == 'std' or cleanFuncName == 'standarddeviation':
			def sampleStandardDeviation(values):
				return UML.calculate.standardDeviation(values, True)
			toCall = sampleStandardDeviation
		elif cleanFuncName == 'samplestd' or cleanFuncName == 'samplestandarddeviation':
			def sampleStandardDeviation(values):
				return UML.calculate.standardDeviation(values, True)
			toCall = sampleStandardDeviation
		elif cleanFuncName == 'populationstd' or cleanFuncName == 'populationstandarddeviation':
			toCall = UML.calculate.standardDeviation

		if axis == 'point':
			ret = self.applyToPoints(toCall, inPlace=False)
			ret.setFeatureName(0, cleanFuncName)
		else:
			ret = self.applyToFeatures(toCall, inPlace=False)
			ret.setPointName(0, cleanFuncName)
		return ret

	############################
	############################
	###   Helper functions   ###
	############################
	############################

	def _arrangeFinalTable(self, pnames, pnamesWidth, dataTable, dataWidths,
			fnames, pnameSep):

		if fnames is not None:
			fnamesWidth = map(len, fnames)
		else:
			fnamesWidth = []

		# We make extensive use of list addition in this helper in order
		# to prepend single values onto lists.

		# glue point names onto the left of the data
		if pnames is not None:
			for i in xrange(len(dataTable)):
				dataTable[i] = [pnames[i], pnameSep] + dataTable[i]
			dataWidths = [pnamesWidth, len(pnameSep)] + dataWidths

		# glue feature names onto the top of the data
		if fnames is not None:
			# adjust with the empty space in the upper left corner, if needed
			if pnames is not None:
				fnames = ["", ""] + fnames
				fnamesWidth = [0, 0] + fnamesWidth

			# make gap row:
			gapRow = [""] * len(fnames)

			dataTable = [fnames, gapRow] + dataTable
			# finalize widths by taking the largest of the two possibilities
			for i in xrange(len(fnames)):
				nameWidth = fnamesWidth[i]
				valWidth = dataWidths[i]
				dataWidths[i] = max(nameWidth, valWidth)

		return dataTable, dataWidths

	def _arrangeFeatureNames(self, maxWidth, nameLength, colSep, colHold, nameHold):
		"""Prepare feature names for string output. Grab only those names that
		fit according to the given width limitation, process them for length,
		omit them if they are default. Returns a list of prepared names, and
		a list of the length of each name in the return.

		"""
		colHoldWidth = len(colHold)
		colHoldTotal = len(colSep) + colHoldWidth
		nameCutIndex = nameLength - len(nameHold)

		lNames, rNames = [], []

		# total width will always include the column placeholder column,
		# until it is shown that it isn't needed
		totalWidth = colHoldTotal

		# going to add indices from the beginning and end of the data until
		# we've used up our available space, or we've gone through all of
		# the columns. currIndex makes use of negative indices, which is
		# why the end condition makes use of an exact stop value, which
		# varies between positive and negative depending on the number of
		# features
		endIndex = self.featureCount / 2
		if self.featureCount % 2 == 1:
			endIndex *= -1
			endIndex -= 1
		currIndex = 0
		numAdded = 0
		while totalWidth < maxWidth and currIndex != endIndex:
			nameIndex = currIndex
			if currIndex < 0:
				nameIndex = self.featureCount + currIndex

			currName = self.getFeatureName(nameIndex)

			if currName.startswith(DEFAULT_PREFIX):
				currName = ""
			if len(currName) > nameLength:
				currName = currName[:nameCutIndex] + nameHold
			currWidth = len(currName)

			currNames = lNames if currIndex >= 0 else rNames

			totalWidth += currWidth + len(colSep)
			# test: total width is under max without column holder
			rawStillUnder = totalWidth - (colHoldTotal) < maxWidth
			# test: the column we are trying to add is the last one possible
			allCols = rawStillUnder and (numAdded == (self.featureCount - 1))
			# only add this column if it won't put us over the limit,
			# OR if it is the last one (and under the limit without the col
			# holder)
			if totalWidth < maxWidth or allCols:
				numAdded += 1
				currNames.append(currName)

				# the width value goes in different lists depending on the index
				if currIndex < 0:
					currIndex = abs(currIndex)
				else:
					currIndex = (-1 * currIndex) - 1
			
		# combine the tables. Have to reverse rTable because entries were appended
		# in a right to left order
		rNames.reverse()
		if numAdded == self.featureCount:
			lNames += rNames
		else:
			lNames += [colHold] + rNames

		return lNames

	def _arrangePointNames(self, maxRows, nameLength, rowHolder, nameHold):
		"""Prepare point names for string output. Grab only those names that
		fit according to the given row limitation, process them for length,
		omit them if they are default. Returns a list of prepared names, and
		a int bounding the length of each name representation.

		"""
		names = []
		pnamesWidth = 0
		nameCutIndex = nameLength - len(nameHold)
		(tRowIDs, bRowIDs) = dataHelpers.indicesSplit(maxRows, self.pointCount)

		# we pull indices from two lists: tRowIDs and bRowIDs
		for sourceIndex in range(2):
			source = list([tRowIDs, bRowIDs])[sourceIndex]
			
			# add in the rowHolder, if needed
			if sourceIndex == 1 and len(bRowIDs) + len(tRowIDs) < self.pointCount:
				names.append(rowHolder)

			for i in source:
				pname = self.getPointName(i)
				# omit default valued names
				if pname.startswith(DEFAULT_PREFIX):
					pname = ""
				
				# truncate names which extend past the given length
				if len(pname) > nameLength:
					pname = pname[:nameCutIndex] + nameHold
				
				names.append(pname)

				# keep track of bound.
				if len(pname) > pnamesWidth:
					pnamesWidth = len(pname)
							
		return names, pnamesWidth

	def _arrangeDataWithLimits(self, maxWidth=80, maxHeight=30, sigDigits=3,
				colSep=' ', colHold='--', rowHold='|'):
		"""
		Arrange the data in this object into a table structure, while
		respecting the given boundaries. If there is more data than
		what fits within the limitations, then omit points or features
		from the middle portions of the data.

		Returns a list of list of strings. The length of the outer list
		is less than or equal to maxHeight. The length of the inner lists
		will all be the same, a length we will designate as n. The sum of
		the individual strings in each inner list will be less than or
		equal to maxWidth - ((n-1) * len(colSep)). 

		"""
		if self.pointCount == 0 or self.featureCount == 0:
			return [[]], []

		if maxHeight < 2 and maxHeight != self.pointCount:
			msg = "If the number of points in this object is two or greater, "
			msg += "then we require that the input argument maxHeight also "
			msg += "be greater than or equal to two."
			raise ArgumentException(msg)

		cHoldWidth = len(colHold)
		cHoldTotal = len(colSep) + cHoldWidth

		#setup a bundle of default values
		if maxHeight is None:
			maxHeight = self.pointCount
		if maxWidth is None:
			maxWidth = float('inf')

		maxRows = min(maxHeight, self.pointCount)
		maxDataRows = maxRows

		(tRowIDs, bRowIDs) = dataHelpers.indicesSplit(maxDataRows, self.pointCount)
		combinedRowIDs = tRowIDs + bRowIDs
		if len(combinedRowIDs) < self.pointCount:
			rowHolderIndex = len(tRowIDs)
		else:
			rowHolderIndex = sys.maxint

		lTable, rTable = [], []
		lColWidths, rColWidths = [], []

		# total width will always include the column placeholder column,
		# until it is shown that it isn't needed
		totalWidth = cHoldTotal

		# going to add indices from the beginning and end of the data until
		# we've used up our available space, or we've gone through all of
		# the columns. currIndex makes use of negative indices, which is
		# why the end condition makes use of an exact stop value, which
		# varies between positive and negative depending on the number of
		# features
		endIndex = self.featureCount / 2
		if self.featureCount % 2 == 1:
			endIndex *= -1
			endIndex -= 1
		currIndex = 0
		numAdded = 0
		while totalWidth < maxWidth and currIndex != endIndex:
			currWidth = 0
			currTable = lTable if currIndex >= 0 else rTable
			currCol = []

			# check all values in this column (in the accepted rows)
			for i in range(len(combinedRowIDs)):
				rID = combinedRowIDs[i]
				val = self[rID, currIndex]
				valFormed = formatIfNeeded(val, sigDigits)
				valLen = len(valFormed)
				if valLen > currWidth:
					currWidth = valLen

				# If these are equal, it is time to add the holders
				if i == rowHolderIndex:
					currCol.append(rowHold)
				
				currCol.append(valFormed)

			totalWidth += currWidth + len(colSep)
			# test: total width is under max without column holder
			allCols = totalWidth - (cHoldTotal) < maxWidth
			# test: the column we are trying to add is the last one possible
			allCols = allCols and (numAdded == (self.featureCount - 1))
			# only add this column if it won't put us over the limit
			if totalWidth < maxWidth or allCols:
				numAdded += 1
				for i in range(len(currCol)):
					if len(currTable) != len(currCol):
						currTable.append([currCol[i]])
					else:
						currTable[i].append(currCol[i])

				# the width value goes in different lists depending on the index
				if currIndex < 0:
					currIndex = abs(currIndex)
					rColWidths.append(currWidth)
				else:
					currIndex = (-1 * currIndex) - 1
					lColWidths.append(currWidth)

		# combine the tables. Have to reverse rTable because entries were appended
		# in a right to left order
		rColWidths.reverse()
		if numAdded == self.featureCount:
			lColWidths += rColWidths
		else:
			lColWidths += [cHoldWidth] + rColWidths
		for rowIndex in range(len(lTable)):
			if len(rTable) > 0:
				rTable[rowIndex].reverse()
				toAdd = rTable[rowIndex]
			else:
				toAdd = []

			if numAdded == self.featureCount:
				lTable[rowIndex] += toAdd	
			else:
				lTable[rowIndex] += [colHold] + toAdd

		return lTable, lColWidths


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
		return self._equalNames(self.getPointNames(), other.getPointNames())

	def _equalFeatureNames(self, other):
		if other is None or not isinstance(other, Base):
			return False
		return self._equalNames(self.getFeatureNames(), other.getFeatureNames())

	def _equalNames(self, selfNames, otherNames):
		"""Private function to determine equality of either pointNames of
		featureNames. It ignores equality of default values, considering only
		whether non default names consistent (position by position) and
		uniquely positioned (if a non default name is present in both, then
		it is in the same position in both).

		"""
		if len(selfNames) != len(otherNames):
			return False

		unequalNames = self._unequalNames(selfNames, otherNames)
		return unequalNames == {}

	def _validateEqualNames(self, leftAxis, rightAxis, callSym, other):
		lnames = self.getPointNames() if leftAxis == 'point' else self.getFeatureNames()
		rnames = other.getPointNames() if rightAxis == 'point' else other.getFeatureNames()
		inconsistencies = self._inconsistentNames(lnames, rnames)

		if inconsistencies != {}:
			table = [['left', 'ID', 'right']]
			for i in sorted(inconsistencies.keys()):
				lname = '"' + lnames[i] + '"'
				rname = '"' + rnames[i] + '"'
				table.append([lname, str(i), rname])

			msg = leftAxis + " to " + rightAxis + " name inconsistencies when "
			msg += "calling left." + callSym + "(right) \n"
			msg += UML.logger.tableString.tableString(table)
			print >>sys.stderr, msg
			raise ArgumentException(msg)

	def _inconsistentNames(self, selfNames, otherNames):
		"""Private function to find and return all name inconsistencies
		between the given two sets. It ignores equality of default values,
		considering only whether non default names consistent (position by
		position) and uniquely positioned (if a non default name is present
		in both, then it is in the same position in both). The return value
		is a dict between integer IDs and the pair of offending names at
		that position in both objects.

		Assumptions: the size of the two name sets is equal.

		"""
		inconsistencies = {}

		def checkFromLeftKeys(ret, leftNames, rightNames):
			for index in xrange(len(leftNames)):
				lname = leftNames[index]
				rname = rightNames[index]
				if not lname.startswith(DEFAULT_PREFIX):
					if not rname.startswith(DEFAULT_PREFIX):
						if lname != rname:
							ret[index] = (lname, rname)
					else:
						# if a name in one is mirrored by a default name,
						# then it must not appear in any other index;
						# and therefore, must not appear at all.
						if rightNames.count(lname) > 0:
							ret[index] = (lname, rname)
							ret[rightNames[lname]] = (lname, rname)


		# check both name directions
		checkFromLeftKeys(inconsistencies, selfNames, otherNames)
		checkFromLeftKeys(inconsistencies, otherNames, selfNames)

		return inconsistencies


	def _unequalNames(self, selfNames, otherNames):
		"""Private function to find and return all name inconsistencies
		between the given two sets. It ignores equality of default values,
		considering only whether non default names consistent (position by
		position) and uniquely positioned (if a non default name is present
		in both, then it is in the same position in both). The return value
		is a dict between integer IDs and the pair of offending names at
		that position in both objects.

		Assumptions: the size of the two name sets is equal.

		"""
		inconsistencies = {}

		def checkFromLeftKeys(ret, leftNames, rightNames):
			for index in xrange(len(leftNames)):
				lname = leftNames[index]
				rname = rightNames[index]
				if not lname.startswith(DEFAULT_PREFIX):
					if not rname.startswith(DEFAULT_PREFIX):
						if lname != rname:
							ret[index] = (lname, rname)
					else:
						ret[index] = (lname, rname)

		# check both name directions
		checkFromLeftKeys(inconsistencies, selfNames, otherNames)
		checkFromLeftKeys(inconsistencies, otherNames, selfNames)

		return inconsistencies


	def _getPointIndex(self, identifier):
		return self._getIndex(identifier, 'point')

	def _getFeatureIndex(self, identifier):
		return self._getIndex(identifier, 'feature')

	def _getIndex(self, identifier, axis):
		names = self.getPointNames() if axis == 'point' else self.getFeatureNames()
		nameGetter = self.getPointIndex if axis == 'point' else self.getFeatureIndex
		
		toReturn = identifier
		if len(names) == 0:
			msg = "There are no valid " + axis + " identifiers; this object has 0 "
			msg += axis + "s"
			raise ArgumentException(msg)
		if identifier is None:
			msg = "An identifier cannot be None."
			raise ArgumentException(msg)
		if (not isinstance(identifier,basestring)) and (not isinstance(identifier,int)):
			axisCount = self.pointCount if axis == 'point' else self.featureCount
			msg = "The identifier must be either a string (a valid " + axis
			msg += " name) or an integer index between 0 and " + str(axisCount-1) 
			msg += " inclusive"
			raise ArgumentException(msg)
		if isinstance(identifier,int):
			if identifier < 0:
				identifier = len(names) + identifier
				toReturn = identifier
			if identifier < 0 or identifier >= len(names):
				msg = "The given index " + str(identifier) + " is outside of the range "
				msg += "of possible indices in the " + axis + " axis (0 to " 
				msg += str(len(names)-1) + ")."
				raise ArgumentException(msg)
		if isinstance(identifier,basestring):
			try:
				toReturn = nameGetter(identifier)
			except KeyError:
				msg = "The " + axis + " name '" + identifier + "' cannot be found."
				raise ArgumentException(msg)
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
		self._addName(pointName, self.pointNames, self.pointNamesInverse, 'point')

	def _addFeatureName(self, featureName):
		self._addName(featureName, self.featureNames, self.featureNamesInverse, 'feature')

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
		self._removeNameAndShift(toRemove, 'point', self.pointNames, self.pointNamesInverse)

	def _removeFeatureNameAndShift(self, toRemove):
		"""
		Removes the specified name from featureNames, changing the indices
		of other featureNames to fill in the missing index.

		toRemove must be a non None string or integer, specifying either a current featureNames
		or the index of a current featureNames in the given axis.
		
		"""
		self._removeNameAndShift(toRemove, 'feature', self.featureNames, self.featureNamesInverse)

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
		index = self._getIndex(toRemove, axis)
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
			return
		if not hasattr(assignments, '__getitem__') or not hasattr(assignments, '__len__'):
			msg = "assignments may only be an ordered container type, with "
			msg += "implentations for both __len__ and __getitem__, where "
			msg += "__getitem__ accepts non-negative integers"
			raise ArgumentException(msg)
		if count == 0:
			if len(assignments) > 0:
				msg = "assignments is too large (" + str(len(assignments))
				msg += "); this axis is empty"
				raise ArgumentException(msg)
			return		
		if len(assignments) != count:
			msg = "assignments may only be an ordered container type, with as "
			msg += "many entries (" + str(len(assignments)) + ") as this axis "
			msg += "is long (" + str(count) + ")"
			raise ArgumentException(msg)
		try:
			assignments[0]
		except IndexError:
			msg = "assignments may only be an ordered container type, with "
			msg += "implentations for both __len__ and __getitem__, where "
			msg += "__getitem__ accepts non-negative integers"
			raise ArgumentException(msg)

		#convert to dict so we only write the checking code once
		temp = {}
		for index in xrange(len(assignments)):
			name = assignments[index]
			# take this to mean fill it in with a default name
			if name is None:
				name = self._nextDefaultName(axis)
			if name in temp:
				raise ArgumentException("Cannot input duplicate names: " + str(name))
			temp[name] = index
		assignments = temp

		self._setNamesFromDict(assignments, count, axis)

	def _setNamesFromDict(self, assignments, count, axis):
		self._validateAxis(axis)
		if assignments is None:
			self._setAllDefault(axis)
			return
		if not isinstance(assignments, dict):
			raise ArgumentException("assignments may only be a dict, with as many entries as this axis is long")
		if count == 0:
			if len(assignments) > 0:
				raise ArgumentException("assignments is too large; this axis is empty ")
			if axis == 'point':
				self.pointNames = {}
				self.pointNamesInverse = {}
			else:
				self.featureNames = {}
				self.featureNamesInverse = {}
			return
		if len(assignments) != count:
			raise ArgumentException("assignments may only be a dict, with as many entries as this axis is long")

		# at this point, the input must be a dict
		#check input before performing any action
		for name in assignments.keys():
			if not None and not isinstance(name, basestring):
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
			self.pointNames = copy.deepcopy(assignments)
			self.pointNamesInverse = reverseDict
		else:
			self.featureNames = copy.deepcopy(assignments)
			self.featureNamesInverse = reverseDict


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


	def _validateValueIsNotNone(self, name, value):
		if value is None:
			msg = "The argument named " + name + " must not have a value of None"
			raise ArgumentException(msg)

	def _validateValueIsUMLDataObject(self, name, value):
		if not isinstance(value, UML.data.Base):
			msg = "The argument named " + name + " must be an instance "
			msg += "of the UML.data.Base class. The value we recieved was "
			msg += str(value) + ", had the type " + str(type(value)) 
			msg += ", and a method resolution order of "
			msg += str(inspect.getmro(value.__class__))
			raise ArgumentException(msg)

	def _shapeCompareString(self, argName, argValue):
		selfPoints = self.pointCount
		sps = "" if selfPoints == 1 else "s"
		selfFeats = self.featureCount
		sfs = "" if selfFeats == 1 else "s"
		argPoints = argValue.pointCount
		aps = "" if argPoints == 1 else "s"
		argFeats = argValue.featureCount
		afs = "" if argFeats == 1 else "s"

		ret = "Yet, " + argName + " has "
		ret += str(argPoints) + " point" + aps + " and "
		ret += str(argFeats) + " feature" + afs + " "
		ret += "while the caller has " 
		ret += str(selfPoints) + " point" + sps + " and "
		ret += str(selfFeats) + " feature" + sfs + "."

		return ret

	def _validateObjHasSameNumberOfFeatures(self, argName, argValue):
		selfFeats = self.featureCount
		argFeats = argValue.featureCount

		if selfFeats != argFeats:
			msg = "The argument named " + argName + " must have the same number "
			msg += "of features as the caller object. "
			msg += self._shapeCompareString(argName, argValue)
			raise ArgumentException(msg)

	def _validateObjHasSameNumberOfPoints(self, argName, argValue):
		selfPoints = self.pointCount
		argValuePoints = argValue.pointCount
		if selfPoints != argValuePoints:
			msg = "The argument named " + argName + " must have the same number "
			msg += "of points as the caller object. "
			msg += self._shapeCompareString(argName, argValue)
			raise ArgumentException(msg)

	def _validateEmptyNamesIntersection(self, axis, argName, argValue):
		if axis == 'point':
			intersection = self._pointNameIntersection(argValue)
			nString = 'pointNames'
		elif axis == 'feature':
			intersection = self._featureNameIntersection(argValue)
			nString = 'featureNames'
		else:
			raise ArgumentException("invalid axis")

		shared = []
		if intersection:
			for name in intersection:
				if not name.startswith(DEFAULT_PREFIX):
					shared.append(name)

		if shared != []:
			truncated = False
			if len(shared) > 10:
				full = len(shared)
				shared = shared[:10]
				truncated = True

			msg = "The argument named " + argName + " must not share any "
			msg += nString + " with the calling object, yet the following "
			msg += "names occured in both: "
			msg += UML.exceptions.prettyListString(shared)
			if truncated:
				msg += "... (only first 10 entries out of " + str(full)
				msg += " total)"
			raise ArgumentException(msg)

	def _validateMatPlotLibImport(self, error, name):
		if error is not None:
			msg = "The module matplotlib is required to be installed "
			msg += "in order to call the " + name + "() method. "
			msg += "However, when trying to import, an ImportError with "
			msg += "the following message was raised: '"
			msg += str(error) + "'"

			raise ImportError(msg)
