"""
Anchors the hierarchy of data representation types, providing stubs and common functions.

"""

# TODO conversions
# TODO who sorts inputs to derived implementations?

from copy import copy
from copy import deepcopy
from ..utility.custom_exceptions import ArgumentException
import UML
from UML.uml_logging.data_set_analyzer import produceFeaturewiseReport
from UML.uml_logging.data_set_analyzer import produceAggregateReport
import random
import math
from abc import ABCMeta
from abc import abstractmethod

# a default seed for testing and predictible trials
DEFAULT_SEED = 'DEFAULTSEED'

# the prefix for default featureNames
DEFAULT_PREFIX = "_DEFAULT_#"


class BaseData(object):
	"""
	Class defining important data manipulation operations and giving functionality
	for the naming the features of that data. A mapping from feature names to feature
	indices is given by the featureNames attribute, the inverse of that mapping is
	given by featureNamesInverse.

	"""

	def __init__(self, featureNames=None, name=None, path=None):
		"""
		Instantiates the featureName book-keeping structures that are defined by this representation.
		
		featureNames may be None if the object is to have default names, or a list or dict defining the
		featureName mapping.

		"""
		super(BaseData, self).__init__()
		self._nextDefaultValue = 0
		self._setAllDefault()
		self._renameMultipleFeatureNames_implementation(featureNames,True)
		if featureNames is not None and len(featureNames) != self.features():
			raise ArgumentException("Cannot have different number of featureNames and features, len(featureNames): " + str(len(featureNames)) + ", self.features(): " + str(self.features()))
		self.name = name
		self.path = path

	########################
	# Low Level Operations #
	########################

	def featureNameDifference(self, other):
		"""
		Returns a set containing those featureNames in this object that are not also in the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, BaseData):
			raise ArgumentException("Must provide another representation type to determine featureName difference")
		
		return self.featureNames.viewkeys() - other.featureNames.viewkeys() 

	def featureNameIntersection(self, other):
		"""
		Returns a set containing only those featureNames that are shared by this object and the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, BaseData):
			raise ArgumentException("Must provide another representation type to determine featureName intersection")
		
		return self.featureNames.viewkeys() & other.featureNames.viewkeys() 

	def featureNameSymmetricDifference(self, other):
		"""
		Returns a set containing only those featureNames not shared between this object and the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, BaseData):
			raise ArgumentException("Must provide another representation type to determine featureName difference")
		
		return self.featureNames.viewkeys() ^ other.featureNames.viewkeys() 

	def featureNameUnion(self,other):
		"""
		Returns a set containing all featureNames in either this object or the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, BaseData):
			raise ArgumentException("Must provide another representation type to determine featureName union")
		
		return self.featureNames.viewkeys() | other.featureNames.viewkeys() 


	def renameFeatureName(self, oldIdentifier, newFeatureName):
		"""
		Changes the featureName specified by previous to the supplied input name.
		
		oldIdentifier must be a non None string or integer, specifying either a current featureName
		or the index of a current featureName. newName may be either a string not currently
		in the featureName set, or None for an default featureName. newName cannot begin with the
		default prefix.

		"""
		self._renameFeatureName_implementation(oldIdentifier, newFeatureName, False)

	def renameMultipleFeatureNames(self, assignments=None):
		"""
		Rename some portion of the featureName set according to the input. 

		assignments may be either a list or dict specifying new names, or None
		to set all featureNames to new default values. If assignment is any other type, or
		if the names are not strings, the names are not unique, the feature
		indices are not integers, or the names begin with the default prefix,
		then an ArgumentException will be raised.

		"""
		self._renameMultipleFeatureNames_implementation(assignments,False)

	def setName(self, name):
		"""
		Copy over the name attribute of this object with the input name

		"""
		if name is not None and not isinstance(name, basestring):
			raise ArgumentException("name must be a string")
		self.name = name

	###########################
	# Higher Order Operations #
	###########################

	def _joinUniqueKeyedOther(self, other, fillUnmatched):
		# other must be data rep obj
		# there must be overlap
		# cannot be complete overlap
		
		# from overlap generate keys from other
		#	- must be unique

		# determine how many features of new data there will be

		# for each new feature
		# apply to each point in self with a lookup function into other

		# .... how do we do lookup?
		# we have all the feature numbers, so its ok.
		#TODO
		raise NotImplementedError

	
	def dropStringValuedFeatures(self):
		"""
		Modify this object so that it no longer contains features which have strings
		as values

		"""
		if self.features() == 0:
			return

		def hasStrings(feature):
			for value in feature:
				if isinstance(value, basestring):
					return True
			return False
	
		self.extractFeatures(hasStrings)


	def featureToBinaryCategoryFeatures(self, featureToConvert):
		"""
		Modify this object so that the choosen feature is removed, and binary valued
		features are added, one for each possible value seen in the original feature.

		"""
		index = self._getIndex(featureToConvert)

		# extract col.
		toConvert = self.extractFeatures([index])

		# MR to get list of values
		def getValue(point):
			return [(point[0],1)]
		def simpleReducer(identifier, valuesList):
			return (identifier,0)

		values = toConvert.mapReduceOnPoints(getValue, simpleReducer)
		values.renameFeatureName(0,'values')
		values = values.extractFeatures([0])

		# Convert to RLD, so we can have easy access
		values = values.toRowListData()

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
			ret = toConvert.applyFunctionToEachPoint(makeFunc(value))
			ret.renameFeatureName(0, varName + "=" + str(value).strip())
			toConvert.appendFeatures(ret)

		# remove the original feature, and combine with self
		toConvert.extractFeatures([varName])
		self.appendFeatures(toConvert)


	def featureToIntegerCategories(self, featureToConvert):
		"""
		Modify this object so that the chosen feature in removed, and a new integer
		valued feature is added with values 0 to n-1, one for each of n values present
		in the original feature. 

		"""
		index = self._getIndex(featureToConvert)

		# extract col.
		toConvert = self.extractFeatures([index])

		# MR to get list of values
		def getValue(point):
			return [(point[0],1)]
		def simpleReducer(identifier, valuesList):
			return (identifier,0)

		values = toConvert.mapReduceOnPoints(getValue, simpleReducer)
		values.renameFeatureName(0,'values')
		values = values.extractFeatures([0])

		# Convert to RLD, so we can have easy access
		values = values.toRowListData()

		mapping = {}
		index = 0
		for point in values.data:
			if point[0] not in mapping:
				mapping[point[0]] = index
				index = index + 1

		# use apply to each to make new feature with the int mappings
		def lookup(point):
			return mapping[point[0]]

		converted = toConvert.applyFunctionToEachPoint(lookup)
		converted.renameFeatureName(0,toConvert.featureNamesInverse[0])		

		self.appendFeatures(converted)


	def _selectConstantOfPointsByValue(self, numToSelect, featureToSelectOver, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of points from
		this object, with the sample limited to a constant number of points
		of each representative value in the specifed feature. Those selected
		values are also removed from this object.

		"""
		random.seed(seed)
		if numToSelect is None:
			raise ArgumentException("The selection constant must not be None")
		if numToSelect <= 0:
			raise ArgumentException("The selection constant must be positive")
		index = self._getIndex(featureToSelectOver)
		#TODO
		raise NotImplementedError


	def _selectPercentOfPointsByValue(self, percentToSelect, featureToSelectOver, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of points from
		this object, with the sample limited to a percentage of points
		of each representative value in the specified feature. Those selected
		values are also removed from this object.

		"""
		random.seed(seed)
		if percentToSelect is None:
			raise ArgumentException("percentToSelect must not be none")
		if percentToSelect <= 0:
			raise ArgumentException("percentToSelect must be greater than 0")
		if percentToSelect >= 100:
			raise ArgumentException("percentToSelect must be less than 100")
		index = self._getIndex(featureToSelectOver)
		#TODO
		raise NotImplementedError

		#MR to find how many of each value
		def mapperCount(point):
			return [(point[index],1)]
		def reducerCount(identifier, values):
			total = 0
			for value in values:
				total += value
			return (identifier, total)
		
#		totalOfEach = self.mapReduceOnPoints(mapperCount,reducerCount)
			

		valueToTotal = {}
		def tagValuesWithID(point):
			key = point[index]
			if key not in valueToTotal:
				valueToTotal[key] = 0
				return 0
			ret = valueToTotal[key]
			valueToTotal[key] += 1
			return ret

#		ids = self.applyFunctionToEachPoint(tagValuesWithID)
#		self.appendFeature(ids)

	
		#apply to all to number the different values		
		#	the storage struc is outside the func, ie we have access to it here
		#apply to all using above to mark selected

	def toListOfLists(self):
		"""
			Extract this object's data and return it as a list of lists, with 
			featureNames as the first row in the list.  If featureNames is blank,
			insert a blank list in the first position of the returned list.
		"""
		rowListContainer = self.toRowListData()
		return rowListContainer.data

	def extractPointsByCoinToss(self, extractionProbability, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of points
		from this object, where a random experient is performed for each
		point, with the chance of selection equal to the extractionProbabilty
		parameter. Those selected values are also removed from this object.

		"""
		if self.points() == 0:
			raise ArgumentException("Cannot extract points from an object with 0 points")

		random.seed(seed)
		if extractionProbability is None:
			raise ArgumentException("Must provide a extractionProbability")
		if extractionProbability <= 0:
			raise ArgumentException("extractionProbability must be greater than zero")
		if extractionProbability >= 1:
			raise ArgumentException("extractionProbability must be less than one")

		def experiment(point):
			return bool(random.random() < extractionProbability)

		def isSelected(point):
			return point[len(point)-1]

		selectionKeys = self.applyFunctionToEachPoint(experiment)
		self.appendFeatures(selectionKeys)
		ret = self.extractPoints(isSelected)
		# remove the experimental data
		if ret.points() > 0:
			ret.extractFeatures([ret.features()-1])
		if self.points() > 0:
			self.extractFeatures([self.features()-1])
		
		return ret


	def foldIterator(self, numFolds, seed=DEFAULT_SEED):
		"""
		Returns an iterator object that iterates through folds of this object

		"""
		# note: we want truncation here
		numInFold = int(self.points() / numFolds)
		if numInFold == 0:
			raise ArgumentException("Must specifiy few enough folds so there is a point in each")

		# randomly select the folded portions
		indices = range(self.points())
		random.seed(seed)
		random.shuffle(indices)
		foldList = []
		for fold in xrange(numFolds):
			start = fold * numInFold
			if fold == numFolds - 1:
				end = self.points()
			else:
				end = (fold + 1) * numInFold
			thisFoldList = indices[start:end]
			thisFoldList.sort()
			foldList.append(thisFoldList)

		# return that lists iterator as the fold iterator 	
		return self._foldIteratorClass(foldList, self)

	def applyFunctionToEachPoint(self, function):
		"""
		Applies the given funciton to each point in this object, collecting the
		output values into a new object that is returned upon completion.

		function must not be none and accept a point as an argument

		"""
		if self.points() == 0:
			raise ArgumentException("Function not callable if there are 0 points of data")
		if function is None:
			raise ArgumentException("function must not be None")
		retData = []
		for point in self.pointViewIterator():
			currOut = function(point)
			retData.append([currOut])
		return UML.data(self.getType(), retData)

	def applyFunctionToEachFeature(self,function):
		"""
		Applies the given funciton to each feature in this object, collecting the
		output values into a new object in the shape of a feature vector that is
		returned upon completion.

		function must not be none and accept a feature as an argument

		"""
		if self.features() == 0:
			raise ArgumentException("Function not callable if there are 0 features of data")
		if function is None:
			raise ArgumentException("function must not be None")
		retData = [[]]
		for feature in self.featureViewIterator():
			currOut = function(feature)
			retData[0].append(currOut)
		return UML.data(self.getType(), retData)


	def mapReduceOnPoints(self, mapper, reducer):
#		if self.points() == 0 or self.features() == 0:
#
		if mapper is None or reducer is None:
			raise ArgumentException("The arguments must not be none")
		if not hasattr(mapper, '__call__'):
			raise ArgumentException("The mapper must be callable")
		if not hasattr(reducer, '__call__'):
			raise ArgumentException("The reducer must be callable")

		mapResults = {}
		# apply the mapper to each point in the data
		for point in self.pointViewIterator():
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
		return UML.data(self.getType(), ret)

	def pointViewIterator(self):
		class pointIt():
			def __init__(self, outer):
				self._outer = outer
				self._position = 0
			def __iter__(self):
				return self
			def next(self):
				while (self._position < self._outer.points()):
					value = self._outer.getPointView(self._position)
					self._position += 1
					return value
				raise StopIteration
		return pointIt(self)

	def featureViewIterator(self):
		class featureIt():
			def __init__(self, outer):
				self._outer = outer
				self._position = 0
			def __iter__(self):
				return self
			def next(self):
				while (self._position < self._outer.features()):
					value = self._outer.getFeatureView(self._position)
					self._position += 1
					return value
				raise StopIteration
		return featureIt(self)

	def transformPoint(self, point, function):
		"""
		Modifies this object so that the specified point is replaced with the results
		of passing each value of the original point to the input function.

		"""
		if self.features() == 0:
			raise ArgumentException("Cannot transform points in a data object with no features")
		if point is None or function is None:
			raise ArgumentException("point and function must not be None")
		if not isinstance(point, int):
			raise ArgumentException("point must be the integer index of the point to modify")

		currView = self.getPointView(point)

		for x in xrange(self.features()):
			currValue = currView[x]
			result = function(currValue)
			currView[x] = result


	def transformFeature(self, feature, function):
		"""
		Modifies this object so that the specified feature is replaced with the results
		of passing each value of the original feature to the input function.

		"""
		if self.points() == 0:
			raise ArgumentException("Cannot transform features in a data object with no points")
		if feature is None or function is None:
			raise ArgumentException("feature and function must not be None")
		index = self._getIndex(feature)

		currView = self.getFeatureView(index)

		for x in xrange(self.points()):
			currValue = currView[x]
			result = function(currValue)
			currView[x] = result


	def computeListOfValuesFromElements(self, valueFunction, skipZeros=False, excludeNoneResultValues=False):
		"""Applies the valueFunction(elementValue) or valueFunction(elementValue, pointNum, featureNum) 
		to each element, and returns a list of the resulting values. 
		If skipZeros=True it does not apply valueFunction to elements in the dataMatrix that are 0.
		If excludeNoneResultValues=True, any time valueFunction() returns None, that None value will not be
		added to the list of resulting values that is returned.
		"""
		oneArg = False
		try:
			valueFunction(0,0,0)
		except TypeError:
			oneArg = True

		valueList = []
		for currPoint in self.pointViewIterator():
			currPointID = currPoint.index()
			for j in xrange(len(currPoint)):
				value = currPoint[j]
				if skipZeros and value == 0:
					continue	
				if oneArg:
					currRet = valueFunction(value)
				else:
					currRet = valueFunction(value, currPointID, j)
				if not (excludeNoneResultValues and currRet is None):
					valueList.append(currRet)

		return valueList

	def hashCode(self):
		"""returns a hash for this matrix, which is a number x in the range 0<= x < 1 billion
		that should almost always change when the values of the matrix are changed by a substantive amount"""
		valueList = self.computeListOfValuesFromElements(lambda elementValue, pointNum, featureNum: ((math.sin(pointNum) + math.cos(featureNum))/2.0) * elementValue, skipZeros=True)
		avg = sum(valueList)/float(self.points()*self.features())
		bigNum = 1000000000
		#this should return an integer x in the range 0<= x < 1 billion
		return int(int(round(bigNum*avg)) % bigNum)	


	def isApproxEquivalent(self, otherDataMatrix):
		"""If it returns False, this DataMatrix and otherDataMatrix definitely don't store equivalent data. 
		If it returns True, they probably do but you can't be absolutely sure.
		Note that only the actual data stored is considered, it doesn't matter whether the data matrix objects 
		passed are of the same type (dense, sparse, etc.)"""
		#first check to make sure they have the same number of rows and columns
		if self.points() != otherDataMatrix.points(): return False
		if self.features() != otherDataMatrix.features(): return False
		#now check if the hashes of each matrix are the same
		if self.hashCode() != otherDataMatrix.hashCode(): return False
		return True


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

	def report(self, displayDigits=2):
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
		self.renameMultipleFeatureNames(None)

	def appendPoints(self, toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object.

		toAppend cannot be None, and must be a kind of data representation object with the same
		number of features as the calling object.
		
		"""
		if toAppend is None:
			raise ArgumentException("toAppend must not be None")
		if not isinstance(toAppend,BaseData):
			raise ArgumentException("toAppend must be a kind of data representation object")
		if not self.features() == toAppend.features():
			raise ArgumentException("toAppend must have the same number of features as this object")
		self._appendPoints_implementation(toAppend)
		
	def appendFeatures(self, toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		toAppend cannot be None, must be a kind of data representation object with the same
		number of points as the calling object, and must not share any feature names with the calling
		object.
		
		"""	
		if toAppend is None:
			raise ArgumentException("toAppend must not be None")
		if not isinstance(toAppend,BaseData):
			raise ArgumentException("toAppend must be a kind of data representation object")
		if not self.points() == toAppend.points():
			raise ArgumentException("toAppend must have the same number of points as this object")
		if self.featureNameIntersection(toAppend):
			raise ArgumentException("toAppend must not share any featureNames with this object")
		self._appendFeatures_implementation(toAppend)

		for i in xrange(toAppend.features()):
			self._addFeatureName(toAppend.featureNamesInverse[i])

	def sortPoints(self, sortBy=None, sortHelper=None):
		""" 
		Modify this object so that the points are sorted in place, where sortBy may
		indicate the feature to sort by or None if the entire point is to be taken as a key,
		sortHelper may either be comparator, a scoring function, or None to indicate the natural
		ordering.
		"""
		# its already sorted in these cases
		if self.features() == 0 or self.points() == 0:
			return

		sortByIndex = sortBy
		if sortBy is not None:
			sortByIndex = self._getIndex(sortBy)
			if sortHelper is not None:
				raise ArgumentException("Cannot specify a feature to sort by and a helper function")
		else:
			if sortHelper is None:
				raise ArgumentException("Either sortBy or sortHelper must not be None")

		self._sortPoints_implementation(sortByIndex, sortHelper)

	def sortFeatures(self, sortBy=None, sortHelper=None):
		""" 
		Modify this object so that the features are sorted in place, where sortBy may
		indicate the feature to sort by or None if the entire point is to be taken as a key,
		sortHelper may either be comparator, a scoring function, or None to indicate the natural
		ordering.

		"""
		# its already sorted in these cases
		if self.features() == 0 or self.points() == 0:
			return
		newFeatureNameOrder = self._sortFeatures_implementation(sortBy, sortHelper)
		self._renameMultipleFeatureNames_implementation(newFeatureNameOrder,True)


	def extractPoints(self, toExtract=None, start=None, end=None, number=None, randomize=False):
		"""
		Modify this object, removing those points that are specified by the input, and returning
		an object containing those removed points.

		toExtract may be a single identifier, a list of identifiers, or a function that when
		given a point will return True if it is to be removed. number is the quantity of points that
		we are to be extracted, the default None means unlimited extracttion. start and end are
		parameters indicating range based extraction: if range based extraction is employed,
		toExtract must be None, and vice versa. If only one of start and end are non-None, the
		other defaults to 0 and self.points() respectibly. randomize indicates whether random
		sampling is to be used in conjunction with the number parameter, if randomize is False,
		the chosen points are determined by point order, otherwise it is uniform random across the
		space of possible removals.

		"""
		if self.points() == 0:
			raise ArgumentException("Cannot extract points from an object with 0 points")

		if toExtract is not None:
			if start is not None or end is not None:
				raise ArgumentException("Range removal is exclusive, to use it, toExtract must be None")
		else:
			if start is None:
				start = 0
			if end is None:
				end = self.points() - 1
			if start < 0 or start > self.points():
				raise ArgumentException("start must be a valid index, in the range of possible points")
			if end < 0 or end > self.points():
				raise ArgumentException("end must be a valid index, in the range of possible points")
			if start > end:
				raise ArgumentException("start cannot be an index greater than end")
			if not randomize and number is not None:
				#then we can do the windowing calculation here
				possibleEnd = start + number -1
				if possibleEnd < end:
					end = possibleEnd

		ret = self._extractPoints_implementation(toExtract, start, end, number, randomize)
		ret._renameMultipleFeatureNames_implementation(self.featureNames,True)
		return ret

	def extractFeatures(self, toExtract=None, start=None, end=None, number=None, randomize=False):
		"""
		Modify this object, removing those features that are specified by the input, and returning
		an object containing those removed features. This particular function only does argument
		checking and modifying the featureNames for this object. It is the job of helper functions in
		the derived class to perform the removal and assign featureNames for the returned object.

		toExtract may be a single identifier, a list of identifiers, or a function that when
		given a feature will return True if it is to be removed. number is the quantity of features that
		are to be extracted, the default None means unlimited extracttion. start and end are
		parameters indicating range based extraction: if range based extraction is employed,
		toExtract must be None, and vice versa. If only one of start and end are non-None, the
		other defaults to 0 and self.features() respectibly. randomize indicates whether random
		sampling is to be used in conjunction with the number parameter, if randomize is False,
		the chosen features are determined by feature order, otherwise it is uniform random across the
		space of possible removals.

		"""
		if self.features() == 0:
			raise ArgumentException("Cannot extract features from an object with 0 features")

		if toExtract is not None:
			if start is not None or end is not None:
				raise ArgumentException("Range removal is exclusive, to use it, toExtract must be None")
		elif start is not None or end is not None:
			if start is None:
				start = 0
			if end is None:
				end = self.features() - 1
			if start < 0 or start > self.features():
				raise ArgumentException("start must be a valid index, in the range of possible features")
			if end < 0 or end > self.features():
				raise ArgumentException("end must be a valid index, in the range of possible features")
			if start > end:
				raise ArgumentException("start cannot be an index greater than end")
			if not randomize and number is not None:
				#then we can do the windowing calculation here
				possibleEnd = start + number -1
				if possibleEnd < end:
					end = possibleEnd

		ret = self._extractFeatures_implementation(toExtract, start, end, number, randomize)
		for key in ret.featureNames.keys():
			self._removeFeatureNameAndShift(key)
		return ret

	def equals(self, other):
		if not self._equalFeatureNames(other):
			return False

		return self._equals_implementation(other)

	def points(self):
		return self._points_implementation()

	def features(self):
		return self._features_implementation()

	def toRowListData(self):
		return self._toRowListData_implementation()

	def toDenseMatrixData(self):
		return self._toDenseMatrixData_implementation()

	def writeFile(self, extension, outPath, includeFeatureNames):
		"""
		Funciton to write the data in this object to a file with the choosen
		extension. outPath is the location where we want to write the output file.
		includeFeatureNames is boolean argument indicating whether the file should
		start with a comment line designating featureNames.

		"""
		if extension.lower() == "csv":
			return self._writeFileCSV_implementation(outPath, includeFeatureNames)
		elif extension.lower() == "mtx":
			return self._writeFileMTX_implementation(outPath, includeFeatureNames)
		else:
			raise ArgumentException("Unrecognized file extension")

	def copyReferences(self, toCopy):
		"""
		Modifies the internal data of this object to refer to the same data as other. In other
		words, the data wrapped by both the self and other objects resides in the
		same place in memory. Other must be an object of the same type as
		the calling object. Also, the shape of other should be consistent with the set
		of featureNames currently in this object

		"""
		# this is called first because it checks the data type
		self._copyReferences_implementation(toCopy)
		self.featureNames = toCopy.featureNames
		self.featureNamesInverse = toCopy.featureNamesInverse


	def duplicate(self):
		"""
		Return a new object which has the same data and featureNames as this object

		"""
		return self._duplicate_implementation()

	def copyPoints(self, points=None, start=None, end=None):
		"""
		Return a new object which consists only of those specified points, without mutating
		this object.
		
		"""
		if isinstance(points, basestring):
			points = [points]
		if points is None:
			if start is not None or end is not None:
				if start is None:
					start = 0
				if end is None:
					end = self.points() - 1
				if start < 0 or start > self.points():
					raise ArgumentException("start must be a valid index, in the range of possible features")
				if end < 0 or end > self.points():
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
				if index < 0 or index >= self.points():
					raise ArgumentException("input must contain only valid indices")

		retObj = self._copyPoints_implementation(points, start, end)
		retObj._renameMultipleFeatureNames_implementation(self.featureNames,True)
		return retObj
	
	def copyFeatures(self, features=None, start=None, end=None):
		"""
		Return a new object which consists only of those specified features, without mutating
		this object.
		
		"""
		if isinstance(features, basestring):
			features = [features]
		indices = None
		if features is None:
			if start is not None or end is not None:
				if start is None:
					start = 0
				if end is None:
					end = self.features() - 1
				if start < 0 or start > self.features():
					raise ArgumentException("start must be a valid index, in the range of possible features")
				if end < 0 or end > self.features():
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
				indices.append(self._getIndex(identifier))

		return self._copyFeatures_implementation(indices, start, end)

	def getType(self):
		"""
			Return a string representing the non-abstract type of this object (e.g. DenseMatrixData,
			CooSparseData, etc.) that can be passed to data() function to create a new object
			of the same type.
		"""
		return self._getType_implementation()

	def __getitem__(self, key):
		try:
			(x,y) = key
		except TypeError:
			raise ArgumentException("Must include a point and feature index")
		if not isinstance(x,int) or x < 0 or x >= self.points():
			raise ArgumentException(str(x) + " is not a valid point ID")

		if isinstance(y,basestring):
			y = self._getIndex(y)

		if not isinstance(y,int) or y < 0 or y >= self.features():
			raise ArgumentException(str(y) + " is not a valid feature ID")

		return self._getitem_implementation(x,y)

	def getPointView(self, ID):
		"""
		Returns a View object into the data of the point with the given ID. See View object
		comments for its capabilities. This View is only valid until the next modification
		to the shape or ordering of the internal data. After such a modification, there is
		no guarantee to the validity of the results.
		"""
		if not isinstance(ID,int):
			raise ArgumentException("Point IDs must be integers")
		return self._getPointView_implementation(ID)

	def getFeatureView(self, ID):
		"""
		Returns a View object into the data of the point with the given ID. See View object
		comments for its capabilities. This View is only valid until the next modification
		to the shape or ordering of the internal data. After such a modification, there is
		no guarantee to the validity of the results.
		"""
		index = self._getIndex(ID)
		return self._getFeatureView_implementation(index)
	

	####################
	# Helper functions #
	####################

	def _equalFeatureNames(self, other):
		"""
		Private function to determine equality of BaseData featureNames. It ignores
		equality of default values, though if default values are present,
		the number of variables and their indices must match up.

		"""
		if other is None:
			return False
		if not isinstance(other,BaseData):
			return False	
		if len(self.featureNames) != len(other.featureNames):
			return False
		if len(self.featureNamesInverse) != len(other.featureNamesInverse):
			return False
		# check both featureName directions
		for featureName in self.featureNames.keys():
			if not featureName.startswith(DEFAULT_PREFIX) and featureName not in other.featureNames:
				return False
			if not featureName.startswith(DEFAULT_PREFIX) and self.featureNames[featureName] != other.featureNames[featureName]:
				return False
		for index in self.featureNamesInverse.keys():
			if index not in other.featureNamesInverse:
				return False
			if not self.featureNamesInverse[index].startswith(DEFAULT_PREFIX):
				if self.featureNamesInverse[index] != self.featureNamesInverse[index]:
					return False
		for featureName in other.featureNames.keys():
			if not featureName.startswith(DEFAULT_PREFIX) and featureName not in self.featureNames:
				return False
			if not featureName.startswith(DEFAULT_PREFIX) and other.featureNames[featureName] != self.featureNames[featureName]:
				return False
		for index in other.featureNamesInverse.keys():
			if index not in self.featureNamesInverse:
				return False
			if not other.featureNamesInverse[index].startswith(DEFAULT_PREFIX):
				if other.featureNamesInverse[index] != other.featureNamesInverse[index]:
					return False
		return True


	def _getIndex(self, identifier):
		toReturn = identifier
		if identifier is None:
			raise ArgumentException("An identifier cannot be None")
		if (not isinstance(identifier,basestring)) and (not isinstance(identifier,int)):
			raise ArgumentException("The indentifier must be either a string or integer index")
		if isinstance(identifier,int):
			if identifier < 0 or identifier >= len(self.featureNamesInverse):
				raise ArgumentException("The index " + str(identifier) +" is outside of the range of possible values")
		if isinstance(identifier,basestring):
			if identifier not in self.featureNames:
				raise ArgumentException("The featureName '" + identifier + "' cannot be found")
			# set as index for return
			toReturn = self.featureNames[identifier]
		return toReturn

	def _nextDefaultFeatureName(self):
		ret = DEFAULT_PREFIX + str(self._nextDefaultValue)
		self._nextDefaultValue = self._nextDefaultValue + 1
		return ret

	def _setAllDefault(self):
		self.featureNames = {}
		self.featureNamesInverse = {}
		for i in xrange(self.features()):
			defaultFeatureName = self._nextDefaultFeatureName()
			self.featureNamesInverse[i] = defaultFeatureName
			self.featureNames[defaultFeatureName] = i


	def _addFeatureName(self, featureName):
		"""
		Name the next feature outside of the current possible range with the given featureName

		featureName may be either a string, or None if you want this next feature to have a default
		name. If the featureName is not a string, or already being used by another feature, an
		ArgumentException will be raised.

		"""
		if featureName is not None and not isinstance(featureName, basestring):
			raise ArgumentException("The featureName must be a string")
		if featureName in self.featureNames:
			raise ArgumentException("This featureName is already in use")
		
		if featureName is None:
			featureName = self._nextDefaultFeatureName()

		if featureName.startswith(DEFAULT_PREFIX):
			featureNameNum = int(featureName[len(DEFAULT_PREFIX):])
			self._nextDefaultValue = max(self._nextDefaultValue + 1, featureNameNum)

		features = len(self.featureNamesInverse)
		self.featureNamesInverse[features] = featureName
		self.featureNames[featureName] = features


	def _removeFeatureNameAndShift(self, toRemove):
		"""
		Removes the specified feature from the featureName set, changing the other featureNames to fill
		in the missing index.

		toRemove must be a non None string or integer, specifying either a current featureName
		or the index of a current featureName.
		
		"""
		#this will throw the appropriate exceptions, if need be
		index = self._getIndex(toRemove)
		featureName = self.featureNamesInverse[index]

		del self.featureNames[featureName]

		features  = len(self.featureNamesInverse)
		# remaping each index starting with the one we removed
		for i in xrange(index, features-1):
			nextFeatureName = self.featureNamesInverse[i+1]
			if featureName is not None:
				self.featureNames[nextFeatureName] = i
			self.featureNamesInverse[i] = nextFeatureName
		#delete the last mapping, that featureName was shifted in the for loop
		del self.featureNamesInverse[features-1]


	def _renameFeatureName_implementation(self, oldIdentifier, newFeatureName, allowDefaults=False):
		"""
		Changes the featureName specified by previous to the supplied input featureName.
		
		oldIdentifier must be a non None string or integer, specifying either a current featureName
		or the index of a current featureName. newFeatureName may be either a string not currently
		in the featureName set, or None for an default featureName. newFeatureName may begin with the
		default prefix

		"""

		#this will throw the appropriate exceptions, if need be
		index = self._getIndex(oldIdentifier)
		if newFeatureName is not None: 
			if not isinstance(newFeatureName,basestring):
				raise ArgumentException("The new featureName must be either None or a string")
			if not allowDefaults and newFeatureName.startswith(DEFAULT_PREFIX):
				raise ArgumentException("Cannot manually add a featureName with the default prefix")
		if newFeatureName in self.featureNames:
			if self.featureNamesInverse[index] == newFeatureName:
				return
			raise ArgumentException("This featureName is already in use")
		
		if newFeatureName is None:
			newFeatureName = self._nextDefaultFeatureName()

		#remove the current featureName
		oldFeatureName = self.featureNamesInverse[index]
		del self.featureNames[oldFeatureName]		

		# setup the new featureName
		self.featureNamesInverse[index] = newFeatureName
		self.featureNames[newFeatureName] = index

		#TODO increment next default if necessary


	def _renameMultipleFeatureNames_implementation(self, assignments=None, allowDefaults=False):
		"""
		Rename some portion of the featureName set according to the input. 

		assignments may be either a list or dict specifying new featureName names, or None
		to set all featureNames to new default values. If assignment is any other type, or
		if the featureNames are not strings, the featureNames are not unique, the feature
		indices are not integers, then an ArgumentException will be raised. If
		allowDefault is False, then none of the new featureNames may begin with the default
		prefix.

		"""
		# if none, we have to set up default values	
		if assignments is None:
			self._setAllDefault()
			return
		# only certain types of input are accepted
		if (not isinstance(assignments,list)) and (not isinstance(assignments,dict)):
			raise ArgumentException("FeatureNames may only be a list or dictionary")

		if isinstance(assignments,list):
			#convert to dict so we only write the checking code once
			temp = {}
			for index in xrange(len(assignments)):
				featureName = assignments[index]
				if featureName in temp:
					raise ArgumentException("Cannot input duplicate featureNames")
				temp[featureName] = index
			assignments = temp

		# at this point, the input must be a dict
		#check input before performing any action
		for featureName in assignments.keys():
			if not isinstance(featureName,basestring):
				raise ArgumentException("FeatureNames must be strings")
			if not isinstance(assignments[featureName],int):
				raise ArgumentException("Indices must be integers")
			if not allowDefaults and featureName.startswith(DEFAULT_PREFIX):
				raise ArgumentException("Cannot manually add a featureName with the default prefix")
	

		# we have to first clear the current featureNames, so if one of our 
		# renames matches a current featureName, it doesn't throw an exception
		for key in assignments.keys():
			if key in self.featureNames:
				index = self.featureNames[key]
				del self.featureNames[key]
				temp = DEFAULT_PREFIX + 'TEMPOARY_FOR_COLUMN:' + str(index)
				self.featureNames[temp] = temp
				self.featureNamesInverse[index] = temp
		for key in assignments.keys():
			self._renameFeatureName_implementation(assignments[key],key,allowDefaults)


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
			copied = self.outerReference.duplicate()
			dataY = copied.extractPoints(self.foldList[self.index])
			dataX = copied
			self.index = self.index +1
			return dataX, dataY



class View():
	__metaclass__ = ABCMeta

	def equals(self, other):
		if not isinstance(other, View):
			return False
		if len(self) != len(other):
			return False
		if self.index() != other.index():
			return False
		if self.name() != other.name():
			return False
		for i in xrange(len(self)):
			if self[i] != other[i]:
				return False
		return True

	@abstractmethod
	def __getitem__(self, index):
		pass
	@abstractmethod
	def __setitem__(self, key, value):
		pass
	@abstractmethod
	def nonZeroIterator(self):
		pass
	@abstractmethod
	def __len__(self):
		pass
	@abstractmethod
	def index(self):
		pass
	@abstractmethod
	def name(self):
		pass

