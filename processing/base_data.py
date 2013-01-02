"""
Anchors the hierarchy of data representation types, providing stubs and common functions.

"""

# TODO conversions
# TODO who sorts inputs to derived implementations?

from copy import copy
from copy import deepcopy
from ..utility.custom_exceptions import ArgumentException
import random
import re
import os


# a default seed for testing and predictible trials
DEFAULT_SEED = 'DEFAULTSEED'

# the prefix for default featureNames
DEFAULT_PREFIX = "_DEFAULT_#"


class BaseData(object):
	"""
	Class defining important data manipulation operations and giving functionality
	for the naming the features of that data. A mapping from feature names to column
	numbers is given by the featureNames attribute, the inverse of that mapping is
	given by featureNamesInverse.

	"""

	def __init__(self, featureNames=None):
		"""
		Instantiates the featureName book-keeping structures that are defined by this representation.
		
		featureNames may be None if the object is to have default names, or a list or dict defining the
		featureName mapping.

 		"""
		super(BaseData, self).__init__()
		self._nextDefaultValue = 0
		self._setAllDefault()
		self._renameMultipleFeatureNames_implementation(featureNames,True)
		if featureNames is not None and len(featureNames) != self.columns():
			raise ArgumentException("Cannot have different number of featureNames and columns")


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
		if the names are not strings, the names are not unique, the column
		indices are not integers, or the names begin with the default prefix,
		then an ArgumentException will be raised.

		"""
		self._renameMultipleFeatureNames_implementation(assignments,False)



	###########################
	# Higher Order Operations #
	###########################

	def duplicate(self):
		"""
		Return a new object which has the same data and featureNames as this object

		"""
		# extract all
		# add extracted back to self
		# return extracted
		#TODO
		raise NotImplementedError

	def duplicatePoints(self, points):
		"""
		Return a new object which consists only of those specified points, without mutating
		this object.
		
		"""
		if points is None:
			raise ArgumentException("Must provide identifiers for the points you want duplicated")
		#TODO
		raise NotImplementedError
	
	def duplicateColumns(self, columns):
		"""
		Return a new object which consists only of those specified columns, without mutating
		this object.
		
		"""
		if columns is None:
			raise ArgumentException("Must provide identifiers for the columns you want duplicated")
		#TODO
		raise NotImplementedError


	def joinUniqueKeyedOther(self, other, fillUnmatched):
		# other must be data rep obj
		# there must be overlap
		# cannot be complete overlap
		
		# from overlap generate keys from other
		#	- must be unique

		# determine how many columns of new data there will be

		# for each new column
		# apply to each point in self with a lookup function into other

		# .... how do we do lookup?
		# we have all the column numbers, so its ok.
		pass

	
	def dropStringValuedColumns(self):
		"""
		Modify this object so that it no longer contains columns which have strings
		as values

		"""
		def hasStrings(column):
			for value in column:
				if isinstance(value, basestring):
					return True
			return False
	
		self.extractColumns(hasStrings)


	def columnToBinaryCategoryColumns(self, columnToConvert):
		"""
		Modify this object so that the choosen column is removed, and binary range
		columns are added, one for each possible value seen in the choosen column.

		"""
		index = self._getIndex(columnToConvert)

		# extract col.
		toConvert = self.extractColumns([index])

		# MR to get list of values
		def getValue(point):
			return [(point[0],1)]
		def simpleReducer(identifier, valuesList):
			return (identifier,0)

		values = toConvert.mapReduceOnPoints(getValue, simpleReducer)
		values.renameFeatureName(0,'values')
		values = values.extractColumns([0])

		# Convert to RLD, so we can have easy access
		values = values.convertToRowListData()

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
			ret.renameFeatureName(0, varName + "=" + str(value))
			toConvert.appendColumns(ret)

		# remove the original column, and combine with self
		toConvert.extractColumns([varName])
		self.appendColumns(toConvert)


	def columnToIntegerCategories(self, columnToConvert):
		"""
		Modify this object so that the chosen column in removed, and a new integer
		valued column is added with values 0 to n-1, one for each of n values present
		in the original column

		"""
		index = self._getIndex(columnToConvert)

		# extract col.
		toConvert = self.extractColumns([index])

		# MR to get list of values
		def getValue(point):
			return [(point[0],1)]
		def simpleReducer(identifier, valuesList):
			return (identifier,0)

		values = toConvert.mapReduceOnPoints(getValue, simpleReducer)
		values.renameFeatureName(0,'values')
		values = values.extractColumns([0])

		# Convert to RLD, so we can have easy access
		values = values.convertToRowListData()

		mapping = {}
		index = 0
		for point in values.data:
			if point[0] not in mapping:
				mapping[point[0]] = index
				index = index + 1

		# use apply to each to make new column with the int mappings
		def lookup(point):
			return mapping[point[0]]

		converted = toConvert.applyFunctionToEachPoint(lookup)
		converted.renameFeatureName(0,toConvert.featureNamesInverse[0])		

		self.appendColumns(converted)


	def selectConstantOfPointsByValue(self, numToSelect, columnToSelectOver, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of points from
		this object, with the sample limited to a constant number of points
		of each representative value in the specifed column. Those selected
		values are also removed from this object.

		"""
		random.seed(seed)
		if numToSelect is None:
			raise ArgumentException("The selection constant must not be None")
		if numToSelect <= 0:
			raise ArgumentException("The selection constant must be positive")
		index = self._getIndex(columnToSelectOver)
		#TODO
		raise NotImplementedError


	def selectPercentOfPointsByValue(self, percentToSelect, columnToSelectOver, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of points from
		this object, with the sample limited to a percentage of points
		of each representative value in the specified column. Those selected
		values are also removed from this object.

		"""
		random.seed(seed)
		if percentToSelect is None:
			raise ArgumentException("percentToSelect must not be none")
		if percentToSelect <= 0:
			raise ArgumentException("percentToSelect must be greater than 0")
		if percentToSelect >= 100:
			raise ArgumentException("percentToSelect must be less than 100")
		index = self._getIndex(columnToSelectOver)
		#TODO
		raise NotImplementedError

		#MR to find how many of each value
		def mapperCount (point):
			return [(point[index],1)]
		def reducerCount (identifier, values):
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
#		self.addColumn(ids)

	
		#apply to all to number the different values		
		#	the storage struc is outside the func, ie we have access to it here
		#apply to all using above to mark selected

	def extractPointsByCoinToss(self, extractionProbability, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of points
		from this object, where a random experient is performed for each
		point, with the chance of selection equal to the extractionProbabilty
		parameter. Those selected values are also removed from this object.

		"""
		random.seed(seed)
		if extractionProbability is None:
			raise ArgumentException("Must provide a extractionProbability")
		if  extractionProbability <= 0:
			raise ArgumentException("extractionProbability must be greater than zero")
		if extractionProbability >= 1:
			raise ArgumentException("extractionProbability must be less than one")

		def experiment(point):
			return bool(random.random() < extractionProbability)

		def isSelected(point):
			return point[len(point)-1]

		selectionKeys = self.applyFunctionToEachPoint(experiment)
		self.appendColumns(selectionKeys)
		ret = self.extractPoints(isSelected)
		# remove the experimental data
		if ret.points() > 0:
			ret.extractColumns([ret.columns()-1])
		if self.points() > 0:
			self.extractColumns([self.columns()-1])
		
		return ret
	
	#################################
	# Functions for derived classes #
	#################################

	def transpose(self):
		"""
		Function to transpose the data, ie invert the column and point indices of the data.
	
		Columns are then given default featureNames.

		"""
		self._transpose_implementation()
		self.renameMultipleFeatureNames(None)

	def appendPoints(self, toAppend):
		"""
		Append the points from the toAppend object to the bottom of the columns in this object.

		toAppend cannot be None, and must be a kind of data representation object with the same
		number of columns as the calling object.
		
		"""
		if toAppend is None:
			raise ArgumentException("toAppend must not be None")
		if not isinstance(toAppend,BaseData):
			raise ArgumentException("toAppend must be a kind of data representation object")
		if not self.columns() == toAppend.columns():
			raise ArgumentException("toAppend must have the same number of columns as this object")
		self._appendPoints_implementation(toAppend)
		
	def appendColumns(self, toAppend):
		"""
		Append the columns from the toAppend object to right ends of the points in this object

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
		self._appendColumns_implementation(toAppend)

		for i in xrange(toAppend.columns()):
			self._addFeatureName(toAppend.featureNamesInverse[i])

	def sortPoints(self, cmp=None, key=None, reverse=False):
		""" 
		Modify this object so that the points are sorted in place, where the input
		arguments are interpreted and employed in the same way as Python list
		sorting.

		"""
		self._sortPoints_implementation(cmp, key, reverse)

	def sortColumns(self, cmp=None, key=None, reverse=False):
		""" 
		Modify this object so that the columns are sorted in place, where the input
		arguments are interpreted and employed in the same way as Python list
		sorting.

		"""
		newFeatureNameOrder = self._sortColumns_implementation(cmp, key, reverse)
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
		if toExtract is not None:
			if start is not None or end is not None:
				raise ArgumentException("Range removal is exclusive, to use it, toExtract must be None")
		elif start is not None or end is not None:
			if start < 0 or start > self.points():
				raise ArgumentException("start must be a valid index, in the range of possible points")
			if end < 0 or end > self.points():
				raise ArgumentException("end must be a valid index, in the range of possible points")
			if start > end:
				raise ArgumentException("start cannot be an index greater than end")

		ret = self._extractPoints_implementation(toExtract, start, end, number, randomize)
		ret._renameMultipleFeatureNames_implementation(self.featureNames,True)
		return ret

	def extractColumns(self, toExtract=None, start=None, end=None, number=None, randomize=False):
		"""
		Modify this object, removing those columns that are specified by the input, and returning
		an object containing those removed columns. This particular function only does argument
		checking and modifying the featureNames for this object. It is the job of helper functions in
		the derived class to perform the removal and assign featureNames for the returned object.

		toExtract may be a single identifier, a list of identifiers, or a function that when
		given a column will return True if it is to be removed. number is the quantity of columns that
		are to be extracted, the default None means unlimited extracttion. start and end are
		parameters indicating range based extraction: if range based extraction is employed,
		toExtract must be None, and vice versa. If only one of start and end are non-None, the
		other defaults to 0 and self.columns() respectibly. randomize indicates whether random
		sampling is to be used in conjunction with the number parameter, if randomize is False,
		the chosen columns are determined by column order, otherwise it is uniform random across the
		space of possible removals.

		"""
		if toExtract is not None:
			if start is not None or end is not None:
				raise ArgumentException("Range removal is exclusive, to use it, toExtract must be None")
		elif start is not None or end is not None:
			if start < 0 or start > self.columns():
				raise ArgumentException("start must be a valid index, in the range of possible columns")
			if end < 0 or end > self.columns():
				raise ArgumentException("end must be a valid index, in the range of possible columns")
			if start > end:
				raise ArgumentException("start cannot be an index greater than end")

		ret = self._extractColumns_implementation(toExtract, start, end, number, randomize)
		for key in ret.featureNames.keys():
			self._removeFeatureNameAndShift(key)
		return ret


	def applyFunctionToEachPoint(self, function):
		"""
		Applies the given funciton to each point in this object, collecting the
		output values into a new object that is returned upon completion.

		function must not be none and accept a point as an argument

		"""
		if function is None:
			raise ArgumentException("function must not be None")
		return self._applyFunctionToEachPoint_implementation(function)

	def applyFunctionToEachColumn(self, function):
		"""
		Applies the given funciton to each column in this object, collecting the
		output values into a new object in the shape of a column vector that is
		returned upon completion.

		function must not be none and accept a column as an argument

		"""
		if function is None:
			raise ArgumentException("function must not be None")
		return self._applyFunctionToEachColumn_implementation(function)


	def mapReduceOnPoints(self, mapper, reducer):
		if mapper is None or reducer is None:
			raise ArgumentException("The arguments must not be none")
		if not hasattr(mapper, '__call__'):
			raise ArgumentException("The mapper must be callable")
		if not hasattr(reducer, '__call__'):
			raise ArgumentException("The reducer must be callable")

		ret = self._mapReduceOnPoints_implementation(mapper, reducer)
		return ret


	def equals(self, other):
		if not self._equalFeatureNames(other):
			return False		

		return self._equals_implementation(other)

	def points(self):
		return self._points_implementation()

	def columns(self):
		return self._columns_implementation()

	def convertToRowListData(self):
		return self._convertToRowListData_implementation()

	def convertToDenseMatrixData(self):
		return self._convertToDenseMatrixData_implementation()

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
		for i in xrange(self.columns()):
			defaultFeatureName = self._nextDefaultFeatureName()
			self.featureNamesInverse[i] = defaultFeatureName
			self.featureNames[defaultFeatureName] = i


	def _addFeatureName(self, featureName):
		"""
		FeatureName the next column outside of the current possible range with the given featureName

		featureName may be either a string, or None if you want this next column to have a default
		name. If the featureName is not a string, or already being used by another column, an
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

		columns  = len(self.featureNamesInverse)
		self.featureNamesInverse[columns] = featureName
		self.featureNames[featureName] = columns


	def _removeFeatureNameAndShift(self, toRemove):
		"""
		Removes the specified column from the featureName set, changing the other featureNames to fill
		in the missing index.

		toRemove must be a non None string or integer, specifying either a current featureName
		or the index of a current featureName.
		
		"""
		#this will throw the appropriate exceptions, if need be
		index = self._getIndex(toRemove)
		featureName = self.featureNamesInverse[index]

		del self.featureNames[featureName]

		columns  = len(self.featureNamesInverse)
		# remaping each index starting with the one we removed
		for i in xrange(index, columns-1):
			nextFeatureName = self.featureNamesInverse[i+1]
			if featureName is not None:
				self.featureNames[nextFeatureName] = i
			self.featureNamesInverse[i] = nextFeatureName
		#delete the last mapping, that featureName was shifted in the for loop
		del self.featureNamesInverse[columns-1]


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
		if the featureNames are not strings, the featureNames are not unique, the column
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



