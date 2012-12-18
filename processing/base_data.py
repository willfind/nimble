"""
Anchors the hierarchy of data representation types, providing stubs and common functions.

"""

# TODO conversions
# TODO natrual join extract label ops
# TODO who sorts inputs to derived implementations?

from copy import copy
from copy import deepcopy
from ..utility.custom_exceptions import ArgumentException
import random
import re
import os


# a default seed for testing and predictible trials
DEFAULT_SEED = 'DEFAULTSEED'

# the prefix for default label names
DEFAULT_PREFIX = "_DEFAULT_#"


class BaseData(object):
	"""
	Class defining important data manipulation operations and giving functionality
	for the labeling of that data. A mapping from labels to column numbers is given
	by the labels attribute, the inverse of that mapping is given by labelsInverse.
	labels may be empty if the object is labelless, or contains no data. labelsInverse
	will be empty only if the object contains no data, otherwise it will map column
	numbers to None.

	"""

	def __init__(self, labels=None):
		"""
		Instantiates the label book-keeping structures that are defined by this representation.
		
		labels may be None if the object is to have default labels, or a list or dict defining the label mapping.

 		"""
		super(BaseData, self).__init__()
		self._nextDefaultValue = 0
		self._setAllDefault()
		self._renameMultipleLabels_implementation(labels,True)
		if labels is not None and len(labels) != self.numColumns():
			raise ArgumentException("Cannot have different number of labels and columns")


	########################
	# Low Level Operations #
	########################

	def labelDifference(self, other):
		"""
		Returns a set containing those labels in this object that are not also in the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, BaseData):
			raise ArgumentException("Must provide another representation type to determine label difference")
		
		return self.labels.viewkeys() - other.labels.viewkeys() 

	def labelIntersection(self, other):
		"""
		Returns a set containing only those labels that are shared by this object and the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, BaseData):
			raise ArgumentException("Must provide another representation type to determine label intersection")
		
		return self.labels.viewkeys() & other.labels.viewkeys() 

	def labelSymmetricDifference(self, other):
		"""
		Returns a set containing only those labels not shared between this object and the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, BaseData):
			raise ArgumentException("Must provide another representation type to determine label difference")
		
		return self.labels.viewkeys() ^ other.labels.viewkeys() 

	def labelUnion(self,other):
		"""
		Returns a set containing all labels in either this object or the input object.

		"""
		if other is None:
			raise ArgumentException("The other object cannot be None")
		if not isinstance(other, BaseData):
			raise ArgumentException("Must provide another representation type to determine label union")
		
		return self.labels.viewkeys() | other.labels.viewkeys() 


	def renameLabel(self, oldIdentifier, newLabel):
		"""
		Changes the label specified by previous to the supplied input label.
		
		oldIdentifier must be a non None string or integer, specifying either a current label
		or the index of a current label. newLabel may be either a string not currently
		in the label set, or None for an default label. newLabel cannot begin with the
		default label prefix.

		"""
		self._renameLabel_implementation(oldIdentifier, newLabel, False)

	def renameMultipleLabels(self, assignments=None):
		"""
		Rename some portion of the label set according to the input. 

		assignments may be either a list or dict specifying new label names, or None
		to set all labels to new default values. If assignment is any other type, or
		if the labels are not strings, the labels are not unique, the column
		indices are not integers, or the labels begin with the default prefix,
		then an ArgumentException will be raised.

		"""
		self._renameMultipleLabels_implementation(assignments,False)



	###########################
	# Higher Order Operations #
	###########################

	def duplicate(self):
		"""
		Return a new object which has the same data and labels as this object

		"""
		# extract all
		# add extracted back to self
		# return extracted
		#TODO
		raise NotImplementedError

	def duplicateRows(self, rows):
		"""
		Return a new object which consists only of those specified rows, without mutating
		this object.
		
		"""
		if rows is None:
			raise ArgumentException("Must provide identifiers for the rows you want duplicated")
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
		# both must be labeled
		# there must be overlap
		# cannot be complete overlap
		
		# from overlap generate keys from other
		#	- must be unique

		# determine how many columns of new data there will be

		# for each new column
		# apply to each row in self with a lookup function into other

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
	
		self.extractSatisfyingColumns(hasStrings)


	def columnToBinaryCategoryColumns(self, columnToConvert):
		"""
		Modify this object so that the choosen column is removed, and binary range
		columns are added, one for each possible value seen in the choosen column.

		"""
		index = self._getIndex(columnToConvert)

		# extract col.
		toConvert = self.extractColumns([index])

		# MR to get list of values
		def getValue(row):
			return [(row[0],1)]
		def simpleReducer(identifier, valuesList):
			return (identifier,0)

		values = toConvert.mapReduceOnRows(getValue, simpleReducer)
		values.renameLabel(0,'values')
		values = values.extractColumns([0])

		# Convert to RLD, so we can have easy access
		values = values.convertToRowListData()

		# for each value run applyToEach to produce a category row for each value
		def makeFunc(value):
			def equalTo(row):
				if row[0] == value:
					return 1
				return 0
			return equalTo

		varName = toConvert.labelsInverse[0]

		for row in values.data:
			value = row[0]
			ret = toConvert.applyToEachRow(makeFunc(value))
			ret.renameLabel(0, varName + "=" + str(value))
			toConvert.addColumns(ret)

		# remove the original column, and combine with self
		toConvert.extractColumns([varName])
		self.addColumns(toConvert)


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
		def getValue(row):
			return [(row[0],1)]
		def simpleReducer(identifier, valuesList):
			return (identifier,0)

		values = toConvert.mapReduceOnRows(getValue, simpleReducer)
		values.renameLabel(0,'values')
		values = values.extractColumns([0])

		# Convert to RLD, so we can have easy access
		values = values.convertToRowListData()

		mapping = {}
		index = 0
		for row in values.data:
			if row[0] not in mapping:
				mapping[row[0]] = index
				index = index + 1

		# use apply to each to make new column with the int mappings
		def lookup(row):
			return mapping[row[0]]

		converted = toConvert.applyToEachRow(lookup)
		converted.renameLabel(0,toConvert.labelsInverse[0])		

		self.addColumns(converted)


	def selectConstantOfRowsByValue(self, numToSelect, columnToSelectOver, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of rows from
		this object, with the sample limited to a constant number of rows
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


	def selectPercentOfRowsByValue(self, percentToSelect, columnToSelectOver, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of rows from
		this object, with the sample limited to a percentage of rows
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
		def mapperCount (row):
			return [(row[index],1)]
		def reducerCount (identifier, values):
			total = 0
			for value in values:
				total += value
			return (identifier, total)
		
#		totalOfEach = self.mapReduceOnRows(mapperCount,reducerCount)
			

		valueToTotal = {}
		def tagValuesWithID(row):
			key = row[index]
			if key not in valueToTotal:
				valueToTotal[key] = 0
				return 0
			ret = valueToTotal[key]
			valueToTotal[key] += 1
			return ret

#		ids = self.applyToEachRow(tagValuesWithID)
#		self.addColumn(ids)

	
		#apply to all to number the different values		
		#	the storage struc is outside the func, ie we have access to it here
		#apply to all using above to mark selected


	def selectPercentOfAllRows(self, percentToSelect, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of rows from
		this object, with the sampe limited to the given percentage of all
		present rows. Those selected values are also removed from this object.

		"""
		random.seed(seed)
		if percentToSelect is None:
			raise ArgumentException("percentToSelect must not be none")
		if percentToSelect <= 0:
			raise ArgumentException("percentToSelect must be greater than 0")
		if percentToSelect >= 100:
			raise ArgumentException("percentToSelect must be less than 100")

		numToSelect = int((percentToSelect/100.0) * self.numRows())
		selectionKeys = range(self.numRows())
		selectionKeys = random.sample(selectionKeys, numToSelect)
		ret = self.extractRows(selectionKeys)
		
		return ret

	def extractRowsByCoinToss(self, extractionProbability, seed=DEFAULT_SEED):
		"""
		Return a new object containing a randomly selected sample of rows
		from this object, where a random experient is performed for each
		row, with the chance of selection equal to the extractionProbabilty
		parameter. Those selected values are also removed from this object.

		"""
		random.seed(seed)
		if extractionProbability is None:
			raise ArgumentException("Must provide a extractionProbability")
		if  extractionProbability <= 0:
			raise ArgumentException("extractionProbability must be greater than zero")
		if extractionProbability >= 1:
			raise ArgumentException("extractionProbability must be less than one")

		def experiment(row):
			return bool(random.random() < extractionProbability)

		def isSelected(row):
			return row[len(row)-1]

		selectionKeys = self.applyToEachRow(experiment)
		self.addColumns(selectionKeys)
		ret = self.extractSatisfyingRows(isSelected)
		# remove the experimental data
		if ret.numRows() > 0:
			ret.extractColumns([ret.numColumns()-1])
		if self.numRows() > 0:
			self.extractColumns([self.numColumns()-1])
		
		return ret
	
	#################################
	# Functions for derived classes #
	#################################

	def transpose(self):
		"""
		Function to transpose the data, ie invert the column and row indices of the data.
	
		Columns are then given default labels.

		"""
		self._transpose_implementation()
		self.renameMultipleLabels(None)

	def addRows(self, toAdd):
		"""
		Append the rows from the toAdd object to the bottom of the columns in this object.

		toAdd cannot be None, and must be a kind of data representation object with the same
		number of columns as the calling object.
		
		"""
		if toAdd is None:
			raise ArgumentException("toAdd must not be None")
		if not isinstance(toAdd,BaseData):
			raise ArgumentException("toAdd must be a kind of data representation object")
		if not self.numColumns() == toAdd.numColumns():
			raise ArgumentException("toAdd must have the same number of columns as this object")
		self._addRows_implementation(toAdd)
		
	def addColumns(self, toAdd):
		"""
		Append the columns from the toAdd object to right ends of the rows in this object

		toAdd cannot be None, must be a kind of data representation object with the same
		number of rows as the calling object, and must not share any labels with the calling
		object.
		
		"""	
		if toAdd is None:
			raise ArgumentException("toAdd must not be None")
		if not isinstance(toAdd,BaseData):
			raise ArgumentException("toAdd must be a kind of data representation object")
		if not self.numRows() == toAdd.numRows():
			raise ArgumentException("toAdd must have the same number of rows as this object")
		if self.labelIntersection(toAdd):
			raise ArgumentException("toAdd must not share any labels with this object")
		self._addColumns_implementation(toAdd)

		for i in xrange(toAdd.numColumns()):
			self._addLabel(toAdd.labelsInverse[i])

	def sortRows(self, cmp=None, key=None, reverse=False):
		""" 
		Modify this object so that the rows are sorted in place, where the input
		arguments are interpreted and employed in the same way as Python list
		sorting.

		"""
		self._sortRows_implementation(cmp, key, reverse)

	def sortColumns(self, cmp=None, key=None, reverse=False):
		""" 
		Modify this object so that the columns are sorted in place, where the input
		arguments are interpreted and employed in the same way as Python list
		sorting.

		"""
		newLabelOrder = self._sortColumns_implementation(cmp, key, reverse)
		self._renameMultipleLabels_implementation(newLabelOrder,True)

	def extractRows(self, toExtract):
		"""
		Modify this object to have only the rows whose identifiers are not given
		in the input, returning an object containing those rows that are.

		toExtract must not be None, and must contain only identifiers that make
		sense in the context of this object

		"""
		if toExtract is None:
			raise ArgumentException("toRemove must not be None")
		ret = self._extractRows_implementation(toExtract)
		ret._renameMultipleLabels_implementation(self.labels,True)
		return ret

	def extractColumns(self, toExtract):
		"""
		Modify this object to have only the columns whose identifiers are not given
		in the input, returning an object containing those columns that are.

		toExtract must not be None, and must contain only identifiers that make
		sense in the context of this object.

		"""
		if toExtract is None:
			raise ArgumentException("toRemove must not be None")
		toExtractIndices = []
		for value in toExtract:
			toExtractIndices.append(self._getIndex(value))

		toExtractIndices.sort()

		ret = self._extractColumns_implementation(toExtractIndices)

		# remove from the end, so as to not disturb the lower indices	
		toExtractIndices.sort()
		toExtractIndices.reverse()
		newLabels = []
		for colNum in toExtractIndices:
			newLabels.append(self.labelsInverse[colNum])			
			self._removeLabelAndShift(colNum)

		newLabels.reverse()
		ret._renameMultipleLabels_implementation(newLabels,True)

		return ret

	def extractSatisfyingRows(self, function):
		"""
		Modify this object to have only the rows that do not satisfy the given function,
		returning an object containing those rows that do.

		function must not be none, accept a row as an argument and return a truth value

		"""
		if function is None:
			raise ArgumentException("function must not be None")
		ret = self._extractSatisfyingRows_implementation(function)
		ret._renameMultipleLabels_implementation(self.labels,True)
		return ret

	def extractSatisfyingColumns(self, function):
		"""
		Modify this object to have only the columns that do not satisfy the given function,
		returning an object containing those columns that do.

		function must not be none, accept a column as an argument and return a truth value

		"""
		if function is None:
			raise ArgumentException("function must not be None")
		ret = self._extractSatisfyingColumns_implementation(function)

		return ret

	def extractRangeRows(self, start, end):
		"""
		Modify this object to have only those rows that are not within the given range,
		inclusive; returning an object containing those rows that are.
	
		start and end must not be null, must be within the range of possible rows,
		and start must not be greater than end

		"""
		if start is None:
			raise ArgumentException("start must be an interger index, not None")
		if start < 0 or start > self.numRows():
			raise ArgumentException("start must be a valid index, in the range of possible rows")
		if end is None:
			raise ArgumentException("end must be an interger index, not None")
		if end < 0 or end > self.numRows():
			raise ArgumentException("end must be a valid index, in the range of possible rows")
		if start > end:
			raise ArgumentException("start cannot be an index greater than end")

		ret = self._extractRangeRows_implementation(start,end)
		ret._renameMultipleLabels_implementation(self.labels,True)
		return ret

	def extractRangeColumns(self, start, end):
		"""
		Modify this object to have only those columns that are not within the given range,
		inclusive; returning an object containing those rows that are.
	
		start and end must not be null, must be within the range of possible columns,
		and start must not be greater than end

		"""
		start = self._getIndex(start)
		end = self._getIndex(end)
		if start is None:
			raise ArgumentException("start must be an interger index, not None")
		if start < 0 or start > self.numColumns():
			raise ArgumentException("start must be a valid index, in the range of possible rows")
		if end is None:
			raise ArgumentException("end must be an interger index, not None")
		if end < 0 or end > self.numRows():
			raise ArgumentException("end must be a valid index, in the range of possible rows")
		if start > end:
			raise ArgumentException("start must come before end")

		ret = self._extractRangeColumns_implementation(start,end)
		for index in reversed(xrange(start,end+1)):
			removedLabel = self.labelsInverse[index]
			self._removeLabelAndShift(removedLabel)
			ret._renameLabel_implementation(index-start, removedLabel, True)
		return ret

	def applyToEachRow(self, function):
		"""
		Applies the given funciton to each row in this object, collecting the
		output values into a new object in the shape of a row vector that is
		returned upon completion.

		function must not be none and accept a row as an argument

		"""
		if function is None:
			raise ArgumentException("function must not be None")
		return self._applyToEachRow_implementation(function)

	def applyToEachColumn(self, function):
		"""
		Applies the given funciton to each column in this object, collecting the
		output values into a new object in the shape of a column vector that is
		returned upon completion.

		function must not be none and accept a column as an argument

		"""
		if function is None:
			raise ArgumentException("function must not be None")
		return self._applyToEachColumn_implementation(function)


	def mapReduceOnRows(self, mapper, reducer):
		if mapper is None or reducer is None:
			raise ArgumentException("The arguments must not be none")
		if not hasattr(mapper, '__call__'):
			raise ArgumentException("The mapper must be callable")
		if not hasattr(reducer, '__call__'):
			raise ArgumentException("The reducer must be callable")

		ret = self._mapReduceOnRows_implementation(mapper, reducer)
		return ret


	def equals(self, other):
		if not self._equalLabels(other):
			return False		

		return self._equals_implementation(other)

	def numRows(self):
		return self._numRows_implementation()

	def numColumns(self):
		return self._numColumns_implementation()

	def convertToRowListData(self):
		return self._convertToRowListData_implementation()

	def convertToDenseMatrixData(self):
		return self._convertToDenseMatrixData_implementation()

	####################
	# Helper functions #
	####################

	def _equalLabels(self, other):
		"""
		Private function to determine equality of BaseData labels. It ignores
		equality of default values, though if default values are present,
		the number of variables and their indices must match up.

		"""
		if other is None:
			return False
		if not isinstance(other,BaseData):
			return False	
		if len(self.labels) != len(other.labels):
			return False
		if len(self.labelsInverse) != len(other.labelsInverse):
			return False
		# check both label directions
		for label in self.labels.keys():
			if not label.startswith(DEFAULT_PREFIX) and label not in other.labels:
				return False
			if not label.startswith(DEFAULT_PREFIX) and self.labels[label] != other.labels[label]:
				return False
		for index in self.labelsInverse.keys():
			if index not in other.labelsInverse:
				return False
			if not self.labelsInverse[index].startswith(DEFAULT_PREFIX):
				if self.labelsInverse[index] != self.labelsInverse[index]:
					return False
		for label in other.labels.keys():
			if not label.startswith(DEFAULT_PREFIX) and label not in self.labels:
				return False
			if not label.startswith(DEFAULT_PREFIX) and other.labels[label] != self.labels[label]:
				return False
		for index in other.labelsInverse.keys():
			if index not in self.labelsInverse:
				return False
			if not other.labelsInverse[index].startswith(DEFAULT_PREFIX):
				if other.labelsInverse[index] != other.labelsInverse[index]:
					return False
		return True


	def _getIndex(self, identifier):
		toReturn = identifier
		if identifier is None:
			raise ArgumentException("An identifier cannot be None")
		if (not isinstance(identifier,basestring)) and (not isinstance(identifier,int)):
			raise ArgumentException("The indentifier must be either a string or integer index")
		if isinstance(identifier,int):
			if identifier < 0 or identifier >= len(self.labelsInverse):
				raise ArgumentException("The index " + str(identifier) +" is outside of the range of possible values")
		if isinstance(identifier,basestring):
			if identifier not in self.labels:
				raise ArgumentException("The label '" + identifier + "' cannot be found")
			# set as index for return
			toReturn = self.labels[identifier]
		return toReturn

	def _nextDefaultLabel(self):
		ret = DEFAULT_PREFIX + str(self._nextDefaultValue)
		self._nextDefaultValue = self._nextDefaultValue + 1
		return ret

	def _setAllDefault(self):
		self.labels = {}
		self.labelsInverse = {}
		for i in xrange(self.numColumns()):
			defaultLabel = self._nextDefaultLabel()
			self.labelsInverse[i] = defaultLabel
			self.labels[defaultLabel] = i


	def _addLabel(self, label):
		"""
		Label the next column outside of the current possible range with the given label

		label may be either a string, or None if you want this next column to be labeless.
		If the label is not a string, or already being used by another column, an
		ArgumentException will be raised.

		"""
		if label is not None and not isinstance(label, basestring):
			raise ArgumentException("The label must be a string")
		if label in self.labels:
			raise ArgumentException("This label is already in use")
		
		if label is None:
			label = self._nextDefaultLabel()

		if label.startswith(DEFAULT_PREFIX):
			labelNum = int(label[len(DEFAULT_PREFIX):])
			self._nextDefaultValue = max(self._nextDefaultValue + 1, labelNum)

		columns  = len(self.labelsInverse)
		self.labelsInverse[columns] = label
		self.labels[label] = columns


	def _removeLabelAndShift(self, toRemove):
		"""
		Removes the specified column from the label set, changing the other labels to fill
		in the missing index.

		toRemove must be a non None string or integer, specifying either a current label
		or the index of a current label.
		
		"""
		#this will throw the appropriate exceptions, if need be
		index = self._getIndex(toRemove)
		label = self.labelsInverse[index]

		del self.labels[label]

		columns  = len(self.labelsInverse)
		# remaping each index starting with the one we removed
		for i in xrange(index, columns-1):
			nextLabel = self.labelsInverse[i+1]
			if label is not None:
				self.labels[nextLabel] = i
			self.labelsInverse[i] = nextLabel
		#delete the last mapping, that label was shifted in the for loop
		del self.labelsInverse[columns-1]


	def _renameLabel_implementation(self, oldIdentifier, newLabel, allowDefaults=False):
		"""
		Changes the label specified by previous to the supplied input label.
		
		oldIdentifier must be a non None string or integer, specifying either a current label
		or the index of a current label. newLabel may be either a string not currently
		in the label set, or None for an default label. newLabel may begin with the
		default label prefix

		"""

		#this will throw the appropriate exceptions, if need be
		index = self._getIndex(oldIdentifier)
		if newLabel is not None: 
			if not isinstance(newLabel,basestring):
				raise ArgumentException("The new label must be either None or a string")
			if not allowDefaults and newLabel.startswith(DEFAULT_PREFIX):
				raise ArgumentException("Cannot manually add a label with the default prefix")
		if newLabel in self.labels:
			raise ArgumentException("This label is already in use")
		
		if newLabel is None:
			newLabel = self._nextDefaultLabel()

		#remove the current label
		oldLabel = self.labelsInverse[index]
		del self.labels[oldLabel]		

		# setup the new label
		self.labelsInverse[index] = newLabel
		self.labels[newLabel] = index

		#TODO increment next default if necessary


	def _renameMultipleLabels_implementation(self, assignments=None, allowDefaults=False):
		"""
		Rename some portion of the label set according to the input. 

		assignments may be either a list or dict specifying new label names, or None
		to set all labels to new default values. If assignment is any other type, or
		if the labels are not strings, the labels are not unique, the column
		indices are not integers, then an ArgumentException will be raised. If
		allowDefault is False, then none of the new labels may begin with the default
		prefix.

		"""
		# if none, we have to set up default values	
		if assignments is None:
			self._setAllDefault()
			return
		# only certain types of input are accepted
		if (not isinstance(assignments,list)) and (not isinstance(assignments,dict)):
			raise ArgumentException("Labels may only be a list or dictionary")

		if isinstance(assignments,list):
			#convert to dict so we only write the checking code once
			temp = {}
			for index in xrange(len(assignments)):
				label = assignments[index]
				if label in temp:
					raise ArgumentException("Cannot input duplicate labels")
				temp[label] = index
			assignments = temp

		# at this point, the input must be a dict
		#check input before performing any action
		for label in assignments.keys():
			if not isinstance(label,basestring):
				raise ArgumentException("Labels must be strings")
			if not isinstance(assignments[label],int):
				raise ArgumentException("Indices must be integers")
			if not allowDefaults and label.startswith(DEFAULT_PREFIX):
				raise ArgumentException("Cannot manually add a label with the default prefix")
	

		# we have to first clear the current labels, so if one of our 
		# renames matches a current label, it doesn't throw an exception
		for key in assignments.keys():
			if key in self.labels:
				index = self.labels[key]
				del self.labels[key]
				temp = DEFAULT_PREFIX + 'TEMPOARY_FOR_COLUMN:' + str(index)
				self.labels[temp] = temp
				self.labelsInverse[index] = temp
		for key in assignments.keys():
			self._renameLabel_implementation(assignments[key],key,allowDefaults)



