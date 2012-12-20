"""
Class extending BaseData, using a list of rows to store data.

Outside of the class, functions are defined for reading and writing RowListData
to files.

"""

from base_data import BaseData
from copy import copy
from copy import deepcopy
from ..utility.custom_exceptions import ArgumentException
import random
import re
import os


class RowListData(BaseData):
	"""
	Class providing implementations of data manipulation operations on data stored
	in a list of lists, representing a list of rows of data. data is the list of
	lists. columns is the integer number of columns in each row.

	"""

	def __init__(self, data=None, labels=None):
		"""
		Instantiate a Row List data using the given data and labels. data may be
		none or an empty list to indicate an empty object, or a fully populated
		list of lists to be encased by this object. labels is passed up to
		the init funciton of BaseData, to be interpreted there.

		"""
		if data is None or len(data) == 0:
			self.columns = 0
			self.data = []
			super(RowListData, self).__init__([])
		else:
			self.columns = len(data[0])
			for row in data:
				if len(row) != self.columns:
					raise ArgumentException("Rows must be of equal size")
			self.data = data
			super(RowListData, self).__init__(labels)


	def _transpose_implementation(self):
		"""
		Function to transpose the data, ie invert the column and row indices of the data.
		
		This is not an in place operation, a new list of lists is constructed.
		"""
		tempColumns = len(self.data)
		transposed = []
		#load the new data with an empty row for each column in the original
		for i in xrange(len(self.data[0])):
			transposed.append([])
		for row in self.data:
			for i in xrange(len(row)):
				transposed[i].append(row[i])
		
		self.data = transposed
		self.columns = tempColumns

	def _appendRows_implementation(self,toAppend):
		"""
		Append the rows from the toAppend object to the bottom of the columns in this object
		
		"""
		for point in toAppend.data:
			self.data.append(point)
	
	def _appendColumns_implementation(self,toAppend):
		"""
		Append the columns from the toAppend object to right ends of the rows in this object

		"""	
		for i in xrange(self.rows()):
			for value in toAppend.data[i]:
				self.data[i].append(value)
		self.columns = self.columns + toAppend.numColumns()

	def _sortRows_implementation(self,cmp, key, reverse):
		""" 
		Modify this object so that the rows are sorted using the built in python
		sort. The input arguments are passed to that function unalterted

		"""
		self.data.sort(cmp, key, reverse)

	def _sortColumns_implementation(self,cmp, key, reverse):
		""" 
		Modify this object so that the columns are sorted using the built in python
		sort on column views. The input arguments are passed to that function unalterted.
		This funciton returns a list of labels indicating the new order of the data.

		"""
		def passThrough(toKey):
			return toKey
		if key is None:
			key = passThrough
		#create key list - to be sorted; and dictionary
		keyList = []
		temp = []
		for i in xrange(self.numColumns()):
			ithView = self.ColumnView(self,i)
			keyList.append(key(ithView))
			temp.append(None)
		keyDict = {}
		for i in xrange(len(keyList)):
			keyDict[keyList[i]] = i
		keyList.sort(cmp,passThrough,reverse)

		#now modify data to correspond to the new order
		for row in self.data:
			#have to copy it out so we don't corrupt the row
			for i in xrange(self.numColumns()):
				currKey = keyList[i]
				oldColNum = keyDict[currKey]
				temp[i] = row[oldColNum]
			for i in xrange(self.numColumns()):
				row[i] = temp[i]

		# have to deal with labels now
		for i in xrange(self.numColumns()):
			currKey = keyList[i]
			oldColNum = keyDict[currKey]
			temp[i] = self.labelsInverse[oldColNum]
		return temp

	def _extractRows_implementation(self,toExtract):
		"""
		Modify this object to have only the rows that are not given in the input,
		returning an object containing those rows that are.

		"""
		toWrite = 0
		satisfying = []
		for i in xrange(self.rows()):
			if i not in toExtract:
				self.data[toWrite] = self.data[i]
				toWrite += 1
			else:
				satisfying.append(self.data[i])

		# blank out the elements beyond our last copy, ie our last wanted row.
		for index in xrange(toWrite,len(self.data)):
			self.data.pop()

		return RowListData(satisfying)

	def _extractColumns_implementation(self,toExtract):
		"""
		Modify this object to have only the columns that are not given in the input,
		returning an object containing those columns that are.

		"""
		toExtract.sort()
		toExtract.reverse()
		extractedData = []
		for row in self.data:
			extractedRow = []
			for index in toExtract:
				extractedRow.append(row.pop(index))
			extractedRow.reverse()
			extractedData.append(extractedRow)

		self.columns = self.columns - len(toExtract)
		return RowListData(extractedData)


	def _extractSatisfyingRows_implementation(self,function):
		"""
		Modify this object to have only the rows that do not satisfy the given function,
		returning an object containing those rows that do.

		"""
		toWrite = 0
		satisfying = []
		# walk through each row, copying the wanted rows back to the toWrite index
		# toWrite is only incremented when we see a wanted row; unwanted rows are copied
		# over
		for index in xrange(len(self.data)):
			row = self.data[index]
			if not function(row):
				self.data[toWrite] = row
				toWrite += 1
			else:
				satisfying.append(row)

		# blank out the elements beyond our last copy, ie our last wanted row.
		for index in xrange(toWrite,len(self.data)):
			self.data.pop()

		return RowListData(satisfying)


	def _extractSatisfyingColumns_implementation(self,function):
		"""
		Modify this object to have only the columns whose views do not satisfy the given
		function, returning an object containing those columns whose views do.

		"""
		toExtract = []		
		for i in xrange(self.numColumns()):
			ithView = self.ColumnView(self,i)
			if function(ithView):
				toExtract.append(i)
		return self.extractColumns(toExtract)


	def _extractRangeRows_implementation(self,start,end):
		"""
		Modify this object to have only those rows that are not within the given range,
		inclusive; returning an object containing those rows that are.
	
		start and end must not be null, must be within the range of possible rows,
		and start must not be greater than end

		"""
		toWrite = start
		inRange = []
		for i in xrange(start,self.rows()):
			if i <= end:
				inRange.append(self.data[i])		
			else:
				self.data[toWrite] = self.data[i]
				toWrite += 1

		# blank out the elements beyond our last copy, ie our last wanted row.
		for index in xrange(toWrite,len(self.data)):
			self.data.pop()

		return RowListData(inRange)

	def _extractRangeColumns_implementation(self, start, end):
		"""
		Modify this object to have only those columns that are not within the given range,
		inclusive; returning an object containing those rows that are.
	
		start and end must not be null, must be within the range of possible columns,
		and start must not be greater than end

		"""
		extractedData = []
		for row in self.data:
			extractedRow = []
			#end + 1 because our ranges are inclusive, xrange's are not
			for index in reversed(xrange(start,end+1)):
				extractedRow.append(row.pop(index))
			extractedRow.reverse()
			extractedData.append(extractedRow)

		self.columns = self.columns - len(extractedRow)
		return RowListData(extractedData)


	def _applyFunctionToEachRow_implementation(self,function):
		"""
		Applies the given funciton to each row in this object, collecting the
		output values into a new object in the shape of a row vector that is
		returned upon completion.

		"""
		retData = []
		for row in self.data:
			currOut = function(row)
			retData.append([currOut])
		return RowListData(retData)

	def _applyFunctionToEachColumn_implementation(self,function):
		"""
		Applies the given funciton to each column in this object, collecting the
		output values into a new object in the shape of a column vector that is
		returned upon completion.

		"""
		retData = [[]]
		for i in xrange(self.numColumns()):
			ithView = self.ColumnView(self,i)
			currOut = function(ithView)
			retData[0].append(currOut)
		return RowListData(retData)


	def _mapReduceOnRows_implementation(self, mapper, reducer):
		mapResults = {}
		# apply the mapper to each row in the data
		for row in self.data:
			currResults = mapper(row)
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
		return RowListData(ret)

	def _numColumns_implementation(self):
		return self.columns

	def _rows_implementation(self):
		return len(self.data)

	def _equals_implementation(self,other):
		if not isinstance(other,RowListData):
			return False
		if self.rows() != other.rows():
			return False
		if self.numColumns() != other.numColumns():
			return False
		for index in xrange(self.rows()):
			if self.data[index] != other.data[index]:
				return False
		return True


	def _convertToRowListData_implementation(self):
		"""	Returns a RowListData object with the same data and labels as this one """
		return RowListData(self.data, self.labels)

	def _convertToDenseMatrixData_implementation(self):
		""" Returns a DenseMatrixData object with the same data and labels as this object """
		from dense_matrix_data import DenseMatrixData as DMD
		return DMD(self.data, self.labels)


	###########
	# Helpers #
	###########

	class ColumnView():
		"""
		Class to simulate direct random access of a column.

		"""
		def __init__(self, outer, colNum):
			self._data = outer.data
			self._colNum = colNum
		def __getitem__(self, index):
			row = self._data[index]
			value = row[self._colNum]
			return value	
		def __setitem__(self,key,value):
			row = self._data[key]
			row[self._colNum] = value



###########
# File IO #
###########

def loadCSV(inPath, lineParser = None):
	"""
	Function to read the CSV formated file at the given path and to output a
	RowListData object representing the data in the file.

	lineParser is an optional argument, which may be set to a function to perform
	the parsing. It must take a string representing a line, and output a list
	of values that were in that line. May be used to correctly parse different
	types of input, because by default all values read are stored as strings.
	"""
	inFile = open(inPath, 'r')
	firstLine = inFile.readline()
	labelList = None

	# test if this is a line defining column labels
	if firstLine[0] == "#":
		# strip '#' from the begining of the line
		scrubbedLine = firstLine[1:]
		# strip newline from end of line
		scrubbedLine = scrubbedLine.rstrip()
		labelList = scrubbedLine.split(',')
		labelMap = {}
		for name in labelList:
			labelMap[name] = labelList.index(name)
	#if not, get the iterator pointed back at the first line again	
	else:
		inFile.close()
		inFile = open(inPath, 'r')

	#list of datapoints in the file, where each data point is a list
	data = []
	for currLine in inFile:
		currLine = currLine.rstrip()
		#ignore empty lines
		if len(currLine) == 0:
			continue

		if lineParser != None:
			data.append(lineParser(currLine))
		else:
			currList = currLine.split(',')
			data.append(currList)

	if labelList == None:
		return RowListData(data)

	inFile.close()

	return RowListData(data,labelMap)


def writeToCSV(toWrite, outPath, includeLabels):
	"""
	Function to write the data in a RowListData to a CSV file at the designated
	path.

	toWrite is the RowListData to write to file. outPath is the location where
	we want to write the output file. includeLabels is boolean argument indicating
	whether the file should start with a comment line designating column labels.

	"""
	outFile = open(outPath, 'w')
	
	if includeLabels and toWrite.labels != None:
		pairs = toWrite.labels.items()
		# sort according to the value, not the key. ie sort by column number
		pairs = sorted(pairs,lambda (a,x),(b,y): x-y)
		for (a,x) in pairs:
			if pairs.index((a,x)) == 0:
				outFile.write('#')
			else:
				outFile.write(',')
			outFile.write(str(a))
		outFile.write('\n')

	for point in toWrite.data:
		first = True
		for value in point:
			if not first:
				outFile.write(',')		
			outFile.write(str(value))
			first = False
		outFile.write('\n')
	outFile.close()




