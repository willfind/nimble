"""
Class extending BaseData, using a numpy dense matrix to store data.

Outside of the class, functions are defined for reading and writing DenseMatrixData
to files.

"""

import numpy

from base_data import *
from copy import copy
from copy import deepcopy
from ..utility.custom_exceptions import ArgumentException

import random
import re
import os


class DenseMatrixData(BaseData):
	"""
	Class providing implementations of data manipulation operations on data stored
	in a numpy dense matrix.

	"""
	#TODO

	def __init__(self, data=None, labels=None):
		self.data = numpy.matrix(data)
		super(DenseMatrixData, self).__init__(labels)
		

	def _transpose_implementation(self):
		"""
		Function to transpose the data, ie invert the column and row indices of the data.
		
		This is not an in place operation, a new list of lists is constructed.
		"""
		self.data = self.data.getT()


	def _appendRows_implementation(self,toAppend):
		"""
		Append the rows from the toAppend object to the bottom of the columns in this object
		
		"""
		self.data = numpy.concatenate((self.data,toAppend.data),0)
		

	def _appendColumns_implementation(self,toAppend):
		"""
		Append the columns from the toAppend object to right ends of the rows in this object

		"""
		self.data = numpy.concatenate((self.data,toAppend.data),1)

	def _sortRows_implementation(self,cmp, key, reverse):
		""" 
		Modify this object so that the rows are sorted using the built in python
		sort. The input arguments are passed to that function unalterted

		"""
		print self.data

		self.data = numpy.sort(self.data,0)
		print self.data

	def _sortColumns_implementation(self,cmp, key, reverse):
		""" 
		Modify this object so that the columns are sorted using the built in python
		sort on column views. The input arguments are passed to that function unalterted
		This funciton returns a list of labels indicating the new order of the data.

		"""
		def passThrough(toKey):
			return toKey
		if key is None:
			key = passThrough	

		#TODO
		raise NotImplementedError

		#return new order

	def _extractRows_implementation(self,toExtract):
		"""
		Modify this object to have only the rows that are not given in the input,
		returning an object containing those rows that are.

		"""
		ret = self.data[toExtract]
		self.data = numpy.delete(self.data,toExtract,0)

		return DenseMatrixData(ret)

	def _extractColumns_implementation(self,toExtract):
		"""
		Modify this object to have only the columns that are not given in the input,
		returning an object containing those columns that are.

		"""
		ret = self.data[:,toExtract]
		self.data = numpy.delete(self.data,toExtract,1)

		return DenseMatrixData(ret)


	def _extractSatisfyingRows_implementation(self,function):
		"""
		Modify this object to have only the rows that do not satisfy the given function,
		returning an object containing those rows that do.

		"""
		results = numpy.apply_along_axis(function,1,self.data)
		ret = self.data[numpy.nonzero(results),:]
		# need to convert our boolean array to to list of rows to be removed	
		toRemove = []
		for i in xrange(len(results)):
			if results[i]:
				toRemove.append(i)
		self.data = numpy.delete(self.data,toRemove,0)

		return DenseMatrixData(ret)


	def _extractSatisfyingColumns_implementation(self,function):
		"""
		Modify this object to have only the columns whose views do not satisfy the given
		function, returning an object containing those columns whose views do.

		"""
		results = numpy.apply_along_axis(function,0,self.data)
		ret = self.data[:,results]
		# need to convert our boolean array to to list of columns to be removed			
		toRemove = []
		for i in xrange(len(results)):
			if results[i]:
				toRemove.append(i)

		return self.extractColumns(toRemove)

	def _extractRangeRows_implementation(self, start, end):
		"""
		Modify this object to have only those rows that are not within the given range,
		inclusive; returning an object containing those rows that are.
	
		start and end must not be null, must be within the range of possible rows,
		and start must not be greater than end

		"""
		# +1 on end in ranges, because our ranges are inclusive
		ret = self.data[start:end+1,:]
		self.data = numpy.delete(self.data, numpy.s_[start:end+1], 0)
		return DenseMatrixData(ret)


	def _extractRangeColumns_implementation(self, start, end):
		"""
		Modify this object to have only those columns that are not within the given range,
		inclusive; returning an object containing those rows that are.
	
		start and end must not be null, must be within the range of possible columns,
		and start must not be greater than end

		"""
		# +1 on end in ranges, because our ranges are inclusive
		ret = self.data[:,start:end+1]
		self.data = numpy.delete(self.data, numpy.s_[start:end+1], 1)
		return DenseMatrixData(ret)


	def _applyFunctionToEachRow_implementation(self,function):
		"""
		Applies the given funciton to each row in this object, collecting the
		output values into a new object in the shape of a row vector that is
		returned upon completion.

		"""
		retData = numpy.apply_along_axis(function,1,self.data)
		retData = numpy.matrix(retData)
		retData = retData.T
		return DenseMatrixData(retData)


	def _applyFunctionToEachColumn_implementation(self,function):
		"""
		Applies the given funciton to each column in this object, collecting the
		output values into a new object in the shape of a column vector that is
		returned upon completion.

		"""
		retData = numpy.apply_along_axis(function,0,self.data)
		return DenseMatrixData(retData)



	def _mapReduceOnRows_implementation(self, mapper, reducer):
		# apply_along_axis() expects a scalar or array of scalars as output,
		# but our mappers output a list of tuples (ie a sequence type)
		# which is not allowed. This packs key value pairs into an array
		def mapperWrapper(row):
			pairs = mapper(row)
			ret = []
			for (k,v) in pairs:
				ret.append(k)
				ret.append(v)
			return numpy.array(ret)

		mapResultsMatrix = numpy.apply_along_axis(mapperWrapper,1,self.data)
		mapResults = {}
		for pairsArray in mapResultsMatrix:
			for i in xrange(len(pairsArray)/2):
				# pairsArray has key value pairs packed back to back
				k = pairsArray[i * 2]
				v = pairsArray[(i * 2) +1]
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
				ret.append([redKey, redValue])
		return DenseMatrixData(ret)


	def _columns_implementation(self):
		shape = numpy.shape(self.data)
		return shape[1]

	def _rows_implementation(self):
		shape = numpy.shape(self.data)
		return shape[0]

	def _equals_implementation(self,other):
		if not isinstance(other,DenseMatrixData):
			return False
		if self.rows() != other.rows():
			return False
		if self.columns() != other.columns():
			return False
		return numpy.array_equal(self.data,other.data)


	def _convertToRowListData_implementation(self):
		"""	Returns a RowListData object with the same data and labels as this one """
		from row_list_data import RowListData as RLD
		return RLD(self.data.tolist(), self.labels)

	def _convertToDenseMatrixData_implementation(self):
		""" Returns a DenseMatrixData object with the same data and labels as this object """
		return DenseMatrixData(self.data, self.labels)


	###########
	# Helpers #
	###########



###########
# File IO #
###########

def loadCSV(inPath, lineParser = None):
	"""
	Function to read the CSV formated file at the given path and to output a
	DenseMatrixData object representing the data in the file.

	"""
	inFile = open(inPath, 'r')
	firstLine = inFile.readline()
	labelList = None
	skip_header = 0

	# test if this is a line defining column labels
	if firstLine[0] == "#":
		# strip '#' from the begining of the line
		scrubbedLine = firstLine[1:]
		# strip newline from end of line
		scrubbedLine = scrubbedLine.rstrip()
		labelList = scrubbedLine.split(',')
		skip_header = 1

	matrix = numpy.genfromtxt(inPath,delimiter=',',skip_header=skip_header)
	return DenseMatrixData(matrix,labelList)

def writeToCSV(toWrite, outPath, includeLabels):
	"""
	Function to write the data in a DenseMatrixData to a CSV file at the designated
	path.

	toWrite is the DenseMatrixData to write to file. outPath is the location where
	we want to write the output file. includeLabels is boolean argument indicating
	whether the file should start with a comment line designating column labels.

	"""
	header = None
	if includeLabels:
		labelString = "#"
		for i in xrange(toWrite.columns()):
			labelString += toWrite.labelsInverse[i]
			if not i == toWrite.columns() - 1:
				labelString += ','
		header = labelString

	outFile = open(outPath,'w')
	if header is not None:
		outFile.write(header + "\n")
	numpy.savetxt(outFile,toWrite.data,delimiter=',')
	outFile.close()



