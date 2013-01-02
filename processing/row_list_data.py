"""
Class extending BaseData, using a list of lists to store data.

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
	in a list of lists, representing a list of points of data. data is the list of
	lists. numFeatures is the integer number of features in each point.

	"""

	def __init__(self, data=None, featureNames=None):
		"""
		Instantiate a Row List data using the given data and featureNames. data may be
		none or an empty list to indicate an empty object, or a fully populated
		list of lists to be encased by this object. featureNames is passed up to
		the init funciton of BaseData, to be interpreted there.

		"""
		if data is None or len(data) == 0:
			self.numFeatures = 0
			self.data = []
			super(RowListData, self).__init__([])
		else:
			self.numFeatures= len(data[0])
			for point in data:
				if len(point) != self.numFeatures:
					raise ArgumentException("Points must be of equal size")
			self.data = data
			super(RowListData, self).__init__(featureNames)


	def _transpose_implementation(self):
		"""
		Function to transpose the data, ie invert the feature and point indices of the data.
		
		This is not an in place operation, a new list of lists is constructed.
		"""
		tempFeatures = len(self.data)
		transposed = []
		#load the new data with an empty point for each feature in the original
		for i in xrange(len(self.data[0])):
			transposed.append([])
		for point in self.data:
			for i in xrange(len(point)):
				transposed[i].append(point[i])
		
		self.data = transposed
		self.numFeatures= tempFeatures

	def _appendPoints_implementation(self,toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object
		
		"""
		for point in toAppend.data:
			self.data.append(point)
	
	def _appendFeatures_implementation(self,toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		"""	
		for i in xrange(self.points()):
			for value in toAppend.data[i]:
				self.data[i].append(value)
		self.numFeatures= self.numFeatures+ toAppend.features()

	def _sortPoints_implementation(self,cmp, key, reverse):
		""" 
		Modify this object so that the points are sorted using the built in python
		sort. The input arguments are passed to that function unalterted

		"""
		self.data.sort(cmp, key, reverse)

	def _sortFeatures_implementation(self,cmp, key, reverse):
		""" 
		Modify this object so that the features are sorted using the built in python
		sort on feature views. The input arguments are passed to that function unalterted.
		This funciton returns a list of featureNames indicating the new order of the data.

		"""
		def passThrough(toKey):
			return toKey
		if key is None:
			key = passThrough
		#create key list - to be sorted; and dictionary
		keyList = []
		temp = []
		for i in xrange(self.features()):
			ithView = self.FeatureView(self,i)
			keyList.append(key(ithView))
			temp.append(None)
		keyDict = {}
		for i in xrange(len(keyList)):
			keyDict[keyList[i]] = i
		keyList.sort(cmp,passThrough,reverse)

		#now modify data to correspond to the new order
		for point in self.data:
			#have to copy it out so we don't corrupt the point
			for i in xrange(self.features()):
				currKey = keyList[i]
				oldColNum = keyDict[currKey]
				temp[i] = point[oldColNum]
			for i in xrange(self.features()):
				point[i] = temp[i]

		# have to deal with featureNames now
		for i in xrange(self.features()):
			currKey = keyList[i]
			oldColNum = keyDict[currKey]
			temp[i] = self.featureNamesInverse[oldColNum]
		return temp

	def _extractPoints_implementation(self, toExtract, start, end, number, randomize):
		"""
		Function to extract points according to the parameters, and return an object containing
		the removed points with default feature names. The actual work is done by further helper
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
				toExtract = random.sample(toExtract, number)
			# else take the first number members of toExtract
			else:
				toExtract = toExtract[:number]
			return self._extractPointsByList_implementation(toExtract)
		# boolean function
		if hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError # TODO randomize in the extractPointByFunction case
			else:
				if number is None:
					number = self.points()		
				return self._extractPointsByFunction_implementation(toExtract, number)
		# by range
		if start is not None or end is not None:
			if start is None:
				start = 0
			if end is None:
				end = self.points()
			if number is None:
				number = end - start
			if randomize:
				return self.extactPointsByList(random.randrange(start,end,number))
			else:
				return self._extractPointsByRange_implementation(start, end)

	def _extractPointsByList_implementation(self, toExtract):
		"""
		Modify this object to have only the points that are not listed in toExtract,
		returning an object containing those points that are.

		"""
		toWrite = 0
		satisfying = []
		for i in xrange(self.points()):
			if i not in toExtract:
				self.data[toWrite] = self.data[i]
				toWrite += 1
			else:
				satisfying.append(self.data[i])

		# blank out the elements beyond our last copy, ie our last wanted point.
		for index in xrange(toWrite,len(self.data)):
			self.data.pop()

		return RowListData(satisfying)

	def _extractPointsByFunction_implementation(self, toExtract, number):
		"""
		Modify this object to have only the points that do not satisfy the given function,
		returning an object containing those points that do.

		"""
		toWrite = 0
		satisfying = []
		# walk through each point, copying the wanted points back to the toWrite index
		# toWrite is only incremented when we see a wanted point; unwanted points are copied
		# over
		for index in xrange(len(self.data)):
			point = self.data[index]
			if number > 0 and toExtract(point):			
				satisfying.append(point)
				number = number - 1
			else:
				self.data[toWrite] = point
				toWrite += 1

		# blank out the elements beyond our last copy, ie our last wanted point.
		for index in xrange(toWrite,len(self.data)):
			self.data.pop()

		return RowListData(satisfying)

	def _extractPointsByRange_implementation(self, start, end):
		"""
		Modify this object to have only those points that are not within the given range,
		inclusive; returning an object containing those points that are.

		"""
		toWrite = start
		inRange = []
		for i in xrange(start,self.points()):
			if i <= end:
				inRange.append(self.data[i])		
			else:
				self.data[toWrite] = self.data[i]
				toWrite += 1

		# blank out the elements beyond our last copy, ie our last wanted point.
		for index in xrange(toWrite,len(self.data)):
			self.data.pop()

		return RowListData(inRange)


	def _extractFeatures_implementation(self, toExtract, start, end, number, randomize):
		"""
		Function to extract features according to the parameters, and return an object containing
		the removed features with their featureNames from this object. The actual work is done by
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
				toExtract = random.sample(toExtract, number)
			# else take the first number members of toExtract
			else:
				toExtract = toExtract[:number]
			# convert IDs if necessary
			toExtractIndices = []
			for value in toExtract:
				toExtractIndices.append(self._getIndex(value))
			return self._extractFeaturesByList_implementation(toExtractIndices)	
		# boolean function
		if hasattr(toExtract, '__call__'):
			if randomize:
				#apply to each
				raise NotImplementedError # TODO randomize in the extractFeaturesByFunction case
			else:
				if number is None:
					number = self.points()		
				return self._extractFeaturesByFunction_implementation(toExtract, number)
		# by range
		if start is not None or end is not None:
			if start is None:
				start = 0
			if end is None:
				end = self.points()
			if number is None:
				number = end - start
			if randomize:
				return self.extactFeaturesByList(random.randrange(start,end,number))
			else:
				return self._extractFeaturesByRange_implementation(start, end)


	def _extractFeaturesByList_implementation(self, toExtract):
		"""
		Modify this object to have only the features that are not given in the input,
		returning an object containing those features that are, with the same featureNames
		they had previously. It does not modify the featureNames for the calling object.

		"""
		toExtract.sort()
		toExtract.reverse()
		extractedData = []
		for point in self.data:
			extractedPoint = []
			for index in toExtract:
				extractedPoint.append(point.pop(index))
			extractedPoint.reverse()
			extractedData.append(extractedPoint)

		self.numFeatures = self.numFeatures - len(toExtract)

		# construct featureName list
		featureNameList = []
		for index in toExtract:
			featureNameList.append(self.featureNamesInverse[index])
		# toExtract was reversed (for efficiency) so we have to rereverse this to get it right
		featureNameList.reverse()
		return RowListData(extractedData, featureNameList)


	def _extractFeaturesByFunction_implementation(self, function, number):
		"""
		Modify this object to have only the features whose views do not satisfy the given
		function, returning an object containing those features whose views do, with the
		same featureNames	they had previously. It does not modify the featureNames for the calling object.

		"""
		# all we're doing is making a list and calling extractFeaturesBy list, no need
		# deal with featureNames or the number of features.
		toExtract = []
		for i in xrange(self.features()):
			ithView = self.FeatureView(self,i)
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

		self.numFeatures = self.numFeatures- len(extractedPoint)

		# construct featureName list
		featureNameList = []
		for index in xrange(start,end+1):
			featureNameList.append(self.featureNamesInverse[index])
	
		return RowListData(extractedData, featureNameList)


	def _applyFunctionToEachPoint_implementation(self, function):
		"""
		Applies the given function to each point in this object, collecting the
		output values into a new object that is returned upon completion.

		"""
		retData = []
		for point in self.data:
			currOut = function(point)
			retData.append([currOut])
		return RowListData(retData)

	def _applyFunctionToEachFeature_implementation(self,function):
		"""
		Applies the given funciton to each feature in this object, collecting the
		output values into a new object in the shape of a feature vector that is
		returned upon completion.

		"""
		retData = [[]]
		for i in xrange(self.features()):
			ithView = self.FeatureView(self,i)
			currOut = function(ithView)
			retData[0].append(currOut)
		return RowListData(retData)


	def _mapReduceOnPoints_implementation(self, mapper, reducer):
		mapResults = {}
		# apply the mapper to each point in the data
		for point in self.data:
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
		return RowListData(ret)

	def _features_implementation(self):
		return self.numFeatures

	def _points_implementation(self):
		return len(self.data)

	def _equals_implementation(self,other):
		if not isinstance(other,RowListData):
			return False
		if self.points() != other.points():
			return False
		if self.features() != other.features():
			return False
		for index in xrange(self.points()):
			if self.data[index] != other.data[index]:
				return False
		return True


	def _convertToRowListData_implementation(self):
		"""	Returns a RowListData object with the same data and featureNames as this one """
		return RowListData(self.data, self.featureNames)

	def _convertToDenseMatrixData_implementation(self):
		""" Returns a DenseMatrixData object with the same data and featureNames as this object """
		from dense_matrix_data import DenseMatrixData as DMD
		return DMD(self.data, self.featureNames)


	###########
	# Helpers #
	###########

	class FeatureView():
		"""
		Class to simulate direct random access of a feature.

		"""
		def __init__(self, outer, colNum):
			self._data = outer.data
			self._colNum = colNum
		def __getitem__(self, index):
			point = self._data[index]
			value = point[self._colNum]
			return value	
		def __setitem__(self,key,value):
			point = self._data[key]
			point[self._colNum] = value



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
	featureNameList = None

	# test if this is a line defining featureNames
	if firstLine[0] == "#":
		# strip '#' from the begining of the line
		scrubbedLine = firstLine[1:]
		# strip newline from end of line
		scrubbedLine = scrubbedLine.rstrip()
		featureNameList = scrubbedLine.split(',')
		featureNameMap = {}
		for name in featureNameList:
			featureNameMap[name] = featureNameList.index(name)
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

	if featureNameList == None:
		return RowListData(data)

	inFile.close()

	return RowListData(data,featureNameMap)


def writeToCSV(toWrite, outPath, includeFeatureNames):
	"""
	Function to write the data in a RowListData to a CSV file at the designated
	path.

	toWrite is the RowListData to write to file. outPath is the location where
	we want to write the output file. includeFeatureNames is boolean argument indicating
	whether the file should start with a comment line designating featureNames.

	"""
	outFile = open(outPath, 'w')
	
	if includeFeatureNames and toWrite.featureNames != None:
		pairs = toWrite.featureNames.items()
		# sort according to the value, not the key. ie sort by feature number
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




