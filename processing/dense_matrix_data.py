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

from scipy.io import mmwrite
from scipy.sparse import isspmatrix
import random
import re
import os


class DenseMatrixData(BaseData):
	"""
	Class providing implementations of data manipulation operations on data stored
	in a numpy dense matrix.

	"""

	def __init__(self, data=None, featureNames=None):
		if isspmatrix(data):
			self.data = data.todense()
		else:
			self.data = numpy.matrix(data)

		# check for any strings in the data
		(x,y) = self.data.shape
		for i in xrange(x):
			for j in xrange(y):
				if isinstance(self.data[i,j], basestring):
					raise ArgumentException("DenseMatrixData does not accept strings in the input")

		super(DenseMatrixData, self).__init__(featureNames)
		

	def _transpose_implementation(self):
		"""
		Function to transpose the data, ie invert the feature and point indices of the data.
		
		This is not an in place operation, a new list of lists is constructed.
		"""
		self.data = self.data.getT()


	def _appendPoints_implementation(self,toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object
		
		"""
		self.data = numpy.concatenate((self.data,toAppend.data),0)
		

	def _appendFeatures_implementation(self,toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		"""
		self.data = numpy.concatenate((self.data,toAppend.data),1)

	def _sortPoints_implementation(self,cmp, key, reverse):
		""" 
		Modify this object so that the points are sorted using the built in python
		sort. The input arguments are passed to that function unalterted

		"""
		print self.data

		self.data = numpy.sort(self.data,0)
		print self.data

	def _sortFeatures_implementation(self,cmp, key, reverse):
		""" 
		Modify this object so that the features are sorted using the built in python
		sort on feature views. The input arguments are passed to that function unalterted
		This funciton returns a list of featureNames indicating the new order of the data.

		"""
		def passThrough(toKey):
			return toKey
		if key is None:
			key = passThrough	

		#TODO
		raise NotImplementedError

		#return new order

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
			if number is None:
				number = end - start
			if randomize:
				toExtract = random.sample(xrange(start,end),number)
				toExtract.sort()
				return self._extractPointsByList_implementation(toExtract)
			else:
				return self._extractPointsByRange_implementation(start, end)


	def _extractPointsByList_implementation(self, toExtract):
		"""
		Modify this object to have only the points that are not given in the input,
		returning an object containing those points that are.

		"""
		ret = self.data[toExtract]
		self.data = numpy.delete(self.data,toExtract,0)

		return DenseMatrixData(ret)

	def _extractPointsByFunction_implementation(self, toExtract, number):
		"""
		Modify this object to have only the points that do not satisfy the given function,
		returning an object containing those points that do.

		"""
		results = numpy.apply_along_axis(toExtract,1,self.data)
		ret = self.data[numpy.nonzero(results),:]
		# need to convert our boolean array to to list of points to be removed	
		toRemove = []
		for i in xrange(len(results)):
			if results[i]:
				toRemove.append(i)
		self.data = numpy.delete(self.data,toRemove,0)

		return DenseMatrixData(ret)

	def _extractPointsByRange_implementation(self, start, end):
		"""
		Modify this object to have only those points that are not within the given range,
		inclusive; returning an object containing those points that are.
	
		"""
		# +1 on end in ranges, because our ranges are inclusive
		ret = self.data[start:end+1,:]
		self.data = numpy.delete(self.data, numpy.s_[start:end+1], 0)
		return DenseMatrixData(ret)

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
		if isinstance(toExtract, int):
			toExtract = [toExtract]	
		# list of identifiers
		if isinstance(toExtract, list):
			if number is None:
				number = len(toExtract)
			# if randomize, use random sample
			if randomize:
				raise NotImplementedError # TODO implement using sample(), but without losing the extraction order
				#toExtract = random.sample(toExtract, number)
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
				raise NotImplementedError # TODO randomize in the extractPointByFunction case
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
				toExtract = random.sample(xrange(start,end),number)
				toExtract.sort()
				return self._extractFeaturesByList_implementation(toExtract)
			else:
				return self._extractFeaturesByRange_implementation(start, end)

	def _extractFeaturesByList_implementation(self, toExtract):
		"""
		Modify this object to have only the features that are not given in the input,
		returning an object containing those features that are.

		"""
		ret = self.data[:,toExtract]
		self.data = numpy.delete(self.data,toExtract,1)

		# construct featureName list
		featureNameList = []
		for index in toExtract:
			featureNameList.append(self.featureNamesInverse[index])

		return DenseMatrixData(ret, featureNameList)

	def _extractFeaturesByFunction_implementation(self, toExtract, number):
		"""
		Modify this object to have only the features whose views do not satisfy the given
		function, returning an object containing those features whose views do.

		"""
		results = numpy.apply_along_axis(toExtract, 0, self.data)
		ret = self.data[:,results]
		# need to convert our boolean array to to list of features to be removed			
		toRemove = []
		for i in xrange(len(results)):
			if results[i]:
				toRemove.append(i)

		return self._extractFeaturesByList_implementation(toRemove)


	def _extractFeaturesByRange_implementation(self, start, end):
		"""
		Modify this object to have only those features that are not within the given range,
		inclusive; returning an object containing those features that are.
	
		start and end must not be null, must be within the range of possible features,
		and start must not be greater than end

		"""
		# +1 on end in ranges, because our ranges are inclusive
		ret = self.data[:,start:end+1]
		self.data = numpy.delete(self.data, numpy.s_[start:end+1], 1)

		# construct featureName list
		featureNameList = []
		for index in xrange(start,end+1):
			featureNameList.append(self.featureNamesInverse[index])

		return DenseMatrixData(ret, featureNameList)


	def _applyFunctionToEachPoint_implementation(self,function):
		"""
		Applies the given funciton to each point in this object, collecting the
		output values into a new object that is returned upon completion.

		"""
		retData = numpy.apply_along_axis(function,1,self.data)
		retData = numpy.matrix(retData)
		retData = retData.T
		return DenseMatrixData(retData)


	def _applyFunctionToEachFeature_implementation(self,function):
		"""
		Applies the given funciton to each feature in this object, collecting the
		output values into a new object in the shape of a feature vector that is
		returned upon completion.

		"""
		retData = numpy.apply_along_axis(function,0,self.data)
		return DenseMatrixData(retData)



	def _mapReduceOnPoints_implementation(self, mapper, reducer):
		# apply_along_axis() expects a scalar or array of scalars as output,
		# but our mappers output a list of tuples (ie a sequence type)
		# which is not allowed. This packs key value pairs into an array
		def mapperWrapper(point):
			pairs = mapper(point)
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


	def _features_implementation(self):
		shape = numpy.shape(self.data)
		return shape[1]

	def _points_implementation(self):
		shape = numpy.shape(self.data)
		return shape[0]

	def _equals_implementation(self,other):
		if not isinstance(other,DenseMatrixData):
			return False
		if self.points() != other.points():
			return False
		if self.features() != other.features():
			return False
		return numpy.array_equal(self.data,other.data)


	def _toRowListData_implementation(self):
		"""	Returns a RowListData object with the same data and featureNames as this one """
		from row_list_data import RowListData as RLD
		return RLD(self.data.tolist(), self.featureNames)

	def _toDenseMatrixData_implementation(self):
		""" Returns a DenseMatrixData object with the same data and featureNames as this object """
		return DenseMatrixData(self.data, self.featureNames)


	def _writeFileCSV_implementation(self, outPath, includeFeatureNames):
		"""
		Function to write the data in this object to a CSV file at the designated
		path.

		"""
		header = None
		if includeFeatureNames:
			featureNameString = "#"
			for i in xrange(self.features()):
				featureNameString += self.featureNamesInverse[i]
				if not i == self.features() - 1:
					featureNameString += ','
			header = featureNameString

		outFile = open(outPath,'w')
		if header is not None:
			outFile.write(header + "\n")
		numpy.savetxt(outFile,self.data,delimiter=',')
		outFile.close()

	def _writeFileMTX_implementation(self, outPath, includeFeatureNames):
		if includeFeatureNames:
			featureNameString = "#"
			for i in xrange(self.features()):
				featureNameString += self.featureNamesInverse[i]
				if not i == self.features() - 1:
					featureNameString += ','
			
			mmwrite(target=outPath, a=self.data, comment=featureNameString)		
		else:
			mmwrite(target=outPath, a=self.data)

	def _copyReferences_implementation(self, other):
		if not isinstance(other, DenseMatrixData):
			raise ArgumentException("Other must be the same type as this object")

		self.data = other.data

	def _duplicate_implementation(self):
		return DenseMatrixData(deepcopy(self.data), deepcopy(self.featureNames))

	def _copyPoints_implementation(self, points):
		ret = self.data[points]

		return DenseMatrixData(ret)

	def _copyFeatures_implementation(self, indices):
		ret = self.data[:,indices]

		# construct featureName list
		featureNameList = []
		for index in indices:
			featureNameList.append(self.featureNamesInverse[index])

		return DenseMatrixData(ret, featureNameList)


