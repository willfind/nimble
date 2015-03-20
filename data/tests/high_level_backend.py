"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

"""

from copy import deepcopy
from nose.tools import *

import numpy
import tempfile

import UML
from UML.data import List
from UML.data import Matrix
from UML.data import Sparse
from UML.exceptions import ArgumentException, ImproperActionException

from UML.data.tests.baseObject import DataTestObject

from UML.randomness import numpyRandom

### Helpers used by tests in the test class ###

def simpleMapper(point):
	idInt = point[0]
	intList = []
	for i in xrange(1, len(point)):
		intList.append(point[i])
	ret = []
	for value in intList:
		ret.append((idInt,value))
	return ret

def simpleReducer(identifier, valuesList):
	total = 0
	for value in valuesList:
		total += value
	return (identifier,total)

def oddOnlyReducer(identifier, valuesList):
	if identifier % 2 == 0:
		return None
	return simpleReducer(identifier,valuesList)

def passThrough(value):
	return value

def plusOne(value):
	return (value + 1)

def plusOneOnlyEven(value):
	if value % 2 == 0:
		return (value + 1)
	else:
		return None


class HighLevelBackend(DataTestObject):

	###########################
	# dropFeaturesContainingType #
	###########################


	def test_dropFeaturesContainingType_emptyTest(self):
		""" Test dropFeaturesContainingType() when the data is empty """
		data = []
		toTest = self.constructor(data)
		unchanged = self.constructor(data)
		ret = toTest.dropFeaturesContainingType(basestring) # RET CHECK
		assert toTest.isIdentical(unchanged)
		assert ret is None
		
	def test_dropFeaturesContainingType_intoFEmpty(self):
		""" Test dropFeaturesContainingType() when dropping all features """
		data = [[1.0],[2.0]]
		toTest = self.constructor(data)
		toTest.dropFeaturesContainingType(float)

		exp = numpy.array([[],[]])
		exp = numpy.array(exp)
		exp = self.constructor(exp)

		assert toTest.isIdentical(exp)

	def test_dropFeaturesContainingType_ListOnlyTest(self):
		""" Test dropFeaturesContainingType() only on List data """
		data = [[1,2],[3,4]]
		toTest = self.constructor(data)
		stringData = [[5, 'six']]
		toAdd = UML.createData('List', stringData)
		if toTest.getTypeString() == 'List':
			toTest.appendPoints(toAdd)
			toTest.dropFeaturesContainingType(basestring)
			assert toTest.featureCount == 1


	#################################
	# replaceFeatureWithBinaryFeatures #
	#################################

	@raises(ImproperActionException)
	def test_replaceFeatureWithBinaryFeatures_PemptyException(self):
		""" Test replaceFeatureWithBinaryFeatures() with a point empty object """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)
		toTest.replaceFeatureWithBinaryFeatures(0)

	@raises(ArgumentException)
	def test_replaceFeatureWithBinaryFeatures_FemptyException(self):
		""" Test replaceFeatureWithBinaryFeatures() with a feature empty object """
		data = [[],[]]
		data = numpy.array(data)
		toTest = self.constructor(data)
		toTest.replaceFeatureWithBinaryFeatures(0)


	def test_replaceFeatureWithBinaryFeatures_handmade(self):
		""" Test replaceFeatureWithBinaryFeatures() against handmade output """
		data = [[1],[2],[3]]
		featureNames = ['col']
		toTest = self.constructor(data, featureNames=featureNames)
		getNames = self.constructor(data, featureNames=featureNames)
		ret = toTest.replaceFeatureWithBinaryFeatures(0) # RET CHECK

		expData = [[1,0,0], [0,1,0], [0,0,1]]
		expFeatureNames = []
		for point in getNames.pointIterator():
			expFeatureNames.append('col=' + str(point[0]))
		exp = self.constructor(expData, featureNames=expFeatureNames)

		assert toTest.isIdentical(exp)
		assert ret is None


	#############################
	# transformFeatureToIntegers #
	#############################

	@raises(ImproperActionException)
	def test_transformFeatureToIntegers_PemptyException(self):
		""" Test transformFeatureToIntegers() with an point empty object """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)
		toTest.transformFeatureToIntegers(0)

	@raises(ArgumentException)
	def test_transformFeatureToIntegers_FemptyException(self):
		""" Test transformFeatureToIntegers() with an feature empty object """
		data = [[],[]]
		data = numpy.array(data)
		toTest = self.constructor(data)
		toTest.transformFeatureToIntegers(0)

	def test_transformFeatureToIntegers_handmade(self):
		""" Test transformFeatureToIntegers() against handmade output """
		data = [[10],[20],[30.5],[20],[10]]
		featureNames = ['col']
		toTest = self.constructor(data, featureNames=featureNames)
		ret = toTest.transformFeatureToIntegers(0)  # RET CHECK

		assert toTest[0,0] == toTest[4,0]
		assert toTest[1,0] == toTest[3,0]
		assert toTest[0,0] != toTest[1,0]
		assert toTest[0,0] != toTest[2,0]
		assert ret is None

	def test_transformFeatureToIntegers_pointNames(self):
		""" Test transformFeatureToIntegers preserves pointNames """
		data = [[10],[20],[30.5],[20],[10]]
		pnames = ['1a', '2a', '3', '2b', '1b']
		fnames = ['col']
		toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
		ret = toTest.transformFeatureToIntegers(0)  # RET CHECK

		assert toTest.getPointName(0) == '1a'
		assert toTest.getPointName(1) == '2a'
		assert toTest.getPointName(2) == '3'
		assert toTest.getPointName(3) == '2b'
		assert toTest.getPointName(4) == '1b'
		assert ret is None

	def test_transformFeatureToIntegers_positioning(self):
		""" Test transformFeatureToIntegers preserves featurename mapping """
		data = [[10,0],[20,1],[30.5,2],[20,3],[10,4]]
		pnames = ['1a', '2a', '3', '2b', '1b']
		fnames = ['col','pos']
		toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
		ret = toTest.transformFeatureToIntegers(0)  # RET CHECK

		assert toTest[0,1] == toTest[4,1]
		assert toTest[1,1] == toTest[3,1]
		assert toTest[0,1] != toTest[1,1]
		assert toTest[0,1] != toTest[2,1]

		assert toTest[0,0] == 0
		assert toTest[1,0] == 1
		assert toTest[2,0] == 2
		assert toTest[3,0] == 3
		assert toTest[4,0] == 4

	#########################
	# extractPointsByCoinToss #
	#########################

#	@raises(ImproperActionException)
#	def test_extractPointsByCoinToss_exceptionEmpty(self):
#		""" Test extractPointsByCoinToss() for ImproperActionException when object is empty """
#		data = []
#		toTest = self.constructor(data)
#		toTest.extractPointsByCoinToss(0.5)

	@raises(ArgumentException)
	def test_extractPointsByCoinToss_exceptionNoneProbability(self):
		""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is None """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		featureNames = ['1','2','3']
		pointNames = ['1', '4', '7']
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		toTest.extractPointsByCoinToss(None)

	@raises(ArgumentException)
	def test_extractPointsByCoinToss_exceptionLEzero(self):
		""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is <= 0 """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		featureNames = ['1','2','3']
		pointNames = ['1', '4', '7']
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		toTest.extractPointsByCoinToss(0)

	@raises(ArgumentException)
	def test_extractPointsByCoinToss_exceptionGEone(self):
		""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is >= 1 """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		featureNames = ['1','2','3']
		pointNames = ['1', '4', '7']
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		toTest.extractPointsByCoinToss(1)

	def test_extractPointsByCoinToss_intoPEmpty(self):
		""" Test extractPointsByCoinToss() when it removes all points """
		data = [[1]]
		toTest = self.constructor(data)
		retExp = self.constructor(data)
		while True:
			ret = toTest.extractPointsByCoinToss(.99)
			if ret.pointCount == 1:
				break

		assert retExp.isIdentical(ret)

		data = [[]]
		data = numpy.array(data).T
		exp = self.constructor(data)

		assert toTest.isIdentical(exp)


	def test_extractPointsByCoinToss_handmade(self):
		""" Test extractPointsByCoinToss() produces sane results (ie a partition) """
		data = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]]
		featureNames = ['a','b','c']
		pointNames = ['1', '2', '3', '4', '5', '6']
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		orig = toTest.copy()
		ret = toTest.extractPointsByCoinToss(0.5)

		def checkEqual(v1, v2):
			assert len(v1) == len(v2)
			for i in range(len(v1)):
				assert v1[i] == v2[i]

		# everything in ret is in orig
		for pIndex in range(ret.pointCount):
			currRetPoint = ret.pointView(pIndex)
			currName = ret.getPointName(pIndex)
			currOrigPoint = orig.pointView(currName)
			checkEqual(currRetPoint, currOrigPoint)

		# everything in toTest is in orig
		for pIndex in range(toTest.pointCount):
			currToTestPoint = toTest.pointView(pIndex)
			currName = toTest.getPointName(pIndex)
			currOrigPoint = orig.pointView(currName)
			checkEqual(currToTestPoint, currOrigPoint)

		# everything in orig in either ret or toTest
		for pIndex in range(orig.pointCount):
			currOrigPoint = orig.pointView(pIndex)
			currName = orig.getPointName(pIndex)
			if currName in ret.getPointNames():
				assert currName not in toTest.getPointNames()
				checkPoint = ret.pointView(currName)
			else:
				assert currName in toTest.getPointNames()
				assert currName not in ret.getPointNames()
				checkPoint = toTest.pointView(currName)

			checkEqual(checkPoint, currOrigPoint)



	####################
	# applyToPoints() #
	####################

	@raises(ImproperActionException)
	def test_applyToPoints_exceptionPEmpty(self):
		""" Test applyToPoints() for ImproperActionException when object is point empty """
		data = [[],[]]
		data = numpy.array(data).T
		origObj = self.constructor(data)

		def emitLower(point):
			return point[origObj.getFeatureIndex('deci')]

		origObj.applyToPoints(emitLower, inPlace=False)

	@raises(ImproperActionException)
	def test_applyToPoints_exceptionFEmpty(self):
		""" Test applyToPoints() for ImproperActionException when object is feature empty """
		data = [[],[]]
		data = numpy.array(data)
		origObj = self.constructor(data)

		def emitLower(point):
			return point[origObj.getFeatureIndex('deci')]

		origObj.applyToPoints(emitLower, inPlace=False)

	@raises(ArgumentException)
	def test_applyToPoints_exceptionInputNone(self):
		""" Test applyToPoints() for ArgumentException when function is None """
		featureNames = {'number':0,'centi':2,'deci':1}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
		origObj.applyToPoints(None)

	def test_applyToPoints_Handmade(self):
		""" Test applyToPoints() with handmade output """
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitLower(point):
			return point[origObj.getFeatureIndex('deci')]

		lowerCounts = origObj.applyToPoints(emitLower, inPlace=False)

		expectedOut = [[0.1], [0.1], [0.1], [0.2]]
		exp = self.constructor(expectedOut)

		assert lowerCounts.isIdentical(exp)


	def test_applyToPoints_HandmadeLimited(self):
		""" Test applyToPoints() with handmade output on a limited portion of points """
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitLower(point):
			return point[origObj.getFeatureIndex('deci')]

		lowerCounts = origObj.applyToPoints(emitLower, points=['three',2], inPlace=False)

		expectedOut = [[0.1], [0.2]]
		exp = self.constructor(expectedOut)

		assert lowerCounts.isIdentical(exp)


	def test_applyToPoints_nonZeroItAndLen(self):
		""" Test applyToPoints() for the correct usage of the nonzero iterator """
		origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
		origObj = self.constructor(deepcopy(origData))

		def emitNumNZ(point):
			ret = 0
			assert len(point) == 3
			for value in point.nonZeroIterator():
				ret += 1
			return ret

		counts = origObj.applyToPoints(emitNumNZ, inPlace=False)

		expectedOut = [[3], [2], [2], [1]]
		exp = self.constructor(expectedOut)

		assert counts.isIdentical(exp)

	def test_applyToPoints_HandmadeInPlace(self):
		""" Test applyToPoints() with handmade output. InPlace """
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllDeci(point):
			value = point[origObj.getFeatureIndex('deci')]
			return [value, value, value]

		lowerCounts = origObj.applyToPoints(emitAllDeci) #RET CHECK

		expectedOut = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
		exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

		assert lowerCounts is None
		assert origObj.isIdentical(exp)

	def test_applyToPoints_HandmadeLimitedInPlace(self):
		""" Test applyToPoints() with handmade output on a limited portion of points. InPlace"""
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllDeci(point):
			value = point[origObj.getFeatureIndex('deci')]
			return [value, value, value]

		origObj.applyToPoints(emitAllDeci, points=[3,'two'])

		expectedOut = [[1,0.1,0.01], [1,0.1,0.02], [0.1,0.1,0.1], [0.2,0.2,0.2]]
		exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

		assert origObj.isIdentical(exp)


	def test_applyToPoints_nonZeroItAndLenInPlace(self):
		""" Test applyToPoints() for the correct usage of the nonzero iterator. InPlace """
		origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
		origObj = self.constructor(deepcopy(origData))

		def emitNumNZ(point):
			ret = 0
			assert len(point) == 3
			for value in point.nonZeroIterator():
				ret += 1
			return [ret, ret, ret]

		origObj.applyToPoints(emitNumNZ)

		expectedOut = [[3,3,3], [2,2,2], [2,2,2], [1,1,1]]
		exp = self.constructor(expectedOut)

		assert origObj.isIdentical(exp)




	#######################
	# applyToFeatures() #
	#######################

	@raises(ImproperActionException)
	def test_applyToFeatures_exceptionPEmpty(self):
		""" Test applyToFeatures() for ImproperActionException when object is point empty """
		data = [[],[]]
		data = numpy.array(data).T
		origObj= self.constructor(data)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return 0
			return 1

		origObj.applyToFeatures(emitAllEqual, inPlace=False)

	@raises(ImproperActionException)
	def test_applyToFeatures_exceptionFEmpty(self):
		""" Test applyToFeatures() for ImproperActionException when object is feature empty """
		data = [[],[]]
		data = numpy.array(data)
		origObj= self.constructor(data)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return 0
			return 1

		origObj.applyToFeatures(emitAllEqual, inPlace=False)

	@raises(ArgumentException)
	def test_applyToFeatures_exceptionInputNone(self):
		""" Test applyToFeatures() for ArgumentException when function is None """
		featureNames = {'number':0,'centi':2,'deci':1}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj= self.constructor(deepcopy(origData), featureNames=featureNames)
		origObj.applyToFeatures(None, inPlace=False)


	def test_applyToFeatures_Handmade(self):
		""" Test applyToFeatures() with handmade output """
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj= self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllEqual(feature):
			first = feature['zero']
			for value in feature:
				if value != first:
					return 0
			return 1

		lowerCounts = origObj.applyToFeatures(emitAllEqual, inPlace=False)
		expectedOut = [[1,0,0]]	
		exp = self.constructor(expectedOut)
		assert lowerCounts.isIdentical(exp)


	def test_applyToFeatures_HandmadeLimited(self):
		""" Test applyToFeatures() with handmade output on a limited portion of features """
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj= self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return 0
			return 1

		lowerCounts = origObj.applyToFeatures(emitAllEqual, features=[0,'centi'], inPlace=False)
		expectedOut = [[1,0]]
		exp = self.constructor(expectedOut)
		assert lowerCounts.isIdentical(exp)



	def test_applyToFeatures_nonZeroItAndLen(self):
		""" Test applyToFeatures() for the correct usage of the nonzero iterator """
		origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
		origObj = self.constructor(deepcopy(origData))

		def emitNumNZ(feature):
			ret = 0
			assert len(feature) == 4
			for value in feature.nonZeroIterator():
				ret += 1
			return ret

		counts = origObj.applyToFeatures(emitNumNZ, inPlace=False)

		expectedOut = [[3, 3, 2]]
		exp = self.constructor(expectedOut)

		assert counts.isIdentical(exp)


	def test_applyToFeatures_HandmadeInPlace(self):
		""" Test applyToFeatures() with handmade output. InPlace """
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj= self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return [0,0,0,0]
			return [1,1,1,1]

		lowerCounts = origObj.applyToFeatures(emitAllEqual) # RET CHECK
		expectedOut = [[1,0,0], [1,0,0], [1,0,0], [1,0,0]]	
		exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)
		
		assert lowerCounts is None
		assert origObj.isIdentical(exp)


	def test_applyToFeatures_HandmadeLimitedInPlace(self):
		""" Test applyToFeatures() with handmade output on a limited portion of features. InPlace """
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj= self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return [0,0,0,0]
			return [1,1,1,1]

		origObj.applyToFeatures(emitAllEqual, features=[0,'centi'])
		expectedOut = [[1,0.1,0], [1,0.1,0], [1,0.1,0], [1,0.2,0]]
		exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

		assert origObj.isIdentical(exp)


	def test_applyToFeatures_nonZeroItAndLenInPlace(self):
		""" Test applyToFeatures() for the correct usage of the nonzero iterator. InPlace """
		origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
		origObj = self.constructor(deepcopy(origData))

		def emitNumNZ(feature):
			ret = 0
			assert len(feature) == 4
			for value in feature.nonZeroIterator():
				ret += 1
			return [ret, ret, ret, ret]

		origObj.applyToFeatures(emitNumNZ)

		expectedOut = [[3,3,2], [3,3,2], [3,3,2], [3,3,2]]
		exp = self.constructor(expectedOut)

		assert origObj.isIdentical(exp)


	#####################
	# mapReducePoints() #
	#####################

	@raises(ImproperActionException)
	def test_mapReducePoints_argumentExceptionNoFeatures(self):
		""" Test mapReducePoints() for ImproperActionException when there are no features  """
		data = [[],[]]
		data = numpy.array(data)
		toTest = self.constructor(data)
		toTest.mapReducePoints(simpleMapper,simpleReducer)


	def test_mapReducePoints_emptyResultNoPoints(self):
		""" Test mapReducePoints() when given point empty data """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)
		ret = toTest.mapReducePoints(simpleMapper,simpleReducer)

		data = numpy.empty(shape=(0,0))
		exp = self.constructor(data)
		assert ret.isIdentical(exp)


	@raises(ArgumentException)
	def test_mapReducePoints_argumentExceptionNoneMap(self):
		""" Test mapReducePoints() for ArgumentException when mapper is None """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.mapReducePoints(None,simpleReducer)

	@raises(ArgumentException)
	def test_mapReducePoints_argumentExceptionNoneReduce(self):
		""" Test mapReducePoints() for ArgumentException when reducer is None """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.mapReducePoints(simpleMapper,None)

	@raises(ArgumentException)
	def test_mapReducePoints_argumentExceptionUncallableMap(self):
		""" Test mapReducePoints() for ArgumentException when mapper is not callable """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.mapReducePoints("hello",simpleReducer)

	@raises(ArgumentException)
	def test_mapReducePoints_argumentExceptionUncallableReduce(self):
		""" Test mapReducePoints() for ArgumentException when reducer is not callable """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.mapReducePoints(simpleMapper,5)


	def test_mapReducePoints_handmade(self):
		""" Test mapReducePoints() against handmade output """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		ret = toTest.mapReducePoints(simpleMapper,simpleReducer)
		
		exp = self.constructor([[1,5],[4,11],[7,17]])

		assert (ret.isIdentical(exp))
		assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))


	def test_mapReducePoints_handmadeNoneReturningReducer(self):
		""" Test mapReducePoints() against handmade output with a None returning Reducer """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		ret = toTest.mapReducePoints(simpleMapper,oddOnlyReducer)
		
		exp = self.constructor([[1,5],[7,17]])

		assert (ret.isIdentical(exp))
		assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))



	#######################
	# pointIterator() #
	#######################

	@raises(ImproperActionException)
	def test_pointIterator_exceptionFempty(self):
		""" Test pointIterator() for ImproperActionException when object is feature empty """
		data = [[],[]]
		data = numpy.array(data)
		toTest = self.constructor(data)
		toTest.pointIterator()

	def test_pointIterator_noNextPempty(self):
		""" test pointIterator() has no next value when object is point empty """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)
		viewIter = toTest.pointIterator()	
		try:
			viewIter.next()
		except StopIteration:
			return
		assert False

	def test_pointIterator_exactValueViaFor(self):
		""" Test pointIterator() gives views that contain exactly the correct data """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)

		viewIter = toTest.pointIterator()

		toCheck = []
		for v in viewIter:
			toCheck.append(v)

		assert toCheck[0][0] == 1
		assert toCheck[0][1] == 2
		assert toCheck[0][2] == 3
		assert toCheck[1][0] == 4
		assert toCheck[1][1] == 5
		assert toCheck[1][2] == 6
		assert toCheck[2][0] == 7
		assert toCheck[2][1] == 8
		assert toCheck[2][2] == 9

	def test_pointIterator_allZeroVectors(self):
		""" Test pointIterator() works when there are all zero points """
		data = [[0,0,0],[4,5,6],[0,0,0],[7,8,9],[0,0,0],[0,0,0]]
		toTest = self.constructor(data)

		viewIter = toTest.pointIterator()
		toCheck = []
		for v in viewIter:
			toCheck.append(v)

		assert len(toCheck) == toTest.pointCount

		assert toCheck[0][0] == 0
		assert toCheck[0][1] == 0
		assert toCheck[0][2] == 0

		assert toCheck[1][0] == 4
		assert toCheck[1][1] == 5
		assert toCheck[1][2] == 6

		assert toCheck[2][0] == 0
		assert toCheck[2][1] == 0
		assert toCheck[2][2] == 0

		assert toCheck[3][0] == 7
		assert toCheck[3][1] == 8
		assert toCheck[3][2] == 9

		assert toCheck[4][0] == 0
		assert toCheck[4][1] == 0
		assert toCheck[4][2] == 0

		assert toCheck[5][0] == 0
		assert toCheck[5][1] == 0
		assert toCheck[5][2] == 0





	#########################
	# featureIterator() #
	#########################

	@raises(ImproperActionException)
	def test_featureIterator_exceptionPempty(self):
		""" Test featureIterator() for ImproperActionException when object is point empty """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)
		toTest.featureIterator()

	def test_featureIterator_noNextFempty(self):
		""" test featureIterator() has no next value when object is feature empty """
		data = [[],[]]
		data = numpy.array(data)
		toTest = self.constructor(data)
		viewIter = toTest.featureIterator()	
		try:
			viewIter.next()
		except StopIteration:
			return
		assert False


	def test_featureIterator_exactValueViaFor(self):
		""" Test featureIterator() gives views that contain exactly the correct data """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		
		viewIter = toTest.featureIterator()

		toCheck = []
		for v in viewIter:
			toCheck.append(v)

		assert toCheck[0][0] == 1
		assert toCheck[0][1] == 4
		assert toCheck[0][2] == 7
		assert toCheck[1][0] == 2
		assert toCheck[1][1] == 5
		assert toCheck[1][2] == 8
		assert toCheck[2][0] == 3
		assert toCheck[2][1] == 6
		assert toCheck[2][2] == 9


	def test_featureIterator_allZeroVectors(self):
		""" Test featureIterator() works when there are all zero points """
		data = [[0,1,0,2,0,3,0,0],[0,4,0,5,0,6,0,0],[0,7,0,8,0,9,0,0]]
		toTest = self.constructor(data)

		viewIter = toTest.featureIterator()
		toCheck = []
		for v in viewIter:
			toCheck.append(v)

		assert len(toCheck) == toTest.featureCount
		assert toCheck[0][0] == 0
		assert toCheck[0][1] == 0
		assert toCheck[0][2] == 0

		assert toCheck[1][0] == 1
		assert toCheck[1][1] == 4
		assert toCheck[1][2] == 7

		assert toCheck[2][0] == 0
		assert toCheck[2][1] == 0
		assert toCheck[2][2] == 0

		assert toCheck[3][0] == 2
		assert toCheck[3][1] == 5
		assert toCheck[3][2] == 8

		assert toCheck[4][0] == 0
		assert toCheck[4][1] == 0
		assert toCheck[4][2] == 0

		assert toCheck[5][0] == 3
		assert toCheck[5][1] == 6
		assert toCheck[5][2] == 9

		assert toCheck[6][0] == 0
		assert toCheck[6][1] == 0
		assert toCheck[6][2] == 0

		assert toCheck[7][0] == 0
		assert toCheck[7][1] == 0
		assert toCheck[7][2] == 0



	#####################################
	# applyToElements() #
	#####################################

	def test_applyToElements_passthrough(self):
		""" test applyToElements can construct a list by just passing values through  """

		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		ret = toTest.applyToElements(passThrough, inPlace=False)
		retRaw = ret.copyAs(format="python list")

		assert [1,2,3] in retRaw
		assert [4,5,6] in retRaw
		assert [7,8,9] in retRaw

		ret = toTest.applyToElements(passThrough) # RET CHECK
		assert ret is None
		retRaw = toTest.copyAs(format="python list")

		assert [1,2,3] in retRaw
		assert [4,5,6] in retRaw
		assert [7,8,9] in retRaw


	def test_applyToElements_plusOnePreserve(self):
		""" test applyToElements can modify elements other than zero  """

		data = [[1,0,3],[0,5,6],[7,0,9]]
		toTest = self.constructor(data)
		ret = toTest.applyToElements(plusOne, inPlace=False, preserveZeros=True)
		retRaw = ret.copyAs(format="python list")

		assert [2,0,4] in retRaw
		assert [0,6,7] in retRaw
		assert [8,0,10] in retRaw

		toTest.applyToElements(plusOne, preserveZeros=True)
		retRaw = toTest.copyAs(format="python list")

		assert [2,0,4] in retRaw
		assert [0,6,7] in retRaw
		assert [8,0,10] in retRaw


	def test_applyToElements_plusOneExclude(self):
		""" test applyToElements() skipNoneReturnValues flag  """

		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		ret = toTest.applyToElements(plusOneOnlyEven, inPlace=False, skipNoneReturnValues=True)
		retRaw = ret.copyAs(format="python list")

		assert [1,3,3] in retRaw
		assert [5,5,7] in retRaw
		assert [7,9,9] in retRaw

		toTest.applyToElements(plusOneOnlyEven, skipNoneReturnValues=True)
		retRaw = toTest.copyAs(format="python list")

		assert [1,3,3] in retRaw
		assert [5,5,7] in retRaw
		assert [7,9,9] in retRaw


	def test_applyToElements_plusOneLimited(self):
		""" test applyToElements() on limited portions of the points and features """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		names = ['one','two','three']
		pnames = ['1', '4', '7']
		toTest = self.constructor(data, pointNames=pnames, featureNames=names)

		ret = toTest.applyToElements(plusOneOnlyEven, points='4', features=[1,'three'], inPlace=False, skipNoneReturnValues=True)
		retRaw = ret.copyAs(format="python list")

		assert [5,7] in retRaw

		toTest.applyToElements(plusOneOnlyEven, points=1, features=[1,'three'], skipNoneReturnValues=True)
		retRaw = toTest.copyAs(format="python list")

		assert [1,2,3] in retRaw
		assert [4,5,7] in retRaw
		assert [7,8,9] in retRaw


	########################
	# isApproximatelyEqual() #
	########################

	def test_isApproximatelyEqual_randomTest(self):
		""" Test isApproximatelyEqual() using randomly generated data """

		for x in xrange(2):
			points = 100
			features = 40
			data = numpy.zeros((points,features))

			for i in xrange(points):
				for j in xrange(features):
					data[i,j] = numpyRandom.rand() * numpyRandom.randint(0,5)

			toTest = self.constructor(data)

			listObj = List(data)
			matrix = Matrix(data)
			sparse = Sparse(data)

			assert toTest.isApproximatelyEqual(listObj)
			assert toTest.hashCode() == listObj.hashCode()

			assert toTest.isApproximatelyEqual(matrix)
			assert toTest.hashCode() == matrix.hashCode()

			assert toTest.isApproximatelyEqual(sparse)
			assert toTest.hashCode() == sparse.hashCode()



	###################
	# shufflePoints() #
	###################

	@raises(ArgumentException)
	def test_shufflePoints_exceptionIndicesPEmpty(self):
		""" tests shufflePoints() throws an ArgumentException when given invalid indices """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)
		toTest.shufflePoints([1,3])


	def test_shufflePoints_noLongerEqual(self):
		""" Tests shufflePoints() results in a changed object """
		data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
		toTest = self.constructor(deepcopy(data))
		toCompare = self.constructor(deepcopy(data))

		# it is possible that it shuffles it into the same configuration.
		# the odds are vanishingly low that it will do so over consecutive calls
		# however. We will pass as long as it changes once
		returns = []
		for i in xrange(5):
			ret = toTest.shufflePoints() # RET CHECK
			returns.append(ret)
			if not toTest.isApproximatelyEqual(toCompare):
				break

		assert not toTest.isApproximatelyEqual(toCompare)

		for ret in returns:
			assert ret is None



	#####################
	# shuffleFeatures() #
	#####################

	@raises(ArgumentException)
	def test_shuffleFeatures_exceptionIndicesFEmpty(self):
		""" tests shuffleFeatures() throws an ArgumentException when given invalid indices """
		data = [[],[]]
		data = numpy.array(data)
		toTest = self.constructor(data)
		toTest.shuffleFeatures([1,3])

	def test_shuffleFeatures_noLongerEqual(self):
		""" Tests shuffleFeatures() results in a changed object """
		data = [[1,2,3,33],[4,5,6,66],[7,8,9,99],[10,11,12,1111111]]
		toTest = self.constructor(deepcopy(data))
		toCompare = self.constructor(deepcopy(data))

		# it is possible that it shuffles it into the same configuration.
		# the odds are vanishly low that it will do so over consecutive calls
		# however. We will pass as long as it changes once
		returns = []
		for i in xrange(5):
			ret = toTest.shuffleFeatures() # RET CHECK
			returns.append(ret)
			if not toTest.isApproximatelyEqual(toCompare):
				break

		assert not toTest.isApproximatelyEqual(toCompare)

		for ret in returns:
			assert ret is None



	######################
	# trainAndTestSets() #
	######################

	# simple sucess - no labels
	def test_trainAndTestSets_simple_nolabels(self):
		data = [[1,5,-1,3,33],[2,5,-2,6,66],[3,5,-2,9,99],[4,5,-4,12,111]]
		featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
		toTest = self.constructor(data, featureNames=featureNames)

		trX, teX = toTest.trainAndTestSets(.5)

		assert trX.pointCount == 2
		assert trX.featureCount == 5
		assert teX.pointCount == 2
		assert teX.featureCount == 5

	# simple sucess - single label
	def test_trainAndTestSets_simple_singlelabel(self):
		data = [[1,5,-1,3,33],[2,5,-2,6,66],[3,5,-2,9,99],[4,5,-4,12,111]]
		featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
		toTest = self.constructor(data, featureNames=featureNames)

		trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=0)

		assert trX.pointCount == 2
		assert trX.featureCount == 4
		assert trY.pointCount == 2
		assert trY.featureCount == 1
		assert teX.pointCount == 2
		assert teX.featureCount == 4
		assert teY.pointCount == 2
		assert teY.featureCount == 1

	# simple sucess - multi label
	def test_trainAndTestSets_simple_multilabel(self):
		data = [[1,5,-1,3,33],[2,5,-2,6,66],[3,5,-2,9,99],[4,5,-4,12,111]]
		featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
		toTest = self.constructor(data, featureNames=featureNames)

		trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=[0, 'labs2'])

		assert trX.pointCount == 2
		assert trX.featureCount == 3
		assert trY.pointCount == 2
		assert trY.featureCount == 2
		assert teX.pointCount == 2
		assert teX.featureCount == 3
		assert teY.pointCount == 2
		assert teY.featureCount == 2

	# edge cases 0/1 test portions
	def test_trainAndTestSets_0or1_testPortion(self):
		data = [[1,2,3,33],[2,5,6,66],[3,8,9,99],[4,11,12,111]]
		toTest = self.constructor(data)

		trX, trY, teX, teY = toTest.trainAndTestSets(0, 0)

		assert trX.pointCount == 4
		assert trY.pointCount == 4
		assert teX.pointCount == 0
		assert teY.pointCount == 0

		trX, trY, teX, teY = toTest.trainAndTestSets(1, 0)

		assert trX.pointCount == 0
		assert trY.pointCount == 0
		assert teX.pointCount == 4
		assert teY.pointCount == 4

	# each returned set independant of calling set
	def test_trainAndTestSets_unconnectedReturn(self):
		data = [[1,1],[2,2],[3,3],[4,4]]
		toTest = self.constructor(data)

		trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0, randomOrder=False)

		assert trX == trY
		assert teX == teY

		def changeFirst(point):
			ret = []
			first = True
			for val in point:
				if first:
					ret.append(-val)
					first = False
				else:
					ret.append(val)
			return ret

		# change the origin data
		toTest.applyToPoints(changeFirst)
		assert toTest[0,0] == -1

		# assert our returned sets are unaffected
		assert trX == trY
		assert teX == teY

	def test_trainAndTestSets_nameAppend_PathPreserve(self):
		data = [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
		toTest = self.constructor(data,)
		tmpFile = tempfile.NamedTemporaryFile(suffix='.csv')
		toTest.writeFile(tmpFile.name, format='csv')

		toTest = self.constructor(tmpFile.name, name='toTest')

		trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0)

		assert trX.name == 'toTest trainX'
		assert trX.path == tmpFile.name
		assert trY.name == 'toTest trainY'
		assert trY.path == tmpFile.name
		assert teX.name == 'toTest testX'
		assert teX.path == tmpFile.name
		assert teY.name == 'toTest testY'
		assert teY.path == tmpFile.name


	def test_trainAndTestSets_PandFnamesPerserved(self):
		data = [[1,5,-1,3,33],[2,5,-2,6,66],[3,5,-2,9,99],[4,5,-4,12,111]]
		pnames = ['one', 'two', 'three', 'four']
		fnames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
		toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

		trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=0, randomOrder=False)

		assert trX.getPointNames() == ['one', 'two']
		assert trX.getFeatureNames() == ['fives', 'labs2', 'bozo', 'long']

		assert trY.getPointNames() == ['one', 'two']
		assert trY.getFeatureNames() == ['labs1']

		assert teX.getPointNames() == ['three', 'four']
		assert teX.getFeatureNames() == ['fives', 'labs2', 'bozo', 'long']

		assert teY.getPointNames() == ['three', 'four']
		assert teY.getFeatureNames() == ['labs1']


	def test_trainAndTestSets_randomOrder(self):
		data = [[1,1],[2,2],[3,3],[4,4]]
		toTest = self.constructor(data)

		for i in xrange(100):
			trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0, randomOrder=False)

			assert trX == trY
			assert trX[0] == 1
			assert trX[1] == 2

			assert teX == teY
			assert teX[0] == 3
			assert teX[1] == 4

		for i in xrange(100):
			trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0)

			# just to make sure everything looks right
			assert trX == trY
			assert teX == teY
			# this means point ordering was randomized, so we return successfully
			if trX[0] != 1:
				return

		assert False  # implausible number of checks for random order were unsucessful
