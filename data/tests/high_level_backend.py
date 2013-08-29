"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

"""

from copy import deepcopy

import numpy

import UML
from UML.data import List
from UML.data import Matrix
from UML.data import Sparse


###########################
# dropFeaturesContainingType #
###########################


def dropFeaturesContainingType_emptyTest(constructor):
	""" Test dropFeaturesContainingType() when the data is empty """
	data = []
	toTest = constructor(data)
	unchanged = constructor(data)
	ret = toTest.dropFeaturesContainingType(basestring)
	assert toTest.isIdentical(unchanged)
	assert toTest == ret


def dropFeaturesContainingType_ListOnlyTest(constructor):
	""" Test dropFeaturesContainingType() only on List data """
	data = [[1,2],[3,4]]
	toTest = constructor(data)
	stringData = [[5, 'six']]
	toAdd = UML.createData('List', stringData)
	if toTest.getTypeString() == 'List':
		toTest.appendPoints(toAdd)
		toTest.dropFeaturesContainingType(basestring)
		assert toTest.features() == 1


#################################
# replaceFeatureWithBinaryFeatures #
#################################


def replaceFeatureWithBinaryFeatures_PemptyException(constructor):
	""" Test replaceFeatureWithBinaryFeatures() with a point empty object """
	data = []
	toTest = constructor(data)
	toTest.replaceFeatureWithBinaryFeatures(0)


#def replaceFeatureWithBinaryFeatures_exception


def replaceFeatureWithBinaryFeatures_handmade(constructor):
	""" Test replaceFeatureWithBinaryFeatures() against handmade output """
	data = [[1],[2],[3]]
	featureNames = ['col']
	toTest = constructor(data,featureNames)
	ret = toTest.replaceFeatureWithBinaryFeatures(0)

	expData = [[1,0,0], [0,1,0], [0,0,1]]
	expFeatureNames = ['col=1','col=2','col=3']
	exp = constructor(expData, expFeatureNames)

#	import pdb
#	pdb.set_trace()
#	print toTest.featureNames
	assert toTest.isIdentical(exp)
	assert toTest == ret


#############################
# transformFeartureToIntegerFeature #
#############################

def transformFeartureToIntegerFeature_emptyException(constructor):
	""" Test transformFeartureToIntegerFeature() with an empty object """
	data = []
	toTest = constructor(data)
	toTest.transformFeartureToIntegerFeature(0)

def transformFeartureToIntegerFeature_handmade(constructor):
	""" Test transformFeartureToIntegerFeature() against handmade output """
	data = [[10],[20],[30.5],[20],[10]]
	featureNames = ['col']
	toTest = constructor(data,featureNames)
	ret = toTest.transformFeartureToIntegerFeature(0)

	assert toTest.data[0] == toTest.data[4]
	assert toTest.data[1] == toTest.data[3]
	assert toTest.data[0] != toTest.data[1]
	assert toTest.data[0] != toTest.data[2]
	assert toTest == ret



###############################
# selectConstantOfPointsByValue #
###############################

def selectConstantOfPointsByValue_exceptionNumToSelectNone(constructor):
	""" Test selectConstantOfPointsByValue() for Argument exception when numToSelect is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfPointsByValue(None,'1')

def selectConstantOfPointsByValue_exceptionNumToSelectLEzero(constructor):
	""" Test selectConstantOfPointsByValue() for Argument exception when numToSelect <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfPointsByValue(0,'1')

def selectConstantOfPointsByValue_handmade(constructor):
	""" Test selectConstantOfPointsByValue() against handmade output """
	data = [[1,2,3],[1,5,6],[1,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfPointsByValue(2,'1')

	expRet = constructor([[1,2,3],[1,8,9]],featureNames)
	expTest = constructor([[1,5,6],],featureNames)

	assert ret.isIdentical(expRet)
	assert expTest.isIdentical(toTest)

def selectConstantOfPointsByValue_handmadeLimit(constructor):
	""" Test selectConstantOfPointsByValue() against handmade output when the constant exceeds the available points """
	data = [[1,2,3],[1,5,6],[1,8,9],[2,11,12]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfPointsByValue(2,'1')

	expRet = constructor([[1,2,3],[1,8,9],[2,11,12]],featureNames)
	expTest = constructor([[1,5,6],],featureNames)

	assert ret.isIdentical(expRet)
	assert expTest.isIdentical(toTest)



##############################
# selectPercentOfPointsByValue #
##############################

def selectPercentOfPointsByValue_exceptionPercentNone(constructor):
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfPointsByValue(None,'1')

def selectPercentOfPointsByValue_exceptionPercentZero(constructor):
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfPointsByValue(0,'1')

def selectPercentOfPointsByValue_exceptionPercentOneHundrend(constructor):
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is >= 100 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfPointsByValue(100,'1')

def selectPercentOfPointsByValue_handmade(constructor):
	""" Test selectPercentOfPointsByValue() against handmade output """
	data = [[1,2,3],[1,5,6],[1,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfPointsByValue(50,'1')

	expRet = constructor([[1,2,3]],featureNames)
	expTest = constructor([[1,5,6],[1,8,9]],featureNames)

	assert ret.isIdentical(expRet)
	assert expTest.isIdentical(toTest)


#########################
# extractPointsByCoinToss #
#########################

def extractPointsByCoinToss_exceptionEmpty(constructor):
	""" Test extractPointsByCoinToss() for ImproperActionException when object is empty """
	data = []
	toTest = constructor(data)
	toTest.extractPointsByCoinToss(0.5)

def extractPointsByCoinToss_exceptionNoneProbability(constructor):
	""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.extractPointsByCoinToss(None)

def extractPointsByCoinToss_exceptionLEzero(constructor):
	""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.extractPointsByCoinToss(0)

def extractPointsByCoinToss_exceptionGEone(constructor):
	""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is >= 1 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.extractPointsByCoinToss(1)

def extractPointsByCoinToss_handmade(constructor):
	""" Test extractPointsByCoinToss() against handmade output with the test seed """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	ret = toTest.extractPointsByCoinToss(0.5)

	expRet = constructor([[4,5,6],[7,8,9]],featureNames)
	expTest = constructor([[1,2,3],[10,11,12]],featureNames)

	assert ret.isIdentical(expRet)
	assert expTest.isIdentical(toTest)



################
# foldIterator #
################

def foldIterator_exceptionEmpty(constructor):
	""" Test foldIterator() for exception when object is empty """
	data = []
	toTest = constructor(data)
	toTest.foldIterator(2)

def foldIterator_exceptionTooManyFolds(constructor):
	""" Test foldIterator() for exception when given too many folds """
	data = [[1],[2],[3],[4],[5]]
	names = ['col']
	toTest = constructor(data,names)
	toTest.foldIterator(6)


def foldIterator_verifyPartitions(constructor):
	""" Test foldIterator() yields the correct number folds and partitions the data """
	data = [[1],[2],[3],[4],[5]]
	names = ['col']
	toTest = constructor(data,names)
	folds = toTest.foldIterator(2)

	(fold1Train, fold1Test) = folds.next()
	(fold2Train, fold2Test) = folds.next()

	try:
		folds.next()
		assert False
	except StopIteration:
		pass

	assert fold1Train.points() + fold1Test.points() == 5
	assert fold2Train.points() + fold2Test.points() == 5

	fold1Train.appendPoints(fold1Test)
	fold2Train.appendPoints(fold2Test)

	#TODO some kind of rigourous partition check




####################
# applyToEachPoint() #
####################

def applyToEachPoint_exceptionEmpty(constructor):
	""" Test applyToEachPoint() for ImproperActionException when object is empty """
	origData = []
	origObj = constructor(origData)

	def emitLower(point):
		return point[origObj.featureNames['deci']]

	lowerCounts = origObj.applyToEachPoint(emitLower)

def applyToEachPoint_exceptionInputNone(constructor):
	""" Test applyToEachPoint() for ArgumentException when function is None """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj = constructor(deepcopy(origData),featureNames)
	origObj.applyToEachPoint(None)

def applyToEachPoint_Handmade(constructor):
	""" Test applyToEachPoint() with handmade output """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj = constructor(deepcopy(origData),featureNames)


	def emitLower(point):
		return point[origObj.featureNames['deci']]

	lowerCounts = origObj.applyToEachPoint(emitLower)

	expectedOut = [[0.1], [0.1], [0.1], [0.2]]
	exp = constructor(expectedOut)

	assert lowerCounts.isIdentical(exp)


def applyToEachPoint_nonZeroItAndLen(constructor):
	""" Test applyToEachPoint() for the correct usage of the nonzero iterator """
	origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
	origObj = constructor(deepcopy(origData))

	def emitNumNZ(point):
		ret = 0
		print len(point)
		assert len(point) == 3
		for value in point.nonZeroIterator():
			ret += 1
		return ret

	counts = origObj.applyToEachPoint(emitNumNZ)

	expectedOut = [[3], [2], [2], [1]]
	exp = constructor(expectedOut)

	assert counts.isIdentical(exp)



#######################
# applyToEachFeature() #
#######################

def applyToEachFeature_exceptionEmpty(constructor):
	""" Test applyToEachFeature() for ImproperActionException when object is empty """
	origData = []
	origObj= constructor(origData)

	def emitAllEqual(feature):
		first = feature[0]
		for value in feature:
			if value != first:
				return 0
		return 1

	lowerCounts = origObj.applyToEachFeature(emitAllEqual)

def applyToEachFeature_exceptionInputNone(constructor):
	""" Test applyToEachFeature() for ArgumentException when function is None """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj= constructor(deepcopy(origData),featureNames)
	origObj.applyToEachFeature(None)

def applyToEachFeature_Handmade(constructor):
	""" Test applyToEachFeature() with handmade output """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj= constructor(deepcopy(origData),featureNames)

	def emitAllEqual(feature):
		first = feature[0]
		for value in feature:
			if value != first:
				return 0
		return 1

	lowerCounts = origObj.applyToEachFeature(emitAllEqual)
	expectedOut = [[1,0,0]]	
	assert lowerCounts.isIdentical(constructor(expectedOut))



def applyToEachFeature_nonZeroItAndLen(constructor):
	""" Test applyToEachFeature() for the correct usage of the nonzero iterator """
	origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
	origObj = constructor(deepcopy(origData))

	def emitNumNZ(feature):
		ret = 0
		assert len(feature) == 4
		for value in feature.nonZeroIterator():
			ret += 1
		return ret

	counts = origObj.applyToEachFeature(emitNumNZ)

	expectedOut = [[3, 3, 2]]
	exp = constructor(expectedOut)

	assert counts.isIdentical(exp)



#####################
# mapReducePoints() #
#####################

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


def mapReducePoints_argumentExceptionNoFeatures(constructor):
	""" Test mapReducePoints() for ImproperActionException when there are no features  """
	data = [[],[],[]]
	toTest = constructor(data)
	toTest.mapReducePoints(simpleMapper,simpleReducer)

def mapReducePoints_argumentExceptionNoneMap(constructor):
	""" Test mapReducePoints() for ArgumentException when mapper is None """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.mapReducePoints(None,simpleReducer)

def mapReducePoints_argumentExceptionNoneReduce(constructor):
	""" Test mapReducePoints() for ArgumentException when reducer is None """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.mapReducePoints(simpleMapper,None)

def mapReducePoints_argumentExceptionUncallableMap(constructor):
	""" Test mapReducePoints() for ArgumentException when mapper is not callable """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.mapReducePoints("hello",simpleReducer)

def mapReducePoints_argumentExceptionUncallableReduce(constructor):
	""" Test mapReducePoints() for ArgumentException when reducer is not callable """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.mapReducePoints(simpleMapper,5)


# inconsistent output?



def mapReducePoints_handmade(constructor):
	""" Test mapReducePoints() against handmade output """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.mapReducePoints(simpleMapper,simpleReducer)
	
	exp = constructor([[1,5],[4,11],[7,17]])

	assert (ret.isIdentical(exp))
	assert (toTest.isIdentical(constructor(data,featureNames)))


def mapReducePoints_handmadeNoneReturningReducer(constructor):
	""" Test mapReducePoints() against handmade output with a None returning Reducer """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.mapReducePoints(simpleMapper,oddOnlyReducer)
	
	exp = constructor([[1,5],[7,17]])

	assert (ret.isIdentical(exp))
	assert (toTest.isIdentical(constructor(data,featureNames)))



#######################
# pointIterator() #
#######################


def pointIterator_exactValueViaFor(constructor):
	""" Test pointIterator() gives views that contain exactly the correct data """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	
#	import pdb
#	pdb.set_trace()

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


#########################
# featureIterator() #
#########################


def featureIterator_exactValueViaFor(constructor):
	""" Test featureIterator() gives views that contain exactly the correct data """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	
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



####################
# transformPoint() #
####################




######################
# transformFeature() #
######################



#####################################
# applyToElements() #
#####################################

def passThrough(value):
	return value

def plusOne(value):
	return (value + 1)

def plusOneOnlyEven(value):
	if value % 2 == 0:
		return (value + 1)
	else:
		return None

def applyToElements_passthrough(constructor):
	""" test applyToElements can construct a list by just passing values through  """

	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.applyToElements(passThrough, inPlace=False)
	retRaw = ret.copy(asType="python list")

	assert [1,2,3] in retRaw
	assert [4,5,6] in retRaw
	assert [7,8,9] in retRaw

	ret = toTest.applyToElements(passThrough)
	assert ret == toTest
	retRaw = ret.copy(asType="python list")

	assert [1,2,3] in retRaw
	assert [4,5,6] in retRaw
	assert [7,8,9] in retRaw


def applyToElements_plusOnePreserve(constructor):
	""" test applyToElements can modify elements other than zero  """

	data = [[1,0,3],[0,5,6],[7,0,9]]
	toTest = constructor(data)
	ret = toTest.applyToElements(plusOne, inPlace=False, preserveZeros=True)
	retRaw = ret.copy(asType="python list")

	assert [2,0,4] in retRaw
	assert [0,6,7] in retRaw
	assert [8,0,10] in retRaw

	ret = toTest.applyToElements(plusOne, preserveZeros=True)
	assert ret == toTest
	retRaw = ret.copy(asType="python list")

	assert [2,0,4] in retRaw
	assert [0,6,7] in retRaw
	assert [8,0,10] in retRaw


def applyToElements_plusOneExclude(constructor):
	""" test applyToElements() skipNoneReturnValues flag  """

	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.applyToElements(plusOneOnlyEven, inPlace=False, skipNoneReturnValues=True)
	retRaw = ret.copy(asType="python list")

	assert [1,3,3] in retRaw
	assert [5,5,7] in retRaw
	assert [7,9,9] in retRaw

	ret = toTest.applyToElements(plusOneOnlyEven, skipNoneReturnValues=True)
	assert ret == toTest
	retRaw = ret.copy(asType="python list")

	assert [1,3,3] in retRaw
	assert [5,5,7] in retRaw
	assert [7,9,9] in retRaw


def applyToElements_plusOneLimited(constructor):
	""" test applyToElements() on limited portions of the points and features """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	names = ['one','two','three']
	toTest = constructor(data, names)

	ret = toTest.applyToElements(plusOneOnlyEven, points=1, features=[1,'three'], inPlace=False, skipNoneReturnValues=True)
	retRaw = ret.copy(asType="python list")

	assert [5,7] in retRaw

	ret = toTest.applyToElements(plusOneOnlyEven, points=1, features=[1,'three'], skipNoneReturnValues=True)
	assert ret == toTest
	retRaw = ret.copy(asType="python list")

	assert [1,2,3] in retRaw
	assert [4,5,7] in retRaw
	assert [7,8,9] in retRaw


########################
# isApproximatelyEqual() #
########################


def isApproximatelyEqual_randomTest(constructor):
	""" Test isApproximatelyEqual() using randomly generated data """

	for x in xrange(1,2):
		points = 100
		features = 40
		data = numpy.zeros((points,features))

		for i in xrange(points):
			for j in xrange(features):
				data[i,j] = numpy.random.rand() * numpy.random.randint(0,5)

	toTest = constructor(data)

	listObj = List(data)
	matrix = Matrix(data)
	sparse = Sparse(data)

	assert toTest.isApproximatelyEqual(listObj)

	assert toTest.isApproximatelyEqual(matrix)

	assert toTest.isApproximatelyEqual(sparse)



###################
# shufflePoints() #
###################


def shufflePoints_noLongerEqual(constructor):
	""" Tests shufflePoints() results in a changed object """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	toTest = constructor(deepcopy(data))
	toCompare = constructor(deepcopy(data))

	# it is possible that it shuffles it into the same configuration.
	# the odds are vanishly low that it will do so over consecutive calls
	# however. We will pass as long as it changes once
	returns = []
	for i in xrange(5):
		ret = toTest.shufflePoints()
		returns.append(ret)
		if not toTest.isApproximatelyEqual(toCompare):
			break

	assert not toTest.isApproximatelyEqual(toCompare)

	for ret in returns:
		assert ret == toTest



#####################
# shuffleFeatures() #
#####################


def shuffleFeatures_noLongerEqual(constructor):
	""" Tests shuffleFeatures() results in a changed object """
	data = [[1,2,3,33],[4,5,6,66],[7,8,9,99],[10,11,12,1111111]]
	toTest = constructor(deepcopy(data))
	toCompare = constructor(deepcopy(data))

	# it is possible that it shuffles it into the same configuration.
	# the odds are vanishly low that it will do so over consecutive calls
	# however. We will pass as long as it changes once
	returns = []
	for i in xrange(5):
		ret = toTest.shuffleFeatures()
		returns.append(ret)
		if not toTest.isApproximatelyEqual(toCompare):
			break

	assert not toTest.isApproximatelyEqual(toCompare)

	for ret in returns:
		assert ret == toTest




