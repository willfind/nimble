"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

"""

from copy import deepcopy

import numpy

from UML.data import List
from UML.data import Dense
from UML.data import Sparse


###########################
# dropStringValuedFeatures #
###########################


#hmmm but this only applies to representations that can have strings.

def dropStringValuedFeatures_emptyTest(constructor):
	""" Test dropStringValuedFeatures() when the data is empty """
	data = []
	toTest = constructor(data)
	unchanged = constructor(data)
	toTest.dropStringValuedFeatures()
	assert toTest.equals(unchanged)


#################################
# featureToBinaryCategoryFeatures #
#################################


def featureToBinaryCategoryFeatures_emptyException(constructor):
	""" Test featureToBinaryCategoryFeatures() with an empty object """
	data = []
	toTest = constructor(data)
	toTest.featureToBinaryCategoryFeatures(0)


def featureToBinaryCategoryFeatures_handmade(constructor):
	""" Test featureToBinaryCategoryFeatures() against handmade output """
	data = [[1],[2],[3]]
	featureNames = ['col']
	toTest = constructor(data,featureNames)
	toTest.featureToBinaryCategoryFeatures(0)

	expData = [[1,0,0], [0,1,0], [0,0,1]]
	expFeatureNames = ['col=1','col=2','col=3']
	exp = constructor(expData, expFeatureNames)

	print toTest.featureNames

	assert toTest.equals(exp)
	


#############################
# featureToIntegerCategories #
#############################

def featureToIntegerCategories_emptyException(constructor):
	""" Test featureToIntegerCategories() with an empty object """
	data = []
	toTest = constructor(data)
	toTest.featureToIntegerCategories(0)

def featureToIntegerCategories_handmade(constructor):
	""" Test featureToIntegerCategories() against handmade output """
	data = [[10],[20],[30.5],[20],[10]]
	featureNames = ['col']
	toTest = constructor(data,featureNames)
	toTest.featureToIntegerCategories(0)

	assert toTest.data[0] == toTest.data[4]
	assert toTest.data[1] == toTest.data[3]
	assert toTest.data[0] != toTest.data[1]
	assert toTest.data[0] != toTest.data[2]




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

	assert ret.equals(expRet)
	assert expTest.equals(toTest)

def selectConstantOfPointsByValue_handmadeLimit(constructor):
	""" Test selectConstantOfPointsByValue() against handmade output when the constant exceeds the available points """
	data = [[1,2,3],[1,5,6],[1,8,9],[2,11,12]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfPointsByValue(2,'1')

	expRet = constructor([[1,2,3],[1,8,9],[2,11,12]],featureNames)
	expTest = constructor([[1,5,6],],featureNames)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)



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

	assert ret.equals(expRet)
	assert expTest.equals(toTest)


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

	assert ret.equals(expRet)
	assert expTest.equals(toTest)



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
# applyFunctionToEachPoint() #
####################

def applyFunctionToEachPoint_exceptionEmpty(constructor):
	""" Test applyFunctionToEachPoint() for ImproperActionException when object is empty """
	origData = []
	origObj = constructor(origData)

	def emitLower(point):
		return point[origObj.featureNames['deci']]

	lowerCounts = origObj.applyFunctionToEachPoint(emitLower)

def applyFunctionToEachPoint_exceptionInputNone(constructor):
	""" Test applyFunctionToEachPoint() for ArgumentException when function is None """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj = constructor(deepcopy(origData),featureNames)
	origObj.applyFunctionToEachPoint(None)

def applyFunctionToEachPoint_Handmade(constructor):
	""" Test applyFunctionToEachPoint() with handmade output """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj = constructor(deepcopy(origData),featureNames)


	def emitLower(point):
		return point[origObj.featureNames['deci']]

	lowerCounts = origObj.applyFunctionToEachPoint(emitLower)

	expectedOut = [[0.1], [0.1], [0.1], [0.2]]
	exp = constructor(expectedOut)

	assert lowerCounts.equals(exp)


def applyFunctionToEachPoint_nonZeroItAndLen(constructor):
	""" Test applyFunctionToEachPoint() for the correct usage of the nonzero iterator """
	origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
	origObj = constructor(deepcopy(origData))

	def emitNumNZ(point):
		ret = 0
		print len(point)
		assert len(point) == 3
		for value in point.nonZeroIterator():
			ret += 1
		return ret

	counts = origObj.applyFunctionToEachPoint(emitNumNZ)

	expectedOut = [[3], [2], [2], [1]]
	exp = constructor(expectedOut)

	assert counts.equals(exp)



#######################
# applyFunctionToEachFeature() #
#######################

def applyFunctionToEachFeature_exceptionEmpty(constructor):
	""" Test applyFunctionToEachFeature() for ImproperActionException when object is empty """
	origData = []
	origObj= constructor(origData)

	def emitAllEqual(feature):
		first = feature[0]
		for value in feature:
			if value != first:
				return 0
		return 1

	lowerCounts = origObj.applyFunctionToEachFeature(emitAllEqual)

def applyFunctionToEachFeature_exceptionInputNone(constructor):
	""" Test applyFunctionToEachFeature() for ArgumentException when function is None """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj= constructor(deepcopy(origData),featureNames)
	origObj.applyFunctionToEachFeature(None)

def applyFunctionToEachFeature_Handmade(constructor):
	""" Test applyFunctionToEachFeature() with handmade output """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj= constructor(deepcopy(origData),featureNames)

	def emitAllEqual(feature):
		first = feature[0]
		for value in feature:
			if value != first:
				return 0
		return 1

	lowerCounts = origObj.applyFunctionToEachFeature(emitAllEqual)
	expectedOut = [[1,0,0]]	
	assert lowerCounts.equals(constructor(expectedOut))



def applyFunctionToEachFeature_nonZeroItAndLen(constructor):
	""" Test applyFunctionToEachFeature() for the correct usage of the nonzero iterator """
	origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
	origObj = constructor(deepcopy(origData))

	def emitNumNZ(feature):
		ret = 0
		assert len(feature) == 4
		for value in feature.nonZeroIterator():
			ret += 1
		return ret

	counts = origObj.applyFunctionToEachFeature(emitNumNZ)

	expectedOut = [[3, 3, 2]]
	exp = constructor(expectedOut)

	assert counts.equals(exp)



#####################
# mapReduceOnPoints() #
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


def mapReduceOnPoints_argumentExceptionNoFeatures(constructor):
	""" Test mapReduceOnPoints() for ImproperActionException when there are no features  """
	data = [[],[],[]]
	toTest = constructor(data)
	toTest.mapReduceOnPoints(simpleMapper,simpleReducer)

def mapReduceOnPoints_argumentExceptionNoneMap(constructor):
	""" Test mapReduceOnPoints() for ArgumentException when mapper is None """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.mapReduceOnPoints(None,simpleReducer)

def mapReduceOnPoints_argumentExceptionNoneReduce(constructor):
	""" Test mapReduceOnPoints() for ArgumentException when reducer is None """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.mapReduceOnPoints(simpleMapper,None)

def mapReduceOnPoints_argumentExceptionUncallableMap(constructor):
	""" Test mapReduceOnPoints() for ArgumentException when mapper is not callable """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.mapReduceOnPoints("hello",simpleReducer)

def mapReduceOnPoints_argumentExceptionUncallableReduce(constructor):
	""" Test mapReduceOnPoints() for ArgumentException when reducer is not callable """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.mapReduceOnPoints(simpleMapper,5)


# inconsistent output?



def mapReduceOnPoints_handmade(constructor):
	""" Test mapReduceOnPoints() against handmade output """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.mapReduceOnPoints(simpleMapper,simpleReducer)
	
	exp = constructor([[1,5],[4,11],[7,17]])

	assert (ret.equals(exp))
	assert (toTest.equals(constructor(data,featureNames)))


def mapReduceOnPoints_handmadeNoneReturningReducer(constructor):
	""" Test mapReduceOnPoints() against handmade output with a None returning Reducer """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.mapReduceOnPoints(simpleMapper,oddOnlyReducer)
	
	exp = constructor([[1,5],[7,17]])

	assert (ret.equals(exp))
	assert (toTest.equals(constructor(data,featureNames)))



#######################
# pointViewIterator() #
#######################


def pointViewIterator_exactValueViaFor(constructor):
	""" Test pointViewIterator() gives views that contain exactly the correct data """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	
#	import pdb
#	pdb.set_trace()

	viewIter = toTest.pointViewIterator()

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
# featureViewIterator() #
#########################


def featureViewIterator_exactValueViaFor(constructor):
	""" Test featureViewIterator() gives views that contain exactly the correct data """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	
	viewIter = toTest.featureViewIterator()

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
# computeListOfValuesFromElements() #
#####################################

def passThrough(value):
	return value

def passThroughEven(value):
	if value % 2 == 0:
		return value
	else:
		return None

def computeList_passthrough(constructor):
	""" test computeListOfValuesFromElements can construct a list by just passing values through  """

	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.computeListOfValuesFromElements(passThrough)

	print toTest.getType()

	assert 1 in ret
	assert 2 in ret
	assert 3 in ret
	assert 4 in ret
	assert 5 in ret
	assert 6 in ret
	assert 7 in ret
	assert 8 in ret
	assert 9 in ret


def computeList_passthroughSkip(constructor):
	""" test computeListOfValuesFromElements can construct a list by just passing values through  """

	data = [[1,0,3],[0,5,6],[7,0,9]]
	toTest = constructor(data)
	ret = toTest.computeListOfValuesFromElements(passThrough, skipZeros=True)

	assert 1 in ret
	assert 3 in ret
	assert 5 in ret
	assert 6 in ret
	assert 7 in ret
	assert 9 in ret


def computeList_passthroughExclude(constructor):
	""" test computeListOfValuesFromElements can construct a list by just passing values through  """

	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.computeListOfValuesFromElements(passThroughEven,excludeNoneResultValues=True)

	assert 2 in ret
	assert 4 in ret
	assert 6 in ret
	assert 8 in ret



########################
# isApproxEquivalent() #
########################


def isApproxEquivalent_randomTest(constructor):
	""" Test isApproxEquivalent() using randomly generated data """

	for x in xrange(1,2):
		points = 200
		features = 80
		data = numpy.zeros((points,features))

		for i in xrange(points):
			for j in xrange(features):
				data[i,j] = numpy.random.rand() * numpy.random.randint(1,5)

	toTest = constructor(data)

	listObj = List(data)
	dense = Dense(data)
	sparse = Sparse(data)

	assert toTest.isApproxEquivalent(listObj)
	assert listObj.isApproxEquivalent(toTest)

	assert toTest.isApproxEquivalent(dense)
	assert dense.isApproxEquivalent(toTest)

	assert toTest.isApproxEquivalent(sparse)
	assert sparse.isApproxEquivalent(toTest)





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
	for i in xrange(5):
		toTest.shufflePoints()
		if not toTest.isApproxEquivalent(toCompare):
			return

	assert not toTest.isApproxEquivalent(toCompare)






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
	for i in xrange(5):
		toTest.shuffleFeatures()
		if not toTest.isApproxEquivalent(toCompare):
			return

	assert not toTest.isApproxEquivalent(toCompare)







