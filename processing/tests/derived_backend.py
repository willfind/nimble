"""
Backend for unit tests of the functions implemented by the derived classes.

These tests rely on having a working .equals method, which must be tested
directly by the class calling this backend.

"""

import tempfile
import os

from ..base_data import *
from copy import deepcopy
from ..row_list_data import RowListData as RLD
from ..dense_matrix_data import DenseMatrixData as DMD

##############
# __init__() #
##############

def init_allEqual(constructor):
	""" Test __init__() so that each possible way to instantiate produces equal objects """
	# instantiate from list of lists
	fromList = constructor(data=[[1,2,3]], featureNames=['one', 'two', 'three'])

	# instantiate from file
	tmpFile = tempfile.NamedTemporaryFile() 
	tmpFile.write("#one,two,three\n")
	tmpFile.write("1,2,3\n")
	tmpFile.flush()
	# TODO -- can we name this file so readFile() can use the extension to determine what to call?
	fromCSV = constructor(file=tmpFile.name)

	# check equality between all pairs
	assert fromList.equals(fromCSV)
	assert fromCSV.equals(fromList)


############
# equals() #
############

def equals_False(constructor):
	""" Test equals() against some non-equal input """
	toTest = constructor([[4,5]])
	assert not toTest.equals(constructor([[1,1],[2,2]]))
	assert not toTest.equals(constructor([[1,2,3]]))
	assert not toTest.equals(constructor([[1,2]]))

def equals_True(constructor):
	""" Test equals() against some actually equal input """
	toTest1 = constructor([[4,5]])
	toTest2 = constructor(deepcopy([[4,5]]))
	assert toTest1.equals(toTest2)
	assert toTest2.equals(toTest1)

def equals_empty(constructor):
	""" Test equals() for empty objects """
	toTest1 = constructor([])
	toTest2 = constructor(deepcopy([]))
	assert toTest1.equals(toTest2)
	assert toTest2.equals(toTest1)

	toTest1 = constructor(None)
	toTest2 = constructor(deepcopy(None))
	assert toTest1.equals(toTest2)
	assert toTest2.equals(toTest1)


###############
# transpose() #
###############

def transpose_handmade(constructor):
	""" Test transpose() function against handmade output """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	dataTrans = [[1,4,7],[2,5,8],[3,6,9]]

	dataObj1 = constructor(deepcopy(data))
	dataObj2 = constructor(deepcopy(data))
	dataObjT = constructor(deepcopy(dataTrans))
	
	dataObj1.transpose()
	assert dataObj1.equals(dataObjT)
	dataObj1.transpose()
	dataObjT.transpose()
	assert dataObj1.equals(dataObj2)
	assert dataObj2.equals(dataObjT)


#############
# appendPoints() #
#############

def appendPoints_exceptionNone(constructor):
	""" Test appendPoints() for ArgumentException when toAppend is None"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendPoints(None)

def appendPoints_exceptionWrongSize(constructor):
	""" Test appendPoints() for ArgumentException when toAppend has too many features """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendPoints([["too", " ", "many", " ", "features"]])

def appendPoints_handmadeSingle(constructor):
	""" Test appendPoints() against handmade output for a single added point """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	dataExpected = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	toTest = constructor(data)
	toAppend = constructor([[10,11,12]])
	expected = constructor(dataExpected)
	toTest.appendPoints(toAppend)
	assert toTest.equals(expected)

def appendPoints_handmadeSequence(constructor):
	""" Test appendPoints() against handmade output for a sequence of additions"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toAppend1 = [[0.1,0.2,0.3]]
	toAppend2 = [[0.01,0.02,0.03],[0,0,0]]
	toAppend3 = [[10,11,12]]

	dataExpected = [[1,2,3],[4,5,6],[7,8,9],[0.1,0.2,0.3],[0.01,0.02,0.03],[0,0,0],[10,11,12]]
	toTest = constructor(data)
	toTest.appendPoints(constructor(toAppend1))
	toTest.appendPoints(constructor(toAppend2))
	toTest.appendPoints(constructor(toAppend3))

	expected = constructor(dataExpected)

	assert toTest.equals(expected)
	

################
# appendFeatures() #
################


def appendFeatures_exceptionNone(constructor):
	""" Test appendFeatures() for ArgumentException when toAppend is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendFeatures(None)

def appendFeatures_exceptionWrongSize(constructor):
	""" Test appendFeatures() for ArgumentException when toAppend has too many points """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendFeatures([["too"], [" "], ["many"], [" "], ["points"]])

def appendFeatures_exceptionSameFeatureName(constructor):
	""" Test appendFeatures() for ArgumentException when toAppend and self have a featureName in common """
	toTest1 = constructor([[1]],["hello"])
	toTest2 = constructor([[1,2]],["hello","goodbye"])
	toTest2.appendFeatures(toTest1)

def appendFeatures_handmadeSingle(constructor):
	""" Test appendFeatures() against handmade output for a single added feature"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)

	toAppend = constructor([[-1],[-2],[-3]],['-1'])

	dataExpected = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	featureNamesExpected = ['1','2','3','-1']
	expected = constructor(dataExpected,featureNamesExpected)

	toTest.appendFeatures(toAppend)
	assert toTest.equals(expected)

def appendFeatures_handmadeSequence(constructor):
	""" Test appendFeatures() against handmade output for a sequence of additions"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)

	toAppend1 = [[0.1],[0.2],[0.3]]
	lab1 =  ['a']
	toAppend2 = [[0.01,0],[0.02,0],[0.03,0]]
	lab2 = ['A','0']
	toAppend3 = [[10],[11],[12]]
	lab3 = ['10']

	toTest.appendFeatures(constructor(toAppend1,lab1))
	toTest.appendFeatures(constructor(toAppend2,lab2))
	toTest.appendFeatures(constructor(toAppend3,lab3))

	featureNamesExpected = ['1','2','3','a','A','0','10']
	dataExpected = [[1,2,3,0.1,0.01,0,10],[4,5,6,0.2,0.02,0,11],[7,8,9,0.3,0.03,0,12]]

	expected = constructor(dataExpected,featureNamesExpected)
	assert toTest.equals(expected)



##############
# sortPoints() #
##############

def sortPoints_handmadeNatural(constructor):
	""" Test sortPoints() against handmade , naturally ordered  output """	
	data = [[7,8,9],[1,2,3],[4,5,6]]
	toTest = constructor(data)

	toTest.sortPoints()

	dataExpected = [[1,2,3],[4,5,6],[7,8,9]]
	objExp = constructor(dataExpected)

	assert toTest.equals(objExp)


def sortPoints_handmadeWithFcn(constructor):
	""" Test sortPoints() against handmade output when given cmp and key functions """	
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)

	def cmpNums(num1,num2):
		return num1 - num2

	def sumModEight(point):
		total = 0
		for value in point:
			total = total + value
		return total % 8

	toTest.sortPoints(cmpNums,sumModEight)

	dataExpected = [[7,8,9],[1,2,3],[4,5,6]]
	objExp = constructor(dataExpected)

	assert toTest.equals(objExp)


def sortPoints_handmade_reverse(constructor):
	assert False

#################
# sortFeatures() #
#################


def sortFeatures_handmadeWithFcn(constructor):
	""" Test sortFeatures() against handmade output when given cmp and key functions """	
	data = [[1,4,7],[2,5,8],[3,6,9]]
	toTest = constructor(data)

	def cmpNums(num1,num2):
		return num1 - num2

	def sumModEight(col):
		total = 0
		for value in col:
			total = total + value
		return total % 8

	toTest.sortFeatures(cmpNums,sumModEight)
	toTest.sortFeatures(key=sumModEight)

	dataExpected = [[7,1,4],[8,2,5],[9,3,6]]
	objExp = constructor(dataExpected)
	assert toTest.equals(objExp)

def sortPoints_handmade_reverse(constructor):
	assert False

#################
# extractPoints() #
#################

def extractPoints_emptyInput(constructor): #TODO 
	""" Test extractPoints() does nothing when not provided with any input """
	pass

def extractPoints_handmadeSingle(constructor):
	""" Test extractPoints() against handmade output when extracting one point """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ext1 = toTest.extractPoints(0)
	exp1 = constructor([[1,2,3]])
	assert ext1.equals(exp1)
	expEnd = constructor([[4,5,6],[7,8,9]])
	assert toTest.equals(expEnd)

def extractPoints_handmadeListSequence(constructor):
	""" Test extractPoints() against handmade output for several list extractions """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	toTest = constructor(data)
	ext1 = toTest.extractPoints(0)
	exp1 = constructor([[1,2,3]])
	assert ext1.equals(exp1)
	ext2 = toTest.extractPoints([1,2])
	exp2 = constructor([[7,8,9],[10,11,12]])
	assert ext2.equals(exp2)
	expEnd = constructor([[4,5,6]])
	assert toTest.equals(expEnd)

def extractPoints_handmadeFunction(constructor):
	""" Test extractPoints() against handmade output for function extraction """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	def oneOrFour(point):
		if 1 in point or 4 in point:
			return True
		return False
	ext = toTest.extractPoints(oneOrFour)
	exp = constructor([[1,2,3],[4,5,6]])
	assert ext.equals(exp)
	expEnd = constructor([[7,8,9]])
	assert toTest.equals(expEnd)

def extractPoints_handmadeFuncionWithFeatureNames(constructor):
	""" Test extractPoints() against handmade output for function extraction with featureNames"""
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	def oneOrFour(point):
		if 1 in point or 4 in point:
			return True
		return False
	ext = toTest.extractPoints(oneOrFour)
	exp = constructor([[1,2,3],[4,5,6]],featureNames)
	assert ext.equals(exp)
	expEnd = constructor([[7,8,9]],featureNames)
	assert toTest.equals(expEnd)

def extractPoints_exceptionStartInvalid(constructor):
	""" Test extracPoints() for ArgumentException when start is not a valid point index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractPoints(start=-1,end=2)

def extractPoints_exceptionEndInvalid(constructor):
	""" Test extractPoints() for ArgumentException when start is not a valid feature index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractPoints(start=1,end=5)

def extractPoints_exceptionInversion(constructor):
	""" Test extractPoints() for ArgumentException when start comes after end """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractPoints(start=2,end=0)

def extractPoints_handmade(constructor):
	""" Test extractPoints() against handmade output for range extraction """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.extractPoints(start=1,end=2)
	
	expectedRet = constructor([[4,5,6],[7,8,9]])
	expectedTest = constructor([[1,2,3]])

	assert expectedRet.equals(ret)
	assert expectedTest.equals(toTest)

def extractPoints_handmadeWithFeatureNames(constructor):
	""" Test extractPoints() against handmade output for range extraction with featureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.extractPoints(start=1,end=2)
	
	expectedRet = constructor([[4,5,6],[7,8,9]],featureNames)
	expectedTest = constructor([[1,2,3]],featureNames)

	assert expectedRet.equals(ret)
	assert expectedTest.equals(toTest)


#TODO an extraction test where all data is removed
#TODO extraction tests for all of the number and randomize combinations


####################
# extractFeatures() #
####################

def extractFeatures_handmadeSingle(constructor):
	""" Test extractFeatures() against handmade output when extracting one feature """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ext1 = toTest.extractFeatures(0)
	exp1 = constructor([[1],[4],[7]])
	assert ext1.equals(exp1)
	expEnd = constructor([[2,3],[5,6],[8,9]])
	assert toTest.equals(expEnd)

def extractFeatures_handmadeListSequence(constructor):
	""" Test extractFeatures() against handmade output for several extractions by list """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	toTest = constructor(data)
	ext1 = toTest.extractFeatures([0])
	exp1 = constructor([[1],[4],[7]])
	assert ext1.equals(exp1)
	ext2 = toTest.extractFeatures([1,2])
	exp2 = constructor([[3,-1],[6,-2],[9,-3]])
	assert ext2.equals(exp2)
	expEndData = [[2],[5],[8]]
	expEnd = constructor(expEndData)
	assert toTest.equals(expEnd)

def extractFeatures_handmadeListWithFeatureName(constructor):
	""" Test extractFeatures() against handmade output for list extraction when specifying featureNames """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	featureNames = ["one","two","three","neg"]
	toTest = constructor(data,featureNames)
	ext1 = toTest.extractFeatures(["one"])
	exp1 = constructor([[1],[4],[7]], ["one"])
	assert ext1.equals(exp1)
	ext2 = toTest.extractFeatures(["three","neg"])
	exp2 = constructor([[3,-1],[6,-2],[9,-3]],["three","neg"])
	assert ext2.equals(exp2)
	expEnd = constructor([[2],[5],[8]], ["two"])
	assert toTest.equals(expEnd)


def extractFeatures_handmadeFunction(constructor):
	""" Test extractFeatures() against handmade output for function extraction """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	toTest = constructor(data)
	def absoluteOne(feature):
		if 1 in feature or -1 in feature:
			return True
		return False
	ext = toTest.extractFeatures(absoluteOne)
	exp = constructor([[1,-1],[4,-2],[7,-3]])
	assert ext.equals(exp)
	expEnd = constructor([[2,3],[5,6],[8,9]])		
	assert toTest.equals(expEnd)


def extractFeatures_handmadeFunctionWithFeatureName(constructor):
	""" Test extractFeatures() against handmade output for function extraction with featureNames """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	featureNames = ["one","two","three","neg"]
	toTest = constructor(data,featureNames)
	def absoluteOne(feature):
		if 1 in feature or -1 in feature:
			return True
		return False

	ext = toTest.extractFeatures(absoluteOne)
	exp = constructor([[1,-1],[4,-2],[7,-3]], ['one','neg'])
	assert ext.equals(exp)
	expEnd = constructor([[2,3],[5,6],[8,9]],["two","three"])	
	assert toTest.equals(expEnd)


def extractFeatures_exceptionStartInvalid(constructor):
	""" Test extractFeatures() for ArgumentException when start is not a valid feature index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractFeatures(start=-1, end=2)

def extractFeatures_exceptionStartInvalidFeatureName(constructor):
	""" Test extractFeatures() for ArgumentException when start is not a valid feature FeatureName """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractFeatures(start="wrong", end=2)

def extractFeatures_exceptionEndInvalid(constructor):
	""" Test extractFeatures() for ArgumentException when start is not a valid feature index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractFeatures(start=0, end=5)

def extractFeatures_exceptionEndInvalidFeatureName(constructor):
	""" Test extractFeatures() for ArgumentException when start is not a valid featureName """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractFeatures(start="two", end="five")

def extractFeatures_exceptionInversion(constructor):
	""" Test extractFeatures() for ArgumentException when start comes after end """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractFeatures(start=2, end=0)

def extractFeatures_exceptionInversionFeatureName(constructor):
	""" Test extractFeatures() for ArgumentException when start comes after end as FeatureNames"""
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractFeatures(start="two", end="one")

def extractFeatures_handmadeRange(constructor):
	""" Test extractFeatures() against handmade output for range extraction """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.extractFeatures(start=1, end=2)
	
	expectedRet = constructor([[2,3],[5,6],[8,9]])
	expectedTest = constructor([[1],[4],[7]])

	assert expectedRet.equals(ret)
	assert expectedTest.equals(toTest)

def extractFeatures_handmadeWithFeatureNames(constructor):
	""" Test extractFeatures() against handmade output for range extraction with FeatureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.extractFeatures(start=1,end=2)
	
	expectedRet = constructor([[2,3],[5,6],[8,9]],["two","three"])
	expectedTest = constructor([[1],[4],[7]],["one"])

	assert expectedRet.equals(ret)
	assert expectedTest.equals(toTest)



####################
# applyFunctionToEachPoint() #
####################

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


	def emitLower (point):
		return point[origObj.featureNames['deci']]

	lowerCounts = origObj.applyFunctionToEachPoint(emitLower)

	expectedOut = [[0.1], [0.1], [0.1], [0.2]]
	exp = constructor(expectedOut)

	assert lowerCounts.equals(exp)



#######################
# applyFunctionToEachFeature() #
#######################

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

	def emitAllEqual (feature):
		first = feature[0]
		for value in feature:
			if value != first:
				return 0
		return 1

	lowerCounts = origObj.applyFunctionToEachFeature(emitAllEqual)
	expectedOut = [[1,0,0]]	
	assert lowerCounts.equals(constructor(expectedOut))



#####################
# mapReduceOnPoints() #
#####################

def simpleMapper(point):
	idInt = point[0]
	intList = point[1:]
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

	




##########################
# toRowListData() #
##########################


def toRowListData_handmade_defaultFeatureNames(constructor):
	""" Test toRowListData with default featureNames """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)

	ret = toTest.toRowListData()
	exp = RLD(data)

	assert ret.equals(exp)
	assert exp.equals(ret)

	
def toRowListData_handmade_assignedFeatureNames(constructor):
	""" Test toRowListData with assigned featureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)

	ret = toTest.toRowListData()
	exp = RLD(data,featureNames)

	assert ret.equals(exp)
	assert exp.equals(ret)



##############################
# toDenseMatrixData() #
##############################

def toDenseMatrixData_handmade_defaultFeatureNames(constructor):
	""" Test toDenseMatrixData with default featureNames """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)

	ret = toTest.toDenseMatrixData()
	exp = DMD(data)

	assert ret.equals(exp)
	assert exp.equals(ret)

	
def toDenseMatrixData_handmade_assignedFeatureNames(constructor):
	""" Test toDenseMatrixData with assigned featureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)

	ret = toTest.toDenseMatrixData()
	exp = DMD(data,featureNames)

	assert ret.equals(exp)
	assert exp.equals(ret)



############
# writeCSV #
############

def writeCSV_handmade(constructor):
	""" Test writeCSV with both data and featureNames """
	tmpFile = tempfile.NamedTemporaryFile()

	# instantiate object
	data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	toWrite = constructor(data, featureNames)

	# call writeCSV
	toWrite.writeCSV(tmpFile.name, includeFeatureNames=True)

	# read it back into a different object, then test equality
	readObj = constructor(file=tmpFile.name)

	assert readObj.equals(toWrite)
	assert toWrite.equals(readObj)


###########
# writeMM #
###########


def writeMM_handmade(constructor):
	""" Test writeCSV with both data and featureNames """
	tmpFile = tempfile.NamedTemporaryFile()

	# instantiate object
	data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	toWrite = constructor(data, featureNames)

	# call writeCSV
	toWrite.writeMM(tmpFile.name, includeFeatureNames=True)

	opened = open(tmpFile.name,'r')
	print opened.read()
	for line in opened:
		print line

	# read it back into a different object, then test equality
	readObj = constructor(file=tmpFile.name)

	assert readObj.equals(toWrite)
	assert toWrite.equals(readObj)


#####################
# copyDataReference #
#####################

def copyDataReference_exceptionInconsistentFeatures(constructor):
	""" Test copyDataReference() throws exception when the number of features doesn't match. """

	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)

	data2 = [[-1,-2,-3,-4]]
	other = constructor(data2)

	orig.copyDataReference(other)


def copyDataReference_exceptionWrongType(constructor):
	""" Test copyDataReference() throws exception when other is not the same type """
	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)

	type1 = RLD(data1,featureNames)
	type2 = DMD(data1,featureNames)

	# at least one of these two will be the wrong type
	orig.copyDataReference(type1)
	orig.copyDataReference(type2)


def copyDataReference_sameReference(constructor):
	""" Test copyDataReference() successfully records the same reference """

	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)

	data2 = [[-1,-2,-3,]]
	other = constructor(data2)

	orig.copyDataReference(other)

	assert orig.data is other.data

