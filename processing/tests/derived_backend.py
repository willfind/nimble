"""
Backend for unit tests of the functions implemented by the derived classes.

These tests rely on having a working .equals method, which must be tested
directly by the class calling this backend.

"""

from ..base_data import *
from copy import deepcopy
from ..row_list_data import RowListData as RLD
from ..dense_matrix_data import DenseMatrixData as DMD

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
	""" Test appendPoints() for ArgumentException when toAppend has too many columns """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendPoints([["too", " ", "many", " ", "columns"]])

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
# appendColumns() #
################


def appendColumns_exceptionNone(constructor):
	""" Test appendColumns() for ArgumentException when toAppend is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendColumns(None)

def appendColumns_exceptionWrongSize(constructor):
	""" Test appendColumns() for ArgumentException when toAppend has too many points """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendColumns([["too"], [" "], ["many"], [" "], ["points"]])

def appendColumns_exceptionSameFeatureName(constructor):
	""" Test appendColumns() for ArgumentException when toAppend and self have a featureName in common """
	toTest1 = constructor([[1]],["hello"])
	toTest2 = constructor([[1,2]],["hello","goodbye"])
	toTest2.appendColumns(toTest1)

def appendColumns_handmadeSingle(constructor):
	""" Test appendColumns() against handmade output for a single added column"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)

	toAppend = constructor([[-1],[-2],[-3]],['-1'])

	dataExpected = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	featureNamesExpected = ['1','2','3','-1']
	expected = constructor(dataExpected,featureNamesExpected)

	toTest.appendColumns(toAppend)
	assert toTest.equals(expected)

def appendColumns_handmadeSequence(constructor):
	""" Test appendColumns() against handmade output for a sequence of additions"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)

	toAppend1 = [[0.1],[0.2],[0.3]]
	lab1 =  ['a']
	toAppend2 = [[0.01,0],[0.02,0],[0.03,0]]
	lab2 = ['A','0']
	toAppend3 = [[10],[11],[12]]
	lab3 = ['10']

	toTest.appendColumns(constructor(toAppend1,lab1))
	toTest.appendColumns(constructor(toAppend2,lab2))
	toTest.appendColumns(constructor(toAppend3,lab3))

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
# sortColumns() #
#################


def sortColumns_handmadeWithFcn(constructor):
	""" Test sortColumns() against handmade output when given cmp and key functions """	
	data = [[1,4,7],[2,5,8],[3,6,9]]
	toTest = constructor(data)

	def cmpNums(num1,num2):
		return num1 - num2

	def sumModEight(col):
		total = 0
		for value in col:
			total = total + value
		return total % 8

	toTest.sortColumns(cmpNums,sumModEight)
	toTest.sortColumns(key=sumModEight)

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
	""" Test extractPoints() for ArgumentException when start is not a valid column index """
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
# extractColumns() #
####################

def extractColumns_handmadeSingle(constructor):
	""" Test extractColumns() against handmade output when extracting one column """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ext1 = toTest.extractColumns(0)
	exp1 = constructor([[1],[4],[7]])
	assert ext1.equals(exp1)
	expEnd = constructor([[2,3],[5,6],[8,9]])
	assert toTest.equals(expEnd)

def extractColumns_handmadeListSequence(constructor):
	""" Test extractColumns() against handmade output for several extractions by list """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	toTest = constructor(data)
	ext1 = toTest.extractColumns([0])
	exp1 = constructor([[1],[4],[7]])
	assert ext1.equals(exp1)
	ext2 = toTest.extractColumns([1,2])
	exp2 = constructor([[3,-1],[6,-2],[9,-3]])
	assert ext2.equals(exp2)
	expEndData = [[2],[5],[8]]
	expEnd = constructor(expEndData)
	assert toTest.equals(expEnd)

def extractColumns_handmadeListWithFeatureName(constructor):
	""" Test extractColumns() against handmade output for list extraction when specifying featureNames """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	featureNames = ["one","two","three","neg"]
	toTest = constructor(data,featureNames)
	ext1 = toTest.extractColumns(["one"])
	exp1 = constructor([[1],[4],[7]], ["one"])
	assert ext1.equals(exp1)
	ext2 = toTest.extractColumns(["three","neg"])
	exp2 = constructor([[3,-1],[6,-2],[9,-3]],["three","neg"])
	assert ext2.equals(exp2)
	expEnd = constructor([[2],[5],[8]], ["two"])
	assert toTest.equals(expEnd)


def extractColumns_handmadeFunction(constructor):
	""" Test extractColumns() against handmade output for function extraction """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	toTest = constructor(data)
	def absoluteOne(column):
		if 1 in column or -1 in column:
			return True
		return False
	ext = toTest.extractColumns(absoluteOne)
	exp = constructor([[1,-1],[4,-2],[7,-3]])
	assert ext.equals(exp)
	expEnd = constructor([[2,3],[5,6],[8,9]])		
	assert toTest.equals(expEnd)


def extractColumns_handmadeFunctionWithFeatureName(constructor):
	""" Test extractColumns() against handmade output for function extraction with featureNames """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	featureNames = ["one","two","three","neg"]
	toTest = constructor(data,featureNames)
	def absoluteOne(column):
		if 1 in column or -1 in column:
			return True
		return False

	ext = toTest.extractColumns(absoluteOne)
	exp = constructor([[1,-1],[4,-2],[7,-3]], ['one','neg'])
	assert ext.equals(exp)
	expEnd = constructor([[2,3],[5,6],[8,9]],["two","three"])	
	assert toTest.equals(expEnd)


def extractColumns_exceptionStartInvalid(constructor):
	""" Test extractColumns() for ArgumentException when start is not a valid column index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractColumns(start=-1, end=2)

def extractColumns_exceptionStartInvalidFeatureName(constructor):
	""" Test extractColumns() for ArgumentException when start is not a valid column FeatureName """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractColumns(start="wrong", end=2)

def extractColumns_exceptionEndInvalid(constructor):
	""" Test extractColumns() for ArgumentException when start is not a valid column index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractColumns(start=0, end=5)

def extractColumns_exceptionEndInvalidFeatureName(constructor):
	""" Test extractColumns() for ArgumentException when start is not a valid featureName """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractColumns(start="two", end="five")

def extractColumns_exceptionInversion(constructor):
	""" Test extractColumns() for ArgumentException when start comes after end """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractColumns(start=2, end=0)

def extractColumns_exceptionInversionFeatureName(constructor):
	""" Test extractColumns() for ArgumentException when start comes after end as FeatureNames"""
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.extractColumns(start="two", end="one")

def extractColumns_handmadeRange(constructor):
	""" Test extractColumns() against handmade output for range extraction """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.extractColumns(start=1, end=2)
	
	expectedRet = constructor([[2,3],[5,6],[8,9]])
	expectedTest = constructor([[1],[4],[7]])

	assert expectedRet.equals(ret)
	assert expectedTest.equals(toTest)

def extractColumns_handmadeWithFeatureNames(constructor):
	""" Test extractColumns() against handmade output for range extraction with FeatureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.extractColumns(start=1,end=2)
	
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
# applyFunctionToEachColumn() #
#######################

def applyFunctionToEachColumn_exceptionInputNone(constructor):
	""" Test applyFunctionToEachColumn() for ArgumentException when function is None """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj= constructor(deepcopy(origData),featureNames)
	origObj.applyFunctionToEachColumn(None)

def applyFunctionToEachColumn_Handmade(constructor):
	""" Test applyFunctionToEachColumn() with handmade output """
	featureNames = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj= constructor(deepcopy(origData),featureNames)

	def emitAllEqual (column):
		first = column[0]
		for value in column:
			if value != first:
				return 0
		return 1

	lowerCounts = origObj.applyFunctionToEachColumn(emitAllEqual)
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
# convertToRowListData() #
##########################


def convertToRowListData_handmade_defaultFeatureNames(constructor):
	""" Test convertToRowListData with default featureNames """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)

	ret = toTest.convertToRowListData()
	exp = RLD(data)

	assert ret.equals(exp)
	assert exp.equals(ret)

	
def convertToRowListData_handmade_assignedFeatureNames(constructor):
	""" Test convertToRowListData with assigned featureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)

	ret = toTest.convertToRowListData()
	exp = RLD(data,featureNames)

	assert ret.equals(exp)
	assert exp.equals(ret)



##############################
# convertToDenseMatrixData() #
##############################

def convertToDenseMatrixData_handmade_defaultFeatureNames(constructor):
	""" Test convertToDenseMatrixData with default featureNames """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)

	ret = toTest.convertToDenseMatrixData()
	exp = DMD(data)

	assert ret.equals(exp)
	assert exp.equals(ret)

	
def convertToDenseMatrixData_handmade_assignedFeatureNames(constructor):
	""" Test convertToDenseMatrixData with assigned featureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)

	ret = toTest.convertToDenseMatrixData()
	exp = DMD(data,featureNames)

	assert ret.equals(exp)
	assert exp.equals(ret)










