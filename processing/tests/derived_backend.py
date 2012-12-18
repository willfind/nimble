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
# appendRows() #
#############

def appendRows_exceptionNone(constructor):
	""" Test appendRows() for ArgumentException when toAppend is None"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendRows(None)

def appendRows_exceptionWrongSize(constructor):
	""" Test appendRows() for ArgumentException when toAppend has too many columns """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendRows([["too", " ", "many", " ", "columns"]])

def appendRows_handmadeSingle(constructor):
	""" Test appendRows() against handmade output for a single added row """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	dataExpected = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	toTest = constructor(data)
	toAppend = constructor([[10,11,12]])
	expected = constructor(dataExpected)
	toTest.appendRows(toAppend)
	assert toTest.equals(expected)

def appendRows_handmadeSequence(constructor):
	""" Test appendRows() against handmade output for a sequence of additions"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toAppend1 = [[0.1,0.2,0.3]]
	toAppend2 = [[0.01,0.02,0.03],[0,0,0]]
	toAppend3 = [[10,11,12]]

	dataExpected = [[1,2,3],[4,5,6],[7,8,9],[0.1,0.2,0.3],[0.01,0.02,0.03],[0,0,0],[10,11,12]]
	toTest = constructor(data)
	toTest.appendRows(constructor(toAppend1))
	toTest.appendRows(constructor(toAppend2))
	toTest.appendRows(constructor(toAppend3))

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
	""" Test appendColumns() for ArgumentException when toAppend has too many rows """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.appendColumns([["too"], [" "], ["many"], [" "], ["rows"]])

def appendColumns_exceptionSameLabel(constructor):
	""" Test appendColumns() for ArgumentException when toAppend and self have a column label in common """
	toTest1 = constructor([[1]],["hello"])
	toTest2 = constructor([[1,2]],["hello","goodbye"])
	toTest2.appendColumns(toTest1)

def appendColumns_handmadeSingle(constructor):
	""" Test appendColumns() against handmade output for a single added column"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)

	toAppend = constructor([[-1],[-2],[-3]],['-1'])

	dataExpected = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	labelsExpected = ['1','2','3','-1']
	expected = constructor(dataExpected,labelsExpected)

	toTest.appendColumns(toAppend)
	assert toTest.equals(expected)

def appendColumns_handmadeSequence(constructor):
	""" Test appendColumns() against handmade output for a sequence of additions"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)

	toAppend1 = [[0.1],[0.2],[0.3]]
	lab1 =  ['a']
	toAppend2 = [[0.01,0],[0.02,0],[0.03,0]]
	lab2 = ['A','0']
	toAppend3 = [[10],[11],[12]]
	lab3 = ['10']

	toTest.appendColumns(constructor(toAppend1,lab1))
	toTest.appendColumns(constructor(toAppend2,lab2))
	toTest.appendColumns(constructor(toAppend3,lab3))

	labelsExpected = ['1','2','3','a','A','0','10']
	dataExpected = [[1,2,3,0.1,0.01,0,10],[4,5,6,0.2,0.02,0,11],[7,8,9,0.3,0.03,0,12]]

	expected = constructor(dataExpected,labelsExpected)
	assert toTest.equals(expected)



##############
# sortRows() #
##############

def sortRows_handmadeNatural(constructor):
	""" Test sortRows() against handmade , naturally ordered  output """	
	data = [[7,8,9],[1,2,3],[4,5,6]]
	toTest = constructor(data)

	toTest.sortRows()

	dataExpected = [[1,2,3],[4,5,6],[7,8,9]]
	objExp = constructor(dataExpected)

	assert toTest.equals(objExp)


def sortRows_handmadeWithFcn(constructor):
	""" Test sortRows() against handmade output when given cmp and key functions """	
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)

	def cmpNums(num1,num2):
		return num1 - num2

	def sumModEight(row):
		total = 0
		for value in row:
			total = total + value
		return total % 8

	toTest.sortRows(cmpNums,sumModEight)

	dataExpected = [[7,8,9],[1,2,3],[4,5,6]]
	objExp = constructor(dataExpected)

	assert toTest.equals(objExp)


def sortRows_handmade_reverse(constructor):
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

def sortRows_handmade_reverse(constructor):
	assert False

#################
# extractRows() #
#################

def extractRows_exceptionNone(constructor):
	""" Test extractRows() for ArgumentException when toRemove is none """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.extractRows(None)

def extractRows_handmadeSingle(constructor):
	""" Test extractRows() against handmade output when extracting one row """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ext1 = toTest.extractRows([0])
	exp1 = constructor([[1,2,3]])
	assert ext1.equals(exp1)
	expEnd = constructor([[4,5,6],[7,8,9]])
	assert toTest.equals(expEnd)

def extractRows_handmadeSequence(constructor):
	""" Test extractRows() against handmade output for several extractions """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	toTest = constructor(data)
	ext1 = toTest.extractRows([0])
	exp1 = constructor([[1,2,3]])
	assert ext1.equals(exp1)
	ext2 = toTest.extractRows([1,2])
	exp2 = constructor([[7,8,9],[10,11,12]])
	assert ext2.equals(exp2)
	expEnd = constructor([[4,5,6]])
	assert toTest.equals(expEnd)


####################
# extractColumns() #
####################

def extractColumns_exceptionNone(constructor):
	""" Test extractColumns() for ArgumentException when toRemove is none """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.extractColumns(None)

def extractColumns_handmadeSingle(constructor):
	""" Test extractColumns() against handmade output when extracting one row """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ext1 = toTest.extractColumns([0])
	exp1 = constructor([[1],[4],[7]])
	assert ext1.equals(exp1)
	expEnd = constructor([[2,3],[5,6],[8,9]])
	assert toTest.equals(expEnd)

def extractColumns_handmadeSequence(constructor):
	""" Test extractColumns() against handmade output for several extractions """
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

def extractColumns_handmadeByLabel(constructor):
	""" Test extractColumns() against handmade output when specifying labels """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	labels = ["one","two","three","neg"]
	toTest = constructor(data,labels)
	ext1 = toTest.extractColumns(["one"])
	exp1 = constructor([[1],[4],[7]], ["one"])
	assert ext1.equals(exp1)
	ext2 = toTest.extractColumns(["three","neg"])
	exp2 = constructor([[3,-1],[6,-2],[9,-3]],["three","neg"])
	assert ext2.equals(exp2)
	expEnd = constructor([[2],[5],[8]], ["two"])
	assert toTest.equals(expEnd)


###########################
# extractSatisfyingRows() #
###########################

def extractSatisfyingRows_exceptionNone(constructor):
	""" Test extractSatisfyingRows() for ArgumentException when toRemove is none """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.extractSatisfyingRows(None)

def extractSatisfyingRows_handmade(constructor):
	""" Test extractSatisfyingRows() against handmade output """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	def oneOrFour(row):
		if 1 in row or 4 in row:
			return True
		return False
	ext = toTest.extractSatisfyingRows(oneOrFour)
	exp = constructor([[1,2,3],[4,5,6]])
	assert ext.equals(exp)
	expEnd = constructor([[7,8,9]])
	assert toTest.equals(expEnd)


def extractSatisfyingRows_handmadeWithLabels(constructor):
	""" Test extractSatisfyingRows() against handmade output with labels"""
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	def oneOrFour(row):
		if 1 in row or 4 in row:
			return True
		return False
	ext = toTest.extractSatisfyingRows(oneOrFour)
	exp = constructor([[1,2,3],[4,5,6]],labels)
	assert ext.equals(exp)
	expEnd = constructor([[7,8,9]],labels)
	assert toTest.equals(expEnd)


def extractSatisfyingRows_handmadeExtractAll(constructor):
	pass
	#TODO
	#print

##############################
# extractSatisfyingColumns() #
##############################

def extractSatisfyingColumns_exceptionNone(constructor):
	""" Test extractSatisfyingColumns() for ArgumentException when toRemove is none """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	toTest.extractSatisfyingColumns(None)


def extractSatisfyingColumns_handmade(constructor):
	""" Test extractSatisfyingColumns() against handmade output """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	toTest = constructor(data)
	def absoluteOne(column):
		if 1 in column or -1 in column:
			return True
		return False
	ext = toTest.extractSatisfyingColumns(absoluteOne)
	exp = constructor([[1,-1],[4,-2],[7,-3]])
	assert ext.equals(exp)
	expEnd = constructor([[2,3],[5,6],[8,9]])	
	assert toTest.equals(expEnd)


def extractSatisfyingColumns_handmadeWithLabel(constructor):
	""" Test extractSatisfyingColumns() against handmade output with labels """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	labels = ["one","two","three","neg"]
	toTest = constructor(data,labels)
	def absoluteOne(column):
		if 1 in column or -1 in column:
			return True
		return False

	ext = toTest.extractSatisfyingColumns(absoluteOne)
	exp = constructor([[1,-1],[4,-2],[7,-3]], ['one','neg'])
	assert ext.equals(exp)
	expEnd = constructor([[2,3],[5,6],[8,9]],["two","three"])	
	assert toTest.equals(expEnd)



######################
# extractRangeRows() #
######################


def extractRangeRows_exceptionStartNone(constructor):
	""" Test extractRangeRows() for ArgumentException when start is None"""
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeRows(None,2)

def extractRangeRows_exceptionStartInvalid(constructor):
	""" Test extractRangeRows() for ArgumentException when start is not a valid row index """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeRows(-1,2)

def extractRangeRows_exceptionEndNone(constructor):
	""" Test extractRangeRows() for ArgumentException when end is None"""
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeRows(1,None)

def extractRangeRows_exceptionEndInvalid(constructor):
	""" Test extractRangeRows() for ArgumentException when start is not a valid column index """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeRows(start=1,end=5)

def extractRangeRows_exceptionInversion(constructor):
	""" Test extractRangeRows() for ArgumentException when start comes after end """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeRows(2,0)

def extractRangeRows_handmade(constructor):
	""" Test extractRangeRows() against handmade output """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.extractRangeRows(1,2)
	
	expectedRet = constructor([[4,5,6],[7,8,9]])
	expectedTest = constructor([[1,2,3]])

	assert expectedRet.equals(ret)
	assert expectedTest.equals(toTest)

def extractRangeRows_handmadeWithLabels(constructor):
	""" Test extractRangeRows() against handmade output with labels """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	ret = toTest.extractRangeRows(1,2)
	
	expectedRet = constructor([[4,5,6],[7,8,9]],labels)
	expectedTest = constructor([[1,2,3]],labels)

	assert expectedRet.equals(ret)
	assert expectedTest.equals(toTest)


#########################
# extractRangeColumns() #
#########################

def extractRangeColumns_exceptionStartNone(constructor):
	""" Test extractRangeColumns() for ArgumentException when start is None"""
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeColumns(None,2)


def extractRangeColumns_exceptionStartInvalid(constructor):
	""" Test extractRangeColumns() for ArgumentException when start is not a valid column index """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeColumns(-1,2)

def extractRangeColumns_exceptionStartInvalidLabel(constructor):
	""" Test extractRangeColumns() for ArgumentException when start is not a valid column Label """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeColumns("wrong",2)

def extractRangeColumns_exceptionEndNone(constructor):
	""" Test extractRangeColumns() for ArgumentException when end is None"""
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeColumns("one",None)

def extractRangeColumns_exceptionEndInvalid(constructor):
	""" Test extractRangeColumns() for ArgumentException when start is not a valid column index """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeColumns(0,5)

def extractRangeColumns_exceptionEndInvalidLabel(constructor):
	""" Test extractRangeColumns() for ArgumentException when start is not a valid column label """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeColumns("two","five")

def extractRangeColumns_exceptionInversion(constructor):
	""" Test extractRangeColumns() for ArgumentException when start comes after end """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeColumns(2,0)

def extractRangeColumns_exceptionInversionLabel(constructor):
	""" Test extractRangeColumns() for ArgumentException when start comes after end as Labels"""
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.extractRangeColumns("two","one")

def extractRangeColumns_handmade(constructor):
	""" Test extractRangeColumns() against handmade output """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.extractRangeColumns(1,2)
	
	expectedRet = constructor([[2,3],[5,6],[8,9]])
	expectedTest = constructor([[1],[4],[7]])

	assert expectedRet.equals(ret)
	assert expectedTest.equals(toTest)

def extractRangeColumns_handmadeWithLabels(constructor):
	""" Test extractRangeColumns() against handmade output with Labels """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	ret = toTest.extractRangeColumns(1,2)
	
	expectedRet = constructor([[2,3],[5,6],[8,9]],["two","three"])
	expectedTest = constructor([[1],[4],[7]],["one"])

	assert expectedRet.equals(ret)
	assert expectedTest.equals(toTest)



####################
# applyToEachRow() #
####################

def applyToEachRow_exceptionInputNone(constructor):
	""" Test applyToEachRow() for ArgumentException when function is None """
	labels = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj = constructor(deepcopy(origData),labels)
	origObj.applyToEachRow(None)

def applyToEachRow_Handmade(constructor):
	""" Test applyToEachRow() with handmade output """
	labels = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj = constructor(deepcopy(origData),labels)


	def emitLower (row):
		return row[origObj.labels['deci']]

	lowerCounts = origObj.applyToEachRow(emitLower)

	expectedOut = [[0.1], [0.1], [0.1], [0.2]]
	exp = constructor(expectedOut)

	assert lowerCounts.equals(exp)



#######################
# applyToEachColumn() #
#######################

def applyToEachColumn_exceptionInputNone(constructor):
	""" Test applyToEachColumn() for ArgumentException when function is None """
	labels = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj= constructor(deepcopy(origData),labels)
	origObj.applyToEachColumn(None)

def applyToEachColumn_Handmade(constructor):
	""" Test applyToEachColumn() with handmade output """
	labels = {'number':0,'centi':2,'deci':1}
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
	origObj= constructor(deepcopy(origData),labels)

	def emitAllEqual (column):
		first = column[0]
		for value in column:
			if value != first:
				return 0
		return 1

	lowerCounts = origObj.applyToEachColumn(emitAllEqual)
	expectedOut = [[1,0,0]]	
	assert lowerCounts.equals(constructor(expectedOut))



#####################
# mapReduceOnRows() #
#####################

def simpleMapper(row):
	idInt = row[0]
	intList = row[1:]
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

def mapReduceOnRows_argumentExceptionNoneMap(constructor):
	""" Test mapReduceOnRows() for ArgumentException when mapper is None """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.mapReduceOnRows(None,simpleReducer)

def mapReduceOnRows_argumentExceptionNoneReduce(constructor):
	""" Test mapReduceOnRows() for ArgumentException when reducer is None """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.mapReduceOnRows(simpleMapper,None)

def mapReduceOnRows_argumentExceptionUncallableMap(constructor):
	""" Test mapReduceOnRows() for ArgumentException when mapper is not callable """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.mapReduceOnRows("hello",simpleReducer)

def mapReduceOnRows_argumentExceptionUncallableReduce(constructor):
	""" Test mapReduceOnRows() for ArgumentException when reducer is not callable """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	toTest.mapReduceOnRows(simpleMapper,5)


# inconsistent output?



def mapReduceOnRows_handmade(constructor):
	""" Test mapReduceOnRows() against handmade output """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	ret = toTest.mapReduceOnRows(simpleMapper,simpleReducer)
	
	exp = constructor([[1,5],[4,11],[7,17]])
	
	assert (ret.equals(exp))
	assert (toTest.equals(constructor(data,labels)))


def mapReduceOnRows_handmadeNoneReturningReducer(constructor):
	""" Test mapReduceOnRows() against handmade output with a None returning Reducer """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)
	ret = toTest.mapReduceOnRows(simpleMapper,oddOnlyReducer)
	
	exp = constructor([[1,5],[7,17]])
	
	assert (ret.equals(exp))
	assert (toTest.equals(constructor(data,labels)))

	




##########################
# convertToRowListData() #
##########################


def convertToRowListData_handmade_defaultLabels(constructor):
	""" Test convertToRowListData with default labels """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)

	ret = toTest.convertToRowListData()
	exp = RLD(data)

	assert ret.equals(exp)
	assert exp.equals(ret)

	
def convertToRowListData_handmade_assignedLabels(constructor):
	""" Test convertToRowListData with assigned labels """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)

	ret = toTest.convertToRowListData()
	exp = RLD(data,labels)

	assert ret.equals(exp)
	assert exp.equals(ret)



##############################
# convertToDenseMatrixData() #
##############################

def convertToDenseMatrixData_handmade_defaultLabels(constructor):
	""" Test convertToDenseMatrixData with default labels """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)

	ret = toTest.convertToDenseMatrixData()
	exp = DMD(data)

	assert ret.equals(exp)
	assert exp.equals(ret)

	
def convertToDenseMatrixData_handmade_assignedLabels(constructor):
	""" Test convertToDenseMatrixData with assigned labels """
	labels = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,labels)

	ret = toTest.convertToDenseMatrixData()
	exp = DMD(data,labels)

	assert ret.equals(exp)
	assert exp.equals(ret)










