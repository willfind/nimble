"""
Backend for unit tests of the functions implemented by the derived classes.

These tests rely on having a working .isIdentical method, which must be tested
directly by the class calling this backend.

"""

import tempfile
import numpy

from copy import deepcopy
from UML import createData
from UML.data import List
from UML.data import Matrix
from UML.data import Sparse
from UML.data.dataHelpers import View

##############
# __init__() #
##############

def init_allEqual(constructor):
	""" Test __init__() that every way to instantiate produces equal objects """
	# instantiate from list of lists
	fromList = constructor(data=[[1,2,3]])

	# instantiate from csv file
	tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
	tmpCSV.write("1,2,3\n")
	tmpCSV.flush()
	fromCSV = constructor(data=tmpCSV.name)

	# instantiate from mtx array file
	tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx")
	tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
	tmpMTXArr.write("1 3\n")
	tmpMTXArr.write("1\n")
	tmpMTXArr.write("2\n")
	tmpMTXArr.write("3\n")
	tmpMTXArr.flush()
	fromMTXArr = constructor(data=tmpMTXArr.name)

	# instantiate from mtx coordinate file
	tmpMTXCoo = tempfile.NamedTemporaryFile(suffix=".mtx")
	tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
	tmpMTXCoo.write("1 3 3\n")
	tmpMTXCoo.write("1 1 1\n")
	tmpMTXCoo.write("1 2 2\n")
	tmpMTXCoo.write("1 3 3\n")
	tmpMTXCoo.flush()
	fromMTXCoo = constructor(data=tmpMTXCoo.name)

	# check equality between all pairs
	assert fromList.isIdentical(fromCSV)
	assert fromMTXArr.isIdentical(fromList)
	assert fromMTXArr.isIdentical(fromCSV)
	assert fromMTXCoo.isIdentical(fromList)
	assert fromMTXCoo.isIdentical(fromCSV)
	assert fromMTXCoo.isIdentical(fromMTXArr)

def init_allEqualWithFeatureNames(constructor):
	""" Test __init__() that every way to instantiate produces equal objects, with featureNames """
	# instantiate from list of lists
	fromList = constructor(data=[[1,2,3]], featureNames=['one', 'two', 'three'])

	# instantiate from csv file
	tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
	tmpCSV.write("#one,two,three\n")
	tmpCSV.write("1,2,3\n")
	tmpCSV.flush()
	fromCSV = constructor(data=tmpCSV.name)

	# instantiate from mtx file
	tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx")
	tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
	tmpMTXArr.write("%#one,two,three\n")
	tmpMTXArr.write("1 3\n")
	tmpMTXArr.write("1\n")
	tmpMTXArr.write("2\n")
	tmpMTXArr.write("3\n")
	tmpMTXArr.flush()
	fromMTXArr = constructor(data=tmpMTXArr.name)

	# instantiate from mtx coordinate file
	tmpMTXCoo = tempfile.NamedTemporaryFile(suffix=".mtx")
	tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
	tmpMTXCoo.write("%#one,two,three\n")
	tmpMTXCoo.write("1 3 3\n")
	tmpMTXCoo.write("1 1 1\n")
	tmpMTXCoo.write("1 2 2\n")
	tmpMTXCoo.write("1 3 3\n")
	tmpMTXCoo.flush()
	fromMTXCoo = constructor(data=tmpMTXCoo.name)

	# check equality between all pairs
	assert fromList.isIdentical(fromCSV)
	assert fromMTXArr.isIdentical(fromList)
	assert fromMTXArr.isIdentical(fromCSV)
	assert fromMTXCoo.isIdentical(fromList)
	assert fromMTXCoo.isIdentical(fromCSV)
	assert fromMTXCoo.isIdentical(fromMTXArr)




############
# isIdentical() #
############

def isIdentical_False(constructor):
	""" Test isIdentical() against some non-equal input """
	toTest = constructor([[4,5]])
	assert not toTest.isIdentical(constructor([[1,1],[2,2]]))
	assert not toTest.isIdentical(constructor([[1,2,3]]))
	assert not toTest.isIdentical(constructor([[1,2]]))

def isIdentical_True(constructor):
	""" Test isIdentical() against some actually equal input """
	toTest1 = constructor([[4,5]])
	toTest2 = constructor(deepcopy([[4,5]]))
	assert toTest1.isIdentical(toTest2)
	assert toTest2.isIdentical(toTest1)


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
	
	ret1 = dataObj1.transpose()
	assert dataObj1.isIdentical(dataObjT)
	assert dataObj1 == ret1
	dataObj1.transpose()
	dataObjT.transpose()
	assert dataObj1.isIdentical(dataObj2)
	assert dataObj2.isIdentical(dataObjT)


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
	toAppend = constructor([[2, 3, 4, 5, 6]])
	toTest.appendPoints(toAppend)

def appendPoints_exceptionMismatchedFeatureNames(constructor):
	""" Test appendPoints() for ArgumentException when toAppend and self's feature names do not match"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,['one','two','three'])
	toAppend = constructor([[11, 12, 13,]], ["two", 'one', 'three'])
	toTest.appendPoints(toAppend)

def appendPoints_handmadeSingle(constructor):
	""" Test appendPoints() against handmade output for a single added point """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	dataExpected = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	toTest = constructor(data)
	toAppend = constructor([[10,11,12]])
	expected = constructor(dataExpected)
	ret = toTest.appendPoints(toAppend)
	assert toTest.isIdentical(expected)
	assert toTest == ret

def appendPoints_handmadeSequence(constructor):
	""" Test appendPoints() against handmade output for a sequence of additions"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toAppend1 = [[0.1,0.2,0.3]]
	toAppend2 = [[0.01,0.02,0.03],[0,0,0]]
	toAppend3 = [[10,11,12]]

	dataExpected = [[1,2,3],[4,5,6],[7,8,9],[0.1,0.2,0.3],[0.01,0.02,0.03],[0,0,0],[10,11,12]]
	toTest = constructor(data)
	ret0 = toTest.appendPoints(constructor(toAppend1))
	ret1 = toTest.appendPoints(constructor(toAppend2))
	ret2 = toTest.appendPoints(constructor(toAppend3))

	expected = constructor(dataExpected)

	assert toTest.isIdentical(expected)
	assert toTest == ret0
	assert toTest == ret1
	assert toTest == ret2
	

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

def appendFeatures_exceptionMismatchedFeatureNames(constructor):
	""" Test appendFeatures() for ArgumentException when toAppend and self do not have equal featureNames """
	toTest1 = constructor([[2,1]],["goodbye","hello"])
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

	ret = toTest.appendFeatures(toAppend)
	assert toTest.isIdentical(expected)
	assert toTest == ret

def appendFeatures_handmadeSequence(constructor):
	""" Test appendFeatures() against handmade output for a sequence of additions"""
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)

	toAppend1 = [[0.1],[0.2],[0.3]]
	lab1 = ['a']
	toAppend2 = [[0.01,0],[0.02,0],[0.03,0]]
	lab2 = ['A','0']
	toAppend3 = [[10],[11],[12]]
	lab3 = ['10']

	ret1 = toTest.appendFeatures(constructor(toAppend1,lab1))
	ret2 = toTest.appendFeatures(constructor(toAppend2,lab2))
	ret3 = toTest.appendFeatures(constructor(toAppend3,lab3))

	featureNamesExpected = ['1','2','3','a','A','0','10']
	dataExpected = [[1,2,3,0.1,0.01,0,10],[4,5,6,0.2,0.02,0,11],[7,8,9,0.3,0.03,0,12]]

	expected = constructor(dataExpected,featureNamesExpected)
	assert toTest.isIdentical(expected)
	assert toTest == ret1
	assert toTest == ret2
	assert toTest == ret3



##############
# sortPoints() #
##############

def sortPoints_exceptionAtLeastOne(constructor):
	""" Test sortPoints() has at least one paramater """
	data = [[7,8,9],[1,2,3],[4,5,6]]
	toTest = constructor(data)

	toTest.sortPoints()

def sortPoints_naturalByFeature(constructor):
	""" Test sortPoints() when we specify a feature to sort by """	
	data = [[1,2,3],[7,1,9],[4,5,6]]
	toTest = constructor(data)

	ret = toTest.sortPoints(sortBy=1)

	dataExpected = [[7,1,9],[1,2,3],[4,5,6]]
	objExp = constructor(dataExpected)

	assert toTest.isIdentical(objExp)
	assert toTest == ret

def sortPoints_scorer(constructor):
	""" Test sortPoints() when we specify a scoring function """
	data = [[7,1,9],[1,2,3],[4,5,6]]
	toTest = constructor(data)

	def numOdds(point):
		assert isinstance(point, View)
		ret = 0
		for val in point:
			if val % 2 != 0:
				ret += 1
		return ret

	ret = toTest.sortPoints(sortHelper=numOdds)

	dataExpected = [[4,5,6],[1,2,3],[7,1,9]]
	objExp = constructor(dataExpected)

	assert toTest.isIdentical(objExp)
	assert toTest == ret

def sortPoints_comparator(constructor):
	""" Test sortPoints() when we specify a comparator function """
	data = [[7,1,9],[1,2,3],[4,5,6]]
	toTest = constructor(data)

	# comparator sort currently disabled for Matrix
	if isinstance(toTest, Matrix):
		return

	def compOdds(point1, point2):
		odds1 = 0
		odds2 = 0
		for val in point1:
			if val % 2 != 0:
				odds1 += 1
		for val in point2:
			if val % 2 != 0:
				odds2 += 1
		return odds1 - odds2

	ret = toTest.sortPoints(sortHelper=compOdds)

	dataExpected = [[4,5,6],[1,2,3],[7,1,9]]
	objExp = constructor(dataExpected)

	assert toTest.isIdentical(objExp)
	assert toTest == ret


#################
# sortFeatures() #
#################


def sortFeatures_exceptionAtLeastOne(constructor):
	""" Test sortFeatures() has at least one paramater """
	data = [[7,8,9],[1,2,3],[4,5,6]]
	toTest = constructor(data)

	toTest.sortFeatures()

def sortFeatures_naturalByPointWithNames(constructor):
	""" Test sortFeatures() when we specify a point to sort by; includes featureNames """	
	data = [[1,2,3],[7,1,9],[4,5,6]]
	names = ["1","2","3"]
	toTest = constructor(data,names)

	ret = toTest.sortFeatures(sortBy=1)

	dataExpected = [[2,1,3],[1,7,9],[5,4,6]]
	namesExp = ["2", "1", "3"]
	objExp = constructor(dataExpected, namesExp)

	assert toTest.isIdentical(objExp)
	assert toTest == ret

def sortFeatures_scorer(constructor):
	""" Test sortFeatures() when we specify a scoring function """
	data = [[7,1,9],[1,2,3],[4,2,9]]
	toTest = constructor(data)

	def numOdds(feature):
		ret = 0
		for val in feature:
			if val % 2 != 0:
				ret += 1
		return ret

	ret = toTest.sortFeatures(sortHelper=numOdds)

	dataExpected = [[1,7,9],[2,1,3],[2,4,9]]
	objExp = constructor(dataExpected)

	assert toTest.isIdentical(objExp)
	assert toTest == ret

def sortFeatures_comparator(constructor):
	""" Test sortFeatures() when we specify a comparator function """
	data = [[7,1,9],[1,2,3],[4,2,9]]
	toTest = constructor(data)

	# comparator sort currently disabled for Matrix
	if isinstance(toTest, Matrix):
		return

	def compOdds(point1, point2):
		odds1 = 0
		odds2 = 0
		for val in point1:
			if val % 2 != 0:
				odds1 += 1
		for val in point2:
			if val % 2 != 0:
				odds2 += 1
		return odds1 - odds2

	ret = toTest.sortFeatures(sortHelper=compOdds)

	dataExpected = [[1,7,9],[2,1,3],[2,4,9]]
	objExp = constructor(dataExpected)

	assert toTest.isIdentical(objExp)
	assert toTest == ret



#################
# extractPoints() #
#################

def extractPoints_handmadeSingle(constructor):
	""" Test extractPoints() against handmade output when extracting one point """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ext1 = toTest.extractPoints(0)
	exp1 = constructor([[1,2,3]])
	assert ext1.isIdentical(exp1)
	expEnd = constructor([[4,5,6],[7,8,9]])
	assert toTest.isIdentical(expEnd)

def extractPoints_handmadeListSequence(constructor):
	""" Test extractPoints() against handmade output for several list extractions """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	toTest = constructor(data)
	ext1 = toTest.extractPoints(0)
	exp1 = constructor([[1,2,3]])
	assert ext1.isIdentical(exp1)
	ext2 = toTest.extractPoints([1,2])
	exp2 = constructor([[7,8,9],[10,11,12]])
	assert ext2.isIdentical(exp2)
	expEnd = constructor([[4,5,6]])
	assert toTest.isIdentical(expEnd)

def extractPoints_handmadeListOrdering(constructor):
	""" Test extractPoints() against handmade output for out of order extraction """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
	toTest = constructor(data)
	ext1 = toTest.extractPoints([3,4,1])
	exp1 = constructor([[10,11,12],[13,14,15],[4,5,6]])
	assert ext1.isIdentical(exp1)
	expEnd = constructor([[1,2,3], [7,8,9]])
	assert toTest.isIdentical(expEnd)


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
	assert ext.isIdentical(exp)
	expEnd = constructor([[7,8,9]])
	assert toTest.isIdentical(expEnd)

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
	assert ext.isIdentical(exp)
	expEnd = constructor([[7,8,9]],featureNames)
	assert toTest.isIdentical(expEnd)

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

def extractPoints_handmadeRange(constructor):
	""" Test extractPoints() against handmade output for range extraction """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.extractPoints(start=1,end=2)
	
	expectedRet = constructor([[4,5,6],[7,8,9]])
	expectedTest = constructor([[1,2,3]])

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)

def extractPoints_handmadeRangeWithFeatureNames(constructor):
	""" Test extractPoints() against handmade output for range extraction with featureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.extractPoints(start=1,end=2)
	
	expectedRet = constructor([[4,5,6],[7,8,9]],featureNames)
	expectedTest = constructor([[1,2,3]],featureNames)

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)

def extractPoints_handmadeRangeRand_FM(constructor):
	""" Test extractPoints() against handmade output for randomized range extraction with featureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.extractPoints(start=0, end=2, number=2, randomize=True)
	
	expectedRet = constructor([[1,2,3],[4,5,6]],featureNames)
	expectedTest = constructor([[7,8,9]],featureNames)
	
	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)

def extractPoints_handmadeRangeDefaults(constructor):
	""" Test extractPoints uses the correct defaults in the case of range based extraction """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.extractPoints(end=1)
	
	expectedRet = constructor([[1,2,3],[4,5,6]],featureNames)
	expectedTest = constructor([[7,8,9]],featureNames)
	
	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)

	toTest = constructor(data,featureNames)
	ret = toTest.extractPoints(start=1)

	expectedTest = constructor([[1,2,3]],featureNames)
	expectedRet = constructor([[4,5,6],[7,8,9]],featureNames)

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)


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

	assert ext1.isIdentical(exp1)
	expEnd = constructor([[2,3],[5,6],[8,9]])
	assert toTest.isIdentical(expEnd)

def extractFeatures_handmadeListSequence(constructor):
	""" Test extractFeatures() against handmade output for several extractions by list """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	toTest = constructor(data)
	ext1 = toTest.extractFeatures([0])
	exp1 = constructor([[1],[4],[7]])
	assert ext1.isIdentical(exp1)
	ext2 = toTest.extractFeatures([1,2])
	exp2 = constructor([[3,-1],[6,-2],[9,-3]])
	assert ext2.isIdentical(exp2)
	expEndData = [[2],[5],[8]]
	expEnd = constructor(expEndData)
	assert toTest.isIdentical(expEnd)

def extractFeatures_handmadeListWithFeatureName(constructor):
	""" Test extractFeatures() against handmade output for list extraction when specifying featureNames """
	data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
	featureNames = ["one","two","three","neg"]
	toTest = constructor(data,featureNames)
	ext1 = toTest.extractFeatures(["one"])
	exp1 = constructor([[1],[4],[7]], ["one"])
	assert ext1.isIdentical(exp1)
	ext2 = toTest.extractFeatures(["three","neg"])
	exp2 = constructor([[3,-1],[6,-2],[9,-3]],["three","neg"])
	assert ext2.isIdentical(exp2)
	expEnd = constructor([[2],[5],[8]], ["two"])
	assert toTest.isIdentical(expEnd)


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
	assert ext.isIdentical(exp)
	expEnd = constructor([[2,3],[5,6],[8,9]])	
	assert toTest.isIdentical(expEnd)


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
	assert ext.isIdentical(exp)
	expEnd = constructor([[2,3],[5,6],[8,9]],["two","three"])	
	assert toTest.isIdentical(expEnd)


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

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)

def extractFeatures_handmadeWithFeatureNames(constructor):
	""" Test extractFeatures() against handmade output for range extraction with FeatureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.extractFeatures(start=1,end=2)
	
	expectedRet = constructor([[2,3],[5,6],[8,9]],["two","three"])
	expectedTest = constructor([[1],[4],[7]],["one"])

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)



############
# writeFile #
############

def writeFileCSV_handmade(constructor):
	""" Test writeFile() for csv extension with both data and featureNames """
	tmpFile = tempfile.NamedTemporaryFile(suffix=".csv")

	# instantiate object
	data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	toWrite = constructor(data, featureNames)

	# call writeFile
	toWrite.writeFile(tmpFile.name, format='csv', includeFeatureNames=True)

#	opened = open(tmpFile.name,'r')
#	print opened.read()
#	for line in opened:
#		print line

	# read it back into a different object, then test equality
	readObj = constructor(data=tmpFile.name)

	assert readObj.isIdentical(toWrite)
	assert toWrite.isIdentical(readObj)


def writeFileMTX_handmade(constructor):
	""" Test writeFile() for mtx extension with both data and featureNames """
	tmpFile = tempfile.NamedTemporaryFile(suffix=".mtx")

	# instantiate object
	data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	toWrite = constructor(data, featureNames)

	# call writeFile
	toWrite.writeFile(tmpFile.name, format='mtx', includeFeatureNames=True)

	# read it back into a different object, then test equality
	readObj = constructor(data=tmpFile.name)

	assert readObj.isIdentical(toWrite)
	assert toWrite.isIdentical(readObj)


#####################
# referenceDataFrom #
#####################

def referenceDataFrom_exceptionWrongType(constructor):
	""" Test referenceDataFrom() throws exception when other is not the same type """
	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)

	type1 = List(data1,featureNames)
	type2 = Matrix(data1,featureNames)

	# at least one of these two will be the wrong type
	orig.referenceDataFrom(type1)
	orig.referenceDataFrom(type2)


def referenceDataFrom_sameReference(constructor):
	""" Test copyReference() successfully records the same reference """

	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)

	data2 = [[-1,-2,-3,]]
	other = constructor(data2)

	ret = orig.referenceDataFrom(other)

	assert orig.data is other.data
	assert orig == ret

#############
# copy #
#############

def copy_withZeros(constructor):
	""" Test copy() produces an equal object and doesn't just copy the references """
	data1 = [[1,2,3,0],[1,0,3,0],[2,4,6,0],[0,0,0,0]]
	featureNames = ['one', 'two', 'three', 'four']
	orig = constructor(data1, featureNames)

	dup = orig.copy()

	assert orig.isIdentical(dup)
	assert dup.isIdentical(orig)

	assert orig.data is not dup.data


def copy_rightTypeTrueCopy(constructor):
	""" Test copy() will return all of the right type and do not show each other's modifications"""

	data = [[1,2,3],[1,0,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data, featureNames)
	sparseObj = createData(retType="Sparse", data=data, featureNames=featureNames)
	listObj = createData(retType="List", data=data, featureNames=featureNames)
	matixObj = createData(retType="Matrix", data=data, featureNames=featureNames)

	pointsShuffleIndices = [3,1,2,0]
	featuresshuffleIndices = [1,2,0]

	copySparse = orig.copy(asType='Sparse')
	assert copySparse.isIdentical(sparseObj)
	assert sparseObj.isIdentical(copySparse)
	assert type(copySparse) == Sparse
	copySparse.setFeatureName('two', '2')
	assert 'two' in orig.featureNames
	copySparse.shufflePoints(pointsShuffleIndices)
	copySparse.shuffleFeatures(featuresshuffleIndices)
	assert orig[0,0] == 1 

	copyList = orig.copy(asType='List')
	assert copyList.isIdentical(listObj)
	assert listObj.isIdentical(copyList)
	assert type(copyList) == List
	copyList.setFeatureName('two', '2')
	assert 'two' in orig.featureNames
	copyList.shufflePoints(pointsShuffleIndices)
	copyList.shuffleFeatures(featuresshuffleIndices)
	assert orig[0,0] == 1 

	copyMatrix = orig.copy(asType='Matrix')
	assert copyMatrix.isIdentical(matixObj)
	assert matixObj.isIdentical(copyMatrix)
	assert type(copyMatrix) == Matrix
	copyMatrix.setFeatureName('two', '2')
	assert 'two' in orig.featureNames
	copyMatrix.shufflePoints(pointsShuffleIndices)
	copyMatrix.shuffleFeatures(featuresshuffleIndices)
	assert orig[0,0] == 1 

	pyList = orig.copy(asType='python list')
	assert type(pyList) == list
	pyList[0][0] = 5
	assert orig[0,0] == 1 

	numpyArray = orig.copy(asType='numpy array')
	assert type(numpyArray) == type(numpy.array([]))
	numpyArray[0,0] = 5
	assert orig[0,0] == 1 

	numpyMatrix = orig.copy(asType='numpy matrix')
	assert type(numpyMatrix) == type(numpy.matrix([]))
	numpyMatrix[0,0] = 5
	assert orig[0,0] == 1 


###################
# copyPoints #
###################

def copyPoints_exceptionNone(constructor):
	""" Test copyPoints() for exception when argument is None """

	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)
	orig.copyPoints(None)

def copyPoints_exceptionNonIndex(constructor):
	""" Test copyPoints() for exception when a value in the input is not a valid index """
	
	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)
	orig.copyPoints([1,'yes'])


def copyPoints_handmadeContents(constructor):
	""" Test copyPoints() returns the correct data """

	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)
	expOrig = constructor(data1, featureNames)

	data2 = [[1,2,3],[2,4,6]]
	expRet = constructor(data2, featureNames)

	ret = orig.copyPoints([1,2])

	assert orig.isIdentical(expOrig)
	assert ret.isIdentical(expRet)

def copyPoints_exceptionStartInvalid(constructor):
	""" Test copyPoints() for ArgumentException when start is not a valid point index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.copyPoints(start=-1,end=2)

def copyPoints_exceptionEndInvalid(constructor):
	""" Test copyPoints() for ArgumentException when start is not a valid feature index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.copyPoints(start=1,end=5)

def copyPoints_exceptionInversion(constructor):
	""" Test copyPoints() for ArgumentException when start comes after end """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.copyPoints(start=2,end=0)

def copyPoints_handmadeRange(constructor):
	""" Test copyPoints() against handmade output for range copying """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.copyPoints(start=1,end=2)
	
	expectedRet = constructor([[4,5,6],[7,8,9]])
	expectedTest = constructor(data)

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)

def copyPoints_handmadeRangeWithFeatureNames(constructor):
	""" Test copyPoints() against handmade output for range copying with featureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.copyPoints(start=1,end=2)
	
	expectedRet = constructor([[4,5,6],[7,8,9]],featureNames)
	expectedTest = constructor(data,featureNames)

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)

def copyPoints_handmadeRangeDefaults(constructor):
	""" Test copyPoints uses the correct defaults in the case of range based copying """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.copyPoints(end=1)
	
	expectedRet = constructor([[1,2,3],[4,5,6]],featureNames)
	expectedTest = constructor(data,featureNames)
	
	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)

	toTest = constructor(data,featureNames)
	ret = toTest.copyPoints(start=1)

	expectedTest = constructor(data,featureNames)
	expectedRet = constructor([[4,5,6],[7,8,9]],featureNames)

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)




#####################
# copyFeatures #
#####################


def copyFeatures_exceptionNone(constructor):
	""" Test copyFeatures() for exception when argument is None """

	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)
	orig.copyFeatures(None)


def copyFeatures_exceptionNonIndex(constructor):
	""" Test copyFeatures() for exception when a value in the input is not a valid index """
	
	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)
	orig.copyFeatures([1,'yes'])


def copyFeatures_handmadeContents(constructor):
	""" Test copyFeatures() returns the correct data """

	data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
	featureNames = ['one', 'two', 'three']
	orig = constructor(data1, featureNames)
	expOrig = constructor(data1, featureNames)

	data2 = [[1,2],[1,2],[2,4],[0,0]]

	expRet = constructor(data2, ['one','two'])

	ret = orig.copyFeatures([0,'two'])

	assert orig.isIdentical(expOrig)
	assert ret.isIdentical(expRet)


####


def copyFeatures_exceptionStartInvalid(constructor):
	""" Test copyFeatures() for ArgumentException when start is not a valid feature index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.copyFeatures(start=-1, end=2)

def copyFeatures_exceptionStartInvalidFeatureName(constructor):
	""" Test copyFeatures() for ArgumentException when start is not a valid feature FeatureName """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.copyFeatures(start="wrong", end=2)

def copyFeatures_exceptionEndInvalid(constructor):
	""" Test copyFeatures() for ArgumentException when start is not a valid feature index """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.copyFeatures(start=0, end=5)

def copyFeatures_exceptionEndInvalidFeatureName(constructor):
	""" Test copyFeatures() for ArgumentException when start is not a valid featureName """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.copyFeatures(start="two", end="five")

def copyFeatures_exceptionInversion(constructor):
	""" Test copyFeatures() for ArgumentException when start comes after end """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.copyFeatures(start=2, end=0)

def copyFeatures_exceptionInversionFeatureName(constructor):
	""" Test copyFeatures() for ArgumentException when start comes after end as FeatureNames"""
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	toTest.copyFeatures(start="two", end="one")

def copyFeatures_handmadeRange(constructor):
	""" Test copyFeatures() against handmade output for range copying """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data)
	ret = toTest.copyFeatures(start=1, end=2)
	
	expectedRet = constructor([[2,3],[5,6],[8,9]])
	expectedTest = constructor(data)

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)

def copyFeatures_handmadeWithFeatureNames(constructor):
	""" Test copyFeatures() against handmade output for range copying with FeatureNames """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)
	ret = toTest.copyFeatures(start=1,end=2)
	
	expectedRet = constructor([[2,3],[5,6],[8,9]],["two","three"])
	expectedTest = constructor(data, featureNames)

	assert expectedRet.isIdentical(ret)
	assert expectedTest.isIdentical(toTest)


##############
# __getitem__#
##############


def getitem_simpleExampeWithZeroes(constructor):
	""" Test __getitem__ returns the correct output for a number of simple queries """
	featureNames = ["one","two","three","zero"]
	data = [[1,2,3,0],[4,5,0,0],[7,0,9,0],[0,0,0,0]]

	toTest = constructor(data, featureNames)

	assert toTest[0,0] == 1
	assert toTest[1,3] == 0
	assert toTest[2,2] == 9
	assert toTest[3,3] == 0

	assert toTest[1,'one'] == 4



################
# pointView #
################

def pointView_isinstance(constructor):
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)

	pView = toTest.pointView(0)

	assert isinstance(pView, View)
	assert pView.name() is None
	assert pView.index() >= 0 and pView.index() < toTest.points()
	assert len(pView) == toTest.features()
	assert pView[0] == 1
	assert pView['two'] == 2
	assert pView['three'] == 3
	pView[0] = -1
	pView['two'] = -2
	pView['three'] = -3
	assert pView[0] == -1
	assert pView['two'] == -2
	assert pView['three'] == -3


##################
# featureView #
##################

def featureView_isinstance(constructor):
	""" Test featureView() returns an instance of the View in dataHelpers """
	featureNames = ["one","two","three"]
	data = [[1,2,3],[4,5,6],[7,8,9]]
	toTest = constructor(data,featureNames)

	fView = toTest.featureView('one')

	assert isinstance(fView, View)
	assert fView.name() is not None
	assert fView.index() >= 0 and fView.index() < toTest.features()
	assert len(fView) == toTest.points()
	assert fView[0] == 1
	assert fView[1] == 4
	assert fView[2] == 7
	fView[0] = -1
	fView[1] = -4
	fView[2] = -7
	assert fView[0] == -1
	assert fView[1] == -4
	assert fView[2] == -7



############
# points() #
############

def points_vectorTest(constructor):
	""" Test points() when we only have row or column vectors of data """
	dataR = [[1,2,3]]
	dataC = [[1], [2], [3]]

	toTestR = constructor(dataR)
	toTestC = constructor(dataC)

	rPoints = toTestR.points()
	cPoints = toTestC.points()

	assert rPoints == 1
	assert cPoints == 3

############
# features() #
############

def features_vectorTest(constructor):
	""" Test features() when we only have row or column vectors of data """
	dataR = [[1,2,3]]
	dataC = [[1], [2], [3]]

	toTestR = constructor(dataR)
	toTestC = constructor(dataC)

	rFeatures = toTestR.features()
	cFeatures = toTestC.features()

	assert rFeatures == 3
	assert cFeatures == 1


