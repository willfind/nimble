"""
Backend for unit tests of the functions implemented by the derived classes.

These tests rely on having a working .isIdentical method, which must be tested
directly by the class calling this backend.

"""

import tempfile
import numpy
from nose.tools import *

from copy import deepcopy
from UML import createData
from UML.data import List
from UML.data import Matrix
from UML.data import Sparse
from UML.data.dataHelpers import View
from UML.exceptions import ArgumentException


class DerivedBackend(object):

	def __init__(self, constructor):
		self.constructor = constructor
#		super(DerivedBackend, self).__init__()

	##############
	# __init__() #
	##############

	def test_init_allEqual(self):
		""" Test __init__() that every way to instantiate produces equal objects """
		# instantiate from list of lists
		fromList = self.constructor(data=[[1,2,3]])

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("1,2,3\n")
		tmpCSV.flush()
		fromCSV = self.constructor(data=tmpCSV.name)

		# instantiate from mtx array file
		tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
		tmpMTXArr.write("1 3\n")
		tmpMTXArr.write("1\n")
		tmpMTXArr.write("2\n")
		tmpMTXArr.write("3\n")
		tmpMTXArr.flush()
		fromMTXArr = self.constructor(data=tmpMTXArr.name)

		# instantiate from mtx coordinate file
		tmpMTXCoo = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
		tmpMTXCoo.write("1 3 3\n")
		tmpMTXCoo.write("1 1 1\n")
		tmpMTXCoo.write("1 2 2\n")
		tmpMTXCoo.write("1 3 3\n")
		tmpMTXCoo.flush()
		fromMTXCoo = self.constructor(data=tmpMTXCoo.name)

		# check equality between all pairs
		assert fromList.isIdentical(fromCSV)
		assert fromMTXArr.isIdentical(fromList)
		assert fromMTXArr.isIdentical(fromCSV)
		assert fromMTXCoo.isIdentical(fromList)
		assert fromMTXCoo.isIdentical(fromCSV)
		assert fromMTXCoo.isIdentical(fromMTXArr)

	def test_init_allEqualWithFeatureNames(self):
		""" Test __init__() that every way to instantiate produces equal objects, with featureNames """
		# instantiate from list of lists
		fromList = self.constructor(data=[[1,2,3]], featureNames=['one', 'two', 'three'])

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("#one,two,three\n")
		tmpCSV.write("1,2,3\n")
		tmpCSV.flush()
		fromCSV = self.constructor(data=tmpCSV.name)

		# instantiate from mtx file
		tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
		tmpMTXArr.write("%#one,two,three\n")
		tmpMTXArr.write("1 3\n")
		tmpMTXArr.write("1\n")
		tmpMTXArr.write("2\n")
		tmpMTXArr.write("3\n")
		tmpMTXArr.flush()
		fromMTXArr = self.constructor(data=tmpMTXArr.name)

		# instantiate from mtx coordinate file
		tmpMTXCoo = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXCoo.write("%%MatrixMarket matrix coordinate integer general\n")
		tmpMTXCoo.write("%#one,two,three\n")
		tmpMTXCoo.write("1 3 3\n")
		tmpMTXCoo.write("1 1 1\n")
		tmpMTXCoo.write("1 2 2\n")
		tmpMTXCoo.write("1 3 3\n")
		tmpMTXCoo.flush()
		fromMTXCoo = self.constructor(data=tmpMTXCoo.name)

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

	def test_isIdentical_False(self):
		""" Test isIdentical() against some non-equal input """
		toTest = self.constructor([[4,5]])
		assert not toTest.isIdentical(self.constructor([[1,1],[2,2]]))
		assert not toTest.isIdentical(self.constructor([[1,2,3]]))
		assert not toTest.isIdentical(self.constructor([[1,2]]))

	def test_isIdentical_True(self):
		""" Test isIdentical() against some actually equal input """
		toTest1 = self.constructor([[4,5]])
		toTest2 = self.constructor(deepcopy([[4,5]]))
		assert toTest1.isIdentical(toTest2)
		assert toTest2.isIdentical(toTest1)



	###############
	# transpose() #
	###############

	def test_transpose_empty(self):
		""" Test transpose() on different kinds of emptiness """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)

		toTest.transpose()

		exp1 = [[],[]]
		exp1 = numpy.array(exp1)
		ret1 = self.constructor(exp1)
		assert ret1.isIdentical(toTest)

		toTest.transpose()

		exp2 = [[],[]]
		exp2 = numpy.array(exp2).T
		ret2 = self.constructor(exp2)
		assert ret2.isIdentical(toTest)


	def test_transpose_handmade(self):
		""" Test transpose() function against handmade output """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		dataTrans = [[1,4,7],[2,5,8],[3,6,9]]

		dataObj1 = self.constructor(deepcopy(data))
		dataObj2 = self.constructor(deepcopy(data))
		dataObjT = self.constructor(deepcopy(dataTrans))
		
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

	@raises(ArgumentException)
	def test_appendPoints_exceptionNone(self):
		""" Test appendPoints() for ArgumentException when toAppend is None"""
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		toTest.appendPoints(None)

	@raises(ArgumentException)
	def test_appendPoints_exceptionWrongSize(self):
		""" Test appendPoints() for ArgumentException when toAppend has too many features """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		toAppend = self.constructor([[2, 3, 4, 5, 6]])
		toTest.appendPoints(toAppend)

	@raises(ArgumentException)
	def test_appendPoints_exceptionMismatchedFeatureNames(self):
		""" Test appendPoints() for ArgumentException when toAppend and self's feature names do not match"""
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,['one','two','three'])
		toAppend = self.constructor([[11, 12, 13,]], ["two", 'one', 'three'])
		toTest.appendPoints(toAppend)

	def test_appendPoints_outOfPEmpty(self):
		""" Test appendPoints() when the calling object is point empty """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)

		data = [[1,2]]
		toAdd = self.constructor(data)
		toExp = self.constructor(data)

		toTest.appendPoints(toAdd)
		assert toTest.isIdentical(toExp)

	def test_appendPoints_handmadeSingle(self):
		""" Test appendPoints() against handmade output for a single added point """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		dataExpected = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
		toTest = self.constructor(data)
		toAppend = self.constructor([[10,11,12]])
		expected = self.constructor(dataExpected)
		ret = toTest.appendPoints(toAppend)
		assert toTest.isIdentical(expected)
		assert toTest == ret

	def test_appendPoints_handmadeSequence(self):
		""" Test appendPoints() against handmade output for a sequence of additions"""
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toAppend1 = [[0.1,0.2,0.3]]
		toAppend2 = [[0.01,0.02,0.03],[0,0,0]]
		toAppend3 = [[10,11,12]]

		dataExpected = [[1,2,3],[4,5,6],[7,8,9],[0.1,0.2,0.3],[0.01,0.02,0.03],[0,0,0],[10,11,12]]
		toTest = self.constructor(data)
		ret0 = toTest.appendPoints(self.constructor(toAppend1))
		ret1 = toTest.appendPoints(self.constructor(toAppend2))
		ret2 = toTest.appendPoints(self.constructor(toAppend3))

		expected = self.constructor(dataExpected)

		assert toTest.isIdentical(expected)
		assert toTest == ret0
		assert toTest == ret1
		assert toTest == ret2
		

	################
	# appendFeatures() #
	################

	@raises(ArgumentException)
	def test_appendFeatures_exceptionNone(self):
		""" Test appendFeatures() for ArgumentException when toAppend is None """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		toTest.appendFeatures(None)

	@raises(ArgumentException)
	def test_appendFeatures_exceptionWrongSize(self):
		""" Test appendFeatures() for ArgumentException when toAppend has too many points """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		toTest.appendFeatures([["too"], [" "], ["many"], [" "], ["points"]])

	@raises(ArgumentException)
	def test_appendFeatures_exceptionSameFeatureName(self):
		""" Test appendFeatures() for ArgumentException when toAppend and self have a featureName in common """
		toTest1 = self.constructor([[1]],["hello"])
		toTest2 = self.constructor([[1,2]],["hello","goodbye"])
		toTest2.appendFeatures(toTest1)

	@raises(ArgumentException)
	def test_appendFeatures_exceptionMismatchedFeatureNames(self):
		""" Test appendFeatures() for ArgumentException when toAppend and self do not have equal featureNames """
		toTest1 = self.constructor([[2,1]],["goodbye","hello"])
		toTest2 = self.constructor([[1,2]],["hello","goodbye"])
		toTest2.appendFeatures(toTest1)

	def test_appendFeatures_outOfPEmpty(self):
		""" Test appendFeatures() when the calling object is feature empty """
		data = [[],[]]
		data = numpy.array(data)
		toTest = self.constructor(data)

		data = [[1],[2]]
		toAdd = self.constructor(data)
		toExp = self.constructor(data)

		toTest.appendFeatures(toAdd)
		assert toTest.isIdentical(toExp)

	def test_appendFeatures_handmadeSingle(self):
		""" Test appendFeatures() against handmade output for a single added feature"""
		data = [[1,2,3],[4,5,6],[7,8,9]]
		featureNames = ['1','2','3']
		toTest = self.constructor(data,featureNames)

		toAppend = self.constructor([[-1],[-2],[-3]],['-1'])

		dataExpected = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
		featureNamesExpected = ['1','2','3','-1']
		expected = self.constructor(dataExpected,featureNamesExpected)

		ret = toTest.appendFeatures(toAppend)
		assert toTest.isIdentical(expected)
		assert toTest == ret

	def test_appendFeatures_handmadeSequence(self):
		""" Test appendFeatures() against handmade output for a sequence of additions"""
		data = [[1,2,3],[4,5,6],[7,8,9]]
		featureNames = ['1','2','3']
		toTest = self.constructor(data,featureNames)

		toAppend1 = [[0.1],[0.2],[0.3]]
		lab1 = ['a']
		toAppend2 = [[0.01,0],[0.02,0],[0.03,0]]
		lab2 = ['A','0']
		toAppend3 = [[10],[11],[12]]
		lab3 = ['10']

		ret1 = toTest.appendFeatures(self.constructor(toAppend1,lab1))
		ret2 = toTest.appendFeatures(self.constructor(toAppend2,lab2))
		ret3 = toTest.appendFeatures(self.constructor(toAppend3,lab3))

		featureNamesExpected = ['1','2','3','a','A','0','10']
		dataExpected = [[1,2,3,0.1,0.01,0,10],[4,5,6,0.2,0.02,0,11],[7,8,9,0.3,0.03,0,12]]

		expected = self.constructor(dataExpected,featureNamesExpected)
		assert toTest.isIdentical(expected)
		assert toTest == ret1
		assert toTest == ret2
		assert toTest == ret3



	##############
	# sortPoints() #
	##############

	@raises(ArgumentException)
	def test_sortPoints_exceptionAtLeastOne(self):
		""" Test sortPoints() has at least one paramater """
		data = [[7,8,9],[1,2,3],[4,5,6]]
		toTest = self.constructor(data)

		toTest.sortPoints()

	def test_sortPoints_naturalByFeature(self):
		""" Test sortPoints() when we specify a feature to sort by """	
		data = [[1,2,3],[7,1,9],[4,5,6]]
		toTest = self.constructor(data)

		ret = toTest.sortPoints(sortBy=1)

		dataExpected = [[7,1,9],[1,2,3],[4,5,6]]
		objExp = self.constructor(dataExpected)

		assert toTest.isIdentical(objExp)
		assert toTest == ret

	def test_sortPoints_scorer(self):
		""" Test sortPoints() when we specify a scoring function """
		data = [[7,1,9],[1,2,3],[4,5,6]]
		toTest = self.constructor(data)

		def numOdds(point):
			assert isinstance(point, View)
			ret = 0
			for val in point:
				if val % 2 != 0:
					ret += 1
			return ret

		ret = toTest.sortPoints(sortHelper=numOdds)

		dataExpected = [[4,5,6],[1,2,3],[7,1,9]]
		objExp = self.constructor(dataExpected)

		assert toTest.isIdentical(objExp)
		assert toTest == ret

	def test_sortPoints_comparator(self):
		""" Test sortPoints() when we specify a comparator function """
		data = [[7,1,9],[1,2,3],[4,5,6]]
		toTest = self.constructor(data)

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
		objExp = self.constructor(dataExpected)

		assert toTest.isIdentical(objExp)
		assert toTest == ret


	#################
	# sortFeatures() #
	#################

	@raises(ArgumentException)
	def test_sortFeatures_exceptionAtLeastOne(self):
		""" Test sortFeatures() has at least one paramater """
		data = [[7,8,9],[1,2,3],[4,5,6]]
		toTest = self.constructor(data)

		toTest.sortFeatures()

	def test_sortFeatures_naturalByPointWithNames(self):
		""" Test sortFeatures() when we specify a point to sort by; includes featureNames """	
		data = [[1,2,3],[7,1,9],[4,5,6]]
		names = ["1","2","3"]
		toTest = self.constructor(data,names)

		ret = toTest.sortFeatures(sortBy=1)

		dataExpected = [[2,1,3],[1,7,9],[5,4,6]]
		namesExp = ["2", "1", "3"]
		objExp = self.constructor(dataExpected, namesExp)

		assert toTest.isIdentical(objExp)
		assert toTest == ret

	def test_sortFeatures_scorer(self):
		""" Test sortFeatures() when we specify a scoring function """
		data = [[7,1,9],[1,2,3],[4,2,9]]
		toTest = self.constructor(data)

		def numOdds(feature):
			ret = 0
			for val in feature:
				if val % 2 != 0:
					ret += 1
			return ret

		ret = toTest.sortFeatures(sortHelper=numOdds)

		dataExpected = [[1,7,9],[2,1,3],[2,4,9]]
		objExp = self.constructor(dataExpected)

		assert toTest.isIdentical(objExp)
		assert toTest == ret

	def test_sortFeatures_comparator(self):
		""" Test sortFeatures() when we specify a comparator function """
		data = [[7,1,9],[1,2,3],[4,2,9]]
		toTest = self.constructor(data)

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
		objExp = self.constructor(dataExpected)

		assert toTest.isIdentical(objExp)
		assert toTest == ret



	#################
	# extractPoints() #
	#################

	def test_extractPoints_handmadeSingle(self):
		""" Test extractPoints() against handmade output when extracting one point """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		ext1 = toTest.extractPoints(0)
		exp1 = self.constructor([[1,2,3]])
		assert ext1.isIdentical(exp1)
		expEnd = self.constructor([[4,5,6],[7,8,9]])
		assert toTest.isIdentical(expEnd)

	def test_extractPoints_ListIntoPEmpty(self):
		""" Test extractPoints() by removing a list of all points """
		data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
		toTest = self.constructor(data)
		expRet = self.constructor(data)
		ret = toTest.extractPoints([0,1,2,3])

		assert ret.isIdentical(expRet)

		data = [[],[],[]]
		data = numpy.array(data).T
		exp = self.constructor(data)

		toTest.isIdentical(exp)


	def test_extractPoints_handmadeListSequence(self):
		""" Test extractPoints() against handmade output for several list extractions """
		data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
		toTest = self.constructor(data)
		ext1 = toTest.extractPoints(0)
		exp1 = self.constructor([[1,2,3]])
		assert ext1.isIdentical(exp1)
		ext2 = toTest.extractPoints([1,2])
		exp2 = self.constructor([[7,8,9],[10,11,12]])
		assert ext2.isIdentical(exp2)
		expEnd = self.constructor([[4,5,6]])
		assert toTest.isIdentical(expEnd)

	def test_extractPoints_handmadeListOrdering(self):
		""" Test extractPoints() against handmade output for out of order extraction """
		data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
		toTest = self.constructor(data)
		ext1 = toTest.extractPoints([3,4,1])
		exp1 = self.constructor([[10,11,12],[13,14,15],[4,5,6]])
		assert ext1.isIdentical(exp1)
		expEnd = self.constructor([[1,2,3], [7,8,9]])
		assert toTest.isIdentical(expEnd)


	def test_extractPoints_functionIntoPEmpty(self):
		""" Test extractPoints() by removing all points using a function """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		expRet = self.constructor(data)
		def allTrue(point):
			return True
		ret = toTest.extractPoints(allTrue)
		assert ret.isIdentical(expRet)

		data = [[],[],[]]
		data = numpy.array(data).T
		exp = self.constructor(data)

		toTest.isIdentical(exp)


	def test_extractPoints_handmadeFunction(self):
		""" Test extractPoints() against handmade output for function extraction """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		def oneOrFour(point):
			if 1 in point or 4 in point:
				return True
			return False
		ext = toTest.extractPoints(oneOrFour)
		exp = self.constructor([[1,2,3],[4,5,6]])
		assert ext.isIdentical(exp)
		expEnd = self.constructor([[7,8,9]])
		assert toTest.isIdentical(expEnd)

	def test_extractPoints_handmadeFuncionWithFeatureNames(self):
		""" Test extractPoints() against handmade output for function extraction with featureNames"""
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		def oneOrFour(point):
			if 1 in point or 4 in point:
				return True
			return False
		ext = toTest.extractPoints(oneOrFour)
		exp = self.constructor([[1,2,3],[4,5,6]],featureNames)
		assert ext.isIdentical(exp)
		expEnd = self.constructor([[7,8,9]],featureNames)
		assert toTest.isIdentical(expEnd)

	@raises(ArgumentException)
	def test_extractPoints_exceptionStartInvalid(self):
		""" Test extracPoints() for ArgumentException when start is not a valid point index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.extractPoints(start=-1,end=2)

	@raises(ArgumentException)
	def test_extractPoints_exceptionEndInvalid(self):
		""" Test extractPoints() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.extractPoints(start=1,end=5)

	@raises(ArgumentException)
	def test_extractPoints_exceptionInversion(self):
		""" Test extractPoints() for ArgumentException when start comes after end """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.extractPoints(start=2,end=0)

	def test_extractPoints_handmadeRange(self):
		""" Test extractPoints() against handmade output for range extraction """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		ret = toTest.extractPoints(start=1,end=2)
		
		expectedRet = self.constructor([[4,5,6],[7,8,9]])
		expectedTest = self.constructor([[1,2,3]])

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_extractPoints_rangeIntoPEmpty(self):
		""" Test extractPoints() removes all points using ranges """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		expRet = self.constructor(data,featureNames)
		ret = toTest.extractPoints(start=0,end=2)

		assert ret.isIdentical(expRet)

		data = [[],[],[]]
		data = numpy.array(data).T
		exp = self.constructor(data, featureNames)

		toTest.isIdentical(exp)


	def test_extractPoints_handmadeRangeWithFeatureNames(self):
		""" Test extractPoints() against handmade output for range extraction with featureNames """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		ret = toTest.extractPoints(start=1,end=2)
		
		expectedRet = self.constructor([[4,5,6],[7,8,9]],featureNames)
		expectedTest = self.constructor([[1,2,3]],featureNames)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_extractPoints_handmadeRangeRand_FM(self):
		""" Test extractPoints() against handmade output for randomized range extraction with featureNames """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		ret = toTest.extractPoints(start=0, end=2, number=2, randomize=True)
		
		expectedRet = self.constructor([[1,2,3],[4,5,6]],featureNames)
		expectedTest = self.constructor([[7,8,9]],featureNames)
		
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_extractPoints_handmadeRangeDefaults(self):
		""" Test extractPoints uses the correct defaults in the case of range based extraction """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		ret = toTest.extractPoints(end=1)
		
		expectedRet = self.constructor([[1,2,3],[4,5,6]],featureNames)
		expectedTest = self.constructor([[7,8,9]],featureNames)
		
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		toTest = self.constructor(data,featureNames)
		ret = toTest.extractPoints(start=1)

		expectedTest = self.constructor([[1,2,3]],featureNames)
		expectedRet = self.constructor([[4,5,6],[7,8,9]],featureNames)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)


	#TODO an extraction test where all data is removed
	#TODO extraction tests for all of the number and randomize combinations


	####################
	# extractFeatures() #
	####################

	def test_extractFeatures_handmadeSingle(self):
		""" Test extractFeatures() against handmade output when extracting one feature """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		ext1 = toTest.extractFeatures(0)
		exp1 = self.constructor([[1],[4],[7]])

		assert ext1.isIdentical(exp1)
		expEnd = self.constructor([[2,3],[5,6],[8,9]])
		assert toTest.isIdentical(expEnd)

	def test_extractFeatures_ListIntoFEmpty(self):
		""" Test extractFeatures() by removing a list of all features """
		data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
		toTest = self.constructor(data)
		expRet = self.constructor(data)
		ret = toTest.extractFeatures([0,1,2])

		assert ret.isIdentical(expRet)

		data = [[],[],[],[]]
		data = numpy.array(data)
		exp = self.constructor(data)

		toTest.isIdentical(exp)


	def test_extractFeatures_handmadeListSequence(self):
		""" Test extractFeatures() against handmade output for several extractions by list """
		data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
		toTest = self.constructor(data)
		ext1 = toTest.extractFeatures([0])
		exp1 = self.constructor([[1],[4],[7]])
		assert ext1.isIdentical(exp1)
		ext2 = toTest.extractFeatures([1,2])
		exp2 = self.constructor([[3,-1],[6,-2],[9,-3]])
		assert ext2.isIdentical(exp2)
		expEndData = [[2],[5],[8]]
		expEnd = self.constructor(expEndData)
		assert toTest.isIdentical(expEnd)

	def test_extractFeatures_handmadeListWithFeatureName(self):
		""" Test extractFeatures() against handmade output for list extraction when specifying featureNames """
		data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
		featureNames = ["one","two","three","neg"]
		toTest = self.constructor(data,featureNames)
		ext1 = toTest.extractFeatures(["one"])
		exp1 = self.constructor([[1],[4],[7]], ["one"])
		assert ext1.isIdentical(exp1)
		ext2 = toTest.extractFeatures(["three","neg"])
		exp2 = self.constructor([[3,-1],[6,-2],[9,-3]],["three","neg"])
		assert ext2.isIdentical(exp2)
		expEnd = self.constructor([[2],[5],[8]], ["two"])
		assert toTest.isIdentical(expEnd)


	def test_extractFeatures_functionIntoFEmpty(self):
		""" Test extractFeatures() by removing all featuress using a function """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		expRet = self.constructor(data)
		def allTrue(point):
			return True
		ret = toTest.extractFeatures(allTrue)
		assert ret.isIdentical(expRet)

		data = [[],[],[]]
		data = numpy.array(data)
		exp = self.constructor(data)

		toTest.isIdentical(exp)


	def test_extractFeatures_handmadeFunction(self):
		""" Test extractFeatures() against handmade output for function extraction """
		data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
		toTest = self.constructor(data)
		def absoluteOne(feature):
			if 1 in feature or -1 in feature:
				return True
			return False
		ext = toTest.extractFeatures(absoluteOne)
		exp = self.constructor([[1,-1],[4,-2],[7,-3]])
		assert ext.isIdentical(exp)
		expEnd = self.constructor([[2,3],[5,6],[8,9]])	
		assert toTest.isIdentical(expEnd)


	def test_extractFeatures_handmadeFunctionWithFeatureName(self):
		""" Test extractFeatures() against handmade output for function extraction with featureNames """
		data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
		featureNames = ["one","two","three","neg"]
		toTest = self.constructor(data,featureNames)
		def absoluteOne(feature):
			if 1 in feature or -1 in feature:
				return True
			return False

		ext = toTest.extractFeatures(absoluteOne)
		exp = self.constructor([[1,-1],[4,-2],[7,-3]], ['one','neg'])
		assert ext.isIdentical(exp)
		expEnd = self.constructor([[2,3],[5,6],[8,9]],["two","three"])	
		assert toTest.isIdentical(expEnd)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionStartInvalid(self):
		""" Test extractFeatures() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.extractFeatures(start=-1, end=2)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionStartInvalidFeatureName(self):
		""" Test extractFeatures() for ArgumentException when start is not a valid feature FeatureName """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.extractFeatures(start="wrong", end=2)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionEndInvalid(self):
		""" Test extractFeatures() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.extractFeatures(start=0, end=5)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionEndInvalidFeatureName(self):
		""" Test extractFeatures() for ArgumentException when start is not a valid featureName """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.extractFeatures(start="two", end="five")

	@raises(ArgumentException)
	def test_extractFeatures_exceptionInversion(self):
		""" Test extractFeatures() for ArgumentException when start comes after end """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.extractFeatures(start=2, end=0)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionInversionFeatureName(self):
		""" Test extractFeatures() for ArgumentException when start comes after end as FeatureNames"""
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.extractFeatures(start="two", end="one")


	def test_extractFeatures_rangeIntoFEmpty(self):
		""" Test extractFeatures() removes all Featuress using ranges """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		expRet = self.constructor(data,featureNames)
		ret = toTest.extractFeatures(start=0,end=2)

		assert ret.isIdentical(expRet)

		data = [[],[],[]]
		data = numpy.array(data)
		exp = self.constructor(data)

		toTest.isIdentical(exp)

	def test_extractFeatures_handmadeRange(self):
		""" Test extractFeatures() against handmade output for range extraction """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		ret = toTest.extractFeatures(start=1, end=2)
		
		expectedRet = self.constructor([[2,3],[5,6],[8,9]])
		expectedTest = self.constructor([[1],[4],[7]])

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_extractFeatures_handmadeWithFeatureNames(self):
		""" Test extractFeatures() against handmade output for range extraction with FeatureNames """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		ret = toTest.extractFeatures(start=1,end=2)
		
		expectedRet = self.constructor([[2,3],[5,6],[8,9]],["two","three"])
		expectedTest = self.constructor([[1],[4],[7]],["one"])

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)



	############
	# writeFile #
	############

	def test_writeFileCSV_handmade(self):
		""" Test writeFile() for csv extension with both data and featureNames """
		tmpFile = tempfile.NamedTemporaryFile(suffix=".csv")

		# instantiate object
		data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		toWrite = self.constructor(data, featureNames)

		# call writeFile
		toWrite.writeFile(tmpFile.name, format='csv', includeFeatureNames=True)

	#	opened = open(tmpFile.name,'r')
	#	print opened.read()
	#	for line in opened:
	#		print line

		# read it back into a different object, then test equality
		readObj = self.constructor(data=tmpFile.name)

		assert readObj.isIdentical(toWrite)
		assert toWrite.isIdentical(readObj)


	def test_writeFileMTX_handmade(self):
		""" Test writeFile() for mtx extension with both data and featureNames """
		tmpFile = tempfile.NamedTemporaryFile(suffix=".mtx")

		# instantiate object
		data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		toWrite = self.constructor(data, featureNames)

		# call writeFile
		toWrite.writeFile(tmpFile.name, format='mtx', includeFeatureNames=True)

		# read it back into a different object, then test equality
		readObj = self.constructor(data=tmpFile.name)

		assert readObj.isIdentical(toWrite)
		assert toWrite.isIdentical(readObj)


	#####################
	# referenceDataFrom #
	#####################

	@raises(ArgumentException)
	def test_referenceDataFrom_exceptionWrongType(self):
		""" Test referenceDataFrom() throws exception when other is not the same type """
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, featureNames)

		type1 = List(data1,featureNames)
		type2 = Matrix(data1,featureNames)

		# at least one of these two will be the wrong type
		orig.referenceDataFrom(type1)
		orig.referenceDataFrom(type2)


	def test_referenceDataFrom_sameReference(self):
		""" Test copyReference() successfully records the same reference """

		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, featureNames)

		data2 = [[-1,-2,-3,]]
		other = self.constructor(data2)

		ret = orig.referenceDataFrom(other)

		assert orig.data is other.data
		assert orig == ret

	#############
	# copy #
	#############

	def test_copy_withZeros(self):
		""" Test copy() produces an equal object and doesn't just copy the references """
		data1 = [[1,2,3,0],[1,0,3,0],[2,4,6,0],[0,0,0,0]]
		featureNames = ['one', 'two', 'three', 'four']
		orig = self.constructor(data1, featureNames)

		dup = orig.copy()

		assert orig.isIdentical(dup)
		assert dup.isIdentical(orig)

		assert orig.data is not dup.data


	def test_copy_Pempty(self):
		""" test copy() produces the correct outputs when given an point empty object """
		data = [[],[]]
		data = numpy.array(data).T

		orig = self.constructor(data)
		sparseObj = createData(retType="Sparse", data=data)
		listObj = createData(retType="List", data=data)
		matixObj = createData(retType="Matrix", data=data)

		copySparse = orig.copyAs(format='Sparse')
		assert copySparse.isIdentical(sparseObj)
		assert sparseObj.isIdentical(copySparse)

		copyList = orig.copyAs(format='List')
		assert copyList.isIdentical(listObj)
		assert listObj.isIdentical(copyList)

		copyMatrix = orig.copyAs(format='Matrix')
		assert copyMatrix.isIdentical(matixObj)
		assert matixObj.isIdentical(copyMatrix)

		pyList = orig.copyAs(format='python list')
		assert pyList == []

		numpyArray = orig.copyAs(format='numpy array')
		assert numpy.array_equal(numpyArray, data)

		numpyMatrix = orig.copyAs(format='numpy matrix')
		assert numpy.array_equal(numpyMatrix, numpy.matrix(data))
	

	def test_copy_Fempty(self):
		""" test copy() produces the correct outputs when given an feature empty object """
		data = [[],[]]
		data = numpy.array(data)

		orig = self.constructor(data)
		sparseObj = createData(retType="Sparse", data=data)
		listObj = createData(retType="List", data=data)
		matixObj = createData(retType="Matrix", data=data)

		copySparse = orig.copyAs(format='Sparse')
		assert copySparse.isIdentical(sparseObj)
		assert sparseObj.isIdentical(copySparse)

		copyList = orig.copyAs(format='List')
		assert copyList.isIdentical(listObj)
		assert listObj.isIdentical(copyList)

		copyMatrix = orig.copyAs(format='Matrix')
		assert copyMatrix.isIdentical(matixObj)
		assert matixObj.isIdentical(copyMatrix)

		pyList = orig.copyAs(format='python list')
		assert pyList == [[],[]]

		numpyArray = orig.copyAs(format='numpy array')
		assert numpy.array_equal(numpyArray, data)

		numpyMatrix = orig.copyAs(format='numpy matrix')
		assert numpy.array_equal(numpyMatrix, numpy.matrix(data))

	def test_copy_Trueempty(self):
		""" test copy() produces the correct outputs when given a point and feature empty object """
		data = numpy.empty(shape=(0,0))

		orig = self.constructor(data)
		sparseObj = createData(retType="Sparse", data=data)
		listObj = createData(retType="List", data=data)
		matixObj = createData(retType="Matrix", data=data)

		copySparse = orig.copyAs(format='Sparse')
		assert copySparse.isIdentical(sparseObj)
		assert sparseObj.isIdentical(copySparse)

		copyList = orig.copyAs(format='List')
		assert copyList.isIdentical(listObj)
		assert listObj.isIdentical(copyList)

		copyMatrix = orig.copyAs(format='Matrix')
		assert copyMatrix.isIdentical(matixObj)
		assert matixObj.isIdentical(copyMatrix)

		pyList = orig.copyAs(format='python list')
		assert pyList == []

		numpyArray = orig.copyAs(format='numpy array')
		assert numpy.array_equal(numpyArray, data)

		numpyMatrix = orig.copyAs(format='numpy matrix')
		assert numpy.array_equal(numpyMatrix, numpy.matrix(data))


	def test_copy_rightTypeTrueCopy(self):
		""" Test copy() will return all of the right type and do not show each other's modifications"""

		data = [[1,2,3],[1,0,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data, featureNames)
		sparseObj = createData(retType="Sparse", data=data, featureNames=featureNames)
		listObj = createData(retType="List", data=data, featureNames=featureNames)
		matixObj = createData(retType="Matrix", data=data, featureNames=featureNames)

		pointsShuffleIndices = [3,1,2,0]
		featuresshuffleIndices = [1,2,0]

		copySparse = orig.copyAs(format='Sparse')
		assert copySparse.isIdentical(sparseObj)
		assert sparseObj.isIdentical(copySparse)
		assert type(copySparse) == Sparse
		copySparse.setFeatureName('two', '2')
		assert 'two' in orig.featureNames
		copySparse.shufflePoints(pointsShuffleIndices)
		copySparse.shuffleFeatures(featuresshuffleIndices)
		assert orig[0,0] == 1 

		copyList = orig.copyAs(format='List')
		assert copyList.isIdentical(listObj)
		assert listObj.isIdentical(copyList)
		assert type(copyList) == List
		copyList.setFeatureName('two', '2')
		assert 'two' in orig.featureNames
		copyList.shufflePoints(pointsShuffleIndices)
		copyList.shuffleFeatures(featuresshuffleIndices)
		assert orig[0,0] == 1 

		copyMatrix = orig.copyAs(format='Matrix')
		assert copyMatrix.isIdentical(matixObj)
		assert matixObj.isIdentical(copyMatrix)
		assert type(copyMatrix) == Matrix
		copyMatrix.setFeatureName('two', '2')
		assert 'two' in orig.featureNames
		copyMatrix.shufflePoints(pointsShuffleIndices)
		copyMatrix.shuffleFeatures(featuresshuffleIndices)
		assert orig[0,0] == 1 

		pyList = orig.copyAs(format='python list')
		assert type(pyList) == list
		pyList[0][0] = 5
		assert orig[0,0] == 1 

		numpyArray = orig.copyAs(format='numpy array')
		assert type(numpyArray) == type(numpy.array([]))
		numpyArray[0,0] = 5
		assert orig[0,0] == 1 

		numpyMatrix = orig.copyAs(format='numpy matrix')
		assert type(numpyMatrix) == type(numpy.matrix([]))
		numpyMatrix[0,0] = 5
		assert orig[0,0] == 1 


	###################
	# copyPoints #
	###################

	@raises(ArgumentException)
	def test_copyPoints_exceptionNone(self):
		""" Test copyPoints() for exception when argument is None """

		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, featureNames)
		orig.copyPoints(None)

	@raises(ArgumentException)
	def test_copyPoints_exceptionNonIndex(self):
		""" Test copyPoints() for exception when a value in the input is not a valid index """
		
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, featureNames)
		orig.copyPoints([1,'yes'])


	def test_copyPoints_FEmpty(self):
		""" Test copyPoints() returns the correct data in a feature empty object """
		data = [[],[]]
		data = numpy.array(data)
		toTest = self.constructor(data)
		ret = toTest.copyPoints([0])

		data = [[]]
		data = numpy.array(data)
		exp = self.constructor(data)
		exp.isIdentical(ret)


	def test_copyPoints_handmadeContents(self):
		""" Test copyPoints() returns the correct data """
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, featureNames)
		expOrig = self.constructor(data1, featureNames)

		data2 = [[1,2,3],[2,4,6]]
		expRet = self.constructor(data2, featureNames)

		ret = orig.copyPoints([1,2])

		assert orig.isIdentical(expOrig)
		assert ret.isIdentical(expRet)

	@raises(ArgumentException)
	def test_copyPoints_exceptionStartInvalid(self):
		""" Test copyPoints() for ArgumentException when start is not a valid point index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.copyPoints(start=-1,end=2)

	@raises(ArgumentException)
	def test_copyPoints_exceptionEndInvalid(self):
		""" Test copyPoints() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.copyPoints(start=1,end=5)

	@raises(ArgumentException)
	def test_copyPoints_exceptionInversion(self):
		""" Test copyPoints() for ArgumentException when start comes after end """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.copyPoints(start=2,end=0)

	def test_copyPoints_handmadeRange(self):
		""" Test copyPoints() against handmade output for range copying """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		ret = toTest.copyPoints(start=1,end=2)
		
		expectedRet = self.constructor([[4,5,6],[7,8,9]])
		expectedTest = self.constructor(data)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_copyPoints_handmadeRangeWithFeatureNames(self):
		""" Test copyPoints() against handmade output for range copying with featureNames """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		ret = toTest.copyPoints(start=1,end=2)
		
		expectedRet = self.constructor([[4,5,6],[7,8,9]],featureNames)
		expectedTest = self.constructor(data,featureNames)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_copyPoints_handmadeRangeDefaults(self):
		""" Test copyPoints uses the correct defaults in the case of range based copying """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		ret = toTest.copyPoints(end=1)
		
		expectedRet = self.constructor([[1,2,3],[4,5,6]],featureNames)
		expectedTest = self.constructor(data,featureNames)
		
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		toTest = self.constructor(data,featureNames)
		ret = toTest.copyPoints(start=1)

		expectedTest = self.constructor(data,featureNames)
		expectedRet = self.constructor([[4,5,6],[7,8,9]],featureNames)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)




	#####################
	# copyFeatures #
	#####################

	@raises(ArgumentException)
	def test_copyFeatures_exceptionNone(self):
		""" Test copyFeatures() for exception when argument is None """

		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, featureNames)
		orig.copyFeatures(None)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionNonIndex(self):
		""" Test copyFeatures() for exception when a value in the input is not a valid index """
		
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, featureNames)
		orig.copyFeatures([1,'yes'])

	def test_copyFeatures_PEmpty(self):
		""" Test copyFeatures() returns the correct data in a point empty object """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)
		ret = toTest.copyFeatures([0])

		data = [[]]
		data = numpy.array(data).T
		exp = self.constructor(data)
		exp.isIdentical(ret)


	def test_copyFeatures_handmadeContents(self):
		""" Test copyFeatures() returns the correct data """

		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, featureNames)
		expOrig = self.constructor(data1, featureNames)

		data2 = [[1,2],[1,2],[2,4],[0,0]]

		expRet = self.constructor(data2, ['one','two'])

		ret = orig.copyFeatures([0,'two'])

		assert orig.isIdentical(expOrig)
		assert ret.isIdentical(expRet)


	####

	@raises(ArgumentException)
	def test_copyFeatures_exceptionStartInvalid(self):
		""" Test copyFeatures() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.copyFeatures(start=-1, end=2)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionStartInvalidFeatureName(self):
		""" Test copyFeatures() for ArgumentException when start is not a valid feature FeatureName """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.copyFeatures(start="wrong", end=2)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionEndInvalid(self):
		""" Test copyFeatures() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.copyFeatures(start=0, end=5)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionEndInvalidFeatureName(self):
		""" Test copyFeatures() for ArgumentException when start is not a valid featureName """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.copyFeatures(start="two", end="five")

	@raises(ArgumentException)
	def test_copyFeatures_exceptionInversion(self):
		""" Test copyFeatures() for ArgumentException when start comes after end """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.copyFeatures(start=2, end=0)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionInversionFeatureName(self):
		""" Test copyFeatures() for ArgumentException when start comes after end as FeatureNames"""
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		toTest.copyFeatures(start="two", end="one")

	def test_copyFeatures_handmadeRange(self):
		""" Test copyFeatures() against handmade output for range copying """
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		ret = toTest.copyFeatures(start=1, end=2)
		
		expectedRet = self.constructor([[2,3],[5,6],[8,9]])
		expectedTest = self.constructor(data)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_copyFeatures_handmadeWithFeatureNames(self):
		""" Test copyFeatures() against handmade output for range copying with FeatureNames """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)
		ret = toTest.copyFeatures(start=1,end=2)
		
		expectedRet = self.constructor([[2,3],[5,6],[8,9]],["two","three"])
		expectedTest = self.constructor(data, featureNames)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)


	##############
	# __getitem__#
	##############


	def test_getitem_simpleExampeWithZeroes(self):
		""" Test __getitem__ returns the correct output for a number of simple queries """
		featureNames = ["one","two","three","zero"]
		data = [[1,2,3,0],[4,5,0,0],[7,0,9,0],[0,0,0,0]]

		toTest = self.constructor(data, featureNames)

		assert toTest[0,0] == 1
		assert toTest[1,3] == 0
		assert toTest[2,2] == 9
		assert toTest[3,3] == 0

		assert toTest[1,'one'] == 4



	################
	# pointView #
	################


	def test_pointView_FEmpty(self):
		""" Test pointView() when accessing a feature empty object """
		data = [[],[]]
		data = numpy.array(data)
		toTest = self.constructor(data)

		v = toTest.pointView(0)

		assert len(v) == 0


	def test_pointView_isinstance(self):
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)

		pView = toTest.pointView(0)

		assert isinstance(pView, View)
		assert pView.name() is None
		assert pView.index() >= 0 and pView.index() < toTest.pointCount
		assert len(pView) == toTest.featureCount
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

	def test_featureView_FEmpty(self):
		""" Test featureView() when accessing a point empty object """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)

		v = toTest.featureView(0)

		assert len(v) == 0

	def test_featureView_isinstance(self):
		""" Test featureView() returns an instance of the View in dataHelpers """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data,featureNames)

		fView = toTest.featureView('one')

		assert isinstance(fView, View)
		assert fView.name() is not None
		assert fView.index() >= 0 and fView.index() < toTest.featureCount
		assert len(fView) == toTest.pointCount
		assert fView[0] == 1
		assert fView[1] == 4
		assert fView[2] == 7
		fView[0] = -1
		fView[1] = -4
		fView[2] = -7
		assert fView[0] == -1
		assert fView[1] == -4
		assert fView[2] == -7



	##############
	# pointCount #
	##############


	def test_pointCount_empty(self):
		""" test pointCount when given different kinds of emptiness """
		data = [[],[]]
		dataPEmpty = numpy.array(data).T
		dataFEmpty = numpy.array(data)

		objPEmpty = self.constructor(dataPEmpty)
		objFEmpty = self.constructor(dataFEmpty)

		assert objPEmpty.pointCount == 0
		assert objFEmpty.pointCount == 2


	def test_pointCount_vectorTest(self):
		""" Test pointCount when we only have row or column vectors of data """
		dataR = [[1,2,3]]
		dataC = [[1], [2], [3]]

		toTestR = self.constructor(dataR)
		toTestC = self.constructor(dataC)

		rPoints = toTestR.pointCount
		cPoints = toTestC.pointCount

		assert rPoints == 1
		assert cPoints == 3

	#################
	# featuresCount #
	#################


	def test_featureCount_empty(self):
		""" test featureCount when given different kinds of emptiness """
		data = [[],[]]
		dataPEmpty = numpy.array(data).T
		dataFEmpty = numpy.array(data)

		pEmpty = self.constructor(dataPEmpty)
		fEmpty = self.constructor(dataFEmpty)

		assert pEmpty.featureCount == 2
		assert fEmpty.featureCount == 0


	def test_featureCount_vectorTest(self):
		""" Test featureCount when we only have row or column vectors of data """
		dataR = [[1,2,3]]
		dataC = [[1], [2], [3]]

		toTestR = self.constructor(dataR)
		toTestC = self.constructor(dataC)

		rFeatures = toTestR.featureCount
		cFeatures = toTestC.featureCount

		assert rFeatures == 3
		assert cFeatures == 1


