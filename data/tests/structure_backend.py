
import tempfile
import numpy
from nose.tools import *

from copy import deepcopy

import UML
from UML import createData
from UML.data import List
from UML.data import Matrix
from UML.data import Sparse
from UML.data.dataHelpers import View
from UML.exceptions import ArgumentException

from UML.data.tests.baseObject import DataTestObject

class StructureBackend(DataTestObject):
	

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

	def test_init_allEqualWithNames(self):
		""" Test __init__() that every way to instantiate produces equal objects, with names """
		# instantiate from list of lists
		fromList = self.constructor(data=[[1,2,3]], pointNames=['1P'], featureNames=['one', 'two', 'three'])

		# instantiate from csv file
		tmpCSV = tempfile.NamedTemporaryFile(suffix=".csv")
		tmpCSV.write("#1P\n")
		tmpCSV.write("#one,two,three\n")
		tmpCSV.write("1,2,3\n")
		tmpCSV.flush()
		fromCSV = self.constructor(data=tmpCSV.name)

		# instantiate from mtx file
		tmpMTXArr = tempfile.NamedTemporaryFile(suffix=".mtx")
		tmpMTXArr.write("%%MatrixMarket matrix array integer general\n")
		tmpMTXArr.write("%#1P\n")
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
		tmpMTXCoo.write("%#1P\n")
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
		
		ret1 = dataObj1.transpose() # RET CHECK
		assert dataObj1.isIdentical(dataObjT)
		assert ret1 is None
		dataObj1.transpose()
		dataObjT.transpose()
		assert dataObj1.isIdentical(dataObj2)
		assert dataObj2.isIdentical(dataObjT)

	def test_transpose_handmadeWithZeros(self):
		""" Test transpose() function against handmade output """
		data = [[1,2,3],[4,5,6],[7,8,9],[0,0,0],[11,12,13]]
		dataTrans = [[1,4,7,0,11],[2,5,8,0,12],[3,6,9,0,13]]

		dataObj1 = self.constructor(deepcopy(data))
		dataObj2 = self.constructor(deepcopy(data))
		dataObjT = self.constructor(deepcopy(dataTrans))

#		print repr(dataObj1)
#		print repr(dataObj1)

		ret1 = dataObj1.transpose() # RET CHECK
#		import pdb
#		pdb.set_trace()
#		print dataObj1[0,4]

		assert dataObj1.isIdentical(dataObjT)
		assert ret1 is None
#		assert False
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
	def test_appendPoints_exceptionSamePointName(self):
		""" Test appendPoints() for ArgumentException when toAppend and self have a pointName in common """
		toTest1 = self.constructor([[1,2]], pointNames=["hello"])
		toTest2 = self.constructor([[1,2],[5,6]], pointNames=["hello","goodbye"])
		toTest2.appendPoints(toTest1)

	@raises(ArgumentException)
	def test_appendPoints_exceptionMismatchedFeatureNames(self):
		""" Test appendPoints() for ArgumentException when toAppend and self's feature names do not match"""
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=['one','two','three'])
		toAppend = self.constructor([[11, 12, 13,]], featureNames=["two", 'one', 'three'])
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
		names = ['1', '4', '7']
		dataExpected = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
		namesExp = ['1', '4', '7', '10']
		toTest = self.constructor(data, pointNames=names)
		toAppend = self.constructor([[10,11,12]], pointNames=['10'])
		expected = self.constructor(dataExpected, pointNames=namesExp)
		ret = toTest.appendPoints(toAppend) # RET CHECK
		assert toTest.isIdentical(expected)
		assert ret is None

	def test_appendPoints_handmadeSequence(self):
		""" Test appendPoints() against handmade output for a sequence of additions"""
		data = [[1,2,3],[4,5,6],[7,8,9]]
		names = ['1', '4', '7']
		toAppend1 = [[0.1,0.2,0.3]]
		n1 = ['d']
		toAppend2 = [[0.01,0.02,0.03],[0,0,0]]
		n2 = ['dd', '0']
		toAppend3 = [[10,11,12]]
		n3 = ['ten']

		dataExpected = [[1,2,3],[4,5,6],[7,8,9],[0.1,0.2,0.3],[0.01,0.02,0.03],[0,0,0],[10,11,12]]
		namesExp = ['1', '4', '7', 'd', 'dd', '0', 'ten']
		toTest = self.constructor(data, pointNames=names)
		toTest.appendPoints(self.constructor(toAppend1, pointNames=n1))
		toTest.appendPoints(self.constructor(toAppend2, pointNames=n2))
		toTest.appendPoints(self.constructor(toAppend3, pointNames=n3))

		expected = self.constructor(dataExpected, pointNames=namesExp)

		assert toTest.isIdentical(expected)
		


	####################
	# appendFeatures() #
	####################

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
		toTest1 = self.constructor([[1]], featureNames=["hello"])
		toTest2 = self.constructor([[1,2]], featureNames=["hello","goodbye"])
		toTest2.appendFeatures(toTest1)

	@raises(ArgumentException)
	def test_appendFeatures_exceptionMismatchedPointNames(self):
		""" Test appendFeatures() for ArgumentException when toAppend and self do not have equal pointNames """
		toTest1 = self.constructor([[2,1]], pointNames=["goodbye"])
		toTest2 = self.constructor([[1,2]], pointNames=["hello"])
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
		toTest = self.constructor(data, featureNames=featureNames)

		toAppend = self.constructor([[-1],[-2],[-3]], featureNames=['-1'])

		dataExpected = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
		featureNamesExpected = ['1','2','3','-1']
		expected = self.constructor(dataExpected, featureNames=featureNamesExpected)

		ret = toTest.appendFeatures(toAppend) # RET CHECK
		assert toTest.isIdentical(expected)
		assert ret is None

	def test_appendFeatures_handmadeSequence(self):
		""" Test appendFeatures() against handmade output for a sequence of additions"""
		data = [[1,2,3],[4,5,6],[7,8,9]]
		featureNames = ['1','2','3']
		toTest = self.constructor(data, featureNames=featureNames)

		toAppend1 = [[0.1],[0.2],[0.3]]
		lab1 = ['a']
		toAppend2 = [[0.01,0],[0.02,0],[0.03,0]]
		lab2 = ['A','0']
		toAppend3 = [[10],[11],[12]]
		lab3 = ['10']

		toTest.appendFeatures(self.constructor(toAppend1, featureNames=lab1))
		toTest.appendFeatures(self.constructor(toAppend2, featureNames=lab2))
		toTest.appendFeatures(self.constructor(toAppend3, featureNames=lab3))

		featureNamesExpected = ['1','2','3','a','A','0','10']
		dataExpected = [[1,2,3,0.1,0.01,0,10],[4,5,6,0.2,0.02,0,11],[7,8,9,0.3,0.03,0,12]]

		expected = self.constructor(dataExpected, featureNames=featureNamesExpected)
		assert toTest.isIdentical(expected)



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
		names = ['1', '7', '4']
		toTest = self.constructor(data, pointNames=names)

		ret = toTest.sortPoints(sortBy=1) # RET CHECK

		dataExpected = [[7,1,9],[1,2,3],[4,5,6]]
		namesExp = ['7', '1', '4']
		objExp = self.constructor(dataExpected, pointNames=namesExp)

		assert toTest.isIdentical(objExp)
		assert ret is None

	def test_sortPoints_scorer(self):
		""" Test sortPoints() when we specify a scoring function """
		data = [[1,2,3],[4,5,6],[7,1,9],[0,0,0]]
		toTest = self.constructor(data)

		def numOdds(point):
			assert isinstance(point, View)
			ret = 0
			for val in point:
				if val % 2 != 0:
					ret += 1
			return ret

		toTest.sortPoints(sortHelper=numOdds)

		dataExpected = [[0,0,0],[4,5,6],[1,2,3],[7,1,9]]
		objExp = self.constructor(dataExpected)

		assert toTest.isIdentical(objExp)
		
	def test_sortPoints_comparator(self):
		""" Test sortPoints() when we specify a comparator function """
		data = [[1,2,3],[4,5,6],[7,1,9],[0,0,0]]
		toTest = self.constructor(data)

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

		toTest.sortPoints(sortHelper=compOdds)

		dataExpected = [[0,0,0],[4,5,6],[1,2,3],[7,1,9]]
		objExp = self.constructor(dataExpected)

		assert toTest.isIdentical(objExp)

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
		toTest = self.constructor(data, featureNames=names)

		ret = toTest.sortFeatures(sortBy=1) # RET CHECK

		dataExpected = [[2,1,3],[1,7,9],[5,4,6]]
		namesExp = ["2", "1", "3"]
		objExp = self.constructor(dataExpected, featureNames=namesExp)

		assert toTest.isIdentical(objExp)
		assert ret is None

	def test_sortFeatures_scorer(self):
		""" Test sortFeatures() when we specify a scoring function """
		data = [[7,1,9,0],[1,2,3,0],[4,2,9,0]]
		names = ["2","1","3","0"]
		toTest = self.constructor(data, featureNames=names)

		def numOdds(feature):
			ret = 0
			for val in feature:
				if val % 2 != 0:
					ret += 1
			return ret

		toTest.sortFeatures(sortHelper=numOdds)

		dataExpected = [[0,1,7,9],[0,2,1,3],[0,2,4,9]]
		namesExp = ['0', '1', '2', '3']
		objExp = self.constructor(dataExpected, featureNames=namesExp)

		assert toTest.isIdentical(objExp)

	def test_sortFeatures_comparator(self):
		""" Test sortFeatures() when we specify a comparator function """
		data = [[7,1,9,0],[1,2,3,0],[4,2,9,0]]
		toTest = self.constructor(data)

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

		toTest.sortFeatures(sortHelper=compOdds)

		dataExpected = [[0,1,7,9],[0,2,1,3],[0,2,4,9]]
		objExp = self.constructor(dataExpected)

		assert toTest.isIdentical(objExp)


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
		names = ['1', '4', '7', '10']
		toTest = self.constructor(data, pointNames=names)
		ext1 = toTest.extractPoints('1')
		exp1 = self.constructor([[1,2,3]], pointNames=['1'])
		assert ext1.isIdentical(exp1)
		ext2 = toTest.extractPoints([1,2])
		exp2 = self.constructor([[7,8,9],[10,11,12]], pointNames=['7', '10'])
		assert ext2.isIdentical(exp2)
		expEnd = self.constructor([[4,5,6]], pointNames=['4'])
		assert toTest.isIdentical(expEnd)

	def test_extractPoints_handmadeListOrdering(self):
		""" Test extractPoints() against handmade output for out of order extraction """
		data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
		names = ['1', '4', '7', '10', '13']
		toTest = self.constructor(data, pointNames=names)
		ext1 = toTest.extractPoints([3,4,1])
		exp1 = self.constructor([[10,11,12],[13,14,15],[4,5,6]], pointNames=['10','13','4'])
		assert ext1.isIdentical(exp1)
		expEnd = self.constructor([[1,2,3], [7,8,9]], pointNames=['1','7'])
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
		toTest = self.constructor(data, featureNames=featureNames)
		def oneOrFour(point):
			if 1 in point or 4 in point:
				return True
			return False
		ext = toTest.extractPoints(oneOrFour)
		exp = self.constructor([[1,2,3],[4,5,6]], featureNames=featureNames)
		assert ext.isIdentical(exp)
		expEnd = self.constructor([[7,8,9]], featureNames=featureNames)
		assert toTest.isIdentical(expEnd)

	@raises(ArgumentException)
	def test_extractPoints_exceptionStartInvalid(self):
		""" Test extracPoints() for ArgumentException when start is not a valid point index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.extractPoints(start=-1,end=2)

	@raises(ArgumentException)
	def test_extractPoints_exceptionEndInvalid(self):
		""" Test extractPoints() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.extractPoints(start=1,end=5)

	@raises(ArgumentException)
	def test_extractPoints_exceptionInversion(self):
		""" Test extractPoints() for ArgumentException when start comes after end """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
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
		pointNames = ['1', '4', '7']
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		expRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints(start=0, end=2)

		assert ret.isIdentical(expRet)

		data = [[],[],[]]
		data = numpy.array(data).T
		exp = self.constructor(data, featureNames=featureNames)

		toTest.isIdentical(exp)


	def test_extractPoints_handmadeRangeWithFeatureNames(self):
		""" Test extractPoints() against handmade output for range extraction with featureNames """
		featureNames = ["one","two","three"]
		pointNames = ['1', '4', '7']
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints(start=1,end=2)
		
		expectedRet = self.constructor([[4,5,6],[7,8,9]], pointNames=['4','7'], featureNames=featureNames)
		expectedTest = self.constructor([[1,2,3]], pointNames=['1'], featureNames=featureNames)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_extractPoints_handmadeRangeRand_FM(self):
		""" Test extractPoints() for correct sizes when using randomized range extraction and featureNames """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		ret = toTest.extractPoints(start=0, end=2, number=2, randomize=True)
		
		assert ret.pointCount == 2
		assert toTest.pointCount == 1

	def test_extractPoints_handmadeRangeDefaults(self):
		""" Test extractPoints uses the correct defaults in the case of range based extraction """
		featureNames = ["one","two","three"]
		pointNames = ['1', '4', '7']
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints(end=1)
		
		expectedRet = self.constructor([[1,2,3],[4,5,6]], pointNames=['1', '4'], featureNames=featureNames)
		expectedTest = self.constructor([[7,8,9]], pointNames=['7'], featureNames=featureNames)
		
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints(start=1)

		expectedTest = self.constructor([[1,2,3]], pointNames=['1'], featureNames=featureNames)
		expectedRet = self.constructor([[4,5,6],[7,8,9]], pointNames=['4', '7'], featureNames=featureNames)

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

	def test_extractFeatures_ListIntoFEmptyOutOfOrder(self):
		""" Test extractFeatures() by removing a list of all features """
		data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
		toTest = self.constructor(data)
		expData = [[3,1,2], [6,4,5], [9,7,8], [12,10,11]]
		expRet = self.constructor(expData)
		ret = toTest.extractFeatures([2,0,1])

		assert ret.isIdentical(expRet)

		data = [[],[],[],[]]
		data = numpy.array(data)
		exp = self.constructor(data)

		toTest.isIdentical(exp)


	def test_extractFeatures_handmadeListSequence(self):
		""" Test extractFeatures() against handmade output for several extractions by list """
		pointNames = ['1', '4', '7']
		data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
		toTest = self.constructor(data, pointNames=pointNames)
		ext1 = toTest.extractFeatures([0])
		exp1 = self.constructor([[1],[4],[7]], pointNames=pointNames)
		assert ext1.isIdentical(exp1)
		ext2 = toTest.extractFeatures([2,1])
		exp2 = self.constructor([[-1,3],[-2,6],[-3,9]], pointNames=pointNames)
		assert ext2.isIdentical(exp2)
		expEndData = [[2],[5],[8]]
		expEnd = self.constructor(expEndData, pointNames=pointNames)
		assert toTest.isIdentical(expEnd)

	def test_extractFeatures_handmadeListWithFeatureName(self):
		""" Test extractFeatures() against handmade output for list extraction when specifying featureNames """
		data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
		featureNames = ["one","two","three","neg"]
		toTest = self.constructor(data, featureNames=featureNames)
		ext1 = toTest.extractFeatures(["one"])
		exp1 = self.constructor([[1],[4],[7]], featureNames=["one"])
		assert ext1.isIdentical(exp1)
		ext2 = toTest.extractFeatures(["three","neg"])
		exp2 = self.constructor([[3,-1],[6,-2],[9,-3]], featureNames=["three","neg"])
		assert ext2.isIdentical(exp2)
		expEnd = self.constructor([[2],[5],[8]], featureNames=["two"])
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
		pointNames = ['1', '4', '7']
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		def absoluteOne(feature):
			if 1 in feature or -1 in feature:
				return True
			return False

		ext = toTest.extractFeatures(absoluteOne)
		exp = self.constructor([[1,-1],[4,-2],[7,-3]], pointNames=pointNames, featureNames=['one','neg'])
		assert ext.isIdentical(exp)
		expEnd = self.constructor([[2,3],[5,6],[8,9]], pointNames=pointNames, featureNames=["two","three"])	
		assert toTest.isIdentical(expEnd)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionStartInvalid(self):
		""" Test extractFeatures() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.extractFeatures(start=-1, end=2)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionStartInvalidFeatureName(self):
		""" Test extractFeatures() for ArgumentException when start is not a valid feature FeatureName """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.extractFeatures(start="wrong", end=2)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionEndInvalid(self):
		""" Test extractFeatures() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.extractFeatures(start=0, end=5)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionEndInvalidFeatureName(self):
		""" Test extractFeatures() for ArgumentException when start is not a valid featureName """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.extractFeatures(start="two", end="five")

	@raises(ArgumentException)
	def test_extractFeatures_exceptionInversion(self):
		""" Test extractFeatures() for ArgumentException when start comes after end """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.extractFeatures(start=2, end=0)

	@raises(ArgumentException)
	def test_extractFeatures_exceptionInversionFeatureName(self):
		""" Test extractFeatures() for ArgumentException when start comes after end as FeatureNames"""
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.extractFeatures(start="two", end="one")


	def test_extractFeatures_rangeIntoFEmpty(self):
		""" Test extractFeatures() removes all Featuress using ranges """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		expRet = self.constructor(data, featureNames=featureNames)
		ret = toTest.extractFeatures(start=0, end=2)

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
		pointNames = ['1', '4', '7']
		data = [[1,2,3],[4,5,6],[7,8,9]]

		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures(start=1,end=2)
		
		expectedRet = self.constructor([[2,3],[5,6],[8,9]], pointNames=pointNames, featureNames=["two","three"])
		expectedTest = self.constructor([[1],[4],[7]], pointNames=pointNames, featureNames=["one"])

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)


	#####################
	# referenceDataFrom #
	#####################

	@raises(ArgumentException)
	def test_referenceDataFrom_exceptionWrongType(self):
		""" Test referenceDataFrom() throws exception when other is not the same type """
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pNames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

		type1 = List(data1, pointNames=pNames, featureNames=featureNames)
		type2 = Matrix(data1, pointNames=pNames, featureNames=featureNames)

		# at least one of these two will be the wrong type
		orig.referenceDataFrom(type1)
		orig.referenceDataFrom(type2)


	def test_referenceDataFrom_sameReference(self):
		""" Test copyReference() successfully records the same reference """

		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pNames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

		data2 = [[-1,-2,-3,]]
		featureNames = ['1', '2', '3']
		pNames = ['-1']
		other = self.constructor(data2, pointNames=pNames, featureNames=featureNames)

		ret = orig.referenceDataFrom(other) # RET CHECK

		assert orig.data is other.data
		assert '-1' in orig.pointNames.keys()
		assert '1' in orig.featureNames.keys()
		assert ret is None


	#############
	# copyAs #
	#############

	def test_copy_withZeros(self):
		""" Test copyAs() produces an equal object and doesn't just copy the references """
		data1 = [[1,2,3,0],[1,0,3,0],[2,4,6,0],[0,0,0,0]]
		featureNames = ['one', 'two', 'three', 'four']
		pointNames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pointNames, featureNames=featureNames)

		dup1 = orig.copy()
		dup2 = orig.copyAs(orig.getTypeString())

		assert orig.isIdentical(dup1)
		assert dup1.isIdentical(orig)

		assert orig.data is not dup1.data

		assert orig.isIdentical(dup2)
		assert dup2.isIdentical(orig)

		assert orig.data is not dup2.data


	def test_copy_Pempty(self):
		""" test copyAs() produces the correct outputs when given an point empty object """
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
		""" test copyAs() produces the correct outputs when given an feature empty object """
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
		""" test copyAs() produces the correct outputs when given a point and feature empty object """
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
		""" Test copyAs() will return all of the right type and do not show each other's modifications"""

		data = [[1,2,3],[1,0,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pointNames = ['1', 'one', '2', '0']
		orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		sparseObj = createData(retType="Sparse", data=data, pointNames=pointNames, featureNames=featureNames)
		listObj = createData(retType="List", data=data, pointNames=pointNames, featureNames=featureNames)
		matixObj = createData(retType="Matrix", data=data, pointNames=pointNames, featureNames=featureNames)

		pointsShuffleIndices = [3,1,2,0]
		featuresshuffleIndices = [1,2,0]

		copySparse = orig.copyAs(format='Sparse')
		assert copySparse.isIdentical(sparseObj)
		assert sparseObj.isIdentical(copySparse)
		assert type(copySparse) == Sparse
		copySparse.setFeatureName('two', '2')
		copySparse.setPointName('one', 'WHAT')
		assert 'two' in orig.featureNames
		assert 'one' in orig.pointNames
		copySparse.shufflePoints(pointsShuffleIndices)
		copySparse.shuffleFeatures(featuresshuffleIndices)
		assert orig[0,0] == 1 

		copyList = orig.copyAs(format='List')
		assert copyList.isIdentical(listObj)
		assert listObj.isIdentical(copyList)
		assert type(copyList) == List
		copyList.setFeatureName('two', '2')
		copyList.setPointName('one', 'WHAT')
		assert 'two' in orig.featureNames
		assert 'one' in orig.pointNames
		copyList.shufflePoints(pointsShuffleIndices)
		copyList.shuffleFeatures(featuresshuffleIndices)
		assert orig[0,0] == 1 

		copyMatrix = orig.copyAs(format='Matrix')
		assert copyMatrix.isIdentical(matixObj)
		assert matixObj.isIdentical(copyMatrix)
		assert type(copyMatrix) == Matrix
		copyMatrix.setFeatureName('two', '2')
		copyMatrix.setPointName('one', 'WHAT')
		assert 'two' in orig.featureNames
		assert 'one' in orig.pointNames
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
		orig = self.constructor(data1, featureNames=featureNames)
		orig.copyPoints(None)

	@raises(ArgumentException)
	def test_copyPoints_exceptionNonIndex(self):
		""" Test copyPoints() for exception when a value in the input is not a valid index """
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		pnames = ['1', 'one', '2', '0']
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, pointNames=pnames, featureNames=featureNames)
		orig.copyPoints([1,'yes'])


	def test_copyPoints_FEmpty(self):
		""" Test copyPoints() returns the correct data in a feature empty object """
		data = [[],[]]
		pnames = ['1', 'one']
		data = numpy.array(data)
		toTest = self.constructor(data, pointNames=pnames)
		ret = toTest.copyPoints([0])

		data = [[]]
		data = numpy.array(data)
		exp = self.constructor(data, pointNames=['0'])
		exp.isIdentical(ret)


	def test_copyPoints_handmadeContents(self):
		""" Test copyPoints() returns the correct data """
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pnames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pnames, featureNames=featureNames)
		expOrig = self.constructor(data1, pointNames=pnames, featureNames=featureNames)

		data2 = [[1,2,3],[2,4,6]]
		expRet = self.constructor(data2, pointNames=['one', '2'], featureNames=featureNames)

		ret = orig.copyPoints([1,2])

		assert orig.isIdentical(expOrig)
		assert ret.isIdentical(expRet)

	def test_copyPoints_handmadeListOrdering(self):
		""" Test copyPoints() against handmade output for out of order indices """
		data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
		names = ['1', '4', '7', '10', '13']
		toTest = self.constructor(data, pointNames=names)
		cop1 = toTest.copyPoints([3,4,1])
		exp1 = self.constructor([[10,11,12],[13,14,15],[4,5,6]], pointNames=['10','13','4'])
		assert cop1.isIdentical(exp1)


	@raises(ArgumentException)
	def test_copyPoints_exceptionStartInvalid(self):
		""" Test copyPoints() for ArgumentException when start is not a valid point index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.copyPoints(start=-1,end=2)

	@raises(ArgumentException)
	def test_copyPoints_exceptionEndInvalid(self):
		""" Test copyPoints() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.copyPoints(start=1,end=5)

	@raises(ArgumentException)
	def test_copyPoints_exceptionInversion(self):
		""" Test copyPoints() for ArgumentException when start comes after end """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
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
		pnames = ['1', '4', '7']
		toTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)
		ret = toTest.copyPoints(start=1,end=2)
		
		expectedRet = self.constructor([[4,5,6],[7,8,9]], pointNames=['4','7'], featureNames=featureNames)
		expectedTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_copyPoints_handmadeRangeDefaults(self):
		""" Test copyPoints uses the correct defaults in the case of range based copying """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		ret = toTest.copyPoints(end=1)
		
		expectedRet = self.constructor([[1,2,3],[4,5,6]], featureNames=featureNames)
		expectedTest = self.constructor(data, featureNames=featureNames)
		
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		toTest = self.constructor(data, featureNames=featureNames)
		ret = toTest.copyPoints(start=1)

		expectedTest = self.constructor(data, featureNames=featureNames)
		expectedRet = self.constructor([[4,5,6],[7,8,9]], featureNames=featureNames)

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
		orig = self.constructor(data1, featureNames=featureNames)
		orig.copyFeatures(None)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionNonIndex(self):
		""" Test copyFeatures() for exception when a value in the input is not a valid index """
		
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		orig = self.constructor(data1, featureNames=featureNames)
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
		pnames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pnames, featureNames=featureNames)
		expOrig = self.constructor(data1, pointNames=pnames, featureNames=featureNames)

		data2 = [[1,2],[1,2],[2,4],[0,0]]

		expRet = self.constructor(data2, pointNames=pnames, featureNames=['one','two'])

		ret = orig.copyFeatures([0,'two'])

		assert orig.isIdentical(expOrig)
		assert ret.isIdentical(expRet)

	def test_copyFeatures_handmadeListOrdering(self):
		""" Test copyFeatures() against handmade output for out of order indices """
		data = [[1,2,3,33],[4,5,6,66],[7,8,9,99],[10,11,12,122]]
		names = ['1', '2', '3', 'dubs']
		toTest = self.constructor(data, featureNames=names)
		cop1 = toTest.copyFeatures([2,3,1])
		exp1 = self.constructor([[3, 33, 2],[6, 66, 5],[9, 99, 8],[12, 122, 11]], featureNames=['3','dubs','2'])
		assert cop1.isIdentical(exp1)


	@raises(ArgumentException)
	def test_copyFeatures_exceptionStartInvalid(self):
		""" Test copyFeatures() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.copyFeatures(start=-1, end=2)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionStartInvalidFeatureName(self):
		""" Test copyFeatures() for ArgumentException when start is not a valid feature FeatureName """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.copyFeatures(start="wrong", end=2)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionEndInvalid(self):
		""" Test copyFeatures() for ArgumentException when start is not a valid feature index """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.copyFeatures(start=0, end=5)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionEndInvalidFeatureName(self):
		""" Test copyFeatures() for ArgumentException when start is not a valid featureName """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.copyFeatures(start="two", end="five")

	@raises(ArgumentException)
	def test_copyFeatures_exceptionInversion(self):
		""" Test copyFeatures() for ArgumentException when start comes after end """
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
		toTest.copyFeatures(start=2, end=0)

	@raises(ArgumentException)
	def test_copyFeatures_exceptionInversionFeatureName(self):
		""" Test copyFeatures() for ArgumentException when start comes after end as FeatureNames"""
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, featureNames=featureNames)
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
		pnames = ['1', '4', '7']
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)
		ret = toTest.copyFeatures(start=1,end=2)
		
		expectedRet = self.constructor([[2,3],[5,6],[8,9]], pointNames=pnames, featureNames=["two","three"])
		expectedTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)



