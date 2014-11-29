
import math
import tempfile
import numpy
from nose.tools import *

from copy import deepcopy

import UML
from UML.data.dataHelpers import View
from UML.data.tests.baseObject import DataTestObject
from UML.data.dataHelpers import formatIfNeeded
from UML.exceptions import ArgumentException


class QueryBackend(DataTestObject):
	
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




	#################
	# isIdentical() #
	#################

	def test_isIdentical_False(self):
		""" Test isIdentical() against some non-equal input """
		toTest = self.constructor([[4,5]])
		assert not toTest.isIdentical(self.constructor([[1,1],[2,2]]))
		assert not toTest.isIdentical(self.constructor([[1,2,3]]))
		assert not toTest.isIdentical(self.constructor([[1,2]]))

	def test_isIdentical_FalseBozoTypes(self):
		""" Test isIdentical() against some non-equal input of crazy types """
		toTest = self.constructor([[4,5]])
		assert not toTest.isIdentical(numpy.matrix([[1,1],[2,2]]))
		assert not toTest.isIdentical('self.constructor([[1,2,3]])')
		assert not toTest.isIdentical(toTest.isIdentical)

	def test_isIdentical_True(self):
		""" Test isIdentical() against some actually equal input """
		toTest1 = self.constructor([[4,5]])
		toTest2 = self.constructor(deepcopy([[4,5]]))
		assert toTest1.isIdentical(toTest2)
		assert toTest2.isIdentical(toTest1)


	############
	# writeFile #
	############

	def test_writeFile_CSVhandmade(self):
		""" Test writeFile() for csv extension with both data and featureNames """
		tmpFile = tempfile.NamedTemporaryFile(suffix=".csv")

		# instantiate object
		data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		pointNames = ['1', 'one', '2', '0']
		featureNames = ['one', 'two', 'three']
		toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

		# call writeFile
		toWrite.writeFile(tmpFile.name, format='csv', includeNames=True)

		# read it back into a different object, then test equality
		readObj = self.constructor(data=tmpFile.name)

		assert readObj.isIdentical(toWrite)
		assert toWrite.isIdentical(readObj)

	def test_writeFile_CSVauto(self):
		""" Test writeFile() will (if needed) autoconvert to Matrix to use its CSV output """
		tmpFile = tempfile.NamedTemporaryFile(suffix=".csv")

		# instantiate object
		data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		pointNames = ['1', 'one', '2', '0']
		featureNames = ['one', 'two', 'three']
		toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

		# cripple all but cannonical implementation
		if self.retType != 'Matrix':
			toWrite._writeFile_implementation = None

		# call writeFile
		toWrite.writeFile(tmpFile.name, format='csv', includeNames=True)

		# read it back into a different object, then test equality
		readObj = self.constructor(data=tmpFile.name)

		assert readObj.isIdentical(toWrite)
		assert toWrite.isIdentical(readObj)


	def test_writeFile_MTXhandmade(self):
		""" Test writeFile() for mtx extension with both data and featureNames """
		tmpFile = tempfile.NamedTemporaryFile(suffix=".mtx")

		# instantiate object
		data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pointNames = ['1', 'one', '2', '0']
		toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

		# call writeFile
		toWrite.writeFile(tmpFile.name, format='mtx', includeNames=True)

		# read it back into a different object, then test equality
		readObj = self.constructor(data=tmpFile.name)

		assert readObj.isIdentical(toWrite)
		assert toWrite.isIdentical(readObj)

	def test_writeFile_MTXauto(self):
		""" Test writeFile() will (if needed) autoconvert to Matrix to use its MTX output """
		tmpFile = tempfile.NamedTemporaryFile(suffix=".mtx")

		# instantiate object
		data = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pointNames = ['1', 'one', '2', '0']
		toWrite = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

		# cripple all but cannonical implementation
		if self.retType != 'Sparse':
			toWrite._writeFile_implementation = None

		# call writeFile
		toWrite.writeFile(tmpFile.name, format='mtx', includeNames=True)

		# read it back into a different object, then test equality
		readObj = self.constructor(data=tmpFile.name)

		assert readObj.isIdentical(toWrite)
		assert toWrite.isIdentical(readObj)

	##############
	# __getitem__#
	##############


	def test_getitem_simpleExampeWithZeroes(self):
		""" Test __getitem__ returns the correct output for a number of simple queries """
		featureNames = ["one","two","three","zero"]
		pnames = ['1', '4', '7', '0']
		data = [[1,2,3,0],[4,5,0,0],[7,0,9,0],[0,0,0,0]]

		toTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)

		assert toTest[0,0] == 1
		assert toTest[1,3] == 0
		assert toTest['7',2] == 9
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
		pointNames = ['1', '4', '7']
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

		pView = toTest.pointView(0)

		assert isinstance(pView, View)
		assert pView.name() == '1'
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
		pointNames = ['1', '4', '7']
		featureNames = ["one","two","three"]
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

		fView = toTest.featureView('one')

		assert isinstance(fView, View)
		assert fView.name() == 'one'
		assert fView.index() >= 0 and fView.index() < toTest.featureCount
		assert len(fView) == toTest.pointCount
		assert fView[0] == 1
		assert fView['4'] == 4
		assert fView['7'] == 7
		fView[0] = -1
		fView['4'] = -4
		fView[2] = -7
		assert fView['1'] == -1
		assert fView['4'] == -4
		assert fView[2] == -7


	################
	# containsZero #
	################


	def test_containsZero_simple(self):
		""" Test containsZero works as expected on simple numerical data """
		dataAll = [[0,0,0], [0,0,0], [0,0,0]]
		dataSome = [[1,1,1], [0.0,1,-4], [2,2,2]]
		dataNone = [[1.1,2,12], [-2,-3,-4], [.0001, .000003, .00000004]]

		dAll = self.constructor(dataAll)
		dSome = self.constructor(dataSome)
		dNone = self.constructor(dataNone)

		assert dAll.containsZero() == True
		assert dSome.containsZero() == True
		assert dNone.containsZero() == False

	##########
	# __eq__ #
	##########

	def test_eq__exactlyisIdentical(self):
		""" Test that __eq__ relies on isIdentical """

		class FlagWrap(object):
			flag = False

		flag1 = FlagWrap()
		flag2 = FlagWrap()
		def fake(other):
			if flag1.flag:
				flag2.flag = True
			flag1.flag = True
			return True

		toTest1 = self.constructor([[4,5]])
		toTest2 = self.constructor(deepcopy([[4,5]]))

		toTest1.isIdentical = fake
		toTest2.isIdentical = fake

		assert toTest1 == toTest2
		assert toTest2 == toTest1
		assert flag1.flag
		assert flag2.flag

	##########
	# __ne__ #
	##########

	def test_ne__exactly__eq__(self):
		""" Test that __ne__ relies on __eq__ """

		class FlagWrap(object):
			flag = False

		flag1 = FlagWrap()
		flag2 = FlagWrap()
		def fake(other):
			if flag1.flag:
				flag2.flag = True
			flag1.flag = True
			return False

		toTest1 = self.constructor([[4,5]])
		toTest2 = self.constructor(deepcopy([[4,5]]))

		toTest1.__eq__ = fake
		toTest2.__eq__ = fake

		assert toTest1 != toTest2
		assert toTest2 != toTest1
		assert flag1.flag
		assert flag2.flag


	############
	# toString #
	############

	def test_toString_dataLocation(self):
		""" test toString under default parameters """
		
		for pNum in [3,9]:
			for fNum in [2,5,8]:
				randGen = UML.createRandomData("List", pNum, fNum, 0, numericType='int')
				raw = randGen.data

				fnames = ['fn0', 'fn1', 'fn2', 'fn3', 'fn4', 'fn5', 'fn6', 'fn7', 'fn8']
				pnames = ['pn0', 'pn1', 'pn2', 'pn3', 'pn4', 'pn5', 'pn6', 'pn7', 'pn8']
				data = UML.createData(self.retType, raw, pointNames=pnames[:pNum], featureNames=fnames[:fNum])
				
				for mw in [40, 60, 80,]:
					for mh in [5, 7, 10]:
						ret = data.toString(includeNames=False, maxWidth=mw, maxHeight=mh)
						checkToStringRet(ret, data)


	##################### #######################
	# pointSimilarities # # featureSimilarities #
	##################### #######################

	@raises(ArgumentException)
	def test_pointSimilaritesInvalidParamType(self):
		""" Test pointSimilarities raise exception for unexpected param type """
		self.backend_InvalidParamType(True)

	@raises(ArgumentException)
	def test_featureSimilaritesInvalidParamType(self):
		""" Test featureSimilarities raise exception for unexpected param type """
		self.backend_InvalidParamType(False)

	def backend_InvalidParamType(self, axis):
		data = [[1,2],[3,4]]
		obj = self.constructor(data)

		if axis:
			obj.pointSimilarities({"hello":5})
		else:
			obj.featureSimilarities({"hello":5})

	@raises(ArgumentException)
	def test_pointSimilaritesUnexpectedString(self):
		""" Test pointSimilarities raise exception for unexpected string value """
		self.backend_UnexpectedString(True)

	@raises(ArgumentException)
	def test_featureSimilaritesUnexpectedString(self):
		""" Test featureSimilarities raise exception for unexpected string value """
		self.backend_UnexpectedString(False)

	def backend_UnexpectedString(self, axis):
		data = [[1,2],[3,4]]
		obj = self.constructor(data)

		if axis:
			obj.pointSimilarities("foo")
		else:
			obj.featureSimilarities("foo")


	# test results covariance
	def test_pointSimilaritesCovarianceResult(self):
		""" Test pointSimilarities returns correct covariance results """
		self.backend_CovarianceResult(True)

	def test_featureSimilaritesCovarianceResult(self):
		""" Test featureSimilarities returns correct covariance results """
		self.backend_CovarianceResult(False)

	def backend_CovarianceResult(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointSimilarities("covariance")
		else:
			orig.transpose()
			ret = orig.featureSimilarities("covariance")
			ret.transpose()
			orig.transpose()

		# hand computed results
		expRow0 = [2,      (4./3),  (2./3) ]
		expRow1 = [(4./3), (15./9), (-2./9)]
		expRow2 = [(2./3), (-2./9), (8./9) ]
		expData = [expRow0, expRow1, expRow2]
		expObj = self.constructor(expData)

		assert expObj == ret
		assert sameAsOrig == orig

	# test results correlation
	def test_pointSimilaritesCorrelationResult(self):
		""" Test pointSimilarities returns correct correlation results """
		self.backend_CorrelationResult(True)

	def test_featureSimilaritesCorrelationResult(self):
		""" Test featureSimilarities returns correct correlation results """
		self.backend_CorrelationResult(False)

	def backend_CorrelationResult(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointSimilarities("correlation")
		else:
			orig.transpose()
			ret = orig.featureSimilarities("correlation")
			ret.transpose()
			orig.transpose()


		tempRow0 = [0, 0, 0]
		tempRow1 = [(-2./3 * math.sqrt(14./9)), (1./3 * math.sqrt(14./9)), (1./3 * math.sqrt(14./9))]
		tempRow2 = [(2./3 * math.sqrt(8./9)), (-1./3 * math.sqrt(8./9)), (-1./3 * math.sqrt(8./9))]
		tempData = [tempRow0, tempRow1, tempRow2]
		tempObj = self.constructor(tempData)

		expObj = tempObj.featureSimilarities("covariance")

		assert expObj == ret
		assert sameAsOrig == orig

	# test results dot product
	def test_pointSimilaritesDotProductResult(self):
		""" Test pointSimilarities returns correct dot product results """
		self.backend_DotProductResult(True)

	def test_featureSimilaritesDotProductResult(self):
		""" Test featureSimilarities returns correct dot product results """
		self.backend_DotProductResult(False)

	def backend_DotProductResult(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = obj.pointSimilarities("dotproduct")
		else:
			orig.transpose()
			ret = obj.featureSimilarities("dotproduct")
			ret.transpose()
			orig.transpose()

		expData = [[3, 2, 1], [2, 2, 0], [1, 0, 1]]
		expObj = self.constructor(expData)

		assert expObj == ret
		assert sameAsOrig == orig

	# test input function validation
	@raises(ArgumentException)
	def test_pointSimilaritesFuncValidation(self):
		""" Test pointSimilarities raises exception for invalid funcitions """
		self.backend_FuncValidation(True)

	@raises(ArgumentException)
	def test_featureSimilaritesFuncValidation(self):
		""" Test featureSimilarities raises exception for invalid funcitions """
		self.backend_FuncValidation(False)

	def backend_FuncValidation(self, axis):
		data = [[1,2],[3,4]]
		obj = self.constructor(data)

		def singleArg(one):
			return one

		if axis:
			obj.pointSimilarities(singleArg)
		else:
			obj.featureSimilarities(singleArg)

	# test results passed function
	def test_pointSimilariteGivenFuncResults(self):
		""" Test pointSimilarities returns correct results for given function """
		self.backend_GivenFuncResults(True)

	def test_featureSimilaritesGivenFuncResults(self):
		""" Test featureSimilarities returns correct results for given function """
		self.backend_GivenFuncResults(False)

	def backend_GivenFuncResults(self, axis):
		assert False
		data = [[1,2],[3,4]]
		obj = self.constructor(data)

		def euclideanDistance(left, right):
			assert False

		if axis:
			obj.pointSimilarities(euclideanDistance)
		else:
			obj.featureSimilarities(euclideanDistance)

	##################### #######################
	# pointStatistics # # featureStatistics #
	##################### #######################



###########
# Helpers #
###########

def checkToStringRet(ret, data):
	cHold = '...'
	rHold = '|'
	sigDigits = 3
	rows = ret.split('\n')
	rows = rows[:(len(rows)-1)]

	negRow = False
	assert len(rows) <= data.pointCount
	for r in range(len(rows)):
		row = rows[r]
		spaceSplit = row.split(' ')
		vals = []
		for possible in spaceSplit:
			if possible != '':
				vals.append(possible)
		if vals[0] == rHold:
			negRow = True
			continue

		rDataIndex = r
		if negRow:
			rDataIndex = -(len(rows) - r)

		negCol = False
		assert len(vals) <= data.featureCount
		for c in range(len(vals)):
			if vals[c] == cHold:
				negCol = True
				continue

			cDataIndex = c
			if negCol:
				cDataIndex = -(len(vals) - c)
			
			wanted = data[rDataIndex, cDataIndex]
			wantedS = formatIfNeeded(wanted, sigDigits)
			have = vals[c]
			assert wantedS == have
