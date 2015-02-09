
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
		assert toTest[3,'zero'] == 0

		assert toTest[1,'one'] == 4

	@raises(ArgumentException)
	def test_getitem_nonIntConvertableFloatSingleKey(self):
		data = [[0,1,2,3]]
		toTest = self.constructor(data)

		assert toTest[0.1] == 0

	@raises(ArgumentException)
	def test_getitem_nonIntConvertableFloatTupleKey(self):
		data = [[0,1],[2,3]]
		toTest = self.constructor(data)

		assert toTest[0,1.1] == 1

	def test_getitem_floatKeys(self):
		""" Test __getitem__ correctly interprets float valued keys """
		featureNames = ["one","two","three","zero"]
		pnames = ['1', '4', '7', '0']
		data = [[1,2,3,0],[4,5,0,0],[7,0,9,0],[0,0,0,0]]

		toTest = self.constructor(data, pointNames=pnames, featureNames=featureNames)

		assert toTest[0.0,0] == 1
		assert toTest[1.0,3.0] == 0

		data = [[0,1,2,3]]
		toTest = self.constructor(data)

		assert toTest[0.0] == 0
		assert toTest[1.0] == 1


	def test_getitem_SinglePoint(self):
		""" Test __getitem__ has vector style access for one point object """
		pnames = ['single']
		fnames = ['a', 'b', 'c', 'd', 'e']
		data = [[0,1,2,3,10]]
		toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

		assert toTest[0] == 0
		assert toTest['a'] == 0
		assert toTest[1] == 1
		assert toTest['b'] == 1
		assert toTest[2] == 2
		assert toTest['c'] == 2
		assert toTest[3] == 3
		assert toTest['d'] == 3
		assert toTest[4] == 10
		assert toTest['e'] == 10

	def test_getitem_SingleFeature(self):
		""" Test __getitem__ has vector style access for one feature object """
		fnames = ['single']
		pnames = ['a', 'b', 'c', 'd', 'e']
		data = [[0], [1], [2], [3], [10]]
		toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

		assert toTest[0] == 0
		assert toTest['a'] == 0
		assert toTest[1] == 1
		assert toTest['b'] == 1
		assert toTest[2] == 2
		assert toTest['c'] == 2
		assert toTest[3] == 3
		assert toTest['d'] == 3
		assert toTest[4] == 10
		assert toTest['e'] == 10


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
		self.backend_Sim_InvalidParamType(True)

	@raises(ArgumentException)
	def test_featureSimilaritesInvalidParamType(self):
		""" Test featureSimilarities raise exception for unexpected param type """
		self.backend_Sim_InvalidParamType(False)

	def backend_Sim_InvalidParamType(self, axis):
		data = [[1,2],[3,4]]
		obj = self.constructor(data)

		if axis:
			obj.pointSimilarities({"hello":5})
		else:
			obj.featureSimilarities({"hello":5})

	@raises(ArgumentException)
	def test_pointSimilaritesUnexpectedString(self):
		""" Test pointSimilarities raise exception for unexpected string value """
		self.backend_Sim_UnexpectedString(True)

	@raises(ArgumentException)
	def test_featureSimilaritesUnexpectedString(self):
		""" Test featureSimilarities raise exception for unexpected string value """
		self.backend_Sim_UnexpectedString(False)

	def backend_Sim_UnexpectedString(self, axis):
		data = [[1,2],[3,4]]
		obj = self.constructor(data)

		if axis:
			obj.pointSimilarities("foo")
		else:
			obj.featureSimilarities("foo")


	# test results covariance
	def test_pointSimilaritesSampleCovarianceResult(self):
		""" Test pointSimilarities returns correct sample covariance results """
		self.backend_Sim_SampleCovarianceResult(True)

	def test_featureSimilaritesSampleCovarianceResult(self):
		""" Test featureSimilarities returns correct sample covariance results """
		self.backend_Sim_SampleCovarianceResult(False)

	def backend_Sim_SampleCovarianceResult(self, axis):
		data = [[3,0,3],[0,0,3], [3,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointSimilarities("covariance")
		else:
			orig.transpose()
			ret = orig.featureSimilarities("samplecovariance")
			ret.transpose()
			orig.transpose()

		# hand computed results
		expRow0 = [3, 1.5, 1.5]
		expRow1 = [1.5, 3, -1.5]
		expRow2 = [1.5, -1.5, 3]
		expData = [expRow0, expRow1, expRow2]
		expObj = self.constructor(expData)

		# numpy computted result -- bias=0 -> divisor of n-1
		npExpRaw = numpy.cov(data, bias=0)
		npExpObj = self.constructor(npExpRaw)
		assert ret.isApproximatelyEqual(npExpObj)

		assert expObj.isApproximatelyEqual(ret)
		assert sameAsOrig == orig

	def test_pointSimilaritesPopulationCovarianceResult(self):
		""" Test pointSimilarities returns correct population covariance results """
		self.backend_Sim_populationCovarianceResult(True)

	def test_featureSimilaritesPopulationCovarianceResult(self):
		""" Test featureSimilarities returns correct population covariance results """
		self.backend_Sim_populationCovarianceResult(False)

	def backend_Sim_populationCovarianceResult(self, axis):
		data = [[3,0,3],[0,0,3], [3,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointSimilarities("populationcovariance")
		else:
			orig.transpose()
			ret = orig.featureSimilarities("populationcovariance")
			ret.transpose()
			orig.transpose()

		# hand computed results
		expRow0 = [2, 1, 1]
		expRow1 = [1, 2, -1]
		expRow2 = [1, -1, 2]
		expData = [expRow0, expRow1, expRow2]
		expObj = self.constructor(expData)

		# numpy computted result -- bias=1 -> divisor of n
		npExpRaw = numpy.cov(data, bias=1)
		npExpObj = self.constructor(npExpRaw)
		assert ret.isApproximatelyEqual(npExpObj)

		assert expObj.isApproximatelyEqual(ret)
		assert sameAsOrig == orig

	def test_pointSimilaritesSTDandVarianceIdentity(self):
		""" Test identity between population covariance and population std of points """
		self.backend_Sim_STDandVarianceIdentity(True)

	def test_featureSimilaritesSTDandVarianceIdentity(self):
		""" Test identity between population covariance and population std of features """
		self.backend_Sim_STDandVarianceIdentity(False)

	def backend_Sim_STDandVarianceIdentity(self, axis):
		data = [[3,0,3],[0,0,3], [3,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointSimilarities("populationcovariance")
			stdVector = orig.pointStatistics("populationstd")
		else:
			orig.transpose()
			ret = orig.featureSimilarities("populationcovariance")
			stdVector = orig.featureStatistics("populationstd")
			ret.transpose()
			orig.transpose()

		numpy.testing.assert_approx_equal(ret[0,0], stdVector[0] * stdVector[0])
		numpy.testing.assert_approx_equal(ret[1,1], stdVector[1] * stdVector[1])
		numpy.testing.assert_approx_equal(ret[2,2], stdVector[2] * stdVector[2])


	# test results correlation
	def test_pointSimilaritesCorrelationResult(self):
		""" Test pointSimilarities returns correct correlation results """
		self.backend_Sim_CorrelationResult(True)

	def test_featureSimilaritesCorrelationResult(self):
		""" Test featureSimilarities returns correct correlation results """
		self.backend_Sim_CorrelationResult(False)

	def backend_Sim_CorrelationResult(self, axis):
		data = [[3,0,3],[0,0,3], [3,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointSimilarities("correlation")
		else:
			orig.transpose()
			ret = orig.featureSimilarities("correlation")
			ret.transpose()
			orig.transpose()

		expRow0 = [1,      (1./2),  (1./2)]
		expRow1 = [(1./2), 1,       (-1./2)]
		expRow2 = [(1./2), (-1./2), 1]
		expData = [expRow0, expRow1, expRow2]
		expObj = self.constructor(expData)

		npExpRaw = numpy.corrcoef(data)
		npExpObj = self.constructor(npExpRaw)

		assert ret.isApproximatelyEqual(npExpObj)
		assert expObj.isApproximatelyEqual(ret)
		assert sameAsOrig == orig

	def test_pointSimilaritesCorrelationHelpersEquiv(self):
		""" Compare pointSimilarities correlation using the various possible helpers """
		self.backend_Sim_CorrelationHelpersEquiv(True)

	def test_featureSimilaritesCorrelationHelpersEquiv(self):
		""" Compare featureSimilarities correlation using the various possible helpers """
		self.backend_Sim_CorrelationHelpersEquiv(False)

	def backend_Sim_CorrelationHelpersEquiv(self, axis):
		data = [[3,0,3],[0,0,3], [3,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		def explicitCorr(X, X_T, sample=True):
			sampleStdVector = X.pointStatistics('samplestd')
			popStdVector = X.pointStatistics('populationstd')
			stdVector = sampleStdVector if sample else popStdVector
						
			stdVector_T = stdVector.copy()
			stdVector_T.transpose()

			if sample:
				cov = X.pointSimilarities('samplecovariance')
			else:
				cov = X.pointSimilarities('populationcovariance')

			stdMatrix = stdVector * stdVector_T
			ret = cov / stdMatrix

			return ret

		if axis:
			ret = orig.pointSimilarities("correlation")
			trans = orig.copy()
			trans.transpose()
			sampRet = explicitCorr(orig, True)
			popRet = explicitCorr(orig, False)
		else:
			trans = orig.copy()
			orig.transpose()
			ret = orig.featureSimilarities("correlation")
			sampRet = explicitCorr(trans, orig, True)
			popRet = explicitCorr(trans, orig, False)
			ret.transpose()
			orig.transpose()

		npExpRawB0 = numpy.corrcoef(data, bias=0)
		npExpRawB1 = numpy.corrcoef(data, bias=1)
		npExpB0 = self.constructor(npExpRawB0)
		npExpB1 = self.constructor(npExpRawB1)

		assert ret.isApproximatelyEqual(sampRet)
		assert ret.isApproximatelyEqual(popRet)
		assert sampRet.isApproximatelyEqual(popRet)
		assert ret.isApproximatelyEqual(npExpB0)
		assert ret.isApproximatelyEqual(npExpB1)
		


	# test results dot product
	def test_pointSimilaritesDotProductResult(self):
		""" Test pointSimilarities returns correct dot product results """
		self.backend_Sim_DotProductResult(True)

	def test_featureSimilaritesDotProductResult(self):
		""" Test featureSimilarities returns correct dot product results """
		self.backend_Sim_DotProductResult(False)

	def backend_Sim_DotProductResult(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointSimilarities("dotproduct")
		else:
			orig.transpose()
			ret = orig.featureSimilarities("dotproduct")
			ret.transpose()
			orig.transpose()

		expData = [[3, 2, 1], [2, 2, 0], [1, 0, 1]]
		expObj = self.constructor(expData)

		assert expObj == ret
		assert sameAsOrig == orig

	# test input function validation
	@raises(ArgumentException)
	def todotest_pointSimilaritesFuncValidation(self):
		""" Test pointSimilarities raises exception for invalid funcitions """
		self.backend_Sim_FuncValidation(True)

	@raises(ArgumentException)
	def todotest_featureSimilaritesFuncValidation(self):
		""" Test featureSimilarities raises exception for invalid funcitions """
		self.backend_Sim_FuncValidation(False)

	def backend_Sim_FuncValidation(self, axis):
		assert False
		data = [[1,2],[3,4]]
		obj = self.constructor(data)

		def singleArg(one):
			return one

		if axis:
			obj.pointSimilarities(singleArg)
		else:
			obj.featureSimilarities(singleArg)

	# test results passed function
	def todotest_pointSimilariteGivenFuncResults(self):
		""" Test pointSimilarities returns correct results for given function """
		self.backend_Sim_GivenFuncResults(True)

	def todotest_featureSimilaritesGivenFuncResults(self):
		""" Test featureSimilarities returns correct results for given function """
		self.backend_Sim_GivenFuncResults(False)

	def backend_Sim_GivenFuncResults(self, axis):
		assert False
		data = [[1,2],[3,4]]
		obj = self.constructor(data)

		def euclideanDistance(left, right):
			assert False

		if axis:
			obj.pointSimilarities(euclideanDistance)
		else:
			obj.featureSimilarities(euclideanDistance)

	################### ####################
	# pointStatistics # #featureStatistics #
	################### ####################

	def test_pointStatistics_max(self):
		""" Test pointStatistics returns correct max results """
		self.backend_Stat_max(True)

	def test_featureStatistics_max(self):
		""" Test featureStatistics returns correct max results """
		self.backend_Stat_max(False)

	def backend_Stat_max(self, axis):
		data = [[1,2,1],[-10,-1,-21], [-1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("MAx")
			assert ret.pointCount == 3
			assert ret.featureCount == 1
		else:
			orig.transpose()
			
			ret = orig.featureStatistics("max")
			assert ret.pointCount == 1
			assert ret.featureCount == 3

			ret.transpose()
			orig.transpose()

		expRaw = [[2],[-1],[0]]
		expObj = self.constructor(expRaw)

		assert expObj == ret
		assert sameAsOrig == orig			

	def test_pointStatistics_mean(self):
		""" Test pointStatistics returns correct mean results """
		self.backend_Stat_mean(True)

	def test_featureStatistics_mean(self):
		""" Test featureStatistics returns correct mean results """
		self.backend_Stat_mean(False)

	def backend_Stat_mean(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("Mean")
			assert ret.pointCount == 3
			assert ret.featureCount == 1
		else:
			orig.transpose()
			ret = orig.featureStatistics("MEAN")
			assert ret.pointCount == 1
			assert ret.featureCount == 3
			ret.transpose()
			orig.transpose()

		expRaw = [[1],[2./3],[1./3]]
		expObj = self.constructor(expRaw)

		assert expObj == ret
		assert sameAsOrig == orig

	def test_pointStatistics_median(self):
		""" Test pointStatistics returns correct median results """
		self.backend_Stat_median(True)

	def test_featureStatistics_median(self):
		""" Test featureStatistics returns correct median results """
		self.backend_Stat_median(False)

	def backend_Stat_median(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("MeDian")
			assert ret.pointCount == 3
			assert ret.featureCount == 1
		else:
			orig.transpose()
			ret = orig.featureStatistics("median")
			assert ret.pointCount == 1
			assert ret.featureCount == 3
			ret.transpose()
			orig.transpose()

		expRaw = [[1],[1],[0]]
		expObj = self.constructor(expRaw)

		assert expObj == ret
		assert sameAsOrig == orig


	def test_pointStatistics_min(self):
		""" Test pointStatistics returns correct min results """
		self.backend_Stat_min(True)

	def test_featureStatistics_min(self):
		""" Test featureStatistics returns correct min results """
		self.backend_Stat_min(False)

	def backend_Stat_min(self, axis):
		data = [[1,2,1],[-10,-1,-21], [-1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("mIN")
			assert ret.pointCount == 3
			assert ret.featureCount == 1
		else:
			orig.transpose()
			ret = orig.featureStatistics("min")
			assert ret.pointCount == 1
			assert ret.featureCount == 3
			ret.transpose()
			orig.transpose()

		expRaw = [[1],[-21],[-1]]
		expObj = self.constructor(expRaw)

		assert expObj == ret
		assert sameAsOrig == orig

	def test_pointStatistics_uniqueCount(self):
		""" Test pointStatistics returns correct uniqueCount results """
		self.backend_Stat_uniqueCount(True)

	def test_featureStatistics_uniqueCount(self):
		""" Test featureStatistics returns correct uniqueCount results """
		self.backend_Stat_uniqueCount(False)

	def backend_Stat_uniqueCount(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,-1]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("uniqueCount")
			assert ret.pointCount == 3
			assert ret.featureCount == 1
		else:
			orig.transpose()
			ret = orig.featureStatistics("UniqueCount")
			assert ret.pointCount == 1
			assert ret.featureCount == 3
			ret.transpose()
			orig.transpose()

		expRaw = [[1],[2],[3]]
		expObj = self.constructor(expRaw)

		assert expObj == ret
		assert sameAsOrig == orig

	def todotest_pointStatistics_proportionMissing(self):
		""" Test pointStatistics returns correct proportionMissing results """
		self.backend_Stat_proportionMissing(True)

	def todotest_featureStatistics_proportionMissing(self):
		""" Test featureStatistics returns correct proportionMissing results """
		self.backend_Stat_proportionMissing(False)

	def backend_Stat_proportionMissing(self, axis):
		data = [[1,None,1],[0,1,float('nan')], [1,float('nan'),None]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("ProportionMissing")
			assert ret.pointCount == 3
			assert ret.featureCount == 1
		else:
			orig.transpose()
			ret = orig.featureStatistics("proportionmissing")
			assert ret.pointCount == 1
			assert ret.featureCount == 3
			ret.transpose()
			orig.transpose()

		expRaw = [[1./3],[1./3],[2./3]]
		expObj = self.constructor(expRaw)

		assert expObj == ret
		assert sameAsOrig == orig

	def test_pointStatistics_proportionZero(self):
		""" Test pointStatistics returns correct proportionZero results """
		self.backend_Stat_proportionZero(True)

	def test_featureStatistics_proportionZero(self):
		""" Test featureStatistics returns correct proportionZero results """
		self.backend_Stat_proportionZero(False)

	def backend_Stat_proportionZero(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("proportionZero")
			assert ret.pointCount == 3
			assert ret.featureCount == 1
		else:
			orig.transpose()
			ret = orig.featureStatistics("proportionZero")
			assert ret.pointCount == 1
			assert ret.featureCount == 3
			ret.transpose()
			orig.transpose()

		expRaw = [[0],[1./3],[2./3]]
		expObj = self.constructor(expRaw)

		assert expObj == ret
		assert sameAsOrig == orig

	def test_pointStatistics_samplestd(self):
		""" Test pointStatistics returns correct sample std results """
		self.backend_Stat_sampleStandardDeviation(True)

	def test_featureStatistics_samplestd(self):
		""" Test featureStatistics returns correct sample std results """
		self.backend_Stat_sampleStandardDeviation(False)

	def backend_Stat_sampleStandardDeviation(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("samplestd")
			assert ret.pointCount == 3
			assert ret.featureCount == 1
		else:
			orig.transpose()
			ret = orig.featureStatistics("standarddeviation")
			assert ret.pointCount == 1
			assert ret.featureCount == 3
			ret.transpose()
			orig.transpose()

		npExpRaw = numpy.std(data, axis=1, ddof=1, keepdims=True)
		npExpObj = self.constructor(npExpRaw)

		assert npExpObj.isApproximatelyEqual(ret)

		expRaw = [[0],[math.sqrt(3./9)],[math.sqrt(3./9)]]
		expObj = self.constructor(expRaw)

		assert expObj.isApproximatelyEqual(ret)
		assert sameAsOrig == orig


	def test_pointStatistics_populationstd(self):
		""" Test pointStatistics returns correct population std results """
		self.backend_Stat_populationStandardDeviation(True)

	def test_featureStatistics_populationstd(self):
		""" Test featureStatistics returns correct population std results """
		self.backend_Stat_populationStandardDeviation(False)

	def backend_Stat_populationStandardDeviation(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("populationstd")
			assert ret.pointCount == 3
			assert ret.featureCount == 1
		else:
			orig.transpose()
			ret = orig.featureStatistics("populationstandarddeviation")
			assert ret.pointCount == 1
			assert ret.featureCount == 3
			ret.transpose()
			orig.transpose()

		npExpRaw = numpy.std(data, axis=1, ddof=0, keepdims=True)
		npExpObj = self.constructor(npExpRaw)

		assert npExpObj.isApproximatelyEqual(ret)

		expRaw = [[0],[math.sqrt(2./9)],[math.sqrt(2./9)]]
		expObj = self.constructor(expRaw)

		assert expObj.isApproximatelyEqual(ret)
		assert sameAsOrig == orig

	@raises(ArgumentException)
	def test_pointStatistics_unexpectedString(self):
		""" Test pointStatistics returns correct std results """
		self.backend_Stat_unexpectedString(True)

	@raises(ArgumentException)
	def test_featureStatistics_unexpectedString(self):
		""" Test featureStatistics returns correct std results """
		self.backend_Stat_unexpectedString(False)

	def backend_Stat_unexpectedString(self, axis):
		data = [[1,1,1],[0,1,1], [1,0,0]]
		orig = self.constructor(data)
		sameAsOrig = self.constructor(data)

		if axis:
			ret = orig.pointStatistics("hello")
		else:
			orig.transpose()
			ret = orig.featureStatistics("meanie")
			ret.transpose()
			orig.transpose()

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
