
import tempfile
import numpy
from nose.tools import *

from copy import deepcopy

from UML.data.dataHelpers import View
from UML.data.tests.baseObject import DataTestObject


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

	def test_isIdentical_True(self):
		""" Test isIdentical() against some actually equal input """
		toTest1 = self.constructor([[4,5]])
		toTest2 = self.constructor(deepcopy([[4,5]]))
		assert toTest1.isIdentical(toTest2)
		assert toTest2.isIdentical(toTest1)


	############
	# writeFile #
	############

	def test_writeFileCSV_handmade(self):
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


	def test_writeFileMTX_handmade(self):
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



