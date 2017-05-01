"""

Methods tested in this file:

In object StructureDataSafe:
copyAs, copyPoints, copyFeatures


In object StructureModifying:
__init__,  transpose, appendPoints, appendFeatures, sortPoints, sortFeatures,
extractPoints, extractFeatures, referenceDataFrom, transformEachPoint,
transformEachFeature, transformEachElement, fillWith



"""


import tempfile
import numpy
import scipy.sparse
import os
import os.path
from nose.tools import *

from copy import deepcopy

import UML
from UML import createData
from UML.data import List
from UML.data import Matrix
from UML.data import Sparse
from UML.data import BaseView
from UML.data.dataHelpers import DEFAULT_PREFIX
from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException

from UML.data.tests.baseObject import DataTestObject

preserveName = "PreserveTestName"
preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
preserveRPath = os.path.relpath(preserveAPath)
preservePair = (preserveAPath,preserveRPath)


### Helpers used by tests in the test class ###

def passThrough(value):
	return value

def plusOne(value):
	return (value + 1)

def plusOneOnlyEven(value):
	if value % 2 == 0:
		return (value + 1)
	else:
		return None


class StructureDataSafe(DataTestObject):

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
		sparseObj = createData(returnType="Sparse", data=data)
		listObj = createData(returnType="List", data=data)
		matixObj = createData(returnType="Matrix", data=data)

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
		sparseObj = createData(returnType="Sparse", data=data)
		listObj = createData(returnType="List", data=data)
		matixObj = createData(returnType="Matrix", data=data)

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
		sparseObj = createData(returnType="Sparse", data=data)
		listObj = createData(returnType="List", data=data)
		matixObj = createData(returnType="Matrix", data=data)

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
		sparseObj = createData(returnType="Sparse", data=data, pointNames=pointNames, featureNames=featureNames)
		listObj = createData(returnType="List", data=data, pointNames=pointNames, featureNames=featureNames)
		matixObj = createData(returnType="Matrix", data=data, pointNames=pointNames, featureNames=featureNames)

		pointsShuffleIndices = [3,1,2,0]
		featuresshuffleIndices = [1,2,0]

		copySparse = orig.copyAs(format='Sparse')
		assert copySparse.isIdentical(sparseObj)
		assert sparseObj.isIdentical(copySparse)
		assert type(copySparse) == Sparse
		copySparse.setFeatureName('two', '2')
		copySparse.setPointName('one', 'WHAT')
		assert 'two' in orig.getFeatureNames()
		assert 'one' in orig.getPointNames()
		copySparse.shufflePoints(pointsShuffleIndices)
		copySparse.shuffleFeatures(featuresshuffleIndices)
		assert orig[0,0] == 1 

		copyList = orig.copyAs(format='List')
		assert copyList.isIdentical(listObj)
		assert listObj.isIdentical(copyList)
		assert type(copyList) == List
		copyList.setFeatureName('two', '2')
		copyList.setPointName('one', 'WHAT')
		assert 'two' in orig.getFeatureNames()
		assert 'one' in orig.getPointNames()
		copyList.shufflePoints(pointsShuffleIndices)
		copyList.shuffleFeatures(featuresshuffleIndices)
		assert orig[0,0] == 1 

		copyMatrix = orig.copyAs(format='Matrix')
		assert copyMatrix.isIdentical(matixObj)
		assert matixObj.isIdentical(copyMatrix)
		assert type(copyMatrix) == Matrix
		copyMatrix.setFeatureName('two', '2')
		copyMatrix.setPointName('one', 'WHAT')
		assert 'two' in orig.getFeatureNames()
		assert 'one' in orig.getPointNames()
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

		spcsc = orig.copyAs(format='scipy csc')
		assert type(spcsc) == type(scipy.sparse.csc_matrix(numpy.matrix([])))
		spcsc[0,0] = 5
		assert orig[0,0] == 1

		spcsr = orig.copyAs(format='scipy csr')
		assert type(spcsr) == type(scipy.sparse.csr_matrix(numpy.matrix([])))
		spcsr[0,0] = 5
		assert orig[0,0] == 1

	def test_copy_rowsArePointsFalse(self):
		""" Test copyAs() will return data in the right places when rowsArePoints is False"""
		data = [[1,2,3],[1,0,3],[2,4,6],[0,0,0]]
		dataT = numpy.array(data).T.tolist()

		featureNames = ['one', 'two', 'three']
		pointNames = ['1', 'one', '2', '0']
		orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

		out = orig.copyAs(orig.getTypeString(), rowsArePoints=False)

		desired = self.constructor(dataT, pointNames=featureNames, featureNames=pointNames)

		assert out == desired

	def test_copy_outputAs1DWrongFormat(self):
		""" Test copyAs will raise exception when given an unallowed format """
		data = [[1,2,3],[1,0,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pointNames = ['1', 'one', '2', '0']
		orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

		try:
			orig.copyAs("List", outputAs1D=True)
			assert False
		except ArgumentException as ae:
			print ae
		try:
			orig.copyAs("Matrix", outputAs1D=True)
			assert False
		except ArgumentException as ae:
			print ae
		try:
			orig.copyAs("Sparse", outputAs1D=True)
			assert False
		except ArgumentException as ae:
			print ae
		try:
			orig.copyAs("numpy matrix", outputAs1D=True)
			assert False
		except ArgumentException as ae:
			print ae
		try:
			orig.copyAs("scipy csr", outputAs1D=True)
			assert False
		except ArgumentException as ae:
			print ae
		try:
			orig.copyAs("scipy csc", outputAs1D=True)
			assert False
		except ArgumentException as ae:
			print ae

	@raises(ArgumentException)
	def test_copy_outputAs1DWrongShape(self):
		""" Test copyAs will raise exception when given an unallowed shape """
		data = [[1,2,3],[1,0,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pointNames = ['1', 'one', '2', '0']
		orig = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

		orig.copyAs("numpy array", outputAs1D=True)


	def test_copyAs_outpuAs1DTrue(self):
		""" Test copyAs() will return successfully output 1d for all allowable possibilities"""
		dataPv = [[1,2, 0, 3]]
		dataFV = [[1],[2],[3],[0]]
		origPV = self.constructor(dataPv)
		origFV = self.constructor(dataFV)

		outPV = origPV.copyAs('python list', outputAs1D=True)
		assert outPV == [1,2,0,3]

		outFV = origFV.copyAs('numpy array', outputAs1D=True)
		assert numpy.array_equal(outFV, numpy.array([1,2,3,0]))

	def test_copyAs_NameAndPath(self):
		""" Test copyAs() will preserve name and path attributes"""

		data = [[1,2,3],[1,0,3],[2,4,6],[0,0,0]]
		name = 'copyAsTestName'
		orig = self.constructor(data)
		with tempfile.NamedTemporaryFile(suffix=".csv") as source:
			orig.writeFile(source.name, 'csv', includeNames=False)
			orig = self.constructor(source.name, name=name)
			path = source.name

		assert orig.name == name
		assert orig.path == path
		assert orig.absolutePath == path
		assert orig.relativePath == os.path.relpath(path)

		copySparse = orig.copyAs(format='Sparse')
		assert copySparse.name == orig.name
		assert copySparse.path == orig.path
		assert copySparse.absolutePath == path
		assert copySparse.relativePath == os.path.relpath(path)
		
		copyList = orig.copyAs(format='List')
		assert copyList.name == orig.name
		assert copyList.path == orig.path
		assert copyList.absolutePath == path
		assert copyList.relativePath == os.path.relpath(path)

		copyMatrix = orig.copyAs(format='Matrix')
		assert copyMatrix.name == orig.name
		assert copyMatrix.path == orig.path
		assert copyMatrix.absolutePath == path
		assert copyMatrix.relativePath == os.path.relpath(path)



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

	def test_copyPoints_list_NamePath(self):
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pnames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pnames, featureNames=featureNames,
				name=preserveName, path=preservePair)

		ret = orig.copyPoints([1,2])

		assert orig.name == preserveName
		assert orig.absolutePath == preserveAPath
		assert orig.relativePath == preserveRPath

		assert ret.nameIsDefault()
		assert ret.absolutePath == preserveAPath
		assert ret.relativePath == preserveRPath


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
		toTest.copyPoints(start=4,end=2)

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

	def test_copyPoints_range_NamePath(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, name=preserveName, path=preservePair)

		ret = toTest.copyPoints(start=1,end=2)

		assert toTest.name == preserveName
		assert toTest.absolutePath == preserveAPath
		assert toTest.relativePath == preserveRPath

		assert ret.nameIsDefault()
		assert ret.absolutePath == preserveAPath
		assert ret.relativePath == preserveRPath
		

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


	def test_copyFeatures_list_NamePath(self):
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pnames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pnames,
				featureNames=featureNames, name=preserveName, path=preservePair)

		ret = orig.copyFeatures([0,'two'])

		assert orig.name == preserveName
		assert orig.absolutePath == preserveAPath
		assert orig.relativePath == preserveRPath

		assert ret.nameIsDefault()
		assert ret.absolutePath == preserveAPath
		assert ret.relativePath == preserveRPath


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

	def test_copyFeatures_range_NamePath(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data, name=preserveName, path=preservePair)

		ret = toTest.copyFeatures(start=1, end=2)
		
		assert toTest.name == preserveName
		assert toTest.absolutePath == preserveAPath
		assert toTest.relativePath == preserveRPath

		assert ret.nameIsDefault()
		assert ret.absolutePath == preserveAPath
		assert ret.relativePath == preserveRPath


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




class StructureModifying(DataTestObject):


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
		tmpCSV.write("\n")
		tmpCSV.write("\n")
		tmpCSV.write("point_names,one,two,three\n")
		tmpCSV.write("1P,1,2,3\n")
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


	@raises(ArgumentException, TypeError)
	def test_init_noThriceNestedListInputs(self):
		self.constructor([[[1,2,3]]])


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

		ret1 = dataObj1.transpose() # RET CHECK

		assert dataObj1.isIdentical(dataObjT)
		assert ret1 is None

		dataObj1.transpose()
		dataObjT.transpose()
		assert dataObj1.isIdentical(dataObj2)
		assert dataObj2.isIdentical(dataObjT)

	def test_transpose_NamePath_preservation(self):
		data = [[1,2,3],[4,5,6],[7,8,9],[0,0,0],[11,12,13]]
		
		dataObj1 = self.constructor(deepcopy(data))

		dataObj1._name = "TestName"
		dataObj1._absPath = "TestAbsPath"
		dataObj1._relPath = "testRelPath"

		dataObj1.transpose()

		assert dataObj1.name == "TestName"
		assert dataObj1.absolutePath == "TestAbsPath"
		assert dataObj1.relativePath == 'testRelPath'




	#####################################
	# appendPoints() / appendFeatures() #
	#####################################

	def backend_append_exceptionNone(self, axis):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		
		if axis == 'point':
			toTest.appendPoints(None)
		else:
			toTest.appendFeatures(None)

	@raises(ArgumentException)
	def test_appendPoints_exceptionNone(self):
		""" Test appendPoints() for ArgumentException when toAppend is None"""
		self.backend_append_exceptionNone('point')

	@raises(ArgumentException)
	def test_appendFeatures_exceptionNone(self):
		""" Test appendFeatures() for ArgumentException when toAppend is None """
		self.backend_append_exceptionNone('feature')



	def backend_append_exceptionWrongSize(self, axis):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		toAppend = self.constructor([[2, 3, 4, 5, 6]])

		if axis == 'point':
			toTest.appendPoints(toAppend)
		else:
			toTest.appendFeatures(toAppend)

	@raises(ArgumentException)
	def test_appendPoints_exceptionWrongSize(self):
		""" Test appendPoints() for ArgumentException when toAppend has too many features """
		self.backend_append_exceptionWrongSize('point')

	@raises(ArgumentException)
	def test_appendFeatures_exceptionWrongSize(self):
		""" Test appendFeatures() for ArgumentException when toAppend has too many points """
		self.backend_append_exceptionWrongSize('feature')



	def backend_append_exception_extendAxis_SameName(self, axis):
		toTest1 = self.constructor([[1,2]], pointNames=["hello"])
		toTest2 = self.constructor([[1,2],[5,6]], pointNames=["hello","goodbye"])

		if axis == 'point':
			toTest2.appendPoints(toTest1)
		else:
			toTest1.transpose()
			toTest2.transpose()
			toTest2.appendFeatures(toTest1)

	@raises(ArgumentException)
	def test_appendPoints_exceptionSamePointName(self):
		""" Test appendPoints() for ArgumentException when toAppend and self have a pointName in common """
		self.backend_append_exception_extendAxis_SameName('point')

	@raises(ArgumentException)
	def test_appendFeatures_exceptionSameFeatureName(self):
		""" Test appendFeatures() for ArgumentException when toAppend and self have a featureName in common """
		self.backend_append_exception_extendAxis_SameName('feature')



	def backend_append_exception_sharedAxis_unsharedName(self, axis):
		toTest1 = self.constructor([[1,2]], featureNames=['1', '2'])
		toTest2 = self.constructor([[2,1],[6,5]], featureNames=['6','1'])

		if axis == 'point':
			toTest2.appendPoints(toTest1)
		else:
			toTest1.transpose()
			toTest2.transpose()
			toTest2.appendFeatures(toTest1)

	@raises(ArgumentException)
	def test_appendPoints_exception_unsharedFeatureName(self):
		""" Test appendPoints() for ArgumentException when toAppend and self have a featureName not in common """
		self.backend_append_exception_sharedAxis_unsharedName('point')

	@raises(ArgumentException)
	def test_appendFeatures_exception_unsharedPointName(self):
		""" Test appendFeatures() for ArgumentException when toAppend and self have a pointName not in common """
		self.backend_append_exception_sharedAxis_unsharedName('feature')



	def backend_append_exceptionNonUMLDataType(self, axis):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)

		if axis == 'point':
			toTest.appendPoints([[1,1,1]])
		else:
			toTest.appendFeatures([[1],[1],[1]])

	@raises(ArgumentException)
	def test_appendPoints_exceptionNonUMLDataType(self):
		self.backend_append_exceptionNonUMLDataType('point')

	@raises(ArgumentException)
	def test_appendFeatures_exceptionNonUMLDataType(self):
		self.backend_append_exceptionNonUMLDataType('feature')



	def backend_append_exceptionDifferentUMLDataType(self, axis):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		if toTest.getTypeString() == 'List':
			other = UML.createData("Matrix", data)
		else:
			other = UML.createData("List", data)

		if axis == 'point':
			toTest.appendPoints(other)
		else:
			toTest.appendFeatures(other)

	@raises(ArgumentException)
	def test_appendPoints_exceptionDifferentUMLDataType(self):
		self.backend_append_exceptionDifferentUMLDataType('point')

	@raises(ArgumentException)
	def test_appendFeatures_exceptionDifferentUMLDataType(self):
		self.backend_append_exceptionDifferentUMLDataType('feature')



	def backend_append_exception_outOfOrder_with_defaults(self, axis):
		toTest1 = self.constructor([[1,2,3]])
		toTest2 = self.constructor([[1,3,2]])

		toTest1.setFeatureName(1,'2')
		toTest1.setFeatureName(2,'3')
		toTest2.setFeatureName(1,'3')
		toTest2.setFeatureName(2,'2')

		if axis == 'point':
			toTest1.appendPoints(toTest2)
		else:
			toTest1.transpose()
			toTest2.transpose()
			toTest1.appendFeatures(toTest2)


	@raises(ArgumentException)
	def test_appendPoints_exception_outOfOrder_with_defaults(self):
		self.backend_append_exception_outOfOrder_with_defaults('point')

	@raises(ArgumentException)
	def test_appendFeatures_exception_outOfOrder_with_defaults(self):
		self.backend_append_exception_outOfOrder_with_defaults('feature')



	def backend_append_emptyObject(self, axis):
		empty = [[],[]]

		if axis == 'point':
			empty = numpy.array(empty).T
			data = [[1,2]]
		else:
			empty = numpy.array(empty)
			data = [[1],[2]]

		toTest = self.constructor(empty)
		toAdd = self.constructor(data)
		toExp = self.constructor(data)

		if axis == 'point':
			toTest.appendPoints(toAdd)
		else:
			toTest.appendFeatures(toAdd)

		assert toTest.isIdentical(toExp)

	def test_appendPoints_fromEmpty(self):
		""" Test appendPoints() when the calling object is point empty """
		self.backend_append_emptyObject('point')

	def test_appendFeatures_fromEmpty(self):
		""" Test appendFeatures() when the calling object is feature empty """
		self.backend_append_emptyObject('feature')		



	def backend_append_handmadeSingle(self, axis):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		offNames = ['o1', 'o2', 'o3']
		names = ['one', 'two', 'three']
		addName = ['new']
		namesExp = ['one', 'two', 'three', 'new']

		if axis == 'point':
			dataExpected = [[1,2,3],[4,5,6],[7,8,9],[-1,-2,-3]]
			toTest = self.constructor(data, pointNames=names, featureNames=offNames)
			toAppend = self.constructor([[-1,-2,-3]], pointNames=addName, featureNames=offNames)
			expected = self.constructor(dataExpected, pointNames=namesExp, featureNames=offNames)
			ret = toTest.appendPoints(toAppend)  # RET CHECK
		else:
			dataExpected = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
			toTest = self.constructor(data, pointNames=offNames, featureNames=names)
			toAppend = self.constructor([[-1],[-2],[-3]], pointNames=offNames, featureNames=addName)
			expected = self.constructor(dataExpected, pointNames=offNames, featureNames=namesExp)
			ret = toTest.appendFeatures(toAppend)  # RET CHECK
		
		assert toTest.isIdentical(expected)
		assert ret is None

	def test_appendPoints_handmadeSingle(self):
		""" Test appendPoints() against handmade output for a single added point """
		self.backend_append_handmadeSingle('point')

	def test_appendFeatures_handmadeSingle(self):
		""" Test appendFeatures() against handmade output for a single added feature"""
		self.backend_append_handmadeSingle('feature')



	def backend_append_handmadeSequence(self, axis):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		offNames = ['o1', 'o2', 'o3']
		names = ['one', 'two', 'three']
		newNames = ['a1', 'b1', 'b2', 'c1']
		namesExp = names + newNames
		toAppend = [[0.1,0.2,0.3],[0.01,0.02,0.03],[0,0,0],[10,11,12]]
		toAppend = self.constructor(toAppend, pointNames=newNames, featureNames=offNames)

		if axis == 'point':
			dataExpected = [[1,2,3],[4,5,6],[7,8,9],[0.1,0.2,0.3],[0.01,0.02,0.03],[0,0,0],[10,11,12]]
			toTest = self.constructor(data, pointNames=names, featureNames=offNames)
			for nextAdd in toAppend.pointIterator():
				toTest.appendPoints(nextAdd)
			expected = self.constructor(dataExpected, pointNames=namesExp, featureNames=offNames)
		else:
			dataExpected = [[1,2,3,0.1,0.01,0,10],[4,5,6,0.2,0.02,0,11],[7,8,9,0.3,0.03,0,12]]
			toTest = self.constructor(data, pointNames=offNames, featureNames=names)
			toAppend.transpose()
			for nextAdd in toAppend.featureIterator():
				toTest.appendFeatures(nextAdd)
			expected = self.constructor(dataExpected, pointNames=offNames, featureNames=namesExp)
		
		assert toTest.isIdentical(expected)

	def test_appendPoints_handmadeSequence(self):	
		""" Test appendPoints() against handmade output for a sequence of additions"""
		self.backend_append_handmadeSequence('point')

	def test_appendFeatures_handmadeSequence(self):
		""" Test appendFeatures() against handmade output for a sequence of additions"""
		self.backend_append_handmadeSequence('feature')



	def backend_append_NamePath_preservation(self, axis):
		data = [[1,2,3],[4,5,6],[7,8,9]]

		names = ['one', 'two', 'three']

		if axis == 'point':
			toTest = self.constructor(data, pointNames=names)
			toAppend = self.constructor([[-1,-2,-3]], pointNames=['new'])
		else:
			toTest = self.constructor(data, featureNames=names)
			toAppend = self.constructor([[-1],[-2],[-3]], featureNames=['new'])

		toTest._name = "TestName"
		toTest._absPath = "TestAbsPath"
		toTest._relPath = "testRelPath"

		toAppend._name = "TestNameOther"
		toAppend._absPath = "TestAbsPathOther"
		toAppend._relPath = "testRelPathOther"

		if axis == 'point':
			toTest.appendPoints(toAppend)
		else:
			toTest.appendFeatures(toAppend)

		assert toTest.name == "TestName"
		assert toTest.absolutePath == "TestAbsPath"
		assert toTest.relativePath == 'testRelPath'

	def test_appendPoints_NamePath_preservation(self):
		self.backend_append_NamePath_preservation('point')

	def test_appendFeatures_NamePath_preservation(self):
		self.backend_append_NamePath_preservation('feature')



	def backend_append_selfAppend(self, axis):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		names = ['one', 'two', 'three']

		if axis == 'point':
			orig = self.constructor(data, featureNames=names)
		else:
			orig = self.constructor(data, pointNames=names)

		dup = orig.copy()

		if axis == 'point':
			dupNames = dup.getPointNames()
			assert orig.getPointNames() == dupNames
			
			orig.appendPoints(orig)
			
			dataExp = [[1,2,3],[4,5,6],[7,8,9],[1,2,3],[4,5,6],[7,8,9]]
			expected = self.constructor(dataExp, featureNames=names)
		else:
			dupNames = dup.getFeatureNames()
			assert orig.getFeatureNames() == dupNames
			
			orig.appendFeatures(orig)

			dataExp = [[1,2,3,1,2,3],[4,5,6,4,5,6],[7,8,9,7,8,9]]
			expected = self.constructor(dataExp, pointNames=names)

		assert orig == expected

		checkNames = orig.getPointNames() if axis == 'point' else orig.getFeatureNames()
		assert checkNames[:3] == dupNames

		lastDefIndex = int(dupNames[2][-1])

		assert checkNames[3] == DEFAULT_PREFIX + str(lastDefIndex + 1)
		assert checkNames[4] == DEFAULT_PREFIX + str(lastDefIndex + 2)
		assert checkNames[5] == DEFAULT_PREFIX + str(lastDefIndex + 3)

	def test_appendPoints_selfAppend(self):
		self.backend_append_selfAppend('point')

	def test_appendFeatures_selfAppend(self):
		self.backend_append_selfAppend('feature')



	def backend_append_automaticReorder(self, axis, defPrimaryNames):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		offNames = ['off1', 'off2', 'off3']
		addOffName = ['off3', 'off2', 'off1']
		if defPrimaryNames:
			names = [None] * 3
			addName = [None]
			namesExp = [None] * 4
		else:
			names = ['one', 'two', 'three']
			addName = ['new']
			namesExp = ['one', 'two', 'three', 'new']

		if axis == 'point':
			toAddData = [[-3,-2,-1]]
			dataExpected = [[1,2,3],[4,5,6],[7,8,9],[-1,-2,-3]]
			toTest = self.constructor(data, pointNames=names, featureNames=offNames)
			toAppend = self.constructor(toAddData, pointNames=addName, featureNames=addOffName)
			expected = self.constructor(dataExpected, pointNames=namesExp, featureNames=offNames)
			toTest.appendPoints(toAppend)
		else:
			toAddData = [[-3],[-2],[-1]]
			dataExpected = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
			toTest = self.constructor(data, pointNames=offNames, featureNames=names)
			toAppend = self.constructor(toAddData, pointNames=addOffName, featureNames=addName)
			expected = self.constructor(dataExpected, pointNames=offNames, featureNames=namesExp)
			toTest.appendFeatures(toAppend)

		assert toTest.isIdentical(expected)



	def test_appendPoints_automaticReorder_fullySpecifiedNames(self):
		self.backend_append_automaticReorder('point', False)

	def test_appendFeatures_automaticReorder_fullySpecifiedNames(self):
		self.backend_append_automaticReorder('feature', False)

	def test_appendPoints_automaticReorder_defaultPointNames(self):
		self.backend_append_automaticReorder('point', True)

	def test_appendFeatures_automaticReorder_defaultFeatureNames(self):
		self.backend_append_automaticReorder('feature', True)




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

	def test_sortPoints_naturalByFeatureName(self):
		""" Test sortPoints() when we specify a feature name to sort by """	
		data = [[1,2,3],[7,1,9],[4,5,6]]
		pnames = ['1', '7', '4']
		fnames = ['1', '2', '3']
		toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

		ret = toTest.sortPoints(sortBy='2') # RET CHECK

		dataExpected = [[7,1,9],[1,2,3],[4,5,6]]
		namesExp = ['7', '1', '4']
		objExp = self.constructor(dataExpected, pointNames=namesExp, featureNames=fnames)

		assert toTest.isIdentical(objExp)
		assert ret is None


	def test_sortPoints_scorer(self):
		""" Test sortPoints() when we specify a scoring function """
		data = [[1,2,3],[4,5,6],[7,1,9],[0,0,0]]
		toTest = self.constructor(data)

		def numOdds(point):
			assert isinstance(point, BaseView)
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

	def test_sortFeatures_naturalByPointNameWithFNames(self):
		""" Test sortFeatures() when we specify a point name to sort by; includes featureNames """	
		data = [[1,2,3],[7,1,9],[4,5,6]]
		pnames = ['1', '7', '4']
		fnames = ["1","2","3"]
		toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

		ret = toTest.sortFeatures(sortBy='7') # RET CHECK

		dataExpected = [[2,1,3],[1,7,9],[5,4,6]]
		namesExp = ["2", "1", "3"]
		objExp = self.constructor(dataExpected, pointNames=pnames, featureNames=namesExp)

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

	def test_extractPoints_index_NamePath_Preserve(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		
		toTest._name = 'testName'
		toTest._absPath = 'testAbsPath'
		toTest._relPath = 'testRelPath'
		
		ext1 = toTest.extractPoints(0)
		
		assert ext1.nameIsDefault()
		assert ext1.path == 'testAbsPath'
		assert ext1.absolutePath == 'testAbsPath'
		assert ext1.relativePath == 'testRelPath'

		assert toTest.name == "testName"
		assert toTest.absolutePath == "testAbsPath"
		assert toTest.relativePath == 'testRelPath'


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

	def test_extractPoints_List_trickyOrdering(self):
		data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
		toExtract = [6, 5, 3, 9]

		toTest = self.constructor(data)

		ret = toTest.extractPoints(toExtract)

		expRaw = [[0],[0],[2],[0]]
		expRet = self.constructor(expRaw)

		expRaw = [[0],[2],[2],[0],[0],[2]]
		expRem = self.constructor(expRaw)

		assert ret == expRet
		assert toTest == expRem

	def test_extractPoints_function_selectionGap(self):
		data = [[0], [2], [2], [2], [0], [0], [0], [0], [2], [0]]
		extractIndices = [3,5,6,9]
		pnames = ['0','1','2','3','4','5','6','7','8','9']

		def sel(point):
			if int(point.getPointName(0)) in extractIndices:
				return True
			else:
				return False

		toTest = self.constructor(data, pointNames=pnames)

		ret = toTest.extractPoints(sel)

		expRaw = [[2],[0],[0],[0]]
		expNames = ['3','5','6','9']
		expRet = self.constructor(expRaw, pointNames=expNames)

		expRaw = [[0],[2],[2],[0],[0],[2]]
		expNames = ['0','1','2','4','7','8']
		expRem = self.constructor(expRaw, pointNames=expNames)

		assert ret == expRet
		assert toTest == expRem


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

	def test_extractPoints_function_returnPointEmpty(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		exp = self.constructor(data)
		def takeNone(point):
			return False

#		import pdb
#		pdb.set_trace()

		ret = toTest.extractPoints(takeNone)

		data = [[],[],[]]
		data = numpy.array(data).T
		expRet = self.constructor(data)

		assert ret.isIdentical(expRet)
		assert toTest.isIdentical(exp)

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

	def test_extractPoints_func_NamePath_preservation(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		def oneOrFour(point):
			if 1 in point or 4 in point:
				return True
			return False
		
		toTest._name = "testName"
		toTest._absPath = "testAbsPath"
		toTest._relPath = "testRelPath"

		ext = toTest.extractPoints(oneOrFour)

		assert toTest.name == "testName"
		assert toTest.absolutePath == "testAbsPath"
		assert toTest.relativePath == 'testRelPath'

		assert ext.nameIsDefault()
		assert ext.absolutePath == 'testAbsPath'
		assert ext.relativePath == 'testRelPath'

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
		toTest.extractPoints(start=1.1,end=2)

	@raises(ArgumentException)
	def test_extractPoints_exceptionEndInvalid(self):
		""" Test extractPoints() for ArgumentException when start is not a valid Point index """
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

	def test_extractPoints_range_NamePath_preservation(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)

		toTest._name = "testName"
		toTest._absPath = "testAbsPath"
		toTest._relPath = "testRelPath"

		ret = toTest.extractPoints(start=1,end=2)
		
		assert toTest.name == "testName"
		assert toTest.absolutePath == "testAbsPath"
		assert toTest.relativePath == 'testRelPath'

		assert ret.nameIsDefault()
		assert ret.absolutePath == 'testAbsPath'
		assert ret.relativePath == 'testRelPath'


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

	def test_extractPoints_handmade_calling_pointNames(self):
		featureNames = ["one","two","three"]
		pointNames = ['1', '4', '7']
		data = [[1,2,3],[4,5,6],[7,8,9]]

		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints(start='4',end='7')

		expectedRet = self.constructor([[4,5,6],[7,8,9]], pointNames=pointNames[1:], featureNames=featureNames)
		expectedTest = self.constructor([[1,2,3]], pointNames=pointNames[:1], featureNames=featureNames)

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_extractPoints_handmadeString(self):
		featureNames = ["one","two","three"]
		pointNames = ['1', '4', '7']
		data = [[1,2,3],[4,5,6],[7,8,9]]

		#test featureName=value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('one=1')
		expectedRet = self.constructor([[1,2,3]], pointNames=pointNames[:1], featureNames=featureNames)
		expectedTest = self.constructor([[4,5,6],[7,8,9]], pointNames=pointNames[1:], featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName==value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('one==1')
		expectedRet = self.constructor([[1,2,3]], pointNames=pointNames[:1], featureNames=featureNames)
		expectedTest = self.constructor([[4,5,6],[7,8,9]], pointNames=pointNames[1:], featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName<value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('one<2')
		expectedRet = self.constructor([[1,2,3]], pointNames=pointNames[:1], featureNames=featureNames)
		expectedTest = self.constructor([[4,5,6],[7,8,9]], pointNames=pointNames[1:], featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName<=value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('one<=1')
		expectedRet = self.constructor([[1,2,3]], pointNames=pointNames[:1], featureNames=featureNames)
		expectedTest = self.constructor([[4,5,6],[7,8,9]], pointNames=pointNames[1:], featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName>value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('one>4')
		expectedRet = self.constructor([[7,8,9]], pointNames=pointNames[-1:], featureNames=featureNames)
		expectedTest = self.constructor([[1,2,3], [4,5,6]], pointNames=pointNames[:-1], featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName>=value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('one>=7')
		expectedRet = self.constructor([[7,8,9]], pointNames=pointNames[-1:], featureNames=featureNames)
		expectedTest = self.constructor([[1,2,3], [4,5,6]], pointNames=pointNames[:-1], featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName!=value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('one!=4')
		expectedRet = self.constructor([[1,2,3],[7,8,9]], pointNames=[pointNames[0], pointNames[-1]], featureNames=featureNames)
		expectedTest = self.constructor([[4,5,6]], pointNames=[pointNames[1]], featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName<value and return back an empty
		assert expectedTest.isIdentical(toTest)
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('one<1')
		expectedRet = self.constructor([], featureNames=featureNames)
		expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName<value and return back all data
		assert expectedTest.isIdentical(toTest)
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('one>0')
		expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		expectedTest = self.constructor([], featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	@raises(ArgumentException)
	def test_extractPoints_handmadeString_featureNotExist(self):
		featureNames = ["one","two","three"]
		pointNames = ['1', '4', '7']
		data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractPoints('four=1')

	def test_extractPoints_numberOnly(self):
		self.back_extract_numberOnly('point')

	def test_extractPoints_numberAndRandomize(self):
		self.back_extract_numberAndRandomize('point')

	#TODO an extraction test where all data is removed
	#TODO extraction tests for all of the number and randomize combinations

	##########################
	# extract common backend #
	##########################

	def back_extract_numberOnly(self, axis):
		if axis == 'point':
			toCall = "extractPoints"
		else:
			toCall = "extractFeatures"

		data = [[1,2,3,33],[4,5,6,66],[7,8,9,99],[10,11,12,14]]
		pnames = ['1', '4', '7', '10']
		fnames = ['a', 'b', 'd', 'gg']
		toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
		ret = getattr(toTest,toCall)(number=3)

		if axis == 'point':
			exp = self.constructor(data[:3], pointNames=pnames[:3], featureNames=fnames)
			rem = self.constructor(data[3:], pointNames=pnames[3:], featureNames=fnames)
		else:
			exp = self.constructor([p[:3] for p in data], pointNames=pnames, featureNames=fnames[:3])
			rem = self.constructor([p[3:] for p in data], pointNames=pnames, featureNames=fnames[3:])

		assert exp.isIdentical(ret)
		assert rem.isIdentical(toTest)

	def back_extract_numberAndRandomize(self, axis):
		if axis == 'point':
			toCall = "extractPoints"
		else:
			toCall = "extractFeatures"

		data = [[1,2,3,33],[4,5,6,66],[7,8,9,99],[10,11,12,14]]
		pnames = ['1', '4', '7', '10']
		fnames = ['a', 'b', 'd', 'gg']
		toTest1 = self.constructor(data, pointNames=pnames, featureNames=fnames)
		toTest2 = self.constructor(data, pointNames=pnames, featureNames=fnames)

		UML.randomness.startAlternateControl()
		ret = getattr(toTest1,toCall)(number=3, randomize=True)
		UML.randomness.endAlternateControl()

		UML.randomness.startAlternateControl()
		retRange = getattr(toTest2,toCall)(start=0, end=3, number=3, randomize=True)
		UML.randomness.endAlternateControl()

		assert ret.isIdentical(retRange)
		assert toTest1.isIdentical(toTest2)


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

	def test_extractFeatures_List_NamePath_Preserve(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)

		toTest._name = "testName"
		toTest._absPath = "testAbsPath"
		toTest._relPath = "testRelPath"

		ext1 = toTest.extractFeatures(0)
		
		assert toTest.path == 'testAbsPath'
		assert toTest.absolutePath == 'testAbsPath'
		assert toTest.relativePath == 'testRelPath'

		assert ext1.nameIsDefault()
		assert ext1.absolutePath == 'testAbsPath'
		assert ext1.relativePath == 'testRelPath'

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


	def test_extractFeatures_List_trickyOrdering(self):
		data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
		toExtract = [6, 5, 3, 9]
#		toExtract = [3,5,6,9]

		toTest = self.constructor(data)

		ret = toTest.extractFeatures(toExtract)

		expRaw = [0,0,1,0]
		expRet = self.constructor(expRaw)

		expRaw = [0,1,1,0,0,1]
		expRem = self.constructor(expRaw)

		assert ret == expRet
		assert toTest == expRem

	def test_extractFeatures_List_reorderingWithFeatureNames(self):
		data = [[1,2,3,10], [4,5,6,11], [7,8,9,12]]
		fnames = ['a','b','c','d']
		test = self.constructor(data, featureNames=fnames)

		expRetRaw = [[1,3,2], [4,6,5], [7,9,8]]
		expRetNames = ['a', 'c', 'b']
		expRet = self.constructor(expRetRaw, featureNames=expRetNames)

		expTestRaw = [[10],[11],[12]]
		expTestNames = ['d']
		expTest = self.constructor(expTestRaw, featureNames=expTestNames)

		ret = test.extractFeatures(expRetNames)
		assert ret == expRet
		assert test == expTest


	def test_extractFeatures_function_selectionGap(self):
		data = [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
		fnames = ['0','1','2','3','4','5','6','7','8','9']

		extractIndices = [3,5,6,9]

		def sel(feature):
			if int(feature.getFeatureName(0)) in extractIndices:
				return True
			else:
				return False

		toTest = self.constructor(data, featureNames=fnames)

		ret = toTest.extractFeatures(sel)

		expRaw = [1,0,0,0]
		expNames = ['3','5','6','9']
		expRet = self.constructor(expRaw, featureNames=expNames)

		expRaw = [0,1,1,0,0,1]
		expNames = ['0','1','2','4','7','8']
		expRem = self.constructor(expRaw, featureNames=expNames)

		assert ret == expRet
		assert toTest == expRem


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

	def test_extractFeatures_function_returnPointEmpty(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		exp = self.constructor(data)
		def takeNone(point):
			return False

		ret = toTest.extractFeatures(takeNone)

		data = [[],[],[]]
		data = numpy.array(data)
		expRet = self.constructor(data)

		assert ret.isIdentical(expRet)
		assert toTest.isIdentical(exp)


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


	def test_extractFeatures_func_NamePath_preservation(self):
		data = [[1,2,3,-1],[4,5,6,-2],[7,8,9,-3]]
		toTest = self.constructor(data)
		
		def absoluteOne(feature):
			if 1 in feature or -1 in feature:
				return True
			return False

		toTest._name = "testName"
		toTest._absPath = "testAbsPath"
		toTest._relPath = "testRelPath"

		ext = toTest.extractFeatures(absoluteOne)

		assert toTest.name == "testName"
		assert toTest.absolutePath == "testAbsPath"
		assert toTest.relativePath == 'testRelPath'

		assert ext.nameIsDefault()
		assert ext.absolutePath == 'testAbsPath'
		assert ext.relativePath == 'testRelPath'

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
		toTest.extractFeatures(start=1.1, end=2)

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

	def test_extractFeatures_range_NamePath_preservation(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		
		toTest._name = "testName"
		toTest._absPath = "testAbsPath"
		toTest._relPath = "testRelPath"

		ret = toTest.extractFeatures(start=1, end=2)
		
		assert toTest.name == "testName"
		assert toTest.absolutePath == "testAbsPath"
		assert toTest.relativePath == 'testRelPath'

		assert ret.nameIsDefault()
		assert ret.absolutePath == 'testAbsPath'
		assert ret.relativePath == 'testRelPath'


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

	def test_extractFeatures_handmade_calling_featureNames(self):
		featureNames = ["one","two","three"]
		pointNames = ['1', '4', '7']
		data = [[1,2,3],[4,5,6],[7,8,9]]

		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures(start="two",end="three")

		expectedRet = self.constructor([[2,3],[5,6],[8,9]], pointNames=pointNames, featureNames=["two","three"])
		expectedTest = self.constructor([[1],[4],[7]], pointNames=pointNames, featureNames=["one"])

		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	def test_extractFeatures_handmadeString(self):
		featureNames = ["one", "two", "three"]
		pointNames = ['p1', 'p2', 'p3']
		data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

		#test pointName=value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('p2=5')
		expectedRet = self.constructor([[2], [5], [8]], pointNames=pointNames, featureNames=[featureNames[1]])
		expectedTest = self.constructor([[1, 3], [4, 6], [7, 9]], pointNames=pointNames, featureNames=[featureNames[0], featureNames[-1]])
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName==value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('p1==1')
		expectedRet = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
		expectedTest = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName<value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('p3<9')
		expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
		expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName<=value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('p3<=8')
		expectedRet = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
		expectedTest = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName>value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('p3>8')
		expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
		expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName>=value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('p3>8.5')
		expectedRet = self.constructor([[3], [6], [9]], pointNames=pointNames, featureNames=[featureNames[-1]])
		expectedTest = self.constructor([[1, 2], [4, 5], [7, 8]], pointNames=pointNames, featureNames=featureNames[:-1])
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName!=value
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('p1!=1.0')
		expectedRet = self.constructor([[2, 3], [5, 6], [8, 9]], pointNames=pointNames, featureNames=featureNames[1:])
		expectedTest = self.constructor([[1], [4], [7]], pointNames=pointNames, featureNames=[featureNames[0]])
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName<value and return back an empty
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('p1<1')
		if self.returnType == 'Sparse':
			expectedRet = self.constructor([], pointNames=pointNames)
		else:
			expectedRet = self.constructor([[], [], []], pointNames=pointNames)
		expectedTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

		#test featureName<value and return back all data
		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('p1>0')
		expectedRet = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		if self.returnType == 'Sparse':
			expectedTest = self.constructor([], pointNames=pointNames)
		else:
			expectedTest = self.constructor([[], [], []], pointNames=pointNames)
		assert expectedRet.isIdentical(ret)
		assert expectedTest.isIdentical(toTest)

	@raises(ArgumentException)
	def test_extractFeatures_handmadeString_pointNotExist(self):
		featureNames = ["one","two","three"]
		pointNames = ['1', '4', '7']
		data = [[1,2,3],[4,5,6],[7,8,9]]

		toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
		ret = toTest.extractFeatures('5=1')

	def test_extractFeatures_numberOnly(self):
		self.back_extract_numberOnly('feature')

	def test_extractFeatures_numberAndRandomize(self):
		self.back_extract_numberAndRandomize('feature')


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


	def test_referenceDataFrom_data_axisNames(self):
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pNames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

		data2 = [[-1,-2,-3,-4]]
		featureNames = ['1', '2', '3', '4']
		pNames = ['-1']
		other = self.constructor(data2, pointNames=pNames, featureNames=featureNames)

		ret = orig.referenceDataFrom(other)  # RET CHECK

		assert orig.data is other.data
		assert '-1' in orig.getPointNames()
		assert '1' in orig.getFeatureNames()
		assert ret is None

	def test_referenceDataFrom_ObjName_Paths(self):
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pNames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

		data2 = [[-1,-2,-3,]]
		featureNames = ['1', '2', '3']
		pNames = ['-1']
		other = self.constructor(data2, pointNames=pNames, featureNames=featureNames)

		orig._name = "testName"
		orig._absPath = "testAbsPath"
		orig._relPath = "testRelPath"

		other._name = "testNameother"
		other._absPath = "testAbsPathother"
		other._relPath = "testRelPathother"

		orig.referenceDataFrom(other)

		assert orig.name == "testName"
		assert orig.absolutePath == "testAbsPathother"
		assert orig.relativePath == 'testRelPathother'

		assert other.name == "testNameother"
		assert other.absolutePath == "testAbsPathother"
		assert other.relativePath == 'testRelPathother'


	def test_referenceDataFrom_allMetadataAttributes(self):
		data1 = [[1,2,3],[1,2,3],[2,4,6],[0,0,0]]
		featureNames = ['one', 'two', 'three']
		pNames = ['1', 'one', '2', '0']
		orig = self.constructor(data1, pointNames=pNames, featureNames=featureNames)

		data2 = [[-1,-2,-3,4,5,3,],[-1,-2,-3,4,5,3,]]
		other = self.constructor(data2,)


		orig.referenceDataFrom(other)

		assert orig._pointCount == other.pointCount
		assert orig._featureCount == other.featureCount

		assert orig._nextDefaultValuePoint == other._nextDefaultValuePoint
		assert orig._nextDefaultValueFeature == other._nextDefaultValueFeature


	########################
	# transformEachPoint() #
	########################

	@raises(ArgumentException)
	def test_transformEachPoint_exceptionInputNone(self):
		featureNames = {'number':0,'centi':2,'deci':1}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
		origObj.transformEachPoint(None)

	@raises(ImproperActionException)
	def test_transformEachPoint_exceptionPEmpty(self):
		data = [[],[]]
		data = numpy.array(data).T
		origObj = self.constructor(data)

		def emitLower(point):
			return point[origObj.getFeatureIndex('deci')]

		origObj.transformEachPoint(emitLower)

	@raises(ImproperActionException)
	def test_transformEachPoint_exceptionFEmpty(self):
		data = [[],[]]
		data = numpy.array(data)
		origObj = self.constructor(data)

		def emitLower(point):
			return point[origObj.getFeatureIndex('deci')]

		origObj.transformEachPoint(emitLower)

	def test_transformEachPoint_Handmade(self):
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllDeci(point):
			value = point[origObj.getFeatureIndex('deci')]
			return [value, value, value]

		lowerCounts = origObj.transformEachPoint(emitAllDeci)  # RET CHECK

		expectedOut = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
		exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

		assert lowerCounts is None
		assert origObj.isIdentical(exp)

	def test_transformEachPoint_NamePath_preservation(self):
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		toTest = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllDeci(point):
			value = point[toTest.getFeatureIndex('deci')]
			return [value, value, value]

		toTest._name = "TestName"
		toTest._absPath = "TestAbsPath"
		toTest._relPath = "testRelPath"

		toTest.transformEachPoint(emitAllDeci)

		assert toTest.name == "TestName"
		assert toTest.absolutePath == "TestAbsPath"
		assert toTest.relativePath == 'testRelPath'

	def test_transformEachPoint_HandmadeLimited(self):
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllDeci(point):
			value = point[origObj.getFeatureIndex('deci')]
			return [value, value, value]

		origObj.transformEachPoint(emitAllDeci, points=[3,'two'])

		expectedOut = [[1,0.1,0.01], [1,0.1,0.02], [0.1,0.1,0.1], [0.2,0.2,0.2]]
		exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

		assert origObj.isIdentical(exp)


	def test_transformEachPoint_nonZeroIterAndLen(self):
		origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
		origObj = self.constructor(deepcopy(origData))

		def emitNumNZ(point):
			ret = 0
			assert len(point) == 3
			for value in point.nonZeroIterator():
				ret += 1
			return [ret, ret, ret]

		origObj.transformEachPoint(emitNumNZ)

		expectedOut = [[3,3,3], [2,2,2], [2,2,2], [1,1,1]]
		exp = self.constructor(expectedOut)

		assert origObj.isIdentical(exp)


	##########################
	# transformEachFeature() #
	##########################

	@raises(ImproperActionException)
	def test_transformEachFeature_exceptionPEmpty(self):
		data = [[],[]]
		data = numpy.array(data).T
		origObj = self.constructor(data)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return 0
			return 1

		origObj.transformEachFeature(emitAllEqual)

	@raises(ImproperActionException)
	def test_transformEachFeature_exceptionFEmpty(self):
		data = [[],[]]
		data = numpy.array(data)
		origObj = self.constructor(data)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return 0
			return 1

		origObj.transformEachFeature(emitAllEqual)

	@raises(ArgumentException)
	def test_transformEachFeature_exceptionInputNone(self):
		featureNames = {'number':0,'centi':2,'deci':1}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
		origObj.transformEachFeature(None)


	def test_transformEachFeature_Handmade(self):
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return [0,0,0,0]
			return [1,1,1,1]

		lowerCounts = origObj.transformEachFeature(emitAllEqual)  # RET CHECK
		expectedOut = [[1,0,0], [1,0,0], [1,0,0], [1,0,0]]	
		exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)
		
		assert lowerCounts is None
		assert origObj.isIdentical(exp)


	def test_transformEachFeature_NamePath_preservation(self):
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		toTest = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return [0,0,0,0]
			return [1,1,1,1]

		toTest._name = "TestName"
		toTest._absPath = "TestAbsPath"
		toTest._relPath = "testRelPath"

		toTest.transformEachFeature(emitAllEqual)

		assert toTest.name == "TestName"
		assert toTest.absolutePath == "TestAbsPath"
		assert toTest.relativePath == 'testRelPath'


	def test_transformEachFeature_HandmadeLimited(self):
		featureNames = {'number':0,'centi':2,'deci':1}
		pointNames = {'zero':0, 'one':1, 'two':2, 'three':3}
		origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02]]
		origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

		def emitAllEqual(feature):
			first = feature[0]
			for value in feature:
				if value != first:
					return [0,0,0,0]
			return [1,1,1,1]

		origObj.transformEachFeature(emitAllEqual, features=[0,'centi'])
		expectedOut = [[1,0.1,0], [1,0.1,0], [1,0.1,0], [1,0.2,0]]
		exp = self.constructor(expectedOut, pointNames=pointNames, featureNames=featureNames)

		assert origObj.isIdentical(exp)


	def test_transformEachFeature_nonZeroIterAndLen(self):
		origData = [[1,1,1], [1,0,2], [1,1,0], [0,2,0]]
		origObj = self.constructor(deepcopy(origData))

		def emitNumNZ(feature):
			ret = 0
			assert len(feature) == 4
			for value in feature.nonZeroIterator():
				ret += 1
			return [ret, ret, ret, ret]

		origObj.transformEachFeature(emitNumNZ)

		expectedOut = [[3,3,2], [3,3,2], [3,3,2], [3,3,2]]
		exp = self.constructor(expectedOut)

		assert origObj.isIdentical(exp)


	##########################
	# transformEachElement() #
	##########################

	def test_transformEachElement_passthrough(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)

		ret = toTest.transformEachElement(passThrough)  # RET CHECK
		assert ret is None
		retRaw = toTest.copyAs(format="python list")

		assert [1,2,3] in retRaw
		assert [4,5,6] in retRaw
		assert [7,8,9] in retRaw


	def test_transformEachElement_NamePath_preservation(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)
		
		toTest._name = "TestName"
		toTest._absPath = "TestAbsPath"
		toTest._relPath = "testRelPath"

		toTest.transformEachElement(passThrough)

		assert toTest.name == "TestName"
		assert toTest.absolutePath == "TestAbsPath"
		assert toTest.relativePath == 'testRelPath'


	def test_transformEachElement_plusOnePreserve(self):
		data = [[1,0,3],[0,5,6],[7,0,9]]
		toTest = self.constructor(data)

		toTest.transformEachElement(plusOne, preserveZeros=True)
		retRaw = toTest.copyAs(format="python list")

		assert [2,0,4] in retRaw
		assert [0,6,7] in retRaw
		assert [8,0,10] in retRaw


	def test_transformEachElement_plusOneExclude(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		toTest = self.constructor(data)

		toTest.transformEachElement(plusOneOnlyEven, skipNoneReturnValues=True)
		retRaw = toTest.copyAs(format="python list")

		assert [1,3,3] in retRaw
		assert [5,5,7] in retRaw
		assert [7,9,9] in retRaw


	def test_transformEachElement_plusOneLimited(self):
		data = [[1,2,3],[4,5,6],[7,8,9]]
		names = ['one','two','three']
		pnames = ['1', '4', '7']
		toTest = self.constructor(data, pointNames=pnames, featureNames=names)

		toTest.transformEachElement(plusOneOnlyEven, points=1, features=[1,'three'], skipNoneReturnValues=True)
		retRaw = toTest.copyAs(format="python list")

		assert [1,2,3] in retRaw
		assert [4,5,7] in retRaw
		assert [7,8,9] in retRaw



	####################
	#### fillWith() ####
	####################

	# fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd)

	def test_fillWith_acceptableValues(self):
		raw = [[1,2],[3,4]]
		toTest = self.constructor(raw)

		try:
			toTest.fillWith(set([1,3]), 0, 0, 0, 1)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException
		
		try:
			toTest.fillWith(lambda x: x*x, 0, 0, 0, 1)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException


	def test_fillWith_sizeMismatch(self):
		raw = [[1,2],[3,4]]
		toTest = self.constructor(raw)

		raw = [[-1,-2]]
		val = self.constructor(raw)

		try:
			toTest.fillWith(val, 0, 0, 1, 1)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException

		val.transpose()

		try:
			toTest.fillWith(val, 0, 0, 1, 1)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException


	def test_fillWith_invalidID(self):
		raw = [[1,2],[3,4]]
		toTest = self.constructor(raw)

		val = 1

		try:
			toTest.fillWith(val, "hello", 0, 1, 1)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException
		try:
			toTest.fillWith(val, 0, "Wrong", 1, 1)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException
		try:
			toTest.fillWith(val, 0, 0, 2, 1)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException
		try:
			toTest.fillWith(val, 0, 0, 1, -12)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException


	def test_fillWith_start_lessThan_end(self):
		raw = [[1,2],[3,4]]
		toTest = self.constructor(raw)

		val = 1

		try:
			toTest.fillWith(val, 1, 0, 0, 1)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException
		try:
			toTest.fillWith(val, 0, 1, 1, 0)
			assert False  # expected ArgumentExcpetion
		except ArgumentException as ae:
			print ae
		except Exception:
			assert False  # expected ArgumentException


	def test_fillWith_fullObjectFill(self):
		raw = [[1,2],[3,4]]
		toTest = self.constructor(raw)

		arg = [[-1,-2],[-3,-4]]
		arg = self.constructor(arg)
		exp = arg.copy()

		ret = toTest.fillWith(arg, 0, 0, toTest.pointCount-1, toTest.featureCount-1)
		assert ret is None

		arg *= 10

		assert toTest == exp
		assert toTest != arg


	def test_fillWith_vectorFill(self):
		raw = [[1,2],[3,4]]
		toTestP = self.constructor(raw)
		toTestF = self.constructor(raw)

		rawP = [[-1,-2]]
		valP = self.constructor(rawP)

		rawF = [[-1],[-3]]
		valF = self.constructor(rawF)

		expP = [[-1,-2],[3,4]]
		expP = self.constructor(expP)

		expF = [[-1,2],[-3,4]]
		expF = self.constructor(expF)

		toTestP.fillWith(valP, 0, 0, 0, 1)
		assert toTestP == expP

		toTestF.fillWith(valF, 0, 0, 1, 0)
		assert toTestF == expF


	def test_fillWith_offsetSquare(self):
		raw = [[11,12,13],[21,22,23], [31,32,33]]
		base = self.constructor(raw)
		trialRaw = [[0,0],[0,0]]
		trial = self.constructor(trialRaw)

		leftCorner = [(0,0),(0,1),(1,0),(1,1)]
		for p,f in leftCorner:
			toTest = base.copy()

			toTest.fillWith(trial, p, f, p+1, f+1)
			assert toTest[p,f] == 0
			assert toTest[p+1,f] == 0
			assert toTest[p,f+1] == 0
			assert toTest[p+1,f+1] == 0


	def test_fillWith_constants(self):
		toTest0 = self.constructor([[0,0,0],[0,0,0],[0,0,0]]) 
		exp0 = self.constructor([[0,1,1],[0,1,1],[0,0,0]])
		toTest0.fillWith(1, 0, 1, 1, 2)
		assert toTest0 == exp0

		toTest1 = self.constructor([[1,1,1],[1,1,1],[1,1,1]])
		exp1 = self.constructor([[1,0,1],[1,0,1],[1,0,1]])
		toTest1.fillWith(0, 0, 1, 2, 1)
		assert toTest1 == exp1

		toTestI = self.constructor([[1,0,0],[0,1,0],[0,0,1]])
		expi = self.constructor([[1,0,2],[0,1,0],[2,0,1]])
		toTestI.fillWith(2, 0,2,0,2)
		toTestI.fillWith(2, 2,0,2,0)
		assert toTestI == expi


	def test_fillWIth_differentType(self):
		raw = [[11,12,13],[21,22,23],[31,32,33]]
		fill = [[0,0],[0,0]]
		exp = [[0,0,13],[0,0,23],[31,32,33]]
		exp = self.constructor(exp)
		for t in UML.data.available:
			toTest = self.constructor(raw)
			arg = UML.createData(t, fill)
			toTest.fillWith(arg, 0,0,1,1)
			assert toTest == exp


class StructureAll(StructureDataSafe, StructureModifying):
	pass
