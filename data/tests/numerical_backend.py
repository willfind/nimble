
import numpy
import copy
import random

from nose.tools import *

import UML

from UML import createData

from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException

from UML.data.tests.baseObject import DataTestObject


class NumericalBackend(DataTestObject):


	########################
	# matrixMultiply #
	########################

	@raises(ArgumentException)
	def test_matrixMultiply_otherObjectExceptions(self):
		""" Test matrixMultiply raises exception when param is not a UML data object """
		data = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data)
		caller.matrixMultiply(data)

	@raises(ArgumentException)
	def test_matrixMultiply_selfNotNumericException(self):
		""" Test matrixMultiply raises exception if self has non numeric data """
		data1 = [['why','2','3'], ['4','5','6'], ['7','8','9']]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		try:
			caller = self.constructor(data1)
			callee = self.constructor(data2)
		except:
			raise ArgumentException("Data type doesn't support non numeric data")
		caller.matrixMultiply(callee)

	@raises(ArgumentException)
	def test_matrixMultiply_otherNotNumericException(self):
		""" Test matrixMultiply raises exception if param object has non numeric data """
		data1 = [[1,2,3], [4,5,6], [7,8,9]]
		data2 = [['why','2','3'], ['4','5','6'], ['7','8','9']]
		caller = self.constructor(data1)
		callee = createData("List", data2)

		caller.matrixMultiply(callee)

	@raises(ArgumentException)
	def test_matrixMultiply_shapeException(self):
		""" Test matrixMultiply raises exception the shapes of the object don't fit correctly """
		data1 = [[1,2], [4,5], [7,8]]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data1)
		callee = self.constructor(data2)

		caller.matrixMultiply(callee)

	@raises(ImproperActionException)
	def test_matrixMultiply_pEmptyException(self):
		""" Test matrixMultiply raises exception for point empty data """
		data = []
		fnames = ['one', 'two']
		caller = self.constructor(data, featureNames=fnames)
		callee = caller.copy()
		callee.transpose()

		caller.matrixMultiply(callee)


	@raises(ImproperActionException)
	def test_matrixMultiply_fEmptyException(self):
		""" Test matrixMultiply raises exception for feature empty data """
		data = [[],[]]
		pnames = ['one', 'two']
		caller = self.constructor(data, pointNames=pnames)
		callee = caller.copy()
		callee.transpose()

		caller.matrixMultiply(callee)


	def test_matrixMultiply_handmade(self):
		""" Test matrixMultiply on handmade data """
		data = [[2,4,8], [4,2,8], [8,2,4]]
		exp1 = [[4,8,16], [8,4,16], [16,4,8]]
		exp2 = [[1,2,4], [2,1,4], [4,1,2]]

		dubs = [[2,0,0], [0,2,0], [0,0,2]]
		quarts = [[.25,0,0], [0,.25,0], [0,0,.25]]

		caller = self.constructor(data)
		unchanged = self.constructor(data)
		dubsObj = self.constructor(dubs)
		ret1 = caller.matrixMultiply(dubsObj)

		exp1Obj = self.constructor(exp1)

		assert exp1Obj.isIdentical(ret1)
		assert unchanged.isIdentical(caller)

		quartsObj = self.constructor(quarts)
		ret2 =ret1.matrixMultiply(quartsObj)

		exp2Obj = self.constructor(exp2)

		assert exp2Obj.isIdentical(ret2)


	def test_matrixMultiply_auto(self):
		""" Test matrixMultiply against automated data """
		trials = 10
		for t in range(trials):
			n = random.randint(1,15)
			m = random.randint(1,15)
			k = random.randint(1,15)

			lhs = numpy.ones((n,m))
			rhs = numpy.ones((m,k))
			result = lhs.dot(rhs)

			lhsObj = self.constructor(lhs)
			rhsObj = self.constructor(rhs)
			expObj = self.constructor(result)
			resObj = lhsObj.matrixMultiply(rhsObj)

			assert expObj.isIdentical(resObj)


		
	def test_matrixMultiply_handmadeDiffTypes(self):
		""" Test matrixMultiply on handmade data with different types of data objects"""
		data = [[2,4,8], [4,2,8], [8,2,4]]
		exp1 = [[4,8,16], [8,4,16], [16,4,8]]

		dubs = [[2,0,0], [0,2,0], [0,0,2]]

		makers = [UML.data.List, UML.data.Matrix, UML.data.Sparse]

		for i in range(len(makers)):
			maker = makers[i]
			caller = self.constructor(data)
			unchanged = self.constructor(data)
			dubsObj = maker(dubs)
			ret1 = caller.matrixMultiply(dubsObj)

			exp1Obj = self.constructor(exp1)

			assert exp1Obj.isIdentical(ret1)
			assert unchanged.isIdentical(caller)

			if type(ret1) != type(caller):
				assert type(ret1) == maker.__class__


	#############################
	# elementwiseMultiply #
	#############################

	@raises(ArgumentException)
	def test_elementwiseMultiply_otherObjectExceptions(self):
		""" Test elementwiseMultiply raises exception when param is not a UML data object """
		data = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data)
		caller.elementwiseMultiply(data)

	@raises(ArgumentException)
	def test_elementwiseMultiply_selfNotNumericException(self):
		""" Test elementwiseMultiply raises exception if self has non numeric data """
		data1 = [['why','2','3'], ['4','5','6'], ['7','8','9']]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		try:
			caller = self.constructor(data1)
			callee = self.constructor(data2)
		except:
			raise ArgumentException("Data type doesn't support non numeric data")
		caller.elementwiseMultiply(callee)


	@raises(ArgumentException)
	def test_elementwiseMultiply_otherNotNumericException(self):
		""" Test elementwiseMultiply raises exception if param object has non numeric data """
		data1 = [[1,2,3], [4,5,6], [7,8,9]]
		data2 = [['one','2','3'], ['4','5','6'], ['7','8','9']]
		caller = self.constructor(data1)
		callee = createData("List", data2)

		caller.elementwiseMultiply(callee)


	@raises(ArgumentException)
	def test_elementwiseMultiply_pShapeException(self):
		""" Test elementwiseMultiply raises exception the shapes of the object don't fit correctly """
		data1 = [[1,2,6], [4,5,3], [7,8,6]]
		data2 = [[1,2,3], [4,5,6], ]
		caller = self.constructor(data1)
		callee = self.constructor(data2)

		caller.elementwiseMultiply(callee)

	@raises(ArgumentException)
	def test_elementwiseMultiply_fShapeException(self):
		""" Test elementwiseMultiply raises exception the shapes of the object don't fit correctly """
		data1 = [[1,2], [4,5], [7,8]]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data1)
		callee = self.constructor(data2)

		caller.elementwiseMultiply(callee)

	@raises(ImproperActionException)
	def test_elementwiseMultiply_pEmptyException(self):
		""" Test elementwiseMultiply raises exception for point empty data """
		data = []
		fnames = ['one', 'two']
		caller = self.constructor(data, featureNames=fnames)
		exp = self.constructor(data, featureNames=fnames)

		caller.elementwiseMultiply(caller)
		assert exp.isIdentical(caller)

	@raises(ImproperActionException)
	def test_elementwiseMultiply_fEmpty(self):
		""" Test elementwiseMultiply raises exception for feature empty data """
		data = [[],[]]
		pnames = ['one', 'two']
		caller = self.constructor(data, pointNames=pnames)
		exp = self.constructor(data, pointNames=pnames)

		caller.elementwiseMultiply(caller)
		assert exp.isIdentical(caller)


	def test_elementwiseMultiply_handmade(self):
		""" Test elementwiseMultiply on handmade data """
		data = [[1,2], [4,5], [7,8]]
		twos = [[2,2], [2,2], [2,2]]
		exp1 = [[2,4], [8,10], [14,16]]
		halves = [[0.5,0.5], [0.5,0.5], [0.5,0.5]]

		caller = self.constructor(data)
		twosObj = self.constructor(twos)
		caller.elementwiseMultiply(twosObj)

		exp1Obj = self.constructor(exp1)

		assert exp1Obj.isIdentical(caller)

		halvesObj = self.constructor(halves)
		caller.elementwiseMultiply(halvesObj)

		exp2Obj = self.constructor(data)

		assert caller.isIdentical(exp2Obj)


	def test_elementwiseMultiply_handmadeDifInputs(self):
		""" Test elementwiseMultiply on handmade data with different input object types"""
		data = [[1,2], [4,5], [7,8]]
		twos = [[2,2], [2,2], [2,2]]
		exp1 = [[2,4], [8,10], [14,16]]
		halves = [[0.5,0.5], [0.5,0.5], [0.5,0.5]]

		makers = [UML.data.List, UML.data.Matrix, UML.data.Sparse]

		for maker in makers:
			caller = self.constructor(data)
			twosObj = maker(twos)
			caller.elementwiseMultiply(twosObj)

			exp1Obj = self.constructor(exp1)

			assert exp1Obj.isIdentical(caller)

			halvesObj = maker(halves)
			caller.elementwiseMultiply(halvesObj)

			exp2Obj = self.constructor(data)

			assert caller.isIdentical(exp2Obj)


	########################
	# scalarMultiply #
	########################


	@raises(ArgumentException)
	def test_scalarMultiply_selfNotNumericException(self):
		""" Test scalarMultiply raises exception if self has non numeric data """
		data1 = [['one','2','3'], ['4','5','6'], ['7','8','9']]
		try:
			caller = self.constructor(data1)
		except:
			raise ArgumentException("Data type doesn't support non numeric data")
		caller.scalarMultiply(2)


	@raises(ArgumentException)
	def test_scalarMultiply_nonNumericParamException(self):
		""" Test scalarMultiply raises exception when param is not a numeric"""
		data = [[1,2], [4,5], [7,8]]
		caller = self.constructor(data)
		scalar = 'why hello there'

		caller.scalarMultiply(scalar)

	@raises(ImproperActionException)
	def test_scalarMultiply_pEmptyException(self):
		""" Test scalarMultiply raises exception for point empty data """
		data = []
		fnames = ['one', 'two']
		caller = self.constructor(data, featureNames=fnames)
		exp = self.constructor(data, featureNames=fnames)

		caller.scalarMultiply(5)
		assert exp.isIdentical(caller)

	@raises(ImproperActionException)
	def test_scalarMultiply_fEmpty(self):
		""" Test scalarMultiply raises exception for feature empty data """
		data = [[],[]]
		pnames = ['one', 'two']
		caller = self.constructor(data, pointNames=pnames)
		exp = self.constructor(data, pointNames=pnames)

		caller.scalarMultiply(5)
		assert exp.isIdentical(caller)


	def test_scalarMultiply_handmade(self):
		""" Test scalarMultiply on handmade data """
		data = [[1,2], [4,5], [7,8]]
		exp1 = [[2,4], [8,10], [14,16]]
		scalar = 2

		caller = self.constructor(copy.deepcopy(data))
		caller.scalarMultiply(scalar)

		exp1Obj = self.constructor(exp1)

		assert exp1Obj.isIdentical(caller)

		caller.scalarMultiply(0.5)

		exp2Obj = self.constructor(data)

		assert caller.isIdentical(exp2Obj)



