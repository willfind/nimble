

from nose.tools import *

from UML import createData

from UML.exceptions import ArgumentException

from UML.data.tests.baseObject import DataTestObject


class NumericalBackend(DataTestObject):


	########################
	# matrixMultiplication #
	########################

	@raises(ArgumentException)
	def test_matrixMultiplication_otherObjectExceptions(self):
		""" Test matrixMultiplication raises exception when param is not a UML data object """
		data = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data)
		caller.matrixMultiplication(data)

	@raises(ArgumentException)
	def test_matrixMultiplication_selfNotNumericException(self):
		""" Test matrixMultiplication raises exception if self has non numeric data """
		data1 = [['why','2','3'], ['4','5','6'], ['7','8','9']]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		try:
			caller = self.constructor(data1)
			callee = self.constructor(data2)
		except:
			raise ArgumentException("Data type doesn't support non numeric data")
		caller.matrixMultiplication(callee)

	@raises(ArgumentException)
	def test_matrixMultiplication_otherNotNumericException(self):
		""" Test matrixMultiplication raises exception if param object has non numeric data """
		data1 = [[1,2,3], [4,5,6], [7,8,9]]
		data2 = [['why','2','3'], ['4','5','6'], ['7','8','9']]
		caller = self.constructor(data1)
		callee = createData("List", data2)

		caller.matrixMultiplication(callee)

	@raises(ArgumentException)
	def test_matrixMultiplication_shapeException(self):
		""" Test matrixMultiplication raises exception the shapes of the object don't fit correctly """
		data1 = [[1,2], [4,5], [7,8]]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data1)
		callee = self.constructor(data2)

		caller.matrixMultiplication(callee)



	#############################
	# elementwiseMultiplication #
	#############################

	@raises(ArgumentException)
	def test_elementwiseMultiplication_otherObjectExceptions(self):
		""" Test elementwiseMultiplication raises exception when param is not a UML data object """
		data = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data)
		caller.elementwiseMultiplication(data)

	@raises(ArgumentException)
	def test_elementwiseMultiplication_selfNotNumericException(self):
		""" Test elementwiseMultiplication raises exception if self has non numeric data """
		data1 = [['why','2','3'], ['4','5','6'], ['7','8','9']]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		try:
			caller = self.constructor(data1)
			callee = self.constructor(data2)
		except:
			raise ArgumentException("Data type doesn't support non numeric data")
		caller.elementwiseMultiplication(callee)


	@raises(ArgumentException)
	def test_elementwiseMultiplication_otherNotNumericException(self):
		""" Test elementwiseMultiplication raises exception if param object has non numeric data """
		data1 = [[1,2,3], [4,5,6], [7,8,9]]
		data2 = [['one','2','3'], ['4','5','6'], ['7','8','9']]
		caller = self.constructor(data1)
		callee = createData("List", data2)

		caller.elementwiseMultiplication(callee)


	@raises(ArgumentException)
	def test_elementwiseMultiplication_pShapeException(self):
		""" Test elementwiseMultiplication raises exception the shapes of the object don't fit correctly """
		data1 = [[1,2,6], [4,5,3], [7,8,6]]
		data2 = [[1,2,3], [4,5,6], ]
		caller = self.constructor(data1)
		callee = self.constructor(data2)

		caller.elementwiseMultiplication(callee)

	@raises(ArgumentException)
	def test_elementwiseMultiplication_fShapeException(self):
		""" Test elementwiseMultiplication raises exception the shapes of the object don't fit correctly """
		data1 = [[1,2], [4,5], [7,8]]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data1)
		callee = self.constructor(data2)

		caller.elementwiseMultiplication(callee)



	########################
	# scalarMultiplication #
	########################


	@raises(ArgumentException)
	def test_scalarMultiplication_selfNotNumericException(self):
		""" Test scalarMultiplication raises exception if self has non numeric data """
		data1 = [['one','2','3'], ['4','5','6'], ['7','8','9']]
		try:
			caller = self.constructor(data1)
		except:
			raise ArgumentException("Data type doesn't support non numeric data")
		caller.scalarMultiplication(2)


	@raises(ArgumentException)
	def test_scalarMultiplication_nonNumericParamException(self):
		""" Test scalarMultiplication raises exception when param is not a numeric"""
		data = [[1,2], [4,5], [7,8]]
		caller = self.constructor(data)
		scalar = 'why hello there'

		caller.scalarMultiplication(scalar)



