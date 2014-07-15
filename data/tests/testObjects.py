
import UML

from UML.data.tests.numerical_backend import NumericalBackend
from UML.data.tests.query_backend import QueryBackend
from UML.data.tests.structure_backend import StructureBackend
from UML.data.tests.high_level_backend import HighLevelBackend
from UML.data.tests.low_level_backend import LowLevelBackend




class TestList(HighLevelBackend, NumericalBackend, QueryBackend, StructureBackend):
	def __init__(self):
		super(TestList, self).__init__('List')

class TestMatrix(HighLevelBackend, NumericalBackend, QueryBackend, StructureBackend):
	def __init__(self):
		super(TestMatrix, self).__init__('Matrix')

class TestSparse(HighLevelBackend, NumericalBackend, QueryBackend, StructureBackend):
	def __init__(self):
		super(TestSparse, self).__init__('Sparse')


class TestBaseOnly(LowLevelBackend):
	def __init__(self):
		def makeConst(num):
			def const(dummy=2):
				return num
			return const
		def makeAndDefine(pointNames=None, featureNames=None, psize=0, fsize=0):			
			""" Make a base data object that will think it has as many features as it has featureNames,
			even though it has no actual data """
			rows = psize if pointNames is None else len(pointNames)
			cols = fsize if featureNames is None else len(featureNames)
			ret = UML.data.Base((rows,cols), pointNames=pointNames, featureNames=featureNames)
			return ret

		self.constructor = makeAndDefine
