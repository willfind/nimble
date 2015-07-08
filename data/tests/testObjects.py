
import UML

from UML.data.tests.numerical_backend import AllNumerical
from UML.data.tests.numerical_backend import NumericalDataSafe

from UML.data.tests.query_backend import QueryBackend

from UML.data.tests.high_level_backend import HighLevelAll
from UML.data.tests.high_level_backend import HighLevelDataSafe

from UML.data.tests.low_level_backend import LowLevelBackend

from UML.data.tests.structure_backend import StructureAll
from UML.data.tests.structure_backend import StructureDataSafe



#class TestListView(HighLevelDataSafe, NumericalDataSafe, QueryBackend, StructureDataSafe):
#	def __init__(self):
#		def maker(data, pointNames=None, featureNames=None, name=None):
#			orig = UML.createData("List", data=data, pointNames=pointNames,
#					featureNames=featureNames, name=name)
#			return orig.view()
#		super(TestListView, self).__init__('ListView', maker)

#class TestMatrixView(HighLevelDataSafe, NumericalDataSafe, QueryBackend, StructureDataSafe):
#	def __init__(self):
#		def maker(data, pointNames=None, featureNames=None, name=None):
#			orig = UML.createData("Matrix", data=data, pointNames=pointNames,
#					featureNames=featureNames, name=name)
#			return orig.view()
#		super(TestMatrixView, self).__init__('MatrixView', maker)

#class TestSparseView(HighLevelDataSafe, NumericalDataSafe, QueryBackend, StructureDataSafe):
#	def __init__(self):
#		def maker(data, pointNames=None, featureNames=None, name=None):
#			orig = UML.createData("Sparse", data=data, pointNames=pointNames,
#					featureNames=featureNames, name=name)
#			return orig.view()
#		super(TestSparseView, self).__init__('SparseView', maker)


class TestList(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
	def __init__(self):
		super(TestList, self).__init__('List')

class TestMatrix(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
	def __init__(self):
		super(TestMatrix, self).__init__('Matrix')

class TestSparse(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
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
