"""
Contains the discoverable test object for all classes in the data hierarchy.

Makes use of multiple inheritance to reuse (non-discoverable) test objects
higher in the the test object hierarchy associated with specific portions
of functionality. For example, View objects only inherit from those test
objects associated with non-destructive methods. Furthermore, each of
those higher objects are collections of unit tests generic over a
construction method, which is provided by the discoverable test objects
defined in this file. 

"""

import UML

from UML.data.tests.numerical_backend import AllNumerical
from UML.data.tests.numerical_backend import NumericalDataSafe

from UML.data.tests.query_backend import QueryBackend

from UML.data.tests.high_level_backend import HighLevelAll
from UML.data.tests.high_level_backend import HighLevelDataSafe

from UML.data.tests.low_level_backend import LowLevelBackend

from UML.data.tests.structure_backend import StructureAll
from UML.data.tests.structure_backend import StructureDataSafe

from UML.data.tests.view_access_backend import ViewAccess

def viewMakerMaker(concreteType):
	"""
	Method to help construct the constructors used in View test objects
	"""
	def maker(
			data, pointNames='automatic', featureNames='automatic', name=None,
			path=(None,None)):
		if isinstance(data, basestring):
			orig = UML.createData(
				concreteType, data=data, pointNames=pointNames,
				featureNames=featureNames, name=name)
		else:
			orig = UML.helpers.initDataObject(
				concreteType, rawData=data, pointNames=pointNames,
				featureNames=featureNames, name=name, path=path,
				keepPoints='all', keepFeatures='all')
		return orig.view()
	return maker


class TestListView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
		StructureDataSafe, ViewAccess):
	def __init__(self):
		super(TestListView, self).__init__('ListView', viewMakerMaker("List"))

class TestMatrixView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
		StructureDataSafe, ViewAccess):
	def __init__(self):
		super(TestMatrixView, self).__init__('MatrixView', viewMakerMaker("Matrix"))

#class TestSparseView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
#		StructureDataSafe, ViewAccess):
#	def __init__(self):
#		super(TestSparseView, self).__init__('SparseView', viewMakerMaker("Sparse"))


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
