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

from __future__ import absolute_import
import UML

from .numerical_backend import AllNumerical
from .numerical_backend import NumericalDataSafe

from .query_backend import QueryBackend

from .high_level_backend import HighLevelAll
from .high_level_backend import HighLevelDataSafe

from .low_level_backend import LowLevelBackend

from .structure_backend import StructureAll
from .structure_backend import StructureDataSafe

from .view_access_backend import ViewAccess


class TestListView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                   StructureDataSafe, ViewAccess):
    def __init__(self):
        super(TestListView, self).__init__('ListView')


class TestMatrixView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                     StructureDataSafe, ViewAccess):
    def __init__(self):
        super(TestMatrixView, self).__init__('MatrixView')


class TestSparseView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                     StructureDataSafe, ViewAccess):
    def __init__(self):
        super(TestSparseView, self).__init__('SparseView')


class TestDataFrameView(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                        StructureDataSafe, ViewAccess):
    def __init__(self):
        super(TestDataFrameView, self).__init__('DataFrameView')


class TestList(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
    def __init__(self):
        super(TestList, self).__init__('List')


class TestMatrix(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
    def __init__(self):
        super(TestMatrix, self).__init__('Matrix')


class TestSparse(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
    def __init__(self):
        super(TestSparse, self).__init__('Sparse')


class TestDataFrame(HighLevelAll, AllNumerical, QueryBackend, StructureAll):
    def __init__(self):
        super(TestDataFrame, self).__init__('DataFrame')


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
            ret = UML.data.Base((rows, cols), pointNames=pointNames, featureNames=featureNames)
            return ret

        self.constructor = makeAndDefine
