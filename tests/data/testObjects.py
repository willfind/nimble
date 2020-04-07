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

import nimble

from .baseObject import startObjectValidation
from .baseObject import stopObjectValidation

from .numerical_backend import AllNumerical
from .numerical_backend import NumericalDataSafe

from .query_backend import QueryBackend

from .high_level_backend import HighLevelAll
from .high_level_backend import HighLevelDataSafe

from .low_level_backend import LowLevelBackend

from .structure_backend import StructureAll
from .structure_backend import StructureDataSafe

from .view_access_backend import ViewAccess

from .stretch_backend import StretchAll
from .stretch_backend import StretchDataSafe

from .high_dimension_backend import HighDimensionAll
from .high_dimension_backend import HighDimensionSafe

class BaseViewChildTests(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                   StructureDataSafe, ViewAccess, StretchDataSafe,
                   HighDimensionSafe):
    def __init__(self, returnType):
        super(BaseViewChildTests, self).__init__(returnType)

class BaseChildTests(HighLevelAll, AllNumerical, QueryBackend, StructureAll,
                 StretchAll, HighDimensionAll):
    def __init__(self, returnType):
        super(BaseChildTests, self).__init__(returnType)

class TestListView(BaseViewChildTests):
    def __init__(self):
        super(TestListView, self).__init__('ListView')


class TestMatrixView(BaseViewChildTests):
    def __init__(self):
        super(TestMatrixView, self).__init__('MatrixView')


class TestSparseView(BaseViewChildTests):
    def __init__(self):
        super(TestSparseView, self).__init__('SparseView')


class TestDataFrameView(BaseViewChildTests):
    def __init__(self):
        super(TestDataFrameView, self).__init__('DataFrameView')


class TestList(BaseChildTests):
    def __init__(self):
        super(TestList, self).__init__('List')


class TestMatrix(BaseChildTests):
    def __init__(self):
        super(TestMatrix, self).__init__('Matrix')


class TestSparse(BaseChildTests):
    def __init__(self):
        super(TestSparse, self).__init__('Sparse')


class TestDataFrame(BaseChildTests):
    def __init__(self):
        super(TestDataFrame, self).__init__('DataFrame')


class TestBaseOnly(LowLevelBackend):
    def __init__(self):

        def makeAndDefine(pointNames=None, featureNames=None, psize=0, fsize=0,
                          shape=None):
            """ Make a base data object that will think it has as many features as it has featureNames,
            even though it has no actual data """
            rows = psize if pointNames is None else len(pointNames)
            cols = fsize if featureNames is None else len(featureNames)
            if shape is None:
                shape = [rows, cols]
            ret = nimble.data.Base(shape, pointNames=pointNames, featureNames=featureNames)
            return ret

        self.constructor = makeAndDefine

    def setUp(self):
        startObjectValidation()

    def tearDown(self):
        stopObjectValidation()
