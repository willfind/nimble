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

from .data_specific_backend import SparseSpecific

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


class TestSparse(BaseChildTests, SparseSpecific):
    def __init__(self):
        super(TestSparse, self).__init__('Sparse')


class TestDataFrame(BaseChildTests):
    def __init__(self):
        super(TestDataFrame, self).__init__('DataFrame')

def concreter(name, *abclasses):
    class concreteCls(*abclasses):
        pass
    concreteCls.__abstractmethods__ = frozenset()
    return type(name, (concreteCls,), {})

def makeAndDefine(shape=None, pointNames=None, featureNames=None,
                  psize=0, fsize=0, name=None):
    """
    Make a Base data object with no actual data but has a shape
    and can have pointNames and featureNames.
    """
    rows = psize if pointNames is None else len(pointNames)
    cols = fsize if featureNames is None else len(featureNames)
    if shape is None:
        shape = [rows, cols]
    # Base, Points, and Features are abstract so they cannot be instantiated,
    # however we can create Dummy concrete classes to test the base
    # functionality.
    BaseDummy = concreter('BaseDummy', nimble.core.data.Base)
    PointsDummy = concreter('PointsDummy', nimble.core.data.Axis,
                            nimble.core.data.Points)
    FeaturesDummy = concreter('FeaturesDummy', nimble.core.data.Axis,
                              nimble.core.data.Features)
    # need to override _getPoints and _getFeatures to use dummies
    class BaseConcreteDummy(BaseDummy):
        def _getPoints(self):
            return PointsDummy(self)
        def _getFeatures(self):
            return FeaturesDummy(self)

    return BaseConcreteDummy(shape, pointNames=pointNames,
                             featureNames=featureNames, name=name)

class TestBaseOnly(LowLevelBackend):
    def __init__(self):
        self.constructor = makeAndDefine

    def setUp(self):
        startObjectValidation()

    def tearDown(self):
        stopObjectValidation()
