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

import pytest

import nimble

from .baseObject import objConstructorMaker
from .baseObject import viewConstructorMaker
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

from .data_specific_backend import SparseSpecificDataSafe
from .data_specific_backend import SparseSpecificAll
from .data_specific_backend import DataFrameSpecificAll
from .data_specific_backend import DataFrameSpecificDataSafe
from .data_specific_backend import ListSpecificAll
from .data_specific_backend import MatrixSpecificAll

listViewConstructor = viewConstructorMaker('List')
matrixViewConstructor = viewConstructorMaker('Matrix')
sparseViewConstructor = viewConstructorMaker('Sparse')
dataframeViewConstructor = viewConstructorMaker('DataFrame')
listConstructor = objConstructorMaker('List')
matrixConstructor = objConstructorMaker('Matrix')
sparseConstructor = objConstructorMaker('Sparse')
dataframeConstructor = objConstructorMaker('DataFrame')

class BaseViewChildTests(HighLevelDataSafe, NumericalDataSafe, QueryBackend,
                         StructureDataSafe, ViewAccess, StretchDataSafe,
                         HighDimensionSafe):
    pass

class BaseChildTests(HighLevelAll, AllNumerical, QueryBackend, StructureAll,
                     StretchAll, HighDimensionAll):
    pass

class TestListView(BaseViewChildTests):
    def constructor(self, *args, **kwargs):
        listViewConstructor = viewConstructorMaker('List')
        return listViewConstructor(*args, **kwargs)


class TestMatrixView(BaseViewChildTests):
    def constructor(self, *args, **kwargs):
        matrixViewConstructor = viewConstructorMaker('Matrix')
        return matrixViewConstructor(*args, **kwargs)


class TestSparseView(BaseViewChildTests, SparseSpecificDataSafe):
    def constructor(self, *args, **kwargs):
        sparseViewConstructor = viewConstructorMaker('Sparse')
        return sparseViewConstructor(*args, **kwargs)


class TestDataFrameView(BaseViewChildTests, DataFrameSpecificDataSafe):
    def constructor(self, *args, **kwargs):
        dataframeViewConstructor = viewConstructorMaker('DataFrame')
        return dataframeViewConstructor(*args, **kwargs)


class TestList(BaseChildTests, ListSpecificAll):
    def constructor(self, *args, **kwargs):
        listConstructor = objConstructorMaker('List')
        return listConstructor(*args, **kwargs)


class TestMatrix(BaseChildTests, MatrixSpecificAll):
    def constructor(self, *args, **kwargs):
        matrixConstructor = objConstructorMaker('Matrix')
        return matrixConstructor(*args, **kwargs)


class TestSparse(BaseChildTests, SparseSpecificAll):
    def constructor(self, *args, **kwargs):
        sparseConstructor = objConstructorMaker('Sparse')
        return sparseConstructor(*args, **kwargs)


class TestDataFrame(BaseChildTests, DataFrameSpecificAll):
    def constructor(self, *args, **kwargs):
        dataframeConstructor = objConstructorMaker('DataFrame')
        return dataframeConstructor(*args, **kwargs)

def concreter(name, *abclasses):
    class concreteCls(*abclasses):
        pass
    concreteCls.__abstractmethods__ = frozenset()
    return type(name, (concreteCls,), {})

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
    def _getPoints(self, names):
        return PointsDummy(self, names)
    def _getFeatures(self, names):
        return FeaturesDummy(self, names)

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

    return BaseConcreteDummy(shape, pointNames=pointNames,
                             featureNames=featureNames, name=name)

class TestBaseOnly(LowLevelBackend):
    def constructor(self, *args, **kwargs):
        return makeAndDefine(*args, **kwargs)

    def setup_method(self):
        startObjectValidation()

    def teardown_method(self):
        stopObjectValidation()
