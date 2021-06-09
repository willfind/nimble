"""
Define data test objects for methods of properties specific to the type

"""
import numpy as np

import nimble
from nimble.random import pythonRandom
from tests.helpers import assertCalled, assertNotCalled
from .baseObject import DataTestObject


def back_sparseScalarZeroPreserving(constructor, nimbleOp):
    data = [[1, 2, 3], [0, 0, 0]]
    toTest = constructor(data)
    rint = pythonRandom.randint(2, 5)
    try:
        with assertCalled(nimble.core.data.Sparse,
                          '_scalarZeroPreservingBinary_implementation'):
            ret = getattr(toTest, nimbleOp)(rint)
    except ZeroDivisionError:
        assert nimbleOp.startswith('__r')

@assertNotCalled(nimble.core.data.Sparse, '_defaultBinaryOperations_implementation')
@assertNotCalled(nimble.core.data.Sparse, '_scalarZeroPreservingBinary_implementation')
def back_sparseScalarOfOne(constructor, nimbleOp):
    """Test Sparse does not call helper functions for these scalar ops """
    data = [[1, 2, 3], [0, 0, 0]]
    toTest = constructor(data)
    rint = 1.0
    # Sparse should use a copy of toTest when exponent is 1, so neither
    # helper function should be called
    ret = getattr(toTest, nimbleOp)(rint)

class SparseSpecificDataSafe(DataTestObject):

    def test_mul_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__mul__')

    def test_mul_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__mul__')

    def test_rmul_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__rmul__')

    def test_rmul_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__rmul__')

    def test_truediv_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__truediv__')

    def test_truediv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__truediv__')

    def test_rtruediv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__rtruediv__')

    def test_floordiv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__floordiv__')

    def test_rfloordiv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__rfloordiv__')

    def test_mod_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__mod__')

    def test_rmod_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__rmod__')

    def test_pow_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__pow__')

    def test_pow_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__pow__')


class SparseSpecificDataModifying(DataTestObject):

    def test_sortInternal_avoidsUnecessary(self):
        data = [[1, 0, 3], [0, 5, 0]]
        obj = self.constructor(data)

        # ensure that our patched target is used
        with assertCalled(np, "lexsort"):
            obj._sortInternal('point')

        obj._sortInternal('point')
        assert obj._sorted['axis'] == 'point'
        assert obj._sorted['indices'] is None

        # call _sortInternal to generate indices on already sorted obj
        with assertNotCalled(np, "lexsort"):
            obj._sortInternal('point', setIndices=True)

        # Confirm the desired action actually took place
        assert obj._sorted['indices'] is not None


    def test_imul_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__imul__')

    def test_imul_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__imul__')

    def test_itruediv_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__itruediv__')

    def test_itruediv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__itruediv__')

    def test_ifloordiv_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__ifloordiv__')

    def test_imod_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__imod__')

    def test_ipow_Sparse_calls_scalarZeroPreservingBinary(self):
        back_sparseScalarZeroPreserving(self.constructor, '__ipow__')

    def test_ipow_Sparse_scalarOfOne(self):
        back_sparseScalarOfOne(self.constructor, '__ipow__')

class SparseSpecificAll(SparseSpecificDataSafe, SparseSpecificDataModifying):
    """Tests for Sparse implementation details """


class DataFrameSpecificDataSafe(DataTestObject):

    def test_dtypes_binaryOperators(self):
        data = [[1, 2, 3.], [-1, -2, -3.], [9, 8, 7.]]
        obj = self.constructor(data)
        startDtypes = tuple(obj._data.dtypes)

        assert startDtypes == (np.dtype(int), np.dtype(int),
                               np.dtype(float))

        # truediv and pow will always convert to floats
        floatDtypes = (np.dtype(float),) * 3

        ret = obj + 2
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == startDtypes

        ret = obj - 2
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == startDtypes

        ret = obj * 2
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == startDtypes

        ret = obj // 2
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == startDtypes

        ret = obj / 2
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == floatDtypes

        ret = obj ** 2
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == floatDtypes

        ret = obj ** -1
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == floatDtypes

        ret = obj + obj
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == startDtypes

        ret = obj - obj
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == startDtypes

        ret = obj * obj
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == startDtypes

        ret = obj // obj
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == startDtypes

        ret = obj / obj
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == floatDtypes

        ret = obj ** obj
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == floatDtypes

        multObj = obj.copy()
        # square multipication can be misleading, use a different shape
        multObj.features.append(obj.copy())
        ret = obj.matrixMultiply(multObj)
        assert tuple(obj._data.dtypes) == startDtypes
        assert tuple(ret._data.dtypes) == floatDtypes * 2

    def test_dtypes_copy(self):
        data = [[1, 2, 3.], [-1, -2, -3.], [9, 8, 7.]]
        obj = self.constructor(data)
        startDtypes = tuple(obj._data.dtypes)
        copy = obj.copy()

        assert tuple(copy._data.dtypes) == startDtypes

class DataFrameSpecificDataModifying(DataTestObject):

    def test_dtypes_binaryOperators_inplace(self):
        data = [[1, 2, 3.], [-1, -2, -3.], [9, 8, 7.]]
        obj = self.constructor(data)
        startDtypes = tuple(obj._data.dtypes)

        assert startDtypes == (np.dtype(int), np.dtype(int),
                               np.dtype(float))

        # truediv and pow will always convert to floats
        floatDtypes = (np.dtype(float),) * 3

        toTest = obj.copy()
        toTest += 2
        assert tuple(toTest._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest -= 2
        assert tuple(toTest._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest *= 2
        assert tuple(toTest._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest //= 2
        assert tuple(toTest._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest /= 2
        assert tuple(toTest._data.dtypes) == floatDtypes

        toTest = obj.copy()
        toTest **= 2
        assert tuple(toTest._data.dtypes) == floatDtypes

        toTest = obj.copy()
        toTest **= -1
        assert tuple(toTest._data.dtypes) == floatDtypes

        toTest = obj.copy()
        toTest += toTest
        assert tuple(toTest._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest -= toTest
        assert tuple(toTest._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest *= toTest
        assert tuple(toTest._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest //= toTest
        assert tuple(obj._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest /= toTest
        assert tuple(toTest._data.dtypes) == floatDtypes

        toTest = obj.copy()
        toTest **= toTest
        assert tuple(toTest._data.dtypes) == floatDtypes

    def test_dtypes_replaceRectangle(self):
        data = [[1, 2, 3.], [-1, -2, -3.], [9, 8, 7.]]
        obj = self.constructor(data)
        startDtypes = tuple(obj._data.dtypes)
        replace = self.constructor([[22, 33], [-22, -33]])

        obj.replaceRectangle(replace, 1, 1, 2, 2)
        assert tuple(obj._data.dtypes) == startDtypes

        obj = self.constructor(data)
        replace = self.constructor([[11., 22], [-11., -22]])

        obj.replaceRectangle(replace, 0, 0, 1, 1)
        expDtypes = (np.dtype(float), np.dtype(int), np.dtype(float))
        assert tuple(obj._data.dtypes) == expDtypes

    def test_dtypes_flattenUnflatten(self):
        data = [[1, 2, 3.], [-1, -2, -3.], [9, 8, 7.]]
        obj = self.constructor(data)
        startDtypes = tuple(obj._data.dtypes)
        obj.flatten('feature')

        expDtypes = tuple(startDtypes[0:2]) * 3 + (startDtypes[2],) * 3
        assert tuple(obj._data.dtypes) == expDtypes

        obj.unflatten((3, 3), 'feature')
        assert tuple(obj._data.dtypes) == startDtypes

        obj.flatten('point')
        assert tuple(obj._data.dtypes) == startDtypes * 3

        obj.unflatten((3, 3), 'point')
        assert tuple(obj._data.dtypes) == startDtypes


    def test_dtypes_structural(self):
        data = [[1, 2, 3., 4.], [-1, -2, -3., -4.],
                [9, 8, 7., 6.], [-9, -8, -7., -6.]]
        obj = self.constructor(data)
        startDtypes = tuple(obj._data.dtypes)
        ftExpDtypes = (np.dtype(int), np.dtype(float))

        ptCp = obj.points.copy([1, 2])
        assert tuple(ptCp._data.dtypes) == startDtypes

        ftCp = obj.features.copy([1, 2])
        assert tuple(ftCp._data.dtypes) == ftExpDtypes

        toTest = obj.copy()
        ptEx = toTest.points.extract([1, 3])
        assert tuple(toTest._data.dtypes) == startDtypes
        assert tuple(ptEx._data.dtypes) == startDtypes

        toTest = obj.copy()

        ftEx = toTest.features.extract([1, 3])
        assert tuple(toTest._data.dtypes) == ftExpDtypes
        assert tuple(ftEx._data.dtypes) == ftExpDtypes

        toTest = obj.copy()
        toTest.points.retain([1, 3])
        assert tuple(toTest._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest.features.retain([1, 3])
        assert tuple(toTest._data.dtypes) == ftExpDtypes

        toTest = obj.copy()
        toTest.points.delete([1, 3])
        assert tuple(toTest._data.dtypes) == startDtypes

        toTest = obj.copy()
        toTest.features.delete([1, 3])
        assert tuple(toTest._data.dtypes) == ftExpDtypes

    def test_dtypes_repeat(self):
        data = [[1, 2, 3.], [-1, -2, -3.], [9, 8, 7.]]
        obj = self.constructor(data)
        startDtypes = tuple(obj._data.dtypes)

        rep = obj.points.repeat(2, False)
        assert tuple(rep._data.dtypes) == startDtypes

        rep = obj.points.repeat(2, True)
        assert tuple(rep._data.dtypes) == startDtypes

        rep = obj.features.repeat(2, False)
        assert tuple(rep._data.dtypes) == startDtypes * 2

        rep = obj.features.repeat(2, True)
        expDtypes = (np.dtype(int),) * 4 + (np.dtype(float),) * 2
        assert tuple(rep._data.dtypes) == expDtypes

    def test_dtypes_splitByCollapsingFeatures(self):
        data = [[0,0,1,2,3,4], [1,1,5,6,7,8], [2,2,-1,-2,-3,-4]]
        ptNames = ["0", "1", "2"]
        ftNames = ["ret0", "ret1", "coll0", "coll1", "coll2", "coll3"]
        toTest = self.constructor(data, pointNames=ptNames,
                                  featureNames=ftNames)

        toCollapse = ["coll0", "coll1", "coll2", "coll3"]
        toTest.points.splitByCollapsingFeatures(toCollapse, "ftNames",
                                                "ftValues")
        expDtypes = (np.dtype(int), np.dtype(int), np.dtype(object),
                     np.dtype(int))
        assert tuple(toTest._data.dtypes) == expDtypes

    def test_dtypes_combineByExpandingFeatures(self):
        data = [["p1", 100, 'r1', 9.5], ["p1", 100, 'r2', 9.9],
                ["p2", 100, 'r1', 6.5], ["p2", 100, 'r2', 6.0],
                ["p3", 100, 'r1', 11], ["p3", 100, 'r2', 11.2],
                ["p1", 200, 'r1', 18.1], ["p1", 200, 'r2', 20.1]]
        pNames = [str(i) for i in range(8)]
        fNames = ['type', 'dist', 'run', 'time']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expDtypes = (np.dtype(object), np.dtype(int), np.dtype(float),
                     np.dtype(float))
        toTest.points.combineByExpandingFeatures('run', 'time')
        assert tuple(toTest._data.dtypes) == expDtypes

    def test_dtypes_splitByParsing(self):
        data = [[0, "a1", 0.], [1, "b2", 1.], [2, "c3", 2.]]
        pNames = ["0", "1", "2"]
        fNames = ["f0", "merged", "f1"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        toTest.features.splitByParsing(1, 1, ["split0", "split1"])
        expDtypes = (np.dtype(int), np.dtype(object),
                     np.dtype(object), np.dtype(float))
        assert tuple(toTest._data.dtypes) == expDtypes

class DataFrameSpecificAll(DataFrameSpecificDataSafe,
                           DataFrameSpecificDataModifying):
    """Tests for DataFrame implementation details """
