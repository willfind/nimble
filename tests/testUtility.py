"""
Tests for nimble._utility submodule
"""

from nose.tools import raises
import numpy as np

from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble._utility import DeferredModuleImport
from nimble._utility import mergeArguments
from nimble._utility import inspectArguments
from nimble._utility import numpy2DArray, is2DArray
from nimble._utility import _setAll

def test_DeferredModuleImport_numpy():
    optNumpy = DeferredModuleImport('numpy')
    assert optNumpy.nimbleAccessible() # performs the import
    # use some top level numpy functions
    arr_data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    opt_arr = optNumpy.array(arr_data)
    concat = optNumpy.concatenate([opt_arr, opt_arr])

    # use some numpy submodule functions
    randInt = optNumpy.random.randint(1, 10)
    determinant = optNumpy.linalg.det(opt_arr)
    inverse = optNumpy.linalg.inv(opt_arr)

    assert isinstance(randInt, int)
    assert 1 <= randInt < 10
    assert determinant == 1

    import numpy
    import numpy as np
    assert numpy == np
    assert numpy != optNumpy

    numpy_arr = np.array(arr_data)
    assert np.array_equal(numpy_arr, opt_arr)
    assert type(numpy_arr) == type(opt_arr)
    assert np.array_equal(inverse, opt_arr)


def test_DeferredModuleImport_numpy_nimbleAccessibleSuccess():
    optNumpy = DeferredModuleImport('numpy')
    if not optNumpy.nimbleAccessible():
        assert False


@raises(AttributeError)
def test_DeferredModuleImport_numpy_noAccessibleCheck():
    np = DeferredModuleImport('numpy')
    # random is a valid numpy module but will get an AttributeError since
    # we never verified numpy is accessible using np.nimbleAccessible()
    np.random


@raises(AttributeError)
def test_DeferredModuleImport_numpy_bogusAttribute():
    optNumpy = DeferredModuleImport('numpy')
    assert optNumpy.nimbleAccessible() # performs the import
    assert hasattr(optNumpy, 'random')
    optNumpy.random.bogus


def test_DeferredModuleImport_bogus_nimbleAccessibleFailure():
    bogus = DeferredModuleImport('bogus')
    if bogus.nimbleAccessible():
        assert False


@raises(InvalidArgumentValueCombination)
def testMergeArgumentsException():
    """ Test utility.mergeArguments will throw the exception it should """
    args = {1: 'a', 2: 'b', 3: 'd'}
    kwargs = {1: 1, 2: 'b'}

    mergeArguments(args, kwargs)


def testMergeArgumentsHand():
    """ Test utility.mergeArguments is correct on hand construsted data """
    args = {1: 'a', 2: 'b', 3: 'd'}
    kwargs = {1: 'a', 4: 'b'}

    ret = mergeArguments(args, kwargs)

    assert ret == {1: 'a', 2: 'b', 3: 'd', 4: 'b'}


def testMergeArgumentsMakesCopy():
    """Test utility.mergeArguments does not modify or return either original dict"""
    emptyA = {}
    emptyB = {}
    dictA = {'foo': 'bar'}
    dictB = {'bar': 'foo'}

    merged0 = mergeArguments(None, emptyB)
    assert emptyB == merged0
    assert id(emptyB) != id(merged0)
    assert emptyB == {}

    merged1 = mergeArguments(emptyA, emptyB)
    assert emptyA == merged1
    assert emptyB == merged1
    assert id(emptyA) != id(merged1)
    assert id(emptyB) != id(merged1)
    assert emptyA == {}
    assert emptyB == {}

    merged2 = mergeArguments(dictA, emptyB)
    assert dictA == merged2
    assert id(dictA) != id(merged2)
    assert dictA == {'foo': 'bar'}
    assert emptyB == {}

    merged3 = mergeArguments(emptyA, dictB)
    assert dictB == merged3
    assert id(dictB) != id(merged3)
    assert emptyA == {}
    assert dictB == {'bar': 'foo'}

    merged4 = mergeArguments(dictA, dictB)
    assert dictA != merged4
    assert dictB != merged4
    assert dictA == {'foo': 'bar'}
    assert dictB == {'bar': 'foo'}


def test_inspectArguments():

    def checkSignature(a, b, c, d=False, e=True, f=None, *sigArgs, **sigKwargs):
        pass

    a, v, k, d = inspectArguments(checkSignature)

    assert a == ['a', 'b', 'c', 'd', 'e', 'f']
    assert v == 'sigArgs'
    assert k == 'sigKwargs'
    assert d == (False, True, None)

def test_numpy2DArray_converts1D():
    raw = [1, 2, 3, 4]
    ret = numpy2DArray(raw)
    fromNumpy = np.array(raw)
    assert len(ret.shape) == 2
    assert not np.array_equal(ret,fromNumpy)

@raises(InvalidArgumentValue)
def test_numpy2DArray_dimensionException():
    raw = [[[1, 2], [3, 4]]]
    ret = numpy2DArray(raw)

def test_is2DArray():
    raw1D = [1, 2, 3]
    arr1D = np.array(raw1D)
    mat1D = np.matrix(raw1D)
    assert not is2DArray(arr1D)
    assert is2DArray(mat1D)
    raw2D = [[1, 2, 3]]
    arr2D = np.array(raw2D)
    mat2D = np.matrix(raw2D)
    assert is2DArray(arr2D)
    assert is2DArray(mat2D)
    raw3D = [[[1, 2, 3]]]
    arr3D = np.array(raw3D)
    assert not is2DArray(arr3D)

def test_setAll():
    import sys as _sys # always ignored module
    from os import path # module
    from re import search, match, findall, compile # functions

    _variable = '_variable' # always ignored local variable
    variable = 'variable'

    variables = vars().copy()

    all1 = _setAll(variables)
    assert all1 == ['compile', 'findall', 'match', 'search', 'variable']

    all2 = _setAll(variables, includeModules=True)
    assert all2 == ['compile', 'findall', 'match', 'path', 'search',
                    'variable']

    all3 = _setAll(variables, ignore=['match'])
    assert all3 == ['compile', 'findall', 'search', 'variable']

    # ignore takes precedence over includeModules
    all4 = _setAll(variables, ignore=['variable', 'path'],
                   includeModules=True)
    assert all4 == ['compile', 'findall', 'match', 'search']
