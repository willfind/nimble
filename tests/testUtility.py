"""
Tests for nimble.utility submodule
"""

from nose.tools import raises

from nimble.utility import OptionalPackage
from nimble.exceptions import PackageException

def test_OptionalPackage_numpy():
    optNumpy = OptionalPackage('numpy')

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

    numpy_arr = numpy.array(arr_data)
    assert numpy.array_equal(numpy_arr, opt_arr)
    assert type(numpy_arr) == type(opt_arr)
    assert numpy.array_equal(inverse, opt_arr)

def test_OptionalPackage_numpy_boolSuccess():
    optNumpy = OptionalPackage('numpy')
    if not optNumpy:
        assert False

@raises(AttributeError)
def test_OptionalPackage_numpy_bogusAttribute():
    optNumpy = OptionalPackage('numpy')
    optNumpy.random.bogus

@raises(PackageException)
def test_OptionalPackage_bogus():
    bogus = OptionalPackage('bogus')
    bogus.foo

def test_OptionalPackage_bogus_boolFailure():
    bogus = OptionalPackage('bogus')
    if bogus:
        assert False
