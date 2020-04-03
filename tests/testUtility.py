"""
Tests for nimble.utility submodule
"""

from nose.tools import raises

from nimble.utility import DeferredModuleImport

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

    numpy_arr = numpy.array(arr_data)
    assert numpy.array_equal(numpy_arr, opt_arr)
    assert type(numpy_arr) == type(opt_arr)
    assert numpy.array_equal(inverse, opt_arr)

def test_DeferredModuleImport_numpy_nimbleAccessibleSuccess():
    optNumpy = DeferredModuleImport('numpy')
    if not optNumpy.nimbleAccessible():
        assert False

@raises(AttributeError)
def test_DeferredModuleImport_numpy_noAccessibleCheck():
    numpy = DeferredModuleImport('numpy')
    # random is a valid numpy module but will get an AttributeError since
    # we never verified numpy is accessible using numpy.nimbleAccessible()
    numpy.random

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
