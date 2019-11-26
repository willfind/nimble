"""
Helper functions that support multiple modules.

Functions here do not import from nimble (except for Exceptions),
allowing for each function to be imported into any file within nimble
without risk of circular imports.
"""

import inspect

import numpy

# nimble.exceptions may be imported
from nimble.exceptions import InvalidArgumentValue


def isFunction(func):
    """
    Return True if an object is a python or cython function
    """
    # inspect only identifies python functions
    if inspect.isfunction(func):
        return True
    return type(func).__name__ == 'cython_function_or_method'


def inheritDocstringsFactory(toInherit):
    """
    Factory to make decorator to copy docstrings from toInherit for
    reimplementations in the wrapped object. Only those functions
    without docstrings will be given the corresponding docstrings from
    toInherit.
    """
    def inheritDocstring(cls):
        writable = cls.__dict__
        for name in writable:
            if isFunction(writable[name]) and hasattr(toInherit, name):
                func = writable[name]
                if not func.__doc__:
                    func.__doc__ = getattr(toInherit, name).__doc__

        return cls
    return inheritDocstring

def numpy2DArray(obj, dtype=None, copy=True, order='K', subok=False):
    """
    Mirror numpy.array() but require the data be two-dimensional.
    """
    ret = numpy.array(obj, dtype=dtype, copy=copy, order=order, subok=subok,
                      ndmin=2)
    if len(ret.shape) > 2:
        raise InvalidArgumentValue('obj cannot be more than two-dimensional')
    return ret

def is2DArray(arr):
    """
    Determine if a numpy ndarray object is two-dimensional.

    Since numpy.matrix inherits from numpy.ndarray, they will always
    return True.
    """
    return isinstance(arr, numpy.ndarray) and len(arr.shape) == 2

def cooMatrixToArray(cooMatrix):
    """
    Helper for coo_matrix.toarray.

    Scipy cannot handle conversions using toarray() when the data is not
    numeric, so in that case we generate the array.
    """
    try:
        return cooMatrix.toarray()
    except ValueError:
        # flexible dtypes, such as strings, when used in scipy sparse
        # object create an implicitly mixed datatype: some values are
        # strings, but the rest are implicitly zero. In order to match
        # that, we must explicitly specify a mixed type for our destination
        # matrix
        retDType = cooMatrix.dtype
        if isinstance(retDType, numpy.flexible):
            retDType = object
        ret = numpy.zeros(cooMatrix.shape, dtype=retDType)
        nz = (cooMatrix.row, cooMatrix.col)
        for (i, j), v in zip(zip(*nz), cooMatrix.data):
            ret[i, j] = v
        return ret
