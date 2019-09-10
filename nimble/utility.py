"""
Helper functions that support multiple modules.

Functions here do not import from nimble (except for Exceptions),
allowing for each function to be imported into any file within nimble
without risk of circular imports.
"""
from __future__ import absolute_import
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

# class Array2D(numpy.ndarray):
#     def __init__(self, *args, **kwargs):
#         super(self, Array2D).__init__(*args, **kwargs)
#         if len(self.shape) == 1:
#             self.reshape((1, -1))
#         elif len(self.shape) > 2:
#             raise InvalidArgumentValue('data cannot exceed two-dimensions')

def numpy2DArray(obj, dtype=None, copy=True, order='K', subok=False):
    ret = numpy.array(obj, dtype=dtype, copy=copy, order=order, subok=subok,
                      ndmin=2)
    if len(ret.shape) > 2:
        raise InvalidArgumentValue('obj cannot be more than two-dimensional')
    return ret

def is2DArray(arr):
    return isinstance(arr, numpy.ndarray) and len(arr.shape) == 2
