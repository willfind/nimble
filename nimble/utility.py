"""
Helper functions that support multiple modules.

Functions here do not import from nimble (except for Exceptions),
allowing for each function to be imported into any file within nimble
without risk of circular imports.
"""
from __future__ import absolute_import
import inspect
import importlib

import numpy

# nimble.exceptions may be imported
from nimble.exceptions import InvalidArgumentValue, PackageException


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
    ret = numpy.array(obj, dtype=dtype, copy=copy, order=order, subok=subok,
                      ndmin=2)
    if len(ret.shape) > 2:
        raise InvalidArgumentValue('obj cannot be more than two-dimensional')
    return ret

def is2DArray(arr):
    return isinstance(arr, numpy.ndarray) and len(arr.shape) == 2

class OptionalPackage(object):
    def __init__(self, name):
        self.name = name
        self.imported = None

    def __bool__(self):
        self._import()
        return self.imported is not None

    def _import(self):
        """
        Attempt to import package and set the imported attribute, if
        unsuccessful raise PackageException.
        """
        if self.imported is None:
            try:
                mod = importlib.import_module(self.name)
                self.imported = mod
            except ImportError:
                pass

    def __getattr__(self, name):
        """
        If the attribute is a submodule, return a new OptionalPackage
        for the submodule, otherwise return the attribute object.  If
        the module has not been imported before attempted to access this
        attribute and import fails, a PackageException will be raised,
        if the module has imported but the attribute does not exist an
        AttributeError will be raised. In all successful cases,the
        attribute is set for this object so it is immediately
        identifiable in the future.
        """
        try:
            asSubmodule = '.'.join([self.name, name])
            submod = importlib.import_module(asSubmodule)
            setattr(self, name, OptionalPackage(asSubmodule))
            return OptionalPackage(asSubmodule)
        except ImportError:
            pass
        self._import()
        if self.imported is None and name != '__wrapped__':
            msg = 'This operation requires the {0} package '.format(self.name)
            msg += 'to be installed'
            raise PackageException(msg)
        ret = getattr(self.imported, name)
        setattr(self, name, ret)
        return ret
