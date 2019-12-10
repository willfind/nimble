"""
Helper functions that support multiple modules.

Functions here do not import from nimble (except for Exceptions),
allowing for each function to be imported into any file within nimble
without risk of circular imports.
"""

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


class ImportModule(object):
    def __init__(self, name):
        self.name = name
        self.imported = None
        self.errorMsg = None

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
            except ImportError as e:
                self.errorMsg = str(e)

    def __getattr__(self, name):
        """
        If the attribute is a submodule, return a new ImportModule
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
            setattr(self, name, ImportModule(asSubmodule))
            return ImportModule(asSubmodule)
        except ImportError:
            pass
        self._import()
        if self.imported is None and name != '__wrapped__':
            msg = "{0} is required to be installed ".format(self.name)
            msg += "in order to complete this operation."
            if self.errorMsg:
                msg += " However, an ImportError with the following message "
                msg += "was raised: '{0}'".format(self.errorMsg)
            raise PackageException(msg)
        ret = getattr(self.imported, name)
        setattr(self, name, ret)
        return ret

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
