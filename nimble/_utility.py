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
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination


def isFunction(func):
    """
    Return True if an object is a python or cython function
    """
    # inspect only identifies python functions
    if inspect.isfunction(func):
        return True
    return type(func).__name__ == 'cython_function_or_method'


def inspectArguments(func):
    """
    To be used in place of inspect.getargspec for Python3 compatibility.
    Return is the tuple (args, varargs, keywords, defaults)
    """
    try:
        # in py>=3.5 inspect.signature can extract the original signature
        # of wrapped functions
        sig = inspect.signature(func)
        a = []
        if inspect.isclass(func) or hasattr(func, '__self__'):
            # self included already for cython function signature
            if 'cython' not in str(type(func)):
                # add self to classes and bounded methods to align
                # with output of getfullargspec
                a.append('self')
        v = None
        k = None
        d = []
        for param in sig.parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD:
                a.append(param.name)
                if param.default is not param.empty:
                    d.append(param.default)
            elif param.kind == param.VAR_POSITIONAL:
                v = param.name
            elif param.kind == param.VAR_KEYWORD:
                k = param.name
        d = tuple(d)
        argspec = tuple([a, v, k, d])
    except AttributeError:
        argspec = inspect.getfullargspec(func)[:4] # py>=3

    return argspec


def mergeArguments(argumentsParam, kwargsParam):
    """
    Takes two dicts and returns a new dict of them merged together. Will
    throw an exception if the two inputs have contradictory values for
    the same key.
    """
    ret = {}
    if argumentsParam is None:
        argumentsParam = {}
    # UniversalInterface uses this helper to merge params a little differently,
    # arguments (which might be None) is passed as kwargsParam so in that case
    # we need to set kwargsParam to {}.
    if kwargsParam is None:
        kwargsParam = {}
    if len(argumentsParam) < len(kwargsParam):
        smaller = argumentsParam
        larger = kwargsParam
    else:
        smaller = kwargsParam
        larger = argumentsParam

    for k in larger:
        ret[k] = larger[k]
    for k in smaller:
        val = smaller[k]
        if k in ret and ret[k] != val:
            msg = "The two dicts disagree. key= " + str(k)
            msg += " | arguments value= " + str(argumentsParam[k])
            msg += " | **kwargs value= " + str(kwargsParam[k])
            raise InvalidArgumentValueCombination(msg)
        ret[k] = val

    return ret


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

class DeferredModuleImport(object):
    """
    Defer import of third-party modules.

    Module must first be determined to accessible to nimble by calling
    the ``nimbleAccessible`` method.
    """
    def __init__(self, name):
        self.name = name
        self.imported = None

    def nimbleAccessible(self):
        """
        Determine if nimble can successfully import the module.
        """
        if self.imported is None:
            try:
                mod = importlib.import_module(self.name)
                self.imported = mod
            except ImportError:
                pass
        return self.imported is not None

    def __getattr__(self, name):
        """
        If the attribute is a submodule, return the submodule, otherwise
        return the attribute object.  If the module has not been
        imported before attemptimg to access this attribute an
        AttributeError will be raised explaining that the accessibility
        of the module has not been determined. In all successful cases,
        the attribute is set for this object so it is immediately
        identifiable in the future.
        """
        if not self.imported:
            msg = "Cannot access attributes for {mod} because the "
            msg += "accessibility of the module has not been determined. "
            msg += "A call must be made to {mod}.nimbleAccessible() first "
            msg += "to determine if nimble is able to import {mod}."
            raise AttributeError(msg.format(mod=self.name))
        try:
            asSubmodule = '.'.join([self.name, name])
            submod = importlib.import_module(asSubmodule)
            setattr(self, name, submod)
            return submod
        except ImportError:
            pass
        ret = getattr(self.imported, name)
        setattr(self, name, ret)
        return ret

####################
# Optional modules #
####################

scipy = DeferredModuleImport('scipy')
pd = DeferredModuleImport('pandas')
matplotlib = DeferredModuleImport('matplotlib')
plt = DeferredModuleImport('matplotlib.pyplot')
requests = DeferredModuleImport('requests')
cloudpickle = DeferredModuleImport('cloudpickle')
h5py = DeferredModuleImport('h5py')

def sparseMatrixToArray(sparseMatrix):
    """
    Helper for coo_matrix.toarray.

    Scipy cannot handle conversions using toarray() when the data is not
    numeric, so in that case we generate the array.
    """
    try:
        return sparseMatrix.toarray()
    except ValueError:
        # flexible dtypes, such as strings, when used in scipy sparse
        # object create an implicitly mixed datatype: some values are
        # strings, but the rest are implicitly zero. In order to match
        # that, we must explicitly specify a mixed type for our destination
        # matrix
        if (scipy.nimbleAccessible()
                and not scipy.sparse.isspmatrix_coo(sparseMatrix)):
            sparseMatrix = sparseMatrix.tocoo()
        retDType = sparseMatrix.dtype
        if isinstance(retDType, numpy.flexible):
            retDType = object
        ret = numpy.zeros(sparseMatrix.shape, dtype=retDType)
        nz = (sparseMatrix.row, sparseMatrix.col)
        for (i, j), v in zip(zip(*nz), sparseMatrix.data):
            ret[i, j] = v
        return ret

def dtypeConvert(obj):
    """
    Most learners need numeric dtypes so attempt to convert from
    object dtype if possible, otherwise return object as-is.
    """
    if hasattr(obj, 'dtype') and obj.dtype == numpy.object_:
        try:
            obj = obj.astype(numpy.float)
        except ValueError:
            pass
    return obj
