"""
Helper functions that support multiple modules.

Functions here do not import from nimble (except for Exceptions),
allowing for each function to be imported into any file within nimble
without risk of circular imports.
"""

import inspect
import importlib
import numbers
import datetime
from types import ModuleType

import numpy as np

# nimble.exceptions may be imported
from nimble.exceptions import InvalidArgumentValue, ImproperObjectAction
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
    if not (argumentsParam or kwargsParam):
        return {}
    if not argumentsParam:
        return kwargsParam.copy()
    if not kwargsParam:
        return argumentsParam.copy()

    ret = {}
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

def allowedNumpyDType(dtype):
    """
    Check if a dtype is one allowed for nimble data.
    """
    return (dtype in [int, float, bool, object]
            or np.issubdtype(dtype, np.number)
            or np.issubdtype(dtype, np.datetime64))

def numpy2DArray(obj, dtype=None, copy=True, order='K', subok=False):
    """
    Mirror np.array() but require the data be two-dimensional.
    """
    ret = np.array(obj, dtype=dtype, copy=copy, order=order, subok=subok,
                   ndmin=2)
    if len(ret.shape) > 2:
        raise InvalidArgumentValue('obj cannot be more than two-dimensional')

    if not allowedNumpyDType(ret.dtype):
        ret = np.array(obj, dtype=np.object_, copy=copy, order=order,
                       subok=subok, ndmin=2)

    return ret

def is2DArray(arr):
    """
    Determine if a numpy ndarray object is two-dimensional.

    Since np.matrix inherits from np.ndarray, they will always
    return True.
    """
    return isinstance(arr, np.ndarray) and len(arr.shape) == 2


class DeferredModuleImport(object):
    """
    Defer import of third-party modules.

    Module must first be determined to accessible to nimble by calling
    the ``nimbleAccessible`` method. For pickling, __getstate__ and
    __setstate__ were defined due to issues loading trained learners
    trained using SparseView objects.
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

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

####################
# Optional modules #
####################

scipy = DeferredModuleImport('scipy')
pd = DeferredModuleImport('pandas')
plt = DeferredModuleImport('matplotlib.pyplot')
requests = DeferredModuleImport('requests')
cloudpickle = DeferredModuleImport('cloudpickle')
h5py = DeferredModuleImport('h5py')
dateutil = DeferredModuleImport('dateutil')

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
        if isinstance(retDType, np.flexible):
            retDType = object
        ret = np.zeros(sparseMatrix.shape, dtype=retDType)
        nonzero = (sparseMatrix.row, sparseMatrix.col)
        for (i, j), v in zip(zip(*nonzero), sparseMatrix.data):
            ret[i, j] = v
        return ret

def dtypeConvert(obj):
    """
    Most learners need numeric dtypes so attempt to convert from
    object dtype if possible, otherwise return object as-is.
    """
    if hasattr(obj, 'dtype') and obj.dtype == np.object_:
        try:
            obj = obj.astype(np.float)
        except ValueError:
            pass
    return obj

def isDatetime(x):
    """
    Determine if a value is a datetime object.
    """
    datetimeTypes = [datetime.datetime, np.datetime64]
    if pd.nimbleAccessible():
        datetimeTypes.append(pd.Timestamp)
    return isinstance(x, tuple(datetimeTypes))

def isAllowedSingleElement(x):
    """
    Determine if an element is an allowed single element.
    """
    if isinstance(x, (numbers.Number, str, np.bool_)):
        return True

    if isDatetime(x):
        return True

    if hasattr(x, '__len__'):#not a single element
        return False

    #None and np.NaN are allowed
    if x is None or x != x:
        return True

    return False

def validateAllAllowedElements(data):
    """
    Validate all values in the data are allowed types.
    """
    if isinstance(data, dict):
        data = data.values()
    if not all(map(isAllowedSingleElement, data)):
        msg = "Number, string, None, nan, and datetime objects are "
        msg += "the only elements allowed in nimble data objects"
        raise ImproperObjectAction(msg)

def pandasDataFrameToList(pdDataFrame):
    """
    Transform a pandas DataFrame into a 2D list.
    """
    return list(map(list, zip(*(col for _, col in pdDataFrame.iteritems()))))

def removeDuplicatesNative(cooObj):
    """
    Creates a new coo_matrix, using summation for numeric data to remove
    duplicates. If there are duplicate entires involving non-numeric
    data, an exception is raised.

    cooObj : the coo_matrix from which the data of the return object
    originates from. It will not be modified by the function.

    Returns : a new coo_matrix with the same data as in the input
    matrix, except with duplicate numerical entries summed to a single
    value. This operation is NOT stable - the row / col attributes are
    not guaranteed to have an ordering related to those from the input
    object. This operation is guaranteed to not introduce any 0 values
    to the data attribute.
    """
    if cooObj.data is None:
        #When cooObj data is not iterable: Empty
        #It will throw TypeError: zip argument #3 must support iteration.
        #Decided just to do this quick check instead of duck typing.
        return cooObj

    seen = {}
    duplicates = False
    zeroInData = False
    for i, j, v in zip(cooObj.row, cooObj.col, cooObj.data):
        if not zeroInData and v == 0:
            zeroInData = True
        if (i, j) not in seen:
            # all types are allowed to be present once
            seen[(i, j)] = v
        else:
            duplicates = True
            try:
                seen[(i, j)] += float(v)
            except ValueError as e:
                msg = 'Unable to represent this configuration of data in '
                msg += 'Sparse object. At least one of the duplicate entries '
                msg += 'is a non-numerical type'
                raise ValueError(msg) from e

    if not duplicates and not zeroInData:
        if not allowedNumpyDType(cooObj.data.dtype):
            cooObj.data = cooObj.data.astype(np.object_)
        return cooObj

    rows = []
    cols = []
    data = []

    for indices, value in seen.items():
        if value != 0:
            i, j = indices
            rows.append(i)
            cols.append(j)
            data.append(value)

    dataNP = np.array(data)
    # if there are mixed strings and numeric values numpy will automatically
    # turn everything into strings. This will check to see if that has
    # happened and use the object dtype instead.
    if len(dataNP) > 0 and isinstance(dataNP[0], np.flexible):
        dataNP = np.array(data, dtype='O')
    cooNew = scipy.sparse.coo_matrix((dataNP, (rows, cols)),
                                     shape=cooObj.shape)

    return cooNew

def _setAll(variables, includeModules=False, ignore=()):
    """
    Will add any attribute in the directory without a leading underscore
    to the list for __all__, except modules when includeModules is False.

    Note: Does not follow standard nimble leading underscore conventions
    because it should not be included in __all__.
    """
    inAll = []
    for name, obj in variables.items():
        if name.startswith('_') or name in ignore:
            continue
        isMod = isinstance(obj, ModuleType)
        if (isMod and includeModules) or not isMod:
            inAll.append(name)
    return sorted(inAll)
