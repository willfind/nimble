
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Helper functions that support multiple modules.

Other than exceptions and _dependencies (because they do contain any
nimble imports), this file should not import from nimble to avoid any
risk of circular imports.
"""

import inspect
import importlib
import numbers
import datetime
from types import ModuleType
from copy import deepcopy
import math

import numpy as np

# only allowed imports from nimble are exceptions and _dependencies
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble._dependencies import checkVersion
import nimble

acceptedStats = ['max', 'mean', 'median', 'min', 'unique count',
            'proportion missing', 'proportion zero', 'standard deviation',
            'std', 'population std', 'population standard deviation',
            'sample std', 'sample standard deviation', 'count', 'mode', 
            'sum', 'variance','median absolute deviation', 'quartiles']

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

    There is a risk that a large string in an object will lead numpy to
    allocate much more memory than necessary. So, if the object is not
    already a numpy array, we need to load as object dtype first then
    determine if a numeric dtype can be applied.
    """
    if dtype is not None:
        dtype = np.dtype(dtype)
        if not allowedNumpyDType(dtype):
            raise ValueError("only numeric dtypes or object dtype are allowed")
        ret = np.array(obj, dtype=dtype, copy=copy, order=order, subok=subok,
                       ndmin=2)
    elif isinstance(obj, np.ndarray):
        if not allowedNumpyDType(obj.dtype):
            dtype = np.object_
        ret = np.array(obj, dtype=dtype, copy=copy, order=order, subok=subok,
                       ndmin=2)
    else:
        ret = np.array(obj, dtype=np.object_, copy=copy, order=order,
                       subok=subok, ndmin=2)
        # check the unique element types to determine dtype
        typed = np.vectorize(type, otypes=[object])(ret.ravel())
        unique = set(typed)
        if (all(issubclass(u, numbers.Number) for u in unique) and
                (len(unique) == 1 or
                 # stay with object dtype to avoid bools being converted
                 not any(issubclass(u, (bool, np.bool_)) for u in unique))):
            ret = np.array(obj, dtype=None, copy=copy, order=order,
                           subok=subok, ndmin=2)

    if len(ret.shape) > 2:
        raise InvalidArgumentValue('obj cannot be more than two-dimensional')

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
    def __init__(self, name, validate=True):
        self.name = name
        self.imported = None
        self.validated = not validate

    def nimbleAccessible(self):
        """
        Determine if nimble can successfully import the module.
        """
        if self.imported is None:
            try:
                mod = importlib.import_module(self.name)
                self.imported = mod
            except ImportError:
                return False
        if not self.validated:
            # may import submodules but need to check version of base module
            if '.' in self.name:
                base = self.name.split('.', 1)[0]
                checkVersion(importlib.import_module(base))
            else:
                checkVersion(self.imported)
            self.validated = True

        return True

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
        ret = getattr(self.imported, name)
        if isinstance(ret, ModuleType):
            asSubmodule = '.'.join([self.name, name])
            try:
                submod = importlib.import_module(asSubmodule)
                ret = submod
            except ImportError:
                pass
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
hyperopt = DeferredModuleImport('hyperopt')
storm_tuner = DeferredModuleImport('storm_tuner')
IPython = DeferredModuleImport('IPython', False)

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
            obj = obj.astype(np.float64)
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
        raise InvalidArgumentValue(msg)

def pandasDataFrameToList(pdDataFrame):
    """
    Transform a pandas DataFrame into a 2D list.
    """
    return list(map(list, zip(*(col for _, col in pdDataFrame.items()))))

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

def _prettyString(obj, useAnd, numberItems, iterator, formatter):
    ret = ""
    length = len(obj)
    for i, item in enumerate(iterator(obj)):
        if i > 0:
            if length > 2 or not useAnd:
                ret += ','
            ret += ' '
            if useAnd and i == length - 1:
                ret += 'and '
        if numberItems:
            ret += '(' + str(i) + ') '
        ret += formatter(item)

    return ret

def prettyListString(inList, useAnd=False, numberItems=False, itemStr=str):
    """
    Display lists in a more appealing way than default.
    """
    return _prettyString(inList, useAnd, numberItems, iter, itemStr)

def prettyDictString(inDict, useAnd=False, numberItems=False, keyStr=str,
                     delimiter='=', valueStr=str):
    """
    Display dicts in a more appealing way than default.
    """
    def itemFormatter(item):
        key, value = item
        return f'{keyStr(key)}{delimiter}{valueStr(value)}'

    return _prettyString(inDict, useAnd, numberItems,
                         iterator=lambda d: d.items(), formatter=itemFormatter)

def quoteStrings(val):
    """
    Wrap string values in quotes.

    A helper function convenient for the formatting (*Str) parameters in
    the pretty*String functions.
    """
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)

def tableString(table, rowHeader=True, colHeaders=None, roundDigits=None,
                columnSeparator="  ", maxRowsToShow=None, snipIndex=None,
                includeTrailingNewLine=True,  rowHeadJustify='right',
                colHeadJustify='center', colValueJustify='center'):
    """
    Take a table (rows and columns of strings) and return a string
    representing a nice visual representation of that table. roundDigits
    is the number of digits to round floats to.
    """
    if len(table) == 0:
        return ""

    justifiers = {'center': lambda s, w: s.center(w),
                  'left': lambda s, w: s.ljust(w),
                  'right': lambda s, w: s.rjust(w)}
    for justify in [rowHeadJustify, colHeadJustify, colValueJustify]:
        if justify not in justifiers:
            msg = 'justifier values must be be "center", "left", or "right"'
            raise ValueError(msg)

    if isinstance(roundDigits, int):
        roundDigits = "." + str(roundDigits) + "f"

    # So that the table doesn't get destroyed in the process!
    table = deepcopy(table)
    colWidths = []
    #rows = len(table)
    cols = 0
    for row in table:
        if len(row) > cols:
            cols = len(row)

    for _ in range(cols):
        colWidths.append(1)

    for i, row in enumerate(table):
        for j, val in enumerate(row):
            if roundDigits is not None and isinstance(val, float):
                toString = format(val, roundDigits)
            else:
                toString = str(val)
            table[i][j] = toString
            if len(toString) > colWidths[j]:
                colWidths[j] = len(toString)

    if colHeaders is not None:
        for j, header in enumerate(colHeaders):
            if colWidths[j] < len(header):
                colWidths[j] = len(header)
        table.insert(0, colHeaders)
        colHeader = True
    else:
        colHeader = False

    # if there is a limit to how many rows we can show, delete the middle rows
    # and replace them with a "..." row
    if maxRowsToShow is not None:
        numToDelete = max(len(table) - maxRowsToShow, 0)
        if numToDelete > 0:
            firstToDelete = int(math.ceil((len(table) / 2.0)
                                          - (numToDelete / 2.0)))
            lastToDelete = firstToDelete + numToDelete - 1
            table = table[:firstToDelete]
            table += [["..."] * len(table[firstToDelete])]
            table += table[lastToDelete + 1:]
    # if we want to imply the existence of more rows, but they're not currently
    # present in the table, we just add an elipses at the specified index
    elif snipIndex is not None and snipIndex > 0:
        table = table[:snipIndex]
        table += [["..."] * len(table[0])]
        table += table[snipIndex + 1:]

    #modify the text in each column to give it the right length
    for i, row in enumerate(table):
        for j, val in enumerate(row):
            if (i > 0 and j > 0):
                table[i][j] = justifiers[colValueJustify](val, colWidths[j])
            elif i == 0 and colHeader:
                table[i][j] = justifiers[colHeadJustify](val, colWidths[j])
            # first column
            elif j == 0 and rowHeader:
                table[i][j] = justifiers[rowHeadJustify](val, colWidths[j])
            else:
                table[i][j] = justifiers[colValueJustify](val, colWidths[j])
            if j != len(table[i]) - 1 and columnSeparator:
                table[i][j] += columnSeparator

    out = ""
    for val in table[:-1]:
        out += "".join(val) + "\n"
    out += "".join(table[-1])
    if includeTrailingNewLine:
        out += '\n'

    return out

def _getStatsFunction(statsFuncName):
    if statsFuncName == 'max':
        toCall = nimble.calculate.maximum
    elif statsFuncName == 'min':
        toCall = nimble.calculate.minimum
    elif statsFuncName == 'mean':
        toCall = nimble.calculate.mean
    elif statsFuncName == 'median':
        toCall = nimble.calculate.median
    elif statsFuncName == 'mode':
        toCall = nimble.calculate.mode
    elif statsFuncName == 'sum':
        toCall = nimble.calculate.sum
    elif statsFuncName == 'variance':
        toCall = nimble.calculate.variance
    elif statsFuncName == 'uniquecount':
        toCall = nimble.calculate.uniqueCount
    elif statsFuncName == 'proportionmissing':
        toCall = nimble.calculate.proportionMissing
    elif statsFuncName == 'proportionzero':
        toCall = nimble.calculate.proportionZero
    elif statsFuncName == 'quartiles':
        toCall = nimble.calculate.quartiles
    elif statsFuncName == 'medianabsolutedeviation':
        toCall = nimble.calculate.medianAbsoluteDeviation
    elif statsFuncName in ['std', 'standarddeviation', 'samplestd',
                             'samplestandarddeviation']:
        toCall = nimble.calculate.standardDeviation
    elif statsFuncName in ['populationstd', 'populationstandarddeviation']:

        def populationStandardDeviation(values):
            return nimble.calculate.standardDeviation(values, False)

        toCall = populationStandardDeviation
    else:
        raise ValueError(f"Invalid statistical method name: {statsFuncName}")

    return toCall



############################
# Unavailable ML Methods  #
###########################

def _customMlGetattrHelper(name):
    """
    Helper for adjusting the __getAttr__ of nimble and TrainedLearner.
    Returns a "Try ... instead" style string for certain name inputs,
    which are then used by the caller to construct an appropriate
    AttributeError.
    """
    fill = None

    if name == 'fit':
        fill = "nimble.train() or the TrainedLearner's .retrain() method"

    if name in ['fit_transform', 'transform']:
        fill = "nimble.normalize or a data object's points/features"
        fill += ".fillMatching() and features.normalize() methods"

    if name == 'predict':
        fill = "the TrainedLearner's .apply() method"

    if name == 'score':
        fill = "the TrainedLearner's .getScores() method"

    if name == 'get_params':
        fill = "nimble.learnerParameters() or the TrainedLearner's "
        fill += ".getAttributes() method"

    if fill is not None:
        return f"Try {fill} instead."
    return None
