
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
Any method, object, or constant that might be used by multiple tests or
the main data wrapper objects defined in this module.
"""

import math
import re
from functools import wraps
import os.path

import numpy as np

import nimble
from nimble import match
from nimble._utility import pd, plt, scipy
from nimble._utility import inspectArguments
from nimble._utility import isAllowedSingleElement
from nimble._utility import is2DArray
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble.core.logger import handleLogging

def binaryOpNamePathMerge(caller, other, ret, nameSource, pathSource):
    """
    Helper to set names and pathes of a return object when dealing
    with some kind of binary operation on data objects. nameSource
    is expected to be either 'self' (indicating take the name from
    the calling object) or None (take a default name). pathSource
    is expected to be either 'self' or 'merge' (meaning to take
    a path only if one of the caller or other has a path specified,
    else use default values)
    """

    # determine return value's name
    if nameSource == 'self':
        ret._name = caller._name
    else:
        ret._name = None

    if pathSource == 'self':
        ret._absPath = caller.absolutePath
        ret._relPath = caller.relativePath
    elif pathSource == 'merge':
        if caller.absolutePath is not None and other.absolutePath is None:
            ret._absPath = caller.absolutePath
        elif caller.absolutePath is None and other.absolutePath is not None:
            ret._absPath = other.absolutePath
        #		elif caller.absolutePath == other.absolutePath:
        #			ret._absPath = caller.absolutePath
        else:
            ret._absPath = None

        if caller.relativePath is not None and other.relativePath is None:
            ret._relPath = caller.relativePath
        elif caller.relativePath is None and other.relativePath is not None:
            ret._relPath = other.relativePath
        #		elif caller.relativePath == other.relativePath:
        #			ret._relPath = caller.relativePath
        else:
            ret._relPath = None
    else:
        ret._absPath = None
        ret._relPath = None


def mergeNames(baseNames, otherNames):
    """
    Combine two lists of names (point or feature) giving
    priority to non-default names. Names may also be None, the non-None
    names are returned or None if both are None.

    If both non-None, assumes the objects are of equal length and valid
    to be merged.
    """
    if otherNames is None:
        return baseNames
    if baseNames is None:
        return otherNames
    ret = [None] * len(baseNames)
    for i, (baseName, otherName) in enumerate(zip(baseNames, otherNames)):
        if baseName is None and otherName is not None:
            ret[i] = otherName
        elif baseName is not None:
            ret[i] = baseName

    return ret


def mergeNonDefaultNames(baseSource, otherSource):
    """
    Merges the point and feature names of the the two source objects,
    returning a double of the merged point names on the left and the
    merged feature names on the right. A merged name is either the
    baseSource's if both have default prefixes (or are equal).
    Otherwise, it is the name which doesn't have a default prefix from
    either source.

    Assumptions: (1) Both objects are the same shape. (2) The point
    names and feature names of both objects are consistent (any
    non-default names in the same positions are equal)
    """
    ptNames = mergeNames(baseSource.points._getNamesNoGeneration(),
                         otherSource.points._getNamesNoGeneration())
    ftNames = mergeNames(baseSource.features._getNamesNoGeneration(),
                         otherSource.features._getNamesNoGeneration())
    return ptNames, ftNames


def looksNumeric(val):
    """
    Check if a value looks numeric.

    div is a good check of your standard numeric objects, and excludes
    things like python lists. Must still explicitly exclude strings
    because of the numpy string implementation.
    """
    if not hasattr(val, '__truediv__') or isinstance(val, str):
        return False
    return True


def checkNumeric(val):
    """
    Check if value looks numeric. Raise ValueError if not.
    """
    if not looksNumeric(val):
        raise ValueError(f"Value '{val}' does not seem to be numeric")


def formatIfNeeded(value, sigDigits):
    """
    Format the value into a string, and in the case of a float typed value,
    limit the output to the given number of significant digits.
    """
    if isinstance(value, (float, np.float64)):
        if value != value:
            return ''
        if sigDigits is not None:
            return format(value, '.' + str(int(sigDigits)) + 'f')
    return str(value)


def indicesSplit(numAllow, iRange):
    """
    Given a limit to how many indices we are allowed to display and a range of
    possible indices, return two lists of indices defining a middle ommision.
    In the tupple return, the first list are positive indices
    growing up from zero. The second list are negative indices
    growing up to negative one.
    """
    total = len(iRange)
    if total > numAllow:
        numAllow -= 1

    if numAllow == 1 or total == 1:
        return ([0], [])

    forward = int(math.ceil(numAllow / 2.0))
    backward = int(math.floor(numAllow / 2.0))

    fIndices = list(range(forward))
    bIndices = list(range(-backward, 0))

    for i, bIdx in enumerate(bIndices):
        bIndices[i] = bIdx + total

    if fIndices[len(fIndices) - 1] == bIndices[0]:
        bIndices = bIndices[1:]

    fIndices = list(map(lambda x: iRange.start + x, fIndices))
    bIndices = list(map(lambda x: iRange.start + x, bIndices))

    return (fIndices, bIndices)

def cleanKeywordInput(string):
    """
    Processes the input string such that it is in lower case, and all
    whitespace is removed. Such a string is then considered 'cleaned'
    and ready for comparison against lists of accepted values of
    keywords.
    """
    string = string.lower()
    string = "".join(string.split())
    return string

def validateInputString(string, accepted, paramName):
    """
    Validate that a string belongs to a set of acceptable values.
    """
    acceptedClean = list(map(cleanKeywordInput, accepted))

    msg = paramName + " must be equivalent to one of the following: "
    msg += str(accepted) + ", but '" + str(string)
    msg += "' was given instead. Note: casing and whitespace is "
    msg += "ignored when checking the " + paramName

    if not isinstance(string, str):
        raise InvalidArgumentType(msg)

    cleanFuncName = cleanKeywordInput(string)

    if cleanFuncName not in acceptedClean:
        raise InvalidArgumentValue(msg)

    return cleanFuncName


def readOnlyException(name):
    """
    The exception to raise for functions that are disallowed in view
    objects.
    """
    msg = "The " + name + " method is disallowed for View objects. View "
    msg += "objects are read only, yet this method modifies the object"
    raise ImproperObjectAction(msg)

# prepend a message that view objects will raise an exception to Base docstring
def exceptionDocstringFactory(cls):
    """
    Modify docstrings for view objects based on a  base class.

    The docstring will acknowledge that the method is not available
    for views, but also provide the docstring for the method for
    future reference.
    """
    def exceptionDocstring(func):
        name = func.__name__
        baseDoc = getattr(cls, name).__doc__
        if baseDoc is not None:
            viewMsg = f"The {name} method is object modifying and will always "
            viewMsg += "raise an exception for view objects.\n\n"
            viewMsg += "For reference, the docstring for this method "
            viewMsg += "when objects can be modified is below:\n"
            func.__doc__ = viewMsg + baseDoc

        return func
    return exceptionDocstring

def denseAxisUniqueArray(obj, axis):
    """
    Get an array of unique data from non sparse types.

    List, Matrix and Dataframe all utilize this helper in the
    _unique_implementation().
    """
    validateAxis(axis)
    if obj.getTypeString() == 'DataFrame':
        data = obj._asNumpyArray()
    else:
        data = np.array(obj._data, dtype=np.object_)
    if axis == 'feature':
        data = data.transpose()

    unique = set()
    uniqueIndices = []
    for i, values in enumerate(data):
        if tuple(values) not in unique:
            unique.add(tuple(values))
            uniqueIndices.append(i)
    uniqueData = data[uniqueIndices]

    if axis == 'feature':
        uniqueData = uniqueData.transpose()

    return uniqueData, uniqueIndices

def valuesToPythonList(values, argName):
    """
    Create a python list of values from an integer (python or numpy),
    string, or an iterable container object
    """
    if isinstance(values, list):
        return values
    if isinstance(values, (int, np.integer, str)):
        return [values]
    valuesList = []
    try:
        for val in values:
            valuesList.append(val)
    except TypeError as e:
        msg = f"The argument '{argName}' is not an integer (python or numpy), "
        msg += "string, or an iterable container object."
        raise InvalidArgumentType(msg) from e

    return valuesList

def constructIndicesList(obj, axis, values, argName=None):
    """
    Construct a list of indices from a valid integer (python or numpy) or
    string, or an iterable, list-like container of valid integers and/or
    strings
    """
    if argName is None:
        argName = axis + 's'
    # pandas DataFrames are iterable but do not iterate through the values
    if pd.nimbleAccessible() and isinstance(values, pd.DataFrame):
        msg = "A pandas DataFrame object is not a valid input "
        msg += f"for '{argName}'. Only one-dimensional objects are accepted."
        raise InvalidArgumentType(msg)

    valuesList = valuesToPythonList(values, argName)
    try:
        axisObj = obj._getAxis(axis)
        axisLen = len(axisObj)
        # faster to bypass getIndex if value is already a valid index
        indicesList = [v if (isinstance(v, (int, np.integer))
                             and 0 <= v < axisLen)
                       else axisObj.getIndex(v) for v in valuesList]
    except InvalidArgumentValue as iav:
        msg = f"Invalid value for the argument '{argName}'. "
        # add more detail to msg; slicing to exclude quotes
        msg += str(iav)[1:-1]
        raise InvalidArgumentValue(msg) from iav

    return indicesList

def fillArrayWithCollapsedFeatures(featuresToCollapse, retainData,
                                   collapseData, currNumPoints, currFtNames,
                                   numRetPoints, numRetFeatures):
    """
    Helper function for modifying the underlying data for
    points.splitByCollapsingFeatures. Used in all non-sparse
    implementations.
    """
    fill = np.empty((numRetPoints, numRetFeatures), dtype=np.object_)
    fill[:, :-2] = np.repeat(retainData, len(featuresToCollapse), axis=0)

    # stack feature names repeatedly to create new feature
    namesAsFeature = np.tile(currFtNames, (1, currNumPoints))
    fill[:, -2] = namesAsFeature
    # flatten values by row then reshape into new feature
    valuesAsFeature = collapseData.flatten()
    fill[:, -1] = valuesAsFeature

    return fill

def fillArrayWithExpandedFeatures(uniqueDict, namesIdx, uniqueNames,
                                  numRetFeatures, numExpanded):
    """
    Helper function for modifying the underlying data for
    combinePointsByExpandingFeatures. Used in all non-sparse
    implementations.
    """
    fill = np.empty(shape=(len(uniqueDict), numRetFeatures), dtype=np.object_)

    endSlice = slice(namesIdx + (len(uniqueNames) * numExpanded), None)
    for i, point in enumerate(uniqueDict):
        fill[i, :namesIdx] = point[:namesIdx]
        for j, name in enumerate(uniqueNames):
            fillStart = namesIdx + (j * numExpanded)
            fillStop = fillStart + numExpanded
            fillSlice = slice(fillStart, fillStop)
            if name in uniqueDict[point]:
                fill[i, fillSlice] = uniqueDict[point][name]
            else:
                fill[i, fillSlice] = [np.nan] * numExpanded
        fill[i, endSlice] = point[namesIdx:]

    return fill


def allDataIdentical(arr1, arr2):
    """
    Checks for equality between all points in the arrays. Arrays
    containing NaN values in the same positions will also be considered
    equal.
    """
    if arr1.shape != arr2.shape:
        return False
    try:
        # check the values that are not equal
        checkPos = arr1 != arr2
        # if values are nan, conversion to float dtype will be successful
        test1 = np.array(arr1[checkPos], dtype=np.float64)
        test2 = np.array(arr2[checkPos], dtype=np.float64)
        return np.isnan(test1).all() and np.isnan(test2).all()
    except ValueError:
        return False

def createListOfDict(data, featureNames):
    """
    Create a list of dictionaries mapping feature names to point values.

    Dictionaries are in point order.
    """
    listofdict = []
    for point in data:
        featureDict = {}
        for i, value in enumerate(point):
            feature = featureNames[i]
            featureDict[feature] = value
        listofdict.append(featureDict)
    return listofdict

def createDictOfList(data, featureNames, nFeatures):
    """
    Create a python dictionary mapping feature names to python lists.

    Each list contains the values in the feature in point order.
    """
    dictoflist = {}
    for i in range(nFeatures):
        feature = featureNames[i]
        valuesList = data[:, i].tolist()
        dictoflist[feature] = valuesList
    return dictoflist


def createDataNoValidation(returnType, data, pointNames=None,
                           featureNames=None, reuseData=False):
    """
    Instantiate a new object without validating the data.

    This function assumes that data being used is already in a format
    acceptable for nimble and the returnType's __init__ method. This
    allows for faster instantiation than through nimble.data. However,
    if the data has not already been processed by nimble, it is not
    recommended to use this function.  Note that this function will
    handle point and feature names, but all other metadata will be set
    to default values.
    """
    if hasattr(data, 'dtype'):
        accepted = (np.number, np.object_, np.bool_)
        if not issubclass(data.dtype.type, accepted):
            msg = "data must have numeric, boolean, or object dtype"
            raise InvalidArgumentType(msg)
    initMethod = getattr(nimble.core.data, returnType)
    if returnType == 'List':
        return initMethod(data, pointNames=pointNames,
                          featureNames=featureNames, reuseData=reuseData,
                          checkAll=False)
    return initMethod(data, pointNames=pointNames, featureNames=featureNames,
                      reuseData=reuseData)


def _limitAndConvertToArray(obj, points, features):
    if points is None and features is None:
        return obj.copy(to='numpyarray')
    pWanted = points if points is not None else slice(None)
    fWanted = features if features is not None else slice(None)
    limited = obj[pWanted, fWanted]
    return limited.copy(to='numpyarray')


def denseCountUnique(obj, points=None, features=None):
    """
    Return dictionary of the unique elements and their values for dense
    data representations.

    np.unique is most efficient but needs data to be numeric, when
    non-numeric data is present we replace every unique value with a
    unique integer and the generated mapping is used to return the
    unique values from the original data.
    """
    if isinstance(obj, nimble.core.data.Base):
        array = _limitAndConvertToArray(obj, points, features)
    elif isinstance(obj, np.ndarray):
        array = obj
    else:
        raise InvalidArgumentValue("obj must be nimble object or numpy array")
    # unique treats every name value as a unique value, we want to treat them
    # as the same
    if issubclass(array.dtype.type, np.number):
        vals, counts = np.unique(array, return_counts=True)
        intMap = None
    else:
        mapping = {}
        nextIdx = [0]
        def mapper(val):
            if val in mapping:
                return mapping[val]

            mapping[val] = nextIdx[0]
            nextIdx[0] += 1
            return mapping[val]

        vectorMap = np.vectorize(mapper)
        array = vectorMap(array)
        intMap = {v: k for k, v in mapping.items()}
        vals, counts = np.unique(array, return_counts=True)
    ret = {}
    nan = np.nan
    for val, count in zip(vals, counts):
        if val != val:
            if nan in ret:
                ret[nan] += count
            else:
                ret[nan] = count
        elif intMap is None:
            ret[val] = count
        else:
            ret[intMap[val]] = count

    return ret


def wrapMatchFunctionFactory(matchFunc, elementwise=True):
    """
    Wrap functions for matchingElements to validate output.

    Function must output a boolean value (True, False, 0, 1).
    """
    if isinstance(matchFunc, str):
        matchFunc = match.QueryString(matchFunc, elementwise)
    try:
        matchFunc(0, 0, 0)

        @wraps(matchFunc)
        def wrappedMatch(value, i, j):
            ret = matchFunc(value, i, j)
            # in [True, False] also covers 0, 1 and numpy number and bool types
            if ret not in [True, False]:
                msg = 'toMatch function must return True, False, 0 or 1'
                raise InvalidArgumentValue(msg)
            return bool(ret) # converts 1 and 0 to True and False

        wrappedMatch.oneArg = False
    except TypeError:

        @wraps(matchFunc)
        def wrappedMatch(value):
            ret = matchFunc(value)
            # in [True, False] also covers 0, 1 and numpy number and bool types
            if ret not in [True, False]:
                msg = 'toMatch function must return True, False, 0 or 1'
                raise InvalidArgumentValue(msg)
            return bool(ret) # converts 1 and 0 to True and False

        wrappedMatch.oneArg = True

    wrappedMatch.convertType = bool

    return wrappedMatch

def validateElementFunction(func, preserveZeros, skipNoneReturnValues,
                            funcName):
    """
    Wrap functions for transformElements to verify and validate output.

    Adjust user function based on preserveZeros and skipNoneReturnValues
    parameters and validate the returned element types.
    """
    def elementValidated(value, *args):
        if preserveZeros and value == 0:
            return float(0)
        ret = func(value, *args)
        if ret is None and skipNoneReturnValues:
            return value
        if skipNoneReturnValues and ret is None:
            return value
        if not isAllowedSingleElement(ret):
            msg = funcName + " can only return numeric, boolean, or string "
            msg += "values, but the returned value was " + str(type(ret))
            raise InvalidArgumentValue(msg)
        return ret

    if isinstance(func, dict):
        func = _getDictionaryMappingFunction(func)

    try:
        a, _, _, d = inspectArguments(func)
        numRequiredArgs = len(a) - len(d)
    except ValueError:
        # functions implemented in C cannot be inspected. We will assume these
        # are builtins and that they take one argument (str, int, float, etc.)
        numRequiredArgs = 1

    if numRequiredArgs == 3:
        oneArg = False

        @wraps(func)
        def wrappedElementFunction(value, i, j):
            return elementValidated(value, i, j)

    else:
        oneArg = True
        # see if we can preserve zeros even if not explicitly set
        if not preserveZeros:
            try:
                if func(0) == 0:
                    preserveZeros = True
            # since it is a user function we cannot predict the exception type
            except Exception: # pylint: disable=broad-except
                pass

        @wraps(func)
        def wrappedElementFunction(value):
            return elementValidated(value)

    wrappedElementFunction.oneArg = oneArg
    wrappedElementFunction.preserveZeros = preserveZeros

    return wrappedElementFunction

def validateAxisFunction(func, axis, allowedLength=None):
    """
    Wrap axis transform and calculate functions to validate types.

    Transform defines allowedLength because the function return must
    have the same length of the axis opposite the one calling transform.
    Calculate allows for objects of varying lengths or single values to
    be returned so allowedLength is None. For both, the return value
    types are validated.
    """
    if func is None:
        raise InvalidArgumentType("'function' must not be None")

    @wraps(func)
    def wrappedAxisFunc(*args, **kwargs):
        ret = func(*args, **kwargs)

        if isinstance(ret, dict):
            msg = "The return of 'function' cannot be a dictionary. The "
            msg += 'returned object must contain only new values for each '
            msg += f'{axis} and it is unclear what the key/value pairs '
            msg += 'represent'
            raise InvalidArgumentValue(msg)

        # for transform, we need to validate the function's returned values
        # for calculate, the validation occurs when the new object is created
        if allowedLength:
            # need specific information if return is string
            # need to say length of return is mismatched with axis length
            oppositeAxis = 'point' if axis == 'feature' else 'feature'
            endmsg = f"elements as {oppositeAxis}s ({allowedLength}) in this object"
            if isinstance(ret, str):
                msg = "'function' returns a string instead of an iterable with as many "
                raise InvalidArgumentValue(msg+endmsg)
            if ((not hasattr(ret, '__len__')
                        or len(ret) != allowedLength)):
                msg = "'function' must return an iterable with as many "
                raise InvalidArgumentValue(msg+endmsg)
            if isAllowedSingleElement(ret):
                wrappedAxisFunc.updateConvertType(type(ret))
                return ret
            notBase = not isinstance(ret, nimble.core.data.Base)
            try:
                for value in ret:
                    # values in Base have already been validated
                    if notBase and not isAllowedSingleElement(value):
                        msg = "The return of 'function' contains an "
                        msg += "invalid value. Numbers, strings, None, or "
                        msg += "nan are the only valid values. This value "
                        msg += "was " + str(type(value))
                        raise InvalidArgumentValue(msg)
                    wrappedAxisFunc.updateConvertType(type(value))
            except TypeError as e:
                msg = "'function' must return a single valid value "
                msg += "(number, string, None, or nan) or an iterable "
                msg += "container of valid values"
                raise InvalidArgumentValue(msg) from e

        return ret

    # None indicates the convertType has not been set. Since functions using
    # this wrapper require non-empty data, convertType will always be set to
    # one of the other 3 typeHierarchy keys.
    typeHierarchy = {None: -1, bool: 0, int: 1, float: 2, object: 3}
    def updateConvertType(valueType):
        # None will be assigned float to align with nan values being floats
        if valueType == type(None):
            valueType = float
        if hasattr(valueType, 'dtype'):
            if np.issubdtype(valueType, np.floating):
                valueType = float
            elif np.issubdtype(valueType, np.integer):
                valueType = int
            elif np.issubdtype(valueType, np.bool_):
                valueType = bool
        currLevel = typeHierarchy[wrappedAxisFunc.convertType]
        if valueType not in typeHierarchy:
            wrappedAxisFunc.convertType = object
        elif typeHierarchy[valueType] > currLevel:
            wrappedAxisFunc.convertType = valueType

    wrappedAxisFunc.convertType = None
    wrappedAxisFunc.updateConvertType = updateConvertType

    return wrappedAxisFunc

def _getDictionaryMappingFunction(dictionary):
    def valueMappingFunction(value):
        if value in dictionary:
            return dictionary[value]
        return value
    return valueMappingFunction

class ElementIterator1D(object):
    """
    Object providing iteration through each item in the axis.
    """
    def __init__(self, source):
        if source:
            self.source = source.copy('python list', outputAs1D=True)
        else:
            self.source = []
        self.sourceLen = len(source)
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.position < self.sourceLen:
            value = self.source[self.position]
            self.position += 1
            return value
        raise StopIteration

class NimbleElementIterator(object):
    """
    Modify data level iterators to extract the correct data and limit
    the iterator output if necessary.

    Parameters
    ----------
    array : np.ndarray
        A numpy array used to construct the iterator.
    order: str
        'point' or 'feature' indicating how the iterator will navigate
        the values.
    only : function, None
        The function that indicates whether __next__ should return the
        value. If None, every value is returned.
    """
    def __init__(self, array, order, only):
        if not isinstance(array, np.ndarray):
            msg = 'a numpy array is required to build the iterator'
            raise InvalidArgumentType(msg)
        if order == 'point':
            iterOrder = 'C'
        else:
            iterOrder = 'F'
        # these flags allow for object dtypes and empty iterators
        flags = ["refs_ok", "zerosize_ok"]
        iterator = np.nditer(array, order=iterOrder, flags=flags)
        self.iterator = iterator
        self.only = only

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # np.nditer returns value as an array type,
            # [()] extracts the actual object we want to return
            val = next(self.iterator)[()]
            if self.only is None or self.only(val):
                return val

def csvCommaFormat(name):
    """
    Prevent the name from being interpreted as two different csv values.
    """
    if isinstance(name, str) and ',' in name:
        if '"' in name:
            name = re.sub(r'"', '""', name)
        name = f'"{name}"'
    return name

def limitedTo2D(method):
    """
    Wrapper for operations only allowed in two-dimensions.
    """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if hasattr(self, '_base'):
            tensorRank = len(self._base._dims)
        else:
            tensorRank = len(self._dims)
        if tensorRank > 2:
            msg = f"{method.__name__} is not permitted when the data has more "
            msg += "than two dimensions"
            raise ImproperObjectAction(msg)
        return method(self, *args, **kwargs)
    return wrapped

def convertToNumpyOrder(order):
    """
    Convert 'point' and 'feature' to the string equivalent in np.
    """
    return 'C' if order == 'point' else 'F'

def arrangeFinalTable(pnames, pnamesWidth, dataTable, dataWidths, fnames,
                      pnameSep):
    """
    Arrange the final table of values for Base string representation.
    """
    # We make extensive use of list addition in this helper in order
    # to prepend single values onto lists.

    # glue point names onto the left of the data
    for i, data in enumerate(dataTable):
        if pnames:
            dataTable[i] = [pnames[i], pnameSep] + data
        else:
            dataTable[i] = ['', ''] + data

    dataWidths = [pnamesWidth, len(pnameSep)] + dataWidths
    # only apply to singletons, not everything
    def tuplify(val):
        if not isinstance(val, tuple):
            return tuple([val])
        return val
    dataWidths = list(map(tuplify, dataWidths))
    fnames = ['', ''] + fnames

    # glue feature names onto the top of the data
    dataTable = [fnames] + dataTable

    return dataTable, dataWidths

def inconsistentNames(selfNames, otherNames):
    """Private function to find and return all name inconsistencies
    between the given two sets. It ignores equality of default
    values, considering only whether non default names consistent
    (position by position) and uniquely positioned (if a non default
    name is present in both, then it is in the same position in
    both). The return value is a dict between integer IDs and the
    pair of offending names at that position in both objects.

    Assumptions: the size of the two name sets is equal.
    """
    inconsistencies = {}

    def checkFromLeftKeys(ret, leftNames, rightNames):
        for index, lname in enumerate(leftNames):
            rname = rightNames[index]
            if lname is not None:
                if rname is not None:
                    if lname != rname:
                        ret[index] = (lname, rname)
                else:
                    # if a name in one is mirrored by a default name,
                    # then it must not appear in any other index;
                    # and therefore, must not appear at all.
                    if rightNames.count(lname) > 0:
                        ret[index] = (lname, rname)
                        ret[rightNames.index(lname)] = (lname, rname)


    # check both name directions
    checkFromLeftKeys(inconsistencies, selfNames, otherNames)
    checkFromLeftKeys(inconsistencies, otherNames, selfNames)

    return inconsistencies

def unequalNames(selfNames, otherNames):
    """Private function to find and return all name inconsistencies
    between the given two sets. It ignores equality of default
    values, considering only whether non default names consistent
    (position by position) and uniquely positioned (if a non default
    name is present in both, then it is in the same position in
    both). The return value is a dict between integer IDs and the
    pair of offending names at that position in both objects.

    Assumptions: the size of the two name sets is equal.
    """
    inconsistencies = {}

    def checkFromLeftKeys(ret, leftNames, rightNames):
        for index, lname in enumerate(leftNames):
            rname = rightNames[index]
            if lname is not None:
                if rname is not None:
                    if lname != rname:
                        ret[index] = (lname, rname)
                else:
                    ret[index] = (lname, rname)

    # check both name directions
    checkFromLeftKeys(inconsistencies, selfNames, otherNames)
    checkFromLeftKeys(inconsistencies, otherNames, selfNames)

    return inconsistencies

def equalNames(selfNames, otherNames):
    """
    Private function to determine equality of either pointNames of
    featureNames. It ignores equality of default values, considering
    only whether non default names consistent (position by position)
    and uniquely positioned (if a non default name is present in
    both, then it is in the same position in both).
    """
    if selfNames is None and otherNames is None:
        return True
    if (selfNames is None
            and all(n is None for n in otherNames)):
        return True
    if (otherNames is None
            and all(n is None for n in selfNames)):
        return True
    if selfNames is None or otherNames is None:
        return False
    if len(selfNames) != len(otherNames):
        return False

    namesUnequal = unequalNames(selfNames, otherNames)
    return not namesUnequal

def validateAxis(axis):
    """
    Check the string value for axis is valid.
    """
    if axis not in ('point', 'feature'):
        msg = 'axis parameter may only be "point" or "feature"'
        raise InvalidArgumentValue(msg)

def validateRangeOrder(startName, startVal, endName, endVal):
    """
    Validate a range where both values are inclusive.
    """
    if startVal > endVal:
        msg = "When specifying a range, the arguments were resolved to "
        msg += "having the values " + startName
        msg += "=" + str(startVal) + " and " + endName + "=" + str(endVal)
        msg += ", yet the starting value is not allowed to be greater "
        msg += "than the ending value (" + str(startVal) + ">"
        msg += str(endVal) + ")"

        raise InvalidArgumentValueCombination(msg)

####################
# Plotting Helpers #
####################

def pyplotRequired(func):
    """
    Wrap plot functions to check that matplotlib.pylot is accessible.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        if not plt.nimbleAccessible():
            msg = 'matplotlib.pyplot is required for plotting'
            raise PackageException(msg)
        # prevent interactive plots from showing until .show() called
        plt.ioff()
        return func(*args, **kwargs)
    return wrapped

def plotFigureHandling(figureID):
    """
    Provide the figure and axis for the plot.

    Use the stored figure and axis if the figureID exists, otherwise
    generate a new figure and axis.
    """
    figures = nimble.core.data._plotFigures
    if figureID is not None and figureID in figures:
        return figures[figureID]

    fig, ax = plt.subplots()
    # tight_layout automatically adjusts margins to accommodate labels
    fig.set_tight_layout(True)
    # Setting a _nimbleAxisLimits attribute is a workaround for properly
    # setting figure axis limits. See plotUpdateAxisLimits docstring.
    ax._nimbleAxisLimits = [None, None, None, None]
    if figureID is not None:
        figures[figureID] = fig, ax
    return fig, ax

def plotOutput(outPath, show):
    """
    Save and/or display a figure, if necessary.
    """
    if outPath is not None:
        outFormat = None
        if isinstance(outPath, str):
            (_, ext) = os.path.splitext(outPath)
            if len(ext) == 0:
                outFormat = 'png'
        plt.savefig(outPath, format=outFormat)
    if show:
        plt.show()
        # once plt.show() is called, existing figures will no longer display on
        # the next plt.show() call, so there is no need to keep _plotFigures
        nimble.core.data._plotFigures = {}

def plotAxisLabels(ax, xAxisLabel, xLabelIfTrue, yAxisLabel, yLabelIfTrue):
    """
    Helper for setting the axis labels on a figure.
    """
    if xAxisLabel is True:
        xAxisLabel = xLabelIfTrue
    if xAxisLabel is False:
        xAxisLabel = None
    ax.set_xlabel(xAxisLabel)
    if yAxisLabel is True:
        yAxisLabel = yLabelIfTrue
    if yAxisLabel is False:
        yAxisLabel = None
    ax.set_ylabel(yAxisLabel)

def plotUpdateAxisLimits(ax, xMin, xMax, yMin, yMax):
    """
    Internally tracks the user-defined axis limit values.

    This must be called BEFORE the plot is generated.
    When plotting multiple plots on the same figure, there can be figure
    axis limits that are defined based on previous plots but should be
    dynamic. See plotAxisLimits docstring. This may prohibit data in the
    additional plots from displaying. Instead, we store any user values
    and ensure that matplotlib can adjust the limit values automatically
    to accommodate all of the data. After the plot is generated, we
    apply any user-defined limits with a call to plotAxisLimits.
    """
    if xMin is not None:
        ax._nimbleAxisLimits[0] = xMin
    if xMax is not None:
        ax._nimbleAxisLimits[1] = xMax
    if yMin is not None:
        ax._nimbleAxisLimits[2] = yMin
    if yMax is not None:
        ax._nimbleAxisLimits[3] = yMax
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)

def plotAxisLimits(ax):
    """
    Apply user-defined axis limits.

    This must be called AFTER the plot is generated.
    Calls to ax.set_[xy]limit will apply defined limits for the axis
    even if one parameter is set to None. This is problematic for
    additional plots on the same figure because we want None to indicate
    that a specific axis limit is still dynamic. For this reason, we use
    plotUpdateAxisLimits before generating the plot to allow the figure
    limits to remain dynamic when adding plots. Then apply the limits
    defined by the user after the plot has been added to the figure.
    """
    xMin, xMax, yMin, yMax = ax._nimbleAxisLimits
    if xMin is not None or xMax is not None:
        ax.set_xlim(left=xMin, right=xMax)
    if yMin is not None or yMax is not None:
        ax.set_ylim(bottom=yMin, top=yMax)

def plotXTickLabels(ax, fig, names, numTicks):
    """
    Helper for setting and orienting the labels for x ticks.
    """
    xtickMax = max(len(name) for name in names)
    # 1 unit of figwidth can contain 9 characters
    tickWidth = int(fig.get_figwidth() / numTicks * 9)
    if xtickMax > tickWidth:
        ax.set_xticklabels(names, rotation='vertical')
    else:
        ax.set_xticklabels(names)

def plotConfidenceIntervalMeanAndError(feature):
    """
    Helper for calculating the mean and error for error bar charts.
    """
    if not scipy.nimbleAccessible():
        msg = 'scipy must be installed for confidence intervals.'
        raise PackageException(msg)
    mean = nimble.calculate.mean(feature)
    std = nimble.calculate.standardDeviation(feature)
    # two tailed 95% CI with n -1 degrees of freedom
    tStat = scipy.stats.t.ppf(0.025, len(feature) - 1)
    error = tStat * (std / np.sqrt(len(feature)))
    error = np.abs(error)

    return mean, error

def plotErrorBars(ax, axisRange, means, errors, horizontal, **kwargs):
    """
    Helper for plotting an error bar chart.
    """
    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'
    if 'capsize' not in kwargs:
        kwargs['capsize'] = 8

    if horizontal:
        ax.errorbar(y=axisRange, x=means, xerr=errors, **kwargs)
    else:
        ax.errorbar(x=axisRange, y=means, yerr=errors, **kwargs)

def plotSingleBarChart(ax, axisRange, heights, horizontal, **kwargs):
    """
    Helper for plotting a single bar chart.
    """
    if horizontal:
        ax.barh(axisRange, heights, **kwargs)
    else:
        ax.bar(axisRange, heights, **kwargs)

def plotMultiBarChart(ax, heights, horizontal, legendTitle, **kwargs):
    """
    Helper for plotting a multiple bar chart.
    """
    # need to manually handle some kwargs with subgroups
    if 'width' in kwargs:
        width = kwargs['width']
        del kwargs['width']
    else:
        width = 0.8 # plt.bar default
    if 'color' in kwargs:
        # sets color array to apply to subgroup bars not group bars
        ax.set_prop_cycle(color=kwargs['color'])
        del kwargs['color']
    elif len(heights) > 10:
        # matplotlib default will repeat colors, need broader colormap
        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1, len(heights))]
        ax.set_prop_cycle(color=colors)
    singleWidth = width / len(heights)
    start = 1 - (width / 2) + (singleWidth / 2)

    for i, (name, height) in enumerate(heights.items()):
        widths = np.arange(start, len(height))
        widths += i * singleWidth
        if horizontal:
            ax.barh(widths, height, height=singleWidth, label=name,
                    **kwargs)
        else:
            ax.bar(widths, height, width=singleWidth, label=name,
                   **kwargs)

    ax.legend(title=legendTitle)

def is2DList(lst):
    """
    Determine if a python list is two-dimensional.
    """
    if not isinstance(lst, list):
        return False
    try:
        return lst[0] == [] or isAllowedSingleElement(lst[0][0])
    except IndexError:
        return lst == []

def isValid2DObject(data):
    """
    Determine validity of object for nimble data object instantiation.
    """
    return is2DArray(data) or is2DList(data)

def modifyNumpyArrayValue(arr, index, newVal):
    """
    Change a single value in a numpy array.

    Will cast to a new dtype if necessary.
    """
    nonNumericNewVal = match.nonNumeric(newVal) and newVal is not None
    floatNewVal = isinstance(newVal, (float, np.floating)) or newVal is None
    if nonNumericNewVal and arr.dtype != np.object_:
        arr = arr.astype(np.object_)
    elif floatNewVal and arr.dtype not in (np.floating, np.object_):
        arr = arr.astype(np.float64)

    arr[index] = newVal

    return arr

def getFeatureDtypes(obj):
    """
    Get a dtype for each feature in the object.

    DataFrames and Lists can return a tuple of heterogeneous types,
    Matrix and Sparse will be homogeneous.
    """
    if hasattr(obj._data, 'dtypes'):
        return tuple(obj._data.dtypes)
    if hasattr(obj._data, 'dtype'):
        return (obj._data.dtype,) * len(obj.features)

    dtypeList = []
    # _data is list or ListPassThrough
    for ft in zip(*obj._data):
        dtype = max(map(np.dtype, map(type, ft)))
        if np.can_cast(dtype, float):
            dtypeList.append(dtype)
        else:
            dtypeList.append(np.object_)

    return tuple(dtypeList)

def prepLog(method):
    """
    Provides logging for methods that manipulate data.
    """
    @wraps(method)
    def wrapped(self, *args, useLog=None, **kwargs):
        ret = method(self, *args, useLog=None, **kwargs)
        handleLogging(useLog, 'prep', self, wrapped.__name__, ret, *args,
                      **kwargs)
        return ret
    return wrapped
