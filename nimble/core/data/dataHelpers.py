"""
Any method, object, or constant that might be used by multiple tests or
the main data wrapper objects defined in this module
"""

import copy
import math
import numbers
import inspect
import re
import operator
from functools import wraps

import numpy

import nimble
from nimble.utility import pd
from nimble import match
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction

# the prefix for default featureNames
DEFAULT_PREFIX = "_DEFAULT_#"
DEFAULT_PREFIX2 = DEFAULT_PREFIX+'%s'
DEFAULT_PREFIX_LENGTH = len(DEFAULT_PREFIX)

DEFAULT_NAME_PREFIX = "OBJECT_#"

defaultObjectNumber = 0

def isAllowedSingleElement(x):
    """
    Determine if an element is an allowed single element.
    """
    if isinstance(x, (numbers.Number, str)):
        return True

    if hasattr(x, '__len__'):#not a single element
        return False

    if x is None or x != x:#None and np.NaN are allowed
        return True

    return False

def nextDefaultObjectName():
    """
    Get the next available default name for an object.
    """
    global defaultObjectNumber
    ret = DEFAULT_NAME_PREFIX + str(defaultObjectNumber)
    defaultObjectNumber = defaultObjectNumber + 1
    return ret


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
        ret._name = nextDefaultObjectName()

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
    Combine two lists or dicts of names (point or feature) giving
    priority to non-default names. Names may also be None, the non-None
    names are returned or None if both are None.

    If both non-None, assumes the objects are of equal length and valid
    to be merged.
    """
    if otherNames is None:
        return baseNames
    if baseNames is None:
        return otherNames
    ret = {}
    for i, baseName in enumerate(baseNames):
        otherName = otherNames[i]
        baseIsDefault = baseName.startswith(DEFAULT_PREFIX)
        otherIsDefault = otherName.startswith(DEFAULT_PREFIX)

        if baseIsDefault and not otherIsDefault:
            ret[otherName] = i
        else:
            ret[baseName] = i

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


def _looksNumeric(val):
    # div is a good check of your standard numeric objects, and excludes things
    # list python lists. We must still explicitly exclude strings because of
    # the numpy string implementation.
    if not hasattr(val, '__truediv__') or isinstance(val, str):
        return False
    return True


def _checkNumeric(val):
    """
    Check if value looks numeric. Raise ValueError if not.
    """
    if not _looksNumeric(val):
        raise ValueError("Value '{}' does not seem to be numeric".format(val))


def formatIfNeeded(value, sigDigits):
    """
    Format the value into a string, and in the case of a float typed value,
    limit the output to the given number of significant digits.
    """
    if isinstance(value, (float, numpy.float)) and sigDigits is not None:
        return format(value, '.' + str(int(sigDigits)) + 'f')
    return str(value)


def indicesSplit(allowed, total):
    """
    Given the total length of a list, and a limit to
    how many indices we are allowed to display, return
    two lists of indices defining a middle ommision.
    In the tupple return, the first list are positive indices
    growing up from zero. The second list are negative indices
    growing up to negative one.
    """
    if total > allowed:
        allowed -= 1

    if allowed == 1 or total == 1:
        return ([0], [])

    forward = int(math.ceil(allowed / 2.0))
    backward = int(math.floor(allowed / 2.0))

    fIndices = list(range(forward))
    bIndices = list(range(-backward, 0))

    for i in range(len(bIndices)):
        bIndices[i] = bIndices[i] + total

    if fIndices[len(fIndices) - 1] == bIndices[0]:
        bIndices = bIndices[1:]

    return (fIndices, bIndices)


def hasNonDefault(obj, axis):
    """
    Determine if an axis has non default names.
    """
    if axis == 'point':
        if not obj.points._namesCreated():
            return False
        possibleIndices = range(len(obj.points))
    else:
        if not obj.features._namesCreated():
            return False
        possibleIndices = range(len(obj.features))

    getter = obj.points.getName if axis == 'point' else obj.features.getName

    for index in possibleIndices:
        if not getter(index).startswith(DEFAULT_PREFIX):
            return True

    return False


def makeNamesLines(indent, maxW, numDisplayNames, count, namesList, nameType):
    """
    Helper for __repr__ in Base.
    """
    if not namesList:
        return ''
    namesString = ""
    (posL, posR) = indicesSplit(numDisplayNames, count)
    possibleIndices = posL + posR

    if namesList is None:
        allDefault = True
    else:
        allDefault = all([namesList[i].startswith(DEFAULT_PREFIX)
                          for i in possibleIndices])

    if allDefault:
        return ""

    currNamesString = indent + nameType + '={'
    newStartString = indent * 2
    prevIndex = -1
    for i in range(len(possibleIndices)):
        currIndex = possibleIndices[i]
        # means there was a gap, need to insert elipses
        if currIndex - prevIndex > 1:
            addition = '..., '
            if len(currNamesString) + len(addition) > maxW:
                namesString += currNamesString + '\n'
                currNamesString = newStartString
            currNamesString += addition
        prevIndex = currIndex

        # get name and truncate if needed
        fullName = namesList[currIndex]
        currName = fullName
        if len(currName) > 11:
            currName = currName[:8] + '...'
        addition = "'" + currName + "':" + str(currIndex)

        # if it isn't the last entry, comma and space. if it is
        # the end-cbrace
        addition += ', ' if i != (len(possibleIndices) - 1) else '}'

        # if adding this would put us above the limit, add the line
        # to namesString before, and start a new line
        if len(currNamesString) + len(addition) > maxW:
            namesString += currNamesString + '\n'
            currNamesString = newStartString

        currNamesString += addition

    namesString += currNamesString + '\n'
    return namesString


def cleanKeywordInput(s):
    """
    Processes the input string such that it is in lower case, and all
    whitespace is removed. Such a string is then considered 'cleaned'
    and ready for comparison against lists of accepted values of keywords.
    """
    s = s.lower()
    s = "".join(s.split())
    return s

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
            viewMsg = "The {0} method is object modifying ".format(name)
            viewMsg += "and will always raise an exception for view "
            viewMsg += "objects.\n\n"
            viewMsg += "For reference, the docstring for this method "
            viewMsg += "when objects can be modified is below:\n"
            func.__doc__ = viewMsg + baseDoc

        return func
    return exceptionDocstring

def nonSparseAxisUniqueArray(obj, axis):
    """
    Get an array of unique data from non sparse types.

    List, Matrix and Dataframe all utilize this helper in the
    _unique_implementation().
    """
    obj._validateAxis(axis)
    if obj.getTypeString() == 'DataFrame':
        # faster than numpy.array(obj.data)
        data = obj.data.values
    else:
        data = numpy.array(obj.data, dtype=numpy.object_)
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

def uniqueNameGetter(obj, axis, uniqueIndices):
    """
    Get the first point or feature names of the object's unique values.
    """
    obj._validateAxis(axis)
    if axis == 'point':
        hasAxisNames = obj._pointNamesCreated()
        hasOffAxisNames = obj._featureNamesCreated()
        getAxisName = obj.points.getName
        getOffAxisNames = obj.features.getNames
    else:
        hasAxisNames = obj._featureNamesCreated()
        hasOffAxisNames = obj._pointNamesCreated()
        getAxisName = obj.features.getName
        getOffAxisNames = obj.points.getNames

    axisNames = False
    offAxisNames = False
    if hasAxisNames:
        axisNames = [getAxisName(i) for i in uniqueIndices]
    if hasOffAxisNames:
        offAxisNames = getOffAxisNames()

    return axisNames, offAxisNames

def valuesToPythonList(values, argName):
    """
    Create a python list of values from an integer (python or numpy),
    string, or an iterable container object
    """
    if isinstance(values, list):
        return values
    if isinstance(values, (int, numpy.integer, str)):
        return [values]
    valuesList = []
    try:
        for val in values:
            valuesList.append(val)
    except TypeError:
        msg = "The argument '{0}' is not an integer ".format(argName)
        msg += "(python or numpy), string, or an iterable container object."
        raise InvalidArgumentType(msg)

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
        msg += "for '{0}'. ".format(argName)
        msg += "Only one-dimensional objects are accepted."
        raise InvalidArgumentType(msg)

    valuesList = valuesToPythonList(values, argName)
    try:
        axisObj = obj._getAxis(axis)
        axisLen = len(axisObj)
        # faster to bypass getIndex if value is already a valid index
        indicesList = [v if (isinstance(v, (int, numpy.integer))
                             and 0 <= v < axisLen)
                       else axisObj.getIndex(v) for v in valuesList]
    except InvalidArgumentValue as iav:
        msg = "Invalid value for the argument '{0}'. ".format(argName)
        # add more detail to msg; slicing to exclude quotes
        msg += str(iav)[1:-1]
        raise InvalidArgumentValue(msg)

    return indicesList

def sortIndexPosition(obj, sortBy, sortHelper, axisAttr):
    """
    Helper for sort() to define new indexPosition list.
    """
    scorer = None
    comparator = None
    if axisAttr == 'points':
        test = obj._base.pointView(0)
    else:
        test = obj._base.featureView(0)
    try:
        sortHelper(test)
        scorer = sortHelper
    except TypeError:
        pass
    try:
        sortHelper(test, test)
        comparator = sortHelper
    except TypeError:
        pass

    if sortHelper is not None and scorer is None and comparator is None:
        msg = "sortHelper is neither a scorer or a comparator"
        raise InvalidArgumentType(msg)

    if comparator is not None:
        # make array of views
        viewArray = []
        for v in obj:
            viewArray.append(v)
        sortIndices = sorted(enumerate(viewArray),
                             key=lambda x:cmp_to_key(comparator)(x[1]))
        indexPosition = [i[0] for i in sortIndices]
    elif hasattr(scorer, 'permuter'):
        scoreArray = scorer.indices
        indexPosition = numpy.argsort(scoreArray)
    else:
        # make array of views
        viewArray = []
        for v in obj:
            viewArray.append(v)

        scoreArray = viewArray
        if scorer is not None:
            # use scoring function to turn views into values
            for i in range(len(viewArray)):
                scoreArray[i] = scorer(viewArray[i])
        else:
            for i in range(len(viewArray)):
                scoreArray[i] = viewArray[i][sortBy]

        # use numpy.argsort to make desired index array
        # this results in an array whose ith entry contains the the
        # index into the data of the value that should be in the ith
        # position.
        indexPosition = numpy.argsort(scoreArray)

    return indexPosition

def cmp_to_key(mycmp):
    """
    Convert a cmp=function for python2 into a key= function for python3.
    """
    class K:
        """
        Object for cmp_to_key.
        """
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def fillArrayWithCollapsedFeatures(featuresToCollapse, retainData,
                                   collapseData, currNumPoints, currFtNames,
                                   numRetPoints, numRetFeatures):
    """
    Helper function for modifying the underlying data for
    points.splitByCollapsingFeatures. Used in all non-sparse
    implementations.
    """
    fill = numpy.empty((numRetPoints, numRetFeatures), dtype=numpy.object_)
    fill[:, :-2] = numpy.repeat(retainData, len(featuresToCollapse), axis=0)

    # stack feature names repeatedly to create new feature
    namesAsFeature = numpy.tile(currFtNames, (1, currNumPoints))
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
    fill = numpy.empty(shape=(len(uniqueDict), numRetFeatures),
                       dtype=numpy.object_)

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
                fill[i, fillSlice] = [numpy.nan] * numExpanded
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
        test1 = numpy.array(arr1[checkPos], dtype=numpy.float_)
        test2 = numpy.array(arr2[checkPos], dtype=numpy.float_)
        return numpy.isnan(test1).all() and numpy.isnan(test2).all()
    except ValueError:
        return False

def createListOfDict(data, featureNames):
    """
    Create a list of dictionaries mapping feature names to point values.

    Dictionaries are in point order.
    """
    listofdict = []
    for point in data:
        feature_dict = {}
        for i, value in enumerate(point):
            feature = featureNames[i]
            feature_dict[feature] = value
        listofdict.append(feature_dict)
    return listofdict

def createDictOfList(data, featureNames, nFeatures):
    """
    Create a python dictionary mapping feature names to python lists.

    Each list contains the values in the feature in point order.
    """
    dictoflist = {}
    for i in range(nFeatures):
        feature = featureNames[i]
        values_list = data[:, i].tolist()
        dictoflist[feature] = values_list
    return dictoflist


def createDataNoValidation(returnType, data, pointNames=None,
                           featureNames=None, reuseData=False):
    """
    Instantiate a new object without validating the data.

    This function assumes that data being used is already in a format
    acceptable for nimble and the returnType's __init__ method. This
    allows for faster instantiation than through createData. However, if
    the data has not already been processed by nimble, it is not
    recommended to use this function.  Note that this function will
    handle point and feature names, but all other metadata will be set
    to default values.
    """
    if hasattr(data, 'dtype'):
        accepted = (numpy.number, numpy.object_, numpy.bool_)
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


def limitAndConvertToArray(obj, points, features):
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

    numpy.unique is most efficient but needs data to be numeric, when
    non-numeric data is present we replace every unique value with a
    unique integer and the generated mapping is used to return the
    unique values from the original data.
    """
    if isinstance(obj, nimble.core.data.Base):
        array = limitAndConvertToArray(obj, points, features)
    elif isinstance(obj, numpy.ndarray):
        array = obj
    else:
        raise InvalidArgumentValue("obj must be nimble object or numpy array")
    if issubclass(array.dtype.type, numpy.number):
        vals, counts = numpy.unique(array, return_counts=True)
        return {val: count for val, count in zip(vals, counts)}

    mapping = {}
    nextIdx = [0]
    def mapper(val):
        if val in mapping:
            return mapping[val]
        else:
            mapping[val] = nextIdx[0]
            nextIdx[0] += 1
            return mapping[val]
    vectorMap = numpy.vectorize(mapper)
    array = vectorMap(array)
    intMap = {v: k for k, v in mapping.items()}
    vals, counts = numpy.unique(array, return_counts=True)
    return {intMap[val]: count for val, count in zip(vals, counts)}


def wrapMatchFunctionFactory(matchFunc):
    """
    Wrap functions for matchingElements to validate output.

    Function must output a boolean value (True, False, 0, 1).
    """
    try:
        matchFunc(0, 0, 0)

        @wraps(matchFunc)
        def wrappedMatch(value, i, j):
            ret = matchFunc(value, i, j)
            # in [True, False] also covers 0 and 1 and numpy number and bool types
            if ret not in [True, False]:
                msg = 'toMatch function must return True, False, 0 or 1'
                raise InvalidArgumentValue(msg)
            return bool(ret) # converts 1 and 0 to True and False

        wrappedMatch.oneArg = False
    except TypeError:

        @wraps(matchFunc)
        def wrappedMatch(value):
            ret = matchFunc(value)
            # in [True, False] also covers 0 and 1 and numpy number and bool types
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
        func = getDictionaryMappingFunction(func)
    try:
        func(0, 0, 0)
        oneArg = False

        @wraps(func)
        def wrappedElementFunction(value, i, j):
            return elementValidated(value, i, j)

    except TypeError:
        oneArg = True
        # see if we can preserve zeros even if not explicitly set
        try:
            if not preserveZeros and func(0) == 0:
                preserveZeros = True
        except TypeError:
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

    Transform defines oppositeAxisInfo because the function return must
    have the same length of the axis opposite the one calling transform.
    Calculate allows for objects of varying lengths or single values to
    be returned so oppositeAxisInfo is None. For both, the return value
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
            msg += '{0} and it is unclear what the key/value '.format(axis)
            msg += 'pairs represent'
            raise InvalidArgumentValue(msg)

        # for transform, we need to validate the function's returned values
        # for calculate, the validation occurs when the new object is created
        if allowedLength:
            if (isinstance(ret, str)
                    or (not hasattr(ret, '__len__')
                        or len(ret) != allowedLength)):
                oppositeAxis = 'point' if axis == 'feature' else 'feature'
                msg = "'function' must return an iterable with as many elements "
                msg += "as {0}s in this object".format(oppositeAxis)
                raise InvalidArgumentValue(msg)
            if isAllowedSingleElement(ret):
                wrappedAxisFunc.updateConvertType(type(ret))
                return ret
            try:
                for value in ret:
                    if not isAllowedSingleElement(value):
                        msg = "The return of 'function' contains an "
                        msg += "invalid value. Numbers, strings, None, or "
                        msg += "nan are the only valid values. This value "
                        msg += "was " + str(type(value))
                        raise InvalidArgumentValue(msg)
                    wrappedAxisFunc.updateConvertType(type(value))
            except TypeError:
                msg = "'function' must return a single valid value "
                msg += "(number, string, None, or nan) or an iterable "
                msg += "container of valid values"
                raise InvalidArgumentValue(msg)

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
            if numpy.issubdtype(valueType, numpy.floating):
                valueType = float
            elif numpy.issubdtype(valueType, numpy.integer):
                valueType = int
            elif numpy.issubdtype(valueType, numpy.bool_):
                valueType = bool
        currLevel = typeHierarchy[wrappedAxisFunc.convertType]
        if valueType not in typeHierarchy:
            wrappedAxisFunc.convertType = object
        elif typeHierarchy[valueType] > currLevel:
            wrappedAxisFunc.convertType = valueType

    wrappedAxisFunc.convertType = None
    wrappedAxisFunc.updateConvertType = updateConvertType

    return wrappedAxisFunc

def getDictionaryMappingFunction(dictionary):
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
    array : numpy.ndarray
        A numpy array used to construct the iterator.
    order: str
        'point' or 'feature' indicating how the iterator will navigate
        the values.
    only : function, None
        The function that indicates whether __next__ should return the
        value. If None, every value is returned.
    """
    def __init__(self, array, order, only):
        if not isinstance(array, numpy.ndarray):
            msg = 'a numpy array is required to build the iterator'
            raise InvalidArgumentType(msg)
        if order == 'point':
            iterOrder = 'C'
        else:
            iterOrder = 'F'
        # these flags allow for object dtypes and empty iterators
        flags = ["refs_ok", "zerosize_ok"]
        iterator = numpy.nditer(array, order=iterOrder, flags=flags)
        self.iterator = iterator
        self.only = only

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # numpy.nditer returns value as an array type,
            # item() extracts the actual object we want to return
            val = next(self.iterator).item()
            if self.only is None or self.only(val):
                return val

def csvCommaFormat(name):
    if isinstance(name, str) and ',' in name:
        return '"{0}"'.format(name)
    return name

operatorDict = {'!=': operator.ne, '==': operator.eq, '=': operator.eq,
                '<=': operator.le, '>=': operator.ge,
                '<': operator.lt, '>': operator.gt}

def isQueryString(value, startswithOperator=True):
    """
    If a value is a query string, returns match object from re module if
    so. startswithOperator denotes whether the operator is expected at
    the beginning (excluding whitespace) of the string, ">=0", or if it
    can be within the string, "ft0<1".
    """
    if not isinstance(value, str):
        return False

    pattern = r'==|=|!=|>=|>|<=|<'
    if startswithOperator:
        match = re.match(pattern, value.strip())
    else:
        match = re.search(pattern, value.strip())
    return match if match else False

def elementQueryFunction(match):
    """
    Convert a re module match object to an element input function.
    """
    func = operatorDict[match.string[:match.end()]]
    matchVal = match.string[match.end():].strip()
    try:
        matchVal = float(matchVal)
    except ValueError:
        pass
    return lambda elem: func(elem, matchVal)

def axisQueryFunction(match, axis, nameChecker):
    """
    Convert a re module match object to an axis input function.
    """
    # to set in for loop
    nameOfPtOrFt = None
    valueOfPtOrFt = None
    optrOperator = None

    optr = match.string[match.start():match.end()]
    targetList = match.string.split(optr)
    #after splitting at the optr, list must have 2 items
    if len(targetList) != 2:
        msg = "the target({0}) is a ".format(string)
        msg += "query string but there is an error"
        raise InvalidArgumentValue(msg)
    nameOfPtOrFt = targetList[0]
    valueOfPtOrFt = targetList[1]
    nameOfPtOrFt = nameOfPtOrFt.strip()
    valueOfPtOrFt = valueOfPtOrFt.strip()

    #when point, check if the feature exists or not
    #when feature, check if the point exists or not
    if not nameChecker(nameOfPtOrFt):
        if axis == 'point':
            offAxis = 'feature'
        else:
            offAxis = 'point'
        msg = "the {0} ".format(offAxis)
        msg += "'{0}' doesn't exist".format(nameOfPtOrFt)
        raise InvalidArgumentValue(msg)

    optrOperator = operatorDict[optr]
    # convert valueOfPtOrFt from a string, if possible
    try:
        valueOfPtOrFt = float(valueOfPtOrFt)
    except ValueError:
        pass
    #convert query string to a function
    def target_f(vector):
        return optrOperator(vector[nameOfPtOrFt], valueOfPtOrFt)

    target_f.vectorized = True
    target_f.nameOfPtOrFt = nameOfPtOrFt
    target_f.valueOfPtOrFt = valueOfPtOrFt
    target_f.optr = optrOperator
    target = target_f

    return target

def limitedTo2D(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if hasattr(self, '_base'):
            tensorRank = len(self._base._shape)
        else:
            tensorRank = len(self._shape)
        if tensorRank > 2:
            msg = "{0} is not permitted when the ".format(method.__name__)
            msg += "data has more than two dimensions"
            raise ImproperObjectAction(msg)
        return method(self, *args, **kwargs)
    return wrapped
