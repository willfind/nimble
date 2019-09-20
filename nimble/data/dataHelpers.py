"""
Any method, object, or constant that might be used by multiple tests or
the main data wrapper objects defined in this module
"""

from __future__ import division
from __future__ import absolute_import
import copy
import math
import numbers
import inspect
import re

import six
from six.moves import range
import numpy

import nimble
from nimble import importModule
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue

pd = importModule('pandas')

# the prefix for default featureNames
DEFAULT_PREFIX = "_DEFAULT_#"
DEFAULT_PREFIX2 = DEFAULT_PREFIX+'%s'
DEFAULT_PREFIX_LENGTH = len(DEFAULT_PREFIX)

DEFAULT_NAME_PREFIX = "OBJECT_#"

defaultObjectNumber = 0

def isAllowedSingleElement(x):
    """
    This function is to determine if an element is an allowed single
    element.
    """
    if isinstance(x, (numbers.Number, six.string_types)):
        return True

    if hasattr(x, '__len__'):#not a single element
        return False

    if x is None or x != x:#None and np.NaN are allowed
        return True

    return

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
    # merge helper
    def mergeNames(baseNames, otherNames):
        ret = {}
        for i in range(len(baseNames)):
            baseName = baseNames[i]
            otherName = otherNames[i]
            baseIsDefault = baseName.startswith(DEFAULT_PREFIX)
            otherIsDefault = otherName.startswith(DEFAULT_PREFIX)

            if baseIsDefault and not otherIsDefault:
                ret[otherName] = i
            else:
                ret[baseName] = i

        return ret

    (retPNames, retFNames) = (None, None)

    if baseSource._pointNamesCreated() and otherSource._pointNamesCreated():
        retPNames = mergeNames(baseSource.points.getNames(),
                               otherSource.points.getNames())
    elif (baseSource._pointNamesCreated()
          and not otherSource._pointNamesCreated()):
        retPNames = baseSource.pointNames
    elif (not baseSource._pointNamesCreated()
          and otherSource._pointNamesCreated()):
        retPNames = otherSource.pointNames
    else:
        retPNames = None

    if (baseSource._featureNamesCreated()
            and otherSource._featureNamesCreated()):
        retFNames = mergeNames(baseSource.features.getNames(),
                               otherSource.features.getNames())
    elif (baseSource._featureNamesCreated()
          and not otherSource._featureNamesCreated()):
        retFNames = baseSource.featureNames
    elif (not baseSource._featureNamesCreated()
          and otherSource._featureNamesCreated()):
        retFNames = otherSource.featureNames
    else:
        retFNames = None

    return (retPNames, retFNames)


def reorderToMatchList(dataObject, matchList, axis):
    """
    Helper which will reorder the data object along the specified axis
    so that instead of being in an order corresponding to a sorted
    version of matchList, it will be in the order of the given
    matchList.

    matchList must contain only indices, not name based identifiers.
    """
    if axis.lower() == "point":
        sortFunc = dataObject.points.sort
    else:
        sortFunc = dataObject.features.sort

    sortedList = copy.copy(matchList)
    sortedList.sort()
    mappedOrig = {}
    for i in range(len(matchList)):
        mappedOrig[matchList[i]] = i

    if axis == 'point':
        def indexGetter(x):
            return dataObject.points.getIndex(x.points.getName(0))
    else:
        def indexGetter(x):
            return dataObject.features.getIndex(x.features.getName(0))

    def scorer(viewObj):
        index = indexGetter(viewObj)
        return mappedOrig[sortedList[index]]

    sortFunc(sortHelper=scorer, useLog=False)

    return dataObject


def _looksNumeric(val):
    # div is a good check of your standard numeric objects, and excludes things
    # list python lists. We must still explicitly exclude strings because of
    # the numpy string implementation.
    if not hasattr(val, '__truediv__') or isinstance(val, six.string_types):
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
    if _looksNumeric(value) and not (value is True or value is False):
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

    if not isinstance(string, six.string_types):
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
    raise TypeError(msg)

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
        try:
            baseDoc = getattr(cls, name).__doc__
            if baseDoc is not None:
                viewMsg = "The {0} method is object modifying ".format(name)
                viewMsg += "and will always raise an exception for view "
                viewMsg += "objects.\n\n"
                viewMsg += "For reference, the docstring for this method "
                viewMsg += "when objects can be modified is below:\n"
                func.__doc__ = viewMsg + baseDoc
        except AttributeError:
            # ignore built-in functions that differ between py2 and py3
            # (__idiv__ vs __itruediv__, __ifloordiv__)
            pass
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
    if isinstance(values, (int, numpy.integer, six.string_types)):
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
    if pd and isinstance(values, pd.DataFrame):
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
        test = obj._source.pointView(0)
    else:
        test = obj._source.featureView(0)
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
                                  numRetFeatures):
    """
    Helper function for modifying the underlying data for
    combinePointsByExpandingFeatures. Used in all non-sparse
    implementations.
    """
    fill = numpy.empty(shape=(len(uniqueDict), numRetFeatures),
                       dtype=numpy.object_)

    for i, point in enumerate(uniqueDict):
        fill[i, :namesIdx] = point[:namesIdx]
        for j, name in enumerate(uniqueNames):
            if name in uniqueDict[point]:
                fill[i, namesIdx + j] = uniqueDict[point][name]
            else:
                fill[i, namesIdx + j] = numpy.nan
        fill[i, namesIdx + len(uniqueNames):] = point[namesIdx:]

    return fill


def allDataIdentical(arr1, arr2):
    """
    Checks for equality between all points in the arrays. Arrays
    containing NaN values in the same positions will also be considered
    equal.
    """
    try:
        # check the values that are not equal
        checkPos = arr1 != arr2
        # if values are nan, conversion to float dtype will be successful
        test1 = numpy.array(arr1[checkPos], dtype=numpy.float_)
        test2 = numpy.array(arr2[checkPos], dtype=numpy.float_)
        return numpy.isnan(test1).all() and numpy.isnan(test2).all()
    except Exception:
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
    initMethod = getattr(nimble.data, returnType)
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
    if isinstance(obj, nimble.data.Base):
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
    def wrappedMatch(value):
        ret = matchFunc(value)
        # in [True, False] also covers 0 and 1 and numpy number and bool types
        if ret not in [True, False]:
            msg = 'toMatch function must return True, False, 0 or 1'
            raise InvalidArgumentValue(msg)
        return bool(ret) # converts 1 and 0 to True and False
    wrappedMatch.oneArg = True
    wrappedMatch.__name__ = matchFunc.__name__
    wrappedMatch.__doc__ = matchFunc.__doc__

    return wrappedMatch
