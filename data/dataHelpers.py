"""
Any method, object, or constant that might be used by multiple tests or
the main data wrapper objects defined in this module

"""

from __future__ import division
from __future__ import absolute_import
import copy
import math
import inspect
import operator

from abc import ABCMeta
from abc import abstractmethod
import six
from six.moves import range
import numpy

from UML.exceptions import ArgumentException

# the prefix for default featureNames
DEFAULT_PREFIX = "_DEFAULT_#"
DEFAULT_PREFIX2 = DEFAULT_PREFIX+'%s'
DEFAULT_PREFIX_LENGTH = len(DEFAULT_PREFIX)

DEFAULT_NAME_PREFIX = "OBJECT_#"

OPTRLIST = ['<=', '>=', '!=', '==', '=', '<', '>']
OPTRDICT = {'<=': operator.le, '>=': operator.ge,
            '!=': operator.ne, '==': operator.eq,
            '<': operator.lt, '>': operator.gt}

defaultObjectNumber = 0


class View(six.with_metaclass(ABCMeta)):
    def equals(self, other):
        if not isinstance(other, View):
            return False
        if len(self) != len(other):
            return False
        if self.index() != other.index():
            return False
        if not self.name().startswith(DEFAULT_PREFIX) and not other.name().startswith(DEFAULT_PREFIX):
            if self.name() != other.name():
                return False
        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        return True

    def __str__(self):
        ret = '['
        for i in range(len(self)):
            if i != 0:
                ret += ', '
            ret += str(self[i])
        ret += ']'
        return ret

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def nonZeroIterator(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def index(self):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def getPointName(self, index):
        pass

    @abstractmethod
    def getFeatureName(self, index):
        pass


def nextDefaultObjectName():
    global defaultObjectNumber
    ret = DEFAULT_NAME_PREFIX + str(defaultObjectNumber)
    defaultObjectNumber = defaultObjectNumber + 1
    return ret


def binaryOpNamePathMerge(caller, other, ret, nameSource, pathSource):
    """Helper to set names and pathes of a return object when dealing
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
    """ Merges the point and feature names of the the two source objects,
    returning a double of the merged point names on the left and the
    merged feature names on the right. A merged name is either the
    baseSource's if both have default prefixes (or are equal). Otherwise,
    it is the name which doesn't have a default prefix from either source.

    Assumptions: (1) Both objects are the same shape. (2) The point names
    and feature names of both objects are consistent (any non-default
    names in the same positions are equal)

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
        retPNames = mergeNames(baseSource.getPointNames(), otherSource.getPointNames())
    elif baseSource._pointNamesCreated() and not otherSource._pointNamesCreated():
        retPNames = baseSource.pointNames
    elif not baseSource._pointNamesCreated() and otherSource._pointNamesCreated():
        retPNames = otherSource.pointNames
    else:
        retPNames = None

    if baseSource._featureNamesCreated() and otherSource._featureNamesCreated():
        retFNames = mergeNames(baseSource.getFeatureNames(), otherSource.getFeatureNames())
    elif baseSource._featureNamesCreated() and not otherSource._featureNamesCreated():
        retFNames = baseSource.featureNames
    elif not baseSource._featureNamesCreated() and otherSource._featureNamesCreated():
        retFNames = otherSource.featureNames
    else:
        retFNames = None

    return (retPNames, retFNames)


def reorderToMatchList(dataObject, matchList, axis):
    """
    Helper which will reorder the data object along the specified axis so that
    instead of being in an order corresponding to a sorted version of matchList,
    it will be in the order of the given matchList.

    matchList must contain only indices, not name based identifiers.

    """
    if axis.lower() == "point":
        sortFunc = dataObject.sortPoints
    else:
        sortFunc = dataObject.sortFeatures

    sortedList = copy.copy(matchList)
    sortedList.sort()
    mappedOrig = {}
    for i in range(len(matchList)):
        mappedOrig[matchList[i]] = i

    if axis == 'point':
        indexGetter = lambda x: dataObject.getPointIndex(x.getPointName(0))
    else:
        indexGetter = lambda x: dataObject.getFeatureIndex(x.getFeatureName(0))

    def scorer(viewObj):
        index = indexGetter(viewObj)
        return mappedOrig[sortedList[index]]

    sortFunc(sortHelper=scorer)

    return dataObject


def _looksNumeric(val):
    # div is a good check of your standard numeric objects, and excludes things
    # list python lists. We must still explicitly exclude strings because of the
    # numpy string implementation.
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
    if _looksNumeric(value):
        if not isinstance(value, int) and sigDigits is not None:
            return format(value, '.' + str(int(sigDigits)) + 'f')
    return str(value)


def indicesSplit(allowed, total):
    """Given the total length of a list, and a limit to
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
    if axis == 'point':
        possibleIndices = range(len(obj.points))
    else:
        possibleIndices = range(len(obj.features))

    getter = obj.getPointName if axis == 'point' else obj.getFeatureName

    ret = False
    for index in possibleIndices:
        if not getter(index).startswith(DEFAULT_PREFIX):
            ret = True

    return ret


def makeNamesLines(indent, maxW, numDisplayNames, count, namesList, nameType):
    if not namesList: return ''
    namesString = ""
    (posL, posR) = indicesSplit(numDisplayNames, count)
    possibleIndices = posL + posR

    allDefault = all([namesList[i].startswith(DEFAULT_PREFIX) for i in possibleIndices])

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
    """Processes the input string such that it is in lower case, and all
    whitespace is removed. Such a string is then considered 'cleaned' and
    ready for comparison against lists of accepted values of keywords.

    """
    s = s.lower()
    s = "".join(s.split())
    return s


def makeConsistentFNamesAndData(fnames, data, dataWidths, colHold):
    """Adjust the inputs to be a consistent length and with
    consistent omission by removing
    values and columns from the middle. Returns None.

    """
    namesOmitIndex = int(math.floor(len(fnames) / 2.0))
    dataOmitIndex = int(math.floor(len(dataWidths) / 2.0))
    namesOmitted = fnames[namesOmitIndex] == colHold
    dataOmitted = False
    if len(data) > 0:
        dataOmitted = data[0][dataOmitIndex] == colHold

    if len(fnames) == len(dataWidths):
        # inputs consistent, don't have to do anything
        if namesOmitted and dataOmitted:
            return
        elif namesOmitted:
            desiredLength = len(dataWidths)
            remNum = 0
            removalVals = data
            removalWidths = dataWidths
        elif dataOmitted:
            desiredLength = len(fnames)
            remNum = 0
            removalVals = [fnames]
            removalWidths = None
        else:  # inputs consistent, don't have to do anything
            return
    elif len(fnames) > len(dataWidths):
        desiredLength = len(dataWidths)
        remNum = len(fnames) - desiredLength
        removalVals = [fnames]
        removalWidths = None
    else:  # len(fnames) < len(dataWidths)
        desiredLength = len(fnames)
        remNum = len(dataWidths) - desiredLength
        removalVals = data
        removalWidths = dataWidths

    if desiredLength % 2 == 0:
        removeIndex = int(math.ceil(desiredLength / 2.0))
    else:  # desiredLength % 2 == 1
        removeIndex = int(math.floor(desiredLength / 2.0))

    # remove values so that we reach the target length
    for row in removalVals:
        for i in range(remNum):
            row.pop(removeIndex)

    # now that we are at the target length, we have to modify
    # a column to indicat that values were omitted
    for row in removalVals:
        row[removeIndex] = colHold

    if removalWidths is not None:
        # remove those widths associated with omitted values
        for i in range(remNum):
            removalWidths.pop(removeIndex)

        # modify the width associated with the colHold
        if removalWidths is not None:
            removalWidths[removeIndex] = len(colHold)


def inheritDocstringsFactory(toInherit):
    """
    Factory to make decorator to copy docstrings from toInherit for reimplementations
    in the wrapped object. Only those functions without docstrings will be given the
    corresponding docstrings from toInherit.
    """
    def inheritDocstring(cls):
        writable = cls.__dict__
        for name in writable:
            if inspect.isfunction(writable[name]) and hasattr(toInherit, name):
                func = writable[name]
                if not func.__doc__:
                    func.__doc__ = getattr(toInherit, name).__doc__

        return cls
    return inheritDocstring

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
        raise ArgumentException(msg)

    return valuesList
