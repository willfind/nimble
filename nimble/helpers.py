"""
Helper functions for any functions defined in core.py

They are separated here so that that (most) top level
user facing functions are contained in core.py without
the distraction of helpers

"""

import csv
import inspect
import importlib
from io import StringIO, BytesIO
import os.path
import re
import datetime
import copy
import sys
import itertools

import numpy

import nimble
from nimble.logger import handleLogging
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import InvalidArgumentValueCombination, PackageException
from nimble.exceptions import ImproperObjectAction
from nimble.exceptions import FileFormatException
from nimble.data import Base
from nimble.data.dataHelpers import isAllowedSingleElement
from nimble.data.sparse import removeDuplicatesNative
from nimble.randomness import pythonRandom
from nimble.randomness import numpyRandom
from nimble.utility import numpy2DArray, is2DArray
from nimble.utility import sparseMatrixToArray
from nimble.utility import scipy, pd, requests, h5py

def findBestInterface(package):
    """
    Attempt to determine the interface.

    Takes the string name of a possible interface provided to some other
    function by a nimble user, and attempts to find the interface which
    best matches that name amoung those available. If it does not match
    any available interfaces, then an exception is thrown.
    """
    for interface in nimble.interfaces.available:
        if (package == interface.getCanonicalName()
                or interface.isAlias(package)):
            return interface
    for interface in nimble.interfaces.predefined:
        if (package == interface.getCanonicalName()
                or interface.isAlias(package)):
            # interface is a predefined one, but instantiation failed
            return interface.provideInitExceptionInfo()
    # if package is not recognized, provide generic exception information
    msg = "package '" + package
    msg += "' was not associated with any of the available package interfaces"
    raise InvalidArgumentValue(msg)


def _learnerQuery(name, queryType):
    """
    Takes a string of the form 'package.learnerName' and a string
    defining a queryType of either 'parameters' or 'defaults' then
    returns the results of either the package's
    getParameters(learnerName) function or the package's
    getDefaultValues(learnerName) function.
    """
    [package, learnerName] = name.split('.')

    if queryType == "parameters":
        toCallName = 'getLearnerParameterNames'
    elif queryType == 'defaults':
        toCallName = 'getLearnerDefaultValues'
    else:
        raise InvalidArgumentValue("Unrecognized queryType: " + queryType)

    interface = findBestInterface(package)
    ret = getattr(interface, toCallName)(learnerName)

    if len(ret) == 1:
        return ret[0]
    return ret


def isAllowedRaw(data, allowLPT=False):
    """
    Verify raw data is one of the accepted types.
    """
    if isinstance(data, Base):
        return True
    if allowLPT and 'PassThrough' in str(type(data)):
        return True
    if scipy.nimbleAccessible() and scipy.sparse.issparse(data):
        return True
    if isinstance(data, (tuple, list, dict, numpy.ndarray)):
        return True

    if pd.nimbleAccessible():
        if isinstance(data, (pd.DataFrame, pd.Series, pd.SparseDataFrame)):
            return True

    return False


def validateReturnType(returnType):
    retAllowed = copy.copy(nimble.data.available)
    retAllowed.append(None)
    if returnType not in retAllowed:
        msg = "returnType must be a value in " + str(retAllowed)
        raise InvalidArgumentValue(msg)


def extractNamesFromRawList(rawData, pnamesID, fnamesID):
    """
    Remove name data from a python list.

    Takes a raw python list of lists and if specified remove those
    rows or columns that correspond to names, returning the remaining
    data, and the two name objects (or None in their place if they were
    not specified for extraction). pnamesID may either be None, or an
    integer ID corresponding to the column of point names. fnamesID
    may eith rbe None, or an integer ID corresponding to the row of
    feature names.
    """
    # we allow a list of values as input, but we assume a list of lists type
    # for this function; take the input list to mean a single point
    addedDim = False
    if rawData == [] or isAllowedSingleElement(rawData[0]):
        rawData = [rawData]
        addedDim = True
    # otherwise, we know we are dealing with (at least) a twice nested list.
    # A thrice or more nested list will cause problems with the extraction
    # helpers, so we validate the contents first.
    elif len(rawData[0]) > 0 and not isAllowedSingleElement(rawData[0][0]):
        msg = "List of lists containing numbers, strings, None, or nan are "
        msg += "the only accepted list formats, yet the (0,0)th element was "
        msg += str(type(rawData[0][0]))
        raise TypeError(msg)
    mustCopy = ['automatic', True]
    if pnamesID in mustCopy or fnamesID is mustCopy:
        # copy rawData to avoid modifying the original user data
        if isinstance(rawData[0], tuple):
            rawData = [list(row) for row in rawData]
        else:
            rawData = [row.copy() for row in rawData]

    firstRow = rawData[0] if len(rawData) > 0 else None
    secondRow = rawData[1] if len(rawData) > 1 else None
    pnamesID, fnamesID = autoDetectNamesFromRaw(pnamesID, fnamesID, firstRow,
                                                secondRow)
    pnamesID = 0 if pnamesID is True else None
    fnamesID = 0 if fnamesID is True else None

    retPNames = None
    if pnamesID is not None:
        temp = []
        for i in range(len(rawData)):
            # grab and remove each value in the feature associated
            # with point names
            currVal = rawData[i].pop(pnamesID)
            # have to skip the index of the feature names, if they are also
            # in the data
            if fnamesID is not None and i != fnamesID:
            # we wrap it with the string constructor in case the
                # values in question AREN'T strings
                temp.append(str(currVal))
        retPNames = temp

    retFNames = None
    if fnamesID is not None:
        # don't have to worry about an overlap entry with point names;
        # if they existed we had already removed those values.
        # Therefore: just pop that entire point
        temp = rawData.pop(fnamesID)
        for i in range(len(temp)):
            temp[i] = str(temp[i])
        retFNames = temp

    if addedDim:
        rawData = rawData[0]

    return (rawData, retPNames, retFNames)


def extractNamesFromPdDataFrame(rawData, pnamesID, fnamesID):
    """
    Output the index of rawData as pointNames.
    Output the columns of rawData as featureNames.
    """
    firstRow = rawData.values[0] if len(rawData) > 0 else None
    secondRow = rawData.values[1] if len(rawData) > 1 else None
    pnamesID, fnamesID = autoDetectNamesFromRaw(pnamesID, fnamesID, firstRow,
                                                secondRow)
    pnamesID = 0 if pnamesID is True else None
    fnamesID = 0 if fnamesID is True else None

    retPNames = None
    if pnamesID is not None:
        retPNames = [str(i) for i in rawData.index.tolist()]

    retFNames = None
    if fnamesID is not None:
        retFNames = [str(i) for i in rawData.columns.tolist()]

    return (rawData, retPNames, retFNames)


def extractNamesFromPdSeries(rawData, pnamesID, fnamesID):
    """
    Output the index of rawData as featureNames.
    """
    retPNames = None
    if pnamesID is True:
        retPNames = [rawData.index[0]]
        rawData = rawData[1:]

    retFNames = None
    if fnamesID is True:
        retFNames = [str(i) for i in rawData.index.tolist()]
        rawData = numpy.empty((0, len(retFNames)))

    return (rawData, retPNames, retFNames)


def createConstantHelper(numpyMaker, returnType, numPoints, numFeatures,
                         pointNames, featureNames, name):
    """
    Create nimble data objects containing constant values.

    Use numpy.ones or numpy.zeros to create constant nimble objects of
    the designated returnType.
    """
    validateReturnType(returnType)
    if numPoints < 0:
        msg = "numPoints must be 0 or greater, yet " + str(numPoints)
        msg += " was given."
        raise InvalidArgumentValue(msg)

    if numFeatures < 0:
        msg = "numFeatures must be 0 or greater, yet " + str(numFeatures)
        msg += " was given."
        raise InvalidArgumentValue(msg)

    if numPoints == 0 and numFeatures == 0:
        msg = "Either one of numPoints (" + str(numPoints) + ") or "
        msg += "numFeatures (" + str(numFeatures) + ") must be non-zero."
        raise InvalidArgumentValueCombination(msg)

    if returnType == 'Sparse':
        if not scipy.nimbleAccessible():
            msg = "scipy is not available"
            raise PackageException(msg)
        if numpyMaker == numpy.ones:
            rawDense = numpyMaker((numPoints, numFeatures))
            rawSparse = scipy.sparse.coo_matrix(rawDense)
        else:  # case: numpyMaker == numpy.zeros
            assert numpyMaker == numpy.zeros
            rawSparse = scipy.sparse.coo_matrix((numPoints, numFeatures))
        return nimble.createData(returnType, rawSparse, pointNames=pointNames,
                                 featureNames=featureNames, name=name,
                                 useLog=False)
    else:
        raw = numpyMaker((numPoints, numFeatures))
        return nimble.createData(returnType, raw, pointNames=pointNames,
                                 featureNames=featureNames, name=name,
                                 useLog=False)


def transposeMatrix(matrixObj):
    """
    This function is similar to np.transpose.
    copy.deepcopy(np.transpose(matrixObj)) may generate a messed data,
    so I created this function.
    """
    return numpy2DArray(list(zip(*matrixObj.tolist())), dtype=matrixObj.dtype)


def extractNames(rawData, pointNames, featureNames):
    """
    Extract the point and feature names from the raw data, if necessary.
    """
    acceptedNameTypes = (str, bool, type(None), list, dict)
    if not isinstance(pointNames, acceptedNameTypes):
        try:
            pointNames = [val for val in pointNames]
        except TypeError:
            msg = "if pointNames are not 'bool' or a 'str', "
            msg += "they should be other 'iterable' object"
            raise InvalidArgumentType(msg)
    if not isinstance(featureNames, acceptedNameTypes):
        try:
            featureNames = [val for val in featureNames]
        except TypeError:
            msg = "if featureNames are not 'bool' or a 'str', "
            msg += "they should be other 'iterable' object"
            raise InvalidArgumentType(msg)
    # 1. convert dict like {'a':[1,2], 'b':[3,4]} to np.array
    # featureNames must be those keys
    # pointNames must be False or automatic
    if isinstance(rawData, dict):
        if rawData:
            featureNames = list(rawData.keys())
            rawData = numpy2DArray(list(rawData.values()), dtype=numpy.object_)
            if len(featureNames) == len(rawData):
                # {'a':[1,3],'b':[2,4],'c':['a','b']} -> keys = ['a', 'c', 'b']
                # np.matrix(values()) = [[1,3], ['a', 'b'], [2,4]]
                # thus transpose is needed
                # {'a':1, 'b':2, 'c':3} --> keys = ['a', 'c', 'b']
                # np.matrix(values()) = [[1,3,2]]
                # transpose is not needed
                rawData = transposeMatrix(rawData)
            pointNames = None

        else: # rawData={}
            featureNames = None
            rawData = numpy.empty([0, 0])
            pointNames = None

    # 2. convert list of dict ie. [{'a':1, 'b':3}, {'a':2, 'b':4}] to np.array
    # featureNames must be those keys
    # pointNames must be False or automatic
    elif (isinstance(rawData, list)
          and len(rawData) > 0
          and isinstance(rawData[0], dict)):
        # double nested list contained list-type forced values from first row
        values = [list(rawData[0].values())]
        # in py3 keys() returns a dict_keys object comparing equality of these
        # objects is valid, but converting to lists for comparison can fail
        keys = rawData[0].keys()
        for i, row in enumerate(rawData[1:]):
            if row.keys() != keys:
                msg = "The keys at index {i} do not match ".format(i=i)
                msg += "the keys at index 0. Each dictionary in the list must "
                msg += "contain the same key values."
                raise InvalidArgumentValue(msg)
            values.append(list(row.values()))
        rawData = values
        featureNames = keys
        pointNames = None

    else:
        # 3. for rawData of other data types
        # check if we need to do name extraction, setup new variables,
        # or modify values for subsequent call to data init method.
        if isinstance(rawData, list):
            func = extractNamesFromRawList
        elif isinstance(rawData, tuple):
            rawData = list(rawData)
            func = extractNamesFromRawList
        elif isinstance(rawData, numpy.ndarray):
            func = extractNamesFromNumpy
        elif scipy.nimbleAccessible() and scipy.sparse.issparse(rawData):
            # all input coo_matrices must have their duplicates removed; all
            # helpers past this point rely on there being single entires only.
            if isinstance(rawData, scipy.sparse.coo_matrix):
                rawData = removeDuplicatesNative(rawData)
            func = extractNamesFromScipySparse
        elif (pd.nimbleAccessible()
                and isinstance(rawData, (pd.DataFrame, pd.SparseDataFrame))):
            func = extractNamesFromPdDataFrame
        elif pd.nimbleAccessible() and isinstance(rawData, pd.Series):
            func = extractNamesFromPdSeries

        rawData, tempPointNames, tempFeatureNames = func(rawData, pointNames,
                                                         featureNames)

        # tempPointNames and tempFeatures may either be None or explicit names.
        # pointNames and featureNames may be True, False, None, 'automatic', or
        # explicit names. False and None have the same behavior.

        # User explicitly did not want names extracted
        if pointNames is False or pointNames is None:
            # assert that data was not accidentally removed
            assert tempPointNames is None
            pointNames = None
        # User explicitly wanted us to extract names
        elif pointNames is True:
            pointNames = tempPointNames
        # We could have extracted names but didn't
        elif pointNames == 'automatic' and tempPointNames is None:
            pointNames = None
        # We could have extracted name and did
        elif pointNames == 'automatic' and tempPointNames is not None:
            pointNames = tempPointNames
        # Point names were provided by user
        else:
            assert tempPointNames is None
            pointNames = pointNames

        # User explicitly did not want names extracted
        if featureNames is False or featureNames is None:
            # assert that data was not accidentally removed
            assert tempFeatureNames is None
            featureNames = None
        # User explicitly wanted us to extract names
        elif featureNames is True:
            featureNames = tempFeatureNames
        # We could have extracted names but didn't
        elif featureNames == 'automatic' and tempFeatureNames is None:
            featureNames = None
        # We could have extracted name and did
        elif featureNames == 'automatic' and tempFeatureNames is not None:
            featureNames = tempFeatureNames
        # Feature names were provided by user
        else:
            assert tempFeatureNames is None
            featureNames = featureNames

    return rawData, pointNames, featureNames


def convertData(returnType, rawData, pointNames, featureNames,
                convertToType):
    """
    Convert data to an object type which is compliant with the
    initializion for the given returnType. Additionally, ensure the data
    is converted to the convertToType. If convertToType is None an
    attempt will be made to convert all the data to floats, if
    unsuccessful, the data will remain the same object type.
    """
    typeMatch = {'List': list,
                 'Matrix': numpy.ndarray}
    if scipy:
        typeMatch['Sparse'] = scipy.sparse.spmatrix
    if pd:
        typeMatch['DataFrame'] = pd.DataFrame

    # perform type conversion if necessary
    # this also guarantees data is in an acceptable format
    rawData = elementTypeConvert(rawData, convertToType)
    try:
        typeMatchesReturn = isinstance(rawData, typeMatch[returnType])
    except KeyError:
        if returnType == 'Sparse':
            package = 'scipy'
        if returnType == 'DataFrame':
            package = 'pandas'
        msg = "{0} must be installed to create a {1} object"
        raise PackageException(msg.format(package, returnType))

    # if the data can be used to instantiate the object we pass it as-is
    # otherwise a 2D array is needed as they are accepted by all init methods
    if typeMatchesReturn:
        if returnType == 'List':
            lenFts = len(featureNames) if featureNames else 0
            if len(rawData) == 0:
                lenPts = len(pointNames) if pointNames else 0
                return numpy.empty([lenPts, lenFts])
            if hasattr(rawData[0], '__len__') and len(rawData[0]) == 0:
                return numpy.empty([len(rawData), lenFts])
        if returnType == 'Matrix' and len(rawData.shape) == 1:
            rawData = numpy2DArray(rawData)
        return rawData
    ret = convertToArray(rawData, convertToType, pointNames, featureNames)
    if returnType == 'Sparse' and ret.dtype == numpy.object_:
        # Sparse will convert None to 0 so we need to use numpy.nan instead
        ret[ret == None] = numpy.nan
    return ret

def convertToArray(rawData, convertToType, pointNames, featureNames):
    if pd.nimbleAccessible() and isinstance(rawData, pd.DataFrame):
        return rawData.values
    if pd.nimbleAccessible() and isinstance(rawData, pd.Series):
        if rawData.empty:
            return numpy.empty((0, rawData.shape[0]))
        return numpy2DArray(rawData.values)
    if scipy.nimbleAccessible() and scipy.sparse.isspmatrix(rawData):
        return sparseMatrixToArray(rawData)
    if isinstance(rawData, numpy.ndarray):
        if not is2DArray(rawData):
            rawData = numpy2DArray(rawData)
        return rawData
    # lists (or other similar objects)
    lenFts = len(featureNames) if featureNames else 0
    if len(rawData) == 0:
        lenPts = len(pointNames) if pointNames else 0
        return numpy.empty([lenPts, lenFts])
    if hasattr(rawData[0], '__len__') and len(rawData[0]) == 0:
        return numpy.empty([len(rawData), lenFts])
    if convertToType is not None:
        arr = numpy2DArray(rawData, dtype=convertToType)
        # run through elementType to convert to object if not accepted type
        return elementTypeConvert(arr, None)
    else:
        arr = numpy2DArray(rawData)
    # The bool dtype is acceptable, but others run the risk of transforming the
    # data so we default to object dtype.
    if arr.dtype == bool:
        return arr
    return numpy2DArray(rawData, dtype=numpy.object_)

def elementTypeConvert(rawData, convertToType):
    """
    Attempt to convert rawData to the specified convertToType.
    """

    def allowedElemType(elemType):
        return (elemType in [int, float, bool, object]
                or numpy.issubdtype(elemType, numpy.number))

    if convertToType is None:
        if hasattr(rawData, 'dtype') and not allowedElemType(rawData.dtype):
            rawData = rawData.astype(numpy.object_)
        return rawData
    if (isinstance(rawData, numpy.ndarray)
            or (scipy and isinstance(rawData, scipy.sparse.spmatrix))
            or (pd and isinstance(rawData, (pd.DataFrame, pd.Series)))):
        try:
            converted = rawData.astype(convertToType)
            if not allowedElemType(converted.dtype):
                converted = rawData.astype(numpy.object_)
            return converted
        except (TypeError, ValueError) as err:
            error = err
    # otherwise we assume data follows conventions of a list
    if convertToType not in [object, numpy.object_] and len(rawData) > 0:
        if not hasattr(rawData[0], '__len__'):
            rawData = [rawData] # make 2D
        try:
            convertedData = []
            for point in rawData:
                convertedData.append(list(map(convertToType, point)))
            return convertedData
        except (ValueError, TypeError) as err:
            error = err
    else:
        return rawData
    # if nothing has been returned, we cannot convert to the convertToType
    msg = 'Unable to convert the data to convertToType '
    msg += "'{0}'. ".format(convertToType) + str(error)
    raise InvalidArgumentValue(msg)

def replaceNumpyValues(data, toReplace, replaceWith):
    """
    Replace values in a numpy array.

    Parameters
    ----------
    data : numpy.ndarray
        A numpy array of data.
    toReplace : list
        A list of values to search and replace in the data.
    replaceWith : value
        The value which will replace any values in ``toReplace``.
    """
    # if data has numeric dtype and replacing with a numeric value, do not
    # process any non-numeric values since it can only contain numeric values
    if (numpy.issubclass_(data.dtype.type, numpy.number)
            and isinstance(replaceWith, (int, float, numpy.number))):
        toReplace = [val for val in toReplace
                     if isinstance(val, (int, float, numpy.number))]
        toReplace = numpy.array(toReplace)
    else:
        toReplace = numpy.array(toReplace, dtype=numpy.object_)

    # numpy.isin cannot handle nan replacement, so if nan is in
    # toReplace we instead set the flag to trigger nan replacement
    replaceNan = any(isinstance(val, float) and numpy.isnan(val)
                     for val in toReplace)

    # try to avoid converting dtype if possible for efficiency.
    try:
        data[numpy.isin(data, toReplace)] = replaceWith
        if replaceNan:
            data[data != data] = replaceWith
    except ValueError:
        data = data.astype(numpy.object_)
        data[numpy.isin(data, toReplace)] = replaceWith
        if replaceNan:
            data[data != data] = replaceWith
    return data


def replaceMissingData(rawData, treatAsMissing, replaceMissingWith):
    """
    Convert any values in rawData found in treatAsMissing with
    replaceMissingWith value.
    """
    # need to convert SparseDataFrame to coo matrix before handling missing
    if pd.nimbleAccessible() and isinstance(rawData, pd.SparseDataFrame):
        rawData = scipy.sparse.coo_matrix(rawData)

    if isinstance(rawData, (list, tuple)):
        handleMissing = numpy.array(rawData, dtype=numpy.object_)
        handleMissing = replaceNumpyValues(handleMissing, treatAsMissing,
                                           replaceMissingWith)
        rawData = handleMissing.tolist()

    elif isinstance(rawData, numpy.ndarray):
        rawData = replaceNumpyValues(rawData, treatAsMissing,
                                           replaceMissingWith)

    elif scipy.sparse.issparse(rawData):
        handleMissing = replaceNumpyValues(rawData.data, treatAsMissing,
                                           replaceMissingWith)
        rawData.data = handleMissing

    elif (pd.nimbleAccessible()
            and isinstance(rawData, (pd.DataFrame, pd.Series))):
        if len(rawData.values) > 0:
            # .where keeps the values that return True, use ~ to replace those
            # values instead
            rawData = rawData.where(~rawData.isin(treatAsMissing),
                                    replaceMissingWith)

    return rawData

class SparseCOORowIterator:
    """
    Iterate through the nonZero values of a coo matrix by row.
    """
    def __init__(self, data):
        self.data = data
        self.rowIdx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.rowIdx < self.data.shape[0]:
            ret = self.data.data[self.data.row == self.rowIdx]
            self.rowIdx += 1
            return ret
        else:
            raise StopIteration

class GenericPointIterator:
    """
    Iterate through a list-like object row by row.

    This iterator optimizes performance when objects contain nimble data
    objects. Recursively iterating through nimble objects is more costly
    so copying to a python list whenever the object is not a point
    vector is much more efficient. If the object does not contain any
    nimble objects this is effectively the same as using iter().
    """
    def __init__(self, data):
        self.iterator = iter(data)

    def __iter__(self):
        return self

    def __next__(self):
        val = next(self.iterator)
        if isinstance(val, Base) and 1 not in val.shape:
            return val.copy('python list')
        return val

def getPointIterator(data):
    """
    Generate an iterator for the points in the object.
    """
    if isinstance(data, Base):
        return data.points
    if isinstance(data, numpy.matrix):
        return iter(numpy.array(data))
    if isinstance(data, dict):
        return iter(data.values())
    if (pd.nimbleAccessible()
            and isinstance(data, (pd.DataFrame, pd.Series))):
        return iter(data.values)
    if scipy.nimbleAccessible() and scipy.sparse.isspmatrix(data):
        return SparseCOORowIterator(data.tocoo(False))
    return GenericPointIterator(data)

def isHighDimensionData(rawData, skipDataProcessing):
    """
    Identify data with more than two-dimensions.
    """
    if scipy.nimbleAccessible() and scipy.sparse.isspmatrix(rawData):
        if not rawData.data.size:
            return False
        rawData = [rawData.data]
    try:
        if isAllowedSingleElement(rawData[0]):
            if (not skipDataProcessing and
                    not all(map(isAllowedSingleElement, rawData))):
                msg = "Numbers, strings, None, and nan are the only values "
                msg += "allowed in nimble data objects"
                raise InvalidArgumentValue(msg)
            return False
        if isAllowedSingleElement(rawData[0][0]):
            if not skipDataProcessing:
                toIter = getPointIterator(rawData)
                firstLength = len(next(toIter))
                for i, point in enumerate(toIter):
                    if not len(point) == firstLength:
                        msg = "All points in the data do not have the same "
                        msg += "number of features. The first point had {0} "
                        msg += "features but the point at index {1} had {2} "
                        msg += "features"
                        msg = msg.format(firstLength, i, len(point))
                        raise InvalidArgumentValue(msg)
                    if not all(map(isAllowedSingleElement, point)):
                        msg = "Numbers, strings, None, and nan are the only "
                        msg += "values allowed in nimble data objects"
                        raise InvalidArgumentValue(msg)
            return False
        else:
            return True
    except KeyError: # rawData or rawData[0] is dict
        return False
    except IndexError: # rawData or rawData[0] is empty
        return False
    except (ImproperObjectAction, InvalidArgumentType): # high dimension Base
        return True
    except TypeError: # invalid non-subscriptable object
        msg = "Numbers, strings, None, and nan are the only "
        msg += "values allowed in nimble data objects"
        raise InvalidArgumentValue(msg)

def highDimensionNames(rawData, pointNames, featureNames):
    """
    Names cannot be extracted at higher dimensions because the elements
    are not strings. If 'automatic' we can set to False, if True an
    exception must be raised. If a list, the length must align with
    the dimensions.
    are not strings so 'automatic' must be set to False and an exception
    must be raised if either is set to True.
    """
    if any(param is True for param in (pointNames, featureNames)):
        failedAxis = []
        if pointNames is True:
            failedAxis.append('point')
        if featureNames is True:
            failedAxis.append('feature')
        if failedAxis:
            axes = ' and '.join(failedAxis)
            msg = '{} names cannot be True for data with more '.format(axes)
            msg += "than two dimensions "
            raise InvalidArgumentValue(msg)

    pointNames = False if pointNames == 'automatic' else pointNames
    featureNames = False if featureNames == 'automatic' else featureNames

    return pointNames, featureNames

def validateDataLength(actual, expected):
    if actual != expected:
        msg = 'Inconsistent data lengths in object. Expected lengths of '
        msg += '{0} based on the first available object at '.format(expected)
        msg += 'dimension, but found length {0}'.format(actual)
        raise InvalidArgumentValue(msg)

def flattenToOneDimension(data, dimensions=False, toFill=None):
    """
    Recursive function to flatten an object.

    Flattened values are placed in toFill and this function also records
    the object's dimensions prior to being flattened. getPointIterator
    always return a point-based iterator for these cases so data is
    flattened point by point.
    """
    if toFill is None:
        toFill = []
    returnDims = dimensions is True
    if returnDims:
        dimensions = [True, [len(data)]]
    elif dimensions is False:
        dimensions = [False]
    elif dimensions[0]:
        dimensions[1].append(len(data))
    try:
        if all(map(isAllowedSingleElement, data)):
            toFill.extend(data)
        else:
            for obj in data:
                flattenToOneDimension(obj, dimensions, toFill)
                dimensions[0] = False
    except TypeError:
        msg = "Numbers, strings, None, and nan are the only "
        msg += "values allowed in nimble data objects"
        raise InvalidArgumentValue(msg)

    if returnDims:
        return toFill, dimensions[1]
    return toFill

def flattenHighDimensionFeatures(rawData):
    """
    Flatten data with multi-dimensional features to vectors.

    Features are flattened point by point whether numpy.reshape or
    flattenToOneDimension are used.
    """
    if isinstance(rawData, numpy.ndarray) and rawData.dtype != numpy.object_:
        origDims = rawData.shape
        newShape = (rawData.shape[0], numpy.prod(rawData.shape[1:]))
        rawData = numpy.reshape(rawData, newShape)
    else:
        if hasattr(rawData, 'shape'):
            numPts = rawData.shape[0]
        else:
            numPts = len(rawData)
        points = getPointIterator(rawData)
        firstPoint = next(points)
        firstPointFlat, ptDims = flattenToOneDimension(firstPoint, True)
        origDims = [numPts] + ptDims
        numFts = len(firstPointFlat)
        rawData = numpy.empty((numPts, numFts), dtype=numpy.object_)
        rawData[0] = firstPointFlat
        for i, point in enumerate(points):
            flat = flattenToOneDimension(point, False)
            numVals = len(flat)
            if numVals != numFts:
                msg = "The number of values in the point at index {0} ({1}) "
                msg += "is not equal to the number of values in first point "
                msg += "({2}). All points must contain an equal number of "
                msg += "values to allow nimble to flatten this data so it "
                msg += "can be represented in our data objects"
                raise InvalidArgumentValue(msg.format(i + 1, numVals, numFts))
            rawData[i + 1] = flat

    return rawData, tuple(origDims)

def initDataObject(
        returnType, rawData, pointNames, featureNames, name=None, path=None,
        keepPoints='all', keepFeatures='all', convertToType=None,
        reuseData=False, treatAsMissing=(float('nan'), numpy.nan, None, '',
                                         'None', 'nan', 'NULL', 'NA'),
        replaceMissingWith=numpy.nan, skipDataProcessing=False,
        extracted=(None, None)):
    """
    1. Setup autoType
    2. Extract Names
    3. Convert to 2D representation
    4. Handle Missing data
    5. Convert to acceptable form for returnType init
    """
    if returnType is None:
        if ((scipy.nimbleAccessible() and scipy.sparse.issparse(rawData))
            or (pd.nimbleAccessible()
                and isinstance(rawData, pd.SparseDataFrame))):
            returnType = 'Sparse'
        else:
            returnType = 'Matrix'

    if isinstance(rawData, Base):
        # point/featureNames, treatAsMissing, etc. may vary
        rawData = rawData.data
    if not reuseData:
        rawData = copy.deepcopy(rawData)

    # record if extraction occurred before we possibly modify *Names parameters
    ptsExtracted = extracted[0] if extracted[0] else pointNames is True
    ftsExtracted = extracted[1] if extracted[1] else featureNames is True

    # If skipping data processing, no modification needs to be made
    # to the data, so we can skip name extraction and missing replacement.
    kwargs = {}
    # convert these types as indexing may cause dimensionality confusion
    if isinstance(rawData, numpy.matrix):
        rawData = numpy.array(rawData)
    if scipy.nimbleAccessible() and scipy.sparse.isspmatrix(rawData):
        rawData = rawData.tocoo()
    import time
    s = time.time()
    if isHighDimensionData(rawData, skipDataProcessing):
        # additional name validation / processing before extractNames
        pointNames, featureNames = highDimensionNames(rawData, pointNames,
                                                      featureNames)
        rawData, tensorShape = flattenHighDimensionFeatures(rawData)
        kwargs['shape'] = tensorShape

    if skipDataProcessing:
        if returnType == 'List':
            kwargs['checkAll'] = False
        pointNames = pointNames if pointNames != 'automatic' else None
        featureNames = featureNames if featureNames != 'automatic' else None
    else:
        rawData, pointNames, featureNames = extractNames(rawData, pointNames,
                                                         featureNames)
        if treatAsMissing is not None:
            rawData = replaceMissingData(rawData, treatAsMissing,
                                         replaceMissingWith)
    # convert to convertToType, if necessary
    rawData = convertData(returnType, rawData, pointNames, featureNames,
                          convertToType)

    pathsToPass = (None, None)
    if path is not None:
        # used in data type unit testing, need a way to specify path values
        if isinstance(path, tuple):
            pathsToPass = path
        else:
            if path.startswith('http'):
                pathsToPass = (path, None)
            elif os.path.isabs(path):
                absPath = path
                relPath = os.path.relpath(path)
                pathsToPass = (absPath, relPath)
            else:
                absPath = os.path.abspath(path)
                relPath = path
                pathsToPass = (absPath, relPath)

    initMethod = getattr(nimble.data, returnType)
    # if limiting data based on keepPoints or keepFeatures,
    # delay name setting because names may be a subset
    if keepPoints == 'all':
        usePNames = pointNames
    else:
        usePNames = True if pointNames is True else None
    if keepFeatures == 'all':
        useFNames = featureNames
    else:
        useFNames = True if featureNames is True else None

    ret = initMethod(rawData, pointNames=usePNames,
                     featureNames=useFNames, name=name, paths=pathsToPass,
                     reuseData=reuseData, **kwargs)

    def makeCmp(keepList, outerObj, axis):
        if axis == 'point':
            def indexGetter(x):
                return outerObj.points.getIndex(x.points.getName(0))
        else:
            def indexGetter(x):
                return outerObj.features.getIndex(x.features.getName(0))
        positions = {}
        for i in range(len(keepList)):
            positions[keepList[i]] = i

        def retCmp(view1, view2):
            i1 = indexGetter(view1)
            i2 = indexGetter(view2)
            if positions[i1] < positions[i2]:
                return -1
            elif positions[i1] > positions[i2]:
                return 1
            else:
                return 0

        return retCmp

    # keep points and features if still needed
    if keepPoints != 'all':
        if not ptsExtracted and len(keepPoints) == len(ret.points):
            _raiseKeepLengthConflict('point')
        # if we have all pointNames, set them now
        if (isinstance(pointNames, (list, dict))
                and len(pointNames) == len(ret.points)):
            ret.points.setNames(pointNames, useLog=False)
            setPtNamesAfter = False
        else:
            _keepIndexValuesValidation('point', keepPoints, pointNames)
            setPtNamesAfter = True
        cleaned = []
        for val in keepPoints:
            converted = ret.points.getIndex(val)
            if converted not in cleaned:
                cleaned.append(converted)
        if len(cleaned) == len(ret.points):
            pCmp = makeCmp(cleaned, ret, 'point')
            ret.points.sort(sortHelper=pCmp)
        else:
            ret = ret.points.copy(cleaned)
        # if we had a subset of pointNames can set now on the cleaned data
        if setPtNamesAfter:
            ret.points.setNames(pointNames, useLog=False)
    if keepFeatures != 'all':
        if not ftsExtracted and len(keepFeatures) == len(ret.features):
            _raiseKeepLengthConflict('feature')
        # if we have all featureNames, set them now
        if (isinstance(featureNames, (list, dict))
                and len(featureNames) == len(ret.features)):
            ret.features.setNames(featureNames, useLog=False)
            setFtNamesAfter = False
        # otherwise we require keepFeatures to be index and set names later
        else:
            _keepIndexValuesValidation('feature', keepFeatures, featureNames)
            setFtNamesAfter = True
        cleaned = []
        for val in keepFeatures:
            converted = ret.features.getIndex(val)
            if converted not in cleaned:
                cleaned.append(converted)

        if len(cleaned) == len(ret.features):
            fCmp = makeCmp(cleaned, ret, 'feature')
            ret.features.sort(sortHelper=fCmp)
        else:
            ret = ret.features.copy(cleaned)
        # if we had a subset of featureNames can set now on the cleaned data
        if setFtNamesAfter:
            ret.features.setNames(featureNames, useLog=False)

    return ret


def createDataFromFile(
        returnType, data, pointNames, featureNames, name,
        ignoreNonNumericalFeatures, keepPoints, keepFeatures, inputSeparator,
        treatAsMissing, replaceMissingWith):
    """
    Helper for createData which deals with the case of loading data
    from a file. Returns a triple containing the raw data, pointNames,
    and featureNames (the later two following the same semantics as
    createData's parameters with the same names).
    """

    def autoFileTypeChecker(ioStream):
        # find first non empty line to check header
        startPosition = ioStream.tell()
        currLine = ioStream.readline()
        while currLine == ['', b'']:
            currLine = ioStream.readline()
        header = currLine
        ioStream.seek(startPosition)

        # check beginning of header for sentinel on string or binary stream
        if isinstance(header, bytes):
            if header[:14] == b"%%MatrixMarket":
                return 'mtx'
            if header.strip().endswith(b'\x89HDF'):
                return 'hdf5'
        if header[:14] == "%%MatrixMarket":
            return 'mtx'
        # we default to csv otherwise
        return 'csv'

    toPass = data
    # Case: string value means we need to open the file, either directly or
    # through an http request
    if isinstance(toPass, str):
        if toPass[:4] == 'http':
            if not requests.nimbleAccessible():
                msg = "To load data from a webpage, the requests module must "
                msg += "be installed"
                raise PackageException(msg)
            response = requests.get(data, stream=True)
            if not response.ok:
                msg = "The data could not be accessed from the webpage. "
                msg += "HTTP Status: {0}, ".format(response.status_code)
                msg += "Reason: {0}".format(response.reason)
                raise InvalidArgumentValue(msg)

            # start with BytesIO since mtx and hdf5 need them
            toPass = BytesIO(response.content)
            extension = autoFileTypeChecker(toPass)
            if extension == 'csv':
                toPass.close()
                toPass = StringIO(response.text, newline=None)
        else:
            toPass = open(data, 'r', newline=None)
            try:
                extension = autoFileTypeChecker(toPass)
            except UnicodeDecodeError:
                toPass.close()
                toPass = open(data, 'rb', newline=None)
                extension = autoFileTypeChecker(toPass)
    # Case: we are given an open file already
    else:
        extension = autoFileTypeChecker(toPass)

    # if the file has a different, valid extension from the one we determined
    # we will defer to the file's extension
    if isinstance(data, str):
        path = data
    else:
        # try getting name attribute from file
        try:
            path = data.name
        except AttributeError:
            path = None
    if path is not None:
        split = path.rsplit('.', 1)
        supportedExtensions = ['csv', 'mtx', 'hdf5', 'h5']
        if len(split) > 1 and split[1].lower() in supportedExtensions:
            extension = split[1].lower()
            if extension == 'h5':
                extension = 'hdf5' # h5 and hdf5 are synonymous

    if extension == 'csv':
        loader = _loadcsvUsingPython
    elif extension == 'mtx':
        loader = _loadmtxForAuto
    elif extension == 'hdf5':
        loader = _loadhdf5ForAuto

    # want to make sure we close the file if loading fails
    try:
        loaded = loader(
            toPass, pointNames, featureNames, ignoreNonNumericalFeatures,
            keepPoints, keepFeatures, inputSeparator=inputSeparator)
    finally:
        toPass.close()

    retData, retPNames, retFNames, selectSuccess = loaded

    # auto set name if unspecified, and is possible
    if isinstance(data, str):
        path = data
    elif hasattr(data, 'name'):
        path = data.name
    else:
        path = None

    if path is not None and name is None:
        tokens = path.rsplit(os.path.sep)
        name = tokens[len(tokens) - 1]

    extracted = (pointNames is True, featureNames is True)
    if selectSuccess:
        keepPoints = 'all'
        keepFeatures = 'all'
    # no guarantee that names were dealt with in this case
    else:
        if retPNames is None and isinstance(pointNames, (list, dict)):
            retPNames = pointNames
        if retFNames is None and isinstance(featureNames, (list, dict)):
            retFNames = featureNames

    return initDataObject(
        returnType, retData, retPNames, retFNames, name, path,
        keepPoints, keepFeatures, treatAsMissing=treatAsMissing,
        replaceMissingWith=replaceMissingWith, extracted=extracted)


def _loadmtxForAuto(
        openFile, pointNames, featureNames, ignoreNonNumericalFeatures,
        keepPoints, keepFeatures, **kwargs):
    """
    Uses scipy helpers to read a matrix market file; returning whatever
    is most appropriate for the file. If it is a matrix market array
    type, a numpy dense matrix is returned as data, if it is a matrix
    market coordinate type, a sparse scipy coo_matrix is returned as
    data. If featureNames are present, they are also read.
    """
    if not scipy.nimbleAccessible():
        msg = "scipy is not available"
        raise PackageException(msg)
    startPosition = openFile.tell()
    seenPNames = False
    retPNames = None
    retFNames = None

    # read through the comment lines
    while True:
        currLine = openFile.readline()
        if currLine[0] != '%':
            break
        if len(currLine) > 1 and currLine[1] == "#":
            # strip '%#' from the begining of the line
            scrubbedLine = currLine[2:]
            # strip newline from end of line
            scrubbedLine = scrubbedLine.rstrip()
            names = scrubbedLine.split(',')
            if not seenPNames:
                retPNames = names if names != [''] else None
                seenPNames = True
            else:
                retFNames = names if names != [''] else None

    openFile.seek(startPosition)
    try:
        data = scipy.io.mmread(openFile)
    except TypeError:
        if hasattr(openFile, 'name'):
            tempName = openFile.name
        else:
            tempName = openFile.inner.name
        data = scipy.io.mmread(tempName)

    temp = (data, None, None)

    if pointNames is True or featureNames is True:
        # the helpers operate based on positional inputs with a None
        # sentinal indicating no extration. So we need to convert from
        # the createData input semantics
#        pNameID = 0 if pointNames is True else None
#        fNameID = 0 if featureNames is True else None
        if scipy.sparse.issparse(data):
            temp = extractNamesFromScipySparse(data, pointNames, featureNames)
        else:
            temp = extractNamesFromNumpy(data, pointNames, featureNames)

    # choose between names extracted automatically from comments
    # (retPNames) vs names extracted explicitly from the data
    # (extPNames). extPNames has priority.
    (data, extPNames, extFNames) = temp
    retPNames = extPNames if retPNames is None else retPNames
    retFNames = extFNames if retFNames is None else retFNames

    return (data, retPNames, retFNames, False)


def _loadhdf5ForAuto(
        openFile, pointNames, featureNames, ignoreNonNumericalFeatures,
        keepPoints, keepFeatures, **kwargs):
    """
    Use h5py module to load high dimension data. The openFile is used
    to create a h5py.File object. Each Group and Dataset in the object
    are considered a deeper dimension. If pointNames is True, the keys
    of the initial file object will be used as points, otherwise, the
    point and feature name validity will be assessed in initDataObject.
    """
    if not h5py.nimbleAccessible():
        msg = 'loading hdf5 files requires the h5py module'
        raise PackageException(msg)

    def extractArray(obj):
        arrays = []
        if isinstance(obj, h5py.Dataset):
            # Ellipsis extracts the numpy array
            return obj[...]
        else:
            for value in obj.values():
                arrays.append(extractArray(value))
        return arrays

    with h5py.File(openFile, 'r') as hdf:
        data = []
        pnames = []
        expShape = None
        for key, val in hdf.items():
            ptData = extractArray(val)
            ptShape = numpy.array(ptData).shape
            if expShape is None:
                expShape = ptShape
            elif expShape != ptShape:
                msg = 'Each point in the data must have the same shape. '
                msg += "The data in key '{k}' had shape {act} but the first "
                msg += 'point had shape {exp}'
                msg = msg.format(k=key, act=ptShape, exp=expShape)
                raise InvalidArgumentValue(msg)
            pnames.append(key)
            data.append(ptData)

    # by default 'automatic' will only assign point names if we identify
    # this was a file generated by nimble where includeNames was True.
    openFile.seek(0)
    includePtNames = openFile.readline().startswith(b'includePointNames')
    if pointNames == 'automatic' and includePtNames:
        pointNames = pnames
    elif pointNames == 'automatic':
        pointNames = None
    # if we only have one Dataset, we will default to returning only the
    # array contained in the Dataset unless the pointNames indicate that
    # we should keep the array contained in a single point
    if len(data) == 1 and pointNames and pointNames is not True:
        numNames = len(pointNames)
        innerShape = data[0].shape[0]
        # point names are for the array in the dataset
        if numNames == innerShape:
            data = data[0]
        elif numNames > 1:
            msg = 'This file contains a single Dataset. The length of '
            msg += 'pointNames can either be 1, indicating the data in the '
            msg += 'Dataset will be loaded as a single point, or '
            msg += '{0} the data in the Dataset will be loaded directly, but '
            msg += 'pointNames contained {1} names'
            raise InvalidArgumentValue(msg.format(innerShape, numNames))
    elif len(data) == 1 and not pointNames:
        data = data[0]

    if pointNames is True or (includePtNames and pointNames is None):
        pointNames = pnames

    return (data, pointNames, featureNames, False)


def extractNamesFromNumpy(data, pnamesID, fnamesID):
    """
    Extract name values from a numpy array.
    """
    # if there are no elements, extraction cannot happen. We return correct
    # results for this case so it is excluded from the subsequent code
    if 0 in data.shape:
        return data, None, None

    # we allow single dimension arrays as input, but we assume 2d from here
    # forward; reshape so that the values constitute a single row.
    addedDim = False
    if len(data.shape) == 1:
        data = data.reshape(1, data.shape[0])
        addedDim = True

    def cleanRow(npRow):
        return list(map(_intFloatOrString, list(numpy.array(npRow).flatten())))
    firstRow = cleanRow(data[0]) if len(data) > 0 else None
    secondRow = cleanRow(data[1]) if len(data) > 1 else None
    pnamesID, fnamesID = autoDetectNamesFromRaw(pnamesID, fnamesID, firstRow,
                                                secondRow)
    pnamesID = 0 if pnamesID is True else None
    fnamesID = 0 if fnamesID is True else None

    retPNames = None
    retFNames = None
    if pnamesID is not None:
        retPNames = numpy.array(data[:, pnamesID]).flatten()
        data = numpy.delete(data, pnamesID, 1)
        if isinstance(fnamesID, int):
            retPNames = numpy.delete(retPNames, fnamesID)
        retPNames = numpy.vectorize(str)(retPNames)
        retPNames = list(retPNames)
    if fnamesID is not None:
        retFNames = numpy.array(data[fnamesID]).flatten()
        data = numpy.delete(data, fnamesID, 0)
        retFNames = numpy.vectorize(str)(retFNames)
        retFNames = list(retFNames)

    if addedDim:
        data = data.reshape(data.shape[1])

    return (data, retPNames, retFNames)


def extractNamesFromScipySparse(rawData, pointNames, featureNames):
    """
    Takes a scipy sparse data object, extracts names if needed, and
    returns a coo_matrix of the remaining data and names (if they were
    extracted).

    Parameters
    ----------
    rawData : a scipy sparse data object
    pointNames : bool, 'automatic', explicit names (which are ignored)
    featureNames : bool, 'automatic', explicit names (which are ignored)

    Returns
    -------
    a triple : coo_matrix; None or a pointnames; None or featureNames
    """
    ret = extractNamesFromCooDirect(rawData, pointNames, featureNames)

    return ret

def extractNamesFromCooDirect(data, pnamesID, fnamesID):
    """
    Extract names from a scipy sparse coo matrix.
    """
    if not scipy.sparse.isspmatrix_coo(data):
        data = scipy.sparse.coo_matrix(data)
    # gather up the first two rows of entries, to check for automatic name
    # extraction.
#    import pdb
#    pdb.set_trace()
    if fnamesID == 'automatic' or pnamesID == 'automatic':
        firstRow = [0] * data.shape[1]
        secondRow = [0] * data.shape[1]
        for i, val in enumerate(data.data):
            if data.row[i] == 0:
                firstRow[data.col[i]] = val
            if data.row[i] == 1:
                secondRow[data.col[i]] = val

        pnamesID, fnamesID = autoDetectNamesFromRaw(pnamesID, fnamesID,
                                                    firstRow, secondRow)

    fnamesID = 0 if fnamesID is True else None
    pnamesID = 0 if pnamesID is True else None

    # run through the entries in the returned coo_matrix to get
    # point / feature Names.

    # We justify this time expense by noting that unless this
    # matrix has an inappropriate number of non-zero entries,
    # the names we find will likely be a significant proportion
    # of the present data.

    # these will be ID -> name mappings
    if not scipy.nimbleAccessible():
        msg = "scipy is not available"
        raise PackageException(msg)
    tempPointNames = {}
    tempFeatureNames = {}

    newLen = len(data.data)
    if pnamesID is not None:
        newLen -= data.shape[0]
    if fnamesID is not None:
        newLen -= data.shape[1]
    if (pnamesID is not None) and (fnamesID is not None):
        newLen += 1

    newRows = numpy.empty(newLen, dtype=data.row.dtype)
    newCols = numpy.empty(newLen, dtype=data.col.dtype)
    newData = numpy.empty(newLen, dtype=data.dtype)
    writeIndex = 0
    # adjust the sentinal value for easier index modification
    pnamesID = sys.maxsize if pnamesID is None else pnamesID
    fnamesID = sys.maxsize if fnamesID is None else fnamesID
    for i in range(len(data.data)):
        row = data.row[i]
        setRow = row if row < fnamesID else row - 1
        col = data.col[i]
        setCol = col if col < pnamesID else col - 1
        val = data.data[i]

        colEq = col == pnamesID
        rowEq = row == fnamesID

        # a true value entry, copy it over
        if not colEq and not rowEq:
            # need to adjust the row/col values if we are past the
            # vector of names
            newRows[writeIndex] = setRow
            newCols[writeIndex] = setCol
            newData[writeIndex] = val
            writeIndex += 1
        # inidicates a point name
        elif colEq and not rowEq:
            if str(val) in tempPointNames:
                msg = "The point name " + str(val) + " was given more "
                msg += "than once in this file"
                raise InvalidArgumentValue(msg)
            tempPointNames[setRow] = str(val)
        # indicates a feature name
        elif rowEq and not colEq:
            if str(val) in tempFeatureNames:
                msg = "The feature name " + str(val) + " was given more "
                msg += "than once in this file"
                raise InvalidArgumentValue(msg)
            tempFeatureNames[setCol] = str(val)
        # intersection of point and feature names. ignore
        else:
            pass

    inTup = (newData, (newRows, newCols))
    rshape = data.shape[0] if fnamesID == sys.maxsize else data.shape[0] - 1
    cshape = data.shape[1] if pnamesID == sys.maxsize else data.shape[1] - 1
    data = scipy.sparse.coo_matrix(inTup, shape=(rshape, cshape))

    # process our results: fill in a zero entry if missing on
    # each axis and validate
    def processTempNames(temp, axisName, axisNum):
        retNames = []
        zeroPlaced = None
        for i in range(data.shape[axisNum]):
            if i not in temp:
                if zeroPlaced is not None:
                    msg = axisName + " names not fully specified in the "
                    msg += "data, at least one of the rows "
                    msg += str(zeroPlaced) + " and " + str(i) + " must "
                    msg += "have a non zero value"
                    raise InvalidArgumentValue(msg)
                # make a zero of the same dtype as the data
                name = str(numpy.array([0], dtype=data.dtype)[0])
                zeroPlaced = i
            else:
                name = temp[i]
            if name in retNames:
                msg = "The " + axisName + " name " + name + " was "
                msg += "given more than once in this file"
                raise InvalidArgumentValue(msg)
            retNames.append(name)
        return retNames

    retPNames = None
    if tempPointNames != {}:
        retPNames = processTempNames(tempPointNames, 'point', 0)
    retFNames = None
    if tempFeatureNames != {}:
        retFNames = processTempNames(tempFeatureNames, 'feature', 1)

    return (data, retPNames, retFNames)


def _intFloatOrString(inString):
    """
    Try to convert strings to numeric types or empty strings to None.
    """
    ret = inString
    try:
        ret = int(inString)
    except ValueError:
        ret = float(inString)
    # this will return an int or float if either of the above are successful
    finally:
        if ret == "":
            return None
        return ret

def _intFloatBoolOrString(inString):
    """
    Expand on _intFloatOrString to convert True/False strings to bool.
    """
    ret = _intFloatOrString(inString)
    if ret == 'True':
        return True
    if ret == 'False':
        return False
    return ret

typeHierarchy = {bool: 0, int: 1, float: 2, str: 3}

def _csvColTypeTracking(row, convertCols, nonNumericFeatures):
    """
    Track the possible types of values in each column.

    Store the most complex type for values in each column according to
    the typeHierarchy.  Boolean columns must be exclusively boolean
    values (ignoring None), otherwise the potential boolean values will
    be assumed to be strings instead. Once a column has been determined
    to have a str type, it is removed from convertCols because it will
    only store columns that require conversion and the columns index is
    added to nonNumericFeatures.
    """
    delKeys = []
    if convertCols:
        for idx, currType in convertCols.items():
            colVal = _intFloatBoolOrString(row[idx])
            colType = type(colVal)
            if colVal is not None and colType != currType:
                if typeHierarchy[colType] > typeHierarchy[currType]:
                    if currType is bool or colType is str:
                        delKeys.append(idx)
                    else:
                        convertCols[idx] = colType
                elif colType is bool:
                    delKeys.append(idx)

    if delKeys:
        for key in delKeys:
            del convertCols[key]
            nonNumericFeatures.append(key)

def _colTypeConversion(row, convertCols):
    """
    Converts values in each row that are in numeric/boolean columns.

    Since convertCols does not contain non-numeric columns, any empty
    string is considered to be a missing value.
    """
    for idx, cType in convertCols.items():
        val = row[idx]
        if val == '':
            row[idx] = None
        elif cType is bool and val == 'False':
            # bool('False') would return True since val is still a string
            row[idx] = False
        else:
            row[idx] = cType(val)

def isEmptyRaw(raw):
    """
    Determine if raw data contains no values.
    """
    if raw is None:
        return True
    if raw == []:
        return True
    if hasattr(raw, 'shape') and raw.shape[0] == 0:
        return True

    return False

def autoDetectNamesFromRaw(pointNames, featureNames, firstValues,
                           secondValues):
    """
    Determine if first row or column contains names.
    """
    failPN = isinstance(pointNames, bool) and pointNames
    failFN = isinstance(featureNames, bool) and featureNames
    if isEmptyRaw(firstValues):
        return (failPN, failFN)
    if isEmptyRaw(secondValues):
        return (failPN, failFN)
    if featureNames is False:
        return (failPN, failFN)

    def teq(double):
        x, y = double
        return not isinstance(x, type(y))

    if ((pointNames is True or pointNames == 'automatic')
            and firstValues[0] == 'pointNames'):
        allText = all(map(lambda x: isinstance(x, str),
                          firstValues[1:]))
        allDiff = all(map(teq, zip(firstValues[1:], secondValues[1:])))
    else:
        allText = all(map(lambda x: isinstance(x, str),
                          firstValues))
        allDiff = all(map(teq, zip(firstValues, secondValues)))

    if featureNames == 'automatic' and allText and allDiff:
        featureNames = True
    # At this point, there is no chance to resolve 'automatic' to True
    if featureNames == 'automatic':
        featureNames = False

    if featureNames is True and pointNames == 'automatic':
        if firstValues[0] == 'pointNames':
            pointNames = True
    # At this point, there is no chance to resolve 'automatic' to True
    if pointNames == 'automatic':
        pointNames = False

    return (pointNames, featureNames)

def _checkCSV_for_Names(openFile, pointNames, featureNames, dialect):
    """
    Will check for triggers to automatically determine the positions of
    the point or feature names if they have not been specified by the
    user. For feature names the trigger is two empty lines prior to
    the first line of data. For point names the trigger is the first
    line of data contains the feature names, and the first value of that
    line is 'pointNames'
    """
    startPosition = openFile.tell()

    # walk past all the comments
    currLine = "#"
    while currLine.startswith('#'):
        currLine = openFile.readline()

    # check for two empty lines in a row to denote that first
    # data line contains feature names
    if currLine.strip() == '':
        currLine = openFile.readline()
        if currLine.strip() == '':
            # only change set value if we allow detection
            if featureNames == 'automatic':
                # we set this so the names are extracted later
                featureNames = True

    # Use the robust csv reader to read the first two lines (if available)
    # these are saved to used in further autodection
    openFile.seek(startPosition)
    rowReader = csv.reader(openFile, dialect)
    try:
        firstDataRow = next(rowReader)
        while firstDataRow == []:
            firstDataRow = next(rowReader)
        secondDataRow = next(rowReader)
        while secondDataRow == []:
            secondDataRow = next(rowReader)
    except StopIteration:
        firstDataRow = None
        secondDataRow = None

    if firstDataRow is not None:
        firstDataRow = list(map(_intFloatOrString, firstDataRow))
    if secondDataRow is not None:
        secondDataRow = list(map(_intFloatOrString, secondDataRow))
    (pointNames, featureNames) = autoDetectNamesFromRaw(
        pointNames, featureNames, firstDataRow, secondDataRow)

    # reset everything to make the loop easier
    openFile.seek(startPosition)

    return (pointNames, featureNames)


def _filterCSVRow(row):
    if len(row) == 0:
        return False
    if row[0] == '\n':
        return False
    return True


def _advancePastComments(openFile):
    """
    Take an open file and advance until we find a line that isn't empty
    and doesn't start with the comment character. Returns the number
    of lines that were skipped
    """
    numSkipped = 0
    while True:
        # If we read a row that isn't a comment line, we
        # have to undo our read
        startPosition = openFile.tell()

        row = openFile.readline()
        if len(row) == 0:
            numSkipped += 1
            continue
        if row[0] == '#':
            numSkipped += 1
            continue
        if row[0] == '\n':
            numSkipped += 1
            continue

        openFile.seek(startPosition)
        break

    return numSkipped


def _namesDictToList(names, kind, paramName):
    if not isinstance(names, dict):
        return names

    ret = [None] * len(names)
    for key in names:
        position = names[key]
        if ret[position] is not None:
            msg = "The dict valued parameter " + paramName + " contained "
            msg += "two keys with the same value. Interpreted as names, "
            msg += "this means that two " + kind + "s had the same name, "
            msg += "which is disallowed."
            raise InvalidArgumentValue(msg)

        if position < 0 or position >= len(ret):
            msg = "The dict valued parameter " + paramName + " contained "
            msg += "a key with a value (" + position + "), yet the only "
            msg += "acceptable possible position values would be in the "
            msg += "range 0 to " + str(len(ret))
            raise InvalidArgumentValue(msg)

        ret[position] = key

    return ret

def _detectDialectFromSeparator(openFile, inputSeparator):
    "find the dialect to pass to csv.reader based on inputSeparator"
    startPosition = openFile.tell()
    # skip commented lines
    _ = _advancePastComments(openFile)
    if inputSeparator == 'automatic':
        # detect the delimiter from the first line of data
        dialect = csv.Sniffer().sniff(openFile.readline())
    elif len(inputSeparator) > 1:
        msg = "inputSeparator must be a single character"
        raise InvalidArgumentValue(msg)
    elif inputSeparator == '\t':
        dialect = csv.excel_tab
    else:
        dialect = csv.excel
        dialect.delimiter = inputSeparator

    # reset everything to make the loop easier
    openFile.seek(startPosition)

    return dialect


def _checkForDuplicates(lst, varName):
    duplicates = set(x for x in lst if lst.count(x) > 1)
    if duplicates:
        msg = "{var} cannot contain duplicate values. "
        if len(duplicates) == 1:
            duplicateString = str(list(duplicates)[0])
            msg += "The value {val} was duplicated"
        else:
            duplicateString = ",".join(map(str, duplicates))
            msg += "The values {val} were duplicated"
        msg = msg.format(var=varName, val=duplicateString)
        raise InvalidArgumentValue(msg)


def _keepIndexValuesValidation(axis, keepList, nameList):
    """
    Preliminary validation when keepPoints/Features can only contain
    index values.
    """
    keep = 'keepPoints' if axis == 'point' else 'keepFeatures'
    for idVal in keepList:
        # cannot determine the index location of the feature by name since
        # featureNames is only defining the names of the returned features
        if isinstance(idVal, str) and nameList:
            msg = "Since {axis}Names were only provided for the values in "
            msg += "{keep}, {keep} can contain only index values referencing "
            msg += "the {axis}'s location in the data. If attempting to use "
            msg += "{keep} to reorder all {axis}s, instead create the object "
            msg += "first then sort the {axis}s."
            msg = msg.format(axis=axis, keep=keep)
            raise InvalidArgumentValue(msg)
        if isinstance(idVal, str):
            msg = "{keep} can contain only index values because no "
            msg += "{axis}Names were provided"
            msg = msg.format(axis=axis, keep=keep)
            raise InvalidArgumentValue(msg)
        if idVal < 0:
            msg = "Negative index values are not permitted, found "
            msg += "{value} in {keep}"
            msg = msg.format(value=idVal, keep=keep)
            raise InvalidArgumentValue(msg)


def _raiseKeepIndexNameConflict(axis, index, name):
    """
    Helper for raising exception when two values in keepPoints/Features
    represent the same point/feature.
    """
    keep = 'keepPoints' if axis == 'point' else 'keepFeatures'
    msg = "{keep} cannot contain duplicate values. The index {index} and the "
    msg += "name '{name}' represent the same {axis} and are both in {keep} "
    msg = msg.format(keep=keep, index=index, name=name, axis=axis)
    raise InvalidArgumentValue(msg)


def _raiseKeepLengthConflict(axis):
    """
    Helper to prevent defining keepPoints/Features for every point or
    feature because it cannot be determined whether the list is defining
    the values in the order of the data or order of keepPoints/Features.
    """
    keep = 'keepPoints' if axis == 'point' else 'keepFeatures'
    msg = "The length of {keep} cannot be the same as the number of {axis}s. "
    msg += "If attempting to use {keep} to keep and/or reorder all {axis}s, "
    msg += "instead create the object using {keep}='all', then sort the "
    msg += "{axis}s."
    msg = msg.format(keep=keep, axis=axis)
    raise InvalidArgumentValue(msg)

def _limitToKeptFeatures(keepFeatures, retFNames):
    """
    Limit the featureNames to only those in keepFeatures.

    Returns a two-tuple of lists. The first list converts all values in
    keepFeatures to indices and the second provides the featureNames for
    the values in keepFeatures.
    """
    keepIndices = []
    keepNames = []
    for ftID in keepFeatures:
        if isinstance(ftID, str) and ftID in retFNames:
            idx = retFNames.index(ftID)
            if idx in keepIndices:
                _raiseKeepIndexNameConflict('feature', idx, ftID)
            keepIndices.append(idx)
            keepNames.append(ftID)
        elif isinstance(ftID, str):
            msg = "The value '{v}' in keepFeatures is not a valid featureName"
            msg = msg.format(v=ftID)
            raise InvalidArgumentValue(msg)
        # index values
        elif 0 <= ftID < len(retFNames):
            name = retFNames[ftID]
            if name in keepNames:
                _raiseKeepIndexNameConflict('feature', ftID, name)
            keepIndices.append(ftID)
            keepNames.append(name)
        elif ftID >= 0:
            msg = "The index {idx} is greater than the number of features in "
            msg += "the data, {numFts}"
            msg = msg.format(idx=ftID, numFts=len(retFNames))
            raise InvalidArgumentValue(msg)
        else:
            msg = "Negative index values are not permitted, found {v} in "
            msg += "keepFeatures"
            msg = msg.format(v=ftID)
            raise InvalidArgumentValue(msg)

    return keepIndices, keepNames


def _loadcsvUsingPython(openFile, pointNames, featureNames,
                        ignoreNonNumericalFeatures, keepPoints, keepFeatures,
                        **kwargs):
    """
    Loads a csv file using a reader from python's csv module.

    Parameters
    ----------
    openFile : open file like object
        The data will be read from where the file currently points to.
    pointNames : 'automatic', bool, list, dict
        May be 'automatic', True, False, a list or a dict. The first
        value indicates to detect whether pointNames should be extracted
        or not. True indicates the first column of values is to be taken
        as the pointNames. False indicates that the names are not
        embedded. Finally, the names may have been provided by the user
        as a list or dict, meaning nothing is extracted, and those
        objects are passed on in the return value.
    featureNames : 'automatic', bool, list, dict
        May be 'automatic', True, False, a list or a dict. The first
        value indicates to detect whether featureNames should be
        extracted or not. True indicates the first row of values is to
        be taken as the featureNames. False indicates that the names are
        not embedded. Finally, the names may have been provided by the
        user as a list or dict, meaning nothing is extracted, and those
        objects are passed on in the return value.
    ignoreNonNumericalFeatures : bool
        Value indicating whether, when loading from a file, features
        containing non numercal data shouldn't be loaded into the final
        object. For example, you may be loading a file which has a
        column of strings; setting this flag to true will allow you to
        load that file into a Matrix object (which may contain floats
        only). If there is point or feature selection occurring, then
        only those values within selected points and features are
        considered when determining whether to apply this operation.
    keepPoints : 'all', list
        The value 'all' indicates that all possible points found in the
        file will be included. Alternatively, may be a list containing
        either names or indices (or a mix) of those points they want to
        be selected from the raw data.
    keepFeatures : 'all', list
        The value 'all' indicates that all possible features found in
        the file will be included. Alternatively, may be a list
        containing either names or indices (or a mix) of those features
        they want to be selected from the raw data.

    Returns
    -------
    tuple
        The data read from the file, pointNames (those extracted from
        the data, or the same value as passed in), featureNames (same
        sematics as pointNames), and either True or False indicating if
        the keepPoints and keepFeatures parameters were applied in this
        function call.
    """
    inputSeparator = kwargs['inputSeparator']
    dialect = _detectDialectFromSeparator(openFile, inputSeparator)

    (pointNames, featureNames) = _checkCSV_for_Names(
        openFile, pointNames, featureNames, dialect)

    pointNames = _namesDictToList(pointNames, 'point', 'pointNames')
    featureNames = _namesDictToList(featureNames, 'feature', 'featureNames')

    # Advance the file past any beginning of file comments, record
    # how many are skipped
    skippedLines = _advancePastComments(openFile)
    # remake the file iterator to ignore empty lines
    filtered = filter(_filterCSVRow, openFile)
    # send that line iterator to the csv reader
    lineReader = csv.reader(filtered, dialect)

    firstRowLength = None
    if featureNames is True:
        retFNames = next(lineReader)
        if pointNames is True:
            retFNames = retFNames[1:]
        firstRowLength = len(retFNames)
        lengthDefiningLine = skippedLines
        skippedLines += 1
    else:
        retFNames = featureNames

    # for many cases, we can determine if values in keepFeatures are valid
    # indices now but when keepFeaturesValidated is False we need to wait
    # until we have access to the first point before we can determine if
    # index values are valid
    keepFeaturesValidated = True

    # at this stage, retFNames is either a list or False
    # modifications are necessary if limiting features
    limitFeatures = keepFeatures != 'all'
    limitPoints = keepPoints != 'all'
    if limitFeatures:
        _checkForDuplicates(keepFeatures, 'keepFeatures')
    if (limitFeatures and retFNames
            and (len(retFNames) != len(keepFeatures) or featureNames is True)):
        # have all featureNames
        keepFeatures, retFNames = _limitToKeptFeatures(keepFeatures, retFNames)
    elif limitFeatures:
        # none or a subset of the featureNames provided
        _keepIndexValuesValidation('feature', keepFeatures, featureNames)
        keepFeaturesValidated = False
    if limitPoints:
        _checkForDuplicates(keepPoints, 'keepPoints')
        # none or a subset of the pointNames provided
        if (not pointNames or
                (pointNames is not True
                 and len(pointNames) == len(keepPoints))):
            _keepIndexValuesValidation('point', keepPoints, pointNames)

    extractedPointNames = []
    nonNumericFeatures = set()
    locatedPoints = []
    retData = []
    totalPoints = 0
    if limitPoints:
        # we want to insert the points and names in the desired order
        retData = [None] * len(keepPoints)
        extractedPointNames = [None] * len(keepPoints)

    convertCols = None
    nonNumericFeatures = []
    # lineReader is now at the first line of data
    for i, row in enumerate(lineReader):
        if pointNames is True:
            ptName = row[0]
            row = row[1:]
        elif pointNames and len(pointNames) > len(keepPoints):
            ptName = pointNames[i]
        else:
            ptName = None
        if not keepFeaturesValidated:
            if any(val >= len(row) for val in keepFeatures):
                for val in keepFeatures:
                    if val > len(row):
                        msg = "The index " + str(val) + " is outside the "
                        msg += "range of possible indices, 0 to "
                        msg += str(len(row) - 1)
                        raise InvalidArgumentValue(msg)

        if keepPoints != 'all' and i in keepPoints and ptName in keepPoints:
            _raiseKeepIndexNameConflict('point', i, ptName)
        # this point will be used
        if keepPoints == 'all' or i in keepPoints or ptName in keepPoints:
            if firstRowLength is None:
                firstRowLength = len(row)
                lengthDefiningLine = skippedLines + i + 1
            elif firstRowLength != len(row):
                delimiter = dialect.delimiter
                msg = "The row on line " + str(skippedLines + i + 1) + " has "
                msg += "length " + str(len(row)) + ". We expected length "
                msg += str(firstRowLength) + ". The expected row length was "
                msg += "defined by looking at the row on line "
                msg += str(lengthDefiningLine) + " and using '" + delimiter
                msg += "' as the separator."
                raise FileFormatException(msg)
            if not keepFeaturesValidated:
                if limitFeatures and len(keepFeatures) == firstRowLength:
                    _raiseKeepLengthConflict('feature')
                # only need to do the validation once
                keepFeaturesValidated = True
            if limitFeatures:
                limitedRow = []
                for ftID in keepFeatures:
                    limitedRow.append(row[ftID])
                row = limitedRow

            if convertCols is None: # first row
                convertCols = {}
                for idx, val in enumerate(row):
                    colType = type(_intFloatBoolOrString(val))
                    if colType is not str:
                        convertCols[idx] = colType
                    else:
                        nonNumericFeatures.append(idx)
            else: # continue to track column types in subsequent rows
                _csvColTypeTracking(row, convertCols, nonNumericFeatures)

            if keepPoints == 'all':
                retData.append(row)
                if pointNames is True:
                    extractedPointNames.append(ptName)
            else:
                if ptName is not None and ptName in keepPoints:
                    locate = ptName
                elif i in keepPoints:
                    locate = i
                else:
                    locate = pointNames[i]
                location = keepPoints.index(locate)
                locatedPoints.append(locate)
                retData[location] = row
                if pointNames is True:
                    extractedPointNames[location] = ptName
        totalPoints = i + 1

    if (keepPoints != 'all' and pointNames
            and pointNames is not True
            and len(retData) == totalPoints == len(keepPoints)):
        _raiseKeepLengthConflict('point')

    if limitPoints and len(keepPoints) != len(locatedPoints):
        unlocated = []
        for ptID in keepPoints:
            if ptID not in locatedPoints:
                unlocated.append(ptID)
        msg = "The points " + ",".join(map(str, unlocated)) + " were not "
        msg += "found in the data"
        raise InvalidArgumentValue(msg)

    if convertCols:
        for row in retData:
            _colTypeConversion(row, convertCols)

    if ignoreNonNumericalFeatures:
        if retFNames:
            retFNames = [retFNames[i] for i in range(len(retFNames))
                         if i not in nonNumericFeatures]
        removeNonNumeric = []
        for row in retData:
            removeNonNumeric.append([row[i] for i in range(len(row))
                                     if i not in nonNumericFeatures])
        retData = removeNonNumeric

    if pointNames is True:
        retPNames = extractedPointNames
    elif pointNames and len(retData) == len(pointNames):
        retPNames = pointNames
    elif pointNames:
        # we need to limit pointNames to kept points
        retPNames = []
        for ptID in keepPoints:
            if isinstance(ptID, str):
                retPNames.append(ptID)
            else:
                retPNames.append(pointNames[ptID])
    else:
        retPNames = pointNames

    return (retData, retPNames, retFNames, True)

def registerCustomLearnerBackend(customPackageName, learnerClassObject, save):
    """
    Backend for registering custom Learners in nimble.

    A save value of true will run saveChanges(), modifying the config
    file. When save is False the changes will exist only for that
    session, unless saveChanges() is called later in the session.
    """
    # detect name collision
    for currInterface in nimble.interfaces.available:
        if not isinstance(currInterface,
                          nimble.interfaces.CustomLearnerInterface):
            if currInterface.isAlias(customPackageName):
                msg = "The customPackageName '" + customPackageName
                msg += "' cannot be used: it is an accepted alias of a "
                msg += "non-custom package"
                raise InvalidArgumentValue(msg)

    # do validation before we potentially construct an interface to a
    # custom package
    nimble.customLearners.CustomLearner.validateSubclass(learnerClassObject)

    try:
        currInterface = findBestInterface(customPackageName)
    except InvalidArgumentValue:
        currInterface = nimble.interfaces.CustomLearnerInterface(
            customPackageName)
        nimble.interfaces.available.append(currInterface)

    currInterface.registerLearnerClass(learnerClassObject)

    opName = customPackageName + "." + learnerClassObject.__name__
    opValue = learnerClassObject.__module__ + '.' + learnerClassObject.__name__

    nimble.settings.set('RegisteredLearners', opName, opValue)
    if save:
        nimble.settings.saveChanges('RegisteredLearners', opName)

    # check if new option names introduced, call sync if needed
    if learnerClassObject.options() != []:
        nimble.configuration.setInterfaceOptions(nimble.settings,
                                                 currInterface, save=save)


def deregisterCustomLearnerBackend(customPackageName, learnerName, save):
    """
    Backend for deregistering custom Learners in nimble.

    A save value of true will run saveChanges(), modifying the config
    file. When save is False the changes will exist only for that
    session, unless saveChanges() is called later in the session.
    """
    currInterface = findBestInterface(customPackageName)
    if not isinstance(currInterface, nimble.interfaces.CustomLearnerInterface):
        msg = "May only attempt to deregister learners from the interfaces of "
        msg += "custom packages. '" + customPackageName
        msg += "' is not a custom package"
        raise InvalidArgumentType(msg)
    origOptions = currInterface.optionNames
    empty = currInterface.deregisterLearner(learnerName)
    newOptions = currInterface.optionNames

    # remove options
    for optName in origOptions:
        if optName not in newOptions:
            nimble.settings.delete(customPackageName, optName)
            if save:
                nimble.settings.saveChanges(customPackageName, optName)

    if empty:
        nimble.interfaces.available.remove(currInterface)
        #remove section
        nimble.settings.delete(customPackageName, None)
        if save:
            nimble.settings.saveChanges(customPackageName)

    regOptName = customPackageName + '.' + learnerName
    # delete from registered learner list
    nimble.settings.delete('RegisteredLearners', regOptName)
    if save:
        nimble.settings.saveChanges('RegisteredLearners', regOptName)


def countWins(predictions):
    """
    Count how many contests were won by each label in the set.  If a
    class label doesn't win any predictions, it will not be included in
    the results.  Return a dictionary: {classLabel: # of contests won}.
    """
    predictionCounts = {}
    for prediction in predictions:
        if prediction in predictionCounts:
            predictionCounts[prediction] += 1
        else:
            predictionCounts[prediction] = 1

    return predictionCounts


def extractWinningPredictionLabel(predictions):
    """
    Provided a list of tournament winners (class labels) for one
    point/row in a test set, choose the label that wins the most
    tournaments.  Returns the winning label.
    """
    #Count how many times each class won
    predictionCounts = countWins(predictions)

    #get the class that won the most tournaments
    #TODO: what if there are ties?
    return max(predictionCounts.keys(),
               key=(lambda key: predictionCounts[key]))


def extractWinningPredictionIndex(predictionScores):
    """
    Provided a list of confidence scores for one point/row in a test
    set, return the index of the column (i.e. label) of the highest
    score.  If no score in the list of predictionScores is a number
    greater than negative infinity, returns None.
    """
    maxScore = float("-inf")
    maxScoreIndex = -1
    for i in range(len(predictionScores)):
        score = predictionScores[i]
        if score > maxScore:
            maxScore = score
            maxScoreIndex = i

    if maxScoreIndex == -1:
        return None
    else:
        return maxScoreIndex


def extractWinningPredictionIndexAndScore(predictionScores, featureNamesItoN):
    """
    Provided a list of confidence scores for one point/row in a test
    set, return the index of the column (i.e. label) of the highest
    score.  If no score in the list of predictionScores is a number
    greater than negative infinity, returns None.
    """
    allScores = extractConfidenceScores(predictionScores, featureNamesItoN)

    if allScores is None:
        return None
    else:
        bestScore = float("-inf")
        bestLabel = None
        for key in allScores:
            value = allScores[key]
            if value > bestScore:
                bestScore = value
                bestLabel = key
        return (bestLabel, bestScore)


def extractConfidenceScores(predictionScores, featureNamesItoN):
    """
    Provided a list of confidence scores for one point/row in a test
    set, and a dict mapping indices to featureNames, return a dict
    mapping featureNames to scores.
    """
    if predictionScores is None or len(predictionScores) == 0:
        return None

    scoreMap = {}
    for i in range(len(predictionScores)):
        score = predictionScores[i]
        label = featureNamesItoN[i]
        scoreMap[label] = score

    return scoreMap


def computeMetrics(dependentVar, knownData, predictedData,
                   performanceFunction):
    """
    Calculate the performance of the learner.

    Using the provided metric, compare the known data or labels to the
    predicted data or labels and calculate the performance of the
    learner which produced the predicted data.

    Parameters
    ----------
    dependentVar : indentifier, list, nimble Base object
        Indicate the feature names or indices in knownData containing
        the known labels, or a data object that contains the known
        labels.
    knownData : nimble Base object
        Data object containing the known labels of the training set, as
        well as the features of the training set. Can be None if
        'dependentVar' is an object containing the labels.
    predictedData : nimble Base object
        Data object containing predicted labels/data. Assumes that the
        predicted label (or labels) in the nth row of predictedLabels
        is associated with the same data point/instance as the label in
        the nth row of knownLabels.
    performanceFunction : function
        A python function that returns a single numeric value evaluating
        performance. The function must take either two or three args.
        In the two arg case, they must be two sets of data or labels to
        be compared. In the three arg case, the first two args are the
        same as in the two arg case, and the third arg must take the
        value of what is to be considered the negative label in this
        binary classification problem. See nimble.calculate for a number
        of builtin options.

    Returns
    -------
    Value
        Measurement of the performance of the learner that produced the
        given data.
    """
    if dependentVar is None or isinstance(dependentVar, Base):
        #The known Indicator argument already contains all known
        #labels, so we do not need to do any further processing
        knownLabels = dependentVar
    else:
        #known Indicator is a feature ID or group of IDs; we extract the
        # columns it indicates from knownValues
        knownLabels = knownData.features.copy(dependentVar, useLog=False)

    result = performanceFunction(knownLabels, predictedData)

    return result


def generateAllPairs(items):
    """
    Given a list of items, generate a list of all possible pairs
    (2-combinations) of items from the list, and return as a list
    of tuples.  Assumes that no two items in the list refer to the same
    object or number.  If there are duplicates in the input list, there
    will be duplicates in the output list.
    """
    if items is None or len(items) == 0:
        return None

    pairs = []
    for i in range(len(items)):
        firstItem = items[i]
        for j in range(i + 1, len(items)):
            secondItem = items[j]
            pair = (firstItem, secondItem)
            pairs.append(pair)

    return pairs


class KFoldCrossValidator():
    """
    Perform k-fold cross-validation and store the results.

    On instantiation, cross-validation will be performed.  The results
    can be accessed through the object's attributes and methods.

    Parameters
    ----------
    learnerName : str
        nimble compliant algorithm name in the form 'package.algorithm'
        e.g. 'sciKitLearn.KNeighborsClassifier'
    X : nimble Base object
        points/features data
    Y : nimble Base object
        labels/data about points in X
    performanceFunction : function
        Premade options are available in nimble.calculate.
        Function used to evaluate the performance score for each run.
        Function is of the form: def func(knownValues, predictedValues).
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To trigger cross-validation using multiple values for arguments,
        specify different values for each parameter using a nimble.CV
        object. eg. {'k': nimble.CV([1,3,5])} will generate an error
        score for  the learner when the learner was passed all three
        values of ``k``, separately. These will be merged any
        kwarguments for the learner.
    folds : int
        The number of folds used in the cross validation. Can't exceed
        the number of points in X, Y.
    scoreMode : str
        Used by computeMetrics.
    useLog : bool, None
        Local control for whether to send results/timing to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. To trigger cross-validation using multiple values for
        arguments, specify different values for each parameter using a
        nimble.CV object.
        eg. arg1=nimble.CV([1,2,3]), arg2=nimble.CV([4,5,6])
        which correspond to permutations/argument states with one
        element from arg1 and one element from arg2, such that an
        example generated permutation/argument state would be
        ``arg1=2, arg2=4``. Will be merged with ``arguments``.

    Attributes
    ----------
    learnerName : str
        The learner used for training.
    performanceFunction : function
        The performance function that will or has been used during
        cross-validation.
    folds : int
        The number of folds that will or has been used during
        cross-validation.
    scoreMode : str
        The scoreMode set for training.
    arguments : dict
        A dictionary of the merged arguments and kwarguments.
    allResults : list
        Each dictionary in the returned list will contain a permutation
        of the arguments and the performance of that permutation. A list
        of dictionaries containing each argument permutation and its
        performance based on the ``performanceFunction``.  The key to
        access the performance value will be the __name__ attribute of
        the ``performanceFunction``. If the ``performanceFunction`` has
        no __name__ attribute or is a lambda function the key will be
        set to 'performance'.
    bestArguments : dict
        The argument permutation names and values which provided the
        optimal result according to the ``performanceFunction``.
    bestResult
        The optimal output value from the ``performanceFunction``.
    """
    def __init__(self, learnerName, X, Y, performanceFunction, arguments=None,
                 folds=10, scoreMode='label', useLog=None, **kwarguments):
        self.learnerName = learnerName
        # detectBestResult will raise exception for invalid performanceFunction
        detected = nimble.calculate.detectBestResult(performanceFunction)
        self.maximumIsOptimal = detected == 'max'
        self.performanceFunction = performanceFunction
        self.folds = folds
        self.scoreMode = scoreMode
        self.arguments = _mergeArguments(arguments, kwarguments)
        self._allResults = None
        self._bestArguments = None
        self._bestResult = None
        self._resultsByFold = []
        self._crossValidate(X, Y, useLog)

    def _crossValidate(self, X, Y, useLog):
        """
        Perform K-fold cross-validation on the data.

        Cross-validation will be performed based on the instantiation
        parameters for this instance.

        Parameters
        ----------
        X : nimble Base object
            points/features data
        Y : nimble Base object
            labels/data about points in X
        useLog : bool, None
            Local control for whether to send results/timing to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        """
        if not isinstance(X, Base):
            raise InvalidArgumentType("X must be a Base object")
        if Y is not None:
            if not isinstance(Y, (Base, int, str, list)):
                msg = "Y must be a Base object or an index (int) from X where "
                msg += "Y's data can be found"
                raise InvalidArgumentType(msg)
            if isinstance(Y, (int, str, list)):
                X = X.copy()
                Y = X.features.extract(Y, useLog=False)

            if len(Y.features) > 1 and self.scoreMode != 'label':
                msg = "When dealing with multi dimensional outputs / "
                msg += "predictions, then the scoreMode flag is required to "
                msg += "be set to 'label'"
                raise InvalidArgumentValueCombination(msg)

            if not len(X.points) == len(Y.points):
                #todo support indexing if Y is an index for X instead
                msg = "X and Y must contain the same number of points"
                raise InvalidArgumentValueCombination(msg)

        #get an iterator for the argument combinations- iterator
        #handles case of merged arguments being {}
        argumentCombinationIterator = ArgumentIterator(self.arguments)

        # we want the folds for each argument combination to be the same
        foldIter = FoldIterator([X, Y], self.folds)

        # setup container for outputs, a tuple entry for each arg set,
        # containing a list for the results of those args on each fold
        numArgSets = argumentCombinationIterator.numPermutations
        performanceOfEachCombination = []
        for i in range(numArgSets):
            performanceOfEachCombination.append([None, []])

        # control variables determining if we save all results before
        # calculating performance or if we can calculate for each fold and
        # then avg the results
        canAvgFolds = (hasattr(self.performanceFunction, 'avgFolds')
                       and self.performanceFunction.avgFolds)

        # folditerator randomized the point order, so if we are collecting all
        # the results, we also have to collect the correct order of the known
        # values
        if not canAvgFolds:
            collectedY = None

        # Folding should be the same for each argset (and is expensive) so
        # iterate over folds first
        for fold in foldIter:
            [(curTrainX, curTestingX), (curTrainY, curTestingY)] = fold
            argSetIndex = 0

            # given this fold, do a run for each argument combination
            for curArgumentCombination in argumentCombinationIterator:
                #run algorithm on the folds' training and testing sets
                curRunResult = nimble.trainAndApply(
                    learnerName=self.learnerName, trainX=curTrainX,
                    trainY=curTrainY, testX=curTestingX,
                    arguments=curArgumentCombination, scoreMode=self.scoreMode,
                    useLog=False)

                performanceOfEachCombination[argSetIndex][0] = (
                    curArgumentCombination)

                # calculate error of prediction, using performanceFunction
                # store fold error to CrossValidationResults
                curPerformance = computeMetrics(curTestingY, None,
                                                curRunResult,
                                                self.performanceFunction)
                self._resultsByFold.append((curArgumentCombination,
                                                 curPerformance))

                if canAvgFolds:
                    performanceOfEachCombination[argSetIndex][1].append(
                        curPerformance)
                else:
                    performanceOfEachCombination[argSetIndex][1].append(
                        curRunResult)

                argSetIndex += 1

            if not canAvgFolds:
                if collectedY is None:
                    collectedY = curTestingY
                else:
                    collectedY.points.append(curTestingY, useLog=False)

            # setup for next iteration
            argumentCombinationIterator.reset()

        # We consume the saved results, either by averaging the individual
        # results calculations for each fold, or combining the saved
        # predictions and calculating performance of the entire set.
        for i, (curArgSet, results) in enumerate(performanceOfEachCombination):
            # average score from each fold (works for one fold as well)
            if canAvgFolds:
                finalPerformance = sum(results) / float(len(results))
            # combine the results objects into one, and then calc performance
            else:
                for resultIndex in range(1, len(results)):
                    results[0].points.append(results[resultIndex],
                                             useLog=False)

                # TODO raise RuntimeError(
                #     "How do we guarantee Y and results are in same order?")
                finalPerformance = computeMetrics(collectedY, None, results[0],
                                                  self.performanceFunction)

            # we use the current results container to be the return value
            performanceOfEachCombination[i] = (curArgSet, finalPerformance)

        # store results
        self._allResults = performanceOfEachCombination

        handleLogging(useLog, 'crossVal', X, Y, self.learnerName,
                      self.arguments, self.performanceFunction,
                      performanceOfEachCombination, self.folds)

    @property
    def allResults(self):
        """
        Each argument permutation and its performance.

        Each dictionary in the returned list will contain a permutation
        of the arguments and the performance of that permutation. A list
        of dictionaries containing each argument permutation and its
        performance based on the ``performanceFunction``.  The key to
        access the performance value will be the __name__ attribute of
        the ``performanceFunction``. If the ``performanceFunction`` has
        no __name__ attribute or is a lambda function the key will be
        set to 'performance'.

        Returns
        -------
        list
            List of dictionaries.

        Examples
        --------
        >>> nimble.setRandomSeed(42)
        >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
        >>> yRaw = [[1], [2], [3],
        ...         [1], [2], [3],
        ...         [1], [2], [3]]
        >>> X = nimble.createData('Matrix', xRaw)
        >>> Y = nimble.createData('Matrix', yRaw)
        >>> crossValidator = KFoldCrossValidator(
        ...    'Custom.KNNClassifier', X, Y, arguments={'k': 3},
        ...    performanceFunction=nimble.calculate.fractionIncorrect,
        ...    folds=3)
        >>> crossValidator.allResults
        [{'k': 3, 'fractionIncorrect': 0.3333333333333333}]
        """
        resultsList = []
        for argSet, result in self._allResults:
            resultDict = argSet.copy()
            if (hasattr(self.performanceFunction, '__name__')
                    and self.performanceFunction.__name__ != '<lambda>'):
                resultDict[self.performanceFunction.__name__] = result
            else:
                resultDict['performance'] = result
            resultsList.append(resultDict)
        return resultsList

    @property
    def bestArguments(self):
        """
        The arguments permutation with the most optimal performance.

        Returns
        -------
        dict
            The argument permutation names and values which provided the
            optimal result according to the ``performanceFunction``.
        """
        if self._bestArguments is not None:
            return self._bestArguments
        bestResults = self._bestArgumentsAndResult()
        self._bestArguments = bestResults[0]
        self._bestResult = bestResults[1]
        return self._bestArguments

    @property
    def bestResult(self):
        """
        The performance value for the best argument permutation.

        Returns
        -------
        value
            The optimal output value from the ``performanceFunction``
            according to ``performanceFunction.optimal``.
        """
        if self._bestResult is not None:
            return self._bestResult
        bestResults = self._bestArgumentsAndResult()
        self._bestArguments = bestResults[0]
        self._bestResult = bestResults[1]
        return self._bestResult

    def getFoldResults(self, arguments=None, **kwarguments):
        """
        The result from each fold for a given permutation of arguments.

        Parameters
        ----------
        arguments : dict
            Dictionary of learner argument names and values. Will be
            merged with any kwarguments. After merge, must match an
            argument permutation generated during cross-validation.
        kwarguments
            Learner argument names and values as keywords. Will be
            merged with ``arguments``. After merge, must match an
            argument permutation generated during cross-validation.

        Returns
        -------
        list
            The ``performanceFunction`` results from each fold for this
            argument permutation.

        Examples
        --------
        >>> nimble.setRandomSeed(42)
        >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
        >>> yRaw = [[1], [2], [3],
        ...         [1], [2], [3],
        ...         [1], [2], [3]]
        >>> X = nimble.createData('Matrix', xRaw)
        >>> Y = nimble.createData('Matrix', yRaw)
        >>> kValues = nimble.CV([1, 3])
        >>> crossValidator = KFoldCrossValidator(
        ...    'Custom.KNNClassifier', X, Y, arguments={},
        ...    performanceFunction=nimble.calculate.fractionIncorrect,
        ...    folds=3, k=kValues)
        >>> crossValidator.getFoldResults(arguments={'k': 1})
        [0.3333333333333333, 0.0, 0.0]
        >>> crossValidator.getFoldResults(k=1)
        [0.3333333333333333, 0.0, 0.0]
        >>> crossValidator.getFoldResults({'k': 3})
        [0.3333333333333333, 0.6666666666666666, 0.0]
        """
        merged = _mergeArguments(arguments, kwarguments)
        foldErrors = []
        # self._resultsByFold is a list of two-tuples (argumentSet, foldScore)
        for argSet, score in self._resultsByFold:
            if argSet == merged:
                foldErrors.append(score)
        if not foldErrors:
            self._noMatchingArguments()
        return foldErrors

    def getResult(self, arguments=None, **kwarguments):
        """
        The result over all folds for a given permutation of arguments.

        Parameters
        ----------
        arguments : dict
            Dictionary of learner argument names and values. Will be
            merged with any kwarguments. After merge, must match an
            argument permutation generated during cross-validation.
        kwarguments
            Learner argument names and values as keywords. Will be
            merged with ``arguments``. After merge, must match an
            argument permutation generated during cross-validation.

        Returns
        -------
        value
            The output value of the ``performanceFunction`` for this
            argument permutation.

        Examples
        --------
        >>> nimble.setRandomSeed(42)
        >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
        >>> yRaw = [[1], [2], [3],
        ...         [1], [2], [3],
        ...         [1], [2], [3]]
        >>> X = nimble.createData('Matrix', xRaw)
        >>> Y = nimble.createData('Matrix', yRaw)
        >>> kValues = nimble.CV([1, 3])
        >>> crossValidator = KFoldCrossValidator(
        ...    'Custom.KNNClassifier', X, Y, arguments={},
        ...    performanceFunction=nimble.calculate.fractionIncorrect,
        ...    folds=3, k=kValues)
        >>> crossValidator.getResult(arguments={'k': 1})
        0.1111111111111111
        >>> crossValidator.getResult(k=1)
        0.1111111111111111
        >>> crossValidator.getResult({'k': 3})
        0.3333333333333333
        """
        merged = _mergeArguments(arguments, kwarguments)
        # self._allResults is a list of two-tuples (argumentSet, totalScore)
        for argSet, result in self._allResults:
            if argSet == merged:
                return result
        self._noMatchingArguments()

    def _bestArgumentsAndResult(self):
        """
        The best argument and result based on the performanceFunction.
        """
        bestArgumentAndScoreTuple = None
        for curResultTuple in self._allResults:
            _, curScore = curResultTuple
            #if curArgument is the first or best we've seen:
            #store its details in bestArgumentAndScoreTuple
            if bestArgumentAndScoreTuple is None:
                bestArgumentAndScoreTuple = curResultTuple
            else:
                if (self.maximumIsOptimal
                        and curScore > bestArgumentAndScoreTuple[1]):
                    bestArgumentAndScoreTuple = curResultTuple
                if (not self.maximumIsOptimal
                        and curScore < bestArgumentAndScoreTuple[1]):
                    bestArgumentAndScoreTuple = curResultTuple

        return bestArgumentAndScoreTuple

    def _noMatchingArguments(self):
        """
        Raise exception when passed arguments are not valid.
        """
        msg = "No matching argument sets found. Available argument sets are: "
        msg += ",".join(str(arg) for arg, _ in self._allResults)
        raise InvalidArgumentValue(msg)


class FoldIterator(object):
    """
    Create and iterate through folds.

    Parameters
    ----------
    dataList : list
        A list of data objects to divide into folds.
    folds : int
        The number of folds to create.
    """
    def __init__(self, dataList, folds):
        self.dataList = dataList
        if folds <= 0:
            msg = "Number of folds must be greater than 0"
            raise InvalidArgumentValue(msg)
        self.folds = folds
        self.foldList = self._makeFoldList()
        self.index = 0
        for dat in self.dataList:
            if dat is not None and dat.getTypeString() == 'Sparse':
                dat._sortInternal('point')

    def __iter__(self):
        return self

    def next(self):
        """
        Get next item.
        """
        if self.index >= len(self.foldList):
            raise StopIteration
        # we're going to be separating training and testing sets through
        # extraction, so we have to copy the data in order not to destroy the
        # original sets across multiple folds
        copiedList = []
        for data in self.dataList:
            if data is None:
                copiedList.append(None)
            else:
                copiedList.append(data.copy())

            # we want each training set to be permuted wrt its ordering in the
            # original data. This is setting up a permutation to be applied to
            # each object
            #		indices = range(0, len(copiedList[0].points)
            #                              - len(self.foldList[self.index])))
            #		pythonRandom.shuffle(indices)
        indices = numpy.arange(0, (len(copiedList[0].points)
                                   - len(self.foldList[self.index])))
        numpyRandom.shuffle(indices)

        resultsList = []
        for copied in copiedList:
            if copied is None:
                resultsList.append((None, None))
            else:
                currTest = copied.points.extract(self.foldList[self.index],
                                                 useLog=False)
                currTrain = copied
                currTrain.points.sort(sortHelper=indices, useLog=False)
                resultsList.append((currTrain, currTest))
        self.index = self.index + 1
        return resultsList

    def __next__(self):
        return self.next()

    def _makeFoldList(self):
        if self.dataList is None:
            raise InvalidArgumentType('dataList may not be None')
        if len(self.dataList) == 0:
            raise InvalidArgumentValue("dataList may not be or empty")

        points = len(self.dataList[0].points)
        for data in self.dataList:
            if data is not None:
                if len(data.points) == 0:
                    msg = "One of the objects has 0 points, it is impossible "
                    msg += "to specify a valid number of folds"
                    raise InvalidArgumentValueCombination(msg)
                if len(data.points) != len(self.dataList[0].points):
                    msg = "All data objects in the list must have the same "
                    msg += "number of points and features"
                    raise InvalidArgumentValueCombination(msg)

        # note: we want truncation here
        numInFold = int(points / self.folds)
        if numInFold == 0:
            msg = "Must specify few enough folds so there is a point in each"
            raise InvalidArgumentValue(msg)

        # randomly select the folded portions
        indices = list(range(points))
        pythonRandom.shuffle(indices)
        foldList = []
        for fold in range(self.folds):
            start = fold * numInFold
            if fold == self.folds - 1:
                end = points
            else:
                end = (fold + 1) * numInFold
            foldList.append(indices[start:end])
        return foldList

class ArgumentIterator(object):
    """
    Create and iterate through argument permutations.

    Parameters
    ----------
    rawArgumentInput : dict
        Mapping of argument names (strings) to values.
        e.g. {'a': CV([1, 2, 3]), 'b': nimble.CV([4,5]), 'c': 6}
    """

    def __init__(self, rawArgumentInput):
        self.rawArgumentInput = rawArgumentInput
        self.index = 0
        if not isinstance(rawArgumentInput, dict):
            msg = "ArgumentIterator objects require dictionary's to "
            msg += "initialize- e.g. {'a':CV([1,2,3]), 'b':CV([4,5])} This "
            msg += "is the form generated by **args in a function argument."
            raise InvalidArgumentType(msg)

        # i.e. if rawArgumentInput == {}
        if len(rawArgumentInput) == 0:
            self.numPermutations = 1
            self.permutationsList = [{}]
        else:
            iterableArgDict = {}
            self.numPermutations = 1
            for key in rawArgumentInput.keys():
                if isinstance(rawArgumentInput[key], nimble.CV):
                    self.numPermutations *= len(rawArgumentInput[key])
                    iterableArgDict[key] = rawArgumentInput[key]
                else: # numPermutations not increased
                    # wrap in iterable so that itertools.product will treat
                    # whatever this value is as a single argument value even
                    # if the value itself is an iterable
                    iterableArgDict[key] = (rawArgumentInput[key],)

            # note: calls to keys() and values() will directly correspond as
            # since no modification is made to iterableArgDict between calls.
            self.permutationsList = []
            for permutation in itertools.product(*iterableArgDict.values()):
                permutationDict = {}
                for i, argument in enumerate(iterableArgDict.keys()):
                    permutationDict[argument] = permutation[i]
                self.permutationsList.append(permutationDict)

            assert len(self.permutationsList) == self.numPermutations

    def __iter__(self):
        return self

    def next(self):
        """
        Get next item.
        """
        if self.index >= self.numPermutations:
            self.index = 0
            raise StopIteration
        else:
            permutation = self.permutationsList[self.index]
            self.index += 1
            return permutation

    def __next__(self):
        return self.next()

    def reset(self):
        """
        Reset index to 0.
        """
        self.index = 0


def generateClassificationData(labels, pointsPer, featuresPer):
    """
    Randomly generate sensible data for a classification problem.
    Returns a tuple of tuples, where the first value is a tuple
    containing (trainX, trainY) and the second value is a tuple
    containing (testX ,testY).
    """
    #add noise to the features only
    trainData, _, noiselessTrainLabels = generateClusteredPoints(
        labels, pointsPer, featuresPer, addFeatureNoise=True,
        addLabelNoise=False, addLabelColumn=False)
    testData, _, noiselessTestLabels = generateClusteredPoints(
        labels, 1, featuresPer, addFeatureNoise=True, addLabelNoise=False,
        addLabelColumn=False)

    return ((trainData, noiselessTrainLabels), (testData, noiselessTestLabels))


def generateRegressionData(labels, pointsPer, featuresPer):
    """
    Randomly generate sensible data for a regression problem. Returns a
    tuple of tuples, where the first value is a tuple containing
    (trainX, trainY) and the second value is a tuple containing
    (testX ,testY).
    """
    #add noise to both the features and the labels
    regressorTrainData, trainLabels, _ = generateClusteredPoints(
        labels, pointsPer, featuresPer, addFeatureNoise=True,
        addLabelNoise=True, addLabelColumn=False)
    regressorTestData, testLabels, _ = generateClusteredPoints(
        labels, 1, featuresPer, addFeatureNoise=True, addLabelNoise=True,
        addLabelColumn=False)

    return ((regressorTrainData, trainLabels), (regressorTestData, testLabels))

# with class-based refactor:
# todo add scale control as paramater for generateClusteredPoints
#  - remember to scale noise term accordingly
def generateClusteredPoints(numClusters, numPointsPerCluster,
                            numFeaturesPerPoint, addFeatureNoise=True,
                            addLabelNoise=True, addLabelColumn=False,
                            returnType='Matrix'):
    """
    Function to generate Data object with arbitrary number of points,
    number of clusters, and number of features.

    The function returns the dataset in an object, 'labels' for each
    point in the dataset (noise optional), and the 'noiseless' labels
    for the points, which is the central value used to define the
    feature values for each point.

    generateClusteredPoints() outputs a dataset of the following format:
    each point associated with a cluster has numFeaturesPerPoint
    features. The value of each entry in the feature vector is
    clusterNumber+noise. Each point in the cluster has the same feature
    vector, with different noise.

    NOTE: if addFeatureNoise and addLabelNoise are false, then the
    'clusters' are actually all contain just repeated points, where each
    point in the cluster has the same features and the same labels.

    Returns
    -------
    tuple of nimble.Base objects:
    (pointsObj, labelsObj, noiselessLabelsObj)
    """

    pointsList = []
    labelsList = []
    clusterNoiselessLabelList = []

    def _noiseTerm():
        return pythonRandom.random() * 0.0001 - 0.00005

    for curCluster in range(numClusters):
        for _ in range(numPointsPerCluster):
            curFeatureVector = [float(curCluster) for x
                                in range(numFeaturesPerPoint)]

            if addFeatureNoise:
                curFeatureVector = [_noiseTerm() + entry for entry
                                    in curFeatureVector]

            if addLabelNoise:
                curLabel = _noiseTerm() + curCluster
            else:
                curLabel = curCluster

            if addLabelColumn:
                curFeatureVector.append(curLabel)

            #append curLabel as a list to maintain dimensionality
            labelsList.append([curLabel])

            pointsList.append(curFeatureVector)
            clusterNoiselessLabelList.append([float(curCluster)])

    pointsArray = numpy.array(pointsList, dtype=numpy.float)
    labelsArray = numpy.array(labelsList, dtype=numpy.float)
    clusterNoiselessLabelArray = numpy.array(clusterNoiselessLabelList,
                                             dtype=numpy.float)
    # todo verify that your list of lists is valid initializer for all
    # datatypes, not just matrix
    # then convert
    # finally make matrix object out of the list of points w/ labels in last
    # column of each vector/entry:
    pointsObj = nimble.createData('Matrix', pointsArray, useLog=False)

    labelsObj = nimble.createData('Matrix', labelsArray, useLog=False)

    # todo change actuallavels to something like associatedClusterCentroid
    noiselessLabelsObj = nimble.createData('Matrix', clusterNoiselessLabelArray,
                                           useLog=False)

    # convert datatype if not matrix
    if returnType.lower() != 'matrix':
        pointsObj = pointsObj.copy(to=returnType)
        labelsObj = labelsObj.copy(to=returnType)
        noiselessLabelsObj = noiselessLabelsObj.copy(to=returnType)

    return (pointsObj, labelsObj, noiselessLabelsObj)


def sumAbsoluteDifference(dataOne, dataTwo):
    """
    Aggregates absolute difference between corresponding entries in base
    objects dataOne and dataTwo.

    Checks to see that the vectors (which must be base objects) are of
    the same shape, first. Next it iterates through the corresponding
    points in each vector/matrix and appends the absolute difference
    between corresponding points to a list.

    Finally, the function returns the sum of the absolute differences.
    """

    #compare shapes of data to make sure a comparison is sensible.
    if len(dataOne.features) != len(dataTwo.features):
        msg = "Can't calculate difference between corresponding entries in "
        msg += "dataOne and dataTwo, the underlying data has different "
        msg += "numbers of features."
        raise InvalidArgumentValueCombination(msg)
    if len(dataOne.points) != len(dataTwo.points):
        msg = "Can't calculate difference between corresponding entries in "
        msg += "dataOne and dataTwo, the underlying data has different "
        msg += "numbers of points."
        raise InvalidArgumentValueCombination(msg)

    numpyOne = dataOne.copy(to='numpyarray')
    numpyTwo = dataTwo.copy(to='numpyarray')

    differences = numpyOne - numpyTwo

    absoluteDifferences = numpy.abs(differences)

    sumAbsoluteDifferences = numpy.sum(absoluteDifferences)

    return sumAbsoluteDifferences


class LearnerInspector:
    """
    Class using heirustics to classify the 'type' of problem an
    algorithm is meant to work on.
    e.g. classification, regression, dimensionality reduction, etc.

    Use:
    A LearnerInspector object generates private datasets that are
    intentionally constructed to invite particular results when an
    algorithm is run on them. Once a user has a LearnerInspector object,
    she can call learnerType(algorithmName) and get the 'best guess'
    type for that algorithm.

    Note:
    If characterizing multiple algorithms, use the SAME LearnerInspector
    object, and call learnerType() once for each algorithm you are
    trying to classify.
    """

    def __init__(self):
        """
        Caches the regressor and classifier datasets, to speed up
        learnerType() calls for multiple learners.
        """
        # TODO why is it this value??? should see how it is used and revise
        self.NEAR_THRESHHOLD = .1
        self.EXACT_THRESHHOLD = .00000001

        #initialize datasets for tests
        self.regressorDataTrain, self.regressorDataTest = (
            self._regressorDataset())
        #todo use classifier
        self.classifierDataTrain, self.classifierDataTest = (
            self._classifierDataset())

    def learnerType(self, learnerName):
        """
        Returns, as a string, the heuristically determined best guess
        for the type of problem the learnerName learner is designed to
        run on.
        Example output: 'classification', 'regression', 'other'
        """
        if not isinstance(learnerName, str):
            raise InvalidArgumentType("learnerName must be a string")
        return self._classifyAlgorithmDecisionTree(learnerName)

    # todo pull from each 'trail' function to find out what possible results
    # it can have then make sure that you've covered all possible combinations
    def _classifyAlgorithmDecisionTree(self, learnerName):
        """
        Implements a decision tree based off of the predicted labels
        returned from the datasets.

        Fundamentally, if the classifier dataset has no error, that
        means the algorithm is likely a classifier, but it could be a
        regressor, if its error is low, however, the algorithm is likely
        a regressor, and if its error is high, or the algorithm crashes
        with the dataset, then the algorithm is likely neither
        classifier nor regressor.

        Next, if the classifier dataset had no error, we want to see if
        the error on the regressor dataset is low. Also, we want to see
        if the algorithm is capable of generating labels that it hasn't
        seen (interpolating a la a regressor).

        If the algorithm doesn't produce any new labels, despite no
        repeated labels, then we assume it is a classifier. If the error
        on the classifier dataset is low, however, and the algorithm
        interpolates labels, then we assume it is a regressor.
        """

        regressorTrialResult = self._regressorTrial(learnerName)
        classifierTrialResult = self._classifierTrial(learnerName)

        # decision tree:
        # if classifier tests gives exact results
        if classifierTrialResult == 'exact':
            # could be classifier or regressor at this point
            # if when given unrepeating labels, algorithm generates duplicate
            # of already seen labels, it is classifer
            if regressorTrialResult == 'repeated_labels':
                return 'classification'
            if regressorTrialResult == 'near':
                return 'regression'
            if regressorTrialResult == 'other':
                return 'classification'
            #should be covered by all cases, raise exception
            msg = 'Decision tree needs to be updated to account for other '
            msg += 'results from regressorTrialResult'
            raise AttributeError(msg)

        # if the classifer data set genereated a low error, but not exact,
        # it is regressor
        elif classifierTrialResult == 'near':
            return 'regression'

        # if the classifier dataset doesn't see classifier or regressor
        # behavior, return other
        # todo this is where to insert future sensors for other types of
        # algorithms, but currently we can only resolve classifiers,
        # regressors, and other.
        else:
            return 'other'

    def _regressorDataset(self):
        """
        Generates clustered points, where the labels of the points
        within a single cluster are all very similar, but non-identical.
        """

        clusterCount = 3
        pointsPer = 10
        featuresPer = 5

        #add noise to both the features and the labels
        regressorTrainData, trainLabels, noiselessTrainLabels = (
            generateClusteredPoints(clusterCount, pointsPer, featuresPer,
                                    addFeatureNoise=True, addLabelNoise=True,
                                    addLabelColumn=False))
        regressorTestData, testLabels, noiselessTestLabels = (
            generateClusteredPoints(clusterCount, 1, featuresPer,
                                    addFeatureNoise=True, addLabelNoise=True,
                                    addLabelColumn=False))

        return ((regressorTrainData, trainLabels, noiselessTrainLabels),
                (regressorTestData, testLabels, noiselessTestLabels))

    def _classifierDataset(self):
        """
        Generates clustered points, hwere the labels of the points
        within each cluster are all identical.
        """

        clusterCount = 3
        pointsPer = 10
        featuresPer = 5

        #add noise to the features only
        trainData, trainLabels, noiselessTrainLabels = (
            generateClusteredPoints(clusterCount, pointsPer, featuresPer,
                                    addFeatureNoise=True, addLabelNoise=False,
                                    addLabelColumn=False))
        testData, testLabels, noiselessTestLabels = (
            generateClusteredPoints(clusterCount, 1, featuresPer,
                                    addFeatureNoise=True, addLabelNoise=False,
                                    addLabelColumn=False))

        return ((trainData, trainLabels, noiselessTrainLabels),
                (testData, testLabels, noiselessTestLabels))

    def _regressorTrial(self, learnerName):
        """
        Run trainAndApply on the regressor dataset and make judgments
        about the learner based on the results of trainAndApply.
        """
        #unpack already-initialized datasets
        regressorTrainData, trainLabels, _ = self.regressorDataTrain
        regressorTestData, _, noiselessTestLabels = self.regressorDataTest

        try:
            runResults = nimble.trainAndApply(
                learnerName, trainX=regressorTrainData, trainY=trainLabels,
                testX=regressorTestData)
        except Exception:
            return 'other'

        try:
            sumError = sumAbsoluteDifference(runResults, noiselessTestLabels)
        except InvalidArgumentValueCombination:
            return 'other'

        # if the labels are repeated from those that were trained on, then
        # it is a classifier so pass back that labels are repeated
        # if runResults are all in trainLabels, then it's repeating:
        alreadySeenLabelsList = []
        for curPointIndex in range(len(trainLabels.points)):
            alreadySeenLabelsList.append(trainLabels[curPointIndex, 0])

        # check if the learner generated any new label
        # (one it hadn't seen in training)
        unseenLabelFound = False
        for curResultPointIndex in range(len(runResults.points)):
            if runResults[curResultPointIndex, 0] not in alreadySeenLabelsList:
                unseenLabelFound = True
                break

        if not unseenLabelFound:
            return 'repeated_labels'

        if sumError > self.NEAR_THRESHHOLD:
            return 'other'
        else:
            return 'near'


    def _classifierTrial(self, learnerName):
        """
        Run trainAndApply on the classifer dataset and make judgments
        about the learner based on the results of trainAndApply.
        """
        #unpack initialized datasets
        trainData, trainLabels, _ = self.classifierDataTrain
        testData, testLabels, _ = self.classifierDataTest

        try:
            runResults = nimble.trainAndApply(learnerName, trainX=trainData,
                                              trainY=trainLabels,
                                              testX=testData)
        except Exception:
            return 'other'

        try:
            sumError = sumAbsoluteDifference(runResults, testLabels) #should be identical to noiselessTestLabels
        except InvalidArgumentValueCombination:
            return 'other'

        if sumError > self.NEAR_THRESHHOLD:
            return 'other'
        elif sumError > self.EXACT_THRESHHOLD:
            return 'near'
        else:
            return 'exact'


def _validScoreMode(scoreMode):
    """
    Check that a scoreMode flag to train() trainAndApply(), etc. is an
    accepted value.
    """
    scoreMode = scoreMode.lower()
    if scoreMode not in ['label', 'bestscore', 'allscores']:
        msg ="scoreMode may only be 'label' 'bestScore' or 'allScores'"
        raise InvalidArgumentValue(msg)


def _validMultiClassStrategy(multiClassStrategy):
    """
    Check that a multiClassStrategy flag to train() trainAndApply(),
    etc. is an accepted value.
    """
    multiClassStrategy = multiClassStrategy.lower()
    if multiClassStrategy not in ['default', 'onevsall', 'onevsone']:
        msg = "multiClassStrategy may be 'default' 'OneVsAll' or 'OneVsOne'"
        raise InvalidArgumentValue(msg)


def _unpackLearnerName(learnerName):
    """
    Split a learnerName parameter into the portion defining the package,
    and the portion defining the learner.
    """
    splitList = learnerName.split('.', 1)
    if len(splitList) < 2:
        msg = "Recieved the ill formed learner name '" + learnerName + "'. "
        msg += "The learner name must identify both the desired package and "
        msg += "learner, separated by a dot. Example:'mlpy.KNN'"
        raise InvalidArgumentValue(msg)
    package = splitList[0]
    learnerName = splitList[1]
    return (package, learnerName)


def _validArguments(arguments):
    """
    Check that an arguments parmeter to train() trainAndApply(), etc. is
    an accepted format.
    """
    if not isinstance(arguments, dict) and arguments is not None:
        msg = "The 'arguments' parameter must be a dictionary or None"
        raise InvalidArgumentType(msg)


def _mergeArguments(argumentsParam, kwargsParam):
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


def _validData(trainX, trainY, testX, testY, testRequired):
    """
    Check that the data parameters to train() trainAndApply(), etc. are
    in accepted formats.
    """
    if not isinstance(trainX, Base):
        msg = "trainX may only be an object derived from Base"
        raise InvalidArgumentType(msg)

    if trainY is not None:
        if not isinstance(trainY, (Base, str, int, numpy.int64)):
            msg = "trainY may only be an object derived from Base, or an "
            msg += "ID of the feature containing labels in testX"
            raise InvalidArgumentType(msg)
        if isinstance(trainY, Base):
        #			if not len(trainY.features) == 1:
        #               msg = "If trainY is a Data object, then it may only "
        #               msg += "have one feature"
        #				raise ArgumentException(msg)
            if len(trainY.points) != len(trainX.points):
                msg = "If trainY is a Data object, then it must have the same "
                msg += "number of points as trainX"
                raise InvalidArgumentValueCombination(msg)

    # testX is allowed to be None, sometimes it is appropriate to have it be
    # filled using the trainX argument (ie things which transform data, or
    # learn internal structure)
    if testRequired[0] and testX is None:
        raise InvalidArgumentType("testX must be provided")
    if testX is not None:
        if not isinstance(testX, Base):
            msg = "testX may only be an object derived from Base"
            raise InvalidArgumentType(msg)

    if testRequired[1] and testY is None:
        raise InvalidArgumentType("testY must be provided")
    if testY is not None:
        if not isinstance(testY, (Base, str, int, int)):
            msg = "testY may only be an object derived from Base, or an ID "
            msg += "of the feature containing labels in testX"
            raise InvalidArgumentType(msg)
        if isinstance(testY, Base):
            if len(testY.points) != len(testX.points):
                msg = "If testY is a Data object, then it must have the same "
                msg += "number of points as testX"
                raise InvalidArgumentValueCombination(msg)


def _2dOutputFlagCheck(X, Y, scoreMode, multiClassStrategy):
    outputData = X if Y is None else Y
    if isinstance(outputData, Base):
        needToCheck = len(outputData.features) > 1
    elif isinstance(outputData, (list, tuple)):
        needToCheck = len(outputData) > 1
    elif isinstance(outputData, bool):
        needToCheck = outputData
    else:
        needToCheck = False

    if needToCheck:
        if scoreMode is not None and scoreMode != 'label':
            msg = "When dealing with multi dimensional outputs / predictions, "
            msg += "the scoreMode flag is required to be set to 'label'"
            raise InvalidArgumentValueCombination(msg)
        if multiClassStrategy is not None and multiClassStrategy != 'default':
            msg = "When dealing with multi dimensional outputs / predictions, "
            msg += "the multiClassStrategy flag is required to be set to "
            msg += "'default'"
            raise InvalidArgumentValueCombination(msg)

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
            if not 'cython' in str(type(func)):
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
        argspec = tuple([a, v, k , d])
    except AttributeError:
        argspec = inspect.getfullargspec(func)[:4] # py>=3

    return argspec
