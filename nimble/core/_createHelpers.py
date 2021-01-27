"""
Helper functions for any functions defined in create.py.

They are separated here so that that (most) top level user-facing
functions are contained in create.py without the distraction of helpers.
"""

import csv
from io import StringIO, BytesIO
import os.path
import copy
import sys
import warnings
import datetime
from contextlib import contextmanager

import numpy

import nimble
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import InvalidArgumentTypeCombination
from nimble.exceptions import InvalidArgumentValueCombination, PackageException
from nimble.exceptions import ImproperObjectAction, FileFormatException
from nimble._utility import removeDuplicatesNative
from nimble._utility import numpy2DArray
from nimble._utility import sparseMatrixToArray, pandasDataFrameToList
from nimble._utility import scipy, pd, requests, h5py, dateutil
from nimble._utility import isAllowedSingleElement, validateAllAllowedElements
from nimble._utility import allowedNumpyDType

###########
# Helpers #
###########

def _isBase(data):
    return isinstance(data, nimble.core.data.Base)

def _isNumpyArray(data):
    return isinstance(data, numpy.ndarray)

def _isNumpyMatrix(data):
    return isinstance(data, numpy.matrix)

def _isPandasObject(data, dataframe=True, series=True, sparse=None):
    if pd.nimbleAccessible():
        if dataframe and series:
            pandasTypes = (pd.DataFrame, pd.Series)
        elif dataframe:
            pandasTypes = pd.DataFrame
        elif series:
            pandasTypes = pd.Series
        if isinstance(data, pandasTypes):
            if sparse is None:
                return True
            sparseAccessor = hasattr(data, 'sparse')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # SparseDataFrame deprecated in pandas >= 1.0
                sparseDF = (hasattr(pd, 'SparseDataFrame')
                            and isinstance(data, pd.SparseDataFrame))
            if not data.empty and sparse and (sparseAccessor or sparseDF):
                return True
            if not sparse and not (sparseAccessor or sparseDF):
                return True
    return False

def _isPandasSparse(data):
    return _isPandasObject(data, sparse=True)

def _isPandasDense(data):
    return _isPandasObject(data, sparse=False)

def _isPandasDataFrame(data):
    return _isPandasObject(data, series=False)

def _isPandasSeries(data):
    return _isPandasObject(data, dataframe=False)

def _isScipySparse(data):
    if scipy.nimbleAccessible():
        return scipy.sparse.isspmatrix(data)
    return False

def isAllowedRaw(data, allowLPT=False):
    """
    Verify raw data is one of the accepted types.
    """
    if _isBase(data):
        return True
    if allowLPT and 'PassThrough' in str(type(data)):
        return True
    if isinstance(data, (tuple, list, dict, numpy.ndarray)):
        return True
    if _isScipySparse(data):
        return True
    if _isPandasObject(data):
        return True

    return False


def validateReturnType(returnType):
    """
    Check returnType argument is valid.
    """
    retAllowed = copy.copy(nimble.core.data.available)
    retAllowed.append(None)
    if returnType not in retAllowed:
        msg = "returnType must be a value in " + str(retAllowed)
        raise InvalidArgumentValue(msg)


def isEmptyRaw(raw):
    """
    Determine if raw data contains no values.
    """
    if raw is None:
        return True
    if hasattr(raw, 'shape'):
        if raw.shape[0] == 0:
            return True
    elif raw == []:
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

    def typeEqual(double):
        x, y = double
        return not isinstance(x, type(y))

    def noDuplicates(row):
        return len(row) == len(set(row))

    if ((pointNames is True or pointNames == 'automatic')
            and firstValues[0] == 'pointNames'):
        allText = (all(map(lambda x: isinstance(x, str),
                           firstValues[1:]))
                   and noDuplicates(firstValues[1:]))
        allDiff = all(map(typeEqual, zip(firstValues[1:], secondValues[1:])))
    else:
        allText = (all(map(lambda x: isinstance(x, str),
                           firstValues))
                   and noDuplicates(firstValues[1:]))
        allDiff = all(map(typeEqual, zip(firstValues, secondValues)))

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
    retPNames = None
    if pnamesID is True:
        temp = []

        for i, ft in enumerate(rawData):
            # grab and remove each value in the feature associated
            # with point names
            currVal = ft.pop(0)
            # if feature names are also in the data, skip index 0
            if fnamesID is not True or (fnamesID is True and i != 0):
            # we wrap it with the string constructor in case the
                # values in question AREN'T strings
                temp.append(str(currVal))
        retPNames = temp

    retFNames = None
    if fnamesID is True:
        # don't have to worry about an overlap entry with point names;
        # if they existed we had already removed those values.
        # Therefore: just pop that entire point
        temp = rawData.pop(0)
        for i, val in enumerate(temp):
            temp[i] = str(val)
        retFNames = temp

    if addedDim:
        rawData = rawData[0]

    return (rawData, retPNames, retFNames)


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

    firstRow = data[0] if len(data) > 0 else None
    secondRow = data[1] if len(data) > 1 else None
    pnamesID, fnamesID = autoDetectNamesFromRaw(pnamesID, fnamesID, firstRow,
                                                secondRow)

    retPNames = None
    retFNames = None
    if pnamesID is True:
        retPNames = numpy.array(data[:, 0]).flatten()
        data = numpy.delete(data, 0, 1)
        if fnamesID is True:
            retPNames = numpy.delete(retPNames, 0)
        retPNames = numpy.vectorize(str)(retPNames)
        retPNames = list(retPNames)
    if fnamesID is True:
        retFNames = numpy.array(data[0]).flatten()
        data = numpy.delete(data, 0, 0)
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
    if not scipy.nimbleAccessible():
        msg = "scipy is not available"
        raise PackageException(msg)
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

    # run through the entries in the returned coo_matrix to get
    # point / feature Names.

    # We justify this time expense by noting that unless this
    # matrix has an inappropriate number of non-zero entries,
    # the names we find will likely be a significant proportion
    # of the present data.

    # these will be ID -> name mappings
    tempPointNames = {}
    tempFeatureNames = {}

    newLen = len(data.data)
    if pnamesID is True:
        newLen -= data.shape[0]
    if fnamesID is True:
        newLen -= data.shape[1]
    if (pnamesID is True) and (fnamesID is True):
        newLen += 1

    newRows = numpy.empty(newLen, dtype=data.row.dtype)
    newCols = numpy.empty(newLen, dtype=data.col.dtype)
    newData = numpy.empty(newLen, dtype=data.dtype)
    writeIndex = 0
    # adjust the sentinal value for easier index modification
    pnamesID = 0 if pnamesID is True else sys.maxsize
    fnamesID = 0 if fnamesID is True else sys.maxsize
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


def extractNamesFromPdDataFrame(rawData, pnamesID, fnamesID):
    """
    Output the index of rawData as pointNames.
    Output the columns of rawData as featureNames.
    """
    firstRow = rawData.iloc[0] if len(rawData) > 0 else None
    secondRow = rawData.iloc[1] if len(rawData) > 1 else None
    pnamesID, fnamesID = autoDetectNamesFromRaw(pnamesID, fnamesID, firstRow,
                                                secondRow)

    retPNames = None
    if pnamesID is True:
        retPNames = [str(i) for i in rawData.index.tolist()]

    retFNames = None
    if fnamesID is True:
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
            pointNames = list(pointNames)
        except TypeError as e:
            msg = "if pointNames are not 'bool' or a 'str', "
            msg += "they should be other 'iterable' object"
            raise InvalidArgumentType(msg) from e
    if not isinstance(featureNames, acceptedNameTypes):
        try:
            featureNames = list(featureNames)
        except TypeError as e:
            msg = "if featureNames are not 'bool' or a 'str', "
            msg += "they should be other 'iterable' object"
            raise InvalidArgumentType(msg) from e
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
        elif _isNumpyArray(rawData):
            func = extractNamesFromNumpy
        elif _isScipySparse(rawData):
            # all input coo_matrices must have their duplicates removed; all
            # helpers past this point rely on there being single entires only.
            if isinstance(rawData, scipy.sparse.coo_matrix):
                rawData = removeDuplicatesNative(rawData)
            func = extractNamesFromScipySparse
        elif _isPandasDataFrame(rawData):
            func = extractNamesFromPdDataFrame
        elif _isPandasSeries(rawData):
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

    return rawData, pointNames, featureNames


def convertData(returnType, rawData, pointNames, featureNames):
    """
    Convert data to an object type which is compliant with the
    initializion for the given returnType.
    """
    typeMatch = {'List': list,
                 'Matrix': numpy.ndarray}
    if scipy.nimbleAccessible():
        typeMatch['Sparse'] = scipy.sparse.spmatrix
    if pd.nimbleAccessible():
        typeMatch['DataFrame'] = pd.DataFrame

    try:
        typeMatchesReturn = isinstance(rawData, typeMatch[returnType])
    except KeyError as e:
        if returnType == 'Sparse':
            package = 'scipy'
        if returnType == 'DataFrame':
            package = 'pandas'
        msg = "{0} must be installed to create a {1} object"
        raise PackageException(msg.format(package, returnType)) from e

    # if the data can be used to instantiate the object we pass it as-is
    # otherwise choose the best option, a 2D list or numpy array, based on
    # data type preservation. Both are accepted by all init methods.
    if typeMatchesReturn:
        if returnType == 'List':
            lenFts = len(featureNames) if featureNames else 0
            if len(rawData) == 0:
                lenPts = len(pointNames) if pointNames else 0
                return numpy.empty([lenPts, lenFts])
            if hasattr(rawData[0], '__len__') and len(rawData[0]) == 0:
                return numpy.empty([len(rawData), lenFts])
        elif returnType == 'Matrix' and len(rawData.shape) == 1:
            rawData = numpy2DArray(rawData)
        return rawData
    ret = convertToBest(rawData, pointNames, featureNames)

    return ret


def convertToBest(rawData, pointNames, featureNames):
    """
    Convert to best object for instantiation. All objects accept python
    lists and numpy arrays for instantiation. Since numpy and scipy
    objects only have a single data type, we use an array. Otherwise, we
    use a list to preserve the data types of the raw values. Arrays are
    also used for any empty objects.
    """
    if _isPandasDataFrame(rawData):
        if rawData.empty:
            return rawData.values
        return pandasDataFrameToList(rawData)
    if _isPandasSeries(rawData):
        if rawData.empty:
            return numpy.empty((0, rawData.shape[0]))
        return [rawData.to_list()]
    if _isScipySparse(rawData):
        return sparseMatrixToArray(rawData)
    if _isNumpyArray(rawData):
        if rawData.size == 0:
            return rawData
        return numpy2DArray(rawData)
    # list objects
    if not isinstance(rawData, list):
        rawData = list(rawData)
    if not rawData: # empty
        lenFts = len(featureNames) if featureNames else 0
        lenPts = len(pointNames) if pointNames else 0
        return numpy.empty((lenPts, lenFts))
    if rawData and isAllowedSingleElement(rawData[0]):
        return [rawData]
    if not all(isinstance(point, list) for point in rawData):
        return [list(point) for point in rawData]

    return rawData

def _parseDatetime(elemType):
    isDatetime = elemType in [datetime.datetime, numpy.datetime64]
    if pd.nimbleAccessible():
        isDatetime = isDatetime or elemType == pd.Timestamp
    if isDatetime and not dateutil.nimbleAccessible():
        msg = 'dateutil package must be installed for datetime conversions'
        raise PackageException(msg)

    return isDatetime

def _numpyArrayDatetimeParse(data, datetimeType):
    data = numpy.vectorize(dateutil.parser.parse)(data)
    if datetimeType is not datetime.datetime:
        data = numpy.vectorize(datetimeType)(data)
        data = data.astype(datetimeType)
    return data

def _valueDatetimeParse(datetimeType):
    def valueParser(value):
        if datetimeType is datetime.datetime:
            return dateutil.parser.parse(value)
        return datetimeType(dateutil.parser.parse(value))
    return valueParser

def elementTypeConvert(data, convertToType):
    """
    Attempt to convert data to the specified convertToType.
    """
    singleType = not isinstance(convertToType, list)
    objectTypes = (object, numpy.object_)
    try:
        if singleType and _isNumpyArray(data):
            if _parseDatetime(convertToType):
                data = _numpyArrayDatetimeParse(data, convertToType)
            else:
                data = data.astype(convertToType)
            if not allowedNumpyDType(data.dtype):
                data = data.astype(numpy.object_)
        elif singleType and _isScipySparse(data):
            if _parseDatetime(convertToType):
                data.data = _numpyArrayDatetimeParse(data.data, convertToType)
            else:
                data.data = data.data.astype(convertToType)
            if not allowedNumpyDType(data.data.dtype):
                data.data = data.data.astype(numpy.object_)
        elif singleType and _isPandasDataFrame(data):
            if _parseDatetime(convertToType):
                data = data.applymap(dateutil.parser.parse)
            else:
                data = data.astype(convertToType)
        elif singleType and data: # 2D list
            # only need to convert if not object type
            if convertToType not in objectTypes:
                if _parseDatetime(convertToType):
                    convertToType = _valueDatetimeParse(convertToType)
                convertedData = []
                for point in data:
                    convertedData.append(list(map(convertToType, point)))
                data = convertedData

        # convertToType is a list of differing types
        elif _isNumpyArray(data):
            for j, feature in enumerate(data.T):
                convType = convertToType[j]
                if convType is None:
                    continue
                data = data.astype(numpy.object_)
                if _parseDatetime(convType):
                    feature = _numpyArrayDatetimeParse(feature, convType)
                data[:, j] = feature.astype(convType)
        elif _isScipySparse(data):
            for col, convType in enumerate(convertToType):
                if convType is None:
                    continue
                data = data.astype(numpy.object_)
                colMask = data.col == col
                if _parseDatetime(convType):
                    feature = _numpyArrayDatetimeParse(data.data[colMask],
                                                       convType)
                    data.data[colMask] = feature
                data.data[colMask] = data.data[colMask].astype(convType)
        elif _isPandasDataFrame(data):
            for i, (idx, ft) in enumerate(data.iteritems()):
                convType = convertToType[i]
                if convType is None:
                    continue
                if _parseDatetime(convType):
                    data[idx] = data[idx].apply(dateutil.parser.parse)
                else:
                    data[idx] = ft.astype(convType)
        elif data: # 2D list
            convertToType = [_valueDatetimeParse(ctt) if _parseDatetime(ctt)
                             else ctt for ctt in convertToType]
            for i, point in enumerate(data):
                zippedConvert = zip(point, convertToType)
                data[i] = [val if (ctype is None or ctype in objectTypes)
                           else ctype(val) for val, ctype in zippedConvert]
        return data

    except (ValueError, TypeError) as error:
        msg = 'Unable to convert the data to convertToType '
        msg += "'{0}'. {1}".format(convertToType, repr(error))
        raise InvalidArgumentValue(msg) from error

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
        replaceLocs = numpy.isin(data, toReplace)
        if replaceLocs.any():
            if data.dtype == bool and not isinstance(replaceWith, bool):
                # numpy will replace with bool(replaceWith) instead
                raise ValueError('replaceWith is not a bool type')
            data[replaceLocs] = replaceWith
        if replaceNan:
            nanLocs = data != data
            if nanLocs.any():
                data[nanLocs] = replaceWith
    except ValueError:
        dtype = type(replaceWith)
        if not allowedNumpyDType(dtype):
            dtype = numpy.object_

        data = data.astype(dtype)
        data[numpy.isin(data, toReplace)] = replaceWith
        if replaceNan:
            data[data != data] = replaceWith
    return data


def replaceMissingData(rawData, treatAsMissing, replaceMissingWith):
    """
    Convert any values in rawData found in treatAsMissing with
    replaceMissingWith value.
    """
    # pandas 1.0: SparseDataFrame still in pd namespace but does not work
    # Sparse functionality now determined by presence of .sparse accessor
    # need to convert sparse objects to coo matrix before handling missing
    if _isPandasSparse(rawData):
        rawData = scipy.sparse.coo_matrix(rawData)

    if isinstance(rawData, (list, tuple)):
        handleMissing = numpy.array(rawData, dtype=numpy.object_)
        handleMissing = replaceNumpyValues(handleMissing, treatAsMissing,
                                           replaceMissingWith)
        rawData = handleMissing.tolist()

    elif _isNumpyArray(rawData):
        rawData = replaceNumpyValues(rawData, treatAsMissing,
                                     replaceMissingWith)

    elif scipy.sparse.issparse(rawData):
        handleMissing = replaceNumpyValues(rawData.data, treatAsMissing,
                                           replaceMissingWith)
        rawData.data = handleMissing

    elif _isPandasDense(rawData):
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
            point = numpy.zeros((self.data.shape[1],))
            row = self.data.row == self.rowIdx
            for val, col in zip(self.data.data[row], self.data.col[row]):
                try:
                    point[col] = val
                except ValueError:
                    point = point.astype(numpy.object_)
                    point[col] = val

            self.rowIdx += 1
            return point

        raise StopIteration


class GenericPointIterator:
    """
    Iterate through all objects in the same manner.

    Iterates through all objects point by point. Iterating through
    non-vector nimble objects is more costly so copying to a python list
    whenever the object is not a point vector is much more efficient. If
    the object does not contain any nimble objects this is effectively
    the same as using iter()
    """
    def __init__(self, data):
        if _isBase(data) and data.shape[0] > 1:
            self.iterator = data.points
        elif _isNumpyMatrix(data):
            self.iterator = iter(numpy.array(data))
        elif isinstance(data, dict):
            self.iterator = iter(data.values())
        elif _isPandasObject(data):
            self.iterator = iter(data.values)
        elif _isScipySparse(data):
            self.iterator = SparseCOORowIterator(data.tocoo(False))
        else:
            self.iterator = iter(data)

    def __iter__(self):
        return self

    def __next__(self):
        val = next(self.iterator)
        if _isBase(val) and 1 not in val.shape:
            return val.copy('python list')
        return val


def _getFirstIndex(data):
    if _isScipySparse(data):
        first = data.data[data.row == 0]
    elif _isPandasObject(data):
        first = data.iloc[0]
    elif _isBase(data) and 1 not in data.shape:
        first = data.points[0]
    elif isinstance(data, dict):
        first = data[list(data.keys())[0]]
    else:
        first = data[0]
    return first


def isHighDimensionData(rawData, skipDataProcessing):
    """
    Identify data with more than two-dimensions.
    """
    if _isScipySparse(rawData):
        if not rawData.data.size:
            return False
        rawData = [rawData.data]
    try:
        indexZero = _getFirstIndex(rawData)
        if isAllowedSingleElement(indexZero):
            if not skipDataProcessing:
                validateAllAllowedElements(rawData)
            return False
        indexZeroZero = _getFirstIndex(indexZero)
        if isAllowedSingleElement(indexZeroZero):
            if not skipDataProcessing:
                toIter = GenericPointIterator(rawData)
                first = next(toIter)
                validateAllAllowedElements(first)
                firstLength = len(first)
                for i, point in enumerate(toIter):
                    if not len(point) == firstLength:
                        msg = "All points in the data do not have the same "
                        msg += "number of features. The first point had {0} "
                        msg += "features but the point at index {1} had {2} "
                        msg += "features"
                        msg = msg.format(firstLength, i, len(point))
                        raise InvalidArgumentValue(msg)
                    validateAllAllowedElements(point)
            return False
        return True
    except IndexError: # rawData or rawData[0] is empty
        return False
    except (ImproperObjectAction, InvalidArgumentType): # high dimension Base
        return True
    except TypeError as e: # invalid non-subscriptable object
        msg = "Numbers, strings, None, and nan are the only "
        msg += "values allowed in nimble data objects"
        raise InvalidArgumentValue(msg) from e


def highDimensionNames(pointNames, featureNames):
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


def _getPointCount(data):
    if _isBase(data):
        return len(data.points)
    if hasattr(data, 'shape'):
        return data.shape[0]
    return len(data)


def flattenToOneDimension(data, toFill=None, dimensions=None):
    """
    Recursive function to flatten an object.

    Flattened values are placed in toFill and this function also records
    the object's dimensions prior to being flattened. getPointIterator
    always return a point-based iterator for these cases so data is
    flattened point by point.
    """
    # if Base and not a vector, use points attribute for __len__ and __iter__
    if _isBase(data) and (len(data._shape) > 2 or data.shape[0] > 1):
        data = data.points
    if toFill is None:
        toFill = []
    if dimensions is None:
        dimensions = [True, [_getPointCount(data)]]
    elif dimensions[0]:
        dimensions[1].append(_getPointCount(data))
    try:
        if all(map(isAllowedSingleElement, GenericPointIterator(data))):
            toFill.extend(data)
        else:
            for obj in GenericPointIterator(data):
                flattenToOneDimension(obj, toFill, dimensions)
                dimensions[0] = False
    except TypeError as e:
        msg = "Numbers, strings, None, and nan are the only "
        msg += "values allowed in nimble data objects"
        raise InvalidArgumentValue(msg) from e

    return toFill, tuple(dimensions[1])


def flattenHighDimensionFeatures(rawData):
    """
    Flatten data with multi-dimensional features to vectors.

    Features are flattened point by point whether numpy.reshape or
    flattenToOneDimension are used.
    """
    if _isNumpyArray(rawData) and rawData.dtype != numpy.object_:
        origDims = rawData.shape
        newShape = (rawData.shape[0], numpy.prod(rawData.shape[1:]))
        rawData = numpy.reshape(rawData, newShape)
    else:
        if hasattr(rawData, 'shape'):
            numPts = rawData.shape[0]
        else:
            numPts = len(rawData)
        points = GenericPointIterator(rawData)
        firstPoint = next(points)
        firstPointFlat, ptDims = flattenToOneDimension(firstPoint)
        origDims = tuple([numPts] + list(ptDims))
        numFts = len(firstPointFlat)
        rawData = numpy.empty((numPts, numFts), dtype=numpy.object_)
        rawData[0] = firstPointFlat
        for i, point in enumerate(points):
            flat, dims = flattenToOneDimension(point)
            if dims != ptDims:
                msg = 'The dimensions of each nested object must equal. The '
                msg += 'first point had dimensions {0}, but point {1} had '
                msg += 'dimensions {2}'
                raise InvalidArgumentValue(msg.format(ptDims, i + 1, dims))
            rawData[i + 1] = flat

    return rawData, origDims

def getKeepIndexValues(axisObj, keepList):
    """
    Get only the index values from a list that could contain axis names.

    Additionally validates that there is not an index and name that
    represent the same point/feature.
    """
    cleaned = []
    for val in keepList:
        converted = axisObj.getIndex(val)
        if converted not in cleaned:
            cleaned.append(converted)
        else:
            # we know no duplicates present so an index and name must match
            msg = "Values in {keep} must represent unique {axis}s. "
            msg += "'{name}' and {idx} represent the same {axis}. "
            axis = axisObj._axis
            keep = 'keepPoints' if axis == 'point' else 'keepFeatures'
            name = axisObj.getName(converted)
            msg = msg.format(axis=axis, keep=keep, name=name, idx=converted)
            raise InvalidArgumentValue(msg)

    return cleaned

def convertToTypeDictToList(convertToType, featuresObj, featureNames):
    """
    Convert the dict containing convertToType values to a list where
    the index indicates the type of that feature. None indicates no
    conversion.
    """
    retFNames = featuresObj._getNamesNoGeneration()
    convertList = [None] * len(featuresObj)
    # if no feature names, we will use the list of None values as the
    # featureNames to allow us to use the same process in either case
    if retFNames is None:
        retFNames = convertList
    for i, ftName in enumerate(retFNames):
        if i in convertToType and ftName in convertToType:
            if convertToType[i] == convertToType[ftName]:
                convertList[i] = convertToType[i]
                del convertToType[i]
                del convertToType[ftName]
            else:
                msg = "The keys '{name}' and {idx} represent the same "
                msg += "feature but have different values"
                raise InvalidArgumentValue(msg.format(name=ftName, idx=i))
        if i in convertToType:
            convertList[i] = convertToType[i]
            del convertToType[i]
        elif ftName in convertToType:
            convertList[i] = convertToType[ftName]
            del convertToType[ftName]
    # if there are any unused values, they must correspond with full data
    if convertToType:
        fail = []
        if featureNames is None:
            fail = list(convertToType.keys())
        else:
            for key in convertToType:
                if not ((isinstance(key, int) and key < len(featureNames))
                        or key in featureNames):
                    fail.append(key)
        if fail:
            msg = 'The key(s) {keys} in convertToType are not valid for '
            msg += 'this object'
            raise InvalidArgumentValue(msg.format(keys=fail))

    return convertList

def initDataObject(
        returnType, rawData, pointNames, featureNames, name=None, path=None,
        keepPoints='all', keepFeatures='all', convertToType=None,
        reuseData=False, treatAsMissing=(float('nan'), numpy.nan, None, '',
                                         'None', 'nan', 'NULL', 'NA'),
        replaceMissingWith=numpy.nan, skipDataProcessing=False,
        extracted=(None, None)):
    """
    1. Argument Validation
    2. Setup autoType
    3. Extract Names
    4. Convert to 2D representation
    5. Handle Missing data
    6. Convert to acceptable form for returnType init
    7. init returnType object
    8. Limit to keepPoints / keepFeatures
    9. Perform any convertToType conversions
    """
    limitPoints = keepPoints != 'all'
    limitFeatures = keepFeatures != 'all'
    if limitPoints:
        _checkForDuplicates(keepPoints, 'keepPoints')
    if limitFeatures:
        _checkForDuplicates(keepFeatures, 'keepFeatures')
    if (limitFeatures and isinstance(convertToType, dict)
            and not all(isinstance(t, str) for t in convertToType.keys())):
        msg = "When limiting the features using keepFeatures, the keys in a "
        msg += "convertToType dict cannot be index values because it is "
        msg += "ambiguous as to whether they reference the full dataset or "
        msg += "limited dataset. Use featureNames as keys or a list to "
        msg += "eliminate the ambiguity. None may be used in the list for "
        msg += "features that do not require conversion."
        raise InvalidArgumentTypeCombination(msg)


    if returnType is None:
        # scipy sparse matrix or a pandas sparse object
        if _isScipySparse(rawData) or _isPandasSparse(rawData):
            returnType = 'Sparse'
        else:
            returnType = 'Matrix'

    if _isBase(rawData):
        # point/featureNames, treatAsMissing, etc. may vary
        rawData = rawData._data
    if not reuseData:
        rawData = copy.deepcopy(rawData)

    # record if extraction occurred before we possibly modify *Names parameters
    ptsExtracted = extracted[0] if extracted[0] else pointNames is True
    ftsExtracted = extracted[1] if extracted[1] else featureNames is True

    # If skipping data processing, no modification needs to be made
    # to the data, so we can skip name extraction and missing replacement.
    kwargs = {}
    # convert these types as indexing may cause dimensionality confusion
    if _isNumpyMatrix(rawData):
        rawData = numpy.array(rawData)
    if _isScipySparse(rawData):
        rawData = rawData.tocoo()

    if isHighDimensionData(rawData, skipDataProcessing):
        # additional name validation / processing before extractNames
        pointNames, featureNames = highDimensionNames(pointNames,
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
    # convert data to a type compatible with the returnType init method
    rawData = convertData(returnType, rawData, pointNames, featureNames)

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

    initMethod = getattr(nimble.core.data, returnType)
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
        for i, keep in enumerate(keepList):
            positions[keep] = i

        def retCmp(view1, view2):
            idx1 = indexGetter(view1)
            idx2 = indexGetter(view2)
            if positions[idx1] < positions[idx2]:
                return -1
            if positions[idx1] > positions[idx2]:
                return 1

            return 0

        return retCmp

    # keep points and features if still needed
    if limitPoints:
        numPts = len(ret.points)
        if not ptsExtracted and len(keepPoints) == numPts:
            _raiseKeepLengthConflict('point')
        # if we have all pointNames, set them now
        if isinstance(pointNames, (list, dict)) and len(pointNames) == numPts:
            ret.points.setNames(pointNames, useLog=False)
            setPtNamesAfter = False
        else:
            _keepIndexValuesValidation('point', keepPoints, pointNames)
            setPtNamesAfter = True
        cleaned = getKeepIndexValues(ret.points, keepPoints)
        if len(cleaned) == numPts:
            pCmp = makeCmp(cleaned, ret, 'point')
            ret.points.sort(by=pCmp)
        else:
            ret = ret.points.copy(cleaned)
        # if we had a subset of pointNames can set now on the cleaned data
        if setPtNamesAfter:
            ret.points.setNames(pointNames, useLog=False)
    if limitFeatures:
        numFts = len(ret.features)
        if not ftsExtracted and len(keepFeatures) == numFts:
            _raiseKeepLengthConflict('feature')
        # if we have all featureNames, set them now
        if (isinstance(featureNames, (list, dict))
                and len(featureNames) == numFts):
            ret.features.setNames(featureNames, useLog=False)
            setFtNamesAfter = False
        # otherwise we require keepFeatures to be index and set names later
        else:
            _keepIndexValuesValidation('feature', keepFeatures, featureNames)
            setFtNamesAfter = True
        cleaned = getKeepIndexValues(ret.features, keepFeatures)
        if isinstance(convertToType, list):
            if len(convertToType) == numFts:
                convertToType = [convertToType[i] for i in cleaned]
            elif len(convertToType) != len(cleaned):
                msg = "Invalid length of convertToType. convertToType must "
                msg += "be either the length of the full dataset ({full}) "
                msg += "or the length of the limited dataset ({limited}), but "
                msg += "was length {actual}."
                msg = msg.format(full=numFts, limited=len(cleaned),
                                 actual=len(convertToType))
                raise InvalidArgumentValue(msg)
        if len(cleaned) == numFts:
            fCmp = makeCmp(cleaned, ret, 'feature')
            ret.features.sort(by=fCmp)
        else:
            ret = ret.features.copy(cleaned)
        # if we had a subset of featureNames can set now on the cleaned data
        if setFtNamesAfter:
            ret.features.setNames(featureNames, useLog=False)

    # To simplify, we will make convertToType dicts into lists of types
    if isinstance(convertToType, dict):
        convertToType = convertToTypeDictToList(convertToType, ret.features,
                                                featureNames)
    elif isinstance(convertToType, list):
        if len(convertToType) != len(ret.features):
            msg = 'A list for convertToType must have many elements as '
            msg += 'features in the data. The object contains {numFts} '
            msg += 'features, but convertToType has {numElems} elements.'
            msg = msg.format(numFts=len(ret.features),
                             numElems=len(convertToType))
            raise InvalidArgumentValue(msg)
        if all(v == convertToType[0] for v in convertToType[1:]):
            convertToType = convertToType[0]

    if convertToType is not None:
        ret._data = elementTypeConvert(ret._data, convertToType)

    return ret

def _autoFileTypeChecker(ioStream):
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

@contextmanager
def _openFileIO(openFile):
    """
    Context manager that prevents an open file from closing on exit.
    """
    yield openFile

def createDataFromFile(
        returnType, data, pointNames, featureNames, name,
        ignoreNonNumericalFeatures, keepPoints, keepFeatures, convertToType,
        inputSeparator, treatAsMissing, replaceMissingWith):
    """
    Helper for nimble.data which deals with the case of loading data
    from a file. Returns a triple containing the raw data, pointNames,
    and featureNames (the later two following the same semantics as
    nimble.data's parameters with the same names).
    """
    # Case: string value means we need to open the file, either directly or
    # through an http request
    if isinstance(data, str):
        if data[:4] == 'http': # webpage
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
            with BytesIO(response.content) as toCheck:
                extension = _autoFileTypeChecker(toCheck)
            if extension != 'csv':
                ioStream = BytesIO(response.content)
            else:
                ioStream = StringIO(response.text, newline=None)
        else: # path to file
            try:
                with open(data, 'r', newline=None) as toCheck:
                    extension = _autoFileTypeChecker(toCheck)
                ioStream = open(data, 'r', newline=None)
            except UnicodeDecodeError:
                with open(data, 'rb', newline=None) as toCheck:
                    extension = _autoFileTypeChecker(toCheck)
                ioStream = open(data, 'rb', newline=None)
        path = data
    # Case: we are given an open file already
    else:
        with _openFileIO(data) as toCheck:
            extension = _autoFileTypeChecker(toCheck)
        ioStream = _openFileIO(data)
        # try getting name attribute from file
        try:
            path = data.name
        except AttributeError:
            path = None

    # if the file has a different, valid extension from the one we determined
    # we will defer to the file's extension
    if path is not None:
        split = path.rsplit('.', 1)
        supportedExtensions = ['csv', 'mtx', 'hdf5', 'h5']
        if len(split) > 1 and split[1].lower() in supportedExtensions:
            extension = split[1].lower()

    with ioStream as toLoad:
        selectSuccess = False
        if extension == 'csv':
            loaded = _loadcsvUsingPython(
                toLoad, pointNames, featureNames, ignoreNonNumericalFeatures,
                keepPoints, keepFeatures, inputSeparator)
            selectSuccess = True
        elif extension == 'mtx':
            loaded = _loadmtxForAuto(toLoad, pointNames, featureNames)
        elif extension in ['hdf5', 'h5']: # h5 and hdf5 are synonymous
            loaded = _loadhdf5ForAuto(toLoad, pointNames, featureNames)

    retData, retPNames, retFNames = loaded

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
        keepPoints, keepFeatures, convertToType=convertToType,
        treatAsMissing=treatAsMissing, replaceMissingWith=replaceMissingWith,
        extracted=extracted)


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
        return nimble.data(returnType, rawSparse, pointNames=pointNames,
                           featureNames=featureNames, name=name, useLog=False)

    raw = numpyMaker((numPoints, numFeatures))
    return nimble.data(returnType, raw, pointNames=pointNames,
                       featureNames=featureNames, name=name, useLog=False)


def _intFloatOrString(inString):
    """
    Try to convert strings to numeric types or empty strings to None.
    """
    ret = inString
    if not inString:
        return None
    try:
        return int(inString)
    except ValueError:
        try:
            return float(inString)
        except ValueError:
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


def _checkCSVForNames(openFile, pointNames, featureNames, dialect):
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


def _loadmtxForAuto(openFile, pointNames, featureNames):
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
        # the nimble.data input semantics
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

    return data, retPNames, retFNames


def _loadhdf5ForAuto(openFile, pointNames, featureNames):
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

    return data, pointNames, featureNames


def _loadcsvUsingPython(openFile, pointNames, featureNames,
                        ignoreNonNumericalFeatures, keepPoints, keepFeatures,
                        inputSeparator):
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
    dialect = _detectDialectFromSeparator(openFile, inputSeparator)

    (pointNames, featureNames) = _checkCSVForNames(openFile, pointNames,
                                                   featureNames, dialect)

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
        # have all featureNames but not keeping all features
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
    lastFtAllMissing = not limitFeatures
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
                    if colType not in (str, type(None)):
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
        if lastFtAllMissing and row[-1] != '':
            lastFtAllMissing = False
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

    # when the last value in every row is '' and all columns are kept,
    # that column will be ignored unless a featureName is provided
    if (lastFtAllMissing and
            (not retFNames or len(retFNames) == firstRowLength - 1
             or (featureNames is True and retFNames[-1] == ''))):
        retData = [row[:-1] for row in retData]
        if featureNames is True:
            retFNames = retFNames[:-1]

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

    return retData, retPNames, retFNames
