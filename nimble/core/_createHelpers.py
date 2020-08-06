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

import numpy

import nimble
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import InvalidArgumentValueCombination, PackageException
from nimble.exceptions import ImproperObjectAction
from nimble.exceptions import FileFormatException
from nimble.core.data import Base
from nimble.core.data._dataHelpers import isAllowedSingleElement
from nimble.core.data._dataHelpers import validateAllAllowedElements
from nimble.core.data.sparse import removeDuplicatesNative
from nimble._utility import numpy2DArray, is2DArray
from nimble._utility import sparseMatrixToArray
from nimble._utility import scipy, pd, requests, h5py
#

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
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return True

    return False


def validateReturnType(returnType):
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
        elif pd.nimbleAccessible() and isinstance(rawData, pd.DataFrame):
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
    if scipy.nimbleAccessible():
        typeMatch['Sparse'] = scipy.sparse.spmatrix
    if pd.nimbleAccessible():
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
        elif returnType == 'Matrix' and len(rawData.shape) == 1:
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
        if dtype not in [int, float, bool, numpy.bool_]:
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
    if (pd.nimbleAccessible()
            and isinstance(rawData, (pd.DataFrame, pd.Series))):
        # pandas 1.0: SparseDataFrame still in pd namespace but does not work
        # Sparse functionality now determined by presence of .sparse accessor
        # need to convert sparse objects to coo matrix before handling missing
        if hasattr(rawData, 'sparse') and not rawData.empty:
            rawData = rawData.sparse.to_coo()
        else:
            try:
                if isinstance(rawData, pd.SparseDataFrame):
                    rawData = scipy.sparse.coo_matrix(rawData)
            except AttributeError:
                pass

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
        else:
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
        if isinstance(data, Base) and data.shape[0] > 1:
            self.iterator = data.points
        elif isinstance(data, numpy.matrix):
            self.iterator = iter(numpy.array(data))
        elif isinstance(data, dict):
            self.iterator = iter(data.values())
        elif (pd.nimbleAccessible()
              and isinstance(data, (pd.DataFrame, pd.Series))):
            self.iterator = iter(data.values)
        elif scipy.nimbleAccessible() and scipy.sparse.isspmatrix(data):
            self.iterator = SparseCOORowIterator(data.tocoo(False))
        else:
            self.iterator = iter(data)

    def __iter__(self):
        return self

    def __next__(self):
        val = next(self.iterator)
        if isinstance(val, Base) and 1 not in val.shape:
            return val.copy('python list')
        return val


def getFirstIndex(data):
    if scipy.nimbleAccessible() and scipy.sparse.isspmatrix_coo(data):
        first = data.data[data.row == 0]
    elif pd.nimbleAccessible() and isinstance(data, (pd.DataFrame, pd.Series)):
        first = data.iloc[0]
    elif isinstance(data, Base) and 1 not in data.shape:
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
    if scipy.nimbleAccessible() and scipy.sparse.isspmatrix(rawData):
        if not rawData.data.size:
            return False
        rawData = [rawData.data]
    try:
        indexZero = getFirstIndex(rawData)
        if isAllowedSingleElement(indexZero):
            if not skipDataProcessing:
                validateAllAllowedElements(rawData)
            return False
        indexZeroZero = getFirstIndex(indexZero)
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
        else:
            return True
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


def getPointCount(data):
    if isinstance(data, Base):
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
    if isinstance(data, Base) and (len(data._shape) > 2 or data.shape[0] > 1):
        data = data.points
    if toFill is None:
        toFill = []
    if dimensions is None:
        dimensions = [True, [getPointCount(data)]]
    elif dimensions[0]:
        dimensions[1].append(getPointCount(data))
    try:
        if all(map(isAllowedSingleElement, GenericPointIterator(data))):
            toFill.extend(data)
        else:
            for obj in GenericPointIterator(data):
                flattenToOneDimension(obj, toFill, dimensions)
                dimensions[0] = False
    except TypeError:
        msg = "Numbers, strings, None, and nan are the only "
        msg += "values allowed in nimble data objects"
        raise InvalidArgumentValue(msg)

    return toFill, tuple(dimensions[1])


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
        # scipy sparse matrix or a pandas sparse object
        if ((scipy.nimbleAccessible() and scipy.sparse.issparse(rawData))
                or (pd.nimbleAccessible()
                    and isinstance(rawData, (pd.Series, pd.DataFrame))
                    # latest pandas versions use pd.DataFrame.sparse accessor,
                    # previous versions used pd.SparseDataFrame
                    and ((hasattr(rawData, 'sparse'))
                         or (hasattr(pd, 'SparseDataFrame')
                             and isinstance(rawData, pd.SparseDataFrame))))):
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
            ret.points.sort(by=pCmp)
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
            ret.features.sort(by=fCmp)
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
    Helper for nimble.data which deals with the case of loading data
    from a file. Returns a triple containing the raw data, pointNames,
    and featureNames (the later two following the same semantics as
    nimble.data's parameters with the same names).
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
    else:
        raw = numpyMaker((numPoints, numFeatures))
        return nimble.data(returnType, raw, pointNames=pointNames,
                           featureNames=featureNames, name=name, useLog=False)


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
