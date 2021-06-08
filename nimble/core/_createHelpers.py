"""
Helper functions for any functions defined in create.py.

They are separated here so that that (most) top level user-facing
functions are contained in create.py without the distraction of helpers.
"""

import csv
from io import StringIO, BytesIO
import os
import copy
import warnings
import datetime
import re
import zipfile
import tarfile
import gzip
import shutil
import urllib.parse
import locale

import numpy as np

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


# The values that will be considered missing in data by default
DEFAULT_MISSING = (float('nan'), np.nan, None, '', 'None', 'nan', 'NULL', 'NA')

###########
# Helpers #
###########

def _isBase(data):
    return isinstance(data, nimble.core.data.Base)

def _isNumpyArray(data):
    return isinstance(data, np.ndarray)

def _isNumpyMatrix(data):
    return isinstance(data, np.matrix)

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

def looksFileLike(toCheck):
    """
    Determine if object appears to be a file object.
    """
    hasRead = hasattr(toCheck, 'read')
    hasWrite = hasattr(toCheck, 'write')
    return hasRead and hasWrite

def isAllowedRaw(data):
    """
    Verify raw data is one of the accepted types.
    """
    if isinstance(data, str) or looksFileLike(data):
        return False
    if hasattr(data, '__iter__') or hasattr(data, '__getitem__'):
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

    def isNum(value):
        return isinstance(value, (bool, int, float, np.number))

    def noDuplicates(row):
        return len(row) == len(set(row))

    if ((pointNames is True or pointNames == 'automatic')
            and firstValues[0] == 'pointNames'):
        allText = (all(map(lambda x: isinstance(x, str),
                           firstValues[1:]))
                   and noDuplicates(firstValues[1:]))
        anyNum = any(map(isNum, secondValues[1:]))
    else:
        allText = (all(map(lambda x: isinstance(x, str),
                           firstValues))
                   and noDuplicates(firstValues[1:]))
        anyNum = any(map(isNum, secondValues))

    if featureNames == 'automatic' and allText and anyNum:
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


def extractNamesFromRawList(rawData, pnamesID, fnamesID, copied):
    """
    Remove name data from a python list.

    Takes a raw python list of lists and if specified remove those
    rows or columns that correspond to names, returning the remaining
    data, and the two name objects (or None in their place if they were
    not specified for extraction). pnamesID may either be None, or an
    integer ID corresponding to the column of point names. fnamesID
    may either be None, or an integer ID corresponding to the row of
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

    firstRow = rawData[0] if len(rawData) > 0 else None
    secondRow = rawData[1] if len(rawData) > 1 else None
    pnamesID, fnamesID = autoDetectNamesFromRaw(pnamesID, fnamesID, firstRow,
                                                secondRow)

    if not copied and (pnamesID is True or fnamesID is True):
        rawData = [lst.copy() for lst in rawData]
        copied = True
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

    return rawData, retPNames, retFNames, copied


def extractNamesFromNumpy(data, pnamesID, fnamesID, copied):
    """
    Extract name values from a numpy array.
    """
    # if there are no elements, extraction cannot happen. We return correct
    # results for this case so it is excluded from the subsequent code
    if 0 in data.shape:
        return data, None, None, copied

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
        retPNames = np.array(data[:, 0]).flatten()
        data = np.delete(data, 0, 1)
        copied = True
        if fnamesID is True:
            retPNames = np.delete(retPNames, 0)
        retPNames = np.vectorize(str)(retPNames)
        retPNames = list(retPNames)
    if fnamesID is True:
        retFNames = np.array(data[0]).flatten()
        data = np.delete(data, 0, 0)
        copied = True
        retFNames = np.vectorize(str)(retFNames)
        retFNames = list(retFNames)

    if addedDim:
        data = data.reshape(data.shape[1])

    return data, retPNames, retFNames, copied


def extractNamesFromScipySparse(data, pnamesID, fnamesID, copied):
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
    if not scipy.nimbleAccessible():
        msg = "scipy is not available"
        raise PackageException(msg)
    if not scipy.sparse.isspmatrix_coo(data):
        data = scipy.sparse.coo_matrix(data)
        copied = True
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

    if not pnamesID is True and not fnamesID is True:
        return data, None, None, copied

    numPts = data.shape[0]
    numFts = data.shape[1]
    newData = data.data
    newRow = data.row
    newCol = data.col
    retPNames = None
    retFNames = None
    if pnamesID is True:
        nameCol = newCol == 0
        index = newRow[nameCol]
        names = newData[nameCol]
        idxNameMap = dict(zip( map(int, index), map(str, names)))
        # retPNames = dict(zip(map(str, names), map(int, index)))
        if fnamesID is not True and len(idxNameMap) < numPts:
            # 0 is one of the names, need to find missing index:
            missing = set(range(numPts)) - idxNameMap.keys()
            if len(missing) > 1:
                msg = 'pointNames must be unique. Found multiple zero values '
                msg += 'in column containing pointNames'
                raise InvalidArgumentValue(msg)
            retPNames = {v:k for k, v in idxNameMap.items()}
            retPNames['0'] = missing.pop()
        elif fnamesID is True:
            if 0 in idxNameMap:
                del idxNameMap[0]
            retPNames = {}
            for key, val in idxNameMap.items():
                retPNames[val] = key - 1
        else:
            retPNames = {v:k for k, v in idxNameMap.items()}

        numFts -= 1
        newData = newData[~nameCol]
        newRow = newRow[~nameCol]
        newCol = newCol[~nameCol] - 1

    if fnamesID is True:
        nameRow = newRow == 0
        index = newCol[nameRow]
        names = newData[nameRow]
        retFNames = dict(zip(map(str, names), map(int, index)))
        if len(retFNames) < numFts:
            # 0 is one of the names, need to find missing index:
            missing = set(range(numFts)) - set(retFNames.values())
            if len(missing) > 1:
                msg = 'featureNames must be unique. Found multiple zero '
                msg += 'values in column containing featureNames'
                raise InvalidArgumentValue(msg)
            retFNames['0'] = missing.pop()
        numPts -= 1
        newData = newData[~nameRow]
        newRow = newRow[~nameRow] - 1
        newCol = newCol[~nameRow]

    data = scipy.sparse.coo_matrix((newData, (newRow, newCol)),
                                   shape=(numPts, numFts))

    return data, retPNames, retFNames, True


def extractNamesFromPdDataFrame(rawData, pnamesID, fnamesID, copied):
    """
    Output the index of rawData as pointNames.
    Output the columns of rawData as featureNames.
    """
    firstRow = rawData.iloc[0] if len(rawData) > 0 else None
    secondRow = rawData.iloc[1] if len(rawData) > 1 else None
    pnamesID, fnamesID = autoDetectNamesFromRaw(pnamesID, fnamesID, firstRow,
                                                secondRow)

    if not copied and (pnamesID is True or fnamesID is True):
        rawData = rawData.copy()
        copied = True
    retPNames = None
    if pnamesID is True:
        retPNames = [str(i) for i in rawData.index.tolist()]

    retFNames = None
    if fnamesID is True:
        retFNames = [str(i) for i in rawData.columns.tolist()]

    return rawData, retPNames, retFNames, copied


def extractNamesFromPdSeries(rawData, pnamesID, fnamesID, copied):
    """
    Output the index of rawData as featureNames.
    """
    if not copied and (pnamesID is True or fnamesID is True):
        rawData = rawData.copy()
        copied = True
    retPNames = None
    if pnamesID is True:
        retPNames = [rawData.index[0]]
        rawData = rawData[1:]

    retFNames = None
    if fnamesID is True:
        retFNames = [str(i) for i in rawData.index.tolist()]
        rawData = np.empty((0, len(retFNames)))

    return rawData, retPNames, retFNames, copied

def _extractNamesFromDict(rawData, pointNames, featureNames, copied):
    if rawData:
        ptNames = list(rawData.keys())
        rawData = list(rawData.values())
        if isinstance(rawData[0], dict):
            # pointNames were keys and rawData is now a list of dicts
            return extractNames(rawData, ptNames, featureNames, copied,
                                False)
        if featureNames is True:
            msg = 'featureNames cannot be True when source is a dict'
            raise InvalidArgumentValue(msg)
        rawData = numpy2DArray(rawData, dtype=np.object_)
        copied = True
        pointNames, rawData = _dictNames('point', pointNames, ptNames,
                                         rawData)
        if len(ptNames) != len(rawData):
            # {'a':[1,3],'b':[2,4],'c':['a','b']} -> keys = ['a', 'c', 'b']
            # np.matrix(values()) = [[1,3], ['a', 'b'], [2,4]]
            # transpose is not needed
            # {'a':1, 'b':2, 'c':3} --> keys = ['a', 'c', 'b']
            # np.matrix(values()) = [[1,3,2]]
            # thus transpose is needed
            rawData = transposeMatrix(rawData)
    else: # rawData={}
        rawData = np.empty([0, 0])
        copied = True
        if pointNames is True:
            msg = 'pointNames cannot be True when data is an empty dict'
            raise InvalidArgumentValue(msg)
        pointNames = None

    if featureNames == 'automatic' or featureNames is False:
        featureNames = None
    if pointNames is False:
        pointNames = None

    return rawData, pointNames, featureNames, copied

def _extractNamesFromListOfDict(rawData, pointNames, featureNames, copied):
    if pointNames is True:
        msg = 'pointNames cannot be True when data is a dict'
        raise InvalidArgumentValue(msg)
    # double nested list contained list-type forced values from first row
    values = []
    # in py3 keys() returns a dict_keys object comparing equality of these
    # objects is valid, but converting to lists for comparison can fail
    keys = rawData[0].keys()
    if not keys: # empty dict
        if featureNames is True:
            msg = 'featureNames cannot be True when data is an empty dict'
            raise InvalidArgumentValue(msg)
        if featureNames is False or featureNames == 'automatic':
            featureNames = None
    ftNames = list(keys)
    featureNames, firstPt = _dictNames('feature', featureNames, ftNames,
                                       list(rawData[0].values()))
    values.append(firstPt)
    if featureNames is not None:
        ftNames = featureNames # names may have been reorderd
    for i, row in enumerate(rawData[1:]):
        if row.keys() != keys:
            msg = "The keys at index {} do not match ".format(i + 1)
            msg += "the keys at index 0. Each dictionary in the list must "
            msg += "contain the same keys."
            raise InvalidArgumentValue(msg)
        values.append([row[name] for name in ftNames])
    rawData = values
    copied = True

    if pointNames == 'automatic' or pointNames is False:
        pointNames = None
    if featureNames is False:
        featureNames = None

    return rawData, pointNames, featureNames, copied

def _extractNamesFromListOfBase(rawData, pointNames, featureNames, copied):
    first = rawData[0]
    transpose = first.shape[1] == 1 and first.shape[0] > 1 # feature vector
    if transpose:
        first = first.T
    ftNames = first.features._getNamesNoGeneration()
    if featureNames is True and ftNames is None:
        msg = 'All objects must have feature names when featureNames=True'
        raise InvalidArgumentValue(msg)
    if featureNames == 'automatic':
        featureNames = ftNames is not None
    if featureNames is True or pointNames is True or pointNames == 'automatic':
        ptNames = []
        numNoPtName = 0
        for i, base in enumerate(rawData):
            if transpose:
                base = base.T
            if pointNames is True or pointNames == 'automatic':
                if base.points._namesCreated():
                    ptNames.append(base.points.getName(0))
                else:
                    ptNames.append(None)
                    numNoPtName += 1
            fNames = base.features._getNamesNoGeneration()
            if featureNames is True and ftNames != fNames:
                msg = 'All objects must have identical feature names. '
                if ftNames is None:
                    msg += 'No feature names were detected in the first  '
                    msg += 'object but the object at index {} has names'
                elif fNames is None:
                    msg += 'No feature names were detected in the object at '
                    msg += 'index {}'
                else:
                    msg += 'The feature names at index {} are different than '
                    msg += 'those in the first feature'
                    raise InvalidArgumentValue(msg.format(i))

        if pointNames is True and ptNames and len(ptNames) == numNoPtName:
            msg = 'pointNames cannot be True when none of the objects '
            msg += 'have point names'
            raise InvalidArgumentValue(msg)
        if pointNames is True:
            pointNames = ptNames
        elif pointNames == 'automatic':
            if len(ptNames) > numNoPtName:
                pointNames = ptNames
            else:
                pointNames = None

        if featureNames is True:
            featureNames = ftNames

    if pointNames is False:
        pointNames = None
    if featureNames is False:
        featureNames = None

    return rawData, pointNames, featureNames, copied

def transposeMatrix(matrixObj):
    """
    This function is similar to np.transpose.
    copy.deepcopy(np.transpose(matrixObj)) may generate a messed data,
    so I created this function.
    """
    return numpy2DArray(list(zip(*matrixObj.tolist())), dtype=matrixObj.dtype)

def _dictNames(axis, names, foundNames, data):
    """
    Checks that point or feature names are consistent with dict keys and
    reorders data, if necessary.
    """
    if isinstance(names, (list, dict)):
        if isinstance(names, dict):
            names = sorted(names, key=names.get)
        if names != foundNames:
            if sorted(names) != sorted(foundNames):
                msg = 'Since dictionaries are unordered, ' + axis + 'Names '
                msg += 'can only be provided if they are a reordering of the '
                msg += 'keys in the dictionary'
                raise InvalidArgumentValue(msg)
            # reordering of features is necessary
            newOrder = [foundNames.index(f) for f in names]
            if isinstance(data, np.ndarray):
                data = data[newOrder]
            else:
                data = [data[i] for i in newOrder]
    elif names is False:
        names = None
    elif names is not None: # 'automatic' or True
        names = foundNames

    return names, data


def extractNames(rawData, pointNames, featureNames, copied, checkNames=True):
    """
    Extract the point and feature names from the raw data, if necessary.
    """
    if checkNames: # can be used recursively and don't need to check again
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
    # pointNames can be same as the keys to define point order
    # featureNames cannot be True
    if isinstance(rawData, dict):
        rawData, pointNames, featureNames, copied = _extractNamesFromDict(
            rawData, pointNames, featureNames, copied)
    # 2. convert list of dict ie. [{'a':1, 'b':3}, {'a':2, 'b':4}] to list
    # featureNames may be same as the keys to define feature order
    # pointNames cannot be True
    elif (isinstance(rawData, list)
          and len(rawData) > 0
          and isinstance(rawData[0], dict)):
        rawData, pointNames, featureNames, copied = (
            _extractNamesFromListOfDict(rawData, pointNames, featureNames,
                                        copied))
    # 3. extract names from a list of Base objects
    # will treat all as point vectors so pointNames must be unique
    # featureNames must be consistent
    elif (isinstance(rawData, list)
          and len(rawData) > 0
          and _isBase(rawData[0])):
        rawData, pointNames, featureNames, copied = (
            _extractNamesFromListOfBase(rawData, pointNames, featureNames,
                                        copied))
    # 4. for rawData of other data types
    # check if we need to do name extraction, setup new variables,
    # or modify values for subsequent call to data init method.
    else:
        if _isNumpyArray(rawData):
            func = extractNamesFromNumpy
        elif _isScipySparse(rawData):
            # all input coo_matrices must have their duplicates removed; all
            # helpers past this point rely on there being single entries only.
            if isinstance(rawData, scipy.sparse.coo_matrix):
                rawData = removeDuplicatesNative(rawData)
            func = extractNamesFromScipySparse
        elif _isPandasDataFrame(rawData):
            func = extractNamesFromPdDataFrame
        elif _isPandasSeries(rawData):
            func = extractNamesFromPdSeries
        else:
            func = extractNamesFromRawList

        rawData, tempPointNames, tempFeatureNames, copied = func(
            rawData, pointNames, featureNames, copied)

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

    return rawData, pointNames, featureNames, copied


def convertData(returnType, rawData, pointNames, featureNames, copied):
    """
    Convert data to an object type which is compliant with the
    initializion for the given returnType.
    """
    typeMatch = {'List': list,
                 'Matrix': np.ndarray}
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
                return np.empty([lenPts, lenFts])
            if (hasattr(rawData[0], '__len__') and
                    not isinstance(rawData[0], str)):
                if len(rawData[0]) == 0:
                    return np.empty([len(rawData), lenFts])
                # list of other iterators
                if not all(isinstance(pt, list) for pt in rawData):
                    if not copied:
                        rawData = rawData.copy()
                    for i, iterator in enumerate(rawData):
                        rawData[i] = list(iterator)
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
            return np.empty((0, rawData.shape[0]))
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
        return np.empty((lenPts, lenFts))
    if rawData and isAllowedSingleElement(rawData[0]):
        return [rawData]
    if not all(isinstance(point, list) for point in rawData):
        return [list(point) for point in rawData]

    return rawData

def _parseDatetime(elemType):
    isDatetime = elemType in [datetime.datetime, np.datetime64]
    if pd.nimbleAccessible():
        isDatetime = isDatetime or elemType == pd.Timestamp
    if isDatetime and not dateutil.nimbleAccessible():
        msg = 'dateutil package must be installed for datetime conversions'
        raise PackageException(msg)

    return isDatetime

def _numpyArrayDatetimeParse(data, datetimeType):
    data = np.vectorize(dateutil.parser.parse)(data)
    if datetimeType is not datetime.datetime:
        data = np.vectorize(datetimeType)(data)
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
    objectTypes = (object, np.object_)
    try:
        if singleType and _isNumpyArray(data):
            if _parseDatetime(convertToType):
                data = _numpyArrayDatetimeParse(data, convertToType)
            else:
                data = data.astype(convertToType)
            if not allowedNumpyDType(data.dtype):
                data = data.astype(np.object_)
        elif singleType and _isScipySparse(data):
            if _parseDatetime(convertToType):
                data.data = _numpyArrayDatetimeParse(data.data, convertToType)
            else:
                data.data = data.data.astype(convertToType)
            if not allowedNumpyDType(data.data.dtype):
                data.data = data.data.astype(np.object_)
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
                data = data.astype(np.object_)
                if _parseDatetime(convType):
                    feature = _numpyArrayDatetimeParse(feature, convType)
                data[:, j] = feature.astype(convType)
        elif _isScipySparse(data):
            for col, convType in enumerate(convertToType):
                if convType is None:
                    continue
                data = data.astype(np.object_)
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

def _replaceMissingData(rawData, treatAsMissing, replaceMissingWith, copied):
    """
    Convert any values in rawData found in treatAsMissing with
    replaceMissingWith value.
    """
    def getNumpyReplaceLocations(data):
        # if data has numeric dtype and replacing with a numeric value,
        # non-numeric values can be removed possible values to replace
        numeric = (int, float, np.number)
        if (np.issubclass_(data.dtype.type, np.number)
                and isinstance(replaceMissingWith, numeric)):
            toReplace = np.array([val for val in treatAsMissing
                                     if isinstance(val, numeric)])
        else:
            toReplace = np.array(treatAsMissing, dtype=np.object_)

        replaceLocs = np.isin(data, toReplace)
        # np.isin cannot handle nan replacement, so if nan is in
        # toReplace we instead set the flag to trigger nan replacement
        replaceNan = any(isinstance(val, float) and np.isnan(val)
                         for val in toReplace)
        if replaceNan:
            nanLocs = data != data
            replaceLocs = replaceLocs | nanLocs

        return replaceLocs

    def replaceNumpyValues(data, replaceLocs):
        # try to avoid converting dtype if possible for efficiency.
        try:
            if data.dtype == bool and not isinstance(replaceMissingWith, bool):
                # numpy will replace with bool(replaceWith) instead
                raise ValueError('replaceWith is not a bool type')
            data[replaceLocs] = replaceMissingWith
        except ValueError:
            dtype = type(replaceMissingWith)
            if not allowedNumpyDType(dtype):
                dtype = np.object_
            data = data.astype(dtype)
            data[replaceLocs] = replaceMissingWith

        return data

    # pandas 1.0: SparseDataFrame still in pd namespace but does not work
    # Sparse functionality now determined by presence of .sparse accessor
    # need to convert sparse objects to coo matrix before handling missing
    if _isPandasSparse(rawData):
        rawData = scipy.sparse.coo_matrix(rawData)
        copied = True

    if _isNumpyArray(rawData):
        replaceLocs = getNumpyReplaceLocations(rawData)
        if replaceLocs.any():
            if not copied:
                rawData = rawData.copy()
                copied = True
            rawData = replaceNumpyValues(rawData, replaceLocs)
    elif scipy.sparse.issparse(rawData):
        replaceLocs = getNumpyReplaceLocations(rawData.data)
        if replaceLocs.any():
            if not copied:
                rawData = rawData.copy()
                copied = True
            rawData.data = replaceNumpyValues(rawData.data, replaceLocs)
    elif _isPandasDense(rawData):
        if len(rawData.values) > 0:
            replaceLocs = getNumpyReplaceLocations(rawData.values)
            if replaceLocs.any():
                # .where keeps the True values, use ~ to replace instead
                rawData = rawData.where(~replaceLocs, replaceMissingWith)
                copied = True
    else:
        array = np.array(rawData, dtype=np.object_)
        replaceLocs = getNumpyReplaceLocations(array)
        if replaceLocs.any():
            replaced = replaceNumpyValues(array, replaceLocs)
            rawData = replaced.tolist()
            copied = True

    return rawData, copied


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
            point = np.zeros((self.data.shape[1],))
            row = self.data.row == self.rowIdx
            for val, col in zip(self.data.data[row], self.data.col[row]):
                try:
                    point[col] = val
                except ValueError:
                    point = point.astype(np.object_)
                    point[col] = val

            self.rowIdx += 1
            return point

        raise StopIteration


class GenericRowIterator:
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
            self.iterator = iter(np.array(data))
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

def _setFlattenedRow(data, row, location):
    # first index is only index because this is a feature vector
    if isinstance(row, dict):
        flat = {k: _getFirstIndex(v) for k, v in row.items()}
    else:
        flat = [_getFirstIndex(v) for v in GenericRowIterator(row)]
    if _isScipySparse(data):
        data.data[data.row == location] = flat
    elif _isPandasObject(data):
        data.iloc[location] = flat
    else:
        data[location] = flat

def isHighDimensionData(rawData, rowsArePoints, skipDataProcessing, copied):
    """
    Identify data with more than two-dimensions.

    If the shape of the data container is not defined, 3D data containers are
    are checked and a dimension is dropped if the 2D objects within the
    container are a vector shape that aligns with the rowsArePoints argument.
    """
    if _isScipySparse(rawData) and not rawData.data.size:
        return rawData, False, copied
    try:
        indexZero = _getFirstIndex(rawData)
        if isAllowedSingleElement(indexZero):
            if not skipDataProcessing:
                validateAllAllowedElements(rawData)
            return rawData, False, copied
        indexZeroZero = _getFirstIndex(indexZero)
        if not isAllowedSingleElement(indexZeroZero):
            if hasattr(indexZeroZero, '__len__'):
                objLen = len(indexZeroZero)
            else:
                objLen = sum(1 for _ in GenericRowIterator(indexZeroZero))
            # not feature shaped so must be high dimension
            if (rowsArePoints or objLen > 1 or
                    (objLen == 1 and not
                     isAllowedSingleElement(_getFirstIndex(indexZeroZero)))):
                return rawData, True, copied
            # feature shaped (including empty feature) and rowsArePoints=False
            # so replace with flattened and will be transposed later
            if objLen == 0: # empty need to calculate shape
                toIter = GenericRowIterator(rawData)
                cols = sum(1 for _ in next(toIter))
                rows = sum(1 for _ in toIter) + 1
                return np.empty((rows, cols)), False, True
            if _isNumpyArray(rawData):
                rawData = rawData.squeeze()
                copied = True
            else:
                if not copied:
                    rawData = copy.deepcopy(rawData)
                    copied = True
                if isinstance(rawData, dict): # don't lose keys for names
                    toIter = rawData.items()
                else:
                    toIter = enumerate(GenericRowIterator(rawData))
                for loc, row in toIter:
                    _setFlattenedRow(rawData, row, loc)
                if _isScipySparse(rawData):
                    rawData.eliminate_zeros()

        if not skipDataProcessing:
            toIter = GenericRowIterator(rawData)
            first = next(toIter)
            # if an invalid value is found at this stage, need to use TypeError
            # to raise exception because ImproperObjectAction will continue
            try:
                validateAllAllowedElements(first)
            except ImproperObjectAction as e:
                raise TypeError(str(e)) from e
            firstLength = len(first)
            for i, row in enumerate(toIter):
                if not len(row) == firstLength:
                    msg = "All rows in the data do not have the same "
                    msg += "number of columns. The first row had {0} "
                    msg += "columns but the row at index {1} had {2} "
                    msg += "columns"
                    msg = msg.format(firstLength, i + 1, len(row))
                    raise InvalidArgumentValue(msg)
                try:
                    validateAllAllowedElements(row)
                except ImproperObjectAction as e:
                    raise TypeError(str(e)) from e

        return rawData, False, copied

    except IndexError: # rawData or rawData[0] is empty
        return rawData, False, copied
    except (ImproperObjectAction, InvalidArgumentType): # high dimension Base
        return rawData, True, copied
    except TypeError as e: # invalid non-subscriptable object
        msg = "Numbers, strings, None, and nan are the only "
        msg += "values allowed in nimble data objects"
        raise InvalidArgumentValue(msg) from e

def highDimensionNames(rawData, pointNames, featureNames):
    """
    Names cannot be extracted at higher dimensions because the elements
    are not strings. If 'automatic' we can set to False, if True an
    exception must be raised. If a list, the length must align with
    the dimensions.
    """
    if isinstance(rawData, dict) and (pointNames is True
                                      or pointNames == 'automatic'):
        pointNames = list(rawData.keys())

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
        if dimensions[1] and isinstance(data, dict):
            msg = 'A dict was found nested beyond the first dimension. This '
            msg += 'is not permitted because it is unclear what the keys '
            msg += 'represent for high dimension data'
            raise InvalidArgumentValue(msg)
        if all(map(isAllowedSingleElement, GenericRowIterator(data))):
            toFill.extend(data)
        else:
            for obj in GenericRowIterator(data):
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

    Features are flattened point by point whether np.reshape or
    flattenToOneDimension are used.
    """
    if _isNumpyArray(rawData) and rawData.dtype != np.object_:
        origDims = rawData.shape
        newShape = (rawData.shape[0], np.prod(rawData.shape[1:]))
        rawData = np.reshape(rawData, newShape)
    else:
        if hasattr(rawData, 'shape'):
            numPts = rawData.shape[0]
        else:
            numPts = len(rawData)
        points = GenericRowIterator(rawData)
        firstPoint = next(points)
        firstPointFlat, ptDims = flattenToOneDimension(firstPoint)
        origDims = tuple([numPts] + list(ptDims))
        numFts = len(firstPointFlat)
        rawData = np.empty((numPts, numFts), dtype=np.object_)
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
        returnType, rawData, pointNames, featureNames, name=None,
        convertToType=None, keepPoints='all', keepFeatures='all',
        treatAsMissing=DEFAULT_MISSING, replaceMissingWith=np.nan,
        rowsArePoints=True, copyData=True, skipDataProcessing=False,
        paths=(None, None), extracted=(None, None)):
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

    # record if extraction occurred before we possibly modify *Names parameters
    ptsExtracted = extracted[0] if extracted[0] else pointNames is True
    ftsExtracted = extracted[1] if extracted[1] else featureNames is True

    copied = False
    kwargs = {}
    # point/featureNames, treatAsMissing, etc. may vary so only use the data
    if _isBase(rawData):
        # only use data; point/featureNames, treatAsMissing, etc. may vary
        # _data is always 2D, but _shape could be higher dimension
        kwargs['shape'] = rawData._shape
        if isinstance(rawData, nimble.core.data.BaseView):
            rawData = rawData.copy()
            copied = True
        rawData = rawData._data
    # convert these types as indexing may cause dimensionality confusion
    elif _isNumpyMatrix(rawData):
        rawData = np.array(rawData)
        copied = True
    elif _isScipySparse(rawData) and not scipy.sparse.isspmatrix_coo(rawData):
        rawData = rawData.tocoo()
        copied = True
    # anything we do not recognize turn into a list, this allows for source to
    # be range, generator, map, iter, etc., and other list-like data structures
    elif not (isinstance(rawData, (list, dict)) or
              _isNumpyArray(rawData) or _isPandasObject(rawData) or
              _isScipySparse(rawData)):
        rawData = list(rawData)
        copied = True

    rawData, highDim, copied = isHighDimensionData(rawData, rowsArePoints,
                                                   skipDataProcessing, copied)

    if copyData is None:
        # signals data was constructed internally and can be modified so it
        # can be considered copied for the purposes of this function
        copied = True
    elif copyData and not copied:
        rawData = copy.deepcopy(rawData)
        copied = True

    if not rowsArePoints:
        pointNames, featureNames = featureNames, pointNames
    if highDim:
        if not rowsArePoints:
            msg = 'rowsArePoints cannot be False. Features are ambiguous for '
            msg += 'data with more than two dimensions so rows cannot be '
            msg += 'processed as features.'
            raise ImproperObjectAction(msg)
        # additional name validation / processing before extractNames
        pointNames, featureNames = highDimensionNames(rawData, pointNames,
                                                      featureNames)
        rawData, tensorShape = flattenHighDimensionFeatures(rawData)
        kwargs['shape'] = tensorShape
        copied = True
    # If skipping data processing, no modification needs to be made
    # to the data, so we can skip name extraction and missing replacement.
    if skipDataProcessing:
        if returnType == 'List':
            kwargs['checkAll'] = False
        pointNames = pointNames if pointNames != 'automatic' else None
        featureNames = featureNames if featureNames != 'automatic' else None
    else:
        try:
            rawData, pointNames, featureNames, copied = extractNames(
                rawData, pointNames, featureNames, copied)
        except InvalidArgumentValue as e:
            if not rowsArePoints:
                def swapAxis(match):
                    return 'feature' if match.group(1) == 'point' else 'point'
                msg = re.sub(r'(point|feature)', swapAxis, str(e))
                msg = re.sub(r'rowsArePoints=True', 'rowsArePoints=False', msg)
                # swap axis when rowsArePoints=False
                raise InvalidArgumentValue(msg) from e
            raise
        if treatAsMissing is not None:
            rawData, copied = _replaceMissingData(rawData, treatAsMissing,
                                                  replaceMissingWith, copied)
    # convert data to a type compatible with the returnType init method
    rawData = convertData(returnType, rawData, pointNames, featureNames,
                          copied)

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

    ret = initMethod(rawData, pointNames=usePNames, featureNames=useFNames,
                     name=name, paths=paths, reuseData=True, **kwargs)

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

    if not rowsArePoints:
        ret.transpose(useLog=False)

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

def _isDownloadable(url):
    """
    Does the url contain a downloadable resource.
    """
    headers = requests.head(url, allow_redirects=True).headers
    contentType = headers.get('content-type').lower()
    return not 'html' in contentType

def _isArchive(filename):
    if not hasattr(filename, 'read'):
        return zipfile.is_zipfile(filename) or tarfile.is_tarfile(filename)
    seekLoc = filename.tell()
    # is_zipfile does not seek back to start
    if zipfile.is_zipfile(filename):
        filename.seek(seekLoc)
        return True

    filename.seek(0)
    # is_tarfile does not support file objects until python 3.9
    try:
        tar = tarfile.open(fileobj=filename)
        tar.close()
        return True
    except tarfile.TarError:
        return False
    finally:
        filename.seek(seekLoc)

def _isGZip(filename):
    if hasattr(filename, 'read'):
        saveLoc = filename.tell()
        ret = filename.read(2) == b'\x1f\x8b'
        filename.seek(saveLoc)
        return ret

    with open(filename, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

def _isUCI(source):
    """
    Check if source is a url from UCI or uses UCI shorthand.
    """
    return 'archive.ics.uci.edu' in source or source.startswith('uci::')

def _isUCIDataFile(name):
    """
    UCI convention is that Index file and files ending with .names
    contain dataset details. All other files will be considered data
    files.
    """
    return name != 'Index' and not name.endswith('.names')

def _processUCISource(source):
    """
    UCI source can be UCI shorthand ('uci::<Dataset name>'), a URL to
    UCI main page, or URL to UCI data page. For the first two, the
    source needs to be transformed to point to the data page.
    """
    subDirectories = ''
    if source.startswith('uci::'):
        prefix = 'https://archive.ics.uci.edu/ml/datasets/'
        # sanitize for url. uci replaces whitespace with + character
        toFind = urllib.parse.quote_plus(source[5:].strip(), safe='/')
        # allow for path to subdirectory, ie uci::heart disease/costs
        # split to find the source of the main data page, and store the
        # subDirectories to add back in once the main data page is located
        toFind = toFind.split('/', 1)
        source = prefix + toFind[0].strip()
        if len(toFind) > 1:
            subDirectories = toFind[1].strip()
    if 'ml/datasets/' in source:
        response = requests.get(source)
        archive = 'https://archive.ics.uci.edu/ml/'
        database = r'machine-learning-databases/.+?/'
        data = re.search(database, response.text).group()
        source = archive + data + subDirectories
        if not source.endswith('/'):
            source += '/'

    return source

def _extractFromArchive(ioStream, fromUCI):
    """
    Extract contents of an archive file.
    """
    if zipfile.is_zipfile(ioStream):
        archiveOpen = zipfile.ZipFile
        nameGetter = 'namelist'
        def extractor(fileObj, name):
            return getattr(fileObj, 'open')(name)
    else:
        archiveOpen = lambda f, mode: tarfile.open(mode=mode, fileobj=f)
        nameGetter = 'getnames'
        def extractor(fileObj, name):
            member = getattr(fileObj, 'getmember')(name)
            return getattr(fileObj, 'extractfile')(member)

    ioStream.seek(0)
    with archiveOpen(ioStream, 'r') as fileObj:
        names = getattr(fileObj, nameGetter)()
        if fromUCI:
            # limit only to files containing data
            names = [n for n in names if _isUCIDataFile(n)]
        if len(names) > 1:
            msg = 'Multiple files found in source'
            raise InvalidArgumentValue(msg)
        extracted = extractor(fileObj, names[0])
        return BytesIO(extracted.read())

def _decompressGZip(ioStream):
    """
    Decompress a gzip file.
    """
    with gzip.open(ioStream, 'rb') as unzipped:
        return BytesIO(unzipped.read())

def createDataFromFile(
        returnType, source, pointNames, featureNames, name, convertToType,
        keepPoints, keepFeatures, treatAsMissing, replaceMissingWith,
        rowsArePoints, ignoreNonNumericalFeatures, inputSeparator):
    """
    Helper for nimble.data which deals with the case of loading data
    from a file. Returns a triple containing the raw data, pointNames,
    and featureNames (the later two following the same semantics as
    nimble.data's parameters with the same names).
    """
    encoding = locale.getpreferredencoding() # default encoding for open()
    isUCI = False
    # Case: string value means we need to open the file, either directly or
    # through an http request
    if isinstance(source, str):
        if os.path.exists(source):
            content = open(source, 'rb', newline=None).read()
            path = source
        else: # webpage
            isUCI = _isUCI(source.lower())
            if not isUCI and source[:4].lower() != 'http':
                msg = 'The source is not a path to an existing file and does '
                msg += 'not start with "http" so it cannot be processed as a '
                msg += 'url'
                raise InvalidArgumentValue(msg)
            if not requests.nimbleAccessible():
                msg = "To load data from a webpage, the requests module "
                msg += "must be installed"
                raise PackageException(msg)
            if isUCI:
                source = _processUCISource(source.lower())
            response = requests.get(source, stream=True)
            if not response.ok:
                msg = "The data could not be accessed from the webpage. "
                msg += "HTTP Status: {0}, ".format(response.status_code)
                msg += "Reason: {0}".format(response.reason)
                raise InvalidArgumentValue(msg)
            if isUCI and not _isDownloadable(source):
                # ignore Parent Directory at index 0 in every repository
                hrefs = re.findall('href="(.+)"', response.text)[1:]
                dataFiles = [href for href in hrefs if _isUCIDataFile(href)]
                if len(dataFiles) > 1:
                    msg = 'This UCI source contains multiple data files. '
                    msg += 'Provide a url to a specific file.'
                    raise InvalidArgumentValue(msg)
                source += dataFiles[0]
                response = requests.get(source, stream=True)

            path = source
            if name is None:
                if "Content-Disposition" in response.headers:
                    contentDisp = response.headers["Content-Disposition"][0]
                    name = contentDisp.split('filename=')[1]
                else:
                    name = source.split("/")[-1]

            content = response.content
            if response.apparent_encoding is not None:
                encoding = response.apparent_encoding

    # Case: we are given an open file already
    else:
        saved = source.tell()
        content = source.read()
        source.seek(saved)

        if not isinstance(content, bytes):
            encoding = source.encoding
            content = bytes(content, encoding=encoding)

        # try getting name attribute from file
        if hasattr(source, 'name'):
            path = source.name
        else:
            path = None

    # check if need to decompress or extract file
    with BytesIO(content) as toCheck:
        if _isGZip(toCheck):
            content = _decompressGZip(toCheck).read()
        elif _isArchive(toCheck):
            content = _extractFromArchive(toCheck, isUCI).read()

    with BytesIO(content) as toCheck:
        extension = _autoFileTypeChecker(toCheck)

    if extension != 'csv':
        ioStream = BytesIO(content)
    else:
        ioStream = StringIO(content.decode(encoding), newline=None)

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
            emptyIsMissing = '' in treatAsMissing
            loaded = _loadcsvUsingPython(
                toLoad, pointNames, featureNames, ignoreNonNumericalFeatures,
                keepPoints, keepFeatures, inputSeparator, emptyIsMissing)
            selectSuccess = True
        elif extension == 'mtx':
            loaded = _loadmtxForAuto(toLoad, pointNames, featureNames,
                                     encoding)
        elif extension in ['hdf5', 'h5']: # h5 and hdf5 are synonymous
            loaded = _loadhdf5ForAuto(toLoad, pointNames, featureNames)

    retData, retPNames, retFNames = loaded

    if path is None:
        pathsToPass = (None, None)
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

    if path is not None and name is None:
        tokens = path.rsplit(os.path.sep)
        name = tokens[-1]

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
        returnType, retData, retPNames, retFNames, name, convertToType,
        keepPoints, keepFeatures, treatAsMissing=treatAsMissing,
        replaceMissingWith=replaceMissingWith, rowsArePoints=rowsArePoints,
        copyData=None, paths=pathsToPass, extracted=extracted)


def createConstantHelper(numpyMaker, returnType, numPoints, numFeatures,
                         pointNames, featureNames, name):
    """
    Create nimble data objects containing constant values.

    Use np.ones or np.zeros to create constant nimble objects of
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
        if numpyMaker is np.ones:
            rawDense = numpyMaker((numPoints, numFeatures))
            rawSparse = scipy.sparse.coo_matrix(rawDense)
        elif numpyMaker is np.zeros:
            rawSparse = scipy.sparse.coo_matrix((numPoints, numFeatures))
        else:
            raise ValueError('numpyMaker must be np.ones or np.zeros')
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


def _checkCSVForNames(ioStream, pointNames, featureNames, dialect):
    """
    Finds the first two lines of data (ignoring comments) to determine whether
    point and/or feature names will be extracted from the data.
    """
    startPosition = ioStream.tell()

    # walk past all the comments
    currLine = "#"
    while currLine.startswith('#'):
        currLine = ioStream.readline()

    # Use the robust csv reader to read the first two lines (if available)
    # these are saved to use in further autodetection
    ioStream.seek(startPosition)
    rowReader = csv.reader(ioStream, dialect)
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
    ioStream.seek(startPosition)

    return (pointNames, featureNames)


def _filterCSVRow(row):
    if len(row) == 0:
        return False
    if row[0] == '\n':
        return False
    return True


def _advancePastComments(ioStream):
    """
    Take an open file and advance until we find a line that isn't empty
    and doesn't start with the comment character. Returns the number
    of lines that were skipped
    """
    numSkipped = 0
    while True:
        # If we read a row that isn't a comment line, we
        # have to undo our read
        startPosition = ioStream.tell()

        row = ioStream.readline()
        if len(row) == 0:
            numSkipped += 1
            continue
        if row[0] == '#':
            numSkipped += 1
            continue
        if row[0] == '\n':
            numSkipped += 1
            continue

        ioStream.seek(startPosition)
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

def _detectDialectFromSeparator(ioStream, inputSeparator):
    "find the dialect to pass to csv.reader based on inputSeparator"
    startPosition = ioStream.tell()
    # skip commented lines
    _ = _advancePastComments(ioStream)
    if inputSeparator == 'automatic':
        # detect the delimiter from the first line of data
        dialect = csv.Sniffer().sniff(ioStream.readline())
    elif len(inputSeparator) > 1:
        msg = "inputSeparator must be a single character"
        raise InvalidArgumentValue(msg)
    elif inputSeparator == '\t':
        dialect = csv.excel_tab
    else:
        dialect = csv.excel
        dialect.delimiter = inputSeparator

    # reset everything to make the loop easier
    ioStream.seek(startPosition)

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


def _loadmtxForAuto(ioStream, pointNames, featureNames, encoding):
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
    startPosition = ioStream.tell()
    seenPNames = False
    retPNames = None
    retFNames = None

    # ioStream will always be bytes
    # read through the comment lines
    while True:
        currLine = ioStream.readline()
        # need slice not index for bytes comparison
        if currLine[:1] != b'%':
            break
        if len(currLine) > 1 and currLine[1:2] in [b"#", '#']:
            # strip '%#' from the begining of the line
            scrubbedLine = currLine[2:]
            # strip newline from end of line
            scrubbedLine = scrubbedLine.rstrip()
            names = list(map(lambda s: s.decode(encoding),
                             scrubbedLine.split(b',')))
            if not seenPNames:
                retPNames = names if names != [''] else None
                seenPNames = True
            else:
                retFNames = names if names != [''] else None

    ioStream.seek(startPosition)
    data = scipy.io.mmread(ioStream)

    if pointNames is True or featureNames is True:
        # the helpers operate based on positional inputs with a None
        # sentinal indicating no extration. So we need to convert from
        # the nimble.data input semantics
#        pNameID = 0 if pointNames is True else None
#        fNameID = 0 if featureNames is True else None
        if scipy.sparse.issparse(data):
            extractor = extractNamesFromScipySparse
        else:
            extractor = extractNamesFromNumpy

        data, extPNames, extFNames, _ = extractor(data, pointNames,
                                                  featureNames, True)

    else:
        extPNames = None
        extFNames = None

    # choose between names extracted automatically from comments
    # (retPNames) vs names extracted explicitly from the data
    # (extPNames). extPNames has priority.
    retPNames = extPNames if retPNames is None else retPNames
    retFNames = extFNames if retFNames is None else retFNames

    return data, retPNames, retFNames


def _loadhdf5ForAuto(ioStream, pointNames, featureNames):
    """
    Use h5py module to load high dimension data. The ioStream is used
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

    with h5py.File(ioStream, 'r') as hdf:
        data = []
        pnames = []
        expShape = None
        for key, val in hdf.items():
            ptData = extractArray(val)
            ptShape = np.array(ptData).shape
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
    ioStream.seek(0)
    includePtNames = ioStream.readline().startswith(b'includePointNames')
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
            msg += 'Dataset will be loaded as a single point, or {0} '
            msg += 'indicating the data in the Dataset will be loaded '
            msg += 'directly, but pointNames contained {1} names'
            raise InvalidArgumentValue(msg.format(innerShape, numNames))
    elif len(data) == 1 and not pointNames:
        data = data[0]

    if pointNames is True or (includePtNames and pointNames is None):
        pointNames = pnames

    return data, pointNames, featureNames


def _loadcsvUsingPython(ioStream, pointNames, featureNames,
                        ignoreNonNumericalFeatures, keepPoints, keepFeatures,
                        inputSeparator, emptyIsMissing):
    """
    Loads a csv file using a reader from python's csv module.

    Parameters
    ----------
    ioStream : open file like object
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
    dialect = _detectDialectFromSeparator(ioStream, inputSeparator)
    pointNames, featureNames = _checkCSVForNames(ioStream, pointNames,
                                                 featureNames, dialect)

    pointNames = _namesDictToList(pointNames, 'point', 'pointNames')
    featureNames = _namesDictToList(featureNames, 'feature', 'featureNames')

    # Advance the file past any beginning of file comments, record
    # how many are skipped
    skippedLines = _advancePastComments(ioStream)
    # remake the file iterator to ignore empty lines
    filtered = filter(_filterCSVRow, ioStream)
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
    # possibly remove last feature if all values are '' and '' is missing value
    lastFtRemovable = not limitFeatures and emptyIsMissing
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
        if lastFtRemovable and row[-1] != '':
            lastFtRemovable = False
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

    # remove last feature of all missing values if no feature name is provided
    if (lastFtRemovable and
            (not retFNames or len(retFNames) == firstRowLength - 1
             or (featureNames is True and retFNames[-1] == ''))):
        for row in retData:
            row.pop()
        if featureNames is True:
            retFNames.pop()

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

def _processArchiveFile(filename, overwrite, allowMultiple):
    """
    Extract contents of an archive file.

    Return a list of paths to the extracted files. If the file contains
    multiple files and allowMultiple is False, the archive file will be
    returned and no extraction will occur.
    """
    if zipfile.is_zipfile(filename):
        archiveObj = zipfile.ZipFile
        nameGetter = 'namelist'
    else:
        archiveObj = tarfile.TarFile
        nameGetter = 'getnames'
    with archiveObj(filename, 'r') as fileObj:
        names = getattr(fileObj, nameGetter)()
        if any(os.path.isabs(name) or '..' in name for name in names):
            # potential security risk will not perform extraction
            return [filename]

        if not allowMultiple and len(names) > 1:
            return [filename]

        location = os.path.dirname(filename)
        paths = [os.path.join(location, name) for name in names]
        # only need to extract if overwrite or any expected files don't exist
        if overwrite or not all(os.path.exists(path) for path in paths):
            fileObj.extractall(location)

        files = []
        for path in paths:
            if os.path.isfile(path):
                files.append(path)

        return files

def _processCompressedFile(filename):
    """
    Decompress a gzip file.
    """
    with gzip.open(filename, 'rb') as fIn:
        if filename.endswith('.gz'):
            unzipped = filename[:-3]
        else:
            unzipped = filename
        with open(unzipped, 'wb') as fOut:
            shutil.copyfileobj(fIn, fOut)
        return [unzipped]

def _findData(url, filename, overwrite, allowMultiple):
    """
    Find data locally or download and store.

    Assumes that url is downloadable. The data may be available locally
    if previously stored, but if not found the data will be downloaded
    and stored.

    Return a list of data file paths.
    """
    if os.path.exists(filename):
        if _isArchive(filename):
            return _processArchiveFile(filename, overwrite, allowMultiple)
        if not overwrite:
            isGZip = _isGZip(filename)
            if isGZip and os.path.exists(filename[:-3]):
                return [filename[:-3]]
            if not isGZip:
                return [filename]

    directory = os.path.split(filename)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    response = requests.get(url, allow_redirects=True)
    with open(filename, 'w+b') as f:
        f.write(response.content)

    try:
        if _isArchive(filename):
            return _processArchiveFile(filename, overwrite, allowMultiple)
        if _isGZip(filename):
            return _processCompressedFile(filename)
    except Exception: # pylint: disable=broad-except
        pass # return the archive file on failure

    return [filename]

def _findUCIData(source, path, currPaths, overwrite, allowMultiple):
    """
    Recursive download for files from a UCI repository.
    """
    response = requests.get(source)
    # ignore Parent Directory at index 0 in every repository
    hrefs = re.findall('href="(.+)"', response.text)[1:]
    # Index file and .names files contain dataset details, not data so they
    # will be downloaded but ignored for return with fetchFile.
    ignore = False
    if not allowMultiple and len(hrefs) > 1:
        if len([href for href in hrefs if _isUCIDataFile(href)]) > 1:
            msg = 'The source contains multiple files. Use nimble.fetchFiles '
            msg += 'or provide the url to a specific file.'
            raise InvalidArgumentValue(msg)
        ignore = True
    for href in hrefs:
        url = source + href
        if _isDownloadable(url):
            urlInfo = urllib.parse.urlparse(url)
            name = os.path.split(urlInfo.path)[1]
            filename = os.path.join(path, name)
            paths = _findData(url, filename, overwrite, allowMultiple)
            if not ignore or _isUCIDataFile(href):
                currPaths.extend(paths)
        else:
            newSource = url
            newPath = os.path.join(path, href)
            currPaths = _findUCIData(newSource, newPath, currPaths, overwrite,
                                     allowMultiple)

    return currPaths

def fileFetcher(source, overwrite, allowMultiple=True):
    """
    Download data from the web and store at a specified location. Files
    are stored in a directory named 'nimbleData' that is placed, by
    default, in the home directory (pathlib.Path.home()). The location
    can be changed in configuration.ini. The dataset is only downloaded
    once. Any subsequent calls for the same source will identify that
    the data is locally available unless overwrite is True.

    The storage location is based on parsing the url. Within the
    nimbleData directory, a directory is created for the domain. A
    directory based on the url path is added to the domain directory and
    the file or extracted archive contents are placed in the path
    directory.
    """
    if not requests.nimbleAccessible():
        raise PackageException('requests must be installed')
    isUCI = _isUCI(source.lower())
    if isUCI:
        source = _processUCISource(source.lower())
    configLoc = nimble.settings.get('fetch', 'location')
    homepath = os.path.join(configLoc, 'nimbleData')
    urlInfo = urllib.parse.urlparse(source)
    netLoc = urlInfo.netloc
    if netLoc.startswith('www.'):
        netLoc = netLoc[4:]
    directory = os.path.join(*urlInfo.path.split('/'))
    dirPath = os.path.join(homepath, netLoc, directory)
    if _isDownloadable(source):
        return _findData(source, dirPath, overwrite, allowMultiple)
    if isUCI:
        return _findUCIData(source, dirPath, [], overwrite, allowMultiple)

    msg = 'source did not provide downloadable data'
    raise InvalidArgumentValue(msg)
