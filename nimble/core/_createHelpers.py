
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
import pickle
import numbers
import itertools
import time
from collections.abc import Sequence

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

NUM_TUPLE = (bool, int, float, np.number)

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


def isNum(value):
    """
    Determine if a value is numerical.
    """
    if type(value) == type:   
        return  issubclass(value, NUM_TUPLE)
    else:
        return isinstance(value, NUM_TUPLE)
    

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
    retPNames = None
    retFNames = None
 
    # Special consideration for Structured Numpy arrays
    def _convertStructuredNumpyToDataFrame(data):
        data = pd.DataFrame(data, columns=retFNames)
        fnamesID = True
        return extractNamesFromPdDataFrame(data, pnamesID, fnamesID, copied)
    
    if data.dtype.fields:
        retFNames = [x for x in data.dtype.fields.keys()]
        reshapedData = [list(data[x]) for  x in range(len(data))]
        allNumeric = list(map(isNum, reshapedData[0]))
        if all(allNumeric):
            data = np.array(reshapedData)
        else:
            if pnamesID == True:
                if all(allNumeric[1:]):
                    retFNames.pop(0)
                    data = list()
                    retPNames = list()
                    for i in reshapedData:
                        data.append(i[1:])
                        retPNames.append(i[0])
                    data = np.array(data)
                    return data, retPNames, retFNames, copied
                else:
                    return _convertStructuredNumpyToDataFrame(reshapedData)
            else:
                return _convertStructuredNumpyToDataFrame(reshapedData)

           
    if len(data.shape) == 1:
        data = data.reshape(1, data.shape[0])
        addedDim = True

    firstRow = data[0] if len(data) > 0 else None
    secondRow = data[1] if len(data) > 1 else None
    pnamesID, fnamesID = autoDetectNamesFromRaw(pnamesID, fnamesID, firstRow,
                                                secondRow)
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
        # Validates consistency between pointNames (passed in by user),
        # ptNames (embedded in data), and rawData (order consistency).
        pointNames, rawData = _dictNames('point', pointNames, ptNames, rawData)

        # This was a feature vector dict, we have extracted the pointNames
        # and there is nothing more to pull out. But we need to transpose
        # rawData so it will be interpreted as a feature vector
        if isAllowedSingleElement(rawData[0]):
            rawData = list(map(lambda val: [val], rawData))
            copied = True
            featureNames = None
            return rawData, pointNames, featureNames, copied

        # if we now have list of lists or the like, featureNames are
        # disallowed from being embedded in them.
        if isinstance(rawData[0], Sequence):
            if featureNames is True:
                msg = 'featureNames cannot be True when source is a dict'
                raise InvalidArgumentValue(msg)

        # rawData is now a list of lists, a list of dicts, or
        # something else. In all these cases we go back to extractNames
        # to handle it.
        return extractNames(rawData, pointNames, featureNames, copied, False)

    # rawData={}
    rawData = np.empty([0, 0])
    copied = True
    if pointNames is True:
        msg = 'pointNames cannot be True when data is an empty dict'
        raise InvalidArgumentValue(msg)

    # empty object has no names.
    return rawData, None, None, copied

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
            msg = f"The keys at index {i + 1} do not match "
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
                    msg += f'The feature names at index {i} are different '
                    msg += 'than those in the first feature'
                    raise InvalidArgumentValue(msg)

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
    # If user has specifified ignoring the embedded names, we do no checks
    # and we mark the name contents as None
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
        msg = f"{package} must be installed to create a {returnType} object"
        raise PackageException(msg) from e

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
        elif (returnType == 'Matrix' and
              (len(rawData.shape) == 1 or
              not allowedNumpyDType(rawData.dtype))):
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
            return rawData.to_numpy()
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
            for i, (idx, ft) in enumerate(data.items()):
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
        msg += f"'{convertToType}'. {repr(error)}"
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

    if _isNumpyArray(rawData):
        replaceLocs = getNumpyReplaceLocations(rawData)
        if replaceLocs.any():
            if not copied:
                rawData = rawData.copy()
                copied = True
            rawData = replaceNumpyValues(rawData, replaceLocs)
    elif scipy.nimbleAccessible() and scipy.sparse.issparse(rawData):
        replaceLocs = getNumpyReplaceLocations(rawData.data)
        if replaceLocs.any():
            if not copied:
                rawData = rawData.copy()
                copied = True
            rawData.data = replaceNumpyValues(rawData.data, replaceLocs)
    elif _isPandasDense(rawData):
        if len(rawData.to_numpy()) > 0:
            replaceLocs = getNumpyReplaceLocations(rawData.to_numpy())
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
            self.iterator = iter(data.to_numpy())
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

    except IndexError: # rawData or rawData[0] is empty
        return rawData, False, copied
    except (ImproperObjectAction, InvalidArgumentType): # high dimension Base
        return rawData, True, copied
    except TypeError as e: # invalid non-subscriptable object
        msg = "Number, string, None, nan, and datetime objects are "
        msg += "the only elements allowed in nimble data objects"
        raise InvalidArgumentValue(msg) from e

    return rawData, False, copied

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
            msg = f'{axes} names cannot be True for data with more than two '
            msg += "dimensions "
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
    if _isBase(data) and (len(data._dims) > 2 or data.shape[0] > 1):
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
                msg += f'first point had dimensions {ptDims}, but point '
                msg += f'{i + 1} had dimensions {dims}'
                raise InvalidArgumentValue(msg)
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
            axis = axisObj._axis
            keep = 'keepPoints' if axis == 'point' else 'keepFeatures'
            name = axisObj.getName(converted)
            msg = f"Values in {keep} must represent unique {axis}s. "
            msg += f"'{name}' and {converted} represent the same {axis}. "
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
                msg = f"The keys '{ftName}' and {i} represent the same "
                msg += "feature but have different values"
                raise InvalidArgumentValue(msg)
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
            msg = f'The key(s) {fail} in convertToType are not valid for '
            msg += 'this object'
            raise InvalidArgumentValue(msg)

    return convertList

def analyzeValues(rawData, returnType, skipDataProcessing):
    """
    Validates the data values and determines the returnType.

    Both validation and returnType determination require iteration
    through the data, but each is only performed if necessary.
    """
    invalid = "Number, string, None, nan, and datetime objects are the "
    invalid += "only elements allowed in nimble data objects"
    toIter = GenericRowIterator(rawData)
    try:
        first = next(toIter)
    except StopIteration:
        if returnType is None:
            returnType = "Matrix"
        return returnType

    # processing has already occurred for 1D objects only need returnType
    if isAllowedSingleElement(first):
        if returnType is None:
            if len(set(map(type, rawData))) <= 1:
                returnType = "Matrix"
            else:
                returnType = "DataFrame"
        return returnType

    # restore first line to front, so it's included in testing and
    # returnType checking
    toIter = itertools.chain([first], toIter)

    firstType = None
    firstLength = None
    for i, row in enumerate(toIter):
        if isinstance(row, dict):
            row = list(row.values())
        if firstLength is None:
            firstLength = len(row)
        elif not skipDataProcessing and not len(row) == firstLength:
            msg = "All rows in the data do not have the same number of "
            msg += f"columns. The first row had {firstLength} columns but the "
            msg += f"row at index {i} had {len(row)} columns"
            raise InvalidArgumentValue(msg)
        for val in row:
            if not skipDataProcessing and not isAllowedSingleElement(val):
                raise InvalidArgumentValue(invalid)
            if returnType is None:
                valType = type(val)
                if firstType is None:
                    firstType = valType
                    if issubclass(firstType, str):
                        returnType = "DataFrame"
                        if skipDataProcessing: # only need the returnType
                            return returnType
                elif firstType != valType:
                    ftBool = issubclass(firstType, (bool, np.bool_))
                    ftNum = issubclass(firstType, (numbers.Number, np.number))
                    vtBool = issubclass(valType, (bool, np.bool_))
                    vtNum = issubclass(valType, (numbers.Number, np.number))
                    # If all non-bool numeric, record umbrella type and
                    # keep checking for bad values / non-numeric
                    if (ftNum and not ftBool) and (vtNum and not vtBool):
                        fDt = np.dtype(firstType)
                        vDt = np.dtype(valType)
                        umbrella = max(fDt, vDt)
                        firstType = firstType if umbrella == fDt else valType
                    # Column with non-numeric value, use DataFrame
                    else:
                        returnType = "DataFrame"
                        if skipDataProcessing: # only need the returnType
                            return returnType
    if returnType is None: # data was homogenous (or all int/float numeric)
        returnType = "Matrix"
    return returnType

def initDataObject(
        rawData, pointNames, featureNames, returnType, name=None,
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

    # record if extraction occurred before we possibly modify *Names parameters
    ptsExtracted = extracted[0] if extracted[0] else pointNames is True
    ftsExtracted = extracted[1] if extracted[1] else featureNames is True

    # When returnType is None, can set it now if rawData is a Base, numpy,
    # pandas, or scipy object. For other types, will determine later.
    copied = False
    kwargs = {}
    # point/featureNames, treatAsMissing, etc. may vary so only use the data
    if _isBase(rawData):
        # only use data; point/featureNames, treatAsMissing, etc. may vary
        # _data is always 2D, but _shape could be higher dimension
        kwargs['shape'] = rawData._dims
        if isinstance(rawData, nimble.core.data.BaseView):
            rawData = rawData.copy()
            copied = True
        if pointNames == 'automatic' or pointNames is True:
            pointNames = rawData.points._getNamesNoGeneration()
        if featureNames == 'automatic' or featureNames is True:
            featureNames = rawData.features._getNamesNoGeneration()
        if returnType is None:
            returnType = rawData.getTypeString()
        rawData = rawData._data
    # convert these types as indexing may cause dimensionality confusion
    elif _isNumpyArray(rawData):
    # decide if numpy structured array should be Nimble DataFrame not Matrix
        if rawData.dtype.fields:
            rowTuple = rawData[0]
            if len(rowTuple) > 0:
                allNumeric = list(map(isNum, rowTuple))
                if not all(allNumeric):
                    returnType = "DataFrame"
                if pointNames is True:
                    if all(allNumeric[1:]):
                        returnType = "Matrix"
        if _isNumpyMatrix(rawData):
            rawData = np.array(rawData)
            copied = True
        if returnType is None:
            returnType = "Matrix"
    elif _isScipySparse(rawData):
        if not scipy.sparse.isspmatrix_coo(rawData):
            rawData = rawData.tocoo()
            copied = True
        if returnType is None:
            returnType = "Sparse"
    elif _isPandasObject(rawData):
        if _isPandasSparse(rawData):
            if hasattr(rawData, 'sparse'):
                rawData = rawData.sparse.to_coo()
            else:
                rawData = rawData.to_coo()
            if returnType is None:
                returnType = "Sparse"
        elif returnType is None:
            returnType = "DataFrame"
    elif returnType is None and not pd.nimbleAccessible():
        returnType = "Matrix" # can't use DataFrame so default to Matrix
    # anything we do not recognize turn into a list, this allows for source to
    # be range, generator, map, iter, etc., and other list-like data structures
    elif not isinstance(rawData, (list, dict)):
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
        if returnType is None:
            returnType = "Matrix"
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
    if not skipDataProcessing or returnType is None:
        returnType = analyzeValues(rawData, returnType, skipDataProcessing)
    
    # Decide if nimble Matrix is better as DataFrame given convertToType input.
    if returnType == "Matrix":
        matrixConvertTypes = list(NUM_TUPLE) + [np.datetime64, None]
        if type(convertToType) == list:
            if len(set(convertToType)) == 1:
                if not isNum(convertToType[0]) and convertToType[0] not in matrixConvertTypes:
                    returnType = "DataFrame"
            else:
                returnType = "DataFrame"
        elif not isNum(convertToType) and convertToType not in matrixConvertTypes:
            returnType = "DataFrame"
    
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
                msg = "Invalid length of convertToType. convertToType must be "
                msg += f"either the length of the full dataset ({numFts}) or "
                msg += f"the length of the limited dataset ({len(cleaned)}), "
                msg += f"but was length {len(convertToType)}."
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
            numFts=len(ret.features)
            numElems=len(convertToType)
            msg = 'A list for convertToType must have many elements as '
            msg += f'features in the data. The object contains {numFts} '
            msg += f'features, but convertToType has {numElems} elements.'
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
    Does the url contain a downloadable resource? Rough check
    is that it does NOT contain an html content type.
    """
    if not requests.nimbleAccessible():
        raise PackageException('requests must be installed')
    try:
        head = requests.head(url, allow_redirects=True)
        head.raise_for_status()
        headers = head.headers
        if 'content-type' in headers.keys():
            contentType = headers.get('content-type')
            return 'html' not in contentType
        return True
    except requests.exceptions.RequestException:
        return False

def _isArchive(filename):
    seekLoc = filename.tell() if hasattr(filename, 'read') else None
    try:
        # is_zipfile does not seek back to start
        if zipfile.is_zipfile(filename):
            return True
    finally:
        if seekLoc is not None:
            filename.seek(seekLoc)

    try:
        if tarfile.is_tarfile(filename):
            return True
    except (TypeError, tarfile.ReadError):
        # filename is definitely a file object at this point
        # is_tarfile does not support file objects until python 3.9
        try:
            with tarfile.open(fileobj=filename):
                return True
        except tarfile.TarError:
            pass

    return False


def _isGZip(filename):
    if hasattr(filename, 'read'):
        saveLoc = filename.tell()
        ret = filename.read(2) == b'\x1f\x8b'
        filename.seek(saveLoc)
        return ret

    with open(filename, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

def _processArchiveFile(filename, extract):
    """
    Extract contents of an archive file.

    Return a list of paths to the extracted files.
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

        location = os.path.dirname(filename)
        paths = [os.path.join(location, name) for name in names]
        # only need to extract if overwrite or any expected files don't exist
        if extract or not all(os.path.exists(path) for path in paths):
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

def _processUCISource(source):
    """
    UCI source can be UCI shorthand ('uci::<Dataset name>'),
    or URL to UCI data page. For the first, the source needs
    to be processed to find the data page. For both, then data
    page is then processed to find the download link.

    If a link directly to a dataset is passed in, it will
    go through this helper untouched.
    """
    if not requests.nimbleAccessible():
        raise PackageException('requests must be installed')
    pagePrefix = "https://archive.ics.uci.edu/"

    source = source.lower()
    if source.startswith('uci::'):
        # sanitize for url. uci replaces whitespace with + character
        toFind = urllib.parse.quote_plus(source[5:].strip(), safe='/')

        # Use search page to find dataset specific page
        searchURL = "https://archive.ics.uci.edu/datasets?search="
        searchURL += toFind.strip()
        searchResponse = requests.get(searchURL)
        # for a normal user this ought not to be very relevant, but in certain
        # automated cases (such as building the online documentation) this rate
        # limits requests to a more civil level.
        time.sleep(1)

        # have to escape out the pluses for the RE
        toFindREForm = toFind.strip().replace("+", r"\+")
        # we're looking for links that includes dataset, then
        # some combination of numbers (the unknown dataset ID),
        # then the known dataset name (with pluses)
        linkMatch = r'dataset/[0-9]+/' + toFindREForm
        suffix = re.search(linkMatch, searchResponse.text).group()

        source = pagePrefix + suffix

    if "dataset" in source:
        # rip the dataset id number from the provided link to make a url
        # directly to the zip file; doing so avoids another request to the
        # website that would be needed to rip the download link directly
        # from the page.
        toFind = source.split("dataset/")
        downloadPrefix = "/static/public/"
        subtarget = toFind[1].strip()
        source = pagePrefix + downloadPrefix + subtarget + ".zip"

    if "static/public" not in source:
        msg = "To access a file from UCI one must pass in a direct link to the file, "
        msg += "a linkt to the dataset's page, or the 'uci::' shorthand with the "
        msg += "dataset's name."
        raise InvalidArgumentValue(msg)

    return source

def _processNimbleSource(source):
    prefix = 'https://www.nimbledata.org/datasets.html'
    if source.lower().startswith('nimble::'):
        source = source[8:]
        source = prefix + '#' + '-'.join(source.lower().split())
    elif '/examples/' in source.lower():
        example = source.split('/examples/')[1].split('.')[0]
        source = prefix + '#' + example.lower()
    return source

def _urlSourceProcessor(source):
    lower = source.lower()
    if 'archive.ics.uci.edu' in lower or lower.startswith('uci::'):
        source = _processUCISource(source)
        database = 'uci'
    elif 'nimbledata.org' in lower or lower.startswith('nimble::'):
        source = _processNimbleSource(source)
        database = 'nimble'
    else:
        database = None
    return source, database

def _getNimbleDataPath(source):
    configLoc = nimble.settings.get('fetch', 'location')
    homepath = os.path.join(configLoc, 'nimbleData')
    urlInfo = urllib.parse.urlparse(source)
    netLoc = urlInfo.netloc
    if netLoc.startswith('www.'):
        netLoc = netLoc[4:]
    directory = os.path.join(*urlInfo.path.split('/'))
    localPath = os.path.join(homepath, netLoc, directory)

    return localPath

def _getURLResponse(source):
    if not requests.nimbleAccessible():
        raise PackageException('requests must be installed')
    response = requests.get(source, allow_redirects=True)
    if not response.ok:
        msg = "The data could not be accessed from the webpage. "
        msg += f"HTTP Status: {response.status_code}, "
        msg += f"Reason: {response.reason}"
        raise InvalidArgumentValue(msg)

    return response


class _DirectURLManager:
    """
    For handling URLs to downloadable files.

    The source must be a single file that can be downloaded. If the file
    is a zip, gzip or tar file, an attempt will be made to extract the
    contents, when allowed.
    """
    def __init__(self, source, checkLocal=True):
        self.source = source
        self.path = _getNimbleDataPath(self.source)
        if checkLocal and os.path.exists(self.path):
            if not os.path.isfile(self.path):
                # need to use _IndirectURLManager
                msg = 'source does not provide downloadable data'
                raise InvalidArgumentValue(msg)
            if _isArchive(self.path):
                self.local = _processArchiveFile(self.path, False)
            else:
                isGZip = _isGZip(self.path)
                if isGZip and os.path.exists(self.path[:-3]):
                    # already decompressed gzip file
                    self.local = [self.path[:-3]]
                elif isGZip:
                    self.local = _processCompressedFile(self.path)
                else:
                    self.local = [self.path]
        else:
            self.local = None
        if self.local is None and not _isDownloadable(self.source):
            msg = 'source does not provide downloadable data'
            raise InvalidArgumentValue(msg)

    def fetch(self, overwrite=True):
        """
        Get from local storage or download.
        """
        # for calls from fileFetcher, overwrite is always True because
        # local storage check has already been performed
        if not overwrite and self.local is not None:
            return self.local

        response = _getURLResponse(self.source)

        directory = os.path.split(self.path)[0]
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.path, 'w+b') as f:
            f.write(response.content)

        self.local = self.path

        try:
            if _isArchive(self.path):
                return _processArchiveFile(self.path, overwrite)
            if _isGZip(self.path):
                return _processCompressedFile(self.path)
        except Exception: # pylint: disable=broad-except
            pass # return the archive file on failure

        return [self.path]

class _IndirectURLManager:
    """
    Support downloading from a page that indicates the data to load.

    Base class for special cases where the URL does not provide a link
    to a downloadable data set, but instead provides a page that relates
    to a given data set.
    """

    def __init__(self, source):
        self.source = source
        self.path = _getNimbleDataPath(self.source)
        self.response = _getURLResponse(self.source)

        self.hrefs = self.getHRefs()
        self.hrefsData = [f for f in self.hrefs if self.isDataFile(f)]
        if not self.hrefsData:
            msg = 'The source did not provide any files to download'
            raise InvalidArgumentValue(msg)

    def getHRefs(self):
        """
        Get the hrefs on this page that point to the files to download.
        """
        return re.findall('href="(.+?)"', self.response.text)

    def isDataFile(self, filename): # pylint: disable=unused-argument
        """
        Determine if, by convention, the file should contain data.
        """
        return True

    def downloadPrefix(self):
        """
        The prefix to attach to an href.
        """
        return self.source

    def fetch(self, overwrite=True):
        """
        Get local files or download and write to local storage.

        The files may already be present, but this is not detectable
        initially as the path is not to a specific file. During
        iteration files already available will not be rewritten when
        overwrite is False.
        """
        if not overwrite and os.path.exists(self.path):
            # The path will be a directory
            dataFiles = []
            for root, _, files in os.walk(self.path):
                for f in files:
                    if self.isDataFile(f):
                        dataFiles.append(os.path.join(root, f))
            return dataFiles

        currPaths = []
        for name in self.hrefs:
            url = self.downloadPrefix() + name
            paths = fileFetcher(url, overwrite)
            # When multiple are files are allowed we only return paths to all
            # files, otherwise we only return the data file
            if self.isDataFile(name):
                currPaths.extend(paths)

        return currPaths

class _IndirectURLManagerNimble(_IndirectURLManager):
    """
    For a url pointing to a Nimble example page.
    """
    def getHRefs(self):
        hrefs = super().getHRefs()
        permalink = '#' + self.source.rsplit('#', 1)[1]
        start = hrefs.index(permalink)
        # page contains hrefs for all examples, get files specific to source
        sourceHRefs = []
        for href in hrefs[start + 1:]:
            if not '_downloads' in href:
                break # end of downloads for this data
            sourceHRefs.append(href)

        return sourceHRefs

    def downloadPrefix(self):
        return 'https://www.nimbledata.org'

def _extractFromArchive(ioStream, dataFilter):
    """
    Extract contents of an archive file.
    """
    archiveKwargs = {'mode': 'r'}
    if zipfile.is_zipfile(ioStream):
        archiveOpen = zipfile.ZipFile
        archiveKwargs['file'] = ioStream
        nameGetter = 'namelist'
        def extractor(fileObj, name):
            return getattr(fileObj, 'open')(name)
    else:
        archiveOpen = tarfile.open
        archiveKwargs['fileobj'] = ioStream
        nameGetter = 'getnames'
        def extractor(fileObj, name):
            member = getattr(fileObj, 'getmember')(name)
            return getattr(fileObj, 'extractfile')(member)

    ioStream.seek(0)
    with archiveOpen(**archiveKwargs) as fileObj:
        names = getattr(fileObj, nameGetter)()
        if dataFilter is not None:
            # limit only to files containing data
            names = [n for n in names if dataFilter(n)]
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

def _guessURLorPath(toCheck):
    """ takes a string, returns whether we think it is a url or a path.
    For a sufficiently giberish string, it isn't clear how which will be
    returned, but most common cases ought to be correct.
    """
    # if its there it's a path
    if os.path.exists(toCheck):
        return 'path'

    # Known starts
    if toCheck.startswith("www."):
        return 'url'
    # home dir, relative, unix root
    if toCheck.startswith("~") or toCheck.startswith('.') or toCheck.startswith('/'):
        return 'path'

    results = urllib.parse.urlparse(toCheck)
    # probably a windows path
    if len(results.scheme) == 1:
        return 'path'
    # looks like it starts with some kind of protocol (https, http, etc.)
    if len(results.scheme) >= 2:
        return 'url'

    # try adding https:// and see if url parse identifies a plausible domain
    results = urllib.parse.urlparse("http://" + toCheck)
    if len(results.netloc) >= 3:
        return 'url'

    # last resort
    return "path"

def createDataFromFile(
        source, pointNames, featureNames, returnType, name, convertToType,
        keepPoints, keepFeatures, treatAsMissing, replaceMissingWith,
        rowsArePoints, ignoreNonNumericalFeatures, inputSeparator):
    """
    Helper for nimble.data which deals with the case of loading data
    from a file. Returns a triple containing the raw data, pointNames,
    and featureNames (the later two following the same semantics as
    nimble.data's parameters with the same names).
    """
    encoding = locale.getpreferredencoding() # default encoding for open()
    dataFilter = None
    # Case: string value means we need to open the file, either directly or
    # through an http request
    if isinstance(source, str):
        toTry = _guessURLorPath(source)
        if toTry == 'path':
            with open(source, 'rb', newline=None) as f:
                content = f.read()
            path = source
        if toTry == 'url':
            source, database = _urlSourceProcessor(source)
            try:
                urlManager = _DirectURLManager(source, False)
            except InvalidArgumentValue:
                if database == 'nimble':
                    urlManager = _IndirectURLManagerNimble(source)
                else:
                    raise
                # only a source with a single data file reaches this point
                source = urlManager.downloadPrefix() + urlManager.hrefsData[0]
                urlManager = _DirectURLManager(source, checkLocal=False)

            response = _getURLResponse(urlManager.source)
            path = source

            content = response.content
            if response.apparent_encoding is not None:
                encoding = response.apparent_encoding

    # Case: we are given an open file already
    else:
        saved = source.tell()
        content = source.read()
        source.seek(saved)

        if not isinstance(content, bytes):
            if source.encoding is not None:
                encoding = source.encoding
            content = bytes(content, encoding=encoding)

        # try getting name attribute from file
        if hasattr(source, 'name'):
            path = source.name
        else:
            path = None

    try:
        ret = pickle.loads(content)
        return initDataObject(
            ret, pointNames, featureNames, returnType, name, convertToType,
            keepPoints, keepFeatures, treatAsMissing=treatAsMissing,
            replaceMissingWith=replaceMissingWith, rowsArePoints=rowsArePoints,
            copyData=None)
    except Exception: # pylint: disable=broad-except
        pass

    # check if need to decompress or extract file
    with BytesIO(content) as toCheck:
        if _isGZip(toCheck):
            content = _decompressGZip(toCheck).read()
        elif _isArchive(toCheck):
            content = _extractFromArchive(toCheck, dataFilter).read()

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
        retData, retPNames, retFNames, returnType, name, convertToType,
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
            rawDense = numpyMaker((numPoints, numFeatures), dtype=int)
            rawSparse = scipy.sparse.coo_matrix(rawDense, dtype=int)
        elif numpyMaker is np.zeros:
            rawSparse = scipy.sparse.coo_matrix((numPoints, numFeatures),
                                                dtype=int)
        else:
            raise ValueError('numpyMaker must be np.ones or np.zeros')
        return nimble.data(rawSparse, pointNames=pointNames,
                           featureNames=featureNames, name=name, useLog=False)

    raw = numpyMaker((numPoints, numFeatures), dtype=int)
    return nimble.data(raw, pointNames=pointNames, featureNames=featureNames,
                       returnType=returnType, name=name, useLog=False)


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

def _csvColTypeTracking(row, convertCols, nonNumericFeatures,
                        allEmptyFeatures):
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

    if allEmptyFeatures:
        for idx in allEmptyFeatures:
            if row[idx] != '':
                delKeys.append(idx)

    if delKeys:
        for key in delKeys:
            if key in convertCols:
                del convertCols[key]
                nonNumericFeatures.add(key)
            else:
                allEmptyFeatures.remove(key)

def _colTypeConversion(row, convertCols, containsMissing):
    """
    Converts values in each row that are in numeric/boolean columns.

    Since convertCols does not contain non-numeric columns, any empty
    string is considered to be a missing value.
    """
    for idx, cType in convertCols.items():
        val = row[idx]
        if val == '':
            row[idx] = np.nan
            containsMissing.add(idx)
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
    _ = _advancePastComments(ioStream)
    # Use the robust csv reader to read the first two lines (if available)
    # these are saved to use in further autodetection
    rowReader = csv.reader(ioStream, dialect)
    possiblePtNames = False
    try:
        firstRow = next(rowReader)
        while firstRow == []:
            firstRow = next(rowReader)
        secondRow = next(rowReader)
        while secondRow == []:
            secondRow = next(rowReader)
        firstDataRow = []
        secondDataRow = []
        for i, (first, second) in enumerate(zip(firstRow, secondRow)):
            first = _intFloatOrString(first)
            second = _intFloatOrString(second)
            # can treat missing first values as possible default names except
            # when second is also missing so don't include possible defaults
            # for autoDetectNamesFromRaw
            if first is not None or second is None:
                firstDataRow.append(first)
                secondDataRow.append(second)
            elif first is None and not i:
                possiblePtNames = True

    except StopIteration:
        firstDataRow = None
        secondDataRow = None

    pointNames, featureNames = autoDetectNamesFromRaw(
        pointNames, featureNames, firstDataRow, secondDataRow)

    # possible pointNames if first was missing and rest could be featureNames
    trackPoints = possiblePtNames and featureNames

    # reset everything to make the loop easier
    ioStream.seek(startPosition)

    return pointNames, featureNames, trackPoints


def _advancePastComments(ioStream):
    """
    Take an open file and advance until we find a line that isn't empty
    and doesn't start with the comment character. Returns the number
    of lines that were skipped
    """
    numSkipped = 0
    while True:
        startPosition = ioStream.tell()
        row = ioStream.readline()
        if not row: # empty row indicates end of file
            raise FileFormatException('No data found in file')
        if row[0] not in ['#', '\n']:
            ioStream.seek(startPosition)
            break

        numSkipped += 1

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
    """
    Find the dialect to pass to csv.reader based on inputSeparator.
    """
    startPosition = ioStream.tell()
    # skip commented lines
    _ = _advancePastComments(ioStream)
    if inputSeparator == 'automatic':
        # detect the delimiter from the first line of data
        inputSeparator = csv.Sniffer().sniff(ioStream.readline()).delimiter
    elif len(inputSeparator) > 1:
        msg = "inputSeparator must be a single character"
        raise InvalidArgumentValue(msg)
    if inputSeparator == '\t':
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
        msg = f"{varName} cannot contain duplicate values. "
        if len(duplicates) == 1:
            duplicateString = str(list(duplicates)[0])
            msg += f"The value {duplicateString} was duplicated"
        else:
            duplicateString = ",".join(map(str, duplicates))
            msg += f"The values {duplicateString} were duplicated"
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
            msg = f"Since {axis}Names were only provided for the values in "
            msg += f"{keep}, {keep} can contain only index values referencing "
            msg += f"the {axis}'s location in the data. If attempting to use "
            msg += f"{keep} to reorder all {axis}s, instead create the object "
            msg += f"first then sort the {axis}s."
            raise InvalidArgumentValue(msg)
        if isinstance(idVal, str):
            msg = f"{keep} can contain only index values because no "
            msg += f"{axis}Names were provided"
            raise InvalidArgumentValue(msg)
        if idVal < 0:
            msg = "Negative index values are not permitted, found "
            msg += f"{idVal} in {keep}"
            raise InvalidArgumentValue(msg)


def _raiseKeepIndexNameConflict(axis, index, name):
    """
    Helper for raising exception when two values in keepPoints/Features
    represent the same point/feature.
    """
    keep = 'keepPoints' if axis == 'point' else 'keepFeatures'
    msg = f"{keep} cannot contain duplicate values. The index {index} and the "
    msg += f"name '{name}' represent the same {axis} and are both in {keep} "
    raise InvalidArgumentValue(msg)


def _raiseKeepLengthConflict(axis):
    """
    Helper to prevent defining keepPoints/Features for every point or
    feature because it cannot be determined whether the list is defining
    the values in the order of the data or order of keepPoints/Features.
    """
    keep = 'keepPoints' if axis == 'point' else 'keepFeatures'
    msg = f"The length of {keep} cannot be the same as the number of {axis}s. "
    msg += f"If attempting to use {keep} to keep and/or reorder all {axis}s, "
    msg += f"instead create the object using {keep}='all', then sort the "
    msg += f"{axis}s."
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
            msg = f"The value '{ftID}' in keepFeatures is not a valid "
            msg += "featureName"
            raise InvalidArgumentValue(msg)
        # index values
        elif 0 <= ftID < len(retFNames):
            name = retFNames[ftID]
            if name in keepNames:
                _raiseKeepIndexNameConflict('feature', ftID, name)
            keepIndices.append(ftID)
            keepNames.append(name)
        elif ftID >= 0:
            msg = f"The index {ftID} is greater than the number of features "
            msg += f"in the data, {len(retFNames)}"
            raise InvalidArgumentValue(msg)
        else:
            msg = f"Negative index values are not permitted, found {ftID} in "
            msg += "keepFeatures"
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
                msg += f"The data in key '{key}' had shape {ptShape} but the "
                msg += f'first point had shape {expShape}'
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
            msg += 'Dataset will be loaded as a single point, or '
            msg += f'{innerShape} indicating the data in the Dataset will be '
            msg += f'loaded directly, but pointNames contained {numNames} '
            msg += 'names'
            raise InvalidArgumentValue(msg)
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
    pointNames, featureNames, trackPoints = _checkCSVForNames(
        ioStream, pointNames, featureNames, dialect)

    pointNames = _namesDictToList(pointNames, 'point', 'pointNames')
    featureNames = _namesDictToList(featureNames, 'feature', 'featureNames')

    # Advance the file past any beginning of file comments, record
    # how many are skipped
    skippedLines = _advancePastComments(ioStream)
    # remake the file iterator to ignore empty lines
    filtered = filter(lambda row: row[0] != '\n', ioStream)
    # send that line iterator to the csv reader
    lineReader = csv.reader(filtered, dialect)

    firstRowLength = None
    if featureNames is True:
        retFNames = [v if v else None for v in next(lineReader)]
        if pointNames is True or trackPoints:
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
    nonNumericFeatures = set()
    allEmptyFeatures = set()
    possiblePtNames = set()
    # lineReader is now at the first line of data
    for i, row in enumerate(lineReader):
        if pointNames is True or trackPoints:
            ptName = row[0]
            row = row[1:]
        elif pointNames and len(pointNames) > len(keepPoints):
            ptName = pointNames[i]
        else:
            ptName = None
        if trackPoints:
            if ptName in possiblePtNames:
                # duplicate value, need to add possible names back in as data
                for idx, name in enumerate(extractedPointNames):
                    if retData[idx] is not None:
                        retData[idx].insert(0, name)
                row.insert(0, ptName)
                retFNames.insert(0, None)
                firstRowLength += 1
                trackPoints = False
                pointNames = False
            else:
                possiblePtNames.add(ptName)
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
                        nonNumericFeatures.add(idx)
                        if val == '':
                            allEmptyFeatures.add(idx)
            else: # continue to track column types in subsequent rows
                _csvColTypeTracking(row, convertCols, nonNumericFeatures,
                                    allEmptyFeatures)

            if keepPoints == 'all':
                retData.append(row)
                if pointNames is True or trackPoints:
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
                if pointNames is True or trackPoints:
                    extractedPointNames[location] = ptName

        totalPoints = i + 1

    if allEmptyFeatures and emptyIsMissing:
        for idx in allEmptyFeatures:
            convertCols[idx] = float
            nonNumericFeatures.remove(idx)
    if trackPoints:
        pointNames = trackPoints
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

    # remove last feature if all values are '' (all rows ended with delimiter),
    # '' is missing value, and no feature name is provided
    lastFtRemovable = (not limitFeatures and emptyIsMissing
                       and firstRowLength - 1 in allEmptyFeatures)
    noLastFtName = (not retFNames or len(retFNames) == firstRowLength - 1
                    or (featureNames is True and retFNames[-1] is None))
    if lastFtRemovable and noLastFtName:
        for row in retData:
            row.pop()
        if featureNames is True:
            retFNames.pop()
        del convertCols[firstRowLength -1]
        firstRowLength -= 1

    if convertCols:
        containsMissing = set()
        # convertCols only contains columns that need numeric conversion, so
        # heterogenous data is indicated by using columns that do not require
        # conversion or convertCols containing more than one numeric type.
        if limitFeatures:
            expectedLength = len(keepFeatures)
        else:
            expectedLength = firstRowLength
        if ignoreNonNumericalFeatures:
            expectedLength -= len(nonNumericFeatures)

        for row in retData:
            _colTypeConversion(row, convertCols, containsMissing)
        # upcast features with missing data to float
        for idx in containsMissing:
            convertCols[idx] = float
        # if the data is all numeric use array o/w dataframe if possible
        convertTypes = set(convertCols.values())
        if (pd.nimbleAccessible()
                and (len(convertCols) < expectedLength
                     or (len(convertTypes) > 1 and bool in convertTypes))):
            constructor = pd.DataFrame
        else:
            constructor = numpy2DArray
    # no conversion means all strings so use a dataframe if possible
    else:
        if pd.nimbleAccessible():
            constructor = pd.DataFrame
        else:
            constructor = numpy2DArray

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

    return constructor(retData), retPNames, retFNames

def fileFetcher(source, overwrite):
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
    source, database = _urlSourceProcessor(source)
    try:
        urlManager = _DirectURLManager(source)
    except InvalidArgumentValue:
        if database == 'nimble':
            urlManager = _IndirectURLManagerNimble(source)
        else:
            raise

    return urlManager.fetch(overwrite)
