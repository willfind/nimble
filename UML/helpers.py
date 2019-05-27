"""
Helper functions for any functions defined in uml.py

They are separated here so that that (most) top level
user facing functions are contained in uml.py without
the distraction of helpers

"""

from __future__ import absolute_import
from __future__ import print_function
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
import six
from six.moves import range
from six.moves import zip

import UML
from UML.logger import handleLogging
from UML.exceptions import InvalidArgumentValue, InvalidArgumentType
from UML.exceptions import InvalidArgumentValueCombination, PackageException
from UML.exceptions import ImproperObjectAction
from UML.exceptions import FileFormatException
from UML.data import Base
from UML.data.dataHelpers import isAllowedSingleElement
from UML.data.sparse import removeDuplicatesNative
from UML.randomness import pythonRandom
from UML.randomness import numpyRandom

scipy = UML.importModule('scipy.io')
pd = UML.importModule('pandas')
requests = UML.importModule('requests')

try:
    intern = sys.intern
    class Py2Key:
        """
        Key for python3.
        """
        __slots__ = ("value", "typestr")

        def __init__(self, value):
            self.value = value
            self.typestr = intern(type(value).__name__)

        def __lt__(self, other):
            try:
                return self.value < other.value
            except TypeError:
                return self.typestr < other.typestr
except Exception:
    Py2Key = None # for python2

#in python3, itertools.ifilter is not there anymore. it is filter.
if not hasattr(itertools, 'ifilter'):
    itertools.ifilter = filter


def findBestInterface(package):
    """
    Attempt to determine the interface.

    Takes the string name of a possible interface provided to some other
    function by a UML user, and attempts to find the interface which
    best matches that name amoung those available. If it does not match
    any available interfaces, then an exception is thrown.
    """
    for interface in UML.interfaces.available:
        if package == interface.getCanonicalName():
            return interface
    for interface in UML.interfaces.available:
        if interface.isAlias(package):
            return interface
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
    return getattr(interface, toCallName)(learnerName)


def isAllowedRaw(data, allowLPT=False):
    """
    Verify raw data is one of the accepted types.
    """
    if allowLPT and 'PassThrough' in str(type(data)):
        return True
    if scipy and scipy.sparse.issparse(data):
        return True
    if isinstance(data, (tuple, list, dict, numpy.ndarray, numpy.matrix)):
        return True

    if pd:
        if isinstance(data, (pd.DataFrame, pd.Series, pd.SparseDataFrame)):
            return True

    return False


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
    Create UML data objects containing constant values.

    Use numpy.ones or numpy.zeros to create constant UML objects of the
    designated returnType.
    """
    retAllowed = copy.copy(UML.data.available)
    if returnType not in retAllowed:
        msg = "returnType must be a value in " + str(retAllowed)
        raise InvalidArgumentValue(msg)

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
        if not scipy:
            msg = "scipy is not available"
            raise PackageException(msg)
        if numpyMaker == numpy.ones:
            rawDense = numpyMaker((numPoints, numFeatures))
            rawSparse = scipy.sparse.coo_matrix(rawDense)
        else:  # case: numpyMaker == numpy.zeros
            assert numpyMaker == numpy.zeros
            rawSparse = scipy.sparse.coo_matrix((numPoints, numFeatures))
        return UML.createData(returnType, rawSparse, pointNames=pointNames,
                              featureNames=featureNames, name=name,
                              useLog=False)
    else:
        raw = numpyMaker((numPoints, numFeatures))
        return UML.createData(returnType, raw, pointNames=pointNames,
                              featureNames=featureNames, name=name,
                              useLog=False)


def transposeMatrix(matrixObj):
    """
    This function is similar to np.transpose.
    copy.deepcopy(np.transpose(matrixObj)) may generate a messed data,
    so I created this function.
    """
    return numpy.matrix(list(zip(*matrixObj.tolist())), dtype=matrixObj.dtype)


def extractNamesAndConvertData(returnType, rawData, pointNames, featureNames,
                               elementType):
    """
    1. If rawData is like {'a':[1,2], 'b':[3,4]}, then convert it to np.matrix
    and extract featureNames from keys
    2. If rawData is like [{'a':1, 'b':3}, {'a':2, 'b':4}]
    3. If pointNames is True, then extract point names from the 1st column in
    rawData if featureNames is True, then extract feature names from the 1st
    row in rawData
    4. Convert data to np matrix
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
    # 1. convert dict like {'a':[1,2], 'b':[3,4]} to np.matrix
    # featureNames must be those keys
    # pointNames must be False or automatic
    if isinstance(rawData, dict):
        if rawData:
            featureNames = list(rawData.keys())
            rawData = numpy.matrix(list(rawData.values()), dtype=elementType)
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
            rawData = numpy.matrix(numpy.empty([0, 0]), dtype=elementType)
            pointNames = None

    # 2. convert list of dict ie. [{'a':1, 'b':3}, {'a':2, 'b':4}] to np.matrix
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
        rawData = numpy.matrix(values, dtype=elementType)
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
        elif isinstance(rawData, (numpy.matrix, numpy.ndarray)):
            func = extractNamesFromNumpy
        elif scipy and scipy.sparse.issparse(rawData):
            # all input coo_matrices must have their duplicates removed; all
            # helpers past this point rely on there being single entires only.
            if isinstance(rawData, scipy.sparse.coo_matrix):
                rawData = removeDuplicatesNative(rawData)
            func = extractNamesFromScipySparse
        elif pd and isinstance(rawData, (pd.DataFrame, pd.SparseDataFrame)):
            func = extractNamesFromPdDataFrame
        elif pd and isinstance(rawData, pd.Series):
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

    # 4. if type(data) doesn't match returnType, then convert data to numpy
    # matrix or coo_matrix. If elementType is not None, then convert each
    # element in data to elementType.
    if (elementType is None
            and isinstance(rawData, list)
            and returnType == 'List'
            and len(rawData) != 0
            and (
                # this list can only be [[]], [1,2,3], ['ab', 'c'], [[1,2,'a'],
                # [4,5,'b']] otherwise, we need to covert the list to matrix,
                # such [np.array([1,2]), np.array(3,4)]
                isAllowedSingleElement(rawData[0])
                or isinstance(rawData[0], list)
                or hasattr(rawData[0], 'setLimit'))):
        # attempt to convert the list to floats to remain consistent with other
        # UML types if unsuccessful we will keep the list as is
        try:
            # 1D list
            rawData = list(map(numpy.float, rawData))
        except (TypeError, ValueError):
            try:
                #2D list
                convertedData = []
                for point in rawData:
                    convertedData.append(list(map(numpy.float, point)))
                rawData = convertedData
            except (ValueError, TypeError):
                pass
    elif (elementType is None and
          pd and isinstance(rawData, pd.DataFrame) and
          not isinstance(rawData, pd.SparseDataFrame) and
          returnType == 'DataFrame'):
        pass
    elif (elementType is None and
          scipy and scipy.sparse.isspmatrix(rawData) and
          returnType == 'Sparse'):
        pass
    elif isinstance(rawData, (numpy.ndarray, numpy.matrix)):
        # if the input data is a np matrix, then convert it anyway to make sure
        # try dtype=float 1st.
        rawData = elementTypeConvert(rawData, elementType)
    elif pd and isinstance(rawData, pd.SparseDataFrame):
        #from sparse to sparse, instead of via np matrix
        rawData = elementTypeConvert(rawData, elementType)
        rawData = scipy.sparse.coo_matrix(rawData)

    elif isinstance(rawData, (list, tuple)):
        # when rawData = [], or feature empty [[]], we need to use pointNames
        # and featureNamesto determine its shape
        lenFts = len(featureNames) if featureNames else 0
        if len(rawData) == 0:
            lenPts = len(pointNames) if pointNames else 0
            rawData = numpy.matrix(numpy.empty([lenPts, lenFts]),
                                   dtype=elementType)
        elif isinstance(rawData[0], list) and len(rawData[0]) == 0:
            rawData = numpy.matrix(numpy.empty([len(rawData), lenFts]),
                                   dtype=elementType)
        # if there are actually elements, we attempt to convert them
        else:
            rawData = elementTypeConvert(rawData, elementType)

    elif pd and isinstance(rawData, (pd.DataFrame, pd.Series)):
        rawData = elementTypeConvert(rawData, elementType)

    elif scipy and scipy.sparse.isspmatrix(rawData):
        rawData = elementTypeConvert(rawData.todense(), elementType)

    if (returnType == 'Sparse'
            and isinstance(rawData, numpy.matrix)
            and rawData.shape[0]*rawData.shape[1] > 0):
    #replace None to np.NaN, o.w. coo_matrix will convert None to 0
        numpy.place(rawData, numpy.vectorize(lambda x: x is None)(rawData),
                    numpy.NaN)

    return rawData, pointNames, featureNames


def elementTypeConvert(rawData, elementType):
    """
    Convert rawData to numpy matrix with dtype = elementType, or try
    dtype=float then try dtype=object.
    """
    if pd and isinstance(rawData, pd.Series) and len(rawData) == 0:
        # make sure pd.Series() converted to matrix([], shape=(0, 0))
        rawData = numpy.empty([0, 0])
    elif pd and isinstance(rawData, pd.DataFrame):
        #for pd.DataFrame, convert it to np.ndarray first then to matrix
        #o.w. copy.deepcopy may generate messed data
        rawData = rawData.values

    if elementType:
        return numpy.matrix(rawData, dtype=elementType)
    else:
        try:
            data = numpy.matrix(rawData, dtype=numpy.float)
        except ValueError:
            data = numpy.matrix(rawData, dtype=object)
        return data

def replaceNumpyValues(data, toReplace, replaceWith):
    """
    Replace values in a numpy matrix or array.

    Parameters
    ----------
    data : numpy.matrix, numpy.array
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

def replaceMissingData(rawData, treatAsMissing, replaceMissingWith,
                       elementType=None):
    """
    Convert any values in rawData found in treatAsMissing with
    replaceMissingWith value.
    """
    if isinstance(rawData, (list, tuple)):
        handleMissing = numpy.array(rawData, dtype=numpy.object_)
        handleMissing = replaceNumpyValues(handleMissing, treatAsMissing,
                                           replaceMissingWith)
        rawData = handleMissing.tolist()

    elif isinstance(rawData, (numpy.matrix, numpy.ndarray)):
        handleMissing = replaceNumpyValues(rawData, treatAsMissing,
                                           replaceMissingWith)
        rawData = elementTypeConvert(handleMissing, elementType)

    elif scipy.sparse.issparse(rawData):
        handleMissing = replaceNumpyValues(rawData.data, treatAsMissing,
                                           replaceMissingWith)
        handleMissing = elementTypeConvert(handleMissing, elementType)
        # elementTypeConvert returns matrix, need a 1D array
        handleMissing = handleMissing.A1
        rawData.data = handleMissing

    elif isinstance(rawData, (pd.DataFrame, pd.Series)):
        if len(rawData.values) > 0:
            # .where keeps the values that return True, use ~ to replace those
            # values instead
            rawData = rawData.where(~rawData.isin(treatAsMissing),
                                    replaceMissingWith)

    return rawData

def initDataObject(
        returnType, rawData, pointNames, featureNames, name, path, keepPoints,
        keepFeatures, elementType=None, reuseData=False,
        treatAsMissing=(float('nan'), numpy.nan, None, '', 'None', 'nan',
                        'NULL', 'NA'),
        replaceMissingWith=numpy.nan):
    """
    1. Set up autoType
    2. Extract names and convert data if necessary
    3. Handle missing values
    """
    if (scipy and scipy.sparse.issparse(rawData)) or \
            (pd and isinstance(rawData, pd.SparseDataFrame)):
        autoType = 'Sparse'
    else:
        autoType = 'Matrix'

    if returnType is None:
        returnType = autoType

    #may need to extract names and may need to convert data to matrix
    rawData, pointNames, featureNames = extractNamesAndConvertData(
        returnType, rawData, pointNames, featureNames, elementType)

    # handle missing values
    if treatAsMissing is not None:
        rawData = replaceMissingData(rawData, treatAsMissing,
                                     replaceMissingWith, elementType)

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

    initMethod = getattr(UML.data, returnType)
    try:
        ret = initMethod(rawData, pointNames=pointNames,
                         featureNames=featureNames, name=name,
                         paths=pathsToPass, elementType=elementType,
                         reuseData=reuseData)
    except Exception:
        einfo = sys.exc_info()
        #something went wrong. instead, try to auto load and then convert
        try:
            autoMethod = getattr(UML.data, autoType)
            ret = autoMethod(rawData, pointNames=pointNames,
                             featureNames=featureNames, name=name,
                             paths=pathsToPass, elementType=elementType,
                             reuseData=reuseData)
            ret = ret.copy(to=returnType)
        # If it didn't work, report the error on the thing the user ACTUALLY
        # wanted
        except Exception:
            six.reraise(einfo[0], einfo[1], einfo[2])


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
    if keepFeatures != 'all':
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

    return ret


def extractNamesFromDataObject(data, pointNamesID, featureNamesID):
    """
    Extracts and sets (if needed) the point and feature names from the
    given UML Base object, returning the modified object. pointNamesID
    may be either None, or an integer ID corresponding to a feature in
    the data object. featureNamesID may b either None, or an integer ID
    corresponding to a point in the data object.
    """
    ret = data
    praw = None
    if pointNamesID is not None:
        # extract the feature of point names
        pnames = ret.features.extract(pointNamesID)
        if featureNamesID is not None:
            # discard the point of feature names that pulled along since we
            # extracted these first
            pnames.points.extract(featureNamesID)
        praw = pnames.copy(to='numpyarray', outputAs1D=True)
        praw = numpy.vectorize(str)(praw)

    fraw = None
    if featureNamesID is not None:
        # extract the point of feature names
        fnames = ret.points.extract(featureNamesID)
        # extracted point names first, so if they existed, they aren't in
        # ret anymore. So we DON'T need to extract them from this object
        fraw = fnames.copy(to='numpyarray', outputAs1D=True)
        fraw = numpy.vectorize(str)(fraw)

    # have to wait for everything to be extracted before we add the names,
    # because otherwise the lenths won't be correct
    if praw is not None:
        ret.points.setNames(list(praw))
    if fraw is not None:
        ret.features.setNames(list(fraw))

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
    # try to find an extension for possible optimizations
    if isinstance(data, six.string_types):
        path = data
    else:
        # try getting name attribute from file
        try:
            path = data.name
        except AttributeError:
            path = None

    # detect extension from the path
    if path is not None:
        split = path.rsplit('.', 1)
        extension = None
        if len(split) > 1:
            extension = split[1].lower()
    else:
        extension = None

    def isMtxFileChecker(ioStream):
        # find first non empty line to check header
        startPosition = ioStream.tell()
        currLine = ioStream.readline()
        while currLine == '':
            currLine = ioStream.readline()
        header = currLine
        ioStream.seek(startPosition)

        # check beginning of header for sentinel on string or binary stream
        return header[:14] in ["%%MatrixMarket", b"%%MatrixMarket"]

    toPass = data
    # Case: string value means we need to open the file, either directly or
    # through an http request
    if isinstance(toPass, six.string_types):
        if toPass[:4] == 'http':
            if requests is None:
                msg = "To load data from a webpage, the requests module must "
                msg += "be installed"
                raise PackageException(msg)
            response = requests.get(data, stream=True)
            if not response.ok:
                msg = "The data could not be accessed from the webpage. "
                msg += "HTTP Status: {0}, ".format(response.status_code)
                msg += "Reason: {0}".format(response.reason)
                raise InvalidArgumentValue(msg)

            # check python version
            py3 = sys.version_info[0] == 3
            if py3:
                toPass = StringIO(response.text, newline=None)
                isMtxFile = isMtxFileChecker(toPass)
                # scipy.io.mmreader needs bytes object
                if isMtxFile:
                    toPass = BytesIO(bytes(response.content,
                                           response.apparent_encoding))
            # in python 2, we can just always use BytesIO
            else:
                # handle universal newline
                content = "\n".join(response.content.splitlines())
                toPass = BytesIO(content)
                isMtxFile = isMtxFileChecker(toPass)
        else:
            toPass = open(data, 'rU')
            isMtxFile = isMtxFileChecker(toPass)
    # Case: we are given an open file already
    else:
        isMtxFile = isMtxFileChecker(toPass)

    loadType = returnType
    if loadType is None:
        loadType = 'Auto' if isMtxFile else 'Matrix'

    # use detected format to override file extension
    if isMtxFile:
        extension = 'mtx'

    # Choose what code to use to load the file. Take into consideration the end
    # result we are trying to load into.
    if loadType is not None and extension is not None:
        directPath = "_load" + extension + "For" + loadType
    else:
        directPath = None

    if directPath in globals():
        loader = globals()[directPath]
        loaded = loader(
            toPass, pointNames, featureNames, ignoreNonNumericalFeatures,
            keepPoints, keepFeatures, inputSeparator=inputSeparator)
    # If we don't know, default to trying to load a value separated file
    else:
        loaded = _loadcsvUsingPython(
            toPass, pointNames, featureNames, ignoreNonNumericalFeatures,
            keepPoints, keepFeatures, inputSeparator=inputSeparator)

    (retData, retPNames, retFNames, selectSuccess) = loaded

    # auto set name if unspecified, and is possible
    if isinstance(data, six.string_types):
        path = data
    elif hasattr(data, 'name'):
        path = data.name
    else:
        path = None

    if path is not None and name is None:
        tokens = path.rsplit(os.path.sep)
        name = tokens[len(tokens) - 1]

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
        replaceMissingWith=replaceMissingWith)


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
    if not scipy:
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
        data = scipy.io.mmread(openFile)#python 2
    except Exception:
        if hasattr(openFile, 'name'):
            tempName = openFile.name
        else:
            tempName = openFile.inner.name
        data = scipy.io.mmread(tempName)#for python3, it may need this.

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


def extractNamesFromNumpy(data, pnamesID, fnamesID):
    """
    Extract name values from a numpy matrix or array.
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
#    try:
#        ret = extractNamesFromScipyConversion(rawData, pointNames,
#                                              featureNames)
#    except (NotImplementedError, TypeError):
    ret = extractNamesFromCooDirect(rawData, pointNames, featureNames)

    return ret

def extractNamesFromScipyConversion(rawData, pointNames, featureNames):
    """
    Extract names from a scipy sparse csr or csc matrix.
    """
    if not isinstance(rawData, scipy.sparse.csr_matrix):
        rawData = scipy.sparse.csr_matrix(rawData)

    if rawData.shape[0] > 0:
        firstRow = rawData[0].toarray().flatten().tolist()
    else:
        firstRow = None
    if rawData.shape[0] > 1:
        secondRow = rawData[1].toarray().flatten().tolist()
    else:
        secondRow = None
    pointNames, featureNames = autoDetectNamesFromRaw(pointNames, featureNames,
                                                      firstRow, secondRow)
    pointNames = 0 if pointNames is True else None
    featureNames = 0 if featureNames is True else None

    retFNames = None
    if featureNames == 0:
        retFNames = rawData[0].toarray().flatten().tolist()
        retFNames = list(map(str, retFNames))
        rawData = rawData[1:]

    retPNames = None
    if pointNames == 0:
        rawData = scipy.sparse.csc_matrix(rawData)
        retPNames = rawData[:, 0].toarray().flatten().tolist()
        retPNames = list(map(str, retPNames))
        rawData = rawData[:, 1:]
        retFNames = retFNames[1:]

    rawData = scipy.sparse.coo_matrix(rawData)
    return rawData, retPNames, retFNames

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
    if not scipy:
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


def _defaultParser(line):
    """
    When given a comma separated value line, it will attempt to convert
    the values first to int, then to float, and if all else fails will
    keep values as strings. Returns list of values.
    """
    ret = []
    lineList = line.split(',')
    for entry in lineList:
        ret.append(_intFloatOrString(entry))
    return ret


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
        allText = all(map(lambda x: isinstance(x, six.string_types),
                          firstValues[1:]))
        allDiff = all(map(teq, zip(firstValues[1:], secondValues[1:])))
    else:
        allText = all(map(lambda x: isinstance(x, six.string_types),
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


def _selectionNameValidation(keep, hasNames, kind):
    """
    Helper for _loadcsvUsingPython to check that when points or features
    are specified, if we have no source of names that only integer
    values are in the specification list. This is generic over whether
    points or features are selected.
    """
    kind = kind.lower()
    paramName = 'keep' + kind.capitalize()
    if keep != 'all':
        if hasNames is False:
            # in this case we have no names for reference,
            # so no strings should be in the list
            if not all(isinstance(val, int) for val in keep):
                msg = "No " + kind + " names were provided by the user, and "
                msg += "they are not being extracted from the data, "
                msg += 'therefore only integer valued indices are '
                msg += 'allowed in the ' + paramName + 's parameter'
                raise InvalidArgumentValue(msg)


def _csv_getFNamesAndAnalyzeRows(pointNames, featureNames, openFile,
                                 lineReader, skippedLines, dialect):
    """
    If needed, take the first row from the lineReader to define the
    feature names. Regardless of whether feature names are desired,
    we will analyze the row we read to determine the number of columns,
    the number of features (columns without point names), the line
    number in the file of the row we use to define number of features /
    columns, and whether or not that row was interpreted as feature
    names or data.
    """
    if featureNames is True:
        fnamesRow = next(lineReader)
        # Number values in a row excluding point names
        numFeatures = len(fnamesRow)
        # Number of value in a row
        numColumns = len(fnamesRow)

        if pointNames is True:
            fnamesRow = fnamesRow[1:]
            numFeatures -= 1
        retFNames = fnamesRow
        columnsDefIndex = (lineReader.line_num - 1) + skippedLines
        columnsDefSrc = "feature names"
    else:
        startPosition = openFile.tell()
        filtered = itertools.ifilter(_filterCSVRow, openFile)
        trialReader = csv.reader(filtered, dialect)
        trialRow = next(trialReader)
        # Number values in a row excluding point names
        numFeatures = len(trialRow)
        # Number of value in a row
        numColumns = len(trialRow)
        openFile.seek(startPosition)
        columnsDefIndex = (trialReader.line_num - 1) + skippedLines
        columnsDefSrc = "row"
        retFNames = copy.copy(featureNames)

    return retFNames, numFeatures, numColumns, columnsDefIndex, columnsDefSrc


def _validateRowLength(row, numColumns, lineIndex, columnsDefSrc,
                       columnsDefIndex, delimiter):
    """
    Given a row read from a csv line reader, and the expected length of
    that row, raise an appropriately worded exception if there is a
    discrepency.
    """
    if len(row) != numColumns:
        msg = "The row on line " + str(lineIndex) + " has a length of "
        msg += str(len(row)) + ". We expected a length of "
        msg += str(numColumns) + ". The expected row length was defined "
        msg += "by looking at the " + columnsDefSrc + " on line "
        msg += str(columnsDefIndex) + " and using '" + delimiter
        msg += "' as the separator."
        raise FileFormatException(msg)


def _setupAndValidationForFeatureSelection(
        keepFeatures, retFNames, removeRecord, numFeatures, featsToRemoveSet):
    """
    Once feature names have been determined, we can validate and clean
    the keep features parameter; transforming any name to an index, and
    checking that all indices are valid. At the same time, we also setup
    the data structures needed to record which features are being
    excluded from every row we will eventually read in.
    """
    if keepFeatures != 'all':
        cleaned = []
        for val in keepFeatures:
            selIndex = val
            # this case can only be true if names were extracted or provided
            if isinstance(val, six.string_types):
                try:
                    selIndex = retFNames.index(val)
                except ValueError:
                    msg = 'keepFeatures included a name (' + val + ') '
                    msg += 'which was not found in the featureNames'
                    raise InvalidArgumentValue(msg)
            cleaned.append(selIndex)
            assert selIndex is not None

        # check for duplicates, and that values are in range
        found = {}
        for i in range(len(cleaned)):
            if cleaned[i] in found:
                msg = "Duplicate values were present in the keepFeatures "
                msg += "parameter, at indices ("
                msg += str(found[cleaned[i]])
                msg += ") and ("
                msg += str(i)
                msg += "). The values were ("
                msg += str(keepFeatures[found[cleaned[i]]])
                msg += ") and ("
                msg += str(keepFeatures[i])
                msg += ") respectably."
                raise InvalidArgumentValue(msg)
            else:
                found[cleaned[i]] = i

            if cleaned[i] < 0 or cleaned[i] >= numFeatures:
                msg = "Invalid value in keepFeatures parameter at index ("
                msg += str(i)
                msg += "). The value ("
                msg += str(cleaned[i])
                msg += ") is not in the range of 0 to "
                msg += str(numFeatures - 1)  # we want inclusive end points
                raise InvalidArgumentValue(msg)

        # initialize, but only if we know we'll be adding something
        keepFeatures = cleaned
        if len(cleaned) > 0:
            removeRecord[0] = []
        for i in range(numFeatures):
            if i not in cleaned:
                featsToRemoveSet.add(i)
                removeRecord[0].append(i)

    return keepFeatures


def _raiseSelectionDuplicateException(kind, i1, i2, values):
    msg = "Duplicate or equivalent values were present in the "
    msg += kind
    msg += " parameter, at indices ("
    msg += str(i1)
    msg += ") and ("
    msg += str(i2)
    msg += "). The values were ("
    msg += str(values[i1])
    msg += ") and ("
    msg += str(values[i2])
    msg += ") respectably."
    raise InvalidArgumentValue(msg)


def _validationForPointSelection(keepPoints, pointNames):
    if keepPoints == 'all':
        return 'all'

    found = {}
    cleaned = []
    for i in range(len(keepPoints)):
        if keepPoints[i] in found:
            _raiseSelectionDuplicateException(
                "keepPoints", found[keepPoints[i]], i, keepPoints)
        else:
            found[keepPoints[i]] = i

        if (not isinstance(keepPoints[i], six.string_types)
                and keepPoints[i] < 0):
            msg = "Invalid value in keepPoints parameter at index ("
            msg += str(i)
            msg += "). The value ("
            msg += str(keepPoints[i])
            msg += ") was less than 0, yet we only allow valid non-negative "
            msg += "integer indices or point names as values."
            raise InvalidArgumentValue(msg)

        if isinstance(pointNames, list):
            if isinstance(keepPoints[i], six.string_types):
                try:
                    cleaned.append(pointNames.index(keepPoints[i]))
                except ValueError:
                    msg = 'keepPoints included a name (' + keepPoints[i] + ') '
                    msg += 'which was not found in the provided pointNames'
                    raise InvalidArgumentValue(msg)
            else:
                cleaned.append(keepPoints[i])

    if cleaned != []:
        found = {}
        for i in range(len(cleaned)):
            if cleaned[i] in found:
                msg = "Duplicate values were present in the keepPoints "
                msg += "parameter, at indices ("
                msg += str(found[cleaned[i]])
                msg += ") and ("
                msg += str(i)
                msg += "). The values were ("
                msg += str(keepPoints[found[cleaned[i]]])
                msg += ") and ("
                msg += str(keepPoints[i])
                msg += ") respectably."
                raise InvalidArgumentValue(msg)
            else:
                found[cleaned[i]] = i

            if cleaned[i] < 0 or cleaned[i] >= len(pointNames):
                msg = "Invalid value in keepPoints parameter at index ("
                msg += str(i)
                msg += "). The value ("
                msg += str(cleaned[i])
                msg += ") is not in the range of 0 to "
                msg += str(len(pointNames) - 1)  # we want inclusive end points
                raise InvalidArgumentValue(msg)

        # only do this if cleaned is non-empty / we have provided pointnames
        return cleaned

    # only get here if cleaned was empty / point names will be extracted
    return keepPoints


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
    filtered = itertools.ifilter(_filterCSVRow, openFile)
    # send that line iterator to the csv reader
    lineReader = csv.reader(filtered, dialect)

    # after _checkCSV_for_Names then both pointNames and featureNames
    # should either be True, False, a list or a dict. In the case of
    # True, we setup an empty list to put extracted names into
    if pointNames is True:
        retPNames = []
    else:
        retPNames = pointNames

    # Extract names from first row if needed, record number of
    # columns in a row, and record a few book-keeping details
    # to be output in case of an exception
    namesAndMore = _csv_getFNamesAndAnalyzeRows(
        pointNames, featureNames, openFile, lineReader, skippedLines, dialect)
    retFNames = namesAndMore[0]
    numFeatures = namesAndMore[1]
    numColumns = namesAndMore[2]
    columnsDefIndex = namesAndMore[3]
    columnsDefSrc = namesAndMore[4]

    # Validation: check that if we have no source of names, and specific
    # values are specified for selection, that they are specified only
    # with integer indices, NOT names.
    hasPointNames = not pointNames is False
    hasFeatureNames = not featureNames is False
    _selectionNameValidation(keepPoints, hasPointNames, 'point')
    _selectionNameValidation(keepFeatures, hasFeatureNames, 'feature')

    _validationForPointSelection(keepPoints, pointNames)
    notYetFoundPoints = None
    if keepPoints != 'all':
        notYetFoundPoints = {}
        for i in range(len(keepPoints)):
            notYetFoundPoints[keepPoints[i]] = i

    # This is a record of the discovery of features to remove. It maps
    # the index of a point in the data to the features discovered to be
    # undesirable at that point. Thus, after all of the file has been read
    # into memory, this record can be used to remove those features from
    # the points prior to their discovery.
    removeRecord = {}

    # A set containing the indices of all features that are being removed
    # from the data.
    featsToRemoveSet = set([])

    # now that we have featureNames, we can do full validation of both
    # the feature names and the keepFeatures parameter. We also setup
    # the removal record wrt columns that do not represent selected
    # features.
    keepFeatures = _setupAndValidationForFeatureSelection(
        keepFeatures, retFNames, removeRecord, numFeatures, featsToRemoveSet)

    # Variables to be used in the main loop for reading the csv data.
    # data is where the data from the file will be placed, in the same
    # order as in the file.
    data = []
    # retData is where the data from the file will be placed, in the
    # order specified by keepPoints if that order doesn't match the
    # order in the file. shares references to the same lists as the
    # variable data, so lists modified through data share changes with
    # retData. Will be None and ignored if no reordering needs to take
    # place.
    retData = None
    if keepPoints != 'all' and keepPoints != sorted(keepPoints, key=Py2Key):
        retData = [None] * len(keepPoints)
        keepPointsValToIndex = {}
        for i in range(len(keepPoints)):
            keepPointsValToIndex[keepPoints[i]] = i

    # incremented at beginning of loop, so starts negative.
    # addedIndex is the index of the list that matches this row in the returned
    # list of lists
    addedIndex = -1
    # pointIndex is the index of potential points read off from lineReader.
    # It is always incremented, even if that point is no selected
    pointIndex = -1
    # lineIndex is the index of the newline sepearated row that
    # the lineReader will return next. Inclueds the lines skipped
    # at the beginning of the file
    lineIndex = skippedLines

    # Read through the csv file, row by row.
    # The csv reader gives a list of string values for each row
    for row in lineReader:
        addedIndex += 1
        pointIndex += 1
        lineIndex = lineReader.line_num + skippedLines

        # Validation: require equal length of all rows
        _validateRowLength(
            row, numColumns, lineIndex, columnsDefSrc, columnsDefIndex,
            delimiter=dialect.delimiter)

        # grab the pointName if needed
        currPName = None
        if pointNames is True:
            currPName = row[0]
            row = row[1:]
            # validate keepPoints, given new information
            if (keepPoints != 'all'
                    and pointIndex in keepPoints
                    and currPName in keepPoints):
                _raiseSelectionDuplicateException(
                    "keepPoints", keepPoints.index(pointIndex),
                    keepPoints.index(currPName), keepPoints)
        elif isinstance(pointNames, list):
            currPName = pointNames[pointIndex]

        # Run through selection criteria, adjusting variables as needed,
        # and raising exceptions if needed.
        isSelected = False
        if keepPoints == 'all':
            isSelected = True
        elif currPName in keepPoints:
            del notYetFoundPoints[currPName]
            if retData is not None:
                currIndex = keepPointsValToIndex[currPName]
                keepPointsValToIndex[pointIndex] = currIndex
                del keepPointsValToIndex[currPName]
            isSelected = True
        elif pointIndex in keepPoints:
            del notYetFoundPoints[pointIndex]
            isSelected = True

        # we only do this if this row is destined for the data, not the
        # feature names
        if isSelected:
            # add point name if needed
            if pointNames is True:
                retPNames.append(currPName)
            # process the remaining data
            toAdd = convertAndFilterRow(row, addedIndex, removeRecord,
                                        featsToRemoveSet,
                                        ignoreNonNumericalFeatures)
            data.append(toAdd)
            if retData is not None:
                retData[keepPointsValToIndex[pointIndex]] = toAdd
        else:
            # not selected, so move on to the next row; and index
            # as if it was never present
            addedIndex -= 1

        # In this case we have grabbed all of the desired points, so
        # we can stop reading the file
        if notYetFoundPoints is not None and len(notYetFoundPoints) == 0:
            break

    # check to see if all of the wanted points in keepPoints were
    # found in the data
    if notYetFoundPoints is not None and len(notYetFoundPoints) > 0:
        msg = "The following entries in keepPoints were not found "
        msg += "in the data:"
        for key in notYetFoundPoints:
            msg += " (" + str(key) + ")"
        raise InvalidArgumentValue(msg)

    # the List form of featsToRemoveSet
    featsToRemoveList = list(featsToRemoveSet)

    # Since the convertAndFilterRow helper only removes unwanted features
    # after they've been discovered, we need to remove the from the earlier
    # part of the data after it has all been read into memory. Also, we
    # may need to adjust the order of features and points due to the
    # selection paramters, and this is a convenient and efficient place to
    # do so. If we don't need to do either of those things, then we
    # don't even enter the helper
    removalNeeded = (list(removeRecord.keys()) != [0]
                     and list(removeRecord.keys()) != [])
    reorderNeeded = (keepFeatures != 'all'
                     and keepFeatures != sorted(keepFeatures, key=Py2Key))
    if removalNeeded or reorderNeeded:
        _removalCleanupAndSelectionOrdering(
            data, removeRecord, featsToRemoveList, keepFeatures)

    # adjust WRT removed columns
    if isinstance(retFNames, list):
        copyIndex = 0
        # ASSUMPTION: featsToRemoveList is a sorted list
        removeListIndex = 0
        for i in range(len(retFNames)):
            # if it is a feature that has been removed from the data,
            # we skip over and don't copy it.
            if (removeListIndex < len(featsToRemoveList)
                    and i == featsToRemoveList[removeListIndex]):
                removeListIndex += 1
            else:
                retFNames[copyIndex] = retFNames[i]
                copyIndex += 1
        retFNames = retFNames[:copyIndex]

        needsRemoval = False  # This was done directly above
        retFNames = _adjustNamesGivenKeepList(
            retFNames, keepFeatures, needsRemoval)

    if isinstance(retPNames, list):
        # we only need to do removal if names were provided. If
        # they were extracted, they were only added if that row was
        # kept
        needsRemoval = isinstance(pointNames, list)
        retPNames = _adjustNamesGivenKeepList(
            retPNames, keepPoints, needsRemoval)

    if retData is None:
        retData = data

    return (retData, retPNames, retFNames, True)


def _adjustNamesGivenKeepList(retNames, keepList, needsRemoval):
    # In this case neither sorting or removal is necessary
    if keepList == 'all':
        return retNames

    # if we're already sorted and we don't need to do removal, we
    # can return.
    if sorted(keepList, key=Py2Key) == keepList and not needsRemoval:
        return retNames

    # if needed, resolve names to indices for easy indexing during
    # the sort
    for i, val in enumerate(keepList):
        if isinstance(val, six.string_types):
            keepList[i] = retNames.index(val)

    newRetFNames = []
    for val in keepList:
        newRetFNames.append(retNames[val])
    retNames = newRetFNames

    return retNames


def _removalCleanupAndSelectionOrdering(
        data, record, fullRemoveList, keepFeatures):
    """
    Adjust the given data so that the features to remove as contained in
    the dict record are appropriately removed from each row. Since
    removal is done only after first sighting of an unwanted value, then
    the rows prior to the first sighting still have that feature.
    Therefore, this function iterates the rows in the data, removing
    features until the point where they were discovered.

    Also: if keepFeatures defines an ordering other than the one present
    in the file, then we will adjust the order of the data in this
    helper. The actual selection has already occured during the csv
    reading loop.

    Because data shares references with retData in _loadcsvUsingPython,
    this does not change the contents of the parameter data, only the
    lists referenced by it.
    """
    # feature order adjustment will take place at the same time as unwanted
    # column removal. This just defines a triggering variable.
    adjustFeatureOrder = False
    if (keepFeatures != 'all'
            and keepFeatures != sorted(keepFeatures, key=Py2Key)):
        adjustFeatureOrder = True
        # maps the index of the column to the position that it should
        # be copied into.
        reverseKeepFeatures = {}
        # since we are doing this lookup after we have selected
        # some features, we have to reindex according to those
        # that are currently present. Since they are stored in the
        # file in lexigraphical order, we use the sorted selection
        # list to define the reindexing
        sortedKeepFeatures = sorted(keepFeatures, key=Py2Key)
        for i in range(len(sortedKeepFeatures)):
            reIndexed = sortedKeepFeatures[i]
            reverseKeepFeatures[i] = keepFeatures.index(reIndexed)

    # need to sort keep points. the index into the row maps into the
    # sorted list, which gives the key to reverseKeepFeatures

    # remaining features to be removed, indexed relative to all possible
    # features in the csv file
    absRemoveList = fullRemoveList
    absRemoveList.sort()
    # remaining features to be removed, reindexed given the knowledge of
    # which points have already been removed
    relRemoveList = copy.copy(absRemoveList)

    copySpace = [None] * len(data[len(data) - 1])

    for rowIndex in range(len(data)):
        # check if some feature was added at this row, and if so, delete
        # those indices from the removalList, adjusting feature IDs to be
        # relative the new length as you go
        if rowIndex in record:
            # ASSUMPTION: the lists of added indices in the record are
            # sorted
            addedIndex = 0  # index into the list held in record[rowIndex]
            shift = 0  # the amount we have to shift each index downward
            copyIndex = 0
            for i in range(len(absRemoveList)):
                if (addedIndex < len(record[rowIndex])
                        and absRemoveList[i] == record[rowIndex][addedIndex]):
                    shift += 1
                    addedIndex += 1
                else:
                    absRemoveList[copyIndex] = absRemoveList[i]
                    relRemoveList[copyIndex] = relRemoveList[i] - shift
                    copyIndex += 1
            absRemoveList = absRemoveList[:copyIndex]
            relRemoveList = relRemoveList[:copyIndex]

        # The following loop will be copying inplace. Note though, that
        # since numCopied will then be used as the index to copy into, and
        # it will always be less than or equal to i, so we never corrupt
        # the values we iterate over.
        remIndex = 0
        # If no feature reordering will take place, this serves as the index
        # to copy into. If feature reordering will take place, this is used
        # as the index of the feature, to do a lookup for the copy index.
        numCopied = 0

        for i in range(len(data[rowIndex])):
            if remIndex < len(relRemoveList) and i == relRemoveList[remIndex]:
                remIndex += 1
            else:
                # the index into data[rowIndex] where you will be copying
                # this particular value
                featureIndex = numCopied
                if adjustFeatureOrder:
                    featureIndex = reverseKeepFeatures[numCopied]

                copySpace[featureIndex] = data[rowIndex][i]
                numCopied += 1

        # copy the finalized point back into the list referenced by data
        for i in range(len(copySpace)):
            data[rowIndex][i] = copySpace[i]

        # TODO: run time trials to compare pop vs copying into new list
        needToPop = len(data[rowIndex]) - numCopied
        for i in range(needToPop):
            data[rowIndex].pop()


def convertAndFilterRow(row, pointIndex, record, toRemoveSet,
                        ignoreNonNumericalFeatures):
    """
    Process a row as read by a python csv reader such that the values
    are converted to numeric types if possible, and the unwanted
    features are filtered (with the appropriate book keeping operations
    performed).

    Parameters
    ----------
    row : list
        A python list of string values
    pointIndex : int
        The index of this row, as counted by the number of rows returned
        by the csv reader, excluding a row if it was used as the
        pointNames. Equivalent to the index of the point matching this
        row in the returned data.
    record : dict
        Maps row indices to those features discovered to be undesirable
        at that row index.
    toRemoveSet : set
        Contains all of the features to ignore, that are known up to
        this row. Any features we discover we want to ignore at this row
        are added to this set in this function.
    ignoreNonNumericalFeatures : bool
        Flag indicating whether features containing non numerical values
        will be removed from the data.
    """
    # We use copying of values and then returning the appropriate range
    # to simulate removal of unwanted features
    copyIndex = 0
    for i in range(len(row)):
        value = row[i]
        processed = _intFloatOrString(value)

        # A known feature to ignore
        if i in toRemoveSet:
            pass
        # A new feature to ignore, have to do book keeping
        elif (isinstance(processed, six.string_types)
              and ignoreNonNumericalFeatures):
            if pointIndex in record:
                record[pointIndex].append(i)
            else:
                record[pointIndex] = [i]

            toRemoveSet.add(i)
        else:
            row[copyIndex] = processed
            copyIndex += 1

    row = row[:copyIndex]
    return row


# Register how you want all the various input output combinations
# to be called; those combinations which are ommited will be loaded
# and then converted using some best guesses.
_loadcsvForList = _loadcsvUsingPython
_loadcsvForMatrix = _loadcsvUsingPython
_loadcsvForSparse = _loadcsvUsingPython
_loadcsvForDataFrame = _loadcsvUsingPython
_loadmtxForList = _loadmtxForAuto
_loadmtxForMatrix = _loadmtxForAuto
_loadmtxForSparse = _loadmtxForAuto
_loadmtxForDataFrame = _loadmtxForAuto


def autoRegisterFromSettings():
    """
    Helper which looks at the learners listed in UML.settings under
    the 'RegisteredLearners' section and makes sure they are registered.
    """
    # query for all entries in 'RegisteredLearners' section
    toRegister = UML.settings.get('RegisteredLearners', None)
    # call register custom learner on them
    for key in toRegister:
        try:
            (packName, _) = key.split('.')
            (modPath, attrName) = toRegister[key].rsplit('.', 1)
        except Exception:
            continue
        try:
            module = importlib.import_module(modPath)
            learnerClass = getattr(module, attrName)
            UML.registerCustomLearnerAsDefault(packName, learnerClass)
        except ImportError:
            msg = "When trying to automatically register a custom "
            msg += "learner at " + key + " we were unable to import "
            msg += "the learner object from the location " + toRegister[key]
            msg += " and have therefore ignored that configuration "
            msg += "entry"
            print(msg, file=sys.stderr)


def registerCustomLearnerBackend(customPackageName, learnerClassObject, save):
    """
    Backend for registering custom Learners in UML.

    A save value of true will run saveChanges(), modifying the config
    file. When save is False the changes will exist only for that
    session, unless saveChanges() is called later in the session.
    """
    # detect name collision
    for currInterface in UML.interfaces.available:
        if not isinstance(currInterface,
                          UML.interfaces.CustomLearnerInterface):
            if currInterface.isAlias(customPackageName):
                msg = "The customPackageName '" + customPackageName
                msg += "' cannot be used: it is an accepted alias of a "
                msg += "non-custom package"
                raise InvalidArgumentValue(msg)

    # do validation before we potentially construct an interface to a
    # custom package
    UML.customLearners.CustomLearner.validateSubclass(learnerClassObject)

    try:
        currInterface = findBestInterface(customPackageName)
    except InvalidArgumentValue:
        currInterface = UML.interfaces.CustomLearnerInterface(
            customPackageName)
        UML.interfaces.available.append(currInterface)

    currInterface.registerLearnerClass(learnerClassObject)

    opName = customPackageName + "." + learnerClassObject.__name__
    opValue = learnerClassObject.__module__ + '.' + learnerClassObject.__name__

    UML.settings.set('RegisteredLearners', opName, opValue)
    if save:
        UML.settings.saveChanges('RegisteredLearners', opName)

    # check if new option names introduced, call sync if needed
    if learnerClassObject.options() != []:
        UML.configuration.setInterfaceOptions(UML.settings, currInterface,
                                              save=save)


def deregisterCustomLearnerBackend(customPackageName, learnerName, save):
    """
    Backend for deregistering custom Learners in UML.

    A save value of true will run saveChanges(), modifying the config
    file. When save is False the changes will exist only for that
    session, unless saveChanges() is called later in the session.
    """
    currInterface = findBestInterface(customPackageName)
    if not isinstance(currInterface, UML.interfaces.CustomLearnerInterface):
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
            UML.settings.delete(customPackageName, optName)
            if save:
                UML.settings.saveChanges(customPackageName, optName)

    if empty:
        UML.interfaces.available.remove(currInterface)
        #remove section
        UML.settings.delete(customPackageName, None)
        if save:
            UML.settings.saveChanges(customPackageName)

    regOptName = customPackageName + '.' + learnerName
    # delete from registered learner list
    UML.settings.delete('RegisteredLearners', regOptName)
    if save:
        UML.settings.saveChanges('RegisteredLearners', regOptName)


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
    return max(six.iterkeys(predictionCounts),
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


def copyLabels(dataSet, dependentVar):
    """
    A helper function to simplify the process of obtaining a
    1-dimensional matrix of class labels from a data matrix.  Useful in
    functions which have an argument that may be a column index or a
    1-dimensional matrix.  If 'dependentVar' is an index, this function
    will return a copy of the column in 'dataSet' indexed by
    'dependentVar'.  If 'dependentVar' is itself a column (1-dimensional
    matrix w/shape (nx1)), dependentVar will be returned.

    Parameters
    ----------
    dataSet : matrix
        Contains labels and, possibly, features.  May be empty if
        'dependentVar' is a 1-column matrix containing labels.
    dependentVar: int, matrix
        Either a column index indicating which column in dataSet
        contains class labels, or a matrix containing 1 column of class
        labels.

    Returns
    -------
    matrix
        A 1-column matrix of class labels.
    """
    if isinstance(dependentVar, Base):
        #The known Indicator argument already contains all known
        #labels, so we do not need to do any further processing
        labels = dependentVar
    elif isinstance(dependentVar, (str, six.text_type, int)):
        #known Indicator is an index; we extract the column it indicates
        #from knownValues
        labels = dataSet.features.copy([dependentVar])
    else:
        msg = "Missing or improperly formatted indicator for known labels "
        msg += "in computeMetrics"
        raise InvalidArgumentType(msg)

    return labels


def applyCodeVersions(functionTextList, inputHash):
    """
    Applies all the different various versions of code that can be
    generated from functionText to each of the variables specified in
    inputHash, where data is plugged into the variable with name
    inputVariableName. Returns the result of each application as a list.
    functionTextList is a list of text objects, each of which defines a
    python function inputHash is of the form:
    {variable1Name:variable1Value, variable2Name:variable2Value, ...}.
    """
    results = []
    for codeText in functionTextList:
        results.append(executeCode(codeText, inputHash))
    return results


def executeCode(code, inputHash):
    """
    Execute the given code stored as text in codeText, starting with the
    variable values specified in inputHash. This function assumes the
    code consists of EITHER an entire function definition OR a single
    line of code with statements seperated by semi-colons OR as a python
    function object.
    """
    #make a copy so we don't modify it... but it doesn't seem necessary
    #inputHash = inputHash.copy()
    if isSingleLineOfCode(code):
        #it's one line of text (with ;'s to separate statements)
        return executeOneLinerCode(code, inputHash)
    elif isinstance(code, (str, six.text_type)):
        #it's the text of a function definition
        return executeFunctionCode(code, inputHash)
    else:
        # assume it's a function itself
        return code(**inputHash)


def executeOneLinerCode(codeText, inputHash):
    """
    Execute the given code stored as text in codeText, starting with the
    variable values specified in inputHash. This function assumes the
    code consists of just one line (with multiple statements seperated
    by semi-colons.
    Note: if the last statement in the line starts X=... then the X=
    gets stripped off (to prevent it from getting broken by A=(X=...)).
    """
    if not isSingleLineOfCode(codeText):
        msg = "The code text was not just one line of code."
        raise InvalidArgumentValue(msg)
    codeText = codeText.strip()
    localVariables = inputHash.copy()
    pieces = codeText.split(";")
    lastPiece = pieces[-1].strip()
    # remove 'X =' if the last statement begins with something like X = ...
    lastPiece = re.sub(r"\A([\w])+[\s]*=", "", lastPiece)
    lastPiece = lastPiece.strip()
    pieces[-1] = "RESULTING_VALUE_ZX7_ = (" + lastPiece + ")"
    codeText = ";".join(pieces)
    #oneLiner = True

    #	print "Code text: "+str(codeText)
    exec(codeText, globals(), localVariables)    #apply the code
    return localVariables["RESULTING_VALUE_ZX7_"]


def executeFunctionCode(codeText, inputHash):
    """
    Execute the given code stored as text in codeText, starting with the
    variable values specified in inputHash. This function assumes the
    code consists of an entire function definition.
    """
    if not "def" in codeText:
        msg = "No function definition was found in this code!"
        raise InvalidArgumentValue(msg)
    localVariables = {}
    # apply the code, which declares the function definition
    exec(codeText, globals(), localVariables)
    #foundFunc = False
    #result = None
    for varValue in six.itervalues(localVariables):
        if "function" in str(type(varValue)):
            return varValue(**inputHash)
    return None


def isSingleLineOfCode(codeText):
    """
    Determine if a code string is a single line.
    """
    if not isinstance(codeText, (str, six.text_type)):
        return False
    codeText = codeText.strip()
    try:
        codeText.strip().index("\n")
        return False
    except ValueError:
        return True


def _incrementTrialWindows(allData, orderedFeature, currEndTrain, minTrainSize,
                           maxTrainSize, stepSize, gap, minTestSize,
                           maxTestSize):
    """
    Helper which will calculate the start and end of the training and
    testing sizes given the current position in the full data set.
    """
    #	set_trace()
    # determine the location of endTrain.
    if currEndTrain is None:
    # points are zero indexed, thus -1 for the num of points case
    #		set_trace()
        endTrain = _jumpForward(allData, orderedFeature, 0, minTrainSize, -1)
    else:
        endTrain = _jumpForward(allData, orderedFeature, currEndTrain,
                                stepSize)

    # the value we don't want to split from the training set
    nonSplit = allData[endTrain, orderedFeature]
    # we're doing a lookahead here, thus -1 from the last possible index,
    # and  +1 to our lookup
    while (endTrain < len(allData.points) - 1
           and allData[endTrain + 1, orderedFeature] == nonSplit):
        endTrain += 1

    if endTrain == len(allData.points) - 1:
        return None

    # we get the start for training by counting back from endTrain
    startTrain = _jumpBack(allData, orderedFeature, endTrain, maxTrainSize, -1)
    if startTrain < 0:
        startTrain = 0

    # we get the start and end of the test set by counting forward from
    # endTrain speciffically, we go forward by one, and as much more forward
    # as specified by gap
    startTest = _jumpForward(allData, orderedFeature, endTrain + 1, gap)
    if startTest >= len(allData.points):
        return None

    endTest = _jumpForward(allData, orderedFeature, startTest, maxTestSize, -1)
    if endTest >= len(allData.points):
        endTest = len(allData.points) - 1
    if _diffLessThan(allData, orderedFeature, startTest, endTest, minTestSize):
        return None

    return (startTrain, endTrain, startTest, endTest)


def _jumpBack(allData, orderedFeature, start, delta, intCaseOffset=0):
    if isinstance(delta, datetime.timedelta):
        endPoint = start
        startVal = datetime.timedelta(float(allData[start, orderedFeature]))
        # loop as long as we don't run off the end of the data
        while endPoint > 0:
            prevVal = float(allData[endPoint - 1, orderedFeature])
            if (startVal - datetime.timedelta(prevVal)) > delta:
                break
            endPoint = endPoint - 1
    else:
        endPoint = start - (delta + intCaseOffset)

    return endPoint


def _jumpForward(allData, orderedFeature, start, delta, intCaseOffset=0):
    if isinstance(delta, datetime.timedelta):
        endPoint = start
        startVal = datetime.timedelta(float(allData[start, orderedFeature]))
        # loop as long as we don't run off the end of the data
        while endPoint < (len(allData.points) - 1):
            nextVal = float(allData[endPoint + 1, orderedFeature])
            if (datetime.timedelta(nextVal) - startVal) > delta:
                break
            endPoint = endPoint + 1
    else:
        endPoint = start + (delta + intCaseOffset)

    return endPoint


def _diffLessThan(allData, orderedFeature, startPoint, endPoint, delta):
    if isinstance(delta, datetime.timedelta):
        startFloat = float(allData[startPoint, orderedFeature])
        startVal = datetime.timedelta(startFloat)
        endFloat = float(allData[endPoint, orderedFeature])
        endVal = datetime.timedelta(endFloat)
        return (endVal - startVal) < delta
    else:
        return (endPoint - startPoint + 1) < delta


#def evaluate(metric, knownData, knownLabels, predictedLabels)

def computeMetrics(dependentVar, knownData, predictedData,
                   performanceFunction):
    """
    Calculate the performance of the learner.

    Using the provided metric, compare the known data or labels to the
    predicted data or labels and calculate the performance of the
    learner which produced the predicted data.

    Parameters
    ----------
    dependentVar : indentifier, list, UML Base object
        Indicate the feature names or indices in knownData containing
        the known labels, or a data object that contains the known
        labels.
    knownData : UML Base object
        Data object containing the known labels of the training set, as
        well as the features of the training set. Can be None if
        'dependentVar' is an object containing the labels.
    predictedData : UML Base object
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
        binary classification problem. See UML.calculate for a number of
        built in examples.

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


def confusion_matrix_generator(knownY, predictedY):
    """
    Given two vectors, one of known class labels (as strings) and one of
    predicted labels, compute the confusion matrix.  Returns a
    2-dimensional dictionary in which outer label is keyed by known
    label, inner label is keyed by predicted label, and the value stored
    is the count of instances for each combination.  Works for an
    indefinite number of class labels.
    """
    confusionCounts = {}
    for known, predicted in zip(knownY, predictedY):
        if confusionCounts[known] is None:
            confusionCounts[known] = {predicted: 1}
        elif confusionCounts[known][predicted] is None:
            confusionCounts[known][predicted] = 1
        else:
            confusionCounts[known][predicted] += 1

    #if there are any entries in the square matrix confusionCounts,
    #then there value must be 0.  Go through and fill them in.
    for knownY in confusionCounts:
        if confusionCounts[knownY][knownY] is None:
            confusionCounts[knownY][knownY] = 0

    return confusionCounts


def print_confusion_matrix(confusionMatrix):
    """
    Print a confusion matrix in human readable form, with rows indexed
    by known labels, and columns indexed by predictedlabels.
    confusionMatrix is a 2-dimensional dictionary, that is also
    primarily indexed by known labels, and secondarily indexed by
    predicted labels, with the value at
    confusionMatrix[knownLabel][predictedLabel] being the count of posts
    that fell into that slot.  Does not need to be sorted.
    """
    #print heading
    print("*" * 30 + "Confusion Matrix" + "*" * 30)
    print("\n\n")

    #print top line - just the column headings for
    #predicted labels
    spacer = " " * 15
    sortedLabels = sorted(confusionMatrix.iterKeys())
    for knownLabel in sortedLabels:
        spacer += " " * (6 - len(knownLabel)) + knownLabel

    print(spacer)
    totalPostCount = 0
    for knownLabel in sortedLabels:
        outputBuffer = knownLabel + " " * (15 - len(knownLabel))
        for predictedLabel in sortedLabels:
            count = confusionMatrix[knownLabel][predictedLabel]
            totalPostCount += count
            outputBuffer += " " * (6 - len(count)) + count
        print(outputBuffer)

    print("Total post count: " + totalPostCount)


def checkPrintConfusionMatrix():
    """
    Check print ouptut of confusion matrix.
    """
    X = {"classLabel": ["A", "B", "C", "C", "B", "C", "A", "B", "C", "C",
                        "B", "C", "A", "B", "C", "C", "B", "C"]}
    Y = ["A", "C", "C", "A", "B", "C", "A", "C", "C", "A", "B", "C", "A",
         "C", "C", "A", "B", "C"]
    functions = [confusion_matrix_generator]
    classLabelIndex = "classLabel"
    confusionMatrixResults = computeMetrics(classLabelIndex, X, Y, functions)
    confusionMatrix = confusionMatrixResults["confusion_matrix_generator"]
    print_confusion_matrix(confusionMatrix)


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
        UML compliant algorithm name in the form 'package.algorithm'
        e.g. 'sciKitLearn.KNeighborsClassifier'
    X : UML Base object
        points/features data
    Y : UML Base object
        labels/data about points in X
    performanceFunction : function
        Premade options are available in UML.calculate.
        Function used to evaluate the performance score for each run.
        Function is of the form: def func(knownValues, predictedValues).
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To trigger cross-validation using multiple values for arguments,
        specify different values for each parameter using a UML.CV
        object. eg. {'k': UML.CV([1,3,5])} will generate an error score
        for  the learner when the learner was passed all three values of
        ``k``, separately. These will be merged any kwarguments for the
        learner.
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
        UML.CV object. eg. arg1=UML.CV([1,2,3]), arg2=UML.CV([4,5,6])
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
        detected = UML.calculate.detectBestResult(performanceFunction)
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
        X : UML Base object
            points/features data
        Y : UML Base object
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
            if not isinstance(Y, (Base, int, six.string_types, list)):
                msg = "Y must be a Base object or an index (int) from X where "
                msg += "Y's data can be found"
                raise InvalidArgumentType(msg)
            if isinstance(Y, (int, six.string_types, list)):
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
                curRunResult = UML.trainAndApply(
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
                    collectedY.points.add(curTestingY, useLog=False)

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
                    results[0].points.add(results[resultIndex], useLog=False)

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
        >>> UML.setRandomSeed(42)
        >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
        >>> yRaw = [[1], [2], [3],
        ...         [1], [2], [3],
        ...         [1], [2], [3]]
        >>> X = UML.createData('Matrix', xRaw)
        >>> Y = UML.createData('Matrix', yRaw)
        >>> crossValidator = KFoldCrossValidator(
        ...    'Custom.KNNClassifier', X, Y, arguments={'k': 3},
        ...    performanceFunction=UML.calculate.fractionIncorrect,
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
        >>> UML.setRandomSeed(42)
        >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
        >>> yRaw = [[1], [2], [3],
        ...         [1], [2], [3],
        ...         [1], [2], [3]]
        >>> X = UML.createData('Matrix', xRaw)
        >>> Y = UML.createData('Matrix', yRaw)
        >>> kValues = UML.CV([1, 3])
        >>> crossValidator = KFoldCrossValidator(
        ...    'Custom.KNNClassifier', X, Y, arguments={},
        ...    performanceFunction=UML.calculate.fractionIncorrect,
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
        >>> UML.setRandomSeed(42)
        >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
        >>> yRaw = [[1], [2], [3],
        ...         [1], [2], [3],
        ...         [1], [2], [3]]
        >>> X = UML.createData('Matrix', xRaw)
        >>> Y = UML.createData('Matrix', yRaw)
        >>> kValues = UML.CV([1, 3])
        >>> crossValidator = KFoldCrossValidator(
        ...    'Custom.KNNClassifier', X, Y, arguments={},
        ...    performanceFunction=UML.calculate.fractionIncorrect,
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
                    msg = "One of the objects has 0 points, it is impossible to "
                    msg += "specify a valid number of folds"
                    raise InvalidArgumentValueCombination(msg)
                if len(data.points) != len(self.dataList[0].points):
                    msg = "All data objects in the list must have the same number "
                    msg += "of points and features"
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
        e.g. {'a': CV([1, 2, 3]), 'b': UML.CV([4,5]), 'c': 6}
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
                if isinstance(rawArgumentInput[key], CV):
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

class CV(object):
    def __init__(self, argumentList):
        try:
            self.argumentTuple = tuple(argumentList)
        except TypeError:
            msg = "argumentList must be iterable."

    def __getitem__(self, key):
        return self.argumentTuple[key]

    def __setitem__(self, key, value):
        raise ImproperObjectAction("CV objects are immutable")

    def __len__(self):
        return len(self.argumentTuple)

    def __str__(self):
        return str(self.argumentTuple)

    def __repr__(self):
        return "CV(" + str(list(self.argumentTuple)) + ")"

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
    tuple of UML.Base objects:
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


    # todo verify that your list of lists is valid initializer for all
    # datatypes, not just matrix
    # then convert
    # finally make matrix object out of the list of points w/ labels in last
    # column of each vector/entry:
    pointsObj = UML.createData('Matrix', pointsList, useLog=False)

    labelsObj = UML.createData('Matrix', labelsList, useLog=False)

    # todo change actuallavels to something like associatedClusterCentroid
    noiselessLabelsObj = UML.createData('Matrix', clusterNoiselessLabelList,
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
        if not isinstance(learnerName, six.string_types):
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
            runResults = UML.trainAndApply(
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
            runResults = UML.trainAndApply(learnerName, trainX=trainData,
                                           trainY=trainLabels, testX=testData)
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
        if not isinstance(trainY, (Base, six.string_types, int, numpy.int64)):
            msg = "trainY may only be an object derived from Base, or an "
            msg += "ID of the feature containing labels in testX"
            raise InvalidArgumentType(msg)
        if isinstance(trainY, Base):
        #			if not len(trainY.features) == 1:
        #               msg = "If trainY is a Data object, then it may only "
        #               msg += "have one feature"
        #				raise ArgumentException(msg)
            if not len(trainY.points) == len(trainX.points):
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
        if not isinstance(testY, (Base, six.string_types, int, int)):
            msg = "testY may only be an object derived from Base, or an ID "
            msg += "of the feature containing labels in testX"
            raise InvalidArgumentType(msg)
        if isinstance(trainY, Base):
        #			if not len(trainY.features) == 1:
        #               msg = "If trainY is a Data object, then it may only "
        #               msg += "have one feature"
        #				raise ArgumentException(msg)
            if not len(trainY.points) == len(trainX.points):
                msg = "If trainY is a Data object, then it must have the same "
                msg += "number of points as trainX"
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


def trainAndApplyOneVsOne(learnerName, trainX, trainY, testX, arguments=None,
                          scoreMode='label', useLog=None, **kwarguments):
    """
    Calls on trainAndApply() to train and evaluate the learner defined
    by 'learnerName.'  Assumes there are multiple (>2) class labels, and
    uses the one vs. one method of splitting the training set into
    2-label subsets. Tests performance using the metric function(s)
    found in performanceMetricFunctions.

    Parameters
    ----------
    trainX: UML Base object
        Data to be used for training.
    trainY: identifier, UML Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another UML Base object containing the labels that
        correspond to ``trainX``.
    testX : UML Base object
        data set on which the trained learner will be applied (i.e.
        performing prediction, transformation, etc. as appropriate to
        the learner).
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To make use of multiple permutations, specify different values
        for a parameter as a tuple. eg. {'k': (1,3,5)} will generate an
        error score for  the learner when the learner was passed all
        three values of ``k``, separately. These will be merged with
        kwarguments for the learner.
    scoreMode : str
        In the case of a classifying learner, this specifies the type of
        output wanted: 'label' if we class labels are desired,
        'bestScore' if both the class label and the score associated
        with that class are desired, or 'allScores' if a matrix
        containing the scores for every class label are desired.
    useLog : bool, None
        Local control for whether to send results/timing to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. To make use of multiple permutations, specify different
        values for parameters as a tuple. eg. arg1=(1,2,3), arg2=(4,5,6)
        which correspond to permutations/argument states with one
        element from arg1 and one element from arg2, such that an
        example generated permutation/argument state would be
        ``arg1=2, arg2=4``. Will be merged with ``arguments``.
    """
    _validData(trainX, trainY, testX, None, [True, False])
    _validArguments(arguments)
    _validArguments(kwarguments)
    merged = _mergeArguments(arguments, kwarguments)

    # we want the data and the labels together in one object or this method
    trainX = trainX.copy()
    if isinstance(trainY, Base):
        trainX.features.add(trainY)
        trainY = len(trainX.features) - 1

    # Get set of unique class labels, then generate list of all 2-combinations
    # of class labels
    labelVector = trainX.features.copy([trainY])
    labelVector.transpose()
    labelSet = list(set(labelVector.copy(to="python list")[0]))
    labelPairs = generateAllPairs(labelSet)

    # For each pair of class labels: remove all points with one of those
    # labels, train a classifier on those points, get predictions based on
    # that model, and put the points back into the data object
    rawPredictions = None
    predictionFeatureID = 0
    for pair in labelPairs:
        #get all points that have one of the labels in pair
        pairData = trainX.points.extract(
            lambda point: ((point[trainY] == pair[0])
                           or (point[trainY] == pair[1])))
        pairTrueLabels = pairData.features.extract(trainY)
        #train classifier on that data; apply it to the test set
        partialResults = UML.trainAndApply(learnerName, pairData,
                                           pairTrueLabels, testX, output=None,
                                           arguments=merged, useLog=useLog)
        #put predictions into table of predictions
        if rawPredictions is None:
            rawPredictions = partialResults.copy(to="List")
        else:
            predName = 'predictions-' + str(predictionFeatureID)
            partialResults.features.setName(0, predName)
            rawPredictions.features.add(partialResults.copy(to="List"))
        pairData.features.add(pairTrueLabels)
        trainX.points.add(pairData)
        predictionFeatureID += 1

    #set up the return data based on which format has been requested
    if scoreMode.lower() == 'label'.lower():
        ret = rawPredictions.points.calculate(extractWinningPredictionLabel)
        ret.features.setName(0, "winningLabel")
        return ret
    elif scoreMode.lower() == 'bestScore'.lower():
        # construct a list of lists, with each row in the list containing the
        # predicted label and score of that label for the corresponding row in
        # rawPredictions
        predictionMatrix = rawPredictions.copy(to="python list")
        tempResultsList = []
        for row in predictionMatrix:
            scores = countWins(row)
            sortedScores = sorted(scores, key=scores.get, reverse=True)
            bestLabel = sortedScores[0]
            tempResultsList.append([bestLabel, scores[bestLabel]])

        #wrap the results data in a List container
        featureNames = ['PredictedClassLabel', 'LabelScore']
        resultsContainer = UML.createData("List", tempResultsList,
                                          featureNames=featureNames,
                                          useLog=False)
        return resultsContainer
    elif scoreMode.lower() == 'allScores'.lower():
        columnHeaders = sorted([str(i) for i in labelSet])
        zipIndexLabel = zip(list(range(len(columnHeaders))), columnHeaders)
        labelIndexDict = {str(v): k for k, v in zipIndexLabel}
        predictionMatrix = rawPredictions.copy(to="python list")
        resultsContainer = []
        for row in predictionMatrix:
            finalRow = [0] * len(columnHeaders)
            scores = countWins(row)
            for label, score in scores.items():
                finalIndex = labelIndexDict[str(label)]
                finalRow[finalIndex] = score
            resultsContainer.append(finalRow)

        return UML.createData(rawPredictions.getTypeString(), resultsContainer,
                              featureNames=columnHeaders, useLog=False)
    else:
        msg = 'Unknown score mode in trainAndApplyOneVsOne: ' + str(scoreMode)
        raise InvalidArgumentValue(msg)


def trainAndApplyOneVsAll(learnerName, trainX, trainY, testX, arguments=None,
                          scoreMode='label', useLog=None, **kwarguments):
    """
    Calls on trainAndApply() to train and evaluate the learner defined
    by 'learnerName.'  Assumes there are multiple (>2) class labels, and
    uses the one vs. all method of splitting the training set into
    2-label subsets. Tests performance using the metric function(s)
    found in performanceMetricFunctions.

    Parameters
    ----------
    trainX: UML Base object
        Data to be used for training.
    trainY: identifier, UML Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another UML Base object containing the labels that
        correspond to ``trainX``.
    testX : UML Base object
        data set on which the trained learner will be applied (i.e.
        performing prediction, transformation, etc. as appropriate to
        the learner).
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To make use of multiple permutations, specify different values
        for a parameter as a tuple. eg. {'k': (1,3,5)} will generate an
        error score for  the learner when the learner was passed all
        three values of ``k``, separately. These will be merged with
        kwarguments for the learner.
    scoreMode : str
        In the case of a classifying learner, this specifies the type of
        output wanted: 'label' if we class labels are desired,
        'bestScore' if both the class label and the score associated
        with that class are desired, or 'allScores' if a matrix
        containing the scores for every class label are desired.
    useLog : bool, None
        Local control for whether to send results/timing to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. To make use of multiple permutations, specify different
        values for parameters as a tuple. eg. arg1=(1,2,3), arg2=(4,5,6)
        which correspond to permutations/argument states with one
        element from arg1 and one element from arg2, such that an
        example generated permutation/argument state would be
        ``arg1=2, arg2=4``. Will be merged with ``arguments``.
    """
    _validData(trainX, trainY, testX, None, [True, False])
    _validArguments(arguments)
    _validArguments(kwarguments)
    merged = _mergeArguments(arguments, kwarguments)

    #Remove true labels from from training set, if not already separated
    if isinstance(trainY, (str, int, int)):
        trainX = trainX.copy()
        trainY = trainX.features.extract(trainY)

    # Get set of unique class labels
    labelVector = trainY.copy()
    labelVector.transpose()
    labelSet = list(set(labelVector.copy(to="python list")[0]))

    # For each class label in the set of labels:  convert the true
    # labels in trainY into boolean labels (1 if the point
    # has 'label', 0 otherwise.)  Train a classifier with the processed
    # labels and get predictions on the test set.
    rawPredictions = None

    def relabeler(point, label=None):
        if point[0] != label:
            return 0
        else:
            return 1

    for label in labelSet:
        relabeler.__defaults__ = (label,)
        trainLabels = trainY.points.calculate(relabeler)
        oneLabelResults = UML.trainAndApply(learnerName, trainX, trainLabels,
                                            testX, output=None,
                                            arguments=merged, useLog=useLog)
        # put all results into one Base container, of the same type as trainX
        if rawPredictions is None:
            rawPredictions = oneLabelResults
            # as it's added to results object, rename each column with its
            # corresponding class label
            rawPredictions.features.setName(0, str(label))
        else:
            # as it's added to results object, rename each column with its
            # corresponding class label
            oneLabelResults.features.setName(0, str(label))
            rawPredictions.features.add(oneLabelResults)

    if scoreMode.lower() == 'label'.lower():
        winningPredictionIndices = rawPredictions.points.calculate(
            extractWinningPredictionIndex).copy(to="python list")
        winningLabels = []
        for [winningIndex] in winningPredictionIndices:
            winningLabels.append([labelSet[int(winningIndex)]])
        return UML.createData(rawPredictions.getTypeString(), winningLabels,
                              featureNames=['winningLabel'], useLog=False)

    elif scoreMode.lower() == 'bestScore'.lower():
        # construct a list of lists, with each row in the list containing the
        # predicted label and score of that label for the corresponding row in
        # rawPredictions
        predictionMatrix = rawPredictions.copy(to="python list")
        indexToLabel = rawPredictions.features.getNames()
        tempResultsList = []
        for row in predictionMatrix:
            bestLabelAndScore = extractWinningPredictionIndexAndScore(
                row, indexToLabel)
            tempResultsList.append([bestLabelAndScore[0],
                                    bestLabelAndScore[1]])
        #wrap the results data in a List container
        featureNames = ['PredictedClassLabel', 'LabelScore']
        resultsContainer = UML.createData("List", tempResultsList,
                                          featureNames=featureNames,
                                          useLog=False)
        return resultsContainer

    elif scoreMode.lower() == 'allScores'.lower():
        #create list of Feature Names/Column Headers for final return object
        columnHeaders = sorted([str(i) for i in labelSet])
        #create map between label and index in list, so we know where to put
        # each value
        zipIndexLabel = zip(list(range(len(columnHeaders))), columnHeaders)
        labelIndexDict = {v: k for k, v in zipIndexLabel}
        featureNamesItoN = rawPredictions.features.getNames()
        predictionMatrix = rawPredictions.copy(to="python list")
        resultsContainer = []
        for row in predictionMatrix:
            finalRow = [0] * len(columnHeaders)
            scores = extractConfidenceScores(row, featureNamesItoN)
            for label, score in scores.items():
                #get numerical index of label in return object
                finalIndex = labelIndexDict[label]
                #put score into proper place in its row
                finalRow[finalIndex] = score
            resultsContainer.append(finalRow)
        #wrap data in Base container
        return UML.createData(rawPredictions.getTypeString(), resultsContainer,
                              featureNames=columnHeaders, useLog=False)
    else:
        msg = 'Unknown score mode in trainAndApplyOneVsAll: ' + str(scoreMode)
        raise InvalidArgumentValue(msg)


def trainAndTestOneVsAny(learnerName, f, trainX, trainY, testX, testY,
                         arguments=None, performanceFunction=None, useLog=None,
                         **kwarguments):
    """
    This function is the base model of function trainAndTestOneVsOne and
    trainAndTestOneVsAll.
    """
    _validData(trainX, trainY, testX, testY, [True, True])
    _validArguments(arguments)
    _validArguments(kwarguments)
    merged = _mergeArguments(arguments, kwarguments)

    # timer = Stopwatch() if useLog else None

    # if testY is in testX, we need to extract it before we call a
    # trainAndApply type function
    if isinstance(testY, (six.string_types, int, int)):
        testX = testX.copy()
        testY = testX.features.extract([testY])

    predictions = f(learnerName, trainX, trainY, testX, merged,
                    scoreMode='label', useLog=False)

    # now compute performance metric(s) for the set of winning predictions
    results = computeMetrics(testY, None, predictions, performanceFunction)

    metrics = {}
    for key, value in zip([performanceFunction], [results]):
        metrics[key.__name__] = value

    # Send this run to the log, if desired
    # if useLog:
    #     if not isinstance(performanceFunction, list):
    #         performanceFunction = [performanceFunction]
    #         results = [results]
    #     UML.logger.active.logRun(f.__name__, trainX, trainY, testX, testY,
    #                              learnerName, merged, metrics, timer)

    return results

def trainAndTestOneVsAll(learnerName, trainX, trainY, testX, testY,
                         arguments=None, performanceFunction=None, useLog=None,
                         **kwarguments):
    """
    Calls on trainAndApply() to train and evaluate the learner defined
    by 'learnerName.'  Assumes there are multiple (>2) class labels, and
    uses the one vs. all method of splitting the training set into
    2-label subsets. Tests performance using the metric function(s)
    found in performanceMetricFunctions.

    Parameters
    ----------
    learnerName : str
        Name of the learner to be called, in the form 'package.learner'
    trainX: UML Base object
        Data to be used for training.
    trainY : identifier, UML Base object
        * identifier - The name or index of the feature in ``trainX``
          containing the labels.
        * UML Base object - contains the labels that correspond to
          ``trainX``.
    testX: UML Base object
        Data to be used for testing.
    testY : identifier, UML Base object
        * identifier - A name or index of the feature in ``testX``
          containing the labels.
        * UML Base object - contains the labels that correspond to
          ``testX``.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To make use of multiple permutations, specify different values
        for a parameter as a tuple. eg. {'k': (1,3,5)} will generate an
        error score for  the learner when the learner was passed all
        three values of ``k``, separately. These will be merged with
        kwarguments for the learner.
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in UML.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
    useLog : bool, None
        Local control for whether to send results/timing to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. To make use of multiple permutations, specify different
        values for parameters as a tuple. eg. arg1=(1,2,3), arg2=(4,5,6)
        which correspond to permutations/argument states with one
        element from arg1 and one element from arg2, such that an
        example generated permutation/argument state would be
        ``arg1=2, arg2=4``. Will be merged with ``arguments``.
    """
    return trainAndTestOneVsAny(learnerName=learnerName, trainX=trainX,
                                trainY=trainY, testX=testX, testY=testY,
                                f=trainAndApplyOneVsAll, arguments=arguments,
                                performanceFunction=performanceFunction,
                                useLog=useLog, **kwarguments)

def trainAndTestOneVsOne(learnerName, trainX, trainY, testX, testY,
                         arguments=None, performanceFunction=None, useLog=None,
                         **kwarguments):
    """
    Wrapper class for trainAndApplyOneVsOne.  Useful if you want the
    entire process of training, testing, and computing performance
    measures to be handled.  Takes in a learner's name and training and
    testing data sets, trains a learner, passes the test data to the
    computed model, gets results, and calculates performance based on
    those results.

    Parameters
    ----------
    learnerName : str
        Name of the learner to be called, in the form 'package.learner'
    trainX: UML Base object
        Data to be used for training.
    trainY : identifier, UML Base object
        * identifier - The name or index of the feature in ``trainX``
          containing the labels.
        * UML Base object - contains the labels that correspond to
          ``trainX``.
    testX: UML Base object
        Data to be used for testing.
    testY : identifier, UML Base object
        * identifier - A name or index of the feature in ``testX``
          containing the labels.
        * UML Base object - contains the labels that correspond to
          ``testX``.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To make use of multiple permutations, specify different values
        for a parameter as a tuple. eg. {'k': (1,3,5)} will generate an
        error score for  the learner when the learner was passed all
        three values of ``k``, separately. These will be merged with
        kwarguments for the learner.
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in UML.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
    useLog : bool, None
        Local control for whether to send results/timing to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. To make use of multiple permutations, specify different
        values for parameters as a tuple. eg. arg1=(1,2,3), arg2=(4,5,6)
        which correspond to permutations/argument states with one
        element from arg1 and one element from arg2, such that an
        example generated permutation/argument state would be
        ``arg1=2, arg2=4``. Will be merged with ``arguments``.

    Returns
    -------
    dict
        A dictionary associating the name or code of performance metrics
        with the results of those metrics, computed using the
        predictions of 'learnerName' on testX.
        Example: { 'fractionIncorrect': 0.21, 'numCorrect': 1020 }
    """
    return trainAndTestOneVsAny(learnerName=learnerName, trainX=trainX,
                                trainY=trainY, testX=testX, testY=testY,
                                f=trainAndApplyOneVsOne,
                                arguments=arguments,
                                performanceFunction=performanceFunction,
                                useLog=useLog, **kwarguments)


def inspectArguments(func):
    """
    To be used in place of inspect.getargspec for Python3 compatibility.
    Return is the tuple (args, varargs, keywords, defaults)
    """
    try:
        # py>=3.3
        # in py>=3.5 inspect.signature can extract the original signature
        # of wrapped functions
        sig = inspect.signature(func)
        a = []
        if inspect.isclass(func) or hasattr(func, '__self__'):
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
        try:
            argspec = inspect.getfullargspec(func)[:4] # p>=3
        except AttributeError:
            argspec = inspect.getargspec(func) # py2
    return argspec
