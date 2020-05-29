"""
Module the user-facing data creation functions for the top level
nimble import.
"""

import numpy

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble.core.logger import handleLogging
from nimble._utility import scipy, cloudpickle
from nimble.core._createHelpers import validateReturnType
from nimble.core._createHelpers import isAllowedRaw
from nimble.core._createHelpers import initDataObject
from nimble.core._createHelpers import createDataFromFile
from nimble.core._createHelpers import createConstantHelper


def data(returnType, data, pointNames='automatic', featureNames='automatic',
         convertToType=None, name=None, path=None, keepPoints='all',
         keepFeatures='all', ignoreNonNumericalFeatures=False,
         reuseData=False, inputSeparator='automatic',
         treatAsMissing=(float('nan'), numpy.nan, None, '', 'None', 'nan',
                         'NULL', 'NA'),
         replaceMissingWith=numpy.nan, useLog=None):
    """
    Function to instantiate one of the nimble data container types.

    Creates a nimble data object based on the ``returnType``.  Data can
    be loaded in a raw form, from a file or from a web page.  Some
    preprocessing of the data can also be done when creating the object
    through the passing various arguments.

    Parameters
    ----------
    returnType : str, None
        Indicates which kind of nimble data type to return. Currently
        accepted are the strings "List", "Matrix", "Sparse" and
        "DataFrame" -- which are **case sensitive**. If None is given,
        nimble will attempt to detect the type most appropriate for the
        data.
    data : object, str
        The source of the data to be loaded into the returned object.
        The source may be any number of in-python objects (lists,
        numpy arrays, numpy matrices, scipy sparse objects, pandas
        dataframes) as long as they specify a 2d matrix of data.
        Alternatively, the data may be read from a file, specified
        either as a string path, or a currently open file-like object.
    pointNames : 'automatic', bool, list, dict
        Specifices the source for point names in the returned object.

        * 'automatic' - the default, indicates that this function should
          attempt to detect the presence of pointNames in the data which
          will only be attempted when loading from a file. If no names
          are found, or data isn't being loaded from a file, then
          default names are assigned.
        * bool - True indicates that point names are embedded in the
          data within the first column. A value of False indicates that
          names are not embedded and that default names should be used.
        * list, dict - all points in the data must be assigned a name
          and the names for each point must be unique. As a list, the
          index of the name will define the point index. As a dict,
          the value mapped to each name will define the point index.
    featureNames : 'automatic', bool, list, dict
        Specifices the source for feature names in the returned object.

        * 'automatic' - the default, indicates that this function should
          attempt to detect the presence of featureNames in the data
          which will only be attempted when loading from a file. If no
          names are found, or data isn't being loaded from a file, then
          default names are assigned.
        * bool - True indicates that feature names are embedded in the
          data within the first row. A value of False indicates that
          names are not embedded and that default names should be used.
        * list, dict - all features in the data must be assigned a name
          and the names for each feature must be unique. As a list, the
          index of the name will define the feature index. As a dict,
          the value mapped to each name will define the feature index.
    convertToType : type
        A one-time conversion of all the data to this type. If unable to
        convert every value to the given type, an exception will be
        raised. The default, None, will retain the object types as is
        when creating the object. Note: This only applies during the
        creation process, nimble will modify types on the backend as
        necessary.
    name : str
        When not None, this value is set as the name attribute of the
        returned object.
    keepPoints : bool
        Allows the user to select which points will be kept in the
        returned object, those not selected will be discarded. By
        default, the value 'all' indicates that all possible points in
        the raw data will be kept. Alternatively, the user may provide a
        list containing either names or indices (or a mix) of those
        points they want to be kept from the raw data. The order of this
        list will determine the order of points in the resultant object.
        In the case of reading data from a file, the selection will be
        done at read time, thus limiting the amount of data read into
        memory.
    keepFeatures : bool
        Allows the user to select which features will be kept in the
        returned object, those not selected will be discarded. By
        default, the value 'all' indicates that all possible features in
        the raw data will be kept. Alternatively, the user may provide a
        list containing either names or indices (or a mix) of those
        features they want to be kept from the raw data. The order of
        this list will determine the order of features in the resultant
        object. In the case of reading data from a file, the selection
        will be done at read time, thus limiting the amount of data read
        into memory. Names and indices are defined with respect to the
        data regardless of filtering by the ignoreNonNumericalFeatures
        flag; just because a feature is removed, the indices of
        subsequent features will not be shifted. The
        ``ignoreNonNumericalFeatures`` flag is only consdered after
        selection: if a selected feature has non-numerical values and
        ignoreNonNumericalFeatures is True valued, then that feature
        will **NOT** be included in the output. Similarly, if a feature
        has only numerical values in points that were selected, then
        even if there are non-numerical values in the points that were
        not selected, then that feature will be included
    ignoreNonNumericalFeatures : bool
        Indicate whether, when loading from a file, features containing
        non numercal data shouldn't be loaded into the final object. For
        example, you may be loading a file which has a column of
        strings; setting this flag to true will allow you to load that
        file into a Matrix object (which may contain floats only).
        Currently only has an effect on csv files, as the matrix market
        format does not support non numerical values. Also, if there is
        point or feature selection occurring, then only those values
        within selected points and features are considered when
        determining whether to apply this operation.
    inputSeparator : str
        The character that is used to separate fields in the input file,
        if necessary. By default, a value of 'automatic' will attempt to
        determine the appropriate separator. Otherwise, a single
        character string of the separator in the file can be passed.
    treatAsMissing : list
        Values that will be treated as missing values in the data. These
        values will be replaced with value from replaceMissingWith
        By default this list is [float('nan'), numpy.nan, None, '',
        'None', 'nan']. Set to None or [] to disable replacing missing
        values.
    replaceMissingWith
        A single value with which to replace any value in
        treatAsMissing. By default this value is numpy.nan.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.

    Returns
    -------
    nimble.core.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    ones, zeros, identity, nimble.random.data

    Examples
    --------
    >>> data = [[1, 2, 3], [4, 5, 6]]
    >>> asList = nimble.data('List', data, name='simple')
    >>> asList
    List(
        [[1 2 3]
         [4 5 6]]
        name="simple"
        )

    Loading data from a file.

    >>> with open('simpleData.csv', 'w') as cd:
    ...     out = cd.write('1,2,3\\n4,5,6')
    >>> fromFile = nimble.data('Matrix', 'simpleData.csv')
    >>> fromFile # doctest: +ELLIPSIS
    Matrix(
        [[1 2 3]
         [4 5 6]]
        name="simpleData.csv"
        path="...simpleData.csv"
        )

    Adding point and feature names.

    >>> data = [['a', 'b', 'c'], [0, 0, 1], [1, 0, 0]]
    >>> asSparse = nimble.data('Sparse', data, pointNames=['1', '2'],
    ...                        featureNames=True)
    >>> asSparse
    Sparse(
        [[0 0 1]
         [1 0 0]]
        pointNames={'1':0, '2':1}
        featureNames={'a':0, 'b':1, 'c':2}
        )

    Replacing missing values.

    >>> data = [[1, 'Missing', 3], [4, 'Missing', 6]]
    >>> ftNames = {'a': 0, 'b': 1, 'c': 2}
    >>> asDataFrame = nimble.data('DataFrame', data,
    ...                           featureNames=ftNames,
    ...                           treatAsMissing=["Missing", 3],
    ...                           replaceMissingWith=-1)
    >>> asDataFrame
    DataFrame(
        [[1 -1 -1]
         [4 -1 6 ]]
        featureNames={'a':0, 'b':1, 'c':2}
        )
    """
    validateReturnType(returnType)

    def looksFileLike(toCheck):
        hasRead = hasattr(toCheck, 'read')
        hasWrite = hasattr(toCheck, 'write')
        return hasRead and hasWrite

    # input is raw data
    if isAllowedRaw(data, allowLPT=True):
        ret = initDataObject(
            returnType=returnType, rawData=data, pointNames=pointNames,
            featureNames=featureNames, convertToType=convertToType, name=name,
            path=path, keepPoints=keepPoints, keepFeatures=keepFeatures,
            reuseData=reuseData, treatAsMissing=treatAsMissing,
            replaceMissingWith=replaceMissingWith)
    # input is an open file or a path to a file
    elif isinstance(data, str) or looksFileLike(data):
        ret = createDataFromFile(
            returnType=returnType, data=data, pointNames=pointNames,
            featureNames=featureNames, name=name, keepPoints=keepPoints,
            keepFeatures=keepFeatures,
            ignoreNonNumericalFeatures=ignoreNonNumericalFeatures,
            inputSeparator=inputSeparator, treatAsMissing=treatAsMissing,
            replaceMissingWith=replaceMissingWith)
    # no other allowed inputs
    else:
        msg = "data must contain either raw data or the path to a file to be "
        msg += "loaded"
        raise InvalidArgumentType(msg)

    handleLogging(useLog, 'load', returnType, len(ret.points),
                  len(ret.features), name, path)
    return ret


def ones(returnType, numPoints, numFeatures, pointNames='automatic',
         featureNames='automatic', name=None):
    """
    Return a data object of the given shape containing all 1 values.

    Parameters
    ----------
    returnType : str
        May be any of the allowed types specified in
        nimble.core.data.available.
    numPoints : int
        The number of points in the returned object.
    numFeatures : int
        The number of features in the returned object.
    pointNames : 'automatic', list, dict
        Names to be associated with the points in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by some list-like or dict-like object, so
        long as all points in the data are assigned a name and the names
        for each point are unique.
    featureNames : 'automatic', list, dict
        Names to be associated with the features in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by some list-like or dict-like object, so
        long as all features in the data are assigned a name and the
        names for each feature are unique.
    name : str
        When not None, this value is set as the name attribute of the
        returned object.

    Returns
    -------
    nimble.core.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    zeros

    Examples
    --------
    Ones with default names.

    >>> ones = nimble.ones('List', 5, 5)
    >>> ones
    List(
        [[1.000 1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000 1.000]]
        )

    Named object of ones with pointNames and featureNames.

    >>> onesDF = nimble.ones('DataFrame', 4, 4,
    ...                      pointNames=['1', '2', '3', '4'],
    ...                      featureNames=['a', 'b', 'c', 'd'],
    ...                      name='ones DataFrame')
    >>> onesDF
    DataFrame(
        [[1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000]]
        pointNames={'1':0, '2':1, '3':2, '4':3}
        featureNames={'a':0, 'b':1, 'c':2, 'd':3}
        name="ones DataFrame"
        )
    """
    return createConstantHelper(numpy.ones, returnType, numPoints, numFeatures,
                                pointNames, featureNames, name)


def zeros(returnType, numPoints, numFeatures, pointNames='automatic',
          featureNames='automatic', name=None):
    """
    Return a data object of the given shape containing all 0 values.

    Parameters
    ----------
    returnType : str
        May be any of the allowed types specified in
        nimble.core.data.available.
    numPoints : int
        The number of points in the returned object.
    numFeatures : int
        The number of features in the returned object.
    pointNames : 'automatic', list, dict
        Names to be associated with the points in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by some list-like or dict-like object, so
        long as all points in the data are assigned a name and the names
        for each point are unique.
    featureNames : 'automatic', list, dict
        Names to be associated with the features in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by some list-like or dict-like object, so
        long as all features in the data are assigned a name and the
        names for each feature are unique.
    name : str
        When not None, this value is set as the name attribute of the
        returned object.

    Returns
    -------
    nimble.core.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    ones

    Examples
    --------
    Zeros with default names.

    >>> zeros = nimble.zeros('Matrix', 5, 5)
    >>> zeros
    Matrix(
        [[0.000 0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000 0.000]]
        )

    Named object of zeros with pointNames and featureNames.

    >>> zerosSparse = nimble.zeros('Sparse', 4, 4,
    ...                            pointNames=['1', '2', '3', '4'],
    ...                            featureNames=['a', 'b', 'c', 'd'],
    ...                            name='Sparse all-zeros')
    >>> zerosSparse
    Sparse(
        [[0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]]
        pointNames={'1':0, '2':1, '3':2, '4':3}
        featureNames={'a':0, 'b':1, 'c':2, 'd':3}
        name="Sparse all-zeros"
        )
    """
    return createConstantHelper(numpy.zeros, returnType, numPoints,
                                numFeatures, pointNames, featureNames, name)


def identity(returnType, size, pointNames='automatic',
             featureNames='automatic', name=None):
    """
    Return a data object representing an identity matrix.

    The returned object will always be a square with the number of
    points and features equal to ``size``.  The main diagonal will have
    values of 1 and every other value will be zero.

    Parameters
    ----------
    returnType : str
        May be any of the allowed types specified in
        nimble.core.data.available.
    size : int
        The number of points and features in the returned object.
    pointNames : 'automatic', list, dict
        Names to be associated with the points in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by some list-like or dict-like object, so
        long as all points in the data are assigned a name and the names
        for each point are unique.
    featureNames : 'automatic', list, dict
        Names to be associated with the features in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by some list-like or dict-like object, so
        long as all features in the data are assigned a name and the
        names for each feature are unique.
    name : str
        When not None, this value is set as the name attribute of the
        returned object.

    Returns
    -------
    nimble.core.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    Examples
    --------
    Identity matrix with default names.

    >>> identity = nimble.identity('Matrix', 5)
    >>> identity
    Matrix(
        [[1.000 0.000 0.000 0.000 0.000]
         [0.000 1.000 0.000 0.000 0.000]
         [0.000 0.000 1.000 0.000 0.000]
         [0.000 0.000 0.000 1.000 0.000]
         [0.000 0.000 0.000 0.000 1.000]]
        )

    Named object of zeros with pointNames and featureNames.

    >>> identityList = nimble.identity('List', 3,
    ...                             pointNames=['1', '2', '3'],
    ...                             featureNames=['a', 'b', 'c'],
    ...                             name='identity matrix list')
    >>> identityList
    List(
        [[1.000 0.000 0.000]
         [0.000 1.000 0.000]
         [0.000 0.000 1.000]]
        pointNames={'1':0, '2':1, '3':2}
        featureNames={'a':0, 'b':1, 'c':2}
        name="identity matrix list"
        )
    """
    validateReturnType(returnType)
    if size <= 0:
        msg = "size must be 0 or greater, yet " + str(size)
        msg += " was given."
        raise InvalidArgumentValue(msg)

    if returnType == 'Sparse':
        if not scipy.nimbleAccessible():
            msg = "scipy is not available"
            raise PackageException(msg)

        assert returnType == 'Sparse'
        rawDiag = scipy.sparse.identity(size)
        rawCoo = scipy.sparse.coo_matrix(rawDiag)
        return nimble.data(returnType, rawCoo, pointNames=pointNames,
                           featureNames=featureNames, name=name,  useLog=False)
    else:
        raw = numpy.identity(size)
        return nimble.data(returnType, raw, pointNames=pointNames,
                           featureNames=featureNames, name=name, useLog=False)


def loadData(inputPath, useLog=None):
    """
    Load nimble Base object.

    Parameters
    ----------
    inputPath : str
        The location (including file name and extension) to find a file
        previously generated by nimble.core.data .save(). Expected file
        extension '.nimd'.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.

    Returns
    -------
    nimble.core.data.Base
        Subclass of Base object.
    """
    if not cloudpickle.nimbleAccessible():
        msg = "To load nimble objects, cloudpickle must be installed"
        raise PackageException(msg)
    if not inputPath.endswith('.nimd'):
        msg = 'file extension for a saved nimble data object should be .nimd'
        raise InvalidArgumentValue(msg)
    with open(inputPath, 'rb') as file:
        ret = cloudpickle.load(file)
    if not isinstance(ret, nimble.core.data.Base):
        msg = 'File does not contain a valid nimble data object.'
        raise InvalidArgumentType(msg)

    handleLogging(useLog, 'load', ret.getTypeString(), len(ret.points),
                  len(ret.features), ret.name, inputPath)
    return ret


def loadTrainedLearner(inputPath, useLog=None):
    """
    Load nimble trainedLearner object.

    Parameters
    ----------
    inputPath : str
        The location (including file name and extension) to find a file
        previously generated for a trainedLearner object. Expected file
        extension '.nimm'.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.

    Returns
    -------
    TrainedLearner

    See Also
    --------
    interfaces.universal_interface.TrainedLearner
    """
    if not cloudpickle.nimbleAccessible():
        msg = "To load nimble models, cloudpickle must be installed"
        raise PackageException(msg)
    if not inputPath.endswith('.nimm'):
        msg = 'File extension for a saved nimble model should be .nimm'
        raise InvalidArgumentValue(msg)
    with open(inputPath, 'rb') as file:
        ret = cloudpickle.load(file)
    if not isinstance(ret,
                      nimble.core.interfaces.universal_interface.TrainedLearner):
        msg = 'File does not contain a nimble valid trainedLearner Object.'
        raise InvalidArgumentType(msg)

    handleLogging(useLog, 'load', "TrainedLearner",
                  learnerName=ret.learnerName, learnerArgs=ret.arguments)
    return ret
