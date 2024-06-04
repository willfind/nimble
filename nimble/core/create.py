
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
Module the user-facing data creation functions for the top level
nimble import.
"""
import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble.core.logger import handleLogging
from nimble._utility import scipy
from nimble.core._createHelpers import validateReturnType
from nimble.core._createHelpers import isAllowedRaw
from nimble.core._createHelpers import initDataObject
from nimble.core._createHelpers import createDataFromFile
from nimble.core._createHelpers import createConstantHelper
from nimble.core._createHelpers import looksFileLike
from nimble.core._createHelpers import fileFetcher
from nimble.core._createHelpers import DEFAULT_MISSING

def data(source, pointNames='automatic', featureNames='automatic',
         returnType=None, name=None, convertToType=None, keepPoints='all',
         keepFeatures='all', treatAsMissing=DEFAULT_MISSING,
         replaceMissingWith=np.nan, rowsArePoints=True,
         ignoreNonNumericalFeatures=False, inputSeparator='automatic',
         copyData=True, *, useLog=None):
    """
    Function to instantiate one of the Nimble data container types.

    All Nimble data objects offer the same methods but offer unique
    implementations based on how each stores data on the backend. This
    creates consistency regardless of ``returnType``, but efficiencies
    can be gained by choosing the best ``returnType`` for the ``source``
    data. For example, highly sparse data would benefit most from
    choosing ``returnType='Sparse'``. Nimble data objects are also
    consistent in that they all allow for point and feature names.
    Additional parameters allow some data preprocessing to be performed
    during object creation.

    Parameters
    ----------
    source : object, str
        The source of the data to be loaded into the returned object.

        * in-python data object - list, dictionary, numpy array or
          matrix, scipy sparse matrix, pandas dataframe or series.
        * open file-like object
        * str - A path or url to the data file.
    pointNames : 'automatic', bool, list, dict
        Specifies the source for point names in the returned object.

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
        Specifies the source for feature names in the returned object.

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
    returnType : str, None
        Indicates which Nimble data object to return. Options are the
        **case sensitive** strings "List", "Matrix", "Sparse" and
        "DataFrame". If None, Nimble will detect the most appropriate
        type from the data and/or packages available in the environment.
    name : str, None
        A string describing the data that will display when printing or
        logging the returned object. This value is also set as the name
        attribute of the returned object.
    convertToType : type, dict, list, None
        A one-time conversion of features to the provided type or types.
        By default, object types within ``source`` are not modified,
        except for features detected to be numeric in a data file.
        Setting this parameter to a single type will convert all of the
        data to that type. For feature-by-feature type setting, a dict
        or list can be used. Dicts map feature identifiers (names and
        indexes) to conversion types. Any feature not included in the
        dict will remain as-is. A list must provide a type or None for
        each feature. Note: The setting of types only applies during the
        creation process, object methods will modify types if necessary.
    keepPoints : 'all', list
        Allows the user to select which points will be kept in the
        returned object, those not selected will be discarded. By
        default, the value 'all' indicates that all possible points in
        the data will be kept. Alternatively, the user may provide a
        list containing either names or indices (or a mix) of those
        points they want to be kept from the data. The order of this
        list will determine the order of points in the resultant object.
        In the case of reading data from a file, the selection will be
        done at read time, thus limiting the amount of data read into
        memory.
    keepFeatures : 'all', list
        Allows the user to select which features will be kept in the
        returned object, those not selected will be discarded. By
        default, the value 'all' indicates that all possible features in
        the data will be kept. Alternatively, the user may provide a
        list containing either names or indices (or a mix) of those
        features they want to be kept from the data. The order of this
        list will determine the order of features in the resultant
        object. In the case of reading data from a file, the selection
        will be done at read time, thus limiting the amount of data read
        into memory. Additionally, ``ignoreNonNumericalFeatures`` takes
        precedent and, when set to True, will remove features included
        in this selection if they contain non-numeric values.
    treatAsMissing : list
        Values that will be treated as missing values in the data. These
        values will be replaced with value from ``replaceMissingWith``
        By default this list is [float('nan'), np.nan, None, '',
        'None', 'nan']. Set to None or [] to disable replacing missing
        values.
    replaceMissingWith
        A single value with which to replace any value in
        ``treatAsMissing``. By default this value is np.nan.
    rowsArePoints : bool
        For ``source`` objects with a ``shape`` attribute, shape[0] indicates
        the rows. Otherwise, rows are defined as the objects returned when
        iterating through ``source``. For one-dimensional rows, each row in
        is processed as a point if True, otherwise rows are processed as
        features. Two-dimensional rows will also be treated as one-dimensional
        if this parameter is False and the row has a feature vector shape (e.g.
        a list of 5 x 1 numpy arrays). This parameter must be True for any
        higher dimension rows.
    ignoreNonNumericalFeatures : bool
        **This only applies when ``source`` is a file.**
        Indicate whether features containing non-numeric data should not
        be loaded into the final object. If there is point or feature
        selection occurring, then only those values within selected
        points and features are considered when determining whether to
        apply this operation.
    inputSeparator : str
        **This only applies when ``source`` is a delimited file.**
        The character that is used to separate fields in the input file,
        if necessary. By default, a value of 'automatic' will attempt to
        determine the appropriate separator. Otherwise, a single
        character string of the separator in the file can be passed.
    copyData : bool
        **This only applies when ``source`` is an in-python data type.**
        When True (the default) the backend data container is guaranteed
        to be a different object than ``source`` because a copy is made
        before processing the data. When False, the initial copy is not
        performed so it is possible (NOT guaranteed) that the ``source``
        data object is used as the backend data container for the
        returned object. In that case, any modifications to either
        object would affect the other object.
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
    >>> asList = nimble.data(data, returnType="List", name='simple')
    >>> asList
    <List "simple" 2pt x 3ft
         0  1  2
       ┌────────
     0 │ 1  2  3
     1 │ 4  5  6
    >

    Loading data from a file.

    >>> with open('simpleData.csv', 'w') as cd:
    ...     out = cd.write('1,2,3\\n4,5,6')
    >>> fromFile = nimble.data('simpleData.csv')
    >>> fromFile # doctest: +ELLIPSIS
    <Matrix 2pt x 3ft
         0  1  2
       ┌────────
     0 │ 1  2  3
     1 │ 4  5  6
    >

    Adding point and feature names.

    >>> data = [['a', 'b', 'c'], [0, 0, 1], [1, 0, 0]]
    >>> asSparse = nimble.data(data, pointNames=['1', '2'],
    ...                        featureNames=True, returnType="Sparse")
    >>> asSparse
    <Sparse 2pt x 3ft
         a  b  c
       ┌────────
     1 │ 0  0  1
     2 │ 1  0  0
    >

    Replacing missing values.

    >>> data = [[1, 'Missing', 3], [4, 'Missing', 6]]
    >>> ftNames = {'a': 0, 'b': 1, 'c': 2}
    >>> asDataFrame = nimble.data(data, featureNames=ftNames,
    ...                           returnType="DataFrame",
    ...                           treatAsMissing=["Missing", 3],
    ...                           replaceMissingWith=-1)
    >>> asDataFrame
    <DataFrame 2pt x 3ft
         a  b   c
       ┌──────────
     0 │ 1  -1  -1
     1 │ 4  -1   6
    >

    Keywords
    --------
    create, make, construct, new, matrix, read, load, open, file,
    in-python data object, list, dictionary, numpy array, pandas
    dataframe, scipy sparse, csv, mtx, hdf5, h5, url, pickle
    """
    validateReturnType(returnType)

    # None is an acceptable value for copyData in initDataObject but that
    # is reserved for internal use
    if copyData not in [True, False]:
        raise InvalidArgumentValue('copyData must be True or False')

    # input is raw data
    if isAllowedRaw(source):
        ret = initDataObject(
            rawData=source, pointNames=pointNames, featureNames=featureNames,
            returnType=returnType, name=name, convertToType=convertToType,
            keepPoints=keepPoints, keepFeatures=keepFeatures,
            treatAsMissing=treatAsMissing,
            replaceMissingWith=replaceMissingWith, rowsArePoints=rowsArePoints,
            copyData=copyData)
    # input is an open file or a path to a file
    elif isinstance(source, str) or looksFileLike(source):
        ret = createDataFromFile(
            source=source, pointNames=pointNames, featureNames=featureNames,
            returnType=returnType, name=name, convertToType=convertToType,
            keepPoints=keepPoints, keepFeatures=keepFeatures,
            treatAsMissing=treatAsMissing,
            replaceMissingWith=replaceMissingWith, rowsArePoints=rowsArePoints,
            ignoreNonNumericalFeatures=ignoreNonNumericalFeatures,
            inputSeparator=inputSeparator)
    # no other allowed inputs
    else:
        msg = "source must contain either raw data or the path to a file to "
        msg += "be loaded"
        raise InvalidArgumentType(msg)

    handleLogging(useLog, 'load', ret, returnType=returnType)
    return ret


def ones(numPoints, numFeatures, pointNames='automatic',
         featureNames='automatic', returnType="Matrix", name=None):
    """
    Return a data object of the given shape containing all 1 values.

    Parameters
    ----------
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
    returnType : str
        May be any of the allowed types specified in
        nimble.core.data.available. Default is "Matrix".
    name : str, None
        A string describing the data that will display when printing or
        logging the returned object. This value is also set as the name
        attribute of the returned object.

    Returns
    -------
    nimble.core.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    zeros, identity

    Examples
    --------
    Ones with default names.

    >>> ones = nimble.ones(5, 5, returnType='List')
    >>> ones
    <List 5pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 1  1  1  1  1
     1 │ 1  1  1  1  1
     2 │ 1  1  1  1  1
     3 │ 1  1  1  1  1
     4 │ 1  1  1  1  1
    >

    Named object of ones with pointNames and featureNames.

    >>> onesDF = nimble.ones(4, 4, pointNames=['1', '2', '3', '4'],
    ...                      featureNames=['a', 'b', 'c', 'd'],
    ...                      returnType="DataFrame",
    ...                      name='ones DataFrame')
    >>> onesDF
    <DataFrame "ones DataFrame" 4pt x 4ft
         a  b  c  d
       ┌───────────
     1 │ 1  1  1  1
     2 │ 1  1  1  1
     3 │ 1  1  1  1
     4 │ 1  1  1  1
    >

    Keywords
    --------
    matrix, 1, 1s
    """
    return createConstantHelper(np.ones, returnType, numPoints, numFeatures,
                                pointNames, featureNames, name)


def zeros(numPoints, numFeatures, pointNames='automatic',
          featureNames='automatic', returnType="Matrix", name=None):
    """
    Return a data object of the given shape containing all 0 values.

    Parameters
    ----------
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
    returnType : str
        May be any of the allowed types specified in
        nimble.core.data.available. Default is "Matrix".
    name : str, None
        A string describing the data that will display when printing or
        logging the returned object. This value is also set as the name
        attribute of the returned object.

    Returns
    -------
    nimble.core.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    ones, identity

    Examples
    --------
    Zeros with default names.

    >>> zeros = nimble.zeros(5, 5)
    >>> zeros
    <Matrix 5pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 0  0  0  0  0
     1 │ 0  0  0  0  0
     2 │ 0  0  0  0  0
     3 │ 0  0  0  0  0
     4 │ 0  0  0  0  0
    >

    Named object of zeros with pointNames and featureNames.

    >>> zerosSparse = nimble.zeros(4, 4, returnType="Sparse",
    ...                            pointNames=['1', '2', '3', '4'],
    ...                            featureNames=['a', 'b', 'c', 'd'],
    ...                            name='Sparse all-zeros')
    >>> zerosSparse
    <Sparse "Sparse all-zeros" 4pt x 4ft
         a  b  c  d
       ┌───────────
     1 │ 0  0  0  0
     2 │ 0  0  0  0
     3 │ 0  0  0  0
     4 │ 0  0  0  0
    >

    Keywords
    --------
    matrix, sparse, 0, 0s
    """
    return createConstantHelper(np.zeros, returnType, numPoints, numFeatures,
                                pointNames, featureNames, name)


def identity(size, pointNames='automatic', featureNames='automatic',
             returnType="Matrix", name=None):
    """
    Return a data object representing an identity matrix.

    The returned object will always be a square with the number of
    points and features equal to ``size``.  The main diagonal will have
    values of 1 and every other value will be zero.

    Parameters
    ----------
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
    returnType : str
        May be any of the allowed types specified in
        nimble.core.data.available. Default is "Matrix".
    name : str, None
        A string describing the data that will display when printing or
        logging the returned object. This value is also set as the name
        attribute of the returned object.

    Returns
    -------
    nimble.core.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    ones, zeros

    Examples
    --------
    Identity matrix with default names.

    >>> identity = nimble.identity(5)
    >>> identity
    <Matrix 5pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 1  0  0  0  0
     1 │ 0  1  0  0  0
     2 │ 0  0  1  0  0
     3 │ 0  0  0  1  0
     4 │ 0  0  0  0  1
    >

    Named object of zeros with pointNames and featureNames.

    >>> identityList = nimble.identity(3, returnType="List",
    ...                                pointNames=['1', '2', '3'],
    ...                                featureNames=['a', 'b', 'c'],
    ...                                name='identity matrix list')
    >>> identityList
    <List "identity matrix list" 3pt x 3ft
         a  b  c
       ┌────────
     1 │ 1  0  0
     2 │ 0  1  0
     3 │ 0  0  1
    >

    Keywords
    --------
    identity matrix, square, diagonal, one, eye
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

        rawDiag = scipy.sparse.identity(size, dtype=int)
        rawCoo = scipy.sparse.coo_matrix(rawDiag)
        return nimble.data(rawCoo, pointNames=pointNames,
                           featureNames=featureNames, returnType=returnType,
                           name=name, useLog=False)

    raw = np.identity(size, dtype=int)
    return nimble.data(raw, pointNames=pointNames, featureNames=featureNames,
                       returnType=returnType, name=name, useLog=False)

def fetchFiles(source, overwrite=False):
    """
    Get data files from the web or local storage.

    Downloads new data files from the web and stores them in a
    "nimbleData" directory placed in a configurable location (see next
    paragraph). Once stored, any subsequent calls to fetch the same
    data will identify that the data is already available locally,
    avoiding repeated downloads. For zip and tar files, extraction
    will be attempted. If successful, the returned list paths will
    include the extracted files, otherwise it will include the archive
    file.

    The location to place the "nimbleData" directory is configurable
    through nimble.settings by setting the "location" option in the
    "fetch" section. By default, the location is the home directory
    (pathlib.Path.home()). The file path within "nimbleData" matches the
    the download url, except for files extracted from zip and tar files.

    Special support for the UCI repository is included. The ``source``
    can be 'uci::<Name of Dataset>' or the url to the main page for a
    specific dataset.

    Parameters
    ----------
    source : str
        Downloadable url or valid string to UCI database (see above).
    overwrite : bool
        If True, will overwrite any files stored locally with the data
        currently available from the source.

    Returns
    -------
    list
        The paths to the available files.

    See Also
    --------
    fetchFile, data

    Examples
    --------
    A single dataset from a downloadable url.

    >>> url = 'https://openml.org/data/get_csv/16826755/phpMYEkMl'
    >>> titanic = nimble.fetchFiles(url) # doctest: +SKIP

    Replacing the path to the root storage location with an ellipsis and
    using a Unix operating system, the ``titanic`` return is
    ``['.../nimbleData/openml.org/data/get_csv/16826755/phpMYEkMl']``.
    Note how the directory structure mirrors the url.

    For the UCI database, two additional options are available. A string
    starting with 'uci:' followed by the name of a UCI dataset or the
    url to the main page of the dataset.

    >>> iris = nimble.fetchFiles('uci::Iris') # doctest: +SKIP
    >>> url = 'https://archive.ics.uci.edu/ml/datasets/Wine+Quality'
    >>> wineQuality = fetchFiles(url) # doctest: +SKIP

    Keywords
    --------
    get, download, local, store, files, url, obtain, retrieve, get,
    open, create, folder
    """
    return fileFetcher(source, overwrite)
