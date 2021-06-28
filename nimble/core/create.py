"""
Module the user-facing data creation functions for the top level
nimble import.
"""

import numpy as np

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
from nimble.core._createHelpers import looksFileLike
from nimble.core._createHelpers import fileFetcher
from nimble.core._createHelpers import DEFAULT_MISSING


def data(returnType, source, pointNames='automatic', featureNames='automatic',
         name=None, convertToType=None, keepPoints='all', keepFeatures='all',
         treatAsMissing=DEFAULT_MISSING, replaceMissingWith=np.nan,
         rowsArePoints=True, ignoreNonNumericalFeatures=False,
         inputSeparator='automatic', copyData=True, useLog=None):
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
    returnType : str, None
        Indicates which Nimble data object to return. Options are the
        **case sensitive** strings "List", "Matrix", "Sparse" and
        "DataFrame". If None is given, Nimble will attempt to detect the
        type most appropriate for the data.
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
    name : str
        When not None, this value is set as the name attribute of the
        returned object.
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

    # None is an acceptable value for copyData in initDataObject but that
    # is reserved for internal use
    if copyData not in [True, False]:
        raise InvalidArgumentValue('copyData must be True or False')

    # input is raw data
    if isAllowedRaw(source):
        ret = initDataObject(
            returnType=returnType, rawData=source, pointNames=pointNames,
            featureNames=featureNames, name=name, convertToType=convertToType,
            keepPoints=keepPoints, keepFeatures=keepFeatures,
            treatAsMissing=treatAsMissing,
            replaceMissingWith=replaceMissingWith, rowsArePoints=rowsArePoints,
            copyData=copyData)
    # input is an open file or a path to a file
    elif isinstance(source, str) or looksFileLike(source):
        ret = createDataFromFile(
            returnType=returnType, source=source, pointNames=pointNames,
            featureNames=featureNames, name=name, convertToType=convertToType,
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

    handleLogging(useLog, 'load', returnType, len(ret.points),
                  len(ret.features), ret.name, ret.path)
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
    return createConstantHelper(np.ones, returnType, numPoints, numFeatures,
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
        [[0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000]]
        pointNames={'1':0, '2':1, '3':2, '4':3}
        featureNames={'a':0, 'b':1, 'c':2, 'd':3}
        name="Sparse all-zeros"
        )
    """
    return createConstantHelper(np.zeros, returnType, numPoints, numFeatures,
                                pointNames, featureNames, name)


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
                           featureNames=featureNames, name=name, useLog=False)

    raw = np.identity(size)
    return nimble.data(returnType, raw, pointNames=pointNames,
                       featureNames=featureNames, name=name, useLog=False)


def loadData(inputPath, useLog=None):
    """
    Load nimble Base object.

    Parameters
    ----------
    inputPath : str
        The location (including file name and extension) to find a file
        previously generated by nimble.core.data.save(). Expected file
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
    nimble.core.interfaces.TrainedLearner
    """
    if not cloudpickle.nimbleAccessible():
        msg = "To load nimble models, cloudpickle must be installed"
        raise PackageException(msg)
    if not inputPath.endswith('.nimm'):
        msg = 'File extension for a saved nimble model should be .nimm'
        raise InvalidArgumentValue(msg)
    with open(inputPath, 'rb') as file:
        ret = cloudpickle.load(file)
    if not isinstance(ret, nimble.core.interfaces.TrainedLearner):
        msg = 'File does not contain a valid Nimble TrainedLearner object.'
        raise InvalidArgumentType(msg)

    handleLogging(useLog, 'load', "TrainedLearner",
                  learnerName=ret.learnerName, learnerArgs=ret.arguments)
    return ret

def fetchFile(source, overwrite=False):
    """
    Get a data file from the web or local storage.

    Downloads a new data file from the web and stores it in a
    "nimbleData" directory placed in a configurable location (see next
    paragraph). Once stored, any subsequent calls to fetch the same
    data will identify that the data is already available locally,
    avoiding repeated downloads. For zip and tar files, extraction
    will be attempted. If successful, the path to the extracted file
    will be returned, otherwise the path to the archive file is
    returned.

    The location to place the "nimbleData" directory is configurable
    through nimble.settings by setting the "location" option in the
    "fetch" section. By default, the location is the home directory
    (pathlib.Path.home()). The file path within "nimbleData" matches the
    the download url, except for files extracted from zip and tar files.

    Special support for the UCI repository is included. The ``source``
    can be ``'uci::<Name of Dataset>'`` or the url to the main page for
    a specific dataset. This function requires that the UCI repository
    contain only a single file. An exception is made if all other files
    in the repository are named 'Index' or end in '.names'. By
    convention, these files are expected to contain information about
    the dataset not data suited for Nimble data objects. All files
    are downloaded and stored but only the path to the data file is
    returned.

    Parameters
    ----------
    source : str
        Downloadable url or valid string to UCI database (see above).
    overwrite : bool
        If True, will overwrite the file stored locally with the data
        currently available from the source.

    Returns
    -------
    str
        The path to the available file.

    See Also
    --------
    fetchFiles

    Examples
    --------
    A downloadable url.

    >>> url = 'https://openml.org/data/get_csv/16826755/phpMYEkMl'
    >>> titanic = nimble.fetchFile(url) # doctest: +SKIP

    Replacing the path to the root storage location with an ellipsis and
    using a Unix operating system, the ``titanic`` return looks like:
    ``'.../nimbleData/openml.org/data/get_csv/16826755/phpMYEkMl'``.
    Note how the directory structure mirrors the url.

    For the UCI database, two additional options are available. A string
    starting with 'uci:' followed by the name of a UCI dataset or the
    url to the main page of the dataset.

    >>> wine = nimble.fetchFile('uci::wine') # doctest: +SKIP
    >>> url = 'https://archive.ics.uci.edu/ml/datasets/Abalone'
    >>> abalone = nimble.fetchFile(url) # doctest: +SKIP
    """
    return fileFetcher(source, overwrite, allowMultiple=False)[0]

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
    fetchFile

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
    """
    return fileFetcher(source, overwrite)
