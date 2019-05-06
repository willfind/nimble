"""
Module containing most of the user facing functions for the top level
uml import.
"""

from __future__ import absolute_import
import copy
import six.moves.configparser

import numpy
import six
from six.moves import range
from six.moves import zip

import UML
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from UML.exceptions import InvalidArgumentValueCombination, PackageException
from UML.logger import handleLogging, startTimer, stopTimer
from UML.logger import stringToDatetime
from UML.helpers import findBestInterface
from UML.helpers import _learnerQuery
from UML.helpers import _validScoreMode
from UML.helpers import _validMultiClassStrategy
from UML.helpers import _unpackLearnerName
from UML.helpers import _validArguments
from UML.helpers import _validData
from UML.helpers import _2dOutputFlagCheck
from UML.helpers import LearnerInspector
from UML.helpers import ArgumentIterator
from UML.helpers import _mergeArguments
from UML.helpers import crossValidateBackend
from UML.helpers import isAllowedRaw
from UML.helpers import initDataObject
from UML.helpers import createDataFromFile
from UML.helpers import createConstantHelper
from UML.helpers import computeMetrics
from UML.randomness import numpyRandom, generateSubsidiarySeed
from UML.randomness import startAlternateControl, endAlternateControl
from UML.calculate import detectBestResult

cloudpickle = UML.importModule('cloudpickle')
scipy = UML.importModule('scipy.sparse')


def createRandomData(
        returnType, numPoints, numFeatures, sparsity, pointNames='automatic',
        featureNames='automatic', elementType='float', name=None, useLog=None):
    """
    Generate a data object with random contents.

    The range of values is dependent on the elementType.

    Parameters
    ----------
    returnType : str
        May be any of the allowed types specified in UML.data.available.
    numPoints : int
        The number of points in the returned object.
    numFeatures : int
        The number of features in the returned object.
    sparsity : float
        The likelihood that the value of a (point,feature) pair is zero.
    elementType : str
        If 'float' (default) then the value of (point, feature) pairs
        are sampled from a normal distribution (location 0, scale 1). If
        elementType is 'int' then value of (point, feature) pairs are
        sampled from uniform integer distribution [1 100]. Zeros are not
        counted in/do not affect the aforementioned sampling
        distribution.
    pointNames : 'automatic', list, dict
        Names to be associated with the points in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by a list-like or dict-like object, so
        long as all points in the data are assigned a name and the names
        for each point are unique.
    featureNames : 'automatic', list, dict
        Names to be associated with the features in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by a list-like or dict-like object, so
        long as all features in the data are assigned a name and the
        names for each feature are unique.
    name : str
        When not None, this value is set as the name attribute of the
        returned object.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.

    Returns
    -------
    UML.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    createData

    Examples
    --------
    Random integers.

    >>> ptNames = ['a', 'b', 'c', 'd', 'e']
    >>> random = UML.createRandomData('Matrix', 5, 5, 0,
    ...                               pointNames=ptNames,
    ...                               elementType='int')
    >>> random
    Matrix(
        [[52.000 93.000 15.000 72.000 61.000]
         [21.000 83.000 87.000 75.000 75.000]
         [88.000 24.000 3.000  22.000 53.000]
         [2.000  88.000 30.000 38.000 2.000 ]
         [64.000 60.000 21.000 33.000 76.000]]
        pointNames={'a':0, 'b':1, 'c':2, 'd':3, 'e':4}
        )

    Random floats, high sparsity.

    >>> sparse = UML.createRandomData('Sparse', 5, 5, .9)
    >>> sparse
    Sparse(
        [[  0    0 0   0   0]
         [  0    0 0   0   0]
         [-0.636 0 0   0   0]
         [  0    0 0 0.291 0]
         [  0    0 0   0   0]]
        )
    """
    if numPoints < 1:
        msg = "must specify a positive nonzero number of points"
        raise InvalidArgumentValue(msg)
    if numFeatures < 1:
        msg = "must specify a positive nonzero number of features"
        raise InvalidArgumentValue(msg)
    if sparsity < 0 or sparsity >= 1:
        msg = "sparsity must be greater than zero and less than one"
        raise InvalidArgumentType(msg)
    if elementType != "int" and elementType != "float":
        raise InvalidArgumentValue("elementType may only be 'int' or 'float'")

    seed = generateSubsidiarySeed()
    startAlternateControl(seed=seed)
    #note: sparse is not stochastic sparsity, it uses rigid density measures
    if returnType.lower() == 'sparse':
        if not scipy:
            msg = "scipy is not available"
            raise PackageException(msg)

        density = 1.0 - float(sparsity)
        numNonZeroValues = int(numPoints * numFeatures * density)

        # We want to sample over positions, not point/feature indices, so
        # we consider the possible possitions as numbered in a row-major
        # order on a grid, and sample that without replacement
        gridSize = numPoints * numFeatures
        nzLocation = numpyRandom.choice(gridSize, size=numNonZeroValues,
                                        replace=False)

        # The point value is determined by counting how many groups of
        # numFeatures fit into the position number
        pointIndices = numpy.floor(nzLocation / numFeatures)
        # The feature value is determined by counting the offset from each
        # point edge.
        featureIndices = nzLocation % numFeatures

        if elementType == 'int':
            dataVector = numpyRandom.randint(low=1, high=100,
                                             size=numNonZeroValues)
        #numeric type is float; distribution is normal
        else:
            dataVector = numpyRandom.normal(0, 1, size=numNonZeroValues)

        #pointIndices and featureIndices are
        randData = scipy.sparse.coo.coo_matrix(
            (dataVector, (pointIndices, featureIndices)),
            (numPoints, numFeatures))

    # non-sparse matrices, generate matrices with sparsity characterics
    else:
        size = (numPoints, numFeatures)
        if elementType == 'int':
            filledIntMatrix = numpyRandom.randint(1, 100, size=size)
        else:
            filledFloatMatrix = numpyRandom.normal(loc=0.0, scale=1.0,
                                                   size=size)

        #if sparsity is zero
        if abs(float(sparsity) - 0.0) < 0.0000000001:
            if elementType == 'int':
                randData = filledIntMatrix
            else:
                randData = filledFloatMatrix
        else:
            binarySparsityMatrix = numpyRandom.binomial(1, 1.0 - sparsity,
                                                        size=size)

            if elementType == 'int':
                randData = binarySparsityMatrix * filledIntMatrix
            else:
                randData = binarySparsityMatrix * filledFloatMatrix
    endAlternateControl()

    handleLogging(useLog, 'load', "Random " + returnType, numPoints,
                  numFeatures, name, sparsity=sparsity, seed=seed)

    return createData(returnType, data=randData, pointNames=pointNames,
                      featureNames=featureNames, name=name, useLog=False)


def ones(returnType, numPoints, numFeatures, pointNames='automatic',
         featureNames='automatic', name=None):
    """
    Return a data object of the given shape containing all 1 values.

    Parameters
    ----------
    returnType : str
        May be any of the allowed types specified in UML.data.available.
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
    UML.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    zeros

    Examples
    --------
    Ones with default names.

    >>> ones = UML.ones('List', 5, 5)
    >>> ones
    List(
        [[1.000 1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000 1.000]
         [1.000 1.000 1.000 1.000 1.000]]
        )

    Named object of ones with pointNames and featureNames.

    >>> onesDF = UML.ones('DataFrame', 4, 4,
    ...                   pointNames=['1', '2', '3', '4'],
    ...                   featureNames=['a', 'b', 'c', 'd'],
    ...                   name='ones DataFrame')
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
        May be any of the allowed types specified in UML.data.available.
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
    UML.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    ones

    Examples
    --------
    Zeros with default names.

    >>> zeros = UML.zeros('Matrix', 5, 5)
    >>> zeros
    Matrix(
        [[0.000 0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000 0.000]
         [0.000 0.000 0.000 0.000 0.000]]
        )

    Named object of zeros with pointNames and featureNames.

    >>> zerosSparse = UML.zeros('Sparse', 4, 4,
    ...                         pointNames=['1', '2', '3', '4'],
    ...                         featureNames=['a', 'b', 'c', 'd'],
    ...                         name='Sparse all-zeros')
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
        May be any of the allowed types specified in UML.data.available.
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
    UML.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    Examples
    --------
    Identity matrix with default names.

    >>> identity = UML.identity('Matrix', 5)
    >>> identity
    Matrix(
        [[1.000 0.000 0.000 0.000 0.000]
         [0.000 1.000 0.000 0.000 0.000]
         [0.000 0.000 1.000 0.000 0.000]
         [0.000 0.000 0.000 1.000 0.000]
         [0.000 0.000 0.000 0.000 1.000]]
        )

    Named object of zeros with pointNames and featureNames.

    >>> identityList = UML.identity('List', 3,
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
    retAllowed = copy.copy(UML.data.available)
    if returnType not in retAllowed:
        msg = "returnType must be a value in " + str(retAllowed)
        raise InvalidArgumentValue(msg)

    if size <= 0:
        msg = "size must be 0 or greater, yet " + str(size)
        msg += " was given."
        raise InvalidArgumentValue(msg)

    if returnType == 'Sparse':
        if not scipy:
            msg = "scipy is not available"
            raise PackageException(msg)

        assert returnType == 'Sparse'
        rawDiag = scipy.sparse.identity(size)
        rawCoo = scipy.sparse.coo_matrix(rawDiag)
        return UML.createData(returnType, rawCoo, pointNames=pointNames,
                              featureNames=featureNames, name=name,
                              useLog=False)
    else:
        raw = numpy.identity(size)
        return UML.createData(returnType, raw, pointNames=pointNames,
                              featureNames=featureNames, name=name,
                              useLog=False)


def normalizeData(learnerName, trainX, trainY=None, testX=None, arguments=None,
                  useLog=None, **kwarguments):
    """
    Modify data according to a produced model.

    Calls on the functionality of a package to train on some data and
    then modify ``trainX`` and ``testX``(if provided) according to the
    results of the trained model.

    Parameters
    ----------
    learnerName : str
        Name of the learner to be called, in the form 'package.learner'
    trainX: UML Base object
        Data to be used for training.
    trainY: identifier, UML Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another UML Base object containing the labels that
        correspond to ``trainX``.
    testX: UML Base object
        Data to be used for testing.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application.
        Example: {'dimensions':5, 'k':5}
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. Same format as the arguments parameter.

    See Also
    --------
    UML.data.Points.normalize, UML.data.Features.normalize

    Examples
    --------
    Normalize a single data set.

    >>> data = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    >>> trainX = UML.createData("Matrix", data)
    >>> orig = trainX.copy()
    >>> UML.normalizeData('scikitlearn.PCA', trainX, n_components=2)
    >>> trainX
    Matrix(
        [[-0.216 0.713 ]
         [-1.005 -0.461]
         [1.221  -0.253]]
        )

    Normalize training and testing data.

    >>> data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    >>> trainX = UML.createData("Matrix", data1)
    >>> data2 = [[-1, 0, 5]]
    >>> testX = UML.createData("Matrix", data2)
    >>> UML.normalizeData('scikitlearn.PCA', trainX, testX=testX,
    ...                   n_components=2)
    >>> # trainX is the same as above example.
    >>> testX
    Matrix(
        [[-1.739 2.588]]
        )
    """
    timer = startTimer(useLog)
    (_, trueLearnerName) = _unpackLearnerName(learnerName)
    merged = _mergeArguments(arguments, kwarguments)

    tl = UML.train(learnerName, trainX, trainY, arguments=merged,
                   useLog=False)
    normalizedTrain = tl.apply(trainX, useLog=False)

    if normalizedTrain.getTypeString() != trainX.getTypeString():
        normalizedTrain = normalizedTrain.copyAs(trainX.getTypeString())

    if testX is not None:
        normalizedTest = tl.apply(testX, useLog=False)
        if normalizedTest.getTypeString() != testX.getTypeString():
            normalizedTest = normalizedTest.copyAs(testX.getTypeString())

    # modify references and names for trainX and testX
    trainX.referenceDataFrom(normalizedTrain, useLog=False)
    trainX.name = trainX.name + " " + trueLearnerName

    if testX is not None:
        testX.referenceDataFrom(normalizedTest, useLog=False)
        testX.name = testX.name + " " + trueLearnerName

    time = stopTimer(timer)
    handleLogging(useLog, 'run', "normalizeData", trainX, trainY, testX, None,
                  learnerName, merged, time=time)


def registerCustomLearnerAsDefault(customPackageName, learnerClassObject):
    """
    Permanently add a customLearner to be made available to UML.

    Register the given customLearner class so that it is callable by the
    top level UML functions through the interface of the specified
    custom package. This operation modifies the saved configuration file
    so that this change will be reflected during future sesssions.

    Parameters
    ----------
    customPackageName : str
        The str name of the package preface you want to use when calling
        the learner. If there is already an interface for a custom
        package with this name, the learner will be accessible through
        that interface. If there is no interface to a custom package of
        that name, then one will be created. You cannot register a
        custom learner to be callable through the interface for a
        non-custom package (such as ScikitLearn or MLPY). Therefore,
        customPackageName cannot be a value which is the accepted alias
        of another package's interface.
    learnerClassObject : class
        The class object implementing the learner you want registered.
        It will be checked using
        UML.interfaces.CustomLearner.validateSubclass to ensure that all
        details of the provided implementation are acceptable.
    """
    UML.helpers.registerCustomLearnerBackend(customPackageName,
                                             learnerClassObject, True)


def registerCustomLearner(customPackageName, learnerClassObject):
    """
    Add a customLearner to be made available to UML for this session.

    Register the given customLearner class so that it is callable by the
    top level UML functions through the interface of the specified
    custom package. Though this operation by itself is temporary, it has
    effects in UML.settings, so subsequent saveChanges operations may
    cause it to be reflected in future sessions.

    Parameters
    ----------
    customPackageName : str
        The str name of the package preface you want to use when calling
        the learner. If there is already an interface for a custom
        package with this name, the learner will be accessible through
        that interface. If there is no interface to a custom package of
        that name, then one will be created. You cannot register a
        custom learner to be callable through the interface for a
        non-custom package (such as ScikitLearn or MLPY). Therefore,
        customPackageName cannot be a value which is the accepted alias
        of another package's interface.
    learnerClassObject : class
        The class object implementing the learner you want registered.
        It will be checked using
        UML.interfaces.CustomLearner.validateSubclass to ensure that all
        details of the provided implementation are acceptable.
    """
    UML.helpers.registerCustomLearnerBackend(customPackageName,
                                             learnerClassObject, False)


def deregisterCustomLearnerAsDefault(customPackageName, learnerName):
    """
    Permanently disable a customLearner from being available to UML.

    Remove accessibility of the learner with the given name from the
    interface of the package with the given name permenantly. This
    operation modifies the saved configuration file so that this
    change will be reflected during future sesssions.

    Parameters
    ----------
    customPackageName : str
        The name of the interface / custom package from which the
        learner named 'learnerName' is to be removed from. If that
        learner was the last one grouped in that custom package, then
        the interface is removed from the UML.interfaces.available list.
    learnerName : str
        The name of the learner to be removed from the
        interface / custom package with the name 'customPackageName'.
    """
    UML.helpers.deregisterCustomLearnerBackend(customPackageName,
                                               learnerName, True)


def deregisterCustomLearner(customPackageName, learnerName):
    """
    Temporarily disable a customLearner from being available to UML.

    Remove accessibility of the learner with the given name from the
    interface of the package with the given name temporarily in this
    session. This has effects in UML.settings, so subsequent saveChanges
    operations may cause it to be reflected in future sessions.

    Parameters
    ----------
    customPackageName : str
        The name of the interface / custom package from which the
        learner named 'learnerName' is to be removed from. If that
        learner was the last one grouped in that custom package, then
        the interface is removed from the UML.interfaces.available list.
    learnerName : str
        The name of the learner to be removed from the
        interface / custom package with the name 'customPackageName'.
    """
    UML.helpers.deregisterCustomLearnerBackend(customPackageName,
                                               learnerName, False)


def learnerParameters(name):
    """
    Get a list of parameters for the learner.

    Returns a list of strings which are the names of the parameters when
    calling this learner. If the name cannot be found within the
    package, then an exception will be thrown. If the name is found, be
    for some reason we cannot determine what the parameters are, then we
    return None. Note that if we have determined that there are no
    parameters, we return an empty list.

    Parameters
    ----------
    name : str
        Package and name in the form 'package.learnerName'.

    Returns
    -------
    list
    """
    return _learnerQuery(name, 'parameters')


def learnerDefaultValues(name):
    """
    Get a dictionary mapping parameter names to their default values.

    Returns a dictionary with strings of the parameter names as keys and
    the parameter's default value as values. If the name cannot be found
    within the package, then an exception will be thrown. If the name is
    found, be for some reason we cannot determine what the parameters
    are, then we return None. Note that if we have determined that there
    are no parameters, we return an empty dict.

    Parameters
    ----------
    name : str
        Package and name in the form 'package.learnerName'.

    Returns
    -------
    dict
    """
    return _learnerQuery(name, 'defaults')


def listLearners(package=None):
    """
    Get a list of learners avaliable to UML or a specific package.

    Returns a list a list of learners that are callable through UML's
    training, applying, and testing functions. If ``package`` is
    specified, the list will contain strings of each learner. If
    ``package`` is None, the list will contain strings in the form of
    'package.learner'. This will differ depending on the packages
    currently available in UML.interfaces.available.

    Parameters
    ----------
    package : str
        The name of the package to list the learners from. If None, each
        learners available to each interface will be listed.

    Returns
    -------
    list
    """
    results = []
    if package is None:
        for interface in UML.interfaces.available:
            packageName = interface.getCanonicalName()
            currResults = interface.listLearners()
            for learnerName in currResults:
                results.append(packageName + "." + learnerName)
    else:
        interface = findBestInterface(package)
        currResults = interface.listLearners()
        for learnerName in currResults:
            results.append(learnerName)

    return results


def createData(
        returnType, data, pointNames='automatic', featureNames='automatic',
        elementType=None, name=None, path=None, keepPoints='all',
        keepFeatures='all', ignoreNonNumericalFeatures=False,
        reuseData=False, inputSeparator='automatic',
        treatAsMissing=(float('nan'), numpy.nan, None, '', 'None', 'nan',
                        'NULL', 'NA'),
        replaceMissingWith=numpy.nan, useLog=None):
    """
    Function to instantiate one of the UML data container types.

    Creates a UML data object based on the ``returnType``.  Data can be
    loaded in a raw form, from a file or from a web page.  Some
    preprocessing of the data can also be done when creating the object
    through the passing various arguments.

    Parameters
    ----------
    returnType : str, None
        Indicates which kind of UML data type to return. Currently
        accepted are the strings "List", "Matrix", "Sparse" and
        "DataFrame" -- which are **case sensitive**. If None is given,
        UML will attempt to detect the type most appropriate for the
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
          data within the first column. A value of False indicates that
          names are not embedded and that default names should be used.
        * list, dict - all features in the data must be assigned a name
          and the names for each feature must be unique. As a list, the
          index of the name will define the feature index. As a dict,
          the value mapped to each name will define the feature index.
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
                                 'None', 'nan'].
        Set to None or [] to disable replacing missing values.
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
    UML.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    createRandomData, ones, zeros, identity

    Examples
    --------
    >>> data = [[1, 2, 3], [4, 5, 6]]
    >>> asList = UML.createData('List', data, name='simple')
    >>> asList
    List(
        [[1.000 2.000 3.000]
         [4.000 5.000 6.000]]
        name="simple"
        )

    Loading data from a file.

    >>> with open('createData.csv', 'w') as cd:
    ...     out = cd.write('1,2,3\\n4,5,6')
    >>> fromFile = UML.createData('Matrix', 'createData.csv')
    >>> fromFile
    Matrix(
        [[1.000 2.000 3.000]
         [4.000 5.000 6.000]]
        name="createData.csv"
        path="/UML/createData.csv"
        )

    Adding point and feature names.

    >>> data = [['a', 'b', 'c'], [0, 0, 1], [1, 0, 0]]
    >>> asSparse = UML.createData('Sparse', data, pointNames=['1', '2'],
    ...                           featureNames=True)
    >>> asSparse
    Sparse(
        [[  0   0 1.000]
         [1.000 0   0  ]]
        pointNames={'1':0, '2':1}
        featureNames={'a':0, 'b':1, 'c':2}
        )

    Replacing missing values.

    >>> data = [[1, 'Missing', 3], [4, 'Missing', 6]]
    >>> ftNames = {'a': 0, 'b': 1, 'c': 2}
    >>> asDataFrame = UML.createData('DataFrame', data,
    ...                              featureNames = ftNames,
    ...                              treatAsMissing=["Missing", 3],
    ...                              replaceMissingWith=-1)
    >>> asDataFrame
    DataFrame(
        [[1.000 -1.000 -1.000]
         [4.000 -1.000 6.000 ]]
        featureNames={'a':0, 'b':1, 'c':2}
        )
    """
    retAllowed = copy.copy(UML.data.available)
    retAllowed.append(None)
    if returnType not in retAllowed:
        msg = "returnType must be a value in " + str(retAllowed)
        raise InvalidArgumentValue(msg)

    def looksFileLike(toCheck):
        hasRead = hasattr(toCheck, 'read')
        hasWrite = hasattr(toCheck, 'write')
        return hasRead and hasWrite

    # input is raw data
    if isAllowedRaw(data, allowLPT=True):
        ret = initDataObject(
            returnType=returnType, rawData=data, pointNames=pointNames,
            featureNames=featureNames, elementType=elementType, name=name,
            path=path, keepPoints=keepPoints, keepFeatures=keepFeatures,
            reuseData=reuseData, treatAsMissing=treatAsMissing,
            replaceMissingWith=replaceMissingWith)
    # input is an open file or a path to a file
    elif isinstance(data, six.string_types) or looksFileLike(data):
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


def crossValidate(learnerName, X, Y, performanceFunction, arguments=None,
                  numFolds=10, scoreMode='label', useLog=None, **kwarguments):
    """
    Perform K-fold cross validation.

    Returns mean performance (float) across numFolds folds on a X Y.

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
        Look in UML.calculate for premade options. Function used by
        computeMetrics to generate a performance score for the run.
        function is of the form: def func(knownValues, predictedValues).
    arguments : dict
        Mapping argument names (strings) to their values. The parameter
        is sent to trainAndApply() through its arguments parameter.
        example: {'dimensions':5, 'k':5}
    numFolds : int
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
        Keyword argument specified variables that are passed to the
        learner. To make use of multiple permutations, specify different
        values for a parameter as a tuple. eg. a=(1,2,3) will generate
        an error score for the learner when the learner was passed all
        three values of a, separately.

    Returns
    -------
    TODO

    Examples
    --------
    TODO
    """
    bestResult = crossValidateReturnBest(learnerName, X, Y,
                                         performanceFunction, arguments,
                                         numFolds, scoreMode, useLog,
                                         **kwarguments)

    return bestResult[1]

#return crossValidateBackend(learnerName, X, Y, performanceFunction, arguments,
#                            numFolds, scoreMode, useLog, **kwarguments)


def crossValidateReturnAll(learnerName, X, Y, performanceFunction,
                           arguments=None, numFolds=10, scoreMode='label',
                           useLog=None, **kwarguments):
    """
    Return all results of K-fold cross validation.

    Calculates the cross validated error for each argument permutation
    that can be generated by the merge of arguments and kwarguments.
    Example kwarguments: {'a':(1,2,3), 'b':(4,5)}
    generates permutations of dict in the format:
    {'a':1, 'b':4}, {'a':2, 'b':4}, {'a':3, 'b':4}, {'a':1, 'b':5},
    {'a':2, 'b':5}, {'a':3, 'b':5}

    For each permutation of 'arguments', crossValidateReturnAll uses
    cross validation to generate a performance score for the algorithm,
    given the particular argument permutation.

    Returns a list of tuples, where every tuple contains a dict
    representing the argument sent to trainAndApply, and a float
    represennting the cross validated error associated with that
    argument dict.
    example list element: ({'arg1':2, 'arg2':'max'}, 89.0000123)

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
        Look in UML.calculate for premade options. Function used by
        computeMetrics to generate a performance score for the run.
        function is of the form: def func(knownValues, predictedValues).
    arguments : dict
        Mapping argument names (strings) to their values. The parameter
        is sent to trainAndApply() through its arguments parameter.
        example: {'dimensions':5, 'k':5}
    numFolds : int
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
        Keyword argument specified variables that are passed to the
        learner. To make use of multiple permutations, specify different
        values for a parameter as a tuple. eg. a=(1,2,3) will generate
        an error score for the learner when the learner was passed all
        three values of a, separately.

    Returns
    -------
    TODO

    Examples
    --------
    TODO
    """
    ret = crossValidateBackend(learnerName, X, Y, performanceFunction,
                               arguments, numFolds, scoreMode, useLog,
                               **kwarguments)

    return ret


def crossValidateReturnBest(learnerName, X, Y, performanceFunction,
                            arguments=None, numFolds=10, scoreMode='label',
                            useLog=None, **kwarguments):
    """
    Return the best result of K-fold cross validation.

    For each possible argument permutation generated by arguments,
    ``crossValidateReturnBest`` runs ``crossValidate`` to compute a mean
    error for the argument combination. ``crossValidateReturnBest`` then
    RETURNS the best argument and error as a tuple: (argument_as_dict,
    cross_validated_performance_float)

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
        Look in UML.calculate for premade options. Function used by
        computeMetrics to generate a performance score for the run.
        function is of the form: def func(knownValues, predictedValues).
    arguments : dict
        Mapping argument names (strings) to their values. The parameter
        is sent to trainAndApply() through its arguments parameter.
        example: {'dimensions':5, 'k':5}
    numFolds : int
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
        Keyword argument specified variables that are passed to the
        learner. To make use of multiple permutations, specify different
        values for a parameter as a tuple. eg. a=(1,2,3) will generate
        an error score for the learner when the learner was passed all
        three values of a, separately.

    Returns
    -------
    TODO

    Examples
    --------
    TODO
    """
    resultsAll = crossValidateReturnAll(learnerName, X, Y, performanceFunction,
                                        arguments, numFolds, scoreMode, useLog,
                                        **kwarguments)

    bestArgumentAndScoreTuple = None

    detected = detectBestResult(performanceFunction)
    if detected == 'max':
        maximumIsBest = True
    elif detected == 'min':
        maximumIsBest = False
    else:
        msg = "Unable to automatically determine whether maximal or "
        msg += "minimal scores are considered optimal for the the "
        msg += "given performanceFunction. "
        msg += "By adding an attribute named 'optimal' to "
        msg += "performanceFunction with either the value 'min' or 'max' "
        msg += "depending on whether minimum or maximum returned values "
        msg += "are associated with correctness, this error should be "
        msg += "avoided."

    for curResultTuple in resultsAll:
        _, curScore = curResultTuple
        #if curArgument is the first or best we've seen:
        #store its details in bestArgumentAndScoreTuple
        if bestArgumentAndScoreTuple is None:
            bestArgumentAndScoreTuple = curResultTuple
        else:
            if (maximumIsBest and curScore > bestArgumentAndScoreTuple[1]):
                bestArgumentAndScoreTuple = curResultTuple
            if not maximumIsBest and curScore < bestArgumentAndScoreTuple[1]:
                bestArgumentAndScoreTuple = curResultTuple

    return bestArgumentAndScoreTuple


def learnerType(learnerNames):
    """
    Attempt to determine learner types.

    Returns the string or list of strings representation of a best guess
    for the type of learner(s) specified by the learner name(s) in
    learnerNames. ``learnerType`` first queries the appropriate
    interface object for a definitive return value. If the interface
    doesn't provide a satisfactory answer, then this method calls a
    backend which generates a series of artificial data sets with
    particular traits to look for heuristic evidence of a classifier,
    regressor, etc.

    Parameters
    ----------
    learnerNames : str, list
        A string or a list of strings in the format 'package.learner'.

    Returns
    -------
    str, list
        string for a single learner or a list for multiple learners.
    """
    #argument checking
    if not isinstance(learnerNames, list):
        learnerNames = [learnerNames]

    resultsList = []
    secondPassLearnerNames = []
    for name in learnerNames:
        if not isinstance(name, str):
            msg = "learnerNames must be a string or a list of strings."
            raise InvalidArgumentType(msg)

        splitTuple = _unpackLearnerName(name)
        currInterface = findBestInterface(splitTuple[0])
        allValidLearnerNames = currInterface.listLearners()
        if not splitTuple[1] in allValidLearnerNames:
            msg = name + " is not a valid learner on your machine."
            raise InvalidArgumentValue(msg)
        result = currInterface.learnerType(splitTuple[1])
        if result == 'UNKNOWN' or result == 'other' or result is None:
            resultsList.append(None)
            secondPassLearnerNames.append(name)
        else:
            resultsList.append(result)
            secondPassLearnerNames.append(None)

    #have valid arguments - a list of learner names
    learnerInspectorObj = LearnerInspector()

    for index in range(len(secondPassLearnerNames)):
        curLearnerName = secondPassLearnerNames[index]
        if curLearnerName is None:
            continue
        resultsList[index] = learnerInspectorObj.learnerType(curLearnerName)

    #if only one algo was requested, return as single string
    if len(resultsList) == 1:
        resultsList = resultsList[0]

    return resultsList


def train(learnerName, trainX, trainY=None, performanceFunction=None,
          arguments=None, scoreMode='label', multiClassStrategy='default',
          numFolds=10, doneValidData=False, doneValidArguments1=False,
          doneValidArguments2=False, doneValidMultiClassStrategy=False,
          done2dOutputFlagCheck=False, useLog=None, storeLog='unset',
          **kwarguments):
    """
    Train a specified learner using the provided data.

    Parameters
    ----------
    learnerName : str
        Name of the learner to be called, in the form 'package.learner'
    trainX: UML Base object
        Data to be used for training.
    trainY: identifier, UML Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another UML Base object containing the labels that
        correspond to ``trainX``.
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in UML.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
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
    multiClassStrategy : str
        May only be 'default' 'OneVsAll' or 'OneVsOne'
    numFolds : int
        The number of folds used in the cross validation. Cannot exceed
        the number of points in ``trainX``. Default 10.
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
    UML.interfaces.universal_interface.TrainedLearner

    See Also
    --------
    trainAndApply, trainAndTest, trainAndTestOnTrainingData

    Examples
    --------
    A single dataset which contains the labels.

    >>> data = [[1, 0, 0, 1],
    ...         [0, 1, 0, 2],
    ...         [0, 0, 1, 3],
    ...         [1, 0, 0, 1],
    ...         [0, 1, 0, 2],
    ...         [0, 0, 1, 3]]
    >>> ftNames = ['a', 'b' ,'c', 'label']
    >>> trainData = UML.createData('Matrix', data, featureNames=ftNames)
    >>> tl = UML.train('Custom.KNNClassifier', trainX=trainData,
    ...                trainY='label')
    >>> print(type(tl))
    <class 'UML.interfaces.universal_interface.TrainedLearner'>

    Passing arguments to the learner. Both the arguments parameter and
    kwarguments can be utilized, they will be merged. Below, ``C`` and
    ``kernel`` are parameters for scikit-learn's SVC learner.

    >>> dataX = [[1, 0, 0],
    ...          [0, 1, 0],
    ...          [0, 0, 1],
    ...          [1, 0, 0],
    ...          [0, 1, 0],
    ...          [0, 0, 1]]
    >>> dataY = [[1], [2], [3], [1], [2], [3]]
    >>> trainX = UML.createData('Matrix', dataX)
    >>> trainY = UML.createData('Matrix', dataY)
    >>> tl = UML.train('sciKitLearn.SVC', trainX=trainX, trainY=trainY,
    ...                arguments={'C': 0.1}, kernel='linear')
    >>> tlAttributes = tl.getAttributes()
    >>> cValue = tlAttributes['C']
    >>> kernelValue = tlAttributes['kernel']
    >>> print(cValue, kernelValue)
    0.1 linear
    """
    timer = startTimer(useLog)
    (package, trueLearnerName) = _unpackLearnerName(learnerName)
    if not doneValidData:
        _validData(trainX, trainY, None, None, [False, False])
    if not doneValidArguments1:
        _validArguments(arguments)
    if not doneValidArguments2:
        _validArguments(kwarguments)
    if not doneValidMultiClassStrategy:
        _validMultiClassStrategy(multiClassStrategy)
    if not done2dOutputFlagCheck:
        _2dOutputFlagCheck(trainX, trainY, None, multiClassStrategy)

    merged = _mergeArguments(arguments, kwarguments)

    # perform CV (if needed)
    argCheck = ArgumentIterator(merged)
    if argCheck.numPermutations != 1:
        if performanceFunction is None:
            msg = "Cross validation was triggered to select the best "
            msg += "parameter set, yet no performanceFunction was specified. "
            msg += "Either one must be specified (see UML.calculate for "
            msg += "out-of-the-box options) or there must be no choices in "
            msg += "the parameters."
            raise InvalidArgumentValueCombination(msg)
        if numFolds > len(trainX.points):
            msg = "There must be a minimum of one fold per point in the data."
            msg += "Cross validation was triggered to select the best "
            msg += "parameter set but the 'numFolds' parameter was set to "
            msg += str(numFolds) + " and trainX only contains "
            msg += str(len(trainX.points)) + " points."
            raise InvalidArgumentValueCombination(msg)
        # useLog value stored if call to train is from another function
        if storeLog == 'unset':
            storeLog = useLog
        # sig (learnerName, X, Y, performanceFunction, arguments=None,
        #      numFolds=10, scoreMode='label', useLog=None, maximize=False,
        #      **kwarguments):
        results = UML.crossValidateReturnBest(learnerName, trainX, trainY,
                                              performanceFunction, merged,
                                              numFolds=numFolds,
                                              scoreMode=scoreMode,
                                              useLog=storeLog)
        bestArgument, _ = results

    else:
        bestArgument = merged

    interface = findBestInterface(package)

    trainedLearner = interface.train(trueLearnerName, trainX, trainY,
                                     multiClassStrategy, bestArgument)
    time = stopTimer(timer)

    funcString = interface.getCanonicalName() + '.' + trueLearnerName
    handleLogging(useLog, "run", "train", trainX, trainY, None, None,
                  funcString, bestArgument, time=time)

    return trainedLearner


def trainAndApply(learnerName, trainX, trainY=None, testX=None,
                  performanceFunction=None, arguments=None, output=None,
                  scoreMode='label', multiClassStrategy='default',
                  numFolds=10, useLog=None, **kwarguments):
    """
    Train a model and apply it to the test data.

    The learner will be trained using the training data, then
    prediction, transformation, etc. as appropriate to the learner will
    be applied to the test data and returned.

    Parameters
    ----------
    learnerName : str
        Name of the learner to be called, in the form 'package.learner'
    trainX: UML Base object
        Data to be used for training.
    trainY: identifier, UML Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another UML Base object containing the labels that
        correspond to ``trainX``.
    testX : UML Base object
        Data set on which the trained learner will be applied (i.e.
        performing prediction, transformation, etc. as appropriate to
        the learner).
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in UML.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To make use of multiple permutations, specify different values
        for a parameter as a tuple. eg. {'k': (1,3,5)} will generate an
        error score for  the learner when the learner was passed all
        three values of ``k``, separately. These will be merged with
        kwarguments for the learner.
    output : str
        The kind of UML Base object that the output of this function
        should be in. Any of the normal string inputs to the createData
        ``returnType`` parameter are accepted here. Alternatively, the
        value 'match' will indicate to use the type of the ``trainX``
        parameter.
    scoreMode : str
        In the case of a classifying learner, this specifies the type of
        output wanted: 'label' if we class labels are desired,
        'bestScore' if both the class label and the score associated
        with that class are desired, or 'allScores' if a matrix
        containing the scores for every class label are desired.
    multiClassStrategy : str
        May only be 'default' 'OneVsAll' or 'OneVsOne'
    numFolds : int
        The number of folds used in the cross validation. Cannot exceed
        the number of points in ``trainX``. Default 10.
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
    results
        The resulting output of applying learner.

    See Also
    --------
    train, trainAndTest, trainAndTestOnTrainingData

    Examples
    --------
    Train dataset which contains the labels.

    >>> rawTrain = [[1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3],
    ...             [1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3]]
    >>> rawTestX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> trainData = UML.createData('Matrix', rawTrain)
    >>> testX = UML.createData('Matrix', rawTestX)
    >>> predict = UML.trainAndApply('Custom.KNNClassifier',
    ...                             trainX=trainData, trainY=3,
    ...                             testX=testX)
    >>> predict
    Matrix(
        [[1.000]
         [2.000]
         [3.000]]
        )

    Passing arguments to the learner. Both the arguments parameter and
    kwarguments can be utilized, they will be merged. Below, ``C`` and
    ``kernel`` are parameters for scikit-learn's SVC learner.

    >>> rawTrainX = [[1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1],
    ...              [1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1]]
    >>> rawTrainY = [[1], [2], [3], [1], [2], [3]]
    >>> rawTestX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> trainX = UML.createData('Matrix', rawTrainX)
    >>> trainY = UML.createData('Matrix', rawTrainY)
    >>> testX = UML.createData('Matrix', rawTestX)
    >>> pred = UML.trainAndApply('sciKitLearn.SVC', trainX=trainX,
    ...                          trainY=trainY, testX=testX,
    ...                          arguments={'C': 0.1}, kernel='linear')
    >>> pred
    Matrix(
        [[1.000]
         [2.000]
         [3.000]]
        )
    """
    timer = startTimer(useLog)
    _validData(trainX, trainY, testX, None, [False, False])
    _validScoreMode(scoreMode)
    _2dOutputFlagCheck(trainX, trainY, scoreMode, multiClassStrategy)
    merged = _mergeArguments(arguments, kwarguments)

    trainedLearner = UML.train(learnerName, trainX, trainY,
                               performanceFunction, merged,
                               scoreMode='label',
                               multiClassStrategy=multiClassStrategy,
                               numFolds=numFolds, useLog=False,
                               storeLog=useLog, doneValidData=True,
                               done2dOutputFlagCheck=True, **kwarguments)

    if testX is None:
        if isinstance(trainY, (six.string_types, int, numpy.integer)):
            testX = trainX.copy()
            testX.features.delete(trainY, useLog=False)
        else:
            testX = trainX

    results = trainedLearner.apply(testX, {}, output, scoreMode, useLog=False)
    time = stopTimer(timer)

    extraInfo = None
    if merged != trainedLearner.arguments:
        extraInfo = {"bestParams": trainedLearner.arguments}
    handleLogging(useLog, "run", "trainAndApply", trainX, trainY, testX, None,
                  learnerName, merged, extraInfo=extraInfo, time=time)

    return results


def trainAndTest(learnerName, trainX, trainY, testX, testY,
                 performanceFunction, arguments=None, output=None,
                 scoreMode='label', multiClassStrategy='default',
                 numFolds=10, useLog=None, **kwarguments):
    """
    Train a model and get the results of its performance.

    For each permutation of the merge of 'arguments' and 'kwarguments'
    (more below), this function uses cross validation to generate a
    performance score for the algorithm, given the particular argument
    permutation. The argument permutation that performed best cross
    validating over the training data is then used as the lone argument
    for training on the whole training data set. Finally, the learned
    model generates predictions for the testing set, an the performance
    of those predictions is calculated and returned. If no additional
    arguments are supplied via arguments or kwarguments, then the
    result is the performance of the algorithm with default arguments on
    the testing data.

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
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in UML.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To make use of multiple permutations, specify different values
        for a parameter as a tuple. eg. {'k': (1,3,5)} will generate an
        error score for  the learner when the learner was passed all
        three values of ``k``, separately. These will be merged with
        kwarguments for the learner.
    output : str
        The kind of UML Base object that the output of this function
        should be in. Any of the normal string inputs to the createData
        ``returnType`` parameter are accepted here. Alternatively, the
        value 'match' will indicate to use the type of the ``trainX``
        parameter.
    scoreMode : str
        In the case of a classifying learner, this specifies the type of
        output wanted: 'label' if we class labels are desired,
        'bestScore' if both the class label and the score associated
        with that class are desired, or 'allScores' if a matrix
        containing the scores for every class label are desired.
    multiClassStrategy : str
        May only be 'default' 'OneVsAll' or 'OneVsOne'
    numFolds : int
        The number of folds used in the cross validation. Cannot exceed
        the number of points in ``trainX``. Default 10.
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
    performance
        The calculated value of the ``performanceFunction`` after the
        test.

    See Also
    --------
    train, trainAndApply, trainAndTestOnTrainingData

    Examples
    --------
    Train and test datasets which contains the labels.

    >>> rawTrain = [[1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3],
    ...             [1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3]]
    >>> rawTest = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    >>> ftNames = ['a', 'b', 'c', 'label']
    >>> trainData = UML.createData('Matrix', rawTrain,
    ...                            featureNames=ftNames)
    >>> testData = UML.createData('Matrix', rawTest,
    ...                           featureNames=ftNames)
    >>> perform = UML.trainAndTest(
    ...     'Custom.KNNClassifier', trainX=trainData, trainY='label',
    ...     testX=testData, testY='label',
    ...     performanceFunction=UML.calculate.fractionIncorrect)
    >>> perform
    0.0

    Passing arguments to the learner. Both the arguments parameter and
    kwarguments can be utilized, they will be merged. Below, ``C`` and
    ``kernel`` are parameters for scikit-learn's SVC learner.

    >>> rawTrainX = [[1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1],
    ...              [1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1]]
    >>> rawTrainY = [[1], [2], [3], [1], [2], [3]]
    >>> rawTestX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> rawTestY = [[1], [2], [3]]
    >>> trainX = UML.createData('Matrix', rawTrainX)
    >>> trainY = UML.createData('Matrix', rawTrainY)
    >>> testX = UML.createData('Matrix', rawTestX)
    >>> testY = UML.createData('Matrix', rawTestY)
    >>> perform = UML.trainAndTest(
    ...     'sciKitLearn.SVC', trainX=trainX, trainY=trainY,
    ...     testX=testX, testY=testY,
    ...     performanceFunction=UML.calculate.fractionIncorrect,
    ...     arguments={'C': 0.1}, kernel='linear')
    >>> perform
    0.0
    """
    timer = startTimer(useLog)
    _validData(trainX, trainY, testX, testY, [True, True])
    _2dOutputFlagCheck(trainX, trainY, scoreMode, multiClassStrategy)

    merged = _mergeArguments(arguments, kwarguments)

    trainedLearner = UML.train(learnerName, trainX, trainY,
                               performanceFunction, merged,
                               scoreMode='label',
                               multiClassStrategy=multiClassStrategy,
                               numFolds=numFolds, useLog=False,
                               storeLog=useLog, doneValidData=True,
                               done2dOutputFlagCheck=True)

    if isinstance(testY, (six.string_types, int, numpy.integer)):
        testX = testX.copy()
        testY = testX.features.extract(testY, useLog=False)
    predictions = trainedLearner.apply(testX, {}, output, scoreMode,
                                       useLog=False)
    performance = computeMetrics(testY, None, predictions, performanceFunction)
    time = stopTimer(timer)

    metrics = {}
    for key, value in zip([performanceFunction], [performance]):
        metrics[key.__name__] = value
    extraInfo = None
    if merged != trainedLearner.arguments:
        extraInfo = {"bestParams": trainedLearner.arguments}
    if trainX == testX:
        name = "trainAndTestOnTrainingData"
        testX = None
        testY = None
    else:
        name = "trainAndTest"
    handleLogging(useLog, "run", name, trainX, trainY, testX, testY,
                  learnerName, merged, metrics, extraInfo, time)

    return performance


def trainAndTestOnTrainingData(learnerName, trainX, trainY,
                               performanceFunction, crossValidationError=False,
                               numFolds=10, arguments=None, output=None,
                               scoreMode='label', multiClassStrategy='default',
                               useLog=None, **kwarguments):
    """
    Train a model using the train data and get the performance results.

    ``trainAndTestOnTrainingData`` is the function for doing learner
    creation and evaluation in a single step with only a single data set
    (no withheld testing set). By default, this will calculate training
    error for the learner trained on that data set. However, cross
    validation error can instead be calculated by setting the parameter
    ``crossVadiationError`` to be True. In that case, we will partition
    the training set into a parameter controlled number of folds, and
    iteratively withhold each single fold to be used as the testing set
    of the learner trained on the rest of the data.

    Parameters
    ----------
    learnerName : str
        Name of the learner to be called, in the form 'package.learner'
    trainX: UML Base object
        Data to be used for training.
    trainY: identifier, UML Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another UML Base object containing the labels that
        correspond to ``trainX``.
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in UML.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
    crossValidationError: bool
        Whether we will calculate cross validation error or training
        error. In True case, the training data is split in the
        ``numFolds`` number of partitions. Each of those is iteratively
        withheld and used as the testing set for a learner trained on
        the combination of all of the non-withheld data. The performance
        results for each of those tests are then averaged together to
        act as the return value. In the False case, we train on the
        training data, and then use the same data as the withheld
        testing data. By default, this flag is set to False.
    numFolds : int
        The number of folds used in the cross validation. Cannot exceed
        the number of points in ``trainX``. Default 10.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To make use of multiple permutations, specify different values
        for a parameter as a tuple. eg. {'k': (1,3,5)} will generate an
        error score for  the learner when the learner was passed all
        three values of ``k``, separately. These will be merged with
        kwarguments for the learner.
    output : str
        The kind of UML Base object that the output of this function
        should be in. Any of the normal string inputs to the createData
        ``returnType`` parameter are accepted here. Alternatively, the
        value 'match' will indicate to use the type of the ``trainX``
        parameter.
    scoreMode : str
        In the case of a classifying learner, this specifies the type of
        output wanted: 'label' if we class labels are desired,
        'bestScore' if both the class label and the score associated
        with that class are desired, or 'allScores' if a matrix
        containing the scores for every class label are desired.
    multiClassStrategy : str
        May only be 'default' 'OneVsAll' or 'OneVsOne'
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
    performance
        The results of the test.

    See Also
    --------
    train, trainAndApply, trainAndTest

    Examples
    --------
    Train and test datasets which contains the labels.

    >>> rawTrain = [[1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3],
    ...             [1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3]]
    >>> ftNames = ['a', 'b', 'c', 'label']
    >>> trainData = UML.createData('Matrix', rawTrain,
    ...                            featureNames=ftNames)
    >>> perform = UML.trainAndTestOnTrainingData(
    ...     'Custom.KNNClassifier', trainX=trainData, trainY='label',
    ...     performanceFunction=UML.calculate.fractionIncorrect)
    >>> perform
    0.0

    Passing arguments to the learner. Both the arguments parameter and
    kwarguments can be utilized, they will be merged. Below, ``C`` and
    ``kernel`` are parameters for scikit-learn's SVC learner.

    >>> rawTrainX = [[1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1],
    ...              [1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1]]
    >>> rawTrainY = [[1], [2], [3], [1], [2], [3]]
    >>> trainX = UML.createData('Matrix', rawTrainX)
    >>> trainY = UML.createData('Matrix', rawTrainY)
    >>> perform = UML.trainAndTestOnTrainingData(
    ...     'sciKitLearn.SVC', trainX=trainX, trainY=trainY,
    ...     performanceFunction=UML.calculate.fractionIncorrect,
    ...     arguments={'C': 0.1}, kernel='linear')
    >>> perform
    0.0
    """
    if crossValidationError:
        merged = _mergeArguments(arguments, kwarguments)
        results = crossValidateReturnBest(learnerName, trainX, trainY,
                                          performanceFunction, merged,
                                          numFolds, scoreMode, useLog)
        performance = results[1]
        metrics = {}
        for key, value in zip([performanceFunction], [performance]):
            metrics[key.__name__] = value
        handleLogging(useLog, "run", 'trainAndTestOnTrainingData', trainX,
                      trainY, None, None, learnerName, merged, metrics,
                      extraInfo={'crossValidationError': True})

    else:
        performance = trainAndTest(learnerName, trainX, trainY, trainX, trainY,
                                   performanceFunction, arguments, output,
                                   scoreMode, multiClassStrategy, numFolds,
                                   useLog, **kwarguments)
    return performance


def log(logType, logInfo):
    """
    Enter an entry into active logger's database file.

    The log entry will include a timestamp and a run number in addition
    to the ``logType`` and ``logInfo``.

    Parameters
    ----------
    logType : str
        A string of the type of log entered. "load", "prep", "run", and
        "crossVal" types have builtin processing for logInfo. A default
        processing of logInfo will be used for unrecognized types.
    logInfo : str, list, dict
        Contains any information to be logged.

    See Also
    --------
    showLog
    """
    if not isinstance(logType, six.string_types):
        msg = "logType must be a string"
        raise InvalidArgumentType(msg)
    elif not isinstance(logInfo, (six.string_types, list, dict)):
        msg = "logInfo must be a python string, list, or dictionary type"
        raise InvalidArgumentType(msg)
    UML.logger.active.log(logType, logInfo)


def showLog(levelOfDetail=2, leastRunsAgo=0, mostRunsAgo=2, startDate=None,
            endDate=None, maximumEntries=100, searchForText=None, regex=False,
            saveToFileName=None, append=False):
    """
    Output contents of the logger's database file.

    Create a human readable interpretation of the log file based on
    the arguments passed and print or write to a file.

    Parameters
    ----------
    levelOfDetail:  int
        The value for the level of detail from 1, the least detail,
        to 3 (most detail). Default is 2.
        * Level 1 - Data loading, data preparation and
          preprocessing, custom user logs.
        * Level 2 - Outputs basic information about each run.
          Includes timestamp, run number, learner name, train and
          test object details, parameter, metric, and timer data if
          available.
        * Level 3 - Include cross-validation data.
    leastRunsAgo : int
        The least number of runs since the most recent run to
        include in the log. Default is 0.
    mostRunsAgo : int
        The most number of runs since the most recent run to
        include in the log. Default is 2.
    startDate :  str, datetime
        A string or datetime object of the date to begin adding runs
        to the log.
        Acceptable formats:
        * "YYYY-MM-DD"
        * "YYYY-MM-DD HH:MM"
        * "YYYY-MM-DD HH:MM:SS"
    endDate : str, datetime
        A string or datetime object of the date to stop adding runs
        to the log.
        See ``startDate`` for formatting.
    maximumEntries : int
        Maximum number of entries to allow before stopping the log.
        None will allow all entries provided from the query. Default
        is 100.
    searchForText :  str, regex
        Search for in each log entry. Default is None.
    saveToFileName : str
        The name of a file to write the human readable log. It will
        be saved in the same directory as the logger database.
        Default is None, showLog will print to standard out.
    append : bool
        Append logs to the file in saveToFileName instead of
        overwriting file. Default is False.

    See Also
    --------
    log
    """
    if levelOfDetail < 1 or levelOfDetail > 3 or levelOfDetail is None:
        msg = "levelOfDetail must be 1, 2, or 3"
        raise InvalidArgumentValue(msg)
    if (startDate is not None
            and endDate is not None
            and startDate > endDate):
        startDate = stringToDatetime(startDate)
        endDate = stringToDatetime(endDate)
        msg = "The startDate must be before the endDate"
        raise InvalidArgumentValueCombination(msg)
    if leastRunsAgo is not None:
        if leastRunsAgo < 0:
            msg = "leastRunsAgo must be greater than zero"
            raise InvalidArgumentValue(msg)
        if mostRunsAgo is not None and mostRunsAgo < leastRunsAgo:
            msg = "mostRunsAgo must be greater than or equal to "
            msg += "leastRunsAgo"
            raise InvalidArgumentValueCombination(msg)
    UML.logger.active.showLog(levelOfDetail, leastRunsAgo, mostRunsAgo,
                              startDate, endDate, maximumEntries,
                              searchForText, regex, saveToFileName, append)


def loadData(inputPath, useLog=None):
    """
    Load UML Base object.

    Parameters
    ----------
    inputPath : str
        The location (including file name and extension) to find a file
        previously generated by UML.data .save(). Expected file
        extension '.umld'.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.

    Returns
    -------
    UML.data.Base
        Subclass of Base object.
    """
    if not cloudpickle:
        msg = "To load UML objects, cloudpickle must be installed"
        raise PackageException(msg)
    if not inputPath.endswith('.umld'):
        msg = 'file extension for a saved UML data object should be .umld'
        raise InvalidArgumentValue(msg)
    with open(inputPath, 'rb') as file:
        ret = cloudpickle.load(file)
    if not isinstance(ret, UML.data.Base):
        msg = 'File does not contain a UML valid data Object.'
        raise InvalidArgumentType(msg)

    handleLogging(useLog, 'load', ret.getTypeString(), len(ret.points),
                  len(ret.features), ret.name, inputPath)
    return ret


def loadTrainedLearner(inputPath, useLog=None):
    """
    Load UML trainedLearner object.

    Parameters
    ----------
    inputPath : str
        The location (including file name and extension) to find a file
        previously generated for a trainedLearner object. Expected file
        extension '.umlm'.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.

    Returns
    -------
    UML.interfaces.UniversalInterface.TrainedLearner
    """
    if not cloudpickle:
        msg = "To load UML models, cloudpickle must be installed"
        raise PackageException(msg)
    if not inputPath.endswith('.umlm'):
        msg = 'File extension for a saved UML model should be .umlm'
        raise InvalidArgumentValue(msg)
    with open(inputPath, 'rb') as file:
        ret = cloudpickle.load(file)
    if not isinstance(ret, UML.interfaces.universal_interface.TrainedLearner):
        msg = 'File does not contain a UML valid trainedLearner Object.'
        raise InvalidArgumentType(msg)

    handleLogging(useLog, 'load', "TrainedLearner",
                  learnerName=ret.learnerName, learnerArgs=ret.arguments)
    return ret


def coo_matrixTodense(origTodense):
    """
    decorator for coo_matrix.todense
    """
    def f(self):
        try:
            return origTodense(self)
        except Exception:
            # flexible dtypes, such as strings, when used in scipy sparse
            # object create an implicitly mixed datatype: some values are
            # strings, but the rest are implicitly zero. In order to match
            # that, we must explicitly specify a mixed type for our destination
            # matrix
            retDType = self.dtype
            if isinstance(retDType, numpy.flexible):
                retDType = object
            ret = numpy.matrix(numpy.zeros(self.shape), dtype=retDType)
            nz = (self.row, self.col)
            for (i, j), v in zip(zip(*nz), self.data):
                ret[i, j] = v
            return ret
    return f

if scipy:
    #monkey patch for coo_matrix.todense
    denseMatrix = coo_matrixTodense(scipy.sparse.coo_matrix.todense)
    scipy.sparse.coo_matrix.todense = denseMatrix
