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
from UML.exceptions import ArgumentException, PackageException
from UML.logger import Stopwatch
from UML.helpers import findBestInterface
from UML.helpers import _learnerQuery
from UML.helpers import _validScoreMode
from UML.helpers import _validMultiClassStrategy
from UML.helpers import _unpackLearnerName
from UML.helpers import _validArguments
from UML.helpers import _validData
from UML.helpers import _2dOutputFlagCheck
from UML.helpers import LearnerInspector
from UML.helpers import copyLabels
from UML.helpers import ArgumentIterator
from UML.helpers import _mergeArguments
from UML.helpers import crossValidateBackend
from UML.helpers import isAllowedRaw
from UML.helpers import initDataObject
from UML.helpers import createDataFromFile
from UML.helpers import createConstantHelper
from UML.randomness import numpyRandom
from UML.calculate import detectBestResult

cloudpickle = UML.importModule('cloudpickle')
scipy = UML.importModule('scipy.sparse')

def createRandomData(
        returnType, numPoints, numFeatures, sparsity, pointNames='automatic',
        featureNames='automatic', elementType='float', name=None):
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
    createData

    Examples
    --------
    TODO
    """

    if numPoints < 1:
        msg = "must specify a positive nonzero number of points"
        raise ArgumentException(msg)
    if numFeatures < 1:
        msg = "must specify a positive nonzero number of features"
        raise ArgumentException(msg)
    if sparsity < 0 or sparsity >= 1:
        msg = "sparsity must be greater than zero and less than one"
        raise ArgumentException(msg)
    if elementType != "int" and elementType != "float":
        raise ArgumentException("elementType may only be 'int' or 'float'")


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
        nzLocation = numpy.random.choice(gridSize, size=numNonZeroValues,
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

    return createData(returnType, data=randData, pointNames=pointNames,
                      featureNames=featureNames, name=name)


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
    TODO
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
    TODO
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
    TODO
    """
    retAllowed = copy.copy(UML.data.available)
    if returnType not in retAllowed:
        msg = "returnType must be a value in " + str(retAllowed)
        raise ArgumentException(msg)

    if size <= 0:
        msg = "size must be 0 or greater, yet " + str(size)
        msg += " was given."
        raise ArgumentException(msg)

    if returnType == 'Sparse':
        if not scipy:
            msg = "scipy is not available"
            raise PackageException(msg)

        assert returnType == 'Sparse'
        rawDiag = scipy.sparse.identity(size)
        rawCoo = scipy.sparse.coo_matrix(rawDiag)
        return UML.createData(returnType, rawCoo, pointNames=pointNames,
                              featureNames=featureNames, name=name)
    else:
        raw = numpy.identity(size)
        return UML.createData(returnType, raw, pointNames=pointNames,
                              featureNames=featureNames, name=name)


def normalizeData(learnerName, trainX, trainY=None, testX=None, arguments=None,
                  **kwarguments):
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
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. Same format as the arguments parameter.

    See Also
    --------
    UML.data.Points.normalize, UML.data.Features.normalize

    Examples
    --------
    TODO
    """
    (_, trueLearnerName) = _unpackLearnerName(learnerName)

    tl = UML.train(learnerName, trainX, trainY, arguments=arguments,
                   **kwarguments)
    normalizedTrain = tl.apply(trainX, arguments=arguments, **kwarguments)
    if normalizedTrain.getTypeString() != trainX.getTypeString():
        normalizedTrain = normalizedTrain.copyAs(trainX.getTypeString())

    if testX is not None:
        normalizedTest = tl.apply(testX, arguments=arguments, **kwarguments)
        if normalizedTest.getTypeString() != testX.getTypeString():
            normalizedTest = normalizedTest.copyAs(testX.getTypeString())

    # modify references and names for trainX and testX
    trainX.referenceDataFrom(normalizedTrain)
    trainX.name = trainX.name + " " + trueLearnerName

    if testX is not None:
        testX.referenceDataFrom(normalizedTest)
        testX.name = testX.name + " " + trueLearnerName


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

    Examples
    --------
    TODO
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

    Examples
    --------
    TODO
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

    Examples
    --------
    TODO
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

    Examples
    --------
    TODO
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

    Examples
    --------
    TODO
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

    Examples
    --------
    TODO
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

    Examples
    --------
    TODO
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
        keepFeatures='all', ignoreNonNumericalFeatures=False, useLog=None,
        reuseData=False, inputSeparator='automatic',
        treatAsMissing=(float('nan'), numpy.nan, None, '', 'None', 'nan'),
        replaceMissingWith=numpy.nan):
    """
    Function to instantiate one of the UML data container types.

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
        By default, a value of 'automatic' indicates that this function
        should attempt to detect the presence of pointNames in the data
        which will only be attempted when loading from a file. In no
        names are found, or data isn't being loaded from a file, then we
        use default names. A value of True indicates that point names
        are embedded in the data within the first column. A value of
        False indicates that names are not embedded and that default
        names should be used. Finally, they may be specified explictly
        by some list-like or dict-like object, so long as all points
        in the data are assigned a name and the names for each point are
        unique.
    featureNames : 'automatic', bool, list, dict
        Specifices the source for feature names in the returned object.
        By default, a value of 'automatic' indicates that this function
        should attempt to detect the presence of featureNames in the
        data which will only be attempted when loading from a file. In
        no names are found, or data isn't being loaded from a file, then
        we use default names. A value of True indicates that feature
        names are embedded in the data within the first column. A value
        of False indicates that names are not embedded and that default
        names should be used. Finally, they may be specified explictly
        by some list-like or dict-like object, so long as all points in
        the data are assigned a name and the names for each point are
        unique.
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
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
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

    Returns
    -------
    UML.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    Examples
    --------
    TODO
    """
    # validation of pointNames and featureNames
    accepted = (bool, list, dict)
    if pointNames != 'automatic' and not isinstance(pointNames, accepted):
        msg = "pointNames may only be the values True, False, 'automatic' or "
        msg += "a list or dict specifying a mapping between names and indices."
        raise ArgumentException(msg)

    if featureNames != 'automatic' and not isinstance(featureNames, accepted):
        msg = "featureNames may only be the values True, False, 'automatic' "
        msg += "or a list or dict specifying a mapping between names and "
        msg += "indices."
        raise ArgumentException(msg)

    retAllowed = copy.copy(UML.data.available)
    retAllowed.append(None)
    if returnType not in retAllowed:
        msg = "returnType must be a value in " + str(retAllowed)
        raise ArgumentException(msg)

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
        return ret
    # input is an open file or a path to a file
    elif isinstance(data, six.string_types) or looksFileLike(data):
        ret = createDataFromFile(
            returnType=returnType, data=data, pointNames=pointNames,
            featureNames=featureNames, name=name, keepPoints=keepPoints,
            keepFeatures=keepFeatures,
            ignoreNonNumericalFeatures=ignoreNonNumericalFeatures,
            inputSeparator=inputSeparator, treatAsMissing=treatAsMissing,
            replaceMissingWith=replaceMissingWith)
        return ret
    # no other allowed inputs
    else:
        msg = "data must contain either raw data or the path to a file to be "
        msg += "loaded"
        raise ArgumentException(msg)


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
    bestResult = crossValidateReturnBest(
        learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode,
        useLog, **kwarguments)

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
    example **kwarguments: {'a':(1,2,3), 'b':(4,5)}
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
    return crossValidateBackend(learnerName, X, Y, performanceFunction,
                                arguments, numFolds, scoreMode, useLog,
                                **kwarguments)


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
        A string or a list of strings in the format 'package.learner'

    Returns
    -------
    str - for a single learner
    list - for multiple learners

    Examples
    --------
    TODO
    """
    #argument checking
    if not isinstance(learnerNames, list):
        learnerNames = [learnerNames]

    resultsList = []
    secondPassLearnerNames = []
    for name in learnerNames:
        if not isinstance(name, str):
            msg = "learnerNames must be a string or a list of strings."
            raise ArgumentException(msg)

        splitTuple = _unpackLearnerName(name)
        currInterface = findBestInterface(splitTuple[0])
        allValidLearnerNames = currInterface.listLearners()
        if not splitTuple[1] in allValidLearnerNames:
            msg = name + " is not a valid learner on your machine."
            raise ArgumentException(msg)
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
          useLog=None, doneValidData=False, doneValidArguments1=False,
          doneValidArguments2=False, doneValidMultiClassStrategy=False,
          done2dOutputFlagCheck=False, **kwarguments):
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
        ``arg1=2, arg2=4``. Will be merged with arguments.

    Returns
    -------
    UML.interfaces.UniversalInterface.TrainedLearner

    See Also
    --------
    trainAndApply, trainAndTest, trainAndTestOnTrainingData

    Examples
    --------
    TODO
    """
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

    if useLog is None:
        useLog = UML.settings.get("logger", "enabledByDefault")
        useLog = useLog.lower() == 'true'

    if useLog:
        timer = Stopwatch()
    else:
        timer = None

    # perform CV (if needed)
    argCheck = ArgumentIterator(merged)
    if argCheck.numPermutations != 1:
        if performanceFunction is None:
            msg = "Cross validation was triggered to select the best "
            msg += "parameter set, yet no performanceFunction was specified. "
            msg += "Either one must be specified (see UML.calculate for "
            msg += "out-of-the-box options) or there must be no choices in "
            msg += "the parameters."
            raise ArgumentException(msg)

        #if we are logging this run, we need to start the timer
        if useLog:
            timer = Stopwatch()
            timer.start('crossValidateReturnBest')
        else:
            timer = None

        #modify numFolds if needed
        numFolds = len(trainX.points) if len(trainX.points) < 10 else 10
        # sig (learnerName, X, Y, performanceFunction, arguments=None,
        #      numFolds=10, scoreMode='label', useLog=None, maximize=False,
        #      **kwarguments):
        results = UML.crossValidateReturnBest(learnerName, trainX, trainY,
                                              performanceFunction, merged,
                                              numFolds=numFolds,
                                              scoreMode=scoreMode,
                                              useLog=useLog)
        bestArgument, _ = results

        if useLog:
            timer.stop('crossValidateReturnBest')
    else:
        bestArgument = merged

    interface = findBestInterface(package)

    trainedLearner = interface.train(trueLearnerName, trainX, trainY,
                                     multiClassStrategy, bestArgument,
                                     useLog, timer)

    if useLog:
        funcString = interface.getCanonicalName() + '.' + trueLearnerName
        UML.logger.active.logRun(trainX, trainY, None, None, funcString, None,
                                 None, None, timer, extraInfo=bestArgument)

    return trainedLearner


def trainAndApply(learnerName, trainX, trainY=None, testX=None,
                  performanceFunction=None, arguments=None, output=None,
                  scoreMode='label', multiClassStrategy='default', useLog=None,
                  **kwarguments):
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
        data set on which the trained learner will be applied (i.e.
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
        ``arg1=2, arg2=4``. Will be merged with arguments.

    Returns
    -------
    results
        The resulting output of applying learner.

    See Also
    --------
    train, trainAndTest, trainAndTestOnTrainingData

    Examples
    --------
    TODO
    """
    _validData(trainX, trainY, testX, None, [False, False])
    _validScoreMode(scoreMode)
    _2dOutputFlagCheck(trainX, trainY, scoreMode, multiClassStrategy)

    if testX is None:
        testX = trainX

    if useLog is None:
        useLog = UML.settings.get("logger", "enabledByDefault")
        useLog = useLog.lower() == 'true'

    if useLog:
        timer = Stopwatch()
        timer.start('trainAndApply')
    else:
        timer = None

    # deepLog = False
    # if useLog and multiClassStrategy != 'default':
    #     deepLog = UML.settings.get('logger',
    #                                'enableMultiClassStrategyDeepLogging')
    #     deepLog = True if deepLog.lower() == 'true' else False


    trainedLearner = UML.train(learnerName, trainX, trainY,
                               performanceFunction, arguments,
                               scoreMode='label',
                               multiClassStrategy=multiClassStrategy,
                               useLog=useLog, doneValidData=True,
                               done2dOutputFlagCheck=True, **kwarguments)
    results = trainedLearner.apply(testX, {}, output, scoreMode, useLog=False)

    if useLog:
        timer.stop('trainAndApply')
        funcString = learnerName
        UML.logger.active.logRun(trainX, trainY, testX, None, funcString, None,
                                 results, None, timer, extraInfo=None)

    return results


def trainAndTest(learnerName, trainX, trainY, testX, testY,
                 performanceFunction, arguments=None, output=None,
                 scoreMode='label', multiClassStrategy='default', useLog=None,
                 **kwarguments):
    """
    Train a model and get the results of its performance.

    For each permutation of the merge of 'arguments' and 'kwarguments'
    (more below), ``trainAndTest`` uses cross validation to generate a
    performance score for the algorithm, given the particular argument
    permutation. The argument permutation that performed best cross
    validating over the training data is then used as the lone argument
    for training on the whole training data set. Finally, the learned
    model generates predictions for the testing set, an the performance
    of those predictions is calculated and returned. If no additional
    arguments are supplied via arguments or **kwarguments, then
    ``trainAndTest`` just returns the performance of the algorithm with
    default arguments on the testing data.

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
    testY : identifier, UML Base object
        A name or index of the feature in ``testX`` containing the
        labels or another UML Base object containing the labels that
        correspond to ``testX``.
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
        ``arg1=2, arg2=4``. Will be merged with arguments.

    Returns
    -------
    performance
        The results of the test.

    See Also
    --------
    train, trainAndApply, trainAndTestOnTrainingData

    Examples
    --------
    TODO
    """

    _2dOutputFlagCheck(trainX, trainY, scoreMode, None)
    trainY = copyLabels(trainX, trainY)
    testY = copyLabels(testX, testY)

    if useLog is None:
        useLog = UML.settings.get("logger", "enabledByDefault")
        useLog = useLog.lower() == 'true'

    #if we are logging this run, we need to start the timer
    if useLog:
        timer = Stopwatch()
        timer.start('trainAndTest')
    else:
        timer = None

    predictions = UML.trainAndApply(learnerName, trainX, trainY, testX,
                                    performanceFunction, arguments, output,
                                    scoreMode='label',
                                    multiClassStrategy=multiClassStrategy,
                                    useLog=useLog, **kwarguments)
    performance = UML.helpers.computeMetrics(testY, None, predictions,
                                             performanceFunction)

    if useLog:
        timer.stop('trainAndTest')
        funcString = learnerName
        UML.logger.active.logRun(trainX, trainY, testX, testY, funcString,
                                 [performanceFunction], predictions,
                                 [performance], timer, None)

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
        the number of points in ``trainX``, ``trainY``. Default is 10.
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
        ``arg1=2, arg2=4``. Will be merged with arguments.

    Returns
    -------
    performance
        The results of the test.

    See Also
    --------
    train, trainAndApply, trainAndTest

    Examples
    --------
    TODO
    """

    performance = trainAndTest(learnerName, trainX, trainY, trainX, trainY,
                               performanceFunction, arguments, output,
                               scoreMode, multiClassStrategy, useLog)
    return performance


def loadData(inputPath):
    """
    Load UML Base object.

    Parameters
    ----------
    inputPath : str
        The location (including file name and extension) to find a file
        previously generated by UML.data .save(). Expected file
        extension '.umld'.

    Returns
    -------
    UML.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    Examples
    --------
    TODO
    """
    if not cloudpickle:
        msg = "To load UML objects, cloudpickle must be installed"
        raise PackageException(msg)
    if not inputPath.endswith('.umld'):
        msg = 'file extension for a saved UML Base object should be .umld'
        raise ArgumentException(msg)
    with open(inputPath, 'rb') as file:
        ret = cloudpickle.load(file)
    if not isinstance(ret, UML.data.Base):
        msg = 'File does not contain a UML valid data Object.'
        raise ArgumentException(msg)
    return ret


def loadTrainedLearner(inputPath):
    """
    Load UML trainedLearner object.

    Parameters
    ----------
    inputPath : str
        The location (including file name and extension) to find a file
        previously generated for a trainedLearner object. Expected file
        extension '.umlm'.

    Returns
    -------
    UML.interfaces.UniversalInterface.TrainedLearner

    Examples
    --------
    TODO
    """
    if not cloudpickle:
        msg = "To load UML models, cloudpickle must be installed"
        raise PackageException(msg)
    if not inputPath.endswith('.umlm'):
        msg = 'File extension for a saved UML model should be .umlm'
        raise ArgumentException(msg)
    with open(inputPath, 'rb') as file:
        ret = cloudpickle.load(file)
    if not isinstance(ret, UML.interfaces.universal_interface.TrainedLearner):
        msg = 'File does not contain a UML valid trainedLearner Object.'
        raise ArgumentException(msg)
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
