"""
Module containing most of the user facing functions for the top level uml import.

"""

from __future__ import absolute_import
import numpy
import inspect
import operator
import re
import datetime
import os
import copy
import six.moves.configparser
import math
from dateutil.parser import parse
import cloudpickle

import UML
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from UML.exceptions import InvalidArgumentValueCombination, PackageException

from UML.logger import enableLogging, logCapture, directCall

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
from UML.helpers import trainAndApplyOneVsAll
from UML.helpers import trainAndApplyOneVsOne
from UML.helpers import _mergeArguments
from UML.helpers import crossValidateBackend
from UML.helpers import isAllowedRaw
from UML.helpers import initDataObject
from UML.helpers import createDataFromFile
from UML.helpers import createConstantHelper
from UML.helpers import computeMetrics

from UML.randomness import numpyRandom

from UML.interfaces.interface_helpers import checkClassificationStrategy

from UML.calculate import detectBestResult
import six
from six.moves import range
from six.moves import zip
cloudpickle = UML.importModule('cloudpickle')
scipy = UML.importModule('scipy.sparse')

UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def createRandomData(
        returnType, numPoints, numFeatures, sparsity, pointNames='automatic',
        featureNames='automatic', elementType='float', name=None):
    """
    Generates a data object with random contents.

    returnType - May only be one of the allowed types specified in
    UML.data.available.

    numPoints - the number of points in the returned object.

    numFeatures - the number of features in the returned object.

    sparsity - is the likelihood that the value of a (point,feature) pair is
    zero.

    elementType - if is 'float' (default) then the value of (point, feature)
    pairs are sampled from a normal distribution (location 0, scale 1). If
    elementType is 'int' then value of (point, feature) pairs are sampled from
    uniform integer distribution [1 100]. Zeros are not counted in/do not
    affect the aforementioned sampling distribution.

    pointNames - names to be associated with the points in the returned object.
    If 'automatic', default names will be assigned.

    featureNames - names to be associated with the features in the returned
    object. If 'automatic', default names will be assigned.

    name - When not None, this value is set as the name attribute of the
    returned object.

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
        nzLocation = numpy.random.choice(gridSize, size=numNonZeroValues, replace=False)

        # The point value is determined by counting how many groups of numFeatures fit into
        # the position number
        pointIndices = numpy.floor(nzLocation / numFeatures)
        # The feature value is determined by counting the offset from each point edge.
        featureIndices = nzLocation % numFeatures

        if elementType == 'int':
            dataVector = numpyRandom.randint(low=1, high=100, size=numNonZeroValues)
        #numeric type is float; distribution is normal
        else:
            dataVector = numpyRandom.normal(0, 1, size=numNonZeroValues)

        #pointIndices and featureIndices are
        randData = scipy.sparse.coo.coo_matrix((dataVector, (pointIndices, featureIndices)), (numPoints, numFeatures))

    #for non-sparse matrices, use numpy to generate matrices with sparsity characterics
    else:
        if elementType == 'int':
            filledIntMatrix = numpyRandom.randint(1, 100, (numPoints, numFeatures))
        else:
            filledFloatMatrix = numpyRandom.normal(loc=0.0, scale=1.0, size=(numPoints, numFeatures))

        #if sparsity is zero
        if abs(float(sparsity) - 0.0) < 0.0000000001:
            if elementType == 'int':
                randData = filledIntMatrix
            else:
                randData = filledFloatMatrix
        else:
            binarySparsityMatrix = numpyRandom.binomial(1, 1.0 - sparsity, (numPoints, numFeatures))

            if elementType == 'int':
                randData = binarySparsityMatrix * filledIntMatrix
            else:
                randData = binarySparsityMatrix * filledFloatMatrix

    return createData(returnType, data=randData, pointNames=pointNames,
                      featureNames=featureNames, name=name)


def ones(returnType, numPoints, numFeatures, pointNames='automatic',
         featureNames='automatic', name=None):
    """
    Return a data object of the given type and size containing only the value 1.

    returnType - May only be one of the allowed types specified in UML.data.available

    numPoints - the number of points in the returned object.

    numFeatures - the number of features in the returned object.

    pointNames - names to be associated with the points in the returned object; if
    the 'automatic' flag is given, default names will be assigned.

    featureNames - names to be associated with the features in the returned object;
    if the 'automatic' flag is given, default names will be assigned.

    name - When not None, this value is set as the name attribute of the
    returned object.

    Returns - a numPoints by numFeatures sized objects where every value is equal
    to 1.

    """
    return createConstantHelper(numpy.ones, returnType, numPoints, numFeatures, pointNames,
                                featureNames, name)


def zeros(returnType, numPoints, numFeatures, pointNames='automatic',
          featureNames='automatic', name=None):
    """
    Return a data object of the given type and size containing only the value 0.

    returnType - May only be one of the allowed types specified in UML.data.available

    numPoints - the number of points in the returned object.

    numFeatures - the number of features in the returned object.

    pointNames - names to be associated with the points in the returned object; if
    'automatic', default names will be assigned.

    featureNames - names to be associated with the features in the returned object;
    if 'automatic', default names will be assigned.

    name - When not None, this value is set as the name attribute of the
    returned object.

    Returns - a numPoints by numFeatures sized object where every value is equal
    to 0.

    """
    return createConstantHelper(numpy.zeros, returnType, numPoints, numFeatures, pointNames,
                                featureNames, name)


def identity(returnType, size, pointNames='automatic', featureNames='automatic', name=None):
    """
    Return a square data object of the given size representing an identity matrix.

    returnType - May only be one of the allowed types specified in UML.data.available

    size - the number of points and features in the returned object.

    pointNames - names to be associated with the points in the returned object; if
    'automatic', default names will be assigned.

    featureNames - names to be associated with the features in the returned object;
    if 'automatic', default names will be assigned.

    name - When not None, this value is set as the name attribute of the
    returned object.

    Returns - a size by size shaped object where every value on the main diagonal is 1
    and every other value is 0.

    """
    retAllowed = copy.copy(UML.data.available)
    if returnType not in retAllowed:
        msg = "returnType must be a value in " + str(retAllowed)
        if not isinstance(returnType, six.string_types):
            raise InvalidArgumentType(msg)
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
        return UML.createData(returnType, rawCoo, pointNames=pointNames, featureNames=featureNames, name=name)
    else:
        raw = numpy.identity(size)
        return UML.createData(returnType, raw, pointNames=pointNames, featureNames=featureNames, name=name)


def normalizeData(learnerName, trainX, trainY=None, testX=None, arguments={}, useLog=None, **kwarguments):
    """
    Calls on the functionality of a package to train on some data and then modify both
    the training data and a set of test data according to the produced model.

    Parameters:

    learnerName : String name of the learner to be called, in the form 'package.learner'

    trainX: data to be used for training (as some form of UML data Base object)

    trainY: used to retrieve the known class labels of the training data. Either
    contains the labels themselves (as a Base object) or an identifier (numerical
    index or string name) that defines their placement in the trainX object as a
    feature ID.

    testX: data set to be used for testing (as some form of Base object)

    arguments : dictionary mapping argument names (strings) to their values,
    to be used during training and application. example: {'dimensions':5, 'k':5}

    **kwarguments : kwargs specified variables that are passed to the learner. Same
    format as the arguments parameter.

    """
    if UML.logger.active.position == 0:
        if enableLogging(useLog):
            wrapped = logCapture(normalizeData)
        else:
            wrapped = directCall(normalizeData)
        return wrapped(learnerName, trainX, trainY, testX, arguments,
                       useLog=False, **kwarguments)

    (packName, trueLearnerName) = _unpackLearnerName(learnerName)

    tl = UML.train(learnerName, trainX, trainY, arguments=arguments, useLog=False, **kwarguments)
    normalizedTrain = tl.apply(trainX, arguments=arguments, useLog=False, **kwarguments)

    if normalizedTrain.getTypeString() != trainX.getTypeString():
        normalizedTrain = normalizedTrain.copyAs(trainX.getTypeString())

    if testX is not None:
        normalizedTest = tl.apply(testX, arguments=arguments, useLog=False, **kwarguments)
        if normalizedTest.getTypeString() != testX.getTypeString():
            normalizedTest = normalizedTest.copyAs(testX.getTypeString())

    # modify references and names for trainX and testX
    trainX.referenceDataFrom(normalizedTrain)
    trainX.name = trainX.name + " " + trueLearnerName

    if testX is not None:
        testX.referenceDataFrom(normalizedTest)
        testX.name = testX.name + " " + trueLearnerName

    merged = _mergeArguments(arguments, kwarguments)
    UML.logger.active.logRun("normalizeData", trainX, trainY, testX, None, learnerName, merged, None)


def registerCustomLearnerAsDefault(customPackageName, learnerClassObject):
    """ Register the given customLearner class so that it is callable by the
    top level UML functions through the interface of the specified custom
    package. This operation modifies the saved configuration file so that
    this change will be reflected during future sesssions.

    customPackageName : The string name of the package preface you want to use when calling
    the learner. If there is already an interface for a custom package with this name, the
    learner will be accessible through that interface. If there is no interface to a custom
    package of that name, then one will be created. You cannot register a custom learner to
    be callable through the interface for a non-custom package (such as ScikitLearn or MLPY).
    Therefore, customPackageName cannot be a value which is the accepted alias of another
    package's interface.

    learnerClassObject : The class object implementing the learner you want registered. It
    will be checked using UML.interfaces.CustomLearner.validateSubclass to ensure that all
    details of the provided implementation are acceptable.

    """
    UML.helpers.registerCustomLearnerBackend(customPackageName, learnerClassObject, True)


def registerCustomLearner(customPackageName, learnerClassObject):
    """
    Register the given customLearner class so that it is callable by the
    top level UML functions through the interface of the specified custom
    package. Though this operation by itself is temporary, it has effects
    in UML.settings, so subsequent saveChanges operations may cause it to
    be reflected in future sessions.

    customPackageName : The string name of the package preface you want to use when calling
    the learner. If there is already an interface for a custom package with this name, the
    learner will be accessible through that interface. If there is no interface to a custom
    package of that name, then one will be created. You cannot register a custom learner to
    be callable through the interface for a non-custom package (such as ScikitLearn or MLPY).
    Therefore, customPackageName cannot be a value which is the accepted alias of another
    package's interface.

    learnerClassObject : The class object implementing the learner you want registered. It
    will be checked using UML.interfaces.CustomLearner.validateSubclass to ensure that all
    details of the provided implementation are acceptable.

    """
    UML.helpers.registerCustomLearnerBackend(customPackageName, learnerClassObject, False)


def deregisterCustomLearnerAsDefault(customPackageName, learnerName):
    """
    Remove accessibility of the learner with the given name from the
    interface of the package with the given name permenantly. This
    operation modifies the saved configuration file so that this
    change will be reflected during future sesssions.

    customPackageName : the name of the interface / custom package from which the learner
    named 'learnerName' is to be removed from. If that learner was the last one grouped in
    that custom package, then the interface is removed from the UML.interfaces.available list.

    learnerName : the name of the learner to be removed from the interface / custom package with
    the name 'customPackageName'

    """
    UML.helpers._deregisterCustomLearnerBackend(customPackageName, learnerName, True)


def deregisterCustomLearner(customPackageName, learnerName):
    """
    Remove accessibility of the learner with the given name from the
    interface of the package with the given name temporarily in this
    session. This has effects in UML.settings, so subsequent saveChanges
    operations may cause it to be reflected in future sessions.

    customPackageName : the name of the interface / custom package from which the learner
    named 'learnerName' is to be removed from. If that learner was the last one grouped in
    that custom package, then the interface is removed from the UML.interfaces.available list.

    learnerName : the name of the learner to be removed from the interface / custom package with
    the name 'customPackageName'

    """
    UML.helpers.deregisterCustomLearnerBacked(customPackageName, learnerName, False)


def learnerParameters(name):
    """
    Takes a string of the form 'package.learnerName' and returns a list of
    strings which are the names of the parameters when calling package.learnerName

    If the name cannot be found within the package, then an exception will be thrown.
    If the name is found, be for some reason we cannot determine what the parameters
    are, then we return None. Note that if we have determined that there are no
    parameters, we return an empty list.

    """
    return _learnerQuery(name, 'parameters')


def learnerDefaultValues(name):
    """
    Takes a string of the form 'package.learnerName' and returns a returns a
    dict mapping of parameter names to their default values when calling
    package.learnerName

    If the name cannot be found within the package, then an exception will be thrown.
    If the name is found, be for some reason we cannot determine what the parameters
    are, then we return None. Note that if we have determined that there are no
    parameters, we return an empty dict.

    """
    return _learnerQuery(name, 'defaults')


def listLearners(package=None):
    """
    Takes the name of a package, and returns a list of learners that are callable
    through that package using UML's training, applying, and testing functions.

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
        elementType=None, name=None, path=None, keepPoints='all', keepFeatures='all',
        ignoreNonNumericalFeatures=False, reuseData=False, inputSeparator='automatic',
        treatAsMissing=[float('nan'), numpy.nan, None, '', 'None', 'nan'],
        replaceMissingWith=numpy.nan, useLog=None):
    """Function to instantiate one of the UML data container types.

    returnType: string (or None) indicating which kind of UML data type you
    want returned. If None is given, UML will attempt to detect the type most
    appropriate for the data. Currently accepted are the strings "List",
    "Matrix", and "Sparse" -- which are case sensitive.

    data: the source of the data to be loaded into the returned object. The
    source may be any number of in-python objects (lists, numpy arrays, numpy
    matrices, scipy sparse objects) as long as they specify a 2d matrix of
    data. Alternatively, the data may be read from a file, specified either
    as a string path, or a currently open file-like object.

    pointNames: specifices the source for point names in the returned object.
    By default, a value of 'automatic' indicates that this function should
    attempt to detect the presence of pointNames in the data which will
    only be attempted when loading from a file. In no names are found, or
    data isn't being loaded from a file, then we use default names. A value
    of True indicates that point names are embedded in the data within the
    first column. A value of False indicates that names are not embedded
    and that default names should be used. Finally, they may be specified
    explictly by some list-like or dict-like object, so long as all points
    in the data are assigned a name and the names for each point are unique.

    featureNames: specifices the source for feature names in the returned
    object. By default, a value of 'automatic' indicates that this function
    should attempt to detect the presence of featureNames in the data which
    will only be attempted when loading from a file. In no names are found,
    or data isn't being loaded from a file, then we use default names. A
    value of True indicates that feature names are embedded in the data
    within the first column. A value of False indicates that names are not
    embedded and that default names should be used. Finally, they may be
    specified explictly by some list-like or dict-like object, so long as
    all points in the data are assigned a name and the names for each point
    are unique.

    name: When not None, this value is set as the name attribute of the
    returned object

    keepPoints: Allows the user to select which points will be kept
    in the returned object, those not selected will be discarded. By
    default, the value 'all' indicates that all possible points in the raw
    data will be kept. Alternatively, the user may provide a list containing
    either names or indices (or a mix) of those points they want to be
    kept from the raw data. The order of this list will determine the order
    of points in the resultant object. In the case of reading data from a
    file, the selection will be done at read time, thus limiting the amount
    of data read into memory.

    keepFeatures: Allows the user to select which features will be kept
    in the returned object, those not selected will be discarded. By
    default, the value 'all' indicates that all possible features in the raw
    data will be kept. Alternatively, the user may provide a list containing
    either names or indices (or a mix) of those features they want to be
    kept from the raw data. The order of this list will determine the order
    of features in the resultant object. In the case of reading data from a
    file, the selection will be done at read time, thus limiting the amount
    of data read into memory. Names and indices are defined with respect to the
    data regardless of filtering by the ignoreNonNumericalFeatures flag; just
    because a feature is removed, the indices of subsequent features will not
    be shifted. The ignoreNonNumericalFeatures flag is only consdered after
    selection: if a selected feature has non-numerical values and
    ignoreNonNumericalFeatures is True valued, then that feature will NOT be
    included in the output. Similarly, if a feature has only numerical values
    in points that were selected, then even if there are non-numerical values
    in the points that were not selected, then that feature will be included

    ignoreNonNumericalFeatures: True or False (default False) value indicating
    whether, when loading from a file, features containing non numercal data
    shouldn't be loaded into the final object. For example, you may be loading
    a file which has a column of strings; setting this flag to true will allow
    you to load that file into a Matrix object (which may contain floats only).
    Currently only has an effect on csv files, as the matrix market format
    does not support non numerical values. Also, if there is point or feature
    selection occurring, then only those values within selected points and
    features are considered when determining whether to apply this operation.

    useLog: True, False (default), or None valued flag indicating whether this
    call should be logged by the UML logger. If None, the configurable	global
    default is used.

    inputSeparator: The character that is used to separate fields in the input
    file, if necessary. By default, a value of 'automatic' will attempt to
    determine the appropriate separator. Otherwise, a single character string
    of the separator in the file can be passed.

    treatAsMissing: A list of values that will be treated as missing values in
    the data. These values will be replaced with value from replaceMissingWith
    By default this list is [float('nan'), numpy.nan, None, '', 'None', 'nan']
    Set to None or [] to disable replacing missing values.

    replaceMissingWith: A single value with which to replace any value in
    treatAsMissing. By default this is numpy.nan
    """
    if UML.logger.active.position == 0:
        if enableLogging(useLog):
            wrapped = logCapture(createData)
        else:
            wrapped = directCall(createData)
        return wrapped(returnType, data, pointNames, featureNames, elementType,
                       name, path, keepPoints, keepFeatures,
                       ignoreNonNumericalFeatures, reuseData, inputSeparator,
                       treatAsMissing, replaceMissingWith, useLog=False)
    # validation of pointNames and featureNames
    if pointNames != 'automatic' and not isinstance(pointNames, (bool, list, dict)):
        msg = "pointNames may only be the values True, False, 'automatic' or "
        msg += "a list or dict specifying a mapping between names and indices."
        raise InvalidArgumentType(msg)

    if featureNames != 'automatic' and not isinstance(featureNames, (bool, list, dict)):
        msg = "featureNames may only be the values True, False, 'automatic' or "
        msg += "a list or dict specifying a mapping between names and indices."
        raise InvalidArgumentType(msg)

    retAllowed = copy.copy(UML.data.available)
    retAllowed.append(None)
    if returnType not in retAllowed:
        msg = "returnType must be a value in " + str(retAllowed)
        raise InvalidArgumentValue(msg)

    def looksFileLike(toCheck):
        hasRead = hasattr(toCheck, 'read')
        hasWrite = hasattr(toCheck, 'write')
        return (hasRead and hasWrite)

    # input is raw data
    if isAllowedRaw(data, allowLPT=True):
        ret = initDataObject(
            returnType=returnType, rawData=data, pointNames=pointNames,
            featureNames=featureNames, elementType=elementType, name=name, path=path,
            keepPoints=keepPoints, keepFeatures=keepFeatures, reuseData=reuseData,
            treatAsMissing=treatAsMissing, replaceMissingWith=replaceMissingWith)
        UML.logger.active.logLoad(returnType, len(ret.points),
                                  len(ret.features), name, path)

        return ret
    # input is an open file or a path to a file
    elif isinstance(data, six.string_types) or looksFileLike(data):
        ret = createDataFromFile(
            returnType=returnType, data=data, pointNames=pointNames, featureNames=featureNames,
            name=name, keepPoints=keepPoints, keepFeatures=keepFeatures,
            ignoreNonNumericalFeatures=ignoreNonNumericalFeatures, inputSeparator=inputSeparator,
            treatAsMissing=treatAsMissing, replaceMissingWith=replaceMissingWith)
        UML.logger.active.logLoad(returnType, len(ret.points),
                                  len(ret.features), name, path)

        return ret
    # no other allowed inputs
    else:
        msg = "data must contain either raw data or the path to a file to be loaded"
        raise InvalidArgumentType(msg)


def crossValidate(learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', useLog=None,
                  **kwarguments):
    """
    K-fold cross validation.
    Returns mean performance (float) across numFolds folds on a X Y.

    Parameters:

    learnerName (string) - UML compliant algorithm name in the form
    'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

    X (UML.Base subclass) - points/features data

    Y (UML.Base subclass or int index for X) - labels/data about points in X

    performanceFunction (function) - Look in UML.calculate for premade options.
    Function used by computeMetrics to generate a performance score for the run.
    function is of the form: def func(knownValues, predictedValues).

    arguments (dict) - dictionary mapping argument names (strings)
    to their values. The parameter is sent to trainAndApply() through its arguments
    parameter. example: {'dimensions':5, 'k':5}

    numFolds (int) - the number of folds used in the cross validation. Can't
    exceed the number of points in X, Y

    scoreMode - used by computeMetrics

    useLog - local control for whether to send results/timing to the logger.
    If None (default), use the value as specified in the "logger"
    "enabledByDefault" configuration option. If True, send to the logger
    regardless of the global option. If False, do NOT send to the logger,
    regardless of the global option.

    kwarguments - kwargs specified variables that are passed to the learner.
    To make use of multiple permutations, specify different values for a
    parameter as a tuple. eg. a=(1,2,3) will generate an error score for
    the learner when the learner was passed all three values of a, separately.

    """
    if UML.logger.active.position == 0:
        if enableLogging(useLog):
            wrapped = logCapture(crossValidate)
        else:
            wrapped = directCall(crossValidate)
        return wrapped(learnerName, X, Y, performanceFunction, arguments,
                       numFolds, scoreMode, useLog, **kwarguments)

    bestResult = crossValidateReturnBest(learnerName, X, Y, performanceFunction, arguments,
                                         numFolds, scoreMode, useLog, **kwarguments)

    return bestResult[1]

#return crossValidateBackend(learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode, useLog, **kwarguments)


def crossValidateReturnAll(learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label',
                           useLog=None, **kwarguments):
    """
    Calculates the cross validated error for each argument permutation that can
    be generated by the merge of arguments and kwarguments.

    example **kwarguments: {'a':(1,2,3), 'b':(4,5)}
    generates permutations of dict in the format:
    {'a':1, 'b':4}, {'a':2, 'b':4}, {'a':3, 'b':4}, {'a':1, 'b':5},
    {'a':2, 'b':5}, {'a':3, 'b':5}

    For each permutation of 'arguments', crossValidateReturnAll uses cross
    validation to generate a performance score for the algorithm, given the
    particular argument permutation.

    Returns a list of tuples, where every tuple contains a dict representing
    the argument sent to trainAndApply, and a float represennting the cross
    validated error associated with that argument dict.
    example list element: ({'arg1':2, 'arg2':'max'}, 89.0000123)

    Arguments:

    learnerName (string) - UML compliant algorithm name in the form
    'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

    X (UML.Base subclass) - points/features data

    Y (UML.Base subclass or int index for X) - labels/data about points in X

    performanceFunction (function) - Look in UML.calculate for premade options.
    Function used by computeMetrics to generate a performance score for the run.
    function is of the form: def func(knownValues, predictedValues).

    arguments (dict) - dictionary mapping argument names (strings)
    to their values, to be merged with kwargs. To make use of multiple
    permutations, specify different values for a parameter as a tuple. eg.
    a=(1,2,3) will generate an error score for  the learner when the learner
    was passed all three values of a, separately.

    numFolds (int) - the number of folds used in the cross validation. Can't
    exceed the number of points in X, Y

    scoreMode - used by computeMetrics

    useLog - local control for whether to send results/timing to the logger.
    If None (default), use the value as specified in the "logger"
    "enabledByDefault" configuration option. If True, send to the logger
    regardless of the global option. If False, do NOT send to the logger,
    regardless of the global option.

    kwarguments - kwargs specified variables that are passed to the learner,
    after being merged with arguments. To make use of multiple permutations,
    specify different values for a parameter as a tuple. eg. a=(1,2,3) will
    generate an error score for the learner when the learner was passed all
    three values of a, separately.

    """
    if UML.logger.active.position == 0:
        if enableLogging(useLog):
            wrapped = logCapture(crossValidateReturnAll)
        else:
            wrapped = directCall(crossValidateReturnAll)
        return wrapped(learnerName, X, Y, performanceFunction, arguments,
                       numFolds, scoreMode, useLog, **kwarguments)

    ret = crossValidateBackend(learnerName, X, Y, performanceFunction, arguments, numFolds,
                                scoreMode, useLog, **kwarguments)

    return ret


def crossValidateReturnBest(learnerName, X, Y, performanceFunction, arguments={},
                            numFolds=10, scoreMode='label', useLog=None, **kwarguments):
    """
    For each possible argument permutation generated by arguments,
    crossValidateReturnBest runs crossValidate to compute a mean error for the
    argument combination.

    crossValidateReturnBest then RETURNS the best argument and error as a tuple:
    (argument_as_dict, cross_validated_performance_float)

    Arguments:
    learnerName (string) - UML compliant algorithm name in the form
    'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

    X (UML.Base subclass) - points/features data

    Y (UML.Base subclass or int index for X) - labels/data about points in X

    performanceFunction (function) - Look in UML.calculate for premade options.
    Function used by computeMetrics to generate a performance score for the run.
    function is of the form: def func(knownValues, predictedValues).

    arguments (dict) - dictionary mapping argument names (strings)
    to their values, to be merged with kwargs. To make use of multiple
    permutations, specify different values for a parameter as a tuple. eg.
    a=(1,2,3) will generate an error score for  the learner when the learner
    was passed all three values of a, separately.

    numFolds (int) - the number of folds used in the cross validation. Can't
    exceed the number of points in X, Y

    scoreMode - used by computeMetrics

    useLog - local control for whether to send results/timing to the logger.
    If None (default), use the value as specified in the "logger"
    "enabledByDefault" configuration option. If True, send to the logger
    regardless of the global option. If False, do NOT send to the logger,
    regardless of the global option.

    kwarguments - kwargs specified variables that are passed to the learner,
    after being merged with arguments. To make use of multiple permutations,
    specify different values for a parameter as a tuple. eg. a=(1,2,3) will
    generate an error score for the learner when the learner was passed all
    three values of a, separately.

    """
    if UML.logger.active.position == 0:
        if enableLogging(useLog):
            wrapped = logCapture(crossValidateReturnBest)
        else:
            wrapped = directCall(crossValidateReturnBest)
        return wrapped(learnerName, X, Y, performanceFunction, arguments,
                       numFolds, scoreMode, useLog, **kwarguments)

    resultsAll = crossValidateReturnAll(learnerName, X, Y, performanceFunction, arguments,
                                        numFolds, scoreMode, useLog, **kwarguments)

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
        curArgument, curScore = curResultTuple
        #if curArgument is the first or best we've seen:
        #store its details in bestArgumentAndScoreTuple
        if bestArgumentAndScoreTuple is None:
            bestArgumentAndScoreTuple = curResultTuple
        else:
            if (maximumIsBest and curScore > bestArgumentAndScoreTuple[1]):
                bestArgumentAndScoreTuple = curResultTuple
            if ((not maximumIsBest) and curScore < bestArgumentAndScoreTuple[1]):
                bestArgumentAndScoreTuple = curResultTuple

    return bestArgumentAndScoreTuple


def learnerType(learnerNames):
    """
    Returns the string or list of strings representation of a best guess for
    the type of learner(s) specified by the learner name(s) in learnerNames.

    If learnerNames is a single string (not a list of strings), then only a single
    result is returned, instead of a list.

    LearnerType first queries the appropriate interface object for a definitive return
    value. If the interface doesn't provide a satisfactory answer, then this method
    calls a backend which generates a series of artificial data sets with particular
    traits to look for heuristic evidence of a classifier, regressor, etc.
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

    #if only one algo was requested, remove type from list an return as single string
    if len(resultsList) == 1:
        resultsList = resultsList[0]

    return resultsList


def train(learnerName, trainX, trainY=None, performanceFunction=None,
          arguments={}, scoreMode='label', multiClassStrategy='default',
          doneValidData=False, doneValidArguments1=False,
          doneValidArguments2=False, doneValidMultiClassStrategy=False,
          done2dOutputFlagCheck=False, useLog=None, storeLog='unset', **kwarguments):
    """
    Trains and returns the specified learner using the provided data. The return value is a
    UniversalInterface.trainedLearner object.

    ARGUMENTS:

    learnerName: algorithm to be called, in the form 'package.learnerName'.

    trainX: data set to be used for training (as some form of Base object)

    trainY: used to retrieve the known class labels of the traing data. Either
    contains the labels themselves (as a Base object) or an index (numerical or string)
    that defines their locale in the trainX object

    performanceFunction: If cross validation is triggered to select from the given
    argument set, then this function will be used to generate a performance
    score for the run. function is of the form: def func(knownValues, predictedValues).
    Look in UML.calculate for pre-made options. Default is None, since if
    there is no parameter selection to be done, it is not used.

    arguments: dict containing the parameters to be passed to the learner, in the
    form of a mapping between (string) parameter names, and values. Will be merged
    with the contents of **kwarguments before being passed on.

    scoreMode: In the case of a classifying learner, this specifies the type of output
    wanted: 'label' if we class labels are desired, 'bestScore' if both the class
    label and the score associated with that class are desired, or 'allScores' if
    a matrix containing the scores for every class label are desired.

    multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'

    useLog - local control for whether to send results/timing to the logger.
    If None (default), use the value as specified in the "logger"
    "enabledByDefault" configuration option. If True, send to the logger
    regardless of the global option. If False, do NOT send to the logger,
    regardless of the global option.

    kwarguments: The collection of extra key:value argument pairs included in this call to
    train(). They will be merged with the arguments dict, and passed on through to the
    learner.

    """
    if UML.logger.active.position == 0:
        if enableLogging(useLog):
            wrapped = logCapture(train)
        else:
            wrapped = directCall(train)
        return wrapped(learnerName, trainX, trainY, performanceFunction,
                       arguments, scoreMode, multiClassStrategy,
                       doneValidData, doneValidArguments1,
                       doneValidArguments2, doneValidMultiClassStrategy,
                       done2dOutputFlagCheck, useLog=False, storeLog=useLog,
                       **kwarguments)

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
    if storeLog != 'unset':
        useLog = storeLog

    # perform CV (if needed)
    argCheck = ArgumentIterator(merged)
    if argCheck.numPermutations != 1:
        if performanceFunction is None:
            msg = "Cross validation was triggered to select the best parameter "
            msg += "set, yet no performanceFunction was specified. Either one "
            msg += "must be specified (see UML.calculate for out-of-the-box "
            msg += "options) or there must be no choices in the parameters."
            raise InvalidArgumentValueCombination(msg)

        #modify numFolds if needed
        numFolds = len(trainX.points) if len(trainX.points) < 10 else 10
        #sig (learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', useLog=None, maximize=False, **kwarguments):
        bestArgument, bestScore = UML.crossValidateReturnBest(learnerName, trainX, trainY, performanceFunction, merged,
                                                              numFolds=numFolds, scoreMode=scoreMode, useLog=useLog)

    else:
        bestArgument = merged

    interface = findBestInterface(package)

    trainedLearner = interface.train(trueLearnerName, trainX, trainY, multiClassStrategy,
                                     bestArgument, useLog)

    funcString = interface.getCanonicalName() + '.' + trueLearnerName
    UML.logger.active.logRun("train", trainX, trainY, None, None, funcString, bestArgument, None)

    return trainedLearner


def trainAndApply(learnerName, trainX, trainY=None, testX=None,
                  performanceFunction=None, arguments={}, output=None, scoreMode='label',
                  multiClassStrategy='default', useLog=None, storeLog='unset', **kwarguments):
    """
    Trains and returns the results of applying the learner to the test data (i.e.
    performing prediction, transformation, etc. as appropriate to the learner).

    ARGUMENTS:

    learnerName: algorithm to be called, in the form 'package.learnerName'.

    trainX: data set to be used for training (as some form of Base object)

    trainY: used to retrieve the known class labels of the training data. Either
    contains the labels themselves (as a Base object) or an index (numerical or string)
    that defines their locale in the trainX object

    testX: data set on which the trained learner will be applied (i.e. performing
    prediction, transformation, etc. as appropriate to the learner). Must be
    some form of UML data Base object.

    performanceFunction: If cross validation is triggered to select from the given
    argument set, then this function will be used to generate a performance
    score for the run. function is of the form: def func(knownValues, predictedValues).
    Look in UML.calculate for pre-made options. Default is None, since if
    there is no parameter selection to be done, it is not used.

    arguments: dict containing the parameters to be passed to the learner, in the
    form of a mapping between (string) parameter names, and values. Will be merged
    with the contents of **kwarguments before being passed on.

    output: The kind of UML data object that the output of this function should be
    in. Any of the normal string inputs to the createData 'returnType' parameter are
    accepted here. Alternatively, the value 'match' will indicate to use the type
    of the 'trainX' parameter.

    scoreMode: In the case of a classifying learner, this specifies the type of output
    wanted: 'label' if we class labels are desired, 'bestScore' if both the class
    label and the score associated with that class are desired, or 'allScores' if
    a matrix containing the scores for every class label are desired.

    multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'

    useLog - local control for whether to send results/timing to the logger.
    If None (default), use the value as specified in the "logger"
    "enabledByDefault" configuration option. If True, send to the logger
    regardless of the global option. If False, do NOT send to the logger,
    regardless of the global option.

    kwarguments: The collection of extra key:value argument pairs included in this call to
    train(). They will be merged with the arguments dict, and passed on through to the
    learner.

    """
    if UML.logger.active.position == 0:
        if enableLogging(useLog):
            wrapped = logCapture(trainAndApply)
        else:
            wrapped = directCall(trainAndApply)
        return wrapped(learnerName, trainX, trainY, testX, performanceFunction,
                       arguments, output, scoreMode, multiClassStrategy,
                       useLog=False, storeLog=useLog, **kwarguments)

    _validData(trainX, trainY, testX, None, [False, False])
    _validScoreMode(scoreMode)
    _2dOutputFlagCheck(trainX, trainY, scoreMode, multiClassStrategy)

    if testX is None:
        testX = trainX
    if storeLog != 'unset':
        useLog = storeLog

    trainedLearner = UML.train(learnerName, trainX, trainY, performanceFunction, arguments, \
                               scoreMode='label', multiClassStrategy=multiClassStrategy, useLog=useLog, \
                               doneValidData=True, done2dOutputFlagCheck=True, **kwarguments)
    results = trainedLearner.apply(testX, {}, output, scoreMode, useLog=useLog)

    merged = _mergeArguments(arguments, kwarguments)
    extraInfo = None
    if merged != trainedLearner.arguments:
        extraInfo = {"bestParams": trainedLearner.arguments}

    UML.logger.active.logRun("trainAndApply", trainX, trainY, testX, None,
                             learnerName,
                             merged, None, extraInfo=extraInfo)

    return results


def trainAndTest(learnerName, trainX, trainY, testX, testY,
                 performanceFunction, arguments={}, output=None,
                 scoreMode='label', multiClassStrategy='default', useLog=None,
                 storeLog='unset', **kwarguments):
    """
    For each permutation of the merge of 'arguments' and 'kwarguments' (more below),
    trainAndTest uses cross validation to generate a performance score for the algorithm,
    given the particular argument permutation. The argument permutation that performed
    best cross validating over the training data is then used as the lone argument for
    training on the whole training data set. Finally, the learned model generates
    predictions for the testing set, an the performance of those predictions is
    calculated and returned.

    If no additional arguments are supplied via arguments or **kwarguments, then
    trainAndTest just returns the performance of the algorithm with default arguments
    on the testing data.

    ARGUMENTS:

    learnerName: training algorithm to be called, in the form 'package.algorithmName'.

    trainX: data set to be used for training (as some form of Base object)

    trainY: used to retrieve the known class labels of the training data. Either
    contains the labels themselves (as a Base object) or an index (numerical or string)
    that defines their locale in the trainX object

    testX: data set to be used for testing (as some form of Base object)

    testY: used to retrieve the known class labels of the test data. Either
    contains the labels themselves (as a Base object) or an index (numerical or string)
    that defines their location in the testX object.

    performanceFunction: Function used by computeMetrics to generate a performance score
    for the run. function is of the form: def func(knownValues, predictedValues).
    Look in UML.calculate for pre-made options.

    arguments: dict containing the parameters to be passed to the learner, in the
    form of a mapping between (string) parameter names, and values. Will be merged
    with the contents of **kwarguments before being passed on. The syntax for prescribing
    different arguments for algorithm: arguments of the form {arg1=(1,2,3), arg2=(4,5,6)}
    correspond to permutations/argument states with one element from arg1 and one element
    from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

    multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'

    useLog - local control for whether to send results/timing to the logger.
    If None (default), use the value as specified in the "logger"
    "enabledByDefault" configuration option. If True, send to the logger
    regardless of the global option. If False, do NOT send to the logger,
    regardless of the global option.

    kwarguments: optional arguments to be passed to the specified learner. Will be merged
    with the arguments parameter before being passed on to the learner.
    The syntax for prescribing different arguments for algorithm:
    **kwarguments of the form arg1=(1,2,3), arg2=(4,5,6)
    correspond to permutations/argument states with one element from arg1 and one element
    from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

    """
    if UML.logger.active.position == 0:
        if enableLogging(useLog):
            wrapped = logCapture(trainAndTest)
        else:
            wrapped = directCall(trainAndTest)
        return wrapped(learnerName, trainX, trainY, testX, testY,
                       performanceFunction, arguments, output, scoreMode,
                       multiClassStrategy, useLog=False, storeLog=useLog,
                       **kwarguments)

    _2dOutputFlagCheck(trainX, trainY, scoreMode, None)#this line is for
    # UML.tests.test_train_apply_test_frontends.test_trainAndTest_scoreMode_disallowed_multioutput
    # and UML.tests.test_train_apply_test_frontends.test_trainAndTestOnTrainingData_scoreMode_disallowed_multioutput

    trainY = copyLabels(trainX, trainY)
    testY = copyLabels(testX, testY)

    if storeLog != 'unset':
        useLog = storeLog
    trainedLearner = UML.train(learnerName, trainX, trainY, performanceFunction, arguments, \
                               scoreMode='label', multiClassStrategy=multiClassStrategy, useLog=useLog, \
                               doneValidData=True, done2dOutputFlagCheck=True, **kwarguments)
    predictions = trainedLearner.apply(testX, {}, output, scoreMode, useLog=useLog)
    performance = computeMetrics(testY, None, predictions, performanceFunction)

    metrics = {}
    for key, value in zip([performanceFunction], [performance]):
        metrics[key.__name__] = value
    merged = _mergeArguments(arguments, kwarguments)
    extraInfo = None
    if merged != trainedLearner.arguments:
        extraInfo = {"bestParams": trainedLearner.arguments}
    if trainX == testX:
        name = "trainAndTestOnTrainingData"
    else:
        name = "trainAndTest"
    UML.logger.active.logRun(name, trainX, trainY, testX, testY, learnerName,
                             merged, metrics, extraInfo=extraInfo)

    return performance


def trainAndTestOnTrainingData(learnerName, trainX, trainY, performanceFunction,
                               crossValidationError=False, numFolds=10, arguments={}, output=None,
                               scoreMode='label', multiClassStrategy='default', useLog=None, **kwarguments):
    """
    trainAndTestOnTrainingData is the function for doing learner creation
    and evaluation in a single step with only a single data set (no withheld
    testing set). By default, this will calculate training error for the
    learner trained on that data set. However, cross validation error can
    instead be calculated by setting the parameter crossVadiationError to be
    true. In that case, we will partition the training set into a parameter
    controlled number of folds, and iteratively withhold each single fold to be
    used as the testing set of the learner trained on the rest of the data.

    ARGUMENTS:

    learnerName: training algorithm to be called, in the form 'package.algorithmName'.

    trainX: data set to be used for training (as some form of Base object)

    trainY: used to retrieve the known class labels of the training data. Either
    contains the labels themselves (as a Base object) or an index (numerical or string)
    that defines their locale in the trainX object

    performanceFunction: Function used by computeMetrics to generate a performance score
    for the run. function is of the form: def func(knownValues, predictedValues).
    Look in UML.calculate for pre-made options.

    crossValidationError: True or False, according to whether we will calculate
    cross validation error or training error. In True case, the training data
    is split in the numFolds number of partitions. Each of those is iteratively
    withheld and used as the testing set for a learner trained on the combination
    of all of the non-withheld data. The performance results for each of those
    tests are then averaged together to act as the return value. In the False
    case, we train on the training data, and then use the same data as the
    withheld testing data. By default, this flag is set to False.

    numFolds: the (int) number of folds used in the cross validation. Can't
    exceed the number of points in X, Y. Default is 10.

    arguments: dict containing the parameters to be passed to the learner, in the
    form of a mapping between (string) parameter names, and values. Will be merged
    with the contents of **kwarguments before being passed on. The syntax for prescribing
    different arguments for algorithm: arguments of the form {arg1=(1,2,3), arg2=(4,5,6)}
    correspond to permutations/argument states with one element from arg1 and one element
    from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

    output: Can be used to force a format on the data object resulting from applying
    the learner to whatever is considered the testing data. This object is not seen
    by the user, but will be used by the input performanceFunction, and if the user
    has prior knowledge of requirements of that function, then they can enforced
    by UML instead of manually. By default, this parameter is set to None, indicating
    to match the formatting of the training data objects.

    multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'

    useLog - local control for whether to send results/timing to the logger.
    If None (default), use the value as specified in the "logger"
    "enabledByDefault" configuration option. If True, send to the logger
    regardless of the global option. If False, do NOT send to the logger,
    regardless of the global option.

    kwarguments: optional arguments to be passed to the specified learner. Will be merged
    with the arguments parameter before being passed on to the learner.
    The syntax for prescribing different arguments for algorithm:
    **kwarguments of the form arg1=(1,2,3), arg2=(4,5,6)
    correspond to permutations/argument states with one element from arg1 and one element
    from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

    """
    performance = trainAndTest(learnerName, trainX, trainY, trainX, trainY, performanceFunction,
                               arguments, output, scoreMode, multiClassStrategy,
                               useLog)
    return performance


def log(logType, logInfo):
    """
    log will enter a log entry into the active logger's database files. The log entry will
    include a timestamp and a run number in addition to the logType and logInfo

    ARGUMENTS:
    logType: A string of the type of log entered. "load", "prep", "run", and "crossVal" types
             have builtin processing for logInfo. A default processing of logInfo will be used
             for unrecognized types
    logInfo: A python string, list or dictionary containing any information to be logged
    """
    if not isinstance(logType, six.string_types):
        msg = "logType must be a string"
        raise ArgumentException(msg)
    elif not isinstance(logInfo, (six.string_types, list, dict)):
        msg = "logInfo must be a python string, list, or dictionary type"
        raise ArgumentException(msg)
    UML.logger.active.log(logType, logInfo)


def showLog(levelOfDetail=2, leastRunsAgo=0, mostRunsAgo=2, startDate=None, endDate=None,
            maximumEntries=100, searchForText=None, regex=False, saveToFileName=None, append=False):
        """
        showLog parses the active logfile based on the arguments passed and prints a
        human readable interpretation of the log file.

        ARGUMENTS:
        levelOfDetail:  The (int) value for the level of detail from 1, the least detail,
                        to 3 (most detail). Default is 2
              Level 1:  Data loading, data preparation and preprocessing, custom user logs
              Level 2:  Outputs basic information about each run. Includes timestamp, run number,
                        learner name, train and test object details, parameter, metric, and
                        timer data if available
              Level 3:  Cross Validation data

        leastRunsAgo:   The integer value for the least number of runs since the most recent
                        run to include in the log. Default is 0

        mostRunsAgo:    The integer value for the least number of runs since the most recent
                        run to include in the log. Default is 2

        startDate:      A string or datetime.datetime object of the date to begin adding runs to the log.
                        Acceptable formats:
                          "YYYY-MM-DD"
                          "YYYY-MM-DD HH:MM"
                          "YYYY-MM-DD HH:MM:SS"

        endDate:        A string or datetime.datetime object of the date to stop adding runs to the log.
                        See startDate for formatting.

        maximumEntries: Maximum number of entries to allow before stopping the log
                        Default is 100. None will allow all entries provided from the query
        searchForText:  string or regular expression to search for in each log entry.
                        Default is None

        saveToFileName: The name of a file where the human readable log will be saved.
                        Default is None, showLog will print to standard out

        append:         Append logs to the file in saveToFileName instead of overwriting file.
                        Default is False
        """
        if levelOfDetail < 1 or levelOfDetail > 3 or levelOfDetail is None:
            msg = "levelOfDetail must be 1, 2, or 3"
            raise ArgumentException(msg)
        if startDate is not None and endDate is not None and startDate > endDate:
            startDate = parse(startDate)
            endDate = parse(endDate)
            msg = "The startDate must be before the endDate"
            raise ArgumentException(msg)
        if leastRunsAgo is not None:
            if leastRunsAgo < 0:
                msg = "leastRunsAgo must be greater than zero"
                raise ArgumentException(msg)
            if mostRunsAgo is not None and mostRunsAgo < leastRunsAgo:
                msg = "mostRunsAgo must be greater than or equal to leastRunsAgo"
                raise ArgumentException(msg)
        UML.logger.active.showLog(levelOfDetail, leastRunsAgo, mostRunsAgo, startDate, endDate,
                                  maximumEntries, searchForText, regex, saveToFileName, append)


def loadData(inputPath):
    """
    Load UML data object.

    inputPath: the location (including file name and extension) where
        to find file previously generated by UML.data .save().

    Expected file extension '.umld'.
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
    return ret


def loadTrainedLearner(inputPath):
    """
    Load UML trainedLearner object.

    inputPath: the location (including file name and extension) where
        to find file previously generated for a trainedLearner object.

    Expected file extension '.umlm'.
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
    return ret


def coo_matrixTodense(origTodense):
    """
    decorator for coo_matrix.todense
    """
    def f(self):
        try:
            return origTodense(self)
        except Exception:
            # flexible dtypes, such as strings, when used in scipy sparse object
            # create an implicitly mixed datatype: some values are strings, but
            # the rest are implicitly zero. In order to match that, we must
            # explicitly specify a mixed type for our destination matrix
            retDType = object if isinstance(self.dtype, numpy.flexible) else self.dtype
            ret = numpy.matrix(numpy.zeros(self.shape), dtype=retDType)
            nz = (self.row, self.col)
            for (i, j), v in zip(zip(*nz), self.data):
                ret[i, j] = v
            return ret
    return f

if scipy:
    #monkey patch for coo_matrix.todense
    scipy.sparse.coo_matrix.todense = coo_matrixTodense(scipy.sparse.coo_matrix.todense)
