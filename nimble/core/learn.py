"""
Module containing the user-facing learner functions for the top level
nimble import.
"""

from types import ModuleType
import time

import numpy as np

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentValue, ImproperObjectAction
from nimble.exceptions import InvalidArgumentValueCombination
from nimble._utility import mergeArguments, tableString
from nimble.core.logger import handleLogging, loggingEnabled
from nimble.core.logger import deepLoggingEnabled
from nimble.core._learnHelpers import findBestInterface
from nimble.core._learnHelpers import _learnerQuery
from nimble.core._learnHelpers import _unpackLearnerName
from nimble.core._learnHelpers import validateLearningArguments
from nimble.core._learnHelpers import trackEntry
from nimble.core._learnHelpers import LearnerInspector
from nimble.core._learnHelpers import ArgumentIterator
from nimble.core._learnHelpers import FoldIterator
from nimble.core._learnHelpers import computeMetrics
from nimble.core._learnHelpers import initAvailablePredefinedInterfaces


def learnerType(learnerNames): # pylint: disable=redefined-outer-name
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
        splitTuple = _unpackLearnerName(name)
        currInterface = splitTuple[0]
        allValidLearnerNames = currInterface.learnerNames()
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

    for index, curLearnerName in enumerate(secondPassLearnerNames):
        if curLearnerName is None:
            continue
        resultsList[index] = learnerInspectorObj.learnerType(curLearnerName)

    #if only one algo was requested, return as single string
    if len(resultsList) == 1:
        resultsList = resultsList[0]

    return resultsList


def learnerNames(package=None):
    """
    Get a list of learners available to nimble or a specific package.

    Returns a list of learners that are callable through nimble's
    training, applying, and testing functions. If ``package`` is
    specified, the list will contain strings of each learner. If
    ``package`` is None, the list will contain strings in the form of
    'package.learner'. This will differ depending on the packages
    currently available in nimble.core.interfaces.available.

    Parameters
    ----------
    package : str
        The name of the package to list the learners from. If None, each
        learners available to each interface will be listed.

    Returns
    -------
    list
    """
    if isinstance(package, ModuleType):
        package = package.__name__
    results = []
    if package is None:
        initAvailablePredefinedInterfaces()
        for packageName, interface in nimble.core.interfaces.available.items():
            currResults = interface.learnerNames()
            for learnerName in currResults:
                results.append(packageName + "." + learnerName)
    else:
        interface = findBestInterface(package)
        currResults = interface.learnerNames()
        for learnerName in currResults:
            results.append(learnerName)

    return results

def showLearnerNames(package=None):
    """
    Print the learners available to nimble or a specific package.

    Prints a list of learners that are callable through nimble's
    training, applying, and testing functions. If ``package`` is
    specified, the list will contain strings of each learner. If
    ``package`` is None, the list will contain strings in the form of
    'package.learner'. This will differ depending on the packages
    currently available in nimble.core.interfaces.available.

    Parameters
    ----------
    package : str
        The name of the package to list the learners from. If None, each
        learners available to each interface will be listed.
    """
    for name in learnerNames(package):
        print(name)

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

def showLearnerParameters(name):
    """
    Print a list of parameters for the learner.

    Prints a list of strings which are the names of the parameters when
    calling this learner. If the name cannot be found within the
    package, then an exception will be thrown.

    Parameters
    ----------
    name : str
        Package and name in the form 'package.learnerName'.
    """
    params = learnerParameters(name)
    if params is None:
        print('learner parameters could not be determined')
    elif params:
        for param in params:
            print(param)

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

def showLearnerDefaultValues(name):
    """
    Get a dictionary mapping parameter names to their default values.

    Returns a dictionary with strings of the parameter names as keys and
    the parameter's default value as values. If the name cannot be found
    within the package, then an exception will be thrown.

    Parameters
    ----------
    name : str
        Package and name in the form 'package.learnerName'.
    """
    defaultValues = learnerDefaultValues(name).items()
    if defaultValues is None:
        print('learner default values could not be determined')
    elif defaultValues:
        defaults = []
        for param, default in defaultValues:
            if isinstance(default, str):
                default = "'{}'".format(default)
            defaults.append([param, default])
        print(tableString(defaults, rowHeadJustify='left',
                          colValueJustify='left'))

@trackEntry
def normalizeData(learnerName, trainX, trainY=None, testX=None, arguments=None,
                  randomSeed=None, useLog=None, **kwarguments):
    """
    Modify data according to a produced model.

    Calls on the functionality of a package to train on some data and
    then return the modified ``trainX`` and ``testX`` (if provided)
    according to the results of the trained model. If only ``trainX`` is
    proved, the normalized ``trainX`` is returned. If ``testX`` is also
    provided a tuple (normalizedTrain, normalizedTest) is returned. The
    name of the learner will be added to each normalized object's
    ``name`` attribute to indicate the normalization that has been
    applied. Point and feature names are preserved when possible.

    Parameters
    ----------
    learnerName : str
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX: nimble Base object
        Data to be used for training.
    trainY: identifier, nimble Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another nimble Base object containing the labels that
        correspond to ``trainX``.
    testX: nimble Base object
        Data to be used for testing.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application.
        Example: {'dimensions':5, 'k':5}
        If an argument requires its own parameters for instantiation,
        use a nimble.Init object.
        Example: {'kernel':nimble.Init('KernelGaussian', width=2.0)}.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed. Ignored if learner does not
       depend on randomness.
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
    nimble.core.data.Features.normalize

    Examples
    --------
    Normalize a single data set.

    >>> data = [[20, 1.97, 89], [28, 1.87, 75], [24, 1.91, 81]]
    >>> trainX = nimble.data("Matrix", data, pointNames=['a', 'b', 'c'],
    ...                      featureNames=['age', 'height', 'weight'])
    >>> normTrainX = nimble.normalizeData('scikitlearn.StandardScaler',
    ...                                   trainX)
    >>> normTrainX
    Matrix(
        [[-1.225 1.298  1.279 ]
         [1.225  -1.136 -1.162]
         [0.000  -0.162 -0.116]]
        pointNames={'a':0, 'b':1, 'c':2}
        featureNames={'age':0, 'height':1, 'weight':2}
        )

    Normalize training and testing data.

    >>> data1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    >>> trainX = nimble.data("Matrix", data1)
    >>> data2 = [[-1, 0, 5]]
    >>> testX = nimble.data("Matrix", data2)
    >>> pcaTrain, pcaTest = nimble.normalizeData('scikitlearn.PCA',
    ...                                          trainX, testX=testX,
    ...                                          n_components=2)
    >>> pcaTrain
    Matrix(
        [[-0.216 0.713 ]
         [-1.005 -0.461]
         [1.221  -0.253]]
        )
    >>> pcaTest
    Matrix(
        [[-1.739 2.588]]
        )
    """
    startTime = time.process_time()
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, trainY, testX, False)
    merged = mergeArguments(arguments, kwarguments)

    tl = train(learnerName, trainX, trainY, arguments=merged,
               randomSeed=randomSeed, useLog=False)
    normalizedTrain = tl.apply(trainX, useLog=False)

    if normalizedTrain.getTypeString() != trainX.getTypeString():
        normalizedTrain = normalizedTrain.copy(to=trainX.getTypeString())

    if len(normalizedTrain.features) == len(trainX.features):
        trainXFtNames = trainX.features._getNamesNoGeneration()
        normalizedTrain.features.setNames(trainXFtNames, useLog=False)

    # return normalized trainX when testX is not included otherwise return will
    # be a tuple (normalizedTrain, normalizedTest)
    if testX is None:
        ret = normalizedTrain
    else:
        normalizedTest = tl.apply(testX, useLog=False)
        if normalizedTest.getTypeString() != testX.getTypeString():
            normalizedTest = normalizedTest.copy(to=testX.getTypeString())
        if len(normalizedTest.features) == len(testX.features):
            testXFtNames = testX.features._getNamesNoGeneration()
            normalizedTest.features.setNames(testXFtNames, useLog=False)

        ret = (normalizedTrain, normalizedTest)

    totalTime = time.process_time() - startTime
    handleLogging(useLog, 'run', "normalizeData", trainX, trainY, testX, None,
                  learnerName, merged, tl.randomSeed, time=totalTime)

    return ret

@trackEntry
def fillMatching(learnerName, matchingElements, trainX, arguments=None,
                 points=None, features=None, useLog=None, **kwarguments):
    """
    Fill matching values using imputation learners.

    Transform the data in the ``trainX`` object to replace all matching
    values with values calculated by the learner.

    Parameters
    ----------
    learnerName : str
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX: nimble Base object
        Data to be used for training.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application.
        Example: {'dimensions': 5, 'k': 5}
        If an argument requires its own parameters for instantiation,
        use a nimble.Init object.
        Example: {'kernel':nimble.Init('KernelGaussian', width=2.0)}.
    points : identifier, list of identifiers, None
        May be a single point name or index, an iterable,
        container of point names and/or indices. None indicates
        application to all points, otherwise only the matching values in
        the specified points will be modified.
    features : identifier, list of identifiers, None
        May be a single feature name or index, an iterable,
        container of feature names and/or indices. None indicates
        application to all features, otherwise only the matching values
        in the specified features will be modified.
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
    nimble.core.data.Points.fillMatching,
    nimble.core.data.Features.fillMatching

    Examples
    --------
    Fill missing values based on k-nearest neighbors classifier.

    >>> raw = [[1, None, None],
    ...        [1, 3, 6],
    ...        [2, 1, 6],
    ...        [1, 3, 7],
    ...        [None, 3, None]]
    >>> data = nimble.data('Matrix', raw)
    >>> toMatch = nimble.match.missing
    >>> nimble.fillMatching('nimble.KNNImputation', toMatch, data,
    ...                     mode='classification', k=3)
    >>> data
    Matrix(
        [[1.000 3.000 6.000]
         [1.000 3.000 6.000]
         [2.000 1.000 6.000]
         [1.000 3.000 7.000]
         [1.000 3.000 6.000]]
        )

    Fill last feature zeros based on k-nearest neighbors regressor.

    >>> raw = [[1, 0, 0],
    ...        [1, 3, 6],
    ...        [2, 1, 6],
    ...        [1, 3, 7],
    ...        [0, 3, 0]]
    >>> data = nimble.data('Sparse', raw)
    >>> toMatch = nimble.match.zero
    >>> nimble.fillMatching('nimble.KNNImputation', toMatch, data,
    ...                     features=-1, k=3, mode='regression')
    >>> data
    Sparse(
        [[1.000 0.000 6.333]
         [1.000 3.000 6.000]
         [2.000 1.000 6.000]
         [1.000 3.000 7.000]
         [0.000 3.000 6.333]]
        )
    """
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, arguments=arguments)
    startTime = time.process_time()
    interface, objectName = _unpackLearnerName(learnerName)
    merged = mergeArguments(arguments, kwarguments)
    if interface.isAlias('sklearn') and 'missing_values' in merged:
        msg = 'The missing_values argument for {objectName} is disallowed '
        msg += 'because nimble handles alternative values via its '
        msg += 'matchingElements argument'
        raise InvalidArgumentValue(msg.format(objectName=objectName))
    checkNans = True
    if isinstance(matchingElements, nimble.core.data.Base):
        if matchingElements.shape != trainX.shape:
            msg = 'The shape of matchingElements and trainX must match'
            raise InvalidArgumentValue(msg)
        matchMatrix = matchingElements
    else:
        matchingElements = match._convertMatchToFunction(matchingElements)
        matchMatrix = trainX.matchingElements(matchingElements, useLog=False)
        if matchingElements(np.nan):
            checkNans = False
    if checkNans:
        nanLocs = trainX.matchingElements(match.missing, useLog=False)
        if matchMatrix | nanLocs != matchMatrix:
            msg = "filling requires all unmatched elements to be non-nan"
            raise ImproperObjectAction(msg)

    # do not fill actual trainX with nans in case trainAndApply fails
    toFill = trainX.copy()
    toFill.features.fillMatching(np.nan, matchMatrix, useLog=False)
    filled = trainAndApply(learnerName, toFill, arguments=merged, useLog=False)

    def transformer(elem, i, j):
        if matchMatrix[i, j]:
            return filled[i, j]
        return elem

    trainX.transformElements(transformer, points, features, useLog=False)

    totalTime = time.process_time() - startTime
    handleLogging(useLog, 'run', "fillMatching", trainX, None, None, None,
                  learnerName, merged, time=totalTime)


def crossValidate(learnerName, X, Y, performanceFunction, arguments=None,
                  folds=10, scoreMode='label', randomSeed=None, useLog=None,
                  **kwarguments):
    """
    Perform K-fold cross validation.

    The object returned provides access to the results. All results, the
    best set of arguments and the best result can be accessed through
    its ``allResults``, ``bestArguments`` and ``bestResult`` attributes,
    respectively.

    Parameters
    ----------
    learnerName : str
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    X : nimble Base object
        points/features data
    Y : nimble Base object
        labels/data about points in X
    performanceFunction : function
        Premade options are available in nimble.calculate.
        Function used to evaluate the performance score for each run.
        Function is of the form: def func(knownValues, predictedValues).
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To trigger cross-validation using multiple values for arguments,
        specify different values for each parameter using a nimble.CV
        object. eg. {'k': nimble.CV([1,3,5])} will generate an error
        score for  the learner when the learner was passed all three
        values of ``k``, separately. If an argument requires its own
        parameters for instantiation, use a nimble.Init object.
        eg. {'kernel':nimble.Init('KernelGaussian', width=2.0)}.
        ``arguments`` will be merged with the learner ``kwarguments`` .
    folds : int
        The number of folds used in the cross validation. Can't exceed
        the number of points in X, Y.
    scoreMode : str
        Used by computeMetrics.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed. Ignored if learner does not
       depend on randomness.
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
        nimble.CV object. eg. arg1=nimble.CV([1,2,3]),
        arg2=nimble.CV([4,5,6]) which correspond to permutations/
        argument states with one element from arg1 and one element from
        arg2, such that an example generated permutation/argument state
        would be ``arg1=2, arg2=4``. Will be merged with ``arguments``.

    Returns
    -------
    KFoldCrossValidator
        Object which performs the cross-validation and provides the
        results which can be accessed through the object's attributes
        and methods.

    See Also
    --------
    nimble.core.learn.KFoldCrossValidator

    Examples
    --------
    >>> nimble.random.setSeed(42)
    >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
    >>> yRaw = [[1], [2], [3],
    ...         [1], [2], [3],
    ...         [1], [2], [3]]
    >>> X = nimble.data('Matrix', xRaw)
    >>> Y = nimble.data('Matrix', yRaw)
    >>> crossValidator = nimble.crossValidate(
    ...    'nimble.KNNClassifier', X, Y,
    ...    performanceFunction=nimble.calculate.fractionIncorrect,
    ...    folds=3, k=3)
    >>> type(crossValidator)
    <class 'nimble.core.learn.KFoldCrossValidator'>
    >>> crossValidator.learnerName
    'nimble.KNNClassifier'
    >>> crossValidator.folds
    3
    >>> crossValidator.bestArguments
    {'k': 3}
    >>> crossValidator.bestResult
    0.3333333333333333
    >>> crossValidator.getFoldResults(crossValidator.bestArguments)
    [0.3333333333333333, 0.6666666666666666, 0.0]
    >>> crossValidator.allResults
    [{'k': 3, 'fractionIncorrect': 0.3333333333333333}]
    """
    if trackEntry.isEntryPoint:
        validateLearningArguments(X, Y, arguments=arguments,
                                  scoreMode=scoreMode)
    kfcv = KFoldCrossValidator(learnerName, X, Y, performanceFunction,
                               arguments, folds, scoreMode, randomSeed,
                               useLog, **kwarguments)
    handleLogging(useLog, 'crossVal', kfcv.learnerName, kfcv.arguments,
                  kfcv.performanceFunction, kfcv._allResults, kfcv.folds,
                  kfcv.randomSeed)

    return kfcv

@trackEntry
def train(learnerName, trainX, trainY=None, performanceFunction=None,
          arguments=None, scoreMode='label', multiClassStrategy='default',
          folds=10, randomSeed=None, useLog=None, **kwarguments):
    """
    Train a specified learner using the provided data.

    Parameters
    ----------
    learnerName : str
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX: nimble Base object
        Data to be used for training.
    trainY: identifier, nimble Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another nimble Base object containing the labels that
        correspond to ``trainX``.
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in nimble.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To trigger cross-validation using multiple values for arguments,
        specify different values for each parameter using a nimble.CV
        object. eg. {'k': nimble.CV([1,3,5])} will generate an error
        score for  the learner when the learner was passed all three
        values of ``k``, separately. If an argument requires its own
        parameters for instantiation, use a nimble.Init object.
        eg. {'kernel':nimble.Init('KernelGaussian', width=2.0)}.
        ``arguments`` will be merged with the learner ``kwarguments``
    scoreMode : str
        In the case of a classifying learner, this specifies the type of
        output wanted: 'label' if we class labels are desired,
        'bestScore' if both the class label and the score associated
        with that class are desired, or 'allScores' if a matrix
        containing the scores for every class label are desired.
    multiClassStrategy : str
        May only be 'default', 'OneVsAll' or 'OneVsOne'.
    folds : int
        The number of folds used in the cross validation. Cannot exceed
        the number of points in ``trainX``. Default 10.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed. Ignored if learner does not
       depend on randomness.
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
        nimble.CV object. eg. arg1=nimble.CV([1,2,3]),
        arg2=nimble.CV([4,5,6]) which correspond to permutations/
        argument states with one element from arg1 and one element from
        arg2, such that an example generated permutation/argument state
        would be ``arg1=2, arg2=4``. Will be merged with ``arguments``.

    Returns
    -------
    TrainedLearner

    See Also
    --------
    trainAndApply, trainAndTest, trainAndTestOnTrainingData, CV,
    nimble.interfaces.TrainedLearner

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
    >>> trainData = nimble.data('Matrix', data, featureNames=ftNames)
    >>> tl = nimble.train('nimble.KNNClassifier', trainX=trainData,
    ...                   trainY='label')
    >>> print(type(tl))
    <class 'nimble.core.interfaces.universal_interface.TrainedLearner'>

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
    >>> trainX = nimble.data('Matrix', dataX)
    >>> trainY = nimble.data('Matrix', dataY)
    >>> tl = nimble.train('sciKitLearn.SVC', trainX=trainX, trainY=trainY,
    ...                   arguments={'C': 0.1}, kernel='linear')
    >>> tlAttributes = tl.getAttributes()
    >>> cValue = tlAttributes['C']
    >>> kernelValue = tlAttributes['kernel']
    >>> print(cValue, kernelValue)
    0.1 linear
    """
    startTime = time.process_time()
    crossValLog = useLog
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, trainY, arguments=arguments,
                                  scoreMode=scoreMode,
                                  multiClassStrategy=multiClassStrategy)
    else:
        useLog = False

    merged = mergeArguments(arguments, kwarguments)
    interface, trueLearnerName = _unpackLearnerName(learnerName)

    # perform CV (if needed)
    argCheck = ArgumentIterator(merged)
    if argCheck.numPermutations != 1:
        if performanceFunction is None:
            msg = "Cross validation was triggered to select the best "
            msg += "parameter set, yet no performanceFunction was specified. "
            msg += "Either one must be specified (see nimble.calculate for "
            msg += "out-of-the-box options) or there must be no choices in "
            msg += "the parameters."
            raise InvalidArgumentValueCombination(msg)
        if folds > len(trainX.points):
            msg = "There must be a minimum of one fold per point in the data."
            msg += "Cross validation was triggered to select the best "
            msg += "parameter set but the 'folds' parameter was set to "
            msg += str(folds) + " and trainX only contains "
            msg += str(len(trainX.points)) + " points."
            raise InvalidArgumentValueCombination(msg)
        crossValidationResults = crossValidate(
            learnerName, trainX, trainY, performanceFunction, merged,
            folds=folds, scoreMode=scoreMode, randomSeed=randomSeed,
            useLog=crossValLog)
        bestArguments = crossValidationResults.bestArguments
    else:
        crossValidationResults = None
        bestArguments = merged

    trainedLearner = interface.train(trueLearnerName, trainX, trainY,
                                     bestArguments, multiClassStrategy,
                                     randomSeed, crossValidationResults)
    totalTime = time.process_time() - startTime

    funcString = interface.getCanonicalName() + '.' + trueLearnerName
    handleLogging(useLog, "run", "train", trainX, trainY, None, None,
                  funcString, bestArguments, trainedLearner.randomSeed,
                  time=totalTime)

    return trainedLearner

@trackEntry
def trainAndApply(learnerName, trainX, trainY=None, testX=None,
                  performanceFunction=None, arguments=None, output=None,
                  scoreMode='label', multiClassStrategy='default',
                  folds=10, randomSeed=None, useLog=None, **kwarguments):
    """
    Train a model and apply it to the test data.

    The learner will be trained using the training data, then
    prediction, transformation, etc. as appropriate to the learner will
    be applied to the test data and returned.

    Parameters
    ----------
    learnerName : str
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX: nimble Base object
        Data to be used for training.
    trainY: identifier, nimble Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another nimble Base object containing the labels that
        correspond to ``trainX``.
    testX : nimble Base object
        Data set on which the trained learner will be applied (i.e.
        performing prediction, transformation, etc. as appropriate to
        the learner).
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in nimble.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To trigger cross-validation using multiple values for arguments,
        specify different values for each parameter using a nimble.CV
        object. eg. {'k': nimble.CV([1,3,5])} will generate an error
        score for  the learner when the learner was passed all three
        values of ``k``, separately. If an argument requires its own
        parameters for instantiation, use a nimble.Init object.
        eg. {'kernel':nimble.Init('KernelGaussian', width=2.0)}.
        ``arguments`` will be merged with the learner ``kwarguments``
    output : str
        The kind of nimble Base object that the output of this function
        should be in. Any of the normal string inputs to the nimble.data
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
        May only be 'default', 'OneVsAll' or 'OneVsOne'.
    folds : int
        The number of folds used in the cross validation. Cannot exceed
        the number of points in ``trainX``. Default 10.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed. Ignored if learner does not
       depend on randomness.
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
        nimble.CV object. eg. arg1=nimble.CV([1,2,3]),
        arg2=nimble.CV([4,5,6]) which correspond to permutations/
        argument states with one element from arg1 and one element from
        arg2, such that an example generated permutation/argument state
        would be ``arg1=2, arg2=4``. Will be merged with ``arguments``.

    Returns
    -------
    results
        The resulting output of applying learner.

    See Also
    --------
    train, trainAndTest, trainAndTestOnTrainingData, CV,
    nimble.core.interfaces.TrainedLearner.apply

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
    >>> trainData = nimble.data('Matrix', rawTrain)
    >>> testX = nimble.data('Matrix', rawTestX)
    >>> predict = nimble.trainAndApply('nimble.KNNClassifier',
    ...                                trainX=trainData, trainY=3,
    ...                                testX=testX)
    >>> predict
    Matrix(
        [[1]
         [2]
         [3]]
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
    >>> trainX = nimble.data('Matrix', rawTrainX)
    >>> trainY = nimble.data('Matrix', rawTrainY)
    >>> testX = nimble.data('Matrix', rawTestX)
    >>> pred = nimble.trainAndApply('sciKitLearn.SVC', trainX=trainX,
    ...                             trainY=trainY, testX=testX,
    ...                             arguments={'C': 0.1}, kernel='linear')
    >>> pred
    Matrix(
        [[1]
         [2]
         [3]]
        )
    """
    startTime = time.process_time()
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, trainY, testX, arguments=arguments,
                                  scoreMode=scoreMode,
                                  multiClassStrategy=multiClassStrategy)
    merged = mergeArguments(arguments, kwarguments)

    trainedLearner = train(learnerName, trainX, trainY, performanceFunction,
                           merged, scoreMode='label',
                           multiClassStrategy=multiClassStrategy, folds=folds,
                           randomSeed=randomSeed, useLog=useLog, **kwarguments)

    if testX is None:
        if isinstance(trainY, (str, int, np.integer)):
            testX = trainX.copy()
            testX.features.delete(trainY, useLog=False)
        else:
            testX = trainX

    results = trainedLearner.apply(testX, {}, output, scoreMode, useLog=False)
    totalTime = time.process_time() - startTime

    extraInfo = None
    if merged != trainedLearner.arguments:
        extraInfo = {"bestParams": trainedLearner.arguments}
    handleLogging(useLog, "run", "trainAndApply", trainX, trainY, testX, None,
                  learnerName, merged, trainedLearner.randomSeed,
                  extraInfo=extraInfo, time=totalTime)

    return results

def _trainAndTestBackend(learnerName, trainX, trainY, testX, testY,
                         performanceFunction, arguments, output, scoreMode,
                         multiClassStrategy, folds, randomSeed, useLog,
                         **kwarguments):
    merged = mergeArguments(arguments, kwarguments)

    trainedLearner = train(learnerName, trainX, trainY, performanceFunction,
                           merged, scoreMode='label',
                           multiClassStrategy=multiClassStrategy, folds=folds,
                           randomSeed=randomSeed, useLog=useLog)

    if isinstance(testY, (str, int, np.integer)):
        testX = testX.copy()
        testY = testX.features.extract(testY, useLog=False)
    performance = trainedLearner.test(testX, testY, performanceFunction, {},
                                      output, scoreMode, useLog=False)

    return performance, trainedLearner, merged

@trackEntry
def trainAndTest(learnerName, trainX, trainY, testX, testY,
                 performanceFunction, arguments=None, output=None,
                 scoreMode='label', multiClassStrategy='default',
                 folds=10, randomSeed=None, useLog=None, **kwarguments):
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
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX: nimble Base object
        Data to be used for training.
    trainY : identifier, nimble Base object
        * identifier - The name or index of the feature in ``trainX``
          containing the labels.
        * nimble Base object - contains the labels that correspond to
          ``trainX``.
    testX: nimble Base object
        Data to be used for testing.
    testY : identifier, nimble Base object
        * identifier - A name or index of the feature in ``testX``
          containing the labels.
        * nimble Base object - contains the labels that correspond to
          ``testX``.
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in nimble.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To trigger cross-validation using multiple values for arguments,
        specify different values for each parameter using a nimble.CV
        object. eg. {'k': nimble.CV([1,3,5])} will generate an error
        score for  the learner when the learner was passed all three
        values of ``k``, separately. If an argument requires its own
        parameters for instantiation, use a nimble.Init object.
        eg. {'kernel':nimble.Init('KernelGaussian', width=2.0)}.
        ``arguments`` will be merged with the learner ``kwarguments``
    output : str
        The kind of nimble Base object that the output of this function
        should be in. Any of the normal string inputs to the nimble.data
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
        May only be 'default', 'OneVsAll' or 'OneVsOne'.
    folds : int
        The number of folds used in the cross validation. Cannot exceed
        the number of points in ``trainX``. Default 10.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed. Ignored if learner does not
       depend on randomness.
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
        nimble.CV object. eg. arg1=nimble.CV([1,2,3]),
        arg2=nimble.CV([4,5,6]) which correspond to permutations/
        argument states with one element from arg1 and one element from
        arg2, such that an example generated permutation/argument state
        would be ``arg1=2, arg2=4``. Will be merged with ``arguments``.

    Returns
    -------
    performance
        The calculated value of the ``performanceFunction`` after the
        test.

    See Also
    --------
    train, trainAndApply, trainAndTestOnTrainingData, CV,
    nimble.core.interfaces.TrainedLearner.test

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
    >>> trainData = nimble.data('Matrix', rawTrain,
    ...                         featureNames=ftNames)
    >>> testData = nimble.data('Matrix', rawTest, featureNames=ftNames)
    >>> perform = nimble.trainAndTest(
    ...     'nimble.KNNClassifier', trainX=trainData, trainY='label',
    ...     testX=testData, testY='label',
    ...     performanceFunction=nimble.calculate.fractionIncorrect)
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
    >>> trainX = nimble.data('Matrix', rawTrainX)
    >>> trainY = nimble.data('Matrix', rawTrainY)
    >>> testX = nimble.data('Matrix', rawTestX)
    >>> testY = nimble.data('Matrix', rawTestY)
    >>> perform = nimble.trainAndTest(
    ...     'sciKitLearn.SVC', trainX=trainX, trainY=trainY,
    ...     testX=testX, testY=testY,
    ...     performanceFunction=nimble.calculate.fractionIncorrect,
    ...     arguments={'C': 0.1}, kernel='linear')
    >>> perform
    0.0
    """
    startTime = time.process_time()
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, trainY, testX, True, testY, True,
                                  arguments, scoreMode, multiClassStrategy)

    performance, trainedLearner, merged = _trainAndTestBackend(
        learnerName, trainX, trainY, testX, testY, performanceFunction,
        arguments, output, scoreMode, multiClassStrategy, folds,
        randomSeed, useLog, **kwarguments)

    totalTime = time.process_time() - startTime

    metrics = {}
    for key, value in zip([performanceFunction], [performance]):
        metrics[key.__name__] = value
    extraInfo = None
    if merged != trainedLearner.arguments:
        extraInfo = {"bestParams": trainedLearner.arguments}
    handleLogging(useLog, "run", 'trainAndTest', trainX, trainY, testX, testY,
                  learnerName, merged, trainedLearner.randomSeed, metrics,
                  extraInfo, totalTime)

    return performance


@trackEntry
def trainAndTestOnTrainingData(learnerName, trainX, trainY,
                               performanceFunction, crossValidationError=False,
                               folds=10, arguments=None, output=None,
                               scoreMode='label', multiClassStrategy='default',
                               randomSeed=None, useLog=None, **kwarguments):
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
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX: nimble Base object
        Data to be used for training.
    trainY: identifier, nimble Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another nimble Base object containing the labels that
        correspond to ``trainX``.
    performanceFunction : function
        If cross validation is triggered to select from the given
        argument set, then this function will be used to generate a
        performance score for the run. Function is of the form:
        def func(knownValues, predictedValues).
        Look in nimble.calculate for pre-made options. Default is None,
        since if there is no parameter selection to be done, it is not
        used.
    crossValidationError: bool
        Whether we will calculate cross validation error or training
        error. In True case, the training data is split in the
        ``folds`` number of partitions. Each of those is iteratively
        withheld and used as the testing set for a learner trained on
        the combination of all of the non-withheld data. The performance
        results for each of those tests are then averaged together to
        act as the return value. In the False case, we train on the
        training data, and then use the same data as the withheld
        testing data. By default, this flag is set to False.
    folds : int
        The number of folds used in the cross validation. Cannot exceed
        the number of points in ``trainX``. Default 10.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application. eg. {'dimensions':5, 'k':5}
        To trigger cross-validation using multiple values for arguments,
        specify different values for each parameter using a nimble.CV
        object. eg. {'k': nimble.CV([1,3,5])} will generate an error
        score for  the learner when the learner was passed all three
        values of ``k``, separately. If an argument requires its own
        parameters for instantiation, use a nimble.Init object.
        eg. {'kernel':nimble.Init('KernelGaussian', width=2.0)}.
        ``arguments`` will be merged with the learner ``kwarguments``
    output : str
        The kind of nimble Base object that the output of this function
        should be in. Any of the normal string inputs to the nimble.data
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
        May only be 'default', 'OneVsAll' or 'OneVsOne'.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed. Ignored if learner does not
       depend on randomness.
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
        nimble.CV object. eg. arg1=nimble.CV([1,2,3]),
        arg2=nimble.CV([4,5,6]) which correspond to permutations/
        argument states with one element from arg1 and one element from
        arg2, such that an example generated permutation/argument state
        would be ``arg1=2, arg2=4``. Will be merged with ``arguments``.

    Returns
    -------
    performance
        The results of the test.

    See Also
    --------
    train, trainAndApply, trainAndTest, CV

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
    >>> trainData = nimble.data('Matrix', rawTrain,
    ...                         featureNames=ftNames)
    >>> perform = nimble.trainAndTestOnTrainingData(
    ...     'nimble.KNNClassifier', trainX=trainData, trainY='label',
    ...     performanceFunction=nimble.calculate.fractionIncorrect)
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
    >>> trainX = nimble.data('Matrix', rawTrainX)
    >>> trainY = nimble.data('Matrix', rawTrainY)
    >>> perform = nimble.trainAndTestOnTrainingData(
    ...     'sciKitLearn.SVC', trainX=trainX, trainY=trainY,
    ...     performanceFunction=nimble.calculate.fractionIncorrect,
    ...     arguments={'C': 0.1}, kernel='linear')
    >>> perform
    0.0
    """
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, trainY, arguments=arguments,
                                  scoreMode=scoreMode,
                                  multiClassStrategy=multiClassStrategy)

    startTime = time.process_time()
    if crossValidationError:
        merged = mergeArguments(arguments, kwarguments)
        results = crossValidate(learnerName, trainX, trainY,
                                performanceFunction, merged, folds,
                                scoreMode, randomSeed, useLog)
        performance = results.bestResult
        logSeed = results.randomSeed
        extraInfo = {'crossValidationError': True}
    else:
        performance, trainedLearner, merged = _trainAndTestBackend(
            learnerName, trainX, trainY, trainX, trainY, performanceFunction,
            arguments, output, scoreMode, multiClassStrategy, folds,
            randomSeed, useLog, **kwarguments)

        logSeed = trainedLearner.randomSeed
        extraInfo = None
        if merged != trainedLearner.arguments:
            extraInfo = {"bestParams": trainedLearner.arguments}

    totalTime = time.process_time() - startTime

    metrics = {}
    for key, value in zip([performanceFunction], [performance]):
        metrics[key.__name__] = value
    handleLogging(useLog, "run", 'trainAndTestOnTrainingData', trainX,
                  trainY, None, None, learnerName, merged,
                  logSeed, metrics, extraInfo, totalTime)

    return performance


class CV(object):
    """
    Provide a list of values to an argument for cross-validation.

    Triggers cross-validation to occur for the learner using each of the
    values provided and scoring each one.

    Parameters
    ----------
    argumentList : list
        A list of values for the argument.
    """
    def __init__(self, argumentList):
        try:
            self.argumentTuple = tuple(argumentList)
        except TypeError as e:
            msg = "argumentList must be iterable."
            raise InvalidArgumentValue(msg) from e

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


class Init(object):
    """
    Provide interface-specific objects as learner arguments.

    Triggers the interface to search for object ``name`` and instantiate
    the object so that it can be used as the argument of the learner.
    Additional instantiation parameters can be provided as keyword
    arguments.

    Parameters
    ----------
    name : str
        The name of the object to find within the interface.
    kwargs
        Any keyword arguments will be used as instantiation parameters.
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        formatKwargs = ["{}={}".format(k, v) for k, v in self.kwargs.items()]
        kwargStr = ", ".join(formatKwargs)
        return "Init({}, {})".format(repr(self.name), kwargStr)

class KFoldCrossValidator(object):
    """
    Returned by nimble.crossValidate to access cross-validation results.

    Provides full access to the outcome of cross-validation including
    results for each argument set and the best argument set and results.

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
    randomSeed : int
        The random seed used for the learner. Only applicable if the
        learner utilizes randomness.
    """
    def __init__(self, learnerName, X, Y, performanceFunction, arguments=None,
                 folds=10, scoreMode='label', randomSeed=None, useLog=None,
                 **kwarguments):
        """
        Perform k-fold cross-validation and store the results.

        On instantiation, cross-validation will be performed.  The results
        can be accessed through the object's attributes and methods.

        Parameters
        ----------
        learnerName : str
            The learner to be called. This can be a string in the form
            'package.learner' or the learner class object.
        X : nimble Base object
            points/features data
        Y : nimble Base object
            labels/data about points in X
        performanceFunction : function
            Premade options are available in nimble.calculate.
            Function used to evaluate the performance score for each
            run. Function is of the form:
            def func(knownValues, predictedValues).
        arguments : dict
            Mapping argument names (strings) to their values, to be used
            during training and application. eg. {'dimensions':5, 'k':5}
            To trigger cross-validation using multiple values for
            arguments, specify different values for each parameter using
            a nimble.CV object. eg. {'k': nimble.CV([1,3,5])} will
            generate an error score for  the learner when the learner
            was passed all three values of ``k``, separately. These will
            be merged any kwarguments for the learner.
        folds : int
            The number of folds used in the cross validation. Can't
            exceed the number of points in X, Y.
        scoreMode : str
            Used by computeMetrics.
        randomSeed : int
           The random seed to apply (when applicable).
        useLog : bool, None
            Local control for whether to send results/timing to the
            logger.
        kwarguments
            Keyword arguments specified variables that are passed to the
            learner. To trigger cross-validation using multiple values
            for arguments, specify different values for each parameter
            using a nimble.CV object.
            eg. arg1=nimble.CV([1,2,3]), arg2=nimble.CV([4,5,6])
            which correspond to permutations/argument states with one
            element from arg1 and one element from arg2, such that an
            example generated permutation/argument state would be
            ``arg1=2, arg2=4``. Will be merged with ``arguments``.
        """
        self.learnerName = learnerName
        # detectBestResult will raise exception for invalid performanceFunction
        detected = nimble.calculate.detectBestResult(performanceFunction)
        self.maximumIsOptimal = detected == 'max'
        self.performanceFunction = performanceFunction
        self.folds = folds
        self.scoreMode = scoreMode
        self.arguments = mergeArguments(arguments, kwarguments)
        self.randomSeed = randomSeed
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
        X : nimble Base object
            points/features data
        Y : nimble Base object
            labels/data about points in X
        useLog : bool, None
            Local control for whether to send results/timing to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        """
        if Y is not None:
            if isinstance(Y, (int, str, list)):
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
        deepLog = loggingEnabled(useLog) and deepLoggingEnabled()
        for foldNum, fold in enumerate(foldIter):
            [(curTrainX, curTestingX), (curTrainY, curTestingY)] = fold
            argSetIndex = 0
            # given this fold, do a run for each argument combination
            for curArgumentCombination in argumentCombinationIterator:
                #run algorithm on the folds' training and testing sets
                startTime = time.process_time()
                curTL = train(
                    self.learnerName, curTrainX, curTrainY,
                    arguments=curArgumentCombination, scoreMode=self.scoreMode,
                    randomSeed=self.randomSeed, useLog=False)
                if self.randomSeed is None: # use same random seed each time
                    self.randomSeed = curTL.randomSeed
                curRunResult = curTL.apply(curTestingX, useLog=False)
                totalTime = time.process_time() - startTime
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

                metrics = {self.performanceFunction.__name__: curPerformance}
                extraInfo = {'Fold': '{}/{}'.format(foldNum + 1, self.folds)}
                handleLogging(deepLog, "runCV", "KFoldCrossValidation",
                              curTrainX, curTrainY, curTestingX, curTestingY,
                              self.learnerName, curArgumentCombination,
                              self.randomSeed, metrics=metrics,
                              extraInfo=extraInfo, time=totalTime)

                argSetIndex += 1

            if not canAvgFolds:
                if collectedY is None:
                    collectedY = curTestingY
                else:
                    collectedY.points.append(curTestingY, useLog=False)

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
                    results[0].points.append(results[resultIndex],
                                             useLog=False)

                # TODO raise RuntimeError(
                #     "How do we guarantee Y and results are in same order?")
                finalPerformance = computeMetrics(collectedY, None, results[0],
                                                  self.performanceFunction)

            # we use the current results container to be the return value
            performanceOfEachCombination[i] = (curArgSet, finalPerformance)

        # store results
        self._allResults = performanceOfEachCombination

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
        >>> nimble.random.setSeed(42)
        >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
        >>> yRaw = [[1], [2], [3],
        ...         [1], [2], [3],
        ...         [1], [2], [3]]
        >>> X = nimble.data('Matrix', xRaw)
        >>> Y = nimble.data('Matrix', yRaw)
        >>> crossValidator = KFoldCrossValidator(
        ...    'nimble.KNNClassifier', X, Y, arguments={'k': 3},
        ...    performanceFunction=nimble.calculate.fractionIncorrect,
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
        >>> nimble.random.setSeed(42)
        >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
        >>> yRaw = [[1], [2], [3],
        ...         [1], [2], [3],
        ...         [1], [2], [3]]
        >>> X = nimble.data('Matrix', xRaw)
        >>> Y = nimble.data('Matrix', yRaw)
        >>> kValues = nimble.CV([1, 3])
        >>> crossValidator = KFoldCrossValidator(
        ...    'nimble.KNNClassifier', X, Y, arguments={},
        ...    performanceFunction=nimble.calculate.fractionIncorrect,
        ...    folds=3, k=kValues)
        >>> crossValidator.getFoldResults(arguments={'k': 1})
        [0.3333333333333333, 0.0, 0.0]
        >>> crossValidator.getFoldResults(k=1)
        [0.3333333333333333, 0.0, 0.0]
        >>> crossValidator.getFoldResults({'k': 3})
        [0.3333333333333333, 0.6666666666666666, 0.0]
        """
        merged = mergeArguments(arguments, kwarguments)
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
        >>> nimble.random.setSeed(42)
        >>> xRaw = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ...         [1, 0, 1], [1, 1, 0], [0, 1, 1]]
        >>> yRaw = [[1], [2], [3],
        ...         [1], [2], [3],
        ...         [1], [2], [3]]
        >>> X = nimble.data('Matrix', xRaw)
        >>> Y = nimble.data('Matrix', yRaw)
        >>> kValues = nimble.CV([1, 3])
        >>> crossValidator = KFoldCrossValidator(
        ...    'nimble.KNNClassifier', X, Y, arguments={},
        ...    performanceFunction=nimble.calculate.fractionIncorrect,
        ...    folds=3, k=kValues)
        >>> crossValidator.getResult(arguments={'k': 1})
        0.1111111111111111
        >>> crossValidator.getResult(k=1)
        0.1111111111111111
        >>> crossValidator.getResult({'k': 3})
        0.3333333333333333
        """
        merged = mergeArguments(arguments, kwarguments)
        # self._allResults is a list of two-tuples (argumentSet, totalScore)
        for argSet, result in self._allResults:
            if argSet == merged:
                return result
        return self._noMatchingArguments()

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
