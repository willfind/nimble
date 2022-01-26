"""
Module containing the user-facing learner functions for the top level
nimble import.
"""

from types import ModuleType
import time
from operator import itemgetter

import numpy as np

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentValue, ImproperObjectAction
from nimble._utility import mergeArguments, tableString
from nimble.core.logger import handleLogging
from nimble.core._learnHelpers import findBestInterface
from nimble.core._learnHelpers import _learnerQuery
from nimble.core._learnHelpers import _unpackLearnerName
from nimble.core._learnHelpers import validateLearningArguments
from nimble.core._learnHelpers import trackEntry
from nimble.core._learnHelpers import LearnerInspector
from nimble.core._learnHelpers import initAvailablePredefinedInterfaces
from nimble.core.tune import Tune, Tuning


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

    See Also
    --------
    train, normalizeData, fillMatching

    Keywords
    --------
    classifier, regressor, regression, classification, clustering,
    machine learning, preditor, model
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

    Returns a list of strings that are allowed inputs to the ``learnerName``
    parameter of nimble's training, applying, and testing functions. If
    ``package`` is specified, the list will contain strings of each learner.
    If ``package`` is None, the list will contain strings in the form of
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

    See Also
    --------
    showLearnerNames, showAvailablePackages, train, normalizeData,
    fillMatching

    Keywords
    --------
    algorithms, estimators, transformers, predictors, models, train,
    apply, test, package, machine learning
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

    See Also
    --------
    learnerNames

    Keywords
    --------
    print, display, algorithms, estimators, transformers, predictors,
    models, train, apply, test, package, machine learning
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

    See Also
    --------
    learnerNames, learnerParameterDefaults, showLearnerParameters

    Keywords
    --------
    arguments, keyword arguments, machine learning, setup, settings,
    options, hyper parameters, hyperparameters
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

    See Also
    --------
    learnerNames, learnerParameterDefaults, learnerParameters

    Keywords
    --------
    print, display, arguments, keyword arguments, machine learning,
    setup, settings, options, hyper parameters, hyperparameters
    """
    params = learnerParameters(name)
    if params is None:
        print('learner parameters could not be determined')
    elif params:
        for param in params:
            print(param)

def learnerParameterDefaults(name):
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

    See Also
    --------
    learnerNames, learnerParameters, showLearnerParameterDefaults

    Keywords
    --------
    parameters, keywords, keyword arguments, settings, machine learning,
    setup, options, hyper parameters, hyperparameters
    """
    return _learnerQuery(name, 'defaults')

def showLearnerParameterDefaults(name):
    """
    Get a dictionary mapping parameter names to their default values.

    Returns a dictionary with strings of the parameter names as keys and
    the parameter's default value as values. If the name cannot be found
    within the package, then an exception will be thrown.

    Parameters
    ----------
    name : str
        Package and name in the form 'package.learnerName'.

    See Also
    --------
    learnerNames, learnerParameterDefaults, learnerParameters

    Keywords
    --------
    print, display, parameters, keywords, keyword arguments, settings,
    machine learning, setup, options, hyper parameters, hyperparameters
    """
    defaultValues = sorted(learnerParameterDefaults(name).items(),
                           key=itemgetter(0))
    if defaultValues:
        defaults = []
        for param, default in defaultValues:
            if isinstance(default, str):
                default = f"'{default}'"
            defaults.append([param, default])
        print(tableString(defaults, rowHeadJustify='left',
                          colValueJustify='left',
                          includeTrailingNewLine=False))

@trackEntry
def normalizeData(learnerName, trainX, trainY=None, testX=None, arguments=None,
                  randomSeed=None, useLog=None, **kwarguments):
    """
    Modify data according to a produced model.

    Calls on the functionality of a package to train on some data and
    then return the modified ``trainX`` and ``testX`` (if provided)
    according to the results of the trained model. If only ``trainX`` is
    provided, the normalized ``trainX`` is returned. If ``testX`` is
    also provided a tuple (normalizedTrain, normalizedTest) is returned.
    The name of the learner will be added to each normalized object's
    ``name`` attribute to indicate the normalization that has been
    applied. Point and feature names are preserved when possible.

    Parameters
    ----------
    learnerName : str
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX : nimble Base object
        Data to be used for training.
    trainY : identifier, nimble Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another nimble Base object containing the labels that
        correspond to ``trainX``.
    testX: nimble Base object
        Data to be used for testing.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application (e.g., {'dimensions':5, 'k':5}).
        To provide an argument that is an object from the same package
        as the learner, use a ``nimble.Init`` object with the object
        name and its instantiation arguments (e.g.,
        {'optimizer': nimble.Init('SGD', learning_rate=0.01}).
        Note: learner arguments can also be passed as ``kwarguments`` so
        this dictionary will be merged with any keyword arguments.
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
        learner. These are combined with the ``arguments`` parameter.
        To provide an argument that is an object from the same package
        as the learner, use a ``nimble.Init`` object with the object
        name and its instantiation arguments (e.g.,
        optimizer=nimble.Init('SGD', learning_rate=0.01)).

    See Also
    --------
    train, Init, nimble.core.data.Features.normalize

    Examples
    --------
    Normalize a single data set.

    >>> lst = [[20, 1.97, 89], [28, 1.87, 75], [24, 1.91, 81]]
    >>> trainX = nimble.data(lst, pointNames=['a', 'b', 'c'],
    ...                      featureNames=['age', 'height', 'weight'],
    ...                      returnType="Matrix")
    >>> normTrainX = nimble.normalizeData('scikitlearn.StandardScaler',
    ...                                   trainX)
    >>> normTrainX
    <Matrix 3pt x 3ft
           'age'  'height' 'weight'
         ┌─────────────────────────
     'a' │ -1.225  1.298    1.279
     'b' │ 1.225   -1.136   -1.162
     'c' │ 0.000   -0.162   -0.116
    >

    Normalize training and testing data.

    >>> lst1 = [[0, 1, 3], [-1, 1, 2], [1, 2, 2]]
    >>> trainX = nimble.data(lst1)
    >>> lst2 = [[-1, 0, 5]]
    >>> testX = nimble.data(lst2)
    >>> pcaTrain, pcaTest = nimble.normalizeData('scikitlearn.PCA',
    ...                                          trainX, testX=testX,
    ...                                          n_components=2)
    >>> pcaTrain
    <Matrix 3pt x 2ft
           0      1
       ┌──────────────
     0 │ -0.216 0.713
     1 │ -1.005 -0.461
     2 │ 1.221  -0.253
    >
    >>> pcaTest
    <Matrix 1pt x 2ft
           0      1
       ┌─────────────
     0 │ -1.739 2.588
    >

    Keywords
    --------
    modify, apply, standardize, scale, rescale, encode, center, mean,
    standard deviation, z-scores, z scores
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
    trainX : nimble Base object
        Data to be used for training.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application (e.g., {'dimensions':5, 'k':5}).
        To provide an argument that is an object from the same package
        as the learner, use a ``nimble.Init`` object with the object
        name and its instantiation arguments (e.g.,
        {'optimizer': nimble.Init('SGD', learning_rate=0.01}).
        Note: learner arguments can also be passed as ``kwarguments`` so
        this dictionary will be merged with any keyword arguments.
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
        learner. These are combined with the ``arguments`` parameter.
        To provide an argument that is an object from the same package
        as the learner, use a ``nimble.Init`` object with the object
        name and its instantiation arguments (e.g.,
        optimizer=nimble.Init('SGD', learning_rate=0.01)).

    See Also
    --------
    train, Init, nimble.fill, nimble.match,
    nimble.core.data.Points.fillMatching,
    nimble.core.data.Features.fillMatching

    Examples
    --------
    Fill missing values based on k-nearest neighbors classifier.

    >>> lst = [[1, None, None],
    ...        [1, 3, 6],
    ...        [2, 1, 6],
    ...        [1, 3, 7],
    ...        [None, 3, None]]
    >>> X = nimble.data(lst)
    >>> toMatch = nimble.match.missing
    >>> nimble.fillMatching('nimble.KNNImputation', toMatch, X,
    ...                     mode='classification', k=3)
    >>> X
    <DataFrame 5pt x 3ft
           0     1     2
       ┌──────────────────
     0 │ 1.000 3.000 6.000
     1 │ 1.000 3.000 6.000
     2 │ 2.000 1.000 6.000
     3 │ 1.000 3.000 7.000
     4 │ 1.000 3.000 6.000
    >

    Fill last feature zeros based on k-nearest neighbors regressor.

    >>> lst = [[1, 0, 0],
    ...        [1, 3, 6],
    ...        [2, 1, 6],
    ...        [1, 3, 7],
    ...        [0, 3, 0]]
    >>> X = nimble.data(lst)
    >>> toMatch = nimble.match.zero
    >>> nimble.fillMatching('nimble.KNNImputation', toMatch, X,
    ...                     features=-1, k=3, mode='regression')
    >>> X
    <Matrix 5pt x 3ft
           0     1     2
       ┌──────────────────
     0 │ 1.000 0.000 6.333
     1 │ 1.000 3.000 6.000
     2 │ 2.000 1.000 6.000
     3 │ 1.000 3.000 7.000
     4 │ 0.000 3.000 6.333
    >

    Keywords
    --------
    inputation, replace, missing, empty, NaN, nan, clean
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


@trackEntry
def train(learnerName, trainX, trainY=None, arguments=None, scoreMode='label',
          multiClassStrategy='default', randomSeed=None, tuning=None,
          performanceFunction=None, useLog=None, **kwarguments):
    """
    Train a specified learner using the provided data.

    Parameters
    ----------
    learnerName : str
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX : nimble Base object
        Data to be used for training.
    trainY : identifier, nimble Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another nimble Base object containing the labels that
        correspond to ``trainX``.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application (e.g., {'dimensions':5, 'k':5}).
        Multiple values for arguments can be provided by using a
        ``Tune`` object (e.g., {'k': Tune([3, 5, 7])}) to initiate
        hyperparameter tuning and return the learner trained on the best
        set of arguments. To provide an argument that is an object from
        the same package as the learner, use a ``nimble.Init`` object
        with the object name and its instantiation arguments (e.g.,
        {'optimizer': nimble.Init('SGD', learning_rate=0.01}).
        Note: learner arguments can also be passed as ``kwarguments`` so
        this dictionary will be merged with any keyword arguments.
    scoreMode : str
        In the case of a classifying learner, this specifies the type of
        output wanted: 'label' if the class labels are desired,
        'bestScore' if both the class label and the score associated
        with that class are desired, or 'allScores' if a matrix
        containing the scores for every class label are desired.
    multiClassStrategy : str
        May only be 'default', 'OneVsAll' or 'OneVsOne'.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed. Ignored if learner does not
       depend on randomness.
    tuning : nimble.Tuning, None
        Used when hyperparameter tuning has been initiated through
        ``Tune`` objects in the arguments. If hyperparameter tuning is
        triggered and this is None, the default Tuning will be applied
        and a performanceFunction must be provided.
    performanceFunction : function, None
        If hyperparameter tuning is triggered and the Tuning does not
        have a performanceFunction set, then this function will be used
        to generate a performance score for each validation run. See
        nimble.calculate for pre-made options. Default is None.
    useLog : bool, None
        Local control for whether to send results/timing to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. These are combined with the ``arguments`` parameter.
        Multiple values for arguments can be provided by using a
        ``Tune`` object (e.g., k=Tune([3, 5, 7])) to initiate
        hyperparameter tuning and return the learner trained on the best
        set of arguments. To provide an argument that is an object from
        the same package as the learner, use a ``nimble.Init`` object
        with the object name and its instantiation arguments (e.g.,
        optimizer=nimble.Init('SGD', learning_rate=0.01)).

    Returns
    -------
    TrainedLearner

    See Also
    --------
    trainAndApply, trainAndTest, trainAndTestOnTrainingData,
    nimble.core.interfaces.TrainedLearner, nimble.Init, nimble.Tune,
    nimble.Tuning



    Examples
    --------
    A single dataset which contains the labels.

    >>> lst = [[1, 0, 0, 1],
    ...        [0, 1, 0, 2],
    ...        [0, 0, 1, 3],
    ...        [1, 0, 0, 1],
    ...        [0, 1, 0, 2],
    ...        [0, 0, 1, 3]]
    >>> ftNames = ['a', 'b' ,'c', 'label']
    >>> trainData = nimble.data(lst, featureNames=ftNames)
    >>> tl = nimble.train('nimble.KNNClassifier', trainX=trainData,
    ...                   trainY='label')
    >>> print(type(tl))
    <class 'nimble.core.interfaces.universal_interface.TrainedLearner'>

    Passing arguments to the learner. Both the arguments parameter and
    kwarguments can be utilized, they will be merged. Below, ``C`` and
    ``kernel`` are parameters for scikit-learn's SVC learner.

    >>> lstX = [[1, 0, 0],
    ...         [0, 1, 0],
    ...         [0, 0, 1],
    ...         [1, 0, 0],
    ...         [0, 1, 0],
    ...         [0, 0, 1]]
    >>> lstY = [[1], [2], [3], [1], [2], [3]]
    >>> trainX = nimble.data(lstX)
    >>> trainY = nimble.data(lstY)
    >>> tl = nimble.train('sciKitLearn.SVC', trainX=trainX, trainY=trainY,
    ...                   arguments={'C': 0.1}, kernel='linear')
    >>> tlAttributes = tl.getAttributes()
    >>> cValue = tlAttributes['C']
    >>> kernelValue = tlAttributes['kernel']
    >>> print(cValue, kernelValue)
    0.1 linear

    Keywords
    --------
    learn, model, regression, classification, neural network,
    clustering, supervised, unsupervised, deep learning, fit, training,
    machine learning
    """
    startTime = time.process_time()
    tuneLog = useLog
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, trainY, arguments=arguments,
                                  scoreMode=scoreMode,
                                  multiClassStrategy=multiClassStrategy)
    else:
        useLog = False

    interface, trueLearnerName = _unpackLearnerName(learnerName)

    # perform hyperparameter optimization (if needed)
    arguments = mergeArguments(arguments, kwarguments)
    if (tuning is not None or
            any(isinstance(arg, Tune) for arg in arguments.values())):
        if tuning is None:
            tuning = Tuning()
        tuning.tune(
            learnerName, trainX, trainY, arguments, performanceFunction,
            randomSeed, tuneLog)
        bestArguments = tuning.bestArguments
        # subsequent tune calls to same Tuning will overwrite, so need to
        # attach a copy to the TrainedLearner
        tuning = tuning.copy()
    else:
        bestArguments = arguments

    if tuning is not None and hasattr(tuning.validator, 'bestTrainedLearner'):
        trainedLearner = tuning.validator.bestTrainedLearner
        trainedLearner.tuning = tuning
    else:
        trainedLearner = interface.train(trueLearnerName, trainX, trainY,
                                         bestArguments, multiClassStrategy,
                                         randomSeed, tuning)
    totalTime = time.process_time() - startTime

    funcString = interface.getCanonicalName() + '.' + trueLearnerName
    handleLogging(useLog, "run", "train", trainX, trainY, None, None,
                  funcString, bestArguments, trainedLearner.randomSeed,
                  time=totalTime)

    return trainedLearner

@trackEntry
def trainAndApply(learnerName, trainX, trainY=None, testX=None, arguments=None,
                  output=None, scoreMode='label', multiClassStrategy='default',
                  randomSeed=None, tuning=None, performanceFunction=None,
                  useLog=None, **kwarguments):
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
    trainX : nimble Base object
        Data to be used for training.
    trainY : identifier, nimble Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another nimble Base object containing the labels that
        correspond to ``trainX``.
    testX : nimble Base object
        Data set on which the trained learner will be applied (i.e.
        performing prediction, transformation, etc. as appropriate to
        the learner).
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application (e.g., {'dimensions':5, 'k':5}).
        Multiple values for arguments can be provided by using a
        ``Tune`` object (e.g., {'k': Tune([3, 5, 7])}) to initiate
        hyperparameter tuning and return the learner trained on the best
        set of arguments. To provide an argument that is an object from
        the same package as the learner, use a ``nimble.Init`` object
        with the object name and its instantiation arguments (e.g.,
        {'optimizer': nimble.Init('SGD', learning_rate=0.01}).
        Note: learner arguments can also be passed as ``kwarguments`` so
        this dictionary will be merged with any keyword arguments.
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
    tuning : nimble.Tuning, None
        Used when hyperparameter tuning has been initiated through
        ``Tune`` objects in the arguments. If hyperparameter tuning is
        triggered and this is None, the default Tuning will be applied
        and a performanceFunction must be provided.
    performanceFunction : function, None
        If hyperparameter tuning is triggered and the Tuning does not
        have a performanceFunction set, then this function will be used
        to generate a performance score for each validation run. See
        nimble.calculate for pre-made options. Default is None.
    useLog : bool, None
        Local control for whether to send results/timing to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. These are combined with the ``arguments`` parameter.
        Multiple values for arguments can be provided by using a
        ``Tune`` object (e.g., k=Tune([3, 5, 7])) to initiate
        hyperparameter tuning and return the learner trained on the best
        set of arguments. To provide an argument that is an object from
        the same package as the learner, use a ``nimble.Init`` object
        with the object name and its instantiation arguments (e.g.,
        optimizer=nimble.Init('SGD', learning_rate=0.01)).

    Returns
    -------
    results
        The resulting output of applying learner.

    See Also
    --------
    train, nimble.Init, nimble.Tune, nimble.Tuning,
    nimble.core.interfaces.TrainedLearner.apply

    Examples
    --------
    Train dataset which contains the labels.

    >>> lstTrain = [[1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3],
    ...             [1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3]]
    >>> lstTestX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> trainData = nimble.data(lstTrain)
    >>> testX = nimble.data(lstTestX)
    >>> predict = nimble.trainAndApply('nimble.KNNClassifier',
    ...                                trainX=trainData, trainY=3,
    ...                                testX=testX)
    >>> predict
    <Matrix 3pt x 1ft
         0
       ┌──
     0 │ 1
     1 │ 2
     2 │ 3
    >

    Passing arguments to the learner. Both the arguments parameter and
    kwarguments can be utilized, they will be merged. Below, ``C`` and
    ``kernel`` are parameters for scikit-learn's SVC learner.

    >>> lstTrainX = [[1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1],
    ...              [1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1]]
    >>> lstTrainY = [[1], [2], [3], [1], [2], [3]]
    >>> lstTestX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> trainX = nimble.data(lstTrainX)
    >>> trainY = nimble.data(lstTrainY)
    >>> testX = nimble.data(lstTestX)
    >>> pred = nimble.trainAndApply('sciKitLearn.SVC', trainX=trainX,
    ...                             trainY=trainY, testX=testX,
    ...                             arguments={'C': 0.1}, kernel='linear')
    >>> pred
    <Matrix 3pt x 1ft
         0
       ┌──
     0 │ 1
     1 │ 2
     2 │ 3
    >

    Keywords
    --------
    predict, prediction, transformation, encode, standardize, model,
    supervised, unsupervised, pred, transform, fit_predict,
    fit_transform, training, machine learning
    """
    startTime = time.process_time()
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, trainY, testX, arguments=arguments,
                                  scoreMode=scoreMode,
                                  multiClassStrategy=multiClassStrategy)
    merged = mergeArguments(arguments, kwarguments)
    trainedLearner = train(
        learnerName, trainX, trainY, arguments=merged, scoreMode=scoreMode,
        multiClassStrategy=multiClassStrategy, randomSeed=randomSeed,
        performanceFunction=performanceFunction, tuning=tuning, useLog=useLog)

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
        extraInfo = {"bestArguments": trainedLearner.arguments}
    handleLogging(useLog, "run", "trainAndApply", trainX, trainY, testX, None,
                  learnerName, merged, trainedLearner.randomSeed,
                  extraInfo=extraInfo, time=totalTime)

    return results

def _trainAndTestBackend(learnerName, trainX, trainY, testX, testY,
                         performanceFunction, arguments, output, scoreMode,
                         multiClassStrategy, randomSeed, tuning, useLog,
                         **kwarguments):
    merged = mergeArguments(arguments, kwarguments)

    trainedLearner = train(learnerName, trainX, trainY, merged,
                           multiClassStrategy=multiClassStrategy,
                           performanceFunction=performanceFunction,
                           randomSeed=randomSeed, tuning=tuning, useLog=useLog)

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
                 randomSeed=None, tuning=None, useLog=None,
                 **kwarguments):
    """
    Train a model and get the results of its performance.

    For each permutation of the merge of 'arguments' and 'kwarguments'
    (more below), this function uses cross validation to generate a
    performance score for the algorithm, given the particular argument
    permutation. The argument permutation that performed best cross
    validating over the training data is then used as the lone argument
    for training on the whole training data set. Finally, the learned
    model generates predictions for the testing set, and the performance
    of those predictions is calculated and returned. If no additional
    arguments are supplied via arguments or kwarguments, then the
    result is the performance of the algorithm with default arguments on
    the testing data.

    Parameters
    ----------
    learnerName : str
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX : nimble Base object
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
        The function used to determine the performance of the learner.
        Look in nimble.calculate for pre-made options. If hyperparameter
        tuning is triggered and the Tuning does not have a set
        performanceFunction, the function will be applied there as well.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application (e.g., {'dimensions':5, 'k':5}).
        Multiple values for arguments can be provided by using a
        ``Tune`` object (e.g., {'k': Tune([3, 5, 7])}) to initiate
        hyperparameter tuning and return the learner trained on the best
        set of arguments. To provide an argument that is an object from
        the same package as the learner, use a ``nimble.Init`` object
        with the object name and its instantiation arguments (e.g.,
        {'optimizer': nimble.Init('SGD', learning_rate=0.01}).
        Note: learner arguments can also be passed as ``kwarguments`` so
        this dictionary will be merged with any keyword arguments.
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
    tuning : nimble.Tuning, None
        Used when hyperparameter tuning has been initiated through
        ``Tune`` objects in the arguments. If hyperparameter tuning is
        triggered and this is None, the default Tuning will be applied.
    useLog : bool, None
        Local control for whether to send results/timing to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. These are combined with the ``arguments`` parameter.
        Multiple values for arguments can be provided by using a
        ``Tune`` object (e.g., k=Tune([3, 5, 7])) to initiate
        hyperparameter tuning and return the learner trained on the best
        set of arguments. To provide an argument that is an object from
        the same package as the learner, use a ``nimble.Init`` object
        with the object name and its instantiation arguments (e.g.,
        optimizer=nimble.Init('SGD', learning_rate=0.01)).

    Returns
    -------
    performance
        The calculated value of the ``performanceFunction`` after the
        test.

    See Also
    --------
    train, trainAndTestOnTrainingData, nimble.Init, nimble.Tune,
    nimble.Tuning, nimble.core.interfaces.TrainedLearner.test

    Examples
    --------
    Train and test datasets which contains the labels.

    >>> lstTrain = [[1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3],
    ...             [1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3]]
    >>> lstTest = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    >>> ftNames = ['a', 'b', 'c', 'label']
    >>> trainData = nimble.data(lstTrain,
    ...                         featureNames=ftNames)
    >>> testData = nimble.data(lstTest, featureNames=ftNames)
    >>> perform = nimble.trainAndTest(
    ...     'nimble.KNNClassifier', trainX=trainData, trainY='label',
    ...     testX=testData, testY='label',
    ...     performanceFunction=nimble.calculate.fractionIncorrect)
    >>> perform
    0.0

    Passing arguments to the learner. Both the arguments parameter and
    kwarguments can be utilized, they will be merged. Below, ``C`` and
    ``kernel`` are parameters for scikit-learn's SVC learner.

    >>> lstTrainX = [[1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1],
    ...              [1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1]]
    >>> lstTrainY = [[1], [2], [3], [1], [2], [3]]
    >>> lstTestX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> lstTestY = [[1], [2], [3]]
    >>> trainX = nimble.data(lstTrainX)
    >>> trainY = nimble.data(lstTrainY)
    >>> testX = nimble.data(lstTestX)
    >>> testY = nimble.data(lstTestY)
    >>> perform = nimble.trainAndTest(
    ...     'sciKitLearn.SVC', trainX=trainX, trainY=trainY,
    ...     testX=testX, testY=testY,
    ...     performanceFunction=nimble.calculate.fractionIncorrect,
    ...     arguments={'C': 0.1}, kernel='linear')
    >>> perform
    0.0

    Keywords
    --------
    performance, testing, supervised, score, model, training,
    machine learning, predict, error, measure, accuracy, performance
    """
    startTime = time.process_time()
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, trainY, testX, True, testY, True,
                                  arguments, scoreMode, multiClassStrategy)

    performance, trainedLearner, merged = _trainAndTestBackend(
        learnerName, trainX, trainY, testX, testY, performanceFunction,
        arguments, output, scoreMode, multiClassStrategy, randomSeed, tuning,
        useLog, **kwarguments)

    totalTime = time.process_time() - startTime

    metrics = {}
    for key, value in zip([performanceFunction], [performance]):
        metrics[key.__name__] = value
    extraInfo = None
    if merged != trainedLearner.arguments:
        extraInfo = {"bestArguments": trainedLearner.arguments}
    handleLogging(useLog, "run", 'trainAndTest', trainX, trainY, testX, testY,
                  learnerName, merged, trainedLearner.randomSeed, metrics,
                  extraInfo, totalTime)

    return performance


@trackEntry
def trainAndTestOnTrainingData(
    learnerName, trainX, trainY, performanceFunction,
    crossValidationError=False, folds=5, arguments=None, output=None,
    scoreMode='label', multiClassStrategy='default', randomSeed=None,
    tuning=None, useLog=None, **kwarguments):
    """
    Train a model using the train data and get the performance results.

    ``trainAndTestOnTrainingData`` is the function for doing learner
    creation and evaluation in a single step with only a single data set
    (no withheld testing set). By default, this will calculate training
    error for the learner trained on that data set. However, cross
    validation error can instead be calculated by setting the parameter
    ``crossValidationError`` to be True. In that case, we will partition
    the training set into a parameter controlled number of folds, and
    iteratively withhold each single fold to be used as the testing set
    of the learner trained on the rest of the data.

    Parameters
    ----------
    learnerName : str
        The learner to be called. This can be a string in the form
        'package.learner' or the learner class object.
    trainX : nimble Base object
        Data to be used for training.
    trainY : identifier, nimble Base object
        A name or index of the feature in ``trainX`` containing the
        labels or another nimble Base object containing the labels that
        correspond to ``trainX``.
    performanceFunction : function
        The function used to determine the performance of the learner.
        Look in nimble.calculate for pre-made options. If hyperparameter
        tuning is triggered and the Tuning does not have a set
        performanceFunction, the function will be applied there as well.
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
        The number of folds used in the cross validation. Defaults to 5,
        cannot exceed the number of points in ``trainX``, and only
        applies when ``crossValidationError`` is True.
    arguments : dict
        Mapping argument names (strings) to their values, to be used
        during training and application (e.g., {'dimensions':5, 'k':5}).
        Multiple values for arguments can be provided by using a
        ``Tune`` object (e.g., {'k': Tune([3, 5, 7])}) to initiate
        hyperparameter tuning and return the learner trained on the best
        set of arguments. To provide an argument that is an object from
        the same package as the learner, use a ``nimble.Init`` object
        with the object name and its instantiation arguments (e.g.,
        {'optimizer': nimble.Init('SGD', learning_rate=0.01}).
        Note: learner arguments can also be passed as ``kwarguments`` so
        this dictionary will be merged with any keyword arguments.
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
    tuning : nimble.Tuning, None
        Used when hyperparameter tuning has been initiated through
        ``Tune`` objects in the arguments. If hyperparameter tuning is
        triggered and this is None, the default Tuning will be applied.
    useLog : bool, None
        Local control for whether to send results/timing to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. These are combined with the ``arguments`` parameter.
        Multiple values for arguments can be provided by using a
        ``Tune`` object (e.g., k=Tune([3, 5, 7])) to initiate
        hyperparameter tuning and return the learner trained on the best
        set of arguments. To provide an argument that is an object from
        the same package as the learner, use a ``nimble.Init`` object
        with the object name and its instantiation arguments (e.g.,
        optimizer=nimble.Init('SGD', learning_rate=0.01)).

    Returns
    -------
    performance
        The results of the test.

    See Also
    --------
    train, trainAndTest, nimble.Init, nimble.Tune, nimble.Tuning

    Examples
    --------
    Train and test datasets which contains the labels.

    >>> lstTrain = [[1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3],
    ...             [1, 0, 0, 1],
    ...             [0, 1, 0, 2],
    ...             [0, 0, 1, 3]]
    >>> ftNames = ['a', 'b', 'c', 'label']
    >>> trainData = nimble.data(lstTrain,
    ...                         featureNames=ftNames)
    >>> perform = nimble.trainAndTestOnTrainingData(
    ...     'nimble.KNNClassifier', trainX=trainData, trainY='label',
    ...     performanceFunction=nimble.calculate.fractionIncorrect)
    >>> perform
    0.0

    Passing arguments to the learner. Both the arguments parameter and
    kwarguments can be utilized, they will be merged. Below, ``C`` and
    ``kernel`` are parameters for scikit-learn's SVC learner.

    >>> lstTrainX = [[1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1],
    ...              [1, 0, 0],
    ...              [0, 1, 0],
    ...              [0, 0, 1]]
    >>> lstTrainY = [[1], [2], [3], [1], [2], [3]]
    >>> trainX = nimble.data(lstTrainX)
    >>> trainY = nimble.data(lstTrainY)
    >>> perform = nimble.trainAndTestOnTrainingData(
    ...     'sciKitLearn.SVC', trainX=trainX, trainY=trainY,
    ...     performanceFunction=nimble.calculate.fractionIncorrect,
    ...     arguments={'C': 0.1}, kernel='linear')
    >>> perform
    0.0

    Keywords
    --------
    performance, in-sample, in sample, supervised, score, model,
    training, machine learning, predict, error, measure, accuracy,
    performance
    """
    if trackEntry.isEntryPoint:
        validateLearningArguments(trainX, trainY, arguments=arguments,
                                  scoreMode=scoreMode,
                                  multiClassStrategy=multiClassStrategy)

    startTime = time.process_time()
    merged = mergeArguments(arguments, kwarguments)
    performance, trainedLearner, merged = _trainAndTestBackend(
        learnerName, trainX, trainY, trainX, trainY, performanceFunction,
        merged, output, scoreMode, multiClassStrategy, randomSeed, tuning,
        useLog)

    if trainedLearner.tuning is not None:
        bestArgs = trainedLearner.arguments
        extraInfo = {"bestArguments": bestArgs}
    else:
        bestArgs = merged
        extraInfo = None
    if crossValidationError:
        # use tuning with only the bestArgs for final cross validation
        cvTuning = Tuning(folds=folds)
        cvTuning.tune(learnerName, trainX, trainY, bestArgs,
                      performanceFunction, randomSeed, useLog)
        performance = cvTuning.bestResult

    logSeed = trainedLearner.randomSeed
    metrics = {}
    for key, value in zip([performanceFunction], [performance]):
        metrics[key.__name__] = value
    totalTime = time.process_time() - startTime
    handleLogging(useLog, "run", 'trainAndTestOnTrainingData', trainX,
                  trainY, None, None, learnerName, merged,
                  logSeed, metrics, extraInfo, totalTime)

    return performance


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

    See Also
    --------
    learnerParameters

    Keywords
    --------
    instantiate, argument, initialize, initialization, create
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        formatKwargs = [f"{k}={v}" for k, v in self.kwargs.items()]
        kwargStr = ", ".join(formatKwargs)
        return f"Init({repr(self.name)}, {kwargStr})"
