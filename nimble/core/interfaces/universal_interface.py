"""
The top level objects and methods which allow nimble to interface with
various python packages or custom learners. Also contains the objects
which store trained learner models and provide functionality for
applying and testing learners.
"""

import copy
import abc
import functools
import sys
import numbers
import warnings

import numpy

import nimble
from nimble.exceptions import InvalidArgumentValue, ImproperObjectAction
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import PackageException
from nimble._utility import inspectArguments
from nimble._utility import inheritDocstringsFactory
from nimble._utility import cloudpickle
from nimble._utility import mergeArguments
from nimble.exceptions import _prettyListString
from nimble.exceptions import _prettyDictString
from nimble.core.logger import handleLogging, startTimer, stopTimer
from nimble.core.configuration import configErrors
from nimble.core._learnHelpers import TrainValidator, computeMetrics
from ._interface_helpers import (
    generateBinaryScoresFromHigherSortedLabelScores,
    calculateSingleLabelScoresFromOneVsOneScores,
    ovaNotOvOFormatted, checkClassificationStrategy, cacheWrapper,
    generateAllPairs, countWins, extractWinningPredictionIndex,
    extractWinningPredictionLabel, extractWinningPredictionIndexAndScore,
    extractConfidenceScores)


def captureOutput(toWrap):
    """
    Decorator which will safely ignore certain warnings.

    Prevent any warnings from displaying that do not apply directly to
    the operation being run. This can be overridden by -W options at the
    command line or the PYTHONWARNINGS environment variable.
    """
    @functools.wraps(toWrap)
    def wrapped(*args, **kwarguments):
        # user has not already provided warnings filters
        if not sys.warnoptions:
            with warnings.catch_warnings():
                # filter out warnings that we do not need users to see
                warnings.simplefilter('ignore', DeprecationWarning)
                warnings.simplefilter('ignore', FutureWarning)
                warnings.simplefilter('ignore', PendingDeprecationWarning)
                warnings.simplefilter('ignore', ImportWarning)

                ret = toWrap(*args, **kwarguments)
        else:
            ret = toWrap(*args, **kwarguments)
        return ret

    return wrapped


class UniversalInterface(metaclass=abc.ABCMeta):
    """
    Metaclass defining methods and abstract methods for specific
    package or custom interfaces.
    """
    _listLearnersCached = None

    def __init__(self):
        ### Validate all the information from abstract functions ###
        # enforce a check that the underlying package is accessible at
        # instantiation, aborting the construction of the interface for this
        # session of nimble if it is not.
        if not self.accessible():
            msg = "The underlying package for " + self.getCanonicalName()
            msg += " was not accessible, aborting instantiation."
            raise ImportError(msg)

        # getCanonicalName
        if not isinstance(self.getCanonicalName(), str):
            msg = "Improper implementation of getCanonicalName(), must return "
            msg += "a string"
            raise TypeError(msg)

        # _configurableOptionNames and _optionDefaults
        optionNames = self._configurableOptionNames()
        if not isinstance(optionNames, list):
            msg = "Improper implementation of _configurableOptionNames(), "
            msg += "must return a list of strings"
            raise TypeError(msg)
        for optionName in optionNames:
            if not isinstance(optionName, str):
                msg = "Improper implementation of _configurableOptionNames(), "
                msg += "must return a list of strings"
                raise TypeError(msg)
            # call _optionDefaults to make sure it doesn't throw an exception
            self._optionDefaults(optionName)

        # _exposedFunctions
        exposedFunctions = self._exposedFunctions()
        if exposedFunctions is None or not isinstance(exposedFunctions, list):
            msg = "Improper implementation of _exposedFunctions(), must "
            msg += "return a list of methods to be bundled with TrainedLearner"
            raise TypeError(msg)
        for exposed in exposedFunctions:
            # is callable
            if not hasattr(exposed, '__call__'):
                msg = "Improper implementation of _exposedFunctions, each "
                msg += "member of the return must have __call__ attribute"
                raise TypeError(msg)
            # has name attribute
            if not hasattr(exposed, '__name__'):
                msg = "Improper implementation of _exposedFunctions, each "
                msg += "member of the return must have __name__ attribute"
                raise TypeError(msg)
            # takes self as attribute
            (args, _, _, _) = inspectArguments(exposed)
            if args[0] != 'self':
                msg = "Improper implementation of _exposedFunctions each "
                msg += "member's first argument must be 'self', interpreted "
                msg += "as a TrainedLearner"
                raise TypeError(msg)

    @property
    def optionNames(self):
        """
        TODO
        """
        return copy.copy(self._configurableOptionNames())


    def isAlias(self, name):
        """
        Determine if the name is an accepted alias for this interface.

        Parameters
        ----------
        name : str
            An interface name as a string

        Returns
        -------
        bool
            True if the name is a accepted alias, False otherwise.
        """
        return name.lower() == self.getCanonicalName().lower()


    @captureOutput
    def train(self, learnerName, trainX, trainY=None, arguments=None,
              multiClassStrategy='default', crossValidationResults=None):
        """
        Fit the learner model using training data.

        learnerName : str
            Name of the learner to be called, in the form
            'package.learner'
        trainX: nimble Base object
            Data to be used for training.
        trainY: identifier, nimble Base object
            A name or index of the feature in ``trainX`` containing the
            labels or another nimble Base object containing the labels
            that correspond to ``trainX``.
        multiClassStrategy : str
            May only be 'default' 'OneVsAll' or 'OneVsOne'
        arguments : dict
            Mapping argument names (strings) to their values, to be used
            during training and application. eg. {'dimensions':5, 'k':5}
            To make use of multiple permutations, specify different
            values for a parameter as a tuple. eg. {'k': (1,3,5)} will
            generate an error score for  the learner when the learner
            was passed all three values of ``k``, separately.
        useLog : bool, None
            Local control for whether to send results/timing to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If False,
            do **NOT** send to the logger, regardless of the global
            option.
        """
        if multiClassStrategy != 'default':
            # TODO reevaluate use of checkClassificationStrategy, the if
            # statements below expect a string output but it looks to output
            # a boolean value. It is also susceptible to failures for binary
            # classifiers and learners without a getScores method implemented.

            # #if we need to do multiclassification by ourselves
            # trialResult = checkClassificationStrategy(self, learnerName,
            #                                           arguments)
            #1 VS All
            if multiClassStrategy == 'OneVsAll':
                # and trialResult != 'OneVsAll':?
                #Remove true labels from from training set, if not separated
                if isinstance(trainY, (str, numbers.Integral)):
                    trainX = trainX.copy()
                    trainY = trainX.features.extract(trainY, useLog=False)

                # Get set of unique class labels
                labelVector = trainY.T
                labelVectorToList = labelVector.copy(to="python list")[0]
                labelSet = list(set(labelVectorToList))

                # For each class label in the set of labels:  convert the true
                # labels in trainY into boolean labels (1 if the point
                # has 'label', 0 otherwise.)  Train a classifier with the
                # processed labels and get predictions on the test set.
                trainedLearners = []
                for label in labelSet:

                    def relabeler(val):
                        if val == label:
                            return 1
                        return 0

                    trainLabels = trainY.calculateOnElements(relabeler,
                                                             useLog=False)
                    trainedLearner = self._train(
                        learnerName, trainX, trainLabels, arguments=arguments)
                    trainedLearner.label = label
                    trainedLearners.append(trainedLearner)

                return TrainedLearners(trainedLearners, 'OneVsAll', labelSet)

            #1 VS 1
            if multiClassStrategy == 'OneVsOne':
                # and trialResult != 'OneVsOne': ?
                # want data and labels together in one object for this method
                trainX = trainX.copy()
                if isinstance(trainY, nimble.core.data.Base):
                    trainX.features.append(trainY, useLog=False)
                    trainY = len(trainX.features) - 1

                # Get set of unique class labels, then generate list of all
                # 2-combinations of class labels
                labelVector = trainX.features.copy([trainY])
                labelVector.transpose(useLog=False)
                labelVectorToList = labelVector.copy(to="python list")[0]
                labelSet = list(set(labelVectorToList))
                labelPairs = generateAllPairs(labelSet)

                # For each pair of class labels: remove all points with one of
                # those labels, train a classifier on those points, get
                # predictions based on that model, and put the points back into
                # the data object
                trainedLearners = []
                for pair in labelPairs:
                    #get all points that have one of the labels in pair
                    pairData = trainX.points.extract(
                        lambda point: point[trainY] in pair, useLog=False)
                    pairTrueLabels = pairData.features.extract(trainY,
                                                               useLog=False)
                    trainedLearners.append(
                        self._train(
                            learnerName, pairData.copy(),
                            pairTrueLabels.copy(), arguments=arguments)
                        )
                    pairData.features.append(pairTrueLabels, useLog=False)
                    trainX.points.append(pairData, useLog=False)

                return TrainedLearners(trainedLearners, 'OneVsOne', labelSet)

        # separate training data / labels if needed
        if isinstance(trainY, (str, int, numpy.integer)):
            trainX = trainX.copy()
            trainY = trainX.features.extract(toExtract=trainY, useLog=False)
        return self._train(learnerName, trainX, trainY, arguments,
                           crossValidationResults)


    @captureOutput
    def _train(self, learnerName, trainX, trainY=None, arguments=None,
               crossValidationResults=None):
        packedBackend = self._trainBackend(learnerName, trainX, trainY,
                                           arguments)
        trainedBackend, transformedInputs, customDict = packedBackend

        has2dOutput = False
        outputData = trainX if trainY is None else trainY
        if isinstance(outputData, nimble.core.data.Base):
            has2dOutput = len(outputData.features) > 1
        elif isinstance(outputData, (list, tuple)):
            has2dOutput = len(outputData) > 1
        trainYNames = None
        if trainY is not None:
            trainYNames = trainY.features._getNamesNoGeneration()
        trainXShape = trainX.shape
        transformedArguments = transformedInputs[3]
        # encapsulate into TrainedLearner object
        return TrainedLearner(learnerName, arguments, transformedArguments,
                              customDict, trainedBackend, self, has2dOutput,
                              crossValidationResults, trainXShape, trainYNames)


    def _confirmValidLearner(self, learnerName):
        allLearners = self.listLearners()
        if not learnerName in allLearners:
            msg = learnerName
            msg += " is not the name of a learner exposed by this interface"
            raise InvalidArgumentValue(msg)
        learnerCall = self.findCallable(learnerName)
        if learnerCall is None:
            msg = learnerName + " was not found in this package"
            raise InvalidArgumentValue(msg)


    def _trainBackend(self, learnerName, trainX, trainY, arguments):
        ### PLANNING ###

        # verify the learner is available
        self._confirmValidLearner(learnerName)

        # the scratch space dictionary that the package implementor may use to
        # pass information between I/O transformation, the trainer and applier
        customDict = {}

        # validate the arguments provided
        self._validateLearnerArgumentValues(learnerName, arguments)

        # execute interface implementor's input transformation.
        transformedInputs = self._inputTransformation(
            learnerName, trainX, trainY, None, arguments, customDict)
        transTrainX, transTrainY, _, transArguments = transformedInputs

        transformedInputs = (transformedInputs[0], transformedInputs[1],
                             transformedInputs[2], transArguments)
        ### LEARNER CREATION / TRAINING ###

        # train the instantiated learner
        trainedBackend = self._trainer(learnerName, transTrainX, transTrainY,
                                       transArguments, customDict)

        return (trainedBackend, transformedInputs, customDict)

    def _validateArgumentValues(self, name, arguments):
        possibleParams = self._getParameterNames(name)
        possibleDefaults = self._getDefaultValues(name)
        bestIndex = self._chooseBestParameterSet(possibleParams,
                                                 possibleDefaults, arguments)
        neededParams = possibleParams[bestIndex]
        availableDefaults = possibleDefaults[bestIndex]
        self._argumentValueValidation(name, possibleParams, possibleDefaults,
                                      arguments, neededParams,
                                      availableDefaults)

    def _validateLearnerArgumentValues(self, name, arguments):

        possibleParams = self.getLearnerParameterNames(name)
        possibleDefaults = self.getLearnerDefaultValues(name)
        bestIndex = self._chooseBestParameterSet(possibleParams,
                                                 possibleDefaults, arguments)
        neededParams = possibleParams[bestIndex]
        availableDefaults = possibleDefaults[bestIndex]
        self._argumentValueValidation(name, possibleParams, possibleDefaults,
                                      arguments, neededParams,
                                      availableDefaults)

    def _argumentValueValidation(self, name, possibleParams, possibleDefaults,
                                 arguments, neededParams, availableDefaults):
        if arguments is None:
            arguments = {}
        check = arguments.copy()
        completeArgs = {}
        for param in neededParams:
            if param in check:
                completeArgs[param] = check[param]
                del check[param]
            elif param not in availableDefaults:
                msg = "MISSING LEARNING PARAMETER! "
                msg += "When trying to validate arguments for " + name
                msg += ", we couldn't find a value for the parameter named "
                msg += "'" + param + "'. "
                msg += "The allowed parameters were: "
                msg += _prettyListString(neededParams, useAnd=True)
                if len(possibleParams) > 1:
                    msg += ". These were choosen as the best guess given the "
                    msg += "inputs out of the following (numbered) list of "
                    msg += "possible parameter sets: "
                    msg += _prettyListString(possibleParams, numberItems=True,
                                             itemStr=_prettyListString)

                if len(availableDefaults) == 0:
                    msg += ". All of the allowed parameters must be specified "
                    msg += "by the user"
                else:
                    msg += ". Out of the allowed parameters, the following "
                    msg += "could be omitted, which would result in the "
                    msg += "associated default value being used: "
                    msg += _prettyDictString(availableDefaults, useAnd=True)

                if len(arguments) == 0:
                    msg += ". However, no arguments were provided."
                else:
                    msg += ". The full mapping of inputs actually provided "
                    msg += "was: " + _prettyDictString(arguments)

                raise InvalidArgumentValue(msg)

        if check:
            msg = "EXTRA LEARNER PARAMETER! "
            msg += "When trying to validate arguments for "
            msg += name + ", the following list of parameter "
            msg += "names were not matched: "
            msg += _prettyListString(list(check.keys()), useAnd=True)
            msg += ". The allowed parameters were: "
            msg += _prettyListString(neededParams, useAnd=True)
            if len(possibleParams) > 1:
                msg += ". These were choosen as the best guess given the "
                msg += "inputs out of the following (numbered) list of "
                msg += "possible parameter sets: "
                msg += _prettyListString(possibleParams, numberItems=True,
                                         itemStr=_prettyListString)

            msg += ". The full mapping of inputs actually provided was: "
            msg += _prettyDictString(arguments) + ". "
            msg += "If extra parameters were intended to be passed to one of "
            msg += "the arguments, be sure to group them using a nimble.Init "
            msg += "object. "

            raise InvalidArgumentValue(msg)


    def _argumentInit(self, toInit):
        """
        Recursive function for instantiating learner subobjects
        """
        initObject = self.findCallable(toInit.name)
        if initObject is None:
            msg = 'Unable to locate "{}" in this interface'.format(toInit.name)
            raise InvalidArgumentValue(msg)

        initArgs = {}
        for arg, val in toInit.kwargs.items():
            # recurse if another subject needs to be instantiated
            if isinstance(val, nimble.Init):
                subObj = self._argumentInit(val)
                initArgs[arg] = subObj
            else:
                initArgs[arg] = val

        # validate arguments
        self._validateArgumentValues(toInit.name, initArgs)

        return initObject(**initArgs)


    def setOption(self, option, value):
        """
        TODO
        """
        if option not in self.optionNames:
            msg = str(option)
            msg += " is not one of the accepted configurable option names"
            raise InvalidArgumentValue(msg)

        nimble.settings.set(self.getCanonicalName(), option, value)


    def getOption(self, option):
        """
        TODO
        """
        if option not in self.optionNames:
            msg = str(option)
            msg += " is not one of the accepted configurable option names"
            raise InvalidArgumentValue(msg)

        # empty string is the sentinal value indicating that the configuration
        # file has an option of that name, but the nimble user hasn't set a
        # value for it.
        ret = ''
        try:
            ret = nimble.settings.get(self.getCanonicalName(), option)
        except configErrors:
            # it is possible that the config file doesn't have an option of
            # this name yet. Just pass through and grab the hardcoded default
            pass
        if ret == '':
            ret = self._optionDefaults(option)
        return ret


    def _chooseBestParameterSet(self, possibleParamsSets, matchingDefaults,
                                arguments):
        success = False
        missing = []
        bestParams = []
        nonDefaults = []
        length = len(possibleParamsSets)
        if length == 1:
            return 0

        for i in range(length):
            missing.append([])
        bestIndex = None
        for i in range(length):
            currParams = possibleParamsSets[i]
            currDefaults = matchingDefaults[i]
            nonDefaults.append([])
            allIn = True
            for param in currParams:
                if param not in currDefaults:
                    nonDefaults[i].append(param)
                if param not in arguments and param not in currDefaults:
                    allIn = False
                    missing[i].append(param)
            if allIn and len(currParams) >= len(bestParams):
                bestIndex = i
                success = True
        if not success:
            msg = "MISSING LEARNING PARAMETER(S)! "
            msg += "When trying to validate arguments, "
            msg += "we must pick the set of required parameters that best "
            msg += "match the given input. However, from each possible "
            msg += "(numbered) parameter set, the following parameter names "
            msg += "were missing "
            msg += _prettyListString(missing, numberItems=True,
                                     itemStr=_prettyListString)
            msg += ". The following lists the required names in each of the "
            msg += "possible (numbered) parameter sets: "
            msg += _prettyListString(nonDefaults, numberItems=True,
                                     itemStr=_prettyListString)
            if len(arguments) == 0:
                msg += ". However, no arguments were inputed."
            else:
                msg += ". The full mapping of inputs actually provided was: "
                msg += _prettyDictString(arguments) + ". "

            raise InvalidArgumentValue(msg)

        return bestIndex

    def _getMethodArguments(self, argNames, newArguments, storedArguments):
        applyArgs = {}
        invalidArguments = []
        for arg, value in newArguments.items():
            # valid argument change
            if arg in argNames:
                applyArgs[arg] = value
            # not a valid argument for method
            else:
                invalidArguments.append(arg)
        if invalidArguments:
            msg = "EXTRA PARAMETER! "
            if argNames:
                msg += "The following parameter names cannot be applied: "
                msg += _prettyListString(invalidArguments, useAnd=True)
                msg += ". The allowed parameters are: "
                msg += _prettyListString(argNames, useAnd=True)
            else:
                msg += "No parameters are accepted for this operation"
            raise InvalidArgumentValue(msg)

        # use stored values for any remaining arguments
        for arg in argNames:
            if arg not in applyArgs and arg in storedArguments:
                applyArgs[arg] = storedArguments[arg]

        return applyArgs


    ##############################################
    ### CACHING FRONTENDS FOR ABSTRACT METHODS ###
    ##############################################

    @captureOutput
    def listLearners(self):
        """
        Return a list of all learners callable through this interface.
        """
        if self._listLearnersCached is None:
            ret = self._listLearnersBackend()
            self._listLearnersCached = ret
        else:
            ret = self._listLearnersCached
        return ret

    @captureOutput
    @cacheWrapper
    def findCallable(self, name):
        """
        Find reference to the callable with the given name.

        Parameters
        ----------
        name : str
            The name of the callable.

        Returns
        -------
        Reference to in-package function or constructor.
        """
        return self._findCallableBackend(name)

    @cacheWrapper
    def _getParameterNames(self, name):
        """
        Find params for instantiation and function calls.

        Parameters
        ----------
        name : str
            The name of the class or function.

        Returns
        -------
        List of list of param names to make the chosen call.
        """
        ret = self._getParameterNamesBackend(name)
        if ret is not None:
            # Backend may contain duplicates but only one value can be assigned
            # to a parameter so we use sets to remove duplicates and sorted
            # to convert the sets back to lists with a consistent order
            ret = [sorted(set(lst)) for lst in ret]

        return ret

    @captureOutput
    @cacheWrapper
    def getLearnerParameterNames(self, learnerName):
        """
        Find learner parameter names for a trainAndApply() call.

        Parameters
        ----------
        learnerName : str
            The name of the learner.

        Returns
        -------
        List of list of param names
        """
        ret = self._getLearnerParameterNamesBackend(learnerName)
        if ret is not None:
            # Backend may contain duplicates but only one value can be assigned
            # to a parameter so we use sets to remove duplicates and sorted
            # to convert the sets back to lists with a consistent order
            ret = [sorted(set(lst)) for lst in ret]

        return ret

    @cacheWrapper
    def _getDefaultValues(self, name):
        """
        Find default values.

        Parameters
        ----------
        name : str
            The name of the class or function.

        Returns
        -------
        List of dict of param names to default values.
        """
        return self._getDefaultValuesBackend(name)

    @captureOutput
    @cacheWrapper
    def getLearnerDefaultValues(self, learnerName):
        """
        Find learner default parameter values for trainAndApply() call.

        Parameters
        ----------
        learnerName : str
            The name of the learner.

        Returns
        -------
        List of dict of param names to default values
        """
        return self._getLearnerDefaultValuesBackend(learnerName)

    ########################
    ### ABSTRACT METHODS ###
    ########################

    @abc.abstractmethod
    def accessible(self):
        """
        Determine the accessibility of the underlying interface package.

        Returns
        -------
        bool
            True if the package currently accessible, False otherwise.
        """
        pass

    @abc.abstractmethod
    def getCanonicalName(self):
        """
        The string name that will uniquely identify this interface.

        Returns
        -------
        str
            The canonical name for this interface.
        """
        pass

    @abc.abstractmethod
    def _listLearnersBackend(self):
        pass

    @abc.abstractmethod
    def _findCallableBackend(self, name):

        pass

    @abc.abstractmethod
    def _getParameterNamesBackend(self, name):
        pass

    @abc.abstractmethod
    def _getLearnerParameterNamesBackend(self, learnerName):
        pass

    @abc.abstractmethod
    def _getDefaultValuesBackend(self, name):
        pass

    @abc.abstractmethod
    def _getLearnerDefaultValuesBackend(self, learnerName):
        pass

    @abc.abstractmethod
    def learnerType(self, name):
        """
        Returns a string referring to the action the learner takes out
        of the possibilities: classifier, regressor, featureSelection,
        dimensionalityReduction
        TODO
        """
        pass

    @abc.abstractmethod
    def _getScores(self, learnerName, learner, testX, newArguments,
                   storedArguments, customDict):
        """
        If the learner is a classifier, then return the scores for each
        class on each data point, otherwise raise an exception.
        """
        pass

    @abc.abstractmethod
    def _getScoresOrder(self, learner):
        """
        If the learner is a classifier, then return a list of the the
        labels corresponding to each column of the return from
        getScores.
        """
        pass

    @abc.abstractmethod
    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):
        """
        Method called before any package level function which transforms
        all parameters provided by a nimble user.

        trainX, trainY, and testX are filled with the values of the
        parameters of the same name to a call to trainAndApply() or
        train() and are sometimes empty when being called by other
        functions. For example, a call to apply() will have trainX and
        trainY be None. The arguments parameter is a dictionary mapping
        names to values of all other parameters associated with the
        learner, each of which may need to be processed.

        The return value of this function must be a tuple mirroring the
        structure of the inputs. Specifically, four values are required:
        the transformed versions of trainX, trainY, testX, and arguments
        in that specific order.
        """
        pass

    @abc.abstractmethod
    def _outputTransformation(self, learnerName, outputValue,
                              transformedInputs, outputType, outputFormat,
                              customDict):
        """
        Method called before any package level function which transforms
        the returned value into a format appropriate for a nimble user.
        """
        pass

    @abc.abstractmethod
    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        """
        Build a learner and perform training with the given data.

        Parameters
        ----------
        learnerName : str
            The name of the learner.
        trainX : nimble.core.data.Base
            The training data.
        trainY : nimble.core.data.Base
            The training labels.
        arguments : dict
            The transformed arguments.
        customDict : dict
            The customizable dictionary that is passed for training a
            learner.

        Returns
        -------
        An in package object to be wrapped by a TrainedLearner object.
        """
        pass

    @abc.abstractmethod
    def _incrementalTrainer(self, learnerName, learner, trainX, trainY,
                            arguments, customDict):
        """
        Extend the training of an already trained online learner.

        Parameters
        ----------
        learnerName : str
            The name of the learner.
        trainX : nimble.core.data.Base
            The training data.
        trainY : nimble.core.data.Base
            The training labels.
        arguments : dict
            The transformed arguments.
        customDict : dict
            The customizable dictionary that is passed for training a
            learner.

        Returns
        -------
        The learner after this batch of training.
        """
        pass


    @abc.abstractmethod
    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        """
        Perform testing/prediction on the test set using TrainedLearner.

        Parameters
        ----------
        learnerName : str
            A TrainedLearner object that can be tested on.
        testX : nimble.core.data.Base
            The testing data.
        arguments : dict
            The transformed arguments.
        customDict : dict
            The customizable dictionary that is passed for applying a
            learner.

        Returns
        -------
        nimble friendly results.
        """
        pass


    @abc.abstractmethod
    def _getAttributes(self, learnerBackend):
        """
        Returns whatever attributes might be available for the given
        learner. For example, in the case of linear regression, TODO
        """
        pass

    @abc.abstractmethod
    def _optionDefaults(self, option):
        """
        Define package default values that will be used for as long as a
        default value hasn't been registered in the nimble configuration
        file. For example, these values will always be used the first
        time an interface is instantiated.
        """
        pass


    @abc.abstractmethod
    def _configurableOptionNames(self):
        """
        Returns a list of strings, where each string is the name of a
        configurable option of this interface whose value will be stored
        in nimble's configuration file.
        """
        pass

    @abc.abstractmethod
    def _exposedFunctions(self):
        """
        Returns a list of references to functions which are to be
        wrapped in I/O transformation, and exposed as attributes of all
        TrainedLearner objects returned by this interface's train()
        function. If None, or an empty list is returned, no functions
        will be exposed. Each function in this list should be a python
        function, the inspect module will be used to retrieve argument
        names, and the value of the function's __name__ attribute will
        be its name in TrainedLearner.
        """
        pass

    @abc.abstractmethod
    def version(self):
        """
        The version of the package accessible to the interface.

        Returns
        -------
        str
            The version of this interface as a string.
        """
        pass

##################
# TrainedLearner #
##################

class TrainedLearner(object):
    """
    Container for a learner model that has been trained.

    Provides methods for applying and testing the trained model.

    Attributes
    ----------
    learnerName : str
        The name of the learner used for training.
    arguments : dict
        The original arguments passed to the learner.
    crossValidation : KFoldCrossValidator
        The object containing the cross-validation results, provided
        cross-validation occurred.  See the Attributes section in
        ``help(nimble.core.learn.KFoldCrossValidator)`` or the Examples
        section in ``help(nimble.crossValidate)`` for more information
        on extracting various results from this object.
    """
    def __init__(self, learnerName, arguments, transformedArguments,
                 customDict, backend, interfaceObject, has2dOutput,
                 crossValidationResults, trainXShape, trainYNames):
        """
        Initialize the object wrapping the trained learner stored in
        backend, and setting up the object methods that may be used to
        modify or query the backend trained learner.

        Parameters
        ----------
        learnerName : str
            The name of the learner used in the backend.
        arguments : dict
            Reference to the original arguments parameter to the
            trainAndApply() function.
        transformedArguments : tuple
            Contains the return value of _inputTransformation() that was
            called when training the learner in the backend.
        customDict : dict
            Reference to the customizable dictionary that is passed to
            I/O transformation, training and applying a learner.
        backend : object
            The return value from _trainer(), a reference to some object
            that is to be used by the package implementor during
            application.
        interfaceObject : nimble.core.interfaces.UniversalInterface
            A reference to the subclass of UniversalInterface from which
            this TrainedLearner is being instantiated.
        has2dOutput : bool
            True if output will be 2-dimensional, False assumes the
            output will be 1-dimensional.
        crossValidationResults : KFoldsCrossValidator
            The object containing the cross-validation results if
            cross-validation occurred while training the learner,
            otherwise None.
        trainXShape : tuple
            The shape, (numPts, numFts), of the trainX object.
        """
        self.learnerName = learnerName
        self.arguments = arguments
        self._transformedArguments = transformedArguments
        self._customDict = customDict
        self._backend = backend
        self._interface = interfaceObject
        self._has2dOutput = has2dOutput
        self.crossValidation = crossValidationResults
        self._trainXShape = trainXShape
        self._trainYNames = trainYNames

        exposedFunctions = self._interface._exposedFunctions()
        for exposed in exposedFunctions:
            methodName = getattr(exposed, '__name__')
            (args, _, _, _) = inspectArguments(exposed)
            doc = 'Wrapped version of the ' + methodName + ' function where '
            if 'trainedLearner' in args:
                wrapped = functools.partial(exposed, trainedLearner=self)
                doc += 'the "trainedLearner" parameter has been fixed as this '
                doc += 'object, and '
            else:
                wrapped = functools.partial(exposed)
            doc += 'the "self" parameter has been fixed to be '
            doc += str(interfaceObject)
            wrapped.__doc__ = doc
            setattr(self, methodName, wrapped)

    @captureOutput
    def test(self, testX, testY, performanceFunction, arguments=None,
             output='match', scoreMode='label', useLog=None, **kwarguments):
        """
        Evaluate the performance of the trained learner.

        Evaluation of predictions of ``testX`` using the argument
        ``performanceFunction`` to do the evaluation. Equivalent to
        having called ``trainAndTest``, as long as the data and
        parameter setup for training was the same.

        Parameters
        ----------
        testX : nimble.core.data.Base
            The object containing the test data.
        testY : identifier, nimble Base object
            * identifier - A name or index of the feature in ``testX``
              containing the labels.
            * nimble Base object - contains the labels that correspond
              to ``testX``.
        performanceFunction : function
            If cross validation is triggered to select from the given
            argument set, then this function will be used to generate a
            performance score for the run. Function is of the form:
            def func(knownValues, predictedValues).
            Look in nimble.calculate for pre-made options. Default is
            None, since if there is no parameter selection to be done,
            it is not used.
        arguments : dict
            Mapping argument names (strings) to their values, to be used
            during training and application. eg. {'dimensions':5, 'k':5}
            To make use of multiple permutations, specify different
            values for a parameter as a tuple. eg. {'k': (1,3,5)} will
            generate an error score for  the learner when the learner
            was passed all three values of ``k``, separately. These will
            be merged with kwarguments for the learner.
        output : str
            The kind of nimble Base object that the output of this
            function should be in. Any of the normal string inputs to
            the nimble.data ``returnType`` parameter are accepted here.
            Alternatively, the value 'match' will indicate to use the
            type of the ``trainX`` parameter.
        scoreMode : str
            In the case of a classifying learner, this specifies the
            type of output wanted: 'label' if we class labels are
            desired, 'bestScore' if both the class label and the score
            associated with that class are desired, or 'allScores' if a
            matrix containing the scores for every class label are
            desired.
        useLog : bool, None
            Local control for whether to send results/timing to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        kwarguments
            Keyword arguments specified variables that are passed to the
            learner. To make use of multiple permutations, specify
            different values for parameters as a tuple.
            eg. arg1=(1,2,3), arg2=(4,5,6) which correspond to
            permutations/argument states with one element from arg1 and
            one element from arg2, such that an example generated
            permutation/argument state would be ``arg1=2, arg2=4``.
            Will be merged with ``arguments``.

        Returns
        -------
        performance
            The calculated value of the ``performanceFunction`` after
            the test.

        See Also
        --------
        nimble.trainAndTest, apply

        Examples
        --------
        TODO
        """
        timer = startTimer(useLog)
        if isinstance(useLog, TrainValidator):
            validator = useLog
            useLog = False # logging occurs where TrainValidator was generated
        else:
            validator = TrainValidator()
        validator.validateTestData(testX, testY, testYRequired=True)
        validator.validateScoreMode(scoreMode)
        validator.validateArguments(arguments)
        validator.validateOutputFlags(self._has2dOutput, None, scoreMode, None)

        mergedArguments = mergeArguments(arguments, kwarguments)

        if not isinstance(testY, nimble.core.data.Base):
            testX = testX.copy()
            testY = testX.features.extract(testY, useLog=False)

        pred = self.apply(testX, mergedArguments, output, scoreMode,
                          useLog=validator)
        performance = computeMetrics(testY, None, pred, performanceFunction)
        time = stopTimer(timer)

        metrics = {}
        for key, value in zip([performanceFunction], [performance]):
            metrics[key.__name__] = value

        fullName = self._interface.getCanonicalName() + self.learnerName
        # Signature:
        # (nimbleFunction, trainData, trainLabels, testData, testLabels,
        # learnerFunction, arguments, metrics, extraInfo=None, folds=None)
        handleLogging(useLog, 'run', "TrainedLearner.test", trainData=None,
                      trainLabels=None, testData=testX, testLabels=testY,
                      learnerFunction=fullName, arguments=mergedArguments,
                      metrics=metrics, extraInfo=None, time=time)

        return performance

    @captureOutput
    def apply(self, testX, arguments=None, output='match', scoreMode='label',
              useLog=None, **kwarguments):
        """
        Apply the learner to the test data.

        Return the application of this learner to the given test data
        (i.e. performing prediction, transformation, etc. as appropriate
        to the learner). If ``testX`` has pointNames and the output
        object has the same number of points, the pointNames from
        ``testX`` will be applied to the output object. Equivalent to
        having called ``trainAndApply``, as long as the data and
        parameter setup for training was the same.

        Parameters
        ----------
        testX : nimble Base object
            Data set on which the trained learner will be applied (i.e.
            performing prediction, transformation, etc. as appropriate
            to the learner).
        arguments : dict
            Mapping argument names (strings) to their values, to be used
            during training and application. eg. {'dimensions':5, 'k':5}
            To make use of multiple permutations, specify different
            values for a parameter as a tuple. eg. {'k': (1,3,5)} will
            generate an error score for  the learner when the learner
            was passed all three values of ``k``, separately. These will
            be merged with kwarguments for the learner.
        output : str
            The kind of nimble Base object that the output of this
            function should be in. Any of the normal string inputs to
            the nimble.data ``returnType`` parameter are accepted here.
            Alternatively, the value 'match' will indicate to use the
            type of the ``trainX`` parameter.
        scoreMode : str
            In the case of a classifying learner, this specifies the
            type of output wanted: 'label' if class labels are desired,
            'bestScore' if both the class label and the score associated
            with that class are desired, or 'allScores' if a matrix
            containing the scores for every class label are desired.
        useLog : bool, None
            Local control for whether to send results/timing to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        kwarguments
            Keyword arguments specified variables that are passed to the
            learner. To make use of multiple permutations, specify
            different values for parameters as a tuple.
            eg. arg1=(1,2,3), arg2=(4,5,6) which correspond to
            permutations/argument states with one element from arg1 and
            one element from arg2, such that an example generated
            permutation/argument state would be ``arg1=2, arg2=4``.
            Will be merged with ``arguments``.

        Returns
        -------
        results
            The resulting output of applying learner.

        See Also
        --------
        nimble.trainAndApply, test

        Examples
        --------
        >>> rawTrain = [[1, 0, 0, 1],
        ...             [0, 1, 0, 2],
        ...             [0, 0, 1, 3],
        ...             [1, 0, 0, 1],
        ...             [0, 1, 0, 2],
        ...             [0, 0, 1, 3]]
        >>> rawTestX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> trainData = nimble.data('Matrix', rawTrain)
        >>> testX = nimble.data('Matrix', rawTestX)
        >>> tl = nimble.train('nimble.KNNClassifier', trainX=trainData,
        ...                   trainY=3)
        >>> predict = tl.apply(testX)
        >>> predict
        Matrix(
            [[1]
             [2]
             [3]]
            )
        """
        timer = startTimer(useLog)
        if isinstance(useLog, TrainValidator):
            validator = useLog
            useLog = False # logging occurs where TrainValidator was generated
        else:
            validator = TrainValidator()
        validator.validateTestData(testX)
        validator.validateScoreMode(scoreMode)
        validator.validateArguments(arguments)
        validator.validateOutputFlags(self._has2dOutput, None, scoreMode, None)

        mergedArguments = mergeArguments(arguments, kwarguments)
        self._validTestDataShape(testX)

        # input transformation
        transformedInputs = self._interface._inputTransformation(
            self.learnerName, None, None, testX, mergedArguments,
            self._customDict)
        transTestX = transformedInputs[2]
        usedArguments = transformedInputs[3]

        # depending on the mode, we need different information.
        labels = None
        if scoreMode != 'label':
            scores = self.getScores(testX, usedArguments)
        if scoreMode != 'allScores':
            labels = self._interface._applier(self.learnerName, self._backend,
                                              transTestX, usedArguments,
                                              self._transformedArguments,
                                              self._customDict)
            labels = self._interface._outputTransformation(
                self.learnerName, labels, usedArguments, output, "label",
                self._customDict)
        # if this application is for a classification or regression learner,
        # we will apply featureNames to the output if possible
        lType = self._interface.learnerType(self.learnerName)
        applyFtNames = lType in ['classification', 'regression']
        if scoreMode == 'label':
            ret = labels
            if applyFtNames:
                ret.features.setNames(self._trainYNames, useLog=False)
        elif scoreMode == 'allScores':
            ret = scores
            if applyFtNames:
                scoreOrder = self._interface._getScoresOrder(self._backend)
                ret.features.setNames(map(str, scoreOrder), useLog=False)
        elif scoreMode == 'bestScore':
            scoreOrder = self._interface._getScoresOrder(self._backend)
            scoreOrder = list(scoreOrder)
            # find scores matching predicted labels
            def grabValue(row):
                pointIndex = scores.points.getIndex(row.points.getName(0))
                rowIndex = scoreOrder.index(labels[pointIndex, 0])
                return row[rowIndex]

            scoreVector = scores.points.calculate(grabValue, useLog=False)
            labels.features.append(scoreVector, useLog=False)

            ret = labels
            if applyFtNames and self._trainYNames is not None:
                ftNames = self._trainYNames.copy()
                ftNames.append('bestScore')
                ret.features.setNames(ftNames, useLog=False)
        else:
            msg = 'scoreMode must be "label", "bestScore", or "allScores"'
            raise InvalidArgumentValue(msg)

        if len(testX.points) == len(ret.points):
            ret.points.setNames(testX.points._getNamesNoGeneration(),
                                useLog=False)

        time = stopTimer(timer)

        fullName = self._interface.getCanonicalName() + self.learnerName
        # Signature:
        # (self, nimbleFunction, trainData, trainLabels, testData, testLabels,
        # learnerFunction, arguments, metrics, extraInfo=None, folds=None
        handleLogging(useLog, 'run', "TrainedLearner.apply", trainData=None,
                      trainLabels=None, testData=testX, testLabels=None,
                      learnerFunction=fullName, arguments=mergedArguments,
                      metrics=None, extraInfo=None, time=time)

        return ret

    def save(self, outputPath):
        """
        Save model to a file.

        Uses the cloudpickle library to serialize this object.

        Parameters
        ----------
        outputPath : str
            The location (including file name and extension) where
            we want to write the output file. If filename extension
            .nimm is not included in file name it would be added to the
            output file.

        Examples
        --------
        TODO
        """
        if not cloudpickle.nimbleAccessible():
            msg = "To save nimble models, cloudpickle must be installed"
            raise PackageException(msg)
        extension = '.nimm'
        if not outputPath.endswith(extension):
            outputPath = outputPath + extension

        with open(outputPath, 'wb') as file:
            cloudpickle.dump(self, file)

    @captureOutput
    def retrain(self, trainX, trainY=None, arguments=None, useLog=None,
                **kwarguments):
        """
        Train the model on new data.

        Adjust the learner model by providing new training data and/or
        changing the learner's argument values. Previously set argument
        values for this learner model will remain the same, unless
        overridden by new arguments or kwarguments. If new data is
        provided, the learner will be trained on that data only,
        discarding the previous data.

        Parameters
        ----------
        trainX: nimble Base object
            Data to be used for training.
        trainY : identifier, nimble Base object
            * identifier - The name or index of the feature in
              ``trainX`` containing the labels.
            * nimble Base object - contains the labels that correspond
              to ``trainX``.
        arguments : dict
            Mapping argument names (strings) to their values, to be used
            during training.  These must be singular values, retrain
            does not implement cross-validation for multiple argument
            sets. Will be merged with kwarguments.
        useLog : bool, None
            Local control for whether to send results/timing to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        kwarguments
            Keyword arguments specified variables that are passed to the
            learner. Must be singular values, retrain does not implement
            cross-validation for multiple arguments sets.
            Will be merged with ``arguments``.

        See Also
        --------
        incrementalTrain

        Examples
        --------
        Changing the training data.

        >>> rawTrainX1 = [[1, 1], [2, 2], [3, 3]]
        >>> trainX1 = nimble.data('Matrix', rawTrainX1)
        >>> rawTrainY1 = [[1], [2], [3]] # mean of 2
        >>> trainY1 = nimble.data('Matrix', rawTrainY1)
        >>> rawTestX = [[8, 8], [-3, -3]]
        >>> testX = nimble.data('Matrix', rawTestX)
        >>> tl = nimble.train('nimble.MeanConstant', trainX1, trainY1)
        >>> tl.apply(testX)
        Matrix(
            [[2.000]
             [2.000]]
            )
        >>> rawTrainX2 = [[4, 4], [5, 5], [6, 6]]
        >>> trainX2 = nimble.data('Matrix', rawTrainX2)
        >>> rawTrainY2 = [[4], [5], [6]] # mean of 5
        >>> trainY2 = nimble.data('Matrix', rawTrainY2)
        >>> tl.retrain(trainX2, trainY2)
        >>> tl.apply(testX)
        Matrix(
            [[5.000]
             [5.000]]
            )

        Changing the learner arguments.

        >>> rawTrainX = [[1, 1], [3, 3], [3, 3]]
        >>> trainX = nimble.data('Matrix', rawTrainX)
        >>> rawTrainY = [[1], [3], [3]]
        >>> trainY = nimble.data('Matrix', rawTrainY)
        >>> rawTestX = [[1, 1], [3, 3]]
        >>> testX = nimble.data('Matrix', rawTestX)
        >>> tl = nimble.train('nimble.KNNClassifier', trainX, trainY,
        ...                   k=1)
        >>> tl.apply(testX)
        Matrix(
            [[1]
             [3]]
            )
        >>> tl.retrain(trainX, trainY, k=3)
        >>> tl.apply(testX)
        Matrix(
            [[3]
             [3]]
            )
        """
        validator = TrainValidator()
        validator.validateTrainData(trainX, trainY)
        validator.validateArguments(arguments)

        has2dOutput = False
        outputData = trainX if trainY is None else trainY
        if isinstance(outputData, nimble.core.data.Base):
            has2dOutput = len(outputData.features) > 1
        elif isinstance(outputData, (list, tuple)):
            has2dOutput = len(outputData) > 1

        merged = mergeArguments(arguments, kwarguments)
        self._interface._validateLearnerArgumentValues(self.learnerName,
                                                       merged)
        for arg, value in merged.items():
            if isinstance(value, nimble.CV):
                msg = "Cannot provide a cross-validation arguments "
                msg += "for parameters to retrain a TrainedLearner. "
                msg += "If wanting to perform cross-validation, use "
                msg += "nimble.train()"
                raise InvalidArgumentValue(msg)
            self.arguments[arg] = value
            self._transformedArguments[arg] = value

        # separate training data / labels if needed
        if isinstance(trainY, (str, int, numpy.integer)):
            trainX = trainX.copy()
            trainY = trainX.features.extract(toExtract=trainY, useLog=False)

        trainedBackend = self._interface._trainBackend(
            self.learnerName, trainX, trainY, self._transformedArguments)

        newBackend = trainedBackend[0]
        transformedInputs = trainedBackend[1]
        customDict = trainedBackend[2]

        self._backend = newBackend
        self._trainXShape = trainX.shape
        self.arguments = merged
        self._transformedArguments = transformedInputs[3]
        self._customDict = customDict
        self._has2dOutput = has2dOutput
        self.crossValidation = None

        handleLogging(useLog, 'run', 'TrainedLearner.retrain', trainX, trainY,
                      None, None, self.learnerName, self.arguments, None)

    @captureOutput
    def incrementalTrain(self, trainX, trainY=None, useLog=None):
        """
        Extend the training of this learner with additional data.

        Using the data the model was previously trained on, continue
        training this learner by supplementing the existing data with
        the provided additional data.

        Parameters
        ----------
        trainX: nimble Base object
            Additional data to be used for training.
        trainY : identifier, nimble Base object
            A name or index of the feature in ``trainX`` containing the
            labels or another nimble Base object containing the labels
            that correspond to ``trainX``.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        """
        validator = TrainValidator()
        validator.validateTrainData(trainX, trainY)

        transformed = self._interface._inputTransformation(
            self.learnerName, trainX, trainY, None, self.arguments,
            self._customDict)
        transformedTrainX = transformed[0]
        transformedTrainY = transformed[1]
        transformedArguments = transformed[3]
        self._backend = self._interface._incrementalTrainer(
            self.learnerName, self._backend, transformedTrainX,
            transformedTrainY, transformedArguments, self._customDict)

        handleLogging(useLog, 'run', 'TrainedLearner.incrementalTrain', trainX,
                      trainY, None, None, self.learnerName, self.arguments,
                      None)

    @captureOutput
    def getAttributes(self):
        """
        The attributes associated with this learner.

        Return the attributes of the trained learner (and sub objects).
        The returned value will be a dict, mapping names of attribtues
        to values of attributes. In the case of collisions (especially
        when getting attributes from nested objects) the attribute names
        may be prefaced with the name of the object from which they
        originate.

        Returns
        -------
        dict
            A mapping of attribute name to values of attributes.
        """
        discovered = self._interface._getAttributes(self._backend)
        inputs = self.arguments

        if inputs is not None:
            for key in inputs.keys():
                value = inputs[key]
                if key in list(discovered.keys()):
                    if value != discovered[key]:
                        newKey = self.learnerName + '.' + key
                        discovered[newKey] = discovered[key]
                    discovered[key] = value

        return discovered

    def _formatScoresToOvA(self, testX, applyResults, rawScores, arguments):
        """
        Helper that takes raw scores in any of the three accepted
        formats (binary case best score, one vs one pairwise tournament
        by natural label ordering, or one vs all by natural label
        ordering) and returns them in a one vs all accepted format.
        """
        order = self._interface._getScoresOrder(self._backend)
        numLabels = len(order)
        if numLabels == 2 and len(rawScores.features) == 1:
            ret = generateBinaryScoresFromHigherSortedLabelScores(rawScores)
            return nimble.data("Matrix", ret, useLog=False)

        if applyResults is None:
            applyResults = self._interface._applier(
                self.learnerName, self._backend, testX, arguments,
                self._transformedArguments, self._customDict)
            applyResults = self._interface._outputTransformation(
                self.learnerName, applyResults, arguments, "match", "label",
                self._customDict)
        if len(rawScores.features) != 3:
            strategy = ovaNotOvOFormatted(rawScores, applyResults, numLabels)
        else:
            strategy = checkClassificationStrategy(
                self._interface, self.learnerName, arguments)
        # want the scores to be per label, regardless of the original format,
        # so we check the strategy, and modify it if necessary
        if not strategy:
            scores = []
            for i in range(len(rawScores.points)):
                combinedScores = calculateSingleLabelScoresFromOneVsOneScores(
                    rawScores.pointView(i), numLabels)
                scores.append(combinedScores)
            scores = numpy.array(scores)
            return nimble.data("Matrix", scores, useLog=False)
        else:
            return rawScores


    @captureOutput
    def getScores(self, testX, arguments=None, **kwarguments):
        """
        The scores for all labels for each data point.

        This is the equivalent of calling TrainedLearner's apply method
        with ``scoreMode="allScores"``.

        Returns
        -------
        nimble.core.data.Matrix
            The label scores.
        """
        validator = TrainValidator()
        validator.validateTestData(testX)
        validator.validateArguments(arguments)
        self._validTestDataShape(testX)
        usedArguments = mergeArguments(arguments, kwarguments)
        (_, _, testX, usedArguments) = self._interface._inputTransformation(
            self.learnerName, None, None, testX, usedArguments,
            self._customDict)

        rawScores = self._interface._getScores(self.learnerName, self._backend,
                                               testX, usedArguments,
                                               self._transformedArguments,
                                               self._customDict)
        nimbleTypeRawScores = self._interface._outputTransformation(
            self.learnerName, rawScores, usedArguments, "Matrix", "allScores",
            self._customDict)
        formatedRawOrder = self._formatScoresToOvA(
            testX, None, nimbleTypeRawScores, usedArguments)
        internalOrder = self._interface._getScoresOrder(self._backend)
        naturalOrder = sorted(internalOrder)
        if numpy.array_equal(naturalOrder, internalOrder):
            return formatedRawOrder
        desiredDict = {}
        for i, label in enumerate(naturalOrder):
            desiredDict[label] = i

        def sortScorer(feature):
            name = feature.features.getName(0)
            index = formatedRawOrder.features.getIndex(name)
            label = internalOrder[index]
            return desiredDict[label]

        formatedRawOrder.features.sort(by=sortScorer, useLog=False)
        return formatedRawOrder


    def _validTestDataShape(self, testX):
        """
        Validate testing data is compatible with the training data.
        """
        if self._trainXShape[1] == len(testX.features):
            return
        trainXIsSquare = self._trainXShape[0] == self._trainXShape[1]
        if trainXIsSquare and len(testX.points) == len(testX.features):
            return
        msg = "The number of features in testX ({0}) must be equal to the "
        msg += "number of features in the training data ({1}) "
        msg = msg.format(len(testX.features), self._trainXShape[1])
        if trainXIsSquare:
            msg += "or the testing data must be square-shaped"
        raise InvalidArgumentValueCombination(msg)


@inheritDocstringsFactory(TrainedLearner)
class TrainedLearners(TrainedLearner):
    """
    Container for learner models trained using a multiClassStrategy.

    Provides methods for applying and testing the trained model.

    Attributes
    ----------
    learnerName : str
        The name of the learner used for training.
    arguments : dict
        The original arguments passed to the learner.
    crossValidation : KFoldCrossValidator
        The object containing the cross-validation results, provided
        cross-validation occurred.  See the Attributes section in
        ``help(nimble.core.learn.KFoldCrossValidator)`` or the Examples
        section in ``help(nimble.crossValidate)`` for more information
        on extracting various results from this object.
    method : str
        The multiClassStrategy used, "OneVsAll" or "OneVsOne".
    """
    def __init__(self, trainedLearners, method, labelSet):
        """
        Initialize the object wrapping the trained learner stored in
        backend, and setting up the object methods that may be used to
        modify or query the backend trained learner.

        Parameters
        ----------
        trainedLearners : list
            The list of TrainedLearner objects.
        method : str
            The multiClassStrategy used, "OneVsAll" or "OneVsOne".
        labelSet : list
            The list of unique labels.
        """
        self._trainedLearnersList = trainedLearners
        self.method = method
        self._labelSet = labelSet

        # we need the TrainedLearner attributes to complete the instantiation
        # we can access them using the first TrainedLearner in trainedLearners
        # because they all have identical attribute values
        trainedLearnerAttrs = trainedLearners[0]
        learnerName = trainedLearnerAttrs.learnerName
        arguments = trainedLearnerAttrs.arguments
        transformedArguments = trainedLearnerAttrs._transformedArguments
        customDict = trainedLearnerAttrs._customDict
        backend = trainedLearnerAttrs._backend
        interfaceObject = trainedLearnerAttrs._interface
        has2dOutput = trainedLearnerAttrs._has2dOutput
        crossValidationResults = trainedLearnerAttrs.crossValidation
        trainXShape = trainedLearnerAttrs._trainXShape
        trainYNames = trainedLearnerAttrs._trainYNames

        super(TrainedLearners, self).__init__(
            learnerName, arguments, transformedArguments, customDict, backend,
            interfaceObject, has2dOutput, crossValidationResults, trainXShape,
            trainYNames)

    @captureOutput
    def apply(self, testX, arguments=None, output='match', scoreMode='label',
              useLog=None, **kwarguments):
        """
        Apply the learner to the test data.

        Return the application of this learner to the given test data
        (i.e. performing prediction, transformation, etc. as appropriate
        to the learner). Equivalent to having called ``trainAndApply``,
        as long as the data and parameter setup for training was the
        same.

        Parameters
        ----------
        testX : nimble Base object
            Data set on which the trained learner will be applied (i.e.
            performing prediction, transformation, etc. as appropriate
            to the learner).
        performanceFunction : function
            If cross validation is triggered to select from the given
            argument set, then this function will be used to generate a
            performance score for the run. Function is of the form:
            def func(knownValues, predictedValues).
            Look in nimble.calculate for pre-made options. Default is
            None, since if there is no parameter selection to be done,
            it is not used.
        arguments : dict
            Mapping argument names (strings) to their values, to be used
            during training and application. eg. {'dimensions':5, 'k':5}
            To make use of multiple permutations, specify different
            values for a parameter as a tuple. eg. {'k': (1,3,5)} will
            generate an error score for  the learner when the learner
            was passed all three values of ``k``, separately. These will
            be merged with kwarguments for the learner.
        output : str
            The kind of nimble Base object that the output of this
            function should be in. Any of the normal string inputs to
            the nimble.data ``returnType`` parameter are accepted here.
            Alternatively, the value 'match' will indicate to use the
            type of the ``trainX`` parameter.
        scoreMode : str
            In the case of a classifying learner, this specifies the
            type of output wanted: 'label' if we class labels are
            desired, 'bestScore' if both the class label and the score
            associated with that class are desired, or 'allScores' if a
            matrix containing the scores for every class label are
            desired.
        useLog : bool, None
            Local control for whether to send results/timing to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        kwarguments
            Keyword arguments specified variables that are passed to the
            learner. To make use of multiple permutations, specify
            different values for parameters as a tuple.
            eg. arg1=(1,2,3), arg2=(4,5,6) which correspond to
            permutations/argument states with one element from arg1 and
            one element from arg2, such that an example generated
            permutation/argument state would be ``arg1=2, arg2=4``.
            Will be merged with ``arguments``.

        Returns
        -------
        results
            The resulting output of applying learner.

        See Also
        --------
        nimble.trainAndApply, test

        Examples
        --------
        TODO
        """
        rawPredictions = None
        #1 VS All
        if self.method == 'OneVsAll':
            for trainedLearner in self._trainedLearnersList:
                oneLabelResults = trainedLearner.apply(testX, arguments,
                                                       output, 'label',
                                                       useLog)
                label = trainedLearner.label
                # put all results into one Base container; same type as trainX
                if rawPredictions is None:
                    rawPredictions = oneLabelResults
                    # as it's added to results object,
                    # rename each column with its corresponding class label
                    rawPredictions.features.setName(0, str(label),
                                                    useLog=False)
                else:
                    # as it's added to results object,
                    # rename each column with its corresponding class label
                    oneLabelResults.features.setName(0, str(label),
                                                     useLog=False)
                    rawPredictions.features.append(oneLabelResults,
                                                   useLog=False)

            if scoreMode.lower() == 'label'.lower():

                getWinningPredictionIndices = rawPredictions.points.calculate(
                    extractWinningPredictionIndex, useLog=False)
                winningPredictionIndices = getWinningPredictionIndices.copy(
                    to="python list")
                winningLabels = []
                for [winningIndex] in winningPredictionIndices:
                    winningLabels.append([self._labelSet[int(winningIndex)]])
                return nimble.data(
                    rawPredictions.getTypeString(), winningLabels,
                    featureNames=['winningLabel'], useLog=False)

            elif scoreMode.lower() == 'bestScore'.lower():
                #construct a list of lists, with each row in the list
                # containing the predicted label and score of that label for
                # the corresponding row in rawPredictions
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
                resultsContainer = nimble.data("List", tempResultsList,
                                               featureNames=featureNames,
                                               useLog=False)
                return resultsContainer

            elif scoreMode.lower() == 'allScores'.lower():
                # create list of Feature Names/Column Headers for final
                # return object
                colHeaders = sorted([str(i) for i in self._labelSet])
                # create map between label and index in list,
                # so we know where to put each value
                colIndices = list(range(len(colHeaders)))
                labelIndexDict = {v: k for k, v in zip(colIndices, colHeaders)}
                featureNamesItoN = rawPredictions.features.getNames()
                predictionMatrix = rawPredictions.copy(to="python list")
                resultsContainer = []
                for row in predictionMatrix:
                    finalRow = [0] * len(colHeaders)
                    scores = extractConfidenceScores(
                        row, featureNamesItoN)
                    for label, score in scores.items():
                        #get numerical index of label in return object
                        finalIndex = labelIndexDict[label]
                        #put score into proper place in its row
                        finalRow[finalIndex] = score
                    resultsContainer.append(finalRow)
                #wrap data in Base container
                return nimble.data(rawPredictions.getTypeString(),
                                   resultsContainer, featureNames=colHeaders,
                                   useLog=False)
            else:
                msg = "scoreMode must be 'label', 'bestScore', or 'allScores'"
                raise InvalidArgumentValue(msg)

        #1 VS 1
        elif self.method == 'OneVsOne':
            predictionFeatureID = 0
            for trainedLearner in self._trainedLearnersList:
                # train classifier on that data; apply it to the test set
                partialResults = trainedLearner.apply(testX, arguments, output,
                                                      'label', useLog)
                # put predictions into table of predictions
                if rawPredictions is None:
                    rawPredictions = partialResults.copy(to="List")
                else:
                    predictionName = 'predictions-' + str(predictionFeatureID)
                    partialResults.features.setName(0, predictionName,
                                                    useLog=False)
                    rawPredictions.features.append(partialResults,
                                                   useLog=False)
                predictionFeatureID += 1
            # set up the return data based on which format has been requested
            if scoreMode.lower() == 'label'.lower():
                ret = rawPredictions.points.calculate(
                    extractWinningPredictionLabel, useLog=False)
                ret.features.setName(0, "winningLabel", useLog=False)
                return ret
            elif scoreMode.lower() == 'bestScore'.lower():
                # construct a list of lists, with each row in the list
                # containing the predicted label and score of that label for
                # the corresponding row in rawPredictions
                predictionMatrix = rawPredictions.copy(to="python list")
                tempResultsList = []
                for row in predictionMatrix:
                    scores = countWins(row)
                    sortedScores = sorted(scores, key=scores.get, reverse=True)
                    bestLabel = sortedScores[0]
                    tempResultsList.append([bestLabel, scores[bestLabel]])

                #wrap the results data in a List container
                featureNames = ['PredictedClassLabel', 'LabelScore']
                resultsContainer = nimble.data("List", tempResultsList,
                                               featureNames=featureNames,
                                               useLog=False)
                return resultsContainer
            elif scoreMode.lower() == 'allScores'.lower():
                colHeaders = sorted([str(float(i)) for i in self._labelSet])
                colIndices = list(range(len(colHeaders)))
                labelIndexDict = {v: k for k, v in zip(colIndices, colHeaders)}
                predictionMatrix = rawPredictions.copy(to="python list")
                resultsContainer = []
                for row in predictionMatrix:
                    finalRow = [0] * len(colHeaders)
                    scores = countWins(row)
                    for label, score in scores.items():
                        finalIndex = labelIndexDict[str(float(label))]
                        finalRow[finalIndex] = score
                    resultsContainer.append(finalRow)

                return nimble.data(rawPredictions.getTypeString(),
                                   resultsContainer, featureNames=colHeaders,
                                   useLog=False)
            else:
                msg = "scoreMode must be 'label', 'bestScore', or 'allScores'"
                raise InvalidArgumentValue(msg)
        else:
            raise ImproperObjectAction('Wrong multiclassification method.')


#######################
# PredefinedInterface #
#######################

pathMessage = """
If {name} installed
{underline}
    Make sure the package is installed in the current environment, append the
    path to the package to sys.path prior to importing nimble, or provide the
    path location in configuration.ini. The path can be set manually or call:
        nimble.settings.setDefault('{name}', 'location', '/path/to/package/')
    replacing '/path/to/package/' with the actual path to the directory
    containing the package."""

def formatPathMessage(name):
    underline = '-' * (len(name) + 13)
    return pathMessage.format(name=name, underline=underline)


class PredefinedInterface(abc.ABC):
    """
    Abstract base class of classmethods for predefined interfaces.

    For predefined interfaces, we need class methods to access certain
    information about the class and provide a detailed exception if the
    user attempts to use the interface, but it failed instantiation.
    """
    @classmethod
    @abc.abstractmethod
    def getCanonicalName(cls):
        pass

    @classmethod
    def isAlias(cls, name):
        return name.lower() == cls.getCanonicalName().lower()

    @classmethod
    def provideInitExceptionInfo(cls):
        """
        Provide traceback and reason the interface is not available.
        """
        name = cls.getCanonicalName()
        try:
            return cls()
        except Exception as e:
            origTraceback = e.__traceback__
            msg = "The " + name + " interface is not available because "
            msg += "the interface object could not be instantiated. "
            # could not import package, provide information to help with import
            if isinstance(e, ImportError):
                msg += cls._installInstructions()
                # instructions for providing path to package
                msg += formatPathMessage(name)
            raise PackageException(msg).with_traceback(origTraceback)

    @classmethod
    @abc.abstractmethod
    def _installInstructions(cls):
        pass
