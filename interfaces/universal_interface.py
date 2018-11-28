"""




"""

from __future__ import absolute_import
import inspect
import copy
import abc
import functools
import numpy
import sys
import numbers

import UML
from UML.exceptions import ArgumentException
from UML.exceptions import prettyListString
from UML.exceptions import prettyDictString
from UML.interfaces.interface_helpers import generateBinaryScoresFromHigherSortedLabelScores
from UML.interfaces.interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
from UML.interfaces.interface_helpers import ovaNotOvOFormatted
from UML.interfaces.interface_helpers import checkClassificationStrategy
from UML.interfaces.interface_helpers import cacheWrapper
from UML.logger import Stopwatch

from UML.helpers import _mergeArguments, generateAllPairs, countWins, inspectArguments
import six
from six.moves import range
import warnings

cloudpickle = UML.importModule('cloudpickle')

def captureOutput(toWrap):
    """Decorator which will safefly redirect standard error within the
    wrapped function to the temp file at UML.capturedErr

    """

    def wrapped(*args, **kwarguments):
        backupErr = sys.stderr
        sys.stderr = UML.capturedErr
        try:
            ret = toWrap(*args, **kwarguments)
        finally:
            sys.stderr = backupErr
        return ret

    return wrapped


class UniversalInterface(six.with_metaclass(abc.ABCMeta, object)):
    """

    """

    _listLearnersCached = None

    def __init__(self):
        """

        """
        ### Validate all the information from abstract functions ###
        # enforce a check that the underlying package is accessible at instantiation,
        # aborting the construction of the interface for this session of UML if
        # it is not.
        if not self.accessible():
            raise ImportError(
                "The underlying package for " + self.getCanonicalName() + " was not accessible, aborting instantiation.")

        # getCanonicalName
        if not isinstance(self.getCanonicalName(), str):
            raise TypeError("Improper implementation of getCanonicalName(), must return a string")

        # _configurableOptionNames and _optionDefaults
        optionNames = self._configurableOptionNames()
        if not isinstance(optionNames, list):
            raise TypeError("Improper implementation of _configurableOptionNames(), must return a list of strings")
        for optionName in optionNames:
            if not isinstance(optionName, str):
                raise TypeError("Improper implementation of _configurableOptionNames(), must return a list of strings")
            # make a call to _optionDefaults just to make sure it doesn't throw an exception
            self._optionDefaults(optionName)

        # _exposedFunctions
        exposedFunctions = self._exposedFunctions()
        if exposedFunctions is None or not isinstance(exposedFunctions, list):
            raise TypeError(
                "Improper implementation of _exposedFunctions(), must return a list of methods to be bundled with TrainedLearner")
        for exposed in exposedFunctions:
            # is callable
            if not hasattr(exposed, '__call__'):
                raise TypeError(
                    "Improper implementation of _exposedFunctions, each member of the return must have __call__ attribute")
            # has name attribute
            if not hasattr(exposed, '__name__'):
                raise TypeError(
                    "Improper implementation of _exposedFunctions, each member of the return must have __name__ attribute")
            # takes self as attribute
            (args, varargs, keywords, defaults) = inspectArguments(exposed)
            if args[0] != 'self':
                raise TypeError(
                    "Improper implementation of _exposedFunctions each member's first argument must be 'self', interpreted as a TrainedLearner")

    @property
    def optionNames(self):
        return copy.copy(self._configurableOptionNames())

    @captureOutput
    def trainAndApply(self, learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label',
                      timer=None):

        learner = self.train(learnerName, trainX, trainY, arguments, timer)
        if timer is not None:
            timer.start('apply')
        # call TrainedLearner's apply method (which is already wrapped to perform transformation)
        ret = learner.apply(testX, {}, output, scoreMode, useLog=False)
        if timer is not None:
            timer.stop('apply')

        return ret

    @captureOutput
    def trainAndTest(self, learnerName, trainX, trainY, testX, testY, performanceFunction, arguments={}, output='match',
                     scoreMode='label', timer=None, **kwarguments):
        learner = self.train(learnerName, trainX, trainY, arguments, timer)
        if timer is not None:
            timer.start('test')
        # call TrainedLearner's test method (which is already wrapped to perform transformation)
        ret = learner.test(testX, testY, performanceFunction, {}, output, scoreMode, useLog=False)
        if timer is not None:
            timer.stop('test')

        return ret

    @captureOutput
    def train(self, learnerName, trainX, trainY=None, multiClassStrategy='default', arguments={}, useLog=None, timer=None):
        """

        """
        if multiClassStrategy != 'default':
            #if we need to do multiclassification by ourselves
            trialResult = checkClassificationStrategy(self, learnerName, arguments)

            #1 VS All
            if multiClassStrategy == 'OneVsAll' and trialResult != 'OneVsAll':
                #Remove true labels from from training set, if not already separated
                if isinstance(trainY, (str, numbers.Integral)):
                    trainX = trainX.copy()
                    trainY = trainX.extractFeatures(trainY)

                # Get set of unique class labels
                labelVector = trainY.copy()
                labelVector.transpose()
                labelSet = list(set(labelVector.copyAs(format="python list")[0]))

                if useLog is None:
                    useLog = UML.settings.get("logger", "enabledByDefault")
                    useLog = True if useLog.lower() == 'true' else False
                deepLog = False
                if useLog:
                    deepLog = UML.settings.get('logger', 'enableMultiClassStrategyDeepLogging')
                    deepLog = True if deepLog.lower() == 'true' else False
                    useLog = deepLog

                #if we are logging this run, we need to start the timer
                if useLog:
                    if timer is None:
                        timer = Stopwatch()

                    timer.start('trainOVA')

                # For each class label in the set of labels:  convert the true
                # labels in trainY into boolean labels (1 if the point
                # has 'label', 0 otherwise.)  Train a classifier with the processed
                # labels and get predictions on the test set.
                trainedLearners = []
                for label in labelSet:
                    relabeler.func_defaults = (label,)
                    trainLabels = trainY.calculateForEachPoint(relabeler)
                    trainedLearner = self._train(learnerName, trainX, trainLabels, arguments=arguments, \
                                                       timer=timer)
                    trainedLearner.label = label
                    trainedLearners.append(trainedLearner)

                if useLog:
                    timer.stop('trainOVA')
                return TrainedLearners(trainedLearners, 'OneVsAll', labelSet)

            #1 VS 1
            if multiClassStrategy == 'OneVsOne' and trialResult != 'OneVsOne':
                # we want the data and the labels together in one object or this method
                if isinstance(trainY, UML.data.Base):
                    trainX = trainX.copy()
                    trainX.addFeatures(trainY)
                    trainY = trainX.fts - 1

                # Get set of unique class labels, then generate list of all 2-combinations of
                # class labels
                labelVector = trainX.copyFeatures([trainY])
                labelVector.transpose()
                labelSet = list(set(labelVector.copyAs(format="python list")[0]))
                labelPairs = generateAllPairs(labelSet)

                if useLog is None:
                    useLog = UML.settings.get("logger", "enabledByDefault")
                    useLog = True if useLog.lower() == 'true' else False
                deepLog = False
                if useLog:
                    deepLog = UML.settings.get('logger', 'enableMultiClassStrategyDeepLogging')
                    deepLog = True if deepLog.lower() == 'true' else False
                    useLog = deepLog

                #if we are logging this run, we need to start the timer
                if useLog:
                    if timer is None:
                        timer = Stopwatch()

                    timer.start('trainOVO')

                # For each pair of class labels: remove all points with one of those labels,
                # train a classifier on those points, get predictions based on that model,
                # and put the points back into the data object
                trainedLearners = []
                for pair in labelPairs:
                    #get all points that have one of the labels in pair
                    pairData = trainX.extractPoints(lambda point: (point[trainY] == pair[0]) or (point[trainY] == pair[1]))
                    pairTrueLabels = pairData.extractFeatures(trainY)
                    trainedLearners.append(self._train(learnerName, pairData.copy(), pairTrueLabels.copy(), arguments=arguments, \
                                                       timer=timer))
                    pairData.addFeatures(pairTrueLabels)
                    trainX.addPoints(pairData)
                if useLog:
                    timer.stop('trainOVO')
                return TrainedLearners(trainedLearners, 'OneVsOne', labelSet)

        return self._train(learnerName, trainX, trainY, arguments, timer)

    @captureOutput
    def _train(self, learnerName, trainX, trainY=None, arguments={}, timer=None):
        (trainedBackend, transformedInputs, customDict) = self._trainBackend(learnerName, trainX, trainY, arguments,
                                                                             timer)

        has2dOutput = False
        outputData = trainX if trainY is None else trainY
        if isinstance(outputData, UML.data.Base):
            has2dOutput = outputData.fts > 1
        elif isinstance(outputData, (list, tuple)):
            has2dOutput = len(outputData) > 1

        # encapsulate into TrainedLearner object
        return TrainedLearner(learnerName, arguments, transformedInputs, customDict, trainedBackend, self,
                                   has2dOutput)

    def _confirmValidLearner(self, learnerName):
        allLearners = self.listLearners()
        if not learnerName in allLearners:
            raise ArgumentException("" + learnerName + " is not the name of a learner exposed by this interface")
        learnerCall = self.findCallable(learnerName)
        if learnerCall is None:
            raise ArgumentException("" + learnerName + " was not found in this package")


    def _trainBackend(self, learnerName, trainX, trainY, arguments, timer):
        ### PLANNING ###

        # verify the learner is available
        self._confirmValidLearner(learnerName)

        #validate argument distributions
        groupedArgsWithDefaults = self._validateArgumentDistribution(learnerName, arguments)

        ### INPUT TRANSFORMATION ###
        #recursively work through arguments, doing in-package object instantiation
        instantiatedInputs = self._instantiateArguments(learnerName, groupedArgsWithDefaults)

        # the scratch space dictionary that the package implementor may use to pass information
        # between I/O transformation, the trainer and applier
        customDict = {}

        # separate training data / labels if needed
        if isinstance(trainY, (six.string_types, int, numpy.int64)):
            trainX = trainX.copy()
            trainY = trainX.extractFeatures(toExtract=trainY)

        # execute interface implementor's input transformation.
        transformedInputs = self._inputTransformation(learnerName, trainX, trainY, None, instantiatedInputs, customDict)
        (transTrainX, transTrainY, transTestX, transArguments) = transformedInputs

        ### LEARNER CREATION / TRAINING ###

        # train the instantiated learner
        if timer is not None:
            timer.start('train')
        trainedBackend = self._trainer(learnerName, transTrainX, transTrainY, transArguments, customDict)
        if timer is not None:
            timer.stop('train')

        return (trainedBackend, transformedInputs, customDict)

    def setOption(self, option, value):
        if option not in self.optionNames:
            raise ArgumentException(str(option) + " is not one of the accepted configurable option names")

        UML.settings.set(self.getCanonicalName(), option, value)

    def getOption(self, option):
        if option not in self.optionNames:
            raise ArgumentException(str(option) + " is not one of the accepted configurable option names")

        # empty string is the sentinal value indicating that the configuration
        # file has an option of that name, but the UML user hasn't set a value
        # for it.
        ret = ''
        try:
            ret = UML.settings.get(self.getCanonicalName(), option)
        except:
            # it is possible that the config file doesn't have an option of
            # this name yet. Just pass through and grab the hardcoded default
            pass
        if ret == '':
            ret = self._optionDefaults(option)
        return ret


    def _validateArgumentDistribution(self, learnerName, arguments):
        """
        We check that each call has all the needed arguments, that in total we are
        using each argument only once, and that we use them all.

        return a copy of the arguments that has been arranged for easy instantiation

        """
        baseCallName = learnerName
        possibleParamSets = self.getLearnerParameterNames(learnerName)
        possibleDefaults = self.getLearnerDefaultValues(learnerName)
        bestIndex = self._chooseBestParameterSet(possibleParamSets, possibleDefaults, arguments)

        (neededParams, availableDefaults) = (possibleParamSets[bestIndex], possibleDefaults[bestIndex])
        available = copy.deepcopy(arguments)

        (ret, ignore) = self._validateArgumentDistributionHelper(baseCallName, neededParams, availableDefaults,
                                                                 available, False, arguments)
        return ret

    def _isInstantiable(self, val, hasDefault, defVal):
        if hasDefault and isinstance(defVal, six.string_types):
            return False

        if isinstance(val, six.string_types):
            tmpCallable = self.findCallable(val)

            # if tmpCallable returned something, so long as it isn't a function
            # or a method, it should be instantiable
            isNone = tmpCallable is None
            isMethod = inspect.ismethod(tmpCallable)
            isFunction = inspect.isfunction(tmpCallable)
            if not isNone and not isMethod and not isFunction:
                return True

        return False

    def _validateArgumentDistributionHelper(self, currCallName, currNeededParams, currDefaults, available, sharedPool,
                                            original):
        """
        Recursive function for actually performing _validateArgumentDistribution. Will recurse
        when encountering shorthand for making in package calls, where the desired arguments
        are in another dictionary.

        """
        ret = {}
        # key: key value in ret for accessing appropriate subargs set
        # value: list of key value pair to replace in that set
        delayedAllocations = {None: []}
        # dict where the key matches the param name of the thing that will be
        # instantiated, and the value is a triple, consisting of the the avaiable value,
        # the values consused from available, and the additions to dellayedAllocations
        delayedInstantiations = {}
        #work through this level's needed parameters
        for paramName in currNeededParams:
            # is the param actually there? Is there a default associated with it?
            present = paramName in available
            hasDefault = paramName in currDefaults

            # In each conditional, we have three main tasks: identifying what values will
            # be used, book keeping for that value (removal, delayed allocation / instantiation),
            # and adding values to ret
            if present and hasDefault:
                paramValue = available[paramName]
                paramDefault = currDefaults[paramName]
                addToDelayedIfNeeded = True
                if self._isInstantiable(paramValue, True, paramDefault) or self._isInstantiable(paramDefault, True,
                                                                                                paramDefault):
                    availableBackup = copy.deepcopy(available)
                    allocationsBackup = copy.deepcopy(delayedAllocations)

                if self._isInstantiable(paramDefault, True, paramDefault):
                    # try recursive call using default value
                    try:
                        self._setupValidationRecursiveCall(paramDefault, available, ret, delayedAllocations, original)
                    except:
                    # if fail, try recursive call using actual value and copied available
                        available = availableBackup
                        self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations, original)
                        del available[paramName]
                        addToDelayedIfNeeded = False
                else:
                    ret[paramName] = paramDefault

                if not self._isInstantiable(paramValue, True, paramDefault):
                    # mark down to use the real value if it isn't allocated elsewhere
                    delayedAllocations[None].append((paramName, paramValue))
                else:
                    if addToDelayedIfNeeded:
                        availableChanges = {}
                        for keyBackup in availableBackup:
                            valueBackup = availableBackup[keyBackup]
                            if keyBackup not in available:
                                availableChanges[keyBackup] = valueBackup
                        delayedInstantiations[paramName] = (available[paramName], availableChanges, allocationsBackup)

            elif present and not hasDefault:
                paramValue = available[paramName]
                # is it something that needs to be instantiated and therefore needs params of its own?
                if self._isInstantiable(paramValue, False, None):
                    self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations, original)
                del available[paramName]
                ret[paramName] = paramValue
            elif not present and hasDefault:
                paramValue = currDefaults[paramName]
                # is it something that needs to be instantiated and therefore needs params of its own?
                # TODO is findCallable really the most reliable trigger for this? maybe we should check
                # that you can get params from it too ....
                #if isInstantiable(paramValue, True, paramValue):
                #	self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations, original)
                ret[paramName] = currDefaults[paramName]
            # not present and no default
            else:
                if currCallName in self.listLearners():
                    subParamGroup = self.getLearnerParameterNames(currCallName)
                else:
                    subParamGroup = self._getParameterNames(currCallName)

                msg = "MISSING LEARNERING PARAMETER! "
                msg += "When trying to validate arguments for "
                msg += currCallName + ", "
                msg += "we couldn't find a value for the parameter named "
                msg += "'" + str(paramName) + "'. "
                msg += "The allowed parameters were: "
                msg += prettyListString(currNeededParams, useAnd=True)
                msg += ". These were choosen as the best guess given the inputs"
                msg += " out of the following (numbered) list of possible parameter sets: "
                msg += prettyListString(subParamGroup, numberItems=True, itemStr=prettyListString)

                if len(currDefaults) == 0:
                    msg += ". Out of the allowed parameters, all required values "
                    msg += "specified by the user"
                else:
                    msg += ". Out of the allowed parameters, the following could be omited, "
                    msg += "which would result in the associated default value being used: "
                    msg += prettyDictString(currDefaults, useAnd=True)

                if len(original) == 0:
                    msg += ". However, no arguments were inputed."
                else:
                    msg += ". The full mapping of inputs actually provided was: "
                    msg += prettyDictString(original) + ". "

                raise ArgumentException(msg)

        # if this pool of arguments is not shared, then this is the last subcall,
        # and we can finalize the allocations
        if not sharedPool:
            # work through list of instantiable arguments which were tentatively called using
            # defaults
            for key in delayedInstantiations.keys():
                (value, used, allocations) = delayedInstantiations[key]
                # check to see if it is still in available
                if key in available:
                    # undo the changes made by the default call
                    used.update(available)
                    # make recursive call instead with the actual value
                    #try:
                    self._setupValidationRecursiveCall(value, used, ret, delayedAllocations, original)
                    available = used
                    ret[key] = value
                    del available[key]
                #except:
                # if fail, keep the results of the call with the default
                #	pass

            # work through a list of possible keys for the delayedAllocations dict,
            # if there are allocations associated with that key, perform them.
            for possibleKey in delayedAllocations.keys():
                changesList = delayedAllocations[possibleKey]
                for (k, v) in changesList:
                    if k in available:
                        if possibleKey is None:
                            ret[k] = v
                        else:
                            ret[possibleKey][k] = v
                        del available[k]

            # at this point, everything should have been used, and then removed.
            if len(available) != 0:
                if currCallName in self.listLearners():
                    subParamGroup = self.getLearnerParameterNames(currCallName)
                else:
                    subParamGroup = self._getParameterNames(currCallName)

                msg = "EXTRA LEARNER PARAMETER! "
                msg += "When trying to validate arguments for "
                msg += currCallName + ", "
                msg += "the following list of parameter names were not matched: "
                msg += prettyListString(list(available.keys()), useAnd=True)
                msg += ". The allowed parameters were: "
                msg += prettyListString(currNeededParams, useAnd=True)
                msg += ". These were choosen as the best guess given the inputs"
                msg += " out of the following (numbered) list of possible parameter sets: "
                msg += prettyListString(subParamGroup, numberItems=True, itemStr=prettyListString)
                msg += ". The full mapping of inputs actually provided was: "
                msg += prettyDictString(original) + ". "

                raise ArgumentException(msg)

            delayedAllocations = {}

        return (ret, delayedAllocations)

    def _setupValidationRecursiveCall(self, paramValue, available, callingRet, callingAllocations, original):
        # are the params for this value in a restricted argument pool?
        if paramValue in available:
            subSource = available[paramValue]
            subShared = False
            # We can and should do this here because if there is ever another key with paramVale
            # as the value, then it will be functionally equivalent to save these args for then
            # as it would be to use them here. So, we do the easy thing, and consume them now.
            del available[paramValue]
        # else, they're in the main, shared, pool
        else:
            subSource = available
            subShared = True

        # where we get the wanted parameter set from depends on the kind of thing that we
        # need to instantiate
        if paramValue in self.listLearners():
            subParamGroup = self.getLearnerParameterNames(paramValue)
            subDefaults = self.getLearnerDefaultValues(paramValue)
        else:
            subParamGroup = self._getParameterNames(paramValue)
            subDefaults = self._getDefaultValues(paramValue)

        bestIndex = self._chooseBestParameterSet(subParamGroup, subDefaults, subSource)
        (subParamGroup, subDefaults) = (subParamGroup[bestIndex], subDefaults[bestIndex])

        (ret, allocations) = self._validateArgumentDistributionHelper(paramValue, subParamGroup, subDefaults, subSource,
                                                                      subShared, original)

        # integrate the returned values into the state of the calling frame
        callingRet[paramValue] = ret
        if subShared:
            for pair in allocations[None]:
                if paramValue not in callingAllocations:
                    callingAllocations[paramValue] = []
                callingAllocations[paramValue].append(pair)


    def _instantiateArguments(self, learnerName, arguments):
        """
        Recursively consumes the contents of the arguments parameter, checking for ones
        that need to be instantiated using in-package calls, and performing that
        action if needed. Returns a new dictionary with the same contents as 'arguments',
        except with the replacement of in-package objects for their string names

        """
        baseCallName = learnerName
        baseNeededParams = self._getParameterNames(learnerName)
        toProcess = copy.deepcopy(arguments)
        return self._instantiateArgumentsHelper(toProcess)

    def _instantiateArgumentsHelper(self, toProcess):
        """
        Recursive function for actually performing _instantiateArguments. Will recurse
        when encountering shorthand for making in package calls, where the desired arguments
        are in another dictionary.

        """
        ignoreKeys = []
        ret = {}
        for paramName in toProcess:
            paramValue = toProcess[paramName]
            if isinstance(paramValue, six.string_types):
                ignoreKeys.append(paramValue)
                toCall = self.findCallable(paramValue)
                # if we can find an object for it, and we've prepped the arguments,
                # then we actually instantiate an object
                if toCall is not None and paramValue in toProcess:
                    subInitParams = toProcess[paramValue]
                    if subInitParams is None:
                        ret[paramName] = paramValue
                        continue
                    instantiatedParams = self._instantiateArgumentsHelper(subInitParams)
                    paramValue = toCall(**instantiatedParams)

            ret[paramName] = paramValue

        # for key in ignoreKeys:
        #     if key in ret:
        #         del ret[key]

        return ret

    def _chooseBestParameterSet(self, possibleParamsSets, matchingDefaults, arguments):
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
            msg = "MISSING LEARNERING PARAMETER(S)! "
            msg += "When trying to validate arguments, "
            msg += "we must pick the set of required parameters that best match "
            msg += "the given input. However, from each possible (numbered) parameter"
            msg += " set, the following parameter names were missing "
            msg += prettyListString(missing, numberItems=True, itemStr=prettyListString)
            msg += ". The following lists the required names in each of the possible "
            msg += "(numbered) parameter sets: "
            msg += prettyListString(nonDefaults, numberItems=True, itemStr=prettyListString)
            if len(arguments) == 0:
                msg += ". However, no arguments were inputed."
            else:
                msg += ". The full mapping of inputs actually provided was: "
                msg += prettyDictString(arguments) + ". "

            raise ArgumentException(msg)

            msg = "Missing required params in each possible set: " + str(missing)
            raise ArgumentException(msg)
        return bestIndex

    def _formatScoresToOvA(self, learnerName, learner, testX, applyResults, rawScores, arguments, customDict):
        """
        Helper that takes raw scores in any of the three accepted formats (binary case best score,
        one vs one pairwise tournament by natural label ordering, or one vs all by natural label
        ordering) and returns them in a one vs all accepted format.

        """
        order = self._getScoresOrder(learner)
        numLabels = len(order)
        if numLabels == 2 and rawScores.fts == 1:
            ret = generateBinaryScoresFromHigherSortedLabelScores(rawScores)
            return UML.createData("Matrix", ret)

        if applyResults is None:
            applyResults = self._applier(learner, testX, arguments, customDict)
            applyResults = self._outputTransformation(learnerName, applyResults, arguments, "match", "label",
                                                      customDict)
        if rawScores.fts != 3:
            strategy = ovaNotOvOFormatted(rawScores, applyResults, numLabels)
        else:
            strategy = checkClassificationStrategy(self, learnerName, arguments)
        # we want the scores to be per label, regardless of the original format, so we
        # check the strategy, and modify it if necessary
        if not strategy:
            scores = []
            for i in range(rawScores.pts):
                combinedScores = calculateSingleLabelScoresFromOneVsOneScores(rawScores.pointView(i), numLabels)
                scores.append(combinedScores)
            scores = numpy.array(scores)
            return UML.createData("Matrix", scores)
        else:
            return rawScores


    ##############################################
    ### CACHING FRONTENDS FOR ABSTRACT METHODS ###
    ##############################################

    @captureOutput
    def listLearners(self):
        """
        Return a list of all learners callable through this interface.

        """
        isCustom = isinstance(self, UML.interfaces.CustomLearnerInterface)
        if self._listLearnersCached is None:
            ret = self._listLearnersBackend()
            if not isCustom:
                self._listLearnersCached = ret
        else:
            ret = self._listLearnersCached
        return ret

    @captureOutput
    @cacheWrapper
    def findCallable(self, name):
        """
        Find reference to the callable with the given name
        TAKES string name
        RETURNS reference to in-package function or constructor
        """
        return self._findCallableBackend(name)

    @cacheWrapper
    def _getParameterNames(self, name):
        """
        Find params for instantiation and function calls
        TAKES string name,
        RETURNS list of list of param names to make the chosen call
        """
        return self._getParameterNamesBackend(name)

    @captureOutput
    @cacheWrapper
    def getLearnerParameterNames(self, learnerName):
        """
        Find all parameters involved in a trainAndApply() call to the given learner
        TAKES string name of a learner,
        RETURNS list of list of param names
        """
        return self._getLearnerParameterNamesBackend(learnerName)

    @cacheWrapper
    def _getDefaultValues(self, name):
        """
        Find default values
        TAKES string name,
        RETURNS list of dict of param names to default values
        """
        return self._getDefaultValuesBackend(name)

    @captureOutput
    @cacheWrapper
    def getLearnerDefaultValues(self, learnerName):
        """
        Find all default values for parameters involved in a trainAndApply() call to the given learner
        TAKES string name of a learner,
        RETURNS list of dict of param names to default values
        """
        return self._getLearnerDefaultValuesBackend(learnerName)

    ########################
    ### ABSTRACT METHODS ###
    ########################

    @abc.abstractmethod
    def accessible(self):
        """
        Return true if the package underlying this interface is currently accessible,
        False otherwise.

        """
        pass

    @abc.abstractmethod
    def _listLearnersBackend(self):
        """
        Return a list of all learners callable through this interface.

        """
        pass

    @abc.abstractmethod
    def _findCallableBackend(self, name):
        """
        Find reference to the callable with the given name
        TAKES string name
        RETURNS reference to in-package function or constructor
        """
        pass

    @abc.abstractmethod
    def _getParameterNamesBackend(self, name):
        """
        Find params for instantiation and function calls
        TAKES string name,
        RETURNS list of list of param names to make the chosen call
        """
        pass

    @abc.abstractmethod
    def _getLearnerParameterNamesBackend(self, learnerName):
        """
        Find all parameters involved in a trainAndApply() call to the given learner
        TAKES string name of a learner,
        RETURNS list of list of param names
        """
        pass

    @abc.abstractmethod
    def _getDefaultValuesBackend(self, name):
        """
        Find default values
        TAKES string name,
        RETURNS list of dict of param names to default values
        """
        pass

    @abc.abstractmethod
    def _getLearnerDefaultValuesBackend(self, learnerName):
        """
        Find all default values for parameters involved in a trainAndApply() call to the given learner
        TAKES string name of a learner,
        RETURNS list of dict of param names to default values
        """
        pass

    @abc.abstractmethod
    def learnerType(self, name):
        """
        Returns a string referring to the action the learner takes out of the possibilities:
        classifier, regressor, featureSelection, dimensionalityReduction
        TODO

        """
        pass

    @abc.abstractmethod
    def _getScores(self, learner, testX, arguments, customDict):
        """
        If the learner is a classifier, then return the scores for each
        class on each data point, otherwise raise an exception.

        """
        pass

    @abc.abstractmethod
    def _getScoresOrder(self, learner):
        """
        If the learner is a classifier, then return a list of the the labels corresponding
        to each column of the return from getScores

        """
        pass

    @abc.abstractmethod
    def isAlias(self, name):
        """
        Returns true if the name is an accepted alias for this interface

        """
        pass


    @abc.abstractmethod
    def getCanonicalName(self):
        """
        Returns the string name that will uniquely identify this interface

        """
        pass

    @abc.abstractmethod
    def _inputTransformation(self, learnerName, trainX, trainY, testX, arguments, customDict):
        """
        Method called before any package level function which transforms all
        parameters provided by a UML user.

        trainX, trainY, and testX are filled with the values of the parameters of the same name
        to a call to trainAndApply() or train() and are sometimes empty when being called
        by other functions. For example, a call to apply() will have trainX and trainY be None.
        The arguments parameter is a dictionary mapping names to values of all other
        parameters associated with the learner, each of which may need to be processed.

        The return value of this function must be a tuple mirroring the structure of
        the inputs. Specifically, four values are required: the transformed versions of
        trainX, trainY, testX, and arguments in that specific order.

        """
        pass

    @abc.abstractmethod
    def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputType, outputFormat, customDict):
        """
        Method called before any package level function which transforms the returned
        value into a format appropriate for a UML user.

        """
        pass

    @abc.abstractmethod
    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        """
        build a learner and perform training with the given data
        TAKES name of learner, transformed arguments
        RETURNS an in package object to be wrapped by a TrainedLearner object
        """
        pass

    @abc.abstractmethod
    def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
        """
        Given an already trained online learner, extend it's training with the given data
        TAKES trained learner, transformed arguments,
        RETURNS the learner after this batch of training
        """
        pass


    @abc.abstractmethod
    def _applier(self, learner, testX, arguments, customDict):
        """
        use the given learner to do testing/prediction on the given test set
        TAKES a TrainedLearner object that can be tested on
        RETURNS UML friendly results
        """
        pass


    @abc.abstractmethod
    def _getAttributes(self, learnerBackend):
        """
        Returns whatever attributes might be available for the given learner. For
        example, in the case of linear regression, TODO

        """
        pass

    @abc.abstractmethod
    def _optionDefaults(self, option):
        """
        Define package default values that will be used for as long as a default
        value hasn't been registered in the UML configuration file. For example,
        these values will always be used the first time an interface is instantiated.

        """
        pass


    @abc.abstractmethod
    def _configurableOptionNames(self):
        """
        Returns a list of strings, where each string is the name of a configurable
        option of this interface whose value will be stored in UML's configuration
        file.

        """
        pass

    @abc.abstractmethod
    def _exposedFunctions(self):
        """
        Returns a list of references to functions which are to be wrapped
        in I/O transformation, and exposed as attributes of all TrainedLearner
        objects returned by this interface's train() function. If None, or an
        empty list is returned, no functions will be exposed. Each function
        in this list should be a python function, the inspect module will be
        used to retrieve argument names, and the value of the function's
        __name__ attribute will be its name in TrainedLearner.

        """
        pass

##################
# TrainedLearner #
##################

class TrainedLearner(object):

    def __init__(self, learnerName, arguments, transformedInputs, customDict, backend, interfaceObject,
                 has2dOutput):
        """
        Initialize the object wrapping the trained learner stored in backend, and setting up
        the object methods that may be used to modify or query the backend trained learner.

        learnerName: the name of the learner used in the backend
        arguments: reference to the original arguments parameter to the trainAndApply() function
        transformedArguments: a tuple containing the return value of _inputTransformation() that was called when training the learner in the backend
        customDict: reference to the customizable dictionary that is passed to I/O transformation, training and applying a learner
        backend: the return value from _trainer(), a reference to a some object that is to be used by the package implementor during application
        interfaceObject: a reference to the subclass of UniversalInterface from which this TrainedLearner is being instantiated.

        """
        self.learnerName = learnerName
        self.arguments = arguments
        self.transformedTrainX = transformedInputs[0]
        self.transformedTrainY = transformedInputs[1]
        self.transformedTestX = transformedInputs[2]
        self.transformedArguments = transformedInputs[3]
        self.customDict = customDict
        self.backend = backend
        self.interface = interfaceObject
        self.has2dOutput = has2dOutput

        exposedFunctions = self.interface._exposedFunctions()
        for exposed in exposedFunctions:
            methodName = getattr(exposed, '__name__')
            (args, varargs, keywords, defaults) = inspect.getargspec(exposed)
            if 'trainedLearner' in args:
                wrapped = functools.partial(exposed, trainedLearner=self)
                wrapped.__doc__ = 'Wrapped version of the ' + methodName + ' function where the "trainedLearner" parameter has been fixed as this object, and the "self" parameter has been fixed to be ' + str(
                    interfaceObject)
            else:
                wrapped = functools.partial(exposed)
                wrapped.__doc__ = 'Wrapped version of the ' + methodName + ' function where the "self" parameter has been fixed to be ' + str(
                    interfaceObject)
            setattr(self, methodName, wrapped)

    @captureOutput
    def test(
            self, testX, testY, performanceFunction, arguments={},
            output='match', scoreMode='label', useLog=None, **kwarguments):
        """
        Returns the evaluation of predictions of testX using the argument
        performanceFunction to do the evalutation. Equivalent to having called
        this interface's trainAndqTest method, as long as the data and parameter
        setup for training was the same.

        """
        if useLog is None:
            useLog = UML.settings.get("logger", "enabledByDefault")
            useLog = True if useLog.lower() == 'true' else False

        timer = None
        if useLog:
            timer = Stopwatch()
            timer.start("test")

        #UML.helpers._2dOutputFlagCheck(self.has2dOutput, None, scoreMode, multiClassStrategy)
        UML.helpers._2dOutputFlagCheck(self.has2dOutput, None, scoreMode, None)

        # need to do this here so we
        mergedArguments = self._mergeWithTrainArguments(arguments, kwarguments)

        pred = self.apply(testX, mergedArguments, output, scoreMode, useLog=False)
        performance = UML.helpers.computeMetrics(testY, None, pred, performanceFunction)

        if useLog:
            timer.stop('test')
            fullName = self.interface.getCanonicalName() + self.learnerName
            # Signature:
            # (self, trainData, trainLabels, testData, testLabels, function,
            # metrics, predictions, performance, timer, extraInfo=None,
            # numFolds=None)
            UML.logger.active.logRun(
                trainData=None, trainLabels=None, testData=testX,
                testLabels=testY, function=fullName,
                metrics=[performanceFunction], predictions=pred,
                performance=[performance], timer=timer,
                extraInfo=mergedArguments, numFolds=None)

        return performance

    def _mergeWithTrainArguments(self, newArguments1, newArguments2):
        """
        When calling apply, merges our fixed arguments with our provided arguments,
        giving the new arguments precedence if needed.

        """
        ret = _mergeArguments(self.transformedArguments, newArguments1)
        ret = _mergeArguments(ret, newArguments2)
        return ret

    @captureOutput
    def apply(
            self, testX, arguments={}, output='match', scoreMode='label',
            useLog=None, **kwarguments):
        """
        Returns the application of this learner to the given test data (i.e. performing
        prediction, transformation, etc. as appropriate to the learner). Equivalent to
        having called trainAndApply with the same same setup as this learner was trained
        on.

        """
        UML.helpers._2dOutputFlagCheck(self.has2dOutput, None, scoreMode, None)

        if useLog is None:
            useLog = UML.settings.get("logger", "enabledByDefault")
            useLog = True if useLog.lower() == 'true' else False

        timer = None
        if useLog:
            timer = Stopwatch()
            timer.start("apply")

        #			self.interface._validateOutputFlag(output)
        #			self.interface._validateScoreModeFlag(scoreMode)
        mergedArguments = self._mergeWithTrainArguments(arguments, kwarguments)

        # input transformation
        (trainX, trainY, transTestX, usedArguments) = self.interface._inputTransformation(self.learnerName, None,
                                                                                          None, testX,
                                                                                          mergedArguments,
                                                                                          self.customDict)

        # depending on the mode, we need different information.
        labels = None
        if scoreMode != 'label':
            scores = self.getScores(testX, usedArguments)
        if scoreMode != 'allScores':
            labels = self.interface._applier(self.backend, transTestX, usedArguments, self.customDict)
            labels = self.interface._outputTransformation(self.learnerName, labels, usedArguments, output, "label",
                                                          self.customDict)

        if scoreMode == 'label':
            ret = labels
        elif scoreMode == 'allScores':
            ret = scores
        else:
            scoreOrder = self.interface._getScoresOrder(self.backend)
            scoreOrder = list(scoreOrder)
            # find scores matching predicted labels
            def grabValue(row):
                pointIndex = scores.getPointIndex(row.getPointName(0))
                rowIndex = scoreOrder.index(labels[pointIndex, 0])
                return row[rowIndex]

            scoreVector = scores.calculateForEachPoint(grabValue)
            labels.addFeatures(scoreVector)

            ret = labels

        if useLog:
            timer.stop('apply')
            fullName = self.interface.getCanonicalName() + self.learnerName
            # Signature:
            # (self, trainData, trainLabels, testData, testLabels, function,
            # metrics, predictions, performance, timer, extraInfo=None,
            # numFolds=None)
            UML.logger.active.logRun(
                trainData=None, trainLabels=None, testData=testX,
                testLabels=None, function=fullName, metrics=None,
                predictions=ret, performance=None, timer=timer,
                extraInfo=mergedArguments, numFolds=None)

        return ret

    def save(self, outputPath):
        """
        Save model to a file.

        outputPath: the location (including file name and extension) where
            we want to write the output file.

        If filename extension .umlm is not included in file name it would
        be added to the output file.

        Uses dill library to serialize it.
        """
        if not cloudpickle:
            msg = "To save UML models, cloudpickle must be installed"
            raise PackageException(msg)
        extension = '.umlm'
        if not outputPath.endswith(extension):
            outputPath = outputPath + extension

        with open(outputPath, 'wb') as file:
            try:
                cloudpickle.dump(self, file)
            except Exception as e:
                raise(e)
        # print('session_' + outputFilename)
        # print(globals())
        # dill.dump_session('session_' + outputFilename)

    @captureOutput
    def retrain(self, trainX, trainY=None):
        has2dOutput = False
        outputData = trainX if trainY is None else trainY
        if isinstance(outputData, UML.data.Base):
            has2dOutput = outputData.fts > 1
        elif isinstance(outputData, (list, tuple)):
            has2dOutput = len(outputData) > 1

        #			(trainX, trainY, testX, arguments) = self.interface._inputTransformation(self.learnerName,trainX, trainY, None, self.arguments, self.customDict)
        (newBackend, transformedInputs, customDict) = self.interface._trainBackend(self.learnerName, trainX, trainY,
                                                                                   self.arguments, None)
        self.backend = newBackend
        self.transformedInputs = transformedInputs
        self.customDict = customDict
        self.has2dOutput = has2dOutput

    @captureOutput
    def incrementalTrain(self, trainX, trainY=None):
        (trainX, trainY, testX, arguments) = self.interface._inputTransformation(self.learnerName, trainX, trainY,
                                                                                 None, self.arguments,
                                                                                 self.customDict)
        self.backend = self.interface._incrementalTrainer(self.backend, trainX, trainY, arguments, self.customDict)

    @captureOutput
    def getAttributes(self):
        """ Returns the attributes of the trained learner (and sub objects).
        The returned value will be a dict, mapping names of attribtues to
        values of attributes. In the case of collisions (especially when getting
        attributes from nested objects) the attribute names may be prefaced with
        the name of the object from which they originate.

        The input learner params provided by the user for initilization and
        training will always be included in the output. If there is a collision
        between an input and an attribute of the same name discovered by the
        learner, and their values do not match, then the discovered attribute's
        name will be prefaced with the learner name. Similarly for nested objects
        such as sub-learners and kenerls.

        """
        discovered = self.interface._getAttributes(self.backend)
        inputs = self.arguments

        for key in inputs.keys():
            value = inputs[key]
            if key in list(discovered.keys()):
                if value != discovered[key]:
                    newKey = self.learnerName + '.' + key
                    discovered[newKey] = discovered[key]
                discovered[key] = value

        return discovered

    @captureOutput
    def getScores(self, testX, arguments={}, **kwarguments):
        """
        Returns the scores for all labels for each data point. If this TrainedLearner
        is named foo, this operation is equivalent to calling foo.apply with
        'scoreMode="allScores"'

        """
        usedArguments = self._mergeWithTrainArguments(arguments, kwarguments)
        (trainX, trainY, testX, usedArguments) = self.interface._inputTransformation(self.learnerName, None, None,
                                                                                     testX, usedArguments,
                                                                                     self.customDict)

        rawScores = self.interface._getScores(self.backend, testX, usedArguments, self.customDict)
        umlTypeRawScores = self.interface._outputTransformation(self.learnerName, rawScores, usedArguments,
                                                                "Matrix", "allScores", self.customDict)
        formatedRawOrder = self.interface._formatScoresToOvA(self.learnerName, self.backend, testX, None,
                                                             umlTypeRawScores, usedArguments, self.customDict)
        internalOrder = self.interface._getScoresOrder(self.backend)
        naturalOrder = sorted(internalOrder)
        if numpy.array_equal(naturalOrder, internalOrder):
            return formatedRawOrder
        desiredDict = {}
        for i in range(len(naturalOrder)):
            label = naturalOrder[i]
            desiredDict[label] = i

        def sortScorer(feature):
            index = formatedRawOrder.getFeatureIndex(feature.getFeatureName(0))
            label = internalOrder[index]
            return desiredDict[label]

        formatedRawOrder.sortFeatures(sortHelper=sortScorer)
        return formatedRawOrder

class TrainedLearners(TrainedLearner):
    """

    """
    def __init__(self, trainedLearners, method, labelSet):
        """

        """
        self.trainedLearnersList = trainedLearners
        self.method = method
        self.labelSet = labelSet
        self.has2dOutput = trainedLearners[0].has2dOutput
        self.transformedArguments = trainedLearners[0].transformedArguments
        self.interface = trainedLearners[0].interface
        self.learnerName = trainedLearners[0].learnerName

    @captureOutput
    def apply(self, testX, arguments={}, output='match', scoreMode='label',
            useLog=None, **kwarguments):
        """

        """
        rawPredictions = None
        # import pdb; pdb.set_trace()
        #1 VS All
        if self.method == 'OneVsAll':
            for trainedLearner in self.trainedLearnersList:
                oneLabelResults = trainedLearner.apply(testX, arguments, output, 'label', useLog)
                label = trainedLearner.label
                #put all results into one Base container, of the same type as trainX
                if rawPredictions is None:
                    rawPredictions = oneLabelResults
                    #as it's added to results object, rename each column with its corresponding class label
                    rawPredictions.setFeatureName(0, str(label))
                else:
                    #as it's added to results object, rename each column with its corresponding class label
                    oneLabelResults.setFeatureName(0, str(label))
                    rawPredictions.addFeatures(oneLabelResults)

            if scoreMode.lower() == 'label'.lower():
                winningPredictionIndices = rawPredictions.calculateForEachPoint(UML.helpers.extractWinningPredictionIndex).copyAs(
                    format="python list")
                winningLabels = []
                for [winningIndex] in winningPredictionIndices:
                    winningLabels.append([self.labelSet[int(winningIndex)]])
                return UML.createData(rawPredictions.getTypeString(), winningLabels, featureNames=['winningLabel'])

            elif scoreMode.lower() == 'bestScore'.lower():
                #construct a list of lists, with each row in the list containing the predicted
                #label and score of that label for the corresponding row in rawPredictions
                predictionMatrix = rawPredictions.copyAs(format="python list")
                indexToLabel = rawPredictions.getFeatureNames()
                tempResultsList = []
                for row in predictionMatrix:
                    bestLabelAndScore = UML.helpers.extractWinningPredictionIndexAndScore(row, indexToLabel)
                    tempResultsList.append([bestLabelAndScore[0], bestLabelAndScore[1]])
                #wrap the results data in a List container
                featureNames = ['PredictedClassLabel', 'LabelScore']
                resultsContainer = UML.createData("List", tempResultsList, featureNames=featureNames)
                return resultsContainer

            elif scoreMode.lower() == 'allScores'.lower():
                #create list of Feature Names/Column Headers for final return object
                columnHeaders = sorted([str(i) for i in self.labelSet])
                #create map between label and index in list, so we know where to put each value
                labelIndexDict = {v: k for k, v in zip(list(range(len(columnHeaders))), columnHeaders)}
                featureNamesItoN = rawPredictions.getFeatureNames()
                predictionMatrix = rawPredictions.copyAs(format="python list")
                resultsContainer = []
                for row in predictionMatrix:
                    finalRow = [0] * len(columnHeaders)
                    scores = UML.helpers.extractConfidenceScores(row, featureNamesItoN)
                    for label, score in scores.items():
                        #get numerical index of label in return object
                        finalIndex = labelIndexDict[label]
                        #put score into proper place in its row
                        finalRow[finalIndex] = score
                    resultsContainer.append(finalRow)
                #wrap data in Base container
                return UML.createData(rawPredictions.getTypeString(), resultsContainer, featureNames=columnHeaders)

        #1 VS 1
        elif self.method == 'OneVsOne':
            predictionFeatureID = 0
            for trainedLearner in self.trainedLearnersList:
                #train classifier on that data; apply it to the test set
                partialResults = trainedLearner.apply(testX, arguments, output, 'label', useLog)
                #put predictions into table of predictions
                if rawPredictions is None:
                    rawPredictions = partialResults.copyAs(format="List")
                else:
                    partialResults.setFeatureName(0, 'predictions-' + str(predictionFeatureID))
                    rawPredictions.addFeatures(partialResults.copyAs(format="List"))
                predictionFeatureID += 1
            #set up the return data based on which format has been requested
            if scoreMode.lower() == 'label'.lower():
                ret = rawPredictions.calculateForEachPoint(UML.helpers.extractWinningPredictionLabel)
                ret.setFeatureName(0, "winningLabel")
                return ret
            elif scoreMode.lower() == 'bestScore'.lower():
                #construct a list of lists, with each row in the list containing the predicted
                #label and score of that label for the corresponding row in rawPredictions
                predictionMatrix = rawPredictions.copyAs(format="python list")
                tempResultsList = []
                for row in predictionMatrix:
                    scores = countWins(row)
                    sortedScores = sorted(scores, key=scores.get, reverse=True)
                    bestLabel = sortedScores[0]
                    tempResultsList.append([bestLabel, scores[bestLabel]])

                #wrap the results data in a List container
                featureNames = ['PredictedClassLabel', 'LabelScore']
                resultsContainer = UML.createData("List", tempResultsList, featureNames=featureNames)
                return resultsContainer
            elif scoreMode.lower() == 'allScores'.lower():
                columnHeaders = sorted([str(i) for i in self.labelSet])
                labelIndexDict = {str(v): k for k, v in zip(list(range(len(columnHeaders))), columnHeaders)}
                predictionMatrix = rawPredictions.copyAs(format="python list")
                resultsContainer = []
                for row in predictionMatrix:
                    finalRow = [0] * len(columnHeaders)
                    scores = countWins(row)
                    for label, score in scores.items():
                        finalIndex = labelIndexDict[str(label)]
                        finalRow[finalIndex] = score
                    resultsContainer.append(finalRow)

                return UML.createData(rawPredictions.getTypeString(), resultsContainer, featureNames=columnHeaders)


        else:
            raise(ArgumentException('Wrong multiclassification method.'))


###########
# Helpers #
###########


def relabeler(point, label=None):
    if point[0] != label:
        return 0
    else:
        return 1
