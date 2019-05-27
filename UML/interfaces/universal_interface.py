"""
The top level objects and methods which allow UML to interface with
various python packages or custom learners. Also contains the objects
which store trained learner models and provide functionality for
applying and testing learners.
"""
from __future__ import absolute_import
import inspect
import copy
import abc
import functools
import sys
import numbers

import numpy
import six
from six.moves import range

import UML
from UML.exceptions import InvalidArgumentValue, ImproperObjectAction
from UML.exceptions import PackageException
from UML.docHelpers import inheritDocstringsFactory
from UML.exceptions import prettyListString
from UML.exceptions import prettyDictString
from UML.interfaces.interface_helpers import (
    generateBinaryScoresFromHigherSortedLabelScores,
    calculateSingleLabelScoresFromOneVsOneScores,
    ovaNotOvOFormatted, checkClassificationStrategy, cacheWrapper)
from UML.logger import handleLogging, startTimer, stopTimer
from UML.helpers import _mergeArguments
from UML.helpers import generateAllPairs, countWins, inspectArguments
from UML.helpers import extractWinningPredictionIndex
from UML.helpers import extractWinningPredictionLabel
from UML.helpers import extractWinningPredictionIndexAndScore

cloudpickle = UML.importModule('cloudpickle')


def captureOutput(toWrap):
    """
    Decorator which will safely redirect standard error within the
    wrapped function to the temp file at UML.capturedErr.
    """
    @functools.wraps(toWrap)
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
    Metaclass defining methods and abstract methods for specific
    package or custom interfaces.
    """
    _listLearnersCached = None

    def __init__(self):
        ### Validate all the information from abstract functions ###
        # enforce a check that the underlying package is accessible at
        # instantiation, aborting the construction of the interface for this
        # session of UML if it is not.
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


    @captureOutput
    def trainAndApply(self, learnerName, trainX, trainY=None, testX=None,
                      arguments=None, output=None, scoreMode='label'):
        """
        Train a model and apply it to the test data.

        The learner will be trained using the training data, then
        prediction, transformation, etc. as appropriate to the learner
        will be applied to the test data and returned.

        Parameters
        ----------
        learnerName : str
            Name of the learner to be called, in the form
            'package.learner'
        trainX: UML Base object
            Data to be used for training.
        trainY: identifier, UML Base object
            A name or index of the feature in ``trainX`` containing the
            labels or another UML Base object containing the labels that
            correspond to ``trainX``.
        testX : UML Base object
            data set on which the trained learner will be applied (i.e.
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
            The kind of UML Base object that the output of this function
            should be in. Any of the normal string inputs to the
            createData ``returnType`` parameter are accepted here.
            Alternatively, the value 'match' will indicate to use the
            type of the ``trainX`` parameter.
        scoreMode : str
            In the case of a classifying learner, this specifies the
            type of output wanted: 'label' if we class labels are
            desired, 'bestScore' if both the class label and the score
            associated with that class are desired, or 'allScores' if a
            matrix containing the scores for every class label are
            desired.

        Returns
        -------
        results
            The resulting output of applying learner.
        """
        learner = self.train(learnerName, trainX, trainY, arguments=arguments)
        # call TrainedLearner's apply method
        # (which is already wrapped to perform transformation)
        ret = learner.apply(testX, {}, output, scoreMode, useLog=False)

        return ret


    @captureOutput
    def trainAndTest(self, learnerName, trainX, trainY, testX, testY,
                     performanceFunction, arguments=None, output='match',
                     scoreMode='label', **kwarguments):
        """
        Train a model and get the results of its performance.

        Uses cross validation to generate a performance score for the
        algorithm, given the particular argument permutation. The
        argument permutation that performed best cross validating over
        the training data is then used as the lone argument for training
        on the whole training data set. Finally, the learned model
        generates predictions for the testing set, an the performance
        of those predictions is calculated and returned. If no
        additional arguments are supplied via arguments, then
        the result is the performance of the algorithm with default
        arguments on the testing data.

        Parameters
        ----------
        learnerName : str
            Name of the learner to be called, in the form
            'package.learner'
        trainX: UML Base object
            Data to be used for training.
        trainY : identifier, UML Base object
            * identifier - The name or index of the feature in
              ``trainX`` containing the labels.
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
            since if there is no parameter selection to be done, it is
            not used.
        arguments : dict
            Mapping argument names (strings) to their values, to be used
            during training and application. eg. {'dimensions':5, 'k':5}
            To make use of multiple permutations, specify different
            values for a parameter as a tuple. eg. {'k': (1,3,5)} will
            generate an error score for  the learner when the learner
            was passed all three values of ``k``, separately. These will
            be merged with kwarguments for the learner.
        output : str
            The kind of UML Base object that the output of this function
            should be in. Any of the normal string inputs to the
            createData ``returnType`` parameter are accepted here.
            Alternatively, the value 'match' will indicate to use the
            type of the ``trainX`` parameter.
        scoreMode : str
            In the case of a classifying learner, this specifies the
            type of output wanted: 'label' if we class labels are
            desired, 'bestScore' if both the class label and the score
            associated with that class are desired, or 'allScores' if a
            matrix containing the scores for every class label are
            desired.
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
        """
        learner = self.train(learnerName, trainX, trainY, arguments=arguments)
        # call TrainedLearner's test method
        # (which is already wrapped to perform transformation)
        ret = learner.test(testX, testY, performanceFunction, {}, output,
                           scoreMode, useLog=False)

        return ret


    @captureOutput
    def train(self, learnerName, trainX, trainY=None, arguments=None,
              multiClassStrategy='default', crossValidationResults=None):
        """
        Fit the learner model using training data.

        learnerName : str
            Name of the learner to be called, in the form
            'package.learner'
        trainX: UML Base object
            Data to be used for training.
        trainY: identifier, UML Base object
            A name or index of the feature in ``trainX`` containing the
            labels or another UML Base object containing the labels that
            correspond to ``trainX``.
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
            #if we need to do multiclassification by ourselves
            trialResult = checkClassificationStrategy(self, learnerName,
                                                      arguments)
            #1 VS All
            if multiClassStrategy == 'OneVsAll' and trialResult != 'OneVsAll':
                #Remove true labels from from training set, if not separated
                if isinstance(trainY, (str, numbers.Integral)):
                    trainX = trainX.copy()
                    trainY = trainX.features.extract(trainY, useLog=False)

                # Get set of unique class labels
                labelVector = trainY.copy()
                labelVector.transpose(useLog=False)
                labelVectorToList = labelVector.copy(to="python list")[0]
                labelSet = list(set(labelVectorToList))

                # For each class label in the set of labels:  convert the true
                # labels in trainY into boolean labels (1 if the point
                # has 'label', 0 otherwise.)  Train a classifier with the
                # processed labels and get predictions on the test set.
                trainedLearners = []
                for label in labelSet:
                    relabeler.__defaults__ = (label,)
                    trainLabels = trainY.points.calculate(relabeler,
                                                          useLog=False)
                    trainedLearner = self._train(
                        learnerName, trainX, trainLabels, arguments=arguments)
                    trainedLearner.label = label
                    trainedLearners.append(trainedLearner)

                return TrainedLearners(trainedLearners, 'OneVsAll', labelSet)

            #1 VS 1
            if multiClassStrategy == 'OneVsOne' and trialResult != 'OneVsOne':
                # want data and labels together in one object for this method
                if isinstance(trainY, UML.data.Base):
                    trainX = trainX.copy()
                    trainX.features.add(trainY, useLog=False)
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
                    pairData.features.add(pairTrueLabels, useLog=False)
                    trainX.points.add(pairData, useLog=False)

                return TrainedLearners(trainedLearners, 'OneVsOne', labelSet)

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
        if isinstance(outputData, UML.data.Base):
            has2dOutput = len(outputData.features) > 1
        elif isinstance(outputData, (list, tuple)):
            has2dOutput = len(outputData) > 1

        # encapsulate into TrainedLearner object
        return TrainedLearner(learnerName, arguments, transformedInputs,
                              customDict, trainedBackend, self, has2dOutput,
                              crossValidationResults)


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

        # validate argument distributions
        groupedArgsWithDefaults = self._validateArgumentDistribution(
            learnerName, arguments)

        ### INPUT TRANSFORMATION ###
        # recursively work through arguments,
        # doing in-package object instantiation
        instantiatedInputs = self._instantiateArguments(
            learnerName, groupedArgsWithDefaults)

        # the scratch space dictionary that the package implementor may use to
        # pass information between I/O transformation, the trainer and applier
        customDict = {}

        # separate training data / labels if needed
        if isinstance(trainY, (six.string_types, int, numpy.int64)):
            trainX = trainX.copy()
            trainY = trainX.features.extract(toExtract=trainY, useLog=False)

        # execute interface implementor's input transformation.
        transformedInputs = self._inputTransformation(
            learnerName, trainX, trainY, None, instantiatedInputs, customDict)
        transTrainX, transTrainY, _, transArguments = transformedInputs

        ### LEARNER CREATION / TRAINING ###

        # train the instantiated learner
        trainedBackend = self._trainer(learnerName, transTrainX, transTrainY,
                                       transArguments, customDict)

        return (trainedBackend, transformedInputs, customDict)


    def setOption(self, option, value):
        """
        TODO
        """
        if option not in self.optionNames:
            msg = str(option)
            msg += " is not one of the accepted configurable option names"
            raise InvalidArgumentValue(msg)

        UML.settings.set(self.getCanonicalName(), option, value)


    def getOption(self, option):
        """
        TODO
        """
        if option not in self.optionNames:
            msg = str(option)
            msg += " is not one of the accepted configurable option names"
            raise InvalidArgumentValue(msg)

        # empty string is the sentinal value indicating that the configuration
        # file has an option of that name, but the UML user hasn't set a value
        # for it.
        ret = ''
        try:
            ret = UML.settings.get(self.getCanonicalName(), option)
        except Exception:
            # it is possible that the config file doesn't have an option of
            # this name yet. Just pass through and grab the hardcoded default
            pass
        if ret == '':
            ret = self._optionDefaults(option)
        return ret


    def _validateArgumentDistribution(self, learnerName, arguments):
        """
        We check that each call has all the needed arguments, that in
        total we are using each argument only once, and that we use them
        all.

        Returns
        -------
        A copy of the arguments
            These have been arranged for easy instantiation.
        """
        if arguments is None:
            arguments = []
        baseCallName = learnerName
        possibleParamSets = self.getLearnerParameterNames(learnerName)
        possibleDefaults = self.getLearnerDefaultValues(learnerName)
        bestIndex = self._chooseBestParameterSet(possibleParamSets,
                                                 possibleDefaults, arguments)

        neededParams = possibleParamSets[bestIndex]
        availableDefaults = possibleDefaults[bestIndex]
        available = copy.deepcopy(arguments)

        ret, _ = self._validateArgumentDistributionHelper(
            baseCallName, neededParams, availableDefaults, available, False,
            arguments)
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

    def _validateArgumentDistributionHelper(
            self, currCallName, currNeededParams, currDefaults, available,
            sharedPool, original):
        """
        Recursive function for actually performing
        _validateArgumentDistribution. Will recurse when encountering
        shorthand for making in package calls, where the desired
        arguments are in another dictionary.
        """
        ret = {}
        # key: key value in ret for accessing appropriate subargs set
        # value: list of key value pair to replace in that set
        delayedAllocations = {None: []}
        # dict where the key matches the param name of the thing that will be
        # instantiated, and the value is a triple, consisting of the the
        # avaiable value, the values consused from available, and the additions
        # to delayedAllocations
        delayedInstantiations = {}
        #work through this level's needed parameters
        for paramName in currNeededParams:
            # is param actually there? Is there a default associated with it?
            present = paramName in available
            hasDefault = paramName in currDefaults

            # In each conditional, we have three main tasks: identifying what
            # values will be used, book keeping for that value (removal,
            # delayed allocation / instantiation), and adding values to ret
            if present and hasDefault:
                paramValue = available[paramName]
                paramDefault = currDefaults[paramName]
                addToDelayedIfNeeded = True
                if (self._isInstantiable(paramValue, True, paramDefault)
                        or self._isInstantiable(
                            paramDefault, True, paramDefault)):
                    availableBackup = copy.deepcopy(available)
                    allocationsBackup = copy.deepcopy(delayedAllocations)

                if self._isInstantiable(paramDefault, True, paramDefault):
                    # try recursive call using default value
                    try:
                        self._setupValidationRecursiveCall(
                            paramDefault, available, ret, delayedAllocations,
                            original)
                    except Exception:
                    # try recursive call with actual value and copied available
                        available = availableBackup
                        self._setupValidationRecursiveCall(
                            paramValue, available, ret, delayedAllocations,
                            original)
                        del available[paramName]
                        addToDelayedIfNeeded = False
                else:
                    ret[paramName] = paramDefault

                if not self._isInstantiable(paramValue, True, paramDefault):
                    # mark to use real value if it isn't allocated elsewhere
                    delayedAllocations[None].append((paramName, paramValue))
                else:
                    if addToDelayedIfNeeded:
                        availableChanges = {}
                        for keyBackup in availableBackup:
                            valueBackup = availableBackup[keyBackup]
                            if keyBackup not in available:
                                availableChanges[keyBackup] = valueBackup
                        delayedInstantiations[paramName] = (
                            available[paramName], availableChanges,
                            allocationsBackup)

            elif present and not hasDefault:
                paramValue = available[paramName]
                # is it something that needs to be instantiated and therefore
                # needs params of its own?
                if self._isInstantiable(paramValue, False, None):
                    self._setupValidationRecursiveCall(paramValue, available,
                                                       ret, delayedAllocations,
                                                       original)
                del available[paramName]
                ret[paramName] = paramValue
            elif not present and hasDefault:
                paramValue = currDefaults[paramName]
                # is it something that needs to be instantiated and therefore
                # needs params of its own?
                # TODO is findCallable really most reliable trigger for this?
                # maybe we should check that you can get params from it too
                # if isInstantiable(paramValue, True, paramValue):
                #    self._setupValidationRecursiveCall(
                #        paramValue, available, ret, delayedAllocations,
                #        original)
                ret[paramName] = currDefaults[paramName]
            # not present and no default
            else:
                if currCallName in self.listLearners():
                    subParamGroup = self.getLearnerParameterNames(currCallName)
                else:
                    subParamGroup = self._getParameterNames(currCallName)

                msg = "MISSING LEARNING PARAMETER! "
                msg += "When trying to validate arguments for " + currCallName
                msg += ", we couldn't find a value for the parameter named "
                msg += "'" + str(paramName) + "'. "
                msg += "The allowed parameters were: "
                msg += prettyListString(currNeededParams, useAnd=True)
                msg += ". These were choosen as the best guess given the "
                msg += "inputs out of the following (numbered) list of "
                msg += "possible parameter sets: "
                msg += prettyListString(subParamGroup, numberItems=True,
                                        itemStr=prettyListString)

                if len(currDefaults) == 0:
                    msg += ". Out of the allowed parameters, all required "
                    msg += "values specified by the user"
                else:
                    msg += ". Out of the allowed parameters, the following "
                    msg += "could be omited, which would result in the "
                    msg += "associated default value being used: "
                    msg += prettyDictString(currDefaults, useAnd=True)

                if len(original) == 0:
                    msg += ". However, no arguments were inputed."
                else:
                    msg += ". The full mapping of inputs actually provided "
                    msg += "was: " + prettyDictString(original) + ". "

                raise InvalidArgumentValue(msg)

        # if this pool of arguments is not shared, then this is the last
        # subcall, and we can finalize the allocations
        if not sharedPool:
            # work through list of instantiable arguments which were
            # tentatively called using defaults
            for key in delayedInstantiations.keys():
                (value, used, _) = delayedInstantiations[key]
                # check to see if it is still in available
                if key in available:
                    # undo the changes made by the default call
                    used.update(available)
                    # make recursive call instead with the actual value
                    #try:
                    self._setupValidationRecursiveCall(value, used, ret,
                                                       delayedAllocations,
                                                       original)
                    available = used
                    ret[key] = value
                    del available[key]
                #except:
                # if fail, keep the results of the call with the default
                #	pass

            # work through a list of possible keys for delayedAllocations dict,
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

            # at this point, everything should have been used and then removed.
            if len(available) != 0:
                if currCallName in self.listLearners():
                    subParamGroup = self.getLearnerParameterNames(currCallName)
                else:
                    subParamGroup = self._getParameterNames(currCallName)

                msg = "EXTRA LEARNER PARAMETER! "
                msg += "When trying to validate arguments for "
                msg += currCallName + ", the following list of parameter "
                msg += "names were not matched: "
                msg += prettyListString(list(available.keys()), useAnd=True)
                msg += ". The allowed parameters were: "
                msg += prettyListString(currNeededParams, useAnd=True)
                msg += ". These were choosen as the best guess given the "
                msg += "inputs out of the following (numbered) list of "
                msg += "possible parameter sets: "
                msg += prettyListString(subParamGroup, numberItems=True,
                                        itemStr=prettyListString)
                msg += ". The full mapping of inputs actually provided was: "
                msg += prettyDictString(original) + ". "

                raise InvalidArgumentValue(msg)

            delayedAllocations = {}

        return (ret, delayedAllocations)

    def _setupValidationRecursiveCall(self, paramValue, available, callingRet,
                                      callingAllocations, original):
        # are the params for this value in a restricted argument pool?
        if paramValue in available:
            subSource = available[paramValue]
            subShared = False
            # We can and should do this here because if there is ever another
            # key with paramVale as the value, then it will be functionally
            # equivalent to save these args for then as it would be to use them
            # here. So, we do the easy thing, and consume them now.
            del available[paramValue]
        # else, they're in the main, shared, pool
        else:
            subSource = available
            subShared = True

        # where we get the wanted parameter set from depends on the kind of
        # thing that we need to instantiate
        if paramValue in self.listLearners():
            subParamGroup = self.getLearnerParameterNames(paramValue)
            subDefaults = self.getLearnerDefaultValues(paramValue)
        else:
            subParamGroup = self._getParameterNames(paramValue)
            subDefaults = self._getDefaultValues(paramValue)

        bestIndex = self._chooseBestParameterSet(subParamGroup, subDefaults,
                                                 subSource)
        subParamGroup = subParamGroup[bestIndex]
        subDefaults = subDefaults[bestIndex]

        (ret, allocations) = self._validateArgumentDistributionHelper(
            paramValue, subParamGroup, subDefaults, subSource, subShared,
            original)

        # integrate the returned values into the state of the calling frame
        callingRet[paramValue] = ret
        if subShared:
            for pair in allocations[None]:
                if paramValue not in callingAllocations:
                    callingAllocations[paramValue] = []
                callingAllocations[paramValue].append(pair)


    def _instantiateArguments(self, learnerName, arguments):
        """
        Recursively consumes the contents of the arguments parameter,
        checking for ones that need to be instantiated using in-package
        calls, and performing that action if needed. Returns a new
        dictionary with the same contents as 'arguments', except with
        the replacement of in-package objects for their string names.
        """
        toProcess = copy.deepcopy(arguments)
        return self._instantiateArgumentsHelper(toProcess)

    def _instantiateArgumentsHelper(self, toProcess):
        """
        Recursive function for actually performing
        _instantiateArguments. Will recurse when encountering shorthand
        for making in package calls, where the desired arguments are in
        another dictionary.
        """
        ignoreKeys = []
        ret = {}
        for paramName in toProcess:
            paramValue = toProcess[paramName]
            if isinstance(paramValue, six.string_types):
                ignoreKeys.append(paramValue)
                toCall = self.findCallable(paramValue)
                # if we can find an object for it, and we've prepped the
                # arguments, then we actually instantiate an object
                if toCall is not None and paramValue in toProcess:
                    subInitParams = toProcess[paramValue]
                    if subInitParams is None:
                        ret[paramName] = paramValue
                        continue
                    instantiatedParams = self._instantiateArgumentsHelper(
                        subInitParams)
                    paramValue = toCall(**instantiatedParams)

            ret[paramName] = paramValue

        # for key in ignoreKeys:
        #     if key in ret:
        #         del ret[key]

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
            msg += prettyListString(missing, numberItems=True,
                                    itemStr=prettyListString)
            msg += ". The following lists the required names in each of the "
            msg += "possible (numbered) parameter sets: "
            msg += prettyListString(nonDefaults, numberItems=True,
                                    itemStr=prettyListString)
            if len(arguments) == 0:
                msg += ". However, no arguments were inputed."
            else:
                msg += ". The full mapping of inputs actually provided was: "
                msg += prettyDictString(arguments) + ". "

            raise InvalidArgumentValue(msg)

        return bestIndex

    def _getMethodArguments(self, argNames, newArguments, storedArguments):
        applyArgs = {}
        invalidArguments = []
        for arg, value in newArguments.items():
            # valid argument change
            if arg in argNames and arg in storedArguments :
                applyArgs[arg] = value
            # not a valid argument for method
            else:
                invalidArguments.append(arg)
        if invalidArguments:
            msg = "EXTRA PARAMETER! "
            if argNames:
                msg += "The following parameter names cannot be applied: "
                msg += prettyListString(invalidArguments, useAnd=True)
                msg += ". The allowed parameters are: "
                msg += prettyListString(argNames, useAnd=True)
            else:
                msg += "No parameters are accepted for this operation"
            raise InvalidArgumentValue(msg)

        # use stored values for any remaining arguments
        for arg in argNames:
            if arg not in applyArgs:
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
        return self._getParameterNamesBackend(name)

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
        return self._getLearnerParameterNamesBackend(learnerName)

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
    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):
        """
        Method called before any package level function which transforms
        all parameters provided by a UML user.

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
        the returned value into a format appropriate for a UML user.
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
        trainX : UML.data.Base
            The training data.
        trainY : UML.data.Base
            The training labels.
        arguments : dict
            The transformed arguments.
        customDict : TODO

        Returns
        -------
        An in package object to be wrapped by a TrainedLearner object.
        """
        pass

    @abc.abstractmethod
    def _incrementalTrainer(self, learner, trainX, trainY, arguments,
                            customDict):
        """
        Extend the training of an already trained online learner.

        Parameters
        ----------
        learnerName : str
            The name of the learner.
        trainX : UML.data.Base
            The training data.
        trainY : UML.data.Base
            The training labels.
        arguments : dict
            The transformed arguments.
        customDict : TODO

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
        testX : UML.data.Base
            The testing data.
        arguments : dict
            The transformed arguments.
        customDict : TODO

        Returns
        -------
        UML friendly results.
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
        default value hasn't been registered in the UML configuration
        file. For example, these values will always be used the first
        time an interface is instantiated.
        """
        pass


    @abc.abstractmethod
    def _configurableOptionNames(self):
        """
        Returns a list of strings, where each string is the name of a
        configurable option of this interface whose value will be stored
        in UML's configuration file.
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
    Container for a learner model that has been trained. Provides
    methods for applying and testing the model.

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
        Reference to the customizable dictionary that is passed to I/O
        transformation, training and applying a learner.
    backend : object
        The return value from _trainer(), a reference to some object
        that is to be used by the package implementor during
        application.
    interfaceObject : UML.interfaces.UniversalInterface
        A reference to the subclass of UniversalInterface from which
        this TrainedLearner is being instantiated.
    has2dOutput : bool
        True if output will be 2-dimensional, False assumes the output
        will be 1-dimensional.
    """
    def __init__(self, learnerName, arguments, transformedInputs, customDict,
                 backend, interfaceObject, has2dOutput,
                 crossValidationResults):
        """
        Initialize the object wrapping the trained learner stored in
        backend, and setting up the object methods that may be used to
        modify or query the backend trained learner.
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
        self.crossValidation = crossValidationResults

        exposedFunctions = self.interface._exposedFunctions()
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
        testX : UML.data.Base
            The object containing the test data.
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
            since if there is no parameter selection to be done, it is
            not used.
        arguments : dict
            Mapping argument names (strings) to their values, to be used
            during training and application. eg. {'dimensions':5, 'k':5}
            To make use of multiple permutations, specify different
            values for a parameter as a tuple. eg. {'k': (1,3,5)} will
            generate an error score for  the learner when the learner
            was passed all three values of ``k``, separately. These will
            be merged with kwarguments for the learner.
        output : str
            The kind of UML Base object that the output of this function
            should be in. Any of the normal string inputs to the
            createData ``returnType`` parameter are accepted here.
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
        UML.trainAndTest, apply

        Examples
        --------
        TODO
        """
        timer = startTimer(useLog)
        #UML.helpers._2dOutputFlagCheck(self.has2dOutput, None, scoreMode,
        #                               multiClassStrategy)
        UML.helpers._2dOutputFlagCheck(self.has2dOutput, None, scoreMode, None)

        mergedArguments = _mergeArguments(arguments, kwarguments)
        pred = self.apply(testX, mergedArguments, output, scoreMode,
                          useLog=False)
        performance = UML.helpers.computeMetrics(testY, None, pred,
                                                 performanceFunction)
        time = stopTimer(timer)

        metrics = {}
        for key, value in zip([performanceFunction], [performance]):
            metrics[key.__name__] = value

        fullName = self.interface.getCanonicalName() + self.learnerName
        # Signature:
        # (umlFunction, trainData, trainLabels, testData, testLabels,
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
        to the learner). Equivalent to having called ``trainAndApply``,
        as long as the data and parameter setup for training was the
        same.

        Parameters
        ----------
        testX : UML Base object
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
            The kind of UML Base object that the output of this function
            should be in. Any of the normal string inputs to the
            createData ``returnType`` parameter are accepted here.
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
        UML.trainAndApply, test

        Examples
        --------
        TODO
        """
        timer = startTimer(useLog)
        UML.helpers._2dOutputFlagCheck(self.has2dOutput, None, scoreMode, None)

        mergedArguments = _mergeArguments(arguments, kwarguments)

        # input transformation
        transformedInputs = self.interface._inputTransformation(
            self.learnerName, None, None, testX, mergedArguments,
            self.customDict)
        transTestX = transformedInputs[2]
        usedArguments = transformedInputs[3]

        # depending on the mode, we need different information.
        labels = None
        if scoreMode != 'label':
            scores = self.getScores(testX, usedArguments)
        if scoreMode != 'allScores':
            labels = self.interface._applier(self.learnerName, self.backend,
                                             transTestX, usedArguments,
                                             self.transformedArguments,
                                             self.customDict)
            labels = self.interface._outputTransformation(
                self.learnerName, labels, usedArguments, output, "label",
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
                pointIndex = scores.points.getIndex(row.points.getName(0))
                rowIndex = scoreOrder.index(labels[pointIndex, 0])
                return row[rowIndex]

            scoreVector = scores.points.calculate(grabValue, useLog=False)
            labels.features.add(scoreVector, useLog=False)

            ret = labels

        time = stopTimer(timer)

        fullName = self.interface.getCanonicalName() + self.learnerName
        # Signature:
        # (self, umlFunction, trainData, trainLabels, testData, testLabels,
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
            .umlm is not included in file name it would be added to the
            output file.

        Examples
        --------
        TODO
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
                raise e
        # print('session_' + outputFilename)
        # print(globals())
        # dill.dump_session('session_' + outputFilename)

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
        trainX: UML Base object
            Data to be used for training.
        trainY : identifier, UML Base object
            * identifier - The name or index of the feature in
              ``trainX`` containing the labels.
            * UML Base object - contains the labels that correspond to
              ``trainX``.
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

        >>> trainX1 = UML.createData('Matrix', [[1, 1], [2, 2], [3, 3]])
        >>> trainY1 = UML.createData('Matrix', [[1], [2], [3]]) # mean=2
        >>> testX = UML.createData('Matrix', [[8, 8], [-3, -3]])
        >>> tl = UML.train('Custom.MeanConstant', trainX1, trainY1)
        >>> tl.apply(testX)
        Matrix(
            [[2.000]
             [2.000]]
            )
        >>> trainX2 = UML.createData('Matrix', [[4, 4], [5, 5], [6, 6]])
        >>> trainY2 = UML.createData('Matrix', [[4], [5], [6]]) # mean=5
        >>> tl.retrain(trainX2, trainY2)
        >>> tl.apply(testX)
        Matrix(
            [[5.000]
             [5.000]]
            )

        Changing the learner arguments.

        >>> trainX = UML.createData('Matrix', [[1, 1], [3, 3], [3, 3]])
        >>> trainY = UML.createData('Matrix', [[1], [3], [3]])
        >>> testX = UML.createData('Matrix', [[1, 1], [3, 3]])
        >>> tl = UML.train('Custom.KNNClassifier', trainX, trainY, k=1)
        >>> tl.apply(testX)
        Matrix(
            [[1.000]
             [3.000]]
            )
        >>> tl.retrain(trainX, trainY, k=3)
        >>> tl.apply(testX)
        Matrix(
            [[3.000]
             [3.000]]
            )
        """
        has2dOutput = False
        outputData = trainX if trainY is None else trainY
        if isinstance(outputData, UML.data.Base):
            has2dOutput = len(outputData.features) > 1
        elif isinstance(outputData, (list, tuple)):
            has2dOutput = len(outputData) > 1

        merged = _mergeArguments(arguments, kwarguments)
        for arg, value in merged.items():
            if isinstance(value, UML.CV):
                msg = "Cannot provide a cross-validation arguments "
                msg += "for parameters to retrain a TrainedLearner. "
                msg += "If wanting to perform cross-validation, use "
                msg += "UML.train()"
                raise InvalidArgumentValue(msg)
            if arg not in self.transformedArguments:
                validArgs = list(self.transformedArguments.keys())
                msg = "The argument '" + arg + "' is not valid. "
                if validArgs:
                    msg += "Valid arguments for retrain are: "
                    msg += prettyListString(validArgs)
                else:
                    msg += "There are no valid arguments to retrain "
                    msg += "this learner"
                raise InvalidArgumentValue(msg)
            self.arguments[arg] = value
            self.transformedArguments[arg] = value

        trainedBackend = self.interface._trainBackend(
            self.learnerName, trainX, trainY, self.transformedArguments)

        newBackend = trainedBackend[0]
        transformedInputs = trainedBackend[1]
        customDict = trainedBackend[2]

        self.backend = newBackend
        self.transformedTrainX = transformedInputs[0]
        self.transformedTrainY = transformedInputs[1]
        self.transformedArguments = transformedInputs[3]
        self.customDict = customDict
        self.has2dOutput = has2dOutput
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
        trainX: UML Base object
            Additional data to be used for training.
        trainY : identifier, UML Base object
            * identifier - The name or index of the feature in
            ``trainX`` containing the labels.
            * UML Base object - contains the labels that correspond to
              ``trainX``.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        """
        transformed = self.interface._inputTransformation(
            self.learnerName, trainX, trainY, None, self.arguments,
            self.customDict)
        transformedTrainX = transformed[0]
        transformedTrainY = transformed[1]
        transformedArguments = transformed[3]
        self.backend = self.interface._incrementalTrainer(
            self.backend, transformedTrainX, transformedTrainY,
            transformedArguments, self.customDict)

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
        discovered = self.interface._getAttributes(self.backend)
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
        order = self.interface._getScoresOrder(self.backend)
        numLabels = len(order)
        if numLabels == 2 and len(rawScores.features) == 1:
            ret = generateBinaryScoresFromHigherSortedLabelScores(rawScores)
            return UML.createData("Matrix", ret, useLog=False)

        if applyResults is None:
            applyResults = self.interface._applier(
                self.learnerName, self.backend, testX, arguments,
                self.transformedArguments, self.customDict)
            applyResults = self.interface._outputTransformation(
                self.learnerName, applyResults, arguments, "match", "label",
                self.customDict)
        if len(rawScores.features) != 3:
            strategy = ovaNotOvOFormatted(rawScores, applyResults, numLabels)
        else:
            strategy = checkClassificationStrategy(
                self.interface, self.learnerName, arguments)
        # want the scores to be per label, regardless of the original format,
        # so we check the strategy, and modify it if necessary
        if not strategy:
            scores = []
            for i in range(len(rawScores.points)):
                combinedScores = calculateSingleLabelScoresFromOneVsOneScores(
                    rawScores.pointView(i), numLabels)
                scores.append(combinedScores)
            scores = numpy.array(scores)
            return UML.createData("Matrix", scores, useLog=False)
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
        UML.data.Matrix
            The label scores.
        """
        usedArguments = _mergeArguments(arguments, kwarguments)
        (_, _, testX, usedArguments) = self.interface._inputTransformation(
            self.learnerName, None, None, testX, usedArguments,
            self.customDict)

        rawScores = self.interface._getScores(self.learnerName, self.backend,
                                              testX, usedArguments,
                                              self.transformedArguments,
                                              self.customDict)
        umlTypeRawScores = self.interface._outputTransformation(
            self.learnerName, rawScores, usedArguments, "Matrix", "allScores",
            self.customDict)
        formatedRawOrder = self._formatScoresToOvA(
            testX, None, umlTypeRawScores, usedArguments)
        internalOrder = self.interface._getScoresOrder(self.backend)
        naturalOrder = sorted(internalOrder)
        if numpy.array_equal(naturalOrder, internalOrder):
            return formatedRawOrder
        desiredDict = {}
        for i in range(len(naturalOrder)):
            label = naturalOrder[i]
            desiredDict[label] = i

        def sortScorer(feature):
            name = feature.features.getName(0)
            index = formatedRawOrder.features.getIndex(name)
            label = internalOrder[index]
            return desiredDict[label]

        formatedRawOrder.features.sort(sortHelper=sortScorer, useLog=False)
        return formatedRawOrder


@inheritDocstringsFactory(TrainedLearner)
class TrainedLearners(TrainedLearner):
    """
    Container for a learner models when the training employed a
    multiClassStrategy. Provides method for applying the models.
    """
    def __init__(self, trainedLearners, method, labelSet):
        self.trainedLearnersList = trainedLearners
        self.method = method
        self.labelSet = labelSet
        self.arguments = trainedLearners[0].arguments
        self.has2dOutput = trainedLearners[0].has2dOutput
        self.transformedArguments = trainedLearners[0].transformedArguments
        self.interface = trainedLearners[0].interface
        self.learnerName = trainedLearners[0].learnerName


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
        testX : UML Base object
            Data set on which the trained learner will be applied (i.e.
            performing prediction, transformation, etc. as appropriate
            to the learner).
        performanceFunction : function
            If cross validation is triggered to select from the given
            argument set, then this function will be used to generate a
            performance score for the run. Function is of the form:
            def func(knownValues, predictedValues).
            Look in UML.calculate for pre-made options. Default is None,
            since if there is no parameter selection to be done, it is
            not used.
        arguments : dict
            Mapping argument names (strings) to their values, to be used
            during training and application. eg. {'dimensions':5, 'k':5}
            To make use of multiple permutations, specify different
            values for a parameter as a tuple. eg. {'k': (1,3,5)} will
            generate an error score for  the learner when the learner
            was passed all three values of ``k``, separately. These will
            be merged with kwarguments for the learner.
        output : str
            The kind of UML Base object that the output of this function
            should be in. Any of the normal string inputs to the
            createData ``returnType`` parameter are accepted here.
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
        UML.trainAndApply, test

        Examples
        --------
        TODO
        """
        rawPredictions = None
        #1 VS All
        if self.method == 'OneVsAll':
            for trainedLearner in self.trainedLearnersList:
                oneLabelResults = trainedLearner.apply(testX, arguments,
                                                       output, 'label',
                                                       useLog)
                label = trainedLearner.label
                # put all results into one Base container; same type as trainX
                if rawPredictions is None:
                    rawPredictions = oneLabelResults
                    # as it's added to results object,
                    # rename each column with its corresponding class label
                    rawPredictions.features.setName(0, str(label), useLog=False)
                else:
                    # as it's added to results object,
                    # rename each column with its corresponding class label
                    oneLabelResults.features.setName(0, str(label), useLog=False)
                    rawPredictions.features.add(oneLabelResults, useLog=False)

            if scoreMode.lower() == 'label'.lower():

                getWinningPredictionIndices = rawPredictions.points.calculate(
                    extractWinningPredictionIndex, useLog=False)
                winningPredictionIndices = getWinningPredictionIndices.copy(
                    to="python list")
                winningLabels = []
                for [winningIndex] in winningPredictionIndices:
                    winningLabels.append([self.labelSet[int(winningIndex)]])
                return UML.createData(rawPredictions.getTypeString(),
                                      winningLabels,
                                      featureNames=['winningLabel'],
                                      useLog=False)

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
                resultsContainer = UML.createData("List", tempResultsList,
                                                  featureNames=featureNames,
                                                  useLog=False)
                return resultsContainer

            elif scoreMode.lower() == 'allScores'.lower():
                # create list of Feature Names/Column Headers for final
                # return object
                colHeaders = sorted([str(i) for i in self.labelSet])
                # create map between label and index in list,
                # so we know where to put each value
                colIndices = list(range(len(colHeaders)))
                labelIndexDict = {v: k for k, v in zip(colIndices, colHeaders)}
                featureNamesItoN = rawPredictions.features.getNames()
                predictionMatrix = rawPredictions.copy(to="python list")
                resultsContainer = []
                for row in predictionMatrix:
                    finalRow = [0] * len(colHeaders)
                    scores = UML.helpers.extractConfidenceScores(
                        row, featureNamesItoN)
                    for label, score in scores.items():
                        #get numerical index of label in return object
                        finalIndex = labelIndexDict[label]
                        #put score into proper place in its row
                        finalRow[finalIndex] = score
                    resultsContainer.append(finalRow)
                #wrap data in Base container
                return UML.createData(rawPredictions.getTypeString(),
                                      resultsContainer,
                                      featureNames=colHeaders, useLog=False)
            else:
                msg = "scoreMode must be 'label', 'bestScore', or 'allScores'"
                raise InvalidArgumentValue(msg)

        #1 VS 1
        elif self.method == 'OneVsOne':
            predictionFeatureID = 0
            for trainedLearner in self.trainedLearnersList:
                # train classifier on that data; apply it to the test set
                partialResults = trainedLearner.apply(testX, arguments, output,
                                                      'label', useLog)
                # put predictions into table of predictions
                if rawPredictions is None:
                    rawPredictions = partialResults.copy(to="List")
                else:
                    predictionName = 'predictions-' + str(predictionFeatureID)
                    partialResults.features.setName(0, predictionName, useLog=False)
                    rawPredictions.features.add(partialResults, useLog=False)
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
                resultsContainer = UML.createData("List", tempResultsList,
                                                  featureNames=featureNames,
                                                  useLog=False)
                return resultsContainer
            elif scoreMode.lower() == 'allScores'.lower():
                colHeaders = sorted([str(i) for i in self.labelSet])
                colIndices = list(range(len(colHeaders)))
                labelIndexDict = {v: k for k, v in zip(colIndices, colHeaders)}
                predictionMatrix = rawPredictions.copy(to="python list")
                resultsContainer = []
                for row in predictionMatrix:
                    finalRow = [0] * len(colHeaders)
                    scores = countWins(row)
                    for label, score in scores.items():
                        finalIndex = labelIndexDict[str(label)]
                        finalRow[finalIndex] = score
                    resultsContainer.append(finalRow)

                return UML.createData(rawPredictions.getTypeString(),
                                      resultsContainer,
                                      featureNames=colHeaders, useLog=False)
            else:
                msg = "scoreMode must be 'label', 'bestScore', or 'allScores'"
                raise InvalidArgumentValue(msg)
        else:
            raise ImproperObjectAction('Wrong multiclassification method.')


###########
# Helpers #
###########

def relabeler(point, label=None):
    """
    Determine if the point contains the label value. Returning 1 if
    True else 0.

    Used with points.calculate to convert a feature of labels into a
    binary feature. The default for label must is set to the actual
    label prior to calling points.calculate.
    """
    if point[0] != label:
        return 0
    else:
        return 1
