
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
This modules contains the objects required to perform hyperparameter
tuning. This process requires two protocols, a tuning protocol and a
validation protocol. Tuning protocols define how the next arguments set
will be chosen. Validation protocols determine how the performance of
each argument set will be measured.
"""

from abc import ABC, abstractmethod
import itertools
import time
import operator
from operator import itemgetter
import numbers
import copy
import sys
from timeit import default_timer
from functools import wraps

from packaging.version import Version
import numpy as np

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble.core.logger import handleLogging, loggingEnabled
from nimble.core.logger import deepLoggingEnabled
from nimble.random import pythonRandom, numpyRandom
from nimble.core._learnHelpers import computeMetrics
from nimble._utility import prettyDictString, prettyListString, quoteStrings
from nimble._utility import mergeArguments
from nimble._utility import hyperopt, storm_tuner


class FoldIterator(ABC):
    """
    Create and iterate through folds.

    Parameters
    ----------
    dataList : list
        A list of data objects to divide into folds.
    folds : int
        The number of folds to create.
    """
    def __init__(self, dataList):
        if dataList is None:
            raise InvalidArgumentType('dataList may not be None')
        if len(dataList) == 0:
            raise InvalidArgumentValue("dataList may not be or empty")
        self.dataList = dataList
        self.points = len(self.dataList[0].points)
        for data in self.dataList:
            if data is not None:
                if len(data.points) == 0:
                    msg = "One of the objects has 0 points, it is impossible "
                    msg += "to specify a valid number of folds"
                    raise InvalidArgumentValueCombination(msg)
                if len(data.points) != self.points:
                    msg = "All data objects in the list must have the same "
                    msg += "number of points and features"
                    raise InvalidArgumentValueCombination(msg)
                if data.getTypeString() == 'Sparse':
                    data._sortInternal('point')
        self.foldList = self.makeFoldList()
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.foldList):
            self.index = 0
            raise StopIteration
        # we're going to be separating training and testing sets through
        # extraction, so we have to copy the data in order not to destroy the
        # original sets across multiple folds
        copiedList = []
        for data in self.dataList:
            if data is None:
                copiedList.append(None)
            else:
                copiedList.append(data.copy())

            # we want each training set to be permuted wrt its ordering in the
            # original data. This is setting up a permutation to be applied to
            # each object
            #		indices = range(0, len(copiedList[0].points)
            #                              - len(self.foldList[self.index])))
            #		pythonRandom.shuffle(indices)
        indices = np.arange(0, (len(copiedList[0].points)
                                   - len(self.foldList[self.index])))
        numpyRandom.shuffle(indices)

        resultsList = []
        for copied in copiedList:
            if copied is None:
                resultsList.append((None, None))
            else:
                copied.name = None
                currTest = copied.points.extract(self.foldList[self.index],
                                                 useLog=False)
                currTrain = copied
                currTrain.points.permute(indices, useLog=False)
                resultsList.append((currTrain, currTest))
        self.index = self.index + 1
        return resultsList

    @abstractmethod
    def makeFoldList(self):
        """
        Generate the lists of indexes that represent each fold.
        """


class KFoldIterator(FoldIterator):
    """
    Separate the data into k number of folds.
    """
    def __init__(self, dataList, folds):
        if folds <= 0:
            msg = "Number of folds must be greater than 0"
            raise InvalidArgumentValue(msg)
        self.folds = folds
        super().__init__(dataList)

    def makeFoldList(self):
        # When number of points does not divide evenly by number of folds
        # some folds will need one more point than minNumInFold.
        # Ex. 10 folds and 98 points, we make 8 folds of 10 then 2 folds of 9
        minNumInFold, numFoldsToAddPoint = divmod(self.points, self.folds)
        if minNumInFold == 0:
            msg = "Must specify few enough folds so there is a point in each"
            raise InvalidArgumentValueCombination(msg)

        # randomly select the folded portions
        indices = list(range(self.points))
        pythonRandom.shuffle(indices)
        foldList = []
        end = 0
        for fold in range(self.folds):
            start = end
            end = (fold + 1) * minNumInFold
            if fold < numFoldsToAddPoint:
                end += fold + 1
            else:
                end += numFoldsToAddPoint
            foldList.append(indices[start:end])

        return foldList


class GroupFoldIterator(FoldIterator):
    """
    Separate the data into folds based on a feature of the X data.
    """
    def __init__(self, dataList, foldFeature):
        # ignore any points with missing values in foldFeature
        keep = [i for i, v in enumerate(foldFeature) if match.nonMissing(v)]
        self.foldFeature = foldFeature.points.copy(keep, useLog=False)
        for i, data in enumerate(dataList):
            if data is not None:
                dataList[i] = data.points.copy(keep, useLog=False)
        super().__init__(dataList)

    def makeFoldList(self):
        foldDict = {}
        for i, val in enumerate(self.foldFeature):
            if val in foldDict:
                foldDict[val].append(i)
            else:
                foldDict[val] = [i]

        return list(foldDict.values())


class Validator(ABC):
    """
    Base class for validation protocols.

    A Validator provides the opportunity to perform repeated validation
    of a learner on the same data with different arguments. Each time a
    validation is run, the results are stored within the object.
    """
    def __init__(self, learnerName, X, Y, performanceFunction, randomSeed,
                 useLog, **logInfo):
        if not hasattr(self, "name"):
            raise AttributeError("A Validator must have a name attribute")

        self.learnerName = learnerName

        if isinstance(Y, (int, str, list)):
            X = X.copy()
            Y = X.features.extract(Y, useLog=False)

        if Y is not None and not len(X.points) == len(Y.points):
            msg = "X and Y must contain the same number of points"
            raise InvalidArgumentValueCombination(msg)
        self.X = X
        self.Y = Y
        self.performanceFunction = performanceFunction
        # detectBestResult will raise exception for invalid performanceFunction
        optimal = nimble.calculate.detectBestResult(performanceFunction)
        self.optimal = optimal
        self._isBest = operator.gt if optimal == 'max' else operator.lt
        # use same random seed each time
        if randomSeed is None:
            self.randomSeed = nimble.random.generateSubsidiarySeed()
        else:
            self.randomSeed = randomSeed
        self.useLog = useLog

        self._results = []
        self._arguments = []
        self._best = None
        # used in __str__ and __repr__
        self._logInfo = logInfo

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        ret = f'{self.__class__.__name__}("{self.learnerName}", '
        ret += f'performanceFunction={self.performanceFunction.__name__}, '
        ret += f'randomSeed={self.randomSeed}'
        if self._logInfo:
            ret += ", "
            ret += prettyDictString(self._logInfo, valueStr=quoteStrings)
        ret += ")"
        return ret

    def validate(self, arguments=None, record=True, **kwarguments):
        """
        Apply the validation to a learner with the given arguments.
        """
        arguments = mergeArguments(arguments, kwarguments)
        performance = self._validate(arguments)
        if record:
            self._results.append(performance)
            self._arguments.append(arguments)
            if self._best is None or self._isBest(performance, self._best[0]):
                self._best = (performance, arguments)

        return performance

    @abstractmethod
    def _validate(self, arguments):
        pass


class CrossValidator(Validator):
    """
    Base class for Validators that perform cross validation.

    Each sublass should provide a FoldIterator and its kwargs by calling
    super().__init__. The FoldIterators are instantiated here, rather
    than in the subclass, so that X and Y can be preprocessed first.
    """
    def __init__(self, foldIterator, learnerName, X, Y, performanceFunction,
                 randomSeed, useLog, **kwargs):
        super().__init__(learnerName, X, Y, performanceFunction, randomSeed,
                         useLog, **kwargs)
        # a result should be added here for every fold
        self._deepResults = []
        self._foldIterator = foldIterator([self.X, self.Y], **kwargs)

    def _validate(self, arguments):
        """
        Cross validate and return the overall performance.

        The per-fold performance is also recorded during validation.
        """
        # fold iterator randomized the point order, so if we are collecting all
        # the results, we also have to collect the correct order of the known
        # values
        collectedY = None
        performances = []
        foldByFold = []
        deepLog = loggingEnabled(self.useLog) and deepLoggingEnabled()
        numFolds = len(self._foldIterator.foldList)
        for foldNum, fold in enumerate(self._foldIterator):
            [(curTrainX, curTestingX), (curTrainY, curTestingY)] = fold
            #run algorithm on the folds' training and testing sets
            startTime = time.process_time()
            curRunResult = nimble.trainAndApply(
                self.learnerName, curTrainX, curTrainY, curTestingX,
                arguments=arguments, randomSeed=self.randomSeed, useLog=False)

            # calculate error of prediction, using performanceFunction
            curPerformance = computeMetrics(curTestingY, None, curRunResult,
                                            self.performanceFunction)

            totalTime = time.process_time() - startTime

            foldByFold.append(curPerformance)
            performances.append(curRunResult)
            if collectedY is None:
                collectedY = curTestingY
            else:
                collectedY.points.append(curTestingY, useLog=False)

            metrics = {self.performanceFunction.__name__: curPerformance}
            extraInfo = {'Fold': f'{foldNum + 1}/{numFolds}'}

            handleLogging(deepLog, "deepRun", self.__class__.__name__,
                          curTrainX, curTrainY, curTestingX, curTestingY,
                          self.learnerName, arguments, self.randomSeed,
                          metrics=metrics, extraInfo=extraInfo, time=totalTime)

        self._deepResults.append(foldByFold)

        # combine the performances objects into one, and then calc performance
        for performanceIdx in range(1, len(performances)):
            performances[0].points.append(performances[performanceIdx],
                                          useLog=False)
        finalPerformance = computeMetrics(
            collectedY, None, performances[0], self.performanceFunction)

        return finalPerformance


class KFold(CrossValidator):
    """
    A cross-validation protocol using k number of folds.

    Randomly separates the X and Y data into "k" folds. For each of the
    "k" iterations a different fold is held out as the validation set
    and the remaining folds are used for training.

    Parameters
    ----------
    X : nimble.core.data.Base
        Data to be used for training.
    Y : identifier, nimble.core.data.Base
        A name or index of the feature in ``X`` containing the labels or
        another nimble Base object containing the labels that correspond
        to ``X``.
    performanceFunction : function
        Funtion used to determine the fold's performance, in the form:
        See nimble.calculate for pre-made options.
    folds : int
        The number of folds to divide the data into.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    """
    name = "cross validation"

    def __init__(self, learnerName, X, Y, performanceFunction, folds=5,
                 randomSeed=None, useLog=None):
        self.folds = folds
        super().__init__(KFoldIterator, learnerName, X, Y, performanceFunction,
                         randomSeed, useLog, folds=folds)


class LeaveOneOut(CrossValidator):
    """
    A cross-validation protocol witholding one point each iteration.

    This is a special case of k-fold cross-validation where "k" is set
    to the number of points in the object. Each iteration, a single
    point is heldout for validation and the remaining points are used
    for training.

    Parameters
    ----------
    X : nimble.core.data.Base
        Data to be used for training.
    Y : identifier, nimble.core.data.Base
        A name or index of the feature in ``X`` containing the labels or
        another nimble Base object containing the labels that correspond
        to ``X``.
    performanceFunction : function
        Funtion used to determine the fold's performance, in the form:
        See nimble.calculate for pre-made options.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    """
    name = "leave one out"

    def __init__(self, learnerName, X, Y, performanceFunction,
                 randomSeed=None, useLog=None):
        super().__init__(KFoldIterator, learnerName, X, Y, performanceFunction,
                         randomSeed, useLog, folds=len(X.points))
        # set in super, but not a parameter
        del self._logInfo['folds']


class LeaveOneGroupOut(CrossValidator):
    """
    A cross-validation protocol creating a fold for each group.

    This is a special case of k-fold cross-validation where "k" is set
    to the number of points in the object. Each iteration, a single
    point is heldout for validation and the remaining points are used
    for training.

    Parameters
    ----------
    X : nimble.core.data.Base
        Data to be used for training.
    Y : identifier, nimble.core.data.Base
        A name or index of the feature in ``X`` containing the labels or
        another nimble Base object containing the labels that correspond
        to ``X``.
    performanceFunction : function
        Funtion used to determine the fold's performance, in the form:
        See nimble.calculate for pre-made options.
    foldFeature : identifier
        A feature name or index to group by for the folds.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    """
    name = "leave one group out"

    def __init__(self, learnerName, X, Y, performanceFunction, foldFeature,
                 randomSeed=None, useLog=None):
        if foldFeature is None:
            msg = "foldFeature cannot be done when using leave one "
            msg += "group out validation"
            raise InvalidArgumentValue(msg)
        self.foldFeature = foldFeature
        if isinstance(foldFeature, (str, int, list)):
            if foldFeature == Y:
                msg = "foldFeature and Y cannot be the same feature"
                raise InvalidArgumentValueCombination(msg)
            foldFeature = X.features[foldFeature]
            if isinstance(Y, (str, int, list)):
                X = X.copy()
                Y = X.features.extract(Y, useLog=False)
        elif len(foldFeature.points) != len(X.points):
            msg = "foldFeature must have the same number of points as the X "
            msg += "data"
            raise InvalidArgumentValue(msg)
        super().__init__(GroupFoldIterator, learnerName, X, Y,
                         performanceFunction, randomSeed, useLog,
                         foldFeature=foldFeature)
        # use type string instead of object
        self._logInfo['foldFeature'] = self.foldFeature


class HoldoutValidator(Validator):
    """
    Base class for holdout validation protocols.

    All subclasses provide the a validateX and validateY data objects.
    """
    def __init__(self, learnerName, X, Y, performanceFunction, randomSeed,
                 useLog, validateX, validateY, logInfo):
        if validateX is None:
            msg = "validateX cannot be None"
            raise InvalidArgumentValue(msg)
        if validateY is None: # use Y value if an identifier
            if not isinstance(Y, (int, str, list)):
                msg = "No data provided for validateY"
                raise InvalidArgumentValue(msg)
            validateY = Y
        if isinstance(validateY, (int, str, list)):
            validateX = validateX.copy()
            validateY = validateX.features.extract(validateY, useLog=False)

        if not len(validateX.points) == len(validateY.points):
            msg = "validateX and validateY must contain the same number of "
            msg += "points"
            raise InvalidArgumentValueCombination(msg)

        if validateX.name is None:
            validateX.name = "validateX"
        if validateY.name is None:
            validateY.name = "validateY"

        self.validateX = validateX
        self.validateY = validateY

        super().__init__(learnerName, X, Y, performanceFunction, randomSeed,
                         useLog, **logInfo)

    def _validate(self, arguments):
        startTime = time.process_time()
        performance = nimble.trainAndTest(
            self.learnerName, self.performanceFunction, self.X, self.Y,
            self.validateX, self.validateY, arguments=arguments,
            randomSeed=self.randomSeed, useLog=False)
        totalTime = time.process_time() - startTime

        metrics = {self.performanceFunction.__name__: performance}
        deepLog = loggingEnabled(self.useLog) and deepLoggingEnabled()

        handleLogging(deepLog, "deepRun", self.__class__.__name__,
                      self.X, self.Y, self.validateX, self.validateY,
                      self.learnerName, arguments, self.randomSeed,
                      metrics=metrics, extraInfo=self._logInfo,
                      time=totalTime)

        return performance

class HoldoutData(HoldoutValidator):
    """
    A holdout protocol that provides the validation data.

    Parameters
    ----------
    X : nimble.core.data.Base
        Data to be used for training.
    Y : identifier, nimble.core.data.Base
        A name or index of the feature in ``X`` containing the labels or
        another nimble Base object containing the labels that correspond
        to ``X``.
    performanceFunction : function
        Funtion used to determine the fold's performance, in the form:
        See nimble.calculate for pre-made options.
    validateX : nimble.core.data.Base
        The data to use for validation.
    validateY : identifier, nimble.core.data.Base
        A name or index of the feature in ``validateX`` containing the
        labels or another nimble Base object containing the labels that
        correspond to ``validateX``.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    """
    name = "data holdout"

    def __init__(self, learnerName, X, Y, performanceFunction, validateX,
                 validateY, allowIncremental=False, randomSeed=None,
                 useLog=None):
        logInfo = {'validateX': validateX, 'validateY': validateY}
        super().__init__(learnerName, X, Y, performanceFunction, randomSeed,
                         useLog, validateX, validateY, logInfo)
        self._trainedLearner = None
        self._trainedLearnerBase = None
        self._incremental = None if allowIncremental else False
        self._updater = None
        self._lastArguments = {}
        self.bestTrainedLearner = None

    def validate(self, arguments=None, record=True, **kwarguments):
        """
        Apply the validation to a learner with the given arguments.
        """
        arguments = mergeArguments(arguments, kwarguments)
        performance = self._validate(arguments)
        if record:
            self._results.append(performance)
            self._arguments.append(arguments)
            if self._best is None or self._isBest(performance, self._best[0]):
                self._best = (performance, arguments)
                self.bestTrainedLearner = copy.copy(self._trainedLearnerBase)

        return performance

    def _updateTrainedLearner(self, trial):
        if trial:
            self._trainedLearner = copy.copy(self._trainedLearnerBase)
        else:
            self._trainedLearner = self._trainedLearnerBase

    def _validate(self, arguments):
        if self._trainedLearner is None:
            self._trainedLearnerBase = nimble.train(
                self.learnerName, self.X, self.Y, arguments, useLog=False)
            self._trainedLearner = self._trainedLearnerBase
            self._lastArguments = arguments
        elif self._updater is None:
            if self._incremental is None:
                try:
                    self._trainedLearner.incrementalTrain(
                        self.X, self.Y, arguments, useLog=False)
                    self._updater = self._trainedLearner.incrementalTrain
                    self._incremental = True
                except TypeError:
                    self._incremental = False
            if not self._incremental:
                self._updater = self._trainedLearner.retrain
                # only retrain if arguments have changed
                if arguments != self._lastArguments:
                    self._trainedLearner.retrain(self.X, self.Y, arguments,
                                                 useLog=False)
            self._lastArguments = arguments
        elif self._incremental or arguments != self._lastArguments:
            self._updater(self.X, self.Y, arguments, useLog=False)
            self._lastArguments = arguments

        performance = self._trainedLearner.test(
            self.performanceFunction, self.validateX, self.validateY,
            useLog=False)

        return performance


class HoldoutProportion(HoldoutValidator):
    """
    A holdout protocol withholding a proportion of the provided data.

    From the provided data, a random selection of the data is removed
    for training and used for validation.

    Parameters
    ----------
    X : nimble.core.data.Base
        Data to be used for training.
    Y : identifier, nimble.core.data.Base
        A name or index of the feature in ``X`` containing the labels or
        another nimble Base object containing the labels that correspond
        to ``X``.
    performanceFunction : function
        Funtion used to determine the fold's performance, in the form:
        See nimble.calculate for pre-made options.
    proportion : float
        A number between 0 and 1 indicating the proportion of data to
        holdout. Default is 0.2.
    randomSeed : int
       Set a random seed for the operation. When None, the randomness is
       controlled by Nimble's random seed.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    """
    name = "holdout proportion"

    def __init__(self, learnerName, X, Y, performanceFunction, proportion=0.2,
                 randomSeed=None, useLog=None):
        if proportion <= 0 or proportion >= 1:
            msg = "proportion must be between 0 and 1 (exclusive)"
            raise InvalidArgumentValue(msg)
        self.proportion = proportion
        number = int(proportion * len(X.points))
        selection = numpyRandom.choice(range(len(X.points)), number,
                                       replace=False)
        X = X.copy()
        validateX = X.points.extract(selection, useLog=False)
        if isinstance(Y, (int, str, list)):
            validateY = Y
        else:
            Y = Y.copy()
            validateY = Y.points.extract(selection, useLog=False)

        logInfo = {"proportion": proportion}
        super().__init__(learnerName, X, Y, performanceFunction, randomSeed,
                         useLog, validateX, validateY, logInfo)


class ArgumentSelector(ABC):
    """
    Base for objects that select sets of hyperparameters.

    An iterator object that provides a new set of arguments on each
    iteration from the possible combinations of ``Tune`` arguments. In
    general, the next argument set will be based on the performance of
    the previous set of arguments, so it is required that the
    ``ArgumentSelector`` know whether 'min' or 'max' values are optimal.
    Subclasses will likely need to implement an ``update`` method. This
    method takes in the performance of the last argument set and
    modifies the object attributes to correctly select the next set of
    arguments.
    """
    name = None

    def __init__(self, arguments, validator, **logInfo):
        if self.name is None:
            msg = "An ArgumentSelector must have a name attribute"
            raise AttributeError(msg)
        self.validator = validator
        self.arguments = arguments
        # these provide information for the logger
        self._logInfo = logInfo

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    def validateAll(self):
        """
        Validate every argument by exhausting the iterator.
        """
        for _ in self:
            pass


class BruteForce(ArgumentSelector):
    """
    Try every possible combination of arguments.

    Each argument combination is tested to determine the best argument
    set for the learner.

    Parameters
    ----------
    arguments : dict
        Mapping argument names (strings) to their values, to be used.
    validator : Validator
        The instance providing the validation method and data.
    """
    name = "brute force"

    def __init__(self, arguments, validator):
        super().__init__(arguments, validator)
        self._index = 0
        if self.arguments:
            iterableArgDict = {}
            self._numCombinations = 1
            for key, value in self.arguments.items():
                if isinstance(value, Tune):
                    self._numCombinations *= len(value)
                    iterableArgDict[key] = value
                else: # self._numCombinations not increased
                    # wrap in iterable so that itertools.product will treat
                    # whatever this value is as a single argument value even
                    # if the value itself is an iterable
                    iterableArgDict[key] = (value,)

            # note: calls to keys() and values() will directly correspond as
            # since no modification is made to iterableArgDict between calls.
            self._combinationsList = []
            for combination in itertools.product(*iterableArgDict.values()):
                combinationDict = {}
                for i, argument in enumerate(iterableArgDict.keys()):
                    combinationDict[argument] = combination[i]
                self._combinationsList.append(combinationDict)
        else:
            self._numCombinations = 1
            self._combinationsList = [{}]

    def __next__(self):
        if self._index >= self._numCombinations:
            self._index = 0 # necessary?
            raise StopIteration
        combination = self._combinationsList[self._index]
        self._index += 1
        self.validator.validate(combination)
        return combination


class Consecutive(ArgumentSelector):
    """
    Tune one argument at a time.

    For each argument that requires tuning, only the value of one
    argument will be changed while holding the others constant. Once the
    first argument has been optimized, the next argument is tuned and so
    on. The order in which the arguments are tuned can be set using the
    ``order`` parameter. By default each argument is tuned once, but
    this process can be repeated using the ``loops`` parameter. This
    allows arguments earlier in the order to be retuned based on the
    optimized values from the previous loop.

    Parameters
    ----------
    arguments : dict
        Mapping argument names (strings) to their values, to be used.
    validator : Validator
        The instance providing the validation method and data.
    loops : int
        The number of time to loop through the argument values being
        tuned.
    order : list
        A list of argument names indicating the order to tune the
        arguments.
    """
    name = "consecutive"

    def __init__(self, arguments, validator, loops=1, order=None):
        if loops < 1:
            msg = "loops must be greater than 1"
            raise InvalidArgumentValue(msg)
        logInfo = {'loops': loops}
        if order is not None:
            logInfo['order'] = order
        super().__init__(arguments, validator, **logInfo)
        self.loops = loops
        self.order = order
        self._currentLoop = 1
        self._tune = {}
        self._target = 0
        self._bestPerformance = None
        self._bestArgs = None
        self._currArgs = None
        self._tried = set()

    def __next__(self):
        if self._currentLoop > self.loops:
            raise StopIteration
        if self._currArgs is None:
            self._currArgs = {}
            if self.order is None:
                createOrder = True
                self.order = []
            else:
                createOrder = False
            for key, val in self.arguments.items():
                if isinstance(val, Tune):
                    self._currArgs[key] = val[0]
                    self._tune[key] = iter(val[1:])
                    if createOrder:
                        self.order.append(key)
                    elif key not in self.order:
                        msg = f"{key} is not in the order list"
                        raise ImproperObjectAction(msg)
                else:
                    self._currArgs[key] = val
        elif not self.order: # no Tune objects
            raise StopIteration
        else:
            target = self.order[self._target]
            try:
                self._currArgs[target] = next(self._tune[target])
            except StopIteration:
                if self._target + 1 < len(self.order): # next argument
                    self._currArgs[target] = self._bestArgs[target]
                    self._target += 1
                    target = self.order[self._target]
                    self._currArgs[target] = next(self._tune[target])
                # new loop
                elif self._currentLoop + 1 > self.loops:
                    raise
                else:
                    self._currentLoop += 1
                    self._target = 0
                    target = self.order[self._target]
                    self._currArgs = self._bestArgs.copy()
                    for key in self.order:
                        iterator = iter(val for val in self.arguments[key]
                                        if val != self._currArgs[key])
                        self._tune[key] = iterator
                    self._currArgs[target] = next(self._tune[target])
        # go to next if currArgs were already tried in a previous loop
        if tuple(self._currArgs.items()) in self._tried:
            return self.__next__()
        # save combo to avoid future loops from repeating combo
        self._tried.add(tuple(self._currArgs.items()))
        performance = self.validator.validate(self._currArgs)
        if (self._bestPerformance is None
                or self.validator._isBest(performance, self._bestPerformance)):
            self._bestPerformance = performance
            self._bestArgs = self._currArgs.copy()
        return self._currArgs.copy()


class Bayesian(ArgumentSelector):
    """
    Apply a bayesian algorithm to select the argument sets.

    This requires the hyperopt library to apply Tree Parzen Estimators
    (TPE) and Expected Improvement (EI) algorithms to determine the
    next best set of arguments to try.

    Parameters
    ----------
    arguments : dict
        Mapping argument names (strings) to their values, to be used.
    validator : Validator
        The instance providing the validation method and data.
    maxIterations : int
        The number of times to evaluate the argument space. Default is
        100.
    timeout : float, int, None
        The maximum amount of time in seconds to spend finding the best
        arguments. Default is ``None``, meaning this will stop based on
        ``maxIterations`` or ``threshold``.
    threshold : float, int, None
        Stop selection if performance is better than or equal to the
        threshold. Default is ``None``, meaning this will stop based on
        ``maxIterations`` or ``timeout``.
    """
    name = "bayesian"

    def __init__(self, arguments, validator, maxIterations=100, timeout=None,
                 threshold=None):
        logInfo = {}
        if maxIterations is not None:
            logInfo['maxIterations'] = maxIterations
        if timeout is not None:
            logInfo['timeout'] = timeout
        if threshold is not None:
            logInfo['threshold'] = threshold
        super().__init__(arguments, validator, **logInfo)
        if not hyperopt.nimbleAccessible():
            msg = "The hyperopt library must be installed to perform "
            msg += "hyperparameter tuning with the Bayesian method"
            raise PackageException(msg)

        if validator.optimal != 'min':
            msg = "The performanceFunction must be minimum optimal for "
            msg += 'Bayesian'
            raise InvalidArgumentValue(msg)
        hp = hyperopt.hp
        if maxIterations is None:
            maxIterations = sys.maxsize
        self.maxIterations = maxIterations
        self.timeout = timeout
        self.threshold = threshold
        space = {}
        self._noTune = True
        # determine the distribution for each argument
        for name, arg in self.arguments.items():
            if isinstance(arg, Tune):
                self._noTune = False
                if arg._changeType == 'add':
                    if all(v % 1 == 0 for v
                           in [arg._start, arg._end, arg._change]):
                        space[name] = hp.uniformint(name, arg._start, arg._end,
                                                    q=arg._change)
                    else:
                        space[name] = hp.quniform(name, arg._start, arg._end,
                                                  arg._change)
                elif arg._changeType == 'exponential':
                    space[name] = hp.qloguniform(name, arg._start, arg._end,
                                                 arg._change)
                else:
                    space[name] = hp.choice(name, arg._values)
            else:
                space[name] = arg

        domain = hyperopt.Domain(self.validator.validate, space)
        self.trials = hyperopt.Trials()

        # 0.2.6 and afterwards only allows for BitGenerator rng sources
        if Version(hyperopt.__version__) >= Version("0.2.6"):
            sourceSeed = nimble.random.generateSubsidiarySeed()
            randSource = np.random.default_rng(sourceSeed)
        else:
            randSource = nimble.random.numpyRandom

        self.iterator = hyperopt.FMinIter(
            hyperopt.tpe.suggest, domain, self.trials,
            randSource, max_evals=self.maxIterations)

        self._stopIteration = False
        self._startTime = None

    def __next__(self):
        # calling next on FMinIter does not properly manage threshold and
        # timing so these are handled internally
        if self._startTime is None:
            self._startTime = default_timer()
        if (self._stopIteration
                or (self._noTune and self.validator._incremental is False)
                or (self.timeout is not None
                    and default_timer() - self._startTime > self.timeout)
                or (self.validator._results and self.threshold is not None
                    and self.validator._results[-1] <= self.threshold)):
            raise StopIteration
        # FMinIter raises StopIteration immediately after the last validation
        # call so need to catch and return the arguments used during that call
        # then stop iteration on the next call
        try:
            _ = next(self.iterator)
        except StopIteration:
            self._stopIteration = True
        return self.validator._arguments[-1]


class Iterative(ArgumentSelector):
    """
    Select next arguments by testing small changes to the current set.

    Each iteration, a higher and lower value from the ``Tune`` arguments
    are validated. If at the upper or lower bound of the arguments, only
    the one side will be tested. The value that provides the best
    increase in performance will be used to update the training model
    for the next iteration. On the next iteration, the best values from
    the previous iteration are the starting points. If a one surrounding
    value provided the same performance, the search for better values
    will continue in that direction. If both provided the same
    performance, the direction will be chosen at random.

    Parameters
    ----------
    arguments : dict
        Mapping argument names (strings) to their values, to be used.
    validator : Validator
        The instance providing the validation method and data.
    maxIterations : int
        The number of times to evaluate the argument space. Default is
        100.
    timeout : float, int, None
        The maximum amount of time in seconds to spend finding the best
        arguments. Default is ``None``, meaning this will stop based on
        ``maxIterations`` or ``threshold``.
    threshold : float, int, None
        Stop selection if performance is better than or equal to the
        threshold. Default is ``None``, meaning this will stop based on
        ``maxIterations`` or ``timeout``.
    """
    name = "iterative"

    def __init__(self, arguments, validator, maxIterations=100, timeout=None,
                 threshold=None):
        logInfo = {}
        if maxIterations is not None:
            logInfo['maxIterations'] = maxIterations
        if timeout is not None:
            logInfo['timeout'] = timeout
        if threshold is not None:
            logInfo['threshold'] = threshold
        super().__init__(arguments, validator, **logInfo)
        if maxIterations is None:
            maxIterations = sys.maxsize
        self.maxIterations = maxIterations
        self.timeout = timeout
        self.threshold = threshold
        self._currIteration = 0

        self._currArgs = {}
        self._tuneIdx = {}
        self._tuneVals = {}
        for name, arg in arguments.items():
            if isinstance(arg, Tune):
                vals = sorted(arg)
                idx = len(vals) // 2 # start with middle value
                self._tuneIdx[name] = idx
                self._tuneVals[name] = vals
                self._currArgs[name] = vals[idx]
            else:
                self._currArgs[name] = arg

        self._startTime = None
        self._bestPerf = None
        self._bestArgs = None

    def __next__(self):
        if self._startTime is None:
            self._startTime = default_timer()
            self._bestPerf = self.validator.validate(self._currArgs)
            self._bestArgs = self._currArgs.copy()
            self._currIteration += 1
            return self._currArgs.copy()

        if (self._currIteration >= self.maxIterations
                or (self.timeout is not None
                    and default_timer() - self._startTime > self.timeout)
                or (self.threshold is not None
                    and (self.validator._isBest(self._bestPerf, self.threshold)
                         or self._bestPerf == self.threshold))):
            raise StopIteration
        self.validator._updateTrainedLearner(trial=False)
        best = {}
        for name, idx in self._tuneIdx.items():
            baseArgs = self._bestArgs.copy()
            values = self._tuneVals[name]
            if idx - 1 >= 0:
                argLow = values[idx - 1]
                baseArgs[name] = argLow
                self.validator._updateTrainedLearner(trial=True)
                perfLow = self.validator.validate(baseArgs, record=False)
                lowBest = self.validator._isBest(perfLow, self._bestPerf)
            else:
                lowBest = False
            if idx + 1 < len(values):
                argHigh = values[idx + 1]
                baseArgs[name] = argHigh
                self.validator._updateTrainedLearner(trial=True)
                perfHigh = self.validator.validate(baseArgs, record=False)
                highBest = self.validator._isBest(perfHigh, self._bestPerf)
            else:
                highBest = False
            if lowBest and highBest: # select better of the two
                if self.validator._isBest(perfHigh, perfLow):
                    best[name] = argLow
                    self._tuneIdx[name] += 1
                else:
                    best[name] = argHigh
                    self._tuneIdx[name] -= 1
            elif lowBest:
                best[name] = argLow
                self._tuneIdx[name] -= 1
            elif highBest:
                best[name] = argHigh
                self._tuneIdx[name] += 1

        if not best and not self.validator._incremental:
            # if it is not possible to incrementally train, these
            # results will not improve from here
            raise StopIteration
        if best:
            self._currArgs.update(best)
            self.validator._updateTrainedLearner(trial=True)
            performance = self.validator.validate(self._currArgs)
            if self.validator._isBest(performance, self._bestPerf):
                self._bestPerf = performance
                self._bestArgs = self._currArgs.copy()
        # even if no new parameters were best, incremental training could still
        # find better future parameters so continue
        self._currIteration += 1
        return self._currArgs.copy()

class StochasticRandomMutator(ArgumentSelector):
    """
    Argument selection with a stochastic random mutator (storm).

    This requires the storm_tuner library to be installed. For arguments
    provided to the learner, ``Tune`` objects can be used as they
    normally are. However, storm_tuner also supports tuning parameters
    that are not parameters of the learner object. For example, changing
    the activation function used within the neural network layers or
    even varying the number of hidden layers in the network. Nimble
    supports this as well and how to do so is explained below.

    Hyperparameter tuning with storm_tuner, requires a function that
    builds the model with adjustable parameters and for Nimble this
    works in a similar way. The difference is that Nimble requires the
    function to return the arguments to build the model instead of a
    model object. The function accepts a HyperParameters instance from
    storm_tuner and is used in the same way. Below, each call to
    hp.Param defines the name and possible options for that
    hyperparameter. The current value of the hyperparameter are also
    accessible through hp.values, a dictionary. The function then
    returns a dictionary that Nimble will merge with any other arguments
    while training and testing. See the example below.

    def modelArguments(hp):
        layers = []
        kernel0 = hp.Param('kernelSize0', [64, 128, 256], ordered=True)
        activation = hp.Param('activation', ['relu', 'elu'])
        # layers created from the current choice for the above arguments
        layers.append(nimble.Init('Dense', units=kernel0))
        layers.append(nimble.Init('Activation', activation=activation))
        # access the current kernelSize0 value to determine future units
        kernelSize = hp.values['kernelSize0']
        # build a variable number of hidden layers
        for x in range(hp.Param('num_layers', [1, 2, 3], ordered=True)):
            kernelSize = int(0.75 * kernelSize)
            layers.append(nimble.Init('Dense', units=kernelSize))
            layers.append(nimble.Init('Activation',
                                      activation=activation))

        layers.append(nimble.Init('Dense', units=3,
                                  activation='sigmoid'))

        return {'layers': layers}

    # Without hyperparameter tuning, layers would need to be provided
    # but here modelArguments will determine the best layers.
    nimble.train('Keras.Sequential', trainX, trainY, epochs=25,
                 tuning=Tuning('storm', 'data', fractionIncorrect,
                               validateX=valX, validateY=valY,
                               learnerArgsFunc=modelArguments),
                 loss='sparse_categorical_crossentropy',
                 optimizer=Tune(['Adam', 'Adamax']),
                 metrics=['accuracy'], verbose=0)

    Parameters
    ----------
    arguments : dict
        Mapping argument names (strings) to their values, to be used.
    validator : Validator
        The instance providing the validation method and data.
    learnerArgsFunc : function, None
        The function defining how each learner is built based on the
        current parameters. Required when building the learner requires
        tuning parameters that are outside the scope of ``Tune``.
    maxIterations : int
        The number of times to evaluate the argument space. Default is
        100.
    timeout : float, int, None
        The maximum amount of time in seconds to spend finding the best
        arguments. Default is ``None``, meaning this will stop based on
        ``maxIterations`` or ``threshold``.
    threshold : float, int, None
        Stop selection if performance is better than or equal to the
        threshold. Default is ``None``, meaning this will stop based on
        ``maxIterations`` or ``timeout``.
    """
    name = 'storm'

    def __init__(self, arguments, validator, learnerArgsFunc=None,
                 initRandom=5, randomizeAxisFactor=0.75, maxIterations=100,
                 timeout=None, threshold=None):
        logInfo = {}
        if learnerArgsFunc is not None:
            logInfo['learnerArgsFunc'] = learnerArgsFunc.__name__
        if initRandom is not None:
            logInfo['initRandom'] = initRandom
        else:
            initRandom = 5
        if randomizeAxisFactor is not None:
            logInfo['randomizeAxisFactor'] = randomizeAxisFactor
        else:
            randomizeAxisFactor = 0.75
        if maxIterations is not None:
            logInfo['maxIterations'] = maxIterations
        if timeout is not None:
            logInfo['timeout'] = timeout
        if threshold is not None:
            logInfo['threshold'] = threshold
        super().__init__(arguments, validator, **logInfo)
        if not storm_tuner.nimbleAccessible():
            msg = "The storm_tuner library must be installed to perform "
            msg += "hyperparameter tuning with the StoRM (Stochastic Random "
            msg += "Mutator) method"
            raise PackageException(msg)
        # override storm_tuner randomness reproducibility
        storm_tuner.tuner.random = pythonRandom
        # storm_tuner saves the best trial result to a project directory and
        # creates one if not provided. We won't support saving, so need to
        # prevent Path.mkdir from generating the default directory
        storm_tuner.tuner.Path.mkdir = lambda *args, **kwargs: None

        self.maxIterations = maxIterations
        self.timeout = timeout
        self.threshold = threshold

        @wraps(learnerArgsFunc)
        def _wrappedBuildFunc(hp):
            # all arguments in one dictionary; convert to hp.Param for Tune
            buildArgs = {}
            logArgs = {}
            if learnerArgsFunc is not None:
                buildArgs.update(learnerArgsFunc(hp))
                logArgs.update(hp.values)
            for name, arg in self.arguments.items():
                if name in buildArgs:
                    msg = f'The "{name}" argument should not be provided '
                    msg += 'because it is being defined by the learnerArgsFunc'
                    raise InvalidArgumentValueCombination(msg)
                if isinstance(arg, Tune):
                    buildArgs[name] = hp.Param(
                        name, list(arg), ordered=arg._changeType is not None)
                    logArgs[name] = buildArgs[name]
                else:
                    buildArgs[name] = arg
                    logArgs[name] = buildArgs[name]

            return buildArgs, logArgs

        class Tuner(storm_tuner.Tuner):
            """
            Tuner to use within nimble.
            """
            def __init__(self, validator, project_dir=None, build_fn=None,
                         randomize_axis_factor=0.75, init_random=5,
                         overwrite=True, max_iters=100, seed=None):
                super().__init__(project_dir, build_fn, randomize_axis_factor,
                                 init_random, validator.optimal, overwrite,
                                 max_iters, seed)
                self._iters_checked = False

            def run_trial(self, trial, selector):
                """
                Use validator to score the trial.
                """
                hp = trial.hyperparameters
                # infinite loop if max_iters is greater than the total number
                # of permutations
                if not self._iters_checked:
                    maxOptions = len(list(itertools.product(
                        *(p.values for p in hp.space.values()))))
                    # pylint: disable=attribute-defined-outside-init
                    self.max_iters = min(self.max_iters, maxOptions)
                buildArgs, logArgs = _wrappedBuildFunc(hp)
                score = selector.validator.validate(buildArgs)
                # the validator stores arguments and best values based on the
                # buildArgs, but want to store the hyperparameters set within
                # _wrappedBuildFunc (which generated the buildArgs)
                selector.validator._arguments[-1] = logArgs
                self.score_trial(trial, score)
                if (selector._bestPerf is None or
                        selector.validator._isBest(score, selector._bestPerf)):
                    selector._bestPerf = score
                    selector.validator._best = (score, logArgs)

        self.tuner = Tuner(self.validator, build_fn=_wrappedBuildFunc,
                           init_random=initRandom, max_iters=maxIterations,
                           randomize_axis_factor=randomizeAxisFactor)

        self._startTime = None
        self._bestPerf = None

    def __next__(self):
        if self._startTime is None:
            self._startTime = default_timer()
        if (len(self.tuner.trials) >= self.tuner.max_iters
                or (self.timeout is not None
                    and default_timer() - self._startTime > self.timeout)
                or (self.threshold is not None and self._bestPerf is not None
                    and (self.validator._isBest(self._bestPerf, self.threshold)
                         or self._bestPerf == self.threshold))):
            raise StopIteration
        # from storm_tuner.Tuner.search
        trial = self.tuner._create_trial()
        trial.hyperparameters.initialized = True
        self.tuner.run_trial(trial, self)
        return self.validator._arguments[-1]


class Tune:
    """
    Triggers hyperparameter optimization to occur during training.

    Provide or generate the possible argument values to use during the
    hyperparameter optimization process (defined by a ``Tuning`` object)
    and the best argument will be used to train the learner. A list of
    predetermined values can be passed as the ``values`` parameter or a
    range of values can be constructed using the ``start``, ``end``,
    ``change`` and ``changeType`` parameters. Only ``end`` is required
    in this case, the other parameters will be assigned default values
    if not explicitly set (see below).

    Parameters
    ----------
    values : list
        A list of argument values to use for tuning. Either this
        or ``end`` must not be None.
    start : int, float
        The inclusive start value of the values in a range. Default to 0
        and only applies when ``end`` is not None.
    end : int, float
        The inclusive end value of the values in a range. Either this or
        ``values`` must not be None.
    change : int, float
        The amount by which to changeType the data in the range. The
        ``changeType`` parameter will dictate whether this uses addition
        or multiplication. Defaults to 1 and only applies when when
        ``end`` is not None.
    changeType : str
        Either 'add' or 'multiply' to indicate how the ``change`` will
        be used to generate the range. Defaults to 'add' and only
        applies when ``end`` is not None.

    See Also
    --------
    Tuning, nimble.train

    Keywords
    --------
    cross-validation, parameter, argument, hyperparameters, tuning,
    optimization, cross validate, learn, hyper parameters,
    hyperparameters, choose, grid search, GridSearchCV
    """

    def __init__(self, values=None, start=None, end=None, change=None,
                 changeType=None):
        if values is None and end is None:
            msg = 'Either values or end must not be None'
            raise InvalidArgumentValueCombination(msg)
        if values is not None and end is not None:
            msg = "If values is provided, "
        self._values = values
        self._start = start
        self._end = end
        self._change = change
        self._changeType = changeType

        # prevents needing to autodetect a linear range
        if isinstance(values, range):
            self._values = None
            self._start = values.start
            self._end = values.stop - 1
            self._change = values.step

        self._args = {'values': values, 'start': start, 'end': end,
                      'change': change, 'changeType': changeType}

        if self._values is None:
            if self._start is None:
                self._start = 0
            if self._change is None:
                self._change = 1
            if self._changeType is None:
                self._changeType = 'add'
                linear = True
            else:
                linear = changeType == 'add'
            if self._changeType == 'add':
                op = operator.add
            elif self._changeType == 'multiply':
                op = operator.mul
            else:
                msg = "changeType must be 'add' or 'multiply'"
                raise InvalidArgumentValue(msg)

            start = self._start
            change = self._change
            self._values = []
            if self._start < self._end:
                if not linear and change < 1:
                    msg = "change must be greater than 1"
                    raise InvalidArgumentValue(msg)
                if linear and change <= 0:
                    msg = "change must be greater than 0"
                    raise InvalidArgumentValue(msg)
                comp = operator.le
            elif self._start > self._end:
                if not linear and (self._change >= 1 or self._change <= 0):
                    msg = "change must be between 0 and 1 (exclusive)"
                    raise InvalidArgumentValue(msg)
                if linear and self._change >= 0:
                    msg = "change must be less than 0"
                    raise InvalidArgumentValue(msg)
                comp = operator.ge

            curr = self._start
            while comp(curr, self._end):
                self._values.append(curr)
                curr = op(curr, self._change)
                # rounding errors can prevent end from being included
                if linear and round(curr, 12) == self._end:
                    curr = self._end

        else:
            # detect if values is linear or exponential, i.e if values was set
            # to range(2, 10) this would identify the data as linear.
            try:
                if (len(values) > 2 and
                    all(isinstance(v, numbers.Number) for v in values)):
                    sort = sorted(values)
                    difLin = sort[1] - sort[0]
                    linDiffs = (sort[i + 1] - sort[i] for i
                                in range(len(sort) - 1))
                    if all(d == difLin for d in linDiffs):
                        self._start = sort[0]
                        self._end = sort[-1]
                        self._change = difLin
                        self._changeType = 'add'
                    elif sort[0]:
                        difExp = sort[1] / sort[0]
                        expDiffs = (sort[i + 1] / sort[i] for i
                                    in range(len(sort) - 1))
                        # round to avoid floating point issues
                        if all(round(d, 12) == difExp for d in expDiffs):
                            self._start = sort[0]
                            self._end = sort[-1]
                            self._change = difExp
                            self._changeType = 'exponential'
            except TypeError as e:
                msg = "values must be iterable."
                raise InvalidArgumentType(msg) from e


    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        raise ImproperObjectAction("Tune objects are immutable")

    def __len__(self):
        return len(self._values)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        ret = "Tune("
        args = {}
        for arg, val in self._args.items():
            if val is not None:
                args[arg] = val
        ret += prettyDictString(args, valueStr=quoteStrings)
        ret += ")"
        return ret


class Tuning:
    """
    Define the method to identify the best values to train the learner.

    This object is passed to the ``tuning`` parameter in Nimble's
    training functions to provide a protocol for evaluating multiple
    arguments to determine the best argument set for the model. To make
    this determination, the training function must know:
    1) the sets of arguments to test (amongst all possible combinations)
    2) how to evaluate the performance of each set
    Multiple arguments are specified by providing ``Tune`` objects as
    learner arguments for the training function.

    The ``selection`` parameter identifies which argument sets to try
    and accepts the following:

        - "brute force" : Try every possible combination
        - "consecutive" : Optimize one argument at a time, holding the
          others constant. Optionally, multiple ``loops`` can occur and
          the ``order`` that the parameters are tuned can be defined.
        - "bayesian" : Apply a bayesian algorithm to the argument space.
          Note: When there is a correlation between an argument value
          and loss, the ``Tune`` objects provided should provide a
          linear or exponential range of values. This allows all values
          in that space to be sampled, otherwise only the provided
          values will be sampled and assumed to have no correlation with
          loss.
        - "iterative" : Beginning with the middle value of the sorted
          arguments, tries the higher and lower values (holding others
          constant) then applies the best (higher, lower, or same)
          argument on the next iteration. Note: This requires arguments
          to be numeric and assumes there is a correlation between the
          values and the performance.
        - "storm" : Apply a stochastic random mutator to the argument
          space. Randomly selects argument sets to begin, then starts
          to optimize the best performing set, while selecting random
          values at some given probability to avoid local optima.
          Note: For ordered numeric values, this assumes there is a
          correlation between the values and the performance.

    The ``validation`` parameter identifies how the performance will be
    evaluated and accepts the following:

       - Cross Validations:
           - "cross validation" : perform k-fold cross-validation with
             the training data. The number of folds can be set using the
             ``folds`` parameter.
           - "leave one out" : A k-fold cross-validation where the
             number of folds is equal to the number of points in the
             training data.
           - "leave one group out" : The folds are determined by a
             feature in the data. This requires a ``foldFeature``.
       - Holdout Validations:
           - "proportion" : A random proportion of the training data is
             held out. Requires the ``proportion`` parameter. As a
             shortcut, ``validation`` can be set directly to a float
             value to trigger this validation.
           - "data" : Provide the data to use for validation. These are
             passed as the ``validateX`` and ``validateY`` parameters.

    Parameters
    ----------
    selection : str
        How the next argument set will be chosen. Accepts "brute force",
        "consecutive", "bayesian" and "iterative".
    validation : str, float
        How each argument set will be validated. Accepts
        "cross validation", "leave one out", "leave one group out",
        "data", and "proportion" as strings or a float between 0 and 1
        will also trigger "proportion" validation. See above for
        descriptions of each validation.
    performanceFunction : function, None
        The function that will be used to validate the performance of
        each argument set. If None, the performance function provided to
        the training function will be applied.
    loops : int
       Applies when ``selection`` is "consecutive". For more than one
       loop, the values for the arguments not being optimized will be
       set to the optimal values from the previous loop.
    order : list
       Applies when ``selection`` is "consecutive". A list of argument
       names defining the order to use when tuning.
    maxIterations : int
        Applies when ``selection`` is "bayesian", "iterative", or
        "storm". The maximum number of times iterate through the
        argument selection process. Default is 100.
    timeout : int, None
        Applies when ``selection`` is "bayesian", "iterative", or
        "storm". The maximum number of seconds to perform the argument
        selection process.
    threshold : float, None
        Applies when ``selection`` is "bayesian", "iterative", or
        "storm". Stop the argument selection process if the performance
        is better than or equal to the threshold.
    learnerArgsFunc : function, None
        Applies when the ``selection`` is "storm". A function defining
        how to build the model with variable hyperparameters. Takes the
        form: learnerArgsFunc(hyperparameters) where hyperparameters
        will be a HyperParameters instance from storm_tuner and must
        return a dictionary to use as the arguments parameter for
        nimble.train.
    initRandom : int
        Applies when the ``selection`` is "storm". The number of initial
        iterations to perform a random search. Recommended value is
        between 3 and 8.
    randomizeAxisFactor : float
        Applies when the ``selection`` is "storm". Controls the tradeoff
        between explorative and exploitative selection. Values closer to
        1 are likely to generate more mutations, while values closer to
        0 are more likely to only perform a single mutation during each
        step.
    folds : int
        Applies when ``validation`` is "cross validation". Default is 5.
    foldFeature : identifier
        Applies when ``validation`` is "leave one group out". The folds
        for cross validation will be created by grouping the data by
        this feature.
    validateX : nimble data object
        Applies when ``validation`` is "data". The validation set to
        use. Can contain the validateY data.
    validateY : nimble data object, identifier
        Applies when ``validation`` is "data". Either an object of
        labels for the validation set, or the name or index of the
        labels in ``validateX``.
    proportion : float
        Applies when ``validation`` is "proportion". A value between 0
        and 1 indicating the random proportion of the training data to
        holdout for validation. A float value can also be passed
        directly to ``validation`` to trigger this same validation.

    See Also
    --------
    Tune, nimble.train
    """
    _selections = ["brute force", "consecutive", "bayesian", "iterative",
                   "storm"]
    _validations = ["cross validation", "leave one out", "leave one group out",
                    "data", "proportion"]
    _selectors = {"bruteforce": BruteForce,
                  "consecutive": Consecutive,
                  "bayesian": Bayesian,
                  "iterative": Iterative,
                  "storm": StochasticRandomMutator}
    _validators = {"crossvalidation": KFold,
                   "leaveoneout": LeaveOneOut,
                   "leaveonegroupout": LeaveOneGroupOut,
                   "data": HoldoutData,
                   "proportion": HoldoutProportion}
    def __init__(self, selection="consecutive", validation="cross validation",
                 performanceFunction=None, loops=1, order=None,
                 maxIterations=100, timeout=None, threshold=None,
                 learnerArgsFunc=None, initRandom=5, randomizeAxisFactor=0.75,
                 folds=5, foldFeature=None, validateX=None, validateY=None,
                 proportion=0.2):
        self.selection = selection
        self.validation = validation
        self.performanceFunction = performanceFunction

        # Arguments are validated by the objects they are passed to not here
        self._selectorArgs = {}
        selection = selection.lower().replace(" ", "")
        selection = selection.lower().replace("-", "")
        if selection not in Tuning._selectors:
            msg = prettyListString(Tuning._selections, useAnd=True)
            msg += " are the only accepted selections."
            raise InvalidArgumentValue(msg)
        self._selection = selection
        self._selector = Tuning._selectors[selection]
        dataOnly = ['bayesian', 'iterative', 'storm']
        if selection == "consecutive":
            self._selectorArgs['loops'] = loops
            self._selectorArgs['order'] = order
        elif selection in dataOnly:
            if maxIterations is None and timeout is None and threshold is None:
                msg = "One of maxIterations, timeout, or threshold must be set"
                raise InvalidArgumentValueCombination(msg)
            if selection == 'storm':
                self._selectorArgs['learnerArgsFunc'] = learnerArgsFunc
                self._selectorArgs['initRandom'] = initRandom
                self._selectorArgs['randomizeAxisFactor'] = randomizeAxisFactor
            self._selectorArgs['maxIterations'] = maxIterations
            self._selectorArgs['timeout'] = timeout
            self._selectorArgs['threshold'] = threshold
        self._validatorArgs = {}
        if isinstance(validation, float):
            validation, proportion = "proportion", validation
        elif not isinstance(validation, str):
            validation = "" # set to a value that will fail
        else:
            validation = validation.lower().replace(" ", "")
            validation = validation.lower().replace("-", "")
        if selection in dataOnly and validation != 'data':
            msg = f"'data' validation is required. '{selection}' "
            msg += "selection cannot use any training data for validation."
            raise InvalidArgumentValueCombination(msg)
        if validation not in Tuning._validators:
            msg = prettyListString(Tuning._validations)
            msg += ", or a float representing a holdout proportion of "
            msg += "the training data are the only accepted _validations."
            raise InvalidArgumentValue(msg)
        self._validation = validation
        self._validator = Tuning._validators[validation]

        if validation == 'crossvalidation':
            self._validatorArgs['folds'] = folds
        elif validation == 'leaveonegroupout':
            self._validatorArgs['foldFeature'] = foldFeature
        elif validation == 'data':
            self._validatorArgs['validateX'] = validateX
            self._validatorArgs['validateY'] = validateY
            if selection == 'iterative':
                self._validatorArgs['allowIncremental'] = True
        elif validation == 'proportion':
            self._validatorArgs['proportion'] = proportion

        # to set during tune
        self.validator = None
        self.arguments = None
        self._sortedResults = None

    def tune(self, learnerName, trainX, trainY, arguments, performanceFunction,
             randomSeed, useLog):
        """
        Run validation on each argument set.

        Prepare the Validator and ArgumentSelector objects based on the
        instantiation parameters and apply them with the provided
        parameters.
        """
        self.arguments = arguments
        if self.performanceFunction is None and performanceFunction is None:
            msg = "A performanceFunction is required when tuning parameters"
            raise InvalidArgumentValue(msg)
        if self.performanceFunction is None:
            self.performanceFunction = performanceFunction
        self.validator = self._validator(
            learnerName, trainX, trainY, self.performanceFunction,
            randomSeed=randomSeed, useLog=useLog, **self._validatorArgs)
        selector = self._selector(arguments, self.validator,
                                  **self._selectorArgs)
        selector.validateAll()

        # reverse based on optimal for performanceFunction
        highestIsBest = self.validator.optimal == 'max'
        toZip = [self.validator._results, self.validator._arguments]
        if hasattr(self.validator, '_deepResults'):
            toZip.append(self.validator._deepResults)
        self._sortedResults = sorted(zip(*toZip), key=itemgetter(0),
            reverse=highestIsBest)

        handleLogging(useLog, 'tuning', selector, self.validator)

    @property
    def bestResult(self):
        """
        The score of the best performance.
        """
        return self.validator._best[0]

    @property
    def bestArguments(self):
        """
        The arguments that provided the best performance.
        """
        return self.validator._best[1]

    @property
    def allResults(self):
        """
        Get the results of each validation that has been run.

        The results will be sorted from best to worst based on the
        optimal value of the performance function.
        """
        return [r[0] for r in self._sortedResults]

    @property
    def allArguments(self):
        """
        Get the argument set for each validation that has been run.

        The arguments will be sorted from best to worst based on the
        optimal value of the performance function.
        """
        return [r[1] for r in self._sortedResults]

    @property
    def deepResults(self):
        """
        If a cross-validation was used, get the fold-by fold results.

        For cross-validation, a list of lists is returned with each
        internal list containing the fold-by-fold performance for each
        argument set. These deep results will be sorted from best to
        worst based on the optimal value of the performance function. If
        a holdout protocol was used, this returns None.
        """
        try:
            return [r[2] for r in self._sortedResults]
        except IndexError:
            return None

    def copy(self):
        """
        A new Tuner with attributes based on the latest tuning.
        """
        ret = Tuning(selection=self.selection, validation=self.validation,
                     performanceFunction=self.performanceFunction,
                     **self._selectorArgs, **self._validatorArgs)
        ret.validator = self.validator
        ret.arguments = self.arguments
        ret._sortedResults = self._sortedResults

        return ret
