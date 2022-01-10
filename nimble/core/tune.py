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

import numpy as np

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction
from nimble.core.logger import handleLogging, loggingEnabled
from nimble.core.logger import deepLoggingEnabled
from nimble.random import pythonRandom, numpyRandom
from nimble.core._learnHelpers import computeMetrics
from nimble._utility import prettyDictString, prettyListString
from nimble._utility import mergeArguments


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

def _quoteStr(val):
    if isinstance(val, str):
        return '"{}"'.format(val)
    return str(val)

class Validator(ABC):
    """
    Base class for validation protocols.

    A Validator provides the opportunity to perform repeated validation
    of a learner on the same data with different arguments. Each time a
    validation is run, the results are stored within the object.
    """
    def __init__(self, learnerName, X, Y, performanceFunction, randomSeed,
                 **kwargs):
        if not hasattr(self, "name"):
            raise AttributeError("A Validator must have a name attribute")

        self.learnerName = learnerName
        if Y is None:
            msg = "Validation can only be performed for supervised learning. "
            msg += "Y data cannot be None"
            raise InvalidArgumentValue(msg)

        if isinstance(Y, (int, str, list)):
            X = X.copy()
            Y = X.features.extract(Y, useLog=False)

        if not len(X.points) == len(Y.points):
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
            self.randomSeed = nimble.random._generateSubsidiarySeed()
        else:
            self.randomSeed = randomSeed

        self._results = []
        self._arguments = []
        self._best = None
        # used in __str__ and __repr__
        self._keywords = kwargs

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        ret = '{}("{}", performanceFunction={}, randomSeed={}'.format(
            self.__class__.__name__, self.learnerName,
            self.performanceFunction.__name__, self.randomSeed)
        if self._keywords:
            ret += ", " + prettyDictString(self._keywords, valueStr=_quoteStr)
        ret += ")"
        return ret

    def validate(self, arguments=None, useLog=None, **kwarguments):
        """
        Apply the validation to a learner with the given arguments.
        """
        arguments = mergeArguments(arguments, kwarguments)
        performance = self._validate(arguments, useLog)
        self._results.append(performance)
        self._arguments.append(arguments)
        if self._best is None or self._isBest(performance, self._best[0]):
            self._best = (performance, arguments)

        return performance

    @abstractmethod
    def _validate(self, arguments, useLog):
        pass


class CrossValidator(Validator):
    """
    Base class for Validators that perform cross validation.

    Each sublass should provide a FoldIterator and its kwargs by calling
    super().__init__. The FoldIterators are instantiated here, rather
    than in the subclass, so that X and Y can be preprocessed first.
    """
    def __init__(self, foldIterator, learnerName, X, Y, performanceFunction,
                 randomSeed, **kwargs):
        super().__init__(learnerName, X, Y, performanceFunction, randomSeed,
                         **kwargs)
        # a result should be added here for every fold
        self._deepResults = []
        self._foldIterator = foldIterator([self.X, self.Y], **kwargs)
        # NOTE: no performance functions have this attribute
        self._canAvgFolds = (hasattr(self.performanceFunction, 'avgFolds')
                             and self.performanceFunction.avgFolds)

    def _validate(self, arguments, useLog):
        """
        Cross validate and return the overall performance.

        The per-fold performance is also recorded during validation.
        """
        # fold iterator randomized the point order, so if we are collecting all
        # the results, we also have to collect the correct order of the known
        # values
        if not self._canAvgFolds:
            collectedY = None

        performances = []
        foldByFold = []
        deepLog = loggingEnabled(useLog) and deepLoggingEnabled()
        numFolds = len(self._foldIterator.foldList)
        for foldNum, fold in enumerate(self._foldIterator):
            [(curTrainX, curTestingX), (curTrainY, curTestingY)] = fold
            #run algorithm on the folds' training and testing sets
            startTime = time.process_time()
            curRunResult = nimble.trainAndApply(
                self.learnerName, curTrainX, curTrainY, curTestingX,
                arguments=arguments, randomSeed=self.randomSeed, useLog=False)

            totalTime = time.process_time() - startTime

            # calculate error of prediction, using performanceFunction
            curPerformance = computeMetrics(curTestingY, None, curRunResult,
                                            self.performanceFunction)

            foldByFold.append(curPerformance)

            if self._canAvgFolds:
                performances.append(curPerformance)
            else:
                performances.append(curRunResult)
                if collectedY is None:
                    collectedY = curTestingY
                else:
                    collectedY.points.append(curTestingY, useLog=False)

            metrics = {self.performanceFunction.__name__: curPerformance}
            extraInfo = {'Fold': '{}/{}'.format(foldNum + 1, numFolds)}

            handleLogging(deepLog, "deepRun", self.__class__.__name__,
                          curTrainX, curTrainY, curTestingX, curTestingY,
                          self.learnerName, arguments, self.randomSeed,
                          metrics=metrics, extraInfo=extraInfo, time=totalTime)

        self._deepResults.append(foldByFold)

        # We consume the saved performances, either by averaging the individual
        # performances calculations for each fold, or combining the saved
        # predictions and calculating performance of the entire set.
        # average score from each fold (works for one fold as well)
        if self._canAvgFolds:
            finalPerformance = sum(performances) / float(len(performances))
        # combine the performances objects into one, and then calc performance
        else:
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
    """
    name = "cross validation"

    def __init__(self, learnerName, X, Y, performanceFunction, folds=5,
                 randomSeed=None):
        self.folds = folds
        super().__init__(KFoldIterator, learnerName, X, Y, performanceFunction,
                         randomSeed, folds=folds)


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
    """
    name = "leave one out"

    def __init__(self, learnerName, X, Y, performanceFunction,
                 randomSeed=None):
        super().__init__(KFoldIterator, learnerName, X, Y, performanceFunction,
                         randomSeed, folds=len(X.points))
        # set in super, but not a parameter
        del self._keywords['folds']


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
    """
    name = "leave one group out"

    def __init__(self, learnerName, X, Y, performanceFunction, foldFeature,
                 randomSeed=None):
        if foldFeature is None:
            msg = "foldFeature cannot be done when using leave one "
            msg += "group out validation"
            raise InvalidArgumentValue(msg)
        if isinstance(foldFeature, (str, int)):
            self.foldFeature = foldFeature
            X = X.copy()
            if isinstance(Y, (str, int)):
                if foldFeature == Y:
                    msg = "foldFeature and Y cannot be the same feature"
                    raise InvalidArgumentValueCombination(msg)
                removed = X.features.extract([Y, foldFeature], useLog=False)
                Y, foldFt = removed.features
            elif isinstance(Y, list):
                Y = X.features.extract(Y + [foldFeature], useLog=False)
                foldFt = Y.features.extract(-1, useLog=False)
            else:
                foldFt = X.features.extract(foldFeature, useLog=False)
        else:
            if len(foldFeature.points) != len(X.points):
                msg = "foldFeature must have the same number of points "
                msg += "as the X data"
                raise InvalidArgumentValue(msg)
            self.foldFeature = foldFeature.getTypeString()
            foldFt = foldFeature
        super().__init__(GroupFoldIterator, learnerName, X, Y,
                         performanceFunction, randomSeed,
                         foldFeature=foldFt)
        # use type string instead of object
        self._keywords['foldFeature'] = self.foldFeature


class HoldoutValidator(Validator):
    """
    Base class for holdout validation protocols.

    All subclasses provide the a validateX and validateY data objects.
    """
    def __init__(self, learnerName, X, Y, performanceFunction, randomSeed,
                 validateX, validateY, keywords):
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

        self._validateX = validateX
        self._validateY = validateY

        super().__init__(learnerName, X, Y, performanceFunction, randomSeed,
                         **keywords)

    def _validate(self, arguments, useLog):
        startTime = time.process_time()
        performance = nimble.trainAndTest(
            self.learnerName, self.X, self.Y, self._validateX, self._validateY,
            self.performanceFunction, arguments=arguments,
            randomSeed=self.randomSeed, useLog=False)

        totalTime = time.process_time() - startTime

        metrics = {self.performanceFunction.__name__: performance}
        deepLog = loggingEnabled(useLog) and deepLoggingEnabled()

        handleLogging(deepLog, "deepRun", self.__class__.__name__,
                      self.X, self.Y, self._validateX, self._validateY,
                      self.learnerName, arguments, self.randomSeed,
                      metrics=metrics, extraInfo=self._keywords,
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
    """
    name = "data holdout"

    def __init__(self, learnerName, X, Y, performanceFunction, validateX,
                 validateY, randomSeed=None):
        self.validateX = validateX.getTypeString()
        try:
            self.validateY = Y.getTypeString()
        except AttributeError:
            self.validateY = Y

        super().__init__(learnerName, X, Y, performanceFunction, randomSeed,
                         validateX, validateY,
                         {'validateX': self.validateX,
                          'validateY': self.validateY})


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
    """
    name = "holdout proportion"

    def __init__(self, learnerName, X, Y, performanceFunction, proportion=0.2,
                 randomSeed=None):
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

        super().__init__(learnerName, X, Y, performanceFunction, randomSeed,
                         validateX, validateY, {"proportion": proportion})


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

    def __init__(self, arguments, **kwargs):
        if self.name is None:
            msg = "An ArgumentSelector must have a name attribute"
            raise AttributeError(msg)
        self.arguments = arguments
        # used for  __str__ and __repr__
        self._keywords = kwargs

    def __iter__(self):
        return self

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        ret = '{}({}'.format(self.__class__.__name__, self.arguments)
        if self._keywords:
            ret += ", " + prettyDictString(self._keywords, valueStr=_quoteStr)
        ret += ')'
        return ret

    @abstractmethod
    def __next__(self):
        pass

    def update(self, performance):
        """
        Adjust attributes using the performance of the last arguments.
        """


class BruteForce(ArgumentSelector):
    """
    Try every possible combination of arguments.

    Each argument combination is tested to determine the best argument
    set for the learner.

    Parameters
    ----------
    arguments : dict
        Mapping argument names (strings) to their values, to be used.
    """
    name = "brute force"

    def __init__(self, arguments):
        super().__init__(arguments)
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
    optimal : str
        Either 'max' or 'min'.
    loops : int
        The number of time to loop through the argument values being
        tuned.
    order : list
        A list of argument names indicating the order to tune the
        arguments.
    kwarguments
        Keyword arguments specified variables that are passed to the
        learner. These are combined with the ``arguments`` parameter.
        Multiple values for arguments can be provided by using a ``Tune``
        object (e.g., k=Tune([3, 5, 7])) to initiate hyperparameter
        tuning and return the learner trained on the best set of
        arguments. To provide an argument that is an object from the
        same package as the learner, use a ``nimble.Init`` object with
        the object name and its instantiation arguments (e.g.,
        kernel=nimble.Init('KernelGaussian', width=2.0)).
    """
    name = "consecutive"

    def __init__(self, arguments, optimal, loops=1, order=None):
        if loops < 1:
            msg = "loops must be greater than 1"
            raise InvalidArgumentValue(msg)
        kwargs = {'loops': loops}
        if order is not None:
            kwargs['order'] = order
        super().__init__(arguments, **kwargs)
        self.optimal = optimal
        self.loops = loops
        self.order = order
        self._currentLoop = 1
        self._tune = {}
        self._target = 0
        self._bestPerformance = None
        self._bestArgs = None
        self._isBest = operator.gt if optimal == 'max' else operator.lt
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
                        msg = "{} is not in the order list".format(key)
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

        return self._currArgs.copy()

    def update(self, performance):
        if (self._bestPerformance is None
                or self._isBest(performance, self._bestPerformance)):
            self._bestPerformance = performance
            self._bestArgs = self._currArgs.copy()

# TODO
class Bayesian(ArgumentSelector):
    """
    Apply a bayesian algorithm to select the argument sets.
    """
    name = "bayesian"


class Iterative(ArgumentSelector):
    """
    Select next arguments by testing small changes to the current set.
    """
    name = "iterative"


class Tune:
    """
    Triggers hyperparameter optimization to occur during training.

    Any arguments in a the list of arguments provided to this object
    will be placed in the hyperparameter optimization protocol
    (defined by a ``Tuning`` object) and the best argument will be used
    to train the learner.

    Parameters
    ----------
    argumentList : list
        A list of values for the argument.

    See Also
    --------
    Tuning, nimble.train

    Keywords
    --------
    cross-validation, parameter, argument, hyperparameters, tuning,
    optimization, cross validate, learn, hyper parameters,
    hyperparameters, choose, grid search, GridSearchCV
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
        raise ImproperObjectAction("Tune objects are immutable")

    def __len__(self):
        return len(self.argumentTuple)

    def __str__(self):
        return str(self.argumentTuple)

    def __repr__(self):
        return "Tune(" + str(list(self.argumentTuple)) + ")"


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
        How the next argument set will be chosen. Accepts "brute force"
        and "consecutive". Note that when optimizing for a single
        argument, "brute force" and "consecutive" are effectively the
        same.
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
    _selections = ["brute force", "consecutive"]
    _validations = ["cross validation", "leave one out", "leave one group out",
                    "data", "proportion"]
    _selectors = {"bruteforce": BruteForce,
                  "consecutive": Consecutive}
    _validators = {"crossvalidation": KFold,
                   "leaveoneout": LeaveOneOut,
                   "leaveonegroupout": LeaveOneGroupOut,
                   "data": HoldoutData,
                   "proportion": HoldoutProportion}
    def __init__(self, selection="consecutive", validation="cross validation",
                 performanceFunction=None, loops=1, order=None, folds=5,
                 foldFeature=None, validateX=None, validateY=None,
                 proportion=None):
        self.selection = selection
        self.validation = validation
        self.performanceFunction = performanceFunction

        # Arguments are validated by the objects they are passed to not here
        self._selectorArgs = {}
        selection = selection.lower().replace(" ", "")
        selection = selection.lower().replace("-", "")
        if selection not in Tuning._selectors.keys():
            msg = prettyListString(Tuning._selections, useAnd=True)
            msg += " are the only accepted _selections."
            raise InvalidArgumentValue(msg)
        self._selection = selection
        self._selector = Tuning._selectors[selection]
        if selection == "consecutive":
            self._selectorArgs['loops'] = loops
            self._selectorArgs['order'] = order
        self._validatorArgs = {}
        if isinstance(validation, float):
            validation, proportion = "proportion", validation
        elif not isinstance(validation, str):
            validation = "" # set to a value that will fail
        else:
            validation = validation.lower().replace(" ", "")
            validation = validation.lower().replace("-", "")
        if validation not in Tuning._validators.keys():
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
        # detectBestResult will raise exception for invalid performanceFunction
        optimal = nimble.calculate.detectBestResult(self.performanceFunction)
        self.validator = self._validator(
            learnerName, trainX, trainY, self.performanceFunction,
            randomSeed=randomSeed, **self._validatorArgs)
        if self._selection != 'bruteforce':
            selector = self._selector(arguments, optimal=optimal,
                                      **self._selectorArgs)
        else:
            selector = self._selector(arguments, **self._selectorArgs)
        for argSet in selector:
            performance = self.validator.validate(argSet, useLog)
            selector.update(performance)

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
