"""
Axis generic methods for Base points and features attributes.

Backend methods and helpers responsible for determining how each
function will operate depending on whether it is being called along the
points or the features axis. This is the top level of this hierarchy and
methods in this object should attempt to handle operations related to
axis names here whenever possible. Additionally, any functionality
generic to axis and object subtype should be included here with abstract
methods defined for axis and object subtype specific implementations.
"""

import copy
from abc import ABC, abstractmethod
import inspect
import sys
from operator import itemgetter
import functools
import re

import numpy

import nimble
from nimble import fill
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import ImproperObjectAction
from nimble.exceptions import InvalidArgumentTypeCombination
from nimble.exceptions import InvalidArgumentValueCombination
from nimble._utility import isAllowedSingleElement, validateAllAllowedElements
from nimble.core.logger import handleLogging
from nimble._utility import inspectArguments
from .points import Points
from .features import Features
from ._dataHelpers import DEFAULT_PREFIX, DEFAULT_PREFIX2
from ._dataHelpers import DEFAULT_PREFIX_LENGTH, isDefaultName
from ._dataHelpers import valuesToPythonList, constructIndicesList
from ._dataHelpers import validateInputString
from ._dataHelpers import operatorDict
from ._dataHelpers import isAllowedSingleElement
from ._dataHelpers import createDataNoValidation
from ._dataHelpers import wrapMatchFunctionFactory
from ._dataHelpers import validateAxisFunction
from ._dataHelpers import inconsistentNames
from ._dataHelpers import pyplotRequired, plotOutput, plotFigureHandling
from ._dataHelpers import plotAxisLabels, plotXTickLabels
from ._dataHelpers import plotConfidenceIntervalMeanAndError, plotErrorBars
from ._dataHelpers import plotSingleBarChart, plotMultiBarChart

class Axis(ABC):
    """
    Differentiate how methods act dependent on the axis.

    This class provides backend methods for methods in Points and
    Features that share a common implementation along both axes and
    across data-types. Additionally, methods which require data-type
    specific implementations are defined here as abstract methods.

    Parameters
    ----------
    base : Base
        The Base instance that will be queried or modified.
    """
    def __init__(self, base, **kwargs):
        self._base = base
        kwargs['base'] = base
        if isinstance(self, Points):
            self._axis = 'point'
            self._isPoint = True
        else:
            self._axis = 'feature'
            self._isPoint = False
        super().__init__(**kwargs)

    def __len__(self):
        if self._isPoint:
            return self._base._pointCount

        return self._base._featureCount

    def __bool__(self):
        return len(self) > 0

    def _iter(self):
        return AxisIterator(self)

    ########################
    # Low Level Operations #
    ########################

    def _getName(self, index):
        if not self._namesCreated():
            self._setAllDefault()
        if self._isPoint:
            return self._base.pointNamesInverse[index]

        return self._base.featureNamesInverse[index]

    def _getNames(self):
        if not self._namesCreated():
            self._setAllDefault()
        if self._isPoint:
            namesList = self._base.pointNamesInverse
        else:
            namesList = self._base.featureNamesInverse

        return copy.copy(namesList)


    def _setName(self, oldIdentifier, newName, useLog=None):
        if self._isPoint:
            namesDict = self._base.pointNames
        else:
            namesDict = self._base.featureNames
        if len(self) == 0:
            msg = "Cannot set any {0} names; this object has no {0}s"
            msg = msg.format(self._axis)
            raise ImproperObjectAction(msg)
        if namesDict is None:
            self._setAllDefault()
        self._setName_implementation(oldIdentifier, newName)

        handleLogging(useLog, 'prep', '{ax}s.setName'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('setName'),
                      oldIdentifier, newName)


    def _setNames(self, assignments, useLog=None):
        if self._isPoint:
            names = 'pointNames'
            namesInverse = 'pointNamesInverse'
        else:
            names = 'featureNames'
            namesInverse = 'featureNamesInverse'
        if assignments is None:
            setattr(self._base, names, None)
            setattr(self._base, namesInverse, None)
        else:
            if not isinstance(assignments, dict):
                assignments = valuesToPythonList(assignments, 'assignments')
            self._setNamesBackend(assignments)

        handleLogging(useLog, 'prep', '{ax}s.setNames'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('setNames'),
                      assignments)

    def _getIndex(self, identifier, allowFloats=False):
        num = len(self)
        if num == 0:
            msg = "There are no valid " + self._axis + " identifiers; "
            msg += "this object has 0 " + self._axis + "s"
            raise IndexError(msg)
        if isinstance(identifier, (int, numpy.integer)):
            if identifier < 0:
                identifier = num + identifier
            if identifier < 0 or identifier >= num:
                msg = "The given index " + str(identifier) + " is outside of "
                msg += "the range of possible indices in the " + self._axis
                msg += " axis (0 to " + str(num - 1) + ")."
                raise IndexError(msg)
        elif isinstance(identifier, str):
            identifier = self._getIndexByName(identifier)
        elif allowFloats and isinstance(identifier, (float, numpy.float)):
            if identifier % 1: # x!=int(x)
                idVal = str(identifier)
                msg = "A float valued key of value x is only accepted if x == "
                msg += "int(x). The given value was " + idVal + " yet int("
                msg += idVal + ") = " + str(int(identifier))
                raise KeyError(msg)
            identifier = int(identifier)
        else:
            msg = "The identifier must be either a string (a valid "
            msg += self._axis + " name) or an integer (python or numpy) index "
            msg += "between 0 and " + str(num - 1) + " inclusive. "
            msg += "Instead we got: " + str(identifier)
            raise InvalidArgumentType(msg)
        return identifier

    def _getIndices(self, names):
        if not self._namesCreated():
            self._setAllDefault()
        if self._isPoint:
            namesDict = self._base.pointNames
        else:
            namesDict = self._base.featureNames

        return [namesDict[n] for n in names]

    def _hasName(self, name):
        try:
            self._getIndex(name)
            return True
        except KeyError:
            return False

    def _getitem(self, key):
        singleKey = isinstance(key, (int, float, str, numpy.integer))
        if singleKey:
            key = [self._getIndex(key, allowFloats=True)]
        else:
            key = self._processMultiple(key)
        if key is None:
            return self._base.copy()

        if singleKey and len(self._base._shape) > 2:
            return self._base.pointView(key[0]).copy()
        return self._structuralBackend_implementation('copy', key)

    def _anyDefaultNames(self):
        if self._namesCreated():
            return any(isDefaultName(name) for name in self._getNames())

        return True

    def _allDefaultNames(self):
        if self._namesCreated():
            return all(isDefaultName(name) for name in self._getNames())

        return True

    #########################
    # Structural Operations #
    #########################
    def _copy(self, toCopy, start, end, number, randomize, useLog=None):
        ret = self._genericStructuralFrontend('copy', toCopy, start, end,
                                              number, randomize)
        if self._isPoint:
            ret.features.setNames(self._base.features._getNamesNoGeneration(),
                                  useLog=False)
        else:
            ret.points.setNames(self._base.points._getNamesNoGeneration(),
                                useLog=False)

        ret._absPath = self._base.absolutePath
        ret._relPath = self._base.relativePath

        handleLogging(useLog, 'prep', '{ax}s.copy'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('copy'),
                      toCopy, start, end, number, randomize)

        return ret


    def _extract(self, toExtract, start, end, number, randomize, useLog=None):
        ret = self._genericStructuralFrontend('extract', toExtract, start, end,
                                              number, randomize)

        self._adjustCountAndNames(ret)

        ret._relPath = self._base.relativePath
        ret._absPath = self._base.absolutePath

        handleLogging(useLog, 'prep', '{ax}s.extract'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('extract'),
                      toExtract, start, end, number, randomize)

        return ret


    def _delete(self, toDelete, start, end, number, randomize, useLog=None):
        ret = self._genericStructuralFrontend('delete', toDelete, start, end,
                                              number, randomize)
        self._adjustCountAndNames(ret)

        handleLogging(useLog, 'prep', '{ax}s.delete'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('delete'),
                      toDelete, start, end, number, randomize)


    def _retain(self, toRetain, start, end, number, randomize, useLog=None):
        ref = self._genericStructuralFrontend('retain', toRetain, start, end,
                                              number, randomize)

        ref._relPath = self._base.relativePath
        ref._absPath = self._base.absolutePath

        self._base._referenceDataFrom(ref)

        handleLogging(useLog, 'prep', '{ax}s.retain'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('retain'),
                      toRetain, start, end, number, randomize)


    def _count(self, condition):
        return self._genericStructuralFrontend('count', condition)


    def _sort(self, by, reverse, useLog=None):
        if by is None:
            self._sortByNames(reverse)
        # identifiers
        elif not callable(by):
            if self._isPoint:
                if len(self._base._shape) > 2:
                    msg = "For object with more than two-dimensions, sorting "
                    msg += "can only be performed on point names or using a "
                    msg += "function."
                    raise ImproperObjectAction(msg)
            self._sortByIdentifier(by, reverse)
        # functions
        elif isinstance(by, itemgetter):
            # extract the items and use the faster _sortByIdentifier
            indices = list(by.__reduce__()[1])
            self._sortByIdentifier(indices, reverse)
        else:
            try:
                args, _, _, _ = inspectArguments(by)
                if len(args) == 2: # comparator function
                    func = functools.cmp_to_key(by)
                elif not args or len(args) > 2:
                    msg = 'by must take one or two positional arguments'
                    raise InvalidArgumentValue(msg)
                else:
                    func = by
            except TypeError:
                # inspect fails when 'by' is already cmp_to_key
                func = by
            self._sortByFunction(func, reverse)

        handleLogging(useLog, 'prep', '{ax}s.sort'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('sort'),
                      by, reverse)


    def _permute(self, order=None, useLog=None):
        if order is None:
            values = len(self)
            order = list(range(values))
            nimble.random.pythonRandom.shuffle(order)
        else:
            order = constructIndicesList(self._base, self._axis, order,
                                         'order')
            if len(order) != len(self):
                msg = "This object contains {0} {1}s, "
                msg += "but order has {2} identifiers"
                msg = msg.format(len(self), self._axis, len(order))
                raise InvalidArgumentValue(msg)
            if len(order) != len(set(order)):
                msg = "This object contains {0} unique identifiers but "
                msg += "but order has {1} {2}s"
                msg = msg.format(len(self), len(set(order)), self._axis)
                raise InvalidArgumentValue(msg)

        # only one possible permutation
        if len(self) <= 1:
            return

        self._permute_implementation(order)

        if self._namesCreated():
            names = self._getNames()
            reorderedNames = [names[idx] for idx in order]
            self._setNames(reorderedNames, useLog=False)

        handleLogging(useLog, 'prep', '{ax}s.permute'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('permute'))


    def _transform(self, function, limitTo, useLog=None):
        if self._base._pointCount == 0:
            msg = "We disallow this function when there are 0 points"
            raise ImproperObjectAction(msg)
        if self._base._featureCount == 0:
            msg = "We disallow this function when there are 0 features"
            raise ImproperObjectAction(msg)

        if self._axis == 'point':
            allowedLength = len(self._base.features)
        else:
            allowedLength = len(self._base.points)
        wrappedFunc = validateAxisFunction(function, self._axis, allowedLength)
        if limitTo is not None:
            limitTo = constructIndicesList(self._base, self._axis, limitTo)

        self._transform_implementation(wrappedFunc, limitTo)

        handleLogging(useLog, 'prep', '{ax}s.transform'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('transform'),
                      function, limitTo)

    ###########################
    # Higher Order Operations #
    ###########################

    def _calculate(self, function, limitTo, useLog=None):
        wrappedFunc = validateAxisFunction(function, self._axis)
        ret = self._calculate_backend(wrappedFunc, limitTo)

        handleLogging(useLog, 'prep', '{ax}s.calculate'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('calculate'),
                      function, limitTo)
        return ret

    def _matching(self, function, useLog=None):
        wrappedMatch = wrapMatchFunctionFactory(function)

        ret = self._calculate_backend(wrappedMatch, None)

        self._setNames(self._getNamesNoGeneration(), useLog=False)
        if hasattr(function, '__name__') and function.__name__ != '<lambda>':
            if self._isPoint:
                ret.features.setNames([function.__name__], useLog=False)
            else:
                ret.points.setNames([function.__name__], useLog=False)

        handleLogging(useLog, 'prep', '{ax}s.matching'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('matching'),
                      function)
        return ret

    def _calculate_backend(self, function, limitTo):
        if len(self._base.points) == 0:
            msg = "We disallow this function when there are 0 points"
            raise ImproperObjectAction(msg)
        if len(self._base.features) == 0:
            msg = "We disallow this function when there are 0 features"
            raise ImproperObjectAction(msg)

        if limitTo is not None:
            limitTo = constructIndicesList(self._base, self._axis, limitTo)
        else:
            limitTo = list(range(len(self)))

        retData, offAxisNames = self._calculate_implementation(function,
                                                               limitTo)

        ret = nimble.data(self._base.getTypeString(), retData, useLog=False)

        ret._absPath = self._base.absolutePath
        ret._relPath = self._base.relativePath

        if self._isPoint:
            axisNameSetter = ret.points.setNames
            offAxisNameSetter = ret.features.setNames
        else:
            ret.transpose(useLog=False)
            axisNameSetter = ret.features.setNames
            offAxisNameSetter = ret.points.setNames
        if len(limitTo) < len(self) and self._namesCreated():
            names = []
            for index in limitTo:
                names.append(self._getName(index))
            axisNameSetter(names, useLog=False)
        elif self._namesCreated():
            axisNameSetter(self._getNames(), useLog=False)
        if offAxisNames is not None:
            offAxisNameSetter(offAxisNames, useLog=False)

        return ret

    def _calculate_implementation(self, function, limitTo):
        retData = []
        # signal to convert to object elementType if function is returning
        # non-numeric values.
        if self._isPoint:
            viewer = self._base.pointView
        else:
            viewer = self._base.featureView

        offAxisNames = None
        for i, axisID in enumerate(limitTo):
            view = viewer(axisID)
            currOut = function(view)
            # the output could have multiple values or be singular.
            if isAllowedSingleElement(currOut):
                currOut = [currOut]
            elif (isinstance(currOut, nimble.core.data.Base)
                  and len(currOut._shape) == 2):
                # make 2D if a vector and axis does not match vector direction
                axisIdx = 0 if self._isPoint else 1
                if 1 in currOut.shape and currOut.shape[axisIdx] != 1:
                    currOut = currOut.copy('python list')
                elif i == 0 and self._isPoint:
                    offAxisNames = currOut.features._getNamesNoGeneration()
                elif i == 0:
                    offAxisNames = currOut.points._getNamesNoGeneration()
            # only point axis can handle multidimensional data
            if not self._isPoint:
                try:
                    validateAllAllowedElements(currOut)
                except ImproperObjectAction as e:
                    msg = "function must return a one-dimensional object "
                    raise ImproperObjectAction(msg) from e
                except TypeError as e:
                    msg = "function must return a valid single element or "
                    msg += "an iterable, but got type " + str(type(currOut))
                    raise InvalidArgumentValue(msg) from e
            retData.append(currOut)

        return retData, offAxisNames


    def _insert(self, insertBefore, toInsert, append=False, useLog=None):
        if not append and insertBefore is None:
            msg = "insertBefore must be an index in range 0 to "
            msg += "{l} or {ax} name".format(l=len(self), ax=self._axis)
            raise InvalidArgumentType(msg)
        if insertBefore is None:
            insertBefore = len(self)
        elif insertBefore != len(self) or len(self) == 0:
            insertBefore = self._getIndex(insertBefore)

        self._validateInsertableData(toInsert, append)
        if self._base.getTypeString() != toInsert.getTypeString():
            toInsert = toInsert.copy(to=self._base.getTypeString())

        offAxis = 'feature' if self._axis == 'point' else 'point'
        toInsert = self._alignNames(offAxis, toInsert)
        self._insert_implementation(insertBefore, toInsert)

        self._setInsertedCountAndNames(toInsert, insertBefore)

        if append:
            handleLogging(useLog, 'prep',
                          '{ax}s.append'.format(ax=self._axis),
                          self._base.getTypeString(), self._sigFunc('append'),
                          toInsert)
        else:
            handleLogging(useLog, 'prep',
                          '{ax}s.insert'.format(ax=self._axis),
                          self._base.getTypeString(), self._sigFunc('insert'),
                          insertBefore, toInsert)


    def _mapReduce(self, mapper, reducer, useLog=None):
        if self._isPoint:
            targetCount = len(self._base.points)
            otherCount = len(self._base.features)
            otherAxis = 'feature'
            viewIter = self._base.points
        else:
            targetCount = len(self._base.features)
            otherCount = len(self._base.points)
            otherAxis = 'point'
            viewIter = self._base.features

        if otherCount == 0:
            msg = "We do not allow operations over {0}s if there are 0 {1}s"
            msg = msg.format(self._axis, otherAxis)
            raise ImproperObjectAction(msg)

        if mapper is None or reducer is None:
            raise InvalidArgumentType("The arguments must not be None")
        if not hasattr(mapper, '__call__'):
            raise InvalidArgumentType("The mapper must be callable")
        if not hasattr(reducer, '__call__'):
            raise InvalidArgumentType("The reducer must be callable")

        if targetCount == 0:
            ret = nimble.data(self._base.getTypeString(),
                              numpy.empty(shape=(0, 0)), useLog=False)
        else:
            mapResults = {}
            # apply the mapper to each point in the data
            for value in viewIter:
                currResults = mapper(value)
                # the mapper will return a list of key value pairs
                for (k, v) in currResults:
                    # if key is new, we must add an empty list
                    if k not in mapResults:
                        mapResults[k] = []
                    # append value to list of values associated with the key
                    mapResults[k].append(v)

            # apply the reducer to the list of values associated with each key
            ret = []
            for mapKey in mapResults:
                mapValues = mapResults[mapKey]
                # the reducer will return a tuple of a key to a value
                redRet = reducer(mapKey, mapValues)
                if redRet is not None:
                    (redKey, redValue) = redRet
                    ret.append([redKey, redValue])
            ret = nimble.data(self._base.getTypeString(), ret, useLog=False)

        ret._absPath = self._base.absolutePath
        ret._relPath = self._base.relativePath

        handleLogging(useLog, 'prep', '{ax}s.mapReduce'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('mapReduce'),
                      mapper, reducer)

        return ret


    def _fillMatching(self, fillWith, matchingElements, limitTo=None,
                      useLog=None, **kwarguments):
        removeKwargs = []
        # use our fill.constant function with the fillWith value
        if not callable(fillWith):
            value = fillWith
            # for consistency use numpy.nan for None and nans
            if value is None or value != value:
                value = numpy.nan
            fillFunc = fill.constant
            kwarguments['constantValue'] = value
            removeKwargs.append('constantValue')
        else:
            fillFunc = fillWith

        # if matchingElements is a boolean matrix we need to provide the
        # corresponding boolean vector for the vector passed to _transform
        try:
            idxOrder = iter(limitTo)
        except TypeError:
            if limitTo is None:
                idxOrder = iter(range(len(self)))
            else: # limitTo is a single identifier
                idxOrder = iter([self._getIndex(limitTo)])

        @functools.wraps(fillFunc)
        def fillFunction(vector):
            isBase = isinstance(matchingElements, nimble.core.data.Base)
            if not isBase:
                matcher = matchingElements
            elif self._axis == 'point':
                matcher = matchingElements[next(idxOrder), :]
            else:
                matcher = matchingElements[:, next(idxOrder)]

            return fillFunc(vector, matcher, **kwarguments)

        self._transform(fillFunction, limitTo, useLog=False)

        # prevent kwargs we added from being logged
        for kwarg in removeKwargs:
            del kwarguments[kwarg]

        funcName = '{ax}s.fillMatching'.format(ax=self._axis)
        handleLogging(useLog, 'prep', funcName, self._base.getTypeString(),
                      self._sigFunc('fillMatching'), fillWith,
                      matchingElements, limitTo, **kwarguments)


    def _repeat(self, totalCopies, copyVectorByVector):
        if not isinstance(totalCopies, (int, numpy.int)) or totalCopies < 1:
            raise InvalidArgumentType("totalCopies must be a positive integer")
        if totalCopies == 1:
            return self._base.copy()

        repeated = self._repeat_implementation(totalCopies, copyVectorByVector)

        if self._isPoint:
            ptNames = self._getNamesNoGeneration()
            namesToRepeat = ptNames
            ftNames = self._base.features._getNamesNoGeneration()
        else:
            ftNames = self._getNamesNoGeneration()
            namesToRepeat = ftNames
            ptNames = self._base.points._getNamesNoGeneration()

        if copyVectorByVector and namesToRepeat is not None:
            origNames = namesToRepeat.copy()
            for idx, name in enumerate(origNames):
                for i in range(totalCopies):
                    currIdx = (totalCopies * idx) + i
                    if currIdx < len(origNames):
                        namesToRepeat[currIdx] = name + "_" + str(i + 1)
                    else:
                        namesToRepeat.append(name + "_" + str(i + 1))
        elif namesToRepeat is not None:
            origNames = namesToRepeat.copy()
            for i in range(totalCopies):
                for idx, name in enumerate(origNames):
                    if i == 0:
                        namesToRepeat[idx] = name + "_" + str(i + 1)
                    else:
                        namesToRepeat.append(name + "_" + str(i + 1))

        ret = createDataNoValidation(self._base.getTypeString(), repeated,
                                     pointNames=ptNames, featureNames=ftNames)
        if self._isPoint:
            ret._shape[1:] = self._base._shape[1:]
        return ret

    ###################
    # Query functions #
    ###################

    def _unique(self):
        ret = self._unique_implementation()
        if self._isPoint:
            ret._shape[1:] = self._base._shape[1:]
        ret._absPath = self._base.absolutePath
        ret._relPath = self._base.relativePath

        return ret


    def _similarities(self, similarityFunction):
        accepted = [
            'correlation', 'covariance', 'dot product', 'sample covariance',
            'population covariance'
            ]
        cleanFuncName = validateInputString(similarityFunction, accepted,
                                            'similarities')

        if cleanFuncName == 'correlation':
            toCall = nimble.calculate.correlation
        elif cleanFuncName in ['covariance', 'samplecovariance']:
            toCall = nimble.calculate.covariance
        elif cleanFuncName == 'populationcovariance':
            # pylint: disable=invalid-name
            def populationCovariance(X, X_T):
                return nimble.calculate.covariance(X, X_T, False)

            toCall = populationCovariance
        elif cleanFuncName == 'dotproduct':
            # pylint: disable=invalid-name
            def dotProd(X, X_T):
                return X.matrixMultiply(X_T)

            toCall = dotProd

        transposed = self._base.T

        if self._isPoint:
            ret = toCall(self._base, transposed)
        else:
            ret = toCall(transposed, self._base)

        # TODO validation or result.

        ret._absPath = self._base.absolutePath
        ret._relPath = self._base.relativePath

        return ret

    def _statistics(self, statisticsFunction, groupByFeature=None):
        accepted = [
            'max', 'mean', 'median', 'min', 'unique count',
            'proportion missing', 'proportion zero', 'standard deviation',
            'std', 'population std', 'population standard deviation',
            'sample std', 'sample standard deviation'
            ]
        cleanFuncName = validateInputString(statisticsFunction, accepted,
                                            'statistics')

        if cleanFuncName == 'max':
            toCall = nimble.calculate.maximum
        elif cleanFuncName == 'mean':
            toCall = nimble.calculate.mean
        elif cleanFuncName == 'median':
            toCall = nimble.calculate.median
        elif cleanFuncName == 'min':
            toCall = nimble.calculate.minimum
        elif cleanFuncName == 'uniquecount':
            toCall = nimble.calculate.uniqueCount
        elif cleanFuncName == 'proportionmissing':
            toCall = nimble.calculate.proportionMissing
        elif cleanFuncName == 'proportionzero':
            toCall = nimble.calculate.proportionZero
        elif cleanFuncName in ['std', 'standarddeviation', 'samplestd',
                               'samplestandarddeviation']:
            toCall = nimble.calculate.standardDeviation
        elif cleanFuncName in ['populationstd', 'populationstandarddeviation']:

            def populationStandardDeviation(values):
                return nimble.calculate.standardDeviation(values, False)

            toCall = populationStandardDeviation

        if self._axis == 'point' or groupByFeature is None:
            return self._statisticsBackend(cleanFuncName, toCall)
        # groupByFeature is only a parameter for .features
        res = self._base.groupByFeature(groupByFeature, useLog=False)
        for k in res:
            res[k] = res[k].features._statisticsBackend(cleanFuncName,
                                                        toCall)
        return res

    def _statisticsBackend(self, cleanFuncName, toCall):
        ret = self._calculate(toCall, limitTo=None, useLog=False)
        if self._isPoint:
            ret.points.setNames(self._getNames(), useLog=False)
            ret.features.setName(0, cleanFuncName, useLog=False)
        else:
            ret.points.setName(0, cleanFuncName, useLog=False)
            ret.features.setNames(self._getNames(), useLog=False)

        return ret

    ############
    # Plotting #
    ############

    @pyplotRequired
    def _plotComparison(
            self, statistic, identifiers, confidenceIntervals, horizontal,
            outPath, show, figureName, title, xAxisLabel, yAxisLabel,
            legendTitle, **kwargs):
        fig, ax = plotFigureHandling(figureName)
        if identifiers is None:
            identifiers = list(range(len(self)))
        axisRange = range(1, len(identifiers) + 1)
        target = self[identifiers]
        if self._isPoint:
            targetAxis = target.points
        else:
            targetAxis = target.features
        names = [self._base._formattedStringID(self._axis, identity)
                 for identity in identifiers]
        if hasattr(statistic, '__name__') and statistic.__name__ != '<lambda>':
            statName = statistic.__name__
        else:
            statName = ''
        if confidenceIntervals:
            means = []
            errors = []
            for vec in targetAxis:
                mean, error = plotConfidenceIntervalMeanAndError(vec)
                means.append(mean)
                errors.append(error)

            plotErrorBars(ax, axisRange, means, errors, horizontal, **kwargs)

            if title is True:
                title = "95% Confidence Intervals for Feature Means"
        else:
            if statistic is None:
                calc = target
            else:
                calc = targetAxis.calculate(statistic, useLog=False)
            if self._isPoint:
                calcAxis = calc.features
            else:
                calcAxis = calc.points
            if len(calcAxis) == 1:
                plotSingleBarChart(ax, axisRange, calc, horizontal, **kwargs)
            else:
                heights = {}
                for i, pt in enumerate(calcAxis):
                    name = calc._formattedStringID(calcAxis._axis, i)
                    heights[name] = pt

                plotMultiBarChart(ax, heights, horizontal, legendTitle,
                                  **kwargs)

            if title is True:
                title = ''
                if self._base.name is not None:
                    title += "{}: ".format(self._base.name)
                title += "Feature Comparison"

        if title is False:
            title = None
        ax.set_title(title)
        if horizontal:
            ax.set_yticks(axisRange)
            ax.set_yticklabels(names)
            yAxisDefault = self._axis.capitalize()
            xAxisDefault = statName
        else:
            ax.set_xticks(axisRange)
            plotXTickLabels(ax, fig, names, len(identifiers))
            xAxisDefault = self._axis.capitalize()
            yAxisDefault = statName

        plotAxisLabels(ax, xAxisLabel, xAxisDefault, yAxisLabel, yAxisDefault)

        plotOutput(outPath, show)

    #####################
    # Low Level Helpers #
    #####################

    def _namesCreated(self):
        if self._isPoint:
            return not self._base.pointNames is None

        return not self._base.featureNames is None

    def _nextDefaultName(self):
        if self._isPoint:
            ret = DEFAULT_PREFIX2%self._base._nextDefaultValuePoint
            self._base._nextDefaultValuePoint += 1
        else:
            ret = DEFAULT_PREFIX2%self._base._nextDefaultValueFeature
            self._base._nextDefaultValueFeature += 1
        return ret

    def _setAllDefault(self):
        if self._isPoint:
            self._base.pointNames = {}
            self._base.pointNamesInverse = []
            names = self._base.pointNames
            invNames = self._base.pointNamesInverse
        else:
            self._base.featureNames = {}
            self._base.featureNamesInverse = []
            names = self._base.featureNames
            invNames = self._base.featureNamesInverse
        for i in range(len(self)):
            defaultName = self._nextDefaultName()
            invNames.append(defaultName)
            names[defaultName] = i

    def _getNamesNoGeneration(self):
        if not self._namesCreated():
            return None
        return self._getNames()

    def _getIndexByName(self, name):
        if not self._namesCreated():
            self._setAllDefault()
        if self._isPoint:
            namesDict = self._base.pointNames
        else:
            namesDict = self._base.featureNames

        if name not in namesDict:
            msg = "The " + self._axis + " name '" + name
            msg += "' cannot be found."
            raise KeyError(msg)
        return namesDict[name]

    def _setName_implementation(self, oldIdentifier, newName):
        if self._isPoint:
            names = self._base.pointNames
            invNames = self._base.pointNamesInverse
        else:
            names = self._base.featureNames
            invNames = self._base.featureNamesInverse

        index = self._getIndex(oldIdentifier)
        if newName is not None:
            if not isinstance(newName, str):
                msg = "The new name must be either None or a string"
                raise InvalidArgumentType(msg)

        if newName in names:
            if invNames[index] == newName:
                return
            msg = "This name '" + newName + "' is already in use"
            raise InvalidArgumentValue(msg)

        if newName is None:
            newName = self._nextDefaultName()

        #remove the current featureName
        oldName = invNames[index]
        del names[oldName]

        # setup the new featureName
        invNames[index] = newName
        names[newName] = index
        self._base._incrementDefaultIfNeeded(newName, self._axis)

    def _setNamesBackend(self, assignments):
        count = len(self)
        if len(assignments) != count:
            msg = "assignments may only be an ordered container type, with as "
            msg += "many entries (" + str(len(assignments)) + ") as this axis "
            msg += "is long (" + str(count) + ")"
            raise InvalidArgumentValue(msg)
        if not isinstance(assignments, dict):
            #convert to dict so we only write the checking code once
            temp = {}
            for index, name in enumerate(assignments):
                # take this to mean fill it in with a default name
                if name is None:
                    name = self._nextDefaultName()
                if not isinstance(name, str):
                    msg = 'assignments must contain only string values'
                    raise InvalidArgumentValue(msg)
                if isDefaultName(name) and name in temp:
                    name = self._nextDefaultName()
                if name in temp:
                    msg = "Cannot input duplicate names: " + str(name)
                    raise InvalidArgumentValue(msg)
                temp[name] = index
            assignments = temp

        if count == 0:
            if self._isPoint:
                self._base.pointNames = {}
                self._base.pointNamesInverse = []
            else:
                self._base.featureNames = {}
                self._base.featureNamesInverse = []
            return

        # at this point, the input must be a dict
        #check input before performing any action
        for name in assignments.keys():
            if not None and not isinstance(name, str):
                raise InvalidArgumentValue("Names must be strings")
            if not isinstance(assignments[name], int):
                raise InvalidArgumentValue("Indices must be integers")
            if assignments[name] < 0 or assignments[name] >= count:
                if self._isPoint:
                    countName = 'points'
                else:
                    countName = 'features'
                msg = "Indices must be within 0 to "
                msg += "len(self." + countName + ") - 1"
                raise InvalidArgumentValue(msg)

        reverseMap = [None] * len(assignments)
        for name in assignments.keys():
            self._base._incrementDefaultIfNeeded(name, self._axis)
            reverseMap[assignments[name]] = name

        # have to copy the input, could be from another object
        if self._isPoint:
            self._base.pointNames = copy.deepcopy(assignments)
            self._base.pointNamesInverse = reverseMap
        else:
            self._base.featureNames = copy.deepcopy(assignments)
            self._base.featureNamesInverse = reverseMap

    def _processMultiple(self, key):
        """
        Helper for Base and Axis __getitem__ when given multiple values.

        If the input is a full slice, copying for __getitem__ can be
        ignored so None is returned. Otherwise the input will be
        transformed to a list.

        Returns
        -------
        list, None
        """
        length = len(self)
        if isinstance(key, slice):
            if key == slice(None): # full slice
                return None
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else length - 1
            step = key.step if key.step is not None else 1

            start = self._getIndex(start, allowFloats=True)
            stop = self._getIndex(stop, allowFloats=True)
            if start == 0 and stop == length - 1 and step == 1: # full slice
                return None
            # our stop is inclusive need to adjust for builtin range below
            if step > 0:
                stop += 1
            else:
                stop -= 1
            return list(range(start, stop, step))

        numBool = sum(isinstance(val, (bool, numpy.bool_)) for val in key)
        # contains all boolean values
        if numBool == length:
            return [i for i, v in enumerate(key) if v]
        if numBool > 0:
            msg = 'The key provided for {ax}s contains boolean values. '
            msg += 'Booleans are only permitted if the key contains only '
            msg += 'boolean type values for every {ax} in this object.'
            raise KeyError(msg.format(ax=self._axis))
        key = [self._getIndex(i, allowFloats=True) for i in key]
        if key == list(range(length)):  # full slice
            return None
        if len(set(key)) != len(key):
            duplicates = set(val for val in key if key.count(val) > 1)
            msg = 'Duplicate values in the key are not allowed. The '
            msg += 'following values were duplicated: {dup}. Duplicate '
            msg += '{ax}s can be generated using the repeat() method of '
            msg += "this object's {ax}s attribute"
            msg = msg.format(dup=duplicates, ax=self._axis)
            raise KeyError(msg)
        return key

    ########################
    #  Structural Helpers  #
    ########################

    def _axisQueryFunction(self, string):
        """
        Convert a query string to an axis input function.
        """
        offAxis = 'feature' if self._isPoint else 'point'
        # positive lookahead catches all ambiguous cases (i.e age == > 20)
        operatorCount = len(re.findall(r'(?=\s(==|!=|>=|>|<=|<)\s)', string))
        if not operatorCount:
            msg = "'{0}' is not a valid {1} name nor a valid query string. "
            msg += "A query string must be in the format {2}NAME OPERATOR "
            msg += "VALUE i.e '{1}3 != 0'"
            offCaps = offAxis.upper()
            raise InvalidArgumentValue(msg.format(string, offAxis, offCaps))
        if operatorCount > 1:
            msg = "Multiple operators in query string. Strings containing "
            msg += "more than one whitespace padded operator cannot be "
            msg += "parsed. Use a function instead or modify the {0}Name or "
            msg += "values that includes a whitespace padded operator."
            raise InvalidArgumentValue(msg.format(offAxis))

        name, optr, value = re.split(r'\s(==|!=|>=|>|<=|<)\s', string)
        name = name.strip()
        value = value.strip()

        hasName = getattr(self._base, offAxis + 's')._hasName
        if not hasName(name):
            msg = "the {0} '{1}' does not exist".format(offAxis, name)
            raise InvalidArgumentValue(msg)

        operatorFunc = operatorDict[optr]
        # convert value from a string, if possible
        try:
            value = float(value)
        except ValueError:
            pass
        #convert query string to a function
        def targetFunc(vector):
            return operatorFunc(vector[name], value)

        targetFunc.vectorized = True
        targetFunc.name = name
        targetFunc.value = value
        targetFunc.operator = operatorFunc

        return targetFunc

    def _genericStructuralFrontend(self, structure, target=None,
                                   start=None, end=None, number=None,
                                   randomize=False):
        axis = self._axis
        axisLength = len(self)

        _validateStructuralArguments(structure, axis, target, start,
                                     end, number, randomize)
        targetList = []
        argName = 'to' + structure.capitalize()
        if target is not None and isinstance(target, str):
            # check if target is a valid name
            if self._hasName(target):
                target = self._getIndex(target)
                targetList.append(target)
            # if not a name then assume it's a query string
            elif len(self._base._shape) > 2:
                msg = "query strings for {0} are not supported for data with "
                msg += "more than two dimensions"
                raise ImproperObjectAction(msg.format(argName))
            else:
                target = self._axisQueryFunction(target)

        # list-like container types
        if target is not None and not hasattr(target, '__call__'):
            targetList = constructIndicesList(self._base, axis, target,
                                              argName)
            if len(set(targetList)) != len(targetList):
                dup = set(v for v in targetList if targetList.count(v) > 1)
                msg = '{name} cannot contain duplicate values. The following '
                msg += 'were duplicated: {dup}. Duplicate {ax}s can be '
                msg += "generated using an object's {ax}s.repeat() method"
                msg = msg.format(name=argName, dup=dup, ax=self._axis)
                raise InvalidArgumentValue(msg)
        # boolean function
        elif target is not None:
            if len(self._base._shape) > 2:
                msg = "functions for {0} are not supported for data with more "
                msg += "than two dimensions"
                raise ImproperObjectAction(msg.format(argName))
            # construct list from function
            for targetID, view in enumerate(self):
                if target(view):
                    targetList.append(targetID)

        elif start is not None or end is not None:
            if start is None:
                start = 0
            else:
                start = self._getIndex(start)
            if end is None:
                end = axisLength - 1
            else:
                end = self._getIndex(end)
            _validateStartEndRange(start, end, axis, axisLength)

            # end + 1 because our range is inclusive
            targetList = list(range(start, end + 1))

        else:
            targetList = list(range(axisLength))

        if number:
            if number > len(targetList):
                msg = "The value for 'number' ({0}) ".format(number)
                msg += "is greater than the number of {0}s ".format(axis)
                msg += "to {0} ({1})".format(structure, len(targetList))
                raise InvalidArgumentValue(msg)
            if randomize:
                targetList = nimble.random.pythonRandom.sample(targetList,
                                                               number)
            else:
                targetList = targetList[:number]

        if structure == 'count':
            return len(targetList)
        ret = self._structuralBackend_implementation(structure, targetList)

        if self._isPoint and ret is not None:
            # retain internal dimensions
            ret._shape[1:] = self._base._shape[1:]

        return ret

    def _getStructuralNames(self, targetList):
        nameList = None
        if self._namesCreated():
            nameList = [self._getName(i) for i in targetList]
        if self._isPoint:
            return nameList, self._base.features._getNamesNoGeneration()

        return self._base.points._getNamesNoGeneration(), nameList

    def _adjustCountAndNames(self, other):
        """
        Adjust the count and names (when names have been generated) for
        this object, removing the names that have been extracted to the
        other object.
        """
        if self._isPoint:
            self._base._pointCount -= len(other.points)
            if self._base._pointNamesCreated():
                idxList = []
                for name in other.points.getNames():
                    idxList.append(self._base.pointNames[name])
                idxList = sorted(idxList)
                for i, val in enumerate(idxList):
                    del self._base.pointNamesInverse[val - i]
                self._base.pointNames = {}
                for idx, pt in enumerate(self._base.pointNamesInverse):
                    self._base.pointNames[pt] = idx

        else:
            self._base._featureCount -= len(other.features)
            if self._base._featureNamesCreated():
                idxList = []
                for name in other.features.getNames():
                    idxList.append(self._base.featureNames[name])
                idxList = sorted(idxList)
                for i, val in enumerate(idxList):
                    del self._base.featureNamesInverse[val - i]
                self._base.featureNames = {}
                for idx, ft in enumerate(self._base.featureNamesInverse):
                    self._base.featureNames[ft] = idx

    def _sortByNames(self, reverse):
        names = self._getNamesNoGeneration()
        if names is None or any(isDefaultName(n) for n in names):
            msg = "When by=None, all {0} names must be defined (non-default). "
            msg += "Either set the {0} names of this object or provide "
            msg += "another argument for by"
            raise InvalidArgumentValue(msg.format(self._axis))
        self._permute(sorted(self._getNames(), reverse=reverse),
                      useLog=False)

    def _sortByIdentifier(self, index, reverse):
        if isinstance(index, list):
            for idx in index[::-1]:
                self._sortByIdentifier(idx, reverse)
        else:
            if self._axis == 'point':
                data = self._base.features[index]
            else:
                data = self._base.points[index]
            sortedIndex = sorted(enumerate(data), key=itemgetter(1),
                                 reverse=reverse)
            self._permute((val[0] for val in sortedIndex), useLog=False)

    def _sortByFunction(self, func, reverse):
        sortedData = sorted(enumerate(self), key=lambda x: func(x[1]),
                            reverse=reverse)
        self._permute((val[0] for val in sortedData), useLog=False)

    ##########################
    #  Higher Order Helpers  #
    ##########################

    def _validateInsertableData(self, toInsert, append):
        """
        Required validation before inserting an object
        """
        if append:
            argName = 'toAppend'
            func = 'append'
        else:
            argName = 'toInsert'
            func = 'insert'
        if not isinstance(toInsert, nimble.core.data.Base):
            msg = "The argument '{arg}' must be an instance of the "
            msg += "nimble.core.data.Base class. The value we received was "
            msg += str(toInsert) + ", had the type " + str(type(toInsert))
            msg += ", and a method resolution order of "
            msg += str(inspect.getmro(toInsert.__class__))
            raise InvalidArgumentType(msg.format(arg=argName))

        if self._isPoint:
            objOffAxisLen = self._base._featureCount
            insertOffAxisLen = len(toInsert.features)
            objHasAxisNames = self._base._pointNamesCreated()
            insertHasAxisNames = toInsert._pointNamesCreated()
            objHasOffAxisNames = self._base._featureNamesCreated()
            insertHasOffAxisNames = toInsert._featureNamesCreated()
            offAxis = 'feature'
            funcName = 'points.' + func
        else:
            objOffAxisLen = self._base._pointCount
            insertOffAxisLen = len(toInsert.points)
            objHasAxisNames = self._base._featureNamesCreated()
            insertHasAxisNames = toInsert._featureNamesCreated()
            objHasOffAxisNames = self._base._pointNamesCreated()
            insertHasOffAxisNames = toInsert._pointNamesCreated()
            offAxis = 'point'
            funcName = 'features.' + func

        if objOffAxisLen != insertOffAxisLen:
            if len(self._base._shape) > 2:
                msg = "Cannot perform {0} operation when data has "
                msg += "different dimensions"
                raise ImproperObjectAction(msg.format(funcName))
            msg = "The argument '{arg}' must have the same number of "
            msg += "{offAxis}s as the caller object. This object contains "
            msg += "{objCount} {offAxis}s and {arg} contains {insertCount} "
            msg += "{offAxis}s."
            msg = msg.format(arg=argName, offAxis=offAxis,
                             objCount=objOffAxisLen,
                             insertCount=insertOffAxisLen)
            raise InvalidArgumentValue(msg)

        # this helper ignores default names - so we can only have an
        # intersection of names when BOTH objects have names created.
        if objHasAxisNames and insertHasAxisNames:
            self._validateEmptyNamesIntersection(argName, toInsert)
        # helper looks for name inconsistency that can be resolved by
        # reordering - definitionally, if one object has all default names,
        # there can be no inconsistency, so both objects must have names
        # assigned for this to be relevant.
        if objHasOffAxisNames and insertHasOffAxisNames:
            self._validateReorderedNames(offAxis, funcName, toInsert)

    def _validateEmptyNamesIntersection(self, argName, argValue):
        intersection = self._nameIntersection(argValue)
        shared = []
        if intersection:
            for name in intersection:
                if name[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                    shared.append(name)

        if shared != []:
            truncated = False
            if len(shared) > 10:
                full = len(shared)
                shared = shared[:10]
                truncated = True

            msg = "The argument named " + argName + " must not share any "
            msg += self._axis + "Names with the calling object, yet the "
            msg += "following names occured in both: "
            msg += nimble.exceptions._prettyListString(shared)
            if truncated:
                msg += "... (only first 10 entries out of " + str(full)
                msg += " total)"
            raise InvalidArgumentValue(msg)

    def _nameIntersection(self, other):
        """
        Returns a set containing only names that are shared along the
        axis between the two objects.
        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, nimble.core.data.Base):
            msg = "The other object must be an instance of base"
            raise InvalidArgumentType(msg)

        axis = self._axis
        self._base._defaultNamesGeneration_NamesSetOperations(other, axis)
        if axis == 'point':
            return (self._base.pointNames.keys()
                    & other.pointNames.keys())

        return (self._base.featureNames.keys()
                & other.featureNames.keys())

    def _validateReorderedNames(self, axis, callSym, other):
        """
        Validate axis names to check to see if they are equal ignoring
        order. Raises an exception if the objects do not share exactly
        the same names, or requires reordering in the presence of
        default names.
        """
        if axis == 'point':
            lnames = self._base.points.getNames()
            rnames = other.points.getNames()
            lGetter = self._base.points.getIndex
            rGetter = other.points.getIndex
        else:
            lnames = self._base.features.getNames()
            rnames = other.features.getNames()
            lGetter = self._base.features.getIndex
            rGetter = other.features.getIndex

        inconsistencies = inconsistentNames(lnames, rnames)

        if len(inconsistencies) != 0:
            # check for the presence of default names; we don't allow
            # reordering in that case.
            msgBase = "When calling caller." + callSym + "(callee) we require "
            msgBase += "that the " + axis + " names all contain the same "
            msgBase += "names, regardless of order. "
            msg = copy.copy(msgBase)
            msg += "However, when default names are present, we don't allow "
            msg += "reordering to occur: either all names must be specified, "
            msg += "or the order must be the same."

            if any(x[:len(DEFAULT_PREFIX)] == DEFAULT_PREFIX for x in lnames):
                raise ImproperObjectAction(msg)
            if any(x[:len(DEFAULT_PREFIX)] == DEFAULT_PREFIX for x in rnames):
                raise ImproperObjectAction(msg)

            ldiff = numpy.setdiff1d(lnames, rnames, assume_unique=True)
            # names are not the same.
            if len(ldiff) != 0:
                rdiff = numpy.setdiff1d(rnames, lnames, assume_unique=True)
                msgBase += "Yet, the following names were unmatched (caller "
                msgBase += "names on the left, callee names on the right):\n"
                msg = copy.copy(msgBase)
                table = [['ID', 'name', '', 'ID', 'name']]
                for lname, rname in zip(ldiff, rdiff):
                    table.append([lGetter(lname), lname, "   ",
                                  rGetter(rname), rname])

                msg += nimble.core.logger.tableString.tableString(table)
                print(msg, file=sys.stderr)

                raise InvalidArgumentValue(msg)

    def _alignNames(self, axis, toInsert):
        """
        Sort the point or feature names of the passed object to match
        this object. If sorting is necessary, a copy will be returned to
        prevent modification of the passed object, otherwise the
        original object will be returned. Assumes validation of the
        names has already occurred.
        """
        if axis == 'point':
            objNamesCreated = self._base._pointNamesCreated()
            toInsertNamesCreated = toInsert._pointNamesCreated()
            objNames = self._base.points.getNames
            toInsertNames = toInsert.points.getNames
            def sorter(obj, names):
                obj.points.permute(names)
        else:
            objNamesCreated = self._base._featureNamesCreated()
            toInsertNamesCreated = toInsert._featureNamesCreated()
            objNames = self._base.features.getNames
            toInsertNames = toInsert.features.getNames
            def sorter(obj, names):
                obj.features.permute(names)

        # This may not look exhaustive, but because of the previous call to
        # _validateInsertableData before this helper, most of the toInsert
        # cases will have already caused an exception
        if objNamesCreated and toInsertNamesCreated:
            objAllDefault = all(isDefaultName(n) for n in objNames())
            toInsertAllDefault = all(isDefaultName(n) for n in toInsertNames())
            reorder = objNames() != toInsertNames()
            if not (objAllDefault or toInsertAllDefault) and reorder:
                # use copy when reordering so toInsert object is not modified
                toInsert = toInsert.copy()
                sorter(toInsert, objNames())

        return toInsert

    def _setInsertedCountAndNames(self, insertedObj, insertedBefore):
        """
        Modify the point or feature count to include the insertedObj. If
        one or both objects have names, names will be set as well.
        """
        if self._isPoint:
            newPtCount = len(self) + len(insertedObj.points)
            # only need to adjust names if names are present
            if not (self._namesCreated()
                    or insertedObj.points._namesCreated()):
                self._base._pointCount = newPtCount
                return
            objNames = self._getNames()
            insertedNames = insertedObj.points.getNames()
            # must change point count AFTER getting names
            self._base._pointCount = newPtCount
            setObjNames = self._setNames
            self._base._shape[0] = newPtCount
        else:
            newFtCount = len(self) + len(insertedObj.features)
            # only need to adjust names if names are present
            if not (self._base._featureNamesCreated()
                    or insertedObj._featureNamesCreated()):
                self._base._featureCount = newFtCount
                return
            objNames = self._getNames()
            insertedNames = insertedObj.features.getNames()
            # must change point count AFTER getting names
            self._base._featureCount = newFtCount
            setObjNames = self._setNames
        # ensure no collision with default names
        adjustedNames = []
        for name in insertedNames:
            if isDefaultName(name):
                adjustedNames.append(self._nextDefaultName())
            else:
                adjustedNames.append(name)
        startNames = objNames[:insertedBefore]
        endNames = objNames[insertedBefore:]

        newNames = startNames + adjustedNames + endNames
        setObjNames(newNames, useLog=False)

    def _sigFunc(self, funcName):
        """
        Get the top level function containing the correct signature.
        """
        if self._isPoint:
            return getattr(Points, funcName)
        return getattr(Features, funcName)

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _permute_implementation(self, indexPosition):
        pass

    @abstractmethod
    def _structuralBackend_implementation(self, structure, targetList):
        pass

    @abstractmethod
    def _insert_implementation(self, insertBefore, toInsert):
        pass

    @abstractmethod
    def _transform_implementation(self, function, limitTo):
        pass

    @abstractmethod
    def _repeat_implementation(self, totalCopies, copyVectorByVector):
        pass

    @abstractmethod
    def _unique_implementation(self):
        pass

###########
# Helpers #
###########

def _validateStructuralArguments(structure, axis, target, start, end,
                                 number, randomize):
    """
    Check for conflicting and co-dependent arguments.
    """
    targetName = 'to' + structure.capitalize()
    if all(param is None for param in [target, start, end, number]):
        msg = "You must provide a value for {0}, ".format(targetName)
        msg += " or start/end, or number."
        raise InvalidArgumentTypeCombination(msg)
    if number is not None and number < 1:
        msg = "number must be greater than zero"
        raise InvalidArgumentValue(msg)
    if number is None and randomize:
        msg = "randomize selects a random subset of "
        msg += "{0}s to {1}. ".format(axis, structure)
        msg += "When randomize=True, the number argument cannot be None"
        raise InvalidArgumentValueCombination(msg)
    if target is not None:
        if start is not None or end is not None:
            msg = "Range removal is exclusive, to use it, "
            msg += "{0} must be None".format(targetName)
            raise InvalidArgumentTypeCombination(msg)

def _validateStartEndRange(start, end, axis, axisLength):
    """
    Check that the start and end values are valid.
    """
    if start < 0 or start > axisLength:
        msg = "start must be a valid index, in the range of possible "
        msg += axis + 's'
        raise InvalidArgumentValue(msg)
    if end < 0 or end > axisLength:
        msg = "end must be a valid index, in the range of possible "
        msg += axis + 's'
        raise InvalidArgumentValue(msg)
    if start > end:
        msg = "The start index cannot be greater than the end index"
        raise InvalidArgumentValueCombination(msg)

class AxisIterator(object):
    """
    Object providing iteration through each item in the axis.
    """
    def __init__(self, axisObj):
        if axisObj._isPoint:
            self.viewer = axisObj._base.pointView
        else:
            self.viewer = axisObj._base.featureView
        self._axisLen = len(axisObj)
        self._position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._position < self._axisLen:
            value = self.viewer(self._position)
            self._position += 1
            return value

        raise StopIteration
