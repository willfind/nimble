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
import operator
import functools
import re

import numpy as np

import nimble
from nimble import fill
from nimble.match import QueryString
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import ImproperObjectAction
from nimble.exceptions import InvalidArgumentTypeCombination
from nimble.exceptions import InvalidArgumentValueCombination
from nimble._utility import isAllowedSingleElement, validateAllAllowedElements
from nimble._utility import prettyListString
from nimble.core.logger import handleLogging
from nimble._utility import inspectArguments
from nimble._utility import tableString
from .points import Points
from .features import Features
from ._dataHelpers import valuesToPythonList, constructIndicesList
from ._dataHelpers import validateInputString
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
    def __init__(self, base, names, **kwargs):
        self._base = base
        if isinstance(self, Points):
            self._axis = 'point'
            self._isPoint = True
        elif isinstance(self, Features):
            self._axis = 'feature'
            self._isPoint = False
        else:
            msg = 'Axis objects must also inherit from Points or Features'
            raise TypeError(msg)

        if names is not None and len(names) != len(self):
            msg = "The length of the {axis}Names ({lenNames}) must match the "
            msg += "{axis}s given in shape ({lenAxis})"
            msg = msg.format(axis=self._axis, lenNames=len(names),
                             lenAxis=len(self))
            raise InvalidArgumentValue(msg)

        # Set up point names
        if names is None:
            self.namesInverse = None
            self.names = None
        elif isinstance(names, dict):
            self._setNames(names, useLog=False)
        else:
            names = valuesToPythonList(names, self._axis + 'Names')
            self._setNames(names, useLog=False)

        super().__init__(base, **kwargs)

    def __len__(self):
        if self._isPoint:
            return self._base.shape[0]
        return self._base.shape[1]

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

        return self.namesInverse[index]

    def _getNames(self):
        if not self._namesCreated():
            self._setAllDefault()

        return copy.copy(self.namesInverse)


    def _setName(self, oldIdentifier, newName, useLog=None):
        if len(self) == 0:
            msg = "Cannot set any {0} names; this object has no {0}s"
            msg = msg.format(self._axis)
            raise ImproperObjectAction(msg)
        if not isinstance(newName, (str, type(None))):
            msg = "The new name must be either None or a string"
            raise InvalidArgumentType(msg)
        if self.names is None:
            self._setAllDefault()

        index = self._getIndex(oldIdentifier)
        if oldIdentifier in self.names:
            oldName = oldIdentifier
        else:
            oldName = self.namesInverse[index]

        if newName in self.names:
            if self.namesInverse[index] == newName:
                return
            msg = "This name '" + newName + "' is already in use"
            raise InvalidArgumentValue(msg)

        #remove the current name
        if oldName is not None:
            del self.names[oldName]

        # setup the new name
        self.namesInverse[index] = newName
        if newName is not None:
            self.names[newName] = index

        handleLogging(useLog, 'prep', '{ax}s.setName'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('setName'),
                      oldIdentifier, newName)


    def _setNames(self, assignments, useLog=None):
        self._setNamesBackend(assignments)

        handleLogging(useLog, 'prep', '{ax}s.setNames'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('setNames'),
                      assignments)

    def _setNamesBackend(self, assignments):
        if assignments is None:
            self.names = None
            self.namesInverse = None
            return
        if not isinstance(assignments, (list, dict)):
            assignments = valuesToPythonList(assignments, 'assignments')
        count = len(self)
        if len(assignments) != count:
            msg = "assignments may only be an ordered container type, with as "
            msg += "many entries (" + str(len(assignments)) + ") as this axis "
            msg += "is long (" + str(count) + ")"
            raise InvalidArgumentValue(msg)
        if count == 0:
            self.names = {}
            self.namesInverse = []
            return
        if not isinstance(assignments, dict):
            # convert to dict so we only write the checking code once
            # validation will occur when generating the inverse list
            names = {}
            for i, name in enumerate(assignments):
                if name in names:
                    msg = "Cannot input duplicate names: " + str(name)
                    raise InvalidArgumentValue(msg)
                if name is not None:
                    names[name] = i
        else:
            # have to copy the input, could be from another object
            names = copy.deepcopy(assignments)
            if None in names:
                del names[None]
        # at this point, the input must be a dict
        # check input before assigning to attributes
        reverseMap = [None] * len(self)
        for name, value in names.items():
            if not isinstance(name, str):
                raise InvalidArgumentValue("Names must be strings")
            if not isinstance(value, int):
                raise InvalidArgumentValue("Indices must be integers")
            if value < 0 or value >= count:
                msg = "Indices must be within 0 to "
                msg += "len(self." + self._axis + "s) - 1"
                raise InvalidArgumentValue(msg)

            reverseMap[value] = name

        self.names = names
        self.namesInverse = reverseMap

    def _getIndex(self, identifier, allowFloats=False):
        num = len(self)
        if num == 0:
            msg = "There are no valid " + self._axis + " identifiers; "
            msg += "this object has 0 " + self._axis + "s"
            raise IndexError(msg)
        if isinstance(identifier, (int, np.integer)):
            if identifier < 0:
                identifier = num + identifier
            if identifier < 0 or identifier >= num:
                msg = "The given index " + str(identifier) + " is outside of "
                msg += "the range of possible indices in the " + self._axis
                msg += " axis (0 to " + str(num - 1) + ")."
                raise IndexError(msg)
        elif isinstance(identifier, str):
            identifier = self._getIndexByName(identifier)
        elif allowFloats and isinstance(identifier, (float, np.float)):
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
        return [self.names[n] for n in names]

    def _hasName(self, name):
        try:
            self._getIndex(name)
            return True
        except KeyError:
            return False

    def _getitem(self, key):
        singleKey = isinstance(key, (int, float, str, np.integer))
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
        return self.names is None or len(self.names) < len(self)

    def _allDefaultNames(self):
        return not self.names

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

        ret._relPath = self._base.relativePath
        ret._absPath = self._base.absolutePath

        handleLogging(useLog, 'prep', '{ax}s.extract'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('extract'),
                      toExtract, start, end, number, randomize)

        return ret


    def _delete(self, toDelete, start, end, number, randomize, useLog=None):
        _ = self._genericStructuralFrontend('delete', toDelete, start, end,
                                              number, randomize)

        handleLogging(useLog, 'prep', '{ax}s.delete'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('delete'),
                      toDelete, start, end, number, randomize)


    def _retain(self, toRetain, start, end, number, randomize, useLog=None):
        ref = self._genericStructuralFrontend('retain', toRetain, start, end,
                                              number, randomize)

        paths = (self._base.absolutePath, self._base.relativePath)
        self._base._referenceFrom(ref, paths=paths)

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
        elif isinstance(by, operator.itemgetter):
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
        if self._base._shape[0] == 0:
            msg = "We disallow this function when there are 0 points"
            raise ImproperObjectAction(msg)
        if self._base._shape[1] == 0:
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
        if self._isPoint:
            ret = nimble.data(self._base.getTypeString(), retData,
                              useLog=False)
            axisNameSetter = ret.points.setNames
            offAxisNameSetter = ret.features.setNames
        else:
            ret = nimble.data(self._base.getTypeString(), retData,
                              rowsArePoints=False, useLog=False)
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

        ret._absPath = self._base.absolutePath
        ret._relPath = self._base.relativePath

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

        toInsert = self._alignNames(toInsert)
        self._insert_implementation(insertBefore, toInsert)

        self._setInsertedCountAndNames(toInsert, insertBefore)

        if append:
            handleLogging(useLog, 'prep', '{ax}s.append'.format(ax=self._axis),
                          self._base.getTypeString(), self._sigFunc('append'),
                          toInsert)
        else:
            handleLogging(useLog, 'prep', '{ax}s.insert'.format(ax=self._axis),
                          self._base.getTypeString(), self._sigFunc('insert'),
                          insertBefore, toInsert)

    def _replace(self, data, locations, useLog=None, **dataKwds):
        if isinstance(data, nimble.core.data.Base):
            if dataKwds:
                msg = 'dataKwds only apply when data is not an instance of '
                msg += 'Base'
                raise InvalidArgumentValue(msg)
            dataObj = data.copy()
        else:
            dataKwds.setdefault('returnType', self._base.getTypeString())
            dataKwds['source'] = data
            dataKwds['useLog'] = False
            # need to delay name setting in case the data requires a transpose
            pnames = []
            fnames = []
            if 'pointNames' in dataKwds:
                pnames.append(dataKwds.pop('pointNames'))
            if 'featureNames' in dataKwds:
                fnames.append(dataKwds.pop('featureNames'))
            dataObj = nimble.data(**dataKwds)
            ax, offAx = (1, 0) if self._isPoint else (0, 1)
            if dataObj.shape[ax] == 1 and dataObj.shape[offAx] > 1:
                # transpose if 1D and not expected vector shape
                dataObj.transpose(useLog=False)
            if pnames:
                dataObj.points.setNames(pnames[0], useLog=False)
            if fnames:
                dataObj.features.setNames(fnames[0], useLog=False)

        dataAxis = dataObj._getAxis(self._axis)
        # locations will take priority over axis name matches
        if locations is None and self._base._shape == dataObj._shape:
            locations = range(len(self))
        elif locations is None:
            if not self._namesCreated():
                msg = '{0}s argument is required when {0}Names do not exist'
                raise InvalidArgumentValue(msg.format(self._axis))
            if not dataAxis._namesCreated():
                msg = 'data must have {0}Names if {0}s is None'
                raise InvalidArgumentValue(msg.format(self._axis))
            locations = dataAxis._getNames()

        locations = constructIndicesList(self._base, self._axis, locations)
        if len(locations) != len(dataAxis):
            msg = 'The number of locations ({0}) must match the number of '
            msg += '{1}s in data ({2})'
            raise InvalidArgumentValue(msg.format(len(locations), self._axis,
                                                  len(dataAxis)))
        iterData = iter(dataAxis)
        for i in locations:
            replacement = next(iterData)
            if len(self._base._shape) > 2:
                replacement = replacement.copy()
                replacement.flatten(useLog=False)
            with self._base._treatAs2D():
                # pylint: disable=cell-var-from-loop
                self._transform(lambda _: replacement, i, useLog=False)

        handleLogging(useLog, 'prep', '{ax}s.replace'.format(ax=self._axis),
                      self._base.getTypeString(), self._sigFunc('replace'),
                      data, locations)

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
                              np.empty(shape=(0, 0)), useLog=False)
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
            for mapKey, mapValues in mapResults.items():
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
            # for consistency use np.nan for None and nans
            if value is None or value != value:
                value = np.nan
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
        if not isinstance(totalCopies, (int, np.int)) or totalCopies < 1:
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
        return self.names is not None

    def _setAllDefault(self):
        self.namesInverse = [None] * len(self)
        self.names = {}

    def _getNamesNoGeneration(self):
        if not self._namesCreated():
            return None
        return self._getNames()

    def _getIndexByName(self, name):
        if not self._namesCreated():
            self._setAllDefault()

        if name not in self.names:
            msg = "The " + self._axis + " name '" + name
            msg += "' cannot be found."
            raise KeyError(msg)
        return self.names[name]

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

        numBool = sum(isinstance(val, (bool, np.bool_)) for val in key)
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
        try:
            query = QueryString(string, elementQuery=False)
        except InvalidArgumentValue as e:
            # positive lookahead catches all ambiguous cases (i.e age == > 20)
            operatorCount = len(re.findall(r'(?=\s(==|!=|>=|>|<=|<)\s)',
                                           string))
            if operatorCount <= 1:
                msg = "'{0}' is not a valid {1} name nor a valid query "
                msg += "string. See help(nimble.match.QueryString) for query "
                msg += "string requirements."
                msg = msg.format(string, offAxis)
            else:
                msg = "Multiple operators in query string. Strings containing "
                msg += "more than one whitespace padded operator cannot be "
                msg += "parsed. Use a function instead or modify the {0}Name "
                msg += "or values that includes a whitespace padded operator."
                msg = msg.format(offAxis)

            raise InvalidArgumentValue(msg) from e

        if not self._base._getAxis(offAxis)._hasName(query.identifier):
            msg = "the {0} '{1}' does not exist".format(offAxis,
                                                        query.identifier)
            raise InvalidArgumentValue(msg)

        return query

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

        # remove names that are no longer in the object
        if structure in ['extract', 'delete']:
            shapeIdx = 0 if self._isPoint else 1
            retAxis = ret._getAxis(self._axis)
            self._base._shape[shapeIdx] -= len(retAxis)
            if self._namesCreated():
                targetSet = set(targetList)
                reindexedInverse = []
                self.names = {}
                for idx, value in enumerate(self.namesInverse):
                    if idx not in targetSet:
                        if value is not None:
                            self.names[value] = len(reindexedInverse)
                        reindexedInverse.append(value)
                self.namesInverse = reindexedInverse

        return ret

    def _getStructuralNames(self, targetList):
        nameList = None
        if self._namesCreated():
            nameList = [self._getName(i) for i in targetList]
        if self._isPoint:
            return nameList, self._base.features._getNamesNoGeneration()

        return self._base.points._getNamesNoGeneration(), nameList

    def _sortByNames(self, reverse):
        if self.names is None or self._anyDefaultNames():
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
            sortedIndex = sorted(enumerate(data), key=operator.itemgetter(1),
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

        shapeIdx = 1 if self._isPoint else 0
        offAxis = 'feature' if self._isPoint else 'point'
        objOffAxis = self._base._getAxis(offAxis)
        toInsertAxis = toInsert._getAxis(self._axis)
        toInsertOffAxis = toInsert._getAxis(offAxis)

        objOffAxisLen = self._base.shape[shapeIdx]
        insertOffAxisLen = len(toInsertOffAxis)
        objHasAxisNames = self._namesCreated()
        insertHasAxisNames = toInsertAxis._namesCreated()
        objHasOffAxisNames = objOffAxis._namesCreated()
        insertHasOffAxisNames = toInsertOffAxis._namesCreated()
        funcName = self._axis + 's.' + func

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
                if name is not None:
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
            msg += prettyListString(shared)
            if truncated:
                msg += "... (only first 10 entries out of " + str(full)
                msg += " total)"
            raise InvalidArgumentValue(msg)

    def _namesSetOperations(self, other, operation):
        """

        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, nimble.core.data.Base):
            msg = "The other object must be an instance of Base"
            raise InvalidArgumentType(msg)

        otherAxis = other._getAxis(self._axis)
        if self.names is None:
            self._setAllDefault()
        if otherAxis.names is None:
            otherAxis._setAllDefault()

        return operation(self.names.keys(), otherAxis.names.keys())

    def _nameIntersection(self, other):
        """
        Returns a set containing only names that are shared along the axis
        between the two objects.
        """
        return self._namesSetOperations(other, operator.and_)

    def _nameDifference(self, other):
        """
        Returns a set containing those names in this object that are not also
        in the input object.
        """
        return self._namesSetOperations(other, operator.sub)

    def _nameSymmetricDifference(self, other):
        """
        Returns a set containing only those names not shared between this
        object and the input object.
        """
        return self._namesSetOperations(other, operator.xor)

    def _nameUnion(self, other):
        """
        Returns a set containing all names in either this object or the input
        object.
        """
        return self._namesSetOperations(other, operator.or_)

    def _validateReorderedNames(self, axis, callSym, other):
        """
        Validate axis names to check to see if they are equal ignoring
        order. Raises an exception if the objects do not share exactly
        the same names, or requires reordering in the presence of
        default names.
        """
        if axis == 'point':
            lAxis = self._base.points
            rAxis = other.points
        else:
            lAxis = self._base.features
            rAxis = other.features

        lnames = lAxis.getNames()
        rnames = rAxis.getNames()
        inconsistencies = inconsistentNames(lnames, rnames)

        if len(inconsistencies) != 0:
            # check for the presence of default names; we don't allow
            # reordering in that case.
            msgBase = "When calling caller." + callSym + "(callee) we require "
            msgBase += "that the " + axis + " names all contain the same "
            msgBase += "names, regardless of order. "

            if lAxis._anyDefaultNames() or rAxis._anyDefaultNames():
                msg = copy.copy(msgBase)
                msg += "However, when default names are present, we don't "
                msg += "allow reordering to occur: either all names must be "
                msg += "specified, or the order must be the same."
                raise ImproperObjectAction(msg)

            ldiff = np.setdiff1d(lnames, rnames, assume_unique=True)
            # names are not the same.
            if len(ldiff) != 0:
                rdiff = np.setdiff1d(rnames, lnames, assume_unique=True)
                msgBase += "Yet, the following names were unmatched (caller "
                msgBase += "names on the left, callee names on the right):\n"
                msg = copy.copy(msgBase)
                table = [['ID', 'name', '', 'ID', 'name']]
                for lname, rname in zip(ldiff, rdiff):
                    table.append([lAxis.getIndex(lname), lname, "   ",
                                  rAxis.getIndex(rname), rname])

                msg += tableString(table)

                raise InvalidArgumentValue(msg)

    def _alignNames(self, toInsert):
        """
        Sort the point or feature names of the passed object to match
        this object. If sorting is necessary, a copy will be returned to
        prevent modification of the passed object, otherwise the
        original object will be returned. Assumes validation of the
        names has already occurred.
        """
        offAxis = 'feature' if self._isPoint else 'point'
        offAxisObj = self._base._getAxis(offAxis)
        toInsertAxis = toInsert._getAxis(offAxis)
        # This may not look exhaustive, but because of the previous call to
        # _validateInsertableData before this helper, most of the toInsert
        # cases will have already caused an exception
        if offAxisObj._namesCreated() and toInsertAxis._namesCreated():
            objAllDefault = offAxisObj._allDefaultNames()
            toInsertAllDefault = toInsertAxis._allDefaultNames()
            reorder = offAxisObj.getNames() != toInsertAxis.getNames()
            if not (objAllDefault or toInsertAllDefault) and reorder:
                # use copy when reordering so toInsert object is not modified
                toInsert = toInsert.copy()
                toInsert._getAxis(offAxis).permute(offAxisObj.getNames())

        return toInsert

    def _setInsertedCountAndNames(self, insertedObj, insertedBefore):
        """
        Modify the point or feature count to include the insertedObj. If
        one or both objects have names, names will be set as well.
        """
        shapeIdx = 0 if self._isPoint else 1
        insertedAxis = insertedObj._getAxis(self._axis)
        newCount = len(self) + len(insertedAxis)
        # only need to adjust names if names are present
        if not (self._namesCreated() or insertedAxis._namesCreated()):
            self._base._shape[shapeIdx] = newCount
            return
        objNames = self._getNames()
        insertedNames = insertedAxis.getNames()
        # must change point count AFTER getting names
        self._base._shape[shapeIdx] = newCount

        startNames = objNames[:insertedBefore]
        endNames = objNames[insertedBefore:]

        newNames = startNames + insertedNames + endNames
        self._setNames(newNames, useLog=False)

    def _uniqueNameGetter(self, uniqueIndices):
        """
        Get the first point or feature names of the object's unique values.
        """
        offAxis = 'feature' if self._isPoint else 'point'
        offAxisObj = self._base._getAxis(offAxis)
        axisNames = False
        offAxisNames = False
        if self._namesCreated():
            axisNames = [self._getName(i) for i in uniqueIndices]
        if offAxisObj._namesCreated():
            offAxisNames = offAxisObj.getNames()

        return axisNames, offAxisNames

    def _getMatchingNames(self, other):
        matches = []
        otherAxis = other._getAxis(self._axis)
        if not self._namesCreated() and otherAxis._namesCreated():
            return matches
        allNames = self._getNames() + otherAxis.getNames()
        if len(set(allNames)) != len(allNames):
            for name in self.names:
                if name in otherAxis.names:
                    matches.append(name)
        return matches

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
