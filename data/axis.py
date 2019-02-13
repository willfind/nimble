"""
Methods and helpers responsible for determining how each function
will operate depending on whether it is being called along the points or
the features axis.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import copy
from abc import abstractmethod
import inspect
import sys
import operator

import six
import numpy

import UML
from UML import fill
from UML.exceptions import InvalidArgumentValue, InvalidArgumentType
from UML.exceptions import ImproperObjectAction
from UML.exceptions import InvalidArgumentTypeCombination
from UML.exceptions import InvalidArgumentValueCombination
from UML.randomness import pythonRandom
from .points import Points
from .dataHelpers import DEFAULT_PREFIX, DEFAULT_PREFIX2, DEFAULT_PREFIX_LENGTH
from .dataHelpers import valuesToPythonList, constructIndicesList
from .dataHelpers import validateInputString, logCaptureFactory

class Axis(object):
    """
    Differentiate how methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    data-type and axis specific operations.

    Parameters
    ----------
    source : UML data object
        The object containing point and feature data.
    """
    def __init__(self, source, **kwargs):
        self._source = source
        kwargs['source'] = source
        if isinstance(self, Points):
            self._axis = 'point'
        else:
            self._axis = 'feature'
        super(Axis, self).__init__(**kwargs)

    def __iter__(self):
        return AxisIterator(self)

    def __len__(self):
        if isinstance(self, Points):
            return self._source._pointCount
        else:
            return self._source._featureCount

    ########################
    # Low Level Operations #
    ########################

    def _getName(self, index):
        if not self._namesCreated():
            self._setAllDefault()
        if isinstance(self, Points):
            return self._source.pointNamesInverse[index]
        else:
            return self._source.featureNamesInverse[index]

    def _getNames(self):
        if not self._namesCreated():
            self._setAllDefault()
        if isinstance(self, Points):
            namesList = self._source.pointNamesInverse
        else:
            namesList = self._source.featureNamesInverse

        return copy.copy(namesList)

    def _setName(self, oldIdentifier, newName):
        if isinstance(self, Points):
            namesDict = self._source.pointNames
        else:
            namesDict = self._source.featureNames
        if len(self) == 0:
            msg = "Cannot set any {0} names; this object has no {0}s"
            msg = msg.format(self._axis)
            raise ImproperObjectAction(msg)
        if namesDict is None:
            self._setAllDefault()
        self._setName_implementation(oldIdentifier, newName)

    def _setNames(self, assignments=None):
        if isinstance(self, Points):
            names = 'pointNames'
            namesInverse = 'pointNamesInverse'
        else:
            names = 'featureNames'
            namesInverse = 'featureNamesInverse'
        if assignments is None:
            setattr(self._source, names, None)
            setattr(self._source, namesInverse, None)
            return
        count = len(self)
        if isinstance(assignments, dict):
            self._setNamesFromDict(assignments, count)
        else:
            assignments = valuesToPythonList(assignments, 'assignments')
            self._setNamesFromList(assignments, count)

    def _getIndex(self, identifier):
        num = len(self)
        accepted = (six.string_types, int, numpy.integer)

        toReturn = identifier
        if num == 0:
            msg = "There are no valid " + self._axis + "identifiers; "
            msg += "this object has 0 " + self._axis + "s"
            raise ImproperObjectAction(msg)
        if identifier is None:
            msg = "An identifier cannot be None."
            raise InvalidArgumentType(msg)
        if not isinstance(identifier, accepted):
            msg = "The identifier must be either a string (a valid "
            msg += self._axis + " name) or an integer (python or numpy) index "
            msg += "between 0 and " + str(num - 1) + " inclusive. "
            msg += "Instead we got: " + str(identifier)
            raise InvalidArgumentType(msg)
        if isinstance(identifier, (int, numpy.integer)):
            if identifier < 0:
                identifier = num + identifier
                toReturn = identifier
            if identifier < 0 or identifier >= num:
                msg = "The given index " + str(identifier) + " is outside of "
                msg += "the range of possible indices in the " + self._axis
                msg += " axis (0 to " + str(num - 1) + ")."
                raise InvalidArgumentValue(msg)
        if isinstance(identifier, six.string_types):
            try:
                toReturn = self._getIndexByName(identifier)
            except KeyError:
                msg = "The " + self._axis + " name '" + identifier
                msg += "' cannot be found."
                raise InvalidArgumentValue(msg)

        return toReturn

    def _getIndices(self, names):
        if not self._namesCreated():
            self._setAllDefault()
        if isinstance(self, Points):
            namesDict = self._source.pointNames
        else:
            namesDict = self._source.featureNames

        return [namesDict[n] for n in names]

    def _hasName(self, name):
        try:
            self._getIndex(name)
            return True
        except InvalidArgumentValue:
            return False

    #########################
    # Structural Operations #
    #########################

    def _copy(self, toCopy, start, end, number, randomize):
        ret = self._genericStructuralFrontend('copy', toCopy, start, end,
                                              number, randomize)
        if isinstance(self, Points):
            ret.features.setNames(self._source.features.getNames())
        else:
            ret.points.setNames(self._source.points.getNames())

        ret._absPath = self._source.absolutePath
        ret._relPath = self._source.relativePath

        return ret

    def _extract(self, toExtract, start, end, number, randomize):
        ret = self._genericStructuralFrontend('extract', toExtract, start, end,
                                              number, randomize)

        self._adjustCountAndNames(ret)

        ret._relPath = self._source.relativePath
        ret._absPath = self._source.absolutePath

        self._source.validate()

        return ret

    def _delete(self, toDelete, start, end, number, randomize):
        ret = self._genericStructuralFrontend('delete', toDelete, start, end,
                                              number, randomize)
        self._adjustCountAndNames(ret)
        self._source.validate()

    def _retain(self, toRetain, start, end, number, randomize):
        ref = self._genericStructuralFrontend('retain', toRetain, start, end,
                                              number, randomize)

        ref._relPath = self._source.relativePath
        ref._absPath = self._source.absolutePath

        self._source.referenceDataFrom(ref)

        self._source.validate()

    def _count(self, condition):
        return self._genericStructuralFrontend('count', condition)

    def _sort(self, sortBy, sortHelper):
        if sortBy is not None and sortHelper is not None:
            msg = "Cannot specify a feature to sort by and a helper function. "
            msg += "Either sortBy or sortHelper must be None"
            raise InvalidArgumentTypeCombination(msg)
        if sortBy is None and sortHelper is None:
            msg = "Either sortBy or sortHelper must not be None"
            raise InvalidArgumentTypeCombination(msg)

        if isinstance(self, Points):
            otherAxis = 'feature'
            axisCount = self._source._pointCount
            otherCount = self._source._featureCount
        else:
            otherAxis = 'point'
            axisCount = self._source._featureCount
            otherCount = self._source._pointCount

        if sortBy is not None and isinstance(sortBy, six.string_types):
            axisObj = self._source._getAxis(otherAxis)
            sortBy = axisObj._getIndex(sortBy)

        if sortHelper is not None and not hasattr(sortHelper, '__call__'):
            indices = constructIndicesList(self._source, self._axis, sortHelper)
            if len(indices) != axisCount:
                msg = "This object contains {0} {1}s, "
                msg += "but sortHelper has {2} identifiers"
                msg = msg.format(axisCount, self._axis, len(indices))
                raise InvalidArgumentValue(msg)
            if len(indices) != len(set(indices)):
                msg = "This object contains {0} {1}s, "
                msg += "but sortHelper has {2} unique identifiers"
                msg = msg.format(axisCount, self._axis, len(set(indices)))
                raise InvalidArgumentValue(msg)

            sortHelper = indices

        # its already sorted in these cases
        if otherCount == 0 or axisCount == 0 or axisCount == 1:
            return

        newNameOrder = self._sort_implementation(sortBy, sortHelper)
        self._setNames(newNameOrder)

        self._source.validate()

    # def _flattenToOne(self):
    #     if self._source._pointCount == 0 or self._source._featureCount == 0:
    #         msg = "Can only flattenToOne when there is one or more {0}s. "
    #         msg += "This object has 0 {0}s."
    #         msg = msg.format(self._axis)
    #         raise ImproperActionException(msg)
    #
    #     # TODO: flatten nameless Objects without the need to generate default
    #     # names for them.
    #     if not self._source._pointNamesCreated():
    #         self._source.points._setAllDefault()
    #     if not self._source._featureNamesCreated():
    #         self._source.features._setAllDefault()
    #
    #     self._flattenToOne_implementation()
    #
    #     if isinstance(self, Points):
    #         axisCount = self._source._pointCount
    #         offAxisCount = self._source._featureCount
    #         setAxisCount = self._source._setpointCount
    #         setOffAxisCount = self._source._setfeatureCount
    #         setAxisNames = self._source.points.setNames
    #         setOffAxisNames = self._source.features.setNames
    #     else:
    #         axisCount = self._source._featureCount
    #         offAxisCount = self._source._pointCount
    #         setAxisCount = self._source._setfeatureCount
    #         setOffAxisCount = self._source._setpointCount
    #         setAxisNames = self._source.features.setNames
    #         setOffAxisNames = self._source.points.setNames
    #
    #     setOffAxisCount(axisCount * offAxisCount)
    #     setAxisCount(1)
    #     setOffAxisNames(self._flattenNames(self._axis))
    #     setAxisNames(['Flattened'])
    #
    # def _unflattenFromOne(self, divideInto):
    #     if isinstance(self, Points):
    #         offAxis = 'feature'
    #         axisCount = self._source._pointCount
    #         offAxisCount = self._source._featureCount
    #         setAxisCount = self._source._setpointCount
    #         setOffAxisCount = self._source._setfeatureCount
    #         setAxisNames = self._source.points.setNames
    #         setOffAxisNames = self._source.features.setNames
    #     else:
    #         offAxis = 'point'
    #         axisCount = self._source._featureCount
    #         offAxisCount = self._source._pointCount
    #         setAxisCount = self._source._setfeatureCount
    #         setOffAxisCount = self._source._setpointCount
    #         setAxisNames = self._source.features.setNames
    #         setOffAxisNames = self._source.points.setNames
    #
    #     if offAxisCount == 0:
    #         msg = "Can only unflattenFromOne when there is one or more "
    #         msg = "{offAxis}s. This object has 0 {offAxis}s."
    #         msg = msg.format(offAxis=offAxis)
    #         raise ImproperActionException(msg)
    #     if axisCount != 1:
    #         msg = "Can only unflattenFromOne when there is only one {axis}. "
    #         msg += "This object has {axisCount} {axis}s."
    #         msg += msg.format(axis=self._axis, axisCount=axisCount)
    #         raise ImproperActionException(msg)
    #     if offAxisCount % divideInto != 0:
    #         msg = "The argument num{axisCap}s ({divideInto}) must be a "
    #         msg += "divisor of this object's {offAxis}Count ({offAxisCount}) "
    #         msg += "otherwise it will not be possible to equally divide the "
    #         msg += "elements into the desired number of {axis}s."
    #         msg = msg.format(axisCap=self._axis.capitalize(),
    #                          divideInto=divideInto, offAxis=offAxis,
    #                          offAxisCount=offAxisCount, axis=self._axis)
    #         raise InvalidArgumentValue(msg)
    #
    #     if not self._source._pointNamesCreated():
    #         self._source.points._setAllDefault()
    #     if not self._source._featureNamesCreated():
    #         self._source.features._setAllDefault()
    #
    #     self._unflattenFromOne_implementation(divideInto)
    #     ret = self._unflattenNames(divideInto)
    #
    #     setOffAxisCount(offAxisCount // divideInto)
    #     setAxisCount(divideInto)
    #     setAxisNames(ret[0])
    #     setOffAxisNames(ret[1])

    def _shuffle(self):
        values = len(self)
        indices = list(range(values))
        pythonRandom.shuffle(indices)

        self._sort(sortBy=None, sortHelper=indices)

    def _transform(self, function, limitTo):
        if self._source._pointCount == 0:
            msg = "We disallow this function when there are 0 points"
            raise ImproperObjectAction(msg)
        if self._source._featureCount == 0:
            msg = "We disallow this function when there are 0 features"
            raise ImproperObjectAction(msg)
        if function is None:
            raise InvalidArgumentType("function must not be None")
        if limitTo is not None:
            limitTo = constructIndicesList(self._source, self._axis, limitTo)

        self._transform_implementation(function, limitTo)

        self._source.validate()

    ###########################
    # Higher Order Operations #
    ###########################

    def _calculate(self, function, limitTo):
        if limitTo is not None:
            limitTo = copy.copy(limitTo)
            limitTo = constructIndicesList(self._source, self._axis, limitTo)
        if len(self._source.points) == 0:
            msg = "We disallow this function when there are 0 points"
            raise ImproperObjectAction(msg)
        if len(self._source.features) == 0:
            msg = "We disallow this function when there are 0 features"
            raise ImproperObjectAction(msg)
        if function is None:
            raise InvalidArgumentType("function must not be None")

        ret = self._calculate_implementation(function, limitTo)

        if isinstance(self, Points):
            if limitTo is not None and self._namesCreated():
                names = []
                for index in sorted(limitTo):
                    names.append(self._getName(index))
                ret.points.setNames(names)
            elif self._namesCreated():
                ret.points.setNames(self._getNamesNoGeneration())
        else:
            if limitTo is not None and self._namesCreated():
                names = []
                for index in sorted(limitTo):
                    names.append(self._getName(index))
                ret.features.setNames(names)
            elif self._namesCreated():
                ret.features.setNames(self._getNamesNoGeneration())

        ret._absPath = self._source.absolutePath
        ret._relPath = self._source.relativePath

        self._source.validate()

        return ret

    def _calculate_implementation(self, function, limitTo):
        retData = []
        for viewID, view in enumerate(self):
            if limitTo is not None and viewID not in limitTo:
                continue
            currOut = function(view)
            # first we branch on whether the output has multiple values
            # or is singular.
            if (hasattr(currOut, '__iter__') and
                    # in python3, string has __iter__ too.
                    not isinstance(currOut, six.string_types)):
                # if there are multiple values, they must be random accessible
                if not hasattr(currOut, '__getitem__'):
                    msg = "function must return random accessible data "
                    msg += "(ie has a __getitem__ attribute)"
                    raise InvalidArgumentType(msg)

                toCopyInto = []
                for value in currOut:
                    toCopyInto.append(value)
                retData.append(toCopyInto)
            # singular return
            else:
                retData.append([currOut])

        ret = UML.createData(self._source.getTypeString(), retData)
        if self._axis != 'point':
            ret.transpose()

        return ret

    def _add(self, toAdd, insertBefore):
        self._validateInsertableData(toAdd)
        if self._source.getTypeString() != toAdd.getTypeString():
            toAdd = toAdd.copyAs(self._source.getTypeString())

        if insertBefore is None:
            insertBefore = len(self)
        else:
            insertBefore = self._getIndex(insertBefore)

        offAxis = 'feature' if self._axis == 'point' else 'point'
        toAdd = self._alignNames(offAxis, toAdd)
        self._add_implementation(toAdd, insertBefore)

        self._setAddedCountAndNames(toAdd, insertBefore)

        self._source.validate()

    def _mapReduce(self, mapper, reducer):
        if isinstance(self, Points):
            targetCount = len(self._source.points)
            otherCount = len(self._source.features)
            otherAxis = 'feature'
            viewIter = self._source.points
        else:
            targetCount = len(self._source.features)
            otherCount = len(self._source.points)
            otherAxis = 'point'
            viewIter = self._source.features

        if targetCount == 0:
            return UML.createData(self._source.getTypeString(),
                                  numpy.empty(shape=(0, 0)), useLog=False)
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

        self._source.validate()

        mapResults = {}
        # apply the mapper to each point in the data
        for value in viewIter:
            currResults = mapper(value)
            # the mapper will return a list of key value pairs
            for (k, v) in currResults:
                # if key is new, we must add an empty list
                if k not in mapResults:
                    mapResults[k] = []
                # append value to the list of values associated with the key
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
        ret = UML.createData(self._source.getTypeString(), ret, useLog=False)

        ret._absPath = self._source.absolutePath
        ret._relPath = self._source.relativePath

        return ret

    def _fill(self, toMatch, toFill, limitTo=None, returnModified=False,
              **kwarguments):
        modified = None
        toTransform = fill.factory(toMatch, toFill, **kwarguments)

        if returnModified:
            def bools(values):
                return [True if toMatch(val) else False for val in values]

            modified = self._calculate(bools, limitTo)
            if isinstance(self, Points):
                currNames = modified.points.getNames()
                modNames = [n + "_modified" for n in currNames]
                modified.points.setNames(modNames)
            else:
                currNames = modified.features.getNames()
                modNames = [n + "_modified" for n in currNames]
                modified.features.setNames(modNames)

        self._transform(toTransform, limitTo)

        self._source.validate()

        return modified

    def _normalize(self, subtract, divide, applyResultTo):
        # used to trigger later conditionals
        alsoIsObj = isinstance(applyResultTo, UML.data.Base)

        # the operation is different when the input is a vector
        # or produces a vector (ie when the input is a statistics
        # string) so during the validation steps we check for
        # those cases
        subIsVec = False
        divIsVec = False

        # check it is within the desired types
        allowedTypes = (int, float, six.string_types, UML.data.Base)
        if subtract is not None:
            if not isinstance(subtract, allowedTypes):
                msg = "The argument named subtract must have a value that is "
                msg += "an int, float, string, or is a UML data object"
                raise InvalidArgumentType(msg)
        if divide is not None:
            if not isinstance(divide, allowedTypes):
                msg = "The argument named divide must have a value that is "
                msg += "an int, float, string, or is a UML data object"
                raise InvalidArgumentType(msg)

        # check that if it is a string, it is one of the accepted values
        accepted = [
            'max', 'mean', 'median', 'min', 'unique count',
            'proportion missing', 'proportion zero', 'standard deviation',
            'std', 'population std', 'population standard deviation',
            'sample std', 'sample standard deviation'
            ]
        if isinstance(subtract, six.string_types):
            validateInputString(subtract, accepted, 'subtract')
        if isinstance(divide, six.string_types):
            validateInputString(divide, accepted, 'divide')

        # arg generic helper to check that objects are of the
        # correct shape/size
        def validateInObjectSize(argname, argval):
            inPC = len(argval.points)
            inFC = len(argval.features)
            objPC = self._source._pointCount
            objFC = self._source._featureCount

            inMainLen = inPC if self._axis == "point" else inFC
            inOffLen = inFC if self._axis == "point" else inPC
            objMainLen = objPC if self._axis == "point" else objFC
            objOffLen = objFC if self._axis == 'point' else objPC

            if inMainLen != objMainLen or inOffLen != objOffLen:
                vecErr = argname + " "
                vecErr += "was a UML object in the shape of a "
                vecErr += "vector (" + str(inPC) + " x "
                vecErr += str(inFC) + "), "
                vecErr += "but the length of long axis did not match "
                vecErr += "the number of " + self._axis + "s in this object ("
                vecErr += str(self._source._pointCount) + ")."
                # treat it as a vector
                if inMainLen == 1:
                    if inOffLen != objMainLen:
                        raise InvalidArgumentValue(vecErr)
                    return True
                # treat it as a vector
                elif inOffLen == 1:
                    if inMainLen != objMainLen:
                        raise InvalidArgumentValue(vecErr)
                    argval.transpose()
                    return True
                # treat it as a mis-sized object
                else:
                    msg = argname + " "
                    msg += "was a UML object with a shape of ("
                    msg += str(inPC) + " x " + str(inFC) + "), "
                    msg += "but it doesn't match the shape of the calling"
                    msg += "object (" + str(objPC) + " x "
                    msg += str(objFC) + ")"
                    raise InvalidArgumentValue(msg)
            return False

        def checkAlsoShape(also, objIn):
            """
            Raises an exception if the normalized axis shape doesn't
            match the calling object, or if when subtract of divide
            takes an object, also doesn't match the shape of the caller
            (this is to be called after) the check that the caller's
            shape matches that of the subtract or divide argument.
            """
            offAxis = 'feature' if self._axis == 'point' else 'point'
            callerP = len(self._source.points)
            callerF = len(self._source.features)
            alsoP = len(also.points)
            alsoF = len(also.features)

            callMainLen = callerP if self._axis == "point" else callerF
            alsoMainLen = alsoP if self._axis == "point" else alsoF
            callOffLen = callerF if self._axis == "point" else callerP
            alsoOffLen = alsoF if self._axis == "point" else alsoP

            if callMainLen != alsoMainLen:
                msg = "applyResultTo must have the same number of "
                msg += self._axis + "s (" + str(alsoMainLen) + ") as the "
                msg += "calling object (" + str(callMainLen) + ")"
                raise InvalidArgumentValue(msg)
            if objIn and callOffLen != alsoOffLen:
                msg = "When a non-vector UML object is given for the subtract "
                msg += "or divide arguments, then applyResultTo "
                msg += "must have the same number of " + offAxis
                msg += "s (" + str(alsoOffLen) + ") as the calling object "
                msg += "(" + str(callOffLen) + ")"
                raise InvalidArgumentValueCombination(msg)

        # actually check that objects are the correct shape/size
        objArg = False
        if isinstance(subtract, UML.data.Base):
            subIsVec = validateInObjectSize("subtract", subtract)
            objArg = True
        if isinstance(divide, UML.data.Base):
            divIsVec = validateInObjectSize("divide", divide)
            objArg = True

        # check the shape of applyResultTo
        if alsoIsObj:
            checkAlsoShape(applyResultTo, objArg)

        # if a statistics string was entered, generate the results
        # of that statistic
        #		if isinstance(subtract, basestring):
        #			if axis == 'point':
        #				subtract = self._statistics(subtract)
        #			else:
        #				subtract = self._statistics(subtract)
        #			subIsVec = True
        #		if isinstance(divide, basestring):
        #			if axis == 'point':
        #				divide = self._statistics(divide)
        #			else:
        #				divide = self._statistics(divide)
        #			divIsVec = True

        if isinstance(self, Points):
            indexGetter = lambda x: self._getIndex(x.points.getName(0))
            if isinstance(subtract, six.string_types):
                subtract = self._statistics(subtract)
                subIsVec = True
            if isinstance(divide, six.string_types):
                divide = self._statistics(divide)
                divIsVec = True
        else:
            indexGetter = lambda x: self._getIndex(x.features.getName(0))
            if isinstance(subtract, six.string_types):
                subtract = self._statistics(subtract)
                subIsVec = True
            if isinstance(divide, six.string_types):
                divide = self._statistics(divide)
                divIsVec = True

        # helper for when subtract is a vector of values
        def subber(currView):
            ret = []
            for val in currView:
                ret.append(val - subtract[indexGetter(currView)])
            return ret

        # helper for when divide is a vector of values
        def diver(currView):
            ret = []
            for val in currView:
                ret.append(val / divide[indexGetter(currView)])
            return ret

        # first perform the subtraction operation
        if subtract is not None and subtract != 0:
            if subIsVec:
                if isinstance(self, Points):
                    self._transform(subber, None)
                    if alsoIsObj:
                        applyResultTo.points.transform(subber)
                else:
                    self._transform(subber, None)
                    if alsoIsObj:
                        applyResultTo.features.transform(subber)
            else:
                self._source -= subtract
                if alsoIsObj:
                    applyResultTo -= subtract

        # then perform the division operation
        if divide is not None and divide != 1:
            if divIsVec:
                if isinstance(self, Points):
                    self._transform(diver, None)
                    if alsoIsObj:
                        applyResultTo.points.transform(diver)
                else:
                    self._transform(diver, None)
                    if alsoIsObj:
                        applyResultTo.features.transform(diver)
            else:
                self._source /= divide
                if alsoIsObj:
                    applyResultTo /= divide

    ###################
    # Query functions #
    ###################

    def _nonZeroIterator(self):
        if self._source._pointCount == 0 or self._source._featureCount == 0:
            return EmptyIt()

        return self._nonZeroIterator_implementation()

    def _unique(self):
        ret = self._unique_implementation()

        ret._absPath = self._source.absolutePath
        ret._relPath = self._source.relativePath

        ret.validate()
        return ret

    ###################
    # Query functions #
    ###################

    def _similarities(self, similarityFunction):
        accepted = [
            'correlation', 'covariance', 'dot product', 'sample covariance',
            'population covariance'
            ]
        cleanFuncName = validateInputString(similarityFunction, accepted,
                                            'similarities')

        if cleanFuncName == 'correlation':
            toCall = UML.calculate.correlation
        elif cleanFuncName in ['covariance', 'samplecovariance']:
            toCall = UML.calculate.covariance
        elif cleanFuncName == 'populationcovariance':
            def populationCovariance(X, X_T):
                return UML.calculate.covariance(X, X_T, False)

            toCall = populationCovariance
        elif cleanFuncName == 'dotproduct':
            def dotProd(X, X_T):
                return X * X_T

            toCall = dotProd

        transposed = self._source.copy()
        transposed.transpose()

        if isinstance(self, Points):
            ret = toCall(self._source, transposed)
        else:
            ret = toCall(transposed, self._source)

        # TODO validation or result.

        ret._absPath = self._source.absolutePath
        ret._relPath = self._source.relativePath

        return ret

    def _statistics(self, statisticsFunction, groupByFeature=None):
        if self._axis == 'point' or groupByFeature is None:
            return self._statisticsBackend(statisticsFunction)
        else:
            # groupByFeature is only a parameter for .features
            res = self._source.groupByFeature(groupByFeature)
            for k in res:
                res[k] = res[k].features._statisticsBackend(statisticsFunction)
            return res

    def _statisticsBackend(self, statisticsFunction):
        accepted = [
            'max', 'mean', 'median', 'min', 'unique count',
            'proportion missing', 'proportion zero', 'standard deviation',
            'std', 'population std', 'population standard deviation',
            'sample std', 'sample standard deviation'
            ]
        cleanFuncName = validateInputString(statisticsFunction, accepted,
                                            'statistics')

        if cleanFuncName == 'max':
            toCall = UML.calculate.maximum
        elif cleanFuncName == 'mean':
            toCall = UML.calculate.mean
        elif cleanFuncName == 'median':
            toCall = UML.calculate.median
        elif cleanFuncName == 'min':
            toCall = UML.calculate.minimum
        elif cleanFuncName == 'uniquecount':
            toCall = UML.calculate.uniqueCount
        elif cleanFuncName == 'proportionmissing':
            toCall = UML.calculate.proportionMissing
        elif cleanFuncName == 'proportionzero':
            toCall = UML.calculate.proportionZero
        elif cleanFuncName in ['std', 'standarddeviation']:
            def sampleStandardDeviation(values):
                return UML.calculate.standardDeviation(values, True)

            toCall = sampleStandardDeviation
        elif cleanFuncName in ['samplestd', 'samplestandarddeviation']:
            def sampleStandardDeviation(values):
                return UML.calculate.standardDeviation(values, True)

            toCall = sampleStandardDeviation
        elif cleanFuncName in ['populationstd', 'populationstandarddeviation']:
            toCall = UML.calculate.standardDeviation

        ret = self._calculate(toCall, limitTo=None)
        if isinstance(self, Points):
            ret.points.setNames(self._getNames())
            ret.features.setName(0, cleanFuncName)
        else:
            ret.points.setName(0, cleanFuncName)
            ret.features.setNames(self._getNames())

        return ret

    #####################
    # Low Level Helpers #
    #####################

    def _namesCreated(self):
        if isinstance(self, Points):
            return not self._source.pointNames is None
        else:
            return not self._source.featureNames is None

    def _nextDefaultName(self):
        if isinstance(self, Points):
            ret = DEFAULT_PREFIX2%self._source._nextDefaultValuePoint
            self._source._nextDefaultValuePoint += 1
        else:
            ret = DEFAULT_PREFIX2%self._source._nextDefaultValueFeature
            self._source._nextDefaultValueFeature += 1
        return ret

    def _setAllDefault(self):
        if isinstance(self, Points):
            self._source.pointNames = {}
            self._source.pointNamesInverse = []
            names = self._source.pointNames
            invNames = self._source.pointNamesInverse
        else:
            self._source.featureNames = {}
            self._source.featureNamesInverse = []
            names = self._source.featureNames
            invNames = self._source.featureNamesInverse
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
        if isinstance(self, Points):
            namesDict = self._source.pointNames
        else:
            namesDict = self._source.featureNames

        return namesDict[name]

    def _setName_implementation(self, oldIdentifier, newName):
        if isinstance(self, Points):
            names = self._source.pointNames
            invNames = self._source.pointNamesInverse
        else:
            names = self._source.featureNames
            invNames = self._source.featureNamesInverse

        index = self._getIndex(oldIdentifier)
        if newName is not None:
            if not isinstance(newName, six.string_types):
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
        self._source._incrementDefaultIfNeeded(newName, self._axis)

    def _setNamesFromList(self, assignments, count):
        if isinstance(self, Points):
            def checkAndSet(val):
                if val >= self._source._nextDefaultValuePoint:
                    self._source._nextDefaultValuePoint = val + 1
        else:
            def checkAndSet(val):
                if val >= self._source._nextDefaultValueFeature:
                    self._source._nextDefaultValueFeature = val + 1

        if assignments is None:
            self._setAllDefault()
            return

        if count == 0:
            if len(assignments) > 0:
                msg = "assignments is too large (" + str(len(assignments))
                msg += "); this axis is empty"
                raise InvalidArgumentValue(msg)
            self._setNamesFromDict({}, count)
            return
        if len(assignments) != count:
            msg = "assignments may only be an ordered container type, with as "
            msg += "many entries (" + str(len(assignments)) + ") as this axis "
            msg += "is long (" + str(count) + ")"
            raise InvalidArgumentValue(msg)

        for name in assignments:
            if name is not None and not isinstance(name, six.string_types):
                msg = 'assignments must contain only string values'
                raise InvalidArgumentValue(msg)
            if name is not None and name.startswith(DEFAULT_PREFIX):
                try:
                    num = int(name[DEFAULT_PREFIX_LENGTH:])
                # Case: default prefix with non-integer suffix. This cannot
                # cause a future integer suffix naming collision, so we
                # can ignore it.
                except ValueError:
                    continue
                checkAndSet(num)

        #convert to dict so we only write the checking code once
        temp = {}
        for index in range(len(assignments)):
            name = assignments[index]
            # take this to mean fill it in with a default name
            if name is None:
                name = self._nextDefaultName()
            if name in temp:
                msg = "Cannot input duplicate names: " + str(name)
                raise InvalidArgumentValue(msg)
            temp[name] = index
        assignments = temp

        self._setNamesFromDict(assignments, count)

    def _setNamesFromDict(self, assignments, count):
        if assignments is None:
            self._setAllDefault()
            return
        if not isinstance(assignments, dict):
            msg = "assignments may only be a dict"
            msg += "with as many entries as this axis is long"
            raise InvalidArgumentType(msg)
        if count == 0:
            if len(assignments) > 0:
                msg = "assignments is too large; this axis is empty"
                raise InvalidArgumentValue(msg)
            if isinstance(self, Points):
                self._source.pointNames = {}
                self._source.pointNamesInverse = []
            else:
                self._source.featureNames = {}
                self._source.featureNamesInverse = []
            return
        if len(assignments) != count:
            msg = "assignments may only be a dict, "
            msg += "with as many entries as this axis is long"
            raise InvalidArgumentValue(msg)

        # at this point, the input must be a dict
        #check input before performing any action
        for name in assignments.keys():
            if not None and not isinstance(name, six.string_types):
                raise InvalidArgumentValue("Names must be strings")
            if not isinstance(assignments[name], int):
                raise InvalidArgumentValue("Indices must be integers")
            if assignments[name] < 0 or assignments[name] >= count:
                if isinstance(self, Points):
                    countName = 'points'
                else:
                    countName = 'features'
                msg = "Indices must be within 0 to "
                msg += "len(self." + countName + ") - 1"
                raise InvalidArgumentValue(msg)

        reverseMap = [None] * len(assignments)
        for name in assignments.keys():
            self._source._incrementDefaultIfNeeded(name, self._axis)
            reverseMap[assignments[name]] = name

        # have to copy the input, could be from another object
        if isinstance(self, Points):
            self._source.pointNames = copy.deepcopy(assignments)
            self._source.pointNamesInverse = reverseMap
        else:
            self._source.featureNames = copy.deepcopy(assignments)
            self._source.featureNamesInverse = reverseMap

    ########################
    #  Structural Helpers  #
    ########################

    def _genericStructuralFrontend(self, structure, target=None,
                                   start=None, end=None, number=None,
                                   randomize=False):
        axis = self._axis
        axisLength = len(self)
        hasNameChecker1 = self._hasName
        if isinstance(self, Points):
            hasNameChecker2 = self._source.features._hasName
        else:
            hasNameChecker2 = self._source.points._hasName

        _validateStructuralArguments(structure, axis, target, start,
                                     end, number, randomize)
        targetList = []
        if target is not None and isinstance(target, six.string_types):
            # check if target is a valid name
            if hasNameChecker1(target):
                target = self._getIndex(target)
                targetList.append(target)
            # if not a name then assume it's a query string
            else:
                target = _stringToFunction(target, axis, hasNameChecker2)

        # list-like container types
        if target is not None and not hasattr(target, '__call__'):
            argName = 'to' + structure.capitalize()
            targetList = constructIndicesList(self._source, axis, target, argName)
        # boolean function
        elif target is not None:
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
            targetList = [value for value in range(axisLength)]

        if number:
            if number > len(targetList):
                msg = "The value for 'number' ({0}) ".format(number)
                msg += "is greater than the number of {0}s ".format(axis)
                msg += "to {0} ({1})".format(structure, len(targetList))
                raise InvalidArgumentValue(msg)
            if randomize:
                targetList = pythonRandom.sample(targetList, number)
            else:
                targetList = targetList[:number]

        if structure == 'count':
            return len(targetList)
        return self._structuralBackend_implementation(structure, targetList)

    def _adjustCountAndNames(self, other):
        """
        Adjust the count and names (when names have been generated) for
        this object, removing the names that have been extracted to the
        other object.
        """
        if isinstance(self, Points):
            self._source._pointCount -= len(other.points)
            if self._source._pointNamesCreated():
                idxList = []
                for name in other.points.getNames():
                    idxList.append(self._source.pointNames[name])
                idxList = sorted(idxList)
                for i, val in enumerate(idxList):
                    del self._source.pointNamesInverse[val - i]
                self._source.pointNames = {}
                for idx, pt in enumerate(self._source.pointNamesInverse):
                    self._source.pointNames[pt] = idx

        else:
            self._source._featureCount -= len(other.features)
            if self._source._featureNamesCreated():
                idxList = []
                for name in other.features.getNames():
                    idxList.append(self._source.featureNames[name])
                idxList = sorted(idxList)
                for i, val in enumerate(idxList):
                    del self._source.featureNamesInverse[val - i]
                self._source.featureNames = {}
                for idx, ft in enumerate(self._source.featureNamesInverse):
                    self._source.featureNames[ft] = idx

    # def _flattenNames(self, discardAxis):
    #     """
    #     Axis names for the unflattened axis after a flatten operation.
    #     """
    #     if discardAxis == 'point':
    #         keepNames = self._source.features.getNames()
    #         dropNames = self._source.points.getNames()
    #     else:
    #         keepNames = self._source.points.getNames()
    #         dropNames = self._source.features.getNames()
    #
    #     ret = []
    #     for d in dropNames:
    #         for k in keepNames:
    #             ret.append(k + ' | ' + d)
    #
    #     return ret
    #
    # def _unflattenNames(self, addedAxisLength):
    #     """
    #     New axis names after an unflattening operation.
    #     """
    #     if isinstance(self, Points):
    #         both = self._source.features.getNames()
    #         keptAxisLength = self._source._featureCount // addedAxisLength
    #     else:
    #         both = self._source.points.getNames()
    #         keptAxisLength = self._source._pointCount // addedAxisLength
    #     allDefault = self._namesAreFlattenFormatConsistent(addedAxisLength,
    #                                                        keptAxisLength)
    #
    #     if allDefault:
    #         addedAxisName = None
    #         keptAxisName = None
    #     else:
    #         # we consider the split of the elements into keptAxisLength chunks
    #         # (of which there will be addedAxisLength number of chunks), and
    #         # want the index of the first of each chunk. We allow that first
    #         # name to be representative for that chunk: all will have the same
    #         # stuff past the vertical bar.
    #         locations = range(0, len(both), keptAxisLength)
    #         addedAxisName = [both[n].split(" | ")[1] for n in locations]
    #         keptAxisName = [n.split(" | ")[0] for n in both[:keptAxisLength]]
    #
    #     return addedAxisName, keptAxisName
    #
    # def _namesAreFlattenFormatConsistent(self, newFLen, newUFLen):
    #     """
    #     Validate the formatting of axis names prior to unflattening.
    #
    #     Will raise ImproperActionException if an inconsistency with the
    #     formatting done by the flatten operations is discovered. Returns
    #     True if all the names along the unflattend axis are default, False
    #     otherwise.
    #     """
    #     if isinstance(self, Points):
    #         flat = self._source.points.getNames()
    #         formatted = self._source.features.getNames()
    #     else:
    #         flat = self._source.features.getNames()
    #         formatted = self._source.points.getNames()
    #
    #     def checkIsDefault(axisName):
    #         ret = False
    #         try:
    #             if axisName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
    #                 int(axisName[DEFAULT_PREFIX_LENGTH:])
    #                 ret = True
    #         except ValueError:
    #             ret = False
    #         return ret
    #
    #     # check the contents of the names along the flattened axis
    #     isDefault = checkIsDefault(flat[0])
    #     isExact = flat == ['Flattened']
    #     msg = "In order to unflatten this object, the names must be "
    #     msg += "consistent with the results from a flatten call. "
    #     if not (isDefault or isExact):
    #         msg += "Therefore, the {axis} name for this object ('{axisName}')"
    #         msg += "must either be a default name or the string 'Flattened'"
    #         msg = msg.format(axis=self._axis, axisName=flat[0])
    #         raise ImproperActionException(msg)
    #
    #     # check the contents of the names along the unflattend axis
    #     msg += "Therefore, the {axis} names for this object must either be "
    #     msg += "all default, or they must be ' | ' split names with name "
    #     msg += "values consistent with the positioning from a flatten call."
    #     msg.format(axis=self._axis)
    #     # each name - default or correctly formatted
    #     allDefaultStatus = None
    #     for name in formatted:
    #         isDefault = checkIsDefault(name)
    #         formatCorrect = len(name.split(" | ")) == 2
    #         if allDefaultStatus is None:
    #             allDefaultStatus = isDefault
    #         else:
    #             if isDefault != allDefaultStatus:
    #                 raise ImproperActionException(msg)
    #
    #         if not (isDefault or formatCorrect):
    #             raise ImproperActionException(msg)
    #
    #     # consistency only relevant if we have non-default names
    #     if not allDefaultStatus:
    #         # seen values - consistent wrt original flattend axis names
    #         for i in range(newFLen):
    #             same = formatted[newUFLen*i].split(' | ')[1]
    #             for name in formatted[newUFLen*i:newUFLen*(i+1)]:
    #                 if same != name.split(' | ')[1]:
    #                     raise ImproperActionException(msg)
    #
    #         # seen values - consistent wrt original unflattend axis names
    #         for i in range(newUFLen):
    #             same = formatted[i].split(' | ')[0]
    #             for j in range(newFLen):
    #                 name = formatted[i + (j * newUFLen)]
    #                 if same != name.split(' | ')[0]:
    #                     raise ImproperActionException(msg)
    #
    #     return allDefaultStatus

    ##########################
    #  Higher Order Helpers  #
    ##########################

    def _validateInsertableData(self, toAdd):
        """
        Required validation before inserting an object
        """
        if toAdd is None:
            msg = "The argument 'toAdd' must not have a value of None"
            raise InvalidArgumentType(msg)
        if not isinstance(toAdd, UML.data.Base):
            msg = "The argument 'toAdd' must be an instance of the "
            msg += "UML.data.Base  class. The value we recieved was "
            msg += str(toAdd) + ", had the type " + str(type(toAdd))
            msg += ", and a method resolution order of "
            msg += str(inspect.getmro(toAdd.__class__))
            raise InvalidArgumentType(msg)

        if isinstance(self, Points):
            objOffAxisLen = self._source._featureCount
            addOffAxisLen = len(toAdd.features)
            objHasAxisNames = self._source._pointNamesCreated()
            addHasAxisNames = toAdd._pointNamesCreated()
            objHasOffAxisNames = self._source._featureNamesCreated()
            addHasOffAxisNames = toAdd._featureNamesCreated()
            offAxis = 'feature'
            funcName = 'points.add'
        else:
            objOffAxisLen = self._source._pointCount
            addOffAxisLen = len(toAdd.points)
            objHasAxisNames = self._source._featureNamesCreated()
            addHasAxisNames = toAdd._featureNamesCreated()
            objHasOffAxisNames = self._source._pointNamesCreated()
            addHasOffAxisNames = toAdd._pointNamesCreated()
            offAxis = 'point'
            funcName = 'features.add'

        if objOffAxisLen != addOffAxisLen:
            msg = "The argument 'toAdd' must have the same number of "
            msg += "{offAxis}s as the caller object. This object contains "
            msg += "{objCount} {offAxis}s and toAdd contains {addCount} "
            msg += "{offAxis}s."
            msg = msg.format(offAxis=offAxis, objCount=objOffAxisLen,
                             addCount=addOffAxisLen)
            raise InvalidArgumentValue(msg)

        # this helper ignores default names - so we can only have an
        # intersection of names when BOTH objects have names created.
        if objHasAxisNames and addHasAxisNames:
            self._validateEmptyNamesIntersection('toAdd', toAdd)
        # helper looks for name inconsistency that can be resolved by
        # reordering - definitionally, if one object has all default names,
        # there can be no inconsistency, so both objects must have names
        # assigned for this to be relevant.
        if objHasOffAxisNames and addHasOffAxisNames:
            self._validateReorderedNames(offAxis, funcName, toAdd)

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
            msg += UML.exceptions.prettyListString(shared)
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
        if not isinstance(other, UML.data.Base):
            msg = "The other object must be an instance of base"
            raise InvalidArgumentType(msg)

        axis = self._axis
        self._source._defaultNamesGeneration_NamesSetOperations(other, axis)
        if axis == 'point':
            return (six.viewkeys(self._source.pointNames)
                    & six.viewkeys(other.pointNames))
        else:
            return (six.viewkeys(self._source.featureNames)
                    & six.viewkeys(other.featureNames))

    def _validateReorderedNames(self, axis, callSym, other):
        """
        Validate axis names to check to see if they are equal ignoring
        order. Raises an exception if the objects do not share exactly
        the same names, or requires reordering in the presence of
        default names.
        """
        if axis == 'point':
            lnames = self._source.points.getNames()
            rnames = other.points.getNames()
            lGetter = self._source.points.getIndex
            rGetter = other.points.getIndex
        else:
            lnames = self._source.features.getNames()
            rnames = other.features.getNames()
            lGetter = self._source.features.getIndex
            rGetter = other.features.getIndex

        inconsistencies = self._source._inconsistentNames(lnames, rnames)

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

                msg += UML.logger.tableString.tableString(table)
                print(msg, file=sys.stderr)

                raise InvalidArgumentValue(msg)

    def _alignNames(self, axis, toAdd):
        """
        Sort the point or feature names of the passed object to match
        this object. If sorting is necessary, a copy will be returned to
        prevent modification of the passed object, otherwise the
        original object will be returned. Assumes validation of the
        names has already occurred.
        """
        if axis == 'point':
            objNamesCreated = self._source._pointNamesCreated()
            toAddNamesCreated = toAdd._pointNamesCreated()
            objNames = self._source.points.getNames
            toAddNames = toAdd.points.getNames
            def sorter(obj, names):
                return obj.points.sort(sortHelper=names)
        else:
            objNamesCreated = self._source._featureNamesCreated()
            toAddNamesCreated = toAdd._featureNamesCreated()
            objNames = self._source.features.getNames
            toAddNames = toAdd.features.getNames
            def sorter(obj, names):
                return obj.features.sort(sortHelper=names)

        # This may not look exhaustive, but because of the previous call to
        # _validateInsertableData before this helper, most of the toAdd cases
        # will have already caused an exception
        if objNamesCreated and toAddNamesCreated:
            objAllDefault = all(n.startswith(DEFAULT_PREFIX)
                                for n in objNames())
            toAddAllDefault = all(n.startswith(DEFAULT_PREFIX)
                                  for n in toAddNames())
            reorder = objNames() != toAddNames()
            if not (objAllDefault or toAddAllDefault) and reorder:
                # use copy when reordering so toAdd object is not modified
                toAdd = toAdd.copy()
                sorter(toAdd, objNames())

        return toAdd

    def _setAddedCountAndNames(self, addedObj, insertedBefore):
        """
        Modify the point or feature count to include the addedObj. If
        one or both objects have names, names will be set as well.
        """
        if isinstance(self, Points):
            newPtCount = len(self) + len(addedObj.points)
            # only need to adjust names if names are present
            if not (self._namesCreated()
                    or addedObj.points._namesCreated()):
                self._source._setpointCount(newPtCount)
                return
            objNames = self._getNames()
            insertedNames = addedObj.points.getNames()
            # must change point count AFTER getting names
            self._source._setpointCount(newPtCount)
            setObjNames = self._setNames
        else:
            newFtCount = len(self) + len(addedObj.features)
            # only need to adjust names if names are present
            if not (self._source._featureNamesCreated()
                    or addedObj._featureNamesCreated()):
                self._source._setfeatureCount(newFtCount)
                return
            objNames = self._getNames()
            insertedNames = addedObj.features.getNames()
            # must change point count AFTER getting names
            self._source._setfeatureCount(newFtCount)
            setObjNames = self._setNames
        # ensure no collision with default names
        adjustedNames = []
        for name in insertedNames:
            if name.startswith(DEFAULT_PREFIX):
                adjustedNames.append(self._nextDefaultName())
            else:
                adjustedNames.append(name)
        startNames = objNames[:insertedBefore]
        endNames = objNames[insertedBefore:]

        newNames = startNames + adjustedNames + endNames
        setObjNames(newNames)

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _sort_implementation(self, sortBy, sortHelper):
        pass

    @abstractmethod
    def _structuralBackend_implementation(self, structure, targetList):
        pass

    @abstractmethod
    def _add_implementation(self, toAdd, insertBefore):
        pass

    # @abstractmethod
    # def _flattenToOne_implementation(self):
    #     pass
    #
    # @abstractmethod
    # def _unflattenFromOne_implementation(self, divideInto):
    #     pass

    @abstractmethod
    def _transform_implementation(self, function, limitTo):
        pass

    @abstractmethod
    def _nonZeroIterator_implementation(self):
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

def _stringToFunction(string, axis, nameChecker):
    """
    Convert a query string into a python function.
    """
    optrDict = {'<=': operator.le, '>=': operator.ge,
                '!=': operator.ne, '==': operator.eq,
                '<': operator.lt, '>': operator.gt}
    # to set in for loop
    nameOfPtOrFt = None
    valueOfPtOrFt = None
    optrOperator = None

    for optr in ['<=', '>=', '!=', '==', '=', '<', '>']:
        if optr in string:
            targetList = string.split(optr)
            # user can use '=' but optrDict only contains '=='
            optr = '==' if optr == '=' else optr
            #after splitting at the optr, list must have 2 items
            if len(targetList) != 2:
                msg = "the target({0}) is a ".format(target)
                msg += "query string but there is an error"
                raise InvalidArgumentValue(msg)
            nameOfPtOrFt = targetList[0]
            valueOfPtOrFt = targetList[1]
            nameOfPtOrFt = nameOfPtOrFt.strip()
            valueOfPtOrFt = valueOfPtOrFt.strip()

            #when point, check if the feature exists or not
            #when feature, check if the point exists or not
            if not nameChecker(nameOfPtOrFt):
                if axis == 'point':
                    offAxis = 'feature'
                else:
                    offAxis = 'point'
                msg = "the {0} ".format(offAxis)
                msg += "'{0}' doesn't exist".format(nameOfPtOrFt)
                raise InvalidArgumentValue(msg)

            optrOperator = optrDict[optr]
            # convert valueOfPtOrFt from a string, if possible
            try:
                valueOfPtOrFt = float(valueOfPtOrFt)
            except ValueError:
                pass
            #convert query string to a function
            def target_f(x):
                return optrOperator(x[nameOfPtOrFt], valueOfPtOrFt)

            target_f.vectorized = True
            target_f.nameOfPtOrFt = nameOfPtOrFt
            target_f.valueOfPtOrFt = valueOfPtOrFt
            target_f.optr = optrOperator
            target = target_f
            break
    # the target can't be converted to a function
    else:
        msg = "'{0}' is not a valid {1} ".format(target, axis)
        msg += 'name nor a valid query string'
        raise InvalidArgumentValue(msg)

    return target

class AxisIterator(object):
    def __init__(self, source):
        self._source = source
        self._position = 0

    def __iter__(self):
        return self

    def next(self):
        """
        Get next item
        """
        if isinstance(self._source, Points):
            viewer = self._source._source.pointView
        else:
            viewer = self._source._source.featureView
        if self._position < len(self._source):
            value = viewer(self._position)
            self._position += 1
            return value
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

class EmptyIt(object):
    """
    Non-zero iterator to return when object is point or feature empty.
    """
    def __iter__(self):
        return self

    def next(self):
        raise StopIteration

    def __next__(self):
        return self.next()
