"""
Methods and helpers responsible for determining how each function
will operate depending on whether it is being called along the points or
the features axis.
"""
from __future__ import absolute_import
from __future__ import division
import copy
from abc import abstractmethod

import six
import numpy

import UML
from UML import fill
from UML.exceptions import ArgumentException, ImproperActionException
from UML.randomness import pythonRandom
from .dataHelpers import OPTRLIST, OPTRDICT
from .dataHelpers import DEFAULT_PREFIX, DEFAULT_PREFIX_LENGTH
from .dataHelpers import valuesToPythonList

class Axis(object):
    """
    Differentiate how methods act dependent on the axis.

    Also provides abstract methods which will be required to perform
    data-type and axis specific operations.

    Parameters
    ----------
    axis : str
        The axis ('point' or 'feature') which the function will be
        applied to.
    source : UML data object
        The object containing point and feature data.
    """
    def __init__(self, axis, source):
        self.axis = axis
        self.source = source
        self._position = 0
        super(Axis, self).__init__()

    def __iter__(self):
        return self

    def next(self):
        """
        Get next item
        """
        if self.axis == 'point':
            while self._position < len(self.source.points):
                value = self.source.pointView(self._position)
                self._position += 1
                return value
            self._position = 0
            raise StopIteration
        else:
            while self._position < len(self.source.features):
                value = self.source.featureView(self._position)
                self._position += 1
                return value
            self._position = 0
            raise StopIteration

    def __next__(self):
        return self.next()

    def __len__(self):
        if self.axis == 'point':
            return self.source._pointCount
        else:
            return self.source._featureCount

    ########################
    # Low Level Operations #
    ########################

    def _getName(self, index):
        if self.axis == 'point':
            hasNames = self.source._pointNamesCreated()
        else:
            hasNames = self.source._featureNamesCreated()

        if not hasNames:
            self.source._setAllDefault(self.axis)

        if self.axis == 'point':
            namesList = self.source.pointNamesInverse
        else:
            namesList = self.source.featureNamesInverse

        return namesList[index]

    def _getNames(self):
        if self.axis == 'point':
            hasNames = self.source._pointNamesCreated()
        else:
            hasNames = self.source._featureNamesCreated()

        if not hasNames:
            self.source._setAllDefault(self.axis)

        if self.axis == 'point':
            namesList = self.source.pointNamesInverse
        else:
            namesList = self.source.featureNamesInverse

        return copy.copy(namesList)

    def _setName(self, oldIdentifier, newName):
        if self.axis == 'point':
            count = len(self.source.points)
            namesDict = self.source.pointNames
        else:
            count = len(self.source.features)
            namesDict = self.source.featureNames
        if count == 0:
            msg = "Cannot set any {0} names; this object has no {0}s"
            msg = msg.format(self.axis)
            raise ArgumentException(msg)
        if namesDict is None:
            self.source._setAllDefault(self.axis)
        self._setName_implementation(oldIdentifier, newName)

    def _setNames(self, assignments=None):
        if self.axis == 'point':
            if assignments is None:
                self.source.pointNames = None
                self.source.pointNamesInverse = None
                return
            count = len(self.source.points)
        else:
            if assignments is None:
                self.source.featureNames = None
                self.source.featureNamesInverse = None
                return
            count = len(self.source.features)
        if isinstance(assignments, dict):
            self._setNamesFromDict(assignments, count)
        else:
            assignments = valuesToPythonList(assignments, 'assignments')
            self._setNamesFromList(assignments, count)

    def _getIndex(self, name):
        if self.axis == 'point':
            hasNames = self.source._pointNamesCreated()
            namesDict = self.source.pointNames
        else:
            hasNames = self.source._featureNamesCreated()
            namesDict = self.source.featureNames

        if not hasNames:
            self.source._setAllDefault(self.axis)

        return namesDict[name]

    def _getIndices(self, names):
        if self.axis == 'point':
            hasNames = self.source._pointNamesCreated()
            namesDict = self.source.pointNames
        else:
            hasNames = self.source._featureNamesCreated()
            namesDict = self.source.featureNames

        if not hasNames:
            self.source._setAllDefault(self.axis)

        return [namesDict[n] for n in names]

    def _hasName(self, name):
        try:
            self.source.getIndex(name)
            return True
        # keyError if not in dict, TypeError if names is None
        except (KeyError, TypeError):
            return False

    #########################
    # Structural Operations #
    #########################

    def _copy(self, toCopy, start, end, number, randomize):
        ret = self._genericStructuralFrontend('copy', toCopy, start, end,
                                              number, randomize)

        if self.axis == 'point':
            ret.setFeatureNames(self.source.getFeatureNames())
        else:
            ret.setPointNames(self.source.getPointNames())

        ret._absPath = self.source.absolutePath
        ret._relPath = self.source.relativePath

        return ret

    def _extract(self, toExtract, start, end, number, randomize):
        ret = self._genericStructuralFrontend('extract', toExtract, start, end,
                                              number, randomize)

        if self.axis == 'point':
            ret.setFeatureNames(self.source.getFeatureNames())
        else:
            ret.setPointNames(self.source.getPointNames())
        _adjustCountAndNames(self.source, self.axis, ret)

        ret._relPath = self.source.relativePath
        ret._absPath = self.source.absolutePath

        self.source.validate()

        return ret

    def _delete(self, toDelete, start, end, number, randomize):
        ret = self._genericStructuralFrontend('delete', toDelete, start, end,
                                              number, randomize)
        _adjustCountAndNames(self.source, self.axis, ret)
        self.source.validate()

    def _retain(self, toRetain, start, end, number, randomize):
        ref = self._genericStructuralFrontend('retain', toRetain, start, end,
                                              number, randomize)
        if self.axis == 'point':
            ref.setFeatureNames(self.source.getFeatureNames())
        else:
            ref.setPointNames(self.source.getPointNames())

        ref._relPath = self.source.relativePath
        ref._absPath = self.source.absolutePath

        self.source.referenceDataFrom(ref)

        self.source.validate()

    def _count(self, condition):
        return self._genericStructuralFrontend('count', condition)

    def _sort(self, sortBy, sortHelper):
        if sortBy is not None and sortHelper is not None:
            msg = "Cannot specify a feature to sort by and a helper function"
            raise ArgumentException(msg)
        if sortBy is None and sortHelper is None:
            msg = "Either sortBy or sortHelper must not be None"
            raise ArgumentException(msg)

        if self.axis == 'point':
            otherAxis = 'feature'
            axisCount = self.source._pointCount
            otherCount = self.source._featureCount
            setNames = self.source.setPointNames
        else:
            otherAxis = 'point'
            axisCount = self.source._featureCount
            otherCount = self.source._pointCount
            setNames = self.source.setFeatureNames

        if sortBy is not None and isinstance(sortBy, six.string_types):
            sortBy = self.source._getIndex(sortBy, otherAxis)

        if sortHelper is not None and not hasattr(sortHelper, '__call__'):
            indices = self.source._constructIndicesList(self.axis, sortHelper)
            if len(indices) != axisCount:
                msg = "This object contains {0} {1}s, "
                msg += "but sortHelper has {2} identifiers"
                msg = msg.format(axisCount, self.axis, len(indices))
                raise ArgumentException(msg)
            if len(indices) != len(set(indices)):
                msg = "This object contains {0} {1}s, "
                msg += "but sortHelper has {2} unique identifiers"
                msg = msg.format(axisCount, self.axis, len(set(indices)))
                raise ArgumentException(msg)

            sortHelper = indices

        # its already sorted in these cases
        if otherCount == 0 or axisCount == 0 or axisCount == 1:
            return

        newNameOrder = self._sort_implementation(sortBy, sortHelper)
        setNames(newNameOrder)

        self.source.validate()

    def _flattenToOne(self):
        if self.source._pointCount == 0 or self.source._featureCount == 0:
            msg = "Can only flattenToOne when there is one or more {0}s. "
            msg += "This object has 0 {0}s."
            msg = msg.format(self.axis)
            raise ImproperActionException(msg)

        # TODO: flatten nameless Objects without the need to generate default
        # names for them.
        if not self.source._pointNamesCreated():
            self.source._setAllDefault('point')
        if not self.source._featureNamesCreated():
            self.source._setAllDefault('feature')

        self._flattenToOne_implementation()

        if self.axis == 'point':
            axisCount = self.source._pointCount
            offAxisCount = self.source._featureCount
            setAxisCount = self.source._setpointCount
            setOffAxisCount = self.source._setfeatureCount
            setAxisNames = self.source.setPointNames
            setOffAxisNames = self.source.setFeatureNames
        else:
            axisCount = self.source._featureCount
            offAxisCount = self.source._pointCount
            setAxisCount = self.source._setfeatureCount
            setOffAxisCount = self.source._setpointCount
            setAxisNames = self.source.setFeatureNames
            setOffAxisNames = self.source.setPointNames

        setOffAxisCount(axisCount * offAxisCount)
        setAxisCount(1)
        setOffAxisNames(self._flattenNames(self.axis))
        setAxisNames(['Flattened'])

    def _unflattenFromOne(self, divideInto):
        if self.axis == 'point':
            offAxis = 'feature'
            axisCount = self.source._pointCount
            offAxisCount = self.source._featureCount
            setAxisCount = self.source._setpointCount
            setOffAxisCount = self.source._setfeatureCount
            setAxisNames = self.source.setPointNames
            setOffAxisNames = self.source.setFeatureNames
        else:
            offAxis = 'point'
            axisCount = self.source._featureCount
            offAxisCount = self.source._pointCount
            setAxisCount = self.source._setfeatureCount
            setOffAxisCount = self.source._setpointCount
            setAxisNames = self.source.setFeatureNames
            setOffAxisNames = self.source.setPointNames

        if offAxisCount == 0:
            msg = "Can only unflattenFromOne when there is one or more "
            msg = "{offAxis}s. This object has 0 {offAxis}s."
            msg = msg.format(offAxis=offAxis)
            raise ImproperActionException(msg)
        if axisCount != 1:
            msg = "Can only unflattenFromOne when there is only one {axis}. "
            msg += "This object has {axisCount} {axis}s."
            msg += msg.format(axis=self.axis, axisCount=axisCount)
            raise ImproperActionException(msg)
        if offAxisCount % divideInto != 0:
            msg = "The argument num{axisCap}s ({divideInto}) must be a "
            msg += "divisor of this object's {offAxis}Count ({offAxisCount}) "
            msg += "otherwise it will not be possible to equally divide the "
            msg += "elements into the desired number of {axis}s."
            msg = msg.format(axisCap=self.axis.capitalize(),
                             divideInto=divideInto, offAxis=offAxis,
                             offAxisCount=offAxisCount, axis=self.axis)
            raise ArgumentException(msg)

        if not self.source._pointNamesCreated():
            self.source._setAllDefault('point')
        if not self.source._featureNamesCreated():
            self.source._setAllDefault('feature')

        self._unflattenFromOne_implementation(divideInto)
        ret = self._unflattenNames(divideInto)

        setOffAxisCount(offAxisCount // divideInto)
        setAxisCount(divideInto)
        setAxisNames(ret[0])
        setOffAxisNames(ret[1])

    def _transform(self, function, limitTo):
        if self.source._pointCount == 0:
            msg = "We disallow this function when there are 0 points"
            raise ImproperActionException(msg)
        if self.source._featureCount == 0:
            msg = "We disallow this function when there are 0 features"
            raise ImproperActionException(msg)
        if function is None:
            raise ArgumentException("function must not be None")
        if limitTo is not None:
            limitTo = self.source._constructIndicesList(self.axis, limitTo)

        self._transform_implementation(function, limitTo)

        self.source.validate()

    ###########################
    # Higher Order Operations #
    ###########################

    def _calculate(self, function, limitTo):
        if limitTo is not None:
            limitTo = copy.copy(limitTo)
            limitTo = self.source._constructIndicesList(self.axis, limitTo)
        if len(self.source.points) == 0:
            msg = "We disallow this function when there are 0 points"
            raise ImproperActionException(msg)
        if len(self.source.features) == 0:
            msg = "We disallow this function when there are 0 features"
            raise ImproperActionException(msg)
        if function is None:
            raise ArgumentException("function must not be None")

        self.source.validate()

        ret = self._calculate_implementation(function, limitTo)

        if self.axis == 'point':
            if limitTo is not None and self.source._pointNamesCreated():
                names = []
                for index in sorted(limitTo):
                    names.append(self.source.getPointName(index))
                ret.setPointNames(names)
            elif self.source._pointNamesCreated():
                ret.setPointNames(self.source.getPointNames())
        else:
            if limitTo is not None and self.source._featureNamesCreated():
                names = []
                for index in sorted(limitTo):
                    names.append(self.source.getFeatureName(index))
                ret.setFeatureNames(names)
            elif self.source._featureNamesCreated():
                ret.setFeatureNames(self.source.getFeatureNames())

        ret._absPath = self.source.absolutePath
        ret._relPath = self.source.relativePath

        return ret

    def _add(self, toAdd, insertBefore):
        _validateInsertableData(self.axis, self.source, toAdd)
        if self.source.getTypeString() != toAdd.getTypeString():
            toAdd = toAdd.copyAs(self.source.getTypeString())

        if self.axis == 'point' and insertBefore is None:
            insertBefore = len(self.source.points)
        elif self.axis == 'feature' and insertBefore is None:
            insertBefore = len(self.source.features)
        else:
            insertBefore = self.source._getIndex(insertBefore, self.axis)

        offAxis = 'feature' if self.axis == 'point' else 'point'
        toAdd = _alignNames(offAxis, self.source, toAdd)
        self._add_implementation(toAdd, insertBefore)

        _setAddedCountAndNames(self.axis, self.source, toAdd, insertBefore)

        self.source.validate()

    def _mapReduce(self, mapper, reducer):
        if self.axis == 'point':
            targetCount = len(self.source.points)
            otherCount = len(self.source.features)
            valueIterator = self.source.pointIterator
            otherAxis = 'feature'
        else:
            targetCount = len(self.source.features)
            otherCount = len(self.source.points)
            valueIterator = self.source.featureIterator
            otherAxis = 'point'

        if targetCount == 0:
            return UML.createData(self.source.getTypeString(),
                                  numpy.empty(shape=(0, 0)))
        if otherCount == 0:
            msg = "We do not allow operations over {0}s if there are 0 {1}s"
            msg = msg.format(self.axis, otherAxis)
            raise ImproperActionException(msg)

        if mapper is None or reducer is None:
            raise ArgumentException("The arguments must not be none")
        if not hasattr(mapper, '__call__'):
            raise ArgumentException("The mapper must be callable")
        if not hasattr(reducer, '__call__'):
            raise ArgumentException("The reducer must be callable")

        self.source.validate()

        mapResults = {}
        # apply the mapper to each point in the data
        for value in valueIterator():
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
        ret = UML.createData(self.source.getTypeString(), ret)

        ret._absPath = self.source.absolutePath
        ret._relPath = self.source.relativePath

        return ret

    def _shuffle(self):
        if self.axis == 'point':
            values = len(self.source.points)
            sorter = self.source.sortPoints
        else:
            values = len(self.source.features)
            sorter = self.source.sortFeatures

        indices = list(range(values))
        pythonRandom.shuffle(indices)

        sorter(sortHelper=indices)

    def _fill(self, toMatch, toFill, arguments=None, limitTo=None,
              returnModified=False):
        modified = None
        toTransform = fill.factory(toMatch, toFill, arguments)

        if returnModified:
            def bools(values):
                return [True if toMatch(val) else False for val in values]
            if self.axis == 'point':
                getNames = 'getPointNames'
                setNames = 'setPointNames'
            else:
                getNames = 'getFeatureNames'
                setNames = 'setFeatureNames'
            modified = self._calculate(bools, limitTo)
            modNames = [n + "_modified" for n in getattr(modified, getNames)()]
            getattr(modified, setNames)(modNames)

        self._transform(toTransform, limitTo)

        self.source.validate()

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
        if subtract is not None:
            if not isinstance(subtract, (int, float, six.string_types, UML.data.Base)):
                msg = "The argument named subtract must have a value that is "
                msg += "an int, float, string, or is a UML data object"
                raise ArgumentException(msg)
        if divide is not None:
            if not isinstance(divide, (int, float, six.string_types, UML.data.Base)):
                msg = "The argument named divide must have a value that is "
                msg += "an int, float, string, or is a UML data object"
                raise ArgumentException(msg)

        # check that if it is a string, it is one of the accepted values
        if isinstance(subtract, six.string_types):
            self.source._validateStatisticalFunctionInputString(subtract)
        if isinstance(divide, six.string_types):
            self.source._validateStatisticalFunctionInputString(divide)

        # arg generic helper to check that objects are of the
        # correct shape/size
        def validateInObjectSize(argname, argval):
            inPC = len(argval.points)
            inFC = len(argval.features)
            objPC = self.source._pointCount
            objFC = self.source._featureCount

            inMainLen = inPC if self.axis == "point" else inFC
            inOffLen = inFC if self.axis == "point" else inPC
            objMainLen = objPC if self.axis == "point" else objFC
            objOffLen = objFC if self.axis == 'point' else objPC

            if inMainLen != objMainLen or inOffLen != objOffLen:
                vecErr = argname + " "
                vecErr += "was a UML object in the shape of a "
                vecErr += "vector (" + str(inPC) + " x "
                vecErr += str(inFC) + "), "
                vecErr += "but the length of long axis did not match "
                vecErr += "the number of " + self.axis + "s in this object ("
                vecErr += str(self.source._pointCount) + ")."
                # treat it as a vector
                if inMainLen == 1:
                    if inOffLen != objMainLen:
                        raise ArgumentException(vecErr)
                    return True
                # treat it as a vector
                elif inOffLen == 1:
                    if inMainLen != objMainLen:
                        raise ArgumentException(vecErr)
                    argval.transpose()
                    return True
                # treat it as a mis-sized object
                else:
                    msg = argname + " "
                    msg += "was a UML obejct with a shape of ("
                    msg += str(inPC) + " x " + str(inFC) + "), "
                    msg += "but it doesn't match the shape of the calling"
                    msg += "object (" + str(objPC) + " x "
                    msg += str(objFC) + ")"
                    raise ArgumentException(msg)
            return False

        def checkAlsoShape(also, objIn):
            """
            Raises an exception if the normalized axis shape doesn't match the
            calling object, or if when subtract of divide takes an object, also
            doesn't match the shape of the caller (this is to be called after)
            the check that the caller's shape matches that of the subtract or
            divide argument.
            """
            offAxis = 'feature' if self.axis == 'point' else 'point'
            callerP = len(self.source.points)
            callerF = len(self.source.features)
            alsoP = len(also.points)
            alsoF = len(also.features)

            callMainLen = callerP if self.axis == "point" else callerF
            alsoMainLen = alsoP if self.axis == "point" else alsoF
            callOffLen = callerF if self.axis == "point" else callerP
            alsoOffLen = alsoF if self.axis == "point" else alsoP

            if callMainLen != alsoMainLen:
                msg = "applyResultTo must have the same number of " + self.axis
                msg += "s (" + str(alsoMainLen) + ") as the calling object "
                msg += "(" + str(callMainLen) + ")"
                raise ArgumentException(msg)
            if objIn and callOffLen != alsoOffLen:
                msg = "When a non-vector UML object is given for the subtract "
                msg += "or divide arguments, then applyResultTo "
                msg += "must have the same number of " + offAxis
                msg += "s (" + str(alsoOffLen) + ") as the calling object "
                msg += "(" + str(callOffLen) + ")"
                raise ArgumentException(msg)

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
        #				subtract = self.pointStatistics(subtract)
        #			else:
        #				subtract = self.featureStatistics(subtract)
        #			subIsVec = True
        #		if isinstance(divide, basestring):
        #			if axis == 'point':
        #				divide = self.pointStatistics(divide)
        #			else:
        #				divide = self.featureStatistics(divide)
        #			divIsVec = True

        if self.axis == 'point':
            indexGetter = lambda x: self._getIndex(x.getPointName(0))
            if isinstance(subtract, six.string_types):
                subtract = self.source.pointStatistics(subtract)
                subIsVec = True
            if isinstance(divide, six.string_types):
                divide = self.source.pointStatistics(divide)
                divIsVec = True
        else:
            indexGetter = lambda x: self._getIndex(x.getFeatureName(0))
            if isinstance(subtract, six.string_types):
                subtract = self.source.featureStatistics(subtract)
                subIsVec = True
            if isinstance(divide, six.string_types):
                divide = self.source.featureStatistics(divide)
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
                if self.axis == 'point':
                    self._transform(subber, None)
                    if alsoIsObj:
                        applyResultTo.points.transform(subber)
                else:
                    self._transform(subber, None)
                    if alsoIsObj:
                        applyResultTo.features.transform(subber)
            else:
                self.source -= subtract
                if alsoIsObj:
                    applyResultTo -= subtract

        # then perform the division operation
        if divide is not None and divide != 1:
            if divIsVec:
                if self.axis == 'point':
                    self._transform(diver, None)
                    if alsoIsObj:
                        applyResultTo.points.transform(diver)
                else:
                    self._transform(diver, None)
                    if alsoIsObj:
                        applyResultTo.features.transform(diver)
            else:
                self.source /= divide
                if alsoIsObj:
                    applyResultTo /= divide

    #####################
    # Low Level Helpers #
    #####################

    def _setName_implementation(self, oldIdentifier, newName):
        if self.axis == 'point':
            names = self.source.pointNames
            invNames = self.source.pointNamesInverse
            index = self.source._getPointIndex(oldIdentifier)
        else:
            names = self.source.featureNames
            invNames = self.source.featureNamesInverse
            index = self.source._getFeatureIndex(oldIdentifier)

        if newName is not None:
            if not isinstance(newName, six.string_types):
                msg = "The new name must be either None or a string"
                raise ArgumentException(msg)

        if newName in names:
            if invNames[index] == newName:
                return
            msg = "This name '" + newName + "' is already in use"
            raise ArgumentException(msg)

        if newName is None:
            newName = self.source._nextDefaultName(self.axis)

        #remove the current featureName
        oldName = invNames[index]
        del names[oldName]

        # setup the new featureName
        invNames[index] = newName
        names[newName] = index
        self.source._incrementDefaultIfNeeded(newName, self.axis)

    def _setNamesFromList(self, assignments, count):
        if self.axis == 'point':
            def checkAndSet(val):
                if val >= self.source._nextDefaultValuePoint:
                    self.source._nextDefaultValuePoint = val + 1
        else:
            def checkAndSet(val):
                if val >= self.source._nextDefaultValueFeature:
                    self.source._nextDefaultValueFeature = val + 1

        if assignments is None:
            self.source._setAllDefault(self.axis)
            return

        if count == 0:
            if len(assignments) > 0:
                msg = "assignments is too large (" + str(len(assignments))
                msg += "); this axis is empty"
                raise ArgumentException(msg)
            self._setNamesFromDict({}, count)
            return
        if len(assignments) != count:
            msg = "assignments may only be an ordered container type, with as "
            msg += "many entries (" + str(len(assignments)) + ") as this axis "
            msg += "is long (" + str(count) + ")"
            raise ArgumentException(msg)

        for name in assignments:
            if name is not None and not isinstance(name, six.string_types):
                msg = 'assignments must contain only string values'
                raise ArgumentException(msg)
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
                name = self.source._nextDefaultName(self.axis)
            if name in temp:
                msg = "Cannot input duplicate names: " + str(name)
                raise ArgumentException(msg)
            temp[name] = index
        assignments = temp

        self._setNamesFromDict(assignments, count)

    def _setNamesFromDict(self, assignments, count):
        if assignments is None:
            self.source._setAllDefault(self.axis)
            return
        if not isinstance(assignments, dict):
            msg = "assignments may only be a dict"
            msg += "with as many entries as this axis is long"
            raise ArgumentException(msg)
        if count == 0:
            if len(assignments) > 0:
                msg = "assignments is too large; this axis is empty"
                raise ArgumentException(msg)
            if self.axis == 'point':
                self.source.pointNames = {}
                self.source.pointNamesInverse = []
            else:
                self.source.featureNames = {}
                self.source.featureNamesInverse = []
            return
        if len(assignments) != count:
            msg = "assignments may only be a dict, "
            msg += "with as many entries as this axis is long"
            raise ArgumentException(msg)

        # at this point, the input must be a dict
        #check input before performing any action
        for name in assignments.keys():
            if not None and not isinstance(name, six.string_types):
                raise ArgumentException("Names must be strings")
            if not isinstance(assignments[name], int):
                raise ArgumentException("Indices must be integers")
            if assignments[name] < 0 or assignments[name] >= count:
                if self.axis == 'point':
                    countName = 'points'
                else:
                    countName = 'features'
                msg = "Indices must be within 0 to "
                msg += "len(self." + countName + ") - 1"
                raise ArgumentException(msg)

        reverseMap = [None] * len(assignments)
        for name in assignments.keys():
            self.source._incrementDefaultIfNeeded(name, self.axis)
            reverseMap[assignments[name]] = name

        # have to copy the input, could be from another object
        if self.axis == 'point':
            self.source.pointNames = copy.deepcopy(assignments)
            self.source.pointNamesInverse = reverseMap
        else:
            self.source.featureNames = copy.deepcopy(assignments)
            self.source.featureNamesInverse = reverseMap

    ########################
    #  Structural Helpers  #
    ########################

    def _genericStructuralFrontend(self, structure, target=None,
                                   start=None, end=None, number=None,
                                   randomize=False):
        axis = self.axis
        if axis == 'point':
            axisLength = len(self.source.points)
            hasNameChecker1 = self.source.hasPointName
            hasNameChecker2 = self.source.hasFeatureName
        else:
            axisLength = len(self.source.features)
            hasNameChecker1 = self.source.hasFeatureName
            hasNameChecker2 = self.source.hasPointName

        _validateStructuralArguments(structure, axis, target, start,
                                     end, number, randomize)
        targetList = []
        if target is not None and isinstance(target, six.string_types):
            # check if target is a valid name
            if hasNameChecker1(target):
                target = self.source._getIndex(target, axis)
                targetList.append(target)
            # if not a name then assume it's a query string
            else:
                target = _stringToFunction(target, self.axis, hasNameChecker2)

        # list-like container types
        if target is not None and not hasattr(target, '__call__'):
            argName = 'to' + structure.capitalize()
            targetList = self.source._constructIndicesList(axis, target,
                                                           argName)
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
                start = self.source._getIndex(start, axis)
            if end is None:
                end = axisLength - 1
            else:
                end = self.source._getIndex(end, axis)
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
                raise ArgumentException(msg)
            if randomize:
                targetList = pythonRandom.sample(targetList, number)
            else:
                targetList = targetList[:number]

        if structure == 'count':
            return len(targetList)
        return self._structuralBackend_implementation(structure, targetList)

    def _flattenNames(self, discardAxis):
        """
        Axis names for the unflattened axis after a flatten operation.
        """
        if discardAxis == 'point':
            keepNames = self.source.getFeatureNames()
            dropNames = self.source.getPointNames()
        else:
            keepNames = self.source.getPointNames()
            dropNames = self.source.getFeatureNames()

        ret = []
        for d in dropNames:
            for k in keepNames:
                ret.append(k + ' | ' + d)

        return ret

    def _unflattenNames(self, addedAxisLength):
        """
        New axis names after an unflattening operation.
        """
        if self.axis == 'point':
            both = self.source.getFeatureNames()
            keptAxisLength = self.source._featureCount // addedAxisLength
        else:
            both = self.source.getPointNames()
            keptAxisLength = self.source._pointCount // addedAxisLength
        allDefault = self._namesAreFlattenFormatConsistent(addedAxisLength,
                                                           keptAxisLength)

        if allDefault:
            addedAxisName = None
            keptAxisName = None
        else:
            # we consider the split of the elements into keptAxisLength chunks
            # (of which there will be addedAxisLength number of chunks), and
            # want the index of the first of each chunk. We allow that first
            # name to be representative for that chunk: all will have the same
            # stuff past the vertical bar.
            locations = range(0, len(both), keptAxisLength)
            addedAxisName = [both[n].split(" | ")[1] for n in locations]
            keptAxisName = [n.split(" | ")[0] for n in both[:keptAxisLength]]

        return addedAxisName, keptAxisName

    def _namesAreFlattenFormatConsistent(self, newFLen, newUFLen):
        """
        Validate the formatting of axis names prior to unflattening.

        Will raise ImproperActionException if an inconsistency with the
        formatting done by the flatten operations is discovered. Returns
        True if all the names along the unflattend axis are default, False
        otherwise.
        """
        if self.axis == 'point':
            flat = self.source.getPointNames()
            formatted = self.source.getFeatureNames()
        else:
            flat = self.source.getFeatureNames()
            formatted = self.source.getPointNames()

        def checkIsDefault(axisName):
            ret = False
            try:
                if axisName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
                    int(axisName[DEFAULT_PREFIX_LENGTH:])
                    ret = True
            except ValueError:
                ret = False
            return ret

        # check the contents of the names along the flattened axis
        isDefault = checkIsDefault(flat[0])
        isExact = flat == ['Flattened']
        msg = "In order to unflatten this object, the names must be "
        msg += "consistent with the results from a flatten call. "
        if not (isDefault or isExact):
            msg += "Therefore, the {axis} name for this object ('{axisName}')"
            msg += "must either be a default name or the string 'Flattened'"
            msg = msg.format(axis=self.axis, axisName=flat[0])
            raise ImproperActionException(msg)

        # check the contents of the names along the unflattend axis
        msg += "Therefore, the {axis} names for this object must either be "
        msg += "all default, or they must be ' | ' split names with name "
        msg += "values consistent with the positioning from a flatten call."
        msg.format(axis=self.axis)
        # each name - default or correctly formatted
        allDefaultStatus = None
        for name in formatted:
            isDefault = checkIsDefault(name)
            formatCorrect = len(name.split(" | ")) == 2
            if allDefaultStatus is None:
                allDefaultStatus = isDefault
            else:
                if isDefault != allDefaultStatus:
                    raise ImproperActionException(msg)

            if not (isDefault or formatCorrect):
                raise ImproperActionException(msg)

        # consistency only relevant if we have non-default names
        if not allDefaultStatus:
            # seen values - consistent wrt original flattend axis names
            for i in range(newFLen):
                same = formatted[newUFLen*i].split(' | ')[1]
                for name in formatted[newUFLen*i:newUFLen*(i+1)]:
                    if same != name.split(' | ')[1]:
                        raise ImproperActionException(msg)

            # seen values - consistent wrt original unflattend axis names
            for i in range(newUFLen):
                same = formatted[i].split(' | ')[0]
                for j in range(newFLen):
                    name = formatted[i + (j * newUFLen)]
                    if same != name.split(' | ')[0]:
                        raise ImproperActionException(msg)

        return allDefaultStatus

    ##########################
    #  Higher Order Helpers  #
    ##########################

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
                    raise ArgumentException(msg)

                toCopyInto = []
                for value in currOut:
                    toCopyInto.append(value)
                retData.append(toCopyInto)
            # singular return
            else:
                retData.append([currOut])

        ret = UML.createData(self.source.getTypeString(), retData)
        if self.axis != 'point':
            ret.transpose()

        return ret

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

    @abstractmethod
    def _flattenToOne_implementation(self):
        pass

    @abstractmethod
    def _unflattenFromOne_implementation(self, divideInto):
        pass

    @abstractmethod
    def _transform_implementation(self, function, limitTo):
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
        raise ArgumentException(msg)
    if number is not None and number < 1:
        msg = "number must be greater than zero"
        raise ArgumentException(msg)
    if number is None and randomize:
        msg = "randomize selects a random subset of "
        msg += "{0}s to {1}. ".format(axis, structure)
        msg += "When randomize=True, the number argument cannot be None"
        raise ArgumentException(msg)
    if target is not None:
        if start is not None or end is not None:
            msg = "Range removal is exclusive, to use it, "
            msg += "{0} must be None".format(targetName)
            raise ArgumentException(msg)

def _validateStartEndRange(start, end, axis, axisLength):
    """
    Check that the start and end values are valid.
    """
    if start < 0 or start > axisLength:
        msg = "start must be a valid index, in the range of possible "
        msg += axis + 's'
        raise ArgumentException(msg)
    if end < 0 or end > axisLength:
        msg = "end must be a valid index, in the range of possible "
        msg += axis + 's'
        raise ArgumentException(msg)
    if start > end:
        msg = "The start index cannot be greater than the end index"
        raise ArgumentException(msg)

def _stringToFunction(string, axis, nameChecker):
    """
    Convert a query string into a python function.
    """
    # to set in for loop
    nameOfPtOrFt = None
    valueOfPtOrFt = None
    optrOperator = None
    for optr in OPTRLIST:
        if optr in string:
            targetList = string.split(optr)
            # user can use '=' but OPTRDICT only contains '=='
            optr = '==' if optr == '=' else optr
            #after splitting at the optr, list must have 2 items
            if len(targetList) != 2:
                msg = "the target({0}) is a ".format(target)
                msg += "query string but there is an error"
                raise ArgumentException(msg)
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
                raise ArgumentException(msg)

            optrOperator = OPTRDICT[optr]
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
        raise ArgumentException(msg)

    return target

def _adjustCountAndNames(source, axis, other):
    """
    Adjust the count and names (when names have been generated) for this
    object, removing the names that have been extracted to the other
    object.
    """
    if axis == 'point':
        source._pointCount -= len(other.points)
        if source._pointNamesCreated():
            idxList = []
            for name in other.getPointNames():
                idxList.append(source.pointNames[name])
            idxList = sorted(idxList)
            for i, val in enumerate(idxList):
                del source.pointNamesInverse[val - i]
            source.pointNames = {}
            for idx, pt in enumerate(source.pointNamesInverse):
                source.pointNames[pt] = idx

    else:
        source._featureCount -= len(other.features)
        if source._featureNamesCreated():
            idxList = []
            for name in other.getFeatureNames():
                idxList.append(source.featureNames[name])
            idxList = sorted(idxList)
            for i, val in enumerate(idxList):
                del source.featureNamesInverse[val - i]
            source.featureNames = {}
            for idx, ft in enumerate(source.featureNamesInverse):
                source.featureNames[ft] = idx

def _alignNames(axis, caller, toAdd):
    """
    Sort the point or feature names of the passed object to match
    this object. If sorting is necessary, a copy will be returned to
    prevent modification of the passed object, otherwise the
    original object will be returned. Assumes validation of the
    names has already occurred.
    """
    caller._validateAxis(axis)
    if axis == 'point':
        namesCreated = caller._pointNamesCreated()
        callerNames = caller.getPointNames
        toAddNames = toAdd.getPointNames
        def sorter(obj, names):
            return getattr(obj, 'sortPoints')(sortHelper=names)
    else:
        namesCreated = caller._featureNamesCreated()
        callerNames = caller.getFeatureNames
        toAddNames = toAdd.getFeatureNames
        def sorter(obj, names):
            return getattr(obj, 'sortFeatures')(sortHelper=names)

    # This may not look exhaustive, but because of the previous call to
    # _validateInsertableData before this helper, most of the toAdd cases will
    # have already caused an exception
    if namesCreated:
        allDefault = all(n.startswith(DEFAULT_PREFIX) for n in callerNames())
        reorder = callerNames() != toAddNames()
        if not allDefault and reorder:
            # use copy when reordering so toAdd object is not modified
            toAdd = toAdd.copy()
            sorter(toAdd, callerNames())

    return toAdd

def _validateInsertableData(axis, caller, toAdd):
    """
    Required validation before inserting an object
    """
    caller._validateAxis(axis)
    caller._validateValueIsNotNone('toAdd', toAdd)
    caller._validateValueIsUMLDataObject('toAdd', toAdd, True)
    if axis == 'point':
        caller._validateObjHasSameNumberOfFeatures('toAdd', toAdd)
        # this helper ignores default names - so we can only have an
        # intersection of names when BOTH objects have names created.
        if caller._pointNamesCreated() and toAdd._pointNamesCreated():
            caller._validateEmptyNamesIntersection(axis, 'toAdd', toAdd)
        # helper looks for name inconsistency that can be resolved by
        # reordering - definitionally, if one object has all default names,
        # there can be no inconsistency, so both objects must have names
        # assigned for this to be relevant.
        if caller._featureNamesCreated() and toAdd._featureNamesCreated():
            caller._validateReorderedNames('feature', 'addPoints', toAdd)
    else:
        caller._validateObjHasSameNumberOfPoints('toAdd', toAdd)
        # this helper ignores default names - so we can only have an
        # intersection of names when BOTH objects have names created.
        if caller._featureNamesCreated() and toAdd._featureNamesCreated():
            caller._validateEmptyNamesIntersection(axis, 'toAdd', toAdd)
        # helper looks for name inconsistency that can be resolved by
        # reordering - definitionally, if one object has all default names,
        # there can be no inconsistency, so both objects must have names
        # assigned for this to be relevant.
        if caller._pointNamesCreated() and toAdd._pointNamesCreated():
            caller._validateReorderedNames('point', 'addFeatures', toAdd)

def _setAddedCountAndNames(axis, caller, addedObj, insertedBefore):
    """
    Modify the point or feature count to include the addedObj. If one or
    both objects have names, names will be set as well.
    """
    caller._validateAxis(axis)
    if axis == 'point':
        # only adjust count if no names in either object
        if not (caller._pointNamesCreated()
                or addedObj._pointNamesCreated()):
            caller._setpointCount(len(caller.points) + len(addedObj.points))
            return
        callerNames = caller.getPointNames()
        insertedNames = addedObj.getPointNames()
        setSelfNames = caller.setPointNames
        caller._setpointCount(len(caller.points) + len(addedObj.points))
    else:
        # only adjust count if no names in either object
        if not (caller._featureNamesCreated()
                or addedObj._featureNamesCreated()):
            newFtCount = len(caller.features) + len(addedObj.features)
            caller._setfeatureCount(newFtCount)
            return
        callerNames = caller.getFeatureNames()
        insertedNames = addedObj.getFeatureNames()
        setSelfNames = caller.setFeatureNames
        caller._setfeatureCount(len(caller.features) + len(addedObj.features))
    # ensure no collision with default names
    adjustedNames = []
    for name in insertedNames:
        if name.startswith(DEFAULT_PREFIX):
            adjustedNames.append(caller._nextDefaultName(axis))
        else:
            adjustedNames.append(name)
    startNames = callerNames[:insertedBefore]
    endNames = callerNames[insertedBefore:]

    newNames = startNames + adjustedNames + endNames
    setSelfNames(newNames)
