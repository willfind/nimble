"""
TODO
"""
from __future__ import absolute_import
import copy

import six

import UML
from UML.exceptions import ArgumentException, ImproperActionException
from UML.randomness import pythonRandom
from .dataHelpers import OPTRLIST, OPTRDICT
from .dataHelpers import DEFAULT_PREFIX

class Axis(object):
    """
    TODO
    """
    def __init__(self):
        self._position = 0
        super(Axis, self).__init__()

    def __iter__(self):
        return self

    def next(self):
        """
        Get next item
        """
        if self.axis == 'point':
            while self._position < self.source.pts:
                value = self.source.pointView(self._position)
                self._position += 1
                return value
            raise StopIteration
        else:
            while self._position < self.source.fts:
                value = self.source.featureView(self._position)
                self._position += 1
                return value
            raise StopIteration

    def __next__(self):
        return self.next()

    #########################
    # Structural Operations #
    #########################

    def copy(self, toCopy=None, start=None, end=None, number=None,
             randomize=False):
        """
        Return a copy of certain members of this object.

        A variety of methods for specifying the members to copy based on
        the provided parameters. If toCopy is not None, start and end
        must be None. If start or end is not None, toCopy must be None.

        Parameters
        ----------
        toCopy : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a member will return True if
              it is to be copied
              b) a filter function, as a string, containing a comparison
              operator between a point name and a value
        start, end : identifier
            Parameters indicating range based copying. Begin the copying
            at the location of ``start``. Finish copying at the
            inclusive ``end`` location. If only one of start and end are
            non-None, the other default to 0 and the number of values in
            each member, respectively.
        number : int
            The quantity of members that are to be copied, the default
            None means unrestricted copying. This can be provided on its
            own (toCopy, start and end are None) to the first ``number``
            of members, or in conjuction with toCopy or  start and end,
            to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen members are determined by member order,
            otherwise it is uniform random across the space of possible
            members.

        Returns
        -------
        UML object

        See Also
        --------
        extract, retain, delete, data.copy, data.copyAs

        Examples
        --------
        TODO
        """
        ret = self._genericStructuralFrontend('copy', toCopy, start, end,
                                              number, randomize)

        if self.axis == 'point':
            ret.setFeatureNames(self.source.getFeatureNames())
        else:
            ret.setPointNames(self.source.getPointNames())

        ret._absPath = self.source.absolutePath
        ret._relPath = self.source.relativePath

        return ret

    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False):
        """
        Move certain members of this object into their own object.

        A variety of methods for specifying the members to extract based
        on the provided parameters. If toExtract is not None, start and
        end must be None. If start or end is not None, toExtract must be
        None.

        Parameters
        ----------
        toExtract : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a member will return True if
              it is to be extracted
              b) a filter function, as a string, containing a comparison
              operator between a point name and a value
        start, end : identifier
            Parameters indicating range based extraction. Begin the
            extraction at the location of ``start``. Finish extracting
            at the inclusive ``end`` location. If only one of start and
            end are non-None, the other default to 0 and the number of
            values in each member, respectively.
        number : int
            The quantity of members that are to be extracted, the
            default None means unrestricted extraction. This can be
            provided on its own (toExtract, start and end are None) to
            the first ``number`` of members, or in conjuction with
            toExtract or  start and end, to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen members are determined by member order,
            otherwise it is uniform random across the space of possible
            members.

        Returns
        -------
        UML object

        See Also
        --------
        retain, delete

        Examples
        --------
        TODO
        """
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

    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False):
        """
        Remove certain members from this object.

        A variety of methods for specifying members to delete based on
        the provided parameters. If toDelete is not None, start and end
        must be None. If start or end is not None, toDelete must be
        None.

        Parameters
        ----------
        toDelete : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a member will return True if
              it is to be deleted
              b) a filter function, as a string, containing a comparison
              operator between a point name and a value
        start, end : identifier
            Parameters indicating range based deletion. Begin the
            deletion at the location of ``start``. Finish deleting at
            the inclusive ``end`` location. If only one of start and
            end are non-None, the other default to 0 and the number of
            values in each member, respectively.
        number : int
            The quantity of members that are to be deleted, the
            default None means unrestricted deletion. This can be
            provided on its own (toDelete, start and end are None) to
            the first ``number`` of members, or in conjuction with
            toDelete or  start and end, to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen members are determined by member order,
            otherwise it is uniform random across the space of possible
            members.

        See Also
        --------
        extract, retain

        Examples
        --------
        TODO
        """
        ret = self._genericStructuralFrontend('delete', toDelete, start, end,
                                              number, randomize)
        _adjustCountAndNames(self.source, self.axis, ret)
        self.source.validate()

    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False):
        """
        Keep only certain members of this object.

        A variety of methods for specifying members to delete based on
        the provided parameters. If toRetain is not None, start and end
        must be None. If start or end is not None, toRetain must be
        None.

        Parameters
        ----------
        toRetain : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a member will return True if
              it is to be retained
              b) a filter function, as a string, containing a comparison
              operator between a point name and a value
        start, end : identifier
            Parameters indicating range based retention. Begin the
            retention at the location of ``start``. Finish retaining at
            the inclusive ``end`` location. If only one of start and
            end are non-None, the other default to 0 and the number of
            values in each member, respectively.
        number : int
            The quantity of members that are to be retained, the
            default None means unrestricted retained. This can be
            provided on its own (toRetain, start and end are None) to
            the first ``number`` of members, or in conjuction with
            toRetain or  start and end, to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen members are determined by member order,
            otherwise it is uniform random across the space of possible
            members.

        See Also
        --------
        extract, retain

        Examples
        --------
        TODO
        """
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

    def count(self, condition):
        """
        The number of members which satisfy the condition.

        Parameters
        ----------
        condition : function
            function - may take two forms:
            a) a function that when given a member will return True if
            it is to be counted
            b) a filter function, as a string, containing a comparison
            operator between a point name and a value

        Returns
        -------
        int

        See Also
        --------
        elements.count, elements.countEachUniqueValue

        Examples
        --------
        TODO
        """
        return self._genericStructuralFrontend('count', condition)

    ###########################
    # Higher Order Operations #
    ###########################

    def calculate(self, function, limitTo=None):
        """
        Return a new object with a calculation applied to each member.

        Calculates the results of the given function on the specified
        members in this object, with output values collected into a new
        object that is returned upon completion.

        Parameters
        ----------
        function : function
            Accepts a view of a member as an argument and returns the
            new values in that member.
        limitTo : member, list of members
            The subset of members to limit the calculation to. If None,
            the calculation will apply to all members.

        Returns
        -------
        UML object

        See also
        --------
        transform : calculate inplace

        Examples
        --------
        TODO
        """
        if limitTo is not None:
            limitTo = copy.copy(limitTo)
            limitTo = self.source._constructIndicesList(self.axis, limitTo)
        if self.source.pts == 0:
            msg = "We disallow this function when there are 0 points"
            raise ImproperActionException(msg)
        if self.source.fts == 0:
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

    def add(self, toAdd, insertBefore=None):
        """
        Insert more members into this object.

        Expand this object by inserting the features of toAdd prior to
        the insertBefore identifier. The points in toAdd do not need to
        be in the same order as in the caller object; the data will
        automatically be placed using the caller object's feature order
        if there is an unambiguous mapping. toAdd will be unaffected by
        caller this method.

        Parameters
        ----------
        toAdd : UML object
            The UML data object whose contents we will be including in
            this object. Must have an equal number of members for the
            opposite axis. Must not share any feature names with the
            caller object. Must have the same point names as the
            caller object, but not necessarily in the same order.

        insertBefore : identifier
            The name or index prior to which the data from ``toAdd``
            will be inserted. The default value, None, indicates that
            the data will be inserted to the right of all features in
            this object, or in other words: appended to the end of the
            current features.

        See Also
        --------
        TODO

        Examples
        --------
        TODO
        """
        self._genericAddFrontend(self.axis, toAdd, insertBefore)

    ########################
    #  Structural Helpers  #
    ########################

    def _genericStructuralFrontend(self, structure, target=None,
                                   start=None, end=None, number=None,
                                   randomize=False):
        axis = self.axis
        if axis == 'point':
            axisLength = self.source.pts
            hasNameChecker1 = self.source.hasPointName
            hasNameChecker2 = self.source.hasFeatureName
        else:
            axisLength = self.source.fts
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

    ##########################
    #  Higher Order Helpers  #
    ##########################

    def _calculate_implementation(self, function, included):
        retData = []
        for viewID, view in enumerate(self):
            if included is not None and viewID not in included:
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

    def _genericAddFrontend(self, axis, toAdd, insertBefore):
        """
        Validation and modifications for all insert operations
        """
        self.source._validateAxis(axis)
        _validateInsertableData(axis, self.source, toAdd)
        if self.source.getTypeString() != toAdd.getTypeString():
            toAdd = toAdd.copyAs(self.source.getTypeString())

        if axis == 'point' and insertBefore is None:
            insertBefore = self.source.pts
        elif axis == 'feature' and insertBefore is None:
            insertBefore = self.source.fts
        else:
            insertBefore = self.source._getIndex(insertBefore, axis)

        offAxis = 'feature' if self.axis == 'point' else 'point'
        toAdd = _alignNames(offAxis, self.source, toAdd)
        self._add_implementation(toAdd, insertBefore)

        _setAddedCountAndNames(self.axis, self.source, toAdd, insertBefore)

        self.source.validate()

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
    nameOfMember = None
    valueOfMember = None
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
            nameOfMember = targetList[0]
            valueOfMember = targetList[1]
            nameOfMember = nameOfMember.strip()
            valueOfMember = valueOfMember.strip()

            #when point, check if the feature exists or not
            #when feature, check if the point exists or not
            if not nameChecker(nameOfMember):
                if axis == 'point':
                    offAxis = 'feature'
                else:
                    offAxis = 'point'
                msg = "the {0} ".format(offAxis)
                msg += "'{0}' doesn't exist".format(nameOfMember)
                raise ArgumentException(msg)

            optrOperator = OPTRDICT[optr]
            # convert valueOfMember from a string, if possible
            try:
                valueOfMember = float(valueOfMember)
            except ValueError:
                pass
            #convert query string to a function
            def target_f(x):
                return optrOperator(x[nameOfMember], valueOfMember)

            target_f.vectorized = True
            target_f.nameOfMember = nameOfMember
            target_f.valueOfMember = valueOfMember
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
        source._pointCount -= other.pts
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
        source._featureCount -= other.fts
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
    caller._validateAxis(axis)
    if axis == 'point':
        # only adjust count if no names in either object
        if not (caller._pointNamesCreated()
                or addedObj._pointNamesCreated()):
            caller._setpointCount(caller.pts + addedObj.pts)
            return
        callerNames = caller.getPointNames()
        insertedNames = addedObj.getPointNames()
        setSelfNames = caller.setPointNames
        caller._setpointCount(caller.pts + addedObj.pts)
    else:
        # only adjust count if no names in either object
        if not (caller._featureNamesCreated()
                or addedObj._featureNamesCreated()):
            caller._setfeatureCount(caller.fts + addedObj.fts)
            return
        callerNames = caller.getFeatureNames()
        insertedNames = addedObj.getFeatureNames()
        setSelfNames = caller.setFeatureNames
        caller._setfeatureCount(caller.fts + addedObj.fts)
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
