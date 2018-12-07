"""
TODO
"""
from __future__ import absolute_import
import copy
import operator

import six

import UML
from UML.exceptions import ArgumentException, ImproperActionException
from UML.randomness import pythonRandom

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

        Offers a variety of methods for specifying the members to copy
        based on the provided parameters. If toCopy is not None, start
        and end must be None. If start or end is not None, toCopy must
        be None.

        Parameters
        ----------
        toCopy : identifier, list of identifiers, function
          * identifier - a name or index
          * list of identifiers - an iterable container of identifiers
          * function - may take two forms:
            a) a function that when given a feature will return True if
            it is to be copied
            b) a filter function, as a string, containing a comparison
            operator between a point name and a value (i.e 'member1<10')
        start, end : identifier
            Parameters indicating range based copying. Begin the copying
            at the location of ``start``. Finish copying by including
            the ``end`` location. If only one of start and end are
            non-None, the other default to 0 and self.fts, respectively.
        number : int
            the quantity of features that are to be copied, the default
            None means unrestricted copying. This can be provided on its
            own (toCopy, start and end are None) to the first ``number``
            of members, or in conjuction with toCopy or  start and end,
            to limit their output.
        randomize : bool
            indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen members are determined by member order,
            otherwise it is uniform random across the space of possible
            members.

        Returns
        -------
        UML object

        See Also
        --------
        data.copy, data.copyAs

        Examples
        --------
        TODO
        """
        ret = self._genericStructuralFrontend('copy', toCopy, start,
                                              end, number, randomize)

        if self.axis == 'point':
            ret.setFeatureNames(self.source.getFeatureNames())
        else:
            ret.setPointNames(self.source.getPointNames())

        ret._absPath = self.source.absolutePath
        ret._relPath = self.source.relativePath

        return ret

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

###########
# Helpers #
###########

def _validateStructuralArguments(structure, axis, target, start, end,
                                 number, randomize):
    """
    Check for conflicting and co-dependent arguments
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
    Check that the start and end values are valid
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
    Convert a query string into a python function
    """
    optrDict = {'<=': operator.le, '>=': operator.ge,
                '!=': operator.ne, '==': operator.eq,
                '<': operator.lt, '>': operator.gt}
    # to set in for loop
    nameOfMember = None
    valueOfMember = None
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

            optrOperator = optrDict[optr]
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
