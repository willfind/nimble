"""
TODO
"""
from __future__ import absolute_import
import copy

import six

import UML
from UML.exceptions import ArgumentException, ImproperActionException

class Axis(object):
    """
    TODO
    """
    def __init__(self, **kwds):
        self._position = 0
        self.axis = kwds['axis']
        self.source = kwds['source']
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

    ###########################
    # Higher Order Operations #
    ###########################

    def calculate(self, function, limitTo=None):
        """
        Return an object with a calculation applied to each member.

        Calculates the results of the given function on the specified
        members in this object, with output values collected into a new
        object that is returned upon completion.

        Parameters
        ----------
        function : function
            Accepts a view of a member as an argument and returns the
            new values in that member.
        limitTo : list-like
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
