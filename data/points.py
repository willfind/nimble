"""
Provide point-based documentation for calls to .points

All point-axis functions are contained here to customize the function
signatures and docstrings.  The functions here do not contain
the code which provides the functionality for the function. The
functionality component is located in axis.py.
"""
from __future__ import absolute_import
from abc import abstractmethod

class Points(object):
    """
    Methods that can be called on the a UML Base objects point axis.
    """
    def __init__(self):
        pass

    ########################
    # Low Level Operations #
    ########################

    def getName(self, index):
        """
        The name of the point at the provided index.

        Parameters
        ----------
        index : int

        Returns
        -------
        str

        See Also
        --------
        getNames, setName, setNames

        Examples
        --------
        TODO
        """
        return self._getName(index)

    def getNames(self):
        """
        The point names ordered by index.

        Returns
        -------
        lst

        See Also
        --------
        getName, setName, setNames

        Examples
        --------
        TODO
        """
        return self._getNames()

    def setName(self, oldIdentifier, newName):
        """
        Set or change a pointName.

        Set the name of the point at ``oldIdentifier`` with the value of
        ``newName``.

        Parameters
        ----------
        oldIdentifier : str, int
            A string or integer, specifying either a current pointName
            or the index of a current pointName.
        newName : str
            May be either a string not currently in the pointName set,
            or None for an default pointName. newName cannot begin with
            the default prefix.

        See Also
        --------
        setNames

        Examples
        --------
        TODO
        """
        self._setName(oldIdentifier, newName)

    def setNames(self, assignments=None):
        """
        Set or rename all of the point names of this object.

        Set the point names of this object according to the values
        specified by the ``assignments`` parameter. If assignments is
        None, then all point names will be given new default values.

        Parameters
        ----------
        assignments : iterable, dict
            * iterable - Given a list-like container, the mapping
              between names and array indices will be used to define the
              point names.
            * dict - The mapping for each point name in the format
              {name:index}

        See Also
        --------
        setName

        Examples
        --------
        TODO
        """
        self._setNames(assignments)

    def getIndex(self, name):
        """
        The index of a point name.

        Return the index location of the provided point ``name``.

        Parameters
        ----------
        name : str
            The name of a point.

        Returns
        -------
        int

        See Also
        --------
        indices

        Examples
        --------
        TODO
        """
        return self._getIndex(name)

    def getIndices(self, names):
        """
        The indices of a list of point names.

        Return a list of the the index locations of the provided point
        ``names``.

        Parameters
        ----------
        names : list
            The names of points.

        Returns
        -------
        list

        See Also
        --------
        index

        Examples
        --------
        TODO
        """
        return self._getIndices(names)

    def hasName(self, name):
        """
        Determine if point name exists.

        Parameters
        ----------
        names : str
            The name of a point.

        Returns
        -------
        bool

        Examples
        --------
        TODO
        """
        return self._hasName(name)

    #########################
    # Structural Operations #
    #########################

    def copy(self, toCopy=None, start=None, end=None, number=None,
             randomize=False):
        """
        Return a copy of certain points of this object.

        A variety of methods for specifying the points to copy based on
        the provided parameters. If toCopy is not None, start and end
        must be None. If start or end is not None, toCopy must be None.

        Parameters
        ----------
        toCopy : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a point will return True if
              it is to be copied
              b) a filter function, as a string, containing a comparison
              operator between a feature name and a value (i.e "ft1<10")
        start, end : identifier
            Parameters indicating range based copying. Begin the copying
            at the location of ``start``. Finish copying at the
            inclusive ``end`` location. If only one of start and end are
            non-None, the other default to 0 and the number of values in
            each point, respectively.
        number : int
            The quantity of points that are to be copied, the default
            None means unrestricted copying. This can be provided on its
            own (toCopy, start and end are None) to the first ``number``
            of points, or in conjuction with toCopy or  start and end,
            to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen points are determined by point order,
            otherwise it is uniform random across the space of possible
            points.

        Returns
        -------
        UML Base object

        See Also
        --------
        extract, retain, delete, data.copy, data.copyAs

        Examples
        --------
        TODO
        """
        return self._copy(toCopy, start, end, number, randomize)

    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False):
        """
        Move certain points of this object into their own object.

        A variety of methods for specifying the points to extract based
        on the provided parameters. If toExtract is not None, start and
        end must be None. If start or end is not None, toExtract must be
        None.

        Parameters
        ----------
        toExtract : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a point will return True if
              it is to be extracted
              b) a filter function, as a string, containing a comparison
              operator between a feature name and a value (i.e "ft1<10")
        start, end : identifier
            Parameters indicating range based extraction. Begin the
            extraction at the location of ``start``. Finish extracting
            at the inclusive ``end`` location. If only one of start and
            end are non-None, the other default to 0 and the number of
            values in each point, respectively.
        number : int
            The quantity of points that are to be extracted, the
            default None means unrestricted extraction. This can be
            provided on its own (toExtract, start and end are None) to
            the first ``number`` of points, or in conjuction with
            toExtract or  start and end, to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen points are determined by point order,
            otherwise it is uniform random across the space of possible
            points.

        Returns
        -------
        UML Base object

        See Also
        --------
        retain, delete

        Examples
        --------
        TODO
        """
        return self._extract(toExtract, start, end, number, randomize)

    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False):
        """
        Remove certain points from this object.

        A variety of methods for specifying points to delete based on
        the provided parameters. If toDelete is not None, start and end
        must be None. If start or end is not None, toDelete must be
        None.

        Parameters
        ----------
        toDelete : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a point will return True if
              it is to be deleted
              b) a filter function, as a string, containing a comparison
              operator between a feature name and a value (i.e "ft1<10")
        start, end : identifier
            Parameters indicating range based deletion. Begin the
            deletion at the location of ``start``. Finish deleting at
            the inclusive ``end`` location. If only one of start and
            end are non-None, the other default to 0 and the number of
            values in each point, respectively.
        number : int
            The quantity of points that are to be deleted, the
            default None means unrestricted deletion. This can be
            provided on its own (toDelete, start and end are None) to
            the first ``number`` of points, or in conjuction with
            toDelete or  start and end, to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen points are determined by point order,
            otherwise it is uniform random across the space of possible
            points.

        See Also
        --------
        extract, retain

        Examples
        --------
        TODO
        """
        self._delete(toDelete, start, end, number, randomize)

    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False):
        """
        Keep only certain points of this object.

        A variety of methods for specifying points to delete based on
        the provided parameters. If toRetain is not None, start and end
        must be None. If start or end is not None, toRetain must be
        None.

        Parameters
        ----------
        toRetain : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a point will return True if
              it is to be retained
              b) a filter function, as a string, containing a comparison
              operator between a feature name and a value (i.e "ft1<10")
        start, end : identifier
            Parameters indicating range based retention. Begin the
            retention at the location of ``start``. Finish retaining at
            the inclusive ``end`` location. If only one of start and
            end are non-None, the other default to 0 and the number of
            values in each point, respectively.
        number : int
            The quantity of points that are to be retained, the
            default None means unrestricted retained. This can be
            provided on its own (toRetain, start and end are None) to
            the first ``number`` of points, or in conjuction with
            toRetain or  start and end, to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen points are determined by point order,
            otherwise it is uniform random across the space of possible
            points.

        See Also
        --------
        extract, retain

        Examples
        --------
        TODO
        """
        self._retain(toRetain, start, end, number, randomize)

    def count(self, condition):
        """
        The number of points which satisfy the condition.

        Parameters
        ----------
        condition : function
            function - may take two forms:
            a) a function that when given a point will return True if
            it is to be counted
            b) a filter function, as a string, containing a comparison
            operator and a value (i.e "<10")

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
        return self._count(condition)

    def sort(self, sortBy=None, sortHelper=None):
        """
        Arrange the points in this object.

        A variety of methods to sort the points. May define either
        ``sortBy`` or ``sortHelper`` parameter, not both.

        Parameters
        ----------
        sortBy : str
            May indicate the feature to sort by or None if the entire
            point is to be taken as a key.
        sortHelper : list, function
            Either an iterable, list-like object of identifiers (names
            and/or indices), a comparator or a scoring function, or None
            to indicate the natural ordering.

        Examples
        --------
        TODO
        """
        self._sort(sortBy, sortHelper)

    # def flattenToOne(self):
    #     """
    #     Modify this object so that its values are in a single point.
    #
    #     Each feature in the result maps to exactly one value from the
    #     original object. The order of values respects the point order
    #     from the original object, if there were n features in the
    #     original, the first n values in the result will exactly match
    #     the first point, the nth to (2n-1)th values will exactly match
    #     the original second point, etc. The feature names will be
    #     transformed such that the value at the intersection of the
    #     "pn_i" named point and "fn_j" named feature from the original
    #     object will have a feature name of "fn_j | pn_i". The single
    #     point will have a name of "Flattened". This is an inplace
    #     operation.
    #
    #     See Also
    #     --------
    #     unflattenFromOne
    #
    #     Examples
    #     --------
    #     TODO
    #     """
    #     self._flattenToOne()
    #
    # def unflattenFromOne(self, numPoints):
    #     """
    #     Adjust a flattened point vector to contain multiple points.
    #
    #     This is an inverse of the method ``flattenToOne``: if an
    #     object foo with n points calls the flatten method, then this
    #     method with n as the argument, the result should be identical to
    #     the original foo. It is not limited to objects that have
    #     previously had ``flattenToOne`` called on them; any object
    #     whose structure and names are consistent with a previous call to
    #     flattenToOnePoint may call this method. This includes objects
    #     with all default names. This is an inplace operation.
    #
    #     Parameters
    #     ----------
    #     numPoints : int
    #         The number of points in the modified object.
    #
    #     See Also
    #     --------
    #     flattenToOnePoint
    #
    #     Examples
    #     --------
    #     TODO
    #     """
    #     self._unflattenFromOne(numPoints)

    def transform(self, function, points=None):
        """
        Modify this object by applying a function to each point.

        Perform an inplace modification of the data in this object
        through the application of the provided ``function`` to the
        points or subset of points in this object.

        Parameters
        ----------
        function
            Must accept the view of a point as an argument.

        points : identifier, list of identifiers
            May be a single point name or index, an iterable,
            container of point names and/or indices. None indicates
            application to all points.

        See Also
        --------
        calculate : return a new object instead of performing inplace

        Examples
        --------
        TODO
        """
        self._transform(function, points)

    ###########################
    # Higher Order Operations #
    ###########################

    def calculate(self, function, points=None):
        """
        Return a new object with a calculation applied to each point.

        Calculates the results of the given function on the specified
        points in this object, with output values collected into a new
        object that is returned upon completion.

        Parameters
        ----------
        function : function
            Accepts a view of a point as an argument and returns the
            new values in that point.
        points : point, list of points
            The subset of points to limit the calculation to. If None,
            the calculation will apply to all points.

        Returns
        -------
        UML Base object

        See also
        --------
        transform : calculate inplace

        Examples
        --------
        TODO
        """
        return self._calculate(function, points)

    def add(self, toAdd, insertBefore=None):
        """
        Insert more points into this object.

        Expand this object by inserting the points of toAdd prior to the
        insertBefore identifier. The features in toAdd do not need to be
        in the same order as in the calling object; the data will
        automatically be placed using the calling object's feature order
        if there is an unambiguous mapping. toAdd will be unaffected by
        calling this method.

        Parameters
        ----------
        toAdd : UML Base object
            The UML Base object whose contents we will be including
            in this object. Must have the same number of features as the
            calling object, but not necessarily in the same order. Must
            not share any point names with the calling object.
        insertBefore : identifier
            The index or point name prior to which the data from
            ``toAdd`` will be inserted. The default value, None,
            indicates that the data will be inserted below all points in
            this object, or in other words: appended to the end of the
            current points.

        See Also
        --------
        TODO

        Examples
        --------
        TODO
        """
        self._add(toAdd, insertBefore)

    def mapReduce(self, mapper, reducer):
        """
        Apply a mapper and reducer function to this object.

        Return a new object containing the results of the given mapper
        and reducer functions

        Parameters
        ----------
        mapper : function
            Input a point and output an iterable containing two-tuple(s)
            of mapping identifiers and feature values.
        reducer : function
            Input the ``mapper`` output and output a two-tuple
            containing the identifier and the reduced value.

        Examples
        --------
        TODO
        """
        return self._mapReduce(mapper, reducer)

    def shuffle(self):
        """
        Permute the indexing of the points to a random order.

        Notes
        -----
        This relies on python's random.shuffle() so may not be
        sufficiently random for large number of points.
        See random.shuffle()'s documentation.

        Examples
        --------
        TODO
        """
        self._shuffle()

    def fill(self, match, fill, arguments=None, points=None,
             returnModified=False):
        """
        Replace given values in each point with other values.

        Fill matching values within each point with a specified value
        based on the values in that point. The ``fill`` value can be
        a constant or a determined based on unmatched values in the
        point. The match and fill modules in UML offer common functions
        for these operations.

        Parameters
        ----------
        match : value, list, or function
            * value - a value to locate within each point
            * list - values to locate within each point
            * function - must accept a single value and return True if
              the value is a match. Certain match types can be imported
              from UML's match module: missing, nonNumeric, zero, etc.
        fill : value or function
            * value - a value to fill each matching value in each point
            * function - must be in the format fill(point, match) or
              fill(point, match, arguments) and return the transformed
              point as a list of values. Certain fill methods can be
              imported from UML's fill module: mean, median, mode,
              forwardFill, backwardFill, interpolation
        arguments : dict
            Any additional arguments being passed to the fill
            function.
        points : identifier or list of identifiers
            Select specific points to apply fill to. If points is None,
            the fill will be applied to all points.
        returnModified : return an object containing True for the
            modified values in each point and False for unmodified
            values.

        See Also
        --------
        match, fill

        Examples
        --------
        TODO
        """
        return self._fill(match, fill, arguments, points, returnModified)

    def normalize(self, subtract=None, divide=None, applyResultTo=None):
        """
        Modify all points in this object using the given operations.

        Normalize the data by applying subtraction and division
        operations. A value of None for subtract or divide implies that
        no change will be made to the data in regards to that operation.

        Parameters
        ----------
        subtract : number, str, UML Base object
            * number - a numerical denominator for dividing the data
            * str -  a statistical function (all of the same ones
              callable though pointStatistics)
            * UML Base object - If a vector shaped object is given, then
              the value associated with each point will be subtracted
              from all values of that point. Otherwise, the values in
              the object are used for elementwise subtraction
        divide : number, str, UML Base object
            * number - a numerical denominator for dividing the data
            * str -  a statistical function (all of the same ones
              callable though pointStatistics)
            * UML Base object - If a vector shaped object is given, then
              the value associated with each point will be used in
              division of all values for that point. Otherwise, the
              values in the object are used for elementwise division.
        applyResultTo : UML Base object, statistical method
            If a UML Base object is given, then perform the same
            operations to it as are applied to the calling object.
            However, if a statistical method is specified as subtract or
            divide, then concrete values are first calculated only from
            querying the calling object, and the operation is performed
            on applyResultTo using the results; as if a UML Base object
            was given for the subtract or divide arguments.

        Examples
        --------
        TODO
        """
        self._normalize(subtract, divide, applyResultTo)

    def nonZeroIterator(self):
        """
        Iterate through each non-zero value following the point order.

        Returns an iterator for all non-zero elements contained in this
        object, where the values in the same point will be
        contiguous, with the earlier indexed points coming
        before the later indexed points.

        Examples
        --------
        TODO
        """
        return self._nonZeroIterator()

    #########################
    # Statistical functions #
    #########################

    def similarities(self, similarityFunction):
        """
        Calculate similarities between points.

        Return a new object containing the results of the
        ``similarityFunction``.

        Parameters
        ----------
        similarityFunction: str
            The name of the function. The accepted strings include:
            'correlation', 'covariance', 'dot product',
            'sample covariance', and 'population covariance'

        Returns
        -------
        UML Base object

        Examples
        --------
        TODO
        """
        return self._similarities(similarityFunction)

    def statistics(self, statisticsFunction):
        """
        Calculate point statistics.

        Parameters
        ----------
        statisticsFunction: str
            The name of the function. The accepted strings include:
            'max', 'mean', 'median', 'min', 'population std',
            'population standard deviation', 'proportion missing',
            'proportion zero', 'sample standard deviation',
            'sample std', 'standard deviation', 'std', 'unique count'

        Returns
        -------
        UML Base object

        Examples
        --------
        TODO
        """
        return self._statistics(statisticsFunction)

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _getName(self, index):
        pass

    @abstractmethod
    def _getNames(self):
        pass

    @abstractmethod
    def _setName(self, oldIdentifier, newName):
        pass

    @abstractmethod
    def _setNames(self, assignments):
        pass

    @abstractmethod
    def _getIndex(self, name):
        pass

    @abstractmethod
    def _getIndices(self, names):
        pass

    @abstractmethod
    def _hasName(self, name):
        pass

    @abstractmethod
    def _copy(self, toCopy, start, end, number, randomize):
        pass

    @abstractmethod
    def _extract(self, toExtract, start, end, number, randomize):
        pass

    @abstractmethod
    def _delete(self, toDelete, start, end, number, randomize):
        pass

    @abstractmethod
    def _retain(self, toRetain, start, end, number, randomize):
        pass

    @abstractmethod
    def _count(self, condition):
        pass

    @abstractmethod
    def _sort(self, sortBy, sortHelper):
        pass

    # @abstractmethod
    # def _flattenToOne(self):
    #     pass
    #
    # @abstractmethod
    # def _unflattenFromOne(self, divideInto):
    #     pass

    @abstractmethod
    def _transform(self, function, points):
        pass

    @abstractmethod
    def _calculate(self, function, points):
        pass

    @abstractmethod
    def _add(self, toAdd, insertBefore):
        pass

    @abstractmethod
    def _mapReduce(self, mapper, reducer):
        pass

    @abstractmethod
    def _shuffle(self):
        pass

    @abstractmethod
    def _fill(self, match, fill, arguments, limitTo, returnModified):
        pass

    @abstractmethod
    def _normalize(self, subtract, divide, applyResultTo):
        pass

    @abstractmethod
    def _nonZeroIterator(self):
        pass

    @abstractmethod
    def _similarities(self, similarityFunction):
        pass

    @abstractmethod
    def _statistics(self, statisticsFunction):
        pass
