"""
Provide feature-based documentation for calls to .features

All feature-axis functions are contained here to customize the function
signatures and docstrings.  The functions here do not contain
the code which provides the functionality for the function. The
functionality component is located in axis.py.
"""
from __future__ import absolute_import
from abc import abstractmethod

class Features(object):
    """
    Methods that can be called on the a UML data objects feature axis.
    """
    def __init__(self):
        pass

    ########################
    # Low Level Operations #
    ########################

    def getName(self, index):
        """
        The name of the feature at the provided index.

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
        The feature names ordered by index.

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
        Set or change a featureName.

        Set the name of the feature at ``oldIdentifier`` with the value
        of ``newName``.

        Parameters
        ----------
        oldIdentifier : str, int
            A string or integer, specifying either a current featureName
            or the index of a current featureName.
        newName : str
            May be either a string not currently in the featureName set,
            or None for an default featureName. newName cannot begin
            with the default prefix.

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
        Set or rename all of the feature names of this object.

        Set the feature names of this object according to the values
        specified by the ``assignments`` parameter. If assignments is
        None, then all feature names will be given new default values.

        Parameters
        ----------
        assignments : iterable, dict
            * iterable - Given a list-like container, the mapping
              between names and array indices will be used to define the
              feature names.
            * dict - The mapping for each feature name in the format
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
        The index of a feature name.

        Return the index location of the provided feature ``name``.

        Parameters
        ----------
        name : str
            The name of a feature.

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
        The indices of a list of feature names.

        Return a list of the the index locations of the provided feature
        ``names``.

        Parameters
        ----------
        names : list
            The names of features.

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
        Determine if feature name exists.

        Parameters
        ----------
        names : str
            The name of a feature.

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
        Return a copy of certain features of this object.

        A variety of methods for specifying the features to copy based
        on the provided parameters. If toCopy is not None, start and end
        must be None. If start or end is not None, toCopy must be None.

        Parameters
        ----------
        toCopy : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a feature will return True
              if it is to be copied
              b) a filter function, as a string, containing a comparison
              operator between a point name and a value (i.e "pt1<10")
        start, end : identifier
            Parameters indicating range based copying. Begin the copying
            at the location of ``start``. Finish copying at the
            inclusive ``end`` location. If only one of start and end are
            non-None, the other default to 0 and the number of values in
            each feature, respectively.
        number : int
            The quantity of features that are to be copied, the default
            None means unrestricted copying. This can be provided on its
            own (toCopy, start and end are None) to the first ``number``
            of features, or in conjuction with toCopy or  start and end,
            to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen features are determined by feature order,
            otherwise it is uniform random across the space of possible
            features.

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
        return self._copy(toCopy, start, end, number, randomize)

    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False):
        """
        Move certain features of this object into their own object.

        A variety of methods for specifying the features to extract
        based on the provided parameters. If toExtract is not None,
        start and end must be None. If start or end is not None,
        toExtract must be None.

        Parameters
        ----------
        toExtract : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a feature will return True
              if it is to be extracted
              b) a filter function, as a string, containing a comparison
              operator between a point name and a value (i.e "pt1<10")
        start, end : identifier
            Parameters indicating range based extraction. Begin the
            extraction at the location of ``start``. Finish extracting
            at the inclusive ``end`` location. If only one of start and
            end are non-None, the other default to 0 and the number of
            values in each feature, respectively.
        number : int
            The quantity of features that are to be extracted, the
            default None means unrestricted extraction. This can be
            provided on its own (toExtract, start and end are None) to
            the first ``number`` of features, or in conjuction with
            toExtract or  start and end, to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen features are determined by feature order,
            otherwise it is uniform random across the space of possible
            features.

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
        return self._extract(toExtract, start, end, number, randomize)

    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False):
        """
        Remove certain features from this object.

        A variety of methods for specifying features to delete based on
        the provided parameters. If toDelete is not None, start and end
        must be None. If start or end is not None, toDelete must be
        None.

        Parameters
        ----------
        toDelete : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a feature will return True
              if it is to be extracted
              b) a filter function, as a string, containing a comparison
              operator between a point name and a value (i.e "pt1<10")
        start, end : identifier
            Parameters indicating range based deletion. Begin the
            deletion at the location of ``start``. Finish deleting at
            the inclusive ``end`` location. If only one of start and
            end are non-None, the other default to 0 and the number of
            values in each feature, respectively.
        number : int
            The quantity of features that are to be deleted, the
            default None means unrestricted deletion. This can be
            provided on its own (toDelete, start and end are None) to
            the first ``number`` of features, or in conjuction with
            toDelete or  start and end, to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen features are determined by feature order,
            otherwise it is uniform random across the space of possible
            features.

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
        Keep only certain features of this object.

        A variety of methods for specifying features to delete based on
        the provided parameters. If toRetain is not None, start and end
        must be None. If start or end is not None, toRetain must be
        None.

        Parameters
        ----------
        toRetain : identifier, list of identifiers, function
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - may take two forms:
              a) a function that when given a feature will return True
              if it is to be extracted
              b) a filter function, as a string, containing a comparison
              operator between a point name and a value (i.e "pt1<10")
        start, end : identifier
            Parameters indicating range based retention. Begin the
            retention at the location of ``start``. Finish retaining at
            the inclusive ``end`` location. If only one of start and
            end are non-None, the other default to 0 and the number of
            values in each feature, respectively.
        number : int
            The quantity of features that are to be retained, the
            default None means unrestricted retained. This can be
            provided on its own (toRetain, start and end are None) to
            the first ``number`` of features, or in conjuction with
            toRetain or  start and end, to limit their output.
        randomize : bool
            Indicates whether random sampling is to be used in
            conjunction with the number parameter. If randomize is
            False, the chosen features are determined by feature order,
            otherwise it is uniform random across the space of possible
            features.

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
        The number of features which satisfy the condition.

        Parameters
        ----------
        condition : function
            function - may take two forms:
            a) a function that when given a feature will return True if
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
        Arrange the features in this object.

        A variety of methods to sort the features. May define either
        ``sortBy`` or ``sortHelper`` parameter, not both.

        Parameters
        ----------
        sortBy : str
            May indicate the feature to sort by or None if the entire
            feature is to be taken as a key.
        sortHelper : list, function
            Either an iterable, list-like object of identifiers (names
            and/or indices), a comparator or a scoring function, or None
            to indicate the natural ordering.

        Examples
        --------
        TODO
        """
        self._sort(sortBy, sortHelper)

    def flattenToOne(self):
        """
        Modify this object so that its values are in a single feature.

        Each point in the result maps to exactly one value from the
        original object. The order of values respects the feature order
        from the original object, if there were n points in the
        original, the first n values in the result will exactly match
        the first feature, the nth to (2n-1)th values will exactly
        match the original second feature, etc. The point names will be
        transformed such that the value at the intersection of the
        "pn_i" named point and "fn_j" named feature from the original
        object will have a point name of "pn_i | fn_j". The single
        feature will have a name of "Flattened". This is an inplace
        operation.

        See Also
        --------
        unflattenFromOne

        Examples
        --------
        TODO
        """
        self._flattenToOne()

    def unflattenFromOne(self, numFeatures):
        """


        See Also
        --------
        flattenToOne

        Examples
        --------
        TODO
        """
        self._unflattenFromOne(numFeatures)

    def transform(self, function, features=None):
        """
        Modify this object by applying a function to each feature.

        Perform an inplace modification of the data in this object
        through the application of the provided ``function`` to the
        features or subset of features in this object.

        Parameters
        ----------
        function
            Must accept the view of a feature as an argument.

        features : identifier, list of identifiers
            May be a single feature name or index, an iterable,
            container of feature names and/or indices. None indicates
            application to all features.

        See Also
        --------
        calculate : return a new object instead of performing inplace

        Examples
        --------
        TODO
        """
        self._transform(function, features)

    ###########################
    # Higher Order Operations #
    ###########################

    def calculate(self, function, features=None):
        """
        Return a new object with a calculation applied to each feature.

        Calculates the results of the given function on the specified
        features in this object, with output values collected into a new
        object that is returned upon completion.

        Parameters
        ----------
        function : function
            Accepts a view of a feature as an argument and returns the
            new values in that feature.
        features : feature, list of features
            The subset of features to limit the calculation to. If None,
            the calculation will apply to all features.

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
        return self._calculate(function, features)

    def add(self, toAdd, insertBefore=None):
        """
        Insert more features into this object.

        Expand this object by inserting the features of toAdd prior to
        the insertBefore identifier. The points in toAdd do not need to
        be in the same order as in the calling object; the data will
        automatically be placed using the calling object's point order
        if there is an unambiguous mapping. toAdd will be unaffected by
        calling this method.

        Parameters
        ----------
        toAdd : UML object
            The UML data object whose contents we will be including
            in this object. Must have the same point names as the
            calling object, but not necessarily in the same order. Must
            not share any feature names with the calling object.
        insertBefore : identifier
            The index or feature name prior to which the data from
            toAdd will be inserted. The default value, None, indicates
            that the data will be inserted to the right of all features
            in this object, or in other words: appended to the end of
            the current features.

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
            Input a feature and output an iterable containing
            two-tuple(s) of mapping identifiers and feature values.
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
        Permute the indexing of the features to a random order.

        Notes
        -----
        This relies on python's random.shuffle() so may not be
        sufficiently random for large number of features.
        See random.shuffle()'s documentation.

        Examples
        --------
        TODO
        """
        self._shuffle()

    def fill(self, match, fill, arguments=None, features=None,
             returnModified=False):
        """
        Replace given values in each feature with other values.

        Fill matching values within each feature with a specified value
        based on the values in that feature. The ``fill`` value can be
        a constant or a determined based on unmatched values in the
        feature. The match and fill modules in UML offer common
        functions for these operations.

        Parameters
        ----------
        match : value, list, or function
            * value - a value to locate within each feature
            * list - values to locate within each feature
            * function - must accept a single value and return True if
              the value is a match. Certain match types can be imported
              from UML's match module: missing, nonNumeric, zero, etc.
        fill : value or function
            * value - a value to fill each matching value in each
              feature
            * function - must be in the format fill(feature, match) or
              fill(feature, match, arguments) and return the transformed
              feature as a list of values. Certain fill methods can be
              imported from UML's fill module: mean, median, mode,
              forwardFill, backwardFill, interpolation
        arguments : dict
            Any additional arguments being passed to the fill
            function.
        features : identifier or list of identifiers
            Select specific features to apply fill to. If features is
            None, the fill will be applied to all features.
        returnModified : return an object containing True for the
            modified values in each feature and False for unmodified
            values.
        """
        return self._fill(match, fill, arguments, features, returnModified)

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

    @abstractmethod
    def _flattenToOne(self):
        pass

    @abstractmethod
    def _unflattenFromOne(self, divideInto):
        pass

    @abstractmethod
    def _transform(self, function, features):
        pass

    @abstractmethod
    def _calculate(self, function, features):
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
