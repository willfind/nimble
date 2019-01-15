"""
Provide feature-based documentation for calls to .features

All feature-axis functions are contained here to customize the function
signatures and docstrings.  The functions here do not contain
the code which provides the functionality for the function. The
functionality component is located in axis.py.
"""
from __future__ import absolute_import
from abc import abstractmethod

import numpy
import six

from UML.exceptions import ArgumentException

class Features(object):
    """
    Methods that can be called on the a UML data objects feature axis.
    """
    def __init__(self, source):
        self._source = source
        super(Features, self).__init__()

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
        UML data object

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
        UML data object

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

    # def flattenToOne(self):
    #     """
    #     Modify this object so that its values are in a single feature.
    #
    #     Each point in the result maps to exactly one value from the
    #     original object. The order of values respects the feature order
    #     from the original object, if there were n points in the
    #     original, the first n values in the result will exactly match
    #     the first feature, the nth to (2n-1)th values will exactly
    #     match the original second feature, etc. The point names will be
    #     transformed such that the value at the intersection of the
    #     "pn_i" named point and "fn_j" named feature from the original
    #     object will have a point name of "pn_i | fn_j". The single
    #     feature will have a name of "Flattened". This is an inplace
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
    # def unflattenFromOne(self, numFeatures):
    #     """
    #     Adjust a flattened feature vector to contain multiple features.
    #
    #     This is an inverse of the method ``flattenToOne``: if an
    #     object foo with n features calls the flatten method, then this
    #     method with n as the argument, the result should be identical to
    #     the original foo. It is not limited to objects that have
    #     previously had ``flattenToOne`` called on them; any
    #     object whose structure and names are consistent with a previous
    #     call to flattenToOnePoint may call this method. This includes
    #     objects with all default names. This is an inplace operation.
    #
    #     Parameters
    #     ----------
    #     numFeatures : int
    #         The number of features in the modified object.
    #
    #     See Also
    #     --------
    #     flattenToOneFeature
    #
    #     Examples
    #     --------
    #     TODO
    #     """
    #     self._unflattenFromOne(numFeatures)

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
        UML data object

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
        toAdd : UML data object
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

        See Also
        --------
        match, fill

        Examples
        --------
        TODO
        """
        return self._fill(match, fill, arguments, features, returnModified)

    def normalize(self, subtract=None, divide=None, applyResultTo=None):
        """
        Modify all features in this object using the given operations.

        Normalize the data by applying subtraction and division
        operations. A value of None for subtract or divide implies that
        no change will be made to the data in regards to that operation.

        Parameters
        ----------
        subtract : number, str, UML data object
            * number - a numerical denominator for dividing the data
            * str -  a statistical function (all of the same ones
              callable though featureStatistics)
            * UML data object - If a vector shaped object is given, then
              the value associated with each feature will be subtracted
              from all values of that feature. Otherwise, the values in
              the object are used for elementwise subtraction
        divide : number, str, UML data object
            * number - a numerical denominator for dividing the data
            * str -  a statistical function (all of the same ones
              callable though featureStatistics)
            * UML data object - If a vector shaped object is given, then
              the value associated with each feature will be used in
              division of all values for that feature. Otherwise, the
              values in the object are used for elementwise division.
        applyResultTo : UML data object, statistical method
            If a UML data object is given, then perform the same
            operations to it as are applied to the calling object.
            However, if a statistical method is specified as subtract or
            divide, then concrete values are first calculated only from
            querying the calling object, and the operation is performed
            on applyResultTo using the results; as if a UML data object
            was given for the subtract or divide arguments.

        Examples
        --------
        TODO
        """
        self._normalize(subtract, divide, applyResultTo)

    def splitByParsing(self, feature, rule, resultingNames):
        """
        Split a feature into multiple features.

        Parse an existing feature and divide it into separate parts.
        Each value must split into a number of values equal to the
        length of ``resultingNames``.

        Parameters
        ----------
        feature : indentifier
            The name or index of the feature to parse and split.
        rule : str, int, list, function
            * string - split the value at any instance of the character
              string. This works in the same way as python's built-in
              split() function; removing this string.
            * integer - the index position where the split will occur.
              Unlike a string, no characters will be removed when using
              integer. All characters before the index will be split
              from characters at and after the index.
            * list - may contain integer and/or string values
            * function - any function accepting a value as input,
              splitting the  value and returning a list of the split
              values.
        resultingNames : list
            Strings defining the names of the split features.

        Notes
        -----
        ``
        data.splitFeatureByParsing(location, ', ', ['city', 'country'])

              data (before)                      data (after)
        +-------------------------+      +-----------+--------------+
        | location                |      | city      | country      |
        +-------------------------+      +-----------+--------------+
        | Cape Town, South Africa |      | Cape Town | South Africa |
        +-------------------------+  ->  +-----------+--------------+
        | Lima, Peru              |      | Lima      | Peru         |
        +-------------------------+      +-----------+--------------+
        | Moscow, Russia          |      | Moscow    | Russia       |
        +-------------------------+      +-----------+--------------+

        data.splitFeatureByParsing(0, 3, ['category', 'id'])

            data (before)                        data (after)
        +---------+----------+          +----------+-----+----------+
        | product | quantity |          | category | id  | quantity |
        +---------+----------+          +----------+-----+----------+
        | AGG932  | 44       |          | AGG      | 932 | 44       |
        +---------+----------+          +----------+-----+----------+
        | AGG734  | 11       |    ->    | AGG      | 734 | 11       |
        +---------+----------+          +----------+-----+----------+
        | HEQ892  | 1        |          | HEQ      | 892 | 1        |
        +---------+----------+          +----------+-----+----------+
        | LEQ331  | 2        |          | LEQ      | 331 | 2        |
        +---------+----------+          +----------+-----+----------+
        ``
        This function was inspired by the separate function from the
        tidyr library created by Hadley Wickham in the R programming
        language.

        Examples
        --------
        TODO
        """
        if not (isinstance(rule, (int, numpy.integer, six.string_types))
                or hasattr(rule, '__iter__')
                or hasattr(rule, '__call__')):
            msg = "rule must be an integer, string, iterable of integers "
            msg += "and/or strings, or a function"
            raise ArgumentException(msg)

        splitList = []
        numResultingFts = len(resultingNames)
        for i, value in enumerate(self._source[:, feature]):
            if isinstance(rule, six.string_types):
                splitList.append(value.split(rule))
            elif isinstance(rule, (int, numpy.number)):
                splitList.append([value[:rule], value[rule:]])
            elif hasattr(rule, '__iter__'):
                split = []
                startIdx = 0
                for item in rule:
                    if isinstance(item, six.string_types):
                        split.append(value[startIdx:].split(item)[0])
                        # find index of value from startIdx on, o.w. will only
                        # ever return first instance. Add len of previous
                        # values to get actual index and add one to bypass this
                        # item in next iteration
                        startIdx = (value[startIdx:].index(item) +
                                    len(value[:startIdx]) + 1)
                    elif isinstance(item, (int, numpy.integer)):
                        split.append(value[startIdx:item])
                        startIdx = item
                    else:
                        msg = "A list of items for rule may only contain "
                        msg += " integers and strings"
                        raise ArgumentException(msg)
                split.append(value[startIdx:])
                splitList.append(split)
            else:
                splitList.append(rule(value))
            if len(splitList[-1]) != numResultingFts:
                msg = "The value at index {0} split into ".format(i)
                msg += "{0} values, ".format(len(splitList[-1]))
                msg += "but resultingNames contains "
                msg += "{0} features".format(numResultingFts)
                raise ArgumentException(msg)

        featureIndex = self._source._getFeatureIndex(feature)
        numRetFeatures = len(self._source.features) - 1 + numResultingFts

        self._splitByParsing_implementation(featureIndex, splitList,
                                            numRetFeatures, numResultingFts)

        self._source._featureCount = numRetFeatures
        fNames = self._source.features.getNames()[:featureIndex]
        fNames.extend(resultingNames)
        fNames.extend(self._source.features.getNames()[featureIndex + 1:])
        self._source.features.setNames(fNames)

        self._source.validate()

    ###################
    # Query functions #
    ###################

    def nonZeroIterator(self):
        """
        Iterate through each non-zero value following the feature order.

        Return an iterator for all non-zero elements contained in this
        object, where the values in the same feature will be
        contiguous, with the earlier indexed features coming
        before the later indexed features.

        Examples
        --------
        TODO
        """
        return self._nonZeroIterator()

    def similarities(self, similarityFunction):
        """
        Calculate similarities between features.

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
        UML data object

        Examples
        --------
        TODO
        """
        return self._similarities(similarityFunction)

    def statistics(self, statisticsFunction, groupByFeature=None):
        """
        Calculate feature statistics.

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
        UML data object

        Examples
        --------
        TODO
        """
        return self._statistics(statisticsFunction, groupByFeature)

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

    @abstractmethod
    def _normalize(self, subtract, divide, applyResultTo):
        pass

    @abstractmethod
    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        pass

    @abstractmethod
    def _nonZeroIterator(self):
        pass

    @abstractmethod
    def _similarities(self, similarityFunction):
        pass

    @abstractmethod
    def _statistics(self, statisticsFunction, groupByFeature):
        pass
