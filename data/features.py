"""
Provide feature-based documentation for calls to .features

All feature-axis functions are contained here to customize the function
signatures and docstrings.  The functions here do not contain
the code which provides the functionality for the function. The
functionality component is located in axis.py.
"""
from __future__ import absolute_import

class Features(object):
    """

    """
    def __init__(self, axisObj, source):
        self.obj = axisObj(source)

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
        return self.obj._copy(toCopy, start, end, number, randomize)

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
        return self.obj._extract(toExtract, start, end, number, randomize)

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
        self.obj._delete(toDelete, start, end, number, randomize)

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
        self.obj._retain(toRetain, start, end, number, randomize)

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
        return self.obj._count(condition)

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
        return self.obj._calculate(function, features)

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
        self.obj._add(toAdd, insertBefore)
