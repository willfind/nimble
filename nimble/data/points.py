"""
Define methods of the points attribute for Base objects.

All user-facing, point axis functions are contained here. Functions
specific to only the point axis will provide their functionality here.
However, most functions are applicable to either axis so only the
signatures and docstrings specific to the point axis are provided here.
The functionality of axis generic methods are defined in axis.py, with a
leading underscore added to the method name. Additionally, the wrapping
of function calls for the logger takes place in here.
"""

from __future__ import absolute_import
from abc import abstractmethod
from collections import OrderedDict

from nimble.logger import handleLogging
from nimble.exceptions import ImproperObjectAction

class Points(object):
    """
    Methods that can be called on the a nimble Base objects point axis.
    """
    def __init__(self, source):
        self._source = source
        super(Points, self).__init__()

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
        >>> data = nimble.identity('Matrix', 4,
        ...                        pointNames=['a', 'b', 'c', 'd'])
        >>> data.points.getName(1)
        'b'
        """
        return self._getName(index)

    def getNames(self):
        """
        The point names ordered by index.

        Returns
        -------
        list

        See Also
        --------
        getName, setName, setNames

        Examples
        --------
        >>> data = nimble.identity('Matrix', 4,
        ...                        pointNames=['a', 'b', 'c', 'd'])
        >>> data.points.getNames()
        ['a', 'b', 'c', 'd']
        """
        return self._getNames()

    def setName(self, oldIdentifier, newName, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        setNames

        Examples
        --------
        >>> data = nimble.identity('Matrix', 4,
        ...                        pointNames=['a', 'b', 'c', 'd'])
        >>> data.points.setName('b', 'new')
        >>> data.points.getNames()
        ['a', 'new', 'c', 'd']
        """
        self._setName(oldIdentifier, newName, useLog)

    def setNames(self, assignments=None, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        setName

        Examples
        --------
        >>> data = nimble.identity('Matrix', 4,
        ...                        pointNames=['a', 'b', 'c', 'd'])
        >>> data.points.setNames(['1', '2', '3', '4'])
        >>> data.points.getNames()
        ['1', '2', '3', '4']
        """
        self._setNames(assignments, useLog)

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
        >>> data = nimble.identity('Matrix', 4,
        ...                        pointNames=['a', 'b', 'c', 'd'])
        >>> data.points.getIndex('c')
        2
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
        >>> data = nimble.identity('Matrix', 4,
        ...                        pointNames=['a', 'b', 'c', 'd'])
        >>> data.points.getIndices(['c', 'a', 'd'])
        [2, 0, 3]
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
        >>> data = nimble.identity('Matrix', 4,
        ...                        pointNames=['a', 'b', 'c', 'd'])
        >>> data.points.hasName('a')
        True
        >>> data.points.hasName('e')
        False
        """
        return self._hasName(name)

    #########################
    # Structural Operations #
    #########################

    def copy(self, toCopy=None, start=None, end=None, number=None,
             randomize=False, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Returns
        -------
        nimble Base object

        See Also
        --------
        Base.copy

        Examples
        --------
        >>> raw = [[1, 1, 1, 1],
        ...        [2, 2, 2, 2],
        ...        [3, 3, 3, 3],
        ...        [4, 4, 4, 4]]
        >>> data = nimble.createData('Matrix', raw,
        ...                          featureNames=['a', 'b', 'c', 'd'],
        ...                          pointNames=['1', '2', '3', '4'])
        >>> single = data.points.copy('1')
        >>> single
        Matrix(
            [[1.000 1.000 1.000 1.000]]
            pointNames={'1':0}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> multiple = data.points.copy(['1', 3])
        >>> multiple
        Matrix(
            [[1.000 1.000 1.000 1.000]
             [4.000 4.000 4.000 4.000]]
            pointNames={'1':0, '4':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> func = data.points.copy(lambda pt: sum(pt) < 10)
        >>> func
        Matrix(
            [[1.000 1.000 1.000 1.000]
             [2.000 2.000 2.000 2.000]]
            pointNames={'1':0, '2':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> strFunc = data.points.copy("a>=3")
        >>> strFunc
        Matrix(
            [[3.000 3.000 3.000 3.000]
             [4.000 4.000 4.000 4.000]]
            pointNames={'3':0, '4':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> startEnd = data.points.copy(start=1, end=2)
        >>> startEnd
        Matrix(
            [[2.000 2.000 2.000 2.000]
             [3.000 3.000 3.000 3.000]]
            pointNames={'2':0, '3':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> numberNoRandom = data.points.copy(number=2)
        >>> numberNoRandom
        Matrix(
            [[1.000 1.000 1.000 1.000]
             [2.000 2.000 2.000 2.000]]
            pointNames={'1':0, '2':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> nimble.randomness.setRandomSeed(42)
        >>> numberRandom = data.points.copy(number=2, randomize=True)
        >>> numberRandom
        Matrix(
            [[1.000 1.000 1.000 1.000]
             [4.000 4.000 4.000 4.000]]
            pointNames={'1':0, '4':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        """
        return self._copy(toCopy, start, end, number, randomize, useLog)

    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Returns
        -------
        nimble Base object

        See Also
        --------
        retain, delete

        Examples
        --------
        Extract a single point.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> single = data.points.extract('a')
        >>> single
        Matrix(
            [[1.000 0.000 0.000]]
            pointNames={'a':0}
            )
        >>> data
        Matrix(
            [[0.000 1.000 0.000]
             [0.000 0.000 1.000]]
            pointNames={'b':0, 'c':1}
            )

        Extract multiple points.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> multiple = data.points.extract(['a', 2])
        >>> multiple
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 0.000 1.000]]
            pointNames={'a':0, 'c':1}
            )
        >>> data
        Matrix(
            [[0.000 1.000 0.000]]
            pointNames={'b':0}
            )

        Extract point when the function returns True.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> func = data.points.extract(lambda pt: pt[2] == 1)
        >>> func
        Matrix(
            [[0.000 0.000 1.000]]
            pointNames={'c':0}
            )
        >>> data
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 1.000 0.000]]
            pointNames={'a':0, 'b':1}
            )

        Extract point when the string filter function returns True.

        >>> data = nimble.identity('Matrix', 3,
        ...                        pointNames=['a', 'b', 'c'],
        ...                        featureNames=['f1', 'f2', 'f3'])
        >>> strFunc = data.points.extract("f2 != 0")
        >>> strFunc
        Matrix(
            [[0.000 1.000 0.000]]
            pointNames={'b':0}
            featureNames={'f1':0, 'f2':1, 'f3':2}
            )
        >>> data
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 0.000 1.000]]
            pointNames={'a':0, 'c':1}
            featureNames={'f1':0, 'f2':1, 'f3':2}
            )

        Extract points from the inclusive start to the inclusive end.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> startEnd = data.points.extract(start=1, end=2)
        >>> startEnd
        Matrix(
            [[0.000 1.000 0.000]
             [0.000 0.000 1.000]]
            pointNames={'b':0, 'c':1}
            )
        >>> data
        Matrix(
            [[1.000 0.000 0.000]]
            pointNames={'a':0}
            )

        Select a set number to extract, starting from the first point.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> numberNoRandom = data.points.extract(number=2)
        >>> numberNoRandom
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 1.000 0.000]]
            pointNames={'a':0, 'b':1}
            )
        >>> data
        Matrix(
            [[0.000 0.000 1.000]]
            pointNames={'c':0}
            )

        Select a set number to extract, choosing points at random.

        >>> nimble.randomness.setRandomSeed(42)
        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> numberRandom = data.points.extract(number=2, randomize=True)
        >>> numberRandom
        Matrix(
            [[0.000 0.000 1.000]
             [1.000 0.000 0.000]]
            pointNames={'c':0, 'a':1}
            )
        >>> data
        Matrix(
            [[0.000 1.000 0.000]]
            pointNames={'b':0}
            )
        """
        return self._extract(toExtract, start, end, number, randomize, useLog)

    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        extract, retain

        Examples
        --------
        Delete a single point.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.delete('a')
        >>> data
        Matrix(
            [[0.000 1.000 0.000]
             [0.000 0.000 1.000]]
            pointNames={'b':0, 'c':1}
            )

        Delete multiple points.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.delete(['a', 2])
        >>> data
        Matrix(
            [[0.000 1.000 0.000]]
            pointNames={'b':0}
            )

        Delete point when the function returns True.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.delete(lambda pt: pt[2] == 1)
        >>> data
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 1.000 0.000]]
            pointNames={'a':0, 'b':1}
            )

        Delete point when the string filter function returns True.

        >>> data = nimble.identity('Matrix', 3,
        ...                        pointNames=['a', 'b', 'c'],
        ...                        featureNames=['f1', 'f2', 'f3'])
        >>> data.points.delete("f2 != 0")
        >>> data
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 0.000 1.000]]
            pointNames={'a':0, 'c':1}
            featureNames={'f1':0, 'f2':1, 'f3':2}
            )

        Delete points from the inclusive start to the inclusive end.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.delete(start=1, end=2)
        >>> data
        Matrix(
            [[1.000 0.000 0.000]]
            pointNames={'a':0}
            )

        Select a set number to delete, starting from the first point.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.delete(number=2)
        >>> data
        Matrix(
            [[0.000 0.000 1.000]]
            pointNames={'c':0}
            )

        Select a set number to delete, choosing points at random.

        >>> nimble.randomness.setRandomSeed(42)
        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.delete(number=2, randomize=True)
        >>> data
        Matrix(
            [[0.000 1.000 0.000]]
            pointNames={'b':0}
            )
        """
        self._delete(toDelete, start, end, number, randomize, useLog)

    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        extract, delete

        Examples
        --------
        Retain a single point.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.retain('a')
        >>> data
        Matrix(
            [[1.000 0.000 0.000]]
            pointNames={'a':0}
            )

        Retain multiple points.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.retain(['a', 2])
        >>> data
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 0.000 1.000]]
            pointNames={'a':0, 'c':1}
            )

        Retain point when the function returns True.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.retain(lambda pt: pt[2] == 1)
        >>> data
        Matrix(
            [[0.000 0.000 1.000]]
            pointNames={'c':0}
            )

        Retain point when the string filter function returns True.

        >>> data = nimble.identity('Matrix', 3,
        ...                        pointNames=['a', 'b', 'c'],
        ...                        featureNames=['f1', 'f2', 'f3'])
        >>> data.points.retain("f2 != 0")
        >>> data
        Matrix(
            [[0.000 1.000 0.000]]
            pointNames={'b':0}
            featureNames={'f1':0, 'f2':1, 'f3':2}
            )

        Retain points from the inclusive start to the inclusive end.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.retain(start=1, end=2)
        >>> data
        Matrix(
            [[0.000 1.000 0.000]
             [0.000 0.000 1.000]]
            pointNames={'b':0, 'c':1}
            )

        Select a set number to retain, starting from the first point.

        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.retain(number=2)
        >>> data
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 1.000 0.000]]
            pointNames={'a':0, 'b':1}
            )

        Select a set number to retain, choosing points at random.

        >>> nimble.randomness.setRandomSeed(42)
        >>> data = nimble.identity('Matrix', 3)
        >>> data.points.setNames(['a', 'b', 'c'])
        >>> data.points.retain(number=2, randomize=True)
        >>> data
        Matrix(
            [[0.000 0.000 1.000]
             [1.000 0.000 0.000]]
            pointNames={'c':0, 'a':1}
            )
        """
        self._retain(toRetain, start, end, number, randomize, useLog)

    def count(self, condition):
        """
        The number of points which satisfy the condition.

        Parameters
        ----------
        condition : function
            May take two forms:
            * a function that when given a point will return True if
              it is to be counted
            * a filter function, as a string, containing a comparison
              operator and a value (i.e "ft1<10")

        Returns
        -------
        int

        See Also
        --------
        Elements.count, Elements.countEachUniqueValue

        Examples
        --------
        Count using a python function.

        >>> def sumIsOne(pt):
        ...     return sum(pt) == 1
        >>> data = nimble.identity('List', 3)
        >>> data.points.count(sumIsOne)
        3

        Count when the string filter function returns True.

        >>> data = nimble.identity('List', 3,
        ...                        featureNames=['ft1', 'ft2', 'ft3'])
        >>> data.points.count("ft1 == 0")
        2
        """
        return self._count(condition)

    def sort(self, sortBy=None, sortHelper=None, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        Sort by a given feature using ``sortBy``.

        >>> raw = [['home', 81, 3],
        ...        ['gard', 98, 10],
        ...        ['home', 14, 1],
        ...        ['home', 11, 3]]
        >>> fts = ['dept', 'ID', 'quantity']
        >>> orders = nimble.createData('DataFrame', raw,
        ...                            featureNames=fts)
        >>> orders.points.sort('ID')
        >>> orders
        DataFrame(
            [[home 11 3 ]
             [home 14 1 ]
             [home 81 3 ]
             [gard 98 10]]
            featureNames={'dept':0, 'ID':1, 'quantity':2}
            )

        Sort with a list of identifiers.

        >>> raw = [['home', 81, 3],
        ...        ['gard', 98, 10],
        ...        ['home', 14, 1],
        ...        ['home', 11, 3]]
        >>> pts = ['o_4', 'o_3', 'o_2', 'o_1']
        >>> orders = nimble.createData('DataFrame', raw, pointNames=pts)
        >>> orders.points.sort(sortHelper=['o_1', 'o_2', 'o_3', 'o_4'])
        >>> orders
        DataFrame(
            [[home 11 3 ]
             [home 14 1 ]
             [gard 98 10]
             [home 81 3 ]]
            pointNames={'o_1':0, 'o_2':1, 'o_3':2, 'o_4':3}
            )


        Sort using a comparator function.

        >>> def compareQuantity(pt1, pt2):
        ...     return pt1['quantity'] - pt2['quantity']
        >>> raw = [['home', 81, 3],
        ...        ['gard', 98, 10],
        ...        ['home', 14, 1],
        ...        ['home', 11, 3]]
        >>> fts = ['dept', 'ID', 'quantity']
        >>> orders = nimble.createData('DataFrame', raw,
        ...                            featureNames=fts)
        >>> orders.points.sort(sortHelper=compareQuantity)
        >>> orders
        DataFrame(
            [[home 14 1 ]
             [home 81 3 ]
             [home 11 3 ]
             [gard 98 10]]
            featureNames={'dept':0, 'ID':1, 'quantity':2}
            )

        Sort using a scoring function.

        >>> def scoreQuantity(pt):
        ...     # multiply by -1 to sort starting with highest quantity.
        ...     return pt['quantity'] * -1
        >>> raw = [['home', 81, 3],
        ...        ['gard', 98, 10],
        ...        ['home', 14, 1],
        ...        ['home', 11, 3]]
        >>> fts = ['dept', 'ID', 'quantity']
        >>> orders = nimble.createData('DataFrame', raw,
        ...                            featureNames=fts)
        >>> orders.points.sort(sortHelper=scoreQuantity)
        >>> orders
        DataFrame(
            [[gard 98 10]
             [home 81 3 ]
             [home 11 3 ]
             [home 14 1 ]]
            featureNames={'dept':0, 'ID':1, 'quantity':2}
            )
        """
        self._sort(sortBy, sortHelper, useLog)

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

    def transform(self, function, points=None, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        calculate

        Examples
        --------
        Transform all points; apply to all features.

        >>> data = nimble.ones('Matrix', 3, 5)
        >>> data.points.transform(lambda pt: pt + 2)
        >>> data
        Matrix(
            [[3.000 3.000 3.000 3.000 3.000]
             [3.000 3.000 3.000 3.000 3.000]
             [3.000 3.000 3.000 3.000 3.000]]
            )

        Transform all points; apply to certain features. Note that the
        function recieves a read-only view of each point, so we need to
        make a copy in order to modify any specific data.

        >>> def transformMiddleFeature(pt):
        ...     ptList = pt.copy(to='python list', outputAs1D=True)
        ...     ptList[2] += 4
        ...     return ptList
        >>> data = nimble.ones('Matrix', 3, 5)
        >>> data.points.transform(transformMiddleFeature)
        >>> data
        Matrix(
            [[1.000 1.000 5.000 1.000 1.000]
             [1.000 1.000 5.000 1.000 1.000]
             [1.000 1.000 5.000 1.000 1.000]]
            )

        Transform a subset of points.

        >>> data = nimble.ones('Matrix', 3, 5)
        >>> data.points.transform(lambda pt: pt + 6, points=[0, 2])
        >>> data
        Matrix(
            [[7.000 7.000 7.000 7.000 7.000]
             [1.000 1.000 1.000 1.000 1.000]
             [7.000 7.000 7.000 7.000 7.000]]
            )
        """
        self._transform(function, points, useLog)

    ###########################
    # Higher Order Operations #
    ###########################

    def calculate(self, function, points=None, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Returns
        -------
        nimble Base object

        See also
        --------
        transform

        Examples
        --------
        Apply calculation to all points; apply to all features.

        >>> data = nimble.ones('Matrix', 3, 5)
        >>> addTwo = data.points.calculate(lambda pt: pt + 2)
        >>> addTwo
        Matrix(
            [[3.000 3.000 3.000 3.000 3.000]
             [3.000 3.000 3.000 3.000 3.000]
             [3.000 3.000 3.000 3.000 3.000]]
            )

        Transform all points; apply to certain features. Note that the
        function recieves a read-only view of each point, so we need to
        make a copy in order to modify any specific data.

        >>> def changeMiddleFeature(pt):
        ...     ptList = pt.copy(to='python list', outputAs1D=True)
        ...     ptList[2] += 4
        ...     return ptList
        >>> data = nimble.ones('Matrix', 3, 5)
        >>> changeMiddle = data.points.calculate(changeMiddleFeature)
        >>> changeMiddle
        Matrix(
            [[1.000 1.000 5.000 1.000 1.000]
             [1.000 1.000 5.000 1.000 1.000]
             [1.000 1.000 5.000 1.000 1.000]]
            )

        Transform a subset of points.

        >>> ptNames = ['p1', 'p2', 'p3']
        >>> data = nimble.identity('Matrix', 3, pointNames=ptNames)
        >>> calc = data.points.calculate(lambda pt: pt + 6,
        ...                              points=[2, 0])
        >>> calc
        Matrix(
            [[6.000 6.000 7.000]
             [7.000 6.000 6.000]]
            pointNames={'p3':0, 'p1':1}
            )
        """
        return self._calculate(function, points, useLog)

    def add(self, toAdd, insertBefore=None, useLog=None):
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
        toAdd : nimble Base object
            The nimble Base object whose contents we will be including
            in this object. Must have the same number of features as the
            calling object, but not necessarily in the same order. Must
            not share any point names with the calling object.
        insertBefore : identifier
            The index or point name prior to which the data from
            ``toAdd`` will be inserted. The default value, None,
            indicates that the data will be inserted below all points in
            this object, or in other words: appended to the end of the
            current points.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        Append added data; default names.

        >>> data = nimble.zeros('Matrix', 2, 3)
        >>> toAdd = nimble.ones('Matrix', 2, 3)
        >>> data.points.add(toAdd)
        >>> data
        Matrix(
            [[0.000 0.000 0.000]
             [0.000 0.000 0.000]
             [1.000 1.000 1.000]
             [1.000 1.000 1.000]]
            )

        Reorder names.

        >>> rawData = [[1, 2, 3], [1, 2, 3]]
        >>> data = nimble.createData('Matrix', rawData,
        ...                          featureNames=['a', 'b', 'c'])
        >>> rawAdd = [[3, 2, 1], [3, 2, 1]]
        >>> toAdd = nimble.createData('Matrix', rawAdd,
        ...                           featureNames=['c', 'b', 'a'])
        >>> data.points.add(toAdd)
        >>> data
        Matrix(
            [[1.000 2.000 3.000]
             [1.000 2.000 3.000]
             [1.000 2.000 3.000]
             [1.000 2.000 3.000]]
            featureNames={'a':0, 'b':1, 'c':2}
            )

        Insert before another point; mixed object types.

        >>> rawData = [[1, 1, 1], [4, 4, 4]]
        >>> data = nimble.createData('Matrix', rawData,
        ...                          pointNames=['1', '4'])
        >>> rawAdd = [[2, 2, 2], [3, 3, 3]]
        >>> toAdd = nimble.createData('List', rawAdd,
        ...                           pointNames=['2', '3'])
        >>> data.points.add(toAdd, insertBefore='4')
        >>> data
        Matrix(
            [[1.000 1.000 1.000]
             [2.000 2.000 2.000]
             [3.000 3.000 3.000]
             [4.000 4.000 4.000]]
            pointNames={'1':0, '2':1, '3':2, '4':3}
            )
        """
        self._add(toAdd, insertBefore, useLog)

    def mapReduce(self, mapper, reducer, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        mapReduce the counts of roof styles in the points.

        >>> def roofMapper(pt):
        ...     style = 'Open'
        ...     if pt['ROOF_TYPE'] == 'Dome':
        ...         style = 'Dome'
        ...     return [(style, 1)]
        >>> def roofReducer(style, totals):
        ...     return (style, sum(totals))
        >>> stadiums = [[61500, 'Open', 'Chicago Bears'],
        ...             [71228, 'Dome', 'Atlanta Falcons'],
        ...             [77000, 'Open', 'Kansas City Chiefs'],
        ...             [72968, 'Dome', 'New Orleans Saints'],
        ...             [76500, 'Open', 'Miami Dolphins']]
        >>> fts = ['CAPACITY', 'ROOF_TYPE', 'TEAM']
        >>> data = nimble.createData('Matrix', stadiums,
        ...                          featureNames=fts)
        >>> data.points.mapReduce(roofMapper, roofReducer)
        Matrix(
            [[Open 3]
             [Dome 2]]
            )
        """
        return self._mapReduce(mapper, reducer, useLog)

    def shuffle(self, useLog=None):
        """
        Permute the indexing of the points to a random order.

        Parameters
        ----------
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Notes
        -----
        This relies on python's random.shuffle() so may not be
        sufficiently random for large number of points.
        See random.shuffle()'s documentation.

        Examples
        --------
        >>> nimble.randomness.setRandomSeed(42)
        >>> raw = [[1, 1, 1, 1],
        ...        [2, 2, 2, 2],
        ...        [3, 3, 3, 3],
        ...        [4, 4, 4, 4]]
        >>> data = nimble.createData('DataFrame', raw)
        >>> data.points.shuffle()
        >>> data
        DataFrame(
            [[3.000 3.000 3.000 3.000]
             [2.000 2.000 2.000 2.000]
             [4.000 4.000 4.000 4.000]
             [1.000 1.000 1.000 1.000]]
            )
        """
        self._shuffle(useLog)

    def fill(self, match, fill, points=None, returnModified=False,
             useLog=None, **kwarguments):
        """
        Replace given values in each point with other values.

        Fill matching values within each point with a specified value
        based on the values in that point. The ``fill`` value can be
        a constant or a determined based on unmatched values in the
        point. The match and fill modules in nimble offer common
        functions for these operations.

        Parameters
        ----------
        match : value, list, or function
            * value - a value to locate within each point
            * list - values to locate within each point
            * function - must accept a single value and return True if
              the value is a match. Certain match types can be imported
              from nimble's match module: missing, nonNumeric, zero, etc
        fill : value or function
            * value - a value to fill each matching value in each point
            * function - must be in the format fill(point, match) or
              fill(point, match, arguments) and return the transformed
              point as a list of values. Certain fill methods can be
              imported from nimble's fill module: mean, median, mode,
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        match, fill

        Examples
        --------
        Fill a value with another value.

        >>> raw = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, 'na'],
        ...        [2, 2, 2],
        ...        ['na', 2, 2]]
        >>> data = nimble.createData('Matrix', raw)
        >>> data.points.fill('na', -1)
        >>> data
        Matrix(
            [[1.000  1.000 1.000 ]
             [1.000  1.000 1.000 ]
             [1.000  1.000 -1.000]
             [2.000  2.000 2.000 ]
             [-1.000 2.000 2.000 ]]
            )

        Fill using nimble's match and fill modules; limit to last point.
        Note: None is converted to numpy.nan in nimble.

        >>> from nimble import match
        >>> from nimble import fill
        >>> raw = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, None],
        ...        [2, 2, 2],
        ...        [None, 2, 2]]
        >>> data = nimble.createData('Matrix', raw)
        >>> data.points.fill(match.missing, fill.mode, points=4)
        >>> data
        Matrix(
            [[1.000 1.000 1.000]
             [1.000 1.000 1.000]
             [1.000 1.000  nan ]
             [2.000 2.000 2.000]
             [2.000 2.000 2.000]]
            )
        """
        return self._fill(match, fill, points, returnModified, useLog,
                          **kwarguments)

    def normalize(self, subtract=None, divide=None, applyResultTo=None,
                  useLog=None):
        """
        Modify all points in this object using the given operations.

        Normalize the data by applying subtraction and division
        operations. A value of None for subtract or divide implies that
        no change will be made to the data in regards to that operation.

        Parameters
        ----------
        subtract : number, str, nimble Base object
            * number - a numerical denominator for dividing the data
            * str -  a statistical function (all of the same ones
              callable though points.statistics)
            * nimble Base object - If a vector shaped object is given,
              then the value associated with each point will be
              subtracted from all values of that point. Otherwise, the
              values in the object are used for elementwise subtraction
        divide : number, str, nimble Base object
            * number - a numerical denominator for dividing the data
            * str -  a statistical function (all of the same ones
              callable though pointStatistics)
            * nimble Base object - If a vector shaped object is given,
              then the value associated with each point will be used in
              division of all values for that point. Otherwise, the
              values in the object are used for elementwise division.
        applyResultTo : nimble Base object, statistical method
            If a nimble Base object is given, then perform the same
            operations to it as are applied to the calling object.
            However, if a statistical method is specified as subtract or
            divide, then concrete values are first calculated only from
            querying the calling object, and the operation is performed
            on applyResultTo using the results; as if a nimble Base
            object was given for the subtract or divide arguments.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        TODO
        """
        self._normalize(subtract, divide, applyResultTo, useLog)


    def splitByCollapsingFeatures(self, featuresToCollapse, featureForNames,
                                  featureForValues, useLog=None):
        """
        Separate feature/value pairs into unique points.

        Split each point in this object into k points, one point for
        each featureName/value pair in featuresToCollapse. For all k
        points, the uncollapsed features are copied from the original
        point. The collapsed features are replaced by only two features
        which are filled with a unique featureName/value pair for each
        of the k points. An object containing n points, m features and k
        features-to-collapse will result in this object containing
        (n * m) points and (m - k + 2) features.

        Parameters
        ----------
        featuresToCollapse : list
            Names and/or indices of the features that will be collapsed.
            The first of the two resulting features will contain the
            names of these features. The second resulting feature will
            contain the values of this feature.
        featureForNames : str
            Describe the feature which will contain the collapsed
            feature names.
        featureForValues : str
            Describe the feature which will contain the values from the
            collapsed features.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Notes
        -----
        A visual representation of the Example::

            temp.points.splitByCollapsingFeatures(['jan', 'feb', 'mar'],
                                                  'month', 'temp')

                  temp (before)                     temp (after)
            +------------------------+       +---------------------+
            | city | jan | feb | mar |       | city | month | temp |
            +------+-----+-----+-----+       +------+-------+------+
            | NYC  | 4   | 5   | 10  |       | NYC  | jan   | 4    |
            +------+-----+-----+-----+  -->  +------+-------+------+
            | LA   | 20  | 21  | 21  |       | NYC  | feb   | 5    |
            +------+-----+-----+-----+       +------+-------+------+
            | CHI  | 0   | 2   | 7   |       | NYC  | mar   | 10   |
            +------+-----+-----+-----+       +------+-------+------+
                                             | LA   | jan   | 20   |
                                             +------+-------+------+
                                             | LA   | feb   | 21   |
                                             +------+-------+------+
                                             | LA   | mar   | 21   |
                                             +------+-------+------+
                                             | CHI  | jan   | 0    |
                                             +------+-------+------+
                                             | CHI  | feb   | 2    |
                                             +------+-------+------+
                                             | CHI  | mar   | 7    |
                                             +------+-------+------+

        This function was inspired by the gather function from the tidyr
        library created by Hadley Wickham [1]_ in the R programming
        language.

        References
        ----------
        .. [1] Wickham, H. (2014). Tidy Data. Journal of Statistical
           Software, 59(10), 1 - 23.
           doi:http://dx.doi.org/10.18637/jss.v059.i10

        Examples
        --------
        >>> raw = [['NYC', 4, 5, 10],
        ...        ['LA', 20, 21, 21],
        ...        ['CHI', 0, 2, 7]]
        >>> fts = ['city', 'jan', 'feb', 'mar']
        >>> temp = nimble.createData('Matrix', raw, featureNames=fts)
        >>> temp.points.splitByCollapsingFeatures(['jan', 'feb', 'mar'],
        ...                                       'month', 'temp')
        >>> temp
        Matrix(
            [[NYC jan 4 ]
             [NYC feb 5 ]
             [NYC mar 10]
             [ LA jan 20]
             [ LA feb 21]
             [ LA mar 21]
             [CHI jan 0 ]
             [CHI feb 2 ]
             [CHI mar 7 ]]
            featureNames={'city':0, 'month':1, 'temp':2}
            )
        """
        features = self._source.features
        numCollapsed = len(featuresToCollapse)
        collapseIndices = [self._source.features.getIndex(ft)
                           for ft in featuresToCollapse]
        retainIndices = [idx for idx in range(len(features))
                         if idx not in collapseIndices]
        currNumPoints = len(self)
        currFtNames = [features.getName(idx) for idx in collapseIndices]
        numRetPoints = len(self) * numCollapsed
        numRetFeatures = len(features) - numCollapsed + 2

        self._splitByCollapsingFeatures_implementation(
            featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures)

        self._source._pointCount = numRetPoints
        self._source._featureCount = numRetFeatures
        ftNames = [features.getName(idx) for idx in retainIndices]
        ftNames.extend([featureForNames, featureForValues])
        features.setNames(ftNames, useLog=False)
        if self._source._pointNamesCreated():
            appendedPts = []
            for name in self.getNames():
                for i in range(numCollapsed):
                    appendedPts.append("{0}_{1}".format(name, i))
            self.setNames(appendedPts, useLog=False)

        handleLogging(useLog, 'prep', 'points.splitByCollapsingFeatures',
                      self._source.getTypeString(),
                      Points.splitByCollapsingFeatures, featuresToCollapse,
                      featureForNames, featureForValues)


    def combineByExpandingFeatures(self, featureWithFeatureNames,
                                   featureWithValues, useLog=None):
        """
        Combine points that are identical except at a given feature.

        Combine any points containing matching values at every feature
        except featureWithFeatureNames and featureWithValues. Each
        combined point will expand its features to include a new feature
        for each unique value invfeatureWithFeatureNames. The
        corresponding featureName/value pairs invfeatureWithFeatureNames
        and featureWithValues from each point willvbecome the values for
        the expanded features for the combined points. If a combined
        point lacks a featureName/value pair for any given feature,
        numpy.nan will be assigned as the value at that feature. The
        combined point name will be assigned the point name of the first
        instance of that point, if point names are present.

        An object containing n points with k being unique at every
        feature except featureWithFeatureNames and featureWithValues,
        m features, and j unique values in featureWithFeatureNames will
        be modified to include k points and (m - 2 + j) features.

        Parameters
        ----------
        featureWithFeatureNames : identifier
            The name or index of the feature containing the values that
            will become the names of the features in the combined
            points.
        featureWithValues : identifier
            The name or index of the feature of values that corresponds
            to the values in featureWithFeatureNames.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Notes
        -----
        A visual representation of the Example::

            sprinters.points.combineByExpandingFeatures('dist', 'time')

                 sprinters (before)                 sprinters (after)
            +-----------+------+-------+    +-----------+------+-------+
            | athlete   | dist | time  |    | athlete   | 100m | 200m  |
            +-----------+------+-------+    +-----------+------+-------+
            | Bolt      | 100m | 9.81  |    | Bolt      | 9.81 | 19.78 |
            +-----------+------+-------+ -> +-----------+------+-------+
            | Bolt      | 200m | 19.78 |    | Gatlin    | 9.89 | nan   |
            +-----------+------+-------+    +-----------+------+-------+
            | Gatlin    | 100m | 9.89  |    | de Grasse | 9.91 | 20.02 |
            +-----------+------+-------+    +-----------+------+-------+
            | de Grasse | 200m | 20.02 |
            +-----------+------+-------+
            | de Grasse | 100m | 9.91  |
            +-----------+------+-------+

        This function was inspired by the spread function from the tidyr
        library created by Hadley Wickham [1]_ in the R programming
        language.

        References
        ----------
        .. [1] Wickham, H. (2014). Tidy Data. Journal of Statistical
           Software, 59(10), 1 - 23.
           doi:http://dx.doi.org/10.18637/jss.v059.i10

        Examples
        --------
        >>> raw = [['Bolt', '100m', 9.81],
        ...        ['Bolt', '200m', 19.78],
        ...        ['Gatlin', '100m', 9.89],
        ...        ['de Grasse', '200m', 20.02],
        ...        ['de Grasse', '100m', 9.91]]
        >>> fts = ['athlete', 'dist', 'time']
        >>> sprinters = nimble.createData('Matrix', raw,
        ...                               featureNames=fts)
        >>> sprinters.points.combineByExpandingFeatures('dist', 'time')
        >>> sprinters
        Matrix(
            [[   Bolt   9.810 19.780]
             [  Gatlin  9.890  nan  ]
             [de Grasse 9.910 20.020]]
            featureNames={'athlete':0, '100m':1, '200m':2}
            )
        """
        namesIdx = self._source.features.getIndex(featureWithFeatureNames)
        valuesIdx = self._source.features.getIndex(featureWithValues)
        uncombinedIdx = [i for i in range(len(self._source.features))
                         if i not in (namesIdx, valuesIdx)]

        # using OrderedDict supports point name setting
        unique = OrderedDict()
        pNames = []
        for idx, row in enumerate(iter(self)):
            uncombined = tuple(row[uncombinedIdx])
            if uncombined not in unique:
                unique[uncombined] = {}
                if self._source._pointNamesCreated():
                    pNames.append(self.getName(idx))
            if row[namesIdx] in unique[uncombined]:
                msg = "The point at index {0} cannot be combined ".format(idx)
                msg += "because there is already a value for the feature "
                msg += "{0} in another point which this ".format(row[namesIdx])
                msg += "point would be combined with."
                raise ImproperObjectAction(msg)
            unique[uncombined][row[namesIdx]] = row[valuesIdx]

        uniqueNames = []
        for name in self._source[:, featureWithFeatureNames]:
            if name not in uniqueNames:
                uniqueNames.append(name)
        numRetFeatures = len(self._source.features) + len(uniqueNames) - 2

        self._combineByExpandingFeatures_implementation(unique, namesIdx,
                                                        uniqueNames,
                                                        numRetFeatures)

        self._source._featureCount = numRetFeatures
        self._source._pointCount = len(unique)

        fNames = [self._source.features.getName(i) for i in uncombinedIdx]
        for name in reversed(uniqueNames):
            fNames.insert(namesIdx, name)
        self._source.features.setNames(fNames, useLog=False)

        if self._source._pointNamesCreated():
            self.setNames(pNames, useLog=False)

        handleLogging(useLog, 'prep', 'points.combineByExpandingFeatures',
                      self._source.getTypeString(),
                      Points.combineByExpandingFeatures,
                      featureWithFeatureNames, featureWithValues)

    def duplicate(self, totalCopies=2, copyPointByPoint=False):
        """
        Create an object using copies of this object's points.

        Copies of this object will be stacked vertically. The returned
        object will have the same number of features as this object and
        the number of points will be equal to the number of points in
        this object times ``totalCopies``. If this object contains
        pointNames, each point name will have "_#" appended, where # is
        the number of the copy made.

        Parameters
        ----------
        totalCopies : int
            The number of times a copy of the data in this object will
            be present in the returned object.
        copyPointByPoint : bool
            When False, the default, copies are made as if iterating
            through each point in this object ``totalCopies`` times.
            When True, copies are made as the object is only iterated
            once, making ``totalCopies`` copies of each point before
            iterating to the next point. Note: when this object is a
            single point, the output will be the same either way.

        Returns
        -------
        nimble Base object
            Object containing the copied data.

        Examples
        --------
        Single point

        >>> data = nimble.createData('Matrix', [[1, 2, 3]])
        >>> data.points.setNames(['a'])
        >>> data.points.duplicate(totalCopies=3)
        Matrix(
            [[1.000 2.000 3.000]
             [1.000 2.000 3.000]
             [1.000 2.000 3.000]]
            pointNames={'a_0':0, 'a_1':1, 'a_2':2}
            )

        Two-dimensional, copyPointByPoint is False

        >>> data = nimble.createData('Matrix', [[1, 2, 3], [4, 5, 6]])
        >>> data.points.setNames(['a', 'b'])
        >>> data.points.duplicate(totalCopies=2, copyPointByPoint=False)
        Matrix(
            [[1.000 2.000 3.000]
             [4.000 5.000 6.000]
             [1.000 2.000 3.000]
             [4.000 5.000 6.000]]
            pointNames={'a_0':0, 'b_0':1, 'a_1':2, 'b_1':3}
            )

        Two-dimensional, copyPointByPoint is True

        >>> data = nimble.createData('Matrix', [[1, 2, 3], [4, 5, 6]])
        >>> data.points.setNames(['a', 'b'])
        >>> data.points.duplicate(totalCopies=2, copyPointByPoint=True)
        Matrix(
            [[1.000 2.000 3.000]
             [1.000 2.000 3.000]
             [4.000 5.000 6.000]
             [4.000 5.000 6.000]]
            pointNames={'a_0':0, 'a_1':1, 'b_0':2, 'b_1':3}
            )
        """
        return self._duplicate(totalCopies, copyPointByPoint)

    ####################
    # Query functions #
    ###################

    def nonZeroIterator(self):
        """
        Iterate through each non-zero value following the point order.

        Returns an iterator for all non-zero elements contained in this
        object, where the values in the same point will be
        contiguous, with the earlier indexed points coming
        before the later indexed points.
        """
        return self._nonZeroIterator()

    def unique(self):
        """
        Only the unique points from this object.

        Any repeated points will be removed from the returned object. If
        point names are present, the point name of the first instance of
        the unique point in this object will be assigned.

        Returns
        -------
        nimble Base object
            The object containing only unique points.

        Examples
        --------
        >>> raw = [['a', 1, 3],
        ...        ['b', 5, 6],
        ...        ['b', 7, 1],
        ...        ['a', 1, 3]]
        >>> ptNames = ['p1', 'p2', 'p3', 'p1_copy']
        >>> data = nimble.createData('Matrix', raw, pointNames=ptNames)
        >>> uniquePoints = data.points.unique()
        >>> uniquePoints
        Matrix(
            [[a 1 3]
             [b 5 6]
             [b 7 1]]
            pointNames={'p1':0, 'p2':1, 'p3':2}
            )
        """
        return self._unique()

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
        nimble Base object
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
        nimble Base object
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
    def _setName(self, oldIdentifier, newName, useLog):
        pass

    @abstractmethod
    def _setNames(self, assignments, useLog):
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
    def _copy(self, toCopy, start, end, number, randomize, useLog=None):
        pass

    @abstractmethod
    def _extract(self, toExtract, start, end, number, randomize, useLog=None):
        pass

    @abstractmethod
    def _delete(self, toDelete, start, end, number, randomize, useLog=None):
        pass

    @abstractmethod
    def _retain(self, toRetain, start, end, number, randomize, useLog=None):
        pass

    @abstractmethod
    def _count(self, condition):
        pass

    @abstractmethod
    def _sort(self, sortBy, sortHelper, useLog=None):
        pass

    # @abstractmethod
    # def _flattenToOne(self):
    #     pass
    #
    # @abstractmethod
    # def _unflattenFromOne(self, divideInto):
    #     pass

    @abstractmethod
    def _transform(self, function, limitTo, useLog=None):
        pass

    @abstractmethod
    def _calculate(self, function, limitTo, useLog=None):
        pass

    @abstractmethod
    def _add(self, toAdd, insertBefore, useLog=None):
        pass

    @abstractmethod
    def _mapReduce(self, mapper, reducer, useLog=None):
        pass

    @abstractmethod
    def _shuffle(self, useLog=None):
        pass

    @abstractmethod
    def _fill(self, match, fill, limitTo, returnModified, useLog=None,
              **kwarguments):
        pass

    @abstractmethod
    def _normalize(self, subtract, divide, applyResultTo, useLog=None):
        pass

    @abstractmethod
    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        pass

    @abstractmethod
    def _combineByExpandingFeatures_implementation(
            self, uniqueDict, namesIdx, uniqueNames, numRetFeatures):
        pass

    @abstractmethod
    def _duplicate(self, totalCopies, copyValueByValue):
        pass

    @abstractmethod
    def _nonZeroIterator(self):
        pass

    @abstractmethod
    def _unique(self):
        pass

    @abstractmethod
    def _similarities(self, similarityFunction):
        pass

    @abstractmethod
    def _statistics(self, statisticsFunction):
        pass
