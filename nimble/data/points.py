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

from abc import abstractmethod
from collections import OrderedDict

import nimble
from nimble.logger import handleLogging
from nimble.exceptions import ImproperObjectAction
from .dataHelpers import limitedTo2D

class Points(object):
    """
    Methods that can be called on a nimble Base objects point axis.

    Parameters
    ----------
    base : Base
        The Base instance that will be queried and modified.
    """
    def __init__(self, base):
        self._base = base
        super(Points, self).__init__()

    def __iter__(self):
        return self._iter()

    def __getitem__(self, key):
        return self._getitem(key)

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

    def setNames(self, assignments, useLog=None):
        """
        Set or rename all of the point names of this object.

        Set the point names of this object according to the values
        specified by the ``assignments`` parameter. If assignments is
        None, then all point names will be given new default values.

        Parameters
        ----------
        assignments : iterable, dict, None
            * iterable - Given a list-like container, the mapping
              between names and array indices will be used to define the
              point names.
            * dict - The mapping for each point name in the format
              {name:index}
            * None - remove names from this object
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

    def getIndex(self, identifier):
        """
        The index of a point name.

        Return the index location of the point ``identifier``. The
        ``identifier`` can be a point name or integer (including
        negative integers).

        Parameters
        ----------
        name : str
            The name of a point.

        Returns
        -------
        int

        See Also
        --------
        getIndices

        Examples
        --------
        >>> data = nimble.identity('Matrix', 4,
        ...                        pointNames=['a', 'b', 'c', 'd'])
        >>> data.points.getIndex('c')
        2
        >>> data.points.getIndex(-1)
        3
        """
        return self._getIndex(identifier)

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
        getIndex

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
            [[1 1 1 1]]
            pointNames={'1':0}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> multiple = data.points.copy(['1', 3])
        >>> multiple
        Matrix(
            [[1 1 1 1]
             [4 4 4 4]]
            pointNames={'1':0, '4':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> func = data.points.copy(lambda pt: sum(pt) < 10)
        >>> func
        Matrix(
            [[1 1 1 1]
             [2 2 2 2]]
            pointNames={'1':0, '2':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> strFunc = data.points.copy("a>=3")
        >>> strFunc
        Matrix(
            [[3 3 3 3]
             [4 4 4 4]]
            pointNames={'3':0, '4':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> startEnd = data.points.copy(start=1, end=2)
        >>> startEnd
        Matrix(
            [[2 2 2 2]
             [3 3 3 3]]
            pointNames={'2':0, '3':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> numberNoRandom = data.points.copy(number=2)
        >>> numberNoRandom
        Matrix(
            [[1 1 1 1]
             [2 2 2 2]]
            pointNames={'1':0, '2':1}
            featureNames={'a':0, 'b':1, 'c':2, 'd':3}
            )
        >>> nimble.randomness.setRandomSeed(42)
        >>> numberRandom = data.points.copy(number=2, randomize=True)
        >>> numberRandom
        Matrix(
            [[1 1 1 1]
             [4 4 4 4]]
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

    @limitedTo2D
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
        Base.countElements, Base.countUniqueElements

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

    @limitedTo2D
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
    @limitedTo2D
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

        See Also
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

        Apply calculation to all points; function modifies a specific
        feature. Note that the function recieves a read-only view of
        each point, so a copy is necessary to modify any specific data.

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

        Apply calculation to a subset of points.

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

    @limitedTo2D
    def matching(self, function, useLog=None):
        """
        Return a boolean value object identifying matching points.

        Apply a function returning a boolean value for each point in
        this object. Common any/all matching functions can be found in
        nimble's match module. Note that the featureName in the returned
        object will be set to the ``__name__`` attribute of ``function``
        unless it is a ``lambda`` function.

        Parameters
        ----------
        function : function
            * function - in the form of function(pointView) which
              returns True, False, 0 or 1.

        Returns
        -------
        nimble Base object
            A feature vector of boolean values.

        Examples
        --------
        >>> from nimble import match
        >>> raw = [[1, -1, 1], [-3, 3, 3], [5, 5, 5]]
        >>> data = nimble.createData('Matrix', raw)
        >>> allPositivePts = data.points.matching(match.allPositive)
        >>> allPositivePts
        Matrix(
            [[False]
             [False]
             [ True]]
            featureNames={'allPositive':0}
            )

        >>> from nimble import match
        >>> raw = [[1, -1, float('nan')], [-3, 3, 3], [5, 5, 5]]
        >>> data = nimble.createData('Matrix', raw)
        >>> ptHasMissing = data.points.matching(match.anyMissing)
        >>> ptHasMissing
        Matrix(
            [[ True]
             [False]
             [False]]
            featureNames={'anyMissing':0}
            )
        """
        return self._matching(function, useLog)

    def insert(self, insertBefore, toInsert, useLog=None):
        """
        Insert more points into this object.

        Expand this object by inserting the points of ``toInsert`` prior
        to the ``insertBefore`` identifier. The features in ``toInsert``
        do not need to be in the same order as in the calling object;
        the data will automatically be placed using the calling object's
        feature order if there is an unambiguous mapping. ``toInsert``
        will be unaffected by calling this method.

        Parameters
        ----------
        insertBefore : identifier
            The index or point name prior to which the data from
            ``toInsert`` will be inserted.
        toInsert : nimble Base object
            The nimble Base object whose contents we will be including
            in this object. Must have the same number of features as the
            calling object, but not necessarily in the same order. Must
            not share any point names with the calling object.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        append

        Examples
        --------
        Insert data; default names.

        >>> data = nimble.zeros('Matrix', 2, 3)
        >>> toInsert = nimble.ones('Matrix', 2, 3)
        >>> data.points.insert(1, toInsert)
        >>> data
        Matrix(
            [[0.000 0.000 0.000]
             [1.000 1.000 1.000]
             [1.000 1.000 1.000]
             [0.000 0.000 0.000]]
            )

        Insert before another point; mixed object types.

        >>> rawData = [[1, 1, 1], [4, 4, 4]]
        >>> data = nimble.createData('Matrix', rawData,
        ...                          pointNames=['1', '4'])
        >>> rawInsert = [[2, 2, 2], [3, 3, 3]]
        >>> toInsert = nimble.createData('List', rawInsert,
        ...                           pointNames=['2', '3'])
        >>> data.points.insert('4', toInsert)
        >>> data
        Matrix(
            [[1 1 1]
             [2 2 2]
             [3 3 3]
             [4 4 4]]
            pointNames={'1':0, '2':1, '3':2, '4':3}
            )

        Reorder names.

        >>> rawData = [[1, 2, 3], [1, 2, 3]]
        >>> data = nimble.createData('Matrix', rawData,
        ...                          featureNames=['a', 'b', 'c'])
        >>> rawInsert = [[3, 2, 1], [3, 2, 1]]
        >>> toInsert = nimble.createData('Matrix', rawInsert,
        ...                              featureNames=['c', 'b', 'a'])
        >>> data.points.insert(0, toInsert)
        >>> data
        Matrix(
            [[1 2 3]
             [1 2 3]
             [1 2 3]
             [1 2 3]]
            featureNames={'a':0, 'b':1, 'c':2}
            )
        """
        self._insert(insertBefore, toInsert, False, useLog)

    def append(self, toAppend, useLog=None):
        """
        Append points to this object.

        Expand this object by appending the points of ``toAppend`` to
        the end of the object. The features in ``toAppend`` do not need
        to be in the same order as in the calling object; the data will
        automatically be placed using the calling object's feature order
        if there is an unambiguous mapping. ``toAppend`` will be
        unaffected by calling this method.

        Parameters
        ----------
        toAppend : nimble Base object
            The nimble Base object whose contents we will be including
            in this object. Must have the same number of features as the
            calling object, but not necessarily in the same order. Must
            not share any point names with the calling object.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        insert

        Examples
        --------
        Append data; default names.

        >>> data = nimble.zeros('Matrix', 2, 3)
        >>> toAppend = nimble.ones('Matrix', 2, 3)
        >>> data.points.append(toAppend)
        >>> data
        Matrix(
            [[0.000 0.000 0.000]
             [0.000 0.000 0.000]
             [1.000 1.000 1.000]
             [1.000 1.000 1.000]]
            )

        Append mixed object types.

        >>> rawData = [[1, 1, 1], [2, 2, 2]]
        >>> data = nimble.createData('Matrix', rawData,
        ...                          pointNames=['1', '2'])
        >>> rawAppend = [[3, 3, 3], [4, 4, 4]]
        >>> toAppend = nimble.createData('List', rawAppend,
        ...                           pointNames=['3', '4'])
        >>> data.points.append(toAppend)
        >>> data
        Matrix(
            [[1 1 1]
             [2 2 2]
             [3 3 3]
             [4 4 4]]
            pointNames={'1':0, '2':1, '3':2, '4':3}
            )

        Reorder names.

        >>> rawData = [[1, 2, 3], [1, 2, 3]]
        >>> data = nimble.createData('Matrix', rawData,
        ...                          featureNames=['a', 'b', 'c'])
        >>> rawAppend = [[3, 2, 1], [3, 2, 1]]
        >>> toAppend = nimble.createData('Matrix', rawAppend,
        ...                              featureNames=['c', 'b', 'a'])
        >>> data.points.append(toAppend)
        >>> data
        Matrix(
            [[1 2 3]
             [1 2 3]
             [1 2 3]
             [1 2 3]]
            featureNames={'a':0, 'b':1, 'c':2}
            )
        """
        self._insert(None, toAppend, True, useLog)

    @limitedTo2D
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

    def permute(self, order=None, useLog=None):
        """
        Permute the indexing of the points.

        Change the arrangement of points in the object. A specific
        permutation can be provided as the ``order`` argument. If
        ``order`` is None, the permutation will be random. Note: a
        random permutation may be the same as the current permutation.

        Parameters
        ----------
        order : list, None
            A list of identifiers indicating the new permutation order.
            If None, the permutation will be random.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Notes
        -----
        Random permutation relies on python's random.shuffle() which may
        not be sufficiently random for large number of points.
        See random.shuffle()'s documentation.

        Examples
        --------
        >>> nimble.randomness.setRandomSeed(42)
        >>> raw = [[1, 1, 1, 1],
        ...        [2, 2, 2, 2],
        ...        [3, 3, 3, 3],
        ...        [4, 4, 4, 4]]
        >>> data = nimble.createData('DataFrame', raw)
        >>> data.points.permute()
        >>> data
        DataFrame(
            [[3 3 3 3]
             [2 2 2 2]
             [4 4 4 4]
             [1 1 1 1]]
            )

        Permute with a list of identifiers.

        >>> raw = [['home', 81, 3],
        ...        ['gard', 98, 10],
        ...        ['home', 14, 1],
        ...        ['home', 11, 3]]
        >>> pts = ['o_4', 'o_3', 'o_2', 'o_1']
        >>> orders = nimble.createData('DataFrame', raw, pointNames=pts)
        >>> orders.points.permute(['o_1', 'o_2', 'o_3', 'o_4'])
        >>> orders
        DataFrame(
            [[home 11 3 ]
             [home 14 1 ]
             [gard 98 10]
             [home 81 3 ]]
            pointNames={'o_1':0, 'o_2':1, 'o_3':2, 'o_4':3}
            )
        """
        self._permute(order, useLog)

    @limitedTo2D
    def fillMatching(self, fillWith, matchingElements, points=None,
                     useLog=None, **kwarguments):
        """
        Replace given values in each point with other values.

        Fill matching values within each point with a specified value
        based on the values in that point. The ``fillWith`` value can be
        a constant or a determined based on unmatched values in the
        point. The match and fill modules in nimble offer common
        functions for these operations.

        Parameters
        ----------
        fillWith : value or function
            * value - a value to fill each matching value in each point
            * function - must be in the format:
              fillWith(point, matchingElements) or
              fillWith(point, matchingElements, \*\*kwarguments)
              and return the transformed point as a list of values.
              Certain fill methods can be imported from nimble's fill
              module.
        matchingElements : value, list, or function
            * value - a value to locate within each point
            * list - values to locate within each point
            * function - must accept a single value and return True if
              the value is a match. Certain match types can be imported
              from nimble's match module.
        points : identifier, list of identifiers, None
            Select specific points to apply the fill to. If points is
            None, the fill will be applied to all points.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        kwarguments
            Provide additional parameters to a ``fillWith`` function.

        See Also
        --------
        nimble.fill, nimble.match

        Examples
        --------
        Fill a value with another value.

        >>> raw = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, 'na'],
        ...        [2, 2, 2],
        ...        ['na', 2, 2]]
        >>> data = nimble.createData('Matrix', raw)
        >>> data.points.fillMatching(-1, 'na')
        >>> data
        Matrix(
            [[1  1 1 ]
             [1  1 1 ]
             [1  1 -1]
             [2  2 2 ]
             [-1 2 2 ]]
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
        >>> data.points.fillMatching(fill.mode, match.missing, points=4)
        >>> data
        Matrix(
            [[1 1  1 ]
             [1 1  1 ]
             [1 1 nan]
             [2 2  2 ]
             [2 2  2 ]]
            )
        """
        return self._fillMatching(fillWith, matchingElements, points,
                                  useLog,  **kwarguments)

    @limitedTo2D
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

    @limitedTo2D
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
        features = self._base.features
        numCollapsed = len(featuresToCollapse)
        collapseIndices = [self._base.features.getIndex(ft)
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

        self._base._pointCount = numRetPoints
        self._base._featureCount = numRetFeatures
        ftNames = [features.getName(idx) for idx in retainIndices]
        ftNames.extend([featureForNames, featureForValues])
        features.setNames(ftNames, useLog=False)
        if self._base._pointNamesCreated():
            appendedPts = []
            for name in self.getNames():
                for i in range(numCollapsed):
                    appendedPts.append("{0}_{1}".format(name, i))
            self.setNames(appendedPts, useLog=False)

        handleLogging(useLog, 'prep', 'points.splitByCollapsingFeatures',
                      self._base.getTypeString(),
                      Points.splitByCollapsingFeatures, featuresToCollapse,
                      featureForNames, featureForValues)

    @limitedTo2D
    def combineByExpandingFeatures(self, featureWithFeatureNames,
                                   featuresWithValues, useLog=None):
        """
        Combine similar points based on a differentiating feature.

        Combine points that share common features, but are currently
        separate points due to a feature, ``featureWithFeatureNames``,
        that is categorizing the values in the remaining unshared
        features, ``featuresWithValues``. The points can be combined to
        a single point by instead representing the categorization of the
        unshared values as different features of the same point. The
        corresponding featureName/value pairs from
        ``featureWithFeatureNames`` and ``featuresWithValues`` in each
        point will become the values for the expanded features for the
        combined points. If a combined point lacks a featureName/value
        pair for any given feature(s), numpy.nan will be assigned as the
        value(s) at that feature(s). The resulting featureNames depends
        on the number of features with values. For a single feature with
        values, the new feature names are the unique values in
        ``featureWithFeatureNames``. However, for two or more features
        with values, the unique values in ``featureWithFeatureNames`` no
        longer cover all combinations so those values are combined with
        the names of the features with values using an underscore. The
        combined point name will be assigned the point name of the first
        instance of that point, if point names are present.

        An object containing m features and n points with k unique
        point combinations amongst shared features, i unique values in
        ``featureWithFeatureNames`` and j ``featuresWithValues`` will
        result in an object with k points and (m - (1 + j) + (i * j))
        features.

        Parameters
        ----------
        featureWithFeatureNames : identifier
            The name or index of the feature containing the values that
            will become the names of the features in the combined
            points.
        featuresWithValues : identifier, list of identifiers
            The names and/or indices of the features of values that
            correspond to the values in ``featureWithFeatureNames``.
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
        namesIdx = self._base.features.getIndex(featureWithFeatureNames)
        if not isinstance(featuresWithValues, list):
            featuresWithValues = [featuresWithValues]
        valuesIdx = [self._base.features.getIndex(idx) for idx
                     in featuresWithValues]
        combinedIdx = [namesIdx] + valuesIdx
        uncombinedIdx = [i for i in range(len(self._base.features))
                         if i not in combinedIdx]

        # using OrderedDict supports point name setting
        unique = OrderedDict()
        pNames = []
        uniqueNames = []
        for idx, row in enumerate(iter(self)):
            uncombined = tuple(row[uncombinedIdx])
            if uncombined not in unique:
                unique[uncombined] = {}
                if self._base._pointNamesCreated():
                    pNames.append(self.getName(idx))
            nameIdxVal = row[namesIdx]
            if nameIdxVal in unique[uncombined]:
                msg = "The point at index {0} cannot be combined ".format(idx)
                msg += "because there is already a value for the feature "
                msg += "{0} in another point which this ".format(nameIdxVal)
                msg += "point would be combined with."
                raise ImproperObjectAction(msg)
            if nameIdxVal not in uniqueNames:
                uniqueNames.append(nameIdxVal)

            unique[uncombined][row[namesIdx]] = row[valuesIdx]

        numExpanded = len(featuresWithValues)
        numRetFeatures = (len(self._base.features)
                          + (len(uniqueNames) * numExpanded)
                          - (1 + numExpanded))

        self._combineByExpandingFeatures_implementation(unique, namesIdx,
                                                        uniqueNames,
                                                        numRetFeatures,
                                                        numExpanded)

        self._base._featureCount = numRetFeatures
        self._base._pointCount = len(unique)

        newFtNames = []
        for prefix in map(str, uniqueNames):
            # if only one feature is expanded we will use the unique values
            # from the featureNames feature, otherwise we will concatenate
            # the feature name with the values feature name.
            if numExpanded > 1:
                for suffix in featuresWithValues:
                    if not isinstance(suffix, str):
                        if self._base.features._namesCreated():
                            suffix = self._base.features.getName(suffix)
                        else:
                            suffix = str(suffix)
                    concat = prefix + '_' + suffix
                    newFtNames.append(concat)
            else:
                newFtNames.append(prefix)

        if self._base.features._namesCreated():
            origFts = self._base.features.getNames()
            keptFts = [origFts[i] for i in uncombinedIdx]
            fNames = keptFts[:namesIdx] + newFtNames + keptFts[namesIdx:]
            self._base.features.setNames(fNames, useLog=False)
        else:
            for i, name in enumerate(newFtNames):
                self._base.features.setName(namesIdx + i, name, useLog=False)
        if self._base._pointNamesCreated():
            self.setNames(pNames, useLog=False)

        handleLogging(useLog, 'prep', 'points.combineByExpandingFeatures',
                      self._base.getTypeString(),
                      Points.combineByExpandingFeatures,
                      featureWithFeatureNames, featuresWithValues)

    def repeat(self, totalCopies, copyPointByPoint):
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
            When False, copies are made as if iterating through the
            points in this object ``totalCopies`` times. When True,
            copies are made as if the object is only iterated once,
            making ``totalCopies`` copies of each point before iterating
            to the next point.

        Returns
        -------
        nimble Base object
            Object containing the copied data.

        Examples
        --------
        Single point

        >>> data = nimble.createData('Matrix', [[1, 2, 3]])
        >>> data.points.setNames(['a'])
        >>> data.points.repeat(totalCopies=3, copyPointByPoint=False)
        Matrix(
            [[1 2 3]
             [1 2 3]
             [1 2 3]]
            pointNames={'a_1':0, 'a_2':1, 'a_3':2}
            )

        Two-dimensional, copyPointByPoint is False

        >>> data = nimble.createData('Matrix', [[1, 2, 3], [4, 5, 6]])
        >>> data.points.setNames(['a', 'b'])
        >>> data.points.repeat(totalCopies=2, copyPointByPoint=False)
        Matrix(
            [[1 2 3]
             [4 5 6]
             [1 2 3]
             [4 5 6]]
            pointNames={'a_1':0, 'b_1':1, 'a_2':2, 'b_2':3}
            )

        Two-dimensional, copyPointByPoint is True

        >>> data = nimble.createData('Matrix', [[1, 2, 3], [4, 5, 6]])
        >>> data.points.setNames(['a', 'b'])
        >>> data.points.repeat(totalCopies=2, copyPointByPoint=True)
        Matrix(
            [[1 2 3]
             [1 2 3]
             [4 5 6]
             [4 5 6]]
            pointNames={'a_1':0, 'a_2':1, 'b_1':2, 'b_2':3}
            )
        """
        return self._repeat(totalCopies, copyPointByPoint)

    ####################
    # Query functions #
    ###################

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
    @limitedTo2D
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

    @limitedTo2D
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
    def _iter(self):
        pass

    @abstractmethod
    def _getitem(self, key):
        pass

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
    def _getIndex(self, identifier):
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
    def _matching(self, function, useLog=None):
        pass

    @abstractmethod
    def _insert(self, insertBefore, toInsert, append=False, useLog=None):
        pass

    @abstractmethod
    def _mapReduce(self, mapper, reducer, useLog=None):
        pass

    @abstractmethod
    def _permute(self, order=None, useLog=None):
        pass

    @abstractmethod
    def _fillMatching(self, match, fill, limitTo, useLog=None, **kwarguments):
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
    def _repeat(self, totalCopies, copyValueByValue):
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
