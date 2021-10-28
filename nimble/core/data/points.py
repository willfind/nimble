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

from abc import ABC, abstractmethod
from collections import OrderedDict

import nimble
from nimble.core.logger import handleLogging
from nimble.exceptions import ImproperObjectAction
from ._dataHelpers import limitedTo2D

class Points(ABC):
    """
    Methods that apply to the points axis of a Base object.

    This object can be used to iterate over the points and contains
    methods that operate over the data in the associated Base object
    point-by-point.

    A point is an abstract slice containing data elements within some
    shared context. In a concrete sense, points can be thought of as the
    data rows but a row can be organized in many ways. To optimize
    for machine learning, each row should be modified to meet the
    definition of a point.
    """
    def __init__(self, base):
        """
        Parameters
        ----------
        base : Base
            The Base instance that will be queried and modified.
        """
        self._base = base
        super().__init__()

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
        >>> X = nimble.identity('Matrix', 4,
        ...                     pointNames=['a', 'b', 'c', 'd'])
        >>> X.points.getName(1)
        'b'

        Keywords
        --------
        row, key, index, header, heading, identifier
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
        >>> X = nimble.identity('Matrix', 4,
        ...                     pointNames=['a', 'b', 'c', 'd'])
        >>> X.points.getNames()
        ['a', 'b', 'c', 'd']

        Keywords
        --------
        rows, keys, indexes, indices, headers, headings, identifiers
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
        >>> X = nimble.identity('Matrix', 4,
        ...                     pointNames=['a', 'b', 'c', 'd'])
        >>> X.points.setName('b', 'new')
        >>> X.points.getNames()
        ['a', 'new', 'c', 'd']

        Keywords
        --------
        row, key, index, header, heading, identifier
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
        >>> X = nimble.identity('Matrix', 4,
        ...                     pointNames=['a', 'b', 'c', 'd'])
        >>> X.points.setNames(['1', '2', '3', '4'])
        >>> X.points.getNames()
        ['1', '2', '3', '4']

        Keywords
        --------
        rows, keys, indexes, indices, headers, headings, identifiers
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
        >>> X = nimble.identity('Matrix', 4,
        ...                     pointNames=['a', 'b', 'c', 'd'])
        >>> X.points.getIndex('c')
        2
        >>> X.points.getIndex(-1)
        3

        Keywords
        --------
        position, spot, location, identifier
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
        >>> X = nimble.identity('Matrix', 4,
        ...                     pointNames=['a', 'b', 'c', 'd'])
        >>> X.points.getIndices(['c', 'a', 'd'])
        [2, 0, 3]

        Keywords
        --------
        positions, spots, locations, identifiers
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
        >>> X = nimble.identity('Matrix', 4,
        ...                     pointNames=['a', 'b', 'c', 'd'])
        >>> X.points.hasName('a')
        True
        >>> X.points.hasName('e')
        False

        Keywords
        --------
        title, header, heading, named
        """
        return self._hasName(name)

    #########################
    # Structural Operations #
    #########################

    def copy(self, toCopy=None, start=None, end=None, number=None,
             randomize=False, useLog=None):
        """
        Copy certain points of this object.

        A variety of methods for specifying the points to copy based on
        the provided parameters. If toCopy is not None, start and end
        must be None. If start or end is not None, toCopy must be None.

        Parameters
        ----------
        toCopy : identifier, list of identifiers, function, query
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - accepts a point as its only argument and
              returns a boolean value to indicate if the point should
              be copied
            * query - string in the format 'FEATURENAME OPERATOR VALUE'
              (i.e "ft1 < 10", "id4 == yes", or "col4 is nonZero") where
              OPERATOR is separated from the FEATURENAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.
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
        >>> lst = [[1, 1, 1, 1],
        ...        [2, 2, 2, 2],
        ...        [3, 3, 3, 3],
        ...        [4, 4, 4, 4]]
        >>> X = nimble.data('Matrix', lst,
        ...                 featureNames=['a', 'b', 'c', 'd'],
        ...                 pointNames=['1', '2', '3', '4'])
        >>> single = X.points.copy('1')
        >>> single
        <Matrix 1pt x 4ft
               'a' 'b' 'c' 'd'
             ┌────────────────
         '1' │  1   1   1   1
        >
        >>> multiple = X.points.copy(['1', 3])
        >>> multiple
        <Matrix 2pt x 4ft
               'a' 'b' 'c' 'd'
             ┌────────────────
         '1' │  1   1   1   1
         '4' │  4   4   4   4
        >
        >>> func = X.points.copy(lambda pt: sum(pt) < 10)
        >>> func
        <Matrix 2pt x 4ft
               'a' 'b' 'c' 'd'
             ┌────────────────
         '1' │  1   1   1   1
         '2' │  2   2   2   2
        >
        >>> strFunc = X.points.copy("a >= 3")
        >>> strFunc
        <Matrix 2pt x 4ft
               'a' 'b' 'c' 'd'
             ┌────────────────
         '3' │  3   3   3   3
         '4' │  4   4   4   4
        >
        >>> startEnd = X.points.copy(start=1, end=2)
        >>> startEnd
        <Matrix 2pt x 4ft
               'a' 'b' 'c' 'd'
             ┌────────────────
         '2' │  2   2   2   2
         '3' │  3   3   3   3
        >
        >>> numberNoRandom = X.points.copy(number=2)
        >>> numberNoRandom
        <Matrix 2pt x 4ft
               'a' 'b' 'c' 'd'
             ┌────────────────
         '1' │  1   1   1   1
         '2' │  2   2   2   2
        >
        >>> nimble.random.setSeed(42)
        >>> numberRandom = X.points.copy(number=2, randomize=True)
        >>> numberRandom
        <Matrix 2pt x 4ft
               'a' 'b' 'c' 'd'
             ┌────────────────
         '1' │  1   1   1   1
         '4' │  4   4   4   4
        >

        Keywords
        --------
        duplicate, replicate, clone
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
        toExtract : identifier, list of identifiers, function, query
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - accepts a point as its only argument and
              returns a boolean value to indicate if the point should
              be extracted
            * query - string in the format 'FEATURENAME OPERATOR VALUE'
              (i.e "ft1 < 10", "id4 == yes", or "col4 is nonZero") where
              OPERATOR is separated from the FEATURENAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.
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

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> single = X.points.extract('a')
        >>> single
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
        >
        >>> X
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
         'c' │ 0.000 0.000 1.000
        >

        Extract multiple points.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> multiple = X.points.extract(['a', 2])
        >>> multiple
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
         'c' │ 0.000 0.000 1.000
        >
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
        >

        Extract point when the function returns True.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> func = X.points.extract(lambda pt: pt[2] == 1)
        >>> func
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'c' │ 0.000 0.000 1.000
        >
        >>> X
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
         'b' │ 0.000 1.000 0.000
        >

        Extract point when the query string returns True.

        >>> X = nimble.identity('Matrix', 3,
        ...                     pointNames=['a', 'b', 'c'],
        ...                     featureNames=['f1', 'f2', 'f3'])
        >>> strFunc = X.points.extract("f2 != 0")
        >>> strFunc
        <Matrix 1pt x 3ft
                'f1'  'f2'  'f3'
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
        >
        >>> X
        <Matrix 2pt x 3ft
                'f1'  'f2'  'f3'
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
         'c' │ 0.000 0.000 1.000
        >

        Extract points from the inclusive start to the inclusive end.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> startEnd = X.points.extract(start=1, end=2)
        >>> startEnd
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
         'c' │ 0.000 0.000 1.000
        >
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
        >

        Select a set number to extract, starting from the first point.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> numberNoRandom = X.points.extract(number=2)
        >>> numberNoRandom
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
         'b' │ 0.000 1.000 0.000
        >
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'c' │ 0.000 0.000 1.000
        >

        Select a set number to extract, choosing points at random.

        >>> nimble.random.setSeed(42)
        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> numberRandom = X.points.extract(number=2, randomize=True)
        >>> numberRandom
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'c' │ 0.000 0.000 1.000
         'a' │ 1.000 0.000 0.000
        >
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
        >

        Keywords
        --------
        move, pull, separate, withdraw, cut, hsplit
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
        toDelete : identifier, list of identifiers, function, query
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - accepts a point as its only argument and
              returns a boolean value to indicate if the point should
              be deleted
            * query - string in the format 'FEATURENAME OPERATOR VALUE'
              (i.e "ft1 < 10", "id4 == yes", or "col4 is nonZero") where
              OPERATOR is separated from the FEATURENAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.
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

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.delete('a')
        >>> X
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
         'c' │ 0.000 0.000 1.000
        >

        Delete multiple points.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.delete(['a', 2])
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
        >

        Delete point when the function returns True.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.delete(lambda pt: pt[2] == 1)
        >>> X
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
         'b' │ 0.000 1.000 0.000
        >

        Delete point when the query string returns True.

        >>> X = nimble.identity('Matrix', 3,
        ...                        pointNames=['a', 'b', 'c'],
        ...                        featureNames=['f1', 'f2', 'f3'])
        >>> X.points.delete("f2 != 0")
        >>> X
        <Matrix 2pt x 3ft
                'f1'  'f2'  'f3'
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
         'c' │ 0.000 0.000 1.000
        >

        Delete points from the inclusive start to the inclusive end.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.delete(start=1, end=2)
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
        >

        Select a set number to delete, starting from the first point.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.delete(number=2)
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'c' │ 0.000 0.000 1.000
        >

        Select a set number to delete, choosing points at random.

        >>> nimble.random.setSeed(42)
        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.delete(number=2, randomize=True)
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
        >

        Keywords
        --------
        remove, drop, exclude, eliminate, destroy, cut
        """
        self._delete(toDelete, start, end, number, randomize, useLog)

    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False, useLog=None):
        """
        Keep only certain points of this object.

        A variety of methods for specifying points to keep based on
        the provided parameters. If toRetain is not None, start and end
        must be None. If start or end is not None, toRetain must be
        None.

        Parameters
        ----------
        toRetain : identifier, list of identifiers, function, query
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - accepts a point as its only argument and
              returns a boolean value to indicate if the point should
              be retained
            * query - string in the format 'FEATURENAME OPERATOR VALUE'
              (i.e "ft1 < 10", "id4 == yes", or "col4 is nonZero") where
              OPERATOR is separated from the FEATURENAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.
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

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.retain('a')
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
        >

        Retain multiple points.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.retain(['a', 2])
        >>> X
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
         'c' │ 0.000 0.000 1.000
        >

        Retain point when the function returns True.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.retain(lambda pt: pt[2] == 1)
        >>> X
        <Matrix 1pt x 3ft
                 0     1     2
             ┌──────────────────
         'c' │ 0.000 0.000 1.000
        >

        Retain point when the query string returns True.

        >>> X = nimble.identity('Matrix', 3,
        ...                        pointNames=['a', 'b', 'c'],
        ...                        featureNames=['f1', 'f2', 'f3'])
        >>> X.points.retain("f2 != 0")
        >>> X
        <Matrix 1pt x 3ft
                'f1'  'f2'  'f3'
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
        >

        Retain points from the inclusive start to the inclusive end.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.retain(start=1, end=2)
        >>> X
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'b' │ 0.000 1.000 0.000
         'c' │ 0.000 0.000 1.000
        >

        Select a set number to retain, starting from the first point.

        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.retain(number=2)
        >>> X
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 1.000 0.000 0.000
         'b' │ 0.000 1.000 0.000
        >

        Select a set number to retain, choosing points at random.

        >>> nimble.random.setSeed(42)
        >>> X = nimble.identity('Matrix', 3)
        >>> X.points.setNames(['a', 'b', 'c'])
        >>> X.points.retain(number=2, randomize=True)
        >>> X
        <Matrix 2pt x 3ft
                 0     1     2
             ┌──────────────────
         'c' │ 0.000 0.000 1.000
         'a' │ 1.000 0.000 0.000
        >
        """
        self._retain(toRetain, start, end, number, randomize, useLog)

    @limitedTo2D
    def count(self, condition):
        """
        The number of points which satisfy the condition.

        Parameters
        ----------
        condition : function, query
            * function - accepts a point as its only argument and
              returns a boolean value to indicate if the point should
              be counted
            * query - string in the format 'FEATURENAME OPERATOR VALUE'
              (i.e "ft1 < 10", "id4 == yes", or "col4 is nonZero") where
              OPERATOR is separated from the FEATURENAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.

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
        >>> X = nimble.identity('List', 3)
        >>> X.points.count(sumIsOne)
        3

        Count when the query string returns True.

        >>> X = nimble.identity('List', 3,
        ...                     featureNames=['ft1', 'ft2', 'ft3'])
        >>> X.points.count("ft1 == 0")
        2

        Keywords
        --------
        number, counts, tally
        """
        return self._count(condition)

    def sort(self, by=None, reverse=False, useLog=None):
        """
        Arrange the points in this object.

        This sort is stable, meaning the initial point order is retained
        for points that evaluate as equal.

        Parameters
        ----------
        by : identifier(s), function, None
            Based on the parameter type:

            * identifier(s) - a single feature index or name or a list
              of feature indices and/or names. For lists, sorting occurs
              according to the first index with ties being broken in the
              order of the subsequent indices. Sort follows the natural
              ordering of the values in the identifier(s).
            * function - a scorer or comparator function. Must take
              either one or two positional arguments accepting point
              views.
            * None - sort by the point names.
        reverse : bool
            Sort as if each comparison were reversed.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        Sort by features.

        >>> lst = [['home', 81, 3, 2.49],
        ...        ['gard', 98, 10, 0.99],
        ...        ['home', 14, 1, 8.99],
        ...        ['home', 11, 3, 3.89]]
        >>> fts = ['dept', 'ID', 'quantity', 'price']
        >>> orders = nimble.data('DataFrame', lst, featureNames=fts)
        >>> orders.points.sort(['quantity', 'dept'])
        >>> orders
        <DataFrame 4pt x 4ft
             'dept' 'ID' 'quantity' 'price'
           ┌───────────────────────────────
         0 │  home   14      1       8.990
         1 │  home   81      3       2.490
         2 │  home   11      3       3.890
         3 │  gard   98      10      0.990
        >

        Sort using a comparator function.

        >>> lst = [['home', 81, 3, 2.49],
        ...        ['gard', 98, 10, 0.99],
        ...        ['home', 14, 1, 8.99],
        ...        ['home', 11, 3, 3.89]]
        >>> fts = ['dept', 'ID', 'quantity', 'price']
        >>> orders = nimble.data('DataFrame', lst, featureNames=fts)
        >>> def incomeDifference(pt1, pt2):
        ...     pt1Income = pt1['quantity'] * pt1['price']
        ...     pt2Income = pt2['quantity'] * pt2['price']
        ...     return pt2Income - pt1Income
        >>> orders.points.sort(incomeDifference)
        >>> orders
        <DataFrame 4pt x 4ft
             'dept' 'ID' 'quantity' 'price'
           ┌───────────────────────────────
         0 │  home   11      3       3.890
         1 │  gard   98      10      0.990
         2 │  home   14      1       8.990
         3 │  home   81      3       2.490
        >

        Sort using a scoring function.

        >>> lst = [['home', 81, 3, 2.49],
        ...        ['gard', 98, 10, 0.99],
        ...        ['home', 14, 1, 8.99],
        ...        ['home', 11, 3, 3.89]]
        >>> fts = ['dept', 'ID', 'quantity', 'price']
        >>> orders = nimble.data('DataFrame', lst, featureNames=fts)
        >>> def weightedQuantity(pt):
        ...     weights = {'home': 2, 'gard': 0.5}
        ...     score = pt['quantity'] * weights[pt['dept']]
        ...     return score
        >>> orders.points.sort(weightedQuantity, reverse=True)
        >>> orders
        <DataFrame 4pt x 4ft
             'dept' 'ID' 'quantity' 'price'
           ┌───────────────────────────────
         0 │  home   81      3       2.490
         1 │  home   11      3       3.890
         2 │  gard   98      10      0.990
         3 │  home   14      1       8.990
        >

        Keywords
        --------
        arrange, order
        """
        self._sort(by, reverse, useLog)

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

        >>> X = nimble.ones('Matrix', 3, 5)
        >>> X.points.transform(lambda pt: pt + 2)
        >>> X
        <Matrix 3pt x 5ft
               0     1     2     3     4
           ┌──────────────────────────────
         0 │ 3.000 3.000 3.000 3.000 3.000
         1 │ 3.000 3.000 3.000 3.000 3.000
         2 │ 3.000 3.000 3.000 3.000 3.000
        >

        Transform all points; apply to certain features. Note that the
        function recieves a read-only view of each point, so we need to
        make a copy in order to modify any specific data.

        >>> def transformMiddleFeature(pt):
        ...     ptList = pt.copy(to='python list', outputAs1D=True)
        ...     ptList[2] += 4
        ...     return ptList
        >>> X = nimble.ones('Matrix', 3, 5)
        >>> X.points.transform(transformMiddleFeature)
        >>> X
        <Matrix 3pt x 5ft
               0     1     2     3     4
           ┌──────────────────────────────
         0 │ 1.000 1.000 5.000 1.000 1.000
         1 │ 1.000 1.000 5.000 1.000 1.000
         2 │ 1.000 1.000 5.000 1.000 1.000
        >

        Transform a subset of points.

        >>> X = nimble.ones('Matrix', 3, 5)
        >>> X.points.transform(lambda pt: pt + 6, points=[0, 2])
        >>> X
        <Matrix 3pt x 5ft
               0     1     2     3     4
           ┌──────────────────────────────
         0 │ 7.000 7.000 7.000 7.000 7.000
         1 │ 1.000 1.000 1.000 1.000 1.000
         2 │ 7.000 7.000 7.000 7.000 7.000
        >
        """
        self._transform(function, points, useLog)

    ###########################
    # Higher Order Operations #
    ###########################
    @limitedTo2D
    def calculate(self, function, points=None, useLog=None):
        """
        Apply a calculation to each point.

        Return a new object that calculates the results of the given
        function on the specified points in this object, with output
        values collected into a new object that is returned upon
        completion.

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

        >>> X = nimble.ones('Matrix', 3, 5)
        >>> addTwo = X.points.calculate(lambda pt: pt + 2)
        >>> addTwo
        <Matrix 3pt x 5ft
               0     1     2     3     4
           ┌──────────────────────────────
         0 │ 3.000 3.000 3.000 3.000 3.000
         1 │ 3.000 3.000 3.000 3.000 3.000
         2 │ 3.000 3.000 3.000 3.000 3.000
        >

        Apply calculation to all points; function modifies a specific
        feature. Note that the function recieves a read-only view of
        each point, so a copy is necessary to modify any specific data.

        >>> def changeMiddleFeature(pt):
        ...     ptList = pt.copy(to='python list', outputAs1D=True)
        ...     ptList[2] += 4
        ...     return ptList
        >>> X = nimble.ones('Matrix', 3, 5)
        >>> changeMiddle = X.points.calculate(changeMiddleFeature)
        >>> changeMiddle
        <Matrix 3pt x 5ft
               0     1     2     3     4
           ┌──────────────────────────────
         0 │ 1.000 1.000 5.000 1.000 1.000
         1 │ 1.000 1.000 5.000 1.000 1.000
         2 │ 1.000 1.000 5.000 1.000 1.000
        >

        Apply calculation to a subset of points.

        >>> ptNames = ['p1', 'p2', 'p3']
        >>> X = nimble.identity('Matrix', 3, pointNames=ptNames)
        >>> calc = X.points.calculate(lambda pt: pt + 6,
        ...                              points=[2, 0])
        >>> calc
        <Matrix 2pt x 3ft
                  0     1     2
              ┌──────────────────
         'p3' │ 6.000 6.000 7.000
         'p1' │ 7.000 6.000 6.000
        >

        Keywords: apply, modify, alter
        """
        return self._calculate(function, points, useLog)

    @limitedTo2D
    def matching(self, function, useLog=None):
        """
        Identifying points matching the given criteria.

        Return a boolean value object by applying a function returning a
        boolean value for each point in this object. Common any/all
        matching functions can be found in nimble's match module. Note
        that the featureName in the returned object will be set to the
        ``__name__`` attribute of ``function`` unless it is a ``lambda``
        function.

        Parameters
        ----------
        function : function
            * function - in the form of function(pointView) which
              returns True, False, 0 or 1.
            * query - string in the format 'POINTNAME OPERATOR VALUE'
              (i.e "pt1 < 10", "id4 == yes", or "row4 is nonZero") where
              OPERATOR is separated from the POINTNAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.

        Returns
        -------
        nimble Base object
            A feature vector of boolean values.

        Examples
        --------
        >>> from nimble import match
        >>> lst = [[1, -1, 1], [-3, 3, 3], [5, 5, 5]]
        >>> X = nimble.data('Matrix', lst)
        >>> allPositivePts = X.points.matching(match.allPositive)
        >>> allPositivePts
        <Matrix 3pt x 1ft
             'allPositive'
           ┌──────────────
         0 │     False
         1 │     False
         2 │      True
        >

        >>> from nimble import match
        >>> lst = [[1, -1, float('nan')], [-3, 3, 3], [5, 5, 5]]
        >>> X = nimble.data('Matrix', lst)
        >>> ptHasMissing = X.points.matching(match.anyMissing)
        >>> ptHasMissing
        <Matrix 3pt x 1ft
             'anyMissing'
           ┌─────────────
         0 │     True
         1 │    False
         2 │    False
        >

        Keywords
        --------
        boolean, equivalent, identical, same, matches, equals, compare,
        comparison, same
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

        >>> X = nimble.zeros('Matrix', 2, 3)
        >>> toInsert = nimble.ones('Matrix', 2, 3)
        >>> X.points.insert(1, toInsert)
        >>> X
        <Matrix 4pt x 3ft
               0     1     2
           ┌──────────────────
         0 │ 0.000 0.000 0.000
         1 │ 1.000 1.000 1.000
         2 │ 1.000 1.000 1.000
         3 │ 0.000 0.000 0.000
        >

        Insert before another point; mixed object types.

        >>> lstData = [[1, 1, 1], [4, 4, 4]]
        >>> X = nimble.data('Matrix', lstData, pointNames=['1', '4'])
        >>> lstInsert = [[2, 2, 2], [3, 3, 3]]
        >>> toInsert = nimble.data('List', lstInsert,
        ...                        pointNames=['2', '3'])
        >>> X.points.insert('4', toInsert)
        >>> X
        <Matrix 4pt x 3ft
               0 1 2
             ┌──────
         '1' │ 1 1 1
         '2' │ 2 2 2
         '3' │ 3 3 3
         '4' │ 4 4 4
        >

        Reorder names.

        >>> lstData = [[1, 2, 3], [1, 2, 3]]
        >>> X = nimble.data('Matrix', lstData,
        ...                 featureNames=['a', 'b', 'c'])
        >>> lstInsert = [[3, 2, 1], [3, 2, 1]]
        >>> toInsert = nimble.data('Matrix', lstInsert,
        ...                        featureNames=['c', 'b', 'a'])
        >>> X.points.insert(0, toInsert)
        >>> X
        <Matrix 4pt x 3ft
             'a' 'b' 'c'
           ┌────────────
         0 │  1   2   3
         1 │  1   2   3
         2 │  1   2   3
         3 │  1   2   3
        >
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

        >>> X = nimble.zeros('Matrix', 2, 3)
        >>> toAppend = nimble.ones('Matrix', 2, 3)
        >>> X.points.append(toAppend)
        >>> X
        <Matrix 4pt x 3ft
               0     1     2
           ┌──────────────────
         0 │ 0.000 0.000 0.000
         1 │ 0.000 0.000 0.000
         2 │ 1.000 1.000 1.000
         3 │ 1.000 1.000 1.000
        >

        Append mixed object types.

        >>> lstData = [[1, 1, 1], [2, 2, 2]]
        >>> X = nimble.data('Matrix', lstData, pointNames=['1', '2'])
        >>> lstAppend = [[3, 3, 3], [4, 4, 4]]
        >>> toAppend = nimble.data('List', lstAppend,
        ...                        pointNames=['3', '4'])
        >>> X.points.append(toAppend)
        >>> X
        <Matrix 4pt x 3ft
               0 1 2
             ┌──────
         '1' │ 1 1 1
         '2' │ 2 2 2
         '3' │ 3 3 3
         '4' │ 4 4 4
        >

        Reorder names.

        >>> lstData = [[1, 2, 3], [1, 2, 3]]
        >>> X = nimble.data('Matrix', lstData,
        ...                 featureNames=['a', 'b', 'c'])
        >>> lstAppend = [[3, 2, 1], [3, 2, 1]]
        >>> toAppend = nimble.data('Matrix', lstAppend,
        ...                        featureNames=['c', 'b', 'a'])
        >>> X.points.append(toAppend)
        >>> X
        <Matrix 4pt x 3ft
             'a' 'b' 'c'
           ┌────────────
         0 │  1   2   3
         1 │  1   2   3
         2 │  1   2   3
         3 │  1   2   3
        >

        Keywords
        --------
        affix, adjoin, concatenate, concat, vstack, add, attach, join,
        merge
        """
        self._insert(None, toAppend, True, useLog)

    def replace(self, data, points=None, useLog=None, **dataKwds):
        """
        Replace the data in one or more of the points in this object.

        If ``points=None``, the data must be a nimble data object with
        matching pointNames, matching pointNames must be specified as a
        ``dataKwds`` argument or the data must replace all points.
        Otherwise, the shape of the ``data`` object must align with the
        ``points`` parameter. Index values in ``points`` will take
        priority over matching pointNames.

        Parameters
        ----------
        data
            The object containing the data to use as a replacement. This
            can be any ``source`` accepted by ``nimble.data``.
        points : identifier, list, None
            The point (name or index) or list of points to replace with
            the provided data.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        dataKwds
            Any keyword arguments accepted by ``nimble.data`` to use
            to construct a nimble data object from ``data``. These only
            apply when ``data`` is not already a nimble data object.

        Examples
        --------
        >>> obj = nimble.zeros('Matrix', 3, 3,
        ...                    pointNames=['a', 'b', 'c'])
        >>> newPt = nimble.ones('Matrix', 1, 3, pointNames=['b'])
        >>> obj.points.replace(newPt, points='b')
        >>> obj
        <Matrix 3pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 0.000 0.000 0.000
         'b' │ 1.000 1.000 1.000
         'c' │ 0.000 0.000 0.000
        >

        >>> obj = nimble.zeros('Sparse', 4, 3)
        >>> replacement = [[1, 2, 3], [9, 8, 7]]
        >>> obj.points.replace(replacement, [1, 2])
        >>> obj
        <Sparse 4pt x 3ft
               0     1     2
           ┌──────────────────
         0 │ 0.000 0.000 0.000
         1 │ 1.000 2.000 3.000
         2 │ 9.000 8.000 7.000
         3 │ 0.000 0.000 0.000
        >

        >>> obj = nimble.zeros('DataFrame', 3, 3,
        ...                    pointNames=['a', 'b', 'c'])
        >>> obj.points.replace([2, 3, 2], pointNames=['b'])
        >>> obj
        <DataFrame 3pt x 3ft
                 0     1     2
             ┌──────────────────
         'a' │ 0.000 0.000 0.000
         'b' │ 2.000 3.000 2.000
         'c' │ 0.000 0.000 0.000
        >

        Keywords
        --------
        change, substitute, alter, transform
        """
        return self._replace(data, points, useLog, **dataKwds)

    @limitedTo2D
    def mapReduce(self, mapper, reducer, useLog=None):
        """
        Apply a mapper and reducer function to this object.

        Return a new object containing the results of the given mapper
        and reducer functions.

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
        >>> X = nimble.data('Matrix', stadiums, featureNames=fts)
        >>> X.points.mapReduce(roofMapper, roofReducer)
        <Matrix 2pt x 2ft
              0   1
           ┌───────
         0 │ Open 3
         1 │ Dome 2
        >

        Keywords
        --------
        map, reduce, apply
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
        >>> nimble.random.setSeed(42)
        >>> lst = [[1, 1, 1, 1],
        ...        [2, 2, 2, 2],
        ...        [3, 3, 3, 3],
        ...        [4, 4, 4, 4]]
        >>> X = nimble.data('DataFrame', lst)
        >>> X.points.permute()
        >>> X
        <DataFrame 4pt x 4ft
             0 1 2 3
           ┌────────
         0 │ 3 3 3 3
         1 │ 2 2 2 2
         2 │ 4 4 4 4
         3 │ 1 1 1 1
        >

        Permute with a list of identifiers.

        >>> lst = [['home', 81, 3],
        ...        ['gard', 98, 10],
        ...        ['home', 14, 1],
        ...        ['home', 11, 3]]
        >>> pts = ['o_4', 'o_3', 'o_2', 'o_1']
        >>> orders = nimble.data('DataFrame', lst, pointNames=pts)
        >>> orders.points.permute(['o_1', 'o_2', 'o_3', 'o_4'])
        >>> orders
        <DataFrame 4pt x 3ft
                  0   1  2
               ┌───────────
         'o_1' │ home 11 3
         'o_2' │ home 14 1
         'o_3' │ gard 98 10
         'o_4' │ home 81 3
        >

        Keywords
        --------
        reorder, rearrange, shuffle
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
            * query - string in the format 'OPERATOR VALUE' representing
              a function (i.e "< 10", "== yes", or "is missing"). See
              ``nimble.match.QueryString`` for string requirements.
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

        >>> lst = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, 'na'],
        ...        [2, 2, 2],
        ...        ['na', 2, 2]]
        >>> X = nimble.data('Matrix', lst)
        >>> X.points.fillMatching(-1, 'na')
        >>> X
        <Matrix 5pt x 3ft
             0  1 2
           ┌────────
         0 │ 1  1 1
         1 │ 1  1 1
         2 │ 1  1 -1
         3 │ 2  2 2
         4 │ -1 2 2
        >

        Fill using nimble's match and fill modules; limit to last point.
        Note: None is converted to np.nan in nimble.

        >>> from nimble import match
        >>> from nimble import fill
        >>> lst = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, None],
        ...        [2, 2, 2],
        ...        [None, 2, 2]]
        >>> X = nimble.data('Matrix', lst)
        >>> X.points.fillMatching(fill.mode, match.missing, points=4)
        >>> X
        <Matrix 5pt x 3ft
               0     1     2
           ┌──────────────────
         0 │ 1.000 1.000 1.000
         1 │ 1.000 1.000 1.000
         2 │ 1.000 1.000
         3 │ 2.000 2.000 2.000
         4 │ 2.000 2.000 2.000
        >
        """
        return self._fillMatching(fillWith, matchingElements, points,
                                  useLog, **kwarguments)

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

        This function was inspired by the pivot_wider function from the
        tidyr library created by Hadley Wickham [1]_ in the R
        programming language.

        References
        ----------
        .. [1] Wickham, H. (2014). Tidy Data. Journal of Statistical
           Software, 59(10), 1 - 23.
           doi:http://dx.doi.org/10.18637/jss.v059.i10

        Examples
        --------
        >>> lst = [['NYC', 4, 5, 10],
        ...        ['LA', 20, 21, 21],
        ...        ['CHI', 0, 2, 7]]
        >>> fts = ['city', 'jan', 'feb', 'mar']
        >>> temp = nimble.data('Matrix', lst, featureNames=fts)
        >>> temp.points.splitByCollapsingFeatures(['jan', 'feb', 'mar'],
        ...                                       'month', 'temp')
        >>> temp
        <Matrix 9pt x 3ft
             'city' 'month' 'temp'
           ┌──────────────────────
         0 │  NYC     jan     4
         1 │  NYC     feb     5
         2 │  NYC     mar     10
         3 │   LA     jan     20
         4 │   LA     feb     21
         5 │   LA     mar     21
         6 │  CHI     jan     0
         7 │  CHI     feb     2
         8 │  CHI     mar     7
        >

        Keywords
        --------
        gather, melt, unpivot, fold, pivot_wider, tidy, tidyr
        """
        features = self._base.features
        numCollapsed = len(featuresToCollapse)
        collapseIndices = [self._base.features.getIndex(ft)
                           for ft in featuresToCollapse]
        retainIndices = [idx for idx in range(len(features))
                         if idx not in collapseIndices]
        currNumPoints = len(self)
        currFtNames = []
        for idx in collapseIndices:
            currName = features.getName(idx)
            if currName is None:
                currName = idx
            currFtNames.append(currName)

        numRetPoints = len(self) * numCollapsed
        numRetFeatures = len(features) - numCollapsed + 2

        self._splitByCollapsingFeatures_implementation(
            featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures)

        self._base._shape = [numRetPoints, numRetFeatures]
        ftNames = [features.getName(idx) for idx in retainIndices]
        ftNames.extend([featureForNames, featureForValues])
        features.setNames(ftNames, useLog=False)
        if self._base.points._namesCreated():
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
                                   featuresWithValues,
                                   modifyDuplicateFeatureNames=False,
                                   useLog=None):
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
        pair for any given feature(s), np.nan will be assigned as the
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
        modifyDuplicateFeatureNames : bool
            Allow modifications featureName strings if two or more unique
            values in ``featureWithFeatureNames`` return the same string.
            Duplicate strings will have the type of the feature appended to the
            string wrapped in parenthesis. For example, if 1 and '1' are both
            in ``featureWithFeatureNames``, the featureNames will become
            '1(int)' and '1(str)', respectively.
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
            | Bolt      | 200m | 19.78 |    | Gatlin    | 9.89 |       |
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
        >>> lst = [['Bolt', '100m', 9.81],
        ...        ['Bolt', '200m', 19.78],
        ...        ['Gatlin', '100m', 9.89],
        ...        ['de Grasse', '200m', 20.02],
        ...        ['de Grasse', '100m', 9.91]]
        >>> fts = ['athlete', 'dist', 'time']
        >>> sprinters = nimble.data('Matrix', lst, featureNames=fts)
        >>> sprinters.points.combineByExpandingFeatures('dist', 'time')
        >>> sprinters
        <Matrix 3pt x 3ft
             'athlete' '100m' '200m'
           ┌────────────────────────
         0 │    Bolt   9.810  19.780
         1 │   Gatlin  9.890
         2 │ de Grasse 9.910  20.020
        >


        Keywords
        --------
        spread, cast, pivot, pivot_longer, unfold, tidy, tidyr
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
                if self._base.points._namesCreated():
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
        # validate new feature names before modifying the object
        prefixes = list(map(str, uniqueNames))
        if len(set(prefixes)) != len(prefixes):
            adjust = []
            for i, n in enumerate(uniqueNames):
                if prefixes.count(str(n)) > 1:
                    adjust.append(i)
            if not modifyDuplicateFeatureNames:
                types = set(type(n) for n in (uniqueNames[i] for i in adjust))
                msg = 'Identical strings returned by classes that are unequal '
                msg += 'in featureWithFeatureNames. This was identified for '
                msg += 'the following classes ' + str(types) + '. '
                msg += 'If no changes to these values are necessary, '
                msg += 'modifyDuplicateFeatureNames can be set to True.'
                raise ImproperObjectAction(msg)
            for i in adjust:
                prefixes[i] += '(' + type(uniqueNames[i]).__name__ + ')'

        self._combineByExpandingFeatures_implementation(
            unique, namesIdx, valuesIdx, uniqueNames, numRetFeatures)

        self._base._shape = [len(unique), numRetFeatures]

        newFtNames = []
        for prefix in prefixes:
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
        if self._base.points._namesCreated():
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

        >>> X = nimble.data('Matrix', [[1, 2, 3]])
        >>> X.points.setNames(['a'])
        >>> X.points.repeat(totalCopies=3, copyPointByPoint=False)
        <Matrix 3pt x 3ft
                 0 1 2
               ┌──────
         'a_1' │ 1 2 3
         'a_2' │ 1 2 3
         'a_3' │ 1 2 3
        >

        Two-dimensional, copyPointByPoint is False

        >>> X = nimble.data('Matrix', [[1, 2, 3], [4, 5, 6]])
        >>> X.points.setNames(['a', 'b'])
        >>> X.points.repeat(totalCopies=2, copyPointByPoint=False)
        <Matrix 4pt x 3ft
                 0 1 2
               ┌──────
         'a_1' │ 1 2 3
         'b_1' │ 4 5 6
         'a_2' │ 1 2 3
         'b_2' │ 4 5 6
        >

        Two-dimensional, copyPointByPoint is True

        >>> X = nimble.data('Matrix', [[1, 2, 3], [4, 5, 6]])
        >>> X.points.setNames(['a', 'b'])
        >>> X.points.repeat(totalCopies=2, copyPointByPoint=True)
        <Matrix 4pt x 3ft
                 0 1 2
               ┌──────
         'a_1' │ 1 2 3
         'a_2' │ 1 2 3
         'b_1' │ 4 5 6
         'b_2' │ 4 5 6
        >
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
        >>> lst = [['a', 1, 3],
        ...        ['b', 5, 6],
        ...        ['b', 7, 1],
        ...        ['a', 1, 3]]
        >>> ptNames = ['p1', 'p2', 'p3', 'p1_copy']
        >>> X = nimble.data('Matrix', lst, pointNames=ptNames)
        >>> uniquePoints = X.points.unique()
        >>> uniquePoints
        <Matrix 3pt x 3ft
                0 1 2
              ┌──────
         'p1' │ a 1 3
         'p2' │ b 5 6
         'p3' │ b 7 1
        >

        Keywords
        --------
        distinct, different
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
            'correlation', 'covariance', 'sample covariance',
            'population covariance' and, 'dot product'. Pearson
            correlation coefficients are used for 'correlation'.

        Returns
        -------
        nimble Base object

        Keywords
        --------
        correlation, covariance, sample covariance,
        population covariance, dot product, similarity, relationship,
        cor, pearson, spearman, correlation coefficient
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

        Keywords
        --------
        max, mean, median, min, population std, population standard
        deviation, proportion missing, proportion zero, sample standard
        deviation, sample std, standard deviation, std, unique count,
        stats, compute, calculate
        """
        return self._statistics(statisticsFunction)

    ############
    # plotting #
    ############

    @limitedTo2D
    def plot(self, points=None, horizontal=False, outPath=None,
             show=True, figureID=None, title=True, xAxisLabel=True,
             yAxisLabel=True, legendTitle=None, **kwargs):
        """
        Bar chart comparing points.

        Each value in the object is considered to be the height of a
        bar in the chart. Bars will be grouped by each point and bar
        colors indicate each feature. If multiple bar colors are
        necessary, a legend mapping the featureName (when available) or
        feature index to its bar color will be added.

        Parameters
        ----------
        points : list of identifiers, None
            List of point names and/or indices to plot. None will
            apply to all points.
        horizontal : bool
            False, the default, draws plot bars vertically. True will
            draw bars horizontally.
        outPath : str, None
            A string of the path to save the current figure.
        show : bool
            If True, display the plot. If False, the figure will not
            display until a plotting function with show=True is called.
            This allows for future plots to placed on the figure with
            the same ``figureID`` before being shown.
        figureID : hashable, None
            A new figure will be generated for None or a new id,
            otherwise the figure with that id will be activated to draw
            the plot on the existing figure.
        title : str, bool
            The title of the plot. If True, a title will automatically
            be generated.
        xAxisLabel : str, bool
            A label for the x axis. If True, a label will automatically
            be generated.
        yAxisLabel : str, bool
            A label for the y axis. If True, a label will automatically
            be generated.
        legendTitle : str, None
            A title for the legend. A legend is only added when multiple
            bar colors are necessary, otherwise this parameter is
            ignored. None will not add a title to the legend.
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``bar`` function.

        See Also
        --------
        matplotlib.pyplot.bar

        Keywords
        --------
        bar chart, graph, visualize, graphics, show, display
        """
        self._plotComparison(
            None, points, None, horizontal, outPath, show, figureID, title,
            xAxisLabel, yAxisLabel, legendTitle, **kwargs)

    @limitedTo2D
    def plotMeans(self, points=None, horizontal=False, outPath=None,
                  show=True, figureID=None, title=True, xAxisLabel=True,
                  yAxisLabel=True, **kwargs):
        """
        Plot point means with 95% confidence interval bars.

        The 95% confidence interval for each point is calculated using
        the critical value from the two-sided Student's t-distribution.

        Parameters
        ----------
        points : list of identifiers, None
            List of point names and/or indices to plot. None will
            apply to all points.
        horizontal : bool
            False, the default, draws plot bars vertically. True will
            draw bars horizontally.
        outPath : str, None
            A string of the path to save the current figure.
        show : bool
            If True, display the plot. If False, the figure will not
            display until a plotting function with show=True is called.
            This allows for future plots to placed on the figure with
            the same ``figureID`` before being shown.
        figureID : hashable, None
            A new figure will be generated for None or a new id,
            otherwise the figure with that id will be activated to draw
            the plot on the existing figure.
        title : str, bool
            The title of the plot. If True, a title will automatically
            be generated.
        xAxisLabel : str, bool
            A label for the x axis. If True, a label will automatically
            be generated.
        yAxisLabel : str, bool
            A label for the y axis. If True, a label will automatically
            be generated.
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``errorbar`` function.

        See Also
        --------
        matplotlib.pyplot.errorbar

        Keywords
        --------
        confidence interval bars, student's t-distribution, t test,
        bar chart, display visualize, graphics
        """
        self._plotComparison(
            nimble.calculate.mean, points, True, horizontal, outPath,
            show, figureID, title, xAxisLabel, yAxisLabel, None, **kwargs)

    @limitedTo2D
    def plotStatistics(
            self, statistic, points=None, horizontal=False, outPath=None,
            show=True, figureID=None, title=True, xAxisLabel=True,
            yAxisLabel=True, legendTitle=None, **kwargs):
        """
        Bar chart comparing an aggregate statistic between points.

        The bars in the plot represent the output of the ``statistic``
        function applied to each point. Typically, functions return a
        single numeric value, however, the function may return a point
        vector. In that case, each feature in the object returned by
        ``statistic`` is considered to be the heights of separate bars
        for that point. Bars will be grouped by each point and
        bar colors indicate each feature. If multiple bar colors are
        necessary, a legend mapping the featureName (when available) or
        feature index to its bar color will be added.

        Parameters
        ----------
        statistic : function
            Functions take a point view the only required argument.
            Common statistic functions can be found in nimble.calculate.
        points : list of identifiers, None
            List of point names and/or indices to plot. None will
            apply to all points.
        horizontal : bool
            False, the default, draws plot bars vertically. True will
            draw bars horizontally.
        outPath : str, None
            A string of the path to save the current figure.
        show : bool
            If True, display the plot. If False, the figure will not
            display until a plotting function with show=True is called.
            This allows for future plots to placed on the figure with
            the same ``figureID`` before being shown.
        figureID : hashable, None
            A new figure will be generated for None or a new id,
            otherwise the figure with that id will be activated to draw
            the plot on the existing figure.
        title : str, bool
            The title of the plot. If True, a title will automatically
            be generated.
        xAxisLabel : str, bool
            A label for the x axis. If True, a label will automatically
            be generated.
        yAxisLabel : str, bool
            A label for the y axis. If True, a label will automatically
            be generated.
        legendTitle : str, None
            A title for the legend. A legend is only added when multiple
            bar colors are necessary, otherwise this parameter is
            ignored. None will not add a title to the legend.
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``bar`` function.

        See Also
        --------
        matplotlib.pyplot.bar

        Keywords
        --------
        bar chart, display, visualize, graphics
        """
        self._plotComparison(
            statistic, points, False, horizontal, outPath, show, figureID,
            title, xAxisLabel, yAxisLabel, legendTitle, **kwargs)

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
    def _getIndex(self, identifier, allowFloats=False):
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
    def _sort(self, by, reverse, useLog=None):
        pass

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

    @limitedTo2D
    def _replace(self, data, locations, useLog=None):
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
    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        pass

    @abstractmethod
    def _combineByExpandingFeatures_implementation(
        self, uniqueDict, namesIdx, valuesIdx, uniqueNames, numRetFeatures):
        pass

    @abstractmethod
    def _repeat(self, totalCopies, copyVectorByVector):
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

    @abstractmethod
    def _plotComparison(self, statistic, identifiers, confidenceIntervals,
                        horizontal, outPath, show, figureID, title,
                        xAxisLabel, yAxisLabel, legendTitle, **kwargs):
        pass
