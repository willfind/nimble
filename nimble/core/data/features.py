
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Define methods of the features attribute for Base objects.

All user-facing, feature axis functions are contained here. Functions
specific to only the feature axis will provide their functionality here.
However, most functions are applicable to either axis so only the
signatures and docstrings specific to the features axis are provided
here. The functionality of axis generic methods are defined in axis.py,
with a leading underscore added to the method name. Additionally, the
wrapping of function calls for the logger takes place in here.
"""

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

import nimble
from nimble.core.logger import handleLogging
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble._utility import prettyListString, quoteStrings
from ._dataHelpers import limitedTo2D, prepLog

class Features(ABC):
    """
    Methods that apply to the features axis of a Base object.

    This object can be used to iterate over the features and contains
    methods that operate over the data in the associated Base object
    feature-by-feature.

    A feature is an abstract slice of all data elements of the same kind
    across different contexts. In a concrete sense, features can be
    thought of as the data columns but a column can be organized in many
    ways. To optimize for machine learning, each column should be
    modified to meet the definition of a feature.
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

    @limitedTo2D
    def __iter__(self):
        return self._iter()

    @limitedTo2D
    def __getitem__(self, key):
        return self._getitem(key)

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
        getNames, setNames

        Examples
        --------
        >>> X = nimble.identity(4, featureNames=['a', 'b', 'c', 'd'])
        >>> X.features.getName(1)
        'b'

        Keywords
        --------
        column, title, header, heading, attribute, identifier
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
        getName, setNames

        Examples
        --------
        >>> X = nimble.identity(4, featureNames=['a', 'b', 'c', 'd'])
        >>> X.features.getNames()
        ['a', 'b', 'c', 'd']

        Keywords
        --------
        columns, titles, headers, headings, attributes, identifiers
        """
        return self._getNames()

    @prepLog
    def setNames(self, assignments, oldIdentifiers=None, *,
                 useLog=None): # pylint: disable=unused-argument
        """
        Set or rename one or more feature names of this object.

        Set the feature names of this object according to the values
        specified by the ``assignments`` parameter. If the number of
        new feature names being passed as assignments is less than the
        number of features in the object, then the ``oldIdentifiers``
        argument must be passed with the corresponding previous feature
        names that are to be changed. If assignments is None, then all feature
        names will be given new default values.

        Parameters
        ----------
        assignments : str, iterable, dict, None
            * str - A string not currently in the featureName set.
            * iterable - Given a list-like container, the mapping
              between names and array indices will be used to define the
              feature names.
            * dict - The mapping for each feature name in the format
              {name:index}
            * None - remove names from this object.
        oldIdentifiers : str, int, iterable, None
            * str - The name of a feature to be renamed.
            * int - The index of a feature to be renamed.
            * iterable - The names or indices of features to be renamed.
            * None - The default when assigning names to all features in the
              data.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        getName, getNames

        Examples
        --------
        >>> X = nimble.identity(4, featureNames=['a', 'b', 'c', 'd'])
        >>> X.features.setNames(['1', '2', '3', '4'])
        >>> X.features.getNames()
        ['1', '2', '3', '4']
        >>> X.features.setNames(['newer', 'sea'], oldIdentifiers=['1', '3'])
        >>> X.features.getNames()
        ['newer', '2', 'sea', '4']
        >>> X.features.setNames('by', oldIdentifiers='2')
        >>> X.features.getNames()
        ['newer', 'by', 'sea', '4']

        Keywords
        --------
        columns, titles, headers, headings, attributes, identifiers
        """
        self._setNames(assignments, oldIdentifiers)

    def getIndex(self, identifier):
        """
        The index of a feature.

        Return the index location of the feature ``identifier``. The
        ``identifier`` can be a feature name or integer (including
        negative integers).

        Parameters
        ----------
        name : str
            The name of a feature.

        Returns
        -------
        int

        See Also
        --------
        getIndices

        Examples
        --------
        >>> X = nimble.identity(4, featureNames=['a', 'b', 'c', 'd'])
        >>> X.features.getIndex('c')
        2
        >>> X.features.getIndex(-1)
        3

        Keywords
        --------
        position, spot, location, identifier
        """
        return self._getIndex(identifier)

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
        getIndex

        Examples
        --------
        >>> X = nimble.identity(4, featureNames=['a', 'b', 'c', 'd'])
        >>> X.features.getIndices(['c', 'a', 'd'])
        [2, 0, 3]

        Keywords
        --------
        positions, spots, locations, identifiers
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
        >>> X = nimble.identity(4, featureNames=['a', 'b', 'c', 'd'])
        >>> X.features.hasName('a')
        True
        >>> X.features.hasName('e')
        False

        Keywords
        --------
        title, header, heading, named
        """
        return self._hasName(name)

    #########################
    # Structural Operations #
    #########################
    @limitedTo2D
    @prepLog
    def copy(self, toCopy=None, start=None, end=None, number=None,
             randomize=False, points=None, *,
             useLog=None): # pylint: disable=unused-argument
        """
        Copy certain features of this object.

        A variety of methods for specifying the features to copy based
        on the provided parameters. If ``toCopy`` is not None, ``start``
        and ``end`` must be None. If ``start`` or ``end`` is not None,
        ``toCopy`` must be None.

        The ``nimble.match`` module contains many helpful functions that
        could be used for ``toCopy``.

        Parameters
        ----------
        toCopy : identifier, list of identifiers, function, query
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - accepts a feature as its only argument and
              returns a boolean value to indicate if the feature should
              be copied. See ``nimble.match`` for common functions.
            * query - string in the format 'POINTNAME OPERATOR VALUE'
              (i.e "pt1 < 10", "id4 == yes", or "row4 is nonZero") where
              OPERATOR is separated from the POINTNAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.
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
        points : None, identifier, list of identifiers
            Only apply the target function to a selection of points in
            each feature. May be a single point name or index, an
            iterable, container of point names and/or indices. None
            indicates application to all points.
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
        Base.copy, nimble.match

        Examples
        --------
        >>> lst = [[0, 1, 2, 3],
        ...        [0, 1, 2, 3],
        ...        [0, 1, 2, 3],
        ...        [0, 1, 2, 3]]
        >>> X = nimble.data(lst,
        ...                 pointNames=['a', 'b', 'c', 'd'],
        ...                 featureNames=['0', '1', '2', '3'])
        >>> single = X.features.copy('0')
        >>> single
        <Matrix 4pt x 1ft
             0
           ┌──
         a │ 0
         b │ 0
         c │ 0
         d │ 0
        >
        >>> multiple = X.features.copy(['0', 3])
        >>> multiple
        <Matrix 4pt x 2ft
             0  3
           ┌─────
         a │ 0  3
         b │ 0  3
         c │ 0  3
         d │ 0  3
        >
        >>> func = X.features.copy(nimble.match.allZero)
        >>> func
        <Matrix 4pt x 1ft
             0
           ┌──
         a │ 0
         b │ 0
         c │ 0
         d │ 0
        >
        >>> strFunc = X.features.copy("a >= 2")
        >>> strFunc
        <Matrix 4pt x 2ft
             2  3
           ┌─────
         a │ 2  3
         b │ 2  3
         c │ 2  3
         d │ 2  3
        >
        >>> startEnd = X.features.copy(start=1, end=3) # Note: Inclusive
        >>> startEnd
        <Matrix 4pt x 3ft
             1  2  3
           ┌────────
         a │ 1  2  3
         b │ 1  2  3
         c │ 1  2  3
         d │ 1  2  3
        >
        >>> numberNoRandom = X.features.copy(number=2)
        >>> numberNoRandom
        <Matrix 4pt x 2ft
             0  1
           ┌─────
         a │ 0  1
         b │ 0  1
         c │ 0  1
         d │ 0  1
        >
        >>> nimble.random.setSeed(42)
        >>> numberRandom = X.features.copy(number=2, randomize=True)
        >>> numberRandom
        <Matrix 4pt x 2ft
             0  3
           ┌─────
         a │ 0  3
         b │ 0  3
         c │ 0  3
         d │ 0  3
        >

        Keywords
        --------
        duplicate, replicate, clone, filter
        """
        return self._copy(toCopy, start, end, number, randomize, points)

    @limitedTo2D
    @prepLog
    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False, points=None, *,
                useLog=None): # pylint: disable=unused-argument
        """
        Move certain features of this object into their own object.

        A variety of methods for specifying the features to extract
        based on the provided parameters. If ``toExtract`` is not None,
        ``start`` and ``end`` must be None. If ``start`` or ``end`` is
        not None, ``toExtract`` must be None.

        The ``nimble.match`` module contains many helpful functions that
        could be used for ``toExtract``.

        Parameters
        ----------
        toExtract : identifier, list of identifiers, function, query
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - accepts a feature as its only argument and
              returns a boolean value to indicate if the feature should
              be extracted. See ``nimble.match`` for common functions.
            * query - string in the format 'POINTNAME OPERATOR VALUE'
              (i.e "pt1 < 10", "id4 == yes", or "row4 is nonZero") where
              OPERATOR is separated from the POINTNAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.
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
        points : None, identifier, list of identifiers
            Only apply the target function to a selection of points in
            each feature. May be a single point name or index, an
            iterable, container of point names and/or indices. None
            indicates application to all points.
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
        Extract a single feature.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> single = X.features.extract('a')
        >>> single
        <Matrix 3pt x 1ft
             a
           ┌──
         0 │ 1
         1 │ 0
         2 │ 0
        >
        >>> X
        <Matrix 3pt x 2ft
             b  c
           ┌─────
         0 │ 0  0
         1 │ 1  0
         2 │ 0  1
        >

        Extract multiple features.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> multiple = X.features.extract(['a', 2])
        >>> multiple
        <Matrix 3pt x 2ft
             a  c
           ┌─────
         0 │ 1  0
         1 │ 0  0
         2 │ 0  1
        >
        >>> X
        <Matrix 3pt x 1ft
             b
           ┌──
         0 │ 0
         1 │ 1
         2 │ 0
        >

        Extract feature when the function returns True.

        >>> X = nimble.data([[1, 2, 3], [4, 5, None], [7, 8, 9]])
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> func = X.features.extract(nimble.match.anyMissing)
        >>> func
        <Matrix 3pt x 1ft
               c
           ┌──────
         0 │ 3.000
         1 │
         2 │ 9.000
        >
        >>> X
        <Matrix 3pt x 2ft
               a      b
           ┌─────────────
         0 │ 1.000  2.000
         1 │ 4.000  5.000
         2 │ 7.000  8.000
        >

        Extract feature when the query string returns True.

        >>> X = nimble.identity(3, featureNames=['a', 'b', 'c'],
        ...                     pointNames=['p1', 'p2', 'p3'])
        >>> strFunc = X.features.extract("p2 != 0")
        >>> strFunc
        <Matrix 3pt x 1ft
              b
            ┌──
         p1 │ 0
         p2 │ 1
         p3 │ 0
        >
        >>> X
        <Matrix 3pt x 2ft
              a  c
            ┌─────
         p1 │ 1  0
         p2 │ 0  0
         p3 │ 0  1
        >

        Extract features from the inclusive start to the inclusive end.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> startEnd = X.features.extract(start=1, end=2)
        >>> startEnd
        <Matrix 3pt x 2ft
             b  c
           ┌─────
         0 │ 0  0
         1 │ 1  0
         2 │ 0  1
        >
        >>> X
        <Matrix 3pt x 1ft
             a
           ┌──
         0 │ 1
         1 │ 0
         2 │ 0
        >

        Select a set number to extract, starting from the first feature.

        >>> nimble.random.setSeed(42)
        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> numberNoRandom = X.features.extract(number=2)
        >>> numberNoRandom
        <Matrix 3pt x 2ft
             a  b
           ┌─────
         0 │ 1  0
         1 │ 0  1
         2 │ 0  0
        >
        >>> X
        <Matrix 3pt x 1ft
             c
           ┌──
         0 │ 0
         1 │ 0
         2 │ 1
        >

        Select a set number to extract, choosing features at random.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> numberRandom = X.features.extract(number=2, randomize=True)
        >>> numberRandom
        <Matrix 3pt x 2ft
             c  a
           ┌─────
         0 │ 0  1
         1 │ 0  0
         2 │ 1  0
        >
        >>> X
        <Matrix 3pt x 1ft
             b
           ┌──
         0 │ 0
         1 │ 1
         2 │ 0
        >

        Keywords
        --------
        move, pull, separate, withdraw, cut, vsplit, filter
        """
        return self._extract(toExtract, start, end, number, randomize, points)

    @limitedTo2D
    @prepLog
    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False, points=None, *,
               useLog=None): # pylint: disable=unused-argument
        """
        Remove certain features from this object.

        A variety of methods for specifying features to delete based on
        the provided parameters. If ``toDelete`` is not None, ``start``
        and ``end`` must be None. If start or end is not None,
        ``toDelete`` must be None.

        The ``nimble.match`` module contains many helpful functions that
        could be used for ``toDelete``.

        Parameters
        ----------
        toDelete : identifier, list of identifiers, function, query
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - accepts a feature as its only argument and
              returns a boolean value to indicate if the feature should
              be deleted. See ``nimble.match`` for common functions.
            * query - string in the format 'POINTNAME OPERATOR VALUE'
              (i.e "pt1 < 10", "id4 == yes", or "row4 is nonZero") where
              OPERATOR is separated from the POINTNAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.
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
        points : None, identifier, list of identifiers
            Only apply the target function to a selection of points in
            each feature. May be a single point name or index, an
            iterable, container of point names and/or indices. None
            indicates application to all points.
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
        Delete a single feature.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.delete('a')
        >>> X
        <Matrix 3pt x 2ft
             b  c
           ┌─────
         0 │ 0  0
         1 │ 1  0
         2 │ 0  1
        >

        Delete multiple features.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.delete(['a', 2])
        >>> X
        <Matrix 3pt x 1ft
             b
           ┌──
         0 │ 0
         1 │ 1
         2 │ 0
        >

        Delete feature when the function returns True.

        >>> X = nimble.data([[1, 2, 3], [4, 5, None], [7, 8, 9]])
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.delete(nimble.match.anyMissing)
        >>> X
        <Matrix 3pt x 2ft
               a      b
           ┌─────────────
         0 │ 1.000  2.000
         1 │ 4.000  5.000
         2 │ 7.000  8.000
        >

        Delete feature when the query string returns True.

        >>> X = nimble.identity(3, featureNames=['a', 'b', 'c'],
        ...                     pointNames=['p1', 'p2', 'p3'])
        >>> X.features.delete("p2 != 0")
        >>> X
        <Matrix 3pt x 2ft
              a  c
            ┌─────
         p1 │ 1  0
         p2 │ 0  0
         p3 │ 0  1
        >

        Delete features from the inclusive start to the inclusive end.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.delete(start=1, end=2)
        >>> X
        <Matrix 3pt x 1ft
             a
           ┌──
         0 │ 1
         1 │ 0
         2 │ 0
        >

        Select a set number to delete, starting from the first feature.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.delete(number=2)
        >>> X
        <Matrix 3pt x 1ft
             c
           ┌──
         0 │ 0
         1 │ 0
         2 │ 1
        >

        Select a set number to delete, choosing features at random.

        >>> nimble.random.setSeed(42)
        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.delete(number=2,  randomize=True)
        >>> X
        <Matrix 3pt x 1ft
             b
           ┌──
         0 │ 0
         1 │ 1
         2 │ 0
        >

        Keywords
        --------
        remove, drop, exclude, eliminate, destroy, cut, filter
        """
        self._delete(toDelete, start, end, number, randomize, points)

    @limitedTo2D
    @prepLog
    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False, points=None, *,
               useLog=None): # pylint: disable=unused-argument
        """
        Keep only certain features of this object.

        A variety of methods for specifying features to keep based on
        the provided parameters. If ``toRetain`` is not None, ``start``
        and ``end`` must be None. If ``start`` or ``end`` is not None,
        ``toRetain`` must be None.

        The ``nimble.match`` module contains many helpful functions that
        could be used for ``toRetain``.

        Parameters
        ----------
        toRetain : identifier, list of identifiers, function, query
            * identifier - a name or index
            * list of identifiers - an iterable container of identifiers
            * function - accepts a feature as its only argument and
              returns a boolean value to indicate if the feature should
              be retained. See ``nimble.match`` for common functions.
            * query - string in the format 'POINTNAME OPERATOR VALUE'
              (i.e "pt1 < 10", "id4 == yes", or "row4 is nonZero") where
              OPERATOR is separated from the POINTNAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.
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
        points : None, identifier, list of identifiers
            Only apply the target function to a selection of points in
            each feature. May be a single point name or index, an
            iterable, container of point names and/or indices. None
            indicates application to all points.
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
        Retain a single feature.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.retain('a')
        >>> X
        <Matrix 3pt x 1ft
             a
           ┌──
         0 │ 1
         1 │ 0
         2 │ 0
        >

        Retain multiple features.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.retain(['a', 2])
        >>> X
        <Matrix 3pt x 2ft
             a  c
           ┌─────
         0 │ 1  0
         1 │ 0  0
         2 │ 0  1
        >

        Retain feature when the function returns True.

        >>> X = nimble.data([[1, None, 3], [None, 5, 6], [7, 8, 9]])
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.retain(nimble.match.allNonMissing)
        >>> X
        <Matrix 3pt x 1ft
               c
           ┌──────
         0 │ 3.000
         1 │ 6.000
         2 │ 9.000
        >

        Retain feature when the query string returns True.

        >>> X = nimble.identity(3, featureNames=['a', 'b', 'c'],
        ...                     pointNames=['p1', 'p2', 'p3'])
        >>> X.features.retain("p2 != 0")
        >>> X
        <Matrix 3pt x 1ft
              b
            ┌──
         p1 │ 0
         p2 │ 1
         p3 │ 0
        >

        Retain features from the inclusive start to the inclusive end.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.retain(start=1, end=2)
        >>> X
        <Matrix 3pt x 2ft
             b  c
           ┌─────
         0 │ 0  0
         1 │ 1  0
         2 │ 0  1
        >

        Select a set number to retain, starting from the first feature.

        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.retain(number=2)
        >>> X
        <Matrix 3pt x 2ft
             a  b
           ┌─────
         0 │ 1  0
         1 │ 0  1
         2 │ 0  0
        >

        Select a set number to retain, choosing features at random.

        >>> nimble.random.setSeed(42)
        >>> X = nimble.identity(3)
        >>> X.features.setNames(['a', 'b', 'c'])
        >>> X.features.retain(number=2, randomize=True)
        >>> X
        <Matrix 3pt x 2ft
             c  a
           ┌─────
         0 │ 0  1
         1 │ 0  0
         2 │ 1  0
        >

        Keywords
        --------
        keep, hold, maintain, preserve, remove, filter
        """
        self._retain(toRetain, start, end, number, randomize, points)

    @limitedTo2D
    def count(self, condition, points=None):
        """
        The number of features which satisfy the condition.

        Parameters
        ----------
        condition : function, query

            * function - accepts a feature as its only argument and
              returns a boolean value to indicate if the feature should
              be counted
            * query - string in the format 'POINTNAME OPERATOR VALUE'
              (i.e "pt1 < 10", "id4 == yes", or "row4 is nonZero") where
              OPERATOR is separated from the POINTNAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.

        points : None, identifier, list of identifiers
            Only apply the condition function to a selection of points
            in each feature. May be a single point name or index, an
            iterable, container of point names and/or indices. None
            indicates application to all points.

        Returns
        -------
        int

        See Also
        --------
        Base.countElements, Base.countUniqueElements, matching, nimble.match

        Examples
        --------
        Count using a python function.

        >>> def sumIsOne(ft):
        ...     return sum(ft) == 1
        >>> X = nimble.identity(3)
        >>> X.features.count(sumIsOne)
        3

        Count when the query string returns True.

        >>> X = nimble.identity(3, pointNames=['pt1', 'pt2', 'pt3'])
        >>> X.features.count("pt1 == 0")
        2

        Keywords
        --------
        number, counts, tally
        """
        return self._count(condition, points)

    @limitedTo2D
    @prepLog
    def sort(self, by=None, reverse=False, *,
             useLog=None): # pylint: disable=unused-argument
        """
        Arrange the features in this object.

        This sort is stable, meaning the initial feature order is
        retained for features that evaluate as equal.

        Parameters
        ----------
        by : identifier(s), function, None
            Based on the parameter type:

            * identifier(s) - a single point index or name or a list
              of point indices and/or names. For lists, sorting occurs
              according to the first index with ties being broken in the
              order of the subsequent indices. Sort follows the natural
              ordering of the values in the identifier(s).
            * function - a scorer or comparator function. Must take
              either one or two positional arguments accepting feature
              views.
            * None - sort by the feature names.
        reverse : bool
            Reverse the sorted order.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        permute

        Examples
        --------
        Sort by feature names.

        >>> lst = [[64, 67, 64],
        ...        [68, 71, 66],
        ...        [73, 71, 70],
        ...        [45, 40, 51]]
        >>> fts = ['Denver', 'Boulder', 'Fort Collins']
        >>> highTemps = nimble.data(lst, featureNames=fts)
        >>> highTemps.features.sort()
        >>> highTemps
        <Matrix 4pt x 3ft
             Boulder  Denver  Fort Collins
           ┌──────────────────────────────
         0 │    67      64         64
         1 │    71      68         66
         2 │    71      73         70
         3 │    40      45         51
        >

        Sort by points.

        >>> lst = [[3, 1, 2, 0, 2],
        ...        [100, 1, 10, 0, 11],
        ...        [200, 2, 20, 0, 21],
        ...        [300, 3, 30, 0, 31]]
        >>> orders = nimble.data(lst)
        >>> orders.features.sort(0)
        >>> orders
        <Matrix 4pt x 5ft
             0  1  2   3    4
           ┌──────────────────
         0 │ 0  1   2   2    3
         1 │ 0  1  10  11  100
         2 │ 0  2  20  21  200
         3 │ 0  3  30  31  300
        >

        Sort using function.

        >>> lst = [[64, 67, 64],
        ...        [68, 71, 66],
        ...        [73, 71, 70],
        ...        [45, 40, 51]]
        >>> fts = ['Denver', 'Boulder', 'Fort Collins']
        >>> highTemps = nimble.data(lst, featureNames=fts)
        >>> def averageHighTemp(ft):
        ...     return nimble.calculate.mean(ft)
        >>> highTemps.features.sort(averageHighTemp, reverse=True)
        >>> highTemps
        <Matrix 4pt x 3ft
             Fort Collins  Denver  Boulder
           ┌──────────────────────────────
         0 │      64         64       67
         1 │      66         68       71
         2 │      70         73       71
         3 │      51         45       40
        >

        Keywords
        --------
        arrange, order
        """
        self._sort(by, reverse)

    @limitedTo2D
    @prepLog
    def transform(self, function, features=None, *,
                  useLog=None): # pylint: disable=unused-argument
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        calculate, Base.transformElements

        Examples
        --------
        Transform all features; apply to all points.

        >>> X = nimble.ones(3, 5)
        >>> X.features.transform(lambda ft: ft + 2)
        >>> X
        <Matrix 3pt x 5ft
             0  1  2  3  4
           ┌──────────────
         0 │ 3  3  3  3  3
         1 │ 3  3  3  3  3
         2 │ 3  3  3  3  3
        >

        Transform all features; apply to certain points. Note that the
        function recieves a read-only view of each feature, so we need
        to make a copy in order to modify any specific data.

        >>> def transformMiddlePoint(ft):
        ...     ftList = ft.copy(to='python list', outputAs1D=True)
        ...     ftList[1] += 4
        ...     return ftList
        >>> X = nimble.ones(3, 5)
        >>> X.features.transform(transformMiddlePoint)
        >>> X
        <Matrix 3pt x 5ft
             0  1  2  3  4
           ┌──────────────
         0 │ 1  1  1  1  1
         1 │ 5  5  5  5  5
         2 │ 1  1  1  1  1
        >

        Transform a subset of features.

        >>> X = nimble.ones(3, 5)
        >>> X.features.transform(lambda ft: ft + 6, features=[1, 3])
        >>> X
        <Matrix 3pt x 5ft
             0  1  2  3  4
           ┌──────────────
         0 │ 1  7  1  7  1
         1 │ 1  7  1  7  1
         2 │ 1  7  1  7  1
        >

        Keywords
        --------
        apply, modify, alter, change, map, compute
        """
        self._transform(function, features)

    ###########################
    # Higher Order Operations #
    ###########################
    @limitedTo2D
    @prepLog
    def calculate(self, function, features=None, *,
                  useLog=None): # pylint: disable=unused-argument
        """
        Apply a calculation to each feature.

        Return a new object that calculates the results of the given
        function on the specified features in this object, with output
        values collected into a new object that is returned upon
        completion.

        Parameters
        ----------
        function : function
            Accepts a view of a feature as an argument and returns the
            new values in that feature.
        features : feature, list of features
            The subset of features to limit the calculation to. If None,
            the calculation will apply to all features.
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
        transform, Base.calculateOnElements

        Examples
        --------
        Apply calculation to all features; apply to all points.

        >>> X = nimble.ones(3, 5)
        >>> addTwo = X.features.calculate(lambda ft: ft + 2)
        >>> addTwo
        <Matrix 3pt x 5ft
             0  1  2  3  4
           ┌──────────────
         0 │ 3  3  3  3  3
         1 │ 3  3  3  3  3
         2 │ 3  3  3  3  3
        >

        Apply calculation to all features; function modifies a specific
        point. Note that the function recieves a read-only view of each
        feature, so a copy is necessary to modify any specific data.

        >>> def changeMiddlePoint(ft):
        ...     ftList = ft.copy(to='python list', outputAs1D=True)
        ...     ftList[1] += 4
        ...     return ftList
        >>> X = nimble.ones(3, 5)
        >>> middleChange = X.features.calculate(changeMiddlePoint)
        >>> middleChange
        <Matrix 3pt x 5ft
             0  1  2  3  4
           ┌──────────────
         0 │ 1  1  1  1  1
         1 │ 5  5  5  5  5
         2 │ 1  1  1  1  1
        >

        Apply calculation to a subset of features.

        >>> ftNames = ['f1', 'f2', 'f3']
        >>> X = nimble.identity(3, featureNames=ftNames)
        >>> calc = X.features.calculate(lambda ft: ft + 6,
        ...                             features=[2, 0])
        >>> calc
        <Matrix 3pt x 2ft
             f3  f1
           ┌───────
         0 │ 6   7
         1 │ 6   6
         2 │ 7   6
        >

        Keywords
        --------
        apply, modify, alter, statistics, stats, compute
        """
        return self._calculate(function, features)

    @limitedTo2D
    @prepLog
    def matching(self, function, *,
                 useLog=None): # pylint: disable=unused-argument
        """
        Identifying features matching the given criteria.

        Return a boolean value object by apply a function returning a
        boolean value for each feature in this object. Common any/all
        matching functions can be found in nimble's match module. Note
        that the pointName in the returned object will be set to the
        ``__name__`` attribute of ``function`` unless it is a ``lambda``
        function.

        Parameters
        ----------
        function : function
            * function - in the form of function(featureView) which
              returns True, False, 0 or 1.
            * query - string in the format 'POINTNAME OPERATOR VALUE'
              (i.e "pt1 < 10", "id4 == yes", or "row4 is nonZero") where
              OPERATOR is separated from the POINTNAME and VALUE by
              whitespace characters. See ``nimble.match.QueryString``
              for string requirements.

        Returns
        -------
        nimble Base object
            A point vector of boolean values.

        See Also
        --------
        count, nimble.match

        Examples
        --------
        >>> from nimble import match
        >>> lst = [[1, -1, 1], [-3, 3, 3]]
        >>> X = nimble.data(lst)
        >>> allPositiveFts = X.features.matching(match.allPositive)
        >>> allPositiveFts
        <Matrix 1pt x 3ft
                         0      1     2
                     ┌───────────────────
         allPositive │ False  False  True
        >

        >>> from nimble import match
        >>> lst = [[1, float('nan'), 1], [-3, 3, 3]]
        >>> X = nimble.data(lst)
        >>> ftHasMissing = X.features.matching(match.anyMissing)
        >>> ftHasMissing
        <Matrix 1pt x 3ft
                        0     1      2
                    ┌───────────────────
         anyMissing │ False  True  False
        >

        Keywords
        --------
        boolean, equivalent, identical, same, matches, equals, compare,
        comparison, same, filter
        """
        return self._matching(function)

    @limitedTo2D
    @prepLog
    def insert(self, insertBefore, toInsert, *,
               useLog=None): # pylint: disable=unused-argument
        """
        Insert more features into this object.

        Expand this object by inserting the features of ``toInsert``
        prior to the ``insertBefore`` identifier. The points in
        ``toInsert`` do not need to be in the same order as in the
        calling object; the data will automatically be placed using the
        calling object's point order if there is an unambiguous mapping.
        ``toInsert`` will be unaffected by calling this method.

        Parameters
        ----------
        insertBefore : identifier
            The index or feature name prior to which the data from
            toInsert will be inserted.
        toInsert : nimble Base object
            The nimble Base object whose contents we will be including
            in this object. Must have the same point names as the
            calling object, but not necessarily in the same order. Must
            not share any feature names with the calling object.
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

        >>> X = nimble.zeros(3, 2)
        >>> toInsert = nimble.ones(3, 2)
        >>> X.features.insert(1, toInsert)
        >>> X
        <Matrix 3pt x 4ft
             0  1  2  3
           ┌───────────
         0 │ 0  1  1  0
         1 │ 0  1  1  0
         2 │ 0  1  1  0
        >

        Insert before another point; mixed object types.

        >>> lstData = [[1, 4], [5, 8]]
        >>> X = nimble.data(lstData, featureNames=['1', '4'])
        >>> lstInsert = [[2, 3], [6, 7]]
        >>> toInsert = nimble.data(lstInsert,
        ...                        featureNames=['2', '3'])
        >>> X.features.insert('4', toInsert)
        >>> X
        <Matrix 2pt x 4ft
             1  2  3  4
           ┌───────────
         0 │ 1  2  3  4
         1 │ 5  6  7  8
        >

        Reorder names.

        >>> lstData = [[1, 2], [5, 6]]
        >>> X = nimble.data(lstData, pointNames=['a', 'b'])
        >>> lstInsert = [[7, 8], [3, 4]]
        >>> toInsert = nimble.data(lstInsert,
        ...                        pointNames=['b', 'a'])
        >>> X.features.insert(2, toInsert)
        >>> X
        <Matrix 2pt x 4ft
             0  1  2  3
           ┌───────────
         a │ 1  2  3  4
         b │ 5  6  7  8
        >

        Keywords
        --------
        embed, include, inject, alter, position
        """
        self._insert(insertBefore, toInsert, False)

    @limitedTo2D
    @prepLog
    def append(self, toAppend, *,
               useLog=None): # pylint: disable=unused-argument
        """
        Append features to this object.

        Expand this object by appending the features of ``toAppend``
        to the end of this object. The points in ``toAppend`` do not
        need to be in the same order as in the calling object; the data
        will automatically be placed using the calling object's point
        order if there is an unambiguous mapping. ``toAppend`` will be
        unaffected by calling this method.

        Parameters
        ----------
        toAppend : nimble Base object
            The nimble Base object whose contents we will be including
            in this object. Must have the same point names as the
            calling object, but not necessarily in the same order. Must
            not share any feature names with the calling object.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        insert, Base.merge

        Examples
        --------
        Append data; default names.

        >>> X = nimble.zeros(3, 2)
        >>> toAppend = nimble.ones(3, 2)
        >>> X.features.append(toAppend)
        >>> X
        <Matrix 3pt x 4ft
             0  1  2  3
           ┌───────────
         0 │ 0  0  1  1
         1 │ 0  0  1  1
         2 │ 0  0  1  1
        >

        Append mixed object types.

        >>> lstData = [[1, 2], [5, 6]]
        >>> X = nimble.data(lstData, featureNames=['1', '2'])
        >>> lstAppend = [[3, 4], [7, 8]]
        >>> toAppend = nimble.data(lstAppend,
        ...                        featureNames=['3', '4'])
        >>> X.features.append(toAppend)
        >>> X
        <Matrix 2pt x 4ft
             1  2  3  4
           ┌───────────
         0 │ 1  2  3  4
         1 │ 5  6  7  8
        >

        Reorder names.

        >>> lstData = [[1, 2], [5, 6]]
        >>> X = nimble.data(lstData, pointNames=['a', 'b'])
        >>> lstAppend = [[7, 8], [3, 4]]
        >>> toAppend = nimble.data(lstAppend,
        ...                        pointNames=['b', 'a'])
        >>> X.features.append(toAppend)
        >>> X
        <Matrix 2pt x 4ft
             0  1  2  3
           ┌───────────
         a │ 1  2  3  4
         b │ 5  6  7  8
        >

        Keywords
        --------
        affix, adjoin, concatenate, concat, hstack, add, attach, join,
        merge
        """
        self._insert(None, toAppend, True)

    @limitedTo2D
    @prepLog
    def replace(self, data, features=None, *,
                useLog=None, # pylint: disable=unused-argument
                **dataKwds):
        """
        Replace the data in one or more of the features in this object.

        If ``features=None``, the data must be a nimble data object with
        matching featureNames, matching featureNames must be specified
        as a ``dataKwds`` argument or the data must replace all
        features. Otherwise, the shape of the ``data`` object must align
        with the ``features`` parameter. Index values in ``features``
        will take priority over matching featureNames.

        Parameters
        ----------
        data
            The object containing the data to use as a replacement. This
            can be any ``source`` accepted by ``nimble.data``.
        features : identifier, list, None
            The feature (name or index) or list of features to replace
            with the provided data.
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

        See Also
        --------
        Base.replaceRectangle

        Examples
        --------
        >>> obj = nimble.zeros(3, 3, featureNames=['a', 'b', 'c'])
        >>> newFt = nimble.ones(3, 1, featureNames=['b'])
        >>> obj.features.replace(newFt, features='b')
        >>> obj
        <Matrix 3pt x 3ft
             a  b  c
           ┌────────
         0 │ 0  1  0
         1 │ 0  1  0
         2 │ 0  1  0
        >

        >>> obj = nimble.zeros(3, 4, returnType='Sparse')
        >>> replacement = [[1, 9], [2, 8], [3, 7]]
        >>> obj.features.replace(replacement, [1, 2])
        >>> obj
        <Sparse 3pt x 4ft
             0  1  2  3
           ┌───────────
         0 │ 0  1  9  0
         1 │ 0  2  8  0
         2 │ 0  3  7  0
        >

        >>> obj = nimble.zeros(3, 3, featureNames=['a', 'b', 'c'],
        ...                    returnType="DataFrame")
        >>> obj.features.replace(['Y', 'N', 'Y'], featureNames=['b'])
        >>> obj
        <DataFrame 3pt x 3ft
             a  b  c
           ┌────────
         0 │ 0  Y  0
         1 │ 0  N  0
         2 │ 0  Y  0
        >

        Keywords
        --------
        change, substitute, alter, transform
        """
        return self._replace(data, features, **dataKwds)

    @limitedTo2D
    @prepLog
    def permute(self, order=None, *,
                useLog=None): # pylint: disable=unused-argument
        """
        Permute the indexing of the features.

        Change the arrangement of features in the object. A specific
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
        not be sufficiently random for large number of features.
        See random.shuffle()'s documentation.

        See Also
        --------
        sort

        Examples
        --------
        >>> nimble.random.setSeed(42)
        >>> lst = [[1, 2, 3, 4],
        ...        [1, 2, 3, 4],
        ...        [1, 2, 3, 4],
        ...        [1, 2, 3, 4]]
        >>> X = nimble.data(lst)
        >>> X.features.permute()
        >>> X
        <Matrix 4pt x 4ft
             0  1  2  3
           ┌───────────
         0 │ 3  2  4  1
         1 │ 3  2  4  1
         2 │ 3  2  4  1
         3 │ 3  2  4  1
        >

        Permute with a list of identifiers.

        >>> lst = [['home', 81, 3],
        ...        ['gard', 98, 10],
        ...        ['home', 14, 1],
        ...        ['home', 11, 3]]
        >>> cols = ['dept', 'ID', 'quantity']
        >>> orders = nimble.data(lst, featureNames=cols)
        >>> orders.features.permute(['ID', 'quantity', 'dept'])
        >>> orders
        <DataFrame 4pt x 3ft
             ID  quantity  dept
           ┌───────────────────
         0 │ 81      3     home
         1 │ 98     10     gard
         2 │ 14      1     home
         3 │ 11      3     home
        >

        Keywords
        --------
        reorder, rearrange, shuffle
        """
        self._permute(order)

    @limitedTo2D
    @prepLog
    def fillMatching(self, fillWith, matchingElements, features=None, *,
                     useLog=None, # pylint: disable=unused-argument
                     **kwarguments):
        """
        Replace given values in each feature with other values.

        Fill matching values within each feature with a specified value
        based on the values in that feature. The ``fill`` value can be
        a constant or a determined based on unmatched values in the
        feature. The match and fill modules in nimble offer common
        functions for these operations.

        Parameters
        ----------
        fillWith : value or function
            * value - a value to fill each matching value in each
              feature
            * function - must be in the format:
              fillWith(feature, matchingElements) or
              fillWith(feature, matchingElements, \*\*kwarguments)
              and return the transformed feature as a list of values.
              Certain fill methods can be imported from nimble's fill
              module.
        matchingElements : value, list, or function
            * value - a value to locate within each feature
            * list - values to locate within each feature
            * function - must accept a single value and return True if
              the value is a match. Certain match types can be imported
              from nimble's match module.
            * query - string in the format 'OPERATOR VALUE' representing
              a function (i.e "< 10", "== yes", or "is missing"). See
              ``nimble.match.QueryString`` for string requirements.
        features : identifier or list of identifiers
            Select specific features to apply fill to. If features is
            None, the fill will be applied to all features.
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
        nimble.fill, nimble.match, Base.replaceRectangle,
        nimble.fillMatching

        Examples
        --------
        Fill a value with another value.

        >>> lst = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, 'na'],
        ...        [2, 2, 2],
        ...        ['na', 2, 2]]
        >>> X = nimble.data(lst)
        >>> X.features.fillMatching(-1, 'na')
        >>> X
        <DataFrame 5pt x 3ft
             0   1  2
           ┌──────────
         0 │  1  1   1
         1 │  1  1   1
         2 │  1  1  -1
         3 │  2  2   2
         4 │ -1  2   2
        >

        Fill using nimble's match and fill modules; limit to first
        feature. Note: None is converted to np.nan in nimble.

        >>> from nimble import match
        >>> from nimble import fill
        >>> lst = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, None],
        ...        [2, 2, 2],
        ...        [None, 2, 2]]
        >>> X = nimble.data(lst)
        >>> X.features.fillMatching(fill.mean, match.missing, features=0)
        >>> X
        <Matrix 5pt x 3ft
               0      1      2
           ┌────────────────────
         0 │ 1.000  1.000  1.000
         1 │ 1.000  1.000  1.000
         2 │ 1.000  1.000
         3 │ 2.000  2.000  2.000
         4 │ 1.250  2.000  2.000
        >
        """
        return self._fillMatching(fillWith, matchingElements, features,
                                  **kwarguments)

    @limitedTo2D
    @prepLog
    def normalize(self, function, applyResultTo=None, features=None, *,
                  useLog=None): # pylint: disable=unused-argument
        """
        Modify all features in this object using the given function.

        Normalize the data by a function that adjusts each feature
        based on the provided function. If the function allows, the
        normalization can also be applied to a second object. Examples
        of normalizations provided in Nimble are ``meanNormalize``, 
        ``percentileNormalize``, and the others in the examples below.
        See nimble.calculate.normalize for all the default provided 
        types of normalization.

        Parameters
        ----------
        function
            The function applying the normalization. Functions must
            accept a feature view and output the normalized feature
            data. When ``applyResultTo`` is not None, the function must
            accept a second feature view and return a two-tuple
            (normalized feature from the calling object,
            normalized feature from applyResultTo).
            Common normalizations can be found in 
            nimble.calculate.normalize.
        applyResultTo : nimble Base object, None
            The secondary object to apply the the normalization to.
            Must have the same number of features as the calling object.
        features : identifier, list of identifiers, None
            Select specific features to apply the normalization to. If
            features is None, the normalization will be applied to all
            features.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        nimble.calculate.normalize, nimble.normalizeData

        Examples
        --------
        Calling object only.

        >>> from nimble.calculate import range0to1Normalize
        >>> lstTrain = [[5, 9.8, 92],
        ...             [3, 6.2, 58],
        ...             [2, 3.0, 29]]
        >>> pts = ['movie1', 'movie2', 'movie3']
        >>> fts = ['review1', 'review2', 'review3']
        >>> train = nimble.data(lstTrain, pts, fts)
        >>> train.features.normalize(range0to1Normalize)
        >>> train
        <Matrix 3pt x 3ft
                  review1  review2  review3
                ┌──────────────────────────
         movie1 │  1.000    1.000    1.000
         movie2 │  0.333    0.471    0.460
         movie3 │  0.000    0.000    0.000
        >

        With applyResultTo.

        >>> from nimble.calculate import meanStandardDeviationNormalize
        >>> lstTrain = [[5, 9.8, 92],
        ...             [3, 6.2, 58],
        ...             [2, 3.0, 10]]
        >>> lstTest = [[4, 9.1, 43],
        ...            [3, 5.1, 88]]
        >>> fts = ['review1', 'review2', 'review3']
        >>> trainPts = ['movie1', 'movie2', 'movie3']
        >>> train = nimble.data(lstTrain, trainPts, fts)
        >>> testPts = ['movie4', 'movie5']
        >>> test = nimble.data(lstTest, testPts, fts)
        >>> train.features.normalize(meanStandardDeviationNormalize,
        ...                          applyResultTo=test)
        >>> train
        <Matrix 3pt x 3ft
                  review1  review2  review3
                ┌──────────────────────────
         movie1 │   1.336    1.248    1.149
         movie2 │  -0.267   -0.048    0.139
         movie3 │  -1.069   -1.200   -1.288
        >
        >>> test
        <Matrix 2pt x 3ft
                  review1  review2  review3
                ┌──────────────────────────
         movie4 │   0.535    0.996   -0.307
         movie5 │  -0.267   -0.444    1.031
        >
        
        With user defined normalization function.

        >>> import numpy as np
        >>> lstTrain = [[482], [30000], [7900], [35],[600]]
        >>> pts = ['user1', 'user2', 'user3', 'user4', 'user5']
        >>> fts = ['miles']
        >>> train = nimble.data(lstTrain, pts, fts)
        >>> train
        <Matrix 5pt x 1ft
                 miles
               ┌──────
         user1 │   482
         user2 │ 30000
         user3 │  7900
         user4 │    35
         user5 │   600
        >
        >>> def logNormalize(ft):
        ...     return np.log(ft)
        >>> train.features.normalize(logNormalize)
        >>> train
        <Matrix 5pt x 1ft
                 miles
               ┌───────
         user1 │  6.178
         user2 │ 10.309
         user3 │  8.975
         user4 │  3.555
         user5 │  6.397
        >

        Keywords
        --------
        standardize, scale, rescale, divide, length
        """
        if not callable(function):
            raise InvalidArgumentType('function must be callable')
        if applyResultTo is None:
            self.transform(function, features=features, useLog=False)
        elif isinstance(applyResultTo, nimble.core.data.Base):
            if len(self) != len(applyResultTo.features):
                msg = 'applyResultTo must have the same number of features as '
                msg += 'the calling object'
                raise InvalidArgumentValue(msg)
            selfNames = self._getNamesNoGeneration()
            applyToNames = applyResultTo.features._getNamesNoGeneration()
            if ((selfNames is not None and applyToNames is not None)
                    and selfNames != applyToNames):
                msg = 'applyResultTo featureNames do not match the '
                msg += 'featureNames of the calling object'
                raise InvalidArgumentValue(msg)
            if features is not None:
                features = list(map(self._getIndex, features))
            zipFeatures = zip(self, applyResultTo.features)
            for i, (ft1, ft2) in enumerate(zipFeatures):
                if features is None or i in features:
                    # pylint: disable=cell-var-from-loop
                    norm1, norm2 = function(ft1, ft2)
                    self.transform(lambda _: norm1, features=i, useLog=False)
                    applyResultTo.features.transform(
                        lambda _: norm2, features=i, useLog=False)
        else:
            msg = 'applyResultTo must be None or an instance of Base'
            raise InvalidArgumentType(msg)


    @limitedTo2D
    @prepLog
    def splitByParsing(self, feature, rule, resultingNames, *,
                       useLog=None): # pylint: disable=unused-argument
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Notes
        -----
        Visual representations of the Examples::

            locations.splitFeatureByParsing('location', ', ',
                                            ['city', 'country'])

                locations (before)                locations (after)
            +-------------------------+    +-----------+--------------+
            | location                |    | city      | country      |
            +-------------------------+    +-----------+--------------+
            | Cape Town, South Africa |    | Cape Town | South Africa |
            +-------------------------+ -> +-----------+--------------+
            | Lima, Peru              |    | Lima      | Peru         |
            +-------------------------+    +-----------+--------------+
            | Moscow, Russia          |    | Moscow    | Russia       |
            +-------------------------+    +-----------+--------------+

            inventory.splitFeatureByParsing(0, 3, ['category', 'id'])

              inventory (before)                  inventory (after)
            +---------+----------+        +----------+-----+----------+
            | product | quantity |        | category | id  | quantity |
            +---------+----------+        +----------+-----+----------+
            | AGG932  | 44       |        | AGG      | 932 | 44       |
            +---------+----------+        +----------+-----+----------+
            | AGG734  | 11       |   ->   | AGG      | 734 | 11       |
            +---------+----------+        +----------+-----+----------+
            | HEQ892  | 1        |        | HEQ      | 892 | 1        |
            +---------+----------+        +----------+-----+----------+
            | LEQ331  | 2        |        | LEQ      | 331 | 2        |
            +---------+----------+        +----------+-----+----------+

        This function was inspired by the separate function from the
        tidyr library created by Hadley Wickham [1]_ in the R
        programming language.

        References
        ----------
        .. [1] Wickham, H. (2014). Tidy Data. Journal of Statistical
           Software, 59(10), 1 - 23.
           doi:http://dx.doi.org/10.18637/jss.v059.i10

        Examples
        --------
        Split with a string for ``rule``.

        >>> lst = [['Cape Town, South Africa'],
        ...        ['Lima, Peru'],
        ...        ['Moscow, Russia']]
        >>> fts = ['location']
        >>> locations = nimble.data(lst, featureNames=fts)
        >>> locations.features.splitByParsing('location', ', ',
        ...                                   ['city', 'country'])
        >>> locations
        <DataFrame 3pt x 2ft
                city      country
           ┌────────────────────────
         0 │ Cape Town  South Africa
         1 │      Lima          Peru
         2 │    Moscow        Russia
        >

        Split with an index for ``rule``.

        >>> lst = [['AGG932', 44],
        ...        ['AGG734', 11],
        ...        ['HEQ892', 1],
        ...        ['LEQ331', 2]]
        >>> fts = ['product', 'quantity']
        >>> inventory = nimble.data(lst, featureNames=fts)
        >>> inventory.features.splitByParsing(0, 3, ['category', 'id'])
        >>> inventory
        <DataFrame 4pt x 3ft
             category   id  quantity
           ┌────────────────────────
         0 │   AGG     932     44
         1 │   AGG     734     11
         2 │   HEQ     892      1
         3 │   LEQ     331      2
        >

        Keywords
        --------
        parse, separate, pattern, break, apart, detect, tidy, tidyr
        """
        if not (isinstance(rule, (int, np.integer, str))
                or hasattr(rule, '__iter__')
                or hasattr(rule, '__call__')):
            msg = "rule must be an integer, string, iterable of integers "
            msg += "and/or strings, or a function"
            raise InvalidArgumentType(msg)

        splitList = []
        numResultingFts = len(resultingNames)
        for i, value in enumerate(self._base[:, feature]):
            if isinstance(rule, str):
                splitList.append(value.split(rule))
            elif isinstance(rule, (int, np.number)):
                splitList.append([value[:rule], value[rule:]])
            elif hasattr(rule, '__iter__'):
                split = []
                startIdx = 0
                for item in rule:
                    if isinstance(item, str):
                        split.append(value[startIdx:].split(item)[0])
                        # find index of value from startIdx on, o.w. will only
                        # ever return first instance. Add len of previous
                        # values to get actual index and add one to bypass this
                        # item in next iteration
                        startIdx = (value[startIdx:].index(item) +
                                    len(value[:startIdx]) + 1)
                    elif isinstance(item, (int, np.integer)):
                        split.append(value[startIdx:item])
                        startIdx = item
                    else:
                        msg = "A list of items for rule may only contain "
                        msg += " integers and strings"
                        raise InvalidArgumentType(msg)
                split.append(value[startIdx:])
                splitList.append(split)
            else:
                splitList.append(rule(value))
            if len(splitList[-1]) != numResultingFts:
                msg = f"The value at index {i} split into "
                msg += f"{len(splitList[-1])} values, but resultingNames "
                msg += f"contains {numResultingFts} features"
                raise InvalidArgumentValueCombination(msg)

        featureIndex = self.getIndex(feature)
        numRetFeatures = len(self) - 1 + numResultingFts

        self._splitByParsing_implementation(featureIndex, splitList,
                                            numRetFeatures, numResultingFts)

        fNames = self.getNames()[:featureIndex]
        fNames.extend(resultingNames)
        fNames.extend(self.getNames()[featureIndex + 1:])
        self._base._dims[1] = numRetFeatures
        self.setNames(fNames, useLog=False)


    @limitedTo2D
    @prepLog
    def repeat(self, totalCopies, copyFeatureByFeature, *,
               useLog=None): # pylint: disable=unused-argument
        """
        Create an object using copies of this object's features.

        Copies of this object will be stacked horizontally. The returned
        object will have the same number of points as this object and
        the number of features will be equal to the number of features
        in this object times ``totalCopies``. If this object contains
        featureNames, each feature name will have "_#" appended, where #
        is the number of the copy made.

        Parameters
        ----------
        totalCopies : int
            The number of times a copy of the data in this object will
            be present in the returned object.
        copyFeatureByFeature : bool
            When False, copies are made as if iterating through the
            features in this object ``totalCopies`` times. When True,
            copies are made as if the object is only iterated once,
            making ``totalCopies`` copies of each feature before
            iterating to the next feature.
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
            Object containing the copied data.

        Examples
        --------
        Single feature

        >>> X = nimble.data([[1], [2], [3]])
        >>> X.features.setNames(['a'])
        >>> X.features.repeat(totalCopies=3,
        ...                      copyFeatureByFeature=False)
        <Matrix 3pt x 3ft
             a_1  a_2  a_3
           ┌──────────────
         0 │  1    1    1
         1 │  2    2    2
         2 │  3    3    3
        >

        Two-dimensional, copyFeatureByFeature is False

        >>> X = nimble.data([[1, 2], [3, 4], [5, 6]])
        >>> X.features.setNames(['a', 'b'])
        >>> X.features.repeat(totalCopies=2,
        ...                      copyFeatureByFeature=False)
        <Matrix 3pt x 4ft
             a_1  b_1  a_2  b_2
           ┌───────────────────
         0 │  1    2    1    2
         1 │  3    4    3    4
         2 │  5    6    5    6
        >

        Two-dimensional, copyFeatureByFeature is True

        >>> X = nimble.data([[1, 2], [3, 4], [5, 6]])
        >>> X.features.setNames(['a', 'b'])
        >>> X.features.repeat(totalCopies=2,
        ...                      copyFeatureByFeature=True)
        <Matrix 3pt x 4ft
             a_1  a_2  b_1  b_2
           ┌───────────────────
         0 │  1    1    2    2
         1 │  3    3    4    4
         2 │  5    5    6    6
        >

        Keywords
        --------
        stack, horizontal, duplicate, replicate, hstack
        """
        return self._repeat(totalCopies, copyFeatureByFeature)

    ###################
    # Query functions #
    ###################
    @limitedTo2D
    def unique(self):
        """
        Only the unique features from this object.

        Any repeated features will be removed from the returned object.
        If feature names are present, the feature name of the first
        instance of the unique feature in this object will be assigned.

        Returns
        -------
        nimble Base object
            The object containing only unique features.

        Examples
        --------
        >>> lst = [['a', 1, 3, 'a'],
        ...        ['a', 5, 6, 'a'],
        ...        ['b', 7, 1, 'b'],
        ...        ['c', 2, 9, 'c']]
        >>> ftNames = ['f1', 'f2', 'f3', 'f1_copy']
        >>> X = nimble.data(lst, featureNames=ftNames)
        >>> uniqueFeatures = X.features.unique()
        >>> uniqueFeatures
        <DataFrame 4pt x 3ft
             f1  f2  f3
           ┌───────────
         0 │ a   1   3
         1 │ a   5   6
         2 │ b   7   1
         3 │ c   2   9
        >

        Keywords
        --------
        distinct, different
        """
        return self._unique()

    @limitedTo2D
    def report(self, basicStatistics=True, extraStatisticFunctions=(), 
               dtypes=False, *, useLog=None):
        """
        Report containing a summary and statistics for each feature.

        Produce a report, as a nimble List object, containing statistic
        and summary information about each feature in this object. The
        default will include mean, mode, minimum, Q1, median, Q3,
        maximum, uniqueCount, count, and standardDeviation.

        Parameters
        ----------
        basicStatistics : bool, list
            True will include mean, mode, minimum, Q1, median, Q3,
            maximum, uniqueCount, count, and standardDeviation. False
            will only use functions in ``extraStatisticFunctions``. To
            limit the report to a selection of basic statistics, a list
            of strings can be provided, e.g.
            ['mean', 'standardDeviation', 'minimum', 'maximum']
        extraStatisticFunctions : list
            A list of functions to include in the report. Functions must
            accept a feature view as the only input and output a single
            value.
        dtypes : bool
            True will add an additional column to the features report
            which would include the data type of each feature in the
            dataset.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        Base.report, nimble.calculate.statistic

        Keywords
        --------
        statistic, summary, information, analyze, statistics, stats,
        columns, info, information, describe, about
        """
        allow = ['mean', 'mode', 'minimum', 'Q1', 'median', 'Q3', 'maximum',
                 'uniqueCount', 'count', 'standardDeviation']
        if basicStatistics is True:
            stats = allow
        elif basicStatistics:
            if any(stat not in allow for stat in basicStatistics):
                allowed = prettyListString(allow, True, itemStr=quoteStrings)
                msg = 'Invalid value found in basicStatistics. Allowed '
                msg += f'values are {allowed}'
                raise InvalidArgumentValue(msg)
            stats = basicStatistics
        else:
            stats = []

        fnames = ['index']
        for stat in stats:
            fnames.append(stat)
        fnames.extend(func.__name__ for func in extraStatisticFunctions)

        counter = Counter(fnames)
        remaining = dict(counter)
        # extra function names could conflict
        if len(set(fnames)) != len(fnames):
            editedNames = []
            # add a unique integer to any duplicate names
            for val in fnames:
                if counter[val] > 1:
                    diff = counter[val] - remaining[val]
                    editedNames.append(f'{val} ({diff})')
                    remaining[val] -= 1
                else:
                    editedNames.append(val)
            fnames = editedNames

        pnames = self._getNamesNoGeneration()

        results = []
        quartiles = {'Q1': 0, 'median': 1, 'Q3': 2}
        for i, ft in enumerate(self):
            row = [i]
            quartileCalcs = None
            for stat in stats:
                try:
                    if stat in quartiles:
                        if quartileCalcs is None:
                            quartileCalcs = nimble.calculate.quartiles(ft)
                        row.append(quartileCalcs[quartiles[stat]])
                    else:
                        func = getattr(nimble.calculate, stat)
                        row.append(func(ft))
                except ValueError:
                    row.append(np.nan)

            for func in extraStatisticFunctions:
                row.append(func(ft))
            results.append(row)

        report = nimble.data(results, pnames, fnames, useLog=False)
        
        def unifyingType(data):
            toEval = data._data
            featureCount = range(len(data.features))

            if hasattr(toEval, 'dtypes'):
                featureTypeList = [[toEval.dtypes.tolist()[i].name] for i in featureCount ]
            
            elif hasattr(toEval, 'dtype'):
                featureTypeList = [[toEval.dtype.name] for i in featureCount]

            else:
                featureTypeList = [['object'] for i in featureCount]
                    
            return featureTypeList
        
        if dtypes is True:
            # if this calculation has not been run previously
            if 'dataType' not in report.features.getNames():
                featureDtype = unifyingType(self._base)
                if featureDtype: 
                    featureTypes = nimble.data(featureDtype, featureNames=['dataType'])
                    report.features.append(featureTypes)

        handleLogging(useLog, 'report', "feature", str(report))

        return report

    #########################
    # Statistical functions #
    #########################
    @limitedTo2D
    def similarities(self, similarityFunction):
        """
        Calculate similarities between features.

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

        See Also
        --------
        statistics, nimble.calculate.similarity

        Keywords
        --------
        correlation, covariance, sample covariance,
        population covariance, dot product, similarity, relationship,
        cor, pearson, spearman, correlation coefficient
        """
        return self._similarities(similarityFunction)

    @limitedTo2D
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
        nimble Base object

        See Also
        --------
        similarities, nimble.calculate.statistic

        Keywords
        --------
        max, mean, median, min, population std, population standard
        deviation, proportion missing, proportion zero, sample standard
        deviation, sample std, standard deviation, std, unique count,
        stats, compute, calculate
        """
        return self._statistics(statisticsFunction, groupByFeature)


    ############
    # plotting #
    ############

    @limitedTo2D
    def plot(self, features=None, horizontal=False, outPath=None,
             show=True, figureID=None, title=True, xAxisLabel=True,
             yAxisLabel=True, legendTitle=None, **kwargs):
        """
        Bar chart comparing features.

        Each value in the object is considered to be the height of a
        bar in the chart. Bars will be grouped by each feature and bar
        colors indicate each point. If multiple bar colors are
        necessary, a legend mapping the pointName (when available) or
        point index to its bar color will be added.

        Parameters
        ----------
        features : list of identifiers, None
            List of feature names and/or indices to plot. None will
            apply to all features.
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
            None, features, None, horizontal, outPath, show, figureID, title,
            xAxisLabel, yAxisLabel, legendTitle, **kwargs)

    @limitedTo2D
    def plotMeans(self, features=None, horizontal=False, outPath=None,
                  show=True, figureID=None, title=True, xAxisLabel=True,
                  yAxisLabel=True, **kwargs):
        """
        Plot feature means with 95% confidence interval bars.

        The 95% confidence interval for each feature is calculated using
        the critical value from the two-sided Student's t-distribution.

        Parameters
        ----------
        features : list of identifiers, None
            List of feature names and/or indices to plot. None will
            apply to all features.
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
            nimble.calculate.mean, features, True, horizontal, outPath,
            show, figureID, title, xAxisLabel, yAxisLabel, None, **kwargs)

    @limitedTo2D
    def plotStatistics(
            self, statistic, features=None, horizontal=False, outPath=None,
            show=True, figureID=None, title=True, xAxisLabel=True,
            yAxisLabel=True, legendTitle=None, **kwargs):
        """
        Bar chart comparing an aggregate statistic between features.

        The bars in the plot represent the output of the ``statistic``
        function applied to each feature. Typically, functions return a
        single numeric value, however, the function may return a feature
        vector. In that case, each point in the object returned by
        ``statistic`` is considered to be the heights of separate bars
        for that feature. Bars will be grouped by each feature and
        bar colors indicate each point. If multiple bar colors are
        necessary, a legend mapping the pointName (when available) or
        point index to its bar color will be added.

        Parameters
        ----------
        statistic : function
            Functions take a feature view the only required argument.
            Common statistic functions can be found in nimble.calculate.
        features : list of identifiers, None
            List of feature names and/or indices to plot. None will
            apply to all features.
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
        matplotlib.pyplot.bar, nimble.calculate.statistic

        Keywords
        --------
        bar chart, display, visualize, graphics
        """
        self._plotComparison(
            statistic, features, False, horizontal, outPath, show, figureID,
            title, xAxisLabel, yAxisLabel, legendTitle, **kwargs)

    ################
    # Stats methods #
    ###############

    @limitedTo2D
    def max(self, groupByFeature=None):
        """
        Returns a nimble object representing the maximum
        value along the features axis.

        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.

        See Also
        --------
        minimum

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.max()
        <Matrix 1pt x 3ft
               0  1   2
             ┌─────────
         max │ 3  22  5
        >
        """
        return self._max(groupByFeature)

    @limitedTo2D
    def mean(self, groupByFeature=None):
        """
        Returns a nimble object representing the mean
        value along the features axis.

        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes 
            the result grouped by the point values along the chosen
            feature axis.

        See Also
        --------
        median, mode

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.mean()
        <Matrix 1pt x 3ft
                  0      1       2
              ┌─────────────────────
         mean │ 1.500  22.000  3.500
        >
        """
        return self._mean(groupByFeature)

    @limitedTo2D
    def median(self, groupByFeature=None):
        """
        Returns a nimble object representing the median
        value along the features axis.

        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.

        See Also
        --------
        mean

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.median()
        <Matrix 1pt x 3ft
                    0      1       2
                ┌─────────────────────
         median │ 1.500  22.000  3.500
        >
        """
        return self._median(groupByFeature)

    @limitedTo2D
    def min(self, groupByFeature=None):
        """
        Returns a nimble object representing the minimum
        value along the features axis.

        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.
                       
        See Also
        --------
        maximum

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.min()
        <Matrix 1pt x 3ft
               0  1   2
             ┌─────────
         min │ 0  22  2
        >
        """
        return self._min(groupByFeature)

    @limitedTo2D
    def uniqueCount(self, groupByFeature=None):
        """
        Returns a nimble object representing the number of unique
        values along the features axis.

        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.uniqueCount()
        <Matrix 1pt x 3ft
                       0  1  2
                     ┌────────
         uniquecount │ 2  1  2
        >
        """
        
        return self._uniqueCount(groupByFeature)

    @limitedTo2D
    def proportionMissing(self, groupByFeature=None):
        """
        Returns a nimble object representing the proportion of
        values that are None or NaN along the features axis.

        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.

        Examples
        --------
        >>> lst = [[float('nan'), 2, 0], [0, 2, float('nan')]]
        >>> X = nimble.data(lst)
        >>> X.features.proportionMissing()
        <Matrix 1pt x 3ft
                               0      1      2
                           ┌────────────────────
         proportionmissing │ 0.500  0.000  0.500
        >
        """
        return self._proportionMissing(groupByFeature)

    @limitedTo2D
    def proportionZero(self, groupByFeature=None):
        """
        Returns a nimble object representing the proportion of values
        that are equal to zero along the features axis.

        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.

        Examples
        --------
        >>> lst = [[1, 2, 0], [0, 0, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.proportionZero()
        <Matrix 1pt x 3ft
                            0      1      2
                        ┌────────────────────
         proportionzero │ 0.500  0.500  0.500
        >
        """
        return self._proportionZero(groupByFeature)

    @limitedTo2D
    def standardDeviation(self, groupByFeature=None):
        """
        Returns a nimble object representing the standard deviation 
        along the features axis.

        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.standardDeviation()
        <Matrix 1pt x 3ft
                               0      1      2
                           ┌────────────────────
         standarddeviation │ 2.121  0.000  2.121
        >
        """
        return self._standardDeviation(groupByFeature)

    @limitedTo2D
    def populationStandardDeviation(self, groupByFeature=None):
        """
        Returns a nimble object representing the population standard
        deviation along the features axis.

        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.populationStandardDeviation()
        <Matrix 1pt x 3ft
                                        0      1      2
                                    ┌────────────────────
         populationstandarddeviati─ │ 1.500  0.000  1.500
        >
        """
        return self._populationStandardDeviation(groupByFeature)
   
    
    @limitedTo2D
    def mode(self,grouByFeature=None):
        """
        Returns a nimble object representing the mode along 
        the features axis.
        
        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.
        
        Examples
        --------
        >>> lst = [[0, 2, 2], [0, 2, 5], [2, 3, 2]]
        >>> X = nimble.data(lst)
        >>> X.features.mode()
        <Matrix 1pt x 3ft
                0  1  2
              ┌────────
         mode │ 0  2  2
        >
        """
        return self._mode(grouByFeature)
    
    @limitedTo2D
    def sum(self, groupByFeature=None):
        """
        Returns a nimble object representing the sum along
        the features axis.
        
        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.
            
        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.sum()
        <Matrix 1pt x 3ft
               0  1   2
             ┌─────────
         sum │ 3  44  7
        >
        """
        return self._sum(groupByFeature)
    
    @limitedTo2D
    def variance(self, groupByFeature=None):
        """
        Returns a nimble object representing the variance
        along the features axis.
        
        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.
            
        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.variance()
        <Matrix 1pt x 3ft
                      0      1      2
                  ┌────────────────────
         variance │ 4.500  0.000  4.500
        >
        """
        return self._variance(groupByFeature)
    
    @limitedTo2D
    def medianAbsoluteDeviation(self, groupByFeature=None):
        """
        Returns a nimble object representing the median absolute
        deviation along the features axis.
        
        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.
        
        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.medianAbsoluteDeviation()
        <Matrix 1pt x 3ft
                                     0      1      2
                                 ┌────────────────────
         medianabsolutedeviation │ 1.500  0.000  1.500
        >
        """
        return self._medianAbsoluteDeviation(groupByFeature)
    
    @limitedTo2D
    def quartiles(self, groupByFeature=None):
        """
        Returns a nimble object representing the quartiles
        along the features axis.
        
        Parameters
        ----------
        groupByFeature : identifier, None
            An optional index or name of the feature that makes
            the result grouped by the point values along the chosen
            feature axis.
        
        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.features.quartiles()
        <Matrix 3pt x 3ft
                       0      1       2
                   ┌─────────────────────
         quartile1 │ 0.750  22.000  2.750
         quartile2 │ 1.500  22.000  3.500
         quartile3 │ 2.250  22.000  4.250
        >
        """
        return self._quartiles(groupByFeature)
    

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
    def _setNames(self, assignments, oldIdentifiers=None):
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
    def _sort(self, by, reverse,):
        pass

    @abstractmethod
    def _transform(self, function, limitTo):
        pass

    @abstractmethod
    def _calculate(self, function, limitTo):
        pass

    @abstractmethod
    def _matching(self, function):
        pass

    @abstractmethod
    def _insert(self, insertBefore, toInsert, append=False):
        pass

    @limitedTo2D
    def _replace(self, data, locations):
        pass

    @abstractmethod
    def _mapReduce(self, mapper, reducer):
        pass

    @abstractmethod
    def _permute(self, order=None):
        pass

    @abstractmethod
    def _fillMatching(self, match, fill, limitTo, **kwarguments):
        pass

    @abstractmethod
    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
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
    def _statistics(self, statisticsFunction, groupByFeature):
        pass

    @abstractmethod
    def _plotComparison(self, statistic, identifiers, confidenceIntervals,
                        horizontal, outPath, show, figureID, title,
                        xAxisLabel, yAxisLabel, legendTitle, **kwargs):
        pass

    @abstractmethod
    def _getNamesNoGeneration(self):
        pass
    
    @abstractmethod
    def _max(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _mean(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _median(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _min(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _uniqueCount(self, groupByFeature=None):
      pass

    @abstractmethod
    def _proportionMissing(self, groupByFeature=None):
        pass

    @abstractmethod
    def _proportionZero(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _standardDeviation(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _populationStandardDeviation(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _mode(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _sum(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _variance(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _medianAbsoluteDeviation(self, groupByFeature=None):
        pass
    
    @abstractmethod
    def _quartiles(self, groupByFeature=None):
        pass
