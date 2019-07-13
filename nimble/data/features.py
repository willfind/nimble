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

from __future__ import absolute_import
from abc import abstractmethod

import numpy
import six

from nimble.logger import handleLogging
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValueCombination

class Features(object):
    """
    Methods that can be called on the a nimble Base objects feature axis.
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
        >>> data = nimble.identity('Matrix', 4,
        ...                        featureNames=['a', 'b', 'c', 'd'])
        >>> data.features.getName(1)
        'b'
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
        >>> data = nimble.identity('Matrix', 4,
        ...                        featureNames=['a', 'b', 'c', 'd'])
        >>> data.features.getNames()
        ['a', 'b', 'c', 'd']
        """
        return self._getNames()

    def setName(self, oldIdentifier, newName, useLog=None):
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
        ...                        featureNames=['a', 'b', 'c', 'd'])
        >>> data.features.setName('b', 'new')
        >>> data.features.getNames()
        ['a', 'new', 'c', 'd']
        """
        self._setName(oldIdentifier, newName, useLog)

    def setNames(self, assignments=None, useLog=None):
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
        ...                        featureNames=['a', 'b', 'c', 'd'])
        >>> data.features.setNames(['1', '2', '3', '4'])
        >>> data.features.getNames()
        ['1', '2', '3', '4']
        """
        self._setNames(assignments, useLog)

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
        >>> data = nimble.identity('Matrix', 4,
        ...                        featureNames=['a', 'b', 'c', 'd'])
        >>> data.features.getIndex('c')
        2
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
        >>> data = nimble.identity('Matrix', 4,
        ...                        featureNames=['a', 'b', 'c', 'd'])
        >>> data.features.getIndices(['c', 'a', 'd'])
        [2, 0, 3]
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
        >>> data = nimble.identity('Matrix', 4,
        ...                        featureNames=['a', 'b', 'c', 'd'])
        >>> data.features.hasName('a')
        True
        >>> data.features.hasName('e')
        False
        """
        return self._hasName(name)

    #########################
    # Structural Operations #
    #########################

    def copy(self, toCopy=None, start=None, end=None, number=None,
             randomize=False, useLog=None):
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
        >>> raw = [[1, 2, 3, 4],
        ...        [1, 2, 3, 4],
        ...        [1, 2, 3, 4],
        ...        [1, 2, 3, 4]]
        >>> data = nimble.createData('Matrix', raw,
        ...                          pointNames=['a', 'b', 'c', 'd'],
        ...                          featureNames=['1', '2', '3', '4'])
        >>> single = data.features.copy('1')
        >>> single
        Matrix(
            [[1.000]
             [1.000]
             [1.000]
             [1.000]]
            pointNames={'a':0, 'b':1, 'c':2, 'd':3}
            featureNames={'1':0}
            )
        >>> multiple = data.features.copy(['1', 3])
        >>> multiple
        Matrix(
            [[1.000 4.000]
             [1.000 4.000]
             [1.000 4.000]
             [1.000 4.000]]
            pointNames={'a':0, 'b':1, 'c':2, 'd':3}
            featureNames={'1':0, '4':1}
            )
        >>> func = data.features.copy(lambda ft: sum(ft) < 10)
        >>> func
        Matrix(
            [[1.000 2.000]
             [1.000 2.000]
             [1.000 2.000]
             [1.000 2.000]]
            pointNames={'a':0, 'b':1, 'c':2, 'd':3}
            featureNames={'1':0, '2':1}
            )
        >>> strFunc = data.features.copy("a>=3")
        >>> strFunc
        Matrix(
            [[3.000 4.000]
             [3.000 4.000]
             [3.000 4.000]
             [3.000 4.000]]
            pointNames={'a':0, 'b':1, 'c':2, 'd':3}
            featureNames={'3':0, '4':1}
            )
        >>> startEnd = data.features.copy(start=1, end=2)
        >>> startEnd
        Matrix(
            [[2.000 3.000]
             [2.000 3.000]
             [2.000 3.000]
             [2.000 3.000]]
            pointNames={'a':0, 'b':1, 'c':2, 'd':3}
            featureNames={'2':0, '3':1}
            )
        >>> numberNoRandom = data.features.copy(number=2)
        >>> numberNoRandom
        Matrix(
            [[1.000 2.000]
             [1.000 2.000]
             [1.000 2.000]
             [1.000 2.000]]
            pointNames={'a':0, 'b':1, 'c':2, 'd':3}
            featureNames={'1':0, '2':1}
            )
        >>> nimble.randomness.setRandomSeed(42)
        >>> numberRandom = data.features.copy(number=2, randomize=True)
        >>> numberRandom
        Matrix(
            [[1.000 4.000]
             [1.000 4.000]
             [1.000 4.000]
             [1.000 4.000]]
            pointNames={'a':0, 'b':1, 'c':2, 'd':3}
            featureNames={'1':0, '4':1}
            )
        """
        return self._copy(toCopy, start, end, number, randomize, useLog)

    def extract(self, toExtract=None, start=None, end=None, number=None,
                randomize=False, useLog=None):
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

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> single = data.features.extract('a')
        >>> single
        List(
            [[1.000]
             [0.000]
             [0.000]]
            featureNames={'a':0}
            )
        >>> data
        List(
            [[0.000 0.000]
             [1.000 0.000]
             [0.000 1.000]]
            featureNames={'b':0, 'c':1}
            )

        Extract multiple features.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> multiple = data.features.extract(['a', 2])
        >>> multiple
        List(
            [[1.000 0.000]
             [0.000 0.000]
             [0.000 1.000]]
            featureNames={'a':0, 'c':1}
            )
        >>> data
        List(
            [[0.000]
             [1.000]
             [0.000]]
            featureNames={'b':0}
            )

        Extract feature when the function returns True.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> func = data.features.extract(lambda ft: ft[2] == 1)
        >>> func
        List(
            [[0.000]
             [0.000]
             [1.000]]
            featureNames={'c':0}
            )
        >>> data
        List(
            [[1.000 0.000]
             [0.000 1.000]
             [0.000 0.000]]
            featureNames={'a':0, 'b':1}
            )

        Extract feature when the string filter function returns True.

        >>> data = nimble.identity('List', 3,
        ...                        featureNames=['a', 'b', 'c'],
        ...                        pointNames=['p1', 'p2', 'p3'])
        >>> strFunc = data.features.extract("p2 != 0")
        >>> strFunc
        List(
            [[0.000]
             [1.000]
             [0.000]]
            pointNames={'p1':0, 'p2':1, 'p3':2}
            featureNames={'b':0}
            )
        >>> data
        List(
            [[1.000 0.000]
             [0.000 0.000]
             [0.000 1.000]]
            pointNames={'p1':0, 'p2':1, 'p3':2}
            featureNames={'a':0, 'c':1}
            )

        Extract features from the inclusive start to the inclusive end.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> startEnd = data.features.extract(start=1, end=2)
        >>> startEnd
        List(
            [[0.000 0.000]
             [1.000 0.000]
             [0.000 1.000]]
            featureNames={'b':0, 'c':1}
            )
        >>> data
        List(
            [[1.000]
             [0.000]
             [0.000]]
            featureNames={'a':0}
            )

        Select a set number to extract, starting from the first feature.

        >>> nimble.randomness.setRandomSeed(42)
        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> numberNoRandom = data.features.extract(number=2)
        >>> numberNoRandom
        List(
            [[1.000 0.000]
             [0.000 1.000]
             [0.000 0.000]]
            featureNames={'a':0, 'b':1}
            )
        >>> data
        List(
            [[0.000]
             [0.000]
             [1.000]]
            featureNames={'c':0}
            )

        Select a set number to extract, choosing features at random.
        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> numberRandom = data.features.extract(number=2,
        ...                                      randomize=True)
        >>> numberRandom
        List(
            [[0.000 1.000]
             [0.000 0.000]
             [1.000 0.000]]
            featureNames={'c':0, 'a':1}
            )
        >>> data
        List(
            [[0.000]
             [1.000]
             [0.000]]
            featureNames={'b':0}
            )
        """
        return self._extract(toExtract, start, end, number, randomize, useLog)

    def delete(self, toDelete=None, start=None, end=None, number=None,
               randomize=False, useLog=None):
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

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.delete('a')
        >>> data
        List(
            [[0.000 0.000]
             [1.000 0.000]
             [0.000 1.000]]
            featureNames={'b':0, 'c':1}
            )

        Delete multiple features.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.delete(['a', 2])
        >>> data
        List(
            [[0.000]
             [1.000]
             [0.000]]
            featureNames={'b':0}
            )

        Delete feature when the function returns True.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.delete(lambda ft: ft[2] == 1)
        >>> data
        List(
            [[1.000 0.000]
             [0.000 1.000]
             [0.000 0.000]]
            featureNames={'a':0, 'b':1}
            )

        Delete feature when the string filter function returns True.

        >>> data = nimble.identity('List', 3,
        ...                        featureNames=['a', 'b', 'c'],
        ...                        pointNames=['p1', 'p2', 'p3'])
        >>> data.features.delete("p2 != 0")
        >>> data
        List(
            [[1.000 0.000]
             [0.000 0.000]
             [0.000 1.000]]
            pointNames={'p1':0, 'p2':1, 'p3':2}
            featureNames={'a':0, 'c':1}
            )

        Delete features from the inclusive start to the inclusive end.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.delete(start=1, end=2)
        >>> data
        List(
            [[1.000]
             [0.000]
             [0.000]]
            featureNames={'a':0}
            )

        Select a set number to delete, starting from the first feature.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.delete(number=2)
        >>> data
        List(
            [[0.000]
             [0.000]
             [1.000]]
            featureNames={'c':0}
            )

        Select a set number to delete, choosing features at random.

        >>> nimble.randomness.setRandomSeed(42)
        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.delete(number=2,  randomize=True)
        >>> data
        List(
            [[0.000]
             [1.000]
             [0.000]]
            featureNames={'b':0}
            )
        """
        self._delete(toDelete, start, end, number, randomize, useLog)

    def retain(self, toRetain=None, start=None, end=None, number=None,
               randomize=False, useLog=None):
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

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.retain('a')
        >>> data
        List(
            [[1.000]
             [0.000]
             [0.000]]
            featureNames={'a':0}
            )

        Retain multiple features.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.retain(['a', 2])
        >>> data
        List(
            [[1.000 0.000]
             [0.000 0.000]
             [0.000 1.000]]
            featureNames={'a':0, 'c':1}
            )

        Retain feature when the function returns True.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.retain(lambda ft: ft[2] == 1)
        >>> data
        List(
            [[0.000]
             [0.000]
             [1.000]]
            featureNames={'c':0}
            )

        Retain feature when the string filter function returns True.

        >>> data = nimble.identity('List', 3,
        ...                        featureNames=['a', 'b', 'c'],
        ...                        pointNames=['p1', 'p2', 'p3'])
        >>> data.features.retain("p2 != 0")
        >>> data
        List(
            [[0.000]
             [1.000]
             [0.000]]
            pointNames={'p1':0, 'p2':1, 'p3':2}
            featureNames={'b':0}
            )

        Retain features from the inclusive start to the inclusive end.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.retain(start=1, end=2)
        >>> data
        List(
            [[0.000 0.000]
             [1.000 0.000]
             [0.000 1.000]]
            featureNames={'b':0, 'c':1}
            )

        Select a set number to retain, starting from the first feature.

        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.retain(number=2)
        >>> data
        List(
            [[1.000 0.000]
             [0.000 1.000]
             [0.000 0.000]]
            featureNames={'a':0, 'b':1}
            )

        Select a set number to retain, choosing features at random.

        >>> nimble.randomness.setRandomSeed(42)
        >>> data = nimble.identity('List', 3)
        >>> data.features.setNames(['a', 'b', 'c'])
        >>> data.features.retain(number=2, randomize=True)
        >>> data
        List(
            [[0.000 1.000]
             [0.000 0.000]
             [1.000 0.000]]
            featureNames={'c':0, 'a':1}
            )
        """
        self._retain(toRetain, start, end, number, randomize, useLog)

    def count(self, condition):
        """
        The number of features which satisfy the condition.

        Parameters
        ----------
        condition : function
            May take two forms:
            * a function that when given a feature will return True if
              it is to be counted
            * a filter function, as a string, containing a comparison
              operator and a value (i.e "pt1<10")

        Returns
        -------
        int

        See Also
        --------
        Elements.count, Elements.countEachUniqueValue

        Examples
        --------
        Count using a python function.

        >>> def sumIsOne(ft):
        ...     return sum(ft) == 1
        >>> data = nimble.identity('Matrix', 3)
        >>> data.features.count(sumIsOne)
        3

        Count when the string filter function returns True.

        >>> data = nimble.identity('Matrix', 3,
        ...                        pointNames=['pt1', 'pt2', 'pt3'])
        >>> data.features.count("pt1 == 0")
        2
        """
        return self._count(condition)

    def sort(self, sortBy=None, sortHelper=None, useLog=None):
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
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        Sort by a given point using ``sortBy``.
        TODO

        Sort with a list of identifiers.

        >>> raw = [['home', 81, 3],
        ...        ['gard', 98, 10],
        ...        ['home', 14, 1],
        ...        ['home', 11, 3]]
        >>> cols = ['dept', 'ID', 'quantity']
        >>> orders = nimble.createData('DataFrame', raw,
        ...                            featureNames=cols)
        >>> orders.features.sort(sortHelper=['ID', 'quantity', 'dept'])
        >>> orders
        DataFrame(
            [[81 3  home]
             [98 10 gard]
             [14 1  home]
             [11 3  home]]
            featureNames={'ID':0, 'quantity':1, 'dept':2}
            )

        Sort using a comparator function.
        TODO

        Sort using a scoring function.
        TODO
        """
        self._sort(sortBy, sortHelper, useLog)

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

    def transform(self, function, features=None, useLog=None):
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
        calculate

        Examples
        --------
        Transform all features; apply to all points.

        >>> data = nimble.ones('Matrix', 3, 5)
        >>> data.features.transform(lambda ft: ft + 2)
        >>> data
        Matrix(
            [[3.000 3.000 3.000 3.000 3.000]
             [3.000 3.000 3.000 3.000 3.000]
             [3.000 3.000 3.000 3.000 3.000]]
            )

        Transform all features; apply to certain points. Note that the
        function recieves a read-only view of each feature, so we need
        to make a copy in order to modify any specific data.

        >>> def transformMiddlePoint(ft):
        ...     ftList = ft.copy(to='python list', outputAs1D=True)
        ...     ftList[1] += 4
        ...     return ftList
        >>> data = nimble.ones('Matrix', 3, 5)
        >>> data.features.transform(transformMiddlePoint)
        >>> data
        Matrix(
            [[1.000 1.000 1.000 1.000 1.000]
             [5.000 5.000 5.000 5.000 5.000]
             [1.000 1.000 1.000 1.000 1.000]]
            )

        Transform a subset of features.

        >>> data = nimble.ones('Matrix', 3, 5)
        >>> data.features.transform(lambda ft: ft + 6, features=[1, 3])
        >>> data
        Matrix(
            [[1.000 7.000 1.000 7.000 1.000]
             [1.000 7.000 1.000 7.000 1.000]
             [1.000 7.000 1.000 7.000 1.000]]
            )
        """
        self._transform(function, features, useLog)

    ###########################
    # Higher Order Operations #
    ###########################

    def calculate(self, function, features=None, useLog=None):
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
        Apply calculation to all features; apply to all points.

        >>> data = nimble.ones('Matrix', 3, 5)
        >>> addTwo = data.features.calculate(lambda ft: ft + 2)
        >>> addTwo
        Matrix(
            [[3.000 3.000 3.000 3.000 3.000]
             [3.000 3.000 3.000 3.000 3.000]
             [3.000 3.000 3.000 3.000 3.000]]
            )

        Apply calculation to all features; apply to certain points. Note
        that the function recieves a read-only view of each feature, so
        a copy is necessary to modify any specific data.

        >>> def changeMiddlePoint(ft):
        ...     ftList = ft.copy(to='python list', outputAs1D=True)
        ...     ftList[1] += 4
        ...     return ftList
        >>> data = nimble.ones('Matrix', 3, 5)
        >>> middleChange = data.features.calculate(changeMiddlePoint)
        >>> middleChange
        Matrix(
            [[1.000 1.000 1.000 1.000 1.000]
             [5.000 5.000 5.000 5.000 5.000]
             [1.000 1.000 1.000 1.000 1.000]]
            )

        Apply calculation to a subset of features.

        >>> ftNames = ['f1', 'f2', 'f3']
        >>> data = nimble.identity('Matrix', 3, featureNames=ftNames)
        >>> calc = data.features.calculate(lambda ft: ft + 6,
        ...                                features=[2, 0])
        >>> calc
        Matrix(
            [[6.000 7.000]
             [6.000 6.000]
             [7.000 6.000]]
            featureNames={'f3':0, 'f1':1}
            )
        """
        return self._calculate(function, features, useLog)

    def add(self, toAdd, insertBefore=None, useLog=None):
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
        toAdd : nimble Base object
            The nimble Base object whose contents we will be including
            in this object. Must have the same point names as the
            calling object, but not necessarily in the same order. Must
            not share any feature names with the calling object.
        insertBefore : identifier
            The index or feature name prior to which the data from
            toAdd will be inserted. The default value, None, indicates
            that the data will be inserted to the right of all features
            in this object, or in other words: appended to the end of
            the current features.
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

        >>> data = nimble.zeros('Matrix', 3, 2)
        >>> toAdd = nimble.ones('Matrix', 3, 2)
        >>> data.features.add(toAdd)
        >>> data
        Matrix(
            [[0.000 0.000 1.000 1.000]
             [0.000 0.000 1.000 1.000]
             [0.000 0.000 1.000 1.000]]
            )

        Reorder names.

        >>> rawData = [[1, 2], [5, 6]]
        >>> data = nimble.createData('Matrix', rawData,
        ...                          pointNames=['a', 'b'])
        >>> rawAdd = [[7, 8], [3, 4]]
        >>> toAdd = nimble.createData('Matrix', rawAdd,
        ...                           pointNames=['b', 'a'])
        >>> data.features.add(toAdd)
        >>> data
        Matrix(
            [[1.000 2.000 3.000 4.000]
             [5.000 6.000 7.000 8.000]]
            pointNames={'a':0, 'b':1}
            )

        Insert before another point; mixed object types.

        >>> rawData = [[1, 4], [5, 8]]
        >>> data = nimble.createData('Matrix', rawData,
        ...                          featureNames=['1', '4'])
        >>> rawAdd = [[2, 3], [6, 7]]
        >>> toAdd = nimble.createData('List', rawAdd,
        ...                           featureNames=['2', '3'])
        >>> data.features.add(toAdd, insertBefore='4')
        >>> data
        Matrix(
            [[1.000 2.000 3.000 4.000]
             [5.000 6.000 7.000 8.000]]
            featureNames={'1':0, '2':1, '3':2, '4':3}
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
            Input a feature and output an iterable containing
            two-tuple(s) of mapping identifiers and feature values.
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
        mapReduce the counts of data types in the features.

        >>> def typeMapper(ft):
        ...     ftType = str(type(ft[0]))
        ...     return [(ftType, 1)]
        >>> def typeReducer(ftType, totals):
        ...     return (ftType, sum(totals))
        >>> raw = [[61500, 'Open', 'Chicago Bears'],
        ...        [71228, 'Dome', 'Atlanta Falcons'],
        ...        [77000, 'Open', 'Kansas City Chiefs'],
        ...        [72968, 'Dome', 'New Orleans Saints'],
        ...        [76500, 'Open', 'Miami Dolphins']]
        >>> ftNames = ['CAPACITY', 'ROOF_TYPE', 'TEAM']
        >>> data = nimble.createData('Matrix', raw,
        ...                          featureNames=ftNames)
        >>> data.features.mapReduce(typeMapper, typeReducer)
        Matrix(
            [[<class 'int'> 1]
             [<class 'str'> 2]]
            )
        """
        return self._mapReduce(mapper, reducer, useLog)

    def shuffle(self, useLog=None):
        """
        Permute the indexing of the features to a random order.

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
        sufficiently random for large number of features.
        See random.shuffle()'s documentation.

        Examples
        --------
        >>> nimble.randomness.setRandomSeed(42)
        >>> raw = [[1, 2, 3, 4],
        ...        [1, 2, 3, 4],
        ...        [1, 2, 3, 4],
        ...        [1, 2, 3, 4]]
        >>> data = nimble.createData('DataFrame', raw)
        >>> data.features.shuffle()
        >>> data
        DataFrame(
            [[3.000 2.000 4.000 1.000]
             [3.000 2.000 4.000 1.000]
             [3.000 2.000 4.000 1.000]
             [3.000 2.000 4.000 1.000]]
            )
        """
        self._shuffle(useLog)

    def fill(self, match, fill, features=None, returnModified=False,
             useLog=None, **kwarguments):
        """
        Replace given values in each feature with other values.

        Fill matching values within each feature with a specified value
        based on the values in that feature. The ``fill`` value can be
        a constant or a determined based on unmatched values in the
        feature. The match and fill modules in nimble offer common
        functions for these operations.

        Parameters
        ----------
        match : value, list, or function
            * value - a value to locate within each feature
            * list - values to locate within each feature
            * function - must accept a single value and return True if
              the value is a match. Certain match types can be imported
              from nimble's match module: missing, nonNumeric, zero, etc
        fill : value or function
            * value - a value to fill each matching value in each
              feature
            * function - must be in the format fill(feature, match) or
              fill(feature, match, arguments) and return the transformed
              feature as a list of values. Certain fill methods can be
              imported from nimble's fill module: mean, median, mode,
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
        >>> data.features.fill('na', -1)
        >>> data
        Matrix(
            [[1.000  1.000 1.000 ]
             [1.000  1.000 1.000 ]
             [1.000  1.000 -1.000]
             [2.000  2.000 2.000 ]
             [-1.000 2.000 2.000 ]]
            )

        Fill using nimble's match and fill modules; limit to first
        feature. Note: None is converted to numpy.nan in nimble.

        >>> from nimble import match
        >>> from nimble import fill
        >>> raw = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, None],
        ...        [2, 2, 2],
        ...        [None, 2, 2]]
        >>> data = nimble.createData('Matrix', raw)
        >>> data.features.fill(match.missing, fill.mean, features=0)
        >>> data
        Matrix(
            [[1.000 1.000 1.000]
             [1.000 1.000 1.000]
             [1.000 1.000  nan ]
             [2.000 2.000 2.000]
             [1.250 2.000 2.000]]
            )
        """
        return self._fill(match, fill, features, returnModified, useLog,
                          **kwarguments)

    def normalize(self, subtract=None, divide=None, applyResultTo=None,
                  useLog=None):
        """
        Modify all features in this object using the given operations.

        Normalize the data by applying subtraction and division
        operations. A value of None for subtract or divide implies that
        no change will be made to the data in regards to that operation.

        Parameters
        ----------
        subtract : number, str, nimble Base object
            * number - a numerical denominator for dividing the data
            * str -  a statistical function (all of the same ones
              callable though featureStatistics)
            * nimble Base object - If a vector shaped object is given,
              then the value associated with each feature will be
              subtracted from all values of that feature. Otherwise, the
              values in the object are used for elementwise subtraction
        divide : number, str, nimble Base object
            * number - a numerical denominator for dividing the data
            * str -  a statistical function (all of the same ones
              callable though featureStatistics)
            * nimble Base object - If a vector shaped object is given,
              then the value associated with each feature will be used
              in division of all values for that feature. Otherwise, the
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
        Using statistics functions.

        >>> raw = [[5, 9.8, 92],
        ...        [3, 6.2, 58],
        ...        [2, 3.0, 29]]
        >>> pts = ['movie1', 'movie2', 'movie3']
        >>> fts = ['review1', 'review2', 'review3']
        >>> reviews = nimble.createData('Matrix', raw, pts, fts)
        >>> reviews.features.normalize(subtract='min')
        >>> reviews.features.normalize(divide='max')
        >>> reviews
        Matrix(
            [[1.000 1.000 1.000]
             [0.333 0.471 0.460]
             [0.000 0.000 0.000]]
            pointNames={'movie1':0, 'movie2':1, 'movie3':2}
            featureNames={'review1':0, 'review2':1, 'review3':2}
            )

        Using nimble objects.

        >>> raw = [[5, 9.8, 92],
        ...        [3, 6.2, 58],
        ...        [2, 3.0, 29]]
        >>> pts = ['movie1', 'movie2', 'movie3']
        >>> fts = ['review1', 'review2', 'review3']
        >>> reviews = nimble.createData('Matrix', raw, pts, fts)
        >>> means = reviews.features.statistics('mean')
        >>> stdDevs = reviews.features.statistics('standard deviation')
        >>> reviews.features.normalize(subtract=means, divide=stdDevs)
        >>> reviews
        Matrix(
            [[1.091  1.019  1.025 ]
             [-0.218 -0.039 -0.053]
             [-0.873 -0.980 -0.973]]
            pointNames={'movie1':0, 'movie2':1, 'movie3':2}
            featureNames={'review1':0, 'review2':1, 'review3':2}
            )
        """
        self._normalize(subtract, divide, applyResultTo, useLog)


    def splitByParsing(self, feature, rule, resultingNames, useLog=None):
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

        >>> raw = [['Cape Town, South Africa'],
        ...        ['Lima, Peru'],
        ...        ['Moscow, Russia']]
        >>> fts = ['location']
        >>> locations = nimble.createData('Matrix', raw,
        ...                               featureNames=fts)
        >>> locations.features.splitByParsing('location', ', ',
        ...                                   ['city', 'country'])
        >>> locations
        Matrix(
            [[Cape Town South Africa]
             [   Lima       Peru    ]
             [  Moscow     Russia   ]]
            featureNames={'city':0, 'country':1}
            )

        Split with an index for ``rule``.

        >>> raw = [['AGG932', 44],
        ...        ['AGG734', 11],
        ...        ['HEQ892', 1],
        ...        ['LEQ331', 2]]
        >>> fts = ['product', 'quantity']
        >>> inventory = nimble.createData('List', raw, featureNames=fts)
        >>> inventory.features.splitByParsing(0, 3, ['category', 'id'])
        >>> inventory
        List(
            [[AGG 932 44]
             [AGG 734 11]
             [HEQ 892 1 ]
             [LEQ 331 2 ]]
            featureNames={'category':0, 'id':1, 'quantity':2}
            )
        """
        if not (isinstance(rule, (int, numpy.integer, six.string_types))
                or hasattr(rule, '__iter__')
                or hasattr(rule, '__call__')):
            msg = "rule must be an integer, string, iterable of integers "
            msg += "and/or strings, or a function"
            raise InvalidArgumentType(msg)

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
                        raise InvalidArgumentType(msg)
                split.append(value[startIdx:])
                splitList.append(split)
            else:
                splitList.append(rule(value))
            if len(splitList[-1]) != numResultingFts:
                msg = "The value at index {0} split into ".format(i)
                msg += "{0} values, ".format(len(splitList[-1]))
                msg += "but resultingNames contains "
                msg += "{0} features".format(numResultingFts)
                raise InvalidArgumentValueCombination(msg)

        featureIndex = self.getIndex(feature)
        numRetFeatures = len(self) - 1 + numResultingFts

        self._splitByParsing_implementation(featureIndex, splitList,
                                            numRetFeatures, numResultingFts)

        fNames = self.getNames()[:featureIndex]
        fNames.extend(resultingNames)
        fNames.extend(self.getNames()[featureIndex + 1:])
        self._source._featureCount = numRetFeatures
        self.setNames(fNames, useLog=False)

        handleLogging(useLog, 'prep', 'features.splitByParsing',
                      self._source.getTypeString(), Features.splitByParsing,
                      feature, rule, resultingNames)

    def duplicate(self, totalCopies=2, copyFeatureByFeature=False):
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
            When False, the default, copies are made as if iterating
            through each feature in this object ``totalCopies`` times.
            When True, copies are made as if the object is only iterated
            once, making ``totalCopies`` copies of each feature before
            iterating to the next feature. Note: when this object is a
            single feature, the output will be the same either way.

        Returns
        -------
        nimble Base object
            Object containing the copied data.

        Examples
        --------
        Single feature

        >>> data = nimble.createData('Matrix', [[1], [2], [3]])
        >>> data.features.setNames(['a'])
        >>> data.features.duplicate(totalCopies=3)
        Matrix(
            [[1.000 1.000 1.000]
             [2.000 2.000 2.000]
             [3.000 3.000 3.000]]
            featureNames={'a_0':0, 'a_1':1, 'a_2':2}
            )

        Two-dimensional, copyFeatureByFeature is False

        >>> data = nimble.createData('Matrix', [[1, 2], [3, 4], [5, 6]])
        >>> data.features.setNames(['a', 'b'])
        >>> data.features.duplicate(totalCopies=2,
        ...                         copyFeatureByFeature=False)
        Matrix(
            [[1.000 2.000 1.000 2.000]
             [3.000 4.000 3.000 4.000]
             [5.000 6.000 5.000 6.000]]
            featureNames={'a_0':0, 'b_0':1, 'a_1':2, 'b_1':3}
            )

        Two-dimensional, copyFeatureByFeature is True

        >>> data = nimble.createData('Matrix', [[1, 2], [3, 4], [5, 6]])
        >>> data.features.setNames(['a', 'b'])
        >>> data.features.duplicate(totalCopies=2,
        ...                         copyFeatureByFeature=True)
        Matrix(
            [[1.000 1.000 2.000 2.000]
             [3.000 3.000 4.000 4.000]
             [5.000 5.000 6.000 6.000]]
            featureNames={'a_0':0, 'a_1':1, 'b_0':2, 'b_1':3}
            )
        """
        return self._duplicate(totalCopies, copyFeatureByFeature)

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
        """
        return self._nonZeroIterator()

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
        >>> raw = [['a', 1, 3, 'a'],
        ...        ['a', 5, 6, 'a'],
        ...        ['b', 7, 1, 'b'],
        ...        ['c', 2, 9, 'c']]
        >>> ftNames = ['f1', 'f2', 'f3', 'f1_copy']
        >>> data = nimble.createData('Matrix', raw,
        ...                          featureNames=ftNames)
        >>> uniqueFeatures = data.features.unique()
        >>> uniqueFeatures
        Matrix(
            [[a 1 3]
             [a 5 6]
             [b 7 1]
             [c 2 9]]
            featureNames={'f1':0, 'f2':1, 'f3':2}
            )
        """
        return self._unique()

    #########################
    # Statistical functions #
    #########################

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
        nimble Base object
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
        nimble Base object
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
    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
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
    def _statistics(self, statisticsFunction, groupByFeature):
        pass
