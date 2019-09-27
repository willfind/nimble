"""
Anchors the hierarchy of data representation types, providing stubs and
common functions.
"""

# TODO conversions
# TODO who sorts inputs to derived implementations?

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import sys
import math
import numbers
import itertools
import os.path
from multiprocessing import Process
from abc import abstractmethod

import numpy
import six
from six.moves import map
from six.moves import range
from six.moves import zip

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.logger import handleLogging
from nimble.logger import produceFeaturewiseReport
from nimble.logger import produceAggregateReport
from nimble.randomness import numpyRandom
from .points import Points
from .features import Features
from .axis import Axis
from .elements import Elements
from . import dataHelpers
# the prefix for default point and feature names
from .dataHelpers import DEFAULT_PREFIX, DEFAULT_PREFIX_LENGTH
from .dataHelpers import DEFAULT_NAME_PREFIX
from .dataHelpers import formatIfNeeded
from .dataHelpers import valuesToPythonList
from .dataHelpers import createListOfDict, createDictOfList
from .dataHelpers import createDataNoValidation

cloudpickle = nimble.importModule('cloudpickle')

mplError = None
try:
    import matplotlib
    import __main__ as main
    # for .show() to work in interactive sessions
    # a backend different than Agg needs to be use
    # The interactive session can choose by default e.g.,
    # in jupyter-notebook inline is the default.
    if hasattr(main, '__file__'):
        # It must be agg  for non-interactive sessions
        # otherwise the combination of matplotlib and multiprocessing
        # produces a segfault.
        # Open matplotlib issue here:
        # https://github.com/matplotlib/matplotlib/issues/8795
        # It applies for both for python 2 and 3
        matplotlib.use('Agg')
except ImportError as e:
    mplError = e

#print('matplotlib backend: {}'.format(matplotlib.get_backend()))

def to2args(f):
    """
    This function is for __pow__. In cython, __pow__ must have 3
    arguments and default can't be used there so this function is used
    to convert a function with 3 arguments to a function with 2
    arguments when it is used in python environment.
    """
    def tmpF(x, y):
        return f(x, y, None)
    return tmpF

def hashCodeFunc(elementValue, pointNum, featureNum):
    """
    Generate hash code.
    """
    return ((math.sin(pointNum) + math.cos(featureNum)) / 2.0) * elementValue

class Base(object):
    """
    Class defining important data manipulation operations and giving
    functionality for the naming the points and features of that data. A
    mapping from names to indices is given by the [point/feature]Names
    attribute, the inverse of that mapping is given by
    [point/feature]NamesInverse.

    Specifically, this includes point and feature names, an object
    name, and originating pathes for the data in this object. Note:
    this method (as should all other __init__ methods in this
    hierarchy) makes use of super().

    Parameters
    ----------
    pointNames : iterable, dict
        A list of point names in the order they appear in the data
        or a dictionary mapping names to indices. None is given if
        default names are desired.
    featureNames : iterable, dict
        A list of feature names in the order they appear in the data
        or a dictionary mapping names to indices. None is given if
        default names are desired.
    name : str
        The name to be associated with this object.
    paths : tuple
        The first entry is taken to be the string representing the
        absolute path to the source file of the data and the second
        entry is taken to be the relative path. Both may be None if
        these values are to be unspecified.
    kwds
        Potentially full of arguments further up the class
        hierarchy, as following best practices for use of super().
        Note, however, that this class is the root of the object
        hierarchy as statically defined.

    Attributes
    ----------
    shape : tuple
        The number of points and features in the object in the format
        (points, features).
    points : Axis object
        An object handling functions manipulating data by points.
    features : Axis object
        An object handling functions manipulating data by features.
    elements : Elements object
        An object handling functions manipulating data by each element.
    name : str
        A name to call this object when printing or logging.
    absolutePath : str
        The absolute path to the data file.
    relativePath : str
        The relative path to the data file.
    path : str
        The path to the data file.
    """

    def __init__(self, shape, pointNames=None, featureNames=None, name=None,
                 paths=(None, None), **kwds):
        self._pointCount = shape[0]
        self._featureCount = shape[1]
        if pointNames is not None and len(pointNames) != shape[0]:
            msg = "The length of the pointNames (" + str(len(pointNames))
            msg += ") must match the points given in shape (" + str(shape[0])
            msg += ")"
            raise InvalidArgumentValue(msg)
        if featureNames is not None and len(featureNames) != shape[1]:
            msg = "The length of the featureNames (" + str(len(featureNames))
            msg += ") must match the features given in shape ("
            msg += str(shape[1]) + ")"
            raise InvalidArgumentValue(msg)

        self._points = self._getPoints()
        self._features = self._getFeatures()
        self._elements = self._getElements()

        # Set up point names
        self._nextDefaultValuePoint = 0
        if pointNames is None:
            self.pointNamesInverse = None
            self.pointNames = None
        elif isinstance(pointNames, dict):
            self._nextDefaultValuePoint = self._pointCount
            self.points.setNames(pointNames, useLog=False)
        else:
            pointNames = valuesToPythonList(pointNames, 'pointNames')
            self._nextDefaultValuePoint = self._pointCount
            self.points.setNames(pointNames, useLog=False)

        # Set up feature names
        self._nextDefaultValueFeature = 0
        if featureNames is None:
            self.featureNamesInverse = None
            self.featureNames = None
        elif isinstance(featureNames, dict):
            self._nextDefaultValueFeature = self._featureCount
            self.features.setNames(featureNames, useLog=False)
        else:
            featureNames = valuesToPythonList(featureNames, 'featureNames')
            self._nextDefaultValueFeature = self._featureCount
            self.features.setNames(featureNames, useLog=False)

        # Set up object name
        if name is None:
            self._name = dataHelpers.nextDefaultObjectName()
        else:
            self._name = name

        # Set up paths
        if paths[0] is not None and not isinstance(paths[0], six.string_types):
            msg = "paths[0] must be None, an absolute path or web link to "
            msg += "the file from which the data originates"
            raise InvalidArgumentType(msg)
        if (paths[0] is not None
                and not os.path.isabs(paths[0])
                and not paths[0].startswith('http')):
            raise InvalidArgumentValue("paths[0] must be an absolute path")
        self._absPath = paths[0]

        if paths[1] is not None and not isinstance(paths[1], six.string_types):
            msg = "paths[1] must be None or a relative path to the file from "
            msg += "which the data originates"
            raise InvalidArgumentType(msg)
        self._relPath = paths[1]

        # call for safety
        super(Base, self).__init__(**kwds)

    #######################
    # Property Attributes #
    #######################

    @property
    def shape(self):
        """
        The number of points and features in the object in the format
        (points, features).
        """
        return self._pointCount, self._featureCount

    def _getPoints(self):
        """
        Get the object containing point-based methods for this object.
        """
        return BasePoints(source=self)

    @property
    def points(self):
        """
        An object handling functions manipulating data by points.
        """
        return self._points

    def _getFeatures(self):
        """
        Get the object containing feature-based methods for this object.
        """
        return BaseFeatures(source=self)

    @property
    def features(self):
        """
        An object handling functions manipulating data by features.
        """
        return self._features

    def _getElements(self):
        """
        Get the object containing element-based methods for this object.
        """
        return BaseElements(source=self)

    @property
    def elements(self):
        """
        An object handling functions manipulating data by each element.
        """
        return self._elements

    def _setpointCount(self, value):
        self._pointCount = value

    def _setfeatureCount(self, value):
        self._featureCount = value

    def _getObjName(self):
        return self._name

    def _setObjName(self, value):
        if value is None:
            self._name = dataHelpers.nextDefaultObjectName()
        else:
            if not isinstance(value, six.string_types):
                msg = "The name of an object may only be a string or None"
                raise ValueError(msg)
            self._name = value

    @property
    def name(self):
        """
        A name to be displayed when printing or logging this object
        """
        return self._getObjName()

    @name.setter
    def name(self, value):
        self._setObjName(value)

    def _getAbsPath(self):
        return self._absPath

    @property
    def absolutePath(self):
        """
        The path to the file this data originated from in absolute form.
        """
        return self._getAbsPath()

    def _getRelPath(self):
        return self._relPath

    @property
    def relativePath(self):
        """
        The path to the file this data originated from in relative form.
        """
        return self._getRelPath()

    def _getPath(self):
        return self.absolutePath

    @property
    def path(self):
        """
        The path to the file this data originated from.
        """
        return self._getPath()

    def _pointNamesCreated(self):
        """
        Returns True if point names have been created/assigned
        to the object.
        If the object does not have points it returns False.
        """
        if self.pointNamesInverse is None:
            return False
        else:
            return True

    def _featureNamesCreated(self):
        """
        Returns True if feature names have been created/assigned
        to the object.
        If the object does not have features it returns False.
        """
        if self.featureNamesInverse is None:
            return False
        else:
            return True

    def _anyDefaultPointNames(self):
        """
        Returns True if any default point names exists or if pointNames
        have not been created.
        """
        if self._pointNamesCreated():
            return any([name.startswith(DEFAULT_PREFIX) for name
                        in self.points.getNames()])
        else:
            return True

    def _anyDefaultFeatureNames(self):
        """
        Returns True if any default feature names exists or if
        featureNames have not been created.
        """
        if self._featureNamesCreated():
            return any([name.startswith(DEFAULT_PREFIX) for name
                        in self.features.getNames()])
        else:
            return True

    def _allDefaultPointNames(self):
        """
        Returns True if all point names are default or have not been
        created.
        """
        if self._pointNamesCreated():
            return all([name.startswith(DEFAULT_PREFIX) for name
                        in self.points.getNames()])
        else:
            return True

    def _allDefaultFeatureNames(self):
        """
        Returns True if all feature names are default or have not been
        created.
        """
        if self._featureNamesCreated():
            return all([name.startswith(DEFAULT_PREFIX) for name
                        in self.features.getNames()])
        else:
            return True

    ########################
    # Low Level Operations #
    ########################

    def __len__(self):
        # ordered such that the larger axis is always printed, even
        # if they are both in the range [0,1]
        if self._pointCount == 0 or self._featureCount == 0:
            return 0
        if self._pointCount == 1:
            return self._featureCount
        if self._featureCount == 1:
            return self._pointCount

        msg = "len() is undefined when the number of points ("
        msg += str(self._pointCount)
        msg += ") and the number of features ("
        msg += str(self._featureCount)
        msg += ") are both greater than 1"
        raise TypeError(msg)

    def nameIsDefault(self):
        """
        Returns True if self.name has a default value
        """
        return self.name.startswith(DEFAULT_NAME_PREFIX)

    ###########################
    # Higher Order Operations #
    ###########################

    def replaceFeatureWithBinaryFeatures(self, featureToReplace, useLog=None):
        """
        Create binary features for each unique value in a feature.

        Modify this object so that the chosen feature is removed, and
        binary valued features are added, one for each unique value
        seen in the original feature.

        Parameters
        ----------
        featureToReplace : int or str
            The index or name of the feature being replaced.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Returns
        -------
        list
            The new feature names after replacement.

        Examples
        --------
        >>> raw = [['a'], ['b'], ['c']]
        >>> data = nimble.createData('Matrix', raw,
        ...                          featureNames=['replace'])
        >>> replaced = data.replaceFeatureWithBinaryFeatures('replace')
        >>> replaced
        ['replace=a', 'replace=b', 'replace=c']
        >>> data
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 1.000 0.000]
             [0.000 0.000 1.000]]
            featureNames={'replace=a':0, 'replace=b':1, 'replace=c':2}
            )
        """
        if self._pointCount == 0:
            msg = "This action is impossible, the object has 0 points"
            raise ImproperObjectAction(msg)

        index = self.features.getIndex(featureToReplace)

        replace = self.features.extract([index], useLog=False)

        uniqueVals = list(replace.elements.countUnique().keys())

        binaryObj = replace._replaceFeatureWithBinaryFeatures_implementation(
            uniqueVals)

        binaryObj.points.setNames(self.points._getNamesNoGeneration(),
                                  useLog=False)
        ftNames = []
        prefix = replace.features.getName(0) + "="
        for val in uniqueVals:
            ftNames.append(prefix + str(val))
        binaryObj.features.setNames(ftNames, useLog=False)

        # by default, put back in same place
        insertBefore = index
        # if extracted last feature, None will append
        if insertBefore == len(self.features):
            insertBefore = None
        self.features.add(binaryObj, insertBefore=insertBefore, useLog=False)

        handleLogging(useLog, 'prep', "replaceFeatureWithBinaryFeatures",
                      self.getTypeString(),
                      Base.replaceFeatureWithBinaryFeatures, featureToReplace)

        return ftNames


    def transformFeatureToIntegers(self, featureToConvert, useLog=None):
        """
        Represent each unique value in a feature with a unique integer.

        Modify this object so that the chosen feature is removed and a
        new integer valued feature is added with values 0 to n-1, one
        for each of n unique values present in the original feature.

        Parameters
        ----------
        featureToConvert : int or str
            The index or name of the feature being replaced.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        >>> raw = [[1, 'a', 1], [2, 'b', 2], [3, 'c', 3]]
        >>> featureNames = ['keep1', 'transform', 'keep2']
        >>> data = nimble.createData('Matrix', raw,
        ...                          featureNames=featureNames)
        >>> mapping = data.transformFeatureToIntegers('transform')
        >>> mapping
        {0: 'a', 1: 'b', 2: 'c'}
        >>> data
        Matrix(
            [[1 0.000 1]
             [2 1.000 2]
             [3 2.000 3]]
            featureNames={'keep1':0, 'transform':1, 'keep2':2}
            )
        """
        if self._pointCount == 0:
            msg = "This action is impossible, the object has 0 points"
            raise ImproperObjectAction(msg)

        ftIndex = self.features.getIndex(featureToConvert)

        mapping = {}
        def applyMap(ft):
            uniqueVals = ft.elements.countUnique()
            integerValue = 0
            if 0 in uniqueVals:
                mapping[0] = 0
                integerValue = 1

            mapped = []
            for val in ft:
                if val in mapping:
                    mapped.append(mapping[val])
                else:
                    mapped.append(integerValue)
                    mapping[val] = integerValue
                    integerValue += 1

            return mapped

        self.features.transform(applyMap, features=ftIndex, useLog=False)

        handleLogging(useLog, 'prep', "transformFeatureToIntegers",
                      self.getTypeString(), Base.transformFeatureToIntegers,
                      featureToConvert)

        return {v: k for k, v in mapping.items()}


    def groupByFeature(self, by, countUniqueValueOnly=False, useLog=None):
        """
        Group data object by one or more features.

        Parameters
        ----------
        by : int, str or list
            * int - the index of the feature to group by
            * str - the name of the feature to group by
            * list - indices or names of features to group by
        countUniqueValueOnly : bool
            Return only the count of points in the group
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Returns
        -------
        dict
            Each unique feature (or group of features) to group by as
            keys. When ``countUniqueValueOnly`` is False,  the value at
            each key is a nimble object containing the ungrouped
            features of points within that group. When
            ``countUniqueValueOnly`` is True, the values are the number
            of points within that group.

        Examples
        --------
        >>> raw = [['ACC', 'Clemson', 15, 0],
        ...        ['SEC', 'Alabama', 14, 1],
        ...        ['Big 10', 'Ohio State', 13, 1],
        ...        ['Big 12', 'Oklahoma', 12, 2],
        ...        ['Independent', 'Notre Dame', 12, 1],
        ...        ['SEC', 'LSU', 10, 3],
        ...        ['SEC', 'Florida', 10, 3],
        ...        ['SEC', 'Georgia', 11, 3]]
        >>> ftNames = ['conference', 'team', 'wins', 'losses']
        >>> top10 = nimble.createData('DataFrame', raw,
        ...                           featureNames=ftNames)
        >>> groupByLosses = top10.groupByFeature('losses')
        >>> list(groupByLosses.keys())
        [0, 1, 2, 3]
        >>> groupByLosses[1]
        DataFrame(
            [[    SEC      Alabama   14]
             [   Big 10   Ohio State 13]
             [Independent Notre Dame 12]]
            featureNames={'conference':0, 'team':1, 'wins':2}
            )
        >>> groupByLosses[3]
        DataFrame(
            [[SEC   LSU   10]
             [SEC Florida 10]
             [SEC Georgia 11]]
            featureNames={'conference':0, 'team':1, 'wins':2}
            )
        """
        def findKey1(point, by):#if by is a string or int
            return point[by]

        def findKey2(point, by):#if by is a list of string or a list of int
            return tuple([point[i] for i in by])

        #if by is a list, then use findKey2; o.w. use findKey1
        if isinstance(by, (six.string_types, numbers.Number)):
            findKey = findKey1
        else:
            findKey = findKey2

        res = {}
        if countUniqueValueOnly:
            for point in self.points:
                k = findKey(point, by)
                if k not in res:
                    res[k] = 1
                else:
                    res[k] += 1
        else:
            for point in self.points:
                k = findKey(point, by)
                if k not in res:
                    res[k] = point.points.getNames()
                else:
                    res[k].extend(point.points.getNames())

            for k in res:
                tmp = self.points.copy(toCopy=res[k], useLog=False)
                tmp.features.delete(by, useLog=False)
                res[k] = tmp

        handleLogging(useLog, 'prep', "groupByFeature",
                      self.getTypeString(), Base.groupByFeature, by,
                      countUniqueValueOnly)

        return res

    def hashCode(self):
        """
        Returns a hash for this matrix.

        The hash is a number x in the range 0<= x < 1 billion that
        should almost always change when the values of the matrix are
        changed by a substantive amount.

        Returns
        -------
        int
        """
        if self._pointCount == 0 or self._featureCount == 0:
            return 0
        valueObj = self.elements.calculate(hashCodeFunc, preserveZeros=True,
                                           outputType='Matrix', useLog=False)
        valueList = valueObj.copy(to="python list")
        avg = (sum(itertools.chain.from_iterable(valueList))
               / float(self._pointCount * self._featureCount))
        bigNum = 1000000000
        #this should return an integer x in the range 0<= x < 1 billion
        return int(int(round(bigNum * avg)) % bigNum)

    def isApproximatelyEqual(self, other):
        """
        Determine if the data in both objects is likely the same.

        If it returns False, this object and the ``other`` object
        definitely do not store equivalent data. If it returns True,
        they likely store equivalent data but it is not possible to be
        absolutely sure. Note that only the actual data stored is
        considered, it doesn't matter whether the data matrix objects
        passed are of the same type (Matrix, Sparse, etc.)

        Parameters
        ----------
        other : nimble Base object
            The object with which to compare approximate equality with
            this object.

        Returns
        -------
        bool
            True if approximately equal, else False.
        """
        #first check to make sure they have the same number of rows and columns
        if self._pointCount != len(other.points):
            return False
        if self._featureCount != len(other.features):
            return False
        #now check if the hashes of each matrix are the same
        if self.hashCode() != other.hashCode():
            return False
        return True


    def trainAndTestSets(self, testFraction, labels=None, randomOrder=True,
                         useLog=None):
        """
        Divide the data into training and testing sets.

        Return either a length 2 or a length 4 tuple. If labels=None,
        then returns a length 2 tuple containing the training object,
        then the testing object (trainX, testX). If labels is non-None,
        a length 4 tuple is returned, containing the training data
        object, then the training labels object, then the testing data
        object, and finally the testing labels
        (trainX, trainY, testX, testY).

        Parameters
        ----------
        testFraction : int or float
            The fraction of the data to be placed in the testing sets.
            If ``randomOrder`` is False, then the points are taken from
            the end of this object.
        labels : identifier or list of identifiers
            The name(s) or index(es) of the data labels, a value of None
            implies this data does not contain labels. This parameter
            will affect the shape of the returned tuple.
        randomOrder : bool
            Control whether the order of the points in the returns sets
            matches that of the original object, or if their order is
            randomized.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Returns
        -------
        tuple
            If ``labels`` is None, a length 2 tuple containing the
            training and testing objects (trainX, testX).
            If ``labels`` is non-None, a length 4 tupes containing the
            training and testing data objects and the training a testing
            labels objects (trainX, trainY, testX, testY).

        Examples
        --------
        Returning a 2-tuple.

        >>> nimble.randomness.setRandomSeed(42)
        >>> raw = [[1, 0, 0],
        ...        [0, 1, 0],
        ...        [0, 0, 1],
        ...        [1, 0, 0],
        ...        [0, 1, 0],
        ...        [0, 0, 1]]
        >>> ptNames = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> data = nimble.createData('Matrix', raw, pointNames=ptNames)
        >>> trainData, testData = data.trainAndTestSets(.34)
        >>> trainData
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 1.000 0.000]
             [0.000 0.000 1.000]
             [0.000 0.000 1.000]]
            pointNames={'a':0, 'b':1, 'f':2, 'c':3}
            )
        >>> testData
        Matrix(
            [[0.000 1.000 0.000]
             [1.000 0.000 0.000]]
            pointNames={'e':0, 'd':1}
            )

        Returning a 4-tuple.

        >>> nimble.randomness.setRandomSeed(42)
        >>> raw = [[1, 0, 0, 1],
        ...        [0, 1, 0, 2],
        ...        [0, 0, 1, 3],
        ...        [1, 0, 0, 1],
        ...        [0, 1, 0, 2],
        ...        [0, 0, 1, 3]]
        >>> ptNames = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> data = nimble.createData('Matrix', raw, pointNames=ptNames)
        >>> fourTuple = data.trainAndTestSets(.34, labels=3)
        >>> trainX, trainY = fourTuple[0], fourTuple[1]
        >>> testX, testY = fourTuple[2], fourTuple[3]
        >>> trainX
        Matrix(
            [[1.000 0.000 0.000]
             [0.000 1.000 0.000]
             [0.000 0.000 1.000]
             [0.000 0.000 1.000]]
            pointNames={'a':0, 'b':1, 'f':2, 'c':3}
            )
        >>> trainY
        Matrix(
            [[1.000]
             [2.000]
             [3.000]
             [3.000]]
            pointNames={'a':0, 'b':1, 'f':2, 'c':3}
            )
        >>> testX
        Matrix(
            [[0.000 1.000 0.000]
             [1.000 0.000 0.000]]
            pointNames={'e':0, 'd':1}
            )
        >>> testY
        Matrix(
            [[2.000]
             [1.000]]
            pointNames={'e':0, 'd':1}
            )
        """
        order = list(range(len(self.points)))
        if randomOrder:
            numpyRandom.shuffle(order)

        testXSize = int(round(testFraction * self._pointCount))
        splitIndex = self._pointCount - testXSize

        #pull out a testing set
        trainX = self.points.copy(order[:splitIndex], useLog=False)
        testX = self.points.copy(order[splitIndex:], useLog=False)

        trainX.name = self.name + " trainX"
        testX.name = self.name + " testX"

        if labels is None:
            ret = trainX, testX
        else:
            # safety for empty objects
            toExtract = labels
            if testXSize == 0:
                toExtract = []

            trainY = trainX.features.extract(toExtract, useLog=False)
            testY = testX.features.extract(toExtract, useLog=False)

            trainY.name = self.name + " trainY"
            testY.name = self.name + " testY"

            ret = trainX, trainY, testX, testY

        handleLogging(useLog, 'prep', "trainAndTestSets", self.getTypeString(),
                      Base.trainAndTestSets, testFraction, labels, randomOrder)

        return ret

    ########################################
    ########################################
    ###   Functions related to logging   ###
    ########################################
    ########################################

    def featureReport(self, maxFeaturesToCover=50, displayDigits=2,
                      useLog=None):
        """
        Report containing a summary and statistics for each feature.

        Produce a report, in a string formatted as a table, containing
        summary and statistical information about each feature in the
        data set, up to 50 features.  If there are more than 50
        features, only information about 50 of those features will be
        reported.

        Parameters
        ----------
        maxFeaturesToCover : int
            The maximum number of features to include in the report.
            Default is 50, which is the maximum allowed for this value.
        displayDigits : int
            The number of digits to display after a decimal point.
            Default is 2.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        """
        ret = produceFeaturewiseReport(
            self, maxFeaturesToCover=maxFeaturesToCover,
            displayDigits=displayDigits)
        handleLogging(useLog, 'data', "feature", ret)
        return ret


    def summaryReport(self, displayDigits=2, useLog=None):
        """
        Report containing information regarding the data in this object.

        Produce a report, in a string formatted as a table, containing
        summary information about the data set contained in this object.
        Includes proportion of missing values, proportion of zero
        values, total # of points, and number of features.

        displayDigits : int
            The number of digits to display after a decimal point.
            Default is 2.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.
        """
        ret = produceAggregateReport(self, displayDigits=displayDigits)
        handleLogging(useLog, 'data', "summary", ret)
        return ret

    ###############################################################
    ###############################################################
    ###   Subclass implemented information querying functions   ###
    ###############################################################
    ###############################################################

    def isIdentical(self, other):
        """
        Check for equality between two objects.

        Return True if all values and names in the other object match
        the values and names in this object.
        """
        if not self._equalFeatureNames(other):
            return False
        if not self._equalPointNames(other):
            return False

        return self._isIdentical_implementation(other)

    def writeFile(self, outPath, fileFormat=None, includeNames=True):
        """
        Write the data in this object to a file in the specified format.

        Parameters
        ----------
        outPath : str
            The location (including file name and extension) where
            we want to write the output file.
        fileFormat : str
            The formating of the file we write. May be None, 'csv', or
            'mtx'; if None, we use the extension of outPath to determine
            the format.
        includeNames : bool
            Indicates whether the file will embed the point and feature
            names into the file. The format of the embedding is
            dependant on the format of the file: csv will embed names
            into the data, mtx will place names in a comment.
        """
        if self._pointCount == 0 or self._featureCount == 0:
            msg = "We do not allow writing to file when an object has "
            msg += "0 points or features"
            raise ImproperObjectAction(msg)

        # if format is not specified, we fall back on the extension in outPath
        if fileFormat is None:
            split = outPath.rsplit('.', 1)
            fileFormat = None
            if len(split) > 1:
                fileFormat = split[1].lower()

        if fileFormat not in ['csv', 'mtx']:
            msg = "Unrecognized file format. Accepted types are 'csv' and "
            msg += "'mtx'. They may either be input as the format parameter, "
            msg += "or as the extension in the outPath"
            raise InvalidArgumentValue(msg)

        includePointNames = includeNames
        if includePointNames:
            seen = False
            for name in self.points.getNames():
                if name[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                    seen = True
            if not seen:
                includePointNames = False

        includeFeatureNames = includeNames
        if includeFeatureNames:
            seen = False
            for name in self.features.getNames():
                if name[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                    seen = True
            if not seen:
                includeFeatureNames = False

        try:
            self._writeFile_implementation(
                outPath, fileFormat, includePointNames, includeFeatureNames)
        except Exception:
            if fileFormat.lower() == "csv":
                toOut = self.copy(to="Matrix")
                toOut._writeFile_implementation(
                    outPath, fileFormat, includePointNames, includeFeatureNames)
                return
            if fileFormat.lower() == "mtx":
                toOut = self.copy(to='Sparse')
                toOut._writeFile_implementation(
                    outPath, fileFormat, includePointNames, includeFeatureNames)
                return

    def save(self, outputPath):
        """
        Save object to a file.

        Uses the cloudpickle library to serialize this object.

        Parameters
        ----------
        outputPath : str
            The location (including file name and extension) where
            we want to write the output file. If filename extension
            .nimd is not included in file name it would be added to the
            output file.
        """
        if not cloudpickle:
            msg = "To save nimble objects, cloudpickle must be installed"
            raise PackageException(msg)

        extension = '.nimd'
        if not outputPath.endswith(extension):
            outputPath = outputPath + extension

        with open(outputPath, 'wb') as file:
            try:
                cloudpickle.dump(self, file)
            except Exception as e:
                raise e
        # TODO: save session
        # print('session_' + outputFilename)
        # print(globals())
        # dill.dump_session('session_' + outputFilename)

    def getTypeString(self):
        """
        The nimble Type of this object.

        A string representing the non-abstract type of this
        object (e.g. Matrix, Sparse, etc.) that can be passed to
        createData() function to create a new object of the same type.

        Returns
        -------
        str
        """
        return self._getTypeString_implementation()

    def __getitem__(self, key):
        """
        Return a copy of a subset of the data.

        Vector-shaped objects can use a single value as the key,
        otherwise the key must be a tuple (pointsToGet, featuresToGet).
        All values in the key are INCLUSIVE. When using a slice, the
        ``stop`` value will be included, unlike the python convention.

        Parameters
        ----------
        key : name, index, list, slice
            * name - the name of the point or feature to get.
            * index - the index of the point or feature to get.
            * list - a list of names or indices to get. Note: the points
              and/or features will be returned in the order of the list.
            * slice - a slice of names or indices to get. Note: nimble
              uses inclusive values.

        Examples
        --------
        >>> raw = [[4132, 41, 'management', 50000, 'm'],
        ...        [4434, 26, 'sales', 26000, 'm'],
        ...        [4331, 26, 'administration', 28000, 'f'],
        ...        [4211, 45, 'sales', 33000, 'm'],
        ...        [4344, 45, 'accounting', 43500, 'f']]
        >>> pointNames = ['michael', 'jim', 'pam', 'dwight', 'angela']
        >>> featureNames = ['id', 'age', 'department', 'salary',
        ...                 'gender']
        >>> office = nimble.createData('Matrix', raw,
        ...                            pointNames=pointNames,
        ...                            featureNames=featureNames)

        Get a single value.

        >>> office['michael', 'age']
        41
        >>> office[0,1]
        41
        >>> office['michael', 1]
        41

        Get based on points only.

        >>> pam = office['pam', :]
        >>> print(pam)
               id  age   department   salary gender
        <BLANKLINE>
        pam   4331  26 administration 28000    f
        <BLANKLINE>
        >>> sales = office[[3, 1], :]
        >>> print(sales)
                  id  age department salary gender
        <BLANKLINE>
        dwight   4211  45   sales    33000    m
           jim   4434  26   sales    26000    m
        <BLANKLINE>

        *Note: retains list order; index 3 placed before index 1*

        >>> nonManagement = office[1:4, :]
        >>> print(nonManagement)
                  id  age   department   salary gender
        <BLANKLINE>
           jim   4434  26     sales      26000    m
           pam   4331  26 administration 28000    f
        dwight   4211  45     sales      33000    m
        angela   4344  45   accounting   43500    f
        <BLANKLINE>

        *Note: slices are inclusive; index 4 ('gender') was included*

        Get based on features only.

        >>> departments = office[:, 2]
        >>> print(departments)
                    department
        <BLANKLINE>
        michael     management
            jim       sales
            pam   administration
         dwight       sales
         angela     accounting
        <BLANKLINE>

        >>> genderAndAge = office[:, ['gender', 'age']]
        >>> print(genderAndAge)
                  gender age
        <BLANKLINE>
        michael     m     41
            jim     m     26
            pam     f     26
         dwight     m     45
         angela     f     45
        <BLANKLINE>

        *Note: retains list order; 'gender' placed before 'age'*

        >>> deptSalary = office[:, 'department':'salary']
        >>> print(deptSalary)
                    department   salary
        <BLANKLINE>
        michael     management   50000
            jim       sales      26000
            pam   administration 28000
         dwight       sales      33000
         angela     accounting   43500
        <BLANKLINE>

        *Note: slices are inclusive; 'salary' was included*

        Get based on points and features.

        >>> femaleSalaryAndDept = office[['pam', 'angela'], [3,2]]
        >>> print(femaleSalaryAndDept)
                 salary   department
        <BLANKLINE>
           pam   28000  administration
        angela   43500    accounting
        <BLANKLINE>

        *Note: list orders retained; 'pam' precedes 'angela' and index 3
        ('salary') precedes index 2 ('department')*

        >>> first3Ages = office[:2, 'age']
        >>> print(first3Ages)
                  age
        <BLANKLINE>
        michael    41
            jim    26
            pam    26
        <BLANKLINE>

        *Note: slices are inclusive; index 2 ('pam') was included*
        """
        # Make it a tuple if it isn't one
        if key.__class__ is tuple:
            x, y = key
        else:
            if self._pointCount == 1:
                x = 0
                y = key
            elif self._featureCount == 1:
                x = key
                y = 0
            else:
                msg = "Must include both a point and feature index; or, "
                msg += "if this is vector shaped, a single index "
                msg += "into the axis whose length > 1"
                raise InvalidArgumentValue(msg)

        #process x
        singleX = False
        if isinstance(x, (int, float, str, numpy.integer)):
            x = self.points._processSingle(x)
            singleX = True
        #process y
        singleY = False
        if isinstance(y, (int, float, str, numpy.integer)):
            y = self.features._processSingle(y)
            singleY = True
        #if it is the simplest data retrieval such as X[1,2],
        # we'd like to return it back in the fastest way.
        if singleX and singleY:
            return self._getitem_implementation(x, y)
        # if not convert x and y to lists of indices
        if singleX:
            x = [x]
        else:
            x = self.points._processMultiple(x)

        if singleY:
            y = [y]
        else:
            y = self.features._processMultiple(y)

        # None is returned by _processMultiple if the axis is a full slice
        # since copying of an axis which is a full slice is unnecessary.
        if x is None and y is None:
            ret = self.copy()
        # use backend directly since values have already been validated
        elif y is None:
            ret = self.points._structuralBackend_implementation('copy', x)
        elif x is None:
            ret = self.features._structuralBackend_implementation('copy', y)
        else:
            ret = self.points._structuralBackend_implementation('copy', x)
            ret = ret.features._structuralBackend_implementation('copy', y)
        return ret

    def pointView(self, ID):
        """
        A read-only view of a single point.

        A BaseView object into the data of the point with the given ID.
        See BaseView object comments for its capabilities. This view is
        only valid until the next modification to the shape or ordering
        of this object's internal data. After such a modification, there
        is no guarantee to the validity of the results.

        Returns
        -------
        BaseView
            The read-only object for this point.
        """
        if self._pointCount == 0:
            msg = "ID is invalid, This object contains no points"
            raise ImproperObjectAction(msg)

        index = self.points.getIndex(ID)
        return self.view(index, index, None, None)

    def featureView(self, ID):
        """
        A read-only view of a single feature.

        A BaseView object into the data of the feature with the given
        ID. See BaseView object comments for its capabilities. This view
        is only valid until the next modification to the shape or
        ordering of this object's internal data. After such a
        modification, there is no guarantee to the validity of the
        results.

        Returns
        -------
        BaseView
            The read-only object for this feature.
        """
        if self._featureCount == 0:
            msg = "ID is invalid, This object contains no features"
            raise ImproperObjectAction(msg)

        index = self.features.getIndex(ID)
        return self.view(None, None, index, index)

    def view(self, pointStart=None, pointEnd=None, featureStart=None,
             featureEnd=None):
        """
        Read-only access into the object data.

        Factory function to create a read only view into the calling
        data object. Views may only be constructed from contiguous,
        in-order points and features whose overlap defines a window into
        the data. The returned View object is part of nimble's datatypes
        hiearchy, and will have access to all of the same methods as
        anything that inherits from Base; though only those
        that do not modify the data can be called without an exception
        being raised. The returned view will also reflect any subsequent
        changes made to the original object. This is the only accepted
        method for a user to construct a View object (it should never be
        done directly), though view objects may be provided to the user,
        for example via user defined functions passed to points.extract
        or features.calculate.

        Parameters
        ----------
        pointStart : int
            The inclusive index of the first point to be accessible in
            the returned view. Is None by default, meaning to include
            from the beginning of the object.
        pointEnd: int
            The inclusive index of the last point to be accessible in
            the returned view. Is None by default, meaning to include up
            to the end of the object.
        featureStart : int
            The inclusive index of the first feature to be accessible in
            the returned view. Is None by default, meaning to include
            from the beginning of the object.
        featureEnd : int
            The inclusive index of the last feature to be accessible in
            the returned view. Is None by default, meaning to include up
            to the end of the object.

        Returns
        -------
        nimble view object
            A nimble Base object with read-only access.

        See Also
        --------
        pointView, featureView
        """
        # transform defaults to mean take as much data as possible,
        # transform end values to be EXCLUSIVE
        if pointStart is None:
            pointStart = 0
        else:
            pointStart = self.points.getIndex(pointStart)

        if pointEnd is None:
            pointEnd = self._pointCount
        else:
            pointEnd = self.points.getIndex(pointEnd)
            # this is the only case that could be problematic and needs
            # checking
            self._validateRangeOrder("pointStart", pointStart,
                                     "pointEnd", pointEnd)
            # make exclusive now that it won't ruin the validation check
            pointEnd += 1

        if featureStart is None:
            featureStart = 0
        else:
            featureStart = self.features.getIndex(featureStart)

        if featureEnd is None:
            featureEnd = self._featureCount
        else:
            featureEnd = self.features.getIndex(featureEnd)
            # this is the only case that could be problematic and needs
            # checking
            self._validateRangeOrder("featureStart", featureStart,
                                     "featureEnd", featureEnd)
            # make exclusive now that it won't ruin the validation check
            featureEnd += 1

        return self._view_implementation(pointStart, pointEnd,
                                         featureStart, featureEnd)

    def validate(self, level=1):
        """
        Check the integrity of the data.

        Validate this object with respect to the limitations and
        invariants that our objects enforce.

        Parameters
        ----------
        level : int
            The extent to which to validate the data.
        """
        if self.points._namesCreated():
            assert self._pointCount == len(self.points.getNames())
        if self.features._namesCreated():
            assert self._featureCount == len(self.features.getNames())

        if level > 0:
            if self.points._namesCreated():
                for key in self.points.getNames():
                    index = self.points.getIndex(key)
                    assert self.points.getName(index) == key
            if self.features._namesCreated():
                for key in self.features.getNames():
                    index = self.features.getIndex(key)
                    assert self.features.getName(index) == key

        self._validate_implementation(level)

    def containsZero(self):
        """
        Evaluate if the object contains one or more zero values.

        True if there is a value that is equal to integer 0
        contained in this object, otherwise False.

        Returns
        -------
        bool
        """
        # trivially False.
        if self._pointCount == 0 or self._featureCount == 0:
            return False
        return self._containsZero_implementation()

    def __eq__(self, other):
        return self.isIdentical(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def toString(self, includeNames=True, maxWidth=79, maxHeight=30,
                 sigDigits=3, maxColumnWidth=19, keepTrailingWhitespace=False):
        """
        A string representation of this object.

        For objects exceeding the ``maxWidth`` and/or ``maxHeight``,
        the string will truncate the output using various placeholders
        to indicate that some data was removed. For width and height,
        the data removed will be from the center printing only the data
        for the first few and last few columns and rows, respectively.

        Parameters
        ----------
        includeNames : bool
            Whether of not to include point and feature names in the
            printed output. True, the default, indidcates names will be
            included. False will not include names in the output.
        maxWidth : int
            A bound on the maximum number of characters allowed on each
            line of the output.
        maxHeight : int
            A bound on the maximum number of lines allowed for the
            output.
        sigDigits : int
            The number of significant digits to display in the output.
        maxColumnWidth : int
            A bound on the maximum number of characters allowed for the
            width of single column (feature) in each line.
        """
        if self._pointCount == 0 or self._featureCount == 0:
            return ""

        # setup a bundle of fixed constants
        colSep = ' '
        colHold = '--'
        rowHold = '|'
        pnameSep = ' '
        nameHolder = '...'
        dataOrientation = 'center'
        pNameOrientation = 'rjust'
        fNameOrientation = 'center'

        #setup a bundle of default values
        maxHeight = self._pointCount + 2 if maxHeight is None else maxHeight
        maxWidth = float('inf') if maxWidth is None else maxWidth
        maxRows = min(maxHeight, self._pointCount)
        maxDataRows = maxRows
        includePNames = False
        includeFNames = False

        if includeNames:
            includePNames = dataHelpers.hasNonDefault(self, 'point')
            includeFNames = dataHelpers.hasNonDefault(self, 'feature')
            if includeFNames:
                # plus or minus 2 because we will be dealing with both
                # feature names and a gap row
                maxRows = min(maxHeight, self._pointCount + 2)
                maxDataRows = maxRows - 2

        # Set up point Names and determine how much space they take up
        pnames = None
        pnamesWidth = None
        maxDataWidth = maxWidth
        if includePNames:
            pnames, pnamesWidth = self._arrangePointNames(
                maxDataRows, maxColumnWidth, rowHold, nameHolder)
            # The available space for the data is reduced by the width of the
            # pnames, a column separator, the pnames separator, and another
            # column separator
            maxDataWidth = (maxWidth
                            - (pnamesWidth + 2 * len(colSep) + len(pnameSep)))


        # Set up data values to fit in the available space including
        # featureNames if includeFNames=True
        dataTable, colWidths, fnames = self._arrangeDataWithLimits(
            maxDataWidth, maxDataRows, includeFNames, sigDigits,
            maxColumnWidth, colSep, colHold, rowHold, nameHolder)

        # combine names into finalized table
        finalTable, finalWidths = self._arrangeFinalTable(
            pnames, pnamesWidth, dataTable, colWidths, fnames, pnameSep)

        # set up output string
        out = ""
        for r in range(len(finalTable)):
            row = finalTable[r]
            for c in range(len(row)):
                val = row[c]
                if c == 0 and includePNames:
                    padded = getattr(val, pNameOrientation)(finalWidths[c])
                elif r == 0 and includeFNames:
                    padded = getattr(val, fNameOrientation)(finalWidths[c])
                else:
                    padded = getattr(val, dataOrientation)(finalWidths[c])
                row[c] = padded
            # for __repr__ output want to retain whitespace
            if keepTrailingWhitespace:
                line = colSep.join(finalTable[r]) + "\n"
            else:
                line = colSep.join(finalTable[r]).rstrip() + "\n"
            out += line

        return out

    def __repr__(self):
        indent = '    '
        maxW = 79
        maxH = 30

        # setup type call
        ret = self.getTypeString() + "(\n"

        # setup data
        # decrease max width for indentation (4) and nested list padding (4)
        stringWidth = maxW - 8
        dataStr = self.toString(includeNames=False, maxWidth=stringWidth,
                                maxHeight=maxH, keepTrailingWhitespace=True)
        byLine = dataStr.split('\n')
        # toString ends with a \n, so we get rid of the empty line produced by
        # the split
        byLine = byLine[:-1]
        # convert self.data into a string with nice format
        newLines = (']\n' + indent + ' [').join(byLine)
        ret += (indent + '[[%s]]\n') % newLines

        numRows = min(self._pointCount, maxH)
        # if non default point names, print all (truncated) point names
        ret += dataHelpers.makeNamesLines(
            indent, maxW, numRows, self._pointCount,
            self.points._getNamesNoGeneration(), 'pointNames')
        # if non default feature names, print all (truncated) feature names
        numCols = 0
        if byLine:
            splited = byLine[0].split(' ')
            for val in splited:
                if val not in ['', '...', '--']:
                    numCols += 1
        elif self._featureCount > 0:
            # if the container is empty, then roughly compute length of
            # the string of feature names, and then calculate numCols
            ftNames = self.features._getNamesNoGeneration()
            if ftNames is None:
                # mock up default looking names to avoid name generation
                ftNames = [DEFAULT_PREFIX + '#'] * len(self.features)
            strLength = (len("___".join(ftNames))
                         + len(''.join([str(i) for i
                                        in range(self._featureCount)])))
            numCols = int(min(1, maxW / float(strLength)) * self._featureCount)
        # because of how dataHelpers.indicesSplit works, we need this to be
        # +1 in some cases this means one extra feature name is displayed. But
        # that's acceptable
        if numCols <= self._featureCount:
            numCols += 1
        elif numCols > self._featureCount:
            numCols = self._featureCount
        ret += dataHelpers.makeNamesLines(
            indent, maxW, numCols, self._featureCount,
            self.features._getNamesNoGeneration(), 'featureNames')

        # if name not None, print
        if not self.name.startswith(DEFAULT_NAME_PREFIX):
            prep = indent + 'name="'
            toUse = self.name
            nonNameLen = len(prep) + 1
            if nonNameLen + len(toUse) > 80:
                toUse = toUse[:(80 - nonNameLen - 3)]
                toUse += '...'

            ret += prep + toUse + '"\n'

        # if path not None, print
        if self.path is not None:
            prep = indent + 'path="'
            toUse = self.path
            nonPathLen = len(prep) + 1
            if nonPathLen + len(toUse) > 80:
                toUse = toUse[:(80 - nonPathLen - 3)]
                toUse += '...'

            ret += prep + toUse + '"\n'

        ret += indent + ')'

        return ret

    def __str__(self):
        return self.toString()

    def show(self, description, includeObjectName=True, includeAxisNames=True,
             maxWidth=79, maxHeight=30, sigDigits=3, maxColumnWidth=19):
        """
        A printed representation of the data.

        Method to simplify printing a representation of this data
        object, with some context. The backend is the ``toString()``
        method, and this method includes control over all of the same
        functionality via arguments. Prior to the names and data, it
        additionally prints a description provided by the user,
        (optionally) this object's name attribute, and the number of
        points and features that are in the data.

        Parameters
        ----------
        description : str
            Printed as-is before the rest of the output, unless None.
        includeObjectName : bool
            True will include printing of the object's ``name``
            attribute, False will not print the object's name.
        includeAxisNames : bool
            True will include printing of the object's point and feature
            names, False will not print the point or feature names.
        maxWidth : int
            A bound on the maximum number of characters allowed on each
            line of the output.
        maxHeight : int
            A bound on the maximum number of lines allowed for the
            output.
        sigDigits : int
            The number of significant digits to display in the output.
        maxColumnWidth : int
            A bound on the maximum number of characters allowed for the
            width of single column (feature) in each line.
        """
        if description is not None:
            print(description)

        if includeObjectName:
            context = self.name + " : "
        else:
            context = ""
        context += str(self._pointCount) + "pt x "
        context += str(self._featureCount) + "ft"
        print(context)
        print(self.toString(includeAxisNames, maxWidth, maxHeight, sigDigits,
                            maxColumnWidth))

    def plot(self, outPath=None, includeColorbar=False):
        self._plot(outPath, includeColorbar)

    def _setupOutFormatForPlotting(self, outPath):
        outFormat = None
        if isinstance(outPath, six.string_types):
            (_, ext) = os.path.splitext(outPath)
            if len(ext) == 0:
                outFormat = 'png'
        return outFormat

    def _matplotlibBackendHandling(self, outPath, plotter, **kwargs):
        if outPath is None:
            if matplotlib.get_backend() == 'agg':
                import matplotlib.pyplot as plt
                plt.switch_backend('TkAgg')
                plotter(**kwargs)
                plt.switch_backend('agg')
            else:
                plotter(**kwargs)
            p = Process(target=lambda: None)
            p.start()
        else:
            p = Process(target=plotter, kwargs=kwargs)
            p.start()
        return p

    def _plot(self, outPath=None, includeColorbar=False):
        self._validateMatPlotLibImport(mplError, 'plot')
        outFormat = self._setupOutFormatForPlotting(outPath)

        def plotter(d):
            import matplotlib.pyplot as plt

            plt.matshow(d, cmap=matplotlib.cm.gray)

            if includeColorbar:
                plt.colorbar()

            if not self.name.startswith(DEFAULT_NAME_PREFIX):
                #plt.title("Heatmap of " + self.name)
                plt.title(self.name)
            plt.xlabel("Feature Values", labelpad=10)
            plt.ylabel("Point Values")

            if outPath is None:
                plt.show()
            else:
                plt.savefig(outPath, format=outFormat)

        # toPlot = self.copy(to='numpyarray')

        # problem if we were to use mutiprocessing with backends
        # different than Agg.
        p = self._matplotlibBackendHandling(outPath, plotter, d=self.data)
        return p

    def plotFeatureDistribution(self, feature, outPath=None, xMin=None,
                                xMax=None):
        """
        Plot a histogram of the distribution of values in a feature.

        Along the x axis of the plot will be the values seen in
        the feature, grouped into bins; along the y axis will be the
        number of values in each bin. Bin width is calculated using
        Freedman-Diaconis' rule. Control over the width of the x axis
        is also given, with the warning that user specified values
        can obscure data that would otherwise be plotted given default
        inputs.

        Parameters
        ----------
        feature : identifier
            The index of name of the feature to plot.
        outPath : str, None
            A string of the path to save the plot output. If None, the
            plot will be displayed.
        xMin : int, float
            The least value shown on the x axis of the resultant plot.
        xMax: int, float
            The largest value shown on the x axis of teh resultant plot.

        Returns
        -------
        plot
            Displayed or written to the ``outPath`` file.
        """
        self._plotFeatureDistribution(feature, outPath, xMin, xMax)

    def _plotFeatureDistribution(self, feature, outPath=None, xMin=None,
                                 xMax=None):
        self._validateMatPlotLibImport(mplError, 'plotFeatureDistribution')
        return self._plotDistribution('feature', feature, outPath, xMin, xMax)

    def _plotDistribution(self, axis, identifier, outPath, xMin, xMax):
        outFormat = self._setupOutFormatForPlotting(outPath)
        axisObj = self._getAxis(axis)
        index = axisObj.getIndex(identifier)
        name = None
        if axis == 'point':
            getter = self.pointView
            if self.points._namesCreated():
                name = self.points.getName(index)
        else:
            getter = self.featureView
            if self.features._namesCreated():
                name = self.features.getName(index)

        toPlot = getter(index)

        quartiles = nimble.calculate.quartiles(toPlot)

        IQR = quartiles[2] - quartiles[0]
        binWidth = (2 * IQR) / (len(toPlot) ** (1. / 3))
        # TODO: replace with calculate points after it subsumes
        # pointStatistics?
        valMax = max(toPlot)
        valMin = min(toPlot)
        if binWidth == 0:
            binCount = 1
        else:
            # we must convert to int, in some versions of numpy, the helper
            # functions matplotlib calls will require it.
            binCount = int(math.ceil((valMax - valMin) / binWidth))

        def plotter(d, xLim):
            import matplotlib.pyplot as plt

            plt.hist(d, binCount)

            if not name or name[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
                titlemsg = '#' + str(index)
            else:
                titlemsg = "named: " + name
            plt.title("Distribution of " + axis + " " + titlemsg)
            plt.xlabel("Values")
            plt.ylabel("Number of values")

            plt.xlim(xLim)

            if outPath is None:
                plt.show()
            else:
                plt.savefig(outPath, format=outFormat)

        # problem if we were to use mutiprocessing with backends
        # different than Agg.
        p = self._matplotlibBackendHandling(outPath, plotter, d=toPlot,
                                            xLim=(xMin, xMax))
        return p

    def plotFeatureAgainstFeatureRollingAverage(
            self, x, y, outPath=None, xMin=None, xMax=None, yMin=None,
            yMax=None, sampleSizeForAverage=20):
        """
        A rolling average of the pairwise combination of feature values.

        Control over the width of the both axes is given, with the
        warning that user specified values can obscure data that would
        otherwise be plotted given default inputs.


        Parameters
        ----------
        x : identifier
            The index or name of the feature from which we draw x-axis
            coordinates.
        y : identifier
            The index or name of the feature from which we draw y-axis
            coordinates.
        outPath : str, None
            A string of the path to save the plot output. If None, the
            plot will be displayed.
        xMin : int, float
            The least value shown on the x axis of the resultant plot.
        xMax: int, float
            The largest value shown on the x axis of teh resultant plot.
        yMin : int, float
            The least value shown on the y axis of the resultant plot.
        yMax: int, float
            The largest value shown on the y axis of teh resultant plot.
        sampleSizeForAverage : int
            The number of samples to use for the caclulation of the
            rolling average.

        Returns
        -------
        plot
            Displayed or written to the ``outPath`` file.
        """
        self._plotFeatureAgainstFeature(x, y, outPath, xMin, xMax, yMin, yMax,
                                        sampleSizeForAverage)

    def plotFeatureAgainstFeature(self, x, y, outPath=None, xMin=None,
                                  xMax=None, yMin=None, yMax=None):
        """
        A scatter plot of the pairwise combination of feature values.

        Control over the width of the both axes is given, with the
        warning that user specified values can obscure data that would
        otherwise be plotted given default inputs.

        Parameters
        ----------
        x : identifier
            The index or name of the feature from which we draw x-axis
            coordinates.
        y : identifier
            The index or name of the feature from which we draw y-axis
            coordinates.
        outPath : str, None
            A string of the path to save the plot output. If None, the
            plot will be displayed.
        xMin : int, float
            The least value shown on the x axis of the resultant plot.
        xMax: int, float
            The largest value shown on the x axis of teh resultant plot.
        yMin : int, float
            The least value shown on the y axis of the resultant plot.
        yMax: int, float
            The largest value shown on the y axis of teh resultant plot.

        Returns
        -------
        plot
            Displayed or written to the ``outPath`` file.
        """
        self._plotFeatureAgainstFeature(x, y, outPath, xMin, xMax, yMin, yMax)

    def _plotFeatureAgainstFeature(self, x, y, outPath=None, xMin=None,
                                   xMax=None, yMin=None, yMax=None,
                                   sampleSizeForAverage=None):
        self._validateMatPlotLibImport(mplError, 'plotFeatureComparison')
        return self._plotCross(x, 'feature', y, 'feature', outPath, xMin, xMax,
                               yMin, yMax, sampleSizeForAverage)

    def _plotCross(self, x, xAxis, y, yAxis, outPath, xMin, xMax, yMin, yMax,
                   sampleSizeForAverage=None):
        outFormat = self._setupOutFormatForPlotting(outPath)
        xAxisObj = self._getAxis(xAxis)
        yAxisObj = self._getAxis(yAxis)
        xIndex = xAxisObj.getIndex(x)
        yIndex = yAxisObj.getIndex(y)

        def customGetter(index, axis):
            if axis == 'point':
                copied = self.points.copy(index, useLog=False)
            else:
                copied = self.features.copy(index, useLog=False)
            return copied.copy(to='numpyarray', outputAs1D=True)

        def pGetter(index):
            return customGetter(index, 'point')

        def fGetter(index):
            return customGetter(index, 'feature')

        xName = None
        yName = None
        if xAxis == 'point':
            xGetter = pGetter
            if self.points._namesCreated():
                xName = self.points.getName(xIndex)
        else:
            xGetter = fGetter
            if self.features._namesCreated():
                xName = self.features.getName(xIndex)

        if yAxis == 'point':
            yGetter = pGetter
            if self.points._namesCreated():
                yName = self.points.getName(yIndex)
        else:
            yGetter = fGetter
            if self.features._namesCreated():
                yName = self.features.getName(yIndex)

        xToPlot = xGetter(xIndex)
        yToPlot = yGetter(yIndex)

        if sampleSizeForAverage:
            #do rolling average
            xToPlot, yToPlot = list(zip(*sorted(zip(xToPlot, yToPlot),
                                                key=lambda x: x[0])))
            convShape = (numpy.ones(sampleSizeForAverage)
                         / float(sampleSizeForAverage))
            startIdx = sampleSizeForAverage-1
            xToPlot = numpy.convolve(xToPlot, convShape)[startIdx:-startIdx]
            yToPlot = numpy.convolve(yToPlot, convShape)[startIdx:-startIdx]

        def plotter(inX, inY, xLim, yLim, sampleSizeForAverage):
            import matplotlib.pyplot as plt
            #plt.scatter(inX, inY)
            plt.scatter(inX, inY, marker='.')

            if not xName or xName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
                xlabel = xAxis + ' #' + str(xIndex)
            else:
                xlabel = xName
            if not yName or yName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
                ylabel = yAxis + ' #' + str(yIndex)
            else:
                ylabel = yName

            xName2 = xName
            yName2 = yName
            if sampleSizeForAverage:
                tmpStr = ' (%s sample average)' % sampleSizeForAverage
                xlabel += tmpStr
                ylabel += tmpStr
                xName2 += ' average'
                yName2 += ' average'

            if self.name.startswith(DEFAULT_NAME_PREFIX):
                titleStr = ('%s vs. %s') % (xName2, yName2)
            else:
                titleStr = ('%s: %s vs. %s') % (self.name, xName2, yName2)


            plt.title(titleStr)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.xlim(xLim)
            plt.ylim(yLim)

            if outPath is None:
                plt.show()
            else:
                plt.savefig(outPath, format=outFormat)

        # problem if we were to use mutiprocessing with backends
        # different than Agg.
        p = self._matplotlibBackendHandling(
            outPath, plotter, inX=xToPlot, inY=yToPlot, xLim=(xMin, xMax),
            yLim=(yMin, yMax), sampleSizeForAverage=sampleSizeForAverage)
        return p

    ##################################################################
    ##################################################################
    ###   Subclass implemented structural manipulation functions   ###
    ##################################################################
    ##################################################################

    def transpose(self, useLog=None):
        """
        Invert the feature and point indices of the data.

        Transpose the data in this object, inplace by inverting the
        feature and point indices. This operations also includes
        inverting the point and feature names.

        Parameters
        ----------
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        >>> raw = [[1, 2, 3], [4, 5, 6]]
        >>> data = nimble.createData('List', raw)
        >>> data
        List(
            [[1.000 2.000 3.000]
             [4.000 5.000 6.000]]
            )
        >>> data.transpose()
        >>> data
        List(
            [[1.000 4.000]
             [2.000 5.000]
             [3.000 6.000]]
            )
        """
        self._transpose_implementation()

        self._pointCount, self._featureCount = (self._featureCount,
                                                self._pointCount)
        ptNames, ftNames = (self.features._getNamesNoGeneration(),
                            self.points._getNamesNoGeneration())
        self.points.setNames(ptNames, useLog=False)
        self.features.setNames(ftNames, useLog=False)

        handleLogging(useLog, 'prep', "transpose", self.getTypeString(),
                      Base.transpose)

    @property
    def T(self):
        """
        Invert the feature and point indices of the data.

        Return this object with inverted feature and point indices,
        including inverting point and feature names, if available.

        Examples
        --------
        >>> raw = [[1, 2, 3], [4, 5, 6]]
        >>> data = nimble.createData('List', raw)
        >>> data
        List(
            [[1.000 2.000 3.000]
             [4.000 5.000 6.000]]
            )
        >>> data.T
        List(
            [[1.000 4.000]
             [2.000 5.000]
             [3.000 6.000]]
            )
        """
        ret = self.copy()
        ret.transpose(useLog=False)
        return ret


    def referenceDataFrom(self, other, useLog=None):
        """
        Redefine the object data using the data from another object.

        Modify the internal data of this object to refer to the same
        data as other. In other words, the data wrapped by both the self
        and ``other`` objects resides in the same place in memory.
        Attributes descrbing this object, not its data, will remain the
        same. For example the object's ``name`` attribute will remain.

        Parameters
        ----------
        other : nimble Base object
            Must be of the same type as the calling object. Also, the
            shape of other should be consistent with the shape of this
            object.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        Reference data from an object of all zero values.

        >>> data = nimble.ones('List', 2, 3, name='data')
        >>> data
        List(
            [[1.000 1.000 1.000]
             [1.000 1.000 1.000]]
            name="data"
            )
        >>> ptNames = ['1', '4']
        >>> ftNames = ['a', 'b', 'c']
        >>> toReference = nimble.zeros('List', 2, 3, pointNames=ptNames,
        ...                            featureNames=ftNames,
        ...                            name='reference')
        >>> data.referenceDataFrom(toReference)
        >>> data
        List(
            [[0.000 0.000 0.000]
             [0.000 0.000 0.000]]
            pointNames={'1':0, '4':1}
            featureNames={'a':0, 'b':1, 'c':2}
            name="data"
            )
        """
        # this is called first because it checks the data type
        self._referenceDataFrom_implementation(other)
        self.pointNames = other.pointNames
        self.pointNamesInverse = other.pointNamesInverse
        self.featureNames = other.featureNames
        self.featureNamesInverse = other.featureNamesInverse

        self._pointCount = other._pointCount
        self._featureCount = other._featureCount

        self._absPath = other.absolutePath
        self._relPath = other.relativePath

        self._nextDefaultValuePoint = other._nextDefaultValuePoint
        self._nextDefaultValueFeature = other._nextDefaultValueFeature

        handleLogging(useLog, 'prep', "referenceDataFrom",
                      self.getTypeString(), Base.referenceDataFrom, other)


    def copy(self, to=None, rowsArePoints=True, outputAs1D=False):
        """
        Duplicate an object. Optionally to another nimble or raw format.

        Return a new object containing the same data as this object.
        When copying to a nimble format, the pointNames and featureNames
        will also be copied, as well as any name and path metadata.

        Parameters
        ----------
        to : str, None
            If None, will return a copy of this object. To return a
            different type of nimble Base object, specify to: 'List',
            'Matrix', 'Sparse' or 'DataFrame'. To specify a raw return
            type (which will not include point or feature names),
            specify to: 'python list', 'numpy array', 'numpy matrix',
            'scipy csr', 'scipy csc', 'scipy coo', 'pandas dataframe',
            'list of dict' or 'dict of list'.
        rowsArePoints : bool
            Define whether the rows of the output object correspond to
            the points in this object. Default is True, if False the
            returned copy will transpose the data filling each row with
            feature data.
        outputAs1D : bool
            Return a one-dimensional object. Default is False, True is
            only a valid input for 'python list' and 'numpy array', all
            other formats must be two-dimensional.

        Returns
        -------
        object
            A copy of this object.  If ``to`` is not None, the copy will
            be in the specified format.

        Examples
        --------
        Copy this object in the same format.

        >>> raw = [[1, 3, 5], [2, 4, 6]]
        >>> ptNames = ['odd', 'even']
        >>> data = nimble.createData('List', raw, pointNames=ptNames,
        ...                          name="odd&even")
        >>> data
        List(
            [[1.000 3.000 5.000]
             [2.000 4.000 6.000]]
            pointNames={'odd':0, 'even':1}
            name="odd&even"
            )
        >>> dataCopy = data.copy()
        >>> dataCopy
        List(
            [[1.000 3.000 5.000]
             [2.000 4.000 6.000]]
            pointNames={'odd':0, 'even':1}
            name="odd&even"
            )

        Copy to other formats.

        >>> ptNames = ['0', '1']
        >>> ftNames = ['a', 'b']
        >>> data = nimble.identity('Matrix', 2, pointNames=ptNames,
        ...                        featureNames=ftNames)
        >>> asDataFrame = data.copy(to='DataFrame')
        >>> asDataFrame
        DataFrame(
            [[1.000 0.000]
             [0.000 1.000]]
            pointNames={'0':0, '1':1}
            featureNames={'a':0, 'b':1}
            )
        >>> asNumpyArray = data.copy(to='numpy array')
        >>> asNumpyArray
        array([[1., 0.],
               [0., 1.]])
        >>> asListOfDict = data.copy(to='list of dict')
        >>> asListOfDict
        [{'a': 1.0, 'b': 0.0}, {'a': 0.0, 'b': 1.0}]
        """
        # make lower case, strip out all white space and periods, except if
        # format is one of the accepted nimble data types
        if to is None:
            to = self.getTypeString()
        if not isinstance(to, six.string_types):
            raise InvalidArgumentType("'to' must be a string")
        if to not in ['List', 'Matrix', 'Sparse', 'DataFrame']:
            to = to.lower()
            to = to.strip()
            tokens = to.split(' ')
            to = ''.join(tokens)
            tokens = to.split('.')
            to = ''.join(tokens)
            accepted = ['pythonlist', 'numpyarray', 'numpymatrix', 'scipycsr',
                        'scipycsc', 'scipycoo', 'pandasdataframe',
                        'listofdict', 'dictoflist']
            if to not in accepted:
                msg = "The only accepted 'to' types are: 'List', 'Matrix', "
                msg += "'Sparse', 'DataFrame', 'python list', 'numpy array', "
                msg += "'numpy matrix', 'scipy csr', 'scipy csc', "
                msg += "'scipy coo', 'pandas dataframe',  'list of dict', "
                msg += "'and dict of list'"
                raise InvalidArgumentValue(msg)

        # only 'numpyarray' and 'pythonlist' are allowed to use outputAs1D flag
        if outputAs1D:
            if to != 'numpyarray' and to != 'pythonlist':
                msg = "Only 'numpy array' or 'python list' can output 1D"
                raise InvalidArgumentValueCombination(msg)
            if self._pointCount != 1 and self._featureCount != 1:
                msg = "To output as 1D there may either be only one point or "
                msg += "one feature"
                raise ImproperObjectAction(msg)
            return self._copy_outputAs1D(to)
        if to == 'pythonlist':
            return self._copy_pythonList(rowsArePoints)
        if to in ['listofdict', 'dictoflist']:
            return self._copy_nestedPythonTypes(to, rowsArePoints)

        # nimble, numpy and scipy types
        ret = self._copy_implementation(to)
        if isinstance(ret, Base):
            if not rowsArePoints:
                ret.transpose(useLog=False)
            ret._name = self.name
            ret._relPath = self.relativePath
            ret._absPath = self.absolutePath
        elif not rowsArePoints:
            ret = ret.transpose()

        return ret

    def _copy_outputAs1D(self, to):
        if self._pointCount == 0 or self._featureCount == 0:
            if to == 'numpyarray':
                return numpy.array([])
            if to == 'pythonlist':
                return []
        raw = self._copy_implementation('numpyarray').flatten()
        if to != 'numpyarray':
            raw = raw.tolist()
        return raw

    def _copy_pythonList(self, rowsArePoints):
        if self._pointCount == 0:
            return []
        if self._featureCount == 0:
            ret = []
            for _ in range(self._pointCount):
                ret.append([])
            return ret
        ret = self._copy_implementation('pythonlist')
        if not rowsArePoints:
            ret = numpy.transpose(ret).tolist()
        return ret

    def _copy_nestedPythonTypes(self, to, rowsArePoints):
        data = self._copy_implementation('numpyarray')
        if rowsArePoints:
            featureNames = self.features.getNames()
            if to == 'listofdict':
                return createListOfDict(data, featureNames)
            return createDictOfList(data, featureNames, self._featureCount)
        else:
            data = data.transpose()
            featureNames = self.points.getNames()
            if to == 'listofdict':
                return createListOfDict(data, featureNames)
            return createDictOfList(data, featureNames, self._pointCount)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()


    def fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd,
                 useLog=None):
        """
        Replace values in the data with other values.

        Revise the contents of the calling object so that it contains
        the provided values in the given location.

        Parameters
        ----------
        values : constant or nimble Base object
            * constant - a constant value with which to fill the data
              seletion.
            * nimble Base object - Size must be consistent with the
              given start and end indices.
        pointStart : int or str
            The inclusive index or name of the first point in the
            calling object whose contents will be modified.
        featureStart : int or str
            The inclusive index or name of the first feature in the
            calling object whose contents will be modified.
        pointEnd : int or str
            The inclusive index or name of the last point in the calling
            object whose contents will be modified.
        featureEnd : int or str
            The inclusive index or name of the last feature in the
            calling object whose contents will be modified.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.


        See Also
        --------
        fillUsingAllData, Points.fill, Features.fill

        Examples
        --------
        An object of ones filled with zeros from (0, 0) to (2, 2).

        >>> data = nimble.ones('Matrix', 5, 5)
        >>> filler = nimble.zeros('Matrix', 3, 3)
        >>> data.fillWith(filler, 0, 0, 2, 2)
        >>> data
        Matrix(
            [[0.000 0.000 0.000 1.000 1.000]
             [0.000 0.000 0.000 1.000 1.000]
             [0.000 0.000 0.000 1.000 1.000]
             [1.000 1.000 1.000 1.000 1.000]
             [1.000 1.000 1.000 1.000 1.000]]
            )
        """
        psIndex = self.points.getIndex(pointStart)
        peIndex = self.points.getIndex(pointEnd)
        fsIndex = self.features.getIndex(featureStart)
        feIndex = self.features.getIndex(featureEnd)

        if psIndex > peIndex:
            msg = "pointStart (" + str(pointStart) + ") must be less than or "
            msg += "equal to pointEnd (" + str(pointEnd) + ")."
            raise InvalidArgumentValueCombination(msg)
        if fsIndex > feIndex:
            msg = "featureStart (" + str(featureStart) + ") must be less than "
            msg += "or equal to featureEnd (" + str(featureEnd) + ")."
            raise InvalidArgumentValueCombination(msg)

        if isinstance(values, Base):
            prange = (peIndex - psIndex) + 1
            frange = (feIndex - fsIndex) + 1
            if len(values.points) != prange:
                msg = "When the values argument is a nimble Base object, the "
                msg += "size of values must match the range of modification. "
                msg += "There are " + str(len(values.points)) + " points in "
                msg += "values, yet pointStart (" + str(pointStart) + ")"
                msg += "and pointEnd (" + str(pointEnd) + ") define a range "
                msg += "of length " + str(prange)
                raise InvalidArgumentValueCombination(msg)
            if len(values.features) != frange:
                msg = "When the values argument is a nimble Base object, the "
                msg += "size of values must match the range of modification. "
                msg += "There are " + str(len(values.features)) + " features "
                msg += "in values, yet featureStart (" + str(featureStart)
                msg += ") and featureEnd (" + str(featureEnd) + ") define a "
                msg += "range of length " + str(frange)
                raise InvalidArgumentValueCombination(msg)
            if values.getTypeString() != self.getTypeString():
                values = values.copy(to=self.getTypeString())

        elif (dataHelpers._looksNumeric(values)
              or isinstance(values, six.string_types)):
            pass  # no modifications needed
        else:
            msg = "values may only be a nimble Base object, or a single "
            msg += "numeric value, yet we received something of "
            msg += str(type(values))
            raise InvalidArgumentType(msg)

        self._fillWith_implementation(values, psIndex, fsIndex,
                                      peIndex, feIndex)

        handleLogging(useLog, 'prep', "fillWith",
                      self.getTypeString(), Base.fillWith, values, pointStart,
                      featureStart, pointEnd, featureEnd)


    def fillUsingAllData(self, match, fill, points=None, features=None,
                         returnModified=False, useLog=None, **kwarguments):
        """
        Replace matching values calculated using the entire data object.

        Fill matching values with values based on the context of the
        entire dataset.

        Parameters
        ----------
        match : value, list, or function
            * value - a value to locate within each feature
            * list - values to locate within each feature
            * function - must accept a single value and return True if
              the value is a match. Certain match types can be imported
              from nimble's match module: missing, nonNumeric, zero, etc
        fill : function
            a function in the format fill(feature, match) or
            fill(feature, match, arguments) and return the transformed
            data as a nimble data object. Certain fill methods can be
            imported from nimble's fill module:
            kNeighborsRegressor, kNeighborsClassifier
        points : identifier or list of identifiers
            Select specific points to apply fill to. If points is None,
            the fill will be applied to all points.
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
        kwarguments
            Any additional arguments being passed to the fill function.

        See Also
        --------
        fillWith, Points.fill, Features.fill

        Examples
        --------
        Fill using the value that occurs most often in each points 3
        nearest neighbors.

        >>> from nimble.fill import kNeighborsClassifier
        >>> raw = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, 'na'],
        ...        [2, 2, 2],
        ...        ['na', 2, 2]]
        >>> data = nimble.createData('Matrix', raw)
        >>> data.fillUsingAllData('na', kNeighborsClassifier,
        ...                       n_neighbors=3)
        >>> data
        Matrix(
            [[  1   1   1  ]
             [  1   1   1  ]
             [  1   1 1.000]
             [  2   2   2  ]
             [1.000 2   2  ]]
            )
        """
        if returnModified:
            modified = self.elements.calculate(match, points=points,
                                               features=features, useLog=False)
            modNames = [name + "_modified" for name
                        in modified.features.getNames()]
            modified.features.setNames(modNames, useLog=False)
            if points is not None and features is not None:
                modified = modified[points, features]
            elif points is not None:
                modified = modified[points, :]
            elif features is not None:
                modified = modified[:, features]
        else:
            modified = None

        if not callable(fill):
            msg = "fill must be callable. If attempting to modify all "
            msg += "matching values to a constant, use either "
            msg += "points.fill or features.fill."
            raise InvalidArgumentType(msg)
        tmpData = fill(self.copy(), match, **kwarguments)
        if points is None and features is None:
            self.referenceDataFrom(tmpData, useLog=False)
        else:
            def transform(value, i, j):
                return tmpData[i, j]
            self.elements.transform(transform, points, features, useLog=False)

        handleLogging(useLog, 'prep', "fillUsingAllData",
                      self.getTypeString(), Base.fillUsingAllData, match, fill,
                      points, features, returnModified, **kwarguments)

        return modified

    def _flattenNames(self, discardAxis):
        """
        Helper calculating the axis names for the unflattend axis after
        a flatten operation.
        """
        self._validateAxis(discardAxis)
        if discardAxis == 'point':
            keepNames = self.features.getNames()
            dropNames = self.points.getNames()
        else:
            keepNames = self.points.getNames()
            dropNames = self.features.getNames()

        ret = []
        for d in dropNames:
            for k in keepNames:
                ret.append(k + ' | ' + d)

        return ret


    def flattenToOnePoint(self, useLog=None):
        """
        Modify this object so that its values are in a single point.

        Each feature in the result maps to exactly one value from the
        original object. The order of values respects the point order
        from the original object, if there were n features in the
        original, the first n values in the result will exactly match
        the first point, the nth to (2n-1)th values will exactly match
        the original second point, etc. The feature names will be
        transformed such that the value at the intersection of the
        "pn_i" named point and "fn_j" named feature from the original
        object will have a feature name of "fn_j | pn_i". The single
        point will have a name of "Flattened". This is an inplace
        operation.

        Parameters
        ----------
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        unflattenFromOnePoint

        Examples
        --------
        >>> raw = [[1, 2],
        ...        [3, 4]]
        >>> ptNames = ['1', '4']
        >>> ftNames = ['a', 'b']
        >>> data = nimble.createData('Matrix', raw, pointNames=ptNames,
        ...                          featureNames=ftNames)
        >>> data.flattenToOnePoint()
        >>> data
        Matrix(
            [[1.000 2.000 3.000 4.000]]
            pointNames={'Flattened':0}
            featureNames={'a | 1':0, 'b | 1':1, 'a | 4':2, 'b | 4':3}
            )
        """
        if self._pointCount == 0:
            msg = "Can only flattenToOnePoint when there is one or more "
            msg += "points. This object has 0 points."
            raise ImproperObjectAction(msg)
        if self._featureCount == 0:
            msg = "Can only flattenToOnePoint when there is one or more "
            msg += "features. This object has 0 features."
            raise ImproperObjectAction(msg)

        # TODO: flatten nameless Objects without the need to generate default
        # names for them.
        if not self._pointNamesCreated():
            self.points._setAllDefault()
        if not self._featureNamesCreated():
            self.features._setAllDefault()

        self._flattenToOnePoint_implementation()

        self._featureCount = self._pointCount * self._featureCount
        self._pointCount = 1
        self.features.setNames(self._flattenNames('point'), useLog=False)
        self.points.setNames(['Flattened'], useLog=False)

        handleLogging(useLog, 'prep', "flattenToOnePoint",
                      self.getTypeString(), Base.flattenToOnePoint)


    def flattenToOneFeature(self, useLog=None):
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

        Parameters
        ----------
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        unflattenFromOneFeature

        Examples
        --------
        >>> raw = [[1, 2],
        ...        [3, 4]]
        >>> ptNames = ['1', '4']
        >>> ftNames = ['a', 'b']
        >>> data = nimble.createData('Matrix', raw, pointNames=ptNames,
        ...                          featureNames=ftNames)
        >>> data.flattenToOneFeature()
        >>> data
        Matrix(
            [[1.000]
             [3.000]
             [2.000]
             [4.000]]
            pointNames={'1 | a':0, '4 | a':1, '1 | b':2, '4 | b':3}
            featureNames={'Flattened':0}
            )
        """
        if self._pointCount == 0:
            msg = "Can only flattenToOnePoint when there is one or more "
            msg += "points. This object has 0 points."
            raise ImproperObjectAction(msg)
        if self._featureCount == 0:
            msg = "Can only flattenToOnePoint when there is one or more "
            msg += "features. This object has 0 features."
            raise ImproperObjectAction(msg)

        # TODO: flatten nameless Objects without the need to generate default
        # names for them.
        if not self._pointNamesCreated():
            self.points._setAllDefault()
        if not self._featureNamesCreated():
            self.features._setAllDefault()

        self._flattenToOneFeature_implementation()

        self._pointCount = self._pointCount * self._featureCount
        self._featureCount = 1
        self.points.setNames(self._flattenNames('feature'), useLog=False)
        self.features.setNames(['Flattened'], useLog=False)

        handleLogging(useLog, 'prep', "flattenToOneFeature",
                      self.getTypeString(), Base.flattenToOneFeature)

    def _unflattenNames(self, addedAxis, addedAxisLength):
        """
        Helper calculating the new axis names after an unflattening
        operation.
        """
        self._validateAxis(addedAxis)
        if addedAxis == 'point':
            both = self.features.getNames()
            keptAxisLength = self._featureCount // addedAxisLength
            allDefault = self._namesAreFlattenFormatConsistent(
                'point', addedAxisLength, keptAxisLength)
        else:
            both = self.points.getNames()
            keptAxisLength = self._pointCount // addedAxisLength
            allDefault = self._namesAreFlattenFormatConsistent(
                'feature', addedAxisLength, keptAxisLength)

        if allDefault:
            addedAxisName = None
            keptAxisName = None
        else:
            # we consider the split of the elements into keptAxisLength chunks
            # (of which there will be addedAxisLength number of chunks), and
            # want the index of the first of each chunk. We allow that first
            # name to be representative for that chunk: all will have the same
            # stuff past the vertical bar.
            locations = range(0, len(both), keptAxisLength)
            addedAxisName = [both[n].split(" | ")[1] for n in locations]
            keptAxisName = [name.split(" | ")[0] for name
                            in both[:keptAxisLength]]

        return addedAxisName, keptAxisName

    def _namesAreFlattenFormatConsistent(self, flatAxis, newFLen, newUFLen):
        """
        Helper which validates the formatting of axis names prior to
        unflattening.

        Will raise ImproperObjectAction if an inconsistency with the
        formatting done by the flatten operations is discovered. Returns
        True if all the names along the unflattend axis are default,
        False otherwise.
        """
        if flatAxis == 'point':
            flat = self.points.getNames()
            formatted = self.features.getNames()
        else:
            flat = self.features.getNames()
            formatted = self.points.getNames()

        def checkIsDefault(axisName):
            ret = False
            try:
                if axisName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
                    int(axisName[DEFAULT_PREFIX_LENGTH:])
                    ret = True
            except ValueError:
                ret = False
            return ret

        # check the contents of the names along the flattened axis
        isDefault = checkIsDefault(flat[0])
        isExact = flat == ['Flattened']
        msg = "In order to unflatten this object, the names must be "
        msg += "consistent with the results from a flatten call. "
        if not (isDefault or isExact):
            msg += "Therefore, the {axis} name for this object ('{axisName}')"
            msg += "must either be a default name or the string 'Flattened'"
            msg = msg.format(axis=flatAxis, axisName=flat[0])
            raise ImproperObjectAction(msg)

        # check the contents of the names along the unflattend axis
        msg += "Therefore, the {axis} names for this object must either be "
        msg += "all default, or they must be ' | ' split names with name "
        msg += "values consistent with the positioning from a flatten call."
        msg.format(axis=flatAxis)
        # each name - default or correctly formatted
        allDefaultStatus = None
        for name in formatted:
            isDefault = checkIsDefault(name)
            formatCorrect = len(name.split(" | ")) == 2
            if allDefaultStatus is None:
                allDefaultStatus = isDefault
            else:
                if isDefault != allDefaultStatus:
                    raise ImproperObjectAction(msg)

            if not (isDefault or formatCorrect):
                raise ImproperObjectAction(msg)

        # consistency only relevant if we have non-default names
        if not allDefaultStatus:
            # seen values - consistent wrt original flattend axis names
            for i in range(newFLen):
                same = formatted[newUFLen*i].split(' | ')[1]
                for name in formatted[newUFLen*i:newUFLen*(i+1)]:
                    if same != name.split(' | ')[1]:
                        raise ImproperObjectAction(msg)

            # seen values - consistent wrt original unflattend axis names
            for i in range(newUFLen):
                same = formatted[i].split(' | ')[0]
                for j in range(newFLen):
                    name = formatted[i + (j * newUFLen)]
                    if same != name.split(' | ')[0]:
                        raise ImproperObjectAction(msg)

        return allDefaultStatus


    def unflattenFromOnePoint(self, numPoints, useLog=None):
        """
        Adjust a flattened point vector to contain multiple points.

        This is an inverse of the method ``flattenToOnePoint``: if an
        object foo with n points calls the flatten method, then this
        method with n as the argument, the result should be identical to
        the original foo. It is not limited to objects that have
        previously had ``flattenToOnePoint`` called on them; any object
        whose structure and names are consistent with a previous call to
        ``flattenToOnePoint`` may call this method. This includes
        objects with all default names. This is an inplace operation.

        Parameters
        ----------
        numPoints : int
            The number of points in the modified object.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        flattenToOnePoint

        Examples
        --------

        With all default names.

        >>> raw = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        >>> data = nimble.createData('Matrix', raw)
        >>> data.unflattenFromOnePoint(3)
        >>> data
        Matrix(
            [[1.000 2.000 3.000]
             [4.000 5.000 6.000]
             [7.000 8.000 9.000]]
            )

        With names consistent with call to ``flattenToOnePoint``.

        >>> raw = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        >>> ptNames = {'Flattened':0}
        >>> ftNames = {'a | 1':0, 'b | 1':1, 'c | 1':2,
        ...            'a | 4':3, 'b | 4':4, 'c | 4':5,
        ...            'a | 7':6, 'b | 7':7, 'c | 7':8}
        >>> data = nimble.createData('Matrix', raw, pointNames=ptNames,
        ...                          featureNames=ftNames)
        >>> data.unflattenFromOnePoint(3)
        >>> data
        Matrix(
            [[1.000 2.000 3.000]
             [4.000 5.000 6.000]
             [7.000 8.000 9.000]]
            pointNames={'1':0, '4':1, '7':2}
            featureNames={'a':0, 'b':1, 'c':2}
            )
        """
        if self._featureCount == 0:
            msg = "Can only unflattenFromOnePoint when there is one or more "
            msg += "features.  This object has 0 features."
            raise ImproperObjectAction(msg)
        if self._pointCount != 1:
            msg = "Can only unflattenFromOnePoint when there is only one "
            msg += "point.  This object has " + str(self._pointCount)
            msg += " points."
            raise ImproperObjectAction(msg)
        if self._featureCount % numPoints != 0:
            msg = "The argument numPoints (" + str(numPoints) + ") must be a "
            msg += "divisor of  this object's featureCount ("
            msg += str(self._featureCount) + ") otherwise  it will not be "
            msg += "possible to equally divide the elements into the desired "
            msg += "number of points."
            raise InvalidArgumentValue(msg)

        if not self._pointNamesCreated():
            self.points._setAllDefault()
        if not self._featureNamesCreated():
            self.features._setAllDefault()

        self._unflattenFromOnePoint_implementation(numPoints)
        ret = self._unflattenNames('point', numPoints)
        self._featureCount = self._featureCount // numPoints
        self._pointCount = numPoints
        self.points.setNames(ret[0], useLog=False)
        self.features.setNames(ret[1], useLog=False)

        handleLogging(useLog, 'prep', "unflattenFromOnePoint",
                      self.getTypeString(), Base.unflattenFromOnePoint,
                      numPoints)


    def unflattenFromOneFeature(self, numFeatures, useLog=None):
        """
        Adjust a flattened feature vector to contain multiple features.

        This is an inverse of the method ``flattenToOneFeature``: if an
        object foo with n features calls the flatten method, then this
        method with n as the argument, the result should be identical to
        the original foo. It is not limited to objects that have
        previously had ``flattenToOneFeature`` called on them; any
        object whose structure and names are consistent with a previous
        call to ``flattenToOneFeature`` may call this method. This
        includes objects with all default names. This is an inplace
        operation.

        Parameters
        ----------
        numFeatures : int
            The number of features in the modified object.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        flattenToOneFeature

        Examples
        --------
        With default names.

        >>> raw = [[1], [4], [7], [2], [5], [8], [3], [6], [9]]
        >>> data = nimble.createData('Matrix', raw)
        >>> data.unflattenFromOneFeature(3)
        >>> data
        Matrix(
            [[1.000 2.000 3.000]
             [4.000 5.000 6.000]
             [7.000 8.000 9.000]]
            )

        With names consistent with call to ``flattenToOneFeature``.

        >>> raw = [[1], [4], [7], [2], [5], [8], [3], [6], [9]]
        >>> ptNames = {'1 | a':0, '4 | a':1, '7 | a':2,
        ...            '1 | b':3, '4 | b':4, '7 | b':5,
        ...            '1 | c':6, '4 | c':7, '7 | c':8}
        >>> ftNames = {'Flattened':0}
        >>> data = nimble.createData('Matrix', raw, pointNames=ptNames,
        ...                          featureNames=ftNames)
        >>> data.unflattenFromOneFeature(3)
        >>> data
        Matrix(
            [[1.000 2.000 3.000]
             [4.000 5.000 6.000]
             [7.000 8.000 9.000]]
            pointNames={'1':0, '4':1, '7':2}
            featureNames={'a':0, 'b':1, 'c':2}
            )
        """
        if self._pointCount == 0:
            msg = "Can only unflattenFromOneFeature when there is one or more "
            msg += "points. This object has 0 points."
            raise ImproperObjectAction(msg)
        if self._featureCount != 1:
            msg = "Can only unflattenFromOneFeature when there is only one "
            msg += "feature. This object has " + str(self._featureCount)
            msg += " features."
            raise ImproperObjectAction(msg)

        if self._pointCount % numFeatures != 0:
            msg = "The argument numFeatures (" + str(numFeatures) + ") must "
            msg += "be a divisor of this object's pointCount ("
            msg += str(self._pointCount) + ") otherwise "
            msg += "it will not be possible to equally divide the elements "
            msg += "into the desired number of features."
            raise InvalidArgumentValue(msg)

        if not self._pointNamesCreated():
            self.points._setAllDefault()
        if not self._featureNamesCreated():
            self.features._setAllDefault()

        self._unflattenFromOneFeature_implementation(numFeatures)
        ret = self._unflattenNames('feature', numFeatures)
        self._pointCount = self._pointCount // numFeatures
        self._featureCount = numFeatures
        self.points.setNames(ret[1], useLog=False)
        self.features.setNames(ret[0], useLog=False)

        handleLogging(useLog, 'prep', "unflattenFromOneFeature",
                      self.getTypeString(), Base.unflattenFromOneFeature,
                      numFeatures)


    def merge(self, other, point='strict', feature='union', onFeature=None,
              useLog=None):
        """
        Merge data from another object into this object.

        Merge data based on point names or a common feature between the
        objects. How the data will be merged is based upon the string
        arguments provided to ``point`` and ``feature``. If
        ``onFeature`` is None, the objects will be merged on the point
        names. Otherwise, the objects will be merged on the feature
        provided. ``onFeature`` allows for duplicate values to be
        present in the provided feature, however, one of the objects
        must contain only unique values for each point when
        ``onFeature`` is provided.

        Parameters
        ----------
        other : Base
            The nimble data object containing the data to merge.
        point, feature : str
            The allowed strings for the point and feature arguments are
            as follows:
            * 'strict' - The points/features in the callee exactly match
              the points/features in the caller, however, they may be in
              a different order. If ``onFeature`` is None and no names
              are provided, it will be assumed the order is the same.
            * 'union' - Return all points/features from the caller and
              callee. If ``onFeature`` is None, unnamed points/features
              will be assumed to be unique. Any missing data from the
              caller and callee will be filled with numpy.NaN.
            * 'intersection': Return only points/features shared between
              the caller  and callee. If ``onFeature`` is None,
              point / feature names are required.
            * 'left': Return only the points/features from the caller.
              Any missing data from the callee will be filled with
              numpy.NaN.
        onFeature : identifier, None
            The name or index of the feature present in both objects to
            merge on.  If None, the merge will be based on point names.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        Points.add, Features.add

        Examples
        --------
        A strict case. In this case we will merge using the point names,
        so ``point='strict'`` requires that the each object has the same
        point names (or one or both have default names) In this example,
        there is one shared feature between objects. If
        ``feature='union'``, all features ("f1"-"f5") will be included,
        if ``feature='intersection'``, only the shared feature ("f3")
        will be included, ``feature='left'`` will only use the features
        from the left object (not shown, in strict cases 'left' will not
        modify the left object at all).
        >>> dataL = [["a", 1, 'X'], ["b", 2, 'Y'], ["c", 3, 'Z']]
        >>> fNamesL = ["f1", "f2", "f3"]
        >>> pNamesL = ["p1", "p2", "p3"]
        >>> left = nimble.createData('Matrix', dataL, pointNames=pNamesL,
        ...                          featureNames=fNamesL)
        >>> dataR = [['Z', "f", 6], ['Y', "e", 5], ['X', "d", 4]]
        >>> fNamesR = ["f3", "f4", "f5"]
        >>> pNamesR = ["p3", "p2", "p1"]
        >>> right = nimble.createData('Matrix', dataR, pointNames=pNamesR,
        ...                           featureNames=fNamesR)
        >>> left.merge(right, point='strict', feature='union')
        >>> left
        Matrix(
            [[a 1 X d 4]
             [b 2 Y e 5]
             [c 3 Z f 6]]
            pointNames={'p1':0, 'p2':1, 'p3':2}
            featureNames={'f1':0, 'f2':1, 'f3':2, 'f4':3, 'f5':4}
            )
        >>> left = nimble.createData('Matrix', dataL, pointNames=pNamesL,
        ...                          featureNames=fNamesL)
        >>> left.merge(right, point='strict', feature='intersection')
        >>> left
        Matrix(
            [[X]
             [Y]
             [Z]]
            pointNames={'p1':0, 'p2':1, 'p3':2}
            featureNames={'f3':0}
            )

        Additional merge combinations. In this example, the feature
        ``"id"`` contains a unique value for each point (just as point
        names do). In the example above we matched based on point names,
        here the ``"id"`` feature will be used to match points.
        >>> dataL = [["a", 1, 'id1'], ["b", 2, 'id2'], ["c", 3, 'id3']]
        >>> fNamesL = ["f1", "f2", "id"]
        >>> left = nimble.createData("DataFrame", dataL,
        ...                          featureNames=fNamesL)
        >>> dataR = [['id3', "x", 7], ['id4', "y", 8], ['id5', "z", 9]]
        >>> fNamesR = ["id", "f4", "f5"]
        >>> right = nimble.createData("DataFrame", dataR,
        ...                           featureNames=fNamesR)
        >>> left.merge(right, point='union', feature='union',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[ a   1  id1 nan nan]
             [ b   2  id2 nan nan]
             [ c   3  id3  x   7 ]
             [nan nan id4  y   8 ]
             [nan nan id5  z   9 ]]
            featureNames={'f1':0, 'f2':1, 'id':2, 'f4':3, 'f5':4}
            )
        >>> left = nimble.createData("DataFrame", dataL,
        ...                          featureNames=fNamesL)
        >>> left.merge(right, point='union', feature='intersection',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[id1]
             [id2]
             [id3]
             [id4]
             [id5]]
            featureNames={'id':0}
            )
        >>> left = nimble.createData("DataFrame", dataL,
        ...                          featureNames=fNamesL)
        >>> left.merge(right, point='union', feature='left',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[ a   1  id1]
             [ b   2  id2]
             [ c   3  id3]
             [nan nan id4]
             [nan nan id5]]
            featureNames={'f1':0, 'f2':1, 'id':2}
            )
        >>> left = nimble.createData("DataFrame", dataL,
        ...                          featureNames=fNamesL)
        >>> left.merge(right, point='intersection', feature='union',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[c 3 id3 x 7]]
            featureNames={'f1':0, 'f2':1, 'id':2, 'f4':3, 'f5':4}
            )
        >>> left = nimble.createData("DataFrame", dataL,
        ...                          featureNames=fNamesL)
        >>> left.merge(right, point='intersection',
        ...            feature='intersection', onFeature="id")
        >>> left
        DataFrame(
            [[id3]]
            featureNames={'id':0}
            )
        >>> left = nimble.createData("DataFrame", dataL,
        ...                          featureNames=fNamesL)
        >>> left.merge(right, point='intersection', feature='left',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[c 3 id3]]
            featureNames={'f1':0, 'f2':1, 'id':2}
            )
        >>> left = nimble.createData("DataFrame", dataL,
        ...                          featureNames=fNamesL)
        >>> left.merge(right, point='left', feature='union',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[a 1 id1 nan nan]
             [b 2 id2 nan nan]
             [c 3 id3  x   7 ]]
            featureNames={'f1':0, 'f2':1, 'id':2, 'f4':3, 'f5':4}
            )
        >>> left = nimble.createData("DataFrame", dataL,
        ...                          featureNames=fNamesL)
        >>> left.merge(right, point='left', feature='intersection',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[id1]
             [id2]
             [id3]]
            featureNames={'id':0}
            )
        >>> left = nimble.createData("DataFrame", dataL,
        ...                          featureNames=fNamesL)
        >>> left.merge(right, point='left', feature='left',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[a 1 id1]
             [b 2 id2]
             [c 3 id3]]
            featureNames={'f1':0, 'f2':1, 'id':2}
            )
        """
        point = point.lower()
        feature = feature.lower()
        valid = ['strict', 'left', 'union', 'intersection']
        if point not in valid:
            msg = "point must be 'strict', 'left', 'union', or 'intersection'"
            raise InvalidArgumentValue(msg)
        if feature not in valid:
            msg = "feature must be 'strict', 'left', 'union', or "
            msg += "'intersection'"
            raise InvalidArgumentValue(msg)

        if point == 'strict' or feature == 'strict':
            self._genericStrictMerge_implementation(other, point, feature,
                                                    onFeature)
        else:
            self._genericMergeFrontend(other, point, feature, onFeature)

        handleLogging(useLog, 'prep', "merge", self.getTypeString(),
                      Base.merge, other, point, feature, onFeature)

    def _genericStrictMerge_implementation(self, other, point, feature,
                                           onFeature):
        """
        Validation and helper function when point or feature is set to
        strict.
        """
        # NOTE could return this object?
        if point == 'strict' and feature == 'strict':
            msg = 'Both point and feature cannot be strict'
            raise InvalidArgumentValueCombination(msg)
        tmpOther = other.copy()
        if point == 'strict':
            axis = 'point'
            countL = len(self.points)
            countR = len(tmpOther.points)
            hasNamesL = not self._allDefaultPointNames()
            hasNamesR = not tmpOther._allDefaultPointNames()
            namesL = self.points.getNames
            namesR = tmpOther.points.getNames
            setNamesL = self.points.setNames
            setNamesR = tmpOther.points.setNames
            point = "intersection"
        else:
            axis = 'feature'
            countL = len(self.features)
            countR = len(tmpOther.features)
            hasNamesL = not self._allDefaultFeatureNames()
            hasNamesR = not tmpOther._allDefaultFeatureNames()
            namesL = self.features.getNames
            namesR = tmpOther.features.getNames
            setNamesL = self.features.setNames
            setNamesR = tmpOther.features.setNames
            feature = "intersection"

        if countL != countR:
            msg = "Both objects must have the same number of "
            msg += "{0}s when {0}='strict'".format(axis)
            raise InvalidArgumentValue(msg)
        if hasNamesL and hasNamesR:
            if sorted(namesL()) != sorted(namesR()):
                msg = "When {0}='strict', the {0}s names ".format(axis)
                msg += "may be in a different order but must match exactly"
                raise InvalidArgumentValue(msg)
        # since strict implies that the points or features are the same,
        # if one object does not have names along the axis, but the length
        # matches, we will assume that the unnamed should have the same names
        if onFeature is None:
            if hasNamesL and not hasNamesR:
                setNamesR(namesL(), useLog=False)
            elif not hasNamesL and hasNamesR:
                setNamesL(namesR(), useLog=False)
            elif not hasNamesL and not hasNamesR:
                strictPrefix = '_STRICT' + DEFAULT_PREFIX
                strictNames = [strictPrefix + str(i) for i in range(countL)]
                setNamesL(strictNames, useLog=False)
                setNamesR(strictNames, useLog=False)
        # if using strict with onFeature instead of point names, we need to
        # make sure each id has a unique match in the other object
        elif axis == 'point':
            try:
                self[0, onFeature]
                tmpOther[0, onFeature]
            except KeyError:
                msg = "could not locate feature '{0}' ".format(onFeature)
                msg += "in both objects"
                raise InvalidArgumentValue(msg)
            if len(set(self[:, onFeature])) != len(self.points):
                msg = "when point='strict', onFeature must contain only "
                msg += "unique values"
                raise InvalidArgumentValueCombination(msg)
            if sorted(self[:, onFeature]) != sorted(tmpOther[:, onFeature]):
                msg = "When point='strict', onFeature must have a unique, "
                msg += "matching value in each object"
                raise InvalidArgumentValueCombination(msg)

        self._genericMergeFrontend(tmpOther, point, feature, onFeature, axis)

    def _genericMergeFrontend(self, other, point, feature, onFeature,
                              strict=None):
        # validation
        bothNamesCreated = (self._pointNamesCreated()
                            and other._pointNamesCreated())
        if ((onFeature is None and point == "intersection")
                and not bothNamesCreated):
            msg = "Point names are required in both objects when "
            msg += "point='intersection'"
            raise InvalidArgumentValueCombination(msg)
        bothNamesCreated = (self._featureNamesCreated()
                            and other._featureNamesCreated())
        if feature == "intersection" and not bothNamesCreated:
            msg = "Feature names are required in both objects when "
            msg += "feature='intersection'"
            raise InvalidArgumentValueCombination(msg)

        if onFeature is not None:
            try:
                self[0, onFeature]
                other[0, onFeature]
            except KeyError:
                msg = "could not locate feature '{0}' ".format(onFeature)
                msg += "in both objects"
                raise InvalidArgumentValue(msg)
            uniqueFtL = len(set(self[:, onFeature])) == len(self.points)
            uniqueFtR = len(set(other[:, onFeature])) == len(other.points)
            if not (uniqueFtL or uniqueFtR):
                msg = "nimble only supports joining on a feature which "
                msg += "contains only unique values in one or both objects."
                raise InvalidArgumentValue(msg)

        matchingFts = self._getMatchingNames('feature', other)
        matchingFtIdx = [[], []]
        for name in matchingFts:
            idxL = self.features.getIndex(name)
            idxR = other.features.getIndex(name)
            matchingFtIdx[0].append(idxL)
            matchingFtIdx[1].append(idxR)

        if self.getTypeString() != other.getTypeString():
            other = other.copy(to=self.getTypeString())
        self._merge_implementation(other, point, feature, onFeature,
                                   matchingFtIdx)

        if strict == 'feature':
            if ('_STRICT' in self.features.getName(0)
                    and '_STRICT' in other.features.getName(0)):
                # objects did not have feature names
                self.featureNames = None
                self.featureNamesInverse = None
            elif '_STRICT' in self.features.getName(0):
                # use feature names from other object
                self.features.setNames(other.features.getNames(), useLog=False)
        elif feature == "intersection":
            if self._featureNamesCreated():
                ftNames = [n for n in self.features.getNames()
                           if n in matchingFts]
                self.features.setNames(ftNames, useLog=False)
        elif feature == "union":
            if self._featureNamesCreated() and other._featureNamesCreated():
                ftNamesL = self.features.getNames()
                ftNamesR = [name for name in other.features.getNames()
                            if name not in matchingFts]
                ftNames = ftNamesL + ftNamesR
                self.features.setNames(ftNames, useLog=False)
            elif self._featureNamesCreated():
                ftNamesL = self.features.getNames()
                ftNamesR = [DEFAULT_PREFIX + str(i) for i
                            in range(len(other.features))]
                ftNames = ftNamesL + ftNamesR
                self.features.setNames(ftNames, useLog=False)
            elif other._featureNamesCreated():
                ftNamesL = [DEFAULT_PREFIX + str(i) for i
                            in range(len(self.features))]
                ftNamesR = other.features.getNames()
                ftNames = ftNamesL + ftNamesR
                self.features.setNames(ftNames, useLog=False)
        # no name setting needed for left

        if strict == 'point':
            if ('_STRICT' in self.points.getName(0)
                    and '_STRICT' in other.points.getName(0)):
                # objects did not have point names
                self.pointNames = None
                self.pointNamesInverse = None
            elif '_STRICT' in self.points.getName(0):
                # use point names from other object
                self.points.setNames(other.points.getNames(), useLog=False)
        elif onFeature is None and point == 'left':
            if self._pointNamesCreated():
                self.points.setNames(self.points.getNames(), useLog=False)
        elif onFeature is None and point == 'intersection':
            # default names cannot be included in intersection
            ptNames = [name for name in self.points.getNames()
                       if name in other.points.getNames()
                       and not name.startswith(DEFAULT_PREFIX)]
            self.points.setNames(ptNames, useLog=False)
        elif onFeature is None:
            # union cases
            if self._pointNamesCreated() and other._pointNamesCreated():
                ptNamesL = self.points.getNames()
                if other._anyDefaultPointNames():
                    # handle default name conflicts
                    ptNamesR = [self.points._nextDefaultName() if
                                n.startswith(DEFAULT_PREFIX) else n
                                for n in self.points.getNames()]
                else:
                    ptNamesR = other.points.getNames()
                ptNames = ptNamesL + [name for name in ptNamesR
                                      if name not in ptNamesL]
                self.points.setNames(ptNames, useLog=False)
            elif self._pointNamesCreated():
                ptNamesL = self.points.getNames()
                ptNamesR = [self.points._nextDefaultName() for _
                            in range(len(other.points))]
                ptNames = ptNamesL + ptNamesR
                self.points.setNames(ptNames, useLog=False)
            elif other._pointNamesCreated():
                ptNamesL = [other.points._nextDefaultName() for _
                            in range(len(self.points))]
                ptNamesR = other.points.getNames()
                ptNames = ptNamesL + ptNamesR
                self.points.setNames(ptNames, useLog=False)
        else:
            self.pointNamesInverse = None
            self.pointNames = None


    def _getMatchingNames(self, axis, other):
        matches = []
        if axis == 'point':
            if not self._pointNamesCreated() or not other._pointNamesCreated():
                return matches
            selfNames = self.points.getNames()
            otherNames = other.points.getNames()
        else:
            if (not self._featureNamesCreated()
                    or not other._featureNamesCreated()):
                return matches
            selfNames = self.features.getNames()
            otherNames = other.features.getNames()
        allNames = selfNames + otherNames
        hasMatching = len(set(allNames)) != len(allNames)
        if hasMatching:
            for name in selfNames:
                if name in otherNames:
                    matches.append(name)
        return matches


    ###################################
    ###################################
    ###   Linear Algebra functions   ###
    ###################################
    ###################################

    def inverse(self, pseudoInverse=False):
        """
        Compute the inverse or pseudo-inverse of an object.

        By default tries to compute the (multiplicative) inverse.
        pseudoInverse uses singular-value decomposition (SVD).

        Parameters
        ----------
        pseudoInverse : bool
            Whether to compute pseudoInverse or multiplicative inverse.
        """

        if pseudoInverse:
            inverse = nimble.calculate.pseudoInverse(self)
        else:
            inverse = nimble.calculate.inverse(self)
        return inverse


    def solveLinearSystem(self, b, solveFunction='solve'):
        """
       Solves the linear equation A * x = b for unknown x.

       Parameters
       ----------
       b : nimble Base object.
        Vector shaped object.
       solveFuction : str
        * 'solve' - assumes square matrix.
        * 'least squares' - Computes object x such that 2-norm |b - Ax|
          is minimized.
        """
        if not isinstance(b, Base):
            msg = "b must be an instance of Base."
            raise InvalidArgumentType(msg)

        msg = "Valid methods are: 'solve' and 'least squares'."

        if not isinstance(solveFunction, str):
            raise InvalidArgumentType(msg)
        elif solveFunction == 'solve':
            return nimble.calculate.solve(self, b)
        elif solveFunction == 'least squares':
            return nimble.calculate.leastSquaresSolution(self, b)
        else:
            raise InvalidArgumentValue(msg)



    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################

    def matrixMultiply(self, other):
        return self.__matmul__(other)

    def __matmul__(self, other):
        """
        Perform matrix multiplication.
        """
        return self._genericMatMul_implementation('__matmul__', other)

    def __rmatmul__(self, other):
        """
        Perform matrix multiplication with this object on the right.
        """
        return self._genericMatMul_implementation('__rmatmul__', other)

    def __imatmul__(self, other):
        """
        Perform in place matrix multiplication.
        """
        ret = self.__matmul__(other)
        if ret is not NotImplemented:
            self.referenceDataFrom(ret, useLog=False)
            ret = self
        return ret

    def _genericMatMul_implementation(self, opName, other):
        if not isinstance(other, Base):
            return NotImplemented
        # Test element type self
        if self._pointCount == 0 or self._featureCount == 0:
            msg = "Cannot do a multiplication when points or features is empty"
            raise ImproperObjectAction(msg)

        # test element type other
        if len(other.points) == 0 or len(other.features) == 0:
            msg = "Cannot do a multiplication when points or features is "
            msg += "empty"
            raise ImproperObjectAction(msg)

        if opName.startswith('__r'):
            caller = other
            callee = self
        else:
            caller = self
            callee = other

        if caller._featureCount != len(callee.points):
            msg = "The number of features in the left hand object must "
            msg += "match the number of points in the right hand side object."
            raise InvalidArgumentValue(msg)

        caller._validateEqualNames('feature', 'point', opName, callee)

        # match types with self (the orignal caller of the __r*__ op)
        if (opName.startswith('__r')
                and self.getTypeString() != other.getTypeString()):
            caller = caller.copy(self.getTypeString())

        try:
            ret = caller._matmul__implementation(callee)
        except Exception as e:
            #TODO: improve how the exception is catch
            dataHelpers.numericValidation(self)
            dataHelpers.numericValidation(other, right=True)
            raise e

        if caller._pointNamesCreated():
            ret.points.setNames(caller.points.getNames(), useLog=False)
        if callee._featureNamesCreated():
            ret.features.setNames(callee.features.getNames(), useLog=False)

        dataHelpers.binaryOpNamePathMerge(caller, callee, ret, None, 'merge')

        return ret

    def __mul__(self, other):
        """
        Perform elementwise multiplication or scalar multiplication,
        depending in the input ``other``.
        """
        return self._genericArithmeticBinary('__mul__', other)

    def __rmul__(self, other):
        """
        Perform elementwise multiplication with this object on the right
        """
        return self._genericArithmeticBinary('__rmul__', other)

    def __imul__(self, other):
        """
        Perform in place elementwise multiplication or scalar
        multiplication, depending in the input ``other``.
        """
        return self._genericArithmeticBinary('__imul__', other)

    def __add__(self, other):
        """
        Perform addition on this object, element wise if 'other' is a
        nimble Base object, or element wise with a scalar if other is
        some kind of numeric value.
        """
        return self._genericArithmeticBinary('__add__', other)

    def __radd__(self, other):
        """
        Perform scalar addition with this object on the right
        """
        return self._genericArithmeticBinary('__radd__', other)

    def __iadd__(self, other):
        """
        Perform in-place addition on this object, element wise if
        ``other`` is a nimble Base object, or element wise with a scalar
        if ``other`` is some kind of numeric value.
        """
        return self._genericArithmeticBinary('__iadd__', other)

    def __sub__(self, other):
        """
        Subtract from this object, element wise if ``other`` is a nimble
        data object, or element wise by a scalar if ``other`` is some
        kind of numeric value.
        """
        return self._genericArithmeticBinary('__sub__', other)

    def __rsub__(self, other):
        """
        Subtract each element of this object from the given scalar.
        """
        return self._genericArithmeticBinary('__rsub__', other)

    def __isub__(self, other):
        """
        Subtract (in place) from this object, element wise if ``other``
        is a nimble Base object, or element wise with a scalar if
        ``other`` is some kind of numeric value.
        """
        return self._genericArithmeticBinary('__isub__', other)

    def __truediv__(self, other):
        """
        Perform true division using this object as the numerator,
        elementwise if ``other`` is a nimble Base object, or elementwise
        by a scalar if other is some kind of numeric value.
        """
        return self._genericArithmeticBinary('__truediv__', other)

    def __rtruediv__(self, other):
        """
        Perform element wise true division using this object as the
        denominator, and the given scalar value as the numerator.
        """
        return self._genericArithmeticBinary('__rtruediv__', other)

    def __itruediv__(self, other):
        """
        Perform true division (in place) using this object as the
        numerator, elementwise if ``other`` is a nimble Base object, or
        elementwise by a scalar if ``other`` is some kind of numeric
        value.
        """
        return self._genericArithmeticBinary('__itruediv__', other)

    def __floordiv__(self, other):
        """
        Perform floor division using this object as the numerator,
        elementwise if ``other`` is a nimble Base object, or elementwise
        by a scalar if ``other`` is some kind of numeric value.
        """
        return self._genericArithmeticBinary('__floordiv__', other)

    def __rfloordiv__(self, other):
        """
        Perform elementwise floor division using this object as the
        denominator, and the given scalar value as the numerator.

        """
        return self._genericArithmeticBinary('__rfloordiv__', other)

    def __ifloordiv__(self, other):
        """
        Perform floor division (in place) using this object as the
        numerator, elementwise if ``other`` is a nimble Base object, or
        elementwise by a scalar if ```other``` is some kind of numeric
        value.
        """
        return self._genericArithmeticBinary('__ifloordiv__', other)

    def __mod__(self, other):
        """
        Perform mod using the elements of this object as the dividends,
        elementwise if ``other`` is a nimble Base object, or elementwise
        by a scalar if other is some kind of numeric value.
        """
        return self._genericArithmeticBinary('__mod__', other)

    def __rmod__(self, other):
        """
        Perform mod using the elements of this object as the divisors,
        and the given scalar value as the dividend.
        """
        return self._genericArithmeticBinary('__rmod__', other)

    def __imod__(self, other):
        """
        Perform mod (in place) using the elements of this object as the
        dividends, elementwise if 'other' is a nimble Base object, or
        elementwise by a scalar if other is some kind of numeric value.
        """
        return self._genericArithmeticBinary('__imod__', other)

    @to2args
    def __pow__(self, other, z):
        """
        Perform exponentiation (iterated __mul__) using the elements of
        this object as the bases, elementwise if ``other`` is a nimble
        data object, or elementwise by a scalar if ``other`` is some
        kind of numeric value.
        """
        return self._genericArithmeticBinary('__pow__', other)

    def __rpow__(self, other):
        """
        Perform elementwise exponentiation (iterated __mul__) using the
        ``other`` scalar value as the bases.
        """
        return self._genericArithmeticBinary('__rpow__', other)

    def __ipow__(self, other):
        """
        Perform in-place exponentiation (iterated __mul__) using the
        elements of this object as the bases, element wise if ``other``
        is a nimble Base object, or elementwise by a scalar if ``other``
        is some kind of numeric value.
        """
        return self._genericArithmeticBinary('__ipow__', other)

    def __pos__(self):
        """
        Return this object.
        """
        ret = self.copy()
        ret._name = dataHelpers.nextDefaultObjectName()

        return ret

    def __neg__(self):
        """
        Return this object where every element has been multiplied by -1
        """
        ret = self.copy()
        ret *= -1
        ret._name = dataHelpers.nextDefaultObjectName()

        return ret

    def __abs__(self):
        """
        Perform element wise absolute value on this object
        """
        ret = self.elements.calculate(abs, useLog=False)
        if self._pointNamesCreated():
            ret.points.setNames(self.points.getNames(), useLog=False)
        else:
            ret.points.setNames(None, useLog=False)
        if self._featureNamesCreated():
            ret.features.setNames(self.features.getNames(), useLog=False)
        else:
            ret.points.setNames(None, useLog=False)

        ret._name = dataHelpers.nextDefaultObjectName()
        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath
        return ret

    def _genericArithmeticBinary_sizeValidation(self, opName, other):
        if self._pointCount != len(other.points):
            msg = "The number of points in each object must be equal. "
            msg += "(self=" + str(self._pointCount) + " vs other="
            msg += str(len(other.points)) + ")"
            raise InvalidArgumentValue(msg)
        if self._featureCount != len(other.features):
            msg = "The number of features in each object must be equal."
            raise InvalidArgumentValue(msg)

        if self._pointCount == 0 or self._featureCount == 0:
            msg = "Cannot do " + opName + " when points or features is empty"
            raise ImproperObjectAction(msg)

    def _genericArithmeticBinary_validation(self, opName, other):
        otherNimble = isinstance(other, Base)
        if not otherNimble and not dataHelpers._looksNumeric(other):
            msg = "'other' must be an instance of a nimble Base object or a "
            msg += "scalar"
            raise InvalidArgumentType(msg)
        if otherNimble:
            self._genericArithmeticBinary_sizeValidation(opName, other)
            self._validateEqualNames('point', 'point', opName, other)
            self._validateEqualNames('feature', 'feature', opName, other)

        dataHelpers.arithmeticValidation(self, opName, other)

    def _genericArithmeticBinary(self, opName, other):
        isStretch = isinstance(other, nimble.data.stretch.Stretch)
        if isStretch:
            return NotImplemented
        self._genericArithmeticBinary_validation(opName, other)
        # figure out return obj's point / feature names
        otherNimble = isinstance(other, Base)
        if otherNimble:
            # everything else that uses this helper is a binary scalar op
            retPNames, retFNames = dataHelpers.mergeNonDefaultNames(self,
                                                                    other)
        else:
            retPNames = self.points._getNamesNoGeneration()
            retFNames = self.features._getNamesNoGeneration()

        ret = self._arithmeticBinary_implementation(opName, other)

        if opName.startswith('__i'):
            absPath, relPath = self._absPath, self._relPath
            self.referenceDataFrom(ret, useLog=False)
            self._absPath, self._relPath = absPath, relPath
            ret = self
        ret.points.setNames(retPNames, useLog=False)
        ret.features.setNames(retFNames, useLog=False)

        nameSource = 'self' if opName.startswith('__i') else None
        pathSource = 'merge' if otherNimble else 'self'
        dataHelpers.binaryOpNamePathMerge(
            self, other, ret, nameSource, pathSource)
        return ret


    def _defaultArithmeticBinary_implementation(self, opName, other):
        selfData = self.copy('numpyarray')
        if isinstance(other, Base):
            otherData = other.copy('numpyarray')
        else:
            otherData = other
        ret = getattr(selfData, opName)(otherData)
        ret = createDataNoValidation(self.getTypeString(), ret)

        return ret


    ############################
    ############################
    ###   Helper functions   ###
    ############################
    ############################

    def _arrangeFinalTable(self, pnames, pnamesWidth, dataTable, dataWidths,
                           fnames, pnameSep):

        if fnames is not None:
            fnamesWidth = list(map(len, fnames))
        else:
            fnamesWidth = []

        # We make extensive use of list addition in this helper in order
        # to prepend single values onto lists.

        # glue point names onto the left of the data
        if pnames is not None:
            for i in range(len(dataTable)):
                dataTable[i] = [pnames[i], pnameSep] + dataTable[i]
            dataWidths = [pnamesWidth, len(pnameSep)] + dataWidths

        # glue feature names onto the top of the data
        if fnames is not None:
            # adjust with the empty space in the upper left corner, if needed
            if pnames is not None:
                fnames = ["", ""] + fnames
                fnamesWidth = [0, 0] + fnamesWidth

            # make gap row:
            gapRow = [""] * len(fnames)

            dataTable = [fnames, gapRow] + dataTable
            # finalize widths by taking the largest of the two possibilities
            for i in range(len(fnames)):
                nameWidth = fnamesWidth[i]
                valWidth = dataWidths[i]
                dataWidths[i] = max(nameWidth, valWidth)

        return dataTable, dataWidths

    def _arrangePointNames(self, maxRows, nameLength, rowHolder, nameHold):
        """
        Prepare point names for string output. Grab only those names
        that fit according to the given row limitation, process them for
        length, omit them if they are default. Returns a list of
        prepared names, and a int bounding the length of each name
        representation.
        """
        names = []
        pnamesWidth = 0
        nameCutIndex = nameLength - len(nameHold)
        (tRowIDs, bRowIDs) = dataHelpers.indicesSplit(maxRows,
                                                      self._pointCount)

        # we pull indices from two lists: tRowIDs and bRowIDs
        for sourceIndex in range(2):
            source = list([tRowIDs, bRowIDs])[sourceIndex]

            # add in the rowHolder, if needed
            if (sourceIndex == 1
                    and len(bRowIDs) + len(tRowIDs) < self._pointCount):
                names.append(rowHolder)

            for i in source:
                pname = self.points.getName(i)
                # omit default valued names
                if pname[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
                    pname = ""

                # truncate names which extend past the given length
                if len(pname) > nameLength:
                    pname = pname[:nameCutIndex] + nameHold

                names.append(pname)

                # keep track of bound.
                if len(pname) > pnamesWidth:
                    pnamesWidth = len(pname)

        return names, pnamesWidth

    def _arrangeDataWithLimits(self, maxWidth, maxHeight, includeFNames=False,
                               sigDigits=3, maxStrLength=19, colSep=' ',
                               colHold='--', rowHold='|', strHold='...'):
        """
        Arrange the data in this object into a table structure, while
        respecting the given boundaries. If there is more data than
        what fits within the limitations, then omit points or features
        from the middle portions of the data.

        Returns three values. The first is list of list of strings representing
        the data values we have space for. The length of the outer list
        is less than or equal to maxHeight. The length of the inner lists
        will all be the same, a length we will designate as n. The sum of
        the individual strings in each inner list will be less than or
        equal to maxWidth - ((n-1) * len(colSep)). The second returned
        value will be a list, where the ith value is of the maximum width taken
        up by the strings or feature name in the ith column of the data. The
        third returned value is a list of feature names matching the columns
        represented in the first return value.
        """
        if self._pointCount == 0 or self._featureCount == 0:
            fNames = None
            if includeFNames:
                fNames = []
            return [[]], [], fNames

        if maxHeight < 2 and maxHeight != self._pointCount:
            msg = "If the number of points in this object is two or greater, "
            msg += "then we require that the input argument maxHeight also "
            msg += "be greater than or equal to two."
            raise InvalidArgumentValue(msg)

        cHoldWidth = len(colHold)
        cHoldTotal = len(colSep) + cHoldWidth
        nameCutIndex = maxStrLength - len(strHold)

        #setup a bundle of default values
        if maxHeight is None:
            maxHeight = self._pointCount
        if maxWidth is None:
            maxWidth = float('inf')

        maxRows = min(maxHeight, self._pointCount)
        maxDataRows = maxRows

        (tRowIDs, bRowIDs) = dataHelpers.indicesSplit(maxDataRows,
                                                      self._pointCount)
        combinedRowIDs = tRowIDs + bRowIDs
        if len(combinedRowIDs) < self._pointCount:
            rowHolderIndex = len(tRowIDs)
        else:
            rowHolderIndex = sys.maxsize

        lTable, rTable = [], []
        lColWidths, rColWidths = [], []
        lFNames, rFNames = [], []
        # total width will always include the column placeholder column,
        # until it is shown that it isn't needed
        totalWidth = cHoldTotal

        # going to add indices from the beginning and end of the data until
        # we've used up our available space, or we've gone through all of
        # the columns. currIndex makes use of negative indices, which is
        # why the end condition makes use of an exact stop value, which
        # varies between positive and negative depending on the number of
        # features
        endIndex = self._featureCount // 2
        if self._featureCount % 2 == 1:
            endIndex *= -1
            endIndex -= 1
        currIndex = 0
        numAdded = 0

        while totalWidth < maxWidth and currIndex != endIndex:
            currTable = lTable if currIndex >= 0 else rTable
            currCol = []
            currWidth = 0
            if includeFNames:
                currFName = self.features.getName(currIndex)
                fNameLen = len(currFName)
                if fNameLen > maxStrLength:
                    currFName = currFName[:nameCutIndex] + strHold
                    fNameLen = maxStrLength
                currWidth = fNameLen

            # check all values in this column (in the accepted rows)
            for i in range(len(combinedRowIDs)):
                rID = combinedRowIDs[i]
                val = self[rID, currIndex]
                valFormed = formatIfNeeded(val, sigDigits)
                if len(valFormed) <= maxStrLength:
                    valLimited = valFormed
                else:
                    valLimited = valFormed[:nameCutIndex] + strHold
                valLen = len(valLimited)
                if valLen > currWidth:
                    currWidth = valLen

                # If these are equal, it is time to add the holders
                if i == rowHolderIndex:
                    currCol.append(rowHold)

                currCol.append(valLimited)

            totalWidth += currWidth
            # only add this column if it won't put us over the limit
            if totalWidth <= maxWidth:
                numAdded += 1
                for i in range(len(currCol)):
                    if len(currTable) != len(currCol):
                        currTable.append([currCol[i]])
                    else:
                        currTable[i].append(currCol[i])
                # the width value goes in different lists depending on index
                if currIndex < 0:
                    if includeFNames:
                        rFNames.append(currFName)
                    currIndex = abs(currIndex)
                    rColWidths.append(currWidth)
                else:
                    if includeFNames:
                        lFNames.append(currFName)
                    currIndex = (-1 * currIndex) - 1
                    lColWidths.append(currWidth)

            # ignore column separator if the next column is the last
            if numAdded == (self._featureCount - 1):
                totalWidth -= cHoldTotal
            totalWidth += len(colSep)

        # combine the tables. Have to reverse rTable because entries were
        # appended in a right to left order
        rColWidths.reverse()
        rFNames.reverse()
        fNames = lFNames + rFNames
        if numAdded == self._featureCount:
            lColWidths += rColWidths
        else:
            lColWidths += [cHoldWidth] + rColWidths
            if includeFNames:
                fNames = lFNames + [colHold] + rFNames
        # return None if fNames is [] (includeFNames=False)
        fNames = fNames if fNames else None
        for rowIndex in range(len(lTable)):
            if len(rTable) > 0:
                rTable[rowIndex].reverse()
                toAdd = rTable[rowIndex]
            else:
                toAdd = []

            if numAdded == self._featureCount:
                lTable[rowIndex] += toAdd
            else:
                lTable[rowIndex] += [colHold] + toAdd

        return lTable, lColWidths, fNames

    def _defaultNamesGeneration_NamesSetOperations(self, other, axis):
        """
        TODO: Find a shorter descriptive name.
        TODO: Should we place this function in dataHelpers.py?
        """
        if axis == 'point':
            if self.pointNames is None:
                self.points._setAllDefault()
            if other.pointNames is None:
                other.points._setAllDefault()
        elif axis == 'feature':
            if self.featureNames is None:
                self.features._setAllDefault()
            if other.featureNames is None:
                other.features._setAllDefault()
        else:
            raise InvalidArgumentValue("invalid axis")

    def _pointNameDifference(self, other):
        """
        Returns a set containing those pointNames in this object that
        are not also in the input object.
        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "pointName difference"
            raise InvalidArgumentType(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) - six.viewkeys(other.pointNames)

    def _featureNameDifference(self, other):
        """
        Returns a set containing those featureNames in this object that
        are not also in the input object.
        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "featureName difference"
            raise InvalidArgumentType(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return (six.viewkeys(self.featureNames)
                - six.viewkeys(other.featureNames))

    def _pointNameIntersection(self, other):
        """
        Returns a set containing only those pointNames that are shared
        by this object and the input object.
        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "pointName intersection"
            raise InvalidArgumentType(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) & six.viewkeys(other.pointNames)

    def _featureNameIntersection(self, other):
        """
        Returns a set containing only those featureNames that are shared
        by this object and the input object.
        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "featureName intersection"
            raise InvalidArgumentType(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return (six.viewkeys(self.featureNames)
                & six.viewkeys(other.featureNames))

    def _pointNameSymmetricDifference(self, other):
        """
        Returns a set containing only those pointNames not shared
        between this object and the input object.
        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "pointName difference"
            raise InvalidArgumentType(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) ^ six.viewkeys(other.pointNames)

    def _featureNameSymmetricDifference(self, other):
        """
        Returns a set containing only those featureNames not shared
        between this object and the input object.
        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "featureName difference"
            raise InvalidArgumentType(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return (six.viewkeys(self.featureNames)
                ^ six.viewkeys(other.featureNames))

    def _pointNameUnion(self, other):
        """
        Returns a set containing all pointNames in either this object or
        the input object.
        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "pointNames union"
            raise InvalidArgumentType(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) | six.viewkeys(other.pointNames)

    def _featureNameUnion(self, other):
        """
        Returns a set containing all featureNames in either this object
        or the input object.
        """
        if other is None:
            raise InvalidArgumentType("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "featureName union"
            raise InvalidArgumentType(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return (six.viewkeys(self.featureNames)
                | six.viewkeys(other.featureNames))

    def _equalPointNames(self, other):
        if other is None or not isinstance(other, Base):
            return False
        return self._equalNames(self.points._getNamesNoGeneration(),
                                other.points._getNamesNoGeneration())

    def _equalFeatureNames(self, other):
        if other is None or not isinstance(other, Base):
            return False
        return (self._equalNames(self.features._getNamesNoGeneration(),
                                 other.features._getNamesNoGeneration()))

    def _equalNames(self, selfNames, otherNames):
        """
        Private function to determine equality of either pointNames of
        featureNames. It ignores equality of default values, considering
        only whether non default names consistent (position by position)
        and uniquely positioned (if a non default name is present in
        both, then it is in the same position in both).
        """
        if selfNames is None and otherNames is None:
            return True
        if (selfNames is None
                and all(n.startswith(DEFAULT_PREFIX) for n in otherNames)):
            return True
        if (otherNames is None
                and all(n.startswith(DEFAULT_PREFIX) for n in selfNames)):
            return True
        if selfNames is None or otherNames is None:
            return False
        if len(selfNames) != len(otherNames):
            return False

        unequalNames = self._unequalNames(selfNames, otherNames)
        return unequalNames == {}

    def _validateEqualNames(self, leftAxis, rightAxis, callSym, other):

        def _validateEqualNames_implementation():
            if leftAxis == 'point':
                lnames = self.points.getNames()
            else:
                lnames = self.features.getNames()
            if rightAxis == 'point':
                rnames = other.points.getNames()
            else:
                rnames = other.features.getNames()
            inconsistencies = self._inconsistentNames(lnames, rnames)

            if inconsistencies != {}:
                table = [['left', 'ID', 'right']]
                for i in sorted(inconsistencies.keys()):
                    lname = '"' + lnames[i] + '"'
                    rname = '"' + rnames[i] + '"'
                    table.append([lname, str(i), rname])

                msg = leftAxis + " to " + rightAxis + " name inconsistencies "
                msg += "when calling left." + callSym + "(right) \n"
                msg += nimble.logger.tableString.tableString(table)
                print(msg, file=sys.stderr)
                raise InvalidArgumentValue(msg)

        if leftAxis == 'point' and rightAxis == 'point':
            if self._pointNamesCreated() or other._pointNamesCreated():
                _validateEqualNames_implementation()
        elif leftAxis == 'feature' and rightAxis == 'feature':
            if self._featureNamesCreated() or other._featureNamesCreated():
                _validateEqualNames_implementation()
        elif leftAxis == 'point' and rightAxis == 'feature':
            if self._pointNamesCreated() or other._featureNamesCreated():
                _validateEqualNames_implementation()
        elif leftAxis == 'feature' and rightAxis == 'point':
            if self._featureNamesCreated() or other._pointNamesCreated():
                _validateEqualNames_implementation()

    def _inconsistentNames(self, selfNames, otherNames):
        """Private function to find and return all name inconsistencies
        between the given two sets. It ignores equality of default
        values, considering only whether non default names consistent
        (position by position) and uniquely positioned (if a non default
        name is present in both, then it is in the same position in
        both). The return value is a dict between integer IDs and the
        pair of offending names at that position in both objects.

        Assumptions: the size of the two name sets is equal.
        """
        inconsistencies = {}

        def checkFromLeftKeys(ret, leftNames, rightNames):
            for index in range(len(leftNames)):
                lname = leftNames[index]
                rname = rightNames[index]
                if lname[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                    if rname[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                        if lname != rname:
                            ret[index] = (lname, rname)
                    else:
                        # if a name in one is mirrored by a default name,
                        # then it must not appear in any other index;
                        # and therefore, must not appear at all.
                        if rightNames.count(lname) > 0:
                            ret[index] = (lname, rname)
                            ret[rightNames.index(lname)] = (lname, rname)


        # check both name directions
        checkFromLeftKeys(inconsistencies, selfNames, otherNames)
        checkFromLeftKeys(inconsistencies, otherNames, selfNames)

        return inconsistencies

    def _unequalNames(self, selfNames, otherNames):
        """Private function to find and return all name inconsistencies
        between the given two sets. It ignores equality of default
        values, considering only whether non default names consistent
        (position by position) and uniquely positioned (if a non default
        name is present in both, then it is in the same position in
        both). The return value is a dict between integer IDs and the
        pair of offending names at that position in both objects.

        Assumptions: the size of the two name sets is equal.
        """
        inconsistencies = {}

        def checkFromLeftKeys(ret, leftNames, rightNames):
            for index in range(len(leftNames)):
                lname = leftNames[index]
                rname = rightNames[index]
                if lname[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                    if rname[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                        if lname != rname:
                            ret[index] = (lname, rname)
                    else:
                        ret[index] = (lname, rname)

        # check both name directions
        checkFromLeftKeys(inconsistencies, selfNames, otherNames)
        checkFromLeftKeys(inconsistencies, otherNames, selfNames)

        return inconsistencies

    def _getAxis(self, axis):
        if axis == 'point':
            return self.points
        else:
            return self.features

    def _validateAxis(self, axis):
        if axis != 'point' and axis != 'feature':
            msg = 'axis parameter may only be "point" or "feature"'
            raise InvalidArgumentValue(msg)

    def _incrementDefaultIfNeeded(self, name, axis):
        self._validateAxis(axis)
        if name[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
            intString = name[DEFAULT_PREFIX_LENGTH:]
            try:
                nameNum = int(intString)
            # Case: default prefix with non-integer suffix. This cannot
            # cause a future integer suffix naming collision, so we
            # return without making any chagnes.
            except ValueError:
                return
            if axis == 'point':
                if nameNum >= self._nextDefaultValuePoint:
                    self._nextDefaultValuePoint = nameNum + 1
            else:
                if nameNum >= self._nextDefaultValueFeature:
                    self._nextDefaultValueFeature = nameNum + 1

    def _validateMatPlotLibImport(self, error, name):
        if error is not None:
            msg = "The module matplotlib is required to be installed "
            msg += "in order to call the " + name + "() method. "
            msg += "However, when trying to import, an ImportError with "
            msg += "the following message was raised: '"
            msg += str(error) + "'"

            raise ImportError(msg)

    def _validateRangeOrder(self, startName, startVal, endName, endVal):
        """
        Validate a range where both values are inclusive.
        """
        if startVal > endVal:
            msg = "When specifying a range, the arguments were resolved to "
            msg += "having the values " + startName
            msg += "=" + str(startVal) + " and " + endName + "=" + str(endVal)
            msg += ", yet the starting value is not allowed to be greater "
            msg += "than the ending value (" + str(startVal) + ">"
            msg += str(endVal) + ")"

            raise InvalidArgumentValueCombination(msg)

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _isIdentical_implementation(self, other):
        pass

    @abstractmethod
    def _writeFile_implementation(self, outPath, fileFormat, includePointNames,
                                  includeFeatureNames):
        pass

    @abstractmethod
    def _getTypeString_implementation(self):
        pass

    @abstractmethod
    def _getitem_implementation(self, x, y):
        pass

    @abstractmethod
    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd):
        pass

    @abstractmethod
    def _validate_implementation(self, level):
        pass

    @abstractmethod
    def _containsZero_implementation(self):
        pass

    @abstractmethod
    def _transpose_implementation(self):
        pass

    @abstractmethod
    def _referenceDataFrom_implementation(self, other):
        pass

    @abstractmethod
    def _copy_implementation(self, to):
        pass

    @abstractmethod
    def _fillWith_implementation(self, values, pointStart, featureStart,
                                 pointEnd, featureEnd):
        pass

    @abstractmethod
    def _flattenToOnePoint_implementation(self):
        pass

    @abstractmethod
    def _flattenToOneFeature_implementation(self):
        pass

    @abstractmethod
    def _unflattenFromOnePoint_implementation(self, numPoints):
        pass

    @abstractmethod
    def _unflattenFromOneFeature_implementation(self, numFeatures):
        pass

    @abstractmethod
    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):
        pass

    @abstractmethod
    def _mul__implementation(self, other):
        pass

class BasePoints(Axis, Points):
    """
    Access for point-based methods.
    """
    pass

class BaseFeatures(Axis, Features):
    """
    Access for feature-based methods.
    """
    pass

class BaseElements(Elements):
    """
    Access for element-based methods.
    """
    pass

def cmp(x, y):
    """
    Comparison function.
    """
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0
