"""
Anchors the hierarchy of data representation types, providing stubs and common functions.

"""

# TODO conversions
# TODO who sorts inputs to derived implementations?

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import six
from six.moves import map
from six.moves import range
from six.moves import zip
import sys
import warnings

import __main__ as main
mplError = None
try:
    import matplotlib
    # for .show() to work in interactive sessions
    # a backend different than Agg needs to be use
    # The interactive session can choose by default e.g.,
    # in jupyter-notebook inline is the default.
    if hasattr(main, '__file__'):
        # It must be agg  for non-interactive sessions
        # otherwise the combination of matplotlib and multiprocessing
        # produces a segfault.
        # Open matplotlib issue here: https://github.com/matplotlib/matplotlib/issues/8795
        # It applies for both for python 2 and 3
        matplotlib.use('Agg')

except ImportError as e:
    mplError = e

#print('matplotlib backend: {}'.format(matplotlib.get_backend()))

import math
import numbers
import itertools
import copy
import numpy
import os.path
import inspect
import operator
from multiprocessing import Process

import UML

pd = UML.importModule('pandas')

cython = UML.importModule('cython')
if cython is None or not cython.compiled:
    from math import sin, cos

from UML.exceptions import ArgumentException, PackageException
from UML.exceptions import ImproperActionException
from UML.logger import produceFeaturewiseReport
from UML.logger import produceAggregateReport
from UML.randomness import pythonRandom

from . import dataHelpers

# the prefix for default point and feature names
from .dataHelpers import DEFAULT_PREFIX, DEFAULT_PREFIX2, DEFAULT_PREFIX_LENGTH

from .dataHelpers import DEFAULT_NAME_PREFIX

from .dataHelpers import formatIfNeeded

from .dataHelpers import makeConsistentFNamesAndData

def to2args(f):
    """
    this function is for __pow__. In cython, __pow__ must have 3 arguments and default can't be used there.
    so this function is used to convert a function with 3 arguments to a function with 2 arguments when it is used
    in python environment.
    """
    def tmpF(x, y):
        return f(x, y, None)
    return tmpF

def hashCodeFunc(elementValue, pointNum, featureNum):
    return ((sin(pointNum) + cos(featureNum)) / 2.0) * elementValue

class Base(object):
    """
    Class defining important data manipulation operations and giving functionality
    for the naming the features of that data. A mapping from feature names to feature
    indices is given by the featureNames attribute, the inverse of that mapping is
    given by featureNamesInverse.

    """

    def __init__(self, shape, pointNames=None, featureNames=None, name=None,
                 paths=(None, None), **kwds):
        """
        Instantiates the book-keeping structures that are taken to be common
        across all data types. Specifically, this includes point and feature
        names, an object name, and originating pathes for the data in this
        object. Note: this method (as should all other __init__ methods in
        this hierarchy) makes use of super()

        pointNames: may be a list or dict mapping names to indices. None is
        given if default names are desired.

        featureNames: may be a list or dict mapping names to indices. None is
        given if default names are desired.

        name: the name to be associated with this object.

        paths: a tuple, where the first entry is taken to be the string
        representing the absolute path to the source file of the data and
        the second entry is taken to be the relative path. Both may be
        None if these values are to be unspecified.

        **kwds: potentially full of arguments further up the class hierarchy,
        as following best practices for use of super(). Note however, that
        this class is the root of the object hierarchy as statically defined.

        """
        self._pointCount = shape[0]
        self._featureCount = shape[1]
        if pointNames is not None and len(pointNames) != shape[0]:
            msg = "The length of the pointNames (" + str(len(pointNames))
            msg += ") must match the points given in shape (" + str(shape[0])
            msg += ")"
            raise ArgumentException(msg)
        if featureNames is not None and len(featureNames) != shape[1]:
            msg = "The length of the featureNames (" + str(len(featureNames))
            msg += ") must match the features given in shape ("
            msg += str(shape[1]) + ")"
            raise ArgumentException(msg)

        # Set up point names
        self._nextDefaultValuePoint = 0
        if pointNames is None:
            self.pointNamesInverse = None
            self.pointNames = None
        elif isinstance(pointNames, list):
            self._nextDefaultValuePoint = self._pointCount
            self.setPointNames(pointNames)
        elif isinstance(pointNames, dict):
            self._nextDefaultValuePoint = self._pointCount
            self.setPointNames(pointNames)
        # could still be an ordered container, pass it on to the list helper
        elif hasattr(pointNames, '__len__') and hasattr(pointNames, '__getitem__'):
            self._nextDefaultValuePoint = self._pointCount
            self.setPointNames(pointNames)
        else:
            raise ArgumentException(
                "pointNames may only be a list, an ordered container, or a dict, defining a mapping between integers and pointNames")

        # Set up feature names
        self._nextDefaultValueFeature = 0
        if featureNames is None:
            self.featureNamesInverse = None
            self.featureNames = None
        elif isinstance(featureNames, list):
            self._nextDefaultValueFeature = self._featureCount
            self.setFeatureNames(featureNames)
        elif isinstance(featureNames, dict):
            self._nextDefaultValueFeature = self._featureCount
            self.setFeatureNames(featureNames)
        # could still be an ordered container, pass it on to the list helper
        elif hasattr(featureNames, '__len__') and hasattr(featureNames, '__getitem__'):
            self._nextDefaultValueFeature = self._featureCount
            self.setFeatureNames(featureNames)
        else:
            raise ArgumentException(
                "featureNames may only be a list, an ordered container, or a dict, defining a mapping between integers and featureNames")

        # Set up object name
        if name is None:
            self._name = dataHelpers.nextDefaultObjectName()
        else:
            self._name = name

        # Set up paths
        if paths[0] is not None and not isinstance(paths[0], six.string_types):
            raise ArgumentException(
                "paths[0] must be None or an absolute path to the file from which the data originates")
        if paths[0] is not None and not os.path.isabs(paths[0]) and not paths[0].startswith('http'):
            raise ArgumentException("paths[0] must be an absolute path")
        self._absPath = paths[0]

        if paths[1] is not None and not isinstance(paths[1], six.string_types):
            raise ArgumentException(
                "paths[1] must be None or a relative path to the file from which the data originates")
        self._relPath = paths[1]

        # call for safety
        super(Base, self).__init__(**kwds)


    #######################
    # Property Attributes #
    #######################

    def _getpointCount(self):
        return self._pointCount

    points = property(_getpointCount, doc="The number of points in this object")

    def _getfeatureCount(self):
        return self._featureCount

    features = property(_getfeatureCount, doc="The number of features in this object")

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
                msg = "The name of an object may only be a string, or the value None"
                raise ValueError(msg)
            self._name = value

    name = property(_getObjName, _setObjName, doc="A name to be displayed when printing or logging this object")

    def _getAbsPath(self):
        return self._absPath

    absolutePath = property(_getAbsPath, doc="The path to the file this data originated from, in absolute form")

    def _getRelPath(self):
        return self._relPath

    relativePath = property(_getRelPath, doc="The path to the file this data originated from, in relative form")

    def _getPath(self):
        return self.absolutePath

    path = property(_getPath, doc="The path to the file this data originated from")

    def _pointNamesCreated(self):
        """
        Returns True if point default names have been created/assigned
        to the object.
        If the object does not have points it returns True.
        """
        if self.pointNamesInverse is None:
            return False
        else:
            return True

    def _featureNamesCreated(self):
        """
        Returns True if feature default names have been created/assigned
        to the object.
        If the object does not have features it returns True.
        """
        if self.featureNamesInverse is None:
            return False
        else:
            return True

    ########################
    # Low Level Operations #
    ########################

    def __len__(self):
        # ordered such that the larger axis is always printed, even
        # if they are both in the range [0,1]
        if self.points == 0 or self.features == 0:
            return 0
        if self.points == 1:
            return self.features
        if self.features == 1:
            return self.points

        msg = "len() is undefined when the number of points ("
        msg += str(self.points)
        msg += ") and the number of features ("
        msg += str(self.features)
        msg += ") are both greater than 1"
        raise ImproperActionException(msg)


    def setPointName(self, oldIdentifier, newName):
        """
        Changes the pointName specified by previous to the supplied input name.

        oldIdentifier must be a non None string or integer, specifying either a current pointName
        or the index of a current pointName. newName may be either a string not currently
        in the pointName set, or None for an default pointName. newName cannot begin with the
        default prefix.

        None is always returned.

        """
        if self.points == 0:
            raise ArgumentException("Cannot set any point names; this object has no points ")
        if self.pointNames is None:
            self._setAllDefault('point')
        self._setName_implementation(oldIdentifier, newName, 'point', False)

    def setFeatureName(self, oldIdentifier, newName):
        """
        Changes the featureName specified by previous to the supplied input name.

        oldIdentifier must be a non None string or integer, specifying either a current featureName
        or the index of a current featureName. newName may be either a string not currently
        in the featureName set, or None for an default featureName. newName cannot begin with the
        default prefix.

        None is always returned.

        """
        if self.features == 0:
            raise ArgumentException("Cannot set any feature names; this object has no features ")
        if self.featureNames is None:
            self._setAllDefault('feature')
        self._setName_implementation(oldIdentifier, newName, 'feature', False)


    def setPointNames(self, assignments=None):
        """
        Rename all of the point names of this object according to the values
        specified by the assignments parameter. If given a list, then we use
        the mapping between names and array indices to define the point
        names. If given a dict, then that mapping will be used to define the
        point names. If assignments is None, then all point names will be
        given new default values. If assignment is an unexpected type, the names
        are not strings, the names are not unique, or point indices are missing,
        then an ArgumentException will be raised. None is always returned.

        """
        if assignments is None:
            self.pointNames = None
            self.pointNamesInverse = None
        elif isinstance(assignments, list):
            self._setNamesFromList(assignments, self.points, 'point')
        elif isinstance(assignments, dict):
            self._setNamesFromDict(assignments, self.points, 'point')
        else:
            msg = "'assignments' parameter may only be a list, a dict, or None, "
            msg += "yet a value of type " + \
                str(type(assignments)) + " was given"
            raise ArgumentException(msg)

    def setFeatureNames(self, assignments=None):
        """
        Rename all of the feature names of this object according to the values
        specified by the assignments parameter. If given a list, then we use
        the mapping between names and array indices to define the feature
        names. If given a dict, then that mapping will be used to define the
        feature names. If assignments is None, then all feature names will be
        given new default values. If assignment is an unexpected type, the names
        are not strings, the names are not unique, or feature indices are missing,
        then an ArgumentException will be raised. None is always returned.

        """
        if assignments is None:
            self.featureNames = None
            self.featureNamesInverse = None
        elif isinstance(assignments, list):
            self._setNamesFromList(assignments, self.features, 'feature')
        elif isinstance(assignments, dict):
            self._setNamesFromDict(assignments, self.features, 'feature')
        else:
            msg = "'assignments' parameter may only be a list, a dict, or None, "
            msg += "yet a value of type " + \
                str(type(assignments)) + " was given"
            raise ArgumentException(msg)

    def nameIsDefault(self):
        """Returns True if self.name has a default value"""
        return self.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)

    def getPointNames(self):
        """Returns a list containing all point names, where their index
        in the list is the same as the index of the point they correspond
        to.

        """
        if not self._pointNamesCreated():
            self._setAllDefault('point')
        return copy.copy(self.pointNamesInverse)

    def getFeatureNames(self):
        """Returns a list containing all feature names, where their index
        in the list is the same as the index of the feature they
        correspond to.

        """
        if not self._featureNamesCreated():
            self._setAllDefault('feature')
        return copy.copy(self.featureNamesInverse)

    def getPointName(self, index):
        if not self._pointNamesCreated():
            self._setAllDefault('point')
        return self.pointNamesInverse[index]

    def getPointIndex(self, name):
        if not self._pointNamesCreated():
            self._setAllDefault('point')
        return self.pointNames[name]

    def getPointIndices(self, names):
        if not self._pointNamesCreated():
            self._setAllDefault('point')
        return [self.pointNames[n] for n in names]

    def hasPointName(self, name):
        try:
            self.getPointIndex(name)
            return True
        except KeyError:
            return False

    def getFeatureName(self, index):
        if not self._featureNamesCreated():
            self._setAllDefault('feature')
        return self.featureNamesInverse[index]

    def getFeatureIndex(self, name):
        if not self._featureNamesCreated():
            self._setAllDefault('feature')
        return self.featureNames[name]

    def getFeatureIndices(self, names):
        if not self._featureNamesCreated():
            self._setAllDefault('feature')
        return [self.featureNames[n] for n in names]

    def hasFeatureName(self, name):
        try:
            self.getFeatureIndex(name)
            return True
        except KeyError:
            return False

    ###########################
    # Higher Order Operations #
    ###########################

    def dropFeaturesContainingType(self, typeToDrop):
        """
        Modify this object so that it no longer contains features which have the specified
        type as values. None is always returned.

        """
        if not isinstance(typeToDrop, (list, tuple)):
            if not isinstance(typeToDrop, type):
                raise ArgumentException(
                    "The only allowed inputs are a list of types or a single type, yet the input is neither a list or a type")
            typeToDrop = [typeToDrop]
        else:
            for value in typeToDrop:
                if not isinstance(value, type):
                    raise ArgumentException("When giving a list as input, every contained value must be a type")

        if self.points == 0 or self.features == 0:
            return

        def hasType(feature):
            for value in feature:
                for typeValue in typeToDrop:
                    if isinstance(value, typeValue):
                        return True
            return False

        removed = self.extractFeatures(hasType)
        return


    def replaceFeatureWithBinaryFeatures(self, featureToReplace):
        """
        Modify this object so that the chosen feature is removed, and binary valued
        features are added, one for each possible value seen in the original feature.
        None is always returned.

        """
        if self.points == 0:
            raise ImproperActionException("This action is impossible, the object has 0 points")

        index = self._getFeatureIndex(featureToReplace)
        # extract col.
        toConvert = self.extractFeatures([index])

        # MR to get list of values
        def getValue(point):
            return [(point[0], 1)]

        def simpleReducer(identifier, valuesList):
            return (identifier, 0)

        values = toConvert.mapReducePoints(getValue, simpleReducer)
        values.setFeatureName(0, 'values')
        values = values.extractFeatures([0])

        # Convert to List, so we can have easy access
        values = values.copyAs(format="List")

        # for each value run calculateForEachPoint to produce a category
        # point for each value
        def makeFunc(value):
            def equalTo(point):
                if point[0] == value:
                    return 1
                return 0

            return equalTo

        varName = toConvert.getFeatureName(0)

        for point in values.data:
            value = point[0]
            ret = toConvert.calculateForEachPoint(makeFunc(value))
            ret.setFeatureName(0, varName + "=" + str(value).strip())
            toConvert.appendFeatures(ret)

        # remove the original feature, and combine with self
        toConvert.extractFeatures([varName])
        self.appendFeatures(toConvert)


    def transformFeatureToIntegers(self, featureToConvert):
        """
        Modify this object so that the chosen feature in removed, and a new integer
        valued feature is added with values 0 to n-1, one for each of n values present
        in the original feature. None is always returned.

        """
        if self.points == 0:
            raise ImproperActionException("This action is impossible, the object has 0 points")

        index = self._getFeatureIndex(featureToConvert)

        # extract col.
        toConvert = self.extractFeatures([index])

        # MR to get list of values
        def getValue(point):
            return [(point[0], 1)]

        def simpleReducer(identifier, valuesList):
            return (identifier, 0)

        values = toConvert.mapReducePoints(getValue, simpleReducer)
        values.setFeatureName(0, 'values')
        values = values.extractFeatures([0])

        # Convert to List, so we can have easy access
        values = values.copyAs(format="List")

        mapping = {}
        index = 0
        for point in values.data:
            if point[0] not in mapping:
                mapping[point[0]] = index
                index = index + 1

        def lookup(point):
            return mapping[point[0]]

        converted = toConvert.calculateForEachPoint(lookup)
        converted.setPointNames(toConvert.getPointNames())
        converted.setFeatureName(0, toConvert.getFeatureName(0))

        self.appendFeatures(converted)

    def extractPointsByCoinToss(self, extractionProbability):
        """
        Return a new object containing a randomly selected sample of points
        from this object, where a random experiment is performed for each
        point, with the chance of selection equal to the extractionProbabilty
        parameter. Those selected values are also removed from this object.

        """
        #		if self.points == 0:
        #			raise ImproperActionException("Cannot extract points from an object with 0 points")

        if extractionProbability is None:
            raise ArgumentException("Must provide a extractionProbability")
        if extractionProbability <= 0:
            raise ArgumentException("extractionProbability must be greater than zero")
        if extractionProbability >= 1:
            raise ArgumentException("extractionProbability must be less than one")

        def experiment(point):
            return bool(pythonRandom.random() < extractionProbability)

        ret = self.extractPoints(experiment)

        return ret


    def calculateForEachPoint(self, function, points=None):
        """
        Calculates the results of the given function on the specified points
        in this object, with output values collected into a new object that
        is returned upon completion.

        function must not be none and accept the view of a point as an argument

        points may be None to indicate application to all points, a single point
        ID or a list of point IDs to limit application only to those specified.

        """
        if points is not None:
            points = copy.copy(points)
        if self.points == 0:
            raise ImproperActionException("We disallow this function when there are 0 points")
        if self.features == 0:
            raise ImproperActionException("We disallow this function when there are 0 features")
        if function is None:
            raise ArgumentException("function must not be None")

        if points is not None and not isinstance(points, list):
            if not isinstance(points, int):
                raise ArgumentException(
                    "Only allowable inputs to 'points' parameter is an int ID, a list of int ID's, or None")
            points = [points]

        if points is not None:
            for i in range(len(points)):
                points[i] = self._getPointIndex(points[i])

        self.validate()

        ret = self._calculateForEach_implementation(function, points, 'point')

        if points is not None:
            setNames = [self.getPointName(x) for x in sorted(points)]
            ret.setPointNames(setNames)
        else:
            ret.setPointNames(self.getPointNames())

        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath

        return ret


    def calculateForEachFeature(self, function, features=None):
        """
        Calculates the results of the given function on the specified features
        in this object, with output values collected into a new object that is
        returned upon completion.

        function must not be none and accept the view of a point as an argument

        features may be None to indicate application to all features, a single
        feature ID or a list of feature IDs to limit application only to those
        specified.

        """
        if features is not None:
            features = copy.copy(features)
        if self.points == 0:
            raise ImproperActionException("We disallow this function when there are 0 points")
        if self.features == 0:
            raise ImproperActionException("We disallow this function when there are 0 features")
        if function is None:
            raise ArgumentException("function must not be None")

        if features is not None and not isinstance(features, list):
            if not (isinstance(features, int) or isinstance(features, six.string_types)):
                raise ArgumentException(
                    "Only allowable inputs to 'features' parameter is an ID, a list of int ID's, or None")
            features = [features]

        if features is not None:
            for i in range(len(features)):
                features[i] = self._getFeatureIndex(features[i])

        self.validate()

        ret = self._calculateForEach_implementation(function, features, 'feature')

        if features is not None:
            setNames = [self.getFeatureName(x) for x in sorted(features)]
            ret.setFeatureNames(setNames)
        else:
            ret.setFeatureNames(self.getFeatureNames())

        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath
        return ret


    def _calculateForEach_implementation(self, function, included, axis):
        if axis == 'point':
            viewIterator = self.pointIterator()
        else:
            viewIterator = self.featureIterator()

        retData = []
        for viewID, view in enumerate(viewIterator):
            if included is not None and viewID not in included:
                continue
            currOut = function(view)
            # first we branch on whether the output has multiple values or is singular.
            if hasattr(currOut, '__iter__') and not isinstance(currOut, six.string_types):#in python3, string has __iter__ too.
                # if there are multiple values, they must be random accessible
                if not hasattr(currOut, '__getitem__'):
                    raise ArgumentException(
                        "function must return random accessible data (ie has a __getitem__ attribute)")

                toCopyInto = []
                for value in currOut:
                    toCopyInto.append(value)
                retData.append(toCopyInto)
            # singular return
            else:
                retData.append([currOut])

        ret = UML.createData(self.getTypeString(), retData)
        if axis != 'point':
            ret.transpose()

        return ret


    def mapReducePoints(self, mapper, reducer):
        """
        Return a new object containing the results of the given mapper and
        reducer functions

        mapper:  a function receiving a point as the input and outputting an
                 iterable containing two-tuple(s) of mapping identifier and
                 point values

        reducer: a function receiving the output of mapper as input and outputting
                 a two-tuple containing the identifier and the reduced value
        """
        return self._mapReduce_implementation('point', mapper, reducer)

    def mapReduceFeatures(self, mapper, reducer):
        """
        Return a new object containing the results of the given mapper and
        reducer functions

        mapper:  a function receiving a feature as the input and outputting an
                 iterable containing two-tuple(s) of mapping identifier and
                 feature values

        reducer: a function receiving the output of mapper as input and outputting
                 a two-tuple containing the identifier and the reduced value
        """
        return self._mapReduce_implementation('feature', mapper, reducer)

    def _mapReduce_implementation(self, axis, mapper, reducer):
        if axis == 'point':
            targetCount = self.points
            otherCount = self.features
            valueIterator = self.pointIterator
            otherAxis = 'feature'
        else:
            targetCount = self.features
            otherCount = self.points
            valueIterator = self.featureIterator
            otherAxis = 'point'

        if targetCount == 0:
            return UML.createData(self.getTypeString(), numpy.empty(shape=(0, 0)))
        if otherCount == 0:
            msg = "We do not allow operations over {0}s if there are 0 {1}s".format(axis, otherAxis)
            raise ImproperActionException(msg)

        if mapper is None or reducer is None:
            raise ArgumentException("The arguments must not be none")
        if not hasattr(mapper, '__call__'):
            raise ArgumentException("The mapper must be callable")
        if not hasattr(reducer, '__call__'):
            raise ArgumentException("The reducer must be callable")

        self.validate()

        mapResults = {}
        # apply the mapper to each point in the data
        for value in valueIterator():
            currResults = mapper(value)
            # the mapper will return a list of key value pairs
            for (k, v) in currResults:
                # if key is new, we must add an empty list
                if k not in mapResults:
                    mapResults[k] = []
                # append this value to the list of values associated with the key
                mapResults[k].append(v)

        # apply the reducer to the list of values associated with each key
        ret = []
        for mapKey in mapResults.keys():
            mapValues = mapResults[mapKey]
            # the reducer will return a tuple of a key to a value
            redRet = reducer(mapKey, mapValues)
            if redRet is not None:
                (redKey, redValue) = redRet
                ret.append([redKey, redValue])
        ret = UML.createData(self.getTypeString(), ret)

        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath

        return ret

    def groupByFeature(self, by, countUniqueValueOnly=False):
        """
        Group data object by one or more features.
        Input:
        by: can be an int, string or a list of int or a list of string
        """
        def findKey1(point, by):#if by is a string or int
            return point[by]

        def findKey2(point, by):#if by is a list of string or a list of int
            return tuple([point[i] for i in by])

        if isinstance(by, (six.string_types, numbers.Number)):#if by is a list, then use findKey2; o.w. use findKey1
            findKey = findKey1
        else:
            findKey = findKey2

        res = {}
        if countUniqueValueOnly:
            for point in self.pointIterator():
                k = findKey(point, by)
                if k not in res:
                    res[k] = 1
                else:
                    res[k] += 1
        else:
            for point in self.pointIterator():
                k = findKey(point, by)
                if k not in res:
                    res[k] = point.getPointNames()
                else:
                    res[k].extend(point.getPointNames())

            for k in res:
                tmp = self.copyPoints(toCopy=res[k])
                tmp.extractFeatures(by)
                res[k] = tmp

        return res

    def countUniqueFeatureValues(self, feature):
        """
        Count unique values for one feature or multiple features combination.
        Input:
        feature: can be an int, string or a list of int or a list of string
        """
        return self.groupByFeature(feature, countUniqueValueOnly=True)

    def pointIterator(self):
    #		if self.features == 0:
    #			raise ImproperActionException("We do not allow iteration over points if there are 0 features")

        class pointIt():
            def __init__(self, outer):
                self._outer = outer
                self._position = 0

            def __iter__(self):
                return self

            def next(self):
                while (self._position < self._outer.points):
                    value = self._outer.pointView(self._position)
                    self._position += 1
                    return value
                raise StopIteration

            def __next__(self):
                return self.next()

        return pointIt(self)

    def featureIterator(self):
    #		if self.points == 0:
    #			raise ImproperActionException("We do not allow iteration over features if there are 0 points")

        class featureIt():
            def __init__(self, outer):
                self._outer = outer
                self._position = 0

            def __iter__(self):
                return self

            def next(self):
                while (self._position < self._outer.features):
                    value = self._outer.featureView(self._position)
                    self._position += 1
                    return value
                raise StopIteration

            def __next__(self):
                return self.next()

        return featureIt(self)


    def calculateForEachElement(self, function, points=None, features=None, preserveZeros=False,
                                skipNoneReturnValues=False, outputType=None):
        """
        Returns a new object containing the results of calling function(elementValue)
        or function(elementValue, pointNum, featureNum) for each element.

        points: Limit to only elements of the specified points; may be None for
        all points, a single ID, or a list of IDs; this will affect the shape
        of the returned object.

        features: Limit to only elements of the specified features; may be None for
        all features, a single ID, or a list of IDs; this will affect the shape
        of the returned object.

        preserveZeros: If True it does not apply the function to elements in
        the data that are 0 and a 0 is placed in its place in the output.

        skipNoneReturnValues: If True, any time function() returns None, the
        value that was input to the function will be put in the output in place
        of None.

        """
        oneArg = False
        try:
            function(0, 0, 0)
        except TypeError:
            oneArg = True

        if points is not None and not isinstance(points, list):
            if not isinstance(points, (int, six.string_types)):
                raise ArgumentException(
                    "Only allowable inputs to 'points' parameter is an int ID, a list of int ID's, or None")
            points = [points]

        if features is not None and not isinstance(features, list):
            if not isinstance(features, (int, six.string_types)):
                raise ArgumentException(
                    "Only allowable inputs to 'features' parameter is an ID, a list of int ID's, or None")
            features = [features]

        if points is not None:
            # points = copy.copy(points)
            points = [self._getPointIndex(i) for i in points]

        if features is not None:
            # features = copy.copy(features)
            features = [self._getFeatureIndex(i) for i in features]

        self.validate()

        points = points if points else list(range(self.points))
        features = features if features else list(range(self.features))
        valueArray = numpy.empty([len(points), len(features)])
        p = 0
        for pi in points:
            f = 0
            for fj in features:
                value = self[pi, fj]
                if preserveZeros and value == 0:
                    valueArray[p, f] = 0
                else:
                    currRet = function(value) if oneArg else function(value, pi, fj)
                    if skipNoneReturnValues and currRet is None:
                        valueArray[p, f] = value
                    else:
                        valueArray[p, f] = currRet
                f += 1
            p += 1

        if outputType is not None:
            optType = outputType
        else:
            optType = self.getTypeString()

        ret = UML.createData(optType, valueArray)

        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath

        return ret


    def countElements(self, function):
        """
        Apply the function onto each element, the result should be True or False, or 1 or 0. Then return back the sum of
        True (1).
        function: can be a function object or a string like '>0'.
        """
        if callable(function):
            ret = self.calculateForEachElement(function=function, outputType='Matrix')
        elif isinstance(function, six.string_types):
            func = lambda x: eval('x'+function)
            ret = self.calculateForEachElement(function=func, outputType='Matrix')
        else:
            raise ArgumentException('function can only be a function or str, not else')
        return int(numpy.sum(ret.data))

    def hashCode(self):
        """returns a hash for this matrix, which is a number x in the range 0<= x < 1 billion
        that should almost always change when the values of the matrix are changed by a substantive amount"""
        if self.points == 0 or self.features == 0:
            return 0
        valueObj = self.calculateForEachElement(hashCodeFunc, preserveZeros=True, outputType='Matrix')
        valueList = valueObj.copyAs(format="python list")
        avg = sum(itertools.chain.from_iterable(valueList)) / float(self.points * self.features)
        bigNum = 1000000000
        #this should return an integer x in the range 0<= x < 1 billion
        return int(int(round(bigNum * avg)) % bigNum)


    def isApproximatelyEqual(self, other):
        """If it returns False, this DataMatrix and otherDataMatrix definitely don't store equivalent data.
        If it returns True, they probably do but you can't be absolutely sure.
        Note that only the actual data stored is considered, it doesn't matter whether the data matrix objects
        passed are of the same type (Matrix, Sparse, etc.)"""
        self.validate()
        #first check to make sure they have the same number of rows and columns
        if self.points != other.points: return False
        if self.features != other.features: return False
        #now check if the hashes of each matrix are the same
        if self.hashCode() != other.hashCode(): return False
        return True


    def shufflePoints(self, indices=None):
        """
        Permute the indexing of the points so they are in a random order. Note: this relies on
        python's random.shuffle() so may not be sufficiently random for large number of points.
        See shuffle()'s documentation. None is always returned.

        """
        if indices is None:
            indices = list(range(0, self.points))
            pythonRandom.shuffle(indices)
        else:
            if len(indices) != self.points:
                raise ArgumentException(
                    "If indices are supplied, it must be a list with all and only valid point indices")
            for value in indices:
                if value < 0 or value > self.points:
                    raise ArgumentException("A value in indices is out of bounds of the valid range of points")

        def permuter(pView):
            return indices[self.getPointIndex(pView.getPointName(0))]

        # permuter.permuter = True
        # permuter.indices = indices
        self.sortPoints(sortHelper=permuter)


    def shuffleFeatures(self, indices=None):
        """
        Permute the indexing of the features so they are in a random order. Note: this relies on
        python's random.shuffle() so may not be sufficiently random for large number of features.
        See shuffle()'s documentation. None is always returned.

        """
        if indices is None:
            indices = list(range(0, self.features))
            pythonRandom.shuffle(indices)
        else:
            if len(indices) != self.features:
                raise ArgumentException(
                    "If indices are supplied, it must be a list with all and only valid features indices")
            for value in indices:
                if value < 0 or value > self.features:
                    raise ArgumentException("A value in indices is out of bounds of the valid range of features")

        def permuter(fView):
            return indices[self.getFeatureIndex(fView.getFeatureName(0))]

        self.sortFeatures(sortHelper=permuter)


    def copy(self):
        """
        Return a new object which has the same data (and featureNames, depending on
        the return type) and in the same UML format as this object.

        """
        return self.copyAs(self.getTypeString())

    def trainAndTestSets(self, testFraction, labels=None, randomOrder=True):
        """Partitions this object into training / testing, data / labels
        sets, returning a new object for each as needed.

        testFraction: the fraction of the data to be placed in the testing
        sets. If randomOrder is False, then the points are taken from the
        end of this object.

        labels: may be None, a single feature ID, or a list of feature
        IDs depending on whether one is dealing with data for unsupervised
        learning, single variable supervised learning, or multi-output
        supervised learning. This parameter will affect the shape of the
        returned tuple.

        randomOrder: controls whether the order of the points in the returns
        sets matches that of the original object, or if their order is
        randomized.

        Returns either a length 2 or a length 4 tuple. If labels=None, then
        returns a length 2 tuple containing the training object, then the
        testing object (trainX, testX). If labels is non-None, a length 4
        tuple is returned, containing the training data object, then the
        training labels object, then the testing data object, and finally
        the testing labels (trainX, trainY, testX, testY).

        """
        toSplit = self.copy()
        if randomOrder:
            toSplit.shufflePoints()

        testXSize = int(round(testFraction * self.points))
        startIndex = self.points - testXSize

        #pull out a testing set
        if testXSize == 0:
            testX = toSplit.extractPoints([])
        else:
            testX = toSplit.extractPoints(start=startIndex)

        if labels is None:
            toSplit.name = self.name + " trainX"
            testX.name = self.name + " testX"
            return toSplit, testX

        # safety for empty objects
        toExtract = labels
        if testXSize == 0:
            toExtract = []

        trainY = toSplit.extractFeatures(toExtract)
        testY = testX.extractFeatures(toExtract)

        toSplit.name = self.name + " trainX"
        trainY.name = self.name + " trainY"
        testX.name = self.name + " testX"
        testY.name = self.name + " testY"

        return toSplit, trainY, testX, testY


    def normalizePoints(self, subtract=None, divide=None, applyResultTo=None):
        """
        Modify all points in this object according to the given
        operations.

        applyResultTo: default None, if a UML object is given, then
        perform the same operations to it as are applied to the calling
        object. However, if a statistical method is specified as subtract
        or divide, then	concrete values are first calculated only from
        querying the calling object, and the operation is performed on
        applyResultTo using the results; as if a UML object was given
        for the subtract or divide arguments.

        subtract: what should be subtracted from data. May be a fixed
        numerical value, a string defining a statistical function (all of
        the same ones callable though pointStatistics), or a UML data
        object. If a vector shaped object is given, then the value
        associated with each point will be subtracted from all values of
        that point. Otherwise, the values in the object are used for
        elementwise subtraction. Default None - equivalent to subtracting
        0.

        divide: defines the denominator for dividing the data. May be a
        fixed numerical value, a string defining a statistical function
        (all of the same ones callable though pointStatistics), or a UML
        data object. If a vector shaped object is given, then the value
        associated with each point will be used in division of all values
        for that point. Otherwise, the values in the object are used for
        elementwise division. Default None - equivalent to dividing by
        1.

        Returns None while having affected the data of the calling
        object and applyResultTo (if non-None).

        """
        self._normalizeGeneric("point", subtract, divide, applyResultTo)

    def normalizeFeatures(self, subtract=None, divide=None, applyResultTo=None):
        """
        Modify all features in this object according to the given
        operations.

        applyResultTo: default None, if a UML object is given, then
        perform the same operations to it as are applied to the calling
        object. However, if a statistical method is specified as subtract
        or divide, then	concrete values are first calculated only from
        querying the calling object, and the operation is performed on
        applyResultTo using the results; as if a UML object was given
        for the subtract or divide arguments.

        subtract: what should be subtracted from data. May be a fixed
        numerical value, a string defining a statistical function (all of
        the same ones callable though featureStatistics), or a UML data
        object. If a vector shaped object is given, then the value
        associated with each feature will be subtracted from all values of
        that feature. Otherwise, the values in the object are used for
        elementwise subtraction. Default None - equivalent to subtracting
        0.

        divide: defines the denominator for dividing the data. May be a
        fixed numerical value, a string defining a statistical function
        (all of the same ones callable though featureStatistics), or a UML
        data object. If a vector shaped object is given, then the value
        associated with each feature will be used in division of all values
        for that feature. Otherwise, the values in the object are used for
        elementwise division. Default None - equivalent to dividing by
        1.

        Returns None while having affected the data of the calling
        object and applyResultTo (if non-None).

        """
        self._normalizeGeneric("feature", subtract, divide, applyResultTo)

    def _normalizeGeneric(self, axis, subtract, divide, applyResultTo):

        # used to trigger later conditionals
        alsoIsObj = isinstance(applyResultTo, UML.data.Base)

        # the operation is different when the input is a vector
        # or produces a vector (ie when the input is a statistics
        # string) so during the validation steps we check for
        # those cases
        subIsVec = False
        divIsVec = False

        # check it is within the desired types
        if subtract is not None:
            if not isinstance(subtract, (int, float, six.string_types, UML.data.Base)):
                msg = "The argument named subtract must have a value that is "
                msg += "an int, float, string, or is a UML data object"
                raise ArgumentException(msg)
        if divide is not None:
            if not isinstance(divide, (int, float, six.string_types, UML.data.Base)):
                msg = "The argument named divide must have a value that is "
                msg += "an int, float, string, or is a UML data object"
                raise ArgumentException(msg)

        # check that if it is a string, it is one of the accepted values
        if isinstance(subtract, six.string_types):
            self._validateStatisticalFunctionInputString(subtract)
        if isinstance(divide, six.string_types):
            self._validateStatisticalFunctionInputString(divide)

        # arg generic helper to check that objects are of the
        # correct shape/size
        def validateInObjectSize(argname, argval):
            inPC = argval.points
            inFC = argval.features
            selfPC = self.points
            selfFC = self.features

            inMainLen = inPC if axis == "point" else inFC
            inOffLen = inFC if axis == "point" else inPC
            selfMainLen = selfPC if axis == "point" else selfFC
            selfOffLen = selfFC if axis == 'point' else selfPC

            if inMainLen != selfMainLen or inOffLen != selfOffLen:
                vecErr = argname + " "
                vecErr += "was a UML object in the shape of a "
                vecErr += "vector (" + str(inPC) + " x "
                vecErr += str(inFC) + "), "
                vecErr += "but the length of long axis did not match "
                vecErr += "the number of " + axis + "s in this object ("
                vecErr += str(self.points) + ")."
                # treat it as a vector
                if inMainLen == 1:
                    if inOffLen != selfMainLen:
                        raise ArgumentException(vecErr)
                    return True
                # treat it as a vector
                elif inOffLen == 1:
                    if inMainLen != selfMainLen:
                        raise ArgumentException(vecErr)
                    argval.transpose()
                    return True
                # treat it as a mis-sized object
                else:
                    msg = argname + " "
                    msg += "was a UML obejct with a shape of ("
                    msg += str(inPC) + " x " + str(inFC) + "), "
                    msg += "but it doesn't match the shape of the calling"
                    msg += "object (" + str(selfPC) + " x "
                    msg += str(selfFC) + ")"
                    raise ArgumentException(msg)
            return False

        def checkAlsoShape(caller, also, objIn, axis):
            """
            Raises an exception if the normalized axis shape doesn't match the
            calling object, or if when subtract of divide takes an object, also
            doesn't match the shape of the caller (this is to be called after)
            the check that the caller's shape matches that of the subtract or
            divide argument.
            """
            offAxis = 'feature' if axis == 'point' else 'point'
            callerP = caller.points
            callerF = caller.features
            alsoP = also.points
            alsoF = also.features

            callMainLen = callerP if axis == "point" else callerF
            alsoMainLen = alsoP if axis == "point" else alsoF
            callOffLen = callerF if axis == "point" else callerP
            alsoOffLen = alsoF if axis == "point" else alsoP

            if callMainLen != alsoMainLen:
                msg = "applyResultTo must have the same number of " + axis
                msg += "s (" + str(alsoMainLen) + ") as the calling object "
                msg += "(" + str(callMainLen) + ")"
                raise ArgumentException(msg)
            if objIn and callOffLen != alsoOffLen:
                msg = "When a non-vector UML object is given for the subtract "
                msg += "or divide arguments, then applyResultTo "
                msg += "must have the same number of " + offAxis
                msg += "s (" + str(alsoOffLen) + ") as the calling object "
                msg += "(" + str(callOffLen) + ")"
                raise ArgumentException(msg)

        # actually check that objects are the correct shape/size
        objArg = False
        if isinstance(subtract, UML.data.Base):
            subIsVec = validateInObjectSize("subtract", subtract)
            objArg = True
        if isinstance(divide, UML.data.Base):
            divIsVec = validateInObjectSize("divide", divide)
            objArg = True

        # check the shape of applyResultTo
        if alsoIsObj:
            checkAlsoShape(self, applyResultTo, objArg, axis)

        # if a statistics string was entered, generate the results
        # of that statistic
        #		if isinstance(subtract, basestring):
        #			if axis == 'point':
        #				subtract = self.pointStatistics(subtract)
        #			else:
        #				subtract = self.featureStatistics(subtract)
        #			subIsVec = True
        #		if isinstance(divide, basestring):
        #			if axis == 'point':
        #				divide = self.pointStatistics(divide)
        #			else:
        #				divide = self.featureStatistics(divide)
        #			divIsVec = True

        if axis == 'point':
            indexGetter = lambda x: self.getPointIndex(x.getPointName(0))
            if isinstance(subtract, six.string_types):
                subtract = self.pointStatistics(subtract)
                subIsVec = True
            if isinstance(divide, six.string_types):
                divide = self.pointStatistics(divide)
                divIsVec = True
        else:
            indexGetter = lambda x: self.getFeatureIndex(x.getFeatureName(0))
            if isinstance(subtract, six.string_types):
                subtract = self.featureStatistics(subtract)
                subIsVec = True
            if isinstance(divide, six.string_types):
                divide = self.featureStatistics(divide)
                divIsVec = True

        # helper for when subtract is a vector of values
        def subber(currView):
            ret = []
            for val in currView:
                ret.append(val - subtract[indexGetter(currView)])
            return ret

        # helper for when divide is a vector of values
        def diver(currView):
            ret = []
            for val in currView:
                ret.append(val / divide[indexGetter(currView)])
            return ret

        # first perform the subtraction operation
        if subtract is not None and subtract != 0:
            if subIsVec:
                if axis == 'point':
                    self.transformEachPoint(subber)
                    if alsoIsObj:
                        applyResultTo.transformEachPoint(subber)
                else:
                    self.transformEachFeature(subber)
                    if alsoIsObj:
                        applyResultTo.transformEachFeature(subber)
            else:
                self -= subtract
                if alsoIsObj:
                    applyResultTo -= subtract

        # then perform the division operation
        if divide is not None and divide != 1:
            if divIsVec:
                if axis == 'point':
                    self.transformEachPoint(diver)
                    if alsoIsObj:
                        applyResultTo.transformEachPoint(diver)
                else:
                    self.transformEachFeature(diver)
                    if alsoIsObj:
                        applyResultTo.transformEachFeature(diver)
            else:
                self /= divide
                if alsoIsObj:
                    applyResultTo /= divide

        # this operation is self modifying, so we return None
        return None


    ########################################
    ########################################
    ###   Functions related to logging   ###
    ########################################
    ########################################


    def featureReport(self, maxFeaturesToCover=50, displayDigits=2):
        """
        Produce a report, in a string formatted as a table, containing summary and statistical
        information about each feature in the data set, up to 50 features.  If there are more
        than 50 features, only information about 50 of those features will be reported.
        """
        return produceFeaturewiseReport(self, maxFeaturesToCover=maxFeaturesToCover, displayDigits=displayDigits)

    def summaryReport(self, displayDigits=2):
        """
        Produce a report, in a string formatted as a table, containing summary
        information about the data set contained in this object.  Includes
        proportion of missing values, proportion of zero values, total # of points,
        and number of features.
        """
        return produceAggregateReport(self, displayDigits=displayDigits)


    ###############################################################
    ###############################################################
    ###   Subclass implemented information querying functions   ###
    ###############################################################
    ###############################################################


    def isIdentical(self, other):
        if not self._equalFeatureNames(other):
            return False
        if not self._equalPointNames(other):
            return False

        return self._isIdentical_implementation(other)


    def writeFile(self, outPath, format=None, includeNames=True):
        """
        Write the data in this object to a file using the specified format.

        outPath: the location (including file name and extension) where
        we want to write the output file.

        format: the formating of the file we write. May be None, 'csv', or
        'mtx'; if None, we use the extension of outPath to determine the format.

        includeNames: True or False indicating whether the file will embed the point
        and feature names into the file. The format of the embedding is dependant
        on the format of the file: csv will embed names into the data, mtx will
        place names in a comment.

        """
        if self.points == 0 or self.features == 0:
            raise ImproperActionException("We do not allow writing to file when an object has 0 points or features")

        self.validate()

        # if format is not specified, we fall back on the extension in outPath
        if format is None:
            split = outPath.rsplit('.', 1)
            format = None
            if len(split) > 1:
                format = split[1].lower()

        if format not in ['csv', 'mtx']:
            msg = "Unrecognized file format. Accepted types are 'csv' and 'mtx'. They may "
            msg += "either be input as the format parameter, or as the extension in the "
            msg += "outPath"
            raise ArgumentException(msg)

        includePointNames = includeNames
        if includePointNames:
            seen = False
            for name in self.getPointNames():
                if name[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                    seen = True
            if not seen:
                includePointNames = False

        includeFeatureNames = includeNames
        if includeFeatureNames:
            seen = False
            for name in self.getFeatureNames():
                if name[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                    seen = True
            if not seen:
                includeFeatureNames = False

        try:
            self._writeFile_implementation(outPath, format, includePointNames, includeFeatureNames)
        except Exception:
            if format.lower() == "csv":
                toOut = self.copyAs("Matrix")
                toOut._writeFile_implementation(outPath, format, includePointNames, includeFeatureNames)
                return
            if format.lower() == "mtx":
                toOut = self.copyAs('Sparse')
                toOut._writeFile_implementation(outPath, format, includePointNames, includeFeatureNames)
                return


    def getTypeString(self):
        """
        Return a string representing the non-abstract type of this object (e.g. Matrix,
        Sparse, etc.) that can be passed to createData() function to create a new object
        of the same type.
        """
        return self._getTypeString_implementation()

    def _processSingleX(self, x):
        """

        """
        length = self._pointCount
        if x.__class__ is int or x.__class__ is numpy.integer:
            if x < -length or x >= length:
                msg = "The given index " + str(x) + " is outside of the range "
                msg += "of possible indices in the point axis (0 to "
                msg += str(length - 1) + ")."
                raise IndexError(msg)
            if x >= 0:
                return x, True
            else:
                return x + length, True

        if x.__class__ is str or x.__class__ is six.text_type:
            return self.getPointIndex(x), True

        if x.__class__ is float:
            if x % 1: # x!=int(x)
                msg = "A float valued key of value x is only accepted if x == "
                msg += "int(x). The given value was " + str(x) + " yet int("
                msg += str(x) + ") = " + str(int(x))
                raise ArgumentException(msg)
            else:
                x = int(x)
                if x < -length or x >= length:
                    msg = "The given index " + str(x) + " is outside of the range "
                    msg += "of possible indices in the point axis (0 to "
                    msg += str(length - 1) + ")."
                    raise IndexError(msg)
                if x >= 0:
                    return x, True
                else:
                    return x + length, True

        return x, False

    def _processSingleY(self, y):
        """

        """
        length = self._featureCount
        if y.__class__ is int or y.__class__ is numpy.integer:
            if y < -length or y >= length:
                msg = "The given index " + str(y) + " is outside of the range "
                msg += "of possible indices in the point axis (0 to "
                msg += str(length - 1) + ")."
                raise IndexError(msg)
            if y >= 0:
                return y, True
            else:
                return y + length, True

        if y.__class__ is str or y.__class__ is six.text_type:
            return self.getFeatureIndex(y), True

        if y.__class__ is float:
            if y % 1: # y!=int(y)
                msg = "A float valued key of value y is only accepted if y == "
                msg += "int(y). The given value was " + str(y) + " yet int("
                msg += str(y) + ") = " + str(int(y))
                raise ArgumentException(msg)
            else:
                y = int(y)
                if y < -length or y >= length:
                    msg = "The given index " + str(y) + " is outside of the range "
                    msg += "of possible indices in the point axis (0 to "
                    msg += str(length - 1) + ")."
                    raise IndexError(msg)
                if y >= 0:
                    return y, True
                else:
                    return y + length, True

        return y, False

    def __getitem__(self, key):
        """
        The followings are allowed:
        X[1, :]            ->    (2d) that just has that one point
        X["age", :]    -> same as above
        X[1:5, :]         -> 4 points (1, 2, 3, 4)
        X[[3,8], :]       -> 2 points (3, 8) IN THE ORDER GIVEN
        X[["age","gender"], :]       -> same as above

        --get based on features only : ALWAYS returns a new copy UML object (2d)
        X[:,2]         -> just has that 1 feature
        X[:,"bob"] -> same as above
        X[:,1:5]    -> 4 features (1,2,3,4)
        X[:,[3,8]]  -> 2 features (3,8) IN THE ORDER GIVEN

        --both features and points : can give a scalar value OR UML object 2d depending on case
        X[1,2]           -> single scalar number value
        X["age","bob"]    -> single scalar number value
        X[1:5,4:7]           -> UML object (2d) that has just that rectangle
        X[[1,2],[3,8]]      -> UML object (2d) that has just 2 points (points 1,2) but only 2 features for each of them (features 3,8)
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
                raise ArgumentException(msg)

        #process x
        x, singleX = self._processSingleX(x)
        #process y
        y, singleY = self._processSingleY(y)
        #if it is the simplest data retrieval such as X[1,2], we'd like to return it back in the fastest way.
        if singleX and singleY:
            return self._getitem_implementation(x, y)

        if not singleX:
            if x.__class__ is slice:
                start = x.start if x.start is not None else 0
                if start < 0:
                    start += self.points
                stop = x.stop if x.stop is not None else self.points
                if stop < 0:
                    stop += self.points
                step = x.step if x.step is not None else 1
                x = [self._processSingleX(xi)[0] for xi in range(start, stop, step)]
            else:
                x = [self._processSingleX(xi)[0] for xi in x]

        if not singleY:
            if y.__class__ is slice:
                start = y.start if y.start is not None else 0
                if start < 0:
                    start += self.features
                stop = y.stop if y.stop is not None else self.features
                if stop < 0:
                    stop += self.features
                step = y.step if y.step is not None else 1
                y = [self._processSingleY(yi)[0] for yi in range(start, stop, step)]
            else:
                y = [self._processSingleY(yi)[0] for yi in y]

        return self.copyPoints(toCopy=x).copyFeatures(toCopy=y)

    def pointView(self, ID):
        """
        Returns a View object into the data of the point with the given ID. See View object
        comments for its capabilities. This View is only valid until the next modification
        to the shape or ordering of the internal data. After such a modification, there is
        no guarantee to the validity of the results.
        """
        if self.points == 0:
            raise ImproperActionException("ID is invalid, This object contains no points")

        index = self._getPointIndex(ID)
        return self.view(index, index, None, None)

    def featureView(self, ID):
        """
        Returns a View object into the data of the point with the given ID. See View object
        comments for its capabilities. This View is only valid until the next modification
        to the shape or ordering of the internal data. After such a modification, there is
        no guarantee to the validity of the results.
        """
        if self.features == 0:
            raise ImproperActionException("ID is invalid, This object contains no features")

        index = self._getFeatureIndex(ID)
        return self.view(None, None, index, index)

    def view(self, pointStart=None, pointEnd=None, featureStart=None,
             featureEnd=None):
        """
        Factory function to create a read only view into the calling data
        object. Views may only be constructed from contiguous, in-order
        points and features whose overlap defines a window into the data.
        The returned View object is part of UML's datatypes hiearchy, and
        will have access to all of the same methods as anything that
        inherits from UML.data.Base; though only those that do not modify
        the data can be called without an exception being raised. The
        returned view will also reflect any subsequent changes made to the
        original object. This is the only accepted method for a user to
        construct a View object (it should never be done directly), though
        view objects may be provided to the user, for example via user
        defined functions passed to extractPoints or calculateForEachFeature.

        pointStart: the inclusive index of the first point to be accessible
        in the returned view. Is None by default, meaning to include from
        the beginning of the object.

        pointEnd: the inclusive index of the last point to be accessible in
        the returned view. Is None by default, meaning to include up to the
        end of the object.

        featureStart: the inclusive index of the first feature to be
        accessible in the returned view. Is None by default, meaning to
        include from the beginning of the object.

        featureEnd: the inclusive index of the last feature to be accessible in
        the returned view. Is None by default, meaning to include up to the
        end of the object.

        """
        # transform defaults to mean take as much data as possible,
        # transform end values to be EXCLUSIVE
        if pointStart is None:
            pointStart = 0
        else:
            pointStart = self._getIndex(pointStart, 'point')

        if pointEnd is None:
            pointEnd = self.points
        else:
            pointEnd = self._getIndex(pointEnd, 'point')
            # this is the only case that could be problematic and needs
            # checking
            self._validateRangeOrder("pointStart", pointStart, "pointEnd", pointEnd)
            # make exclusive now that it won't ruin the validation check
            pointEnd += 1

        if featureStart is None:
            featureStart = 0
        else:
            featureStart = self._getIndex(featureStart, 'feature')

        if featureEnd is None:
            featureEnd = self.features
        else:
            featureEnd = self._getIndex(featureEnd, 'feature')
            # this is the only case that could be problematic and needs
            # checking
            self._validateRangeOrder("featureStart", featureStart, "featureEnd", featureEnd)
            # make exclusive now that it won't ruin the validation check
            featureEnd += 1

        return self._view_implementation(pointStart, pointEnd, featureStart,
                                         featureEnd)

    def validate(self, level=1):
        """
        Checks the integrity of the data with respect to the limitations and invariants
        that our objects enforce.
        """
        if self._pointNamesCreated():
            assert self.points == len(self.getPointNames())
        if self._featureNamesCreated():
            assert self.features == len(self.getFeatureNames())

        if level > 0:
            if self._pointNamesCreated():
                for key in self.getPointNames():
                    assert self.getPointName(self.getPointIndex(key)) == key
            if self._featureNamesCreated():
                for key in self.getFeatureNames():
                    assert self.getFeatureName(self.getFeatureIndex(key)) == key

        self._validate_implementation(level)


    def containsZero(self):
        """
        Returns True if there is a value that is equal to integer 0 contained
        in this object. False otherwise

        """
        # trivially False.
        if self.points == 0 or self.features == 0:
            return False
        return self._containsZero_implementation()

    def __eq__(self, other):
        return self.isIdentical(other)

    def __ne__(self, other):
        return not self.__eq__(other)


    def toString(self, includeNames=True, maxWidth=120, maxHeight=30,
                 sigDigits=3, maxColumnWidth=19):

        if self.points == 0 or self.features == 0:
            return ""

        # setup a bundle of fixed constants
        colSep = ' '
        colHold = '--'
        rowHold = '|'
        pnameSep = ' '
        nameHolder = '...'
        holderOrientation = 'center'
        dataOrientation = 'center'
        pNameOrientation = 'rjust'
        fNameOrientation = 'center'

        #setup a bundle of default values
        maxHeight = self.points + 2 if maxHeight is None else maxHeight
        maxWidth = float('inf') if maxWidth is None else maxWidth
        maxRows = min(maxHeight, self.points)
        maxDataRows = maxRows
        includePNames = False
        includeFNames = False

        if includeNames:
            includePNames = dataHelpers.hasNonDefault(self, 'point')
            includeFNames = dataHelpers.hasNonDefault(self, 'feature')
            if includeFNames:
                # plus or minus 2 because we will be dealing with both
                # feature names and a gap row
                maxRows = min(maxHeight, self.points + 2)
                maxDataRows = maxRows - 2

        # Set up point Names and determine how much space they take up
        pnames = None
        pnamesWidth = None
        maxDataWidth = maxWidth
        if includePNames:
            pnames, pnamesWidth = self._arrangePointNames(maxDataRows, maxColumnWidth,
                                                          rowHold, nameHolder)
            # The available space for the data is reduced by the width of the
            # pnames, a column separator, the pnames seperator, and another
            # column seperator
            maxDataWidth = maxWidth - (pnamesWidth + 2 * len(colSep) + len(pnameSep))

        # Set up data values to fit in the available space
        dataTable, colWidths = self._arrangeDataWithLimits(maxDataWidth, maxDataRows,
                                                           sigDigits, maxColumnWidth, colSep, colHold, rowHold,
                                                           nameHolder)

        # set up feature names list, record widths
        fnames = None
        if includeFNames:
            fnames = self._arrangeFeatureNames(maxWidth, maxColumnWidth,
                                               colSep, colHold, nameHolder)

            # adjust data or fnames according to the more restrictive set
            # of col widths
            makeConsistentFNamesAndData(fnames, dataTable, colWidths, colHold)

        # combine names into finalized table
        finalTable, finalWidths = self._arrangeFinalTable(pnames, pnamesWidth,
                                                          dataTable, colWidths, fnames, pnameSep)

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
            line = colSep.join(finalTable[r]) + "\n"
            out += line

        return out


    def __repr__(self):
        indent = '    '
        maxW = 120
        maxH = 40

        # setup type call
        ret = self.getTypeString() + "(\n"

        # setup data
        dataStr = self.toString(includeNames=False, maxWidth=maxW, maxHeight=maxH)
        byLine = dataStr.split('\n')
        # toString ends with a \n, so we get rid of the empty line produced by
        # the split
        byLine = byLine[:-1]
        # convert self.data into a string with nice format
        newLines = (']\n' + indent + ' [').join(byLine)
        ret += (indent + '[[%s]]\n') % newLines

        numRows = min(self.points, maxW)
        # if exists non default point names, print all (truncated) point names
        ret += dataHelpers.makeNamesLines(indent, maxW, numRows, self.points,
                                          self.getPointNames(), 'pointNames')
        # if exists non default feature names, print all (truncated) feature names
        numCols = 0
        if byLine:
            splited = byLine[0].split(' ')
            for val in splited:
                if val != '' and val != '...':
                    numCols += 1
        elif self.features > 0:
            # if the container is empty, then roughly compute length of
            # the string of feature names, and then calculate numCols
            strLength = len("___".join(self.getFeatureNames())) + \
                        len(''.join([str(i) for i in range(self.features)]))
            numCols = int(min(1, maxW / float(strLength)) * self.features)
        # because of how dataHelers.indicesSplit works, we need this to be plus one
        # in some cases this means one extra feature name is displayed. But that's
        # acceptable
        if numCols <= self.features:
            numCols += 1
        ret += dataHelpers.makeNamesLines(indent, maxW, numCols, self.features,
                                          self.getFeatureNames(), 'featureNames')

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

    def show(self, description, includeObjectName=True, includeAxisNames=True, maxWidth=120,
             maxHeight=30, sigDigits=3, maxColumnWidth=19):
        """Method to simplify printing a representation of this data object,
        with some context. The backend is the toString() method, and this
        method includes control over all of the same functionality via
        arguments. Prior to the names and data, it additionally prints a
        description provided by the user, (optionally) this object's name
        attribute, and the number of points and features that are in the
        data.

        description: Unless None, this is printed as-is before the rest of
        the output.

        includeObjectName: if True, the object's name attribute will be
        printed.

        includeAxisNames: if True, the point and feature names will be
        printed.

        maxWidth: a bound on the maximum number of characters printed on
        each line of the output.

        maxHeight: a bound on the maximum number of lines printed in the
        outout.

        sigDigits: the number of decimal places to show when printing
        float valued data.

        nameLength: a bound on the maximum number of characters we allow
        for each point or feature name.

        """
        if description is not None:
            print(description)

        if includeObjectName:
            context = self.name + " : "
        else:
            context = ""
        context += str(self.points) + "pt x "
        context += str(self.features) + "ft"
        print(context)
        print(self.toString(includeAxisNames, maxWidth, maxHeight, sigDigits, maxColumnWidth))


    def plot(self, outPath=None, includeColorbar=False):
        self._plot(outPath, includeColorbar)


    def _setupOutFormatForPlotting(self, outPath):
        outFormat = None
        if isinstance(outPath, six.string_types):
            (path, ext) = os.path.splitext(outPath)
            if len(ext) == 0:
                outFormat = 'png'
        return outFormat

    def _matplotlibBackendHandleing(self, outPath, plotter, **kwargs):
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

        # toPlot = self.copyAs('numpyarray')

        # problem if we were to use mutiprocessing with backends
        # different than Agg.
        p = self._matplotlibBackendHandleing(outPath, plotter, d=self.data)
        return p

    def plotFeatureDistribution(self, feature, outPath=None, xMin=None, xMax=None):
        """Plot a histogram of the distribution of values in the specified
        Feature. Along the x axis of the plot will be the values seen in
        the feature, grouped into bins; along the y axis will be the number
        of values in each bin. Bin width is calculated using
        Freedman-Diaconis' rule. Control over the width of the x axis
        is also given, with the warning that user specified values
        can obscure data that would otherwise be plotted given default
        inputs.

        feature: the identifier (index of name) of the feature to show

        xMin: the least value shown on the x axis of the resultant plot.

        xMax: the largest value shown on the x axis of teh resultant plot

        """
        self._plotFeatureDistribution(feature, outPath, xMin, xMax)

    def _plotFeatureDistribution(self, feature, outPath=None, xMin=None, xMax=None):
        self._validateMatPlotLibImport(mplError, 'plotFeatureDistribution')
        return self._plotDistribution('feature', feature, outPath, xMin, xMax)

    def _plotDistribution(self, axis, identifier, outPath, xMin, xMax):
        outFormat = self._setupOutFormatForPlotting(outPath)
        index = self._getIndex(identifier, axis)
        if axis == 'point':
            getter = self.pointView
            name = self.getPointName(index)
        else:
            getter = self.featureView
            name = self.getFeatureName(index)

        toPlot = getter(index)

        quartiles = UML.calculate.quartiles(toPlot)

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

            if name[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
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
        p= self._matplotlibBackendHandleing(outPath, plotter, d=toPlot, xLim=(xMin, xMax))
        return p

    def plotFeatureAgainstFeatureRollingAverage(self, x, y, outPath=None, xMin=None, xMax=None, yMin=None,
                                  yMax=None, sampleSizeForAverage=20):

        self._plotFeatureAgainstFeature(x, y, outPath, xMin, xMax, yMin, yMax, sampleSizeForAverage)

    def plotFeatureAgainstFeature(self, x, y, outPath=None, xMin=None, xMax=None, yMin=None,
                                  yMax=None):
        """Plot a scatter plot of the two input features using the pairwise
        combination of their values as coordinates. Control over the width
        of the both axes is given, with the warning that user specified
        values can obscure data that would otherwise be plotted given default
        inputs.

        x: the identifier (index of name) of the feature from which we
        draw x-axis coordinates

        y: the identifier (index of name) of the feature from which we
        draw y-axis coordinates

        xMin: the least value shown on the x axis of the resultant plot.

        xMax: the largest value shown on the x axis of the resultant plot

        yMin: the least value shown on the y axis of the resultant plot.

        yMax: the largest value shown on the y axis of the resultant plot

        """
        self._plotFeatureAgainstFeature(x, y, outPath, xMin, xMax, yMin, yMax)

    def _plotFeatureAgainstFeature(self, x, y, outPath=None, xMin=None, xMax=None, yMin=None,
                                   yMax=None, sampleSizeForAverage=None):
        self._validateMatPlotLibImport(mplError, 'plotFeatureComparison')
        return self._plotCross(x, 'feature', y, 'feature', outPath, xMin, xMax, yMin, yMax, sampleSizeForAverage)

    def _plotCross(self, x, xAxis, y, yAxis, outPath, xMin, xMax, yMin, yMax, sampleSizeForAverage=None):
        outFormat = self._setupOutFormatForPlotting(outPath)
        xIndex = self._getIndex(x, xAxis)
        yIndex = self._getIndex(y, yAxis)

        def customGetter(index, axis):
            copied = self.copyPoints(index) if axis == 'point' else self.copyFeatures(index)
            return copied.copyAs('numpyarray', outputAs1D=True)

        def pGetter(index):
            return customGetter(index, 'point')

        def fGetter(index):
            return customGetter(index, 'feature')

        if xAxis == 'point':
            xGetter = pGetter
            xName = self.getPointName(xIndex)
        else:
            xGetter = fGetter
            xName = self.getFeatureName(xIndex)

        if yAxis == 'point':
            yGetter = pGetter
            yName = self.getPointName(yIndex)
        else:
            yGetter = fGetter
            yName = self.getFeatureName(yIndex)

        xToPlot = xGetter(xIndex)
        yToPlot = yGetter(yIndex)

        if sampleSizeForAverage:
            #do rolling average
            xToPlot, yToPlot = list(zip(*sorted(zip(xToPlot, yToPlot), key=lambda x: x[0])))
            convShape = numpy.ones(sampleSizeForAverage)/float(sampleSizeForAverage)
            startIdx = sampleSizeForAverage-1
            xToPlot = numpy.convolve(xToPlot, convShape)[startIdx:-startIdx]
            yToPlot = numpy.convolve(yToPlot, convShape)[startIdx:-startIdx]

        def plotter(inX, inY, xLim, yLim, sampleSizeForAverage):
            import matplotlib.pyplot as plt
            #plt.scatter(inX, inY)
            plt.scatter(inX, inY, marker='.')

            xlabel = xAxis + ' #' + str(xIndex) if xName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX else xName
            ylabel = yAxis + ' #' + str(yIndex) if yName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX else yName

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
        p= self._matplotlibBackendHandleing(outPath, plotter, inX=xToPlot, inY=yToPlot,
                                             xLim=(xMin, xMax), yLim=(yMin, yMax),
                                             sampleSizeForAverage=sampleSizeForAverage)
        return p

    def nonZeroIterator(self, iterateBy='points'):
        """
        Returns an iterator for all non-zero elements contained in this
        object, where the values in the same point|feature will be contiguous,
        with the earlier indexed points|features coming before the later indexed
        points|features.

        iterateBy: Genereate an iterator over 'points' or 'features'. Default is 'points'.

        If the object is one dimensional, iterateBy is ignored.
        """

        class EmptyIt(object):
            def __iter__(self):
                return self

            def next(self):
                raise StopIteration

            def __next__(self):
                return self.next()

        if self.points == 0 or self.features == 0:
            return EmptyIt()

        if self.points == 1:
            return self._nonZeroIteratorPointGrouped_implementation()
        if self.features == 1:
            return self._nonZeroIteratorFeatureGrouped_implementation()

        if iterateBy == 'points':
            return self._nonZeroIteratorPointGrouped_implementation()
        elif iterateBy == 'features':
            return self._nonZeroIteratorFeatureGrouped_implementation()
        else:
            msg = "iterateBy can just be 'points' or 'features'"
            raise ArgumentException(msg)

    ##################################################################
    ##################################################################
    ###   Subclass implemented structural manipulation functions   ###
    ##################################################################
    ##################################################################

    def transpose(self):
        """
        Function to transpose the data, ie invert the feature and point indices of the data.
        The point and feature names are also swapped. None is always returned.

        """
        self._transpose_implementation()

        self._pointCount, self._featureCount = self._featureCount, self._pointCount

        if self._pointNamesCreated() and self._featureNamesCreated():
            self.pointNames, self.featureNames = self.featureNames, self.pointNames
            self.setFeatureNames(self.featureNames)
            self.setPointNames(self.pointNames)
        elif self._pointNamesCreated():
            self.featureNames = self.pointNames
            self.pointNames = None
            self.pointNamesInverse = None
            self.setFeatureNames(self.featureNames)
        elif self._featureNamesCreated():
            self.pointNames = self.featureNames
            self.featureNames = None
            self.featureNamesInverse = None
            self.setPointNames(self.pointNames)
        else:
            pass

        self.validate()

    def appendPoints(self, toAppend):
        """
        Expand this object by appending the points from the toAppend object
        after the points currently in this object, merging together their
        features. The features in toAppend do not need to be in the same
        order as in the calling object; the data will automatically be
        placed using the calling object's feature order if there is an
        unambiguous mapping. toAppend will be unaffected by calling this
        method.

        toAppend - the UML data object whose contents we will be including
        in this object. Must have the same number of features as the calling
        object. Must not share any point names with the calling object. Must
        have the same feature names as the calling object, but not necessarily
        in the same order.

        """
        self._append_implementation('point', toAppend)


    def appendFeatures(self, toAppend):
        """
        Expand this object by appending the features from the toAppend object
        after the features currently in this object, merging together their
        points. The points in toAppend do not need to be in the same
        order as in the calling object; the data will automatically be
        placed using the calling object's point order if there is an
        unambiguous mapping. toAppend will be unaffected by calling this
        method.

        toAppend - the UML data object whose contents we will be including
        in this object. Must have the same number of points as the calling
        object. Must not share any feature names with the calling object.
        Must have the same point names as the calling object, but not
        necessarily in the same order.

        """
        self._append_implementation('feature', toAppend)


    def _append_implementation(self, axis, toAppend):
        self._validateValueIsNotNone("toAppend", toAppend)
        self._validateValueIsUMLDataObject("toAppend", toAppend, True)
        self._validateEmptyNamesIntersection(axis, "toAppend", toAppend)

        if axis == 'point':
            self._validateObjHasSameNumberOfFeatures("toAppend", toAppend)
            # need this in case we are self appending
            origCountS = self.points
            origCountTA = toAppend.points

            otherAxis = 'feature'
            funcString = 'appendPoints'
            selfSetName = self.setPointName
            toAppendGetName = toAppend.getPointName
            selfAppendImplementation = self._appendPoints_implementation
            selfSetCount = self._setpointCount
            selfCount = self._pointCount
            toAppendCount = toAppend.points
            selfAddName = self._addPointName
        else:
            self._validateObjHasSameNumberOfPoints("toAppend", toAppend)
            # need this in case we are self appending
            origCountS = self.features
            origCountTA = toAppend.features

            otherAxis = 'point'
            funcString = 'appendFeatures'
            selfSetName = self.setFeatureName
            selfAppendImplementation = self._appendFeatures_implementation
            toAppendGetName = toAppend.getFeatureName
            selfSetName = self.setFeatureName
            selfSetCount = self._setfeatureCount
            selfCount = self._featureCount
            toAppendCount = toAppend.features
            selfAddName = self._addFeatureName

        # Two cases will require us to use a generic implementation with
        # reordering capabilities: the names are consistent but out of order,
        # or the type of the objects is different.
        isReordered = self._validateReorderedNames(otherAxis, funcString, toAppend)
        differentType = self.getTypeString() != toAppend.getTypeString()

        if isReordered or differentType:
            self._appendReorder_implementation(axis, toAppend)
        else:
            selfAppendImplementation(toAppend)
            selfSetCount(selfCount + toAppendCount)

        # Have to make sure the point/feature names match the extended data. In
        # the case that the reordering implementation is used, the new points or
        # features already have default name assignments, so we set the correct
        # names. In the case of the standard implementation, we must use Base's
        # helper to add new names.
        for i in range(origCountTA):
            currName = toAppendGetName(i)
            # This insures there is no name collision with defaults already
            # present in the original object
            if currName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
                currName = self._nextDefaultName(axis)
            if isReordered or differentType:
                selfSetName(origCountS + i, currName)
            else:
                selfAddName(currName)

        self.validate()


    def _appendReorder_implementation(self, axis, toAppend):
        if axis == 'point':
            newPointNames = self.getPointNames() + ([None] * toAppend.points)
            newFeatureNames = toAppend.getFeatureNames()
            newPointSize = self.points + toAppend.points
            newFeatureSize = self.features
        else:
            newPointNames = toAppend.getPointNames()
            newFeatureNames = self.getFeatureNames() + ([None] * toAppend.features)
            newPointSize = self.points
            newFeatureSize = self.features + toAppend.features

        newObj = UML.zeros(self.getTypeString(), newPointSize, newFeatureSize,
                           pointNames=newPointNames, featureNames=newFeatureNames, name=self.name)

        if axis == 'point':
            newObj.fillWith(toAppend, self.points, 0, newObj.points - 1, newObj.features - 1)
            resortOrder = [self.getFeatureIndex(toAppend.getFeatureName(i)) for i in range(self.features)]
            orderObj = UML.createData(self.getTypeString(), resortOrder)
            newObj.fillWith(orderObj, self.points - 1, 0, self.points - 1, newObj.features - 1)
            newObj.sortFeatures(sortBy=self.points - 1)
            newObj.fillWith(self, 0, 0, self.points - 1, newObj.features - 1)
            self.referenceDataFrom(newObj)
        else:
            newObj.fillWith(toAppend, 0, self.features, newObj.points - 1, newObj.features - 1)
            resortOrder = [self.getPointIndex(toAppend.getPointName(i)) for i in range(self.points)]
            orderObj = UML.createData(self.getTypeString(), resortOrder)
            orderObj.transpose()
            newObj.fillWith(orderObj, 0, self.features - 1, newObj.points - 1, self.features - 1)
            newObj.sortPoints(sortBy=self.features - 1)
            newObj.fillWith(self, 0, 0, newObj.points - 1, self.features - 1)
            self.referenceDataFrom(newObj)


    def sortPoints(self, sortBy=None, sortHelper=None):
        """
        Modify this object so that the points are sorted in place, where sortBy may
        indicate the feature to sort by or None if the entire point is to be taken as a key,
        sortHelper may either be comparator, a scoring function, or None to indicate the natural
        ordering. None is always returned.
        """
        # its already sorted in these cases
        if self.features == 0 or self.points == 0 or self.points == 1:
            return
        if sortBy is not None and sortHelper is not None:
            raise ArgumentException("Cannot specify a feature to sort by and a helper function")
        if sortBy is None and sortHelper is None:
            raise ArgumentException("Either sortBy or sortHelper must not be None")

        if sortBy is not None and isinstance(sortBy, six.string_types):
            sortBy = self._getFeatureIndex(sortBy)

        newPointNameOrder = self._sortPoints_implementation(sortBy, sortHelper)
        self.setPointNames(newPointNameOrder)

        self.validate()

    def sortFeatures(self, sortBy=None, sortHelper=None):
        """
        Modify this object so that the features are sorted in place, where sortBy may
        indicate the feature to sort by or None if the entire point is to be taken as a key,
        sortHelper may either be comparator, a scoring function, or None to indicate the natural
        ordering.  None is always returned.

        """
        # its already sorted in these cases
        if self.features == 0 or self.points == 0 or self.features == 1:
            return
        if sortBy is not None and sortHelper is not None:
            raise ArgumentException("Cannot specify a feature to sort by and a helper function")
        if sortBy is None and sortHelper is None:
            raise ArgumentException("Either sortBy or sortHelper must not be None")

        if sortBy is not None and isinstance(sortBy, six.string_types):
            sortBy = self._getPointIndex(sortBy)

        newFeatureNameOrder = self._sortFeatures_implementation(sortBy, sortHelper)
        self.setFeatureNames(newFeatureNameOrder)

        self.validate()


    def extractPoints(self, toExtract=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, removing those points that are specified by the input, and returning
        an object containing those removed points.

        toExtract may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a point will return True if it is to be extracted, or a
        filter function, as a string, containing a comparison operator between a feature name
        and a value (i.e 'feat1<10')

        number is the quantity of points that are to be extracted, the default None means
        unrestricted extraction.

        start and end are parameters indicating range based extraction: if range based
        extraction is employed, toExtract must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.points respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen points are determined by point order,
        otherwise it is uniform random across the space of possible removals.

        """
        ret = self._genericStructuralFrontend('extract', 'point', toExtract, start, end,
                                              number, randomize)

        self._pointCount -= ret.points
        ret.setFeatureNames(self.getFeatureNames())
        for key in ret.getPointNames():
            self._removePointNameAndShift(key)

        ret._relPath = self.relativePath
        ret._absPath = self.absolutePath

        self.validate()
        return ret


    def extractFeatures(self, toExtract=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, removing those features that are specified by the input, and returning
        an object containing those removed features.

        toExtract may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a feature will return True if it is to be extracted, or a
        filter function, as a string, containing a comparison operator between a point name
        and a value (i.e 'point1<10')

        number is the quantity of features that are to be extracted, the default None means
        unrestricted extraction.

        start and end are parameters indicating range based extraction: if range based
        extraction is employed, toExtract must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.features respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen features are determined by feature order,
        otherwise it is uniform random across the space of possible removals.

        """
        ret = self._genericStructuralFrontend('extract', 'feature', toExtract, start, end,
                                              number, randomize)

        self._featureCount -= ret.features
        if ret.features != 0:
            ret.setPointNames(self.getPointNames())
        for key in ret.getFeatureNames():
            self._removeFeatureNameAndShift(key)

        ret._relPath = self.relativePath
        ret._absPath = self.absolutePath

        self.validate()
        return ret

    def deletePoints(self, toDelete=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, removing those points that are specified by the input.

        toDelete may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a point will return True if it is to be deleted, or a
        filter function, as a string, containing a comparison operator between a feature name
        and a value (i.e 'feat1<10')

        number is the quantity of points that are to be deleted, the default None means
        unrestricted deletion.

        start and end are parameters indicating range based deletion: if range based
        deletion is employed, toDelete must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.points respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen points are determined by point order,
        otherwise it is uniform random across the space of possible removals.

        """
        ret = self.extractPoints(toExtract=toDelete, start=start, end=end, number=number, randomize=randomize)


    def deleteFeatures(self, toDelete=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, removing those features that are specified by the input.

        toDelete may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a feature will return True if it is to be deleted, or a
        filter function, as a string, containing a comparison operator between a point name
        and a value (i.e 'point1<10')

        number is the quantity of features that are to be deleted, the default None means
        unrestricted deleted.

        start and end are parameters indicating range based deletion: if range based
        deletion is employed, toDelete must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.features respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen features are determined by feature order,
        otherwise it is uniform random across the space of possible removals.

        """
        ret = self.extractFeatures(toExtract=toDelete, start=start, end=end, number=number, randomize=randomize)


    def retainPoints(self, toRetain=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, retaining those points that are specified by the input.

        toRetain may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a point will return True if it is to be retained, or a
        filter function, as a string, containing a comparison operator between a feature name
        and a value (i.e 'feat1<10')

        number is the quantity of points that are to be retained, the default None means
        unrestricted retention.

        start and end are parameters indicating range based retention: if range based
        retention is employed, toRetain must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.points respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen points are determined by point order,
        otherwise it is uniform random across the space of possible retentions.

        """
        self._retain_implementation('retain', 'point', toRetain, start, end, number, randomize)


    def retainFeatures(self, toRetain=None, start=None, end=None, number=None, randomize=False):
        """
        Modify this object, retaining those features that are specified by the input.

        toRetain may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a feature will return True if it is to be deleted, or a
        filter function, as a string, containing a comparison operator between a point name
        and a value (i.e 'point1<10')

        number is the quantity of features that are to be retained, the default None means
        unrestricted retention.

        start and end are parameters indicating range based retention: if range based
        retention is employed, toRetain must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.features respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen features are determined by feature order,
        otherwise it is uniform random across the space of possible retentions.

        """
        self._retain_implementation('retain', 'feature', toRetain, start, end, number, randomize)


    def _retain_implementation(self, structure, axis, toRetain, start, end, number, randomize):
        """Implements retainPoints or retainFeatures based on the axis. The complements
        of toRetain are identified to use the extract backend, this is done within this
        implementation except for functions which are complemented within the next helper
        function
        """
        if axis == 'point':
            hasName = self.hasPointName
            getNames = self.getPointNames
            getIndex = self._getPointIndex
            values = self.points
            shuffleValues = self.shufflePoints
        else:
            hasName = self.hasFeatureName
            getNames = self.getFeatureNames
            getIndex = self._getFeatureIndex
            values = self.features
            shuffleValues = self.shuffleFeatures

        # extract points not in toRetain
        if toRetain is not None:
            if isinstance(toRetain, six.string_types):
                if hasName(toRetain):
                    toExtract = [value for value in getNames() if value != toRetain]
                else:
                    # toRetain is a function passed as a string
                    toExtract = toRetain

            elif isinstance(toRetain, (int, numpy.int, numpy.int64)):
                toExtract = [value for value in range(values) if value != toRetain]

            elif isinstance(toRetain, list):
                toRetain = [self._getIndex(value, axis) for value in toRetain]
                toExtract = [value for value in range(values) if value not in toRetain]
                # change the index order of the values to match toRetain
                reindex = toRetain + toExtract
                indices = [None for _ in range(values)]
                for idx, value in enumerate(reindex):
                    indices[value] = idx
                shuffleValues(indices)
                # extract any values after the toRetain values
                extractValues = range(len(toRetain), values)
                toExtract = list(extractValues)

            else:
                # toRetain is a function
                toExtract = toRetain

            ret = self._genericStructuralFrontend('retain', axis, toExtract, start, end, number,
                                                  False)
            self._adjustNamesAndValidate(ret, axis)

        # convert start and end to indexes
        if start is not None and end is not None:
            start = getIndex(start)
            end = getIndex(end)
            if start > end:
                msg = "the value for start ({0}) exceeds the value of end ({1})".format(start,end)
                raise ArgumentException(msg)
            else:
                # adjust end and values for start values that will be removed
                end -= start
                values -= start
        elif start is not None:
            start = getIndex(start)
        elif end is not None:
            end = getIndex(end)

        # extract points not between start and end
        if start is not None:
            # only need to perform if start is not the first value
            if start - 1 >= 0:
                ret = self._genericStructuralFrontend('retain', axis, None, 0, start - 1,
                                                          None, False)
                self._adjustNamesAndValidate(ret, axis)
        if end is not None:
            # only need to perform if end is not the last value
            if end + 1 <= values - 1:
                ret = self._genericStructuralFrontend('retain', axis, None, end + 1, values - 1,
                                                          None, False)
                self._adjustNamesAndValidate(ret, axis)

        if randomize:
            indices = list(range(0, values))
            pythonRandom.shuffle(indices)
            shuffleValues(indices)

        if number is not None:
            start = number
            end = values - 1
            ret = self._genericStructuralFrontend('retain', axis, None, start, end,
                                                      None, False)
            self._adjustNamesAndValidate(ret, axis)


    def countPoints(self, condition):
        """
        Similar to function extractPoints. Here we return back the number of points which satisfy the condition.
        condition: can be a string or a function object.
        """
        return self._genericStructuralFrontend('count', 'point', condition)


    def countFeatures(self, condition):
        """
        Similar to function extractFeatures. Here we return back the number of features which satisfy the condition.
        condition: can be a string or a function object.
        """
        return self._genericStructuralFrontend('count', 'feature', condition)


    def referenceDataFrom(self, other):
        """
        Modifies the internal data of this object to refer to the same data as other. In other
        words, the data wrapped by both the self and other objects resides in the
        same place in memory. Other must be an object of the same type as
        the calling object. Also, the shape of other should be consistent with the set
        of featureNames currently in this object. None is always returned.

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

        self.validate()


    def copyAs(self, format, rowsArePoints=True, outputAs1D=False):
        """
        Return a new object which has the same data (and featureNames, depending on
        the return type) as this object. To return a specific kind of UML data
        object, one may specify the format parameter to be 'List', 'Matrix', or
        'Sparse'. To specify a raw return type (which will not include feature names),
        one may specify 'python list', 'numpy array', or 'numpy matrix', 'scipy csr',
        'scypy csc', 'list of dict' or 'dict of list'.

        """
        #make lower case, strip out all white space and periods, except if format
        # is one of the accepted UML data types
        if format not in ['List', 'Matrix', 'Sparse', 'DataFrame']:
            format = format.lower()
            format = format.strip()
            tokens = format.split(' ')
            format = ''.join(tokens)
            tokens = format.split('.')
            format = ''.join(tokens)
            if format not in ['pythonlist', 'numpyarray', 'numpymatrix', 'scipycsr', 'scipycsc',
                              'listofdict', 'dictoflist']:
                msg = "The only accepted asTypes are: 'List', 'Matrix', 'Sparse'"
                msg += ", 'python list', 'numpy array', 'numpy matrix', 'scipy csr', 'scipy csc'"
                msg += ", 'list of dict', and 'dict of list'"
                raise ArgumentException(msg)

        # we only allow 'numpyarray' and 'pythonlist' to be used with the outpuAs1D flag
        if outputAs1D:
            if format != 'numpyarray' and format != 'pythonlist':
                raise ArgumentException("Cannot output as 1D if format != 'numpy array' or 'python list'")
            if self.points != 1 and self.features != 1:
                raise ArgumentException("To output as 1D there may either be only one point or one feature")

        # certain shapes and formats are incompatible
        if format.startswith('scipy'):
            if self.points == 0 or self.features == 0:
                raise ArgumentException('Cannot output a point or feature empty object in a scipy format')

        ret = self._copyAs_implementation_base(format, rowsArePoints, outputAs1D)

        if isinstance(ret, UML.data.Base):
            ret._name = self.name
            ret._relPath = self.relativePath
            ret._absPath = self.absolutePath

        return ret

    def _copyAs_implementation_base(self, format, rowsArePoints, outputAs1D):
        # in copyAs, we've already limited outputAs1D to the 'numpyarray' and 'python list' formats
        if outputAs1D:
            if self.points == 0 or self.features == 0:
                if format == 'numpyarray':
                    return numpy.array([])
                if format == 'pythonlist':
                    return []
            raw = self._copyAs_implementation('numpyarray').flatten()
            if format != 'numpyarray':
                raw = raw.tolist()
            return raw

        # we enforce very specific shapes in the case of emptiness along one
        # or both axes
        if format == 'pythonlist':
            if self.points == 0:
                return []
            if self.features == 0:
                ret = []
                for i in range(self.points):
                    ret.append([])
                return ret

        if format in ['listofdict', 'dictoflist']:
            ret = self._copyAs_implementation('numpyarray')
        else:
            ret = self._copyAs_implementation(format)
            if isinstance(ret, UML.data.Base):
                self._copyNames(ret)

        def _createListOfDict(data, featureNames):
            # creates a list of dictionaries mapping feature names to the point's values
            # dictionaries are in point order
            listofdict = []
            for point in data:
                feature_dict = {}
                for i, value in enumerate(point):
                    feature = featureNames[i]
                    feature_dict[feature] = value
                listofdict.append(feature_dict)
            return listofdict

        def _createDictOfList(data, featureNames, nFeatures):
            # creates a python dict maps feature names to python lists containing
            # all of that feature's values
            dictoflist = {}
            for i in range(nFeatures):
                feature = featureNames[i]
                values_list = data[:,i].tolist()
                dictoflist[feature] = values_list
            return dictoflist

        if not rowsArePoints:
            if format in ['List', 'Matrix', 'Sparse', 'DataFrame']:
                ret.transpose()
            elif format == 'listofdict':
                ret = ret.transpose()
                ret = _createListOfDict(data=ret, featureNames=self.getPointNames())
                return ret
            elif format == 'dictoflist':
                ret = ret.transpose()
                ret = _createDictOfList(data=ret, featureNames=self.getPointNames(), nFeatures=self.points)
                return ret
            elif format != 'pythonlist':
                ret = ret.transpose()
            else:
                ret = numpy.transpose(ret).tolist()

        if format == 'listofdict':
            ret = _createListOfDict(data=ret, featureNames=self.getFeatureNames())
        if format == 'dictoflist':
            ret = _createDictOfList(data=ret, featureNames=self.getFeatureNames(), nFeatures=self.features)

        return ret

    def _copyNames (self, CopyObj):
        if self._pointNamesCreated():
            CopyObj.pointNamesInverse = self.getPointNames()
            CopyObj.pointNames = copy.copy(self.pointNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.index = self.getPointNames()
        else:
            CopyObj.pointNamesInverse = None
            CopyObj.pointNames = None

        if self._featureNamesCreated():
            CopyObj.featureNamesInverse = self.getFeatureNames()
            CopyObj.featureNames = copy.copy(self.featureNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.columns = self.getFeatureNames()
        else:
            CopyObj.featureNamesInverse = None
            CopyObj.featureNames = None

        CopyObj._nextDefaultValueFeature = self._nextDefaultValueFeature
        CopyObj._nextDefaultValuePoint = self._nextDefaultValuePoint

    def copyPoints(self, toCopy=None, start=None, end=None, number=None, randomize=False):
        """
        Returns an object containing those points that are specified by the input, without
        modification to this object.

        toCopy may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a point will return True if it is to be copied, or a
        filter function, as a string, containing a comparison operator between a feature name
        and a value (i.e 'feat1<10')

        number is the quantity of points that are to be copied, the default None means
        unrestricted copying.

        start and end are parameters indicating range based copying: if range based
        copying is employed, toCopy must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.points respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen points are determined by point order,
        otherwise it is uniform random across the space of possible points.

        """
        ret = self._genericStructuralFrontend('copy', 'point', toCopy, start, end,
                                              number, randomize)

        ret.setFeatureNames(self.getFeatureNames())

        ret._relPath = self.relativePath
        ret._absPath = self.absolutePath

        self.validate()
        return ret


    def copyFeatures(self, toCopy=None, start=None, end=None, number=None, randomize=False):
        """
        Returns an object containing those features that are specified by the input, without
        modification to this object.

        toCopy may be a single identifier (name and/or index), a list of identifiers,
        a function that when given a feature will return True if it is to be copied, or a
        filter function, as a string, containing a comparison operator between a point name
        and a value (i.e 'point1<10')

        number is the quantity of features that are to be copied, the default None means
        unrestricted copying.

        start and end are parameters indicating range based copying: if range based
        copying is employed, toCopy must be None, and vice versa. If only one of start
        and end are non-None, the other defaults to 0 and self.features respectably.

        randomize indicates whether random sampling is to be used in conjunction with the number
        parameter, if randomize is False, the chosen features are determined by feature order,
        otherwise it is uniform random across the space of possible features.

        """
        ret = self._genericStructuralFrontend('copy', 'feature', toCopy, start, end,
                                              number, randomize)

        ret.setPointNames(self.getPointNames())

        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath

        return ret


    def transformEachPoint(self, function, points=None):
        """
        Modifies this object to contain the results of the given function
        calculated on the specified points in this object.

        function must not be none and accept the view of a point as an argument

        points may be None to indicate application to all points, a single point
        ID or a list of point IDs to limit application only to those specified.

        """
        if self.points == 0:
            raise ImproperActionException("We disallow this function when there are 0 points")
        if self.features == 0:
            raise ImproperActionException("We disallow this function when there are 0 features")
        if function is None:
            raise ArgumentException("function must not be None")

        if points is not None and not isinstance(points, list):
            if not isinstance(points, int):
                raise ArgumentException(
                    "Only allowable inputs to 'points' parameter is an int ID, a list of int ID's, or None")
            points = [points]

        if points is not None:
            points = copy.copy(points)
            for i in range(len(points)):
                points[i] = self._getPointIndex(points[i])

        self.validate()

        self._transformEachPoint_implementation(function, points)


    def transformEachFeature(self, function, features=None):
        """
        Modifies this object to contain the results of the given function
        calculated on the specified features in this object.

        function must not be none and accept the view of a feature as an argument

        features may be None to indicate application to all features, a single
        feature ID or a list of feature IDs to limit application only to those
        specified.

        """
        if self.points == 0:
            raise ImproperActionException("We disallow this function when there are 0 points")
        if self.features == 0:
            raise ImproperActionException("We disallow this function when there are 0 features")
        if function is None:
            raise ArgumentException("function must not be None")

        if features is not None and not isinstance(features, list):
            if not (isinstance(features, int) or isinstance(features, six.string_types)):
                raise ArgumentException(
                    "Only allowable inputs to 'features' parameter is an ID, a list of int ID's, or None")
            features = [features]

        if features is not None:
            features = copy.copy(features)
            for i in range(len(features)):
                features[i] = self._getFeatureIndex(features[i])

        self.validate()

        self._transformEachFeature_implementation(function, features)


    def transformEachElement(self, toTransform, points=None, features=None, preserveZeros=False,
                             skipNoneReturnValues=False):
        """
        Modifies this object to contain the results of toTransform for each element.

        toTransform: May be a function in the form of
        toTransform(elementValue) or toTransform(elementValue, pointNum, featureNum),
        or a dictionary mapping the current element [key] to the transformed element [value].

        points: Limit to only elements of the specified points; may be None for
        all points, a single ID, or a list of IDs.

        features: Limit to only elements of the specified features; may be None for
        all features, a single ID, or a list of IDs.

        preserveZeros: If True it does not apply toTransform to elements in
        the data that are 0, and that 0 is not modified.

        skipNoneReturnValues: If True, any time toTransform() returns None, the
        value originally in the data will remain unmodified.

        """
        if points is not None and not isinstance(points, list):
            if not isinstance(points, (int, six.string_types)):
                raise ArgumentException(
                    "Only allowable inputs to 'points' parameter is an int ID, a list of int ID's, or None")
            points = [points]

        if features is not None and not isinstance(features, list):
            if not isinstance(features, (int, six.string_types)):
                raise ArgumentException(
                    "Only allowable inputs to 'features' parameter is an ID, a list of int ID's, or None")
            features = [features]

        if points is not None:
            points = copy.copy(points)
            for i in range(len(points)):
                points[i] = self._getPointIndex(points[i])

        if features is not None:
            features = copy.copy(features)
            for i in range(len(features)):
                features[i] = self._getFeatureIndex(features[i])

        self.validate()

        self._transformEachElement_implementation(toTransform, points, features, preserveZeros, skipNoneReturnValues)


    def fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd):
        """
        Revise the contents of the calling object so that it contains the provided
        values in the given location.

        values - Either a constant value or a UML object whose size is consistent
        with the given start and end indices.

        pointStart - the inclusive ID of the first point in the calling object
        whose contents will be modified.

        featureStart - the inclusive ID of the first feature in the calling object
        whose contents will be modified.

        pointEnd - the inclusive ID of the last point in the calling object
        whose contents will be modified.

        featureEnd - the inclusive ID of the last feature in the calling object
        whose contents will be modified.

        """
        psIndex = self._getPointIndex(pointStart)
        peIndex = self._getPointIndex(pointEnd)
        fsIndex = self._getFeatureIndex(featureStart)
        feIndex = self._getFeatureIndex(featureEnd)

        if psIndex > peIndex:
            msg = "pointStart (" + str(pointStart) + ") must be less than or "
            msg += "equal to pointEnd (" + str(pointEnd) + ")."
            raise ArgumentException(msg)
        if fsIndex > feIndex:
            msg = "featureStart (" + str(featureStart) + ") must be less than or "
            msg += "equal to featureEnd (" + str(featureEnd) + ")."
            raise ArgumentException(msg)

        if isinstance(values, UML.data.Base):
            prange = (peIndex - psIndex) + 1
            frange = (feIndex - fsIndex) + 1
            if values.points != prange:
                msg = "When the values argument is a UML data object, the size "
                msg += "of values must match the range of modification. There are "
                msg += str(values.points) + " points in values, yet pointStart ("
                msg += str(pointStart) + ") and pointEnd ("
                msg += str(pointEnd) + ") define a range of length " + str(prange)
                raise ArgumentException(msg)
            if values.features != frange:
                msg = "When the values argument is a UML data object, the size "
                msg += "of values must match the range of modification. There are "
                msg += str(values.features) + " features in values, yet featureStart ("
                msg += str(featureStart) + ") and featureEnd ("
                msg += str(featureEnd) + ") define a range of length " + str(frange)
                raise ArgumentException(msg)
            if values.getTypeString() != self.getTypeString():
                values = values.copyAs(self.getTypeString())

        elif dataHelpers._looksNumeric(values) or isinstance(values, six.string_types):
            pass  # no modificaitons needed
        else:
            msg = "values may only be a UML data object, or a single numeric value, yet "
            msg += "we received something of " + str(type(values))
            raise ArgumentException(msg)

        self._fillWith_implementation(values, psIndex, fsIndex, peIndex, feIndex)
        self.validate()


    def handleMissingValues(self, method='remove points', features=None, arguments=None, alsoTreatAsMissing=[], markMissing=False):
        """
        This function is to remove, replace or impute missing values in an UML container data object.

        method - a str. It can be 'remove points', 'remove features', 'feature mean', 'feature median', 'feature mode', 'zero', 'constant', 'forward fill'
        'backward fill', 'interpolate'

        features - can be None to indicate all features, or a str to indicate the name of a single feature, or an int to indicate
        the index of a feature, or a list of feature names, or a list of features' indices, or a list of mix of feature names and feature indices. In this function, only those features in the input 'features'
        will be processed.

        arguments - for some kind of methods, we need to setup arguments.
        for method = 'remove points', 'remove features', arguments can be 'all' or 'any' etc.
        for method = 'constant', arguments must be a value
        for method = 'interpolate', arguments can be a dict which stores inputs for numpy.interp

        alsoTreatAsMissing -  a list. In this function, numpy.NaN and None are always treated as missing. You can add extra values which
        should be treated as missing values too, in alsoTreatAsMissing.

        markMissing: True or False. If it is True, then extra columns for those features will be added, in which 0 (False) or 1 (True) will be filled to
        indicate if the value in a cell is originally missing or not.
        """
        #convert features to a list of index
        msg = 'features can only be a str, an int, or a list of str or a list of int'
        if features is None:
            featuresList = list(range(self._getfeatureCount()))
        elif isinstance(features, six.string_types):
            featuresList = [self.getFeatureIndex(features)]
        elif isinstance(features, int):
            featuresList = [features]
        elif isinstance(features, list):
            featuresList = []
            for i in features:
                if isinstance(i, six.string_types):
                    featuresList.append(self.getFeatureIndex(i))
                elif isinstance(i, int):
                    featuresList.append(i)
                else:
                    raise ArgumentException(msg)
        else:
            raise ArgumentException(msg)

        #convert single value alsoTreatAsMissing to a list
        if not hasattr(alsoTreatAsMissing, '__len__') or isinstance(alsoTreatAsMissing, six.string_types):
            alsoTreatAsMissing = [alsoTreatAsMissing]

        if isinstance(self, UML.data.DataFrame):
            #for DataFrame, pass column names instead of indices
            featuresList = [self.getFeatureName(i) for i in featuresList]

        self._handleMissingValues_implementation(method, featuresList, arguments, alsoTreatAsMissing, markMissing)


    def _flattenNames(self, discardAxis):
        """
        Helper calculating the axis names for the unflattend axis after a flatten operation.

        """
        self._validateAxis(discardAxis)
        if discardAxis == 'point':
            keepNames = self.getFeatureNames()
            dropNames = self.getPointNames()
        else:
            keepNames = self.getPointNames()
            dropNames = self.getFeatureNames()

        ret = []
        for d in dropNames:
            for k in keepNames:
                ret.append(k + ' | ' + d)

        return ret

    def flattenToOnePoint(self):
        """
        Adjust this object in place so that the same values are all in a single point.

        Each feature in the result maps to exactly one value from the original object.
        The order of values respects the point order from the original object,
        if there were n features in the original, the first n values in the result
        will exactly match the first point, the nth to (2n-1)th values will exactly
        match the original second point, etc. The feature names will be transformed
        such that the value at the intersection of the "pn_i" named point and "fn_j"
        named feature from the original object will have a feature name of "fn_j | pn_i".
        The single point will have a name of "Flattened".

        Raises: ImproperActionException if an axis has length 0

        """
        if self.points == 0:
            msg = "Can only flattenToOnePoint when there is one or more points. " \
                  "This object has 0 points."
            raise ImproperActionException(msg)
        if self.features == 0:
            msg = "Can only flattenToOnePoint when there is one or more features. " \
                  "This object has 0 features."
            raise ImproperActionException(msg)

        # TODO: flatten nameless Objects without the need to generate default names for them.
        if not self._pointNamesCreated():
            self._setAllDefault('point')
        if not self._featureNamesCreated():
            self._setAllDefault('feature')

        self._flattenToOnePoint_implementation()

        self._featureCount = self.points * self.features
        self._pointCount = 1
        self.setFeatureNames(self._flattenNames('point'))
        self.setPointNames(['Flattened'])


    def flattenToOneFeature(self):
        """
        Adjust this object in place so that the same values are all in a single feature.

        Each point in the result maps to exactly one value from the original object.
        The order of values respects the feature order from the original object,
        if there were n points in the original, the first n values in the result
        will exactly match the first feature, the nth to (2n-1)th values will exactly
        match the original second feature, etc. The point names will be transformed
        such that the value at the intersection of the "pn_i" named point and "fn_j"
        named feature from the original object will have a point name of "pn_i | fn_j".
        The single feature will have a name of "Flattened".

        Raises: ImproperActionException if an axis has length 0

        """
        if self.points == 0:
            msg = "Can only flattenToOneFeature when there is one or more points. " \
                  "This object has 0 points."
            raise ImproperActionException(msg)
        if self.features == 0:
            msg = "Can only flattenToOneFeature when there is one or more features. " \
                  "This object has 0 features."
            raise ImproperActionException(msg)

        # TODO: flatten nameless Objects without the need to generate default names for them.
        if not self._pointNamesCreated():
            self._setAllDefault('point')
        if not self._featureNamesCreated():
            self._setAllDefault('feature')

        self._flattenToOneFeature_implementation()

        self._pointCount = self.points * self.features
        self._featureCount = 1
        self.setPointNames(self._flattenNames('feature'))
        self.setFeatureNames(['Flattened'])


    def _unflattenNames(self, addedAxis, addedAxisLength):
        """
        Helper calculating the new axis names after an unflattening operation.

        """
        self._validateAxis(addedAxis)
        if addedAxis == 'point':
            both = self.getFeatureNames()
            keptAxisLength = self.features // addedAxisLength
            allDefault = self._namesAreFlattenFormatConsistent('point', addedAxisLength, keptAxisLength)
        else:
            both = self.getPointNames()
            keptAxisLength = self.points // addedAxisLength
            allDefault = self._namesAreFlattenFormatConsistent('feature', addedAxisLength, keptAxisLength)

        if allDefault:
            addedAxisName = None
            keptAxisName = None
        else:
            # we consider the split of the elements into keptAxisLength chunks (of
            # which there will be addedAxisLength number of chunks), and want the
            # index of the first of each chunk. We allow that first name to be
            # representative for that chunk: all will have the same stuff past
            # the vertical bar.
            locations = range(0, len(both), keptAxisLength)
            addedAxisName = [both[n].split(" | ")[1] for n in locations]
            keptAxisName = [name.split(" | ")[0] for name in both[:keptAxisLength]]

        return addedAxisName, keptAxisName

    def _namesAreFlattenFormatConsistent(self, flatAxis, newFLen, newUFLen):
        """
        Helper which validates the formatting of axis names prior to unflattening.

        Will raise ImproperActionException if an inconsistency with the formatting
        done by the flatten operations is discovered. Returns True if all the names
        along the unflattend axis are default, False otherwise.

        """
        flat = self.getPointNames() if flatAxis == 'point' else self.getFeatureNames()
        formatted = self.getFeatureNames() if flatAxis == 'point' else self.getPointNames()

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
        if not (isDefault or isExact):
            msg = "In order to unflatten this object, the names must be " \
                  "consistent with the results from a flatten call. Therefore, " \
                  "the " + flatAxis + " name for this object ('" + flat[0] + "') must " \
                  "either be a default name or exactly the string 'Flattened'"
            raise ImproperActionException(msg)

        # check the contents of the names along the unflattend axis
        msg = "In order to unflatten this object, the names must be " \
              "consistent with the results from a flatten call. Therefore, " \
              "the " + flatAxis + " names for this object must either be all " \
              "default, or they must be ' | ' split names with name values " \
              "consistent with the positioning from a flatten call."
        # each name - default or correctly formatted
        allDefaultStatus = None
        for name in formatted:
            isDefault = checkIsDefault(name)
            formatCorrect = len(name.split(" | ")) == 2
            if allDefaultStatus is None:
                allDefaultStatus = isDefault
            else:
                if isDefault != allDefaultStatus:
                    raise ImproperActionException(msg)

            if not (isDefault or formatCorrect):
                raise ImproperActionException(msg)

        # consistency only relevant if we have non-default names
        if not allDefaultStatus:
            # seen values - consistent wrt original flattend axis names
            for i in range(newFLen):
                same = formatted[newUFLen*i].split(' | ')[1]
                for name in formatted[newUFLen*i:newUFLen*(i+1)]:
                    if same != name.split(' | ')[1]:
                        raise ImproperActionException(msg)

            # seen values - consistent wrt original unflattend axis names
            for i in range(newUFLen):
                same = formatted[i].split(' | ')[0]
                for j in range(newFLen):
                    name = formatted[i + (j * newUFLen)]
                    if same != name.split(' | ')[0]:
                        raise ImproperActionException(msg)

        return allDefaultStatus


    def unflattenFromOnePoint(self, numPoints):
        """
        Adjust this point vector in place to an object that it could have been flattend from.

        This is an inverse of the method flattenToOnePoint: if an object foo with n
        points calls the flatten method, then this method with n as the argument, the
        result should be identical to the original foo. It is not limited to objects
        that have previously had flattenToOnePoint called on them; any object whose
        structure and names are consistent with a previous call to flattenToOnePoint
        may call this method. This includes objects with all default names.

        Raises: ArgumentException if numPoints does not divide the length of the point
        vector.

        Raises: ImproperActionException if an axis has length 0, there is more than one
        point, or the names are inconsistent with a previous call to flattenToOnePoint.

        """
        if self.features == 0:
            msg = "Can only unflattenFromOnePoint when there is one or more features. " \
                  "This object has 0 features."
            raise ImproperActionException(msg)
        if self.points != 1:
            msg = "Can only unflattenFromOnePoint when there is only one point. " \
                  "This object has " + str(self.points) + " points."
            raise ImproperActionException(msg)
        if self.features % numPoints != 0:
            msg = "The argument numPoints (" + str(numPoints) + ") must be a divisor of " \
                  "this object's featureCount (" + str(self.features) + ") otherwise " \
                  "it will not be possible to equally divide the elements into the desired " \
                  "number of points."
            raise ArgumentException(msg)

        if not self._pointNamesCreated():
            self._setAllDefault('point')
        if not self._featureNamesCreated():
            self._setAllDefault('feature')

        self._unflattenFromOnePoint_implementation(numPoints)
        ret = self._unflattenNames('point', numPoints)
        self._featureCount = self.features // numPoints
        self._pointCount = numPoints
        self.setPointNames(ret[0])
        self.setFeatureNames(ret[1])


    def unflattenFromOneFeature(self, numFeatures):
        """
        Adjust this feature vector in place to an object that it could have been flattend from.

        This is an inverse of the method flattenToOneFeature: if an object foo with n
        features calls the flatten method, then this method with n as the argument, the
        result should be identical to the original foo. It is not limited to objects
        that have previously had flattenToOneFeature called on them; any object whose
        structure and names are consistent with a previous call to flattenToOneFeature
        may call this method. This includes objects with all default names.

        Raises: ArgumentException if numPoints does not divide the length of the point
        vector.

        Raises: ImproperActionException if an axis has length 0, there is more than one
        point, or the names are inconsistent with a previous call to flattenToOnePoint.

        """
        if self.points == 0:
            msg = "Can only unflattenFromOneFeature when there is one or more points. " \
                  "This object has 0 points."
            raise ImproperActionException(msg)
        if self.features != 1:
            msg = "Can only unflattenFromOneFeature when there is only one feature. " \
                  "This object has " + str(self.features) + " features."
            raise ImproperActionException(msg)

        if self.points % numFeatures != 0:
            msg = "The argument numFeatures (" + str(numFeatures) + ") must be a divisor of " \
                  "this object's pointCount (" + str(self.points) + ") otherwise " \
                  "it will not be possible to equally divide the elements into the desired " \
                  "number of features."
            raise ArgumentException(msg)

        if not self._pointNamesCreated():
            self._setAllDefault('point')
        if not self._featureNamesCreated():
            self._setAllDefault('feature')

        self._unflattenFromOneFeature_implementation(numFeatures)
        ret = self._unflattenNames('feature', numFeatures)
        self._pointCount = self.points // numFeatures
        self._featureCount = numFeatures
        self.setPointNames(ret[1])
        self.setFeatureNames(ret[0])


    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################

    def elementwiseMultiply(self, other):
        """
        Perform element wise multiplication of this UML data object against the
        provided other UML data object, with the result being stored in-place in
        the calling object. Both objects must contain only numeric data. The
        pointCount and featureCount of both objects must be equal. The types of
        the two objects may be different. None is always returned.

        """
        if not isinstance(other, UML.data.Base):
            raise ArgumentException("'other' must be an instance of a UML data object")
        # Test element type self
        if self.points > 0:
            for val in self.pointView(0):
                if not dataHelpers._looksNumeric(val):
                    raise ArgumentException("This data object contains non numeric data, cannot do this operation")

        # test element type other
        if other.points > 0:
            for val in other.pointView(0):
                if not dataHelpers._looksNumeric(val):
                    raise ArgumentException("This data object contains non numeric data, cannot do this operation")

        if self.points != other.points:
            raise ArgumentException("The number of points in each object must be equal.")
        if self.features != other.features:
            raise ArgumentException("The number of features in each object must be equal.")

        if self.points == 0 or self.features == 0:
            raise ImproperActionException("Cannot do elementwiseMultiply when points or features is emtpy")

        self._validateEqualNames('point', 'point', 'elementwiseMultiply', other)
        self._validateEqualNames('feature', 'feature', 'elementwiseMultiply', other)

        self._elementwiseMultiply_implementation(other)

        (retPNames, retFNames) = dataHelpers.mergeNonDefaultNames(self, other)
        self.setPointNames(retPNames)
        self.setFeatureNames(retFNames)
        self.validate()

    def elementwisePower(self, other):
        # other is UML or single numerical value
        singleValue = dataHelpers._looksNumeric(other)
        if not singleValue and not isinstance(other, UML.data.Base):
            raise ArgumentException("'other' must be an instance of a UML data object or a single numeric value")

        # Test element type self
        if self.points > 0:
            for val in self.pointView(0):
                if not dataHelpers._looksNumeric(val):
                    raise ArgumentException("This data object contains non numeric data, cannot do this operation")

        # test element type other
        if isinstance(other, UML.data.Base):
            if other.points > 0:
                for val in other.pointView(0):
                    if not dataHelpers._looksNumeric(val):
                        raise ArgumentException("This data object contains non numeric data, cannot do this operation")

            # same shape
            if self.points != other.points:
                raise ArgumentException("The number of points in each object must be equal.")
            if self.features != other.features:
                raise ArgumentException("The number of features in each object must be equal.")

        if self.points == 0 or self.features == 0:
            raise ImproperActionException("Cannot do elementwiseMultiply when points or features is emtpy")

        if isinstance(other, UML.data.Base):
            def powFromRight(val, pnum, fnum):
                return val ** other[pnum, fnum]

            self.transformEachElement(powFromRight)
        else:
            def powFromRight(val, pnum, fnum):
                return val ** other

            self.transformEachElement(powFromRight)

        self.validate()


    def __mul__(self, other):
        """
        Perform matrix multiplication or scalar multiplication on this object depending on
        the input 'other'

        """
        if not isinstance(other, UML.data.Base) and not dataHelpers._looksNumeric(other):
            return NotImplemented

        if self.points == 0 or self.features == 0:
            raise ImproperActionException("Cannot do a multiplication when points or features is empty")

        # Test element type self
        if self.points > 0:
            for val in self.pointView(0):
                if not dataHelpers._looksNumeric(val):
                    raise ArgumentException("This data object contains non numeric data, cannot do this operation")

        # test element type other
        if isinstance(other, UML.data.Base):
            if other.points == 0 or other.features == 0:
                raise ImproperActionException("Cannot do a multiplication when points or features is empty")

            if other.points > 0:
                for val in other.pointView(0):
                    if not dataHelpers._looksNumeric(val):
                        raise ArgumentException("This data object contains non numeric data, cannot do this operation")

            if self.features != other.points:
                raise ArgumentException("The number of features in the calling object must "
                                        + "match the point in the callee object.")

            self._validateEqualNames('feature', 'point', '__mul__', other)

        ret = self._mul__implementation(other)

        if isinstance(other, UML.data.Base):
            ret.setPointNames(self.getPointNames())
            ret.setFeatureNames(other.getFeatureNames())

        pathSource = 'merge' if isinstance(other, UML.data.Base) else 'self'

        dataHelpers.binaryOpNamePathMerge(self, other, ret, None, pathSource)

        return ret

    def __rmul__(self, other):
        """	Perform scalar multiplication with this object on the right """
        if dataHelpers._looksNumeric(other):
            return self.__mul__(other)
        else:
            return NotImplemented

    def __imul__(self, other):
        """
        Perform in place matrix multiplication or scalar multiplication, depending in the
        input 'other'

        """
        ret = self.__mul__(other)
        if ret is not NotImplemented:
            self.referenceDataFrom(ret)
            ret = self

        return ret

    def __add__(self, other):
        """
        Perform addition on this object, element wise if 'other' is a UML data
        object, or element wise with a scalar if other is some kind of numeric
        value.

        """
        return self._genericNumericBinary('__add__', other)

    def __radd__(self, other):
        """ Perform scalar addition with this object on the right """
        return self._genericNumericBinary('__radd__', other)

    def __iadd__(self, other):
        """
        Perform in-place addition on this object, element wise if 'other' is a UML data
        object, or element wise with a scalar if other is some kind of numeric
        value.

        """
        return self._genericNumericBinary('__iadd__', other)

    def __sub__(self, other):
        """
        Subtract from this object, element wise if 'other' is a UML data
        object, or element wise by a scalar if other is some kind of numeric
        value.

        """
        return self._genericNumericBinary('__sub__', other)

    def __rsub__(self, other):
        """
        Subtract each element of this object from the given scalar

        """
        return self._genericNumericBinary('__rsub__', other)

    def __isub__(self, other):
        """
        Subtract (in place) from this object, element wise if 'other' is a UML data
        object, or element wise with a scalar if other is some kind of numeric
        value.

        """
        return self._genericNumericBinary('__isub__', other)

    def __div__(self, other):
        """
        Perform division using this object as the numerator, element wise if 'other'
        is a UML data object, or element wise by a scalar if other is some kind of
        numeric value.

        """
        return self._genericNumericBinary('__div__', other)

    def __rdiv__(self, other):
        """
        Perform element wise division using this object as the denominator, and the
        given scalar value as the numerator

        """
        return self._genericNumericBinary('__rdiv__', other)

    def __idiv__(self, other):
        """
        Perform division (in place) using this object as the numerator, element
        wise if 'other' is a UML data object, or element wise by a scalar if other
        is some kind of numeric value.

        """
        return self._genericNumericBinary('__idiv__', other)

    def __truediv__(self, other):
        """
        Perform true division using this object as the numerator, element wise
        if 'other' is a UML data object, or element wise by a scalar if other is
        some kind of numeric value.

        """
        return self._genericNumericBinary('__truediv__', other)

    def __rtruediv__(self, other):
        """
        Perform element wise true division using this object as the denominator,
        and the given scalar value as the numerator

        """
        return self._genericNumericBinary('__rtruediv__', other)

    def __itruediv__(self, other):
        """
        Perform true division (in place) using this object as the numerator, element
        wise if 'other' is a UML data object, or element wise by a scalar if other
        is some kind of numeric value.

        """
        return self._genericNumericBinary('__itruediv__', other)

    def __floordiv__(self, other):
        """
        Perform floor division using this object as the numerator, element wise
        if 'other' is a UML data object, or element wise by a scalar if other is
        some kind of numeric value.

        """
        return self._genericNumericBinary('__floordiv__', other)

    def __rfloordiv__(self, other):
        """
        Perform element wise floor division using this object as the denominator,
        and the given scalar value as the numerator

        """
        return self._genericNumericBinary('__rfloordiv__', other)

    def __ifloordiv__(self, other):
        """
        Perform floor division (in place) using this object as the numerator, element
        wise if 'other' is a UML data object, or element wise by a scalar if other
        is some kind of numeric value.

        """
        return self._genericNumericBinary('__ifloordiv__', other)

    def __mod__(self, other):
        """
        Perform mod using the elements of this object as the dividends, element wise
        if 'other' is a UML data object, or element wise by a scalar if other is
        some kind of numeric value.

        """
        return self._genericNumericBinary('__mod__', other)

    def __rmod__(self, other):
        """
        Perform mod using the elements of this object as the divisors, and the
        given scalar value as the dividend

        """
        return self._genericNumericBinary('__rmod__', other)

    def __imod__(self, other):
        """
        Perform mod (in place) using the elements of this object as the dividends,
        element wise if 'other' is a UML data object, or element wise by a scalar
        if other is some kind of numeric value.

        """
        return self._genericNumericBinary('__imod__', other)

    @to2args
    def __pow__(self, other, z):
        """
        Perform exponentiation (iterated __mul__) using the elements of this object
        as the bases, element wise if 'other' is a UML data object, or element wise
        by a scalar if other is some kind of numeric value.

        """
        if self.points == 0 or self.features == 0:
            raise ImproperActionException("Cannot do ** when points or features is empty")
        if not dataHelpers._looksNumeric(other):
            raise ArgumentException("'other' must be an instance of a scalar")
        if other != int(other):
            raise ArgumentException("other may only be an integer type")
        if other < 0:
            raise ArgumentException("other must be greater than zero")

        retPNames = self.getPointNames()
        retFNames = self.getFeatureNames()

        if other == 1:
            ret = self.copy()
            ret._name = dataHelpers.nextDefaultObjectName()
            return ret

        # exact conditions in which we need to instantiate this object
        if other == 0 or other % 2 == 0:
            identity = UML.createData(self.getTypeString(), numpy.eye(self.points),
                                      pointNames=retPNames, featureNames=retFNames)
        if other == 0:
            return identity

        # this means that we don't start with a multiplication at the ones place,
        # so we need to reserve the identity as the in progress return value
        if other % 2 == 0:
            ret = identity
        else:
            ret = self.copy()

        # by setting up ret, we've taken care of the original ones place
        curr = other >> 1
        # the running binary exponent we've calculated. We've done the ones
        # place, so this is just a copy
        running = self.copy()

        while curr != 0:
            running = running._matrixMultiply_implementation(running)
            if (curr % 2) == 1:
                ret = ret._matrixMultiply_implementation(running)

            # shift right to put the next digit in the ones place
            curr = curr >> 1

        ret.setPointNames(retPNames)
        ret.setFeatureNames(retFNames)

        ret._name = dataHelpers.nextDefaultObjectName()

        return ret

    def __ipow__(self, other):
        """
        Perform in-place exponentiation (iterated __mul__) using the elements
        of this object as the bases, element wise if 'other' is a UML data
        object, or element wise by a scalar if other is some kind of numeric
        value.

        """
        ret = self.__pow__(other)
        self.referenceDataFrom(ret)
        return self

    def __pos__(self):
        """ Return this object. """
        ret = self.copy()
        ret._name = dataHelpers.nextDefaultObjectName()

        return ret

    def __neg__(self):
        """ Return this object where every element has been multiplied by -1 """
        ret = self.copy()
        ret *= -1
        ret._name = dataHelpers.nextDefaultObjectName()

        return ret

    def __abs__(self):
        """ Perform element wise absolute value on this object """
        ret = self.calculateForEachElement(abs)
        ret.setPointNames(self.getPointNames())
        ret.setFeatureNames(self.getFeatureNames())

        ret._name = dataHelpers.nextDefaultObjectName()
        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath
        return ret

    def _genericNumericBinary_validation(self, opName, other):
        isUML = isinstance(other, UML.data.Base)

        if not isUML and not dataHelpers._looksNumeric(other):
            raise ArgumentException("'other' must be an instance of a UML data object or a scalar")

        # Test element type self
        if self.points > 0:
            for val in self.pointView(0):
                if not dataHelpers._looksNumeric(val):
                    raise ArgumentException("This data object contains non numeric data, cannot do this operation")

        # test element type other
        if isUML:
            if opName.startswith('__r'):
                return NotImplemented
            if other.points > 0:
                for val in other.pointView(0):
                    if not dataHelpers._looksNumeric(val):
                        raise ArgumentException("This data object contains non numeric data, cannot do this operation")

            if self.points != other.points:
                msg = "The number of points in each object must be equal. "
                msg += "(self=" + str(self.points) + " vs other="
                msg += str(other.points) + ")"
                raise ArgumentException(msg)
            if self.features != other.features:
                raise ArgumentException("The number of features in each object must be equal.")

        if self.points == 0 or self.features == 0:
            raise ImproperActionException("Cannot do " + opName + " when points or features is empty")

        # check name restrictions
        if isUML:
            if self._pointNamesCreated() and other._pointNamesCreated is not None:
                self._validateEqualNames('point', 'point', opName, other)
            if self._featureNamesCreated() and other._featureNamesCreated():
                self._validateEqualNames('feature', 'feature', opName, other)

        divNames = ['__div__', '__rdiv__', '__idiv__', '__truediv__', '__rtruediv__',
                    '__itruediv__', '__floordiv__', '__rfloordiv__', '__ifloordiv__',
                    '__mod__', '__rmod__', '__imod__', ]
        if isUML and opName in divNames:
            if other.containsZero():
                raise ZeroDivisionError("Cannot perform " + opName + " when the second argument"
                                        + "contains any zeros")
            if isinstance(other, UML.data.Matrix):
                if False in numpy.isfinite(other.data):
                    raise ArgumentException("Cannot perform " + opName + " when the second argument"
                                            + "contains any NaNs or Infs")
        if not isUML and opName in divNames:
            if other == 0:
                msg = "Cannot perform " + opName + " when the second argument"
                msg += + "is zero"
                raise ZeroDivisionError(msg)

    def _genericNumericBinary(self, opName, other):
        ret = self._genericNumericBinary_validation(opName, other)
        if ret == NotImplemented:
            return ret

        isUML = isinstance(other, UML.data.Base)

        # figure out return obj's point / feature names
        # if unary:
        (retPNames, retFNames) = (None, None)

        if opName in ['__pos__', '__neg__', '__abs__'] or not isUML:
            if self._pointNamesCreated():
                retPNames = self.getPointNames()
            if self._featureNamesCreated():
                retFNames = self.getFeatureNames()
        # else (everything else that uses this helper is a binary scalar op)
        else:
            (retPNames, retFNames) = dataHelpers.mergeNonDefaultNames(self, other)

        ret = self._genericNumericBinary_implementation(opName, other)

        if retPNames is not None:
            ret.setPointNames(retPNames)
        else:
            ret.setPointNames(None)

        if retFNames is not None:
            ret.setFeatureNames(retFNames)
        else:
            ret.setFeatureNames(None)

        nameSource = 'self' if opName.startswith('__i') else None
        pathSource = 'merge' if isUML else 'self'
        dataHelpers.binaryOpNamePathMerge(
            self, other, ret, nameSource, pathSource)
        return ret

    def _genericNumericBinary_implementation(self, opName, other):
        startType = self.getTypeString()
        implName = opName[1:] + 'implementation'
        if startType == 'Matrix' or startType == 'DataFrame':
            toCall = getattr(self, implName)
            ret = toCall(other)
        else:
            selfConv = self.copyAs("Matrix")
            toCall = getattr(selfConv, implName)
            ret = toCall(other)
            if opName.startswith('__i'):
                ret = ret.copyAs(startType)
                self.referenceDataFrom(ret)
                ret = self
            else:
                ret = UML.createData(startType, ret.data)

        return ret

    #################################
    #################################
    ###   Statistical functions   ###
    #################################
    #################################


    def pointSimilarities(self, similarityFunction):
        """ """
        return self._axisSimilaritiesBackend(similarityFunction, 'point')

    def featureSimilarities(self, similarityFunction):
        """ """
        return self._axisSimilaritiesBackend(similarityFunction, 'feature')

    def _axisSimilaritiesBackend(self, similarityFunction, axis):
        acceptedPretty = [
            'correlation', 'covariance', 'dot product', 'sample covariance',
            'population covariance'
        ]
        accepted = list(map(dataHelpers.cleanKeywordInput, acceptedPretty))

        msg = "The similarityFunction must be equivaltent to one of the "
        msg += "following: "
        msg += str(acceptedPretty) + ", but '" + str(similarityFunction)
        msg += "' was given instead. Note: casing and whitespace is "
        msg += "ignored when checking the input."

        if not isinstance(similarityFunction, six.string_types):
            raise ArgumentException(msg)

        cleanFuncName = dataHelpers.cleanKeywordInput(similarityFunction)

        if cleanFuncName not in accepted:
            raise ArgumentException(msg)

        if cleanFuncName == 'correlation':
            toCall = UML.calculate.correlation
        elif cleanFuncName == 'covariance' or cleanFuncName == 'samplecovariance':
            toCall = UML.calculate.covariance
        elif cleanFuncName == 'populationcovariance':
            def populationCovariance(X, X_T):
                return UML.calculate.covariance(X, X_T, False)

            toCall = populationCovariance
        elif cleanFuncName == 'dotproduct':
            def dotProd(X, X_T):
                return X * X_T

            toCall = dotProd

        transposed = self.copy()
        transposed.transpose()

        if axis == 'point':
            ret = toCall(self, transposed)
        else:
            ret = toCall(transposed, self)

        # TODO validation or result.

        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath

        return ret

    def pointStatistics(self, statisticsFunction):
        """ """
        return self._axisStatisticsBackend(statisticsFunction, 'point')

    def featureStatistics(self, statisticsFunction, groupByFeature=None):
        """ """
        if groupByFeature is None:
            return self._axisStatisticsBackend(statisticsFunction, 'feature')
        else:
            res = self.groupByFeature(groupByFeature)
            for k in res:
                res[k] = res[k]._axisStatisticsBackend(statisticsFunction, 'feature')
            return res

    def _axisStatisticsBackend(self, statisticsFunction, axis):
        cleanFuncName = self._validateStatisticalFunctionInputString(statisticsFunction)

        if cleanFuncName == 'max':
            toCall = UML.calculate.maximum
        elif cleanFuncName == 'mean':
            toCall = UML.calculate.mean
        elif cleanFuncName == 'median':
            toCall = UML.calculate.median
        elif cleanFuncName == 'min':
            toCall = UML.calculate.minimum
        elif cleanFuncName == 'uniquecount':
            toCall = UML.calculate.uniqueCount
        elif cleanFuncName == 'proportionmissing':
            toCall = UML.calculate.proportionMissing
        elif cleanFuncName == 'proportionzero':
            toCall = UML.calculate.proportionZero
        elif cleanFuncName == 'std' or cleanFuncName == 'standarddeviation':
            def sampleStandardDeviation(values):
                return UML.calculate.standardDeviation(values, True)

            toCall = sampleStandardDeviation
        elif cleanFuncName == 'samplestd' or cleanFuncName == 'samplestandarddeviation':
            def sampleStandardDeviation(values):
                return UML.calculate.standardDeviation(values, True)

            toCall = sampleStandardDeviation
        elif cleanFuncName == 'populationstd' or cleanFuncName == 'populationstandarddeviation':
            toCall = UML.calculate.standardDeviation

        if axis == 'point':
            ret = self.calculateForEachPoint(toCall)
            ret.setPointNames(self.getPointNames())
            ret.setFeatureName(0, cleanFuncName)
        else:
            ret = self.calculateForEachFeature(toCall)
            ret.setPointName(0, cleanFuncName)
            ret.setFeatureNames(self.getFeatureNames())
        return ret

    ############################
    ############################
    ###   Helper functions   ###
    ############################
    ############################

    def _genericStructuralFrontend(self, structure, axis, target=None, start=None,
                                   end=None, number=None, randomize=False):
        if axis == 'point':
            getIndex = self._getPointIndex
            axisLength = self.points
            hasNameChecker1, hasNameChecker2 = self.hasPointName, self.hasFeatureName
            viewIterator = self.copy().pointIterator
        else:
            getIndex = self._getFeatureIndex
            axisLength = self.features
            hasNameChecker1, hasNameChecker2 = self.hasFeatureName, self.hasPointName
            viewIterator = self.copy().featureIterator

        if number is not None and number < 1:
            msg = "number must be greater than zero"
            raise ArgumentException(msg)
        if target is not None:
            if start is not None or end is not None:
                raise ArgumentException("Range removal is exclusive, to use it, target must be None")
            if isinstance(target, six.string_types):
                if hasNameChecker1(target):
                    target = getIndex(target)
                    targetList = [target]
                #if axis=point and target is not a point name, or
                # if axis=feature and target is not a feature name,
                # then check if it's a valid query string
                else:
                    optrDict = {'<=': operator.le, '>=': operator.ge, '!=': operator.ne, '==': operator.eq, \
                                        '=': operator.eq, '<': operator.lt, '>': operator.gt}
                    for optr in ['<=', '>=', '!=', '==', '=', '<', '>']:
                        if optr in target:
                            targetList = target.split(optr)
                            optr = '==' if optr == '=' else optr
                            #after splitting at the optr, 2 items must be in the list
                            if len(targetList) != 2:
                                msg = "the target(%s) is a query string but there is an error" % target
                                raise ArgumentException(msg)
                            nameOfFeatureOrPoint, valueOfFeatureOrPoint = targetList
                            nameOfFeatureOrPoint = nameOfFeatureOrPoint.strip()
                            valueOfFeatureOrPoint = valueOfFeatureOrPoint.strip()

                            #when axis=point, check if the feature exists or not
                            #when axis=feature, check if the point exists or not
                            if not hasNameChecker2(nameOfFeatureOrPoint):
                                msg = "the %s %s doesn't exist" % (
                                'feature' if axis == 'point' else 'point', nameOfFeatureOrPoint)
                                raise ArgumentException(msg)

                            optrOperator = optrDict[optr]
                            #convert valueOfFeatureOrPoint from a string, if possible
                            try:
                                valueOfFeatureOrPoint = float(valueOfFeatureOrPoint)
                            except ValueError:
                                pass
                            #convert query string to a function
                            def target_f(x):
                                return optrOperator(x[nameOfFeatureOrPoint], valueOfFeatureOrPoint)

                            target_f.vectorized = True
                            target_f.nameOfFeatureOrPoint = nameOfFeatureOrPoint
                            target_f.valueOfFeatureOrPoint = valueOfFeatureOrPoint
                            target_f.optr = optrOperator
                            target = target_f
                            break
                    #if the target can't be converted to a function
                    if isinstance(target, six.string_types):
                        msg = 'the target is not a valid point name nor a valid query string'
                        raise ArgumentException(msg)
            if isinstance(target, (int, numpy.int, numpy.int64)):
                targetList = [target]
            if isinstance(target, list):
                #verify everything in list is a valid index and convert names into indices
                targetList = []
                for identifier in target:
                    targetList.append(getIndex(identifier))
            # boolean function
            elif hasattr(target, '__call__'):
                if structure == 'retain':
                    targetFunction = target
                    def complement(*args):
                        return not targetFunction(*args)
                    target = complement
                # construct list from function
                targetList = []
                for targetID, view in enumerate(viewIterator()):
                    if target(view):
                        targetList.append(targetID)

        elif start is not None or end is not None:
            start = 0 if start is None else getIndex(start)
            end = axisLength - 1 if end is None else getIndex(end)

            if start < 0 or start > axisLength:
                msg = "start must be a valid index, in the range of possible "
                msg += axis + 's'
                raise ArgumentException(msg)
            if end < 0 or end > axisLength:
                msg = "end must be a valid index, in the range of possible "
                msg += axis + 's'
                raise ArgumentException(msg)
            if start > end:
                raise ArgumentException("The start index cannot be greater than the end index")

            # end + 1 because our range is inclusive
            targetList = list(range(start,end + 1))

        elif number is not None:
            targetList = list(range(0, number))
        else:
            targetName = "to" + structure.capitalize()
            msg = "You must provide a value for " + targetName + ", or start/end, or "
            msg += "number. "
            raise ArgumentException(msg)

        if randomize:
            targetList = pythonRandom.sample(targetList, number)
            targetList.sort()
        if number is not None:
            targetList = targetList[:number]

        if structure == 'count':
            return len(targetList)
        else:
            return self._structuralBackend_implementation(structure, axis, targetList)


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

    def _arrangeFeatureNames(self, maxWidth, nameLength, colSep, colHold, nameHold):
        """Prepare feature names for string output. Grab only those names that
        fit according to the given width limitation, process them for length,
        omit them if they are default. Returns a list of prepared names, and
        a list of the length of each name in the return.

        """
        colHoldWidth = len(colHold)
        colHoldTotal = len(colSep) + colHoldWidth
        nameCutIndex = nameLength - len(nameHold)

        lNames, rNames = [], []

        # total width will always include the column placeholder column,
        # until it is shown that it isn't needed
        totalWidth = colHoldTotal

        # going to add indices from the beginning and end of the data until
        # we've used up our available space, or we've gone through all of
        # the columns. currIndex makes use of negative indices, which is
        # why the end condition makes use of an exact stop value, which
        # varies between positive and negative depending on the number of
        # features
        endIndex = self.features // 2
        if self.features % 2 == 1:
            endIndex *= -1
            endIndex -= 1
        currIndex = 0
        numAdded = 0
        while totalWidth < maxWidth and currIndex != endIndex:
            nameIndex = currIndex
            if currIndex < 0:
                nameIndex = self.features + currIndex

            currName = self.getFeatureName(nameIndex)

            if currName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
                currName = ""
            if len(currName) > nameLength:
                currName = currName[:nameCutIndex] + nameHold
            currWidth = len(currName)

            currNames = lNames if currIndex >= 0 else rNames

            totalWidth += currWidth + len(colSep)
            # test: total width is under max without column holder
            rawStillUnder = totalWidth - (colHoldTotal) < maxWidth
            # test: the column we are trying to add is the last one possible
            allCols = rawStillUnder and (numAdded == (self.features - 1))
            # only add this column if it won't put us over the limit,
            # OR if it is the last one (and under the limit without the col
            # holder)
            if totalWidth < maxWidth or allCols:
                numAdded += 1
                currNames.append(currName)

                # the width value goes in different lists depending on the index
                if currIndex < 0:
                    currIndex = abs(currIndex)
                else:
                    currIndex = (-1 * currIndex) - 1

        # combine the tables. Have to reverse rTable because entries were appended
        # in a right to left order
        rNames.reverse()
        if numAdded == self.features:
            lNames += rNames
        else:
            lNames += [colHold] + rNames

        return lNames

    def _arrangePointNames(self, maxRows, nameLength, rowHolder, nameHold):
        """Prepare point names for string output. Grab only those names that
        fit according to the given row limitation, process them for length,
        omit them if they are default. Returns a list of prepared names, and
        a int bounding the length of each name representation.

        """
        names = []
        pnamesWidth = 0
        nameCutIndex = nameLength - len(nameHold)
        (tRowIDs, bRowIDs) = dataHelpers.indicesSplit(maxRows, self.points)

        # we pull indices from two lists: tRowIDs and bRowIDs
        for sourceIndex in range(2):
            source = list([tRowIDs, bRowIDs])[sourceIndex]

            # add in the rowHolder, if needed
            if sourceIndex == 1 and len(bRowIDs) + len(tRowIDs) < self.points:
                names.append(rowHolder)

            for i in source:
                pname = self.getPointName(i)
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

    def _arrangeDataWithLimits(self, maxWidth, maxHeight, sigDigits=3,
                               maxStrLength=19, colSep=' ', colHold='--', rowHold='|', strHold='...'):
        """
        Arrange the data in this object into a table structure, while
        respecting the given boundaries. If there is more data than
        what fits within the limitations, then omit points or features
        from the middle portions of the data.

        Returns a list of list of strings. The length of the outer list
        is less than or equal to maxHeight. The length of the inner lists
        will all be the same, a length we will designate as n. The sum of
        the individual strings in each inner list will be less than or
        equal to maxWidth - ((n-1) * len(colSep)).

        """
        if self.points == 0 or self.features == 0:
            return [[]], []

        if maxHeight < 2 and maxHeight != self.points:
            msg = "If the number of points in this object is two or greater, "
            msg += "then we require that the input argument maxHeight also "
            msg += "be greater than or equal to two."
            raise ArgumentException(msg)

        cHoldWidth = len(colHold)
        cHoldTotal = len(colSep) + cHoldWidth
        nameCutIndex = maxStrLength - len(strHold)

        #setup a bundle of default values
        if maxHeight is None:
            maxHeight = self.points
        if maxWidth is None:
            maxWidth = float('inf')

        maxRows = min(maxHeight, self.points)
        maxDataRows = maxRows

        (tRowIDs, bRowIDs) = dataHelpers.indicesSplit(maxDataRows, self.points)
        combinedRowIDs = tRowIDs + bRowIDs
        if len(combinedRowIDs) < self.points:
            rowHolderIndex = len(tRowIDs)
        else:
            rowHolderIndex = sys.maxsize

        lTable, rTable = [], []
        lColWidths, rColWidths = [], []

        # total width will always include the column placeholder column,
        # until it is shown that it isn't needed
        totalWidth = cHoldTotal

        # going to add indices from the beginning and end of the data until
        # we've used up our available space, or we've gone through all of
        # the columns. currIndex makes use of negative indices, which is
        # why the end condition makes use of an exact stop value, which
        # varies between positive and negative depending on the number of
        # features
        endIndex = self.features // 2
        if self.features % 2 == 1:
            endIndex *= -1
            endIndex -= 1
        currIndex = 0
        numAdded = 0
        while totalWidth < maxWidth and currIndex != endIndex:
            currWidth = 0
            currTable = lTable if currIndex >= 0 else rTable
            currCol = []

            # check all values in this column (in the accepted rows)
            for i in range(len(combinedRowIDs)):
                rID = combinedRowIDs[i]
                val = self[rID, currIndex]
                valFormed = formatIfNeeded(val, sigDigits)
                valLimited = valFormed if len(valFormed) < maxStrLength else valFormed[:nameCutIndex] + strHold
                valLen = len(valLimited)
                if valLen > currWidth:
                    currWidth = valLen

                # If these are equal, it is time to add the holders
                if i == rowHolderIndex:
                    currCol.append(rowHold)

                currCol.append(valLimited)

            totalWidth += currWidth + len(colSep)
            # test: total width is under max without column holder
            allCols = totalWidth - (cHoldTotal) < maxWidth
            # test: the column we are trying to add is the last one possible
            allCols = allCols and (numAdded == (self.features - 1))
            # only add this column if it won't put us over the limit
            if totalWidth < maxWidth or allCols:
                numAdded += 1
                for i in range(len(currCol)):
                    if len(currTable) != len(currCol):
                        currTable.append([currCol[i]])
                    else:
                        currTable[i].append(currCol[i])

                # the width value goes in different lists depending on the index
                if currIndex < 0:
                    currIndex = abs(currIndex)
                    rColWidths.append(currWidth)
                else:
                    currIndex = (-1 * currIndex) - 1
                    lColWidths.append(currWidth)

        # combine the tables. Have to reverse rTable because entries were appended
        # in a right to left order
        rColWidths.reverse()
        if numAdded == self.features:
            lColWidths += rColWidths
        else:
            lColWidths += [cHoldWidth] + rColWidths
        for rowIndex in range(len(lTable)):
            if len(rTable) > 0:
                rTable[rowIndex].reverse()
                toAdd = rTable[rowIndex]
            else:
                toAdd = []

            if numAdded == self.features:
                lTable[rowIndex] += toAdd
            else:
                lTable[rowIndex] += [colHold] + toAdd

        return lTable, lColWidths

    def _defaultNamesGeneration_NamesSetOperations(self, other, axis):
        """
        TODO: Find a shorter descriptive name.
        TODO: Should we place this function in dataHelpers.py?
        """
        if axis == 'point':
            if self.pointNames is None:
                self._setAllDefault('point')
            if other.pointNames is None:
                other._setAllDefault('point')
        elif axis == 'feature':
            if self.featureNames is None:
                self._setAllDefault('feature')
            if other.featureNames is None:
                other._setAllDefault('feature')
        else:
            raise ArgumentException("invalid axis")

    def _pointNameDifference(self, other):
        """
        Returns a set containing those pointNames in this object that are not also in the input object.

        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            raise ArgumentException("Must provide another representation type to determine pointName difference")

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) - six.viewkeys(other.pointNames)

    def _featureNameDifference(self, other):
        """
        Returns a set containing those featureNames in this object that are not also in the input object.

        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            raise ArgumentException("Must provide another representation type to determine featureName difference")

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return six.viewkeys(self.featureNames) - six.viewkeys(other.featureNames)

    def _pointNameIntersection(self, other):
        """
        Returns a set containing only those pointNames that are shared by this object and the input object.

        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            raise ArgumentException("Must provide another representation type to determine pointName intersection")

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) & six.viewkeys(other.pointNames)

    def _featureNameIntersection(self, other):
        """
        Returns a set containing only those featureNames that are shared by this object and the input object.

        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            raise ArgumentException("Must provide another representation type to determine featureName intersection")

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return six.viewkeys(self.featureNames) & six.viewkeys(other.featureNames)


    def _pointNameSymmetricDifference(self, other):
        """
        Returns a set containing only those pointNames not shared between this object and the input object.

        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            raise ArgumentException("Must provide another representation type to determine pointName difference")

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) ^ six.viewkeys(other.pointNames)

    def _featureNameSymmetricDifference(self, other):
        """
        Returns a set containing only those featureNames not shared between this object and the input object.

        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            raise ArgumentException("Must provide another representation type to determine featureName difference")

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return six.viewkeys(self.featureNames) ^ six.viewkeys(other.featureNames)

    def _pointNameUnion(self, other):
        """
        Returns a set containing all pointNames in either this object or the input object.

        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            raise ArgumentException("Must provide another representation type to determine pointNames union")

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) | six.viewkeys(other.pointNames)

    def _featureNameUnion(self, other):
        """
        Returns a set containing all featureNames in either this object or the input object.

        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            raise ArgumentException("Must provide another representation type to determine featureName union")

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return six.viewkeys(self.featureNames) | six.viewkeys(other.featureNames)


    def _equalPointNames(self, other):
        if other is None or not isinstance(other, Base):
            return False
        return self._equalNames(self.getPointNames(), other.getPointNames())

    def _equalFeatureNames(self, other):
        if other is None or not isinstance(other, Base):
            return False
        return self._equalNames(self.getFeatureNames(), other.getFeatureNames())

    def _equalNames(self, selfNames, otherNames):
        """Private function to determine equality of either pointNames of
        featureNames. It ignores equality of default values, considering only
        whether non default names consistent (position by position) and
        uniquely positioned (if a non default name is present in both, then
        it is in the same position in both).

        """
        if len(selfNames) != len(otherNames):
            return False

        unequalNames = self._unequalNames(selfNames, otherNames)
        return unequalNames == {}

    def _validateEqualNames(self, leftAxis, rightAxis, callSym, other):

        def _validateEqualNames_implementation():
            lnames = self.getPointNames() if leftAxis == 'point' else self.getFeatureNames()
            rnames = other.getPointNames() if rightAxis == 'point' else other.getFeatureNames()
            inconsistencies = self._inconsistentNames(lnames, rnames)

            if inconsistencies != {}:
                table = [['left', 'ID', 'right']]
                for i in sorted(inconsistencies.keys()):
                    lname = '"' + lnames[i] + '"'
                    rname = '"' + rnames[i] + '"'
                    table.append([lname, str(i), rname])

                msg = leftAxis + " to " + rightAxis + " name inconsistencies when "
                msg += "calling left." + callSym + "(right) \n"
                msg += UML.logger.tableString.tableString(table)
                print(msg, file=sys.stderr)
                raise ArgumentException(msg)

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
        between the given two sets. It ignores equality of default values,
        considering only whether non default names consistent (position by
        position) and uniquely positioned (if a non default name is present
        in both, then it is in the same position in both). The return value
        is a dict between integer IDs and the pair of offending names at
        that position in both objects.

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
                            ret[rightNames[lname]] = (lname, rname)


        # check both name directions
        checkFromLeftKeys(inconsistencies, selfNames, otherNames)
        checkFromLeftKeys(inconsistencies, otherNames, selfNames)

        return inconsistencies


    def _unequalNames(self, selfNames, otherNames):
        """Private function to find and return all name inconsistencies
        between the given two sets. It ignores equality of default values,
        considering only whether non default names consistent (position by
        position) and uniquely positioned (if a non default name is present
        in both, then it is in the same position in both). The return value
        is a dict between integer IDs and the pair of offending names at
        that position in both objects.

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


    def _validateReorderedNames(self, axis, callSym, other):
        """
        Validate axis names to check to see if they are equal ignoring order.
        Returns True if the names are equal but reordered, False if they are
        equal and the same order (ignoring defaults), and raises an exception
        if they do not share exactly the same names, or requires reordering in
        the presence of default names.
        """
        if axis == 'point':
            lnames = self.getPointNames()
            rnames = other.getPointNames()
            lGetter = self.getPointIndex
            rGetter = other.getPointIndex
        else:
            lnames = self.getFeatureNames()
            rnames = other.getFeatureNames()
            lGetter = self.getFeatureIndex
            rGetter = other.getFeatureIndex

        inconsistencies = self._inconsistentNames(lnames, rnames)

        if len(inconsistencies) != 0:
            # check for the presence of default names; we don't allow reordering
            # in that case.
            msgBase = "When calling caller." + callSym + "(callee) we require that the "
            msgBase += axis + " names all contain the same names, regardless of order."
            msg = copy.copy(msgBase)
            msg += "However, when default names are present, we don't allow reordering "
            msg += "to occur: either all names must be specified, or the order must be "
            msg += "the same."

            if True in [x[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX for x in lnames]:
                raise ArgumentException(msg)
            if True in [x[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX for x in rnames]:
                raise ArgumentException(msg)

            ldiff = numpy.setdiff1d(lnames, rnames, assume_unique=True)
            # names are not the same.
            if len(ldiff) != 0:
                rdiff = numpy.setdiff1d(rnames, lnames, assume_unique=True)
                msgBase += "Yet, the following names were unmatched (caller names "
                msgBase += "on the left, callee names on the right):\n"

                table = [['ID', 'name', '', 'ID', 'name']]
                for i, (lname, rname) in enumerate(zip(ldiff, rdiff)):
                    table.append([lGetter(lname), lname, "   ", rGetter(rname), rname])

                msg += UML.logger.tableString.tableString(table)
                print(msg, file=sys.stderr)

                raise ArgumentException(msg)
            else:  # names are not different, but are reordered
                return True
        else:  # names exactly equal
            return False


    def _getPointIndex(self, identifier):
        return self._getIndex(identifier, 'point')

    def _getFeatureIndex(self, identifier):
        return self._getIndex(identifier, 'feature')

    def _getIndex(self, identifier, axis):
        num = len(self.getPointNames()) if axis == 'point' else len(self.getFeatureNames())
        nameGetter = self.getPointIndex if axis == 'point' else self.getFeatureIndex
        accepted = (six.string_types, int, numpy.integer)

        toReturn = identifier
        if num == 0:
            msg = "There are no valid " + axis + " identifiers; this object has 0 "
            msg += axis + "s"
            raise ArgumentException(msg)
        if identifier is None:
            msg = "An identifier cannot be None."
            raise ArgumentException(msg)
        if not isinstance(identifier, accepted):
            axisCount = self.points if axis == 'point' else self.features
            msg = "The identifier must be either a string (a valid " + axis
            msg += " name) or an integer (python or numpy) index between 0 and "
            msg += str(axisCount - 1) + " inclusive. Instead we got: " + str(identifier)
            raise ArgumentException(msg)
        if isinstance(identifier, (int, numpy.integer)):
            if identifier < 0:
                identifier = num + identifier
                toReturn = identifier
            if identifier < 0 or identifier >= num:
                msg = "The given index " + str(identifier) + " is outside of the range "
                msg += "of possible indices in the " + axis + " axis (0 to "
                msg += str(num - 1) + ")."
                raise ArgumentException(msg)
        if isinstance(identifier, six.string_types):
            try:
                toReturn = nameGetter(identifier)
            except KeyError:
                msg = "The " + axis + " name '" + identifier + "' cannot be found."
                raise ArgumentException(msg)
        return toReturn


    def _nextDefaultName(self, axis):
        self._validateAxis(axis)
        if axis == 'point':
            ret = DEFAULT_PREFIX2%self._nextDefaultValuePoint
            self._nextDefaultValuePoint += 1
        else:
            ret = DEFAULT_PREFIX2%self._nextDefaultValueFeature
            self._nextDefaultValueFeature += 1
        return ret

    def _setAllDefault(self, axis):
        self._validateAxis(axis)
        if axis == 'point':
            self.pointNames = {}
            self.pointNamesInverse = []
            names = self.pointNames
            invNames = self.pointNamesInverse
            count = self._pointCount
        else:
            self.featureNames = {}
            self.featureNamesInverse = []
            names = self.featureNames
            invNames = self.featureNamesInverse
            count = self._featureCount
        for i in range(count):
            defaultName = self._nextDefaultName(axis)
            invNames.append(defaultName)
            names[defaultName] = i

    def _addPointName(self, pointName):
        if not self._pointNamesCreated():
            self._setAllDefault('point')
        self._addName(pointName, self.pointNames, self.pointNamesInverse, 'point')

    def _addFeatureName(self, featureName):
        if not self._featureNamesCreated():
            self._setAllDefault('feature')
        self._addName(featureName, self.featureNames, self.featureNamesInverse, 'feature')

    def _addName(self, name, selfNames, selfNamesInv, axis):
        """
        Name the next vector outside of the current possible range on the given axis using the
        provided name.

        name may be either a string, or None if you want a default name. If the name is
        not a string, or already being used as another name on this axis, an
        ArgumentException will be raised.

        """
        if name is not None and not isinstance(name, six.string_types):
            raise ArgumentException("The name must be a string")
        if name in selfNames:
            raise ArgumentException("This name is already in use")

        if name is None:
            name = self._nextDefaultName(axis)

        self._incrementDefaultIfNeeded(name, axis)

        numInAxis = len(selfNamesInv)
        selfNamesInv.append(name)
        selfNames[name] = numInAxis

    def _removePointNameAndShift(self, toRemove):
        """
        Removes the specified name from pointNames, changing the indices
        of other pointNames to fill in the missing index.

        toRemove must be a non None string or integer, specifying either a current pointName
        or the index of a current pointName in the given axis.

        """
        self._removeNameAndShift(toRemove, 'point', self.pointNames, self.pointNamesInverse)

    def _removeFeatureNameAndShift(self, toRemove):
        """
        Removes the specified name from featureNames, changing the indices
        of other featureNames to fill in the missing index.

        toRemove must be a non None string or integer, specifying either a current featureNames
        or the index of a current featureNames in the given axis.

        """
        self._removeNameAndShift(toRemove, 'feature', self.featureNames, self.featureNamesInverse)

    def _removeNameAndShift(self, toRemove, axis, selfNames, selfNamesInv):
        """
        Removes the specified name from the name set for the given axis, changing the indices
        of other names to fill in the missing index.

        toRemove must be a non None string or integer, specifying either a current name
        or the index of a current name in the given axis.

        axis must be either 'point' or 'feature'

        selfNames must be the names dict associated with the provided axis in this object

        selfNamesInv must be the indices to names dict associated with the provided axis
        in this object

		"""
        #this will throw the appropriate exceptions, if need be
        index = self._getIndex(toRemove, axis)
        name = selfNamesInv[index]
        numInAxis = len(selfNamesInv)

        del selfNames[name]
        # remapping each index starting with the one we removed
        for i in range(index, numInAxis - 1):
            nextName = selfNamesInv[i + 1]
            selfNames[nextName] = i

        #delete from inverse, since list, del will deal with 'remapping'
        del selfNamesInv[index]

    def _setName_implementation(self, oldIdentifier, newName, axis, allowDefaults=False):
        """
        Changes the featureName specified by previous to the supplied input featureName.

        oldIdentifier must be a non None string or integer, specifying either a current featureName
        or the index of a current featureName. newFeatureName may be either a string not currently
        in the featureName set, or None for an default featureName. newFeatureName may begin with the
        default prefix

        """
        self._validateAxis(axis)
        if axis == 'point':
            names = self.pointNames
            invNames = self.pointNamesInverse
            index = self._getPointIndex(oldIdentifier)
        else:
            names = self.featureNames
            invNames = self.featureNamesInverse
            index = self._getFeatureIndex(oldIdentifier)

        if newName is not None:
            if not isinstance(newName, six.string_types):
                raise ArgumentException("The new name must be either None or a string")
            #			if not allowDefaults and newFeatureName.startswith(DEFAULT_PREFIX):
            #				raise ArgumentException("Cannot manually add a featureName with the default prefix")
        if newName in names:
            if invNames[index] == newName:
                return
            raise ArgumentException("This name '" + newName + "' is already in use")

        if newName is None:
            newName = self._nextDefaultName(axis)

        #remove the current featureName
        oldName = invNames[index]
        del names[oldName]

        # setup the new featureName
        invNames[index] = newName
        names[newName] = index
        self._incrementDefaultIfNeeded(newName, axis)

    def _setNamesFromList(self, assignments, count, axis):
        if axis == 'point':
            def checkAndSet(val):
                if val >= self._nextDefaultValuePoint:
                    self._nextDefaultValuePoint = val + 1
        else:
            def checkAndSet(val):
                if val >= self._nextDefaultValueFeature:
                    self._nextDefaultValueFeature = val + 1

        self._validateAxis(axis)
        if assignments is None:
            self._setAllDefault(axis)
            return
        if not hasattr(assignments, '__getitem__') or not hasattr(assignments, '__len__'):
            msg = "assignments may only be an ordered container type, with "
            msg += "implentations for both __len__ and __getitem__, where "
            msg += "__getitem__ accepts non-negative integers"
            raise ArgumentException(msg)
        if count == 0:
            if len(assignments) > 0:
                msg = "assignments is too large (" + str(len(assignments))
                msg += "); this axis is empty"
                raise ArgumentException(msg)
            self._setNamesFromDict({}, count, axis)
            return
        if len(assignments) != count:
            msg = "assignments may only be an ordered container type, with as "
            msg += "many entries (" + str(len(assignments)) + ") as this axis "
            msg += "is long (" + str(count) + ")"
            raise ArgumentException(msg)
        try:
            assignments[0]
        except IndexError:
            msg = "assignments may only be an ordered container type, with "
            msg += "implentations for both __len__ and __getitem__, where "
            msg += "__getitem__ accepts non-negative integers"
            raise ArgumentException(msg)

        # adjust nextDefaultValue as needed given contents of assignments
        for name in assignments:
            if name is not None and name.startswith(DEFAULT_PREFIX):
                try:
                    num = int(name[DEFAULT_PREFIX_LENGTH:])
                # Case: default prefix with non-integer suffix. This cannot
                # cause a future integer suffix naming collision, so we
                # can ignore it.
                except ValueError:
                    continue
                checkAndSet(num)

        #convert to dict so we only write the checking code once
        temp = {}
        for index in range(len(assignments)):
            name = assignments[index]
            # take this to mean fill it in with a default name
            if name is None:
                name = self._nextDefaultName(axis)
            if name in temp:
                raise ArgumentException("Cannot input duplicate names: " + str(name))
            temp[name] = index
        assignments = temp

        self._setNamesFromDict(assignments, count, axis)

    def _setNamesFromDict(self, assignments, count, axis):
        self._validateAxis(axis)
        if assignments is None:
            self._setAllDefault(axis)
            return
        if not isinstance(assignments, dict):
            raise ArgumentException("assignments may only be a dict, with as many entries as this axis is long")
        if count == 0:
            if len(assignments) > 0:
                raise ArgumentException("assignments is too large; this axis is empty ")
            if axis == 'point':
                self.pointNames = {}
                self.pointNamesInverse = []
            else:
                self.featureNames = {}
                self.featureNamesInverse = []
            return
        if len(assignments) != count:
            raise ArgumentException("assignments may only be a dict, with as many entries as this axis is long")

        # at this point, the input must be a dict
        #check input before performing any action
        for name in assignments.keys():
            if not None and not isinstance(name, six.string_types):
                raise ArgumentException("Names must be strings")
            if not isinstance(assignments[name], int):
                raise ArgumentException("Indices must be integers")
            if assignments[name] < 0 or assignments[name] >= count:
                countName = 'pointCount' if axis == 'point' else 'featureCount'
                raise ArgumentException("Indices must be within 0 to self." + countName + " - 1")

        reverseMap = [None] * len(assignments)
        for name in assignments.keys():
            self._incrementDefaultIfNeeded(name, axis)
            reverseMap[assignments[name]] = name

        # have to copy the input, could be from another object
        if axis == 'point':
            self.pointNames = copy.deepcopy(assignments)
            self.pointNamesInverse = reverseMap
        else:
            self.featureNames = copy.deepcopy(assignments)
            self.featureNamesInverse = reverseMap


    def _constructIndicesList(self, axis, values):
        """
        Construct a list of indices from a valid integer (python or numpy) or
        string, or a one-dimensional, iterable container of valid integers
        and/or strings

        """
        if isinstance(values, (int, numpy.integer, six.string_types)):
            values = self._getIndex(values, axis)
            return [values]
        if pd and isinstance(values, pd.DataFrame):
            msg = "A pandas DataFrame object is not a valid input "
            msg += "for '{0}s'. ".format(axis)
            msg += "Only one-dimensional objects are accepted."
            raise ArgumentException(msg)
        valuesList = []
        try:
            for val in values:
                valuesList.append(val)
            indicesList = [self._getIndex(val, axis) for val in valuesList]
        except TypeError:
            msg = "The argument '{0}s' is not iterable.".format(axis)
            raise ArgumentException(msg)
        # _getIndex failed, use it's descriptive message
        except ArgumentException as ae:
            msg = "Invalid index value for the argument '{0}s'. ".format(axis)
            msg += str(ae)[1:-1]
            raise ArgumentException(msg)

        return indicesList


    def _validateAxis(self, axis):
        if axis != 'point' and axis != 'feature':
            raise ArgumentException('axis parameter may only be "point" or "feature"')

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


    def _validateValueIsNotNone(self, name, value):
        if value is None:
            msg = "The argument named " + name + " must not have a value of None"
            raise ArgumentException(msg)

    def _validateValueIsUMLDataObject(self, name, value, same):
        if not isinstance(value, UML.data.Base):
            msg = "The argument named " + name + " must be an instance "
            msg += "of the UML.data.Base class. The value we recieved was "
            msg += str(value) + ", had the type " + str(type(value))
            msg += ", and a method resolution order of "
            msg += str(inspect.getmro(value.__class__))
            raise ArgumentException(msg)

    def _shapeCompareString(self, argName, argValue):
        selfPoints = self.points
        sps = "" if selfPoints == 1 else "s"
        selfFeats = self.features
        sfs = "" if selfFeats == 1 else "s"
        argPoints = argValue.points
        aps = "" if argPoints == 1 else "s"
        argFeats = argValue.features
        afs = "" if argFeats == 1 else "s"

        ret = "Yet, " + argName + " has "
        ret += str(argPoints) + " point" + aps + " and "
        ret += str(argFeats) + " feature" + afs + " "
        ret += "while the caller has "
        ret += str(selfPoints) + " point" + sps + " and "
        ret += str(selfFeats) + " feature" + sfs + "."

        return ret

    def _validateObjHasSameNumberOfFeatures(self, argName, argValue):
        selfFeats = self.features
        argFeats = argValue.features

        if selfFeats != argFeats:
            msg = "The argument named " + argName + " must have the same number "
            msg += "of features as the caller object. "
            msg += self._shapeCompareString(argName, argValue)
            raise ArgumentException(msg)

    def _validateObjHasSameNumberOfPoints(self, argName, argValue):
        selfPoints = self.points
        argValuePoints = argValue.points
        if selfPoints != argValuePoints:
            msg = "The argument named " + argName + " must have the same number "
            msg += "of points as the caller object. "
            msg += self._shapeCompareString(argName, argValue)
            raise ArgumentException(msg)

    def _validateEmptyNamesIntersection(self, axis, argName, argValue):
        if axis == 'point':
            intersection = self._pointNameIntersection(argValue)
            nString = 'pointNames'
        elif axis == 'feature':
            intersection = self._featureNameIntersection(argValue)
            nString = 'featureNames'
        else:
            raise ArgumentException("invalid axis")

        shared = []
        if intersection:
            for name in intersection:
                if name[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                    shared.append(name)

        if shared != []:
            truncated = False
            if len(shared) > 10:
                full = len(shared)
                shared = shared[:10]
                truncated = True

            msg = "The argument named " + argName + " must not share any "
            msg += nString + " with the calling object, yet the following "
            msg += "names occured in both: "
            msg += UML.exceptions.prettyListString(shared)
            if truncated:
                msg += "... (only first 10 entries out of " + str(full)
                msg += " total)"
            raise ArgumentException(msg)

    def _validateMatPlotLibImport(self, error, name):
        if error is not None:
            msg = "The module matplotlib is required to be installed "
            msg += "in order to call the " + name + "() method. "
            msg += "However, when trying to import, an ImportError with "
            msg += "the following message was raised: '"
            msg += str(error) + "'"

            raise ImportError(msg)

    def _validateStatisticalFunctionInputString(self, statisticsFunction):
        acceptedPretty = [
            'max', 'mean', 'median', 'min', 'unique count', 'proportion missing',
            'proportion zero', 'standard deviation', 'std', 'population std',
            'population standard deviation', 'sample std',
            'sample standard deviation'
        ]
        accepted = list(map(dataHelpers.cleanKeywordInput, acceptedPretty))

        msg = "The statisticsFunction must be equivaltent to one of the "
        msg += "following: "
        msg += str(acceptedPretty) + ", but '" + str(statisticsFunction)
        msg += "' was given instead. Note: casing and whitespace is "
        msg += "ignored when checking the statisticsFunction."

        if not isinstance(statisticsFunction, six.string_types):
            raise ArgumentException(msg)

        cleanFuncName = dataHelpers.cleanKeywordInput(statisticsFunction)

        if cleanFuncName not in accepted:
            raise ArgumentException(msg)

        return cleanFuncName

    def _validateRangeOrder(self, startName, startVal, endName, endVal):
        """
        Validate a range where both values are inclusive.
        """
        if startVal > endVal:
            msg = "When specifying a range, the arguments were resolved to "
            msg += "having the values " + startName
            msg += "=" + str(startVal) + " and " + endName + "=" + str(endVal)
            msg += ", yet the starting value is not allowed to be greater than "
            msg += "the ending value (" + str(startVal) + ">" + str(endVal)
            msg += ")"

            raise ArgumentException(msg)


    def _adjustNamesAndValidate(self, ret, axis):
        if axis == 'point':
            self._pointCount -= ret.points
            for key in ret.getPointNames():
                self._removePointNameAndShift(key)
        else:
            self._featureCount -= ret.features
            for key in ret.getFeatureNames():
                self._removeFeatureNameAndShift(key)
        self.validate()

def cmp_to_key(mycmp):
    """Convert a cmp= function for python2 into a key= function for python3"""
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def cmp(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0
