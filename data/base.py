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
import copy
import os.path
from multiprocessing import Process
from abc import abstractmethod
from collections import OrderedDict

import numpy
import six
from six.moves import map
from six.moves import range
from six.moves import zip

import UML
from UML.exceptions import ArgumentException, PackageException
from UML.exceptions import ImproperActionException
from UML.logger import enableLogging, directCall
from UML.logger import produceFeaturewiseReport
from UML.logger import produceAggregateReport
from .points import Points
from .features import Features
from .axis import Axis
from .elements import Elements
from . import dataHelpers
# the prefix for default point and feature names
from .dataHelpers import DEFAULT_PREFIX, DEFAULT_PREFIX2, DEFAULT_PREFIX_LENGTH
from .dataHelpers import DEFAULT_NAME_PREFIX
from .dataHelpers import formatIfNeeded
from .dataHelpers import makeConsistentFNamesAndData
from .dataHelpers import valuesToPythonList
from .dataHelpers import logCaptureFactory

pd = UML.importModule('pandas')

cython = UML.importModule('cython')
if cython is None or not cython.compiled:
    from math import sin, cos

cloudpickle = UML.importModule('cloudpickle')

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

logCapture = logCaptureFactory()

#print('matplotlib backend: {}'.format(matplotlib.get_backend()))

def to2args(f):
    """
    this function is for __pow__. In cython, __pow__ must have 3
    arguments and default can't be used there so this function is used
    to convert a function with 3 arguments to a function with 2
    arguments when it is used in python environment.
    """
    def tmpF(x, y):
        return f(x, y, None)
    return tmpF

def hashCodeFunc(elementValue, pointNum, featureNum):
    return ((sin(pointNum) + cos(featureNum)) / 2.0) * elementValue

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
            raise ArgumentException(msg)
        if featureNames is not None and len(featureNames) != shape[1]:
            msg = "The length of the featureNames (" + str(len(featureNames))
            msg += ") must match the features given in shape ("
            msg += str(shape[1]) + ")"
            raise ArgumentException(msg)

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
            self.points.setNames(pointNames)
        else:
            pointNames = valuesToPythonList(pointNames, 'pointNames')
            self._nextDefaultValuePoint = self._pointCount
            self.points.setNames(pointNames)

        # Set up feature names
        self._nextDefaultValueFeature = 0
        if featureNames is None:
            self.featureNamesInverse = None
            self.featureNames = None
        elif isinstance(featureNames, dict):
            self._nextDefaultValueFeature = self._featureCount
            self.features.setNames(featureNames)
        else:
            featureNames = valuesToPythonList(featureNames, 'featureNames')
            self._nextDefaultValueFeature = self._featureCount
            self.features.setNames(featureNames)

        # Set up object name
        if name is None:
            self._name = dataHelpers.nextDefaultObjectName()
        else:
            self._name = name

        # Set up paths
        if paths[0] is not None and not isinstance(paths[0], six.string_types):
            msg = "paths[0] must be None, an absolute path or web link to "
            msg += "the file from which the data originates"
            raise ArgumentException(msg)
        if (paths[0] is not None
                and not os.path.isabs(paths[0])
                and not paths[0].startswith('http')):
            raise ArgumentException("paths[0] must be an absolute path")
        self._absPath = paths[0]

        if paths[1] is not None and not isinstance(paths[1], six.string_types):
            msg = "paths[1] must be None or a relative path to the file from "
            msg += "which the data originates"
            raise ArgumentException(msg)
        self._relPath = paths[1]

        # call for safety
        super(Base, self).__init__(**kwds)

    #######################
    # Property Attributes #
    #######################

    @property
    def shape(self):
        return self._pointCount, self._featureCount

    def _getPoints(self):
        return BasePoints(source=self)

    @property
    def points(self):
        return self._points

    def _getFeatures(self):
        return BaseFeatures(source=self)

    @property
    def features(self):
        return self._features

    def _getElements(self):
        return BaseElements(source=self)

    @property
    def elements(self):
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
        return self._setObjName(value)

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
    relativePath = property(_getRelPath, doc="")

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
        Returns True if any default point names exists or if pointNames have
        not been created.
        """
        if self._pointNamesCreated():
            return any([name.startswith(DEFAULT_PREFIX) for name
                        in self.points.getNames()])
        else:
            return True

    def _anyDefaultFeatureNames(self):
        """
        Returns True if any default feature names exists or if featureNames have
        not been created.
        """
        if self._featureNamesCreated():
            return any([name.startswith(DEFAULT_PREFIX) for name
                        in self.features.getNames()])
        else:
            return True

    def _allDefaultPointNames(self):
        """
        Returns True if all point names are default or have not been created
        """
        if self._pointNamesCreated():
            return all([name.startswith(DEFAULT_PREFIX) for name
                        in self.points.getNames()])
        else:
            return True

    def _allDefaultFeatureNames(self):
        """
        Returns True if all feature names are default or have not been created
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
        raise ImproperActionException(msg)

    def nameIsDefault(self):
        """Returns True if self.name has a default value"""
        return self.name.startswith(UML.data.dataHelpers.DEFAULT_NAME_PREFIX)

    def hasPointName(self, name):
        try:
            self.points.getIndex(name)
            return True
        except KeyError:
            return False

    def hasFeatureName(self, name):
        try:
            self.features.getIndex(name)
            return True
        except KeyError:
            return False

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

        Returns
        -------
        list
            The new feature names after replacement.

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.replaceFeatureWithBinaryFeatures)
            else:
                wrapped = directCall(self.replaceFeatureWithBinaryFeatures)
            return wrapped(featureToReplace, useLog=False)

        if self._pointCount == 0:
            msg = "This action is impossible, the object has 0 points"
            raise ImproperActionException(msg)

        index = self._getFeatureIndex(featureToReplace)
        # extract col.
        toConvert = self.features.extract([index])

        # MR to get list of values
        def getValue(point):
            return [(point[0], 1)]

        def simpleReducer(identifier, valuesList):
            return (identifier, 0)

        values = toConvert.points.mapReduce(getValue, simpleReducer)
        values.features.setName(0, 'values')
        values = values.features.extract([0])

        # Convert to List, so we can have easy access
        values = values.copyAs(format="List")

        # for each value run points.calculate to produce a category
        # point for each value
        def makeFunc(value):
            def equalTo(point):
                if point[0] == value:
                    return 1
                return 0

            return equalTo

        varName = toConvert.features.getName(0)

        for point in values.data:
            value = point[0]
            ret = toConvert.points.calculate(makeFunc(value))
            ret.features.setName(0, varName + "=" + str(value).strip())
            toConvert.features.add(ret)

        # remove the original feature, and combine with self
        toConvert.features.extract([varName])
        self.features.add(toConvert)

        return toConvert.features.getNames()

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

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.transformFeatureToIntegers)
            else:
                wrapped = directCall(self.transformFeatureToIntegers)
            return wrapped(featureToConvert, useLog=False)

        if self._pointCount == 0:
            msg = "This action is impossible, the object has 0 points"
            raise ImproperActionException(msg)

        index = self._getFeatureIndex(featureToConvert)

        # extract col.
        toConvert = self.features.extract([index])

        # MR to get list of values
        def getValue(point):
            return [(point[0], 1)]

        def simpleReducer(identifier, valuesList):
            return (identifier, 0)

        values = toConvert.points.mapReduce(getValue, simpleReducer)
        values.features.setName(0, 'values')
        values = values.features.extract([0])

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

        converted = toConvert.points.calculate(lookup)
        converted.points.setNames(toConvert.points.getNames())
        converted.features.setName(0, toConvert.features.getName(0))

        self.features.add(converted)

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

        Returns
        -------
        dict
            Each unique feature (or group of features) to group by as
            keys. When ``countUniqueValueOnly`` is False,  the value at
            each key is a UML object containing the ungrouped features
            of points within that group. When ``countUniqueValueOnly``
            is True, the values are the number of points within that
            group.

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.groupByFeature)
            else:
                wrapped = directCall(self.groupByFeature)
            return wrapped(by, countUniqueValueOnly, useLog=False)

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
                tmp = self.points.copy(toCopy=res[k])
                tmp.features.delete(by)
                res[k] = tmp

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

        Examples
        --------
        TODO
        """
        if self._pointCount == 0 or self._featureCount == 0:
            return 0
        valueObj = self.elements.calculate(hashCodeFunc, preserveZeros=True,
                                           outputType='Matrix')
        valueList = valueObj.copyAs(format="python list")
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
        other : UML data object
            The object with which to compare approximate equality with
            this object.

        Returns
        -------
        bool
            True if approximately equal, else False.

        Examples
        --------
        TODO
        """
        self.validate()
        #first check to make sure they have the same number of rows and columns
        if self._pointCount != len(other.points):
            return False
        if self._featureCount != len(other.features):
            return False
        #now check if the hashes of each matrix are the same
        if self.hashCode() != other.hashCode():
            return False
        return True

    def copy(self):
        """
        Return a new object which has the same data.

        The copy will be in the same UML format as this object and in
        addition to copying this objects data, the name and path
        metadata will by copied as well.

        Returns
        -------
        UML data object
            A copy of this object

        Examples
        --------
        TODO
        """
        return self.copyAs(self.getTypeString())

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
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.trainAndTestSets)
            else:
                wrapped = directCall(self.trainAndTestSets)
            return wrapped(testFraction, labels, randomOrder,
                           useLog=False)

        toSplit = self.copy()
        if randomOrder:
            toSplit.points.shuffle()

        testXSize = int(round(testFraction * self._pointCount))
        startIndex = self._pointCount - testXSize

        #pull out a testing set
        if testXSize == 0:
            testX = toSplit.points.extract([])
        else:
            testX = toSplit.points.extract(start=startIndex)

        if labels is None:
            toSplit.name = self.name + " trainX"
            testX.name = self.name + " testX"

            return toSplit, testX

        # safety for empty objects
        toExtract = labels
        if testXSize == 0:
            toExtract = []

        trainY = toSplit.features.extract(toExtract)
        testY = testX.features.extract(toExtract)

        toSplit.name = self.name + " trainX"
        trainY.name = self.name + " trainY"
        testX.name = self.name + " testX"
        testY.name = self.name + " testY"

        return toSplit, trainY, testX, testY

    ########################################
    ########################################
    ###   Functions related to logging   ###
    ########################################
    ########################################

    def featureReport(self, maxFeaturesToCover=50, displayDigits=2, useLog=None):
        """
        Produce a report, in a string formatted as a table, containing
        summary and statistical information about each feature in the
        data set, up to 50 features.  If there are more than 50
        features, only information about 50 of those features will be
        reported.
        """
        logger = UML.logger.active
        logger.position += 1
        ret = produceFeaturewiseReport(
            self, maxFeaturesToCover=maxFeaturesToCover,
            displayDigits=displayDigits)
        logger.position -= 1
        if enableLogging(useLog):
            logger.logData("feature", ret)
            logger.log(logger.logType, logger.logInfo)
        return ret

    def summaryReport(self, displayDigits=2, useLog=None):
        """
        Produce a report, in a string formatted as a table, containing summary
        information about the data set contained in this object.  Includes
        proportion of missing values, proportion of zero values, total # of
        points, and number of features.
        """
        logger = UML.logger.active
        logger.position += 1
        ret = produceAggregateReport(self, displayDigits=displayDigits)
        logger.position -= 1
        if enableLogging(useLog):
            logger.logData("summary", ret)
            logger.log(logger.logType, logger.logInfo)
        return ret

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
        Write the data in this object to a file in the specified format.

        Parameters
        ----------
        outPath : str
            The location (including file name and extension) where
            we want to write the output file.
        format : str
            The formating of the file we write. May be None, 'csv', or
            'mtx'; if None, we use the extension of outPath to determine
            the format.
        includeNames : bool
            Indicates whether the file will embed the point and feature
            names into the file. The format of the embedding is
            dependant on the format of the file: csv will embed names
            into the data, mtx will place names in a comment.

        Examples
        --------
        TODO
        """
        if self._pointCount == 0 or self._featureCount == 0:
            msg = "We do not allow writing to file when an object has "
            msg += "0 points or features"
            raise ImproperActionException(msg)

        self.validate()

        # if format is not specified, we fall back on the extension in outPath
        if format is None:
            split = outPath.rsplit('.', 1)
            format = None
            if len(split) > 1:
                format = split[1].lower()

        if format not in ['csv', 'mtx']:
            msg = "Unrecognized file format. Accepted types are 'csv' and "
            msg += "'mtx'. They may either be input as the format parameter, "
            msg += "or as the extension in the outPath"
            raise ArgumentException(msg)

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
                outPath, format, includePointNames, includeFeatureNames)
        except Exception:
            if format.lower() == "csv":
                toOut = self.copyAs("Matrix")
                toOut._writeFile_implementation(
                    outPath, format, includePointNames, includeFeatureNames)
                return
            if format.lower() == "mtx":
                toOut = self.copyAs('Sparse')
                toOut._writeFile_implementation(
                    outPath, format, includePointNames, includeFeatureNames)
                return

    def save(self, outputPath):
        """
        Save object to a file.

        Uses the dill library to serialize this object.

        Parameters
        ----------
        outputPath : str
            The location (including file name and extension) where
            we want to write the output file. If filename extension
            .umld is not included in file name it would be added to the
            output file.

        Examples
        --------
        TODO
        """
        if not cloudpickle:
            msg = "To save UML objects, cloudpickle must be installed"
            raise PackageException(msg)

        extension = '.umld'
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
        Return a string representing the non-abstract type of this
        object (e.g. Matrix, Sparse, etc.) that can be passed to
        createData() function to create a new object of the same type.
        """
        return self._getTypeString_implementation()

    def _processSingleX(self, x):
        """
        Helper for __getitem__ when given a single value for x.
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
            return self.points.getIndex(x), True

        if x.__class__ is float:
            if x % 1: # x!=int(x)
                msg = "A float valued key of value x is only accepted if x == "
                msg += "int(x). The given value was " + str(x) + " yet int("
                msg += str(x) + ") = " + str(int(x))
                raise ArgumentException(msg)
            else:
                x = int(x)
                if x < -length or x >= length:
                    msg = "The given index " + str(x) + " is outside of the "
                    msg += "range of possible indices in the point axis (0 to "
                    msg += str(length - 1) + ")."
                    raise IndexError(msg)
                if x >= 0:
                    return x, True
                else:
                    return x + length, True

        return x, False

    def _processSingleY(self, y):
        """
        Helper for __getitem__ when given a single value for y.
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
            return self.features.getIndex(y), True

        if y.__class__ is float:
            if y % 1: # y!=int(y)
                msg = "A float valued key of value y is only accepted if y == "
                msg += "int(y). The given value was " + str(y) + " yet int("
                msg += str(y) + ") = " + str(int(y))
                raise ArgumentException(msg)
            else:
                y = int(y)
                if y < -length or y >= length:
                    msg = "The given index " + str(y) + " is outside of the "
                    msg += "range of possible indices in the point axis (0 to "
                    msg += str(length - 1) + ")."
                    raise IndexError(msg)
                if y >= 0:
                    return y, True
                else:
                    return y + length, True

        return y, False

    def __getitem__(self, key):
        """
        The following are allowed:
        X[1, :]            ->    (2d) that just has that one point
        X["age", :]    -> same as above
        X[1:5, :]         -> 4 points (1, 2, 3, 4)
        X[[3,8], :]       -> 2 points (3, 8) IN THE ORDER GIVEN
        X[["age","gender"], :]       -> same as above

        --get based on features only : ALWAYS returns a new copy UML
        object (2d)
        X[:,2]         -> just has that 1 feature
        X[:,"bob"] -> same as above
        X[:,1:5]    -> 4 features (1,2,3,4)
        X[:,[3,8]]  -> 2 features (3,8) IN THE ORDER GIVEN

        --both features and points : can give a scalar value OR UML
        object 2d depending on case
        X[1,2]         -> single scalar number value
        X["age","bob"] -> single scalar number value
        X[1:5,4:7]     -> UML object (2d) that has just that rectangle
        X[[1,2],[3,8]] -> UML object (2d) that has just 2 points
                          (points 1,2) but only 2 features for each of
                          them (features 3,8)
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
        #if it is the simplest data retrieval such as X[1,2],
        # we'd like to return it back in the fastest way.
        if singleX and singleY:
            return self._getitem_implementation(x, y)

        if not singleX:
            if x.__class__ is slice:
                start = x.start if x.start is not None else 0
                if start < 0:
                    start += self._pointCount
                stop = x.stop if x.stop is not None else self._pointCount
                if stop < 0:
                    stop += self._pointCount
                step = x.step if x.step is not None else 1
                x = [self._processSingleX(xi)[0] for xi
                     in range(start, stop, step)]
            else:
                x = [self._processSingleX(xi)[0] for xi in x]

        if not singleY:
            if y.__class__ is slice:
                start = y.start if y.start is not None else 0
                if start < 0:
                    start += self._featureCount
                stop = y.stop if y.stop is not None else self._featureCount
                if stop < 0:
                    stop += self._featureCount
                step = y.step if y.step is not None else 1
                y = [self._processSingleY(yi)[0] for yi
                     in range(start, stop, step)]
            else:
                y = [self._processSingleY(yi)[0] for yi in y]
        ret = self.points.copy(toCopy=x, useLog=False)
        ret = ret.features.copy(toCopy=y, useLog=False)
        return ret

    def pointView(self, ID):
        """
        Returns a View object into the data of the point with the given
        ID. See View object comments for its capabilities. This View is
        only valid until the next modification to the shape or ordering
        of the internal data. After such a modification, there is no
        guarantee to the validity of the results.
        """
        if self._pointCount == 0:
            msg = "ID is invalid, This object contains no points"
            raise ImproperActionException(msg)

        index = self._getPointIndex(ID)
        return self.view(index, index, None, None)

    def featureView(self, ID):
        """
        Returns a View object into the data of the point with the given
        ID. See View object comments for its capabilities. This View is
        only valid until the next modification to the shape or ordering
        of the internal data. After such a modification, there is no
        guarantee to the validity of the results.
        """
        if self._featureCount == 0:
            msg = "ID is invalid, This object contains no features"
            raise ImproperActionException(msg)

        index = self._getFeatureIndex(ID)
        return self.view(None, None, index, index)

    def view(self, pointStart=None, pointEnd=None, featureStart=None,
             featureEnd=None):
        """
        Read-only access into the object data.

        Factory function to create a read only view into the calling
        data object. Views may only be constructed from contiguous,
        in-order points and features whose overlap defines a window into
        the data. The returned View object is part of UML's datatypes
        hiearchy, and will have access to all of the same methods as
        anything that inherits from UML.data.Base; though only those
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
        UML view object
            A UML data object with read-only access.

        See Also
        --------
        pointView, featureView

        Examples
        --------
        TODO
        """
        # transform defaults to mean take as much data as possible,
        # transform end values to be EXCLUSIVE
        if pointStart is None:
            pointStart = 0
        else:
            pointStart = self._getIndex(pointStart, 'point')

        if pointEnd is None:
            pointEnd = self._pointCount
        else:
            pointEnd = self._getIndex(pointEnd, 'point')
            # this is the only case that could be problematic and needs
            # checking
            self._validateRangeOrder("pointStart", pointStart,
                                     "pointEnd", pointEnd)
            # make exclusive now that it won't ruin the validation check
            pointEnd += 1

        if featureStart is None:
            featureStart = 0
        else:
            featureStart = self._getIndex(featureStart, 'feature')

        if featureEnd is None:
            featureEnd = self._featureCount
        else:
            featureEnd = self._getIndex(featureEnd, 'feature')
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

        Examples
        --------
        TODO
        """
        if self._pointNamesCreated():
            assert self._pointCount == len(self.points.getNames())
        if self._featureNamesCreated():
            assert self._featureCount == len(self.features.getNames())

        if level > 0:
            if self._pointNamesCreated():
                for key in self.points.getNames():
                    index = self.points.getIndex(key)
                    assert self.points.getName(index) == key
            if self._featureNamesCreated():
                for key in self.features.getNames():
                    index = self.features.getIndex(key)
                    assert self.features.getName(index) == key

        self._validate_implementation(level)

    def containsZero(self):
        """
        Evaluate if the object contains one or more zero values.

        Return True if there is a value that is equal to integer 0
        contained in this object, otherwise False.

        Examples
        --------
        TODO
        """
        # trivially False.
        if self._pointCount == 0 or self._featureCount == 0:
            return False
        return self._containsZero_implementation()

    def __eq__(self, other):
        return self.isIdentical(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def toString(self, includeNames=True, maxWidth=120, maxHeight=30,
                 sigDigits=3, maxColumnWidth=19):
        """
        TODO
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
            # pnames, a column separator, the pnames seperator, and another
            # column seperator
            maxDataWidth = (maxWidth
                            - (pnamesWidth + 2 * len(colSep) + len(pnameSep)))

        # Set up data values to fit in the available space
        dataTable, colWidths = self._arrangeDataWithLimits(
            maxDataWidth, maxDataRows, sigDigits, maxColumnWidth, colSep,
            colHold, rowHold, nameHolder)

        # set up feature names list, record widths
        fnames = None
        if includeFNames:
            fnames = self._arrangeFeatureNames(maxWidth, maxColumnWidth,
                                               colSep, colHold, nameHolder)

            # adjust data or fnames according to the more restrictive set
            # of col widths
            makeConsistentFNamesAndData(fnames, dataTable, colWidths, colHold)

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
        dataStr = self.toString(includeNames=False, maxWidth=maxW,
                                maxHeight=maxH)
        byLine = dataStr.split('\n')
        # toString ends with a \n, so we get rid of the empty line produced by
        # the split
        byLine = byLine[:-1]
        # convert self.data into a string with nice format
        newLines = (']\n' + indent + ' [').join(byLine)
        ret += (indent + '[[%s]]\n') % newLines

        numRows = min(self._pointCount, maxW)
        # if non default point names, print all (truncated) point names
        ret += dataHelpers.makeNamesLines(
            indent, maxW, numRows, self._pointCount, self.points.getNames(),
            'pointNames')
        # if non default feature names, print all (truncated) feature names
        numCols = 0
        if byLine:
            splited = byLine[0].split(' ')
            for val in splited:
                if val != '' and val != '...':
                    numCols += 1
        elif self._featureCount > 0:
            # if the container is empty, then roughly compute length of
            # the string of feature names, and then calculate numCols
            strLength = (len("___".join(self.features.getNames()))
                         + len(''.join([str(i) for i
                                        in range(self._featureCount)])))
            numCols = int(min(1, maxW / float(strLength)) * self._featureCount)
        # because of how dataHelers.indicesSplit works, we need this to be plus
        # one in some cases this means one extra feature name is displayed. But
        # that's acceptable
        if numCols <= self._featureCount:
            numCols += 1
        ret += dataHelpers.makeNamesLines(
            indent, maxW, numCols, self._featureCount,
            self.features.getNames(), 'featureNames')

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
             maxWidth=120, maxHeight=30, sigDigits=3, maxColumnWidth=19):
        """
        Method to simplify printing a representation of this data object,
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

        # toPlot = self.copyAs('numpyarray')

        # problem if we were to use mutiprocessing with backends
        # different than Agg.
        p = self._matplotlibBackendHandling(outPath, plotter, d=self.data)
        return p

    def plotFeatureDistribution(self, feature, outPath=None, xMin=None,
                                xMax=None):
        """
        Plot a histogram of the distribution of values in the specified
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

    def _plotFeatureDistribution(self, feature, outPath=None, xMin=None,
                                 xMax=None):
        self._validateMatPlotLibImport(mplError, 'plotFeatureDistribution')
        return self._plotDistribution('feature', feature, outPath, xMin, xMax)

    def _plotDistribution(self, axis, identifier, outPath, xMin, xMax):
        outFormat = self._setupOutFormatForPlotting(outPath)
        index = self._getIndex(identifier, axis)
        if axis == 'point':
            getter = self.pointView
            name = self.points.getName(index)
        else:
            getter = self.featureView
            name = self.features.getName(index)

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
        p = self._matplotlibBackendHandling(outPath, plotter, d=toPlot,
                                            xLim=(xMin, xMax))
        return p

    def plotFeatureAgainstFeatureRollingAverage(
            self, x, y, outPath=None, xMin=None, xMax=None, yMin=None,
            yMax=None, sampleSizeForAverage=20):
        """
        TODO
        """
        self._plotFeatureAgainstFeature(x, y, outPath, xMin, xMax, yMin, yMax,
                                        sampleSizeForAverage)

    def plotFeatureAgainstFeature(self, x, y, outPath=None, xMin=None,
                                  xMax=None, yMin=None, yMax=None):
        """
        Plot a scatter plot of the two input features using the pairwise
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

    def _plotFeatureAgainstFeature(self, x, y, outPath=None, xMin=None,
                                   xMax=None, yMin=None, yMax=None,
                                   sampleSizeForAverage=None):
        self._validateMatPlotLibImport(mplError, 'plotFeatureComparison')
        return self._plotCross(x, 'feature', y, 'feature', outPath, xMin, xMax,
                               yMin, yMax, sampleSizeForAverage)

    def _plotCross(self, x, xAxis, y, yAxis, outPath, xMin, xMax, yMin, yMax,
                   sampleSizeForAverage=None):
        outFormat = self._setupOutFormatForPlotting(outPath)
        xIndex = self._getIndex(x, xAxis)
        yIndex = self._getIndex(y, yAxis)

        def customGetter(index, axis):
            if axis == 'point':
                copied = self.points.copy(index)
            else:
                copied = self.features.copy(index)
            return copied.copyAs('numpyarray', outputAs1D=True)

        def pGetter(index):
            return customGetter(index, 'point')

        def fGetter(index):
            return customGetter(index, 'feature')

        if xAxis == 'point':
            xGetter = pGetter
            xName = self.points.getName(xIndex)
        else:
            xGetter = fGetter
            xName = self.features.getName(xIndex)

        if yAxis == 'point':
            yGetter = pGetter
            yName = self.points.getName(yIndex)
        else:
            yGetter = fGetter
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

            if xName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
                xlabel = xAxis + ' #' + str(xIndex)
            else:
                xlabel = xName
            if yName[:DEFAULT_PREFIX_LENGTH] == DEFAULT_PREFIX:
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

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.transpose)
            else:
                wrapped = directCall(self.transpose)
            return wrapped(useLog=False)

        self._transpose_implementation()

        self._pointCount, self._featureCount = (self._featureCount,
                                                self._pointCount)

        if self._pointNamesCreated() and self._featureNamesCreated():
            self.pointNames, self.featureNames = (self.featureNames,
                                                  self.pointNames)
            self.features.setNames(self.featureNames)
            self.points.setNames(self.pointNames)
        elif self._pointNamesCreated():
            self.featureNames = self.pointNames
            self.pointNames = None
            self.pointNamesInverse = None
            self.features.setNames(self.featureNames)
        elif self._featureNamesCreated():
            self.pointNames = self.featureNames
            self.featureNames = None
            self.featureNamesInverse = None
            self.points.setNames(self.pointNames)
        else:
            pass

        self.validate()

    def referenceDataFrom(self, other, useLog=None):
        """
        Redefine the object data using the data from another object.

        Modify the internal data of this object to refer to the same
        data as other. In other words, the data wrapped by both the self
        and ``other`` objects resides in the same place in memory.

        Parameters
        ----------
        other : UML data object
            Must be of the same type as the calling object. Also, the
            shape of other should be consistent with the shape of this
            object.

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.referenceDataFrom)
            else:
                wrapped = directCall(self.referenceDataFrom)
            return wrapped(other, useLog=False)

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
        Duplicate an object in another format.

        Return a new object which has the same data (and names,
        depending on the return type) as this object.

        Parameters
        ----------
        format : str
            To return a specific kind of UML data object, one may
            specify the format parameter to be 'List', 'Matrix', or
            'Sparse'. To specify a raw return type (which will not
            include feature names), one may specify 'python list',
            'numpy array', or 'numpy matrix', 'scipy csr', 'scypy csc',
            'list of dict' or 'dict of list'.
        """
        #make lower case, strip out all white space and periods, except if
        # format is one of the accepted UML data types
        if format not in ['List', 'Matrix', 'Sparse', 'DataFrame']:
            format = format.lower()
            format = format.strip()
            tokens = format.split(' ')
            format = ''.join(tokens)
            tokens = format.split('.')
            format = ''.join(tokens)
            if format not in ['pythonlist', 'numpyarray', 'numpymatrix',
                              'scipycsr', 'scipycsc', 'listofdict',
                              'dictoflist']:
                msg = "The only accepted asTypes are: 'List', 'Matrix', "
                msg += "'Sparse', 'python list', 'numpy array', "
                msg += "'numpy matrix', 'scipy csr', 'scipy csc', "
                msg += "'list of dict', and 'dict of list'"
                raise ArgumentException(msg)

        # only 'numpyarray' and 'pythonlist' are allowed to use outputAs1D flag
        if outputAs1D:
            if format != 'numpyarray' and format != 'pythonlist':
                msg = "Only 'numpy array' or 'python list' can output 1D"
                raise ArgumentException(msg)
            if self._pointCount != 1 and self._featureCount != 1:
                msg = "To output as 1D there may either be only one point or "
                msg += " one feature"
                raise ArgumentException(msg)

        # certain shapes and formats are incompatible
        if format.startswith('scipy'):
            if self._pointCount == 0 or self._featureCount == 0:
                msg = "scipy formats cannot output point or feature empty "
                msg += "objects"
                raise ArgumentException(msg)

        ret = self._copyAs_implementation_base(format, rowsArePoints,
                                               outputAs1D)

        if isinstance(ret, UML.data.Base):
            ret._name = self.name
            ret._relPath = self.relativePath
            ret._absPath = self.absolutePath

        return ret

    def _copyAs_implementation_base(self, format, rowsArePoints, outputAs1D):
        # in copyAs, we've already limited outputAs1D to the 'numpyarray'
        # and 'python list' formats
        if outputAs1D:
            if self._pointCount == 0 or self._featureCount == 0:
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
            if self._pointCount == 0:
                return []
            if self._featureCount == 0:
                ret = []
                for _ in range(self._pointCount):
                    ret.append([])
                return ret

        if format in ['listofdict', 'dictoflist']:
            ret = self._copyAs_implementation('numpyarray')
        else:
            ret = self._copyAs_implementation(format)
            if isinstance(ret, UML.data.Base):
                self._copyNames(ret)

        def _createListOfDict(data, featureNames):
            # creates a list of dictionaries mapping feature names to the
            # point's values dictionaries are in point order
            listofdict = []
            for point in data:
                feature_dict = {}
                for i, value in enumerate(point):
                    feature = featureNames[i]
                    feature_dict[feature] = value
                listofdict.append(feature_dict)
            return listofdict

        def _createDictOfList(data, featureNames, nFeatures):
            # creates a python dict maps feature names to python lists
            # containing all of that feature's values
            dictoflist = {}
            for i in range(nFeatures):
                feature = featureNames[i]
                values_list = data[:, i].tolist()
                dictoflist[feature] = values_list
            return dictoflist

        if not rowsArePoints:
            if format in ['List', 'Matrix', 'Sparse', 'DataFrame']:
                ret.transpose()
            elif format == 'listofdict':
                ret = ret.transpose()
                ret = _createListOfDict(data=ret,
                                        featureNames=self.points.getNames())
                return ret
            elif format == 'dictoflist':
                ret = ret.transpose()
                ret = _createDictOfList(data=ret,
                                        featureNames=self.points.getNames(),
                                        nFeatures=self._pointCount)
                return ret
            elif format != 'pythonlist':
                ret = ret.transpose()
            else:
                ret = numpy.transpose(ret).tolist()

        if format == 'listofdict':
            ret = _createListOfDict(data=ret,
                                    featureNames=self.features.getNames())
        if format == 'dictoflist':
            ret = _createDictOfList(data=ret,
                                    featureNames=self.features.getNames(),
                                    nFeatures=self._featureCount)

        return ret

    def _copyNames(self, CopyObj):
        if self._pointNamesCreated():
            CopyObj.pointNamesInverse = self.points.getNames()
            CopyObj.pointNames = copy.copy(self.pointNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.index = self.points.getNames()
        else:
            CopyObj.pointNamesInverse = None
            CopyObj.pointNames = None

        if self._featureNamesCreated():
            CopyObj.featureNamesInverse = self.features.getNames()
            CopyObj.featureNames = copy.copy(self.featureNames)
            # if CopyObj.getTypeString() == 'DataFrame':
            #     CopyObj.data.columns = self.features.getNames()
        else:
            CopyObj.featureNamesInverse = None
            CopyObj.featureNames = None

        CopyObj._nextDefaultValueFeature = self._nextDefaultValueFeature
        CopyObj._nextDefaultValuePoint = self._nextDefaultValuePoint

    def fillWith(self, values, pointStart, featureStart, pointEnd, featureEnd,
                 useLog=None):
        """
        Replace values in the data with other values.

        Revise the contents of the calling object so that it contains
        the provided values in the given location.

        Parameters
        ----------
        values : constant or UML data object
            * constant - a constant value with which to fill the data
              seletion.
            * UML data object - Size must be consistent with the given
              start and end indices.
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


        See Also
        --------
        fillUsingAllData, points.fill, features.fill

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.fillWith)
            else:
                wrapped = directCall(self.fillWith)
            return wrapped(values, pointStart, featureStart, pointEnd,
                           featureEnd, useLog=False)

        psIndex = self._getPointIndex(pointStart)
        peIndex = self._getPointIndex(pointEnd)
        fsIndex = self._getFeatureIndex(featureStart)
        feIndex = self._getFeatureIndex(featureEnd)

        if psIndex > peIndex:
            msg = "pointStart (" + str(pointStart) + ") must be less than or "
            msg += "equal to pointEnd (" + str(pointEnd) + ")."
            raise ArgumentException(msg)
        if fsIndex > feIndex:
            msg = "featureStart (" + str(featureStart) + ") must be less than "
            msg += "or equal to featureEnd (" + str(featureEnd) + ")."
            raise ArgumentException(msg)

        if isinstance(values, UML.data.Base):
            prange = (peIndex - psIndex) + 1
            frange = (feIndex - fsIndex) + 1
            if len(values.points) != prange:
                msg = "When the values argument is a UML data object, the "
                msg += "size of values must match the range of modification. "
                msg += "There are " + str(len(values.points)) + " points in "
                msg += "values, yet pointStart (" + str(pointStart) + ")"
                msg += "and pointEnd (" + str(pointEnd) + ") define a range "
                msg += "of length " + str(prange)
                raise ArgumentException(msg)
            if len(values.features) != frange:
                msg = "When the values argument is a UML data object, the "
                msg += "size of values must match the range of modification. "
                msg += "There are " + str(len(values.features)) + " features "
                msg += "in values, yet featureStart (" + str(featureStart)
                msg += ") and featureEnd (" + str(featureEnd) + ") define a "
                msg += "range of length " + str(frange)
                raise ArgumentException(msg)
            if values.getTypeString() != self.getTypeString():
                values = values.copyAs(self.getTypeString())

        elif (dataHelpers._looksNumeric(values)
              or isinstance(values, six.string_types)):
            pass  # no modificaitons needed
        else:
            msg = "values may only be a UML data object, or a single numeric "
            msg += "value, yet we received something of " + str(type(values))
            raise ArgumentException(msg)

        self._fillWith_implementation(values, psIndex, fsIndex,
                                      peIndex, feIndex)
        self.validate()

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
              from UML's match module: missing, nonNumeric, zero, etc.
        fill : function
            a function in the format fill(feature, match) or
            fill(feature, match, arguments) and return the transformed
            data as a UML data object. Certain fill methods can be
            imported from UML's fill module:
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
        kwarguments
            Any additional arguments being passed to the fill function.

        See Also
        --------
        fillWith, points.fill, features.fill

        Examples
        --------
        Fill using the value that occurs most often in each points
        3 nearest neighbors.
        from UML.fill import kNeighborsClassifier
        >>> raw = [[1, 1, 1],
        ...        [1, 1, 1],
        ...        [1, 1, 'na'],
        ...        [2, 2, 2],
        ...        ['na', 2, 2]]
        >>> data = UML.createData('Matrix', raw)
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
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.fillUsingAllData)
            else:
                wrapped = directCall(self.fillUsingAllData)
            return wrapped(match, fill, arguments, points, features,
                           returnModified, useLog=False)

        if returnModified:
            modified = self.elements.calculate(match, points=points,
                                               features=features)
            modNames = [name + "_modified" for name
                        in modified.features.getNames()]
            modified.features.setNames(modNames)
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
            msg += "fillUsingPoints or fillUsingFeatures."
            raise ArgumentException(msg)
        tmpData = fill(self.copy(), match, **kwarguments)
        if points is None and features is None:
            self.referenceDataFrom(tmpData)
        else:
            def transform(value, i, j):
                return tmpData[i, j]
            self.elements.transform(transform, points, features)

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

        See Also
        --------
        unflattenFromOnePoint

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.flattenToOnePoint)
            else:
                wrapped = directCall(self.flattenToOnePoint)
            return wrapped(useLog=False)

        if self._pointCount == 0:
            msg = "Can only flattenToOnePoint when there is one or more "
            msg += "points. This object has 0 points."
            raise ImproperActionException(msg)
        if self._featureCount == 0:
            msg = "Can only flattenToOnePoint when there is one or more "
            msg += "features. This object has 0 features."
            raise ImproperActionException(msg)

        # TODO: flatten nameless Objects without the need to generate default
        # names for them.
        if not self._pointNamesCreated():
            self._setAllDefault('point')
        if not self._featureNamesCreated():
            self._setAllDefault('feature')
        self._flattenToOnePoint_implementation()

        self._featureCount = self._pointCount * self._featureCount
        self._pointCount = 1
        self.features.setNames(self._flattenNames('point'))
        self.points.setNames(['Flattened'])

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

        See Also
        --------
        unflattenFromOneFeature

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.flattenToOneFeature)
            else:
                wrapped = directCall(self.flattenToOneFeature)
            return wrapped(useLog=False)

        if self._pointCount == 0:
            msg = "Can only flattenToOnePoint when there is one or more "
            msg += "points. This object has 0 points."
            raise ImproperActionException(msg)
        if self._featureCount == 0:
            msg = "Can only flattenToOnePoint when there is one or more "
            msg += "features. This object has 0 features."
            raise ImproperActionException(msg)

        # TODO: flatten nameless Objects without the need to generate default
        # names for them.
        if not self._pointNamesCreated():
            self._setAllDefault('point')
        if not self._featureNamesCreated():
            self._setAllDefault('feature')

        self._flattenToOneFeature_implementation()

        self._pointCount = self._pointCount * self._featureCount
        self._featureCount = 1
        self.points.setNames(self._flattenNames('feature'))
        self.features.setNames(['Flattened'])

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

        Will raise ImproperActionException if an inconsistency with the
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
            raise ImproperActionException(msg)

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

    def unflattenFromOnePoint(self, numPoints, useLog=None):
        """
        Adjust a flattened point vector to contain multiple points.

        This is an inverse of the method ``flattenToOnePoint``: if an
        object foo with n points calls the flatten method, then this
        method with n as the argument, the result should be identical to
        the original foo. It is not limited to objects that have
        previously had ``flattenToOnePoint`` called on them; any object
        whose structure and names are consistent with a previous call to
        flattenToOnePoint may call this method. This includes objects
        with all default names. This is an inplace operation.

        Parameters
        ----------
        numPoints : int
            The number of points in the modified object.

        See Also
        --------
        flattenToOnePoint

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.unflattenFromOnePoint)
            else:
                wrapped = directCall(self.unflattenFromOnePoint)
            return wrapped(numPoints, useLog=False)

        if self._featureCount == 0:
            msg = "Can only unflattenFromOnePoint when there is one or more "
            msg += "features.  This object has 0 features."
            raise ImproperActionException(msg)
        if self._pointCount != 1:
            msg = "Can only unflattenFromOnePoint when there is only one "
            msg += "point.  This object has " + str(self._pointCount)
            msg += "points."
            raise ImproperActionException(msg)
        if self._featureCount % numPoints != 0:
            msg = "The argument numPoints (" + str(numPoints) + ") must be a "
            msg += "divisor of  this object's featureCount ("
            msg += str(self._featureCount) + ") otherwise  it will not be "
            msg += "possible to equally divide the elements into the desired "
            msg += "number of points."
            raise ArgumentException(msg)

        if not self._pointNamesCreated():
            self._setAllDefault('point')
        if not self._featureNamesCreated():
            self._setAllDefault('feature')

        self._unflattenFromOnePoint_implementation(numPoints)
        ret = self._unflattenNames('point', numPoints)
        self._featureCount = self._featureCount // numPoints
        self._pointCount = numPoints
        self.points.setNames(ret[0])
        self.features.setNames(ret[1])

    def unflattenFromOneFeature(self, numFeatures, useLog=None):
        """
        Adjust a flattened feature vector to contain multiple features.

        This is an inverse of the method ``flattenToOneFeature``: if an
        object foo with n features calls the flatten method, then this
        method with n as the argument, the result should be identical to
        the original foo. It is not limited to objects that have
        previously had ``flattenToOneFeature`` called on them; any
        object whose structure and names are consistent with a previous
        call to flattenToOnePoint may call this method. This includes
        objects with all default names. This is an inplace operation.

        Parameters
        ----------
        numFeatures : int
            The number of features in the modified object.

        See Also
        --------
        flattenToOneFeature

        Examples
        --------
        TODO
        """
        if UML.logger.active.position == 0:
            if enableLogging(useLog):
                wrapped = logCapture(self.unflattenFromOneFeature)
            else:
                wrapped = directCall(self.unflattenFromOneFeature)
            return wrapped(numFeatures, useLog=False)

        if self._pointCount == 0:
            msg = "Can only unflattenFromOneFeature when there is one or more "
            msg += "points. This object has 0 points."
            raise ImproperActionException(msg)
        if self._featureCount != 1:
            msg = "Can only unflattenFromOneFeature when there is only one "
            msg += "feature. This object has " + str(self._featureCount)
            msg += " features."
            raise ImproperActionException(msg)

        if self._pointCount % numFeatures != 0:
            msg = "The argument numFeatures (" + str(numFeatures) + ") must "
            msg += "be a divisor of this object's pointCount ("
            msg += str(self._pointCount) + ") otherwise "
            msg += "it will not be possible to equally divide the elements "
            msg += "into the desired number of features."
            raise ArgumentException(msg)

        if not self._pointNamesCreated():
            self._setAllDefault('point')
        if not self._featureNamesCreated():
            self._setAllDefault('feature')

        self._unflattenFromOneFeature_implementation(numFeatures)
        ret = self._unflattenNames('feature', numFeatures)
        self._pointCount = self._pointCount // numFeatures
        self._featureCount = numFeatures
        self.points.setNames(ret[1])
        self.features.setNames(ret[0])

    def merge(self, other, point='strict', feature='union', onFeature=None):
        """
        Merge data from another object into this object based on point names or
        a common feature between the objects. How the data will be merged is
        based upon the string arguments provided to point and feature.

        If onFeature is None, the objects will be merged on the point names.
        Otherwise, the objects will be merged on the feature provided.
        onFeature allows for duplicate values to be present in the provided
        feature, however, one of the objects must contain only unique values
        for each point when onFeature is provided.

        The allowed strings for the point and feature arguments are as follows:

        'strict': The points/features in the callee exactly match the
          points/features in the caller, however, they may be in a different
          order. If onFeature is None and no names are provided, it will be
          assumed the order is the same.

        'union': Return all points/features from the caller and callee. If
          onFeature is None, unnamed points/features will be assumed to be
          unique. Any missing data from the caller and callee will be filled
          with numpy.NaN

        'intersection': Return only points/features shared between the caller
          and callee. If onFeature is None, point/feature names are required.

        'left': Return only the points/features from the caller. Any
          missing data from the callee will be filled with numpy.NaN
        """
        point = point.lower()
        feature = feature.lower()
        valid = ['strict', 'left', 'union', 'intersection']
        if point not in valid:
            msg = "point must be 'strict', 'left', 'union', or 'intersection'"
            raise ArgumentException(msg)
        if feature not in valid:
            msg = "feature must be 'strict', 'left', 'union', or 'intersection'"
            raise ArgumentException(msg)

        if point == 'strict' or feature == 'strict':
            return self._genericStrictMerge_implementation(other, point, feature, onFeature)
        else:
            return self._genericMergeFrontend(other, point, feature, onFeature)

    def _genericStrictMerge_implementation(self, other, point, feature, onFeature):
        """
        Validation and helper function when point or feature is set to strict
        """
        # NOTE could return this object?
        if point == 'strict' and feature == 'strict':
            msg = 'Both point and feature cannot be strict'
            raise ArgumentException(msg)
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
            raise ArgumentException(msg)
        if hasNamesL and hasNamesR:
            if sorted(namesL()) != sorted(namesR()):
                msg = "When {0}='strict', the {0}s names ".format(axis)
                msg += "may be in a different order but must match exactly"
                raise ArgumentException(msg)
        # since strict implies that the points or features are the same,
        # if one object does not have names along the axis, but the length
        # matches, we will assume that the unnamed should have the same names
        if onFeature is None:
            if hasNamesL and not hasNamesR:
                setNamesR(namesL())
            elif not hasNamesL and hasNamesR:
                setNamesL(namesR())
            elif not hasNamesL and not hasNamesR:
                strictNames = ['_STRICT' + DEFAULT_PREFIX + str(i) for i in range(countL)]
                setNamesL(strictNames)
                setNamesR(strictNames)
        # if using strict with onFeature instead of point names, we need to
        # make sure each id has a unique match in the other object
        elif axis == 'point':
            try:
                self[0, onFeature]
                tmpOther[0, onFeature]
            except KeyError:
                msg = "could not locate feature '{0}' in both objects".format(onFeature)
                raise ArgumentException(msg)
            if len(set(self[:, onFeature])) != len(self.points):
                msg = "when point='strict', onFeature must contain only unique values"
                raise ArgumentException(msg)
            if sorted(self[:, onFeature]) != sorted(tmpOther[:, onFeature]):
                msg = "When point='strict', onFeature must have a unique, "
                msg += "matching value in each object"
                raise ArgumentException(msg)

        return self._genericMergeFrontend(tmpOther, point, feature, onFeature, axis)

    def _genericMergeFrontend(self, other, point, feature, onFeature, strict=None):
        # validation
        if (onFeature is None and point == "intersection") and not (self._pointNamesCreated() and other._pointNamesCreated()):
            msg = "Point names are required in both objects when "
            msg += "point='intersection'"
            raise ArgumentException(msg)
        if feature == "intersection" and not (self._featureNamesCreated() and other._featureNamesCreated()):
            msg = "Feature names are required in both objects when "
            msg += "feature='intersection'"
            raise ArgumentException(msg)
        defaultPtsL = self._anyDefaultPointNames() and not self._allDefaultPointNames()
        defaultPtsR = other._anyDefaultPointNames() and not other._allDefaultPointNames()
        if onFeature is not None:
            try:
                self[0, onFeature]
                other[0, onFeature]
            except KeyError:
                msg = "could not locate feature '{0}' in both objects".format(onFeature)
                raise ArgumentException(msg)
            uniqueFtL = len(set(self[:, onFeature])) == len(self.points)
            uniqueFtR = len(set(other[:, onFeature])) == len(other.points)
            if not (uniqueFtL or uniqueFtR):
                msg = "UML only supports joining on a feature which "
                msg += "contains only unique values in one or both objects."
                raise ArgumentException(msg)

        numPtsL = len(self.points)
        numFtsL = len(self.features)
        matchingFts = self._getMatchingNames('feature', other)
        matchingFtIdx = [[], []]
        for name in matchingFts:
            idxL = self.features.getIndex(name)
            idxR = other.features.getIndex(name)
            matchingFtIdx[0].append(idxL)
            matchingFtIdx[1].append(idxR)

        if self.getTypeString() != other.getTypeString():
            other = other.copyAs(self.getTypeString())
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
                self.features.setNames(other.features.getNames())
        elif feature == "intersection":
            if self._featureNamesCreated():
                ftNames = [n for n in self.features.getNames()
                           if n in matchingFts]
                self.features.setNames(ftNames)
        elif feature == "union":
            if self._featureNamesCreated() and other._featureNamesCreated():
                ftNamesL = self.features.getNames()
                ftNamesR = [name for name in other.features.getNames()
                            if name not in matchingFts]
                ftNames = ftNamesL + ftNamesR
                self.features.setNames(ftNames)
            elif self._featureNamesCreated():
                ftNamesL = self.features.getNames()
                ftNamesR = [DEFAULT_PREFIX + str(i) for i
                            in range(len(other.features))]
                ftNames = ftNamesL + ftNamesR
                self.features.setNames(ftNames)
            elif other._featureNamesCreated():
                ftNamesL = [DEFAULT_PREFIX + str(i) for i
                            in range(len(self.features))]
                ftNamesR = other.features.getNames()
                ftNames = ftNamesL + ftNamesR
                self.features.setNames(ftNames)
        # no name setting needed for left

        if strict == 'point':
            if ('_STRICT' in self.points.getName(0)
                  and '_STRICT' in other.points.getName(0)):
                # objects did not have point names
                self.pointNames = None
                self.pointNamesInverse = None
            elif '_STRICT' in self.points.getName(0):
                # use point names from other object
                self.points.setNames(other.points.getNames())
        elif onFeature is None and point == 'left':
            if self._pointNamesCreated():
                self.points.setNames(self.points.getNames())
        elif onFeature is None and point == 'intersection':
            # default names cannot be included in intersection
            ptNames = [name for name in self.points.getNames()
                       if name in other.points.getNames()
                       and not name.startswith(DEFAULT_PREFIX)]
            self.points.setNames(ptNames)
        elif onFeature is None:
            # union cases
            if self._pointNamesCreated() and other._pointNamesCreated():
                ptNamesL = self.points.getNames()
                if other._anyDefaultPointNames():
                    # handle default name conflicts
                    ptNamesR = [self._nextDefaultName('point') if
                                n.startswith(DEFAULT_PREFIX) else n
                                for n in self.points.getNames()]
                else:
                    ptNamesR = other.points.getNames()
                ptNames = ptNamesL + [name for name in ptNamesR
                                      if name not in ptNamesL]
                self.points.setNames(ptNames)
            elif self._pointNamesCreated():
                ptNamesL = self.points.getNames()
                ptNamesR = [self._nextDefaultName('point') for _
                            in range(len(other.points))]
                ptNames = ptNamesL + ptNamesR
                self.points.setNames(ptNames)
            elif other._pointNamesCreated():
                ptNamesL = [other._nextDefaultName('point') for _
                            in range(len(self.points))]
                ptNamesR = other.points.getNames()
                ptNames = ptNamesL + ptNamesR
                self.points.setNames(ptNames)
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

    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################

    def __mul__(self, other):
        """
        Perform matrix multiplication or scalar multiplication on this
        object depending on the input ``other``.
        """
        if (not isinstance(other, UML.data.Base)
                and not dataHelpers._looksNumeric(other)):
            return NotImplemented

        # Test element type self
        if self._pointCount == 0 or self._featureCount == 0:
            msg = "Cannot do a multiplication when points or features is empty"
            raise ImproperActionException(msg)

        # test element type other
        if isinstance(other, UML.data.Base):
            if len(other.points) == 0 or len(other.features) == 0:
                msg = "Cannot do a multiplication when points or features is "
                msg += "empty"
                raise ImproperActionException(msg)

            if self._featureCount != len(other.points):
                msg = "The number of features in the calling object must "
                msg += "match the point in the callee object."
                raise ArgumentException(msg)

            self._validateEqualNames('feature', 'point', '__mul__', other)

        try:
            ret = self._mul__implementation(other)
        except Exception as e:
            #TODO: improve how the exception is catch
            self._numericValidation()
            other._numericValidation()
            raise e

        if isinstance(other, UML.data.Base):
            if self._pointNamesCreated():
                ret.points.setNames(self.points.getNames())
            if other._featureNamesCreated():
                ret.features.setNames(other.features.getNames())

        pathSource = 'merge' if isinstance(other, UML.data.Base) else 'self'

        dataHelpers.binaryOpNamePathMerge(self, other, ret, None, pathSource)

        return ret

    def __rmul__(self, other):
        """
        Perform scalar multiplication with this object on the right
        """
        if dataHelpers._looksNumeric(other):
            return self.__mul__(other)
        else:
            return NotImplemented

    def __imul__(self, other):
        """
        Perform in place matrix multiplication or scalar multiplication,
        depending in the input ``other``.
        """
        ret = self.__mul__(other)
        if ret is not NotImplemented:
            self.referenceDataFrom(ret)
            ret = self

        return ret

    def __add__(self, other):
        """
        Perform addition on this object, element wise if 'other' is a
        UML data object, or element wise with a scalar if other is some
        kind of numeric value.
        """
        return self._genericNumericBinary('__add__', other)

    def __radd__(self, other):
        """
        Perform scalar addition with this object on the right
        """
        return self._genericNumericBinary('__radd__', other)

    def __iadd__(self, other):
        """
        Perform in-place addition on this object, element wise if
        ``other`` is a UML data object, or element wise with a scalar if
        ``other`` is some kind of numeric value.
        """
        return self._genericNumericBinary('__iadd__', other)

    def __sub__(self, other):
        """
        Subtract from this object, element wise if ``other`` is a UML
        data object, or element wise by a scalar if ``other`` is some
        kind of numeric value.
        """
        return self._genericNumericBinary('__sub__', other)

    def __rsub__(self, other):
        """
        Subtract each element of this object from the given scalar.
        """
        return self._genericNumericBinary('__rsub__', other)

    def __isub__(self, other):
        """
        Subtract (in place) from this object, element wise if ``other``
        is a UML data object, or element wise with a scalar if ``other``
        is some kind of numeric value.
        """
        return self._genericNumericBinary('__isub__', other)

    def __div__(self, other):
        """
        Perform division using this object as the numerator, elementwise
        if ``other`` is a UML data object, or element wise by a scalar
        if other is some kind of numeric value.
        """
        return self._genericNumericBinary('__div__', other)

    def __rdiv__(self, other):
        """
        Perform element wise division using this object as the
        denominator, and the given scalar value as the numerator.
        """
        return self._genericNumericBinary('__rdiv__', other)

    def __idiv__(self, other):
        """
        Perform division (in place) using this object as the numerator,
        elementwise if ``other`` is a UML data object, or elementwise by
        a scalar if ``other`` is some kind of numeric value.
        """
        return self._genericNumericBinary('__idiv__', other)

    def __truediv__(self, other):
        """
        Perform true division using this object as the numerator,
        elementwise if ``other`` is a UML data object, or element wise
        by a scalar if other is some kind of numeric value.
        """
        return self._genericNumericBinary('__truediv__', other)

    def __rtruediv__(self, other):
        """
        Perform element wise true division using this object as the
        denominator, and the given scalar value as the numerator.
        """
        return self._genericNumericBinary('__rtruediv__', other)

    def __itruediv__(self, other):
        """
        Perform true division (in place) using this object as the
        numerator, elementwise if ``other`` is a UML data object, or
        elementwise by a scalar if ``other`` is some kind of numeric
        value.
        """
        return self._genericNumericBinary('__itruediv__', other)

    def __floordiv__(self, other):
        """
        Perform floor division using this object as the numerator,
        elementwise if ``other`` is a UML data object, or elementwise by
        a scalar if ``other`` is some kind of numeric value.
        """
        return self._genericNumericBinary('__floordiv__', other)

    def __rfloordiv__(self, other):
        """
        Perform elementwise floor division using this object as the
        denominator, and the given scalar value as the numerator.

        """
        return self._genericNumericBinary('__rfloordiv__', other)

    def __ifloordiv__(self, other):
        """
        Perform floor division (in place) using this object as the
        numerator, elementwise if ``other`` is a UML data object, or
        elementwise by a scalar if ```other``` is some kind of numeric
        value.
        """
        return self._genericNumericBinary('__ifloordiv__', other)

    def __mod__(self, other):
        """
        Perform mod using the elements of this object as the dividends,
        elementwise if ``other`` is a UML data object, or elementwise by
        a scalar if other is some kind of numeric value.
        """
        return self._genericNumericBinary('__mod__', other)

    def __rmod__(self, other):
        """
        Perform mod using the elements of this object as the divisors,
        and the given scalar value as the dividend.
        """
        return self._genericNumericBinary('__rmod__', other)

    def __imod__(self, other):
        """
        Perform mod (in place) using the elements of this object as the
        dividends, elementwise if 'other' is a UML data object, or
        elementwise by a scalar if other is some kind of numeric value.
        """
        return self._genericNumericBinary('__imod__', other)

    @to2args
    def __pow__(self, other, z):
        """
        Perform exponentiation (iterated __mul__) using the elements of
        this object as the bases, elemen wise if ``other`` is a UML data
        object, or elementwise by a scalar if ``other`` is some kind of
        numeric value.
        """
        if self._pointCount == 0 or self._featureCount == 0:
            msg = "Cannot do ** when points or features is empty"
            raise ImproperActionException(msg)
        if not dataHelpers._looksNumeric(other):
            msg = "'other' must be an instance of a scalar"
            raise ArgumentException(msg)
        if other != int(other):
            raise ArgumentException("other may only be an integer type")
        if other < 0:
            raise ArgumentException("other must be greater than zero")

        if self._pointNamesCreated():
            retPNames = self.points.getNames()
        else:
            retPNames = None
        if self._featureNamesCreated():
            retFNames = self.features.getNames()
        else:
            retFNames = None

        if other == 1:
            ret = self.copy()
            ret._name = dataHelpers.nextDefaultObjectName()
            return ret

        # exact conditions in which we need to instantiate this object
        if other == 0 or other % 2 == 0:
            identityPNames = 'automatic' if retPNames is None else retPNames
            identityFNames = 'automatic' if retFNames is None else retFNames
            identity = UML.createData(self.getTypeString(),
                                      numpy.eye(self._pointCount),
                                      pointNames=identityPNames,
                                      featureNames=identityFNames,
                                      useLog=False)
        if other == 0:
            return identity

        # this means that we don't start with a multiplication at the ones
        # place, so we need to reserve the identity as the in progress return
        # value
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

        ret.points.setNames(retPNames)
        ret.features.setNames(retFNames)

        ret._name = dataHelpers.nextDefaultObjectName()

        return ret

    def __ipow__(self, other):
        """
        Perform in-place exponentiation (iterated __mul__) using the
        elements of this object as the bases, element wise if ``other``
        is a UML data object, or elementwise by a scalar if ``other`` is
        some kind of numeric value.
        """
        ret = self.__pow__(other)
        self.referenceDataFrom(ret)
        return self

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
        ret = self.elements.calculate(abs)
        if self._pointNamesCreated():
            ret.points.setNames(self.points.getNames())
        else:
            ret.points.setNames(None)
        if self._featureNamesCreated():
            ret.features.setNames(self.features.getNames())
        else:
            ret.points.setNames(None)

        ret._name = dataHelpers.nextDefaultObjectName()
        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath
        return ret

    def _numericValidation(self):
        if self._pointCount > 0:
            try:
                self.elements.calculate(dataHelpers._checkNumeric)
            except ValueError:
                msg = "This data object contains non numeric data, cannot do "
                msg += "this operation"
                raise ArgumentException(msg)

    def _genericNumericBinary_sizeValidation(self, opName, other):
        if self._pointCount != len(other.points):
            msg = "The number of points in each object must be equal. "
            msg += "(self=" + str(self._pointCount) + " vs other="
            msg += str(len(other.points)) + ")"
            raise ArgumentException(msg)
        if self._featureCount != len(other.features):
            msg = "The number of features in each object must be equal."
            raise ArgumentException(msg)

        if self._pointCount == 0 or self._featureCount == 0:
            msg = "Cannot do " + opName + " when points or features is empty"
            raise ImproperActionException(msg)

    def _genericNumericBinary_validation(self, opName, other):
        isUML = isinstance(other, UML.data.Base)

        if not isUML and not dataHelpers._looksNumeric(other):
            msg = "'other' must be an instance of a UML data object or a "
            msg += "scalar"
            raise ArgumentException(msg)

        # Test element type self
        self._numericValidation()

        # test element type other
        if isUML:
            other._numericValidation()

        divNames = ['__div__', '__rdiv__', '__idiv__', '__truediv__',
                    '__rtruediv__', '__itruediv__', '__floordiv__',
                    '__rfloordiv__', '__ifloordiv__', '__mod__', '__rmod__',
                    '__imod__', ]
        if isUML and opName in divNames:
            if other.containsZero():
                msg = "Cannot perform " + opName + " when the second argument "
                msg += "contains any zeros"
                raise ZeroDivisionError(msg)
            if isinstance(other, UML.data.Matrix):
                if False in numpy.isfinite(other.data):
                    msg = "Cannot perform " + opName + " when the second "
                    msg += "argument contains any NaNs or Infs"
                    raise ArgumentException(msg)
        if not isUML and opName in divNames:
            if other == 0:
                msg = "Cannot perform " + opName + " when the second argument "
                msg += "is zero"
                raise ZeroDivisionError(msg)

    def _genericNumericBinary(self, opName, other):

        isUML = isinstance(other, UML.data.Base)

        if isUML:
            if opName.startswith('__r'):
                return NotImplemented

            self._genericNumericBinary_sizeValidation(opName, other)
            self._validateEqualNames('point', 'point', opName, other)
            self._validateEqualNames('feature', 'feature', opName, other)

        # figure out return obj's point / feature names
        # if unary:
        (retPNames, retFNames) = (None, None)

        if opName in ['__pos__', '__neg__', '__abs__'] or not isUML:
            if self._pointNamesCreated():
                retPNames = self.points.getNames()
            if self._featureNamesCreated():
                retFNames = self.features.getNames()
        # else (everything else that uses this helper is a binary scalar op)
        else:
            (retPNames, retFNames) = dataHelpers.mergeNonDefaultNames(self,
                                                                      other)
        try:
            ret = self._genericNumericBinary_implementation(opName, other)
        except Exception as e:
            self._genericNumericBinary_validation(opName, other)
            raise e


        if retPNames is not None:
            ret.points.setNames(retPNames)
        else:
            ret.points.setNames(None)

        if retFNames is not None:
            ret.features.setNames(retFNames)
        else:
            ret.features.setNames(None)

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

    def _arrangeFeatureNames(self, maxWidth, nameLength, colSep, colHold,
                             nameHold):
        """
        Prepare feature names for string output. Grab only those names
        that fit according to the given width limitation, process them
        for length, omit them if they are default. Returns a list of
        prepared names, and a list of the length of each name in the
        return.
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
        endIndex = self._featureCount // 2
        if self._featureCount % 2 == 1:
            endIndex *= -1
            endIndex -= 1
        currIndex = 0
        numAdded = 0
        while totalWidth < maxWidth and currIndex != endIndex:
            nameIndex = currIndex
            if currIndex < 0:
                nameIndex = self._featureCount + currIndex

            currName = self.features.getName(nameIndex)

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
            allCols = rawStillUnder and (numAdded == (self._featureCount - 1))
            # only add this column if it won't put us over the limit,
            # OR if it is the last one (and under the limit without the col
            # holder)
            if totalWidth < maxWidth or allCols:
                numAdded += 1
                currNames.append(currName)

                # the width value goes in different lists depending on index
                if currIndex < 0:
                    currIndex = abs(currIndex)
                else:
                    currIndex = (-1 * currIndex) - 1

        # combine the tables. Have to reverse rTable because entries were
        # appended in a right to left order
        rNames.reverse()
        if numAdded == self._featureCount:
            lNames += rNames
        else:
            lNames += [colHold] + rNames

        return lNames

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

    def _arrangeDataWithLimits(self, maxWidth, maxHeight, sigDigits=3,
                               maxStrLength=19, colSep=' ', colHold='--',
                               rowHold='|', strHold='...'):
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
        if self._pointCount == 0 or self._featureCount == 0:
            return [[]], []

        if maxHeight < 2 and maxHeight != self._pointCount:
            msg = "If the number of points in this object is two or greater, "
            msg += "then we require that the input argument maxHeight also "
            msg += "be greater than or equal to two."
            raise ArgumentException(msg)

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
            currWidth = 0
            currTable = lTable if currIndex >= 0 else rTable
            currCol = []

            # check all values in this column (in the accepted rows)
            for i in range(len(combinedRowIDs)):
                rID = combinedRowIDs[i]
                val = self[rID, currIndex]
                valFormed = formatIfNeeded(val, sigDigits)
                if len(valFormed) < maxStrLength:
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

            totalWidth += currWidth + len(colSep)
            # test: total width is under max without column holder
            allCols = totalWidth - (cHoldTotal) < maxWidth
            # test: the column we are trying to add is the last one possible
            allCols = allCols and (numAdded == (self._featureCount - 1))
            # only add this column if it won't put us over the limit
            if totalWidth < maxWidth or allCols:
                numAdded += 1
                for i in range(len(currCol)):
                    if len(currTable) != len(currCol):
                        currTable.append([currCol[i]])
                    else:
                        currTable[i].append(currCol[i])

                # the width value goes in different lists depending on index
                if currIndex < 0:
                    currIndex = abs(currIndex)
                    rColWidths.append(currWidth)
                else:
                    currIndex = (-1 * currIndex) - 1
                    lColWidths.append(currWidth)

        # combine the tables. Have to reverse rTable because entries were
        # appended in a right to left order
        rColWidths.reverse()
        if numAdded == self._featureCount:
            lColWidths += rColWidths
        else:
            lColWidths += [cHoldWidth] + rColWidths
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
        Returns a set containing those pointNames in this object that
        are not also in the input object.
        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "pointName difference"
            raise ArgumentException(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) - six.viewkeys(other.pointNames)

    def _featureNameDifference(self, other):
        """
        Returns a set containing those featureNames in this object that
        are not also in the input object.
        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "featureName difference"
            raise ArgumentException(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return (six.viewkeys(self.featureNames)
                - six.viewkeys(other.featureNames))

    def _pointNameIntersection(self, other):
        """
        Returns a set containing only those pointNames that are shared
        by this object and the input object.
        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "pointName intersection"
            raise ArgumentException(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) & six.viewkeys(other.pointNames)

    def _featureNameIntersection(self, other):
        """
        Returns a set containing only those featureNames that are shared
        by this object and the input object.
        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "featureName intersection"
            raise ArgumentException(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return (six.viewkeys(self.featureNames)
                & six.viewkeys(other.featureNames))

    def _pointNameSymmetricDifference(self, other):
        """
        Returns a set containing only those pointNames not shared
        between this object and the input object.
        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "pointName difference"
            raise ArgumentException(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) ^ six.viewkeys(other.pointNames)

    def _featureNameSymmetricDifference(self, other):
        """
        Returns a set containing only those featureNames not shared
        between this object and the input object.
        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "featureName difference"
            raise ArgumentException(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return (six.viewkeys(self.featureNames)
                ^ six.viewkeys(other.featureNames))

    def _pointNameUnion(self, other):
        """
        Returns a set containing all pointNames in either this object or
        the input object.
        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "pointNames union"
            raise ArgumentException(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'point')

        return six.viewkeys(self.pointNames) | six.viewkeys(other.pointNames)

    def _featureNameUnion(self, other):
        """
        Returns a set containing all featureNames in either this object
        or the input object.
        """
        if other is None:
            raise ArgumentException("The other object cannot be None")
        if not isinstance(other, Base):
            msg = "Must provide another representation type to determine "
            msg += "featureName union"
            raise ArgumentException(msg)

        self._defaultNamesGeneration_NamesSetOperations(other, 'feature')

        return (six.viewkeys(self.featureNames)
                | six.viewkeys(other.featureNames))

    def _equalPointNames(self, other):
        if other is None or not isinstance(other, Base):
            return False
        return self._equalNames(self.points.getNames(),
                                other.points.getNames())

    def _equalFeatureNames(self, other):
        if other is None or not isinstance(other, Base):
            return False
        return (self._equalNames(self.features.getNames(),
                                 other.features.getNames()))

    def _equalNames(self, selfNames, otherNames):
        """
        Private function to determine equality of either pointNames of
        featureNames. It ignores equality of default values, considering
        only whether non default names consistent (position by position)
        and uniquely positioned (if a non default name is present in
        both, then it is in the same position in both).
        """
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

    def _getPointIndex(self, identifier):
        return self._getIndex(identifier, 'point')

    def _getFeatureIndex(self, identifier):
        return self._getIndex(identifier, 'feature')

    def _getIndex(self, identifier, axis):
        if axis == 'point':
            num = self._pointCount
            axisObj = getattr(self, 'points')
        else:
            num = self._featureCount
            axisObj = getattr(self, 'features')
        accepted = (six.string_types, int, numpy.integer)

        toReturn = identifier
        if num == 0:
            msg = "There are no valid " + axis + "identifiers; "
            msg += "this object has 0 " + axis + "s"
            raise ArgumentException(msg)
        if identifier is None:
            msg = "An identifier cannot be None."
            raise ArgumentException(msg)
        if not isinstance(identifier, accepted):
            msg = "The identifier must be either a string (a valid "
            msg += axis + " name) or an integer (python or numpy) index "
            msg += "between 0 and " + str(num - 1) + " inclusive. "
            msg += "Instead we got: " + str(identifier)
            raise ArgumentException(msg)
        if isinstance(identifier, (int, numpy.integer)):
            if identifier < 0:
                identifier = num + identifier
                toReturn = identifier
            if identifier < 0 or identifier >= num:
                msg = "The given index " + str(identifier) + " is outside of "
                msg += "the range of possible indices in the " + axis
                msg += " axis (0 to " + str(num - 1) + ")."
                raise ArgumentException(msg)
        if isinstance(identifier, six.string_types):
            try:
                toReturn = axisObj.getIndex(identifier)
            except KeyError:
                msg = "The " + axis + " name '" + identifier
                msg += "' cannot be found."
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

    def _constructIndicesList(self, axis, values, argName=None):
        """
        Construct a list of indices from a valid integer (python or numpy) or
        string, or an iterable, list-like container of valid integers and/or
        strings

        """
        if argName is None:
            argName = axis + 's'
        # pandas DataFrames are iterable but do not iterate through the values
        if pd and isinstance(values, pd.DataFrame):
            msg = "A pandas DataFrame object is not a valid input "
            msg += "for '{0}'. ".format(argName)
            msg += "Only one-dimensional objects are accepted."
            raise ArgumentException(msg)

        valuesList = valuesToPythonList(values, argName)
        try:
            indicesList = [self._getIndex(val, axis) for val in valuesList]
        except ArgumentException as ae:
            msg = "Invalid value for the argument '{0}'. ".format(argName)
            # add more detail to msg; slicing to exclude quotes
            msg += str(ae)[1:-1]
            raise ArgumentException(msg)

        return indicesList

    def _validateAxis(self, axis):
        if axis != 'point' and axis != 'feature':
            msg = 'axis parameter may only be "point" or "feature"'
            raise ArgumentException(msg)

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
            raise ArgumentException(msg)

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _isIdentical_implementation(self, other):
        pass

    @abstractmethod
    def _writeFile_implementation(self, outPath, format, includePointNames,
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
    def _copyAs_implementation(self, format):
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
    def _mul__implementation(self, other):
        pass

class BasePoints(Axis, Points):
    pass

class BaseFeatures(Axis, Features):
    pass

class BaseElements(Elements):
    pass

def cmp(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0
