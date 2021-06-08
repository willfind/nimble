"""
Anchors the hierarchy of data representation types, providing stubs and
common functions.
"""

# TODO conversions
# TODO who sorts inputs to derived implementations?

import sys
import math
import numbers
import itertools
import os.path
from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.core.logger import handleLogging
from nimble.core.logger import produceFeaturewiseReport
from nimble.core.logger import produceAggregateReport
from nimble._utility import cloudpickle, h5py, plt
from nimble._utility import isDatetime
from .stretch import Stretch
from ._dataHelpers import formatIfNeeded
from ._dataHelpers import constructIndicesList
from ._dataHelpers import createListOfDict, createDictOfList
from ._dataHelpers import createDataNoValidation
from ._dataHelpers import csvCommaFormat
from ._dataHelpers import validateElementFunction, wrapMatchFunctionFactory
from ._dataHelpers import ElementIterator1D
from ._dataHelpers import elementQueryFunction
from ._dataHelpers import limitedTo2D
from ._dataHelpers import arrangeFinalTable
from ._dataHelpers import inconsistentNames, equalNames
from ._dataHelpers import validateRangeOrder
from ._dataHelpers import pyplotRequired, plotOutput, plotFigureHandling
from ._dataHelpers import plotUpdateAxisLimits, plotAxisLimits
from ._dataHelpers import plotAxisLabels, plotXTickLabels
from ._dataHelpers import plotConfidenceIntervalMeanAndError, plotErrorBars
from ._dataHelpers import plotSingleBarChart, plotMultiBarChart
from ._dataHelpers import looksNumeric, checkNumeric
from ._dataHelpers import mergeNames, mergeNonDefaultNames
from ._dataHelpers import makeNamesLines
from ._dataHelpers import binaryOpNamePathMerge
from ._dataHelpers import indicesSplit

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

class Base(ABC):
    """
    The base class for all nimble data objects.

    All data types inherit their functionality from this object, then
    apply their own implementations of its methods (when necessary).
    Methods in this object apply to the entire object or its elements
    and its ``points`` and ``features`` attributes provide addtional
    methods that apply point-by-point and feature-by-feature,
    respectively.

    Attributes
    ----------
    points
        Access to methods applying to points.
    features
        Access to methods applying to features.
    shape : tuple
        The number of points and features in the object in the format
        (points, features).
    dimensions : tuple
        The actual dimensions of the data in the object. All data is
        stored two-dimensionally, so for objects with more than two-
        dimensions this value will differ from the ``shape`` attribute.
    name : str
        A name to call this object when printing or logging.
    absolutePath : str
        The absolute path to the data file.
    relativePath : str
        The relative path to the data file.
    path : str
        The path to the data file.
    """
    _id = 0

    def __init__(self, shape, pointNames=None, featureNames=None, name=None,
                 paths=(None, None), **kwds):
        """
        Class defining important data manipulation operations.

        Specifically, this includes setting the object _id, name, shape,
        originating paths for the data, and sets the point and feature axis
        objects. Note: this method (as should all other __init__ methods in
        this hierarchy) makes use of super().

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
        """
        self._id = Base._id
        Base._id += 1

        self._shape = list(shape)
        self._name = name

        self._points = self._getPoints(pointNames)
        self._features = self._getFeatures(featureNames)

        # Set up paths
        if paths[0] is not None and not isinstance(paths[0], str):
            msg = "paths[0] must be None, an absolute path or web link to "
            msg += "the file from which the data originates"
            raise InvalidArgumentType(msg)
        if (paths[0] is not None
                and not os.path.isabs(paths[0])
                and not paths[0].startswith('http')):
            raise InvalidArgumentValue("paths[0] must be an absolute path")
        self._absPath = paths[0]

        if paths[1] is not None and not isinstance(paths[1], str):
            msg = "paths[1] must be None or a relative path to the file from "
            msg += "which the data originates"
            raise InvalidArgumentType(msg)
        self._relPath = paths[1]

        # call for safety
        super().__init__(**kwds)

    #######################
    # Property Attributes #
    #######################

    @property
    def shape(self):
        """
        The number of points and features in the object in the format
        (points, features).
        """
        if len(self._shape) > 2:
            return self._shape[0], np.prod(self._shape[1:])
        return self._shape[0], self._shape[1]

    @property
    def dimensions(self):
        """
        The true dimensions of this object.
        """
        return tuple(self._shape)

    @property
    def points(self):
        """
        An object handling functions manipulating data by points.

        See Also
        --------
        Points
        """
        return self._points

    @property
    def features(self):
        """
        An object handling functions manipulating data by features.

        See Also
        --------
        Features
        """
        return self._features

    @property
    def name(self):
        """
        A name to be displayed when printing or logging this object
        """
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str) and value is not None:
            msg = "The name of an object may only be a string or None"
            raise ValueError(msg)
        self._name = value

    @property
    def absolutePath(self):
        """
        The path to the file this data originated from in absolute form.
        """
        return self._absPath

    @property
    def relativePath(self):
        """
        The path to the file this data originated from in relative form.
        """
        return self._relPath

    @property
    def path(self):
        """
        The path to the file this data originated from.
        """
        return self.absolutePath

    @contextmanager
    def _treatAs2D(self):
        """
        This can be applied when dimensionality does not affect an
        operation, but an method call within the operation is blocked
        when the data has more than two dimensions due to the ambiguity
        in the definition of elements.
        """
        if len(self._shape) > 2:
            savedShape = self._shape
            self._shape = [len(self.points), len(self.features)]
            try:
                yield self
            finally:
                self._shape = savedShape
        else:
            yield self

    ########################
    # Low Level Operations #
    ########################
    @limitedTo2D
    def __len__(self):
        # ordered such that the larger axis is always printed, even
        # if they are both in the range [0,1]
        if len(self.points) == 0 or len(self.features) == 0:
            return 0
        if len(self.points) == 1:
            return len(self.features)
        if len(self.features) == 1:
            return len(self.points)

        msg = "len() is undefined when the number of points ("
        msg += str(len(self.points))
        msg += ") and the number of features ("
        msg += str(len(self.features))
        msg += ") are both greater than 1"
        raise ImproperObjectAction(msg)

    @limitedTo2D
    def __iter__(self):
        if len(self.points) in [0, 1] or len(self.features) in [0, 1]:
            return ElementIterator1D(self)

        msg = "Cannot iterate over two-dimensional objects because the "
        msg = "iteration order is arbitrary. Try the iterateElements() method."
        raise ImproperObjectAction(msg)

    def __bool__(self):
        return self._shape[0] > 0 and self._shape[-1] > 0

    @limitedTo2D
    def iterateElements(self, order='point', only=None):
        """
        Iterate over each element in this object.

        Provide an iterator which returns elements in the designated
        ``order``. Optionally, the output of the iterator can be
        restricted to certain elements by the ``only`` function.

        Parameters
        ----------
        order : str
           'point' or 'feature' to indicate how the iterator will access
           the elements in this object.
        only : function, None
           If None, the default, all elements will be returned by the
           iterator. If a function, it must return True or False
           indicating whether the iterator should return the element.

        See Also
        --------
        Points, Features

        Examples
        --------
        >>> from nimble.match import nonZero, positive
        >>> rawData = [[0, 1, 2], [-2, -1, 0]]
        >>> data = nimble.data('Matrix', rawData)
        >>> list(data.iterateElements(order='point'))
        [0, 1, 2, -2, -1, 0]
        >>> list(data.iterateElements(order='feature'))
        [0, -2, 1, -1, 2, 0]
        >>> list(data.iterateElements(order='point', only=nonZero))
        [1, 2, -2, -1]
        >>> list(data.iterateElements(order='feature', only=positive))
        [1, 2]
        """
        if order not in ['point', 'feature']:
            msg = "order must be the string 'point' or 'feature'"
            if not isinstance(order, str):
                raise InvalidArgumentType(msg)
            raise InvalidArgumentValue(msg)
        if only is not None and not callable(only):
            raise InvalidArgumentType('if not None, only must be callable')
        return self._iterateElements_implementation(order, only)


    ###########################
    # Higher Order Operations #
    ###########################
    @limitedTo2D
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
        >>> data = nimble.data('Matrix', raw, featureNames=['replace'])
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
        if len(self.points) == 0:
            msg = "This action is impossible, the object has 0 points"
            raise ImproperObjectAction(msg)

        index = self.features.getIndex(featureToReplace)

        replace = self.features.extract([index], useLog=False)

        uniqueVals = replace.countUniqueElements().keys()
        uniqueIdx = {key: i for i, key in enumerate(uniqueVals)}

        binaryObj = replace._replaceFeatureWithBinaryFeatures_implementation(
            uniqueIdx)

        binaryObj.points.setNames(self.points._getNamesNoGeneration(),
                                  useLog=False)
        ftNames = []

        if replace.features._namesCreated():
            prefix = replace.features.getName(0) + "="
        else:
            prefix = str(index) + '='
        for val in uniqueVals:
            ftNames.append(prefix + str(val))
        binaryObj.features.setNames(ftNames, useLog=False)

        # must use append if the object is now feature empty
        if len(self.features) == 0:
            self.features.append(binaryObj, useLog=False)
        else:
            # insert data at same index of the original feature
            self.features.insert(index, binaryObj, useLog=False)

        handleLogging(useLog, 'prep', "replaceFeatureWithBinaryFeatures",
                      self.getTypeString(),
                      Base.replaceFeatureWithBinaryFeatures, featureToReplace)

        return ftNames

    @limitedTo2D
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
        >>> data = nimble.data('Matrix', raw, featureNames=featureNames)
        >>> mapping = data.transformFeatureToIntegers('transform')
        >>> mapping
        {0: 'a', 1: 'b', 2: 'c'}
        >>> data
        Matrix(
            [[1 0 1]
             [2 1 2]
             [3 2 3]]
            featureNames={'keep1':0, 'transform':1, 'keep2':2}
            )
        """
        if len(self.points) == 0:
            msg = "This action is impossible, the object has 0 points"
            raise ImproperObjectAction(msg)

        ftIndex = self.features.getIndex(featureToConvert)

        mapping = {}
        def applyMap(ft):
            integerValue = 0
            mapped = []
            for val in ft:
                if val in mapping:
                    mapped.append(mapping[val])
                elif val == 0:
                    # will preserve zeros if present in the feature
                    # increment all values that occurred before this zero
                    for key in mapping:
                        mapping[key] += 1
                    mapped = [v + 1 for v in mapped]
                    integerValue += 1
                    mapping[0] = 0
                    mapped.append(0)
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

    @limitedTo2D
    def transformElements(self, toTransform, points=None, features=None,
                          preserveZeros=False, skipNoneReturnValues=False,
                          useLog=None):
        """
        Modify each element using a function or mapping.

        Perform an inplace modification of the elements or subset of
        elements in this object.

        Parameters
        ----------
        toTransform : function, dict
            * function - in the form of toTransform(elementValue)
              or toTransform(elementValue, pointIndex, featureIndex)
            * dictionary -  map the current element [key] to the
              transformed element [value].
        points : identifier, list of identifiers
            May be a single point name or index, an iterable,
            container of point names and/or indices. None indicates
            application to all points.
        features : identifier, list of identifiers
            May be a single feature name or index, an iterable,
            container of feature names and/or indices. None indicates
            application to all features.
        preserveZeros : bool
            If True it does not apply toTransform to elements in the
            data that are 0, and that 0 is not modified.
        skipNoneReturnValues : bool
            If True, any time toTransform() returns None, the value
            originally in the data will remain unmodified.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        calculateOnElements, Points.transform, Features.transform

        Examples
        --------
        Simple transformation to all elements.

        >>> data = nimble.ones('Matrix', 5, 5)
        >>> data.transformElements(lambda elem: elem + 1)
        >>> data
        Matrix(
            [[2.000 2.000 2.000 2.000 2.000]
             [2.000 2.000 2.000 2.000 2.000]
             [2.000 2.000 2.000 2.000 2.000]
             [2.000 2.000 2.000 2.000 2.000]
             [2.000 2.000 2.000 2.000 2.000]]
            )

        Transform while preserving zero values.

        >>> data = nimble.identity('Sparse', 5)
        >>> data.transformElements(lambda elem: elem + 10,
        ...                        preserveZeros=True)
        >>> data
        Sparse(
            [[11.000 0.000  0.000  0.000  0.000 ]
             [0.000  11.000 0.000  0.000  0.000 ]
             [0.000  0.000  11.000 0.000  0.000 ]
             [0.000  0.000  0.000  11.000 0.000 ]
             [0.000  0.000  0.000  0.000  11.000]]
            )

        Transforming a subset of points and features.

        >>> data = nimble.ones('List', 4, 4)
        >>> data.transformElements(lambda elem: elem + 1,
        ...                        points=[0, 1], features=[0, 2])
        >>> data
        List(
            [[2.000 1.000 2.000 1.000]
             [2.000 1.000 2.000 1.000]
             [1.000 1.000 1.000 1.000]
             [1.000 1.000 1.000 1.000]]
            )

        Transforming with None return values. With the ``addTenToEvens``
        function defined below, An even values will be return a value,
        while an odd value will return None. If ``skipNoneReturnValues``
        is False, the odd values will be replaced with None (or nan
        depending on the object type) if set to True the odd values will
        remain as is. Both cases are presented.

        >>> def addTenToEvens(elem):
        ...     if elem % 2 == 0:
        ...         return elem + 10
        ...     return None
        >>> raw = [[1, 2, 3],
        ...        [4, 5, 6],
        ...        [7, 8, 9]]
        >>> dontSkip = nimble.data('Matrix', raw)
        >>> dontSkip.transformElements(addTenToEvens)
        >>> dontSkip
        Matrix(
            [[ nan   12.000  nan  ]
             [14.000  nan   16.000]
             [ nan   18.000  nan  ]]
            )
        >>> skip = nimble.data('Matrix', raw)
        >>> skip.transformElements(addTenToEvens,
        ...                        skipNoneReturnValues=True)
        >>> skip
        Matrix(
            [[1  12 3 ]
             [14 5  16]
             [7  18 9 ]]
            )
        """
        if points is not None:
            points = constructIndicesList(self, 'point', points)
        if features is not None:
            features = constructIndicesList(self, 'feature', features)

        transformer = validateElementFunction(toTransform, preserveZeros,
                                              skipNoneReturnValues,
                                              'toTransform')

        self._transform_implementation(transformer, points, features)

        handleLogging(useLog, 'prep', 'transformElements',
                      self.getTypeString(), Base.transformElements,
                      toTransform, points, features, preserveZeros,
                      skipNoneReturnValues)

    @limitedTo2D
    def calculateOnElements(self, toCalculate, points=None, features=None,
                            preserveZeros=False, skipNoneReturnValues=False,
                            outputType=None, useLog=None):
        """
        Return a new object with a calculation applied to each element.

        Apply a function or mapping to each element in this object or
        subset of points and features in this  object.

        Parameters
        ----------
        toCalculate : function, dict
            * function - in the form of toCalculate(elementValue)
              or toCalculate(elementValue, pointIndex, featureIndex)
            * dictionary -  map the current element [key] to the
              transformed element [value].
        points : point, list of points
            The subset of points to limit the calculation to. If None,
            the calculation will apply to all points.
        features : feature, list of features
            The subset of features to limit the calculation to. If None,
            the calculation will apply to all features.
        preserveZeros : bool
            Bypass calculation on zero values
        skipNoneReturnValues : bool
            Bypass values when ``toCalculate`` returns None. If False,
            the value None will replace the value if None is returned.
        outputType: nimble data type
            Return an object of the specified type. If None, the
            returned object will have the same type as the calling
            object.
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
        transformElements, Points.calculate, Features.calculate

        Examples
        --------
        Simple calculation on all elements.

        >>> data = nimble.ones('Matrix', 5, 5)
        >>> twos = data.calculateOnElements(lambda elem: elem + 1)
        >>> twos
        Matrix(
            [[2.000 2.000 2.000 2.000 2.000]
             [2.000 2.000 2.000 2.000 2.000]
             [2.000 2.000 2.000 2.000 2.000]
             [2.000 2.000 2.000 2.000 2.000]
             [2.000 2.000 2.000 2.000 2.000]]
            )

        Calculate while preserving zero values.

        >>> data = nimble.identity('Sparse', 5)
        >>> addTen = data.calculateOnElements(lambda x: x + 10,
        ...                                   preserveZeros=True)
        >>> addTen
        Sparse(
            [[11.000 0.000  0.000  0.000  0.000 ]
             [0.000  11.000 0.000  0.000  0.000 ]
             [0.000  0.000  11.000 0.000  0.000 ]
             [0.000  0.000  0.000  11.000 0.000 ]
             [0.000  0.000  0.000  0.000  11.000]]
            )

        Calculate on a subset of points and features.

        >>> data = nimble.ones('List', 4, 4)
        >>> calc = data.calculateOnElements(lambda elem: elem + 1,
        ...                                 points=[0, 1],
        ...                                 features=[0, 2])
        >>> calc
        List(
            [[2.000 2.000]
             [2.000 2.000]]
            )

        Calculating with None return values. With the ``addTenToEvens``
        function defined below, An even values will be return a value,
        while an odd value will return None. If ``skipNoneReturnValues``
        is False, the odd values will be replaced with None (or nan
        depending on the object type) if set to True the odd values will
        remain as is. Both cases are presented.

        >>> def addTenToEvens(elem):
        ...     if elem % 2 == 0:
        ...         return elem + 10
        ...     return None
        >>> raw = [[1, 2, 3],
        ...        [4, 5, 6],
        ...        [7, 8, 9]]
        >>> data = nimble.data('Matrix', raw)
        >>> dontSkip = data.calculateOnElements(addTenToEvens)
        >>> dontSkip
        Matrix(
            [[nan  12 nan]
             [ 14 nan  16]
             [nan  18 nan]]
            )
        >>> skip = data.calculateOnElements(addTenToEvens,
        ...                                 skipNoneReturnValues=True)
        >>> skip
        Matrix(
            [[1  12 3 ]
             [14 5  16]
             [7  18 9 ]]
            )
        """
        calculator = validateElementFunction(toCalculate, preserveZeros,
                                             skipNoneReturnValues,
                                             'toCalculate')

        ret = self._calculate_backend(calculator, points, features,
                                      preserveZeros, outputType)

        handleLogging(useLog, 'prep', 'calculateOnElements',
                      self.getTypeString(), Base.calculateOnElements,
                      toCalculate, points, features, preserveZeros,
                      skipNoneReturnValues, outputType)

        return ret

    @limitedTo2D
    def matchingElements(self, toMatch, points=None, features=None,
                         useLog=None):
        """
        Return an object of boolean values identifying matching values.

        Common matching functions can be found in nimble's match module.

        Parameters
        ----------
        toMatch: value, function, query
            * value - elements equal to the value return True
            * function - accepts an element as its only argument and
              returns a boolean value to indicate if the element is a
              match
            * query - string in the format 'OPERATOR VALUE' (i.e "< 10")
              where OPERATOR can be ==, !=, <, >, <=, or >=
        points : point, list of points
            The subset of points to limit the matching to. If None,
            the matching will apply to all points.
        features : feature, list of features
            The subset of features to limit the matching to. If None,
            the matching will apply to all features.
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
            This object will only contain boolean values.

        Examples
        --------
        >>> from nimble import match
        >>> raw = [[1, -1, 1], [-3, 3, -3]]
        >>> data = nimble.data('Matrix', raw)
        >>> isNegativeOne = data.matchingElements(-1)
        >>> isNegativeOne
        Matrix(
            [[False  True False]
             [False False False]]
            )

        >>> from nimble import match
        >>> raw = [[1, -1, None], [None, 3, -3]]
        >>> data = nimble.data('Matrix', raw)
        >>> isMissing = data.matchingElements(match.missing)
        >>> isMissing
        Matrix(
            [[False False  True]
             [ True False False]]
            )

        >>> from nimble import match
        >>> raw = [[1, -1, 1], [-3, 3, -3]]
        >>> data = nimble.data('Matrix', raw)
        >>> isPositive = data.matchingElements(">0")
        >>> isPositive
        Matrix(
            [[ True False  True]
             [False  True False]]
            )
        """
        matchArg = toMatch # preserve toMatch in original state for log
        if not callable(matchArg):
            query = elementQueryFunction(matchArg)
            if query is not None:
                func = query
            # if not a comparison string, element must equal matchArg
            else:
                matchVal = matchArg
                func = lambda elem: elem == matchVal
            matchArg = func

        wrappedMatch = wrapMatchFunctionFactory(matchArg)

        ret = self._calculate_backend(wrappedMatch, points, features,
                                      allowBoolOutput=True)

        pnames = self.points._getNamesNoGeneration()
        if pnames is not None and points is not None:
            ptIdx = constructIndicesList(self, 'point', points)
            pnames = [pnames[i] for i in ptIdx]
        fnames = self.features._getNamesNoGeneration()
        if fnames is not None and features is not None:
            ftIdx = constructIndicesList(self, 'feature', features)
            fnames = [fnames[j] for j in ftIdx]
        ret.points.setNames(pnames, useLog=False)
        ret.features.setNames(fnames, useLog=False)

        handleLogging(useLog, 'prep', 'matchingElements', self.getTypeString(),
                      Base.matchingElements, toMatch, points, features)

        return ret

    def _calculate_backend(self, calculator, points=None, features=None,
                           preserveZeros=False, outputType=None,
                           allowBoolOutput=False):
        if points is not None:
            points = constructIndicesList(self, 'point', points)
        if features is not None:
            features = constructIndicesList(self, 'feature', features)

        if outputType is not None:
            optType = outputType
        else:
            optType = self.getTypeString()
        # Use vectorized for functions with oneArg
        if calculator.oneArg:
            vectorized = np.vectorize(calculator)
            values = self._calculate_implementation(
                vectorized, points, features, preserveZeros)

        else:
            if not points:
                points = list(range(len(self.points)))
            if not features:
                features = list(range(len(self.features)))
            # if unable to vectorize, iterate over each point
            values = np.empty([len(points), len(features)])
            if allowBoolOutput:
                values = values.astype(np.bool_)
            pIdx = 0
            for i in points:
                fIdx = 0
                for j in features:
                    value = self[i, j]
                    currRet = calculator(value, i, j)
                    if (match.nonNumeric(currRet) and currRet is not None
                            and values.dtype != np.object_):
                        values = values.astype(np.object_)
                    values[pIdx, fIdx] = currRet
                    fIdx += 1
                pIdx += 1

        ret = nimble.data(optType, values, treatAsMissing=[None], useLog=False)

        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath

        return ret

    def _calculate_genericVectorized(self, function, points, features):
        # need points/features as arrays for indexing
        if points:
            points = np.array(points)
        else:
            points = np.array(range(len(self.points)))
        if features:
            features = np.array(features)
        else:
            features = np.array(range(len(self.features)))
        toCalculate = self.copy(to='numpyarray')
        # array with only desired points and features
        toCalculate = toCalculate[points[:, None], features]
        try:
            return function(toCalculate)
        except Exception: # pylint: disable=broad-except
            # change output type of vectorized function to object to handle
            # nonnumeric data
            function.otypes = [np.object_]
            return function(toCalculate)

    @limitedTo2D
    def countElements(self, condition):
        """
        The number of values which satisfy the condition.

        Parameters
        ----------
        condition : function, query
            * function - accepts an element as its only argument and
              returns a boolean value to indicate if the element should
              be counted
            * query - string in the format 'OPERATOR VALUE' (i.e "< 10")
              where OPERATOR can be ==, !=, <, >, <=, or >=

        Returns
        -------
        int

        See Also
        --------
        Points.count, Features.count

        Examples
        --------
        Using a python function.

        >>> def greaterThanZero(elem):
        ...     return elem > 0
        >>> data = nimble.identity('Matrix', 5)
        >>> numGreaterThanZero = data.countElements(greaterThanZero)
        >>> numGreaterThanZero
        5

        Using a string filter function.

        >>> numLessThanOne = data.countElements("<1")
        >>> numLessThanOne
        20
        """
        query = elementQueryFunction(condition)
        if query is not None:
            condition = query
        elif not hasattr(condition, '__call__'):
            msg = 'condition can only be a function or string containing a '
            msg += 'comparison operator and a value'
            raise InvalidArgumentType(msg)

        ret = self.calculateOnElements(condition, outputType='Matrix',
                                       useLog=False)
        return int(np.sum(ret._data))

    @limitedTo2D
    def countUniqueElements(self, points=None, features=None):
        """
        Count of each unique value in the data.

        Parameters
        ----------
        points : identifier, list of identifiers
            May be None indicating application to all points, a single
            name or index or an iterable of points and/or indices.
        features : identifier, list of identifiers
            May be None indicating application to all features, a single
            name or index or an iterable of names and/or indices.

        Returns
        -------
        dict
            Each unique value as keys and the number of times that
            value occurs as values.

        See Also
        --------
        nimble.calculate.uniqueCount

        Examples
        --------
        Count for all elements.

        >>> data = nimble.identity('Matrix', 5)
        >>> unique = data.countUniqueElements()
        >>> unique
        {0.0: 20, 1.0: 5}

        Count for a subset of elements.

        >>> data = nimble.identity('Matrix', 5)
        >>> unique = data.countUniqueElements(points=0,
        ...                                   features=[0, 1, 2])
        >>> unique
        {0.0: 2, 1.0: 1}
        """
        return self._countUnique_implementation(points, features)

    @limitedTo2D
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
        >>> top10 = nimble.data('DataFrame', raw, featureNames=ftNames)
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
            return tuple(point[i] for i in by)

        #if by is a list, then use findKey2; o.w. use findKey1
        if isinstance(by, (str, numbers.Number)):
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
                    res[k] = point.copy()
                else:
                    res[k].points.append(point.copy(), useLog=False)

            for k in res:
                res[k].features.delete(by, useLog=False)

        handleLogging(useLog, 'prep', "groupByFeature",
                      self.getTypeString(), Base.groupByFeature, by,
                      countUniqueValueOnly)

        return res

    @limitedTo2D
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
        if len(self.points) == 0 or len(self.features) == 0:
            return 0
        valueObj = self.calculateOnElements(hashCodeFunc, preserveZeros=True,
                                            outputType='Matrix', useLog=False)
        valueList = valueObj.copy(to="python list")
        avg = (sum(itertools.chain.from_iterable(valueList))
               / float(len(self.points) * len(self.features)))
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
        #first check to make sure they have the same dimensions
        if self._shape != other._shape:
            return False
        #now check if the hashes of each matrix are the same

        with self._treatAs2D():
            with other._treatAs2D():
                return self.hashCode() == other.hashCode()

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
        labels : Base object, identifier, list of identifiers, None
            A separate Base object containing the labels for this data
            or the name(s) or index(es) of the data labels within this object.
            A value of None implies this data does not contain labels. This
            parameter will affect the shape of the returned tuple.
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

        >>> nimble.random.setSeed(42)
        >>> raw = [[1, 0, 0],
        ...        [0, 1, 0],
        ...        [0, 0, 1],
        ...        [1, 0, 0],
        ...        [0, 1, 0],
        ...        [0, 0, 1]]
        >>> ptNames = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> data = nimble.data('Matrix', raw, pointNames=ptNames)
        >>> trainData, testData = data.trainAndTestSets(.34)
        >>> trainData
        Matrix(
            [[1 0 0]
             [0 1 0]
             [0 0 1]
             [0 0 1]]
            pointNames={'a':0, 'b':1, 'f':2, 'c':3}
            name="train"
            )
        >>> testData
        Matrix(
            [[0 1 0]
             [1 0 0]]
            pointNames={'e':0, 'd':1}
            name="test"
            )

        Returning a 4-tuple.

        >>> nimble.random.setSeed(42)
        >>> raw = [[1, 0, 0, 1],
        ...        [0, 1, 0, 2],
        ...        [0, 0, 1, 3],
        ...        [1, 0, 0, 1],
        ...        [0, 1, 0, 2],
        ...        [0, 0, 1, 3]]
        >>> ptNames = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> data = nimble.data('Matrix', raw, pointNames=ptNames)
        >>> fourTuple = data.trainAndTestSets(.34, labels=3)
        >>> trainX, trainY = fourTuple[0], fourTuple[1]
        >>> testX, testY = fourTuple[2], fourTuple[3]
        >>> trainX
        Matrix(
            [[1 0 0]
             [0 1 0]
             [0 0 1]
             [0 0 1]]
            pointNames={'a':0, 'b':1, 'f':2, 'c':3}
            name="trainX"
            )
        >>> trainY
        Matrix(
            [[1]
             [2]
             [3]
             [3]]
            pointNames={'a':0, 'b':1, 'f':2, 'c':3}
            name="trainY"
            )
        >>> testX
        Matrix(
            [[0 1 0]
             [1 0 0]]
            pointNames={'e':0, 'd':1}
            name="testX"
            )
        >>> testY
        Matrix(
            [[2]
             [1]]
            pointNames={'e':0, 'd':1}
            name="testY"
            )
        """
        order = list(range(len(self.points)))
        if randomOrder:
            nimble.random.numpyRandom.shuffle(order)

        if not 0 <= testFraction <= 1:
            msg = 'testFraction must be between 0 and 1 (inclusive)'
            raise InvalidArgumentValue(msg)
        testXSize = int(round(testFraction * len(self.points)))
        splitIndex = len(self.points) - testXSize

        #pull out a testing set
        trainX = self.points.copy(order[:splitIndex], useLog=False)
        testX = self.points.copy(order[splitIndex:], useLog=False)

        if labels is None:
            ret = trainX, testX
            if self.name is not None:
                trainX.name = self.name + " train"
                testX.name = self.name + " test"
            else:
                trainX.name = "train"
                testX.name = "test"
        elif len(self._shape) > 2:
            msg = "labels parameter must be None when the data has more "
            msg += "than two dimensions"
            raise ImproperObjectAction(msg)
        else:
            if isinstance(labels, Base):
                if len(labels.points) != len(self.points):
                    msg = 'labels must have the same number of points ({0}) '
                    msg += 'as the calling object ({1})'
                    msg = msg.format(len(labels.points), len(self.points))
                    raise InvalidArgumentValue(msg)
                try:
                    self._validateEqualNames('point', 'point', '', labels)
                except InvalidArgumentValue as e:
                    msg = 'labels and calling object pointNames must be equal'
                    raise InvalidArgumentValue(msg) from e
                trainY = labels.points.copy(order[:splitIndex], useLog=False)
                testY = labels.points.copy(order[splitIndex:], useLog=False)
            else:
                # safety for empty objects
                toExtract = labels
                if testXSize == 0:
                    toExtract = []

                trainY = trainX.features.extract(toExtract, useLog=False)
                testY = testX.features.extract(toExtract, useLog=False)

            if self.name is not None:
                trainX.name = self.name + " trainX"
                testX.name = self.name + " testX"
                trainY.name = self.name + " trainY"
                testY.name = self.name + " testY"
            else:
                trainX.name = "trainX"
                testX.name = "testX"
                trainY.name = "trainY"
                testY.name = "testY"

            ret = trainX, trainY, testX, testY

        handleLogging(useLog, 'prep', "trainAndTestSets", self.getTypeString(),
                      Base.trainAndTestSets, testFraction, labels, randomOrder)

        return ret

    ########################################
    ########################################
    ###   Functions related to logging   ###
    ########################################
    ########################################
    @limitedTo2D
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
        if not isinstance(other, Base):
            return False
        if self._shape != other._shape:
            return False
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
        if len(self.points) == 0 or len(self.features) == 0:
            msg = "We do not allow writing to file when an object has "
            msg += "0 points or features"
            raise ImproperObjectAction(msg)

        # if format is not specified, we fall back on the extension in outPath
        if fileFormat is None:
            split = outPath.rsplit('.', 1)
            fileFormat = None
            if len(split) > 1:
                fileFormat = split[1].lower()

        if fileFormat not in ['csv', 'mtx', 'hdf5', 'h5']:
            msg = "Unrecognized file format. Accepted types are 'csv', "
            msg += "'mtx', 'hdf5', and 'h5'. They may either be input as the "
            msg += "format parameter, or as the extension in the outPath"
            raise InvalidArgumentValue(msg)

        includePointNames = includeNames
        if includePointNames:
            seen = False
            if self.points._getNamesNoGeneration() is not None:
                for name in self.points.getNames():
                    if name is not None:
                        seen = True
            if not seen:
                includePointNames = False

        includeFeatureNames = includeNames
        if includeFeatureNames:
            seen = False
            if self.features._getNamesNoGeneration() is not None:
                for name in self.features.getNames():
                    if name is not None:
                        seen = True
            if not seen:
                includeFeatureNames = False


        if fileFormat.lower() in ['hdf5', 'h5']:
            self._writeFileHDF_implementation(outPath, includePointNames)
        elif len(self._shape) > 2:
            msg = 'Data with more than two dimensions can only be written '
            msg += 'to .hdf5 or .h5 formats otherwise the dimensionality '
            msg += 'would be lost'
            raise InvalidArgumentValue(msg)
        elif fileFormat.lower() == "csv":
            self._writeFileCSV_implementation(
                outPath, includePointNames, includeFeatureNames)
        elif fileFormat.lower() == "mtx":
            self._writeFileMTX_implementation(
                outPath, includePointNames, includeFeatureNames)

    def _writeFeatureNamesToCSV(self, openFile, includePointNames):
        fnames = list(map(csvCommaFormat, self.features.getNames()))
        if includePointNames:
            fnames.insert(0, 'pointNames')
        fnamesLine = ','.join(fnames)
        fnamesLine += '\n'
        openFile.write(fnamesLine)

    def _writeFileHDF_implementation(self, outPath, includePointNames):
        if not h5py.nimbleAccessible():
            msg = 'h5py must be installed to write to an hdf file'
            raise PackageException(msg)
        if includePointNames:
            pnames = self.points.getNames()
            userblockSize = 512
        else:
            pnames = [str(i) for i in range(len(self.points))]
            userblockSize = 0
        with h5py.File(outPath, 'w', userblock_size=userblockSize) as hdf:
            for name, point in zip(pnames, self.points):
                point._convertToNumericTypes()
                asArray = point.copy('numpy array')
                _ = hdf.create_dataset(name, data=asArray)
                hdf.flush()
        if includePointNames:
            with open(outPath, 'rb+') as f:
                f.write(b'includePointNames ')
                f.flush()

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
        if not cloudpickle.nimbleAccessible():
            msg = "To save nimble objects, cloudpickle must be installed"
            raise PackageException(msg)

        extension = '.nimd'
        if not outputPath.endswith(extension):
            outputPath = outputPath + extension

        with open(outputPath, 'wb') as file:
            cloudpickle.dump(self, file)
        # TODO: save session
        # print('session_' + outputFilename)
        # print(globals())
        # dill.dump_session('session_' + outputFilename)

    def getTypeString(self):
        """
        The nimble Type of this object.

        A string representing the non-abstract type of this
        object (e.g. Matrix, Sparse, etc.) that can be passed to
        nimble.data() function to create a new object of the same type.

        Returns
        -------
        str
        """
        return self._getTypeString_implementation()

    @limitedTo2D
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
        >>> office = nimble.data('Matrix', raw, pointNames=pointNames,
        ...                      featureNames=featureNames)

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
            if len(self.points) == 1:
                x = 0
                y = key
            elif len(self.features) == 1:
                x = key
                y = 0
            else:
                msg = "Must include both a point and feature index; or, "
                msg += "if this is vector shaped, a single index "
                msg += "into the axis whose length > 1"
                raise InvalidArgumentType(msg)

        #process x
        singleX = False
        if isinstance(x, (int, float, str, np.integer)):
            x = self.points._getIndex(x, allowFloats=True)
            singleX = True
        #process y
        singleY = False
        if isinstance(y, (int, float, str, np.integer)):
            y = self.features._getIndex(y, allowFloats=True)
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

    def pointView(self, identifier):
        """
        A read-only view of a single point.

        A BaseView object into the data of the point with the given
        identifier. See BaseView object comments for its capabilities.
        This view is only valid until the next modification to the shape
        or ordering of this object's internal data. After such a
        modification, there is no guarantee to the validity of the
        results.

        Returns
        -------
        BaseView
            The read-only object for this point.
        """
        if len(self.points) == 0:
            msg = "identifier is invalid, This object contains no points"
            raise ImproperObjectAction(msg)

        index = self.points.getIndex(identifier)
        ret = self._view_backend(index, index, None, None, True)
        return ret

    @limitedTo2D
    def featureView(self, identifier):
        """
        A read-only view of a single feature.

        A BaseView object into the data of the feature with the given
        identifier. See BaseView object comments for its capabilities.
        This view is only valid until the next modification to the shape
        or ordering of this object's internal data. After such a
        modification, there is no guarantee to the validity of the
        results.

        Returns
        -------
        BaseView
            The read-only object for this feature.
        """
        if len(self.features) == 0:
            msg = "identifier is invalid, This object contains no features"
            raise ImproperObjectAction(msg)

        index = self.features.getIndex(identifier)
        return self._view_backend(None, None, index, index)

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
        return self._view_backend(pointStart, pointEnd, featureStart,
                                  featureEnd)

    def _view_backend(self, pointStart, pointEnd, featureStart, featureEnd,
                      dropDimension=False):
        # transform defaults to mean take as much data as possible,
        # transform end values to be EXCLUSIVE
        if pointStart is None:
            pointStart = 0
        else:
            pointStart = self.points.getIndex(pointStart)

        if pointEnd is None:
            pointEnd = len(self.points)
        else:
            pointEnd = self.points.getIndex(pointEnd)
            # this is the only case that could be problematic and needs
            # checking
            validateRangeOrder("pointStart", pointStart, "pointEnd", pointEnd)
            # make exclusive now that it won't ruin the validation check
            pointEnd += 1

        if featureStart is None:
            featureStart = 0
        else:
            featureStart = self.features.getIndex(featureStart)

        if featureEnd is None:
            featureEnd = len(self.features)
        else:
            featureEnd = self.features.getIndex(featureEnd)
            # this is the only case that could be problematic and needs
            # checking
            validateRangeOrder("featureStart", featureStart,
                               "featureEnd", featureEnd)
            # make exclusive now that it won't ruin the validation check
            featureEnd += 1

        if len(self._shape) > 2:
            if featureStart != 0 or featureEnd != len(self.features):
                msg = "feature limited views are not allowed for data with "
                msg += "more than two dimensions."
                raise ImproperObjectAction(msg)

        return self._view_implementation(pointStart, pointEnd,
                                         featureStart, featureEnd,
                                         dropDimension)

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
            assert len(self.points) == len(self.points.getNames())
        if self.features._namesCreated():
            assert len(self.features) == len(self.features.getNames())

        if level > 0:
            if self.points._namesCreated():
                for i, key in enumerate(self.points.getNames()):
                    if key is not None:
                        assert self.points.getIndex(key) == i
                        assert self.points.getName(i) == key
                    else:
                        assert self.points.getName(i) is None
            if self.features._namesCreated():
                for i, key in enumerate(self.features.getNames()):
                    if key is not None:
                        assert self.features.getIndex(key) == i
                        assert self.features.getName(i) == key
                    else:
                        assert self.features.getName(i) is None

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
        if len(self.points) == 0 or len(self.features) == 0:
            return False
        with self._treatAs2D():
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
        if len(self.points) == 0 or len(self.features) == 0:
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
        maxHeight = len(self.points) + 2 if maxHeight is None else maxHeight
        maxWidth = float('inf') if maxWidth is None else maxWidth
        maxRows = min(maxHeight, len(self.points))
        maxDataRows = maxRows
        includePNames = False
        includeFNames = False

        if includeNames:
            includePNames = not self.points._allDefaultNames()
            includeFNames = not self.features._allDefaultNames()

            if includeFNames:
                # plus or minus 2 because we will be dealing with both
                # feature names and a gap row
                maxRows = min(maxHeight, len(self.points) + 2)
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
        with self._treatAs2D():
            dataTable, colWidths, fnames = self._arrangeDataWithLimits(
                maxDataWidth, maxDataRows, includeFNames, sigDigits,
                maxColumnWidth, colSep, colHold, rowHold, nameHolder)

        # combine names into finalized table
        finalTable, finalWidths = arrangeFinalTable(
            pnames, pnamesWidth, dataTable, colWidths, fnames, pnameSep)

        # set up output string
        out = ""
        for i, row in enumerate(finalTable):
            for j, val in enumerate(row):
                if j == 0 and includePNames:
                    padded = getattr(val, pNameOrientation)(finalWidths[j])
                elif i == 0 and includeFNames:
                    padded = getattr(val, fNameOrientation)(finalWidths[j])
                else:
                    padded = getattr(val, dataOrientation)(finalWidths[j])
                row[j] = padded
            # for __repr__ output want to retain whitespace
            if keepTrailingWhitespace:
                line = colSep.join(finalTable[i]) + "\n"
            else:
                line = colSep.join(finalTable[i]).rstrip() + "\n"
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
        # convert self._data into a string with nice format
        newLines = (']\n' + indent + ' [').join(byLine)
        ret += (indent + '[[%s]]\n') % newLines

        numRows = min(len(self.points), maxH)
        # if non default point names, print all (truncated) point names
        ret += makeNamesLines(indent, maxW, numRows, len(self.points),
                              self.points._getNamesNoGeneration(),
                              'pointNames')
        # if non default feature names, print all (truncated) feature names
        numCols = 0
        if byLine:
            splited = byLine[0].split(' ')
            for val in splited:
                if val not in ['', '...', '--']:
                    numCols += 1
        elif len(self.features) > 0:
            # if the container is empty, then roughly compute length of
            # the string of feature names, and then calculate numCols
            ftNames = self.features._getNamesNoGeneration()
            if ftNames is None:
                # mock up default looking names to avoid name generation
                ftNames = [None] * len(self.features)
            strLength = (len("___".join(ftNames))
                         + len(''.join([str(i) for i
                                        in range(len(self.features))])))
            numCols = int(min(1, maxW / float(strLength)) * len(self.features))
        # because of how _dataHelpers.indicesSplit works, we need this to be
        # +1 in some cases this means one extra feature name is displayed. But
        # that's acceptable
        if numCols <= len(self.features):
            numCols += 1
        elif numCols > len(self.features):
            numCols = len(self.features)
        ret += makeNamesLines(indent, maxW, numCols, len(self.features),
                              self.features._getNamesNoGeneration(),
                              'featureNames')

        # if name not None, print
        if self.name is not None:
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

        if includeObjectName and self.name is not None:
            context = self.name + " : "
        else:
            context = ""
        if len(self._shape) > 2:
            context += " x ".join(map(str, self._shape))
        else:
            context += str(len(self.points)) + "pt x "
            context += str(len(self.features)) + "ft"
        print(context, '\n')
        print(self.toString(includeAxisNames, maxWidth, maxHeight, sigDigits,
                            maxColumnWidth))

    @limitedTo2D
    def plotHeatMap(self, includeColorbar=False, outPath=None, show=True,
                    title=True, xAxisLabel=True, yAxisLabel=True, **kwargs):
        """
        Display a plot of the data.

        Parameters
        ----------
        includeColorbar : bool
            Add a colorbar to the plot.
        outPath : str, None
            A string of the path to save the current figure.
        show : bool
            If True, display the plot. If False, the figure will not
            display until a plotting function with show=True is called.
        title : str, bool
            The title of the plot. If True and this object has a name,
            the title will be the object name.
        xAxisLabel : str, bool
            A label for the x axis. If True, the label will be "Feature
            Values".
        yAxisLabel : str, bool
            A label for the x axis. If True, the label will be "Point
            Values".
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``matshow`` function.
        """
        self._plot(includeColorbar, outPath, show, title, xAxisLabel,
                   yAxisLabel, **kwargs)

    @pyplotRequired
    def _plot(self, includeColorbar, outPath, show, title, xAxisLabel,
              yAxisLabel, **kwargs):
        self._convertToNumericTypes(allowBool=False)

        if 'cmap' not in kwargs:
            kwargs['cmap'] = "gray"

        # matshow generates a new figure b/c existing axes are an issue.
        plt.matshow(self.copy('numpyarray'), **kwargs)

        if includeColorbar:
            plt.colorbar()

        if title is True and self.name is not None:
            title = self.name
        elif title is True:
            title = None
        elif title is False:
            title = None
        plt.title(title)

        if xAxisLabel is True:
            xAxisLabel = "Feature Values"
        if yAxisLabel is True:
            yAxisLabel = "Point Values"
        plt.xlabel(xAxisLabel, labelpad=10)
        plt.ylabel(yAxisLabel)

        plotOutput(outPath, show)

    @limitedTo2D
    def plotFeatureDistribution(self, feature, outPath=None, show=True,
                                figureName=None, title=True, xAxisLabel=True,
                                yAxisLabel=True, xMin=None, xMax=None,
                                **kwargs):
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
            A string of the path to save the current figure.
        show : bool
            If True, display the plot. If False, the figure will not
            display until a plotting function with show=True is called.
            This allows for future plots to placed on the figure with
            the same ``figureName`` before being shown.
        figureName : str, None
            A new figure will be generated when None or a new name,
            otherwise the figure with that name will be activated to
            draw the plot on an existing figure.
        title : str, None
            The title of the plot. If True, the title will identify the
            feature presented in the distribution.
        xAxisLabel : str
            A label for the x axis. If True, the label will be "Values".
        yAxisLabel : str
            A label for the x axis. If True, the label will be "Number
            of Values".
        xMin : int, float
            The minimum value shown on the x axis of the resultant plot.
        xMax : int, float
            The maximum value shown on the x axis of the resultant plot.
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``hist`` function.
        """
        self._plotFeatureDistribution(feature, outPath, show, figureName,
                                      title, xAxisLabel, yAxisLabel, xMin,
                                      xMax, **kwargs)

    def _plotFeatureDistribution(self, feature, outPath, show, figureName,
                                 title, xAxisLabel, yAxisLabel, xMin, xMax,
                                 **kwargs):
        return self._plotDistribution('feature', feature, outPath, show,
                                      figureName, title, xAxisLabel,
                                      yAxisLabel, xMin, xMax, **kwargs)

    @pyplotRequired
    def _plotDistribution(self, axis, identifier, outPath, show, figureName,
                          title, xAxisLabel, yAxisLabel, xMin, xMax, **kwargs):
        _, ax = plotFigureHandling(figureName)
        plotUpdateAxisLimits(ax, xMin, xMax, None, None)

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

        if title is True:
            title = "Distribution of " + axis + " "
            if not name:
                title += '#' + str(index)
            else:
                title += "named: " + name
        elif title is False:
            title = None
        ax.set_title(title)
        toPlot = getter(index)

        if 'bins' not in kwargs:
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
            kwargs['bins'] = binCount

        ax.hist(toPlot, **kwargs)
        if 'label' in kwargs:
            ax.legend()

        plotAxisLimits(ax)
        plotAxisLabels(ax, xAxisLabel, "Values", yAxisLabel,
                       "Number of Values")

        plotOutput(outPath, show)

    @limitedTo2D
    def plotFeatureAgainstFeatureRollingAverage(
            self, x, y, sampleSizeForAverage=20, trend=None, outPath=None,
            show=True, figureName=None, title=True, xAxisLabel=True,
            yAxisLabel=True, xMin=None, xMax=None, yMin=None, yMax=None,
            **kwargs):
        """
        A rolling average of the pairwise combination of feature values.

        The points in the scatter plot have coordinates defined by
        taking the average of pairwise values from ``x`` and ``y`` within
        a rolling window of length ``sampleSizeForAverage`` according
        to an ordering defined by the sorted values of ``x``.

        Control over the visual width of the both axes is given, with the
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
        sampleSizeForAverage : int
            The number of samples to use for the calculation of the
            rolling average.
        trend : str, None
            Specify a trendline type. Currently "linear" is the only
            supported string option. None will not add a trendline.
        outPath : str, None
            A string of the path to save the current figure.
        show : bool
            If True, display the plot. If False, the figure will not
            display until a plotting function with show=True is called.
            This allows for future plots to placed on the figure with
            the same ``figureName`` before being shown.
        figureName : str, None
            A new figure will be generated when None or a new name,
            otherwise the figure with that name will be activated to
            draw the plot on an existing figure.
        title : str, None
            The title of the plot. If True, the title will identify the
            two features presented in the plot.
        xAxisLabel : str
            A label for the x axis. If True, the label will be "Values".
        yAxisLabel : str
            A label for the x axis. If True, the label will be "Number
            of Values".
        xMin : int, float
            The minimum value shown on the x axis of the resultant plot.
        xMax : int, float
            The maximum value shown on the x axis of the resultant plot.
        yMin : int, float
            The minimum value shown on the y axis of the resultant plot.
        yMax : int, float
            The maximum value shown on the y axis of the resultant plot.
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``plot`` function.
        """
        self._plotFeatureAgainstFeature(
            x, y, sampleSizeForAverage, trend, outPath, show, figureName,
            title, xAxisLabel, yAxisLabel, xMin, xMax, yMin, yMax, **kwargs)

    @limitedTo2D
    def plotFeatureAgainstFeature(
            self, x, y, trend=None, outPath=None, show=True, figureName=None,
            title=True, xAxisLabel=True, yAxisLabel=True, xMin=None, xMax=None,
            yMin=None, yMax=None, **kwargs):
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
        trend : str, None
            Specify a trendline type. Currently "linear" is the only
            supported string option. None will not add a trendline.
        outPath : str, None
            A string of the path to save the current figure.
        show : bool
            If True, display the plot. If False, the figure will not
            display until a plotting function with show=True is called.
            This allows for future plots to placed on the figure with
            the same ``figureName`` before being shown.
        figureName : str, None
            A new figure will be generated when None or a new name,
            otherwise the figure with that name will be activated to
            draw the plot on an existing figure.
        title : str, None
            The title of the plot. If True, the title will identify the
            two features presented in the plot.
        xAxisLabel : str
            A label for the x axis. If True, the label will be the x
            feature name or index.
        yAxisLabel : str
            A label for the x axis. If True, the label will be the y
            feature name or index.
        xMin : int, float
            The minimum value shown on the x axis of the resultant plot.
        xMax : int, float
            The maximum value shown on the x axis of the resultant plot.
        yMin : int, float
            The minimum value shown on the y axis of the resultant plot.
        yMax : int, float
            The maximum value shown on the y axis of the resultant plot.
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``plot`` function.
        """
        self._plotFeatureAgainstFeature(
            x, y, None, trend, outPath, show, figureName, title, xAxisLabel,
            yAxisLabel, xMin, xMax, yMin, yMax, **kwargs)

    def _plotFeatureAgainstFeature(
            self, x, y, sampleSizeForAverage, trend, outPath, show, figureName,
            title, xAxisLabel, yAxisLabel, xMin, xMax, yMin, yMax, **kwargs):
        return self._plotCross(
            x, 'feature', y, 'feature', sampleSizeForAverage, trend, outPath,
            show, figureName, title, xAxisLabel, yAxisLabel, xMin, xMax, yMin,
            yMax, **kwargs)

    def _formattedStringID(self, axis, identifier):
        if axis == 'point':
            namesAxis = self.points
        else:
            namesAxis = self.features
        if not isinstance(identifier, str):
            names = namesAxis._getNamesNoGeneration()
            if names is None or names[identifier] is None:
                identifier = axis.capitalize() + ' #' + str(identifier)
            else:
                identifier = names[identifier]

        return identifier

    @pyplotRequired
    def _plotCross(self, x, xAxis, y, yAxis, sampleSizeForAverage, trend,
                   outPath, show, figureName, title, xAxisLabel, yAxisLabel,
                   xMin, xMax, yMin, yMax, **kwargs):
        _, ax = plotFigureHandling(figureName)
        plotUpdateAxisLimits(ax, xMin, xMax, yMin, yMax)

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

        xToPlot = customGetter(xIndex, xAxis)
        yToPlot = customGetter(yIndex, yAxis)

        xName = xlabel = self._formattedStringID(xAxis, xIndex)
        yName = ylabel = self._formattedStringID(yAxis, yIndex)

        if sampleSizeForAverage is not None:
            #do rolling average
            xToPlot, yToPlot = list(zip(*sorted(zip(xToPlot, yToPlot),
                                                key=lambda x: x[0])))
            convShape = (np.ones(sampleSizeForAverage)
                         / float(sampleSizeForAverage))
            startIdx = sampleSizeForAverage-1
            xToPlot = np.convolve(xToPlot, convShape)[startIdx:-startIdx]
            yToPlot = np.convolve(yToPlot, convShape)[startIdx:-startIdx]

            tmpStr = ' (%s sample average)' % sampleSizeForAverage
            xlabel += tmpStr
            ylabel += tmpStr
            xName += ' average'
            yName += ' average'

        if 'marker' not in kwargs:
            kwargs['marker'] = '.'
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = ''

        ax.plot(xToPlot, yToPlot, **kwargs)
        if 'label' in kwargs:
            ax.legend()

        plotAxisLimits(ax)

        if trend is not None and trend.lower() == 'linear':
            meanX = np.mean(xToPlot)
            meanY = np.mean(yToPlot)
            errorX = meanX - xToPlot
            sumSquareErrorX = sum((errorX) ** 2)
            sumErrorXY = sum(errorX * (meanY - yToPlot))
            slope = sumErrorXY / sumSquareErrorX
            intercept = meanY - slope * meanX
            xVals = ax.get_xlim()
            yVals = list(map(lambda x: slope * x + intercept, xVals))
            ax.plot(xVals, yVals, scalex=False, scaley=False)

        elif trend is not None:
            msg = 'invalid trend value. "linear" is the only value supported '
            msg += 'at this time'
            raise InvalidArgumentValue(msg)

        if title is True and self.name is None:
            title = ('%s vs. %s') % (xName, yName)
        elif title is True:
            title = ('%s: %s vs. %s') % (self.name, xName, yName)
        elif title is False:
            title = None
        ax.set_title(title)

        plotAxisLabels(ax, xAxisLabel, xlabel, yAxisLabel, ylabel)

        plotOutput(outPath, show)

    @limitedTo2D
    def plotFeatureGroupMeans(
            self, feature, groupFeature, horizontal=False, outPath=None,
            show=True, figureName=None, title=True, xAxisLabel=True,
            yAxisLabel=True, **kwargs):
        """
        Plot the means of a feature grouped by another feature.

        The plot will include 95% confidence interval bars. The 95%
        confidence interval for each feature is calculated using the
        critical value from the two-sided Student's t-distribution.

        Parameters
        ----------
        feature : identifier
            The feature name or index that will be grouped.
        groupFeature : identifier
            The feature name or index that defines the groups in
            ``feature``
        horizontal : bool
            False, the default, draws plot bars vertically. True will
            draw bars horizontally.
        outPath : str, None
            A string of the path to save the current figure.
        show : bool
            If True, display the plot. If False, the figure will not
            display until a plotting function with show=True is called.
            This allows for future plots to placed on the figure with
            the same ``figureName`` before being shown.
        figureName : str, None
            A new figure will be generated when None or a new name,
            otherwise the figure with that name will be activated to
            draw the plot on an existing figure.
        title : str, bool
            The title of the plot. If True, a title will automatically
            be generated.
        xAxisLabel : str, bool
            A label for the x axis. If True, a label will automatically
            be generated.
        yAxisLabel : str, bool
            A label for the x axis. If True, a label will automatically
            be generated.
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``errorbar`` function.
        """
        self._plotFeatureGroupStatistics(
            nimble.calculate.mean, feature, groupFeature, None, True,
            horizontal, outPath, show, figureName, title, xAxisLabel,
            yAxisLabel, **kwargs)

    @limitedTo2D
    def plotFeatureGroupStatistics(
            self, statistic, feature, groupFeature, subgroupFeature=None,
            horizontal=False, outPath=None, show=True, figureName=None,
            title=True, xAxisLabel=True, yAxisLabel=True, **kwargs):
        """
        Plot an aggregate statistic for each group of a feature.

        A bar chart where each bar in the plot represent the output of
        the statistic function applied to each group (or subgroup, when
        applicable).

        Parameters
        ----------
        statistic : function
            Functions must take a feature view as an argument and return
            a single numeric value. Common statistic functions can be
            found in nimble.calculate.
        feature : identifier
            The feature name or index that will be grouped.
        groupFeature : identifier
            The feature name or index that defines the groups in
            ``feature``
        subgroupFeature : identifier, None
            An optional subgrouping feature. When not None, the bar for
            each group defined by ``groupFeature`` will be subdivided
            based on this feature with a unique colored bar for each
            subgroup. This may be the same as ``feature`` if its values
            define the subgroups.
        horizontal : bool
            False, the default, draws plot bars vertically. True will
            draw bars horizontally.
        outPath : str, None
            A string of the path to save the current figure.
        show : bool
            If True, display the plot. If False, the figure will not
            display until a plotting function with show=True is called.
            This allows for future plots to placed on the figure with
            the same ``figureName`` before being shown.
        figureName : str, None
            A new figure will be generated when None or a new name,
            otherwise the figure with that name will be activated to
            draw the plot on an existing figure.
        title : str, bool
            The title of the plot. If True, a title will automatically
            be generated.
        xAxisLabel : str, bool
            A label for the x axis. If True, a label will automatically
            be generated.
        yAxisLabel : str, bool
            A label for the x axis. If True, a label will automatically
            be generated.
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``bar`` function.
        """
        self._plotFeatureGroupStatistics(
            statistic, feature, groupFeature, subgroupFeature, False,
            horizontal, outPath, show, figureName, title, xAxisLabel,
            yAxisLabel, **kwargs)

    @pyplotRequired
    def _plotFeatureGroupStatistics(
            self, statistic, feature, groupFeature, subgroupFeature,
            confidenceIntervals, horizontal, outPath, show, figureName, title,
            xAxisLabel, yAxisLabel, **kwargs):
        fig, ax = plotFigureHandling(figureName)
        featureName = self._formattedStringID('feature', feature)
        if hasattr(statistic, '__name__') and statistic.__name__ != '<lambda>':
            statName = statistic.__name__
        else:
            statName = ''

        toGroup = self.features[[groupFeature, feature]]
        if subgroupFeature:
            subgroupFt = self.features[subgroupFeature]
            # remove name in case is same as feature
            subgroupFt.features.setName(0, None)
            toGroup.features.insert(1, subgroupFt)
        grouped = toGroup.groupByFeature(0, useLog=False)

        axisRange = range(1, len(grouped) + 1)
        names = []
        if confidenceIntervals:
            means = []
            errors = []
            for name, ft in grouped.items():
                names.append(name)
                mean, error = plotConfidenceIntervalMeanAndError(ft)
                means.append(mean)
                errors.append(error)

            plotErrorBars(ax, axisRange, means, errors, horizontal, **kwargs)

            if title is True:
                title = "95% Confidence Intervals for Mean of " + featureName

        elif subgroupFeature:
            heights = {}
            for i, (name, group) in enumerate(grouped.items()):
                names.append(str(name))
                subgrouped = group.groupByFeature(0, useLog=False)
                for subname, subgroup in subgrouped.items():
                    if subname not in heights:
                        heights[subname] = [0] * len(grouped)
                    heights[subname][i] = statistic(subgroup)
            subgroup = self._formattedStringID('feature', subgroupFeature)
            if title is True:
                title = "{feat} {stat} by {group}"
                group = self._formattedStringID('feature', groupFeature)
                title = title.format(feat=featureName, stat=statName,
                                     group=group)

            plotMultiBarChart(ax, heights, horizontal, subgroup, **kwargs)

        else:
            heights = []
            for name, group in grouped.items():
                names.append(name)
                heights.append(statistic(group))
            plotSingleBarChart(ax, axisRange, heights, horizontal, **kwargs)

            if title is True:
                title = "{feat} {stat} by {group}"
                group = self._formattedStringID('feature', groupFeature)
                title = title.format(feat=featureName, stat=statName,
                                     group=group)

        if title is False:
            title = None
        ax.set_title(title)
        if horizontal:
            ax.set_yticks(axisRange)
            ax.set_yticklabels(names)
            yAxisDefault = self._formattedStringID('feature', groupFeature)
            xAxisDefault = statName
        else:
            ax.set_xticks(axisRange)
            plotXTickLabels(ax, fig, names, len(grouped))
            xAxisDefault = self._formattedStringID('feature', groupFeature)
            yAxisDefault = statName

        plotAxisLabels(ax, xAxisLabel, xAxisDefault, yAxisLabel, yAxisDefault)

        plotOutput(outPath, show)


    ##################################################################
    ##################################################################
    ###   Subclass implemented structural manipulation functions   ###
    ##################################################################
    ##################################################################
    @limitedTo2D
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
        >>> data = nimble.data('List', raw)
        >>> data
        List(
            [[1 2 3]
             [4 5 6]]
            )
        >>> data.transpose()
        >>> data
        List(
            [[1 4]
             [2 5]
             [3 6]]
            )
        """
        self._transpose_implementation()

        self._shape = [len(self.features), len(self.points)]
        ptNames, ftNames = (self.features._getNamesNoGeneration(),
                            self.points._getNamesNoGeneration())
        self.points.setNames(ptNames, useLog=False)
        self.features.setNames(ftNames, useLog=False)

        handleLogging(useLog, 'prep', "transpose", self.getTypeString(),
                      Base.transpose)

    @property
    @limitedTo2D
    def T(self): # pylint: disable=invalid-name
        """
        Invert the feature and point indices of the data.

        Return this object with inverted feature and point indices,
        including inverting point and feature names, if available.

        Examples
        --------
        >>> raw = [[1, 2, 3], [4, 5, 6]]
        >>> data = nimble.data('List', raw)
        >>> data
        List(
            [[1 2 3]
             [4 5 6]]
            )
        >>> data.T
        List(
            [[1 4]
             [2 5]
             [3 6]]
            )
        """
        ret = self.copy()
        ret.transpose(useLog=False)
        return ret

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
        >>> data = nimble.data('List', raw, pointNames=ptNames,
        ...                    name="odd&even")
        >>> data
        List(
            [[1 3 5]
             [2 4 6]]
            pointNames={'odd':0, 'even':1}
            name="odd&even"
            )
        >>> dataCopy = data.copy()
        >>> dataCopy
        List(
            [[1 3 5]
             [2 4 6]]
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
        origTo = to
        if not isinstance(to, str):
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

        if len(self._shape) > 2:
            if to in ['listofdict', 'dictoflist', 'scipycsr', 'scipycsc']:
                msg = 'Objects with more than two dimensions cannot be '
                msg += 'copied to {0}'.format(origTo)
                raise ImproperObjectAction(msg)
            if outputAs1D or not rowsArePoints:
                if outputAs1D:
                    param = 'outputAs1D'
                    value = False
                elif not rowsArePoints:
                    param = 'rowsArePoints'
                    value = True
                msg = '{0} must be {1} when the data '.format(param, value)
                msg += 'has more than two dimensions'
                raise ImproperObjectAction(msg)
        # only 'numpyarray' and 'pythonlist' are allowed to use outputAs1D flag
        if outputAs1D:
            if to not in ('numpyarray', 'pythonlist'):
                msg = "Only 'numpy array' or 'python list' can output 1D"
                raise InvalidArgumentValueCombination(msg)
            if len(self.points) != 1 and len(self.features) != 1:
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
            ret._shape = self._shape.copy()
            if not rowsArePoints:
                ret.transpose(useLog=False)
            ret._name = self.name
            ret._relPath = self.relativePath
            ret._absPath = self.absolutePath
        elif not rowsArePoints:
            ret = ret.transpose()

        return ret

    def _copy_outputAs1D(self, to):
        if to == 'numpyarray':
            if len(self.points) == 0 or len(self.features) == 0:
                return np.array([])
            return self._copy_implementation('numpyarray').flatten()

        if len(self.points) == 0 or len(self.features) == 0:
            return []
        list2d = self._copy_implementation('pythonlist')
        return list(itertools.chain.from_iterable(list2d))

    def _copy_pythonList(self, rowsArePoints):
        ret = self._copy_implementation('pythonlist')
        if len(self._shape) > 2:
            ret = np.reshape(ret, self._shape).tolist()
        if not rowsArePoints:
            ret = np.transpose(ret).tolist()
        return ret

    def _copy_nestedPythonTypes(self, to, rowsArePoints):
        data = self._copy_implementation('numpyarray')
        if rowsArePoints:
            featureNames = self.features.getNames()
            featureCount = len(self.features)
        else:
            data = data.transpose()
            featureNames = self.points.getNames()
            featureCount = len(self.points)
        if to == 'listofdict':
            return createListOfDict(data, featureNames)
        return createDictOfList(data, featureNames, featureCount)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    @limitedTo2D
    def replaceRectangle(self, replaceWith, pointStart, featureStart, pointEnd,
                         featureEnd, useLog=None):
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
        Points.fill, Features.fill

        Examples
        --------
        An object of ones filled with zeros from (0, 0) to (2, 2).

        >>> data = nimble.ones('Matrix', 5, 5)
        >>> filler = nimble.zeros('Matrix', 3, 3)
        >>> data.replaceRectangle(filler, 0, 0, 2, 2)
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

        if isinstance(replaceWith, Base):
            prange = (peIndex - psIndex) + 1
            frange = (feIndex - fsIndex) + 1
            raiseException = False
            if len(replaceWith.points) != prange:
                raiseException = True
                axis = 'point'
                axisLen = len(replaceWith.points)
                start = pointStart
                end = pointEnd
                rangeLen = prange
            elif len(replaceWith.features) != frange:
                raiseException = True
                axis = 'feature'
                axisLen = len(replaceWith.features)
                start = featureStart
                end = featureEnd
                rangeLen = frange
            if raiseException:
                msg = "When the replaceWith argument is a nimble Base object, "
                msg += "the size of replaceWith must match the range of "
                msg += "modification. There are {axisLen} {axis}s in "
                msg += "replaceWith, yet {axis}Start ({start}) and {axis}End "
                msg += "({end}) define a range of length {rangeLen}"
                msg = msg.format(axis=axis, axisLen=axisLen, start=start,
                                 end=end, rangeLen=rangeLen)
                raise InvalidArgumentValueCombination(msg)
            if replaceWith.getTypeString() != self.getTypeString():
                replaceWith = replaceWith.copy(to=self.getTypeString())

        elif not (looksNumeric(replaceWith) or isinstance(replaceWith, str)):
            msg = "replaceWith may only be a nimble Base object, or a single "
            msg += "numeric value, yet we received something of "
            msg += str(type(replaceWith))
            raise InvalidArgumentType(msg)

        self._replaceRectangle_implementation(replaceWith, psIndex, fsIndex,
                                              peIndex, feIndex)

        handleLogging(useLog, 'prep', "replaceRectangle",
                      self.getTypeString(), Base.replaceRectangle, replaceWith,
                      pointStart, featureStart, pointEnd, featureEnd)


    def _flattenNames(self, order):
        """
        Helper calculating the axis names for the unflattend axis after
        a flatten operation.
        """
        pNames = self.points.getNames()
        fNames = self.features.getNames()

        ret = []
        pointNumber = '_PT#{}'
        featureNumber = '_FT#{}'
        if order == 'point':
            for (i, p), (j, f) in itertools.product(enumerate(pNames),
                                                    enumerate(fNames)):
                if p is None:
                    p = pointNumber.format(i)
                if f is None:
                    f = featureNumber.format(j)
                ret.append(' | '.join([p, f]))
        else:
            for (j, f), (i, p) in itertools.product(enumerate(fNames),
                                                    enumerate(pNames)):
                if f is None:
                    f = featureNumber.format(j)
                if p is None:
                    p = pointNumber.format(i)
                ret.append(' | '.join([p, f]))
        return ret

    def flatten(self, order='point', useLog=None):
        """
        Modify this object so that its values are in a single point.

        Each value in the result maps to exactly one value from the
        original object. For data in two-dimensions, ``order`` may be
        'point' or 'feature'. If order='point', the first n values in
        the result will match the original first point, the nth to
        (2n-1)th values will match the original second point and so on.
        If order='feature', the first n values in the result will match
        the original first feature, the nth to (2n-1)th values will
        match the original second feature and so on. For higher
        dimension data, 'point' is the only accepted ``order``. If
        pointNames and/or featureNames are present. The feature names of
        the flattened result will be formatted as "ptName | ftName".
        This is an inplace operation.

        Parameters
        ----------
        order : str
            Either 'point' or 'feature'.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        unflatten

        Examples
        --------
        >>> raw = [[1, 2],
        ...        [3, 4]]
        >>> ptNames = ['1', '3']
        >>> ftNames = ['a', 'b']
        >>> data = nimble.data('Matrix', raw, pointNames=ptNames,
        ...                    featureNames=ftNames)
        >>> data.flatten()
        >>> data
        Matrix(
            [[1 2 3 4]]
            pointNames={'Flattened':0}
            featureNames={'1 | a':0, '1 | b':1, '3 | a':2, '3 | b':3}
            )

        >>> raw = [[1, 2],
        ...        [3, 4]]
        >>> ptNames = ['1', '3']
        >>> ftNames = ['a', 'b']
        >>> data = nimble.data('Matrix', raw, pointNames=ptNames,
        ...                    featureNames=ftNames)
        >>> data.flatten(order='feature')
        >>> data
        Matrix(
            [[1 3 2 4]]
            pointNames={'Flattened':0}
            featureNames={'1 | a':0, '3 | a':1, '1 | b':2, '3 | b':3}
            )
        """
        if order not in ['point', 'feature']:
            msg = "order must be the string 'point' or 'feature'"
            if not isinstance(order, str):
                raise InvalidArgumentType(msg)
            raise InvalidArgumentValue(msg)
        if len(self.points) == 0:
            msg = "Can only flatten when there is one or more "
            msg += "points. This object has 0 points."
            raise ImproperObjectAction(msg)
        if len(self.features) == 0:
            msg = "Can only flatten when there is one or more "
            msg += "features. This object has 0 features."
            raise ImproperObjectAction(msg)
        if order == 'feature' and len(self._shape) > 2:
            msg = "order='feature' is not allowed for flattening objects with "
            msg += 'more than two dimensions'
            raise ImproperObjectAction(msg)

        fNames = None
        if self.points._namesCreated() or self.features._namesCreated():
            fNames = self._flattenNames(order)
        self._shape = list(self.shape) # make 2D before flattening
        self._flatten_implementation(order)
        self._shape = [1, len(self.points) * len(self.features)]

        self.features.setNames(fNames, useLog=False)
        self.points.setNames(['Flattened'], useLog=False)

        handleLogging(useLog, 'prep', "flatten", self.getTypeString(),
                      Base.flatten, order)


    def _unflattenNames(self):
        """
        Helper calculating the new axis names after an unflattening
        operation.
        """
        possibleNames = None
        if len(self.points) > 1 and self.points._namesCreated():
            possibleNames = self.points.getNames()
        elif len(self.features) > 1 and self.features._namesCreated():
            possibleNames = self.features.getNames()
        if not possibleNames:
            return (None, None)
        splitNames = []
        for name in possibleNames:
            if name is None:
                return (None, None)
            splitName = name.split(' | ')
            if len(splitName) == 2:
                splitNames.append(splitName)
            else:
                return (None, None)

        pNames = []
        fNames = []
        allPtDefault = True
        allFtDefault = True
        for pName, fName in splitNames:
            if pName not in pNames:
                if pName.startswith('_PT#'):
                    pNames.append(None)
                else:
                    pNames.append(pName)
                    if allPtDefault:
                        allPtDefault = False
            if fName not in fNames:
                if fName.startswith('_FT#'):
                    fNames.append(None)
                else:
                    fNames.append(fName)
                    if allFtDefault:
                        allFtDefault = False

        if allPtDefault:
            pNames = None
        if allFtDefault:
            fNames = None

        return pNames, fNames


    @limitedTo2D
    def unflatten(self, dataDimensions, order='point', useLog=None):
        """
        Adjust a single point or feature to contain multiple points.

        The flat object is reshaped to match the dimensions of
        ``dataDimensions``. ``order`` determines whether point or
        feature vectors are created from the data. Provided
        ``dataDimensions`` of (m, n), the first n values become the
        first point when ``order='point'`` or the first m values become
        the first feature when ``order='feature'``. For higher dimension
        data, 'point' is the only accepted ``order``. If pointNames or
        featureNames match the format established in ``flatten``,
        "ptName | ftName", and they align eith the ``dataDimensions``,
        point and feature names will be unflattened as well, otherwise
        the result will not have pointNames or featureNames.
        This is an inplace operation.

        Parameters
        ----------
        dataDimensions : tuple, list
            The dimensions of the unflattend object.
        order : str
            Either 'point' or 'feature'.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        flatten

        Examples
        --------

        Unflatten a point in point order with default names.

        >>> raw = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        >>> data = nimble.data('Matrix', raw)
        >>> data.unflatten((3, 3))
        >>> data
        Matrix(
            [[1 2 3]
             [4 5 6]
             [7 8 9]]
            )

        Unflatten a point in feature order with default names.

        >>> raw = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        >>> data = nimble.data('Matrix', raw)
        >>> data.unflatten((3, 3), order='feature')
        >>> data
        Matrix(
            [[1 4 7]
             [2 5 8]
             [3 6 9]]
            )

        Unflatten a feature in feature order with default names.

        >>> raw = [[1], [4], [7], [2], [5], [8], [3], [6], [9]]
        >>> data = nimble.data('Matrix', raw)
        >>> data.unflatten((3, 3), order='feature')
        >>> data
        Matrix(
            [[1 2 3]
             [4 5 6]
             [7 8 9]]
            )

        Unflatten a feature in point order with default names.

        >>> raw = [[1], [4], [7], [2], [5], [8], [3], [6], [9]]
        >>> data = nimble.data('Matrix', raw)
        >>> data.unflatten((3, 3), order='point')
        >>> data
        Matrix(
            [[1 4 7]
             [2 5 8]
             [3 6 9]]
            )

        Unflatten a point with names that can be unflattened.

        >>> raw = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        >>> ftNames = ['1 | a', '1 | b', '1 | c',
        ...            '4 | a', '4 | b', '4 | c',
        ...            '7 | a', '7 | b', '7 | c']
        >>> data = nimble.data('Matrix', raw, featureNames=ftNames)
        >>> data.unflatten((3, 3))
        >>> data
        Matrix(
            [[1 2 3]
             [4 5 6]
             [7 8 9]]
            pointNames={'1':0, '4':1, '7':2}
            featureNames={'a':0, 'b':1, 'c':2}
            )
        """
        if order not in ['point', 'feature']:
            msg = "order must be the string 'point' or 'feature'"
            if not isinstance(order, str):
                raise InvalidArgumentType(msg)
            raise InvalidArgumentValue(msg)
        if len(self.features) == 0 or len(self.points) == 0:
            msg = "Cannot unflatten when there are 0 points or features."
            raise ImproperObjectAction(msg)
        if len(self.points) != 1 and len(self.features) != 1:
            msg = "Can only unflatten when there is only one point or feature."
            raise ImproperObjectAction(msg)
        if not isinstance(dataDimensions, (list, tuple)):
            raise InvalidArgumentType('dataDimensions must be a list or tuple')
        if len(dataDimensions) < 2:
            msg = "dataDimensions must contain a minimum of 2 values"
            raise InvalidArgumentValue(msg)
        if self.shape[0] * self.shape[1] != np.prod(dataDimensions):
            msg = "The product of the dimensions must be equal to the number "
            msg += "of values in this object"
            raise InvalidArgumentValue(msg)

        if len(dataDimensions) > 2:
            if order == 'feature':
                msg = "order='feature' is not allowed when unflattening to "
                msg += 'more than two dimensions'
                raise ImproperObjectAction(msg)
            shape2D = (dataDimensions[0], np.prod(dataDimensions[1:]))
        else:
            shape2D = dataDimensions

        self._unflatten_implementation(shape2D, order)

        if len(dataDimensions) == 2:
            pNames, fNames = self._unflattenNames()
            if pNames and len(pNames) != shape2D[0]:
                pNames = None
            if fNames and len(fNames) != shape2D[1]:
                fNames = None
        else:
            pNames, fNames = (None, None)

        self._shape = list(dataDimensions)
        self.points.setNames(pNames, useLog=False)
        self.features.setNames(fNames, useLog=False)

        handleLogging(useLog, 'prep', "unflatten", self.getTypeString(),
                      Base.unflatten, dataDimensions, order)


    @limitedTo2D
    def merge(self, other, point='strict', feature='union', onFeature=None,
              force=False, useLog=None):
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
              caller and callee will be filled with np.NaN.
            * 'intersection': Return only points/features shared between
              the caller  and callee. If ``onFeature`` is None,
              point / feature names are required.
            * 'left': Return only the points/features from the caller.
              Any missing data from the callee will be filled with
              np.NaN.
        onFeature : identifier, None
            The name or index of the feature present in both objects to
            merge on.  If None, the merge will be based on point names.
        force : bool
            When True, the ``point`` or ``feature`` parameter set to
            'strict' does not require names along that axis. The merge
            is forced to continue as True acknowledges that each index
            along that axis is equal between the two objects. Has no
            effect if neither parameter is set to 'strict'.
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
        >>> left = nimble.data('Matrix', dataL, pointNames=pNamesL,
        ...                    featureNames=fNamesL)
        >>> dataR = [['Z', "f", 6], ['Y', "e", 5], ['X', "d", 4]]
        >>> fNamesR = ["f3", "f4", "f5"]
        >>> pNamesR = ["p3", "p2", "p1"]
        >>> right = nimble.data('Matrix', dataR, pointNames=pNamesR,
        ...                     featureNames=fNamesR)
        >>> left.merge(right, point='strict', feature='union')
        >>> left
        Matrix(
            [[a 1 X d 4]
             [b 2 Y e 5]
             [c 3 Z f 6]]
            pointNames={'p1':0, 'p2':1, 'p3':2}
            featureNames={'f1':0, 'f2':1, 'f3':2, 'f4':3, 'f5':4}
            )
        >>> left = nimble.data('Matrix', dataL, pointNames=pNamesL,
        ...                    featureNames=fNamesL)
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
        >>> left = nimble.data("DataFrame", dataL, featureNames=fNamesL)
        >>> dataR = [['id3', "x", 7], ['id4', "y", 8], ['id5', "z", 9]]
        >>> fNamesR = ["id", "f4", "f5"]
        >>> right = nimble.data("DataFrame", dataR,
        ...                     featureNames=fNamesR)
        >>> left.merge(right, point='union', feature='union',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[ a  1.000 id1 nan  nan ]
             [ b  2.000 id2 nan  nan ]
             [ c  3.000 id3  x  7.000]
             [nan  nan  id4  y  8.000]
             [nan  nan  id5  z  9.000]]
            featureNames={'f1':0, 'f2':1, 'id':2, 'f4':3, 'f5':4}
            )
        >>> left = nimble.data("DataFrame", dataL, featureNames=fNamesL)
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
        >>> left = nimble.data("DataFrame", dataL, featureNames=fNamesL)
        >>> left.merge(right, point='union', feature='left',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[ a  1.000 id1]
             [ b  2.000 id2]
             [ c  3.000 id3]
             [nan  nan  id4]
             [nan  nan  id5]]
            featureNames={'f1':0, 'f2':1, 'id':2}
            )
        >>> left = nimble.data("DataFrame", dataL, featureNames=fNamesL)
        >>> left.merge(right, point='intersection', feature='union',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[c 3 id3 x 7]]
            featureNames={'f1':0, 'f2':1, 'id':2, 'f4':3, 'f5':4}
            )
        >>> left = nimble.data("DataFrame", dataL, featureNames=fNamesL)
        >>> left.merge(right, point='intersection',
        ...            feature='intersection', onFeature="id")
        >>> left
        DataFrame(
            [[id3]]
            featureNames={'id':0}
            )
        >>> left = nimble.data("DataFrame", dataL, featureNames=fNamesL)
        >>> left.merge(right, point='intersection', feature='left',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[c 3 id3]]
            featureNames={'f1':0, 'f2':1, 'id':2}
            )
        >>> left = nimble.data("DataFrame", dataL, featureNames=fNamesL)
        >>> left.merge(right, point='left', feature='union',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[a 1 id1 nan  nan ]
             [b 2 id2 nan  nan ]
             [c 3 id3  x  7.000]]
            featureNames={'f1':0, 'f2':1, 'id':2, 'f4':3, 'f5':4}
            )

        >>> left = nimble.data("DataFrame", dataL, featureNames=fNamesL)
        >>> left.merge(right, point='left', feature='intersection',
        ...            onFeature="id")
        >>> left
        DataFrame(
            [[id1]
             [id2]
             [id3]]
            featureNames={'id':0}
            )
        >>> left = nimble.data("DataFrame", dataL, featureNames=fNamesL)
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
                                                    onFeature, force)
        else:
            self._genericMergeFrontend(other, point, feature, onFeature)

        handleLogging(useLog, 'prep', "merge", self.getTypeString(),
                      Base.merge, other, point, feature, onFeature, force)

    def _genericStrictMerge_implementation(self, other, point, feature,
                                           onFeature, force):
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
            lAxis = self.points
            rAxis = tmpOther.points
            point = "intersection"
        else:
            axis = 'feature'
            lAxis = self.features
            rAxis = tmpOther.features
            feature = "intersection"

        if len(lAxis) != len(rAxis):
            msg = "Both objects must have the same number of "
            msg += "{0}s when {0}='strict'".format(axis)
            raise InvalidArgumentValue(msg)
        # if using strict with onFeature instead of point names, we need to
        # make sure each id has a unique match in the other object
        if onFeature is not None and axis == 'point':
            try:
                feat = self[:, onFeature]
                if len(set(feat)) != len(self.points):
                    msg = "when point='strict', onFeature must contain only "
                    msg += "unique values"
                    raise InvalidArgumentValueCombination(msg)
                if sorted(feat) != sorted(tmpOther[:, onFeature]):
                    msg = "When point='strict', onFeature must have a unique, "
                    msg += "matching value in each object"
                    raise InvalidArgumentValueCombination(msg)
            except KeyError as e:
                msg = "could not locate feature '{0}' ".format(onFeature)
                msg += "in both objects"
                raise InvalidArgumentValue(msg) from e

            self._genericMergeFrontend(tmpOther, point, feature, onFeature)
        else:
            lNames = lAxis._getNamesNoGeneration()
            rNames = rAxis._getNamesNoGeneration()
            lAllNames = not lAxis._anyDefaultNames()
            rAllNames = not rAxis._anyDefaultNames()
            # strict implies that the points or features are the same
            # with all names reordering can occur otherwise each index
            # is treated as equal.
            if lAllNames and rAllNames:
                if sorted(lNames) != sorted(rNames):
                    msg = "When {0}='strict', the {0} names may ".format(axis)
                    msg += "be in a different order but must match exactly"
                    raise InvalidArgumentValue(msg)
                endNames = lNames
            elif force and lAllNames and rNames is None:
                rAxis.setNames(lNames, useLog=False)
                endNames = lNames
            elif force and lNames is None and rAllNames:
                lAxis.setNames(rNames, useLog=False)
                endNames = rNames
            elif force: # default names present
                endNames = mergeNames(lAxis.getNames(), rAxis.getNames())
                # need to alter default names so that _genericMergeFrontend
                # treats them as non-default and equal. After the data is
                # merged, the names are reset to their default state.
                try:
                    strictNames = ['_STRICT' + str(i) if n is None else n
                                   for i, n in enumerate(endNames)]
                    lAxis.setNames(strictNames, useLog=False)
                    rAxis.setNames(strictNames, useLog=False)
                except InvalidArgumentValue as e:
                    msg = "When {axis}='strict' and default {axis} names "
                    msg += "exist, names cannot be reordered. The {axis} "
                    msg += "names do not match at each index where both  "
                    msg += "objects have non-default {axis} names."
                    msg = msg.format(axis=axis)
                    raise InvalidArgumentValue(msg) from e
            else: # default names and not force
                msg = 'Non-default {axis} names are required in both objects '
                msg += 'unless the force parameter is set to True to indicate '
                msg += 'that each {axis} is equal between the two objects'
                raise InvalidArgumentValue(msg.format(axis=axis))

            self._genericMergeFrontend(tmpOther, point, feature, onFeature)

            # only reset names if we did not generate them
            if lNames is None and rNames is None:
                lAxis.setNames(None, useLog=False)
            else:
                lAxis.setNames(endNames, useLog=False)

    def _genericMergeFrontend(self, other, point, feature, onFeature):
        # validation
        bothPtNamesCreated = (self.points._namesCreated()
                              and other.points._namesCreated())
        if onFeature is None and not bothPtNamesCreated:
            msg = "Point names are required in both objects when "
            msg += "onFeature is None"
            raise InvalidArgumentValueCombination(msg)
        bothFtNamesCreated = (self.features._namesCreated()
                              and other.features._namesCreated())
        if not bothFtNamesCreated:
            msg = "Feature names are required in both objects"
            raise InvalidArgumentValueCombination(msg)

        if onFeature is not None:
            if not isinstance(onFeature, str):
                # index allowed only if we can verify feature names match
                ftName = self.features.getName(onFeature)
                if (ftName != other.features.getName(onFeature)
                        or ftName is None):
                    msg = 'The feature names at index {0} '.format(onFeature)
                    msg += 'do not match in each object'
                    raise InvalidArgumentValue(msg)
                onFeature = ftName
            try:
                uniqueFtL = len(set(self[:, onFeature])) == len(self.points)
                uniqueFtR = len(set(other[:, onFeature])) == len(other.points)
                if not (uniqueFtL or uniqueFtR):
                    msg = "nimble only supports joining on a feature which "
                    msg += "contains only unique values in one or both objects"
                    raise InvalidArgumentValue(msg)
            except KeyError as e:
                msg = "could not locate feature '{0}' ".format(onFeature)
                msg += "in both objects"
                raise InvalidArgumentValue(msg) from e

        matchingFts = self.features._getMatchingNames(other)
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

        lFtNames = self.features._namesCreated()
        rFtNames = other.features._namesCreated()
        if feature == "intersection":
            if lFtNames:
                ftNames = [n for n in self.features.getNames()
                           if n in matchingFts]
                self.features.setNames(ftNames, useLog=False)
        elif feature == "union":
            if lFtNames and rFtNames:
                ftNamesL = self.features.getNames()
                ftNamesR = [name for name in other.features.getNames()
                            if name not in matchingFts]
                ftNames = ftNamesL + ftNamesR
                self.features.setNames(ftNames, useLog=False)
            elif lFtNames:
                ftNamesL = self.features.getNames()
                ftNamesR = [None] * len(other.features)
                ftNames = ftNamesL + ftNamesR
                self.features.setNames(ftNames, useLog=False)
            elif rFtNames:
                ftNamesL = [None] * len(self.features)
                ftNamesR = other.features.getNames()
                ftNames = ftNamesL + ftNamesR
                self.features.setNames(ftNames, useLog=False)
        # no name setting needed for left

        lPtNames = self.points._namesCreated()
        rPtNames = other.points._namesCreated()
        if onFeature is None and point == 'left':
            if lPtNames:
                self.points.setNames(self.points.getNames(), useLog=False)
        elif onFeature is None and point == 'intersection':
            # default names cannot be included in intersection
            ptNames = [name for name in self.points.getNames()
                       if name in other.points.getNames()
                       and name is not None]
            self.points.setNames(ptNames, useLog=False)
        elif onFeature is None:
            # union cases
            if lPtNames and rPtNames:
                ptNamesL = self.points.getNames()
                ptNamesR = other.points.getNames()
                ptNames = ptNamesL + [name for name in ptNamesR
                                      if name is None or name not in ptNamesL]
                self.points.setNames(ptNames, useLog=False)
            elif lPtNames:
                ptNamesL = self.points.getNames()
                ptNamesR = [None] * len(other.points)
                ptNames = ptNamesL + ptNamesR
                self.points.setNames(ptNames, useLog=False)
            elif rPtNames:
                ptNamesL = [None] * len(self.points)
                ptNamesR = other.points.getNames()
                ptNames = ptNamesL + ptNamesR
                self.points.setNames(ptNames, useLog=False)
        else:
            self.points.namesInverse = None
            self.points.names = None

    #############################
    #  Linear Algebra functions #
    #############################
    @limitedTo2D
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

    @limitedTo2D
    def solveLinearSystem(self, b, solveFunction='solve'):
        """
        Solves the linear equation A * x = b for unknown x.

        Parameters
        ----------
        b : nimble Base object.
            Vector shaped object.
        solveFuction : str

            * 'solve' - assumes square matrix.
            * 'least squares' - Computes object x such that 2-norm
              determinant of b - A x is minimized.
        """
        if not isinstance(b, Base):
            msg = "b must be an instance of Base."
            raise InvalidArgumentType(msg)

        msg = "Valid methods are: 'solve' and 'least squares'."

        if not isinstance(solveFunction, str):
            raise InvalidArgumentType(msg)
        if solveFunction == 'solve':
            return nimble.calculate.solve(self, b)
        if solveFunction == 'least squares':
            return nimble.calculate.leastSquaresSolution(self, b)

        raise InvalidArgumentValue(msg)



    ###############################################################
    ###############################################################
    ###   Subclass implemented numerical operation functions    ###
    ###############################################################
    ###############################################################
    @limitedTo2D
    def matrixMultiply(self, other):
        """
        Perform matrix multiplication.
        """
        return self.__matmul__(other)

    @limitedTo2D
    def __matmul__(self, other):
        """
        Perform matrix multiplication.
        """
        return self._genericMatMul_implementation('__matmul__', other)

    @limitedTo2D
    def __rmatmul__(self, other):
        """
        Perform matrix multiplication with this object on the right.
        """
        return self._genericMatMul_implementation('__rmatmul__', other)

    @limitedTo2D
    def __imatmul__(self, other):
        """
        Perform in place matrix multiplication.
        """
        ret = self.__matmul__(other)
        if ret is not NotImplemented:
            self._referenceFrom(ret)
            ret = self
        return ret

    def _genericMatMul_implementation(self, opName, other):
        if not isinstance(other, Base):
            return NotImplemented
        # Test element type self
        if len(self.points) == 0 or len(self.features) == 0:
            msg = "Cannot do a multiplication when points or features is empty"
            raise ImproperObjectAction(msg)

        # test element type other
        if len(other.points) == 0 or len(other.features) == 0:
            msg = "Cannot do a multiplication when points or features is "
            msg += "empty"
            raise ImproperObjectAction(msg)

        try:
            self._convertToNumericTypes()
        except ImproperObjectAction:
            self._numericValidation()
        try:
            other._convertToNumericTypes()
        except ImproperObjectAction:
            other._numericValidation(right=True)

        if opName.startswith('__r'):
            caller = other
            callee = self
        else:
            caller = self
            callee = other

        if len(caller.features) != len(callee.points):
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
        except TypeError:
            # help determine the source of the error
            self._numericValidation()
            other._numericValidation(right=True)
            raise # exception should be raised above, but just in case

        if caller.points._namesCreated():
            ret.points.setNames(caller.points.getNames(), useLog=False)
        if callee.features._namesCreated():
            ret.features.setNames(callee.features.getNames(), useLog=False)

        binaryOpNamePathMerge(caller, callee, ret, None, 'merge')

        return ret

    @limitedTo2D
    def matrixPower(self, power):
        """
        Perform matrix power operations on a square matrix.

        For positive power values, return the result of repeated matrix
        multiplication over this object. For power of zero, return an
        identity matrix. For negative power values, return the result of
        repeated matrix multiplication over the inverse of this object,
        provided the object can be inverted using
        nimble.calculate.inverse.
        """
        if not isinstance(power, (int, np.int)):
            msg = 'power must be an integer'
            raise InvalidArgumentType(msg)
        if not len(self.points) == len(self.features):
            msg = 'Cannot perform matrix power operations with this object. '
            msg += 'Matrix power operations require square objects '
            msg += '(number of points is equal to number of features)'
            raise ImproperObjectAction(msg)
        if power == 0:
            operand = nimble.identity(self.getTypeString(), len(self.points))
        elif power > 0:
            operand = self.copy()
            # avoid name conflict in matrixMultiply; names set later
            operand.points.setNames(None, useLog=False)
            operand.features.setNames(None, useLog=False)
        else:
            try:
                operand = nimble.calculate.inverse(self)
            except (InvalidArgumentType, InvalidArgumentValue) as e:
                exceptionType = type(e)
                msg = "Failed to calculate the matrix inverse using "
                msg += "nimble.calculate.inverse. For safety and efficiency, "
                msg += "matrixPower does not attempt to use pseudoInverse but "
                msg += "it is available to users in nimble.calculate. "
                msg += "The inverse operation failed because: " + e.message
                raise exceptionType(msg) from e

        ret = operand
        # loop only applies when abs(power) > 1
        for _ in range(abs(power) - 1):
            ret = ret.matrixMultiply(operand)

        ret.points.setNames(self.points._getNamesNoGeneration(), useLog=False)
        ret.features.setNames(self.features._getNamesNoGeneration(),
                              useLog=False)

        return ret

    def __mul__(self, other):
        """
        Perform elementwise multiplication or scalar multiplication,
        depending in the input ``other``.
        """
        return self._genericBinaryOperations('__mul__', other)

    def __rmul__(self, other):
        """
        Perform elementwise multiplication with this object on the right
        """
        return self._genericBinaryOperations('__rmul__', other)

    def __imul__(self, other):
        """
        Perform in place elementwise multiplication or scalar
        multiplication, depending in the input ``other``.
        """
        return self._genericBinaryOperations('__imul__', other)

    def __add__(self, other):
        """
        Perform addition on this object, element wise if 'other' is a
        nimble Base object, or element wise with a scalar if other is
        some kind of numeric value.
        """
        return self._genericBinaryOperations('__add__', other)

    def __radd__(self, other):
        """
        Perform scalar addition with this object on the right
        """
        return self._genericBinaryOperations('__radd__', other)

    def __iadd__(self, other):
        """
        Perform in-place addition on this object, element wise if
        ``other`` is a nimble Base object, or element wise with a scalar
        if ``other`` is some kind of numeric value.
        """
        return self._genericBinaryOperations('__iadd__', other)

    def __sub__(self, other):
        """
        Subtract from this object, element wise if ``other`` is a nimble
        data object, or element wise by a scalar if ``other`` is some
        kind of numeric value.
        """
        return self._genericBinaryOperations('__sub__', other)

    def __rsub__(self, other):
        """
        Subtract each element of this object from the given scalar.
        """
        return self._genericBinaryOperations('__rsub__', other)

    def __isub__(self, other):
        """
        Subtract (in place) from this object, element wise if ``other``
        is a nimble Base object, or element wise with a scalar if
        ``other`` is some kind of numeric value.
        """
        return self._genericBinaryOperations('__isub__', other)

    def __truediv__(self, other):
        """
        Perform true division using this object as the numerator,
        elementwise if ``other`` is a nimble Base object, or elementwise
        by a scalar if other is some kind of numeric value.
        """
        return self._genericBinaryOperations('__truediv__', other)

    def __rtruediv__(self, other):
        """
        Perform element wise true division using this object as the
        denominator, and the given scalar value as the numerator.
        """
        return self._genericBinaryOperations('__rtruediv__', other)

    def __itruediv__(self, other):
        """
        Perform true division (in place) using this object as the
        numerator, elementwise if ``other`` is a nimble Base object, or
        elementwise by a scalar if ``other`` is some kind of numeric
        value.
        """
        return self._genericBinaryOperations('__itruediv__', other)

    def __floordiv__(self, other):
        """
        Perform floor division using this object as the numerator,
        elementwise if ``other`` is a nimble Base object, or elementwise
        by a scalar if ``other`` is some kind of numeric value.
        """
        return self._genericBinaryOperations('__floordiv__', other)

    def __rfloordiv__(self, other):
        """
        Perform elementwise floor division using this object as the
        denominator, and the given scalar value as the numerator.

        """
        return self._genericBinaryOperations('__rfloordiv__', other)

    def __ifloordiv__(self, other):
        """
        Perform floor division (in place) using this object as the
        numerator, elementwise if ``other`` is a nimble Base object, or
        elementwise by a scalar if ```other``` is some kind of numeric
        value.
        """
        return self._genericBinaryOperations('__ifloordiv__', other)

    def __mod__(self, other):
        """
        Perform mod using the elements of this object as the dividends,
        elementwise if ``other`` is a nimble Base object, or elementwise
        by a scalar if other is some kind of numeric value.
        """
        return self._genericBinaryOperations('__mod__', other)

    def __rmod__(self, other):
        """
        Perform mod using the elements of this object as the divisors,
        and the given scalar value as the dividend.
        """
        return self._genericBinaryOperations('__rmod__', other)

    def __imod__(self, other):
        """
        Perform mod (in place) using the elements of this object as the
        dividends, elementwise if 'other' is a nimble Base object, or
        elementwise by a scalar if other is some kind of numeric value.
        """
        return self._genericBinaryOperations('__imod__', other)

    @to2args
    def __pow__(self, other, z): # pylint: disable=unused-argument
        """
        Perform exponentiation (iterated __mul__) using the elements of
        this object as the bases, elementwise if ``other`` is a nimble
        data object, or elementwise by a scalar if ``other`` is some
        kind of numeric value.
        """
        return self._genericBinaryOperations('__pow__', other)

    def __rpow__(self, other):
        """
        Perform elementwise exponentiation (iterated __mul__) using the
        ``other`` scalar value as the bases.
        """
        return self._genericBinaryOperations('__rpow__', other)

    def __ipow__(self, other):
        """
        Perform in-place exponentiation (iterated __mul__) using the
        elements of this object as the bases, element wise if ``other``
        is a nimble Base object, or elementwise by a scalar if ``other``
        is some kind of numeric value.
        """
        return self._genericBinaryOperations('__ipow__', other)

    def __pos__(self):
        """
        Return this object.
        """
        ret = self.copy()
        ret._name = None

        return ret

    def __neg__(self):
        """
        Return this object where every element has been multiplied by -1
        """
        ret = self.copy()
        ret *= -1
        ret._name = None

        return ret

    def __abs__(self):
        """
        Perform element wise absolute value on this object
        """
        with self._treatAs2D():
            ret = self.calculateOnElements(abs, useLog=False)
        ret._shape = self._shape.copy()
        if self.points._namesCreated():
            ret.points.setNames(self.points.getNames(), useLog=False)
        else:
            ret.points.setNames(None, useLog=False)
        if self.features._namesCreated():
            ret.features.setNames(self.features.getNames(), useLog=False)
        else:
            ret.points.setNames(None, useLog=False)

        ret._name = None
        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath
        return ret

    def _numericValidation(self, right=False):
        """
        Validate the object elements are all numeric.
        """
        try:
            self.calculateOnElements(checkNumeric, useLog=False)
        except ValueError as e:
            msg = "The object on the {0} contains non numeric data, "
            msg += "cannot do this operation"
            if right:
                msg = msg.format('right')
                raise InvalidArgumentValue(msg) from e
            msg = msg.format('left')
            raise ImproperObjectAction(msg) from e

    def _genericBinary_sizeValidation(self, opName, other):
        if self._shape != other._shape:
            msg = "The dimensions of the objects must be equal."
            raise InvalidArgumentValue(msg)
        if len(self.points) != len(other.points):
            msg = "The number of points in each object must be equal. "
            msg += "(self=" + str(len(self.points)) + " vs other="
            msg += str(len(other.points)) + ")"
            raise InvalidArgumentValue(msg)
        if len(self.features) != len(other.features):
            msg = "The number of features in each object must be equal."
            raise InvalidArgumentValue(msg)

        if len(self.points) == 0 or len(self.features) == 0:
            msg = "Cannot do " + opName + " when points or features is empty"
            raise ImproperObjectAction(msg)

    def _validateDivMod(self, opName, other):
        """
        Validate values will not lead to zero division.
        """
        if opName.startswith('__r'):
            toCheck = self
        else:
            toCheck = other

        if isinstance(toCheck, Base):
            if toCheck.containsZero():
                msg = "Cannot perform " + opName + " when the second argument "
                msg += "contains any zeros"
                raise ZeroDivisionError(msg)
        elif toCheck == 0:
            msg = "Cannot perform " + opName + " when the second argument "
            msg += "is zero"
            raise ZeroDivisionError(msg)

    def _diagnoseFailureAndRaiseException(self, opName, other, error):
        """
        Raise exceptions explaining why an arithmetic operation could
        not be performed successfully between two objects.
        """
        if 'pow' in opName and isinstance(error, FloatingPointError):
            if 'divide by zero' in str(error):
                msg = 'Zeros cannot be raised to negative exponents'
                raise ZeroDivisionError(msg)
            msg = "Complex number results are not allowed"
            raise ImproperObjectAction(msg)
        # Test element type self
        self._numericValidation()
        # test element type other
        if isinstance(other, Base):
            other._numericValidation(right=True)

    def _genericBinary_validation(self, opName, other):
        otherBase = isinstance(other, Base)
        if otherBase:
            self._genericBinary_sizeValidation(opName, other)
        elif not (looksNumeric(other) or isDatetime(other)):
            msg = "'other' must be an instance of a nimble Base object or a "
            msg += "scalar"
            raise InvalidArgumentType(msg)
        # divmod operations inconsistently raise exceptions for zero division
        # it is more efficient to validate now than validate after operation
        if 'div' in opName or 'mod' in opName:
            self._validateDivMod(opName, other)

    def _genericBinary_axisNames(self, opName, other, conversionKwargs):
        """
        Determines axis names for operations between two Base objects.

        When both Base objects, axis names determine the operations
        that can be performed. When both axes are determined to have
        equal names, the operation can be performed and the axis names
        of the returned object can be set based on the left and right
        objects. When only one axis has equal names, the other axis must
        have disjoint names and the names of the disjoint axis cannot be
        used to set names for the returned object. In all other cases,
        the operation is disallowed and an exception will be raised.
        """
        # inplace operations require both point and feature name equality
        try:
            self._validateEqualNames('feature', 'feature', opName, other)
            ftNamesEqual = True
        except InvalidArgumentValue:
            if opName.startswith('__i'):
                raise
            ftNamesEqual = False
        try:
            self._validateEqualNames('point', 'point', opName, other)
            ptNamesEqual = True
        except InvalidArgumentValue:
            if opName.startswith('__i'):
                raise
            ptNamesEqual = False
        # for *NamesEqual to be False, the left and right objects must have
        # at least one unequal non-default name along that axis.
        if not ftNamesEqual and not ptNamesEqual:
            msg = "When both point and feature names are present, {0} "
            msg += "requires either the point names or feature names to "
            msg += "be equal between the left and right objects"
            raise InvalidArgumentValue(msg.format(opName))

        # determine axis names for returned object
        try:
            other._convertToNumericTypes(**conversionKwargs)
        except ImproperObjectAction:
            other._numericValidation(right=True)
        # everything else that uses this helper is a binary scalar op
        retPNames, retFNames = mergeNonDefaultNames(self, other)
        # in these cases we cannot define names for the disjoint axis
        if ftNamesEqual and not ptNamesEqual:
            self._genericBinary_axisNamesDisjoint('point', other, opName)
            retPNames = None
        elif ptNamesEqual and not ftNamesEqual:
            self._genericBinary_axisNamesDisjoint('feature', other, opName)
            retFNames = None

        return retPNames, retFNames

    def _genericBinary_axisNamesDisjoint(self, axis, other, opName):
        """
        Verify that the axis names for this object and the other object
        are disjoint. Equal default names do not imply the same point or
        feature, so default names are ignored.
        """
        def nonDefaultNames(names):
            return (n for n in names if n is not None)

        if axis == 'point':
            sNames = nonDefaultNames(self.points.getNames())
            oNames = nonDefaultNames(other.points.getNames())
            equalAxis = 'feature'
        else:
            sNames = nonDefaultNames(self.features.getNames())
            oNames = nonDefaultNames(other.features.getNames())
            equalAxis = 'point'

        if not set(sNames).isdisjoint(oNames):
            matches = [n for n in sNames if n in oNames]
            msg = "{0} between objects with equal {1} names must have "
            msg += "unique {2} names. However, the {2} names {3} were "
            msg += "found in both the left and right objects"
            msg = msg.format(opName, equalAxis, axis, matches)
            raise InvalidArgumentValue(msg)

    def _convertToNumericTypes(self, allowInt=True, allowBool=True):
        """
        Convert the data, inplace, to numeric type if necessary.
        """
        usableTypes = [float]
        if not all(isinstance(a, bool) for a in (allowInt, allowBool)):
            msg = 'all arguments for _convertToNumericTypes must be bools'
            raise InvalidArgumentValue(msg)
        if allowInt:
            usableTypes.append(int)
        if allowBool:
            usableTypes.append(bool)
        usableTypes = tuple(usableTypes)
        try:
            return self._convertToNumericTypes_implementation(usableTypes)
        except (ValueError, TypeError) as e:
            msg = 'Unable to coerce the data to the type required for this '
            msg += 'operation.'
            raise ImproperObjectAction(msg) from e

    def _genericBinaryOperations(self, opName, other):
        conversionKwargs = {}
        if 'pow' in opName:
            conversionKwargs['allowInt'] = False
            conversionKwargs['allowBool'] = False

        if opName.startswith('__i'):
            obj = self
        else:
            obj = self.copy()

        try:
            obj._convertToNumericTypes(**conversionKwargs)
        except ImproperObjectAction:
            obj._numericValidation()
        if isinstance(other, Stretch):
            # __ipow__ does not work if return NotImplemented
            if opName == '__ipow__':
                return pow(self, other)
            return NotImplemented

        obj._genericBinary_validation(opName, other)

        # figure out return obj's point / feature names
        otherBase = isinstance(other, Base)
        if otherBase:
            other = other.copy()
            retPNames, retFNames = obj._genericBinary_axisNames(
                opName, other, conversionKwargs)
        else:
            retPNames = obj.points._getNamesNoGeneration()
            retFNames = obj.features._getNamesNoGeneration()

        try:
            useOp = opName
            if opName.startswith('__i'):
                # inplace operations will modify the data even if op fails
                # use not inplace operation, setting to inplace occurs after
                useOp = opName[:2] + opName[3:]
            with np.errstate(divide='raise', invalid='raise'):
                ret = obj._binaryOperations_implementation(useOp, other)
        except (TypeError, ValueError, FloatingPointError) as error:
            obj._diagnoseFailureAndRaiseException(opName, other, error)
            raise # backup; should be diagnosed and raised above

        ret._shape = self._shape
        if opName.startswith('__i'):
            self._referenceFrom(ret, paths=(self._absPath, self._relPath))
            ret = self
        ret.points.setNames(retPNames, useLog=False)
        ret.features.setNames(retFNames, useLog=False)

        nameSource = 'self' if opName.startswith('__i') else None
        pathSource = 'merge' if otherBase else 'self'
        binaryOpNamePathMerge(obj, other, ret, nameSource, pathSource)
        return ret


    def _defaultBinaryOperations_implementation(self, opName, other):
        with self._treatAs2D():
            selfData = self.copy('numpyarray')
        if isinstance(other, Base):
            with other._treatAs2D():
                otherData = other.copy('numpyarray')
        else:
            otherData = other
        data = getattr(selfData, opName)(otherData)
        ret = createDataNoValidation(self.getTypeString(), data)

        return ret

    @limitedTo2D
    def __and__(self, other):
        return self._genericLogicalBinary('__and__', other)

    @limitedTo2D
    def __or__(self, other):
        return self._genericLogicalBinary('__or__', other)

    @limitedTo2D
    def __xor__(self, other):
        return self._genericLogicalBinary('__xor__', other)

    @limitedTo2D
    def __invert__(self):
        boolObj = self._logicalValidationAndConversion()
        ret = boolObj.matchingElements(lambda v: not v, useLog=False)
        ret.points.setNames(self.points._getNamesNoGeneration(), useLog=False)
        ret.features.setNames(self.features._getNamesNoGeneration(),
                              useLog=False)
        return ret

    def _genericLogicalBinary(self, opName, other):
        if isinstance(other, Stretch):
            return getattr(other, opName)(self)
        if not isinstance(other, Base):
            msg = 'other must be an instance of a nimble Base object'
            raise InvalidArgumentType(msg)
        self._genericBinary_sizeValidation(opName, other)
        lhsBool = self._logicalValidationAndConversion()
        rhsBool = other._logicalValidationAndConversion()

        return lhsBool._genericBinaryOperations(opName, rhsBool)

    def _isBooleanData(self):
        return False

    def _logicalValidationAndConversion(self):
        if not self._isBooleanData():
            validValues = match.allValues([True, False, 0, 1])
            if not validValues(self):
                msg = 'logical operations can only be performed on data '
                msg += 'containing True, False, 0 and 1 values'
                raise ImproperObjectAction(msg)

            ret = self.matchingElements(bool, useLog=False)
            ret.points.setNames(self.points._getNamesNoGeneration(),
                                useLog=False)
            ret.features.setNames(self.features._getNamesNoGeneration(),
                                  useLog=False)
            return ret

        return self

    @property
    @limitedTo2D
    def stretch(self):
        """
        Extend along a one-dimensional axis to fit another object.

        This attribute allows arithmetic operations to occur between
        objects of different shapes (sometimes referred to as
        broadcasting). The operation will pair the point or feature in
        this object with each point or feature in the other object.
        Operations can occur with a nimble Base object or a stretched
        object which is one-dimensional along the opposite axis. Note
        the operation will always return a Base object of the same type
        as the left-hand operand.

        Examples
        --------
        Nimble Base object with a stretched point.

        >>> rawBase = [[1, 2, 3], [4, 5, 6], [0, -1, -2]]
        >>> rawPt = [1, 2, 3]
        >>> baseObj = nimble.data('Matrix', rawBase)
        >>> pointObj = nimble.data('List', rawPt)
        >>> baseObj * pointObj.stretch
        Matrix(
            [[1 4  9 ]
             [4 10 18]
             [0 -2 -6]]
            )

        Stretched feature with nimble Base object.

        >>> rawBase = [[1, 2, 3], [4, 5, 6], [0, -1, -2]]
        >>> rawFt = [[1], [2], [3]]
        >>> baseObj = nimble.data('Matrix', rawBase)
        >>> featObj = nimble.data('List', rawFt)
        >>> featObj.stretch + baseObj
        List(
            [[2 3 4]
             [6 7 8]
             [3 2 1]]
            )

        Two stretched objects.

        >>> rawPt = [[1, 2, 3]]
        >>> rawFt = [[1], [2], [3]]
        >>> pointObj = nimble.data('Matrix', rawPt)
        >>> featObj = nimble.data('List', rawFt)
        >>> pointObj.stretch - featObj.stretch
        Matrix(
            [[0  1  2]
             [-1 0  1]
             [-2 -1 0]]
            )
        >>> featObj.stretch - pointObj.stretch
        List(
            [[0 -1 -2]
             [1 0  -1]
             [2 1  0 ]]
            )
        """
        return Stretch(self)

    ############################
    ############################
    ###   Helper functions   ###
    ############################
    ############################

    def _referenceFrom(self, other, **kwargs):
        """
        Reference data and metadata data from another object.

        This method use the following defaults:
            name=self.name, paths=(other._absPath, other._relPath),
            pointNames=other.points.names, featureNames=other.features.names,
            reuseData=True
        These can be modified through the keyword arguments.

        When not a view, the default reuseData=True means that the _data
        attribute for this object and the other object will be the same in
        memory.
        """
        if not self.getTypeString() == other.getTypeString():
            msg = 'to reference another object, they must be the same type'
            raise InvalidArgumentType(msg)
        if 'data' in kwargs:
            msg = 'Cannot provide "data" keyword, the data will be other._data'
            raise InvalidArgumentValue(msg)
        if 'shape' in kwargs:
            msg = 'Cannot provide "shape" keyword, the shape is determined by '
            msg = 'other._shape'
            raise InvalidArgumentValue(msg)
        if hasattr(other, '_source'): # view
            other = other.copy()

        kwargs['data'] = other._data
        kwargs['shape'] = other._shape
        # setdefault only sets if the key is not already present
        kwargs.setdefault('name', self.name)
        kwargs.setdefault('paths', (other._absPath, other._relPath))
        kwargs.setdefault('pointNames', other.points.namesInverse)
        kwargs.setdefault('featureNames', other.features.namesInverse)
        kwargs.setdefault('reuseData', True)

        self._referenceFrom_implementation(other, kwargs)

    def _referenceFrom_implementation(self, other, kwargs):
        """
        Reinitialize the object with the new keyword arguments.

        __init__ affects _id for this object and Base. self._id should stay
        the same and Base._id should be not increment.
        """
        # pylint: disable=unused-argument
        idVal = self._id
        self.__init__(**kwargs)
        self._id = idVal
        Base._id -= 1

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
        (tRowIDs, bRowIDs) = indicesSplit(maxRows, len(self.points))

        # we pull indices from two lists: tRowIDs and bRowIDs
        for sourceIndex in range(2):
            source = list([tRowIDs, bRowIDs])[sourceIndex]

            # add in the rowHolder, if needed
            if (sourceIndex == 1
                    and len(bRowIDs) + len(tRowIDs) < len(self.points)):
                names.append(rowHolder)

            for i in source:
                pname = self.points.getName(i)
                # omit default valued names
                if pname is None:
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
        if len(self.points) == 0 or len(self.features) == 0:
            fNames = None
            if includeFNames:
                fNames = []
            return [[]], [], fNames

        if maxHeight < 2 and maxHeight != len(self.points):
            msg = "If the number of points in this object is two or greater, "
            msg += "then we require that the input argument maxHeight also "
            msg += "be greater than or equal to two."
            raise InvalidArgumentValue(msg)

        cHoldWidth = len(colHold)
        cHoldTotal = len(colSep) + cHoldWidth
        nameCutIndex = maxStrLength - len(strHold)

        #setup a bundle of default values
        if maxHeight is None:
            maxHeight = len(self.points)
        if maxWidth is None:
            maxWidth = float('inf')

        maxRows = min(maxHeight, len(self.points))
        maxDataRows = maxRows

        (tRowIDs, bRowIDs) = indicesSplit(maxDataRows, len(self.points))
        combinedRowIDs = tRowIDs + bRowIDs
        if len(combinedRowIDs) < len(self.points):
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
        endIndex = len(self.features) // 2
        if len(self.features) % 2 == 1:
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
                if currFName is None:
                    currFName = ''
                fNameLen = len(currFName)
                if fNameLen > maxStrLength:
                    currFName = currFName[:nameCutIndex] + strHold
                    fNameLen = maxStrLength
                currWidth = fNameLen

            # check all values in this column (in the accepted rows)
            for i, rID in enumerate(combinedRowIDs):
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
                for i, val in enumerate(currCol):
                    if len(currTable) != len(currCol):
                        currTable.append([val])
                    else:
                        currTable[i].append(val)
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
            if numAdded == (len(self.features) - 1):
                totalWidth -= cHoldTotal
            totalWidth += len(colSep)

        # combine the tables. Have to reverse rTable because entries were
        # appended in a right to left order
        rColWidths.reverse()
        rFNames.reverse()
        fNames = lFNames + rFNames
        if numAdded == len(self.features):
            lColWidths += rColWidths
        else:
            lColWidths += [cHoldWidth] + rColWidths
            if includeFNames:
                fNames = lFNames + [colHold] + rFNames
        # return None if fNames is [] (includeFNames=False)
        fNames = fNames if fNames else None
        # pylint: disable=consider-using-enumerate
        for rowIndex in range(len(lTable)):
            if len(rTable) > 0:
                rTable[rowIndex].reverse()
                toAdd = rTable[rowIndex]
            else:
                toAdd = []

            if numAdded == len(self.features):
                lTable[rowIndex] += toAdd
            else:
                lTable[rowIndex] += [colHold] + toAdd

        return lTable, lColWidths, fNames

    def _equalPointNames(self, other):
        return equalNames(self.points._getNamesNoGeneration(),
                          other.points._getNamesNoGeneration())

    def _equalFeatureNames(self, other):
        return equalNames(self.features._getNamesNoGeneration(),
                          other.features._getNamesNoGeneration())

    def _validateEqualNames(self, leftAxis, rightAxis, callSym, other):

        if leftAxis == 'point':
            lnamesCreated = self.points._namesCreated()
        else:
            lnamesCreated = self.features._namesCreated()
        if rightAxis == 'point':
            rnamesCreated = self.points._namesCreated()
        else:
            rnamesCreated = self.features._namesCreated()

        if lnamesCreated or rnamesCreated:
            if leftAxis == 'point':
                lnames = self.points._getNamesNoGeneration()
            else:
                lnames = self.features._getNamesNoGeneration()
            if rightAxis == 'point':
                rnames = other.points._getNamesNoGeneration()
            else:
                rnames = other.features._getNamesNoGeneration()
            if lnames is None or rnames is None:
                return
            inconsistencies = inconsistentNames(lnames, rnames)

            if inconsistencies != {}:
                table = [['left', 'ID', 'right']]
                for i in sorted(inconsistencies.keys()):
                    lname = lnames[i]
                    rname = rnames[i]
                    lname = str(None) if lname is None else '"' + lname + '"'
                    rname = str(None) if rname is None else '"' + rname + '"'
                    table.append([lname, str(i), rname])

                msg = leftAxis + " to " + rightAxis + " name inconsistencies "
                msg += "when calling left." + callSym + "(right) \n"
                msg += nimble.core.logger.tableString.tableString(table)
                print(msg, file=sys.stderr)
                raise InvalidArgumentValue(msg)

    def _getAxis(self, axis):
        if axis == 'point':
            return self.points

        return self.features

    ####################
    # Abstract Methods #
    ####################

    @abstractmethod
    def _getPoints(self, names):
        """
        Get the object containing point-based methods for this object.
        """

    @abstractmethod
    def _getFeatures(self, names):
        """
        Get the object containing feature-based methods for this object.
        """

    @abstractmethod
    def _isIdentical_implementation(self, other):
        pass

    @abstractmethod
    def _writeFileCSV_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
        pass

    @abstractmethod
    def _writeFileMTX_implementation(self, outPath, includePointNames,
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
                             featureEnd, dropDimension):
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
    def _copy_implementation(self, to):
        pass

    @abstractmethod
    def _replaceRectangle_implementation(self, replaceWith, pointStart,
                                         featureStart, pointEnd, featureEnd):
        pass

    @abstractmethod
    def _flatten_implementation(self, order):
        pass

    @abstractmethod
    def _unflatten_implementation(self, reshape, order):
        pass

    @abstractmethod
    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):
        pass

    @abstractmethod
    def _convertToNumericTypes_implementation(self, usableTypes):
        pass

    @abstractmethod
    def _iterateElements_implementation(self, order, only):
        pass

    @abstractmethod
    def _transform_implementation(self, toTransform, points, features):
        pass

    @abstractmethod
    def _calculate_implementation(self, function, points, features,
                                  preserveZeros):
        pass

    @abstractmethod
    def _countUnique_implementation(self, points, features):
        pass

    @abstractmethod
    def _binaryOperations_implementation(self, opName, other):
        pass
