
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
import shutil
import re

import numpy as np
from scipy.sparse import coo_matrix


import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.core.logger import handleLogging, LogID
from nimble.match import QueryString
from nimble._utility import cloudpickle, h5py, plt, IPython, pd
from nimble._utility import isDatetime
from nimble._utility import tableString, prettyListString, quoteStrings
from nimble._utility import _getStatsFunction, acceptedStats
from .stretch import Stretch
from ._dataHelpers import formatIfNeeded, validateInputString
from ._dataHelpers import constructIndicesList
from ._dataHelpers import createListOfDict, createDictOfList
from ._dataHelpers import createDataNoValidation
from ._dataHelpers import csvCommaFormat
from ._dataHelpers import validateElementFunction, wrapMatchFunctionFactory
from ._dataHelpers import ElementIterator1D
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
from ._dataHelpers import binaryOpNamePathMerge
from ._dataHelpers import indicesSplit
from ._dataHelpers import prepLog


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

def checkNumericListType(listData):
    '''
    Function dedicated to seeing if list of lists that make up
    base._data contains entirely numeric data or not.
    '''
    if isinstance(listData, list):
      for item in listData:
        if isinstance(item, list):
            if not all(isinstance(sub_item, (int, float)) for sub_item in item):
                return False
        else:
            if not isinstance(item, (int, float)):
                return False
      return True
    return False

def isNumeric(base):
    '''
    Helper for certain stats methods to know if feature columns contain invalid
    values.
    '''
    def is_numeric_column(column):
        return pd.api.types.is_numeric_dtype(column)

    if pd.nimbleAccessible() and isinstance(base._data, pd.DataFrame):
        return all(is_numeric_column(column) for column in base._data.dtypes)
    elif isinstance(base._data, np.ndarray):
        return np.issubdtype(base._data.dtype, np.number)
    elif isinstance(base._data, coo_matrix):
        return base._data.dtype.kind in 'ifc'
    elif isinstance(base._data, list):
        return checkNumericListType(base._data)
    else:
        raise ValueError("Unsupported data type")

class Base(ABC):
    """
    The base class for all nimble data objects.

    All data types inherit their functionality from this object, then
    apply their own implementations of its methods (when necessary).
    Methods in this object apply to the entire object or its elements
    and its ``points`` and ``features`` attributes provide addtional
    methods that apply point-by-point and feature-by-feature,
    respectively.
    """
    # comments below for Sphinx docstring
    #: Identifier for this object within the log.
    #:
    #: An identifier unique within the current logging session is generated
    #: when the logID attribute is first accessed. Each ``logID`` string begins
    #: with "NIMBLE\_" and is followed an integer value. The integer values
    #: start at 0 and increment by 1. Assuming logging is enabled, this occurs
    #: when the object is created whenever possible, otherwise it will be
    #: generated when the first loggable method is called or triggered by user
    #: access. Note: User access can generate ``logID`` values that do not
    #: appear in the log, otherwise searching for the ``logID`` text in the log
    #: will locate all logged usages of this object.
    #:
    #: Examples
    #: --------
    #: >>> noName = nimble.data([])
    #: >>> noName.name
    #: >>> noName.logID
    #: '_NIMBLE_0_'
    #: >>> withName = nimble.data([], name='data')
    #: >>> withName.name
    #: 'data'
    #: >>> withName.logID
    #: '_NIMBLE_1_'
    logID = LogID('NIMBLE')

    def __init__(self, shape, pointNames=None, featureNames=None, name=None,
                 paths=(None, None), **kwds):
        """
        Class defining important data manipulation operations.

        Specifically, this includes setting the object name, shape,
        originating paths for the data, and sets the point and feature
        axis objects. Note: this method (as should all other __init__
        methods in this hierarchy) makes use of super().

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
        self._dims = list(shape)
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
        The number of points and features in the object.

        Return is a tuple in the format (# of points, # of features).

        See Also
        --------
        dimensions, Points, Features
        """
        if len(self._dims) > 2:
            return self._dims[0], np.prod(self._dims[1:])
        return self._dims[0], self._dims[1]
    
    @shape.setter
    def shape(self, value):
        raise AttributeError("User cannot directly set 'shape'. The shape of a base object is determined by the points and features.")

    @property
    def dimensions(self):
        """
        The true dimensions of this object.

        See Also
        --------
        shape, flatten, unflatten
        """
        return tuple(self._dims)
    
    @dimensions.setter
    def dimensions(self, value):
        raise AttributeError("User cannot directly set 'dimensions'. The dimensions of a base object are determined by its shape.")
    

    @property
    def points(self):
        """
        An object handling functions manipulating data by points.

        A point is an abstract slice containing data elements within
        some shared context. In a concrete sense, points can be thought
        of as the data rows but a row can be organized in many ways. To
        optimize for machine learning, each row should be modified to
        meet the definition of a point.

        This attribute is an object that can be used to iterate over the
        points (rows) and contains methods that operate over the data in
        this object point-by-point.

        See Also
        --------
        Points, features

        Examples
        --------
        >>> lst = [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]]
        >>> X = nimble.data(lst)
        >>> len(X.points)
        3
        >>> X.points.permute([2, 1, 0])
        >>> X
        <Matrix 3pt x 4ft
             0  1  2  3
           ┌───────────
         0 │ 0  0  0  0
         1 │ 5  6  7  8
         2 │ 1  2  3  4
        >

        Keywords
        --------
        rows, index, data points, instances, observations, values,
        iterate, iterrows
        """
        return self._points

    @points.setter
    def points(self, value):
        raise AttributeError("User cannot directly set 'points' of a base object.")

    @property
    def features(self):
        """
        An object handling functions manipulating data by features.

        A feature is an abstract slice of all data elements of the same
        kind across different contexts. In a concrete sense, features
        can be thought of as the data columns but a column can be
        organized in many ways. To optimize for machine learning, each
        column should be modified to meet the definition of a feature.

        This attribute is an object that can be used to iterate over the
        features (columns) and contains methods that operate over the
        data in this object feature-by-feature.

        See Also
        --------
        Features, points

        Examples
        --------
        >>> lst = [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]]
        >>> X = nimble.data(lst)
        >>> len(X.features)
        4
        >>> X.features.permute([3, 2, 1, 0])
        >>> X
        <Matrix 3pt x 4ft
             0  1  2  3
           ┌───────────
         0 │ 4  3  2  1
         1 │ 8  7  6  5
         2 │ 0  0  0  0
        >

        Keywords
        --------
        columns, variables, dimensions, attributes, predictors, iterate,
        items
        """
        return self._features

    @features.setter
    def features(self, value):
        raise AttributeError("User cannot directly set 'features' of a base object.")

    @property
    def name(self):
        """
        A name to be displayed when printing or logging this object.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            if not isinstance(value, str):
                msg = "The name of an object may only be a string or None"
                raise InvalidArgumentType(msg)
            if re.search(r'\s', value):
                msg = "Names are not permitted to have whitespace"
                raise InvalidArgumentValue(msg)
        self._name = value

    @property
    def absolutePath(self):
        """
        The path to the file this data originated from in absolute form.

        See Also
        --------
        relativePath

        Keywords
        --------
        file, location
        """
        return self._absPath

    @property
    def relativePath(self):
        """
        The path to the file this data originated from in relative form.

        See Also
        --------
        absolutePath

        Keywords
        --------
        file, location
        """
        return self._relPath

    @property
    def path(self):
        """
        The path to the file this data originated from.

        See Also
        --------
        absolutePath, relativePath
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
        if len(self._dims) > 2:
            savedShape = self._dims
            self._dims = [len(self.points), len(self.features)]
            try:
                yield self
            finally:
                self._dims = savedShape
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
        return self._dims[0] > 0 and self._dims[-1] > 0

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
        Points, Features, nimble.match

        Examples
        --------
        >>> from nimble.match import nonZero, positive
        >>> lst = [[0, 1, 2], [-2, -1, 0]]
        >>> X = nimble.data(lst)
        >>> list(X.iterateElements(order='point'))
        [0, 1, 2, -2, -1, 0]
        >>> list(X.iterateElements(order='feature'))
        [0, -2, 1, -1, 2, 0]
        >>> list(X.iterateElements(order='point', only=nonZero))
        [1, 2, -2, -1]
        >>> list(X.iterateElements(order='feature', only=positive))
        [1, 2]

        Keywords
        --------
        loop, for, for each, iteration, while, all, values
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
    @prepLog
    def replaceFeatureWithBinaryFeatures(self, featureToReplace, *,
                                         useLog=None): # pylint: disable=unused-argument
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
        >>> lst = [['a'], ['b'], ['c']]
        >>> X = nimble.data(lst, featureNames=['replace'])
        >>> replaced = X.replaceFeatureWithBinaryFeatures('replace')
        >>> replaced
        ['replace=a', 'replace=b', 'replace=c']
        >>> X
        <DataFrame 3pt x 3ft
             replace=a  replace=b  replace=c
           ┌────────────────────────────────
         0 │   1.000      0.000      0.000
         1 │   0.000      1.000      0.000
         2 │   0.000      0.000      1.000
        >

        Keywords
        --------
        dummy, dummies, dummy variables, indicator, one-hot encoding,
        1 hot encoding, one hot encoding, onehot encoding, categories,
        category, split, nominal, categorical, get_dummies
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

        return ftNames

    @limitedTo2D
    @prepLog
    def transformFeatureToIntegers(self, featureToConvert, *,
                                   useLog=None): # pylint: disable=unused-argument
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
        >>> lst = [[1, 'a', 1], [2, 'b', 2], [3, 'c', 3]]
        >>> featureNames = ['keep1', 'transform', 'keep2']
        >>> X = nimble.data(lst, featureNames=featureNames)
        >>> mapping = X.transformFeatureToIntegers('transform')
        >>> mapping
        {0: 'a', 1: 'b', 2: 'c'}
        >>> X
        <DataFrame 3pt x 3ft
             keep1  transform  keep2
           ┌────────────────────────
         0 │   1        0        1
         1 │   2        1        2
         2 │   3        2        3
        >

        Keywords
        --------
        map, categories, nominal, categorical, ordinal, unencode
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

        return {v: k for k, v in mapping.items()}

    @limitedTo2D
    @prepLog
    def transformElements(self, toTransform, points=None, features=None,
                          preserveZeros=False, skipNoneReturnValues=False, *,
                          useLog=None): # pylint: disable=unused-argument
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

        >>> X = nimble.ones(3, 3)
        >>> X.transformElements(lambda elem: elem + 1)
        >>> X
        <Matrix 3pt x 3ft
             0  1  2
           ┌────────
         0 │ 2  2  2
         1 │ 2  2  2
         2 │ 2  2  2
        >

        Transform while preserving zero values.

        >>> X = nimble.identity(3, returnType="Sparse")
        >>> X.transformElements(lambda elem: elem + 10,
        ...                     preserveZeros=True)
        >>> X
        <Sparse 3pt x 3ft
             0   1   2
           ┌───────────
         0 │ 11   0   0
         1 │  0  11   0
         2 │  0   0  11
        >

        Transforming a subset of points and features.

        >>> X = nimble.ones(4, 4, returnType="List")
        >>> X.transformElements(lambda elem: elem + 1, points=[0, 1],
        ...                     features=[0, 2])
        >>> X
        <List 4pt x 4ft
             0  1  2  3
           ┌───────────
         0 │ 2  1  2  1
         1 │ 2  1  2  1
         2 │ 1  1  1  1
         3 │ 1  1  1  1
        >

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
        >>> lst = [[1, 2, 3],
        ...        [4, 5, 6],
        ...        [7, 8, 9]]
        >>> dontSkip = nimble.data(lst)
        >>> dontSkip.transformElements(addTenToEvens)
        >>> dontSkip
        <Matrix 3pt x 3ft
               0       1       2
           ┌───────────────────────
         0 │         12.000
         1 │ 14.000          16.000
         2 │         18.000
        >
        >>> skip = nimble.data(lst)
        >>> skip.transformElements(addTenToEvens,
        ...                        skipNoneReturnValues=True)
        >>> skip
        <Matrix 3pt x 3ft
             0   1   2
           ┌───────────
         0 │  1  12   3
         1 │ 14   5  16
         2 │  7  18   9
        >

        Keywords
        --------
        apply, map, change, modify, replace, transformation,
        recalculate, alter, applymap
        """
        if points is not None:
            points = constructIndicesList(self, 'point', points)
        if features is not None:
            features = constructIndicesList(self, 'feature', features)

        transformer = validateElementFunction(toTransform, preserveZeros,
                                              skipNoneReturnValues,
                                              'toTransform')

        self._transform_implementation(transformer, points, features)


    @limitedTo2D
    @prepLog
    def calculateOnElements(self, toCalculate, points=None, features=None,
                            preserveZeros=False, skipNoneReturnValues=False,
                            outputType=None, *,
                            useLog=None): # pylint: disable=unused-argument
        """
        Apply a calculation to each element.

        Return a new object with a function or mapping applied to each
        element in this object or subset of points and features in this
        object.

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

        >>> X = nimble.ones(3, 3)
        >>> twos = X.calculateOnElements(lambda elem: elem + 1)
        >>> twos
        <Matrix 3pt x 3ft
             0  1  2
           ┌────────
         0 │ 2  2  2
         1 │ 2  2  2
         2 │ 2  2  2
        >

        Calculate while preserving zero values.

        >>> X = nimble.identity(3, returnType="Sparse")
        >>> addTen = X.calculateOnElements(lambda x: x + 10,
        ...                                preserveZeros=True)
        >>> addTen
        <Sparse 3pt x 3ft
             0   1   2
           ┌───────────
         0 │ 11   0   0
         1 │  0  11   0
         2 │  0   0  11
        >

        Calculate on a subset of points and features.

        >>> X = nimble.ones(4, 4, returnType="List")
        >>> calc = X.calculateOnElements(lambda elem: elem + 1,
        ...                              points=[0, 1],
        ...                              features=[0, 2])
        >>> calc
        <List 2pt x 2ft
             0  1
           ┌─────
         0 │ 2  2
         1 │ 2  2
        >

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
        >>> lst = [[1, 2, 3],
        ...        [4, 5, 6],
        ...        [7, 8, 9]]
        >>> X = nimble.data(lst)
        >>> dontSkip = X.calculateOnElements(addTenToEvens)
        >>> dontSkip
        <Matrix 3pt x 3ft
             0   1   2
           ┌───────────
         0 │     12
         1 │ 14      16
         2 │     18
        >
        >>> skip = X.calculateOnElements(addTenToEvens,
        ...                              skipNoneReturnValues=True)
        >>> skip
        <Matrix 3pt x 3ft
             0   1   2
           ┌───────────
         0 │  1  12   3
         1 │ 14   5  16
         2 │  7  18   9
        >

        Keywords
        --------
        apply, map, mapping, compute, measure, statistics, stats,
        applymap
        """
        calculator = validateElementFunction(toCalculate, preserveZeros,
                                             skipNoneReturnValues,
                                             'toCalculate')

        ret = self._calculate_backend(calculator, points, features,
                                      preserveZeros, outputType)

        return ret

    @limitedTo2D
    @prepLog
    def matchingElements(self, toMatch, points=None, features=None, *,
                         useLog=None): # pylint: disable=unused-argument
        """
        Identify values meeting the provided criteria.

        Return a new object with True at every location found to be a
        match, otherwise False. Common matching functions can be found
        in nimble's match module.

        Parameters
        ----------
        toMatch: value, function, query
            * value - elements equal to the value return True
            * function - accepts an element as its only argument and
              returns a boolean value to indicate if the element is a
              match
            * query - string in the format 'OPERATOR VALUE' representing
              a function (i.e "< 10", "== yes", or "is missing"). See
              ``nimble.match.QueryString`` for string requirements.
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

        See Also
        --------
        Points.matching, Features.matching, nimble.match, countElements

        Examples
        --------
        >>> from nimble import match
        >>> lst = [[1, -1, 1], [-3, 3, -3]]
        >>> X = nimble.data(lst)
        >>> isNegativeOne = X.matchingElements(-1)
        >>> isNegativeOne
        <Matrix 2pt x 3ft
               0      1      2
           ┌────────────────────
         0 │ False   True  False
         1 │ False  False  False
        >

        >>> from nimble import match
        >>> lst = [[1, -1, None], [None, 3, -3]]
        >>> X = nimble.data(lst)
        >>> isMissing = X.matchingElements(match.missing)
        >>> isMissing
        <Matrix 2pt x 3ft
               0      1      2
           ┌────────────────────
         0 │ False  False   True
         1 │  True  False  False
        >

        >>> from nimble import match
        >>> lst = [[1, -1, 1], [-3, 3, -3]]
        >>> X = nimble.data(lst)
        >>> isPositive = X.matchingElements(">0")
        >>> isPositive
        <Matrix 2pt x 3ft
               0      1      2
           ┌────────────────────
         0 │  True  False   True
         1 │ False   True  False
        >

        Keywords
        --------
        equivalent, identical, same, matches, equals, compare,
        comparison, same
        """
        matchArg = toMatch # preserve toMatch in original state for log
        if not callable(matchArg):
            try:
                func = QueryString(matchArg, elementQuery=True)
            except (InvalidArgumentValue, InvalidArgumentType) : 
                # if not a query string, element must equal matchArg
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

        ret = nimble.data(values, returnType=optType, treatAsMissing=[None],
                          useLog=False)

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
            * query - string in the format 'OPERATOR VALUE' representing
              a function (i.e "< 10", "== yes", or "is missing"). See
              ``nimble.match.QueryString`` for string requirements.

        Returns
        -------
        int

        See Also
        --------
        Points.count, Features.count, matchingElements, nimble.match

        Examples
        --------
        Using a python function.

        >>> def greaterThanZero(elem):
        ...     return elem > 0
        >>> X = nimble.identity(5)
        >>> numGreaterThanZero = X.countElements(greaterThanZero)
        >>> numGreaterThanZero
        5

        Using a string filter function.

        >>> numLessThanOne = X.countElements("<1")
        >>> numLessThanOne
        20

        Keywords
        --------
        number, counts, tally
        """
        if not hasattr(condition, '__call__'):
            condition = QueryString(condition, elementQuery=True)
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

        >>> X = nimble.identity(5)
        >>> unique = X.countUniqueElements()
        >>> unique
        {0: 20, 1: 5}

        Count for a subset of elements.

        >>> X = nimble.identity(5)
        >>> unique = X.countUniqueElements(points=0,
        ...                                   features=[0, 1, 2])
        >>> unique
        {0: 2, 1: 1}

        Keywords
        --------
        counts, tally, distinct, different, different
        """
        return self._countUnique_implementation(points, features)

    @limitedTo2D
    @prepLog
    def groupByFeature(self, by, calculate=None, countUniqueValueOnly=False, *,
                       useLog=None): # pylint: disable=unused-argument
        """
        Group data object by one or more features. This results in a 
        dictionary where the keys are the unique values of the target
        feature(s) and the values are the nimble base objects that
        correspond to the group.

        Parameters
        ----------
        by : int, str or list
            * int - the index of the feature to group by
            * str - the name of the feature to group by
            * list - indices or names of features to group by
        calculate : str, None
            The name of the statistical function to apply to each group. 
            If None, the default, no calculation will be applied.
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

        See Also
        --------
        plotFeatureAgainstFeature

        Examples
        --------
        >>> lst = [['ACC', 'Clemson', 15, 0],
        ...        ['SEC', 'Alabama', 14, 1],
        ...        ['Big 10', 'Ohio State', 13, 1],
        ...        ['Big 12', 'Oklahoma', 12, 2],
        ...        ['Independent', 'Notre Dame', 12, 1],
        ...        ['SEC', 'LSU', 10, 3],
        ...        ['SEC', 'Florida', 10, 3],
        ...        ['SEC', 'Georgia', 11, 3]]
        >>> ftNames = ['conference', 'team', 'wins', 'losses']  
        >>> top10 = nimble.data(lst, featureNames=ftNames)
        >>> groupByLosses = top10.groupByFeature('losses')
        >>> list(groupByLosses.keys())
        [0, 1, 2, 3]
        >>> groupByLosses[1]
        <DataFrame 3pt x 3ft
              conference     team     wins
           ┌──────────────────────────────
         0 │         SEC     Alabama   14
         1 │      Big 10  Ohio State   13
         2 │ Independent  Notre Dame   12
        >
        >>> groupByLosses[3]
        <DataFrame 3pt x 3ft
             conference    team   wins
           ┌──────────────────────────
         0 │    SEC          LSU   10
         1 │    SEC      Florida   10
         2 │    SEC      Georgia   11
        >
        
        Using the calculate parameter we can find the maximum of 
        other features within each group.
        
        >>> top10.groupByFeature('conference', calculate='max')  
        {'ACC': <DataFrame 1pt x 3ft
               team   wins   losses
             ┌─────────────────────
         max │       15.000  0.000
        >, 'SEC': <DataFrame 1pt x 3ft
               team   wins   losses
             ┌─────────────────────
         max │       14.000  3.000
        >, 'Big 10': <DataFrame 1pt x 3ft
               team   wins   losses
             ┌─────────────────────
         max │       13.000  1.000
        >, 'Big 12': <DataFrame 1pt x 3ft
               team   wins   losses
             ┌─────────────────────
         max │       12.000  2.000
        >, 'Independent': <DataFrame 1pt x 3ft
               team   wins   losses
             ┌─────────────────────
         max │       12.000  1.000
        >}
        
        Adding a new point to the data object with a missing value
        in a target feature will result in a new group with the key
        'NaN'.
        >>> lst.append(['', 'Auburn', 9, 4])
        >>> top10 = nimble.data(lst, featureNames=ftNames)
        >>> top10.groupByFeature('conference')
        {'ACC': <DataFrame 1pt x 3ft
               team   wins  losses
           ┌──────────────────────
         0 │ Clemson   15     0
        >, 'SEC': <DataFrame 4pt x 3ft
               team   wins  losses
           ┌──────────────────────
         0 │ Alabama   14     1
         1 │     LSU   10     3
         2 │ Florida   10     3
         3 │ Georgia   11     3
        >, 'Big 10': <DataFrame 1pt x 3ft
                team     wins  losses
           ┌─────────────────────────
         0 │ Ohio State   13     1
        >, 'Big 12': <DataFrame 1pt x 3ft
               team    wins  losses
           ┌───────────────────────
         0 │ Oklahoma   12     2
        >, 'Independent': <DataFrame 1pt x 3ft
                team     wins  losses
           ┌─────────────────────────
         0 │ Notre Dame   12     1
        >, 'NaN': <DataFrame 1pt x 3ft
              team   wins  losses
           ┌─────────────────────
         0 │ Auburn   9      4
        >}

        Keywords
        --------
        split, organize, categorize, groupby, variable, dimension,
        attribute, predictor
        """
        # Numbers coming from a float dtyped object that are equivalent to
        # ints are assumed to be int valued labels, and formatted as such.
        def prettyKey(val):
            undefined = ['', np.NaN, None, np.nan, np.NAN]
            if isinstance(val, str):
                return val
            if isinstance(val, numbers.Number):
                if val in undefined:
                    # check that the feature axis doesn't have that string.
                    if "NaN" not in list(self.features.copy(by)):
                        return "NaN"
                    return 'numpy.nan'
                iVal = int(val)
                return iVal if iVal == val else val
            return val

        def findKey1(point, by):#if by is a string or int
            return prettyKey(point[by])

        def findKey2(point, by):#if by is a list of string or a list of int
            return tuple(prettyKey(point[i]) for i in by)

        #if by is a list, then use findKey2; o.w. use findKey1
        if isinstance(by, (str, numbers.Number)):
            findKey = findKey1
        else:
            findKey = findKey2

        res = {}
        calc = {}
        accepted = acceptedStats
        
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
                    
            for obj in res.values():
                obj.features.delete(by, useLog=False)
                
            if calculate is not None:  
                cleanFuncName = validateInputString(calculate, accepted,
                                            'statistics') 
                toCall = _getStatsFunction(cleanFuncName)
                # different paradigm that puts accepted also into getStatsFunction 
                #cleanFuncName, toCall = _getStatsFunction(calculate)
                    
                for k in list(res.keys()):
                    
                    calc[k] = res[k].features._statisticsBackend(cleanFuncName, toCall)
                
                return calc
        return res

    @limitedTo2D
    def hashCode(self):
        """
        Returns a hash for this matrix.

        The hash is a number x in the range 0 <= x < 1 billion that
        should almost always change when the values of the matrix are
        changed by a substantive amount.

        Returns
        -------
        int

        Keywords
        --------
        id, identifier, unique, number, hash function
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

        See Also
        --------
        isIdentical

        Keywords
        --------
        equivalent, matches, equals, compare, comparison, same, similar
        """
        #first check to make sure they have the same dimensions
        if self._dims != other._dims:
            return False
        #now check if the hashes of each matrix are the same

        with self._treatAs2D():
            with other._treatAs2D():
                return self.hashCode() == other.hashCode()

    @prepLog
    def trainAndTestSets(self, testFraction, labels=None, randomOrder=True, *,
                         useLog=None): # pylint: disable=unused-argument
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
        labels : Base object, identifier, list of identifiers, or None
            A separate Base object containing the labels for this data
            or the feature axis name(s) or index(es) of the data labels 
            within this object.
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
            If ``labels`` is non-None, a length 4 tuple containing the
            training and testing data objects and the training a testing
            labels objects (trainX, trainY, testX, testY).

        See Also
        --------
        nimble.train, Points.extract, Features.extract

        Examples
        --------
        Returning a 2-tuple.

        >>> nimble.random.setSeed(42)
        >>> lst = [[1, 0, 0],
        ...        [0, 1, 0],
        ...        [0, 0, 1],
        ...        [1, 0, 0],
        ...        [0, 1, 0],
        ...        [0, 0, 1]]
        >>> ptNames = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> X = nimble.data(lst, pointNames=ptNames)
        >>> trainData, testData = X.trainAndTestSets(.34)
        >>> trainData
        <Matrix "train" 4pt x 3ft
             0  1  2
           ┌────────
         a │ 1  0  0
         b │ 0  1  0
         f │ 0  0  1
         c │ 0  0  1
        >
        >>> testData
        <Matrix "test" 2pt x 3ft
             0  1  2
           ┌────────
         e │ 0  1  0
         d │ 1  0  0
        >

        Returning a 4-tuple.

        >>> nimble.random.setSeed(42)
        >>> lst = [[1, 0, 0, 1],
        ...        [0, 1, 0, 2],
        ...        [0, 0, 1, 3],
        ...        [1, 0, 0, 1],
        ...        [0, 1, 0, 2],
        ...        [0, 0, 1, 3]]
        >>> ptNames = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> X = nimble.data(lst, pointNames=ptNames)
        >>> fourTuple = X.trainAndTestSets(.34, labels=3)
        >>> trainX, trainY = fourTuple[0], fourTuple[1]
        >>> testX, testY = fourTuple[2], fourTuple[3]
        >>> trainX
        <Matrix "trainX" 4pt x 3ft
             0  1  2
           ┌────────
         a │ 1  0  0
         b │ 0  1  0
         f │ 0  0  1
         c │ 0  0  1
        >
        >>> trainY
        <Matrix "trainY" 4pt x 1ft
             0
           ┌──
         a │ 1
         b │ 2
         f │ 3
         c │ 3
        >
        >>> testX
        <Matrix "testX" 2pt x 3ft
             0  1  2
           ┌────────
         e │ 0  1  0
         d │ 1  0  0
        >
        >>> testY
        <Matrix "testY" 2pt x 1ft
             0
           ┌──
         e │ 2
         d │ 1
        >

        Keywords
        --------
        split, data, prepare, training, testing, data points, divide,
        validation, splitting, train_test_split
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
                trainX.name = self.name + "_train"
                testX.name = self.name + "_test"
            else:
                trainX.name = "train"
                testX.name = "test"

        else:
            if isinstance(labels, Base):
                if len(labels.points) != len(self.points):
                    msg = 'labels must have the same number of points '
                    msg += f'({len(labels.points)}) as the calling object '
                    msg += f'({len(self.points)})'
                    raise InvalidArgumentValue(msg)
                try:
                    self._validateEqualNames('point', 'point', '', labels)
                except InvalidArgumentValue as e:
                    msg = 'labels and calling object pointNames must be equal'
                    raise InvalidArgumentValue(msg) from e 
                trainY = labels.points.copy(order[:splitIndex], useLog=False)
                testY = labels.points.copy(order[splitIndex:], useLog=False)
                
                if self.features._namesCreated() and \
                        labels.features._namesCreated():
                    labelNames = labels.features.getNames()
                    # filter names for ones that overlap with self
                    namesInBoth = [name for name in labelNames
                                   if self.features.hasName(name)]
                    if namesInBoth:
                        trainX.features.delete(namesInBoth, useLog=False)
                        testX.features.delete(namesInBoth, useLog=False)
            else:
                if len(self._dims) > 2:
                    msg = "labels parameter must be None when the data has "
                    msg += "more than two dimensions"
                    raise ImproperObjectAction(msg)

                # safety for empty objects
                toExtract = labels
                if testXSize == 0:
                    toExtract = []

                trainY = trainX.features.extract(toExtract, useLog=False)
                testY = testX.features.extract(toExtract, useLog=False)

            if self.name is not None:
                trainX.name = self.name + "_trainX"
                testX.name = self.name + "_testX"
                trainY.name = self.name + "_trainY"
                testY.name = self.name + "_testY"
            else:
                trainX.name = "trainX"
                testX.name = "testX"
                trainY.name = "trainY"
                testY.name = "testY"

            ret = trainX, trainY, testX, testY

        return ret

    ########################################
    ########################################
    ###   Functions related to reports   ###
    ########################################
    ########################################

    def report(self, *, useLog=None):
        """
        Report containing information regarding the data in this object.

        Produce a report, as a nimble List object, containing summary
        information about the data in this object. Includes the total
        number of values in the object, the number of points and number
        of features (or dimensions for high dimension data), the
        proportion of missing values, and the proportion of zero values.

        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        See Also
        --------
        Features.report

        Keywords
        --------
        summary, information, description, analyze, statistics, stats,
        columns, info, information, describe, about
        """
        results = []
        fnames = []
        fnames.append('Values')
        results.append(np.prod(self._dims))
        if len(self._dims) > 2:
            fnames.append('Dimensions')
            results.append(' x '.join(map(str, self._dims)))
        else:
            fnames.extend(['Points', 'Features'])
            results.extend([self.shape[0], self.shape[1]])

        funcs = (nimble.calculate.proportionZero,
                 nimble.calculate.proportionMissing)

        with self._treatAs2D():
            for func in funcs:
                fnames.append(func.__name__)
                calc = sum(self.features.calculate(func, useLog=False)
                           / len(self.features))
                results.append(calc)

        report = nimble.data(results, featureNames=fnames,
                             returnType=self.getTypeString(), useLog=False)

        handleLogging(useLog, 'report', "summary", str(report))

        return report

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

        See Also
        --------
        isApproximatelyEqual

        Keywords
        --------
        equality, match, matches, equals, compare, comparison, same
        """
        if not isinstance(other, Base):
            return False
        if self._dims != other._dims:
            return False
        if not self._equalFeatureNames(other):
            return False
        if not self._equalPointNames(other):
            return False

        return self._isIdentical_implementation(other)

    def save(self, outPath, fileFormat=None, includeNames=True):
        """
        Write the data in this object to a file in the specified format.

        Can write the data object to a csv, mtx, hdf5, or pickle file.
        If the ``outPath`` does not have an extension and ``fileFormat``
        is None, this will default to writing a csv file.

        Parameters
        ----------
        outPath : str
            The location to write the output file.
        fileFormat : str
            The formatting of the file to write. If None, the extension
            of outPath determines the format. Otherwise can be 'csv',
            'mtx', 'hdf5', 'h5', 'pickle', 'p', or 'pkl'.
        includeNames : bool
            Indicates whether the file will embed the point and feature
            names into the file. The format of the embedding is
            dependent on the format of the file. Names will always be
            included for pickle files.

        See Also
        --------
        nimble.data

        Keywords
        --------
        save, output, disk, location, path
        """
        if len(self.points) == 0 or len(self.features) == 0:
            msg = "We do not allow writing to file when an object has "
            msg += "0 points or features"
            raise ImproperObjectAction(msg)

        # if format is not specified, we fall back on the extension in outPath
        if fileFormat is None:
            extension = os.path.splitext(outPath)[1]
            if extension:
                fileFormat = extension[1:].lower() # remove leading '.'
            else:
                fileFormat = 'csv'

        acceptedFormats = ['csv', 'mtx', 'hdf5', 'h5', 'pickle', 'p', 'pkl']
        if fileFormat not in acceptedFormats:
            accepted = prettyListString(acceptedFormats, useAnd=True,
                                        itemStr=quoteStrings)
            msg = f"Unrecognized file format. Accepted types are {accepted}"
            msg += "It may either be input as the format parameter, or as the "
            msg += "extension in the outPath"
            raise InvalidArgumentValue(msg)

        if fileFormat in ['pickle', 'p', 'pkl']:
            if not cloudpickle.nimbleAccessible():
                msg = "To pickle nimble objects, cloudpickle must be installed"
                raise PackageException(msg)

            with open(outPath, 'wb') as file:
                return cloudpickle.dump(self, file)

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
            return self._saveHDF_implementation(outPath, includePointNames)
        if len(self._dims) > 2:
            msg = 'Data with more than two dimensions can only be written '
            msg += 'to .hdf5 or .h5 formats otherwise the dimensionality '
            msg += 'would be lost'
            raise InvalidArgumentValue(msg)
        if fileFormat.lower() == "csv":
            return self._saveCSV_implementation(
                outPath, includePointNames, includeFeatureNames)

        return self._saveMTX_implementation(
            outPath, includePointNames, includeFeatureNames)


    def _writeFeatureNamesToCSV(self, openFile, includePointNames):
        fnames = list(map(csvCommaFormat, self.features.getNames()))
        if includePointNames:
            fnames.insert(0, 'pointNames')
        fnamesLine = ','.join(fnames)
        fnamesLine += '\n'
        openFile.write(fnamesLine)

    def _saveHDF_implementation(self, outPath, includePointNames):
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

    def getTypeString(self):
        """
        The nimble Type of this object.

        A string representing the non-abstract type of this
        object (e.g. Matrix, Sparse, etc.) that can be passed to
        nimble.data() function to create a new object of the same type.

        Returns
        -------
        str

        See Also
        --------
        copy, nimble.data

        Keywords
        --------
        kind, class
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
        >>> lst = [[4132, 41, 'management', 50000, 'm'],
        ...        [4434, 26, 'sales', 26000, 'm'],
        ...        [4331, 26, 'administration', 28000, 'f'],
        ...        [4211, 45, 'sales', 33000, 'm'],
        ...        [4344, 45, 'accounting', 43500, 'f']]
        >>> pointNames = ['michael', 'jim', 'pam', 'dwight', 'angela']
        >>> featureNames = ['id', 'age', 'department', 'salary',
        ...                 'gender']
        >>> office = nimble.data(lst, pointNames=pointNames, 
        ...                      featureNames=featureNames)

        Get a single value.

        >>> office['michael', 'age']
        41
        >>> office[0,1]
        41
        >>> office['michael', 1]
        41

        Get based on points only.

        >>> office['pam', :]
        <DataFrame 1pt x 5ft
                id   age    department    salary  gender
             ┌──────────────────────────────────────────
         pam │ 4331   26  administration  28000     f
        >
        >>> office[[3, 1], :]
        <DataFrame 2pt x 5ft
                   id   age  department  salary  gender
                ┌──────────────────────────────────────
         dwight │ 4211   45    sales     33000     m
            jim │ 4434   26    sales     26000     m
        >

        *Note: retains list order; index 3 placed before index 1*

        >>> office[1:4, :]
        <DataFrame 4pt x 5ft
                   id   age    department    salary  gender
                ┌──────────────────────────────────────────
            jim │ 4434   26           sales  26000     m
            pam │ 4331   26  administration  28000     f
         dwight │ 4211   45           sales  33000     m
         angela │ 4344   45      accounting  43500     f
        >

        *Note: slices are inclusive; index 4 ('gender') was included*

        Get based on features only.

        >>> office[:, 2]
        <DataFrame 5pt x 1ft
                     department
                 ┌───────────────
         michael │     management
             jim │          sales
             pam │ administration
          dwight │          sales
          angela │     accounting
        >
        >>> office[:, ['gender', 'age']]
        <DataFrame 5pt x 2ft
                   gender  age
                 ┌────────────
         michael │   m      41
             jim │   m      26
             pam │   f      26
          dwight │   m      45
          angela │   f      45
        >

        *Note: retains list order; 'gender' placed before 'age'*

        >>> office[:, 'department':'salary']
        <DataFrame 5pt x 2ft
                     department    salary
                 ┌───────────────────────
         michael │     management  50000
             jim │          sales  26000
             pam │ administration  28000
          dwight │          sales  33000
          angela │     accounting  43500
        >


        *Note: slices are inclusive; 'salary' was included*

        Get based on points and features.

        >>> office[['pam', 'angela'], [3,2]]
        <DataFrame 2pt x 2ft
                  salary    department
                ┌───────────────────────
            pam │ 28000   administration
         angela │ 43500       accounting
        >

        *Note: list orders retained; 'pam' precedes 'angela' and index 3
        ('salary') precedes index 2 ('department')*

        >>> office[:2, 'age']
        <DataFrame 3pt x 1ft
                   age
                 ┌────
         michael │  41
             jim │  26
             pam │  26
        >

        *Note: slices are inclusive; index 2 ('pam') was included*
        """
        # if axis names do not exist provide a list to 
        # compare with. calling ._getNames() creates default names
        # and breaks some tests that rely on the absence of names
        pointsNamesList = []
        featuresNamesList = []
        if self._points._namesCreated():
            pointsNamesList = self._points._getNames()
        if self._features._namesCreated():
            featuresNamesList = self._features._getNames()
            
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
            elif key in pointsNamesList and key not in featuresNamesList:
                    x = key
                    y = slice(None)
                    key = (x, y)
            elif key in featuresNamesList and key not in pointsNamesList:
                    y = key
                    x = slice(None)
                    key = (x, y)
            elif key in pointsNamesList and key in featuresNamesList:
                msg = f"Using '{key}' as an identifier is ambiguous as" 
                msg += " it is both a point and feature name."
                raise InvalidArgumentType(msg)    
            else:
                msg = "Must include both a point and feature index; or, "
                msg += "if this is a one-dimensional data object, a single index "
                msg += "for the axis with length greater than 1."
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
    
    @limitedTo2D
    def __setitem__(self, key, value):
        """
        Set a single item in the data structure.

        Parameters
        ----------
        key : tuple, slice, or combination
            Describes the subset of data to set.
        value : single value, list, or ndarray
            The new values to set. Can be a single value to apply to all selected points
            and features, or an array/list of values matching the size of the selection.

        Examples
        --------
        >>> lst = [[4132, 41, 'management'],
        ...        [4434, 26, 'sales'],
        ...        [4331, 26, 'administration'],
        ...        [4211, 45, 'sales']]
        >>> pointNames = ['michael', 'jim', 'pam', 'dwight']
        >>> featureNames = ['id', 'age', 'department']
        >>> office = nimble.data(lst, pointNames=pointNames,
        ...                      featureNames=featureNames)
        >>> office
        <DataFrame 4pt x 3ft
                    id   age    department
                 ┌──────────────────────────
         michael │ 4132   41      management
             jim │ 4434   26           sales
             pam │ 4331   26  administration
          dwight │ 4211   45           sales
        >
        >>> office['michael', 'age'] = 42
        >>> office['michael']
        <DataFrame 1pt x 3ft
                    id   age  department
                 ┌──────────────────────
         michael │ 4132   42  management
        >
        >>> office[3,1] = 85
        >>> office[1, 'department'] = 'management'
        >>> office
        <DataFrame 4pt x 3ft
                    id   age    department
                 ┌──────────────────────────
         michael │ 4132   42      management
             jim │ 4434   26      management
             pam │ 4331   26  administration
          dwight │ 4211   85           sales
        >
        
        """
        
        # If key is not a tuple, convert it to a tuple
        if not isinstance(key, tuple):
            key = (key, slice(None))  # Assuming the first dimension is 'x' and the second is 'y'
        
        # Extract x and y indices from the key
        x, y = key
        
        # Validate single or multiple indices for 'x'
        single_x = isinstance(x, (int, float, str, np.integer))
        if single_x:
            x = self.points._getIndex(x, allowFloats=True)
        else:
            x = self.points._processMultiple(x)
        
        # Validate single or multiple indices for 'y'
        single_y = isinstance(y, (int, float, str, np.integer))
        if single_y:
            y = self.features._getIndex(y, allowFloats=True)
        else:
            y = self.features._processMultiple(y)
        
        # Perform setting operation based on single or multiple indices
        if single_x and single_y:
            self._setitem_implementation(x, y, value)
        else:
            for i in x:
                for j in y:
                    self._setitem_implementation(i, j, value)
        
    
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

        See Also
        --------
        view

        Keywords
        --------
        data point, instance, observation, value, locked, read only,
        immutable, unchangeable, uneditable
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

        See Also
        --------
        view

        Keywords
        --------
        variables, dimensions, attributes, predictors, locked,
        read only, immutable
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

        Keywords
        --------
        read only, locked, immutable, unchangeable, uneditable
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

        if len(self._dims) > 2:
            if featureStart != 0 or featureEnd != len(self.features):
                msg = "feature limited views are not allowed for data with "
                msg += "more than two dimensions."
                raise ImproperObjectAction(msg)

        return self._view_implementation(pointStart, pointEnd,
                                         featureStart, featureEnd,
                                         dropDimension)

    def checkInvariants(self, level=2):
        """
        Check the integrity of the data.

        Validate this object with respect to the limitations and
        invariants that our objects enforce.

        Parameters
        ----------
        level : int
            The extent to which to validate the data. Higher numbers are more
            stringent checks. Allowed: 0 to 2 inclusive.
        """
        if self.points._namesCreated():
            assert self.shape[0] == len(self.points.namesInverse)
        if self.features._namesCreated():
            assert self.shape[1] == len(self.features.namesInverse)

        if level > 0:
            def checkAxisNameState(axis):
                if axis._namesCreated():
                    for i, key in enumerate(axis.namesInverse):
                        if key is not None:
                            assert axis.names[key] == i
                            assert axis.namesInverse[i] == key
                        else:
                            assert axis.namesInverse[i] is None
                            assert i not in axis.names.values()
                else:
                    assert axis.names is None
                    assert axis.namesInverse is None

            checkAxisNameState(self.points)
            checkAxisNameState(self.features)

        self._checkInvariants_implementation(level)

    def containsZero(self):
        """
        Evaluate if the object contains one or more zero values.

        True if there is a value that is equal to integer 0
        contained in this object, otherwise False.

        Returns
        -------
        bool

        See Also
        --------
        countElements, matchingElements, nimble.match.zero

        Keywords
        --------
        zeros, sparse
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

    def toString(self, points=None, features=None, lineLimit=30,
                 lineWidthLimit=79,  sigDigits=3, columnWidthLimit=19,
                 includePointNames=True, includeFeatureNames=True,
                 indent='', quoteNames=False):
        """
        A string representation of this object.

        For objects exceeding the ``maxWidth`` and/or ``maxHeight``,
        the string will truncate the output using various placeholders
        to indicate that some data was removed. For width and height,
        the data removed will be from the center printing only the data
        for the first few and last few columns and rows, respectively.

        Parameters
        ----------
        points : range, int, None
            A filter for specifying a specific range or number of points.
            If None, lineLimit will determine what is shown.
        features : range, int, None
            A filter for specifying a specific range or number of features.
            If None, lineWidthLimit will determine what is shown.
        lineLimit : int
            A bound on the maximum number of lines allowed for the
            output.
        lineWidthLimit : int
            A bound on the maximum number of characters allowed on each
            line of the output.
        sigDigits : int
            The number of significant digits to display in the output.
        columnWidthLimit : int
            A bound on the maximum number of characters allowed for the
            width of single column (feature) in each line.
        includePointNames : bool
            Used to control whether the point names are printed alongside the
            data in the points rows. If set to 'False' the indices
            of the points will be displayed instead of the names.
        includeFeatureNames : bool
            Used to control whether the feature names are printed alongside
            the data in the features column. If set to 'False' the indices
            of the features will be displayed instead of the names.
        indent : str
            The string to use as indentation.
        quoteNames : bool
            Whether single quotes are included around the axis names.

        See Also
        --------
        show

        Keywords
        --------
        text, write, display, stringify
        """
        # setup a bundle of fixed constants
        colSep = '  '
        colHold = '\u2500\u2500'
        rowHold = '\u2502'
        pnameSep = '\u2502'
        pnameSepColSep = ' ' # special case of colSep for the gaps left and right of pnameSep
        fnameSep = '\u2500'
        corner = '\u250C'
        nameHolder = '\u2500'
        dataOrientation = 'center'
        dataRelativeOrientation = 'rjust'
        pNameOrientation = 'rjust'
        fNameOrientation = 'center'
        holderOrientation = 'center'

        # Resolve default values for points, features, and related
        # variables
        pointsLen = len(self.points)
        featuresLen = len(self.features)
        msg = "The {} argument may only be a range, an int, or None"
        if isinstance(points, range):
            points = self._validateAndTruncatePrintTarget(points, pointsLen)
            numPoints = len(points)
            pRange = points
        elif isinstance(points, int):
            numPoints = min(points, pointsLen)
            pRange = range(pointsLen)
        elif points is None:
            numPoints = pointsLen
            pRange = range(pointsLen)
        else:
            raise InvalidArgumentType(msg.format("points"))

        if isinstance(features, range):
            features = self._validateAndTruncatePrintTarget(features, featuresLen)
        elif isinstance(features, int):
            pass # no action needed
        elif features is None:
            features = len(self.features)
        else:
            raise InvalidArgumentType(msg.format("features"))

        if lineLimit is None:
            maxDataRows = numPoints
        else:
            maxDataRows = min(lineLimit - 2, numPoints)
            if maxDataRows <= 1 and numPoints > maxDataRows:
                msg = 'The minimum lineLimit for this data is '
                msg += str(2 + min(numPoints, 2))
                raise InvalidArgumentValue(msg)

        if lineWidthLimit is None:
            lineWidthLimit = float('inf')
        maxDataWidth = lineWidthLimit - len(indent)
        # we want to only prepare pointNames if "include=True"
        # else table should include data but no pointnames + space for it
        pnames, pnamesWidth = self._arrangePointNames(pRange,
            maxDataRows, columnWidthLimit, rowHold, nameHolder, includePointNames,
            quoteNames)
        # The available space for the data is reduced by the width of the
        # pnames, a pnames column separator, the pnames separator, and another
        # pnames column separator
        maxDataWidth -= pnamesWidth + 2 * len(pnameSepColSep) + len(pnameSep)

        # Set up data values to fit in the available space
        with self._treatAs2D():
            dataTable, colWidths, fnames = self._arrangeDataWithLimits(
                numPoints, pRange, features,
                maxDataWidth, maxDataRows, sigDigits, columnWidthLimit, colSep,
                colHold, rowHold, nameHolder, includeFeatureNames, quoteNames)
        # combine names into finalized table
        finalTable, finalWidths = arrangeFinalTable(
            pnames, pnamesWidth, dataTable, colWidths, fnames, pnameSep)
        # set up output string
        out = ""
        for i, row in enumerate(finalTable):
            for j, val in enumerate(row):
                # point names
                if j == 0:
                    padded = getattr(val, pNameOrientation)(max(finalWidths[j]))
                # feature Names
                elif i == 0:
                    padded = getattr(val, fNameOrientation)(max(finalWidths[j]))
                # seperators between pnames and values (already fully filling
                # the available space)
                elif j == 1 and val == rowHold:
                    padded = val
                # row placeholder: centered between the width of the adjacent
                # values in the column
                elif val == rowHold:
                    # finalTable is being padded as we go, have to strip the
                    # previously added whitespace from the row above
                    aboveVal = (finalTable[i-1][j]).strip()
                    # in certain height limited conditions, the holder row may
                    # be the last, so there is no value below. Only check if
                    # we're sure there's another row.
                    belowVal = aboveVal
                    if i < len(finalTable)-1:
                        belowVal = finalTable[i+1][j]
                    valWidthMax = max(len(aboveVal), len(belowVal))
                    # if we are in a column with all missing values, align to
                    # center of the column
                    if valWidthMax == 0:
                        valWidthMax = max(finalWidths[j])
                    valPadded = getattr(val, holderOrientation)(valWidthMax)
                    padded = getattr(valPadded, dataRelativeOrientation)(finalWidths[j][1])
                    padded = getattr(padded, dataOrientation)(max(finalWidths[j]))
#                    padded = (' ' * max((finalWidths[j])) - valWidthMax) + valPadded
                elif val == colHold:
                    padded = getattr(val, holderOrientation)(max(finalWidths[j]))
                # normal values
                else:
                    padded = getattr(val, dataRelativeOrientation)(finalWidths[j][1])
                    padded = getattr(padded, dataOrientation)(max(finalWidths[j]))
                row[j] = padded
            line = indent + pnameSepColSep.join(finalTable[i][:3]) + colSep + colSep.join(finalTable[i][3:])
            out += line.rstrip() + "\n"
            if i == 0: # add separator row
                out += indent
                # spaces for each character of point IDs max plus column seperator
                blank =  (' ' * (max(finalWidths[0]))) + pnameSepColSep
                ftWidths = [max(w) for w in finalWidths[2:] if w]
                sepStr = (fnameSep * len(colSep)).join([fnameSep * w for w in ftWidths])
                out += blank + corner + fnameSep + sepStr + '\n'

        return out

    def _validateAndTruncatePrintTarget(self, target, axisLength):
        """
        Validate that given target argument for printing is
        a contiguous range of indices within the length of
        the axis. If it extends out from the available indices,
        return a truncated version.
        """

        if target.step != 1:
            msg = "range inputs for printing must have a step of 1, not "
            msg += f"{target.step}"

        desired = (target.start, target.stop)
        if desired[0] < 0:
            desired[0] = 0
        if desired[1] > axisLength:
            desired[1] = axisLength

        return range(desired[0], desired[1])

    def _getNotebookTerminalSize(self, _):
        """
        Aspirational helper for detecting the available space for printing
        within a notebook. However, no viable method has been found to
        do so. We therefore return a handpicked default, chosen to be
        an acceptable max width even on a 1360x768 laptop screen.
        """
        # Sane default
        return (117, 30)


    def _show(self, description=None, points=None, features=None,
              lineLimit='automatic', lineWidthLimit='automatic',
              sigDigits=3, columnWidthLimit='automatic',
              includePointNames=True, includeFeatureNames=True,
              includeObjectName=True, indent='', quoteNames=False):

        # Check if we're in IPython / a Notebook
        if IPython.nimbleAccessible():
            # even if IPython is accessible, it's only operational if
            # we are able to grab the currently used shell
            shell = IPython.core.getipython.get_ipython()
        else:
            shell = None

        # Resolve the 'automatic' values for lineWidthLimit and lineLimit
        if shell is not None:
            terminalSize = self._getNotebookTerminalSize(shell)
        else:
            terminalSize = shutil.get_terminal_size()

        if features is None and lineWidthLimit == 'automatic':
            lineWidthLimit = max(79, terminalSize[0] - 1)
        else:
            lineWidthLimit = None

        if lineLimit == 'automatic':
            lineLimit = max(30, terminalSize[1] - 1)
        if lineLimit is not None:
            # subtract lines for data details and last line
            lineLimit -= 2

        ret = ''
        if description is not None:
            ret += indent + description + '\n'
            if lineLimit is not None:
                lineLimit -= 1

        if columnWidthLimit == 'automatic':
            if isinstance(lineWidthLimit, int):
                # If we just have point indices can know how much space they'll take up
                if not self.points.names or len(self.points.names) == None:
                    pNamesEst = math.ceil(math.log(len(self.points), 10))
                    columnWidthLimit = ((lineWidthLimit-pNamesEst) // len(self.features)) + 7
                # For assigned names, we just let it take up another column's worth
                else:
                    columnWidthLimit = (lineWidthLimit // (len(self.features)+1)) + 7
                if columnWidthLimit < 8:
                    columnWidthLimit = 8 # lower limit for dynamic column width
            else:
                columnWidthLimit = 19 # sane default, matches toString

        if includeObjectName and self.name is not None:
            ret += f'"{self._name}" '
        if len(self._dims) > 2:
            ret += " x ".join(map(str, self._dims))
            ret += " dimensions encoded as "
        ret += str(len(self.points)) + "pt x "
        ret += str(len(self.features)) + "ft"
        ret += '\n'

        ret += self.toString(points, features, lineLimit, lineWidthLimit,
            sigDigits, columnWidthLimit, includePointNames=includePointNames,
            includeFeatureNames=includeFeatureNames, indent=indent,
            quoteNames=quoteNames)

        return ret

    def __str__(self):
        return self._show()

    def __repr__(self):
        ret = f"<{self.getTypeString()} "
        indent = ' '
        # remove leading and trailing newlines
        ret += self._show(indent=indent)
        # if self.path is not None:
        #     ret += indent + "path=" + self.path + '>'
        # else:
        ret += '>'

        return ret
    def show(self, description=None, points=None, features=None,
             lineLimit='automatic', lineWidthLimit='automatic', sigDigits=3,
             columnWidthLimit='automatic', includePointNames=True,
             includeFeatureNames=True, includeObjectName=True):
        """
        A printed representation of the data.

        Method to simplify printing a representation of this data
        object, with some context. Prior to the names and data, it
        additionally prints a description provided by the user,
        (optionally) this object's name attribute, and the number of
        points and features that are in the data.

        Parameters
        ----------
        description : str, None
            Printed as-is before the rest of the output, unless None.
        points : range, int, None
            A filter for which points to include in the output. If given
            a range, only those those within the range will be output. An
            int will allow up to that amount of points to be shown,
            working inwards from the first and last point. If None (default),
            there is no specification, and points will be chosen according to
            lineLimit.
        features : range, int, None
            A filter for which features to include in the output. If given
            a range, only those those within the range will be output. An
            int will allow up to that int's amount of features to be shown,
            working inwards from the first and last feature. If None (default),
            there is no specification, and features will be chosen according to
            lineWidthLimit.
        lineLimit : 'automatic', int, None
            A bound on the maximum number of lines allowed for the
            output.  The default, 'automatic', enforces a minimum
            height of 30 lines but expands dynamically when the height
            of the terminal is greater than 30, or will be ignored if
            the points parameter is specified. None will disable the
            bound on height.
        lineWidthLimit : 'automatic', int, None
            A bound on the maximum number of characters allowed on each line
            of the output. In CPython, the default value 'automatic', resolves
            to a width of 79 but expands dynamically when the width of the
            terminal is greater than 80 characters. In IPython / Notebooks
            'automatic' always resolves to 117. This limit will be ignored if
            the features parameter is specified. A value of None will allow an
            unbounded width.
        sigDigits : int
            The number of significant digits to display in the output.
        columnWidthLimit : int
            A bound on the maximum number of characters allowed for the
            width of single printed column (feature) in each line.
            If the column text is too long for the set bound, 3 characters
            will be used up for the ellipses during truncation.
        includePointNames : bool
            Used to control whether the point names are printed alongside the
            data in the points rows. If set to 'False' the indices
            of the points will be displayed instead of the names.
        includeFeatureNames : bool
            Used to control whether the feature names are printed alongside 
            the data in the features column. If set to 'False' the indices
            of the features will be displayed instead of the names.
        includeObjectName : bool
            True will include printing of the object's ``name``
            attribute, False will not print the object's name.
            
        Keywords
        --------
        print, representation, visualize, out, stdio, visualize, output,
        write, text, repr, represent, display, terminal
        """
        print(self._show(description, points, features, lineLimit,
                         lineWidthLimit, sigDigits, columnWidthLimit,
                         includePointNames, includeFeatureNames,
                         includeObjectName))

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
            A label for the y axis. If True, the label will be "Point
            Values".
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``matshow`` function.

        See Also
        --------
        matplotlib.pyplot.matshow

        Keywords
        --------
        chart, figure, image, graphics, visualization, matrix, map
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
                                figureID=None, title=True, xAxisLabel=True,
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
            the same ``figureID`` before being shown.
        figureID : hashable, None
            A new figure will be generated for None or a new id,
            otherwise the figure with that id will be activated to draw
            the plot on the existing figure.
        title : str, None
            The title of the plot. If True, the title will identify the
            feature presented in the distribution.
        xAxisLabel : str
            A label for the x axis. If True, the label will be "Values".
        yAxisLabel : str
            A label for the y axis. If True, the label will be "Number
            of Values".
        xMin : int, float
            The minimum value shown on the x axis of the resultant plot.
        xMax : int, float
            The maximum value shown on the x axis of the resultant plot.
        kwargs
            Any keyword arguments accepted by matplotlib.pyplot's
            ``hist`` function.

        See Also
        --------
        matplotlib.pyplot.hist

        Keywords
        --------
        histogram, chart, figure, image, graphics, kde, density,
        probability density function, visualization
        """
        if isinstance(feature, list):     
            for listItem in feature:
                self._plotFeatureDistribution(listItem, outPath, show, figureID,
                                        title, xAxisLabel, yAxisLabel, xMin,
                                        xMax, **kwargs)
        else: 
            self._plotFeatureDistribution(feature, outPath, show, figureID,
                                          title, xAxisLabel, yAxisLabel, xMin,
                                          xMax, **kwargs)

    def _plotFeatureDistribution(self, feature, outPath, show, figureID,
                                 title, xAxisLabel, yAxisLabel, xMin, xMax,
                                 **kwargs):
        return self._plotDistribution('feature', feature, outPath, show,
                                      figureID, title, xAxisLabel,
                                      yAxisLabel, xMin, xMax, **kwargs)

    @pyplotRequired
    def _plotDistribution(self, axis, identifier, outPath, show, figureID,
                          title, xAxisLabel, yAxisLabel, xMin, xMax, **kwargs):
        _, ax = plotFigureHandling(figureID)
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
              
        valMax = max(toPlot)
        if type(valMax) in [float, int]:
            if 'bins' not in kwargs:
                quartiles = nimble.calculate.quartiles(toPlot)
                IQR = quartiles[2] - quartiles[0]
                binWidth = (2 * IQR) / (len(toPlot) ** (1. / 3))
                # TODO: replace with calculate points after it subsumes
                # pointStatistics?
                valMin = min(toPlot)
                if binWidth == 0:
                    binCount = 1
                else:
                    # we must convert to int, in some versions of numpy, the helper
                    # functions matplotlib calls will require it.
                    binCount = int(math.ceil((valMax - valMin) / binWidth))
                kwargs['bins'] = binCount
        else:
            toPlot = sorted(list(toPlot))
          
        ax.hist(toPlot, **kwargs)
        if 'label' in kwargs:
            ax.legend()

        plotAxisLimits(ax)
        plotAxisLabels(ax, xAxisLabel, "Values", yAxisLabel,
                       "Number of Values")

        plotOutput(outPath, show)

    @limitedTo2D
    def plotFeatureAgainstFeatureRollingAverage(
            self, x, y, sampleSizeForAverage=20, groupByFeature=None,
            trend=None, outPath=None, show=True, figureID=None, title=True,
            xAxisLabel=True, yAxisLabel=True, xMin=None, xMax=None, yMin=None,
            yMax=None, **kwargs):
        """
        A rolling average of the pairwise combination of feature values.

        The points in the plot have coordinates defined by taking the
        average of pairwise values from ``x`` and ``y`` within a rolling
        window of length ``sampleSizeForAverage`` according to an
        ordering defined by the sorted values of ``x``. Keyword
        arguments for matplotlib.pyplot's ``plot`` function can be
        passed for further customization of the plot.

        Control over the visual width of the both axes is given, with
        the warning that user specified values can obscure data that
        would otherwise be plotted given default inputs.

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
        groupByFeature : identifier, None
            An optional index or name of the feature that divides the x
            and y values into groups. Each group will be plotted with a
            different color and a legend will be added. To use custom
            colors provide a dictionary of labels mapped to matplotlib
            color values as the 'color' keyword argument.
        trend : str, None
            Specify a trendline type. Currently "linear" is the only
            supported string option. None will not add a trendline.
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
        title : str, None
            The title of the plot. If True, the title will identify the
            two features presented in the plot.
        xAxisLabel : str
            A label for the x axis. If True, the label will be "Values".
        yAxisLabel : str
            A label for the y axis. If True, the label will be "Number
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

        See Also
        --------
        matplotlib.pyplot.plot, matplotlib.colors, matplotlib.markers,
        plotFeatureAgainstFeature

        Keywords
        --------
        graph, scatter, line, chart, relationship, figure, image,
        graphics, visualization
        """
        self._plotFeatureAgainstFeature(
            x, y, groupByFeature, sampleSizeForAverage, trend, outPath, show,
            figureID, title, xAxisLabel, yAxisLabel, xMin, xMax, yMin, yMax,
            **kwargs)

    @limitedTo2D
    def plotFeatureAgainstFeature(
            self, x, y, groupByFeature=None, trend=None, outPath=None,
            show=True, figureID=None, title=True, xAxisLabel=True,
            yAxisLabel=True, xMin=None, xMax=None, yMin=None, yMax=None,
            **kwargs):
        """
        A plot of the pairwise combination of feature values.

        Control over the width of the both axes is given, with the
        warning that user specified values can obscure data that would
        otherwise be plotted given default inputs. Keyword arguments for
        matplotlib.pyplot's ``plot`` function can be passed for further
        customization of the plot.

        Parameters
        ----------
        x : identifier
            The index or name of the feature from which we draw x-axis
            coordinates.
        y : identifier
            The index or name of the feature from which we draw y-axis
            coordinates.
        groupByFeature : identifier, None
            An optional index or name of the feature that divides the x
            and y values into groups. Each group will be plotted with a
            different color and a legend will be added. To use custom
            colors provide a dictionary of labels mapped to matplotlib
            color values as the 'color' keyword argument.
        trend : str, None
            Specify a trendline type. Currently "linear" is the only
            supported string option. None will not add a trendline.
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
        title : str, None
            The title of the plot. If True, the title will identify the
            two features presented in the plot.
        xAxisLabel : str
            A label for the x axis. If True, the label will be the x
            feature name or index.
        yAxisLabel : str
            A label for the y axis. If True, the label will be the y
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

        See Also
        --------
        matplotlib.pyplot.plot, matplotlib.colors, matplotlib.markers,
        plotFeatureAgainstFeatureRollingAverage

        Keywords
        --------
        graph, scatter, line, chart, relationship, figure, image,
        graphics, visualization
        """
        self._plotFeatureAgainstFeature(
            x, y, groupByFeature, None, trend, outPath, show, figureID,
            title, xAxisLabel, yAxisLabel, xMin, xMax, yMin, yMax, **kwargs)


    def _plotFeatureAgainstFeature(
            self, x, y, groupByFeature, sampleSizeForAverage, trend, outPath,
            show, figureID, title, xAxisLabel, yAxisLabel, xMin, xMax, yMin,
            yMax, **kwargs):
        if groupByFeature is None:
            self._plotCross(
                x, 'feature', y, 'feature', sampleSizeForAverage, trend,
                outPath, show, figureID, title, xAxisLabel, yAxisLabel, xMin,
                xMax, yMin, yMax, **kwargs)
        else:
            grouped = self.groupByFeature(groupByFeature)
            labels = list(grouped.keys())
            lastLabel = labels[-1]
            if 'color' in kwargs:
                # need dict of labels mapped to colors
                if not isinstance(kwargs['color'], dict):
                    msg = "When groupByFeature is used, the color argument "
                    msg += "must be a dict mapping labels to colors"
                    raise InvalidArgumentType(msg)
                colors = kwargs['color'].copy()
                del kwargs['color']
            else:
                colors = None

            if show and not figureID:
                # need a figure name for plotting loop
                figureID = 'Nimble Figure'
            for label in labels:
                showFig = show and label == lastLabel
                if colors:
                    try:
                        kwargs['color'] = colors[label]
                    except KeyError as e:
                        msg = "color did not contain a key for the label {}"
                        raise KeyError(msg.format(label)) from e
                grouped[label]._plotCross(
                    x, 'feature', y, 'feature', sampleSizeForAverage, trend,
                    outPath, showFig, figureID, title, xAxisLabel,
                    yAxisLabel, xMin, xMax, yMin, yMax, label=label, **kwargs)


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
                   outPath, show, figureID, title, xAxisLabel, yAxisLabel,
                   xMin, xMax, yMin, yMax, **kwargs):
        _, ax = plotFigureHandling(figureID)
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

            tmpStr = f' ({sampleSizeForAverage} sample average)'
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
            title = f'{xName} vs. {yName}'
        elif title is True:
            title = f'{self.name}: {xName} vs. {yName}'
        elif title is False:
            title = None
        ax.set_title(title)

        plotAxisLabels(ax, xAxisLabel, xlabel, yAxisLabel, ylabel)

        plotOutput(outPath, show)

    @limitedTo2D
    def plotFeatureGroupMeans(
            self, feature, groupFeature, horizontal=False, outPath=None,
            show=True, figureID=None, title=True, xAxisLabel=True,
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
        confidence interval bars, student's t-distribution, centers,
        compare, chart, figure, image, graphics, side by side,
        bar chart, t-test, uncertainty, groups, visualization
        """
        self._plotFeatureGroupStatistics(
            nimble.calculate.mean, feature, groupFeature, None, True,
            horizontal, outPath, show, figureID, title, xAxisLabel,
            yAxisLabel, **kwargs)

    @limitedTo2D
    def plotFeatureGroupStatistics(
            self, statistic, feature, groupFeature, subgroupFeature=None,
            horizontal=False, outPath=None, show=True, figureID=None,
            title=True, xAxisLabel=True, yAxisLabel=True, **kwargs):
        """
        Plot an aggregate statistic for each group of a feature.

        A bar chart where each bar in the plot represents the output of
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
            ``bar`` function.

        See Also
        --------
        nimble.calculate.statistic, matplotlib.pyplot.bar

        Keywords
        --------
        bar chart, centers, compare, chart, figure, image, graphics,
        side by side, bar chart, t-test, uncertainty, groups, stats,
        visualization
        """
        self._plotFeatureGroupStatistics(
            statistic, feature, groupFeature, subgroupFeature, False,
            horizontal, outPath, show, figureID, title, xAxisLabel,
            yAxisLabel, **kwargs)

    @pyplotRequired
    def _plotFeatureGroupStatistics(
            self, statistic, feature, groupFeature, subgroupFeature,
            confidenceIntervals, horizontal, outPath, show, figureID, title,
            xAxisLabel, yAxisLabel, **kwargs):
        fig, ax = plotFigureHandling(figureID)
        featureName = self._formattedStringID('feature', feature)
        if hasattr(statistic, '__name__') and statistic.__name__ != '<lambda>':
            statName = statistic.__name__
        else:
            statName = ''

        toGroup = self.features[[groupFeature, feature]]
        if subgroupFeature:
            subgroupFt = self.features[subgroupFeature]
            # remove name in case is same as feature
            subgroupFt.features.setNames(None, oldIdentifiers=0)
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
                group = self._formattedStringID('feature', groupFeature)
                title = f"{featureName} {statName} by {group}"

            plotMultiBarChart(ax, heights, horizontal, subgroup, **kwargs)

        else:
            heights = []
            for name, group in grouped.items():
                names.append(name)
                heights.append(statistic(group))
            plotSingleBarChart(ax, axisRange, heights, horizontal, **kwargs)

            if title is True:
                group = self._formattedStringID('feature', groupFeature)
                title = f"{featureName} {statName} by {group}"

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
    @prepLog
    def transpose(self, *,
                  useLog=None): # pylint: disable=unused-argument
        """
        Invert the feature and point indices of the data.

        Transpose the data in this object, inplace, by inverting the
        feature and point indices. This includes inverting the point and
        feature names, when available.

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
        T : Return the transposed object as a new object.

        Examples
        --------
        >>> lst = [[1, 2, 3], [4, 5, 6]]
        >>> X = nimble.data(lst)
        >>> X
        <Matrix 2pt x 3ft
             0  1  2
           ┌────────
         0 │ 1  2  3
         1 │ 4  5  6
        >
        >>> X.transpose()
        >>> X
        <Matrix 3pt x 2ft
             0  1
           ┌─────
         0 │ 1  4
         1 │ 2  5
         2 │ 3  6
        >

        Keywords
        --------
        invert, .T, reflect, inverse
        """
        self._transpose_implementation()

        self._dims = [len(self.features), len(self.points)]
        ptNames, ftNames = (self.features._getNamesNoGeneration(),
                            self.points._getNamesNoGeneration())
        self.points.setNames(ptNames, useLog=False)
        self.features.setNames(ftNames, useLog=False)

    @property
    @limitedTo2D
    def T(self): # pylint: disable=invalid-name
        """
        Invert the feature and point indices of the data.

        Return a new object that is the transpose of the calling object.
        The feature and point indices will be inverted, including
        inverting point and feature names, if available.

        See Also
        --------
        transpose : transpose the object inplace.

        Examples
        --------
        >>> lst = [[1, 2, 3], [4, 5, 6]]
        >>> X = nimble.data(lst)
        >>> X
        <Matrix 2pt x 3ft
             0  1  2
           ┌────────
         0 │ 1  2  3
         1 │ 4  5  6
        >
        >>> X.T
        <Matrix 3pt x 2ft
             0  1
           ┌─────
         0 │ 1  4
         1 │ 2  5
         2 │ 3  6
        >
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
        A deep copy is made in all cases, no references are shared
        between this object and the copy.

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

        See Also
        --------
        Points.copy, Features.copy

        Examples
        --------
        Copy this object in the same format.

        >>> lst = [[1, 3, 5], [2, 4, 6]]
        >>> ptNames = ['odd', 'even']
        >>> X = nimble.data(lst, pointNames=ptNames,
        ...                 name="odd&even")
        >>> X
        <Matrix "odd&even" 2pt x 3ft
                0  1  2
              ┌────────
          odd │ 1  3  5
         even │ 2  4  6
        >
        >>> XCopy = X.copy()
        >>> XCopy
        <Matrix "odd&even" 2pt x 3ft
                0  1  2
              ┌────────
          odd │ 1  3  5
         even │ 2  4  6
        >

        Copy to other formats.

        >>> ptNames = ['0', '1']
        >>> ftNames = ['a', 'b']
        >>> X = nimble.identity(2, pointNames=ptNames,
        ...                     featureNames=ftNames)
        >>> asDataFrame = X.copy(to='DataFrame')
        >>> asDataFrame
        <DataFrame 2pt x 2ft
             a  b
           ┌─────
         0 │ 1  0
         1 │ 0  1
        >
        >>> asNumpyArray = X.copy(to='numpy array')
        >>> asNumpyArray
        array([[1, 0],
               [0, 1]])
        >>> asListOfDict = X.copy(to='list of dict')
        >>> asListOfDict
        [{'a': 1, 'b': 0}, {'a': 0, 'b': 1}]

        Keywords
        --------
        duplicate, raw, replicate, convert, change, clone
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

        if len(self._dims) > 2:
            if to in ['listofdict', 'dictoflist', 'scipycsr', 'scipycsc']:
                msg = 'Objects with more than two dimensions cannot be '
                msg += f'copied to {origTo}'
                raise ImproperObjectAction(msg)
            if outputAs1D or not rowsArePoints:
                if outputAs1D:
                    param = 'outputAs1D'
                    value = False
                elif not rowsArePoints:
                    param = 'rowsArePoints'
                    value = True
                msg = f'{param} must be {value} when the data has more than '
                msg += 'two dimensions'
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
            ret._dims = self._dims.copy()
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
        if len(self._dims) > 2:
            ret = np.reshape(ret, self._dims).tolist()
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
    @prepLog
    def replaceRectangle(self, replaceWith, pointStart, featureStart,
                         pointEnd=None, featureEnd=None, *,
                         useLog=None): # pylint: disable=unused-argument
        """
        Replace values in the data with other values.

        Revise the contents of the calling object so that it contains
        the provided values in the given location. When ``replaceWith``
        is a Nimble object, the rectangle shape is defined so only the
        ``pointStart`` and ``featureStart`` are required to indicate
        where the top-left corner of the new data is placed within this
        object. When ``replaceWith`` is a constant, ``pointEnd`` and
        ``featureEnd`` must also be set to define the bottom-right
        corner location of the replacement rectangle within this object.

        Parameters
        ----------
        replaceWith : constant or nimble Base object
            * constant - a constant numeric value with which to fill the
              data selection.
            * nimble Base object - Size must be consistent with the
              given start and end indices.
        pointStart : int or str
            The inclusive index or name of the first point in the
            calling object whose contents will be modified.
        featureStart : int or str
            The inclusive index or name of the first feature in the
            calling object whose contents will be modified.
        pointEnd : int, str, None
            The inclusive index or name of the last point in the calling
            object whose contents will be modified. Required when
            ``replaceWith`` is a constant.
        featureEnd : int, str, None
            The inclusive index or name of the last feature in the
            calling object whose contents will be modified. Required
            when ``replaceWith`` is a constant.
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

        >>> X = nimble.ones(4, 4)
        >>> filler = nimble.zeros(2, 2)
        >>> X.replaceRectangle(filler, 0, 0, 1, 1)
        >>> X
        <Matrix 4pt x 4ft
             0  1  2  3
           ┌───────────
         0 │ 0  0  1  1
         1 │ 0  0  1  1
         2 │ 1  1  1  1
         3 │ 1  1  1  1
        >

        Keywords
        --------
        revise, insert, square, alter, stamp, cut out
        """
        psIndex = self.points.getIndex(pointStart)
        fsIndex = self.features.getIndex(featureStart)
        if isinstance(replaceWith, Base):
            excMsg =  "When the replaceWith argument is a nimble Base object, "
            excMsg += "the size of replaceWith must match the range of "
            excMsg += "modification. There are {axisLen} {axis}s in "
            excMsg += "replaceWith, yet {axis}Start ({start}) and {axis}End "
            excMsg += "({end}) define a range of length {rangeLen}"
            if pointEnd is None:
                peIndex = psIndex + len(replaceWith.points) - 1
            else:
                peIndex = self.points.getIndex(pointEnd)
                prange = (peIndex - psIndex) + 1
                if len(replaceWith.points) != prange:
                    msg = excMsg.format(
                        axis='point', axisLen=len(replaceWith.points),
                        start=pointStart, end=pointEnd, rangeLen=prange)
                    raise InvalidArgumentValueCombination(msg)
            if featureEnd is None:
                feIndex = fsIndex + len(replaceWith.points) - 1
            else:
                feIndex = self.features.getIndex(featureEnd)
                frange = (feIndex - fsIndex) + 1
                if len(replaceWith.features) != frange:
                    msg = excMsg.format(
                        axis='feature', axisLen=len(replaceWith.features),
                        start=featureStart, end=featureEnd, rangeLen=frange)
                    raise InvalidArgumentValueCombination(msg)
            if replaceWith.getTypeString() != self.getTypeString():
                replaceWith = replaceWith.copy(to=self.getTypeString())
        elif looksNumeric(replaceWith):
            if pointEnd is None:
                msg = "pointEnd is required when replaceWith is a constant"
                raise InvalidArgumentValue(msg)
            if featureEnd is None:
                msg = "featureEnd is required when replaceWith is a constant"
                raise InvalidArgumentValue(msg)
            peIndex = self.points.getIndex(pointEnd)
            feIndex = self.features.getIndex(featureEnd)
            if psIndex > peIndex:
                msg = "pointStart (" + str(pointStart) + ") must be less than "
                msg += "or equal to pointEnd (" + str(pointEnd) + ")."
                raise InvalidArgumentValueCombination(msg)
            if fsIndex > feIndex:
                msg = "featureStart (" + str(featureStart) + ") must be less "
                msg += "than or equal to featureEnd (" + str(featureEnd) + ")."
                raise InvalidArgumentValueCombination(msg)
        else:
            msg = "replaceWith may only be a nimble Base object, or a single "
            msg += "numeric value, yet we received something of "
            msg += str(type(replaceWith))
            raise InvalidArgumentType(msg)

        self._replaceRectangle_implementation(replaceWith, psIndex, fsIndex,
                                              peIndex, feIndex)


    def _flattenNames(self, order):
        """
        Helper calculating the axis names for the unflattend axis after
        a flatten operation.
        """
        pNames = self.points.getNames()
        fNames = self.features.getNames()

        ret = []
        pointNumber = '_PT#{}'.format
        featureNumber = '_FT#{}'.format
        if order == 'point':
            for (i, p), (j, f) in itertools.product(enumerate(pNames),
                                                    enumerate(fNames)):
                if p is None:
                    p = pointNumber(i)
                if f is None:
                    f = featureNumber(j)
                ret.append(' | '.join([p, f]))
        else:
            for (j, f), (i, p) in itertools.product(enumerate(fNames),
                                                    enumerate(pNames)):
                if f is None:
                    f = featureNumber(j)
                if p is None:
                    p = pointNumber(i)
                ret.append(' | '.join([p, f]))
        return ret

    @prepLog
    def flatten(self, order='point', *,
                useLog=None): # pylint: disable=unused-argument
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
        unflatten, shape, dimensions

        Examples
        --------
        >>> lst = [[1, 2],
        ...        [3, 4]]
        >>> ptNames = ['1', '3']
        >>> ftNames = ['a', 'b']
        >>> X = nimble.data(lst, pointNames=ptNames,
        ...                 featureNames=ftNames)
        >>> X.flatten()
        >>> X
        <Matrix 1pt x 4ft
                     1 | a  1 | b  3 | a  3 | b
                   ┌───────────────────────────
         Flattened │   1      2      3      4
        >

        >>> lst = [[1, 2],
        ...        [3, 4]]
        >>> ptNames = ['1', '3']
        >>> ftNames = ['a', 'b']
        >>> X = nimble.data(lst, pointNames=ptNames,
        ...                 featureNames=ftNames)
        >>> X.flatten(order='feature')
        >>> X
        <Matrix 1pt x 4ft
                     1 | a  3 | a  1 | b  3 | b
                   ┌───────────────────────────
         Flattened │   1      3      2      4
        >

        Keywords
        --------
        reshape, restructure, ravel, one dimensional, 1 dimensional
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
        if order == 'feature' and len(self._dims) > 2:
            msg = "order='feature' is not allowed for flattening objects with "
            msg += 'more than two dimensions'
            raise ImproperObjectAction(msg)

        fNames = None
        if self.points._namesCreated() or self.features._namesCreated():
            fNames = self._flattenNames(order)
        self._dims = list(self.shape) # make 2D before flattening
        self._flatten_implementation(order)
        self._dims = [1, len(self.points) * len(self.features)]

        self.features.setNames(fNames, useLog=False)
        self.points.setNames(['Flattened'], useLog=False)


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
    @prepLog
    def unflatten(self, dataDimensions, order='point', *,
                  useLog=None): # pylint: disable=unused-argument
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
        flatten, shape, dimensions

        Examples
        --------

        Unflatten a point in point order with default names.

        >>> lst = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        >>> X = nimble.data(lst)
        >>> X.unflatten((3, 3))
        >>> X
        <Matrix 3pt x 3ft
             0  1  2
           ┌────────
         0 │ 1  2  3
         1 │ 4  5  6
         2 │ 7  8  9
        >

        Unflatten a point in feature order with default names.

        >>> lst = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        >>> X = nimble.data(lst)
        >>> X.unflatten((3, 3), order='feature')
        >>> X
        <Matrix 3pt x 3ft
             0  1  2
           ┌────────
         0 │ 1  4  7
         1 │ 2  5  8
         2 │ 3  6  9
        >

        Unflatten a feature in feature order with default names.

        >>> lst = [[1], [4], [7], [2], [5], [8], [3], [6], [9]]
        >>> X = nimble.data(lst)
        >>> X.unflatten((3, 3), order='feature')
        >>> X
        <Matrix 3pt x 3ft
             0  1  2
           ┌────────
         0 │ 1  2  3
         1 │ 4  5  6
         2 │ 7  8  9
        >

        Unflatten a feature in point order with default names.

        >>> lst = [[1], [4], [7], [2], [5], [8], [3], [6], [9]]
        >>> X = nimble.data(lst)
        >>> X.unflatten((3, 3), order='point')
        >>> X
        <Matrix 3pt x 3ft
             0  1  2
           ┌────────
         0 │ 1  4  7
         1 │ 2  5  8
         2 │ 3  6  9
        >

        Unflatten a point with names that can be unflattened.

        >>> lst = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        >>> ftNames = ['1 | a', '1 | b', '1 | c',
        ...            '4 | a', '4 | b', '4 | c',
        ...            '7 | a', '7 | b', '7 | c']
        >>> X = nimble.data(lst, featureNames=ftNames)
        >>> X.unflatten((3, 3))
        >>> X
        <Matrix 3pt x 3ft
             a  b  c
           ┌────────
         1 │ 1  2  3
         4 │ 4  5  6
         7 │ 7  8  9
        >

        Keywords
        --------
        reshape, shape, dimensions, 2d, 3d, two dimensional,
        three dimensional, multi dimensional
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

        self._dims = list(dataDimensions)
        self.points.setNames(pNames, useLog=False)
        self.features.setNames(fNames, useLog=False)


    @limitedTo2D
    @prepLog
    def merge(self, other, point='strict', feature='union', onFeature=None,
              force=False, *,
              useLog=None): # pylint: disable=unused-argument
        """
        Combine data from another object with this object.

        Merge can be based on point names or a common feature between
        the objects. How the data will be merged is based upon the
        string arguments provided to ``point`` and ``feature``. If
        ``onFeature`` is None, the objects will be merged on the point
        names. Otherwise, ``onFeature`` must contain only unique values
        in one or both objects.

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
        Points.append, Features.append

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

        >>> lstL = [["a", 1, 'X'], ["b", 2, 'Y'], ["c", 3, 'Z']]
        >>> fNamesL = ["f1", "f2", "f3"]
        >>> pNamesL = ["p1", "p2", "p3"]
        >>> left = nimble.data(lstL, pointNames=pNamesL,
        ...                    featureNames=fNamesL)
        >>> lstR = [['Z', "f", 6], ['Y', "e", 5], ['X', "d", 4]]
        >>> fNamesR = ["f3", "f4", "f5"]
        >>> pNamesR = ["p3", "p2", "p1"]
        >>> right = nimble.data(lstR, pointNames=pNamesR,
        ...                     featureNames=fNamesR)
        >>> left.merge(right, point='strict', feature='union')
        >>> left
        <DataFrame 3pt x 5ft
              f1  f2  f3  f4  f5
            ┌───────────────────
         p1 │ a   1   X   d   4
         p2 │ b   2   Y   e   5
         p3 │ c   3   Z   f   6
        >
        >>> left = nimble.data(lstL, pointNames=pNamesL,
        ...                    featureNames=fNamesL)
        >>> left.merge(right, point='strict', feature='intersection')
        >>> left
        <DataFrame 3pt x 1ft
              f3
            ┌───
         p1 │ X
         p2 │ Y
         p3 │ Z
        >

        Additional merge combinations. In this example, the feature
        ``"id"`` contains a unique value for each point (just as point
        names do). In the example above we matched based on point names,
        here the ``"id"`` feature will be used to match points.

        >>> lstL = [["a", 1, 'id1'], ["b", 2, 'id2'], ["c", 3, 'id3']]
        >>> fNamesL = ["f1", "f2", "id"]
        >>> left = nimble.data(lstL, featureNames=fNamesL)
        >>> lstR = [['id3', "x", 7], ['id4', "y", 8], ['id5', "z", 9]]
        >>> fNamesR = ["id", "f4", "f5"]
        >>> right = nimble.data(lstR, featureNames=fNamesR)
        >>> left.merge(right, point='union', feature='union',
        ...            onFeature="id")
        >>> left
        <DataFrame 5pt x 5ft
             f1    f2    id  f4    f5
           ┌──────────────────────────
         0 │ a   1.000  id1
         1 │ b   2.000  id2
         2 │ c   3.000  id3  x   7.000
         3 │            id4  y   8.000
         4 │            id5  z   9.000
        >
        >>> left = nimble.data(lstL, featureNames=fNamesL)
        >>> left.merge(right, point='union', feature='intersection',
        ...            onFeature="id")
        >>> left
        <DataFrame 5pt x 1ft
              id
           ┌────
         0 │ id1
         1 │ id2
         2 │ id3
         3 │ id4
         4 │ id5
        >
        >>> left = nimble.data(lstL, featureNames=fNamesL)
        >>> left.merge(right, point='union', feature='left',
        ...            onFeature="id")
        >>> left
        <DataFrame 5pt x 3ft
             f1    f2    id
           ┌───────────────
         0 │ a   1.000  id1
         1 │ b   2.000  id2
         2 │ c   3.000  id3
         3 │            id4
         4 │            id5
        >
        >>> left = nimble.data(lstL, featureNames=fNamesL)
        >>> left.merge(right, point='intersection', feature='union',
        ...            onFeature="id")
        >>> left
        <DataFrame 1pt x 5ft
             f1  f2   id  f4  f5
           ┌────────────────────
         0 │ c   3   id3  x   7
        >
        >>> left = nimble.data(lstL, featureNames=fNamesL)
        >>> left.merge(right, point='intersection',
        ...            feature='intersection', onFeature="id")
        >>> left
        <DataFrame 1pt x 1ft
              id
           ┌────
         0 │ id3
        >
        >>> left = nimble.data(lstL, featureNames=fNamesL)
        >>> left.merge(right, point='intersection', feature='left',
        ...            onFeature="id")
        >>> left
        <DataFrame 1pt x 3ft
             f1  f2   id
           ┌────────────
         0 │ c   3   id3
        >
        >>> left = nimble.data(lstL, featureNames=fNamesL)
        >>> left.merge(right, point='left', feature='union',
        ...            onFeature="id")
        >>> left
        <DataFrame 3pt x 5ft
             f1  f2   id  f4    f5
           ┌───────────────────────
         0 │ a   1   id1
         1 │ b   2   id2
         2 │ c   3   id3  x   7.000
        >
        >>> left = nimble.data(lstL, featureNames=fNamesL)
        >>> left.merge(right, point='left', feature='intersection',
        ...            onFeature="id")
        >>> left
        <DataFrame 3pt x 1ft
              id
           ┌────
         0 │ id1
         1 │ id2
         2 │ id3
        >
        >>> left = nimble.data(lstL, featureNames=fNamesL)
        >>> left.merge(right, point='left', feature='left',
        ...            onFeature="id")
        >>> left
        <DataFrame 3pt x 3ft
             f1  f2   id
           ┌────────────
         0 │ a   1   id1
         1 │ b   2   id2
         2 │ c   3   id3
        >

        Keywords
        --------
        combine, join, unite, group, fuse, consolidate, meld, union,
        intersection, inner, outer, full outer, left
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
        
        if point == 'union':
            if onFeature:
                self.points.sort(by=onFeature, useLog=False)
        
            
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
            msg += f"{axis}s when {axis}='strict'"
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
                msg = f"could not locate feature '{onFeature}' in both objects"
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
                    msg = "When {axis}='strict', the {axis} names may be in a "
                    msg += "different order but must match exactly"
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
                    msg = f"When {axis}='strict' and default {axis} names "
                    msg += f"exist, names cannot be reordered. The {axis} "
                    msg += "names do not match at each index where both  "
                    msg += f"objects have non-default {axis} names."
                    raise InvalidArgumentValue(msg) from e
            else: # default names and not force
                msg = f'Non-default {axis} names are required in both objects '
                msg += 'unless the force parameter is set to True to indicate '
                msg += f'that each {axis} is equal between the two objects'
                raise InvalidArgumentValue(msg)

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
                    msg = f'The feature names at index {onFeature} do not '
                    msg += 'match in each object'
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
                msg = f"could not locate feature '{onFeature}' in both objects"
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

        See Also
        --------
        nimble.calculate.inverse, nimble.calculate.pseudoInverse

        Keywords
        --------
        invert, reciprocal, solve, pseudo-inverse, multiplicative
        inverse, singular-value decomposition (SVD)
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

        Keywords
        --------
        linear algebra, ordinary least squares, solver, invert, inverse,
        L2, pseudoinverse, pseudo inverse, equations
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

        Keywords
        --------
        times, product, multiplication
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

        Keywords
        --------
        exponent, raise, square, squared, raised
        """
        if not isinstance(power, (int, np.int_)):
            msg = 'power must be an integer'
            raise InvalidArgumentType(msg)
        if not len(self.points) == len(self.features):
            msg = 'Cannot perform matrix power operations with this object. '
            msg += 'Matrix power operations require square objects '
            msg += '(number of points is equal to number of features)'
            raise ImproperObjectAction(msg)
        if power == 0:
            operand = nimble.identity(len(self.points),
                                      returnType=self.getTypeString())
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
        ret._dims = self._dims.copy()
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
        if self._dims != other._dims:
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
            msg = f"When both point and feature names are present, {opName} "
            msg += "requires either the point names or feature names to "
            msg += "be equal between the left and right objects"
            raise InvalidArgumentValue(msg)

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
            msg = f"{opName} between objects with equal {equalAxis} names "
            msg += f"must have unique {axis} names. However, the {axis} names "
            msg += f"{matches} were found in both the left and right objects"
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

        ret._dims = self._dims
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

        >>> lst3x3 = [[1, 2, 3], [4, 5, 6], [0, -1, -2]]
        >>> lst1x3 = [1, 2, 3]
        >>> baseObj = nimble.data(lst3x3)
        >>> pointObj = nimble.data(lst1x3)
        >>> baseObj * pointObj.stretch
        <Matrix 3pt x 3ft
             0  1   2
           ┌──────────
         0 │ 1   4   9
         1 │ 4  10  18
         2 │ 0  -2  -6
        >

        Stretched feature with nimble Base object.

        >>> lst3x3 = [[1, 2, 3], [4, 5, 6], [0, -1, -2]]
        >>> lst1x3 = [[1], [2], [3]]
        >>> baseObj = nimble.data(lst3x3)
        >>> featObj = nimble.data(lst1x3)
        >>> featObj.stretch + baseObj
        <Matrix 3pt x 3ft
             0  1  2
           ┌────────
         0 │ 2  3  4
         1 │ 6  7  8
         2 │ 3  2  1
        >

        Two stretched objects.

        >>> lst1x3 = [[1, 2, 3]]
        >>> lst3x1 = [[1], [2], [3]]
        >>> pointObj = nimble.data(lst1x3)
        >>> featObj = nimble.data(lst3x1)
        >>> pointObj.stretch - featObj.stretch
        <Matrix 3pt x 3ft
             0   1   2
           ┌──────────
         0 │  0   1  2
         1 │ -1   0  1
         2 │ -2  -1  0
        >
        >>> featObj.stretch - pointObj.stretch
        <Matrix 3pt x 3ft
             0  1   2
           ┌──────────
         0 │ 0  -1  -2
         1 │ 1   0  -1
         2 │ 2   1   0
        >

        Keywords
        --------
        broadcast, broadcasting, expand, match, matching, resize, grow,
        shape, dimensions, reshape, resize, spread, restructure
        """
        return Stretch(self)
    
    ############################
    ############################
    ###  dunder functions ###
    ############################
    ############################
        
    def __getattr__(self, name):
        # Want to pull msg out from raise, since that line is included
        # in the traceback and otherwise would clutter the output.
        base = f"Attribute {name} does not exist for Nimble data objects. "
        if name == "hist":
            msg = "Try .plotFeatureDistribution() instead."
            raise AttributeError(base + msg)
        elif name in ["describe", "summary"]:
            msg = "Try .report() instead."
            raise AttributeError(base + msg)
        elif name in ["columns", "colnames"]:
            msg = "Try .features.getNames() instead."
            raise AttributeError(base + msg)
        elif name in ["isna", "isnull"]:
            msg = "Look up the Nimble QueryString object in the documentation."
            raise AttributeError(base + msg)
        elif name in ["sum", "cumsum", "cor", "corr", "nunique"]:
            msg = "Check for similar methods available in Nimble.calculate."
            raise AttributeError(base + msg)
        elif name in ["scatter_matrix","scatter_pairs"]:
            msg = "Try .plotFeatureAgainstFeature()instead."
            raise AttributeError(base + msg)
        elif name in ["dropna", "omit"]:
            msg = "Look up how to combine .[points/features].delete with the Nimble.match in the documentation."
            raise AttributeError(base + msg)
        elif name == "fillna":
            msg = "Look up how to use Nimble.fill in the documentation."
            raise AttributeError(base + msg)
        elif name == "insert":
            msg = "Try .[points/features].insert() instead."
            raise AttributeError(base + msg)
        elif name in ["sort_values", "sort"]:
            msg = "Try .[points/features].sort() instead."
            raise AttributeError(base + msg)
        else:
            # this re-triggers standard handling without any recursion, which
            # includes proper instantion of the error so that name suggestions
            # can be displayed.
            return self.__getattribute__(name)

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
            msg = 'other._dims'
            raise InvalidArgumentValue(msg)
        if hasattr(other, '_source'): # view
            other = other.copy()

        kwargs['data'] = other._data
        kwargs['shape'] = other._dims
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
        """
        # pylint: disable=unused-argument
        self.__init__(**kwargs)

    def _arrangePointNames(self, pRange, maxRows, nameLength, rowHolder, nameHold,
                           includePointNames, quoteNames):
        """
        Prepare point names for string output. Grab only section of
        those names that fit according to the given row limitation,
        and process them for length. Returns a list of prepared names
        and an int bounding the length of each name representation.
        """
        if len(self.points) == 0:
            return [], 0

        names = []
        pnamesWidth = 0
        nameCutIndex = nameLength - len(nameHold)

        tRowIDs, bRowIDs = indicesSplit(maxRows, pRange)

        def getNameString(index):
            if self.points._allDefaultNames() or includePointNames is False:
                return str(index)

            name = self.points.getName(index)
            if name is None:
                return str(index)

            if quoteNames:
                if len(name) <= nameLength - 2:
                    finalName = name
                else:
                    finalName = name[:nameCutIndex - 2] + nameHold
                return "'" + finalName + "'"
            else:
                if len(name) <= nameLength:
                    finalName = name
                else:
                    finalName = name[:nameCutIndex] + nameHold
                return (finalName)

        topNames = list(map(getNameString, tRowIDs))
        bottomNames = list(map(getNameString, bRowIDs))
        maxWidth = max(map(len, topNames))
        if bottomNames:
            maxWidth = max(maxWidth, max(map(len, bottomNames)))
        pnamesWidth = min(nameLength, maxWidth)
        names = topNames
        if len(topNames) + len(bottomNames) < len(pRange):
            names.append(rowHolder)
        names.extend(bottomNames)

        return names, pnamesWidth

    def _arrangeDataWithLimits(self, numPts, pRange, features,
                               maxWidth, maxHeight, sigDigits,
                               maxStrLength, colSep, colHold, rowHold,
                               strHold, includeFeatureNames, quoteNames):
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
        if maxHeight < 2 and maxHeight != len(self.points):
            msg = "If the number of points in this object is two or greater, "
            msg += "then we require that the input argument maxHeight also "
            msg += "be greater than or equal to two."
            raise InvalidArgumentValue(msg)

        cHoldWidth = len(colHold)
        cHoldTotal = len(colSep) + cHoldWidth
        nameCutIndex = maxStrLength - len(strHold)

        # At the beginning of this method,features may only be a
        # range or an int
        if isinstance(features, range):
            numFts = len(features)
            if numFts != 0:
                fOffset = features[0]
            else:
                fOffset = 0
        else:
            numFts = features
            fOffset = 0

        #setup a bundle of default values
        if maxHeight is None:
            maxHeight = numPts
        if maxWidth is None:
            maxWidth = float('inf')

        maxRows = min(maxHeight, numPts)
        maxDataRows = maxRows

        if numFts == 0:
            return [[]] * maxDataRows, [], []

        if numPts > 0:
            tRowIDs, bRowIDs = indicesSplit(maxDataRows, pRange)
        else:
            tRowIDs, bRowIDs = [], []
        combinedRowIDs = tRowIDs + bRowIDs

        if len(combinedRowIDs) < numPts:
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
        endIndex = numFts // 2
        if numFts % 2 == 1:
            endIndex *= -1
            endIndex -= 1
        currIndex = 0
        numAdded = 0

        # due to the possibility of ranges that don't start at zero,
        # we define the access index to be the absolute position in
        # the object, not the numerical position relative to within
        # the possible range.
        accessIndex = fOffset

        if self.features._allDefaultNames() or includeFeatureNames is False:
            fnames = None
        else:
            fnames = self.features.getNames()

        while totalWidth < maxWidth and currIndex != endIndex:
            currTable = lTable if currIndex >= 0 else rTable
            currCol = []

            if fnames is None:
                currFName = None
            else:
                currFName = fnames[accessIndex]
            if currFName is None:
                currFName = str(accessIndex)
            elif quoteNames:
                if len(currFName) > maxStrLength - 2:
                    currFName = currFName[:nameCutIndex - 2] + strHold
                currFName = "'" + currFName + "'"
            else:
                if len(currFName) > maxStrLength:
                    currFName = currFName[:nameCutIndex] + strHold

            nameWidth = len(currFName)
            valWidth = 0

            # check all values in this column (in the accepted rows)
            i = 0
            for rID in combinedRowIDs:
                val = self[rID, accessIndex]
                valFormed = formatIfNeeded(val, sigDigits)
                if len(valFormed) <= maxStrLength:
                    valLimited = valFormed
                else:
                    valLimited = valFormed[:nameCutIndex] + strHold

                if len(valLimited) > valWidth:
                    valWidth = len(valLimited)

                # If these are equal, it is time to add the holders
                if i == rowHolderIndex:
                    currCol.append(rowHold)

                currCol.append(valLimited)
                i += 1

            # The max width used by this column
            currWidths = (nameWidth, valWidth)

            # placeholder row needs to be placed at bottom
            if i == rowHolderIndex:
                currCol.append(rowHold)

            totalWidth += max(currWidths)
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
                    rFNames.append(currFName)
                    currIndex = abs(currIndex)
                    accessIndex = fOffset + currIndex
                    rColWidths.append(currWidths)
                else:
                    lFNames.append(currFName)
                    currIndex = (-1 * currIndex) - 1
                    accessIndex = fOffset + numFts + currIndex
                    lColWidths.append(currWidths)

            # ignore column separator if the next column is the last
            if numAdded == (numFts - 1):
                totalWidth -= cHoldTotal
            totalWidth += len(colSep)

        # combine the tables. Have to reverse rTable because entries were
        # appended in a right to left order
        rColWidths.reverse()
        rFNames.reverse()
        fNames = lFNames + rFNames
        if numAdded == numFts:
            lColWidths += rColWidths
        else:
            lColWidths += [cHoldWidth] + rColWidths
            fNames = lFNames + [colHold] + rFNames
        # return None if fNames is []
        fNames = fNames if fNames else None
        # pylint: disable=consider-using-enumerate
        for rowIndex in range(len(lTable)):
            if len(rTable) > 0:
                rTable[rowIndex].reverse()
                toAdd = rTable[rowIndex]
            else:
                toAdd = []

            if numAdded == numFts:
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

            if inconsistencies:
                table = [['left', 'ID', 'right']]
                for i in sorted(inconsistencies.keys()):
                    lname = lnames[i]
                    rname = rnames[i]
                    lname = str(None) if lname is None else '"' + lname + '"'
                    rname = str(None) if rname is None else '"' + rname + '"'
                    table.append([lname, str(i), rname])

                msg = leftAxis + " to " + rightAxis + " name inconsistencies "
                msg += "when calling left." + callSym + "(right) \n"
                msg += tableString(table)

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
    def _saveCSV_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
        pass

    @abstractmethod
    def _saveMTX_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
        pass

    @abstractmethod
    def _getTypeString_implementation(self):
        pass

    @abstractmethod
    def _getitem_implementation(self, x, y):
        pass
    
    @abstractmethod
    def _setitem_implementation(self, x, y, value):
        pass

    @abstractmethod
    def _view_implementation(self, pointStart, pointEnd, featureStart,
                             featureEnd, dropDimension):
        pass

    @abstractmethod
    def _checkInvariants_implementation(self, level):
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

    ################
    # Stats methods #
    ###############

    def _vectorize(self):
        baseVector = self.copy()
        baseVector.flatten(useLog=False)
        return baseVector

    def max(self):
        """
        Returns the maximum value in the nimble object.

        See Also
        --------
        minimum

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.max()
        22
        """
        if not isNumeric(self):
            return(np.nan)   
        
        return nimble.calculate.maximum(self._vectorize())

    def mean(self):
        """
        Returns the mean value in the nimble object.

        See Also
        --------
        median

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.mean()
        9.0
        """
        if not isNumeric(self):
            return(np.nan) 
        return nimble.calculate.mean(self._vectorize())
        #return nimble.calculate.mean(self)

    def median(self):
        """
        Returns the median value in the nimble object.

        See Also
        --------
        mean

        Examples
        --------
        >>> lst = [[0, 22, 2], [33, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.median()
        13.5
        """
        if not isNumeric(self):
            return(np.nan) 
        return nimble.calculate.median(self._vectorize())
        #return nimble.calculate.median(self)

    def min(self):
        """
        Returns the minimum value in the nimble object.

        See Also
        --------
        maximum

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.min()
        0
        """
        if not isNumeric(self):
            return(np.nan) 
        return nimble.calculate.minimum(self._vectorize())

    def uniqueCount(self):
        """
        Returns the number of unique values in the nimble object.

        Examples
        --------
        >>> lst = [[0, 22, 2], [3, 22, 5]]
        >>> X = nimble.data(lst)
        >>> X.uniqueCount()
        5
        """
        if not isNumeric(self):
            return(np.nan) 
        return nimble.calculate.uniqueCount(self._vectorize())

    def proportionMissing(self):
        """
        Returns a number representing the proportion of
        values that are None or NaN in the nimble object.

        Examples
        --------
        >>> lst = [[float('nan'), float('nan'), 0], [0, 2, float('nan')]]
        >>> X = nimble.data(lst)
        >>> X.proportionMissing()
        0.5
        """
        if not isNumeric(self):
            calcListByFeature = list(self.features.proportionMissing())
            propMissing = sum(calcListByFeature)/len(self.features)
            return round(propMissing, 3)
        return nimble.calculate.proportionMissing(self._vectorize())

    def proportionZero(self):
        """
        Returns a number representing the proportion of values
        that are equal to zero in the nimble object.

        Examples
        --------
        >>> lst = [[1, 2, 0], [0, 0, 5]]
        >>> X = nimble.data(lst)
        >>> X.proportionZero()
        0.5
        """
        if not isNumeric(self):
            calcListByFeature = list(self.features.proportionZero())
            propZero = sum(calcListByFeature)/len(self.features)
            return round(propZero, 3)
        return nimble.calculate.proportionZero(self._vectorize())
    
    def mode(self):
        """
        Returns a number representing the mode of the nimble object.
        
        Examples
        --------
        >>> lst = [[1, 2, 0], [0, 0, 5]]
        >>> X = nimble.data(lst)
        >>> X.mode()
        0
        """
        return nimble.calculate.mode(self._vectorize())
    
    def sum(self):
        """
        Returns a number representing the sum of the nimble object.
        
        Examples
        --------
        >>> lst = [[1, 2, 0], [0, 0, 5]]
        >>> X = nimble.data(lst)
        >>> X.sum()
        8
        """
        if not isNumeric(self):
            return(np.nan) 
        return nimble.calculate.sum(self._vectorize())
    
    def variance(self):
        """
        Returns a number representing the variance of the nimble object.
        
        Examples
        --------
        >>> lst = [[1, 2, 0], [0, 0, 5]]
        >>> X = nimble.data(lst)
        >>> X.variance()
        3.866666666666667
        """
        if not isNumeric(self):
            return(np.nan) 
        return nimble.calculate.variance(self._vectorize())
    
    def medianAbsoluteDeviation(self):
        """
        Returns a number representing the median absolute deviation 
        of the nimble object.
        
        Examples
        --------
        >>> lst = [[1, 2, 0], [0, 0, 5]]
        >>> X = nimble.data(lst)
        >>> X.medianAbsoluteDeviation()
        0.5
        """
        if not isNumeric(self):
            return(np.nan) 
        return nimble.calculate.medianAbsoluteDeviation(self._vectorize())
    
    def quartiles(self):
        """
        Returns a list of numbers representing the quartiles of 
        the nimble object.
        
        Examples
        --------
        >>> lst = [[1, 2, 0], [0, 0, 5]]
        >>> X = nimble.data(lst)
        >>> X.quartiles()
        (0.0, 0.5, 1.75)
        """
        if not isNumeric(self):
            return(np.nan) 
        return nimble.calculate.quartiles(self._vectorize())
