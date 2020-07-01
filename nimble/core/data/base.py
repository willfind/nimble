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
from abc import abstractmethod
from contextlib import contextmanager

import numpy

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.core.logger import handleLogging
from nimble.core.logger import produceFeaturewiseReport
from nimble.core.logger import produceAggregateReport
from nimble._utility import cloudpickle, h5py
from .points import Points
from .features import Features
from .axis import Axis
from .stretch import Stretch
from . import _dataHelpers
# the prefix for default point and feature names
from ._dataHelpers import DEFAULT_PREFIX, DEFAULT_PREFIX_LENGTH
from ._dataHelpers import DEFAULT_NAME_PREFIX
from ._dataHelpers import formatIfNeeded
from ._dataHelpers import valuesToPythonList, constructIndicesList
from ._dataHelpers import createListOfDict, createDictOfList
from ._dataHelpers import createDataNoValidation
from ._dataHelpers import csvCommaFormat
from ._dataHelpers import validateElementFunction, wrapMatchFunctionFactory
from ._dataHelpers import ElementIterator1D
from ._dataHelpers import isQueryString, elementQueryFunction
from ._dataHelpers import limitedTo2D
from ._dataHelpers import plotPlotter, distributionPlotter, crossPlotter
from ._dataHelpers import matplotlibBackendHandling


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

    def __init__(self, shape, pointNames=None, featureNames=None, name=None,
                 paths=(None, None), **kwds):
        """
        Class defining important data manipulation operations and giving
        functionality for the naming the points and features of that
        data. A mapping from names to indices is given by the
        [point/feature]Names attribute, the inverse of that mapping is
        given by [point/feature]NamesInverse.

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
        """
        self._shape = list(shape)
        if pointNames is not None and len(pointNames) != self._pointCount:
            msg = "The length of the pointNames (" + str(len(pointNames))
            msg += ") must match the points given in shape (" + str(shape[0])
            msg += ")"
            raise InvalidArgumentValue(msg)
        if (featureNames is not None
                and len(featureNames) != self._featureCount):
            msg = "The length of the featureNames (" + str(len(featureNames))
            msg += ") must match the features given in shape ("
            msg += str(shape[1]) + ")"
            raise InvalidArgumentValue(msg)

        self._points = self._getPoints()
        self._features = self._getFeatures()

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
            self._name = _dataHelpers.nextDefaultObjectName()
        else:
            self._name = name

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
        super(Base, self).__init__(**kwds)

    #######################
    # Property Attributes #
    #######################

    @property
    def _pointCount(self):
        return self._shape[0]

    @_pointCount.setter
    def _pointCount(self, value):
        self._shape[0] = value

    @property
    def _featureCount(self):
        if len(self._shape) > 2:
            return int(numpy.prod(self._shape[1:]))
        return self._shape[1]

    @_featureCount.setter
    @limitedTo2D
    def _featureCount(self, value):
        self._shape[1] = value

    @property
    def shape(self):
        """
        The number of points and features in the object in the format
        (points, features).
        """
        return self._pointCount, self._featureCount

    @property
    def dimensions(self):
        """
        The true dimensions of this object.
        """
        return tuple(self._shape)

    def _getPoints(self):
        """
        Get the object containing point-based methods for this object.
        """
        return BasePoints(base=self)

    @property
    def points(self):
        """
        An object handling functions manipulating data by points.

        See Also
        --------
        Points
        """
        return self._points

    def _getFeatures(self):
        """
        Get the object containing feature-based methods for this object.
        """
        return BaseFeatures(base=self)

    @property
    def features(self):
        """
        An object handling functions manipulating data by features.

        See Also
        --------
        Features
        """
        return self._features

    def _setpointCount(self, value):
        self._pointCount = value

    def _setfeatureCount(self, value):
        self._featureCount = value

    def _getObjName(self):
        return self._name

    def _setObjName(self, value):
        if value is None:
            self._name = _dataHelpers.nextDefaultObjectName()
        else:
            if not isinstance(value, str):
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
            self._shape = [self._pointCount, self._featureCount]
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
        raise ImproperObjectAction(msg)

    @limitedTo2D
    def __iter__(self):
        if self._pointCount in [0, 1] or self._featureCount in [0, 1]:
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

    def nameIsDefault(self):
        """
        Returns True if self.name has a default value
        """
        return self.name.startswith(DEFAULT_NAME_PREFIX)

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
        if self._pointCount == 0:
            msg = "This action is impossible, the object has 0 points"
            raise ImproperObjectAction(msg)

        index = self.features.getIndex(featureToReplace)

        replace = self.features.extract([index], useLog=False)

        uniqueVals = list(replace.countUniqueElements().keys())

        binaryObj = replace._replaceFeatureWithBinaryFeatures_implementation(
            uniqueVals)

        binaryObj.points.setNames(self.points._getNamesNoGeneration(),
                                  useLog=False)
        ftNames = []
        prefix = replace.features.getName(0) + "="
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
        if self._pointCount == 0:
            msg = "This action is impossible, the object has 0 points"
            raise ImproperObjectAction(msg)

        ftIndex = self.features.getIndex(featureToConvert)

        mapping = {}
        def applyMap(ft):
            uniqueVals = ft.countUniqueElements()
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
            [[11.000   0      0      0      0   ]
             [  0    11.000   0      0      0   ]
             [  0      0    11.000   0      0   ]
             [  0      0      0    11.000   0   ]
             [  0      0      0      0    11.000]]
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
            [[None  12  None]
             [ 14  None  16 ]
             [None  18  None]]
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
            [[11.000   0      0      0      0   ]
             [  0    11.000   0      0      0   ]
             [  0      0    11.000   0      0   ]
             [  0      0      0    11.000   0   ]
             [  0      0      0      0    11.000]]
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
                                      preserveZeros, skipNoneReturnValues,
                                      outputType)

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
        toMatch
            * value - elements equal to the value return True
            * function - in the form of toMatch(elementValue) that
              returns True, False, 0 or 1.
            * str - a comparison operator and a value (i.e ">=0")
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
            query = isQueryString(matchArg)
            if query:
                func = elementQueryFunction(query)
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
                           preserveZeros=False, skipNoneReturnValues=False,
                           outputType=None, allowBoolOutput=False):
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
            vectorized = numpy.vectorize(calculator)
            values = self._calculate_implementation(
                vectorized, points, features, preserveZeros, optType)

        else:
            if not points:
                points = list(range(len(self.points)))
            if not features:
                features = list(range(len(self.features)))
            # if unable to vectorize, iterate over each point
            values = numpy.empty([len(points), len(features)])
            if allowBoolOutput:
                values = values.astype(numpy.bool_)
            p = 0
            for pi in points:
                f = 0
                for fj in features:
                    value = self[pi, fj]
                    currRet = calculator(value, pi, fj)
                    if (match.nonNumeric(currRet) and currRet is not None
                            and values.dtype != numpy.object_):
                        values = values.astype(numpy.object_)
                    values[p, f] = currRet
                    f += 1
                p += 1

        ret = nimble.data(optType, values, treatAsMissing=[None], useLog=False)

        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath

        return ret

    def _calculate_genericVectorized(
            self, function, points, features, outputType):
        # need points/features as arrays for indexing
        if points:
            points = numpy.array(points)
        else:
            points = numpy.array(range(len(self.points)))
        if features:
            features = numpy.array(features)
        else:
            features = numpy.array(range(len(self.features)))
        toCalculate = self.copy(to='numpyarray')
        # array with only desired points and features
        toCalculate = toCalculate[points[:, None], features]
        try:
            return function(toCalculate)
        except Exception:
            # change output type of vectorized function to object to handle
            # nonnumeric data
            function.otypes = [numpy.object_]
            return function(toCalculate)

    @limitedTo2D
    def countElements(self, condition):
        """
        The number of values which satisfy the condition.

        Parameters
        ----------
        condition : function
            function - may take two forms:
            a) a function that accepts an element value as input and
            will return True if it is to be counted
            b) a filter function, as a string, containing a comparison
            operator and a value

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
        query = isQueryString(condition)
        if query:
            condition = elementQueryFunction(query)
        elif not hasattr(condition, '__call__'):
            msg = 'condition can only be a function or string containing a '
            msg += 'comparison operator and a value'
            raise InvalidArgumentType(msg)

        ret = self.calculateOnElements(condition, outputType='Matrix',
                                       useLog=False)
        return int(numpy.sum(ret.data))

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
            return tuple([point[i] for i in by])

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
        if self._pointCount == 0 or self._featureCount == 0:
            return 0
        valueObj = self.calculateOnElements(hashCodeFunc, preserveZeros=True,
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
            )
        >>> testData
        Matrix(
            [[0 1 0]
             [1 0 0]]
            pointNames={'e':0, 'd':1}
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
            )
        >>> trainY
        Matrix(
            [[1]
             [2]
             [3]
             [3]]
            pointNames={'a':0, 'b':1, 'f':2, 'c':3}
            )
        >>> testX
        Matrix(
            [[0 1 0]
             [1 0 0]]
            pointNames={'e':0, 'd':1}
            )
        >>> testY
        Matrix(
            [[2]
             [1]]
            pointNames={'e':0, 'd':1}
            )
        """
        order = list(range(len(self.points)))
        if randomOrder:
            nimble.random.numpyRandom.shuffle(order)

        if not 0 <= testFraction <= 1:
            msg = 'testFraction must be between 0 and 1 (inclusive)'
            raise InvalidArgumentValue(msg)
        testXSize = int(round(testFraction * self._pointCount))
        splitIndex = self._pointCount - testXSize

        #pull out a testing set
        trainX = self.points.copy(order[:splitIndex], useLog=False)
        testX = self.points.copy(order[splitIndex:], useLog=False)

        trainX.name = self.name + " trainX"
        testX.name = self.name + " testX"

        if labels is None:
            ret = trainX, testX
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
                except InvalidArgumentValue:
                    msg = 'labels and calling object pointNames must be equal'
                    raise InvalidArgumentValue(msg)
                trainY = labels.points.copy(order[:splitIndex], useLog=False)
                testY = labels.points.copy(order[splitIndex:], useLog=False)
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
                    if name[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                        seen = True
            if not seen:
                includePointNames = False

        includeFeatureNames = includeNames
        if includeFeatureNames:
            seen = False
            if self.features._getNamesNoGeneration() is not None:
                for name in self.features.getNames():
                    if name[:DEFAULT_PREFIX_LENGTH] != DEFAULT_PREFIX:
                        seen = True
            if not seen:
                includeFeatureNames = False


        if fileFormat.lower() in ['hdf5', 'h5']:
            self._writeFileHDF_implementation(outPath, includePointNames,
                                              includeFeatureNames)
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

    def _writeFileHDF_implementation(self, outPath, includePointNames,
                                     includeFeatureNames):
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
                point._convertUnusableTypes(float, (int, float, bool), False)
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
                raise InvalidArgumentType(msg)

        #process x
        singleX = False
        if isinstance(x, (int, float, str, numpy.integer)):
            x = self.points._getIndex(x, allowFloats=True)
            singleX = True
        #process y
        singleY = False
        if isinstance(y, (int, float, str, numpy.integer)):
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
        ret = self._view_backend(index, index, None, None, True)
        return ret

    @limitedTo2D
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

        if len(self._shape) > 2:
            if featureStart != 0 or featureEnd != self._featureCount:
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
            includePNames = _dataHelpers.hasNonDefault(self, 'point')
            includeFNames = _dataHelpers.hasNonDefault(self, 'feature')
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
        with self._treatAs2D():
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
        ret += _dataHelpers.makeNamesLines(
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
        # because of how _dataHelpers.indicesSplit works, we need this to be
        # +1 in some cases this means one extra feature name is displayed. But
        # that's acceptable
        if numCols <= self._featureCount:
            numCols += 1
        elif numCols > self._featureCount:
            numCols = self._featureCount
        ret += _dataHelpers.makeNamesLines(
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
        if len(self._shape) > 2:
            context += " x ".join(map(str, self._shape))
        else:
            context += str(self._pointCount) + "pt x "
            context += str(self._featureCount) + "ft"
        print(context, '\n')
        print(self.toString(includeAxisNames, maxWidth, maxHeight, sigDigits,
                            maxColumnWidth))

    @limitedTo2D
    def plot(self, outPath=None, includeColorbar=False):
        """
        Display a plot of the data.
        """
        self._plot(outPath, includeColorbar)

    def _setupOutFormatForPlotting(self, outPath):
        outFormat = None
        if isinstance(outPath, str):
            (_, ext) = os.path.splitext(outPath)
            if len(ext) == 0:
                outFormat = 'png'
        return outFormat

    def _plot(self, outPath=None, includeColorbar=False):
        plotKwargs = {}
        plotKwargs['name'] = self.name
        plotKwargs['includeColorbar'] = includeColorbar

        outFormat = self._setupOutFormatForPlotting(outPath)
        plotKwargs['outFormat'] = outFormat

        toPlot = self._convertUnusableTypes(float, usableTypes=(int, float))
        plotKwargs['d'] = toPlot
        # problem if we were to use mutiprocessing with backends
        # different than Agg.
        p = matplotlibBackendHandling(outPath, plotPlotter, **plotKwargs)

        return p

    @limitedTo2D
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
        return self._plotDistribution('feature', feature, outPath, xMin, xMax)

    def _plotDistribution(self, axis, identifier, outPath, xMin, xMax):
        plotKwargs = {}
        plotKwargs['axis'] = axis
        plotKwargs['xLim'] = (xMin, xMax)

        outFormat = self._setupOutFormatForPlotting(outPath)
        plotKwargs['outFormat'] = outFormat

        axisObj = self._getAxis(axis)
        index = axisObj.getIndex(identifier)
        plotKwargs['index'] = index
        name = None
        if axis == 'point':
            getter = self.pointView
            if self.points._namesCreated():
                name = self.points.getName(index)
        else:
            getter = self.featureView
            if self.features._namesCreated():
                name = self.features.getName(index)
        plotKwargs['name'] = name
        toPlot = getter(index)
        plotKwargs['d'] = toPlot

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
        plotKwargs['binCount'] = binCount

        # problem if we were to use mutiprocessing with backends
        # different than Agg.
        p = matplotlibBackendHandling(outPath, distributionPlotter,
                                      **plotKwargs)

        return p

    @limitedTo2D
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
            The number of samples to use for the calculation of the
            rolling average.

        Returns
        -------
        plot
            Displayed or written to the ``outPath`` file.
        """
        self._plotFeatureAgainstFeature(x, y, outPath, xMin, xMax, yMin, yMax,
                                        sampleSizeForAverage)

    @limitedTo2D
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
        return self._plotCross(x, 'feature', y, 'feature', outPath, xMin, xMax,
                               yMin, yMax, sampleSizeForAverage)

    def _plotCross(self, x, xAxis, y, yAxis, outPath, xMin, xMax, yMin, yMax,
                   sampleSizeForAverage=None):
        plotKwargs = {}
        plotKwargs['name'] = self.name
        plotKwargs['xAxis'] = xAxis
        plotKwargs['yAxis'] = yAxis
        plotKwargs['xLim'] = (xMin, xMax)
        plotKwargs['yLim'] = (yMin, yMax)
        plotKwargs['sampleSizeForAverage'] = sampleSizeForAverage

        outFormat = self._setupOutFormatForPlotting(outPath)
        plotKwargs['outFormat'] = outFormat

        xAxisObj = self._getAxis(xAxis)
        yAxisObj = self._getAxis(yAxis)
        xIndex = xAxisObj.getIndex(x)
        yIndex = yAxisObj.getIndex(y)
        plotKwargs['xIndex'] = xIndex
        plotKwargs['yIndex'] = yIndex

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
        plotKwargs['xName'] = xName
        plotKwargs['yName'] = yName

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

        plotKwargs['inX'] = xToPlot
        plotKwargs['inY'] = yToPlot

        # problem if we were to use mutiprocessing with backends
        # different than Agg.
        p = matplotlibBackendHandling(outPath, crossPlotter, **plotKwargs)

        return p

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

        self._pointCount, self._featureCount = (self._featureCount,
                                                self._pointCount)
        ptNames, ftNames = (self.features._getNamesNoGeneration(),
                            self.points._getNamesNoGeneration())
        self.points.setNames(ptNames, useLog=False)
        self.features.setNames(ftNames, useLog=False)

        handleLogging(useLog, 'prep', "transpose", self.getTypeString(),
                      Base.transpose)

    @property
    @limitedTo2D
    def T(self):
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

        self._shape = other._shape

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
        ret = self._copy_implementation('pythonlist')
        if len(self._shape) > 2:
            ret = numpy.reshape(ret, self._shape).tolist()
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

        elif (_dataHelpers._looksNumeric(replaceWith)
              or isinstance(replaceWith, str)):
            pass  # no modifications needed
        else:
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

        if order == 'point':
            ret = (a + ' | ' + b for a, b in itertools.product(pNames, fNames))
        else:
            ret = (b + ' | ' + a for a, b in itertools.product(fNames, pNames))
        return list(ret)

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
        if self._pointCount == 0:
            msg = "Can only flatten when there is one or more "
            msg += "points. This object has 0 points."
            raise ImproperObjectAction(msg)
        if self._featureCount == 0:
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
        self._featureCount = self._pointCount * self._featureCount
        self._pointCount = 1

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
        splitNames = [name.split(' | ') for name in possibleNames]
        if not all(len(split) == 2 for split in splitNames):
            return (None, None)

        pNames = []
        fNames = []
        allPtDefault = True
        allFtDefault = True
        for pName, fName in splitNames:
            if pName not in pNames:
                pNames.append(pName)
            if allPtDefault and not pName.startswith(DEFAULT_PREFIX):
                allPtDefault = False
            if fNames is not None:
                if fName not in fNames:
                    fNames.append(fName)
            if allFtDefault and not fName.startswith(DEFAULT_PREFIX):
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
        if self._featureCount == 0 or self._pointCount == 0:
            msg = "Cannot unflatten when there are 0 points or features."
            raise ImproperObjectAction(msg)
        if self._pointCount != 1 and self._featureCount != 1:
            msg = "Can only unflatten when there is only one point or feature."
            raise ImproperObjectAction(msg)
        if not isinstance(dataDimensions, (list, tuple)):
            raise InvalidArgumentType('dataDimensions must be a list or tuple')
        if len(dataDimensions) < 2:
            msg = "dataDimensions must contain a minimum of 2 values"
            raise InvalidArgumentValue(msg)
        if self.shape[0] * self.shape[1] != numpy.prod(dataDimensions):
            msg = "The product of the dimensions must be equal to the number "
            msg += "of values in this object"
            raise InvalidArgumentValue(msg)

        if len(dataDimensions) > 2:
            if order == 'feature':
                msg = "order='feature' is not allowed when unflattening to "
                msg += 'more than two dimensions'
                raise ImproperObjectAction(msg)
            shape2D = (dataDimensions[0], numpy.prod(dataDimensions[1:]))
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
            [[ a   1  id1 nan nan]
             [ b   2  id2 nan nan]
             [ c   3  id3  x   7 ]
             [nan nan id4  y   8 ]
             [nan nan id5  z   9 ]]
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
            [[ a   1  id1]
             [ b   2  id2]
             [ c   3  id3]
             [nan nan id4]
             [nan nan id5]]
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
            [[a 1 id1 nan nan]
             [b 2 id2 nan nan]
             [c 3 id3  x   7 ]]
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

        try:
            self._convertUnusableTypes(float, (int, float, bool), False)
        except ImproperObjectAction:
            self._numericValidation()
        try:
            other._convertUnusableTypes(float, (int, float, bool), False)
        except ImproperObjectAction:
            other._numericValidation(right=True)

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
        except TypeError:
            # help determine the source of the error
            self._numericValidation()
            other._numericValidation(right=True)
            raise # exception should be raised above, but just in case

        if caller._pointNamesCreated():
            ret.points.setNames(caller.points.getNames(), useLog=False)
        if callee._featureNamesCreated():
            ret.features.setNames(callee.features.getNames(), useLog=False)

        _dataHelpers.binaryOpNamePathMerge(caller, callee, ret, None, 'merge')

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
        if not isinstance(power, (int, numpy.int)):
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
                raise exceptionType(msg)

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
    def __pow__(self, other, z):
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
        ret._name = _dataHelpers.nextDefaultObjectName()

        return ret

    def __neg__(self):
        """
        Return this object where every element has been multiplied by -1
        """
        ret = self.copy()
        ret *= -1
        ret._name = _dataHelpers.nextDefaultObjectName()

        return ret

    def __abs__(self):
        """
        Perform element wise absolute value on this object
        """
        with self._treatAs2D():
            ret = self.calculateOnElements(abs, useLog=False)
        ret._shape = self._shape.copy()
        if self._pointNamesCreated():
            ret.points.setNames(self.points.getNames(), useLog=False)
        else:
            ret.points.setNames(None, useLog=False)
        if self._featureNamesCreated():
            ret.features.setNames(self.features.getNames(), useLog=False)
        else:
            ret.points.setNames(None, useLog=False)

        ret._name = _dataHelpers.nextDefaultObjectName()
        ret._absPath = self.absolutePath
        ret._relPath = self.relativePath
        return ret

    def _numericValidation(self, right=False):
        """
        Validate the object elements are all numeric.
        """
        try:
            self.calculateOnElements(_dataHelpers._checkNumeric, useLog=False)
        except ValueError:
            msg = "The object on the {0} contains non numeric data, "
            msg += "cannot do this operation"
            if right:
                msg = msg.format('right')
                raise InvalidArgumentValue(msg)
            msg = msg.format('left')
            raise ImproperObjectAction(msg)

    def _genericBinary_sizeValidation(self, opName, other):
        if self._shape != other._shape:
            msg = "The dimensions of the objects must be equal."
            raise InvalidArgumentValue(msg)
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
            else:
                msg = "Complex number results are not allowed"
                raise ImproperObjectAction(msg)
        # Test element type self
        self._numericValidation()
        # test element type other
        if isinstance(other, Base):
            other._numericValidation(right=True)

        raise # backup; should be diagnosed in _numericValidation

    def _genericBinary_validation(self, opName, other):
        otherBase = isinstance(other, Base)
        if not otherBase and not _dataHelpers._looksNumeric(other):
            msg = "'other' must be an instance of a nimble Base object or a "
            msg += "scalar"
            raise InvalidArgumentType(msg)
        if otherBase:
            self._genericBinary_sizeValidation(opName, other)
        # divmod operations inconsistently raise exceptions for zero division
        # it is more efficient to validate now than validate after operation
        if 'div' in opName or 'mod' in opName:
            self._validateDivMod(opName, other)

    def _genericBinary_axisNames(self, opName, other, usableTypes):
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
            other._convertUnusableTypes(float, usableTypes, False)
        except ImproperObjectAction:
            other._numericValidation(right=True)
        # everything else that uses this helper is a binary scalar op
        retPNames, retFNames = _dataHelpers.mergeNonDefaultNames(self,
                                                                other)
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
            return (n for n in names if not n.startswith(DEFAULT_PREFIX))

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

    def _convertUnusableTypes(self, convertTo, usableTypes, returnCopy=True):
        """
        Convert the data if necessary.

        Convert any type not in usableTypes to the convertTo type.
        Conversion is done inplace if returnCopy is set to False
        """
        try:
            ret = self._convertUnusableTypes_implementation(convertTo,
                                                            usableTypes)
        except (ValueError, TypeError):
            msg = 'Unable to coerce the data to the type required for this '
            msg += 'operation.'
            raise ImproperObjectAction(msg)
        if returnCopy:
            return ret
        self.data = ret

    def _genericBinaryOperations(self, opName, other):
        if 'pow' in opName:
            usableTypes = (float,)
        else:
            usableTypes = (int, float, bool)
        try:
            self._convertUnusableTypes(float, usableTypes, False)
        except ImproperObjectAction:
            self._numericValidation()
        if isinstance(other, Stretch):
            # __ipow__ does not work if return NotImplemented
            if opName == '__ipow__':
                return pow(self, other)
            return NotImplemented

        self._genericBinary_validation(opName, other)

        # figure out return obj's point / feature names
        otherBase = isinstance(other, Base)
        if otherBase:
            retPNames, retFNames = self._genericBinary_axisNames(opName, other,
                                                                 usableTypes)
        else:
            retPNames = self.points._getNamesNoGeneration()
            retFNames = self.features._getNamesNoGeneration()

        try:
            useOp = opName
            if opName.startswith('__i'):
                # inplace operations will modify the data even if op fails
                # use not inplace operation, setting to inplace occurs after
                useOp = opName[:2] + opName[3:]
            with numpy.errstate(divide='raise', invalid='raise'):
                ret = self._binaryOperations_implementation(useOp, other)
        except (TypeError, ValueError, FloatingPointError) as error:
            self._diagnoseFailureAndRaiseException(opName, other, error)

        ret._shape = self._shape
        if opName.startswith('__i'):
            absPath, relPath = self._absPath, self._relPath
            self.referenceDataFrom(ret, useLog=False)
            self._absPath, self._relPath = absPath, relPath
            ret = self
        ret.points.setNames(retPNames, useLog=False)
        ret.features.setNames(retFNames, useLog=False)

        nameSource = 'self' if opName.startswith('__i') else None
        pathSource = 'merge' if otherBase else 'self'
        _dataHelpers.binaryOpNamePathMerge(
            self, other, ret, nameSource, pathSource)
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

    def _logicalValidationAndConversion(self):
        if (not hasattr(self.data, 'dtype')
                or self.data.dtype not in [bool, numpy.bool_]):
            validValues = match.allValues([True, False, 0, 1])
            if not validValues(self):
                msg = 'logical operations can only be performed on data '
                msg += 'containing True, False, 0 and 1 values'
                raise ImproperObjectAction(msg)

            ret = self.matchingElements(lambda v: bool(v), useLog=False)
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
            [[1.000 4.000  9.000 ]
             [4.000 10.000 18.000]
             [0.000 -2.000 -6.000]]
            )

        Stretched feature with nimble Base object.

        >>> rawBase = [[1, 2, 3], [4, 5, 6], [0, -1, -2]]
        >>> rawFt = [[1], [2], [3]]
        >>> baseObj = nimble.data('Matrix', rawBase)
        >>> featObj = nimble.data('List', rawFt)
        >>> featObj.stretch + baseObj
        List(
            [[2.000 3.000 4.000]
             [6.000 7.000 8.000]
             [3.000 2.000 1.000]]
            )

        Two stretched objects.

        >>> rawPt = [[1, 2, 3]]
        >>> rawFt = [[1], [2], [3]]
        >>> pointObj = nimble.data('Matrix', rawPt)
        >>> featObj = nimble.data('List', rawFt)
        >>> pointObj.stretch - featObj.stretch
        Matrix(
            [[0.000  1.000  2.000]
             [-1.000 0.000  1.000]
             [-2.000 -1.000 0.000]]
            )
        >>> featObj.stretch - pointObj.stretch
        List(
            [[0.000 -1.000 -2.000]
             [1.000 0.000  -1.000]
             [2.000 1.000  0.000 ]]
            )
        """
        return Stretch(self)

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
        (tRowIDs, bRowIDs) = _dataHelpers.indicesSplit(maxRows,
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

        (tRowIDs, bRowIDs) = _dataHelpers.indicesSplit(maxDataRows,
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
        TODO: Should we place this function in _dataHelpers.py?
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

        return self.pointNames.keys() - other.pointNames.keys()

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

        return (self.featureNames.keys()
                - other.featureNames.keys())

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

        return self.pointNames.keys() ^ other.pointNames.keys()

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

        return (self.featureNames.keys()
                ^ other.featureNames.keys())

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

        return self.pointNames.keys() | other.pointNames.keys()

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

        return (self.featureNames.keys()
                | other.featureNames.keys())

    def _equalPointNames(self, other):
        return self._equalNames(self.points._getNamesNoGeneration(),
                                other.points._getNamesNoGeneration())

    def _equalFeatureNames(self, other):
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
                lnames = self.points._getNamesNoGeneration()
            else:
                lnames = self.features._getNamesNoGeneration()
            if rightAxis == 'point':
                rnames = other.points._getNamesNoGeneration()
            else:
                rnames = other.features._getNamesNoGeneration()
            if lnames is None or rnames is None:
                return
            inconsistencies = self._inconsistentNames(lnames, rnames)

            if inconsistencies != {}:
                table = [['left', 'ID', 'right']]
                for i in sorted(inconsistencies.keys()):
                    lname = '"' + lnames[i] + '"'
                    rname = '"' + rnames[i] + '"'
                    table.append([lname, str(i), rname])

                msg = leftAxis + " to " + rightAxis + " name inconsistencies "
                msg += "when calling left." + callSym + "(right) \n"
                msg += nimble.core.logger.tableString.tableString(table)
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
    def _replaceRectangle_implementation(self, replaceWith, pointStart,
                                         featureStart, pointEnd, featureEnd):
        pass

    @abstractmethod
    def _flatten_implementation(self, order):
        pass

    @abstractmethod
    def _unflatten_implementation(self, dataDimensions, order):
        pass

    @abstractmethod
    def _merge_implementation(self, other, point, feature, onFeature,
                              matchingFtIdx):
        pass

    @abstractmethod
    def _mul__implementation(self, other):
        pass

    @abstractmethod
    def _convertUnusableTypes_implementation(self, convertTo, usableTypes):
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
