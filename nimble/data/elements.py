"""
Define methods of the features attribute for Base objects.

Methods and helpers responsible for determining how each function will
operate over each element. This is the top level of this hierarchy and
methods in this object should attempt to handle operations related to
axis names here whenever possible. Additionally, any functionality
generic to object subtype should be included here with abstract methods
defined for object subtype specific implementations. Additionally, the
wrapping of function calls for the logger takes place in here.
"""

from __future__ import absolute_import
from abc import abstractmethod

import numpy
import six

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction
from nimble.logger import handleLogging
from . import dataHelpers
from .dataHelpers import valuesToPythonList, constructIndicesList


class Elements(object):
    """
    Differentiate how methods act on each element.

    Also includes abstract methods which will be required to perform
    data-type specific operations.

    Parameters
    ----------
    source : nimble Base object
        The object containing the elements.
    kwds
        Included due to best practices so args may automatically be
        passed further up into the hierarchy if needed.
    """
    def __init__(self, source, **kwds):
        self._source = source
        self._ptPosition = 0
        self._ftPosition = 0
        super(Elements, self).__init__(**kwds)

    def __iter__(self):
        return self

    def next(self):
        """
        Get next item
        """
        while self._ptPosition < len(self._source.points):
            while self._ftPosition < len(self._source.features):
                value = self._source[self._ptPosition, self._ftPosition]
                self._ftPosition += 1
                return value
            self._ptPosition += 1
            self._ftPosition = 0
        raise StopIteration

    def __next__(self):
        return self.next()

    #########################
    # Structural Operations #
    #########################

    def transform(self, toTransform, points=None, features=None,
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
        calculate, Points.transform, Features.transform

        Examples
        --------
        Simple transformation to all elements.

        >>> data = nimble.ones('Matrix', 5, 5)
        >>> data.elements.transform(lambda elem: elem + 1)
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
        >>> data.elements.transform(lambda elem: elem + 10,
        ...                         preserveZeros=True)
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
        >>> data.elements.transform(lambda elem: elem + 1,
        ...                         points=[0, 1], features=[0, 2])
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
        >>> dontSkip = nimble.createData('Matrix', raw)
        >>> dontSkip.elements.transform(addTenToEvens)
        >>> dontSkip
        Matrix(
            [[ nan   12.000  nan  ]
             [14.000  nan   16.000]
             [ nan   18.000  nan  ]]
            )
        >>> skip = nimble.createData('Matrix', raw)
        >>> skip.elements.transform(addTenToEvens,
        ...                         skipNoneReturnValues=True)
        >>> skip
        Matrix(
            [[1.000  12.000 3.000 ]
             [14.000 5.000  16.000]
             [7.000  18.000 9.000 ]]
            )
        """
        if points is not None:
            points = constructIndicesList(self._source, 'point', points)
        if features is not None:
            features = constructIndicesList(self._source, 'feature', features)

        transformer = validateElementFunction(toTransform, preserveZeros,
                                              skipNoneReturnValues,
                                              'toTransform')

        self._transform_implementation(transformer, points, features)

        handleLogging(useLog, 'prep', 'elements.transform',
                      self._source.getTypeString(), Elements.transform,
                      toTransform, points, features, preserveZeros,
                      skipNoneReturnValues)

    ###########################
    # Higher Order Operations #
    ###########################

    def calculate(self, toCalculate, points=None, features=None,
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

        See also
        --------
        transform, Points.calculate, Features.calculate

        Examples
        --------
        Simple calculation on all elements.

        >>> data = nimble.ones('Matrix', 5, 5)
        >>> twos = data.elements.calculate(lambda elem: elem + 1)
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
        >>> addTenDiagonal = data.elements.calculate(lambda x: x + 10,
        ...                                          preserveZeros=True)
        >>> addTenDiagonal
        Sparse(
            [[11.000   0      0      0      0   ]
             [  0    11.000   0      0      0   ]
             [  0      0    11.000   0      0   ]
             [  0      0      0    11.000   0   ]
             [  0      0      0      0    11.000]]
            )

        Calculate on a subset of points and features.

        >>> data = nimble.ones('List', 4, 4)
        >>> calc = data.elements.calculate(lambda elem: elem + 1,
        ...                                points=[0, 1],
        ...                                features=[0, 2])
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
        >>> data = nimble.createData('Matrix', raw)
        >>> dontSkip = data.elements.calculate(addTenToEvens)
        >>> dontSkip
        Matrix(
            [[ nan   12.000  nan  ]
             [14.000  nan   16.000]
             [ nan   18.000  nan  ]]
            )
        >>> skip = data.elements.calculate(addTenToEvens,
        ...                                skipNoneReturnValues=True)
        >>> skip
        Matrix(
            [[1.000  12.000 3.000 ]
             [14.000 5.000  16.000]
             [7.000  18.000 9.000 ]]
            )
        """
        calculator = validateElementFunction(toCalculate, preserveZeros,
                                             skipNoneReturnValues,
                                             'toCalculate')

        if points is not None:
            points = constructIndicesList(self._source, 'point', points)
        if features is not None:
            features = constructIndicesList(self._source, 'feature', features)

        if outputType is not None:
            optType = outputType
        else:
            optType = self._source.getTypeString()

        # Use vectorized for functions with oneArg
        if calculator.oneArg:
            vectorized = numpy.vectorize(calculator)
            ret = self._calculate_implementation(vectorized, points, features,
                                                 preserveZeros, optType)

        else:
            # if unable to vectorize, iterate over each point
            if not points:
                points = list(range(len(self._source.points)))
            if not features:
                features = list(range(len(self._source.features)))
            valueArray = numpy.empty([len(points), len(features)])
            p = 0
            for pi in points:
                f = 0
                for fj in features:
                    value = self._source[pi, fj]
                    if calculator.oneArg:
                        currRet = calculator(value)
                    else:
                        currRet = calculator(value, pi, fj)
                    valueArray[p, f] = currRet
                    f += 1
                p += 1

            ret = nimble.createData(optType, valueArray, useLog=False)

        ret._absPath = self._source.absolutePath
        ret._relPath = self._source.relativePath

        handleLogging(useLog, 'prep', 'elements.calculate',
                      self._source.getTypeString(), Elements.calculate,
                      toCalculate, points, features, preserveZeros,
                      skipNoneReturnValues, outputType)

        return ret

    def count(self, condition):
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
        >>> numGreaterThanZero = data.elements.count(greaterThanZero)
        >>> numGreaterThanZero
        5

        Using a string filter function.

        >>> numLessThanOne = data.elements.count("<1")
        >>> numLessThanOne
        20
        """
        if hasattr(condition, '__call__'):
            ret = self.calculate(condition, outputType='Matrix', useLog=False)
        elif isinstance(condition, six.string_types):
            func = lambda x: eval('x'+condition)
            ret = self.calculate(func, outputType='Matrix', useLog=False)
        else:
            msg = 'function can only be a function or string containing a '
            msg += 'comparison operator and a value'
            raise InvalidArgumentType(msg)
        return int(numpy.sum(ret.data))

    def countUnique(self, points=None, features=None):
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
        >>> unique = data.elements.countUnique()
        >>> unique
        {0.0: 20, 1.0: 5}

        Count for a subset of elements.

        >>> data = nimble.identity('Matrix', 5)
        >>> unique = data.elements.countUnique(points=0,
        ...                                    features=[0, 1, 2])
        >>> unique
        {0.0: 2, 1.0: 1}
        """
        return self._countUnique_implementation(points, features)

    ########################
    # Numerical Operations #
    ########################

    def multiply(self, other, useLog=None):
        """
        Multiply objects element-wise.

        Perform element-wise multiplication of this nimble Base object
        against the provided ``other`` nimble Base object, with the
        result being stored in-place in the calling object. Both objects
        must contain only numeric data. The pointCount and featureCount
        of both objects must be equal. The types of the two objects may
        be different.

        Parameters
        ----------
        other : nimble object
            The object containing the elements to multiply with the
            elements in this object.
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        >>> raw1 = [[4, 6],
        ...         [2, 3]]
        >>> raw2 = [[3, 2],
        ...         [6, 4]]
        >>> data1 = nimble.createData('Matrix', raw1)
        >>> data2 = nimble.createData('Matrix', raw2)
        >>> data1.elements.multiply(data2)
        >>> data1
        Matrix(
            [[12.000 12.000]
             [12.000 12.000]]
            )
        """
        if not isinstance(other, nimble.data.Base):
            msg = "'other' must be an instance of a nimble data object"
            raise InvalidArgumentType(msg)

        if len(self._source.points) != len(other.points):
            msg = "The number of points in each object must be equal."
            raise InvalidArgumentValue(msg)
        if len(self._source.features) != len(other.features):
            msg = "The number of features in each object must be equal."
            raise InvalidArgumentValue(msg)

        if len(self._source.points) == 0 or len(self._source.features) == 0:
            msg = "Cannot do elements.multiply with empty points or features"
            raise ImproperObjectAction(msg)

        self._source._validateEqualNames('point', 'point',
                                         'elements.multiply', other)
        self._source._validateEqualNames('feature', 'feature',
                                         'elements.multiply', other)

        try:
            self._multiply_implementation(other)
        except Exception as e:
            #TODO: improve how the exception is catch
            self._source._numericValidation()
            other._numericValidation(right=True)
            raise e

        retNames = dataHelpers.mergeNonDefaultNames(self._source, other)
        retPNames = retNames[0]
        retFNames = retNames[1]
        self._source.points.setNames(retPNames, useLog=False)
        self._source.features.setNames(retFNames, useLog=False)

        handleLogging(useLog, 'prep', 'elements.multiply',
                      self._source.getTypeString(), Elements.multiply, other)


    def power(self, other, useLog=None):
        """
        Raise the elements of this object to a power.

        The power to raise each element to can be either a nimble object
        or a single numerical value. If ``other`` is an object, both
        must contain only numeric data. The pointCount and featureCount
        of both objects must be equal. The types of the two objects may
        be different.

        Parameters
        ----------
        other : numerical value, nimble object
            * numerical value - the power to raise each element to
            * The object containing the elements to raise the elements
              in this object to
        useLog : bool, None
            Local control for whether to send object creation to the
            logger. If None (default), use the value as specified in the
            "logger" "enabledByDefault" configuration option. If True,
            send to the logger regardless of the global option. If
            False, do **NOT** send to the logger, regardless of the
            global option.

        Examples
        --------
        >>> raw1 = [[4, 8],
        ...         [2, 64]]
        >>> raw2 = [[3, 2],
        ...         [6, 1]]
        >>> data1 = nimble.createData('Matrix', raw1)
        >>> data2 = nimble.createData('Matrix', raw2)
        >>> data1.elements.power(data2)
        >>> data1
        Matrix(
            [[64.000 64.000]
             [64.000 64.000]]
            )
        """
        # other is nimble or single numerical value
        singleValue = dataHelpers._looksNumeric(other)
        if not singleValue and not isinstance(other, nimble.data.Base):
            msg = "'other' must be an instance of a nimble Base object "
            msg += "or a single numeric value"
            raise InvalidArgumentType(msg)

        if isinstance(other, nimble.data.Base):
            # same shape
            if len(self._source.points) != len(other.points):
                msg = "The number of points in each object must be equal."
                raise InvalidArgumentValue(msg)
            if len(self._source.features) != len(other.features):
                msg = "The number of features in each object must be equal."
                raise InvalidArgumentValue(msg)

        if len(self._source.points) == 0 or len(self._source.features) == 0:
            msg = "Cannot do elements.power when points or features is emtpy"
            raise ImproperObjectAction(msg)

        if isinstance(other, nimble.data.Base):
            def powFromRight(val, pnum, fnum):
                try:
                    return val ** other[pnum, fnum]
                except Exception as e:
                    self._source._numericValidation()
                    other._numericValidation(right=True)
                    raise e
            self.transform(powFromRight, useLog=False)
        else:
            def powFromRight(val, pnum, fnum):
                try:
                    return val ** other
                except Exception as e:
                    self._source._numericValidation()
                    other._numericValidation(right=True)
                    raise e
            self.transform(powFromRight, useLog=False)

        handleLogging(useLog, 'prep', 'elements.power',
                      self._source.getTypeString(), Elements.power, other)

    ########################
    # Higher Order Helpers #
    ########################

    def _calculate_genericVectorized(
            self, function, points, features, outputType):
        # need points/features as arrays for indexing
        if points:
            points = numpy.array(points)
        else:
            points = numpy.array(range(len(self._source.points)))
        if features:
            features = numpy.array(features)
        else:
            features = numpy.array(range(len(self._source.features)))
        toCalculate = self._source.copy(to='numpyarray')
        # array with only desired points and features
        toCalculate = toCalculate[points[:, None], features]
        try:
            values = function(toCalculate)
            # check if values has numeric dtype
            if numpy.issubdtype(values.dtype, numpy.number):
                return nimble.createData(outputType, values, useLog=False)

            return nimble.createData(outputType, values,
                                     elementType=numpy.object_, useLog=False)
        except Exception:
            # change output type of vectorized function to object to handle
            # nonnumeric data
            function.otypes = [numpy.object_]
            values = function(toCalculate)
            return nimble.createData(outputType, values,
                                     elementType=numpy.object_, useLog=False)

    #####################
    # Abstract Methods  #
    #####################

    @abstractmethod
    def _calculate_implementation(self, function, points, features,
                                  preserveZeros, outputType):
        pass

    @abstractmethod
    def _multiply_implementation(self, other):
        pass

    @abstractmethod
    def _transform_implementation(self, toTransform, points, features,
                                  preserveZeros, skipNoneReturnValues):
        pass

###########
# Helpers #
###########

def validateElementFunction(func, preserveZeros, skipNoneReturnValues,
                            funcName):
    def elementValidated(value, *args):
        if preserveZeros and value == 0:
            return 0
        ret = func(value, *args)
        if skipNoneReturnValues and ret is None:
            return value
        if not dataHelpers.isAllowedSingleElement(ret):
            msg = funcName + " can only return numeric, boolean, or string "
            msg += "values, but the returned value was " + str(type(ret))
            raise InvalidArgumentValue(msg)
        return ret

    if isinstance(func, dict):
        func = getDictionaryMappingFunction(func)
    try:
        func(0, 0, 0)
        oneArg = False

        def wrappedElementFunction(value, i, j):
            return elementValidated(value, i, j)

    except TypeError:
        oneArg = True
        # see if we can preserve zeros even if not explicitly set
        try:
            if not preserveZeros and func(0) == 0:
                preserveZeros = True
        except TypeError:
            pass

        def wrappedElementFunction(value):
            return elementValidated(value)

    wrappedElementFunction.oneArg = oneArg
    wrappedElementFunction.preserveZeros = preserveZeros

    return wrappedElementFunction

def getDictionaryMappingFunction(dictionary):
    def valueMappingFunction(value):
        if value in dictionary:
            return dictionary[value]
        return value
    return valueMappingFunction
