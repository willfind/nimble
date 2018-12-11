"""

"""
from __future__ import absolute_import

import numpy
import six

import UML
from UML.exceptions import ArgumentException, ImproperActionException
from . import dataHelpers
from .dataHelpers import OPTRLIST, OPTRDICT
from .dataHelpers import valuesToPythonList

class Elements(object):
    """

    """
    def __init__(self):
        super(Elements, self).__init__()

    # TODO iterator???

    ###########################
    # Higher Order Operations #
    ###########################

    def calculate(self, function, points=None, features=None,
                  preserveZeros=False, skipNoneReturnValues=False,
                  outputType=None):
        """
        Return a new object with a calculation applied to each element.

        Calculates the results of the given function on the specified
        elements in this object, with output values collected into a new
        object that is returned upon completion.

        Parameters
        ----------
        function : function
            Take a value as input and return the desired value.
        points : point, list of points
            The subset of points to limit the calculation to. If None,
            the calculation will apply to all points.
        features : feature, list of features
            The subset of features to limit the calculation to. If None,
            the calculation will apply to all features.
        preserveZeros : bool
            Bypass calculation on zero values
        skipNoneReturnValues : bool
            Bypass values when ``function`` returns None. If False, the
            value None will replace the value if None is returned.
        outputType: UML data type
            Return an object of the specified type. If None, the
            returned object will have the same type as the calling
            object.

        Returns
        -------
        UML object

        See also
        --------
        transform : calculate inplace

        Examples
        --------
        TODO
        """
        oneArg = False
        try:
            function(0, 0, 0)
        except TypeError:
            oneArg = True

        if points is not None:
            points = self.source._constructIndicesList('point', points)
        if features is not None:
            features = self.source._constructIndicesList('feature', features)

        if outputType is not None:
            optType = outputType
        else:
            optType = self.source.getTypeString()

        # Use vectorized for functions with oneArg
        if oneArg:
            if not preserveZeros:
                # check if the function preserves zero values
                preserveZeros = function(0) == 0
            def functionWrap(value):
                if preserveZeros and value == 0:
                    return 0
                currRet = function(value)
                if skipNoneReturnValues and currRet is None:
                    return value

                return currRet

            vectorized = numpy.vectorize(functionWrap)
            ret = self._calculate_implementation(vectorized, points, features,
                                                 preserveZeros, optType)

        else:
            # if unable to vectorize, iterate over each point
            points = points if points else list(range(self.source.pts))
            features = features if features else list(range(self.source.fts))
            valueArray = numpy.empty([len(points), len(features)])
            p = 0
            for pi in points:
                f = 0
                for fj in features:
                    value = self.source[pi, fj]
                    if preserveZeros and value == 0:
                        valueArray[p, f] = 0
                    else:
                        if oneArg:
                            currRet = function(value)
                        else:
                            currRet = function(value, pi, fj)
                        if skipNoneReturnValues and currRet is None:
                            valueArray[p, f] = value
                        else:
                            valueArray[p, f] = currRet
                    f += 1
                p += 1

            ret = UML.createData(optType, valueArray)

        ret._absPath = self.source.absolutePath
        ret._relPath = self.source.relativePath

        self.source.validate()

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
        points.count, features.count

        Examples
        --------
        TODO
        """
        if callable(condition):
            ret = self.calculate(function=condition, outputType='Matrix')
        elif isinstance(condition, six.string_types):
            for optr in OPTRLIST:
                if optr in condition:
                    value = float(condition[len(optr):])
                    optr = '==' if optr == '=' else optr
                    optrOperator = OPTRDICT[optr]
                    break
            func = lambda x: optrOperator(x, value)
            ret = self.calculate(function=func, outputType='Matrix')
        else:
            msg = 'function can only be a function or str, not else'
            raise ArgumentException(msg)
        return int(numpy.sum(ret.data))

    def countUnique(self, points=None, features=None):
        """
        Count of each unique value in the data.

        Returns a dictionary containing each unique value as a key and
        the number of times that value occurs as the value.

        Parameters
        ----------
        points
            May be None indicating application to all points, a single
            name or index or an iterable of points and/or indices.
        features : identifier, list of identifiers
            May be None indicating application to all features, a single
            name or index or an iterable of names and/or indices.

        Returns
        -------
        dict

        See Also
        --------
        calculate.uniqueCount

        Examples
        --------
        TODO
        """
        uniqueCount = {}
        if points is None:
            points = [i for i in range(self.source.pts)]
        if features is None:
            features = [i for i in range(self.source.fts)]
        points = valuesToPythonList(points, 'points')
        features = valuesToPythonList(features, 'features')
        for i in points:
            for j in features:
                val = self.source[i, j]
                temp = uniqueCount.get(val,0)
                uniqueCount[val] = temp + 1

        return uniqueCount

    def multiply(self, other):
        """
        Multiply objects element-wise.

        Perform element-wise multiplication of this UML data object
        against the provided ``other`` UML data object, with the result
        being stored in-place in the calling object. Both objects must
        contain only numeric data. The pointCount and featureCount of
        both objects must be equal. The types of the two objects may be
        different.

        Parameters
        ----------
        other : UML object
            The object containing the elements to multiply with the
            elements in this object.

        Examples
        --------
        TODO
        """
        if not isinstance(other, UML.data.Base):
            msg = "'other' must be an instance of a UML data object"
            raise ArgumentException(msg)

        if self.source.pts != other.pts:
            msg = "The number of points in each object must be equal."
            raise ArgumentException(msg)
        if self.source.fts != other.fts:
            msg = "The number of features in each object must be equal."
            raise ArgumentException(msg)

        if self.source.pts == 0 or self.source.fts == 0:
            msg = "Cannot do elements.multiply with empty points or features"
            raise ImproperActionException(msg)

        self.source._validateEqualNames('point', 'point',
                                        'elements.multiply', other)
        self.source._validateEqualNames('feature', 'feature',
                                        'elements.multiply', other)

        try:
            self._multiply_implementation(other)
        except Exception as e:
            #TODO: improve how the exception is catch
            self.source._numericValidation()
            other._numericValidation()
            raise(e)

        retNames = dataHelpers.mergeNonDefaultNames(self.source, other)
        retPNames = retNames[0]
        retFNames = retNames[1]
        self.source.setPointNames(retPNames)
        self.source.setFeatureNames(retFNames)
        self.source.validate()

    def power(self, other):
        """
        Raise the elements of this object to a power.

        The power to raise each element to can be either a UML object or
        a single numerical value. If ``other`` is an object, both must
        contain only numeric data. The pointCount and featureCount of
        both objects must be equal. The types of the two objects may be
        different.

        Parameters
        ----------
        other : numerical value, UML object
            * numerical value - the power to raise each element to
            * The object containing the elements to raise the elements
              in this object to

        Examples
        --------
        TODO
        """
        # other is UML or single numerical value
        singleValue = dataHelpers._looksNumeric(other)
        if not singleValue and not isinstance(other, UML.data.Base):
            msg = "'other' must be an instance of a UML data object "
            msg += "or a single numeric value"
            raise ArgumentException(msg)

        if isinstance(other, UML.data.Base):
            # same shape
            if self.source.pts != other.pts:
                msg = "The number of points in each object must be equal."
                raise ArgumentException(msg)
            if self.source.fts != other.fts:
                msg = "The number of features in each object must be equal."
                raise ArgumentException(msg)

        if self.source.pts == 0 or self.source.fts == 0:
            msg = "Cannot do elements.power when points or features is emtpy"
            raise ImproperActionException(msg)

        if isinstance(other, UML.data.Base):
            def powFromRight(val, pnum, fnum):
                try:
                    return val ** other[pnum, fnum]
                except Exception as e:
                    self.source._numericValidation()
                    other._numericValidation()
                    raise(e)
            self.source.transformEachElement(powFromRight)
        else:
            def powFromRight(val, pnum, fnum):
                try:
                    return val ** other
                except Exception as e:
                    self.source._numericValidation()
                    other._numericValidation()
                    raise(e)
            self.source.transformEachElement(powFromRight)

        self.source.validate()

    ########################
    # Higher Order Helpers #
    ########################

    def _calculateForEachElementGenericVectorized(
            self, function, points, features, outputType):
        # need points/features as arrays for indexing
        if points:
            points = numpy.array(points)
        else:
            points = numpy.array(range(self.source.pts))
        if features:
            features = numpy.array(features)
        else:
            features = numpy.array(range(self.source.fts))
        toCalculate = self.source.copyAs('numpyarray')
        # array with only desired points and features
        toCalculate = toCalculate[points[:, None], features]
        try:
            values = function(toCalculate)
            # check if values has numeric dtype
            if numpy.issubdtype(values.dtype, numpy.number):
                return UML.createData(outputType, values)

            return UML.createData(outputType, values,
                                  elementType=numpy.object_)
        except Exception:
            # change output type of vectorized function to object to handle
            # nonnumeric data
            function.otypes = [numpy.object_]
            values = function(toCalculate)
            return UML.createData(outputType, values,
                                  elementType=numpy.object_)
