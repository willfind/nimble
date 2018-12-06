"""
TODO
"""
from __future__ import absolute_import

import numpy

import UML

class Elements(object):
    """
    TODO
    """
    def __init__(self, **kwds):
        self.source = kwds['source']
        super(Elements, self).__init__()

    # TODO iterator???

    def calculate(self, function, points=None, features=None,
                  preserveZeros=False, skipNoneReturnValues=False,
                  outputType=None):
        """
        Returns a new object containing the results of calling function(elementValue)
        or function(elementValue, pointNum, featureNum) for each element.

        points: Limit to only elements of the specified points; may be None for
        all points, a single identifier (name or index), or an iterable,
        list-like container of identifiers; this will affect the shape of the
        returned object.

        features: Limit to only elements of the specified features; may be None
        for all features, a single identifier (name or index), or an iterable,
        list-like container of identifiers; this will affect the shape of the
        returned object.

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
                else:
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
        toCalculate = toCalculate[points[:,None], features]
        try:
            values = function(toCalculate)
            # check if values has numeric dtype
            if numpy.issubdtype(values.dtype, numpy.number):
                return UML.createData(outputType, values)
            else:
                return UML.createData(outputType, values,
                                      elementType=numpy.object_)
        except Exception:
            # change output type of vectorized function to object to handle
            # nonnumeric data
            function.otypes = [numpy.object_]
            values = function(toCalculate)
            return UML.createData(outputType, values, elementType=numpy.object_)
