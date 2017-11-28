import cython
cdef extern from "math.h":
    cpdef double sin(double x)
    cpdef double cos(double x)

cpdef double hashCodeFunc(double elementValue, double pointNum, double featureNum)

cdef class Base:
    cdef public int _pointCount, _featureCount
    cdef public _nextDefaultValuePoint, _nextDefaultValueFeature, _name, pointNames, featureNames, pointNamesInverse, featureNamesInverse
    cdef public _absPath, _relPath
    cpdef _processSingleY(self, y)
    cpdef _processSingleX(self, x)

    cpdef _getpointCount(self)
    cpdef _getfeatureCount(self)
    cpdef _calculateForEach_implementation(self, function, included, axis)
    cpdef _arrangeDataWithLimits(self, maxWidth, maxHeight, sigDigits=*, maxStrLength=*, colSep=*, colHold=*, rowHold=*, strHold=*)
    cpdef _nextDefaultName(self, axis)
    cpdef _setAllDefault(self, axis)
    cpdef _validateAxis(self, axis)

    @cython.locals(avg=cython.double, bigNum=cython.long)
    cpdef hashCode(self)

    @cython.locals(p=cython.int, f=cython.int, pi=cython.int, fj=cython.int)
    cpdef calculateForEachElement(self, function, points=*, features=*, preserveZeros=*,skipNoneReturnValues=*, outputType=*)

