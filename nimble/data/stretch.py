"""
Stretch object TODO
"""
import numpy

import nimble
from nimble.exceptions import ImproperObjectAction
from .dataHelpers import createDataNoValidation
from nimble.importExternalLibraries import importModule

pd = importModule('pandas')

class Stretch(object):
    def __init__(self, source):
        if len(source.points) > 1 and len(source.features) > 1:
            msg = "Only one-dimensional objects can be stretched. This "
            msg += "object has shape " + str(source.shape)
            raise ImproperObjectAction(msg)
        self._source = source
        self._type = source.getTypeString()

    def _TODO_set_names(self, obj, other):
        obj.points.setNames(None)
        obj.features.setNames(None)

    def __add__(self, other):
        return self._genericArithmetic('__add__', other)

    def __radd__(self, other):
        return self._genericArithmetic('__radd__', other)

    def __sub__(self, other):
        return self._genericArithmetic('__sub__', other)

    def __rsub__(self, other):
        return self._genericArithmetic('__rsub__', other)

    def __mul__(self, other):
        return self._genericArithmetic('__mul__', other)

    def __rmul__(self, other):
        return self._genericArithmetic('__rmul__', other)

    def __truediv__(self, other):
        return self._genericArithmetic('__truediv__', other)

    def __rtruediv__(self, other):
        return self._genericArithmetic('__rtruediv__', other)

    def __floordiv__(self, other):
        return self._genericArithmetic('__floordiv__', other)

    def __rfloordiv__(self, other):
        return self._genericArithmetic('__rfloordiv__', other)

    def __mod__(self, other):
        return self._genericArithmetic('__mod__', other)

    def __rmod__(self, other):
        return self._genericArithmetic('__rmod__', other)

    def __pow__(self, other):
        return self._genericArithmetic('__pow__', other)

    def __rpow__(self, other):
        return self._genericArithmetic('__rpow__', other)

    def _genericArithmetic(self, opName, other):
        isNimble = isinstance(other, nimble.data.Base)
        if not (isNimble or isinstance(other, Stretch)):
            msg = 'stretch operations can only be performed with nimble '
            msg += 'Base objects or another Stretch object'
            raise ImproperObjectAction(msg)
        if isinstance(other, Stretch):
            sPts = len(self._source.points)
            sFts = len(self._source.features)
            oPts = len(other._source.points)
            oFts = len(other._source.features)
            if not ((sPts == 1 and oFts == 1) or (sFts == 1 and oPts == 1)):
                msg = "Operations using two stretched objects can only be "
                msg += "performed if one object is a single point and the "
                msg += "other is a single feature"
                raise ImproperObjectAction(msg)

        divNames = ['__truediv__', '__rtruediv__', '__floordiv__',
                    '__rfloordiv__', '__ifloordiv__', '__mod__', '__rmod__']
        if opName.startswith('__r'):
            toCheck = self._source
        else:
            if isNimble:
                toCheck = other
            else:
                toCheck = other._source
        if opName in divNames:
            if toCheck.containsZero():
                msg = "Cannot perform " + opName + " when the second argument "
                msg += "contains any zeros"
                raise ZeroDivisionError(msg)
            unique = toCheck.elements.countUnique()
            if any(val != val or numpy.isinf(val) for val in unique):
                msg = "Cannot perform " + opName + " when the second "
                msg += "argument contains any NaNs or Infs"
                raise InvalidArgumentValue(msg)

        return self._genericArithmetic_implementation(opName, other)

    def _genericArithmetic_implementation(self, opName, other):
        sourceData = self._source.copy('numpyarray')
        if isinstance(other, nimble.data.Base):
            ret = getattr(sourceData, opName)(other.copy('numpyarray'))
        else:
            ret = getattr(sourceData, opName)(other._source.copy('numpyarray'))
        data = numpy.matrix(ret)
        ret = createDataNoValidation(self._type, data)
        self._TODO_set_names(ret, other)
        return ret


class StretchSparse(Stretch):
    def _genericArithmetic_implementation(self, opName, other):
        if isinstance(other, nimble.data.Base):
            if self._source.shape[0] == 1:
                stretched = self._source.points.repeat(other.shape[0], True)
            else:
                stretched = self._source.features.repeat(other.shape[1], True)
            ret = getattr(stretched, opName)(other)
        # other is Stretch
        elif len(self._source.points) == 1:
            selfFts = len(self._source.features)
            otherPts = len(other._source.points)
            stretchPt = self._source.points.repeat(otherPts, True)
            stretchFt = other._source.features.repeat(selfFts, True)
            ret = getattr(stretchPt, opName)(stretchFt)
        else:
            selfPts = len(self._source.points)
            otherFts = len(other._source.features)
            stretchPt = other._source.points.repeat(selfPts, True)
            stretchFt = self._source.features.repeat(otherFts, True)
            ret = getattr(stretchFt, opName)(stretchPt)

        self._TODO_set_names(ret, other)

        return ret
