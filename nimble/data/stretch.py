"""
Stretch object to allow for broadcasting operations.
"""
import numpy

import nimble
from nimble.exceptions import InvalidArgumentValue, ImproperObjectAction
from nimble.importExternalLibraries import importModule
from . import dataHelpers
from .dataHelpers import createDataNoValidation

class Stretch(object):
    """
    Stretch a one-dimensional object along one axis.

    The stretched axis is detemined by the other object being used in
    conjunction with operation being called. All operations return a
    nimble Base subclass.

    Parameters
    ----------
    source : nimble Base object
        Must be one-dimensional.
    """
    def __init__(self, source):
        self._source = source
        self._type = source.getTypeString()
        self._numPts = len(source.points)
        self._numFts = len(source.features)
        if self._numPts > 1 and self._numFts > 1:
            msg = "Only one-dimensional objects can be stretched. This "
            msg += "object has shape " + str(source.shape)
            raise ImproperObjectAction(msg)
        if self._numPts == 0 or self._numFts == 0:
            msg = "Point or feature empty objects cannot be stretched"
            raise ImproperObjectAction(msg)

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

    def setOutputNames(self, toSet, other):
        """
        Set names of objects output from stretch operations.
        """
        sPtNames = self._source.points._getNamesNoGeneration()
        sFtNames = self._source.features._getNamesNoGeneration()
        setNumPts = len(toSet.points)
        setNumFts = len(toSet.features)
        oStretch = isinstance(other, Stretch)
        if oStretch:
            oPtNames = other._source.points._getNamesNoGeneration()
            oFtNames = other._source.features._getNamesNoGeneration()
            oNumPts = other._numPts
            oNumFts = other._numFts
        else:
            oPtNames = other.points._getNamesNoGeneration()
            oFtNames = other.features._getNamesNoGeneration()
            oNumPts = len(other.points)
            oNumFts = len(other.features)

        if sPtNames == oPtNames:
            setPts = sPtNames # includes both names as None
        elif sPtNames is None and oStretch and oNumPts != setNumPts:
            oName = oPtNames[0]
            setPts = [oName + '_' + str(i + 1) for i in range(setNumPts)]
        elif sPtNames is None:
            setPts = oPtNames
        elif oPtNames is None and (self._numPts > 1 or setNumPts == 1):
            setPts = sPtNames
        elif oPtNames is None:
            sName = sPtNames[0]
            setPts = [sName + '_' + str(i + 1) for i in range(setNumPts)]
        else:
            setPts = None

        if sFtNames == oFtNames:
            setFts = sFtNames # includes both names as None
        elif sFtNames is None and oStretch and oNumFts != setNumFts:
            oName = oFtNames[0]
            setFts = [oName + '_' + str(i + 1) for i in range(setNumFts)]
        elif sFtNames is None:
            setFts = oFtNames
        elif oFtNames is None and (self._numFts > 1 or setNumFts == 1):
            setFts = sFtNames
        elif oFtNames is None:
            sName = sFtNames[0]
            setFts = [sName + '_' + str(i + 1) for i in range(setNumFts)]
        else:
            setFts = None

        toSet.points.setNames(setPts)
        toSet.features.setNames(setFts)

    def _genericArithmetic_validation(self, opName, other):
        otherNimble = isinstance(other, nimble.data.Base)
        if not (otherNimble or isinstance(other, Stretch)):
            msg = 'stretch operations can only be performed with nimble '
            msg += 'Base objects or another Stretch object'
            raise ImproperObjectAction(msg)
        if isinstance(other, Stretch):
            validStretch1 = self._numPts == 1 and other._numFts == 1
            validStretch2 = self._numFts == 1 and other._numPts == 1
            if not (validStretch1 or validStretch2):
                msg = "Operations using two stretched objects can only be "
                msg += "performed if one object is a single point and the "
                msg += "other is a single feature"
                raise ImproperObjectAction(msg)

        if otherNimble:
            otherSource = other
        else:
            otherSource = other._source

        if self._numPts == 1:
            fullSelf = self._source.points.repeat(len(otherSource.points),
                                                  True)
        else:
            fullSelf = self._source.features.repeat(len(otherSource.features),
                                                    True)
        if otherNimble:
            fullOther = other
        elif other._numPts == 1:
            fullOther = otherSource.points.repeat(self._numPts, True)
        else:
            fullOther = otherSource.features.repeat(self._numFts, True)

        dataHelpers.arithmeticValidation(fullSelf, opName, fullOther)

    def _genericArithmetic(self, opName, other):
        self._genericArithmetic_validation(opName, other)
        ret = self._arithmetic_implementation(opName, other)
        self.setOutputNames(ret, other)

        return ret

    def _arithmetic_implementation(self, opName, other):
        sourceData = self._source.copy('numpyarray')
        if isinstance(other, nimble.data.Base):
            otherData = other.copy('numpyarray')
        else:
            otherData = other._source.copy('numpyarray')
        ret = getattr(sourceData, opName)(otherData)
        data = numpy.array(ret)
        ret = createDataNoValidation(self._type, data)

        return ret


class StretchSparse(Stretch):
    def _arithmetic_implementation(self, opName, other):
        if isinstance(other, nimble.data.Base):
            if self._source.shape[0] == 1:
                lhs = self._source.points.repeat(other.shape[0], True)
            else:
                lhs = self._source.features.repeat(other.shape[1], True)
            rhs = other.copy()
        # other is Stretch
        elif self._numPts == 1:
            selfFts = len(self._source.features)
            otherPts = len(other._source.points)
            lhs = self._source.points.repeat(otherPts, True)
            rhs = other._source.features.repeat(selfFts, True)
        else:
            selfPts = len(self._source.points)
            otherFts = len(other._source.features)
            rhs = other._source.points.repeat(selfPts, True)
            lhs = self._source.features.repeat(otherFts, True)
        # names may conflict so remove and set later
        lhs.points.setNames(None)
        lhs.features.setNames(None)
        rhs.points.setNames(None)
        rhs.features.setNames(None)
        ret = getattr(lhs, opName)(rhs)

        return ret
