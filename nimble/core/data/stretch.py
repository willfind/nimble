
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
Stretch object to allow for broadcasting operations.
"""
import numpy as np

import nimble
from nimble.exceptions import ImproperObjectAction
from nimble.exceptions import InvalidArgumentValueCombination
from ._dataHelpers import mergeNames

class Stretch(object):
    """
    Stretch a one-dimensional Base object along one axis.

    The stretched axis is determined by the other object being used in
    conjunction with operation being called. All operations return a
    nimble Base subclass.

    Parameters
    ----------
    source : nimble Base object
        Must be one-dimensional.
    """
    def __init__(self, source):
        self._source = source
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
        return self._stretchArithmetic('__add__', other)

    def __radd__(self, other):
        return self._stretchArithmetic('__radd__', other)

    def __sub__(self, other):
        return self._stretchArithmetic('__sub__', other)

    def __rsub__(self, other):
        return self._stretchArithmetic('__rsub__', other)

    def __mul__(self, other):
        return self._stretchArithmetic('__mul__', other)

    def __rmul__(self, other):
        return self._stretchArithmetic('__rmul__', other)

    def __truediv__(self, other):
        return self._stretchArithmetic('__truediv__', other)

    def __rtruediv__(self, other):
        return self._stretchArithmetic('__rtruediv__', other)

    def __floordiv__(self, other):
        return self._stretchArithmetic('__floordiv__', other)

    def __rfloordiv__(self, other):
        return self._stretchArithmetic('__rfloordiv__', other)

    def __mod__(self, other):
        return self._stretchArithmetic('__mod__', other)

    def __rmod__(self, other):
        return self._stretchArithmetic('__rmod__', other)

    def __pow__(self, other):
        return self._stretchArithmetic('__pow__', other)

    def __rpow__(self, other):
        return self._stretchArithmetic('__rpow__', other)

    def _getOutputNames(self, other):
        """
        Set names of objects output from stretch operations.
        """
        sPtAxis = self._source.points
        sFtAxis = self._source.features
        oStretch = isinstance(other, Stretch)
        if oStretch:
            oPtAxis = other._source.points
            oFtAxis = other._source.features
            oNumPts = other._numPts
            oNumFts = other._numFts
        else:
            oPtAxis = other.points
            oFtAxis = other.features
            oNumPts = len(other.points)
            oNumFts = len(other.features)
        setNumPts = max(self._numPts, oNumPts)
        setNumFts = max(self._numFts, oNumFts)

        def getNames(sAxis, oAxis, sNum, oNum, setNum):
            sNames = sAxis._getNamesNoGeneration()
            oNames = oAxis._getNamesNoGeneration()
            sDefault = sAxis._allDefaultNames()
            oDefault = oAxis._allDefaultNames()
            if sNames == oNames: # includes both names are None
                return sNames
            if sDefault and oStretch and oNum != setNum and oNames:
                oName = oNames[0]
                return [oName + '_' + str(i + 1) for i in range(setNum)]
            if sNames and sDefault and not oNames:
                return sNames
            if sDefault:
                return oNames
            if oDefault and (sNum > 1 or setNum == 1):
                return sNames
            if oDefault:
                sName = sNames[0]
                return [sName + '_' + str(i + 1) for i in range(setNum)]
            if len(sNames) == len(oNames): # some default names present
                return mergeNames(sNames, oNames)
            return None

        setPts = getNames(sPtAxis, oPtAxis, self._numPts, oNumPts, setNumPts)
        setFts = getNames(sFtAxis, oFtAxis, self._numFts, oNumFts, setNumFts)

        return setPts, setFts

    def _stretchArithmetic_validation(self, opName, other):
        otherBase = isinstance(other, nimble.core.data.Base)
        if not (otherBase or isinstance(other, Stretch)):
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
            # divmod operations inconsistently raise errors for zero division
            if 'div' in opName or 'mod' in opName:
                self._source._validateDivMod(opName, other._source)
        else:
            stretchPossible = True
            if self._numPts == 1 and self._numFts == 1:
                matchAxis = 'point' if len(other.points) > 1 else 'feature'
                if not 1 in other.shape:
                    stretchPossible = False
            elif self._numPts == 1:
                matchAxis = 'feature'
                if self._numFts != len(other.features):
                    stretchPossible = False
            else:
                matchAxis = 'point'
                if self._numPts != len(other.points):
                    stretchPossible = False

            if not stretchPossible:
                msg = "Unable to stretch this object to fit. The lengths of "
                msg += "one of the axes must align between objects"
                raise InvalidArgumentValueCombination(msg)

            self._source._validateEqualNames(matchAxis, matchAxis, opName,
                                             other)
            # divmod operations inconsistently raise errors for zero division
            if 'div' in opName or 'mod' in opName:
                self._source._validateDivMod(opName, other)

    def _stretchArithmetic(self, opName, other):
        conversionKwargs = {}
        if 'pow' in opName:
            conversionKwargs['allowInt'] = False
            conversionKwargs['allowBool'] = False
        try:
            self._source._convertToNumericTypes(**conversionKwargs)
        except ImproperObjectAction:
            self._source._numericValidation()
        if isinstance(other, Stretch):
            otherSource = other._source
        else:
            otherSource = other
        try:
            otherSource._convertToNumericTypes(**conversionKwargs)
        except ImproperObjectAction:
            otherSource._numericValidation(right=True)

        self._stretchArithmetic_validation(opName, other)

        try:
            with np.errstate(divide='raise', invalid='raise'):
                ret = self._stretchArithmetic_implementation(opName, other)
        except (TypeError, ValueError, FloatingPointError) as error:
            self._source._diagnoseFailureAndRaiseException(opName, otherSource,
                                                           error)
            raise # bakckup should be diagnosed and raised above

        if (opName.startswith('__r')
                and ret.getTypeString() != other.getTypeString()):
            ret = ret.copy(other.getTypeString())

        setPts, setFts = self._getOutputNames(other)
        ret.points.setNames(setPts, useLog=False)
        ret.features.setNames(setFts, useLog=False)

        return ret

    def _stretchArithmetic_implementation(self, opName, other):
        if isinstance(other, Stretch):
            other = other._source
        return self._source._binaryOperations_implementation(opName, other)

    def __and__(self, other):
        return self._stretchLogical('__and__', other)

    def __or__(self, other):
        return self._stretchLogical('__or__', other)

    def __xor__(self, other):
        return self._stretchLogical('__xor__', other)

    def _stretchLogical(self, opName, other):
        self._stretchArithmetic_validation(opName, other)
        if isinstance(other, Stretch):
            oBase = other._source
        else:
            oBase = other
        lhsBool = self._source._logicalValidationAndConversion()
        rhsBool = oBase._logicalValidationAndConversion()

        return lhsBool.stretch._stretchArithmetic_implementation(opName,
                                                                 rhsBool)

class StretchSparse(Stretch):
    """
    Stretch object optimized for Sparse operations.
    """
    def _stretchArithmetic_implementation(self, opName, other):
        if not isinstance(other, Stretch):
            if self._source.shape[0] == 1 and other.shape[0] > 1:
                lhs = self._source.points.repeat(other.shape[0], True,
                                                 useLog=False)
            elif self._source.shape[1] == 1 and other.shape[1] > 1:
                lhs = self._source.features.repeat(other.shape[1], True,
                                                   useLog=False)
            else:
                lhs = self._source
            rhs = other.copy()
        # other is Stretch
        elif self._numPts == 1:
            selfFts = len(self._source.features)
            otherPts = len(other._source.points)
            lhs = self._source.points.repeat(otherPts, True, useLog=False)
            rhs = other._source.features.repeat(selfFts, True, useLog=False)
        else:
            selfPts = len(self._source.points)
            otherFts = len(other._source.features)
            rhs = other._source.points.repeat(selfPts, True, useLog=False)
            lhs = self._source.features.repeat(otherFts, True, useLog=False)

        return lhs._binaryOperations_implementation(opName, rhs)
