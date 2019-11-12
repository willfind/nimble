"""
Tests for using stretch attribute to trigger broadcasting operations
"""
import operator

from nose.tools import raises
import numpy

import nimble
from nimble.exceptions import ImproperObjectAction, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.randomness import pythonRandom
from .baseObject import DataTestObject
from ..assertionHelpers import assertNoNamesGenerated

class StretchDataSafe(DataTestObject):

    ##############
    # Exceptions #
    ##############

    @raises(ImproperObjectAction)
    def test_stretch_exception_ptEmpty(self):
        empty = numpy.empty((0, 3))
        emptyObj = self.constructor(empty)
        emptyObj.stretch

    @raises(ImproperObjectAction)
    def test_stretch_exception_ftEmpty(self):
        empty = numpy.empty((3, 0))
        emptyObj = self.constructor(empty)
        emptyObj.stretch

    @raises(ImproperObjectAction)
    def test_stretch_exception_2D(self):
        toStretch = self.constructor([[1, 2], [3, 4]])
        toStretch.stretch

    @raises(InvalidArgumentValueCombination)
    def test_stretch_exception_shapeMismatch_point(self):
        toStretch = self.constructor([1, 1, 1])
        baseObj = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8]])

        toStretch.stretch + baseObj

    @raises(InvalidArgumentValueCombination)
    def test_stretch_exception_shapeMismatch_feature(self):
        toStretch = self.constructor([[1], [1], [1]])
        baseObj = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8]])

        toStretch.stretch + baseObj

    @raises(InvalidArgumentValueCombination)
    def test_stretch_exception_1x1(self):
        toStretch = self.constructor([1])
        baseObj = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8]])

        toStretch.stretch + baseObj

    ######################
    # Base / Stretch 1x1 #
    ######################

    def test_stretch_1x1_withVectors(self):
        toStretch = self.constructor([1])
        pointVect = self.constructor([1, 2, 3, 4])
        exp1 = self.constructor([2, 3, 4, 5])

        ret1 = toStretch.stretch + pointVect
        assert ret1 == exp1

        featureVect = self.constructor([[1], [2], [3], [4]])
        exp2 = self.constructor([[0], [-1], [-2], [-3]])

        ret2 = toStretch.stretch - featureVect
        assert ret2 == exp2

    ########################
    # Base / Stretch Point #
    ########################

    # zero safe

    def test_handmade_Base_Stretch_add_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])
        exp = self.constructor([[2, 4, 3, 3], [6, 8, 7, 7], [1, 1, -2, -4]])
        ret1 = base + toStretch.stretch
        ret2 = toStretch.stretch + base

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)
        assertNoNamesGenerated(base)

    def test_handmade_Base_Stretch_sub_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[0, 0, 3, 5], [4, 4, 7, 9], [-1, -3, -2, -2]])
        exp_r = self.constructor([[0, 0, -3, -5], [-4, -4, -7, -9], [1, 3, 2, 2]])
        ret1 = base - toStretch.stretch
        ret2 = toStretch.stretch - base

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base)

    def test_handmade_Base_Stretch_mul_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])
        exp = self.constructor([[1, 4, 0, -4], [5, 12, 0, -8], [0, -2, 0, 3]])
        ret1 = base * toStretch.stretch
        ret2 = toStretch.stretch * base

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)
        assertNoNamesGenerated(base)

    # zero exception

    def test_handmade_Base_Stretch_truediv_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])
        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        stretch2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        exp_r = self.constructor([[1, 1, (-1/3), (-1/2)], [(1/5), (1/3), (-1/7), (-1/4)],
                                  [-1, -1, (1/3), (1/2)]])
        ret1 = base1 / stretch1.stretch
        ret2 = stretch1.stretch / base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base1)

        try:
            base2 / stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch / base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_floordiv_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])
        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        stretch2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        exp_r = self.constructor([[1, 1, -1, -1], [0, 0, -1, -1], [-1, -1, 0, 0]])
        ret1 = base1 // stretch1.stretch
        ret2 = stretch1.stretch // base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base1)

        try:
            base2 // stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch // base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_mod_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])
        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        stretch2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        exp_r = self.constructor([[0, 0, 2, 2], [1, 2, 6, 6], [0, 0, -1, -2]])
        ret1 = base1 % stretch1.stretch
        ret2 = stretch1.stretch % base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base1)

        try:
            base2 % stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch % base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_pow_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])
        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, 0]])
        stretch2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 4, (1/3), (1/16)], [5, 36, (1/7), (1/64)], [-1, 4, (-1/3), (1/16)]])
        exp_r = self.constructor([[1, 4, -1, 16], [1, 64, -1, 256], [1, (1/4), -1, (1/16)]])
        ret1 = base1 ** stretch1.stretch
        ret2 = stretch1.stretch ** base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base1)

        try:
            base2 ** stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch ** base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    ##########################
    # Base / Stretch Feature #
    ##########################

    # zero safe

    def test_handmade_Base_Stretch_add_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])
        exp = self.constructor([[2, 3, 4], [6, 7, 8], [7, 8, 9], [-1, -2, -3]])
        ret1 = base + toStretch.stretch
        ret2 = toStretch.stretch + base

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)
        assertNoNamesGenerated(base)

    def test_handmade_Base_Stretch_sub_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[0, 1, 2], [2, 3, 4], [7, 8, 9], [1, 0, -1]])
        exp_r = self.constructor([[0, -1, -2], [-2, -3, -4], [-7, -8, -9], [-1, 0, 1]])
        ret1 = base - toStretch.stretch
        ret2 = toStretch.stretch - base

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base)

    def test_handmade_Base_Stretch_mul_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])
        exp = self.constructor([[1, 2, 3], [8, 10, 12], [0, 0, 0], [0, 1, 2]])
        ret1 = base * toStretch.stretch
        ret2 = toStretch.stretch * base

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)
        assertNoNamesGenerated(base)

    # zero exception

    def test_handmade_Base_Stretch_truediv_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])
        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[1, 2, 3], [2, (5/2), 3], [-7, -8, -9], [(1/2), 1, (3/2)]])
        exp_r = self.constructor([[1, (1/2), (1/3)], [(1/2), (2/5), (1/3)],
                                  [(-1/7), (-1/8), (-1/9)], [2, 1, (2/3)]])
        ret1 = base1 / stretch1.stretch
        ret2 = stretch1.stretch / base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base1)

        try:
            base2 / stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch / base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_floordiv_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])
        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[1, 2, 3], [2, 2, 3], [-7, -8, -9], [0, 1, 1]])
        exp_r = self.constructor([[1, 0, 0], [0, 0, 0], [-1, -1, -1], [2, 1, 0]])
        ret1 = base1 // stretch1.stretch
        ret2 = stretch1.stretch // base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base1)

        try:
            base2 // stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch // base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_mod_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])
        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[0, 0, 0], [0, 1, 0], [0, 0, 0], [-1, 0, -1]])
        exp_r = self.constructor([[0, 1, 1], [2, 2, 2], [6, 7, 8], [0, 0, -2]])
        ret1 = base1 % stretch1.stretch
        ret2 = stretch1.stretch % base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base1)

        try:
            base2 % stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch % base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_pow_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])
        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, -1], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[1, 2, 3], [16, 25, 36], [(1/7), (1/8), (1/9)], [1, (1/4), (1/9)]])
        exp_r = self.constructor([[1, 1, 1], [16, 32, 64], [-1, 1, -1], [(-1/2), (1/4), (-1/8)]])
        ret1 = base1 ** stretch1.stretch
        ret2 = stretch1.stretch ** base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(base1)

        try:
            base2 ** stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch ** base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    #####################
    # Stretch / Stretch #
    #####################
    @raises(ImproperObjectAction)
    def test_stretch_stretch_bothPoints_exception(self):
        pt1 = self.constructor([[1, 2]])
        pt2 = self.constructor([[3, 4]])

        pt1.stretch + pt2.stretch

    @raises(ImproperObjectAction)
    def test_stretch_stretch_bothFeatures_exception(self):
        ft1 = self.constructor([[1], [2]])
        ft2 = self.constructor([[3], [4]])

        ft1.stretch + ft2.stretch

    # zero safe

    def test_handmade_Stretch_Stretch_add(self):
        ft = self.constructor([[0], [1], [2], [-1]])
        pt = self.constructor([[1, 2, 0, -1]])
        exp = self.constructor([[1, 2, 0, -1], [2, 3, 1, 0], [3, 4, 2, 1], [0, 1, -1, -2]])
        ret1 = ft.stretch + pt.stretch
        ret2 = pt.stretch + ft.stretch
        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)

    def test_handmade_Stretch_Stretch_sub(self):
        ft = self.constructor([[0], [1], [2], [-1]])
        pt = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[-1, -2, 0, 1], [0, -1, 1, 2], [1, 0, 2, 3], [-2, -3, -1, 0]])
        exp_r = self.constructor([[1, 2, 0, -1], [0, 1, -1, -2], [-1, 0, -2, -3], [2, 3, 1, 0]])
        ret1 = ft.stretch - pt.stretch
        ret2 = pt.stretch - ft.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)

    def test_handmade_Stretch_Stretch_mul(self):
        ft = self.constructor([[0], [1], [2], [-1]])
        pt = self.constructor([[1, 2, 0, -1]])
        exp = self.constructor([[0, 0, 0, 0], [1, 2, 0, -1], [2, 4, 0, -2], [-1, -2, 0, 1]])
        ret1 = ft.stretch * pt.stretch
        ret2 = pt.stretch * ft.stretch

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)

    # zero exception

    def test_handmade_Stretch_Stretch_truediv(self):
        ft1 = self.constructor([[1], [2], [3], [-1]])
        pt1 = self.constructor([[1, 2, -2, -1]])
        ft2 = self.constructor([[0], [1], [2], [-1]])
        pt2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, (1/2), (-1/2), -1], [2, 1, -1, -2],
                                  [3, (3/2), (-3/2), -3], [-1, (-1/2), (1/2), 1]])
        exp_r = self.constructor([[1, 2, -2, -1], [(1/2), 1, -1, (-1/2)],
                                  [(1/3), (2/3), (-2/3), (-1/3)], [-1, -2, 2, 1]])
        ret1 = ft1.stretch / pt1.stretch
        ret2 = pt1.stretch / ft1.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)

        try:
            ft2.stretch / pt2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            pt2.stretch / ft2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Stretch_Stretch_floordiv(self):
        ft1 = self.constructor([[1], [2], [3], [-1]])
        pt1 = self.constructor([[1, 2, -2, -1]])
        ft2 = self.constructor([[0], [1], [2], [-1]])
        pt2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 0, -1, -1], [2, 1, -1, -2], [3, 1, -2, -3], [-1, -1, 0, 1]])
        exp_r = self.constructor([[1, 2, -2, -1], [0, 1, -1, -1], [0, 0, -1, -1], [-1, -2, 2, 1]])
        ret1 = ft1.stretch // pt1.stretch
        ret2 = pt1.stretch // ft1.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)

        try:
            ft2.stretch // pt2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            pt2.stretch // ft2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Stretch_Stretch_mod(self):
        ft1 = self.constructor([[1], [2], [3], [-1]])
        pt1 = self.constructor([[1, 2, -2, -1]])
        ft2 = self.constructor([[0], [1], [2], [-1]])
        pt2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[0, 1, -1, 0], [0, 0, 0, 0], [0, 1, -1, 0], [0, 1, -1, 0]])
        exp_r = self.constructor([[0, 0, 0, 0], [1, 0, 0, 1], [1, 2, 1, 2], [0, 0, 0, 0]])
        ret1 = ft1.stretch % pt1.stretch
        ret2 = pt1.stretch % ft1.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)

        try:
            ft2.stretch % pt2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            pt2.stretch % ft2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Stretch_Stretch_pow(self):
        ft1 = self.constructor([[1], [2], [3], [-1]])
        pt1 = self.constructor([[1, 2, -2, -1]])
        ft2 = self.constructor([[0], [1], [-1], [0]])
        pt2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 1, 1, 1], [2, 4, (1/4), (1/2)], [3, 9, (1/9), (1/3)], [-1, 1, 1, -1]])
        exp_r = self.constructor([[1, 2, -2, -1], [1, 4, 4, 1], [1, 8, -8, -1], [1, (1/2), (-1/2), -1]])
        ret1 = ft1.stretch ** pt1.stretch
        ret2 = pt1.stretch ** ft1.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)

        try:
            ft2.stretch ** pt2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            pt2.stretch ** ft2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_stretch_differentObjectTypes(self):
        matrixObj = self.constructor([[1, 2, 3], [4, 5, 6]])
        pointVect = self.constructor([9, 8, 7])
        featureVect = self.constructor([[8], [9]])
        otherTypes = [t for t in nimble.data.available if t != matrixObj.getTypeString()]
        for oType in otherTypes:
            possibleOps = [operator.add, operator.sub, operator.mul, operator.truediv,
                           operator.floordiv, operator.mod, operator.pow]
            randOp = possibleOps[pythonRandom.randint(0, 6)]
            matDiff = matrixObj.copy(oType)
            pvDiff = pointVect.copy(oType)
            fvDiff = featureVect.copy(oType)

            out1 = randOp(matrixObj, pvDiff.stretch)
            assert out1.getTypeString() == matrixObj.getTypeString()

            out2 = randOp(matrixObj, fvDiff.stretch)
            assert out2.getTypeString() == matrixObj.getTypeString()

            out3 = randOp(pointVect.stretch, matDiff)
            assert out3.getTypeString() == pointVect.getTypeString()

            out4 = randOp(featureVect.stretch, matDiff)
            assert out4.getTypeString() == pointVect.getTypeString()

            out5 = randOp(pointVect.stretch, fvDiff.stretch)
            assert out5.getTypeString() == pointVect.getTypeString()

            out6 = randOp(featureVect.stretch, pvDiff.stretch)
            assert out5.getTypeString() == pointVect.getTypeString()

    def back_stretchSetNames(self, obj1, obj2, expPts, expFts):
        # random operation for each, the output is not important only the names
        possibleOps = [operator.add, operator.sub, operator.mul, operator.truediv,
                       operator.floordiv, operator.mod, operator.pow]
        op1 = possibleOps[pythonRandom.randint(0, 6)]
        op2 = possibleOps[pythonRandom.randint(0, 6)]
        ret1 = op1(obj1, obj2)
        ret2 = op2(obj2, obj1)

        if expPts is None:
            assert not ret1.points._namesCreated()
            assert not ret2.points._namesCreated()
        else:
            assert ret1.points.getNames() == expPts
            assert ret2.points.getNames() == expPts
        if expFts is None:
            assert not ret1.features._namesCreated()
            assert not ret2.features._namesCreated()
        else:
            assert ret1.features.getNames() == expFts
            assert ret2.features.getNames() == expFts

    def back_stretchSetNamesException(self, obj1, obj2):
        # random operation for each, the output is not important only the names
        possibleOps = [operator.add, operator.sub, operator.mul, operator.truediv,
                       operator.floordiv, operator.mod, operator.pow]
        op1 = possibleOps[pythonRandom.randint(0, 6)]
        op2 = possibleOps[pythonRandom.randint(0, 6)]

        try:
            ret1 = op1(obj1, obj2)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

        try:
            ret2 = op2(obj2, obj1)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

    def test_stretchSetNames(self):
        pNames = ['p0', 'p1', 'p2']
        fNames = ['f0', 'f1', 'f2', 'f3']
        offFNames = ['fA', 'fB', 'fC', 'fD']
        offPNames = ['pA', 'pB', 'pC']
        single = ['s']
        single3 = ['s_1', 's_2', 's_3']
        single4 = ['s_1', 's_2', 's_3', 's_4']

        baseRaw = [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]]
        baseNoNames = self.constructor(baseRaw)
        basePtNames = self.constructor(baseRaw, pointNames=pNames)
        baseFtNames = self.constructor(baseRaw, featureNames=fNames)
        baseAllNames = self.constructor(baseRaw, pointNames=pNames, featureNames=fNames)

        stretchPt_Raw = [1, 2, -1, -2]
        stretchPt_NoNames = self.constructor(stretchPt_Raw).stretch
        pt_DefaultFtNames = self.constructor(stretchPt_Raw)
        defFtNames = pt_DefaultFtNames.features.getNames()
        assert pt_DefaultFtNames.features._namesCreated()
        stretchPt_DefaultFtNames = pt_DefaultFtNames.stretch
        stretchPt_MatchFtNames = self.constructor(stretchPt_Raw, featureNames=fNames).stretch
        stretchPt_NoMatchFtNames = self.constructor(stretchPt_Raw, featureNames=offFNames).stretch
        stretchPt_WithPtName = self.constructor(stretchPt_Raw, pointNames=single).stretch
        stretchPt_AllNamesFtMatch = self.constructor(stretchPt_Raw, pointNames=single,
                                                  featureNames=fNames).stretch
        stretchPt_AllNamesNoFtMatch = self.constructor(stretchPt_Raw, pointNames=single,
                                                    featureNames=offFNames).stretch

        stretchFt_Raw = [[1], [2], [-1]]
        stretchFt_NoNames = self.constructor(stretchFt_Raw).stretch
        ft_DefaultPtNames = self.constructor(stretchFt_Raw)
        defPtNames = ft_DefaultPtNames.points.getNames()
        assert ft_DefaultPtNames.points._namesCreated()
        stretchFt_DefaultPtNames = ft_DefaultPtNames.stretch
        stretchFt_MatchPtNames = self.constructor(stretchFt_Raw, pointNames=pNames).stretch
        stretchFt_NoMatchPtNames = self.constructor(stretchFt_Raw, pointNames=offPNames).stretch
        stretchFt_WithFtName = self.constructor(stretchFt_Raw, featureNames=single).stretch
        stretchFt_AllNamesPtMatch = self.constructor(stretchFt_Raw, pointNames=pNames,
                                                  featureNames=single).stretch
        stretchFt_AllNamesNoPtMatch = self.constructor(stretchFt_Raw, pointNames=offPNames,
                                                    featureNames=single).stretch

        ### Base no names ###
        self.back_stretchSetNames(baseNoNames, stretchPt_NoNames, None, None)
        self.back_stretchSetNames(baseNoNames, stretchFt_NoNames, None, None)
        self.back_stretchSetNames(baseNoNames, stretchPt_DefaultFtNames, None, defFtNames)
        self.back_stretchSetNames(baseNoNames, stretchFt_DefaultPtNames, defPtNames, None)
        self.back_stretchSetNames(baseNoNames, stretchPt_MatchFtNames, None, fNames)
        self.back_stretchSetNames(baseNoNames, stretchFt_MatchPtNames, pNames, None)
        self.back_stretchSetNames(baseNoNames, stretchPt_WithPtName, single3, None)
        self.back_stretchSetNames(baseNoNames, stretchFt_WithFtName, None, single4)
        self.back_stretchSetNames(baseNoNames, stretchPt_AllNamesFtMatch, single3, fNames)
        self.back_stretchSetNames(baseNoNames, stretchFt_AllNamesPtMatch, pNames, single4)

        ### Base pt names ###
        self.back_stretchSetNames(basePtNames, stretchPt_NoNames, pNames, None)
        self.back_stretchSetNames(basePtNames, stretchFt_NoNames, pNames, None)
        self.back_stretchSetNames(basePtNames, stretchPt_DefaultFtNames, pNames, defFtNames)
        self.back_stretchSetNames(basePtNames, stretchFt_DefaultPtNames, pNames, None)
        self.back_stretchSetNames(basePtNames, stretchPt_MatchFtNames, pNames, fNames)
        self.back_stretchSetNames(basePtNames, stretchFt_MatchPtNames, pNames, None)
        self.back_stretchSetNames(basePtNames, stretchPt_WithPtName, None, None)
        self.back_stretchSetNames(basePtNames, stretchFt_WithFtName, pNames, single4)
        self.back_stretchSetNames(basePtNames, stretchPt_AllNamesFtMatch, None, fNames)
        self.back_stretchSetNames(basePtNames, stretchFt_AllNamesPtMatch, pNames, single4)
        self.back_stretchSetNamesException(basePtNames, stretchFt_NoMatchPtNames)
        self.back_stretchSetNamesException(basePtNames, stretchFt_AllNamesNoPtMatch)

        ### Base ft names ###
        self.back_stretchSetNames(baseFtNames, stretchPt_NoNames, None, fNames)
        self.back_stretchSetNames(baseFtNames, stretchFt_NoNames, None, fNames)
        self.back_stretchSetNames(baseFtNames, stretchPt_DefaultFtNames, None, fNames)
        self.back_stretchSetNames(baseFtNames, stretchFt_DefaultPtNames, defPtNames, fNames)
        self.back_stretchSetNames(baseFtNames, stretchPt_MatchFtNames, None, fNames)
        self.back_stretchSetNames(baseFtNames, stretchFt_MatchPtNames, pNames, fNames)
        self.back_stretchSetNames(baseFtNames, stretchPt_WithPtName, single3, fNames)
        self.back_stretchSetNames(baseFtNames, stretchFt_WithFtName, None, None)
        self.back_stretchSetNames(baseFtNames, stretchPt_AllNamesFtMatch, single3, fNames)
        self.back_stretchSetNames(baseFtNames, stretchFt_AllNamesPtMatch, pNames, None)
        self.back_stretchSetNamesException(baseFtNames, stretchPt_NoMatchFtNames)
        self.back_stretchSetNamesException(baseFtNames, stretchPt_AllNamesNoFtMatch)

        ### Base all names ###
        self.back_stretchSetNames(baseAllNames, stretchPt_NoNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchFt_NoNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchPt_DefaultFtNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchFt_DefaultPtNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchPt_MatchFtNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchFt_MatchPtNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchPt_WithPtName, None, fNames)
        self.back_stretchSetNames(baseAllNames, stretchFt_WithFtName, pNames, None)
        self.back_stretchSetNames(baseAllNames, stretchPt_AllNamesFtMatch, None, fNames)
        self.back_stretchSetNames(baseAllNames, stretchFt_AllNamesPtMatch, pNames, None)
        self.back_stretchSetNamesException(baseAllNames, stretchPt_NoMatchFtNames)
        self.back_stretchSetNamesException(baseAllNames, stretchFt_NoMatchPtNames)
        self.back_stretchSetNamesException(baseAllNames, stretchPt_AllNamesNoFtMatch)
        self.back_stretchSetNamesException(baseAllNames, stretchFt_AllNamesNoPtMatch)

        ### Both Stretch ###
        self.back_stretchSetNames(stretchPt_NoNames, stretchFt_NoNames, None, None)
        self.back_stretchSetNames(stretchPt_NoNames, stretchFt_DefaultPtNames, defPtNames, None)
        self.back_stretchSetNames(stretchPt_NoNames, stretchFt_MatchPtNames, pNames, None)
        self.back_stretchSetNames(stretchPt_NoNames, stretchFt_WithFtName, None, single4)
        self.back_stretchSetNames(stretchPt_NoNames, stretchFt_AllNamesPtMatch, pNames, single4)

        self.back_stretchSetNames(stretchPt_MatchFtNames, stretchFt_NoNames, None, fNames)
        self.back_stretchSetNames(stretchPt_MatchFtNames, stretchFt_DefaultPtNames, defPtNames, fNames)
        self.back_stretchSetNames(stretchPt_MatchFtNames, stretchFt_MatchPtNames, pNames, fNames)
        self.back_stretchSetNames(stretchPt_MatchFtNames, stretchFt_WithFtName, None, None)
        self.back_stretchSetNames(stretchPt_MatchFtNames, stretchFt_AllNamesPtMatch, pNames, None)

        self.back_stretchSetNames(stretchPt_WithPtName, stretchFt_NoNames, single3, None)
        self.back_stretchSetNames(stretchPt_WithPtName, stretchFt_DefaultPtNames, single3, None)
        self.back_stretchSetNames(stretchPt_WithPtName, stretchFt_MatchPtNames, None, None)
        self.back_stretchSetNames(stretchPt_WithPtName, stretchFt_WithFtName, single3, single4)
        self.back_stretchSetNames(stretchPt_WithPtName, stretchFt_AllNamesPtMatch, None, single4)

        self.back_stretchSetNames(stretchPt_AllNamesFtMatch, stretchFt_NoNames, single3, fNames)
        self.back_stretchSetNames(stretchPt_AllNamesFtMatch, stretchFt_DefaultPtNames, single3, fNames)
        self.back_stretchSetNames(stretchPt_AllNamesFtMatch, stretchFt_MatchPtNames, None, fNames)
        self.back_stretchSetNames(stretchPt_AllNamesFtMatch, stretchFt_WithFtName, single3, None)
        self.back_stretchSetNames(stretchPt_AllNamesFtMatch, stretchFt_AllNamesPtMatch, None, None)


    def test_stretch_chainedOperators(self):
        raw1 = [[1, 2, 3], [4, 5, 6]]
        raw2 = [3, 3, 3]
        raw3 = [[2], [2]]
        obj1 = self.constructor(raw1)
        obj2 = self.constructor(raw2)
        obj3 = self.constructor(raw3)

        exp1 = [[4, 5, 6], [7, 7, 8]]
        expObj1 = self.constructor(exp1)
        chain1 = obj1 + obj2.stretch - obj3.stretch * obj1 // obj2.stretch ** obj3.stretch

        assert chain1 == expObj1

        exp2 = [[7, 6, 7], [9, 11, 12]]
        expObj2 = self.constructor(exp2)
        chain2 = (obj2.stretch + obj3.stretch) // obj1 + obj3.stretch * obj1

        exp3 = [[4, 14, 30], [52, 80, 114]]
        expObj3 = self.constructor(exp3)
        chain3 = obj1 + obj2.stretch * obj1 ** obj3.stretch
        assert chain3 == expObj3

        exp4 = [[4.5, 4.5, 4.5], [4.5, 4.5, 4.5]]
        expObj4 = self.constructor(exp4)
        chain4 = obj2.stretch * obj3.stretch - obj2.stretch / obj3.stretch
        assert chain4 == expObj4


class StretchDataModifying(DataTestObject):
    ########################
    # Base / Stretch Point #
    ########################

    # zero safe

    def test_handmade_Base_Stretch_iadd_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])

        try:
            toStretch.stretch += base
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[2, 4, 3, 3], [6, 8, 7, 7], [1, 1, -2, -4]])
        base += toStretch.stretch

        assert base.isIdentical(exp)
        assertNoNamesGenerated(base)

    def test_handmade_Base_Stretch_isub_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])

        try:
            toStretch.stretch -= base
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[0, 0, 3, 5], [4, 4, 7, 9], [-1, -3, -2, -2]])
        base -= toStretch.stretch

        assert base.isIdentical(exp)
        assertNoNamesGenerated(base)


    def test_handmade_Base_Stretch_imul_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])

        try:
            toStretch.stretch *= base
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[1, 4, 0, -4], [5, 12, 0, -8], [0, -2, 0, 3]])
        base *= toStretch.stretch

        assert base.isIdentical(exp)
        assertNoNamesGenerated(base)

    # zero exception

    def test_handmade_Base_Stretch_itruediv_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])

        try:
            stretch1.stretch /= base1
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        base1 /= stretch1.stretch

        assert base1.isIdentical(exp)
        assertNoNamesGenerated(base1)

        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        stretch2 = self.constructor([[1, 2, 0, -1]])

        try:
            base2 /= stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_ifloordiv_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])

        try:
            stretch1.stretch //= base1
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        base1 //= stretch1.stretch

        assert base1.isIdentical(exp)
        assertNoNamesGenerated(base1)

        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        stretch2 = self.constructor([[1, 2, 0, -1]])

        try:
            base2 //= stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_imod_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])

        try:
            stretch1.stretch %= base1
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        base1 %= stretch1.stretch

        assert base1.isIdentical(exp)
        assertNoNamesGenerated(base1)

        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        stretch2 = self.constructor([[1, 2, 0, -1]])

        try:
            base2 %= stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_ipow_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])

        try:
            stretch1.stretch **= base1
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[1, 4, (1/3), (1/16)], [5, 36, (1/7), (1/64)], [-1, 4, (-1/3), (1/16)]])
        base1 **= stretch1.stretch

        assert base1.isIdentical(exp)
        assertNoNamesGenerated(base1)

        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, 0]])
        stretch2 = self.constructor([[1, 2, 0, -1]])

        try:
            base2 **= stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    ##########################
    # Base / Stretch Feature #
    ##########################

    # zero safe

    def test_handmade_Base_Stretch_iadd_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])

        try:
            toStretch.stretch += base
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[2, 3, 4], [6, 7, 8], [7, 8, 9], [-1, -2, -3]])
        base += toStretch.stretch

        assert base.isIdentical(exp)
        assertNoNamesGenerated(base)

    def test_handmade_Base_Stretch_isub_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])

        try:
            toStretch.stretch -= base
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[0, 1, 2], [2, 3, 4], [7, 8, 9], [1, 0, -1]])
        base -= toStretch.stretch

        assert base.isIdentical(exp)
        assertNoNamesGenerated(base)


    def test_handmade_Base_Stretch_imul_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])

        try:
            toStretch.stretch *= base
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[1, 2, 3], [8, 10, 12], [0, 0, 0], [0, 1, 2]])
        base *= toStretch.stretch

        assert base.isIdentical(exp)
        assertNoNamesGenerated(base)

    # zero exception

    def test_handmade_Base_Stretch_itruediv_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])

        try:
            stretch1.stretch /= base1
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[1, 2, 3], [2, (5/2), 3], [-7, -8, -9], [(1/2), 1, (3/2)]])
        base1 /= stretch1.stretch

        assert base1.isIdentical(exp)
        assertNoNamesGenerated(base1)

        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])

        try:
            base2 /= stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_ifloordiv_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])

        try:
            stretch1.stretch //= base1
            assert False # expected AttributeError
        except AttributeError:
            pass

        exp = self.constructor([[1, 2, 3], [2, 2, 3], [-7, -8, -9], [0, 1, 1]])
        base1 //= stretch1.stretch

        assert base1.isIdentical(exp)
        assertNoNamesGenerated(base1)

        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])

        try:
            base2 //= stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_imod_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])

        try:
            stretch1.stretch %= base1
            assert False # expected AttributeError
        except AttributeError:
            pass

        base1 %= stretch1.stretch
        exp = self.constructor([[0, 0, 0], [0, 1, 0], [0, 0, 0], [-1, 0, -1]])

        assert base1.isIdentical(exp)
        assertNoNamesGenerated(base1)

        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])

        try:
            base2 %= stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_ipow_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])

        try:
            stretch1.stretch **= base1
            assert False # expected AttributeError
        except AttributeError:
            pass

        base1 **= stretch1.stretch
        exp = self.constructor([[1, 2, 3], [16, 25, 36], [(1/7), (1/8), (1/9)], [1, (1/4), (1/9)]])

        assert base1.isIdentical(exp)
        assertNoNamesGenerated(base1)

        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, -1], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])

        try:
            base2 **= stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

class StretchAll(StretchDataSafe, StretchDataModifying):
    pass
