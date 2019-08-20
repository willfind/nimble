"""
Tests for using stretch attribute to trigger broadcasting operations
"""
from nose.tools import raises

from nimble.exceptions import ImproperObjectAction
from .baseObject import DataTestObject

# TODO
# point and feature names?
  # If base has? if stretch has?
 # Chained operations

class StretchBackend(DataTestObject):

    ##############
    # Exceptions #
    ##############

    @raises(ImproperObjectAction)
    def test_2D_stretch_exception(self):
        toStretch = self.constructor([[1, 2], [3, 4]])
        toStretch.stretch

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

    def test_handmade_Base_Stretch_sub_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[0, 0, 3, 5], [4, 4, 7, 9], [-1, -3, -2, -2]])
        exp_r = self.constructor([[0, 0, -3, -5], [-4, -4, -7, -9], [1, 3, 2, 2]])
        ret1 = base - toStretch.stretch
        ret2 = toStretch.stretch - base
        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)

    def test_handmade_Base_Stretch_mul_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])
        exp = self.constructor([[1, 4, 0, -4], [5, 12, 0, -8], [0, -2, 0, 3]])
        ret1 = base * toStretch.stretch
        ret2 = toStretch.stretch * base
        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)

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

        try:
            base2 ** stretch2.stretch
            assert False # expected FloatingPointError
        except FloatingPointError:
            pass

        try:
            stretch2.stretch ** base2
            assert False # expected FloatingPointError
        except FloatingPointError:
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

    def test_handmade_Base_Stretch_sub_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[0, 1, 2], [2, 3, 4], [7, 8, 9], [1, 0, -1]])
        exp_r = self.constructor([[0, -1, -2], [-2, -3, -4], [-7, -8, -9], [-1, 0, 1]])
        ret1 = base - toStretch.stretch
        ret2 = toStretch.stretch - base
        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)

    def test_handmade_Base_Stretch_mul_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])
        exp = self.constructor([[1, 2, 3], [8, 10, 12], [0, 0, 0], [0, 1, 2]])
        ret1 = base * toStretch.stretch
        ret2 = toStretch.stretch * base
        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)

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

        try:
            base2 ** stretch2.stretch
            assert False # expected FloatingPointError
        except FloatingPointError:
            pass

        try:
            stretch2.stretch ** base2
            assert False # expected FloatingPointError
        except FloatingPointError:
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
            assert False # expected FloatingPointError
        except FloatingPointError:
            pass

        try:
            pt2.stretch ** ft2.stretch
            assert False # expected FloatingPointError
        except FloatingPointError:
            pass
