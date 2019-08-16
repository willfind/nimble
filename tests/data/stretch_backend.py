"""
Tests for using stretch attribute to trigger broadcasting operations
"""

from nose.tools import raises

from nimble.exceptions import ImproperObjectAction
from .baseObject import DataTestObject

# TODO point and feature names?
# If base has? if stretch has?

# TODO test shapes are acceptable

class StretchSharedBackend(DataTestObject):

    ##################
    # Base / Stretch #
    ##################

    def backend_zeroSafe_Base_Stretch_Point(self, opName, expected):
        lhs = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        rhs = self.constructor([[1, 2, 0, -1]])

        ret = getattr(lhs, opName)(rhs.stretch)

        if opName.startswith('__i'):
            assert lhs.isIdentical(expected)
        else:
            assert ret.isIdentical(expected)


    def backend_zeroSafe_Base_Stretch_Feature(self, opName, expected):
        lhs = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        rhs = self.constructor([[1], [2], [0], [-1]])

        ret = getattr(lhs, opName)(rhs.stretch)

        if opName.startswith('__i'):
            assert lhs.isIdentical(expected)
        else:
            assert ret.isIdentical(expected)


    def backend_zeroException_Base_Stretch_Point(self, opName, expected):
        lhs1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        rhs1 = self.constructor([[1, 2, -1, -2]])
        lhs2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        rhs2 = self.constructor([[1, 2, 0, -1]])


        ret1 = getattr(lhs1, opName)(rhs1.stretch)
        if opName.startswith('__i'):
            assert lhs1.isIdentical(expected)
        else:
            assert ret1.isIdentical(expected)

        try:
            getattr(lhs2, opName)(rhs2.stretch)
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass


    def backend_zeroException_Base_Stretch_Feature(self, opName, expected):
        lhs1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        rhs1 = self.constructor([[1], [2], [-1], [-2]])
        lhs2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        rhs2 = self.constructor([[1], [2], [0], [-1]])


        ret1 = getattr(lhs1, opName)(rhs1.stretch)
        if opName.startswith('__i'):
            assert lhs1.isIdentical(expected)
        else:
            assert ret1.isIdentical(expected)

        try:
            getattr(lhs2, opName)(rhs2.stretch)
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    #####################
    # Stretch / Stretch #
    #####################

    def backend_zeroSafe_Stretch_Stretch(self, opName, expected):
        ft = self.constructor([[0], [1], [2], [-1]])
        pt = self.constructor([[1, 2, 0, -1]])

        ret = getattr(ft.stretch, opName)(pt.stretch)

        if opName.startswith('__i'):
            assert ft.isIdentical(expected)
        else:
            assert ret.isIdentical(expected)


    def backend_zeroException_Stretch_Stretch(self, opName, expected):
        ft1 = self.constructor([[1], [2], [3], [-1]])
        pt1 = self.constructor([[1, 2, -2, -1]])
        ft2 = self.constructor([[0], [1], [2], [-1]])
        pt2 = self.constructor([[1, 2, 0, -1]])

        ret1 = getattr(ft1.stretch, opName)(pt1.stretch)

        if opName.startswith('__i'):
            assert ft1.isIdentical(expected)
        else:
            assert ret1.isIdentical(expected)

        try:
            getattr(ft2.stretch, opName)(pt2.stretch)
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass


class StretchDataSafe(StretchSharedBackend):

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
        exp = self.constructor([[2, 4, 3, 3], [6, 8, 7, 7], [1, 1, -2, -4]])
        self.backend_zeroSafe_Base_Stretch_Point('__add__', exp)
        self.backend_zeroSafe_Base_Stretch_Point('__radd__', exp)

    def test_handmade_Base_Stretch_sub_point(self):
        exp_l = self.constructor([[0, 0, 3, 5], [4, 4, 7, 9], [-1, -3, -2, -2]])
        self.backend_zeroSafe_Base_Stretch_Point('__sub__', exp_l)

        exp_r = self.constructor([[0, 0, -3, -5], [-4, -4, -7, -9], [1, 3, 2, 2]])
        self.backend_zeroSafe_Base_Stretch_Point('__rsub__', exp_r)

    def test_handmade_Base_Stretch_mul_point(self):
        exp = self.constructor([[1, 4, 0, -4], [5, 12, 0, -8], [0, -2, 0, 3]])
        self.backend_zeroSafe_Base_Stretch_Point('__mul__', exp)
        self.backend_zeroSafe_Base_Stretch_Point('__rmul__', exp)

    # zero exception

    def test_handmade_Base_Stretch_truediv_point(self):
        exp_l = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        self.backend_zeroException_Base_Stretch_Point('__truediv__', exp_l)

        exp_r = self.constructor([[1, 1, (-1/3), (-1/2)], [(1/5), (1/3), (-1/7), (-1/4)],
                                  [-1, -1, (1/3), (1/2)]])
        self.backend_zeroException_Base_Stretch_Point('__rtruediv__', exp_r)

    def test_handmade_Base_Stretch_floordiv_point(self):
        exp_l = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        self.backend_zeroException_Base_Stretch_Point('__floordiv__', exp_l)

        exp_r = self.constructor([[1, 1, -1, -1], [0, 0, -1, -1], [-1, -1, 0, 0]])
        self.backend_zeroException_Base_Stretch_Point('__rfloordiv__', exp_r)

    def test_handmade_Base_Stretch_mod_point(self):
        exp_l = self.constructor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.backend_zeroException_Base_Stretch_Point('__mod__', exp_l)

        exp_r = self.constructor([[0, 0, 2, 2], [1, 2, 6, 6], [0, 0, -1, -2]])
        self.backend_zeroException_Base_Stretch_Point('__rmod__', exp_r)

    ##########################
    # Base / Stretch Feature #
    ##########################

    # zero safe

    def test_handmade_Base_Stretch_add_feature(self):
        exp = self.constructor([[2, 3, 4], [6, 7, 8], [7, 8, 9], [-1, -2, -3]])
        self.backend_zeroSafe_Base_Stretch_Feature('__add__', exp)
        self.backend_zeroSafe_Base_Stretch_Feature('__radd__', exp)

    def test_handmade_Base_Stretch_sub_feature(self):
        exp_l = self.constructor([[0, 1, 2], [2, 3, 4], [7, 8, 9], [1, 0, -1]])
        self.backend_zeroSafe_Base_Stretch_Feature('__sub__', exp_l)

        exp_r = self.constructor([[0, -1, -2], [-2, -3, -4], [-7, -8, -9], [-1, 0, 1]])
        self.backend_zeroSafe_Base_Stretch_Feature('__rsub__', exp_r)

    def test_handmade_Base_Stretch_mul_feature(self):
        exp = self.constructor([[1, 2, 3], [8, 10, 12], [0, 0, 0], [0, 1, 2]])
        self.backend_zeroSafe_Base_Stretch_Feature('__mul__', exp)
        self.backend_zeroSafe_Base_Stretch_Feature('__rmul__', exp)

    # zero exception

    def test_handmade_Base_Stretch_truediv_feature(self):
        exp_l = self.constructor([[1, 2, 3], [2, (5/2), 3], [-7, -8, -9], [(1/2), 1, (3/2)]])
        self.backend_zeroException_Base_Stretch_Feature('__truediv__', exp_l)

        exp_r = self.constructor([[1, (1/2), (1/3)], [(1/2), (2/5), (1/3)], [(-1/7), (-1/8), (-1/9)], [2, 1, (2/3)]])
        self.backend_zeroException_Base_Stretch_Feature('__rtruediv__', exp_r)

    def test_handmade_Base_Stretch_floordiv_feature(self):
        exp_l = self.constructor([[1, 2, 3], [2, 2, 3], [-7, -8, -9], [0, 1, 1]])
        self.backend_zeroException_Base_Stretch_Feature('__floordiv__', exp_l)

        exp_r = self.constructor([[1, 0, 0], [0, 0, 0], [-1, -1, -1], [2, 1, 0]])
        self.backend_zeroException_Base_Stretch_Feature('__rfloordiv__', exp_r)

    def test_handmade_Base_Stretch_mod_feature(self):
        exp_l = self.constructor([[0, 0, 0], [0, 1, 0], [0, 0, 0], [-1, 0, -1]])
        self.backend_zeroException_Base_Stretch_Feature('__mod__', exp_l)

        exp_r = self.constructor([[0, 1, 1], [2, 2, 2], [6, 7, 8], [0, 0, -2]])
        self.backend_zeroException_Base_Stretch_Feature('__rmod__', exp_r)

    #####################
    # Stretch / Stretch #
    #####################

    # zero safe

    def test_handmade_Stretch_Stretch_add(self):
        exp = self.constructor([[1, 2, 0, -1], [2, 3, 1, 0], [3, 4, 2, 1], [0, 1, -1, -2]])
        self.backend_zeroSafe_Stretch_Stretch('__add__', exp)
        self.backend_zeroSafe_Stretch_Stretch('__radd__', exp)

    def test_handmade_Stretch_Stretch_sub(self):
        exp_l = self.constructor([[-1, -2, 0, 1], [0, 1, -1, -2], [1, 0, 2, 3], [-2, -3, -1, 0]])
        self.backend_zeroSafe_Stretch_Stretch('__sub__', exp_l)
        exp_r = self.constructor([[1, 2, 0, -1], [0, -1, 1, 2], [-1, 0, -2, -3], [2, 3, 1, 0]])
        self.backend_zeroSafe_Stretch_Stretch('__rsub__', exp_r)

    def test_handmade_Stretch_Stretch_mul(self):
        exp = self.constructor([[0, 0, 0, 0], [1, 2, 0, -1], [2, 4, 0, -2], [-1, -2, 0, 1]])
        self.backend_zeroSafe_Stretch_Stretch('__mul__', exp)
        self.backend_zeroSafe_Stretch_Stretch('__rmul__', exp)

    # zero exception

    def test_handmade_Stretch_Stretch_truediv(self):
        exp_l = self.constructor([[1, (1/2), (-1/2), -1], [2, 1, -1, -2],
                                  [3, (3/2), (-3/2), -3], [-1, (-1/2), (1/2), 1]])
        self.backend_zeroException_Stretch_Stretch('__truediv__', exp_l)
        exp_r = self.constructor([[1, 2, -2, -1], [(1/2), 1, -1, (-1/2)],
                                  [(1/3), (2/3), (-2/3), (-1/2)], [-1, -2, 2, 1]])
        self.backend_zeroException_Stretch_Stretch('__rtruediv__', exp_r)

    def test_handmade_Stretch_Stretch_floordiv(self):
        exp_l = self.constructor([[1, 0, -1, -1], [2, 1, -1, -2], [3, 1, -2, -3], [-1, -1, 0, 1]])
        self.backend_zeroException_Stretch_Stretch('__floordiv__', exp_l)
        exp_r = self.constructor([[1, 2, -2, -1], [0, 1, -1, -1], [0, 0, -1, -1], [-1, -2, 2, 1]])
        self.backend_zeroException_Stretch_Stretch('__rfloordiv__', exp_r)

    def test_handmade_Stretch_Stretch_mod(self):
        exp_l = self.constructor([[0, 1, -1, 0], [0, 0, 0, 0], [0, 1, -1, 0], [0, 1, -1, 0]])
        self.backend_zeroException_Stretch_Stretch('__mod__', exp_l)
        exp_r = self.constructor([[0, 0, 0, 0], [1, 0, 0, 1], [1, 2, 1, 2], [0, 0, 0, 0]])
        self.backend_zeroException_Stretch_Stretch('__rmod__', exp_r)


class StretchModifying(StretchSharedBackend):

    ########################
    # Base / Stretch Point #
    ########################

    # zero safe

    def test_handmade_Base_Stretch_iadd_point(self):
        exp = self.constructor([[2, 4, 3, 3], [6, 8, 7, 7], [1, 1, -2, -4]])
        self.backend_zeroSafe_Base_Stretch_Point('__iadd__', exp)

    def test_handmade_Base_Stretch_isub_point(self):
        exp = self.constructor([[0, 0, 3, 5], [4, 4, 7, 9], [-1, -3, -2, -2]])
        self.backend_zeroSafe_Base_Stretch_Point('__isub__', exp)

    def test_handmade_Base_Stretch_imul_point(self):
        exp = self.constructor([[1, 4, 0, -4], [5, 12, 0, -8], [0, -2, 0, 3]])
        self.backend_zeroSafe_Base_Stretch_Point('__imul__', exp)

    # zero exception

    def test_handmade_Base_Stretch_itruediv_point(self):
        exp = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        self.backend_zeroException_Base_Stretch_Point('__itruediv__', exp)

    def test_handmade_Base_Stretch_ifloordiv_point(self):
        exp = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        self.backend_zeroException_Base_Stretch_Point('__ifloordiv__', exp)

    def test_handmade_Base_Stretch_imod_point(self):
        exp = self.constructor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.backend_zeroException_Base_Stretch_Point('__imod__', exp)

    ##########################
    # Base / Stretch Feature #
    ##########################

    # zero safe

    def test_handmade_Base_Stretch_iadd_feature(self):
        exp = self.constructor([[2, 3, 4], [6, 7, 8], [7, 8, 9], [-1, -2, -3]])
        self.backend_zeroSafe_Base_Stretch_Feature('__iadd__', exp)

    def test_handmade_Base_Stretch_isub_feature(self):
        exp = self.constructor([[0, 1, 2], [2, 3, 4], [7, 8, 9], [1, 0, -1]])
        self.backend_zeroSafe_Base_Stretch_Feature('__isub__', exp)

    def test_handmade_Base_Stretch_imul_feature(self):
        exp = self.constructor([[1, 2, 3], [8, 10, 12], [0, 0, 0], [0, 1, 2]])
        self.backend_zeroSafe_Base_Stretch_Feature('__imul__', exp)

    # zero exception

    def test_handmade_Base_Stretch_itruediv_feature(self):
        exp = self.constructor([[1, 2, 3], [2, (5/2), 3], [-7, -8, -9], [(1/2), 1, (3/2)]])
        self.backend_zeroException_Base_Stretch_Feature('__itruediv__', exp)

    def test_handmade_Base_Stretch_ifloordiv_feature(self):
        exp = self.constructor([[1, 2, 3], [2, 2, 3], [-7, -8, -9], [0, 1, 1]])
        self.backend_zeroException_Base_Stretch_Feature('__ifloordiv__', exp)

    def test_handmade_Base_Stretch_imod_feature(self):
        exp = self.constructor([[0, 0, 0], [0, 1, 0], [0, 0, 0], [-1, 0, -1]])
        self.backend_zeroException_Base_Stretch_Feature('__imod__', exp)

    #####################
    # Stretch / Stretch #
    #####################

    # zero safe

    def test_handmade_Stretch_Stretch_iadd(self):
        exp = self.constructor([[1, 2, 0, -1], [2, 3, 1, 0], [3, 4, 2, 1], [0, 1, -1, -2]])
        self.backend_zeroSafe_Stretch_Stretch('__iadd__', exp)

    def test_handmade_Stretch_Stretch_isub(self):
        exp = self.constructor([[-1, -2, 0, 1], [0, 1, -1, -2], [1, 0, 2, 3], [-2, -3, -1, 0]])
        self.backend_zeroSafe_Stretch_Stretch('__isub__', exp)

    def test_handmade_Stretch_Stretch_imul(self):
        exp = self.constructor([[0, 0, 0, 0], [1, 2, 0, -1], [2, 4, 0, -2], [-1, -2, 0, 1]])
        self.backend_zeroSafe_Stretch_Stretch('__imul__', exp)

    # zero exception

    def test_handmade_Stretch_Stretch_itruediv(self):
        exp = self.constructor([[1, (1/2), (-1/2), -1], [2, 1, -1, -2],
                                [3, (3/2), (-3/2), -3], [-1, (-1/2), (1/2), 1]])
        self.backend_zeroException_Stretch_Stretch('__itruediv__', exp)

    def test_handmade_Stretch_Stretch_ifloordiv(self):
        exp = self.constructor([[1, 0, -1, -1], [2, 1, -1, -2], [3, 1, -2, -3], [-1, -1, 0, 1]])
        self.backend_zeroException_Stretch_Stretch('__ifloordiv__', exp)

    def test_handmade_Stretch_Stretch_imod(self):
        exp = self.constructor([[0, 1, -1, 0], [0, 0, 0, 0], [0, 1, -1, 0], [0, 1, -1, 0]])
        self.backend_zeroException_Stretch_Stretch('__imod__', exp)
