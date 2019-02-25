from __future__ import absolute_import
import re

import numpy

import UML
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue, InvalidArgumentValueCombination

scipy = UML.importModule('scipy')

def inverse(A):
    """
       Compute the (multiplicative) inverse of a UML Base object.

    """
    if scipy is None:
        msg = "scipy must be installed in order to use the inverse function."
        raise PackageException(msg)
    if not isinstance(A, UML.data.Base):
        raise InvalidArgumentType(
            "Object must be derived class of UML.data.Base")
    if len(A.points) == 0 and len(A.features) == 0:
        return A.copy()
    if len(A.points) != len(A.features):
        msg = 'Object has to be square (Number of features and points needs to be equal).'
        raise InvalidArgumentValue(msg)

    def _handleSingularCase(e):
        if re.match('.*singular.*', str(e), re.I):
            msg = 'Object non-invertible (Singular)'
            raise InvalidArgumentValue(msg)
        else:
            raise(e)

    if A.getTypeString() in ['Matrix', 'DataFrame', 'List']:
        inv_obj = A.copyAs('Matrix')
        try:
            inv_data = scipy.linalg.inv(inv_obj.data)
        except scipy.linalg.LinAlgError as e:
            _handleSingularCase(e)
        except ValueError as e:
            if re.match('.*object arrays*', str(e), re.I):
                msg = 'Elements types in object data are not supported.'
                raise InvalidArgumentType(msg)
    else:
        inv_obj = A.copyAs('Sparse')
        try:
            inv_data = scipy.sparse.linalg.inv(inv_obj.data.tocsc())
        except RuntimeError as e:
            _handleSingularCase(e)
        except TypeError as e:
            if re.match('.*no supported conversion*', str(e), re.I):
                msg = 'Elements types in object data are not supported.'
                raise InvalidArgumentType(msg)

    inv_obj.transpose()
    inv_obj.data = inv_data
    if A.getTypeString() != inv_obj.getTypeString:
        inv_obj = inv_obj.copyAs(A.getTypeString())
    return inv_obj


def pseudoInverse(A, method='svd'):
    """
        Compute the (Moore-Penrose) pseudo-inverse of a UML object.
        Method: 'svd' or 'least-squares'.
        Uses singular-value decomposition by default. Least squares solver included as an option.
    """
    if scipy is None:
        msg = "scipy must be installed in order to use the pseudoInverse function."
        raise PackageException(msg)
    if not isinstance(A, UML.data.Base):
        raise InvalidArgumentType(
            "Object must be derived class of UML.data.Base.")
    if len(A.points) == 0 and len(A.features) == 0:
        return A
    if method not in ['least-squares', 'svd']:
        raise InvalidArgumentValue(
            "Supported methods are 'least-squares' and 'svd'.")

    def _handleNonSupportedTypes(e):
        if re.match('.*object arrays*', str(e), re.I):
            msg = 'Elements types in object data are not supported.'
            raise InvalidArgumentType(msg)

    pinv_obj = A.copyAs('Matrix')
    if method == 'svd':
        try:
            pinv_data = scipy.linalg.pinv2(pinv_obj.data)
        except ValueError as e:
            _handleNonSupportedTypes(e)
    else:
        pinv_data = scipy.linalg.pinv(pinv_obj.data)
    pinv_obj.transpose()
    pinv_obj.data = numpy.asmatrix(pinv_data)
    if A.getTypeString() != 'Matrix':
        pinv_obj = pinv_obj.copyAs(A.getTypeString())
    return pinv_obj


def solve(A, b):
    """
        Solves the linear equation set A * x = b for the unknown vector x.
        A should be a square a object.
    """
    if scipy is None:
        msg = "scipy must be installed in order to use the solve function."
        raise PackageException(msg)
    if not isinstance(A, UML.data.Base):
        raise InvalidArgumentType(
            "Left hand side object must be derived class of UML.data.Base.")
    if not isinstance(b, UML.data.Base):
        raise InvalidArgumentType(
            "Right hand side object must be derived class of UML.data.Base.")
    if len(A.points) != len(A.features):
        msg = 'Object A has to be square (Number of features and points needs to be equal).'
        raise InvalidArgumentValue(msg)
    if len(b.points) != 1 and len(b.features) != 1:
        raise InvalidArgumentValue("b should be a vector")
    elif len(b.points) == 1 and len(b.features) > 1:
        if len(A.points) != len(b.features):
            raise InvalidArgumentValueCombination('A and b have incompatible dimensions.')
        else:
            b = b.copy()
            b.flattenToOneFeature()
    elif len(b.points) > 1 and len(b.features) == 1:
        if len(A.points) != len(b.points):
            raise InvalidArgumentValueCombination('A and b have incompatible dimensions.')

    A_original_type = A.getTypeString()
    if A.getTypeString() in ['DataFrame', 'List']:
        A = A.copyAs('Matrix')
    if b.getTypeString() in ['DataFrame', 'List', 'Sparse']:
        b = b.copyAs('Matrix')

    sol = A[0, :].copyAs('Matrix')
    sol.points.setNames(['b'])

    if A.getTypeString() == 'Matrix':
        solution = scipy.linalg.solve(A.data, b.data)
        sol.data = solution.T
    elif isinstance(A, UML.data.sparse.SparseView):
        A_copy = A.copy()
        solution = scipy.sparse.linalg.spsolve(A_copy.data, numpy.asarray(b.data))
        sol.data = numpy.asmatrix(solution)
    elif A.getTypeString() == 'Sparse':
        solution = scipy.sparse.linalg.spsolve(A.data, numpy.asarray(b.data))
        sol.data = numpy.asmatrix(solution)

    return sol.copyAs(A_original_type)


def leastSquaresSolution(A, b):
    """
        Compute least-squares solution to equation Ax = b
    """
    if scipy is None:
        msg = "scipy must be installed in order to use the leastSquaresSolution function."
        raise PackageException(msg)
    if not isinstance(A, UML.data.Base):
        raise InvalidArgumentType(
            "Left hand side object must be derived class of UML.data.Base.")
    if not isinstance(b, UML.data.Base):
        raise InvalidArgumentType(
            "Right hand side object must be derived class of UML.data.Base.")
    if len(b.points) != 1 and len(b.features) != 1:
        raise InvalidArgumentValue("b should be a vector")
    elif len(b.points) == 1 and len(b.features) > 1:
        if len(A.points) != len(b.features):
            raise InvalidArgumentValueCombination('A and b have incompatible dimensions.')
        else:
            b = b.copy()
            b.flattenToOneFeature()
    elif len(b.points) > 1 and len(b.features) == 1:
        if len(A.points) != len(b.points):
            raise InvalidArgumentValueCombination('A and b have incompatible dimensions.')

    A_original_type = A.getTypeString()
    if A.getTypeString() in ['DataFrame', 'List']:
        A = A.copyAs('Matrix')
    if b.getTypeString() in ['DataFrame', 'List', 'Sparse']:
        b = b.copyAs('Matrix')

    sol = A[0, :].copyAs('Matrix')
    sol.points.setNames(['b'])
    if A.getTypeString() == 'Matrix':
        solution = scipy.linalg.lstsq(A.data, b.data)
        sol.data = solution[0].T
    elif isinstance(A, UML.data.sparse.SparseView):
        A_copy = A.copy()
        solution = scipy.sparse.linalg.lsqr(A_copy.data, numpy.asarray(b.data))
        sol.data = numpy.asmatrix(solution[0])
    elif A.getTypeString() == 'Sparse':
        solution = scipy.sparse.linalg.lsqr(A.data, numpy.asarray(b.data))
        sol.data = numpy.asmatrix(solution[0])
    return sol.copyAs(A_original_type)
