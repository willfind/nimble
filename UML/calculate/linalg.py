"""
Linear algebra functions that can be used with UML base objects.
"""

from __future__ import absolute_import
import re
import numpy
import scipy

import UML
from UML.exceptions import InvalidArgumentType, \
    InvalidArgumentValue, \
    InvalidArgumentValueCombination


def inverse(A):
    """
       Compute the (multiplicative) inverse of a UML Base object.

    """
    if not isinstance(A, UML.data.Base):
        raise InvalidArgumentType(
            "Object must be derived class of UML.data.Base")
    if not len(A.points) and not len(A.features):
        return A.copy()
    if len(A.points) != len(A.features):
        msg = 'Object has to be square \
        (Number of features and points needs to be equal).'
        raise InvalidArgumentValue(msg)

    def _handleSingularCase(exception):
        if re.match('.*singular.*', str(exception), re.I):
            msg = 'Object non-invertible (Singular)'
            raise InvalidArgumentValue(msg)
        else:
            raise exception

    if A.getTypeString() in ['Matrix', 'DataFrame', 'List']:
        invObj = A.copyAs('Matrix')
        try:
            invData = scipy.linalg.inv(invObj.data)
        except scipy.linalg.LinAlgError as exception:
            _handleSingularCase(exception)
        except ValueError as exception:
            if re.match('.*object arrays*', str(exception), re.I):
                msg = 'Elements types in object data are not supported.'
                raise InvalidArgumentType(msg)
    else:
        invObj = A.copyAs('Sparse')
        try:
            invData = scipy.sparse.linalg.inv(invObj.data.tocsc())
        except RuntimeError as exception:
            _handleSingularCase(exception)
        except TypeError as exception:
            if re.match('.*no supported conversion*', str(exception), re.I):
                msg = 'Elements types in object data are not supported.'
                raise InvalidArgumentType(msg)

    invObj.transpose()
    invObj.data = invData
    if A.getTypeString() != invObj.getTypeString:
        invObj = invObj.copyAs(A.getTypeString())
    return invObj


def pseudoInverse(A, method='svd'):
    """
        Compute the (Moore-Penrose) pseudo-inverse of a UML object.
        Method: 'svd' or 'least-squares'.
        Uses singular-value decomposition by default.
        Least squares solver included as an option.
    """
    if not isinstance(A, UML.data.Base):
        raise InvalidArgumentType(
            "Object must be derived class of UML.data.Base.")
    if not len(A.points) and not len(A.features):
        return A
    if method not in ['least-squares', 'svd']:
        raise InvalidArgumentValue(
            "Supported methods are 'least-squares' and 'svd'.")

    def _handleNonSupportedTypes(exception):
        if re.match('.*object arrays*', str(exception), re.I):
            msg = 'Elements types in object data are not supported.'
            raise InvalidArgumentType(msg)

    pinvObj = A.copyAs('Matrix')
    if method == 'svd':
        try:
            pinvData = scipy.linalg.pinv2(pinvObj.data)
        except ValueError as exception:
            _handleNonSupportedTypes(exception)
    else:
        pinvData = scipy.linalg.pinv(pinvObj.data)
    pinvObj.transpose()
    pinvObj.data = numpy.asmatrix(pinvData)
    if A.getTypeString() != 'Matrix':
        pinvObj = pinvObj.copyAs(A.getTypeString())
    return pinvObj


def solve(A, b):
    """
        Solves the linear equation set A * x = b for the unknown vector x.
        A should be a square a object.
    """
    if not isinstance(A, UML.data.Base):
        raise InvalidArgumentType(
            "Left hand side object must be derived class of UML.data.Base.")
    if not isinstance(b, UML.data.Base):
        raise InvalidArgumentType(
            "Right hand side object must be derived class of UML.data.Base.")
    if len(A.points) != len(A.features):
        msg = 'Object A has to be square \
        (Number of features and points needs to be equal).'
        raise InvalidArgumentValue(msg)
    if len(b.points) != 1 and len(b.features) != 1:
        raise InvalidArgumentValue("b should be a vector")
    elif len(b.points) == 1 and len(b.features) > 1:
        if len(A.points) != len(b.features):
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')
        else:
            b = b.copy()
            b.flattenToOneFeature()
    elif len(b.points) > 1 and len(b.features) == 1:
        if len(A.points) != len(b.points):
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')

    aOriginalType = A.getTypeString()
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
        aCopy = A.copy()
        solution = scipy.sparse.linalg.spsolve(aCopy.data,
                                               numpy.asarray(b.data))
        sol.data = numpy.asmatrix(solution)
    elif A.getTypeString() == 'Sparse':
        solution = scipy.sparse.linalg.spsolve(A.data,
                                               numpy.asarray(b.data))
        sol.data = numpy.asmatrix(solution)

    return sol.copyAs(aOriginalType)


def leastSquaresSolution(A, b):
    """
        Compute least-squares solution to equation Ax = b
    """
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
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')
        else:
            b = b.copy()
            b.flattenToOneFeature()
    elif len(b.points) > 1 and len(b.features) == 1:
        if len(A.points) != len(b.points):
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')

    aOriginalType = A.getTypeString()
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
        aCopy = A.copy()
        solution = scipy.sparse.linalg.lsqr(aCopy.data, numpy.asarray(b.data))
        sol.data = numpy.asmatrix(solution[0])
    elif A.getTypeString() == 'Sparse':
        solution = scipy.sparse.linalg.lsqr(A.data, numpy.asarray(b.data))
        sol.data = numpy.asmatrix(solution[0])
    return sol.copyAs(aOriginalType)
