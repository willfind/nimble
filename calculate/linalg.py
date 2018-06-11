from __future__ import absolute_import
import scipy
import re


import UML
from UML.exceptions import ArgumentException


def inverse(A):
    """
       Compute the (multiplicative) inverse of an UML object
    """
    if not isinstance(A, UML.data.Base):
        raise ArgumentException("Object must be derived class of UML.data.Base")
    if A.points == 0 and A.features == 0:
        return A.copy()
    if A.points != A.features:
        msg = 'Object has to be square (Number of features and points needs to be equal).'
        raise ArgumentException(msg)

    def _handleSingularCase(e):
        if re.match('.*singular.*', str(e), re.I):
            msg = 'Object non-invertible (Singular)'
            raise ArgumentException(msg)
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
                raise ArgumentException(msg)
    else:
        inv_obj = A.copyAs('Sparse')
        try:
            inv_data = scipy.sparse.linalg.inv(inv_obj.data.tocsc())
        except RuntimeError as e:
            _handleSingularCase(e)
        except TypeError as e:
            if re.match('.*no supported conversion*', str(e), re.I):
                msg = 'Elements types in object data are not supported.'
                raise ArgumentException(msg)

    inv_obj.transpose()
    inv_obj.data = inv_data
    if A.getTypeString() != inv_obj.getTypeString:
        inv_obj = inv_obj.copyAs(A.getTypeString())
    return inv_obj


def pseudoInverse(A, method='least-squares'):
    """
        Compute the (Moore-Penrose) pseudo-inverse of a UML object.
        Method: 'least-squares' or 'svd. 
        Uses least squares solver by default and supports singular-value decomposition.

    """
    if not isinstance(A, UML.data.Base):
        raise ArgumentException("Object must be derived class of UML.data.Base.")
    if A.points == 0 and A.features == 0:
        return A
    if method not in ['least-squares', 'svd']:
        raise ArgumentException("Supported methods are 'least-squares' and 'svd'.")

    def _handleNonSupportedTypes(e):
        if re.match('.*object arrays*', str(e), re.I):
                msg = 'Elements types in object data are not supported.'
                raise ArgumentException(msg)

    pinv_obj = A.copyAs('Matrix')
    if method == 'least-squares':
        try:
            pinv_data = scipy.linalg.pinv(pinv_obj.data)
        except ValueError as e:
            _handleNonSupportedTypes(e)
    else:
        pinv_data = scipy.linalg.pinv2(pinv_obj.data)
    pinv_obj.transpose()
    pinv_obj.data = pinv_data
    if A.getTypeString() != 'Matrix':
        pinv_obj = pinv_obj.copyAs(A.getTypeString())
    return pinv_obj

