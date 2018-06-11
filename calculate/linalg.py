from __future__ import absolute_import
import scipy
import re


import UML
from UML.exceptions import ArgumentException


def inverse(umlObject):
    """
       Compute the (multiplicative) inverse of an UML object
    """
    if not isinstance(umlObject, UML.data.Base):
        raise ArgumentException("Object must be derived class of UML.data.Base")
    if umlObject.points == 0 and umlObject.features == 0:
        return umlObject.copy()
    if umlObject.points != umlObject.features:
        msg = 'Object has to be square (Number of features and points needs to be equal).'
        raise ArgumentException(msg)

    def _handleSingularCase(e):
        if re.match('.*singular.*', str(e), re.I):
            msg = 'Object non-invertible (Singular)'
            raise ArgumentException(msg)
        else:
            raise(e)

    if umlObject.getTypeString() in ['Matrix', 'DataFrame', 'List']:
        inv_obj = umlObject.copyAs('Matrix')
        try:
            inv_data = scipy.linalg.inv(inv_obj.data)
        except scipy.linalg.LinAlgError as e:
            _handleSingularCase(e)
    else:
        inv_obj = umlObject.copyAs('Sparse')
        try:
            inv_data = scipy.sparse.linalg.inv(inv_obj.data.tocsr()).tocoo()
        except RuntimeError as e:
            _handleSingularCase(e)

    inv_obj.transpose()
    inv_obj.data = inv_data
    if umlObject.getTypeString() != inv_obj.getTypeString:
        inv_obj = inv_obj.copyAs(umlObject.getTypeString())
    return inv_obj


def pseudoInverse(umlObject, method='least-squares'):
    """
        Compute the (Moore-Penrose) pseudo-inverse of a UML object.
        method: 'least-squares' or 'svd'. 
        Uses least squares solver by default and supports singular-value decomposition.

    """
    if not isinstance(umlObject, UML.data.Base):
        raise ArgumentException("Object must be derived class of UML.data.Base.")
    if umlObject.points == 0 and umlObject.features == 0:
        return umlObject
    if method not in ['least-squares', 'svd']:
        raise ArgumentException("Supported methods are 'least-squares' and 'svd'.")

    pinv_obj = umlObject.copyAs('Matrix')
    if method == 'least-squares':
        pinv_data = scipy.linalg.pinv(pinv_obj.data)
    else:
        pinv_data = scipy.linalg.pinv2(pinv_obj.data)
    pinv_obj.transpose()
    pinv_obj.data = pinv_data
    if umlObject.getTypeString() != 'Matrix':
        pinv_obj = pinv_obj.copyAs(umlObject.getTypeString())
    return pinv_obj
