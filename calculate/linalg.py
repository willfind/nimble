from __future__ import absolute_import
import numpy
import scipy


import UML
from UML.exceptions import ArgumentException


def inverse(umlObject):
    if not isinstance(umlObject, UML.data.Base):
        raise ArgumentException("Object must be derived class of UML.data.Base")
    if umlObject.getTypeString() in ['Matrix', 'DataFrame', 'List']:
        inv_obj = umlObject.copyAs('Matrix')
        try:
            inv_data = numpy.linalg.inv(inv_obj.data)
        except numpy.linalg.LinAlgError:
            msg = 'Object has to be square (Number of features and points needs to be equal).'
            raise ArgumentException(msg)
    else:
        inv_obj = umlObject.copyAs('Sparse')
        try:
            inv_data = scipy.sparse.linalg.inv(inv_obj.data.tocsr()).tocoo()
        except ValueError:
            msg = 'Object has to be square (Number of features and points needs to be equal).'
            raise ArgumentException(msg)

    inv_obj.transpose()
    inv_obj.data = inv_data
    if umlObject.getTypeString() != inv_obj.getTypeString:
        inv_obj = inv_obj.copyAs(umlObject.getTypeString())
    return inv_obj


def pseudoInverse(umlObject):
    if not isinstance(umlObject, UML.data.Base):
        raise ArgumentException("Object must be derived class of UML.data.Base")
    pinv_obj = umlObject.copyAs('Matrix')
    pinv_data = numpy.linalg.pinv(pinv_obj.data)
    pinv_obj.transpose()
    pinv_obj.data = pinv_data
    if umlObject.getTypeString() != 'Matrix':
        pinv_obj = pinv_obj.copyAs(umlObject.getTypeString())
    return pinv_obj
