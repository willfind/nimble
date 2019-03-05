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


def inverse(aObj):
    """
    Compute the (multiplicative) inverse of a UML Base object.

    Parameters
    ----------
    aObj : UML Base object.
        Square object to be inverted.

    Returns
    -------
    aInv : UML Base object.
        Inverse of the object `aObj`

    Raises
    ------
    InvalidArgumentType:
        If `aObj` is not a UML Base Object.
        If `aObj` elements types are not supported.

    InvalidArgumentValue:
        If `aObj` is not square.
        If `aObj` is not invertible (Singular).

    Examples
    --------
    >>> from UML.calculate import inverse
    >>> raw = [[1, 2], [3, 4]]
    >>> data = UML.createData('Matrix', raw)
    >>> data
    Matrix(
    [[1.000 2.000]
     [3.000 4.000]]
    )
    >>> inverse(data)
    Matrix(
    [[-2.000 1.000 ]
     [1.500  -0.500]]
    )
    """
    if not isinstance(aObj, UML.data.Base):
        raise InvalidArgumentType(
            "Object must be derived class of UML.data.Base")
    if not len(aObj.points) and not len(aObj.features):
        return aObj.copy()
    if len(aObj.points) != len(aObj.features):
        msg = 'Object has to be square \
        (Number of features and points needs to be equal).'
        raise InvalidArgumentValue(msg)

    def _handleSingularCase(exception):
        if re.match('.*singular.*', str(exception), re.I):
            msg = 'Object non-invertible (Singular)'
            raise InvalidArgumentValue(msg)
        else:
            raise exception

    if aObj.getTypeString() in ['Matrix', 'DataFrame', 'List']:
        invObj = aObj.copyAs('Matrix')
        try:
            invData = scipy.linalg.inv(invObj.data)
        except scipy.linalg.LinAlgError as exception:
            _handleSingularCase(exception)
        except ValueError as exception:
            if re.match('.*object arrays*', str(exception), re.I):
                msg = 'Elements types in object data are not supported.'
                raise InvalidArgumentType(msg)
    else:
        invObj = aObj.copyAs('Sparse')
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
    if aObj.getTypeString() != invObj.getTypeString:
        invObj = invObj.copyAs(aObj.getTypeString())
    return invObj


def pseudoInverse(aObj, method='svd'):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a UML Base object.

    Calculate a generalized inverse of a matrix using singular-value
    decomposition (default) or least squares solver.

    Parameters
    ----------
    aObj : UML Base object.
        Square object to be pseudo-inverted.
    method : str.
        - 'svd'. Uses singular-value decomposition by default.
        - 'least-squares'.  Uses least squares solver included.

    Returns
    -------
    aPInv : UML Base object.
        Pseudo-inverse of the object `aObj`

    Raises
    ------
    InvalidArgumentType:
        If `aObj` is not a UML Base Object.
        If `aObj` elements types are not supported.

    InvalidArgumentValue:
        If `method` name is not supported.

    Examples
    --------
    >>> from UML.calculate import pseudoInverse
    >>> data = UML.createRandomData('Matrix',
    >>>                              numPoints=4,
    >>>                              numFeatures=3,
    >>>                              sparsity=0.5)
    >>> data
    Matrix(
        [[0.000  -0.000 -0.000]
         [0.774  -0.538 -0.000]
         [-0.000 -0.000 0.000 ]
         [0.582  0.888  0.894 ]]
    )
    >>> pseudoInverse(data)
    Matrix(
        [[0.000  0.881  0.000 0.315]
         [-0.000 -0.592 0.000 0.452]
         [-0.000 0.014  0.000 0.464]]
    )
    """
    if not isinstance(aObj, UML.data.Base):
        raise InvalidArgumentType(
            "Object must be derived class of UML.data.Base.")
    if not len(aObj.points) and not len(aObj.features):
        return aObj
    if method not in ['least-squares', 'svd']:
        raise InvalidArgumentValue(
            "Supported methods are 'least-squares' and 'svd'.")

    def _handleNonSupportedTypes(exception):
        if re.match('.*object arrays*', str(exception), re.I):
            msg = 'Elements types in object data are not supported.'
            raise InvalidArgumentType(msg)

    pinvObj = aObj.copyAs('Matrix')
    if method == 'svd':
        try:
            pinvData = scipy.linalg.pinv2(pinvObj.data)
        except ValueError as exception:
            _handleNonSupportedTypes(exception)
    else:
        pinvData = scipy.linalg.pinv(pinvObj.data)
    pinvObj.transpose()
    pinvObj.data = numpy.asmatrix(pinvData)
    if aObj.getTypeString() != 'Matrix':
        pinvObj = pinvObj.copyAs(aObj.getTypeString())
    return pinvObj


def solve(aObj, bObj):
    """
    Solves the linear equation set A x = b for the unknown vector x.

    Parameters
    ----------
    aObj : (M, M) UML Base object.
        Square object.
    bObj : (M) UML Base object.
        Right-hand side UML Base object in A x = b.

    Returns
    -------
    xObj : (M) UML Base object.
        Solution to the system A x = b. Shape of `xObj` matches `bObj`.

    Raises
    ------
    InvalidArgumentType:
        If `aObj` or `bObj` is not a UML Base Object.
        If `aObj` elements types are not supported.

    InvalidArgumentValue:
        If `aObj` is not squared.
        If `bObj`is not a vector. 1-D.

    InvalidArgumentValueCombination:
        If `aObj` and `bObj` have incompatible dimensions.

    Examples
    --------
    >>> from UML.calculate import solve
    >>> aData = [[3,2,0],[1,-1,0],[0,5,1]]
    >>> aObj = UML.createData('Matrix', aData)
    >>> bData = [2,4,-1]
    >>> bObj = UML.createData('Matrix', bData)
    >>> aObj
    Matrix(
        [[3.000 2.000  0.000]
         [1.000 -1.000 0.000]
         [0.000 5.000  1.000]]
    )
    >>> bObj
    Matrix(
        [[2.000 4.000 -1.000]]
    )
    >>> xObj = solve(aObj, bObj)
    >>> xObj
    Matrix(
        [[2.000 -2.000 9.000]]
    )
    """
    if not isinstance(aObj, UML.data.Base):
        raise InvalidArgumentType(
            "Left hand side object must be derived class of UML.data.Base.")
    if not isinstance(bObj, UML.data.Base):
        raise InvalidArgumentType(
            "Right hand side object must be derived class of UML.data.Base.")
    if len(aObj.points) != len(aObj.features):
        msg = 'Object A has to be square \
        (Number of features and points needs to be equal).'
        raise InvalidArgumentValue(msg)
    if len(bObj.points) != 1 and len(bObj.features) != 1:
        raise InvalidArgumentValue("b should be a vector")
    elif len(bObj.points) == 1 and len(bObj.features) > 1:
        if len(aObj.points) != len(bObj.features):
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')
        else:
            bObj = bObj.copy()
            bObj.flattenToOneFeature()
    elif len(bObj.points) > 1 and len(bObj.features) == 1:
        if len(aObj.points) != len(bObj.points):
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')

    aOriginalType = aObj.getTypeString()
    if aObj.getTypeString() in ['DataFrame', 'List']:
        aObj = aObj.copyAs('Matrix')
    if bObj.getTypeString() in ['DataFrame', 'List', 'Sparse']:
        bObj = bObj.copyAs('Matrix')

    sol = aObj[0, :].copyAs('Matrix')
    sol.points.setNames(['b'])

    if aObj.getTypeString() == 'Matrix':
        solution = scipy.linalg.solve(aObj.data, bObj.data)
        sol.data = solution.T
    elif isinstance(aObj, UML.data.sparse.SparseView):
        aCopy = aObj.copy()
        solution = scipy.sparse.linalg.spsolve(aCopy.data,
                                               numpy.asarray(bObj.data))
        sol.data = numpy.asmatrix(solution)
    elif aObj.getTypeString() == 'Sparse':
        solution = scipy.sparse.linalg.spsolve(aObj.data,
                                               numpy.asarray(bObj.data))
        sol.data = numpy.asmatrix(solution)

    return sol.copyAs(aOriginalType)


def leastSquaresSolution(aObj, bObj):
    """
        Compute least-squares solution to equation Ax = b
    """
    if not isinstance(aObj, UML.data.Base):
        raise InvalidArgumentType(
            "Left hand side object must be derived class of UML.data.Base.")
    if not isinstance(bObj, UML.data.Base):
        raise InvalidArgumentType(
            "Right hand side object must be derived class of UML.data.Base.")
    if len(bObj.points) != 1 and len(bObj.features) != 1:
        raise InvalidArgumentValue("b should be a vector")
    elif len(bObj.points) == 1 and len(bObj.features) > 1:
        if len(aObj.points) != len(bObj.features):
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')
        else:
            bObj = bObj.copy()
            bObj.flattenToOneFeature()
    elif len(bObj.points) > 1 and len(bObj.features) == 1:
        if len(aObj.points) != len(bObj.points):
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')

    aOriginalType = aObj.getTypeString()
    if aObj.getTypeString() in ['DataFrame', 'List']:
        aObj = aObj.copyAs('Matrix')
    if bObj.getTypeString() in ['DataFrame', 'List', 'Sparse']:
        bObj = bObj.copyAs('Matrix')

    sol = aObj[0, :].copyAs('Matrix')
    sol.points.setNames(['b'])
    if aObj.getTypeString() == 'Matrix':
        solution = scipy.linalg.lstsq(aObj.data, bObj.data)
        sol.data = solution[0].T
    elif isinstance(aObj, UML.data.sparse.SparseView):
        aCopy = aObj.copy()
        solution = scipy.sparse.linalg.lsqr(aCopy.data,
                                            numpy.asarray(bObj.data))
        sol.data = numpy.asmatrix(solution[0])
    elif aObj.getTypeString() == 'Sparse':
        solution = scipy.sparse.linalg.lsqr(aObj.data,
                                            numpy.asarray(bObj.data))
        sol.data = numpy.asmatrix(solution[0])
    return sol.copyAs(aOriginalType)
