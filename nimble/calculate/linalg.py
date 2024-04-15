
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
Linear algebra functions that can be used with nimble base objects.
"""

import re

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination, PackageException
from nimble._utility import scipy
from nimble._utility import dtypeConvert

def inverse(aObj):
    """
    Compute the (multiplicative) inverse of a nimble Base object.

    Parameters
    ----------
    aObj : nimble Base object.
        Square object to be inverted.

    Returns
    -------
    aInv : nimble Base object.
        Inverse of the object ``aObj``

    Raises
    ------
    InvalidArgumentType
        If ``aObj`` is not a nimble Base Object.
        If ``aObj`` elements types are not supported.
    InvalidArgumentValue
        If ``aObj`` is not square.
        If ``aObj`` is not invertible (Singular).

    Examples
    --------
    >>> lst = [[1, 2], [3, 4]]
    >>> X = nimble.data(lst)
    >>> X
    <Matrix 2pt x 2ft
         0  1
       ┌─────
     0 │ 1  2
     1 │ 3  4
    >
    >>> inverse(X)
    <Matrix 2pt x 2ft
           0       1
       ┌───────────────
     0 │ -2.000   1.000
     1 │  1.500  -0.500
    >
    """
    if not scipy.nimbleAccessible():
        msg = "scipy must be installed in order to use the inverse function."
        raise PackageException(msg)
    if not isinstance(aObj, nimble.core.data.Base):
        raise InvalidArgumentType(
            "Object must be derived class of nimble.core.data.Base")
    if not aObj.points and not aObj.features:
        return aObj.copy()
    if len(aObj.points) != len(aObj.features):
        msg = 'Object has to be square \
        (Number of features and points needs to be equal).'
        raise InvalidArgumentValue(msg)

    def _handleSingularCase(exception):
        if re.match('.*singular.*', str(exception), re.I):
            msg = 'Object non-invertible (Singular)'
            raise InvalidArgumentValue(msg)
        raise exception

    if aObj.getTypeString() in ['Matrix', 'DataFrame', 'List']:
        try:
            invObj = dtypeConvert(aObj.copy(to='numpy array'))
            invData = scipy.linalg.inv(invObj.data)
        except scipy.linalg.LinAlgError as exception:
            _handleSingularCase(exception)
        except ValueError as exception:
            if re.match('.*object arrays.*', str(exception), re.I):
                msg = 'Elements types in object data are not supported.'
                raise InvalidArgumentType(msg) from exception
            if re.match('(.*infs.*)|(.*nans.*)', str(exception), re.I):
                msg = 'Infs or NaNs values are not supported.'
                raise InvalidArgumentValue(msg) from exception
            raise exception

    else:
        try:
            invObj = aObj.copy(to='scipy csc')
            invData = scipy.sparse.linalg.inv(invObj)
        except RuntimeError as exception:
            _handleSingularCase(exception)
        except TypeError as exception:
            if re.match('.*no supported conversion*', str(exception), re.I):
                msg = 'Elements types in object data are not supported.'
                raise InvalidArgumentType(msg) from exception
            raise exception

    return nimble.data(invData, returnType=aObj.getTypeString(), useLog=False)


def pseudoInverse(aObj):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a nimble Base object.

    Calculate a generalized inverse of a matrix using singular-value
    decomposition solver.

    Parameters
    ----------
    aObj : nimble Base object.
        Square object to be pseudo-inverted.

    Returns
    -------
    aPInv : nimble Base object.
        Pseudo-inverse of the object ``aObj``

    Raises
    ------
    InvalidArgumentType
        If ``aObj`` is not a nimble Base Object.
        If ``aObj`` elements types are not supported.

    Examples
    --------
    >>> lst = [[1, 2], [3, 4]]
    >>> X = nimble.data(lst)
    >>> X
    <Matrix 2pt x 2ft
         0  1
       ┌─────
     0 │ 1  2
     1 │ 3  4
    >
    >>> pseudoInverse(X)
    <Matrix 2pt x 2ft
           0       1
       ┌───────────────
     0 │ -2.000   1.000
     1 │  1.500  -0.500
    >
    """
    if not scipy.nimbleAccessible():
        msg = "scipy must be installed in order to use the pseudoInverse "
        msg += "function."
        raise PackageException(msg)
    if not isinstance(aObj, nimble.core.data.Base):
        raise InvalidArgumentType(
            "Object must be derived class of nimble.core.data.Base.")
    if not aObj.points and not aObj.features:
        return aObj

    def _handleNonSupportedTypes(exception):
        if re.match('.*object arrays*', str(exception), re.I):
            msg = 'Elements types in object data are not supported.'
            raise InvalidArgumentType(msg) from exception
        if re.match('(.*infs.*)|(.*nans.*)', str(exception), re.I):
            msg = 'Infs or NaNs values are not supported.'
            raise InvalidArgumentValue(msg) from exception
        raise exception


    pinvObj = dtypeConvert(aObj.copy(to='numpy array'))
    try:
        pinvData = scipy.linalg.pinv(pinvObj)
    except ValueError as exception:
        _handleNonSupportedTypes(exception)

    return nimble.data(pinvData, returnType=aObj.getTypeString(), useLog=False)


def solve(aObj, bObj):
    """
    Solves the linear equation set A x = b for the unknown vector x.

    Parameters
    ----------
    aObj : (M, M) nimble Base object.
        Square object.
    bObj : (M) nimble Base object.
        Right-hand side nimble Base object in A x = b.

    Returns
    -------
    xObj : (M) nimble Base object.
        Solution to the system A x = b. Shape of ``xObj`` matches
        ``bObj``.

    Raises
    ------
    InvalidArgumentType
        If ``aObj`` or ``bObj`` is not a nimble Base Object.
        If ``aObj`` elements types are not supported.
    InvalidArgumentValue
        If ``aObj`` is not squared.
        If ``bObj`` is not a vector. 1-D.
    InvalidArgumentValueCombination
        If ``aObj`` and ``bObj`` have incompatible dimensions.

    Examples
    --------
    >>> aData = [[3,2,0],[1,-1,0],[0,5,1]]
    >>> aObj = nimble.data(aData)
    >>> bData = [2,4,-1]
    >>> bObj = nimble.data(bData)
    >>> aObj
    <Matrix 3pt x 3ft
         0  1   2
       ┌─────────
     0 │ 3   2  0
     1 │ 1  -1  0
     2 │ 0   5  1
    >
    >>> bObj
    <Matrix 1pt x 3ft
         0  1  2
       ┌─────────
     0 │ 2  4  -1
    >
    >>> xObj = solve(aObj, bObj)
    >>> xObj
    <Matrix 1pt x 3ft
           0      1       2
       ┌─────────────────────
     0 │ 2.000  -2.000  9.000
    >
    """
    return _backendSolvers(aObj, bObj, solve)


def leastSquaresSolution(aObj, bObj):
    """
    Compute least-squares solution to equation A x = b.

    Compute a vector x such that the 2-norm determinant of b - Ax is
    minimized. The matrix A may be square or rectangular
    (over-determined or under-determined).

    Parameters
    ----------
    aObj : (M, N) nimble Base object.
        Left hand side object in A x = b.
    bObj : (M) nimble Base object.
        Right-hand side nimble Base object in A x = b. (1-D)

    Returns
    -------
    xObj : (N) nimble Base object.
        Least-squares solution.

    Raises
    ------
    InvalidArgumentType
        If ``aObj`` or ``bObj`` is not a nimble Base Object.
        If ``aObj`` elements types are not supported.
    InvalidArgumentValue
        If ``bObj`` is not a vector. 1-D.
    InvalidArgumentValueCombination
        If ``aObj`` and ``bObj`` have incompatible dimensions.

    Examples
    --------
    >>> a = [[0, 1],
    ...      [1, 1],
    ...      [2, 1],
    ...      [3, 1],
    ...      [4, 1]]
    >>> b = [6, 9, 12, 15, 18]
    >>> aObj = nimble.data(a)
    >>> bObj = nimble.data(b)
    >>> nimble.calculate.leastSquaresSolution(aObj, bObj)
    <Matrix 1pt x 2ft
           0      1
       ┌─────────────
     0 │ 3.000  6.000
    >
    """
    return _backendSolvers(aObj, bObj, leastSquaresSolution)

def _backendSolvers(aObj, bObj, solverFunction):
    if not scipy.nimbleAccessible():
        msg = "scipy must be installed in order to use the pseudoInverse "
        msg += "function."
        raise PackageException(msg)
    bObj = _backendSolversValidation(aObj, bObj, solverFunction)

    aOriginalType = aObj.getTypeString()
    if aObj.getTypeString() in ['DataFrame', 'List']:
        aObj = aObj.copy(to='Matrix')
    if bObj.getTypeString() in ['DataFrame', 'List', 'Sparse']:
        bObj = bObj.copy(to='Matrix')

    # Solve
    if aObj.getTypeString() == 'Matrix':
        aData = dtypeConvert(aObj._data)
        bData = dtypeConvert(bObj._data)
        if solverFunction.__name__ == 'solve':
            solution = scipy.linalg.solve(aData, bData)
            solution = solution.T
        else:
            solution = scipy.linalg.lstsq(aData, bData)
            solution = solution[0].T

    elif aObj.getTypeString() == 'Sparse':
        aCopy = aObj.copy()
        aData = dtypeConvert(aCopy._data)
        bData = dtypeConvert(np.asarray(bObj._data))
        if solverFunction.__name__ == 'solve':
            solution = scipy.sparse.linalg.spsolve(aData, bData)
        else:
            solution = scipy.sparse.linalg.lsqr(aData, bData)
            solution = solution[0]

    sol = nimble.data(solution, featureNames=aObj.features.getNames(),
                      returnType=aOriginalType, useLog=False)
    return sol



def _backendSolversValidation(aObj, bObj, solverFunction):
    if not isinstance(aObj, nimble.core.data.Base):
        msg = "Left hand side object must be derived class of "
        msg += "nimble.core.data.Base"
        raise InvalidArgumentType(msg)
    if not isinstance(bObj, nimble.core.data.Base):
        msg = "Right hand side object must be derived class of "
        msg += "nimble.core.data.Base"
        raise InvalidArgumentType(msg)

    if (solverFunction.__name__ == 'solve'
            and len(aObj.points) != len(aObj.features)):
        msg = 'Object A has to be square (Number of features and points needs '
        msg += 'to be equal).'
        raise InvalidArgumentValue(msg)

    if len(bObj.points) != 1 and len(bObj.features) != 1:
        raise InvalidArgumentValue("b should be a vector")
    if len(bObj.points) == 1 and len(bObj.features) > 1:
        if len(aObj.points) != len(bObj.features):
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')
        bObj = bObj.T
    if len(bObj.points) > 1 and len(bObj.features) == 1:
        if len(aObj.points) != len(bObj.points):
            raise InvalidArgumentValueCombination(
                'A and b have incompatible dimensions.')
    return bObj
