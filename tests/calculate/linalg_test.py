"""
Tests for linear algebra functions part of the calculate module.

Functions tested in this file:
inverse, pseudoInverse, solve, leastSquaresSolution.
"""

import numpy as np
from nose.tools import raises

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.calculate import inverse, pseudoInverse, leastSquaresSolution, solve

from tests.helpers import getDataConstructors
###########
# inverse #
###########

def testInverseSquareObject():
    """
    Test success of inverse for a square object and
    test that input object is not modified.
    """
    data = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
    pnames = ['p1', 'p2', 'p3']
    fnames = ['f1', 'f2', 'f3']

    for constructor in getDataConstructors():
        identityObj = constructor(nimble.identity('Matrix', 3))
        origObj = constructor(data, pointNames=pnames, featureNames=fnames)
        obj = constructor(data, pointNames=pnames, featureNames=fnames)
        objNoNames = constructor(data)

        objInv = inverse(obj)

        assert objNoNames @ objInv == identityObj
        assert origObj == obj


def testInverseEmptyObject():
    """
    Test inverse for an empty object.
    """
    data = []

    for constructor in getDataConstructors():
        obj = constructor(data)
        objInv = inverse(obj)
        assert objInv == obj


def testInverseNonSquareObject():
    """
    Test inverse for non square object.
    """
    data = [[1, 2, 3], [4, 5, 6]]

    for constructor in getDataConstructors():
        obj = constructor(data)
        try:
            inverse(obj)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass


def testNonInvertibleObject():
    """
    Test inverse for non invertible object.
    """
    data = [[1, 1], [1, 1]]

    for constructor in getDataConstructors():
        obj = constructor(data)

        try:
            inverse(obj)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass


#################
# pseudoInverse #
#################


def testPseudoInverseObject():
    """
    Test success of pseudoInverse for valid objects cases.
    This includes, square objects, non-square objects and
    singular objects.
    """
    def _testPseudoInverseCreateObjects(constructor):

        objsList = []

        data = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
        pnames = ['p1', 'p2', 'p3']
        fnames = ['f1', 'f2', 'f3']
        obj = constructor(data, pointNames=pnames, featureNames=fnames)
        objsList.append(obj)

        data = [[1, 2, 3], [3, 4, 5]]
        pnames = ['p1', 'p2']
        fnames = ['f1', 'f2', 'f3']
        obj = constructor(data, pointNames=pnames, featureNames=fnames)
        objsList.append(obj)

        data = [[1, 2], [3, 4], [5, 6]]
        pnames = ['p1', 'p2', 'p3']
        fnames = ['f1', 'f2']
        obj = constructor(data, pointNames=pnames, featureNames=fnames)
        objsList.append(obj)

        data = [[1, 1], [1, 1]]
        obj = constructor(data)
        objsList.append(obj)

        data = [[1, 2]]
        obj = constructor(data)
        objsList.append(obj)

        return objsList

    def _pseudoInverseTestImplementation(obj, method):
        origObj = obj.copy()
        identity = nimble.identity(obj.getTypeString(),
                                   min(len(obj.points), len(obj.features)))
        objPinv = pseudoInverse(obj, method=method)

        if len(obj.points) <= len(obj.features):
            identityFromPinv = obj @ objPinv
        else:
            identityFromPinv = objPinv @ obj
        assert identityFromPinv.isApproximatelyEqual(identity)
        assert origObj == obj

    for constructor in getDataConstructors():
        for method in ['least-squares', 'svd']:
            objList = _testPseudoInverseCreateObjects(constructor)
            for obj in objList:
                _pseudoInverseTestImplementation(obj, method)


def testPseudoInverseEmptyObject():
    """
    Test pseudoInverse for an empty object.
    """
    data = []

    for constructor in getDataConstructors():
        obj = constructor(data)
        objInv = pseudoInverse(obj)
        assert objInv == obj

#########
# solve #
#########


def testSolveSuccess():
    """
    Test success for solve
    """
    _backendSolverSuccess(solve)


#######################
# leastSquareSolution #
#######################

def testLeastSquareSolutionExact():
    """
    Test success for leastSquareSolution exact case.
    """
    _backendSolverSuccess(leastSquaresSolution)


def testLeastSquareSolutionOverdetermined():
    """
    Test success for leastSquareSolution overdetermined system.
    """
    aArray = np.array([[1, 2], [4, 5], [3, 4]])
    bArrays = [np.array([1, 2, 3]), np.array([[1], [2], [3]])]
    featureNames = ['f1', 'f2']

    _backendNonSquareSolverSucces(aArray, bArrays, featureNames)


def testLeastSquareSolutionUnderdetermined():
    """
    Test success for leastSquareSolution under-determined system.
    """
    aArray = np.array([[1, 2, 3], [4, 5, 6]])
    bArrays = [np.array([1, 2]), np.array([[1], [2]])]
    featureNames = ['f1', 'f2', 'f3']

    _backendNonSquareSolverSucces(aArray, bArrays, featureNames)


def _backendSolverSuccess(solverFunction):
    aArray = np.array([[1, 20], [-30, 4]])
    bArrays = [np.array([-30, 4]), np.array([[-30], [4]])]

    for constructor in getDataConstructors():
        for constructorB in getDataConstructors():
            for bArray in bArrays:
                A = constructor(aArray, featureNames=['f1', 'f2'])
                b = constructorB(bArray)
                sol = solverFunction(A, b)
                aInv = inverse(A)
                if len(b.features) > 1:
                    reference = (aInv @ b.T)
                else:
                    reference = (aInv @ b)
                reference.transpose()
                # reference.points.setNames(['b'])
                assert sol.isApproximatelyEqual(reference)
                assert A.features.getNames() == sol.features.getNames()
                # assert sol.points.getNames() == ['b']
                assert sol.getTypeString() == A.getTypeString()

def _backendNonSquareSolverSucces(aArray,  bArrays, featureNames):
    for constructor in getDataConstructors():
        for constructorB in getDataConstructors():
            for bArray in bArrays:
                aOrig = constructor(aArray, featureNames=featureNames)
                A = constructor(aArray, featureNames=featureNames)
                bOrig = constructorB(bArray)
                b = constructorB(bArray)
                sol = leastSquaresSolution(A, b)

                assert A == aOrig
                assert b == bOrig

                aPinv = pseudoInverse(A)
                if len(b.features) > 1:
                    reference = (aPinv @ b.T)
                else:
                    reference = (aPinv @ b)
                reference.transpose()
                assert sol.isApproximatelyEqual(reference)
                assert A.features.getNames() == sol.features.getNames()
                assert sol.getTypeString() == A.getTypeString()
