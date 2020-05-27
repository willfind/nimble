"""
Tests for linear algebra functions part of the calculate module.

Functions tested in this file:
inverse, pseudoInverse, solve, leastSquaresSolution.
"""

import numpy
from nose.tools import raises
import nimble
from nimble import createData
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.calculate import inverse, pseudoInverse, leastSquaresSolution, solve


###########
# inverse #
###########

def testInverseSquareObject():
    """
        Test success of inverse for a square object and
        test that input object is not modified.mm
    """
    data = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
    pnames = ['p1', 'p2', 'p3']
    fnames = ['f1', 'f2', 'f3']

    for dataType in nimble.core.data.available:
        identityObj = nimble.identity(dataType, 3)
        origObj = createData(
            dataType, data, pointNames=pnames, featureNames=fnames)
        obj = createData(dataType, data, pointNames=pnames,
                         featureNames=fnames)
        objNoNames = createData(dataType, data)

        objInv = inverse(obj)

        assert objNoNames @ objInv == identityObj
        assert origObj == obj


def testInverseEmptyObject():
    """
    Test inverse for an empty object.
    """
    data = []

    for dataType in nimble.core.data.available:
        obj = createData(dataType, data)
        objInv = inverse(obj)
        assert objInv == obj


def testInverseNonSquareObject():
    """
    Test inverse for non square object.
    """
    data = [[1, 2, 3], [4, 5, 6]]

    for dataType in nimble.core.data.available:
        obj = createData(dataType, data)
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

    for dataType in nimble.core.data.available:
        obj = createData(dataType, data)

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
    def _testPseudoInverseCreateObjects(dataType):

        objsList = []

        data = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
        pnames = ['p1', 'p2', 'p3']
        fnames = ['f1', 'f2', 'f3']
        obj = createData(dataType, data, pointNames=pnames,
                         featureNames=fnames)
        objsList.append(obj)

        data = [[1, 2, 3], [3, 4, 5]]
        pnames = ['p1', 'p2']
        fnames = ['f1', 'f2', 'f3']
        obj = createData(dataType, data, pointNames=pnames,
                         featureNames=fnames)
        objsList.append(obj)

        data = [[1, 2], [3, 4], [5, 6]]
        pnames = ['p1', 'p2', 'p3']
        fnames = ['f1', 'f2']
        obj = createData(dataType, data, pointNames=pnames,
                         featureNames=fnames)
        objsList.append(obj)

        data = [[1, 1], [1, 1]]
        obj = createData(dataType, data)
        objsList.append(obj)

        data = [[1, 2]]
        obj = createData(dataType, data)
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

    for dataType in nimble.core.data.available:
        for method in ['least-squares', 'svd']:
            objList = _testPseudoInverseCreateObjects(dataType)
            for obj in objList:
                _pseudoInverseTestImplementation(obj, method)


def testPseudoInverseEmptyObject():
    """
        Test pseudoInverse for an empty object.
    """
    data = []

    for dataType in nimble.core.data.available:
        obj = createData(dataType, data)
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
    aArray = numpy.array([[1, 2], [4, 5], [3, 4]])
    bArrays = [numpy.array([1, 2, 3]), numpy.array([[1], [2], [3]])]
    featureNames = ['f1', 'f2']

    _backendNonSquareSolverSucces(aArray, bArrays, featureNames)


def testLeastSquareSolutionUnderdetermined():
    """
        Test success for leastSquareSolution under-determined system.
    """
    aArray = numpy.array([[1, 2, 3], [4, 5, 6]])
    bArrays = [numpy.array([1, 2]), numpy.array([[1], [2]])]
    featureNames = ['f1', 'f2', 'f3']

    _backendNonSquareSolverSucces(aArray, bArrays, featureNames)


def _backendSolverSuccess(solverFunction):
    aArray = numpy.array([[1, 20], [-30, 4]])
    bArrays = [numpy.array([-30, 4]), numpy.array([[-30], [4]])]

    for dataType in nimble.core.data.available:
        for dataTypeB in nimble.core.data.available:
            for bArray in bArrays:
                A = createData(dataType, aArray, featureNames=['f1', 'f2'])
                b = createData(dataTypeB, bArray)
                sol = solverFunction(A, b)
                aInv = inverse(A)
                if len(b.features) > 1:
                    b.transpose()
                reference = (aInv @ b)
                reference.transpose()
                # reference.points.setNames(['b'])
                assert sol.isApproximatelyEqual(reference)
                assert A.features.getNames() == sol.features.getNames()
                # assert sol.points.getNames() == ['b']
                assert sol.getTypeString() == A.getTypeString()

def _backendNonSquareSolverSucces(aArray,  bArrays, featureNames):
    for dataType in nimble.core.data.available:
        for dataTypeB in nimble.core.data.available:
            for bArray in bArrays:
                aOrig = createData(
                    dataType, aArray, featureNames=featureNames)
                A = createData(dataType, aArray,
                               featureNames=featureNames)
                bOrig = createData(dataTypeB, bArray)
                b = createData(dataTypeB, bArray)
                sol = leastSquaresSolution(A, b)

                assert A == aOrig
                assert b == bOrig

                aPinv = pseudoInverse(A)
                if len(b.features) > 1:
                    b.transpose()
                reference = (aPinv @ b)
                reference.transpose()
                assert sol.isApproximatelyEqual(reference)
                assert A.features.getNames() == sol.features.getNames()
                assert sol.getTypeString() == A.getTypeString()
