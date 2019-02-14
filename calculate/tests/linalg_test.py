from __future__ import absolute_import
import numpy

from nose.tools import raises
import UML
from UML import createData
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from UML.calculate import inverse, pseudoInverse, leastSquaresSolution, solve


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

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        identity_obj = UML.identity(
            dataType, 3, pointNames=pnames, featureNames=pnames)
        orig_obj = createData(
            dataType, data, pointNames=pnames, featureNames=fnames)
        obj = createData(dataType, data, pointNames=pnames,
                         featureNames=fnames)
        obj_inv = inverse(obj)

        assert obj_inv.points.getNames() == obj.features.getNames()
        assert obj_inv.features.getNames() == obj.points.getNames()
        assert (obj * obj_inv) == identity_obj
        assert orig_obj == obj


def testInverseEmptyObject():
    """
        Test inverse for an empty object.
    """
    data = []

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        obj = createData(dataType, data)
        obj_inv = inverse(obj)
        assert obj_inv == obj

@raises(InvalidArgumentValue)
def testInverseNonSquareObject():
    """
        Test inverse for non square object.
    """
    data = [[1, 2, 3], [4, 5, 6]]

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        obj = createData(dataType, data)

        inverse(obj)


@raises(InvalidArgumentValue)
def testNonInvertibleObject():
    """
        Test inverse for non invertible object.
    """
    data = [[1, 1], [1, 1]]

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        obj = createData(dataType, data)

        inverse(obj)


#################
# pseudoInverse #
#################


def testPseudoInverseObject():
    """
        Test success of pseudoInverse for valid objects cases.
        This includes, square objects, non-square objects and
        singular objects.
    """
    def _testPseudoInverse_createObjects(dataType):

        objs_list = []

        data = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
        pnames = ['p1', 'p2', 'p3']
        fnames = ['f1', 'f2', 'f3']
        obj = createData(dataType, data, pointNames=pnames,
                         featureNames=fnames)
        objs_list.append(obj)

        data = [[1, 2, 3], [3, 4, 5]]
        pnames = ['p1', 'p2']
        fnames = ['f1', 'f2', 'f3']
        obj = createData(dataType, data, pointNames=pnames,
                         featureNames=fnames)
        objs_list.append(obj)

        data = [[1, 2], [3, 4], [5, 6]]
        pnames = ['p1', 'p2', 'p3']
        fnames = ['f1', 'f2']
        obj = createData(dataType, data, pointNames=pnames,
                         featureNames=fnames)
        objs_list.append(obj)

        data = [[1, 1], [1, 1]]
        obj = createData(dataType, data)
        objs_list.append(obj)

        data = [[1, 2]]
        obj = createData(dataType, data)
        objs_list.append(obj)

        return objs_list

    def _pseudoInverseTest_implementation(obj, method):
        orig_obj = obj.copy()
        identity = UML.identity(obj.getTypeString(),
                                min(len(obj.points), len(obj.features)))
        obj_pinv = pseudoInverse(obj, method=method)

        if len(obj.points) <= len(obj.features):
            identity_from_pinv = obj * obj_pinv
        else:
            identity_from_pinv = obj_pinv * obj
        assert obj_pinv.points.getNames() == obj.features.getNames()
        assert obj_pinv.features.getNames() == obj.points.getNames()
        assert identity_from_pinv.isApproximatelyEqual(identity)
        assert orig_obj == obj

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        for method in ['least-squares', 'svd']:
            obj_list = _testPseudoInverse_createObjects(dataType)
            for obj in obj_list:
                _pseudoInverseTest_implementation(obj, method)


def testPseudoInverseEmptyObject():
    """
        Test pseudoInverse for an empty object.
    """
    data = []

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        obj = createData(dataType, data)
        obj_inv = pseudoInverse(obj)
        assert obj_inv == obj

#########
# solve #
#########


def testSolve_succes():
    """
        Test success for solve
    """

    a = numpy.array([[1, 20], [-30, 4]])
    b_s = [numpy.array([-30, 4]), numpy.array([[-30], [4]])]

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        for dataType_b in ['Matrix', 'Sparse', 'DataFrame', 'List']:
            for b_np in b_s:
                A = createData(dataType, a, featureNames=['f1', 'f2'])
                b = createData(dataType_b, b_np)
                sol = solve(A, b)
                A_inv = inverse(A)
                if len(b.features) > 1:
                    b.transpose()
                reference = (A_inv * b)
                reference.transpose()
                reference.points.setNames(['b'])
                assert sol.isApproximatelyEqual(reference)
                assert A.features.getNames() == sol.features.getNames()
                assert sol.points.getNames() == ['b']
                assert sol.getTypeString() == A.getTypeString()


#######################
# leastSquareSolution #
#######################

def testLeastSquareSolution_exact():
    """
        Test success for leastSquareSolution exact case.
    """
    a = numpy.array([[1, 20], [-30, 4]])
    b_s = [numpy.array([-30, 4]), numpy.array([[-30], [4]])]

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        for dataType_b in ['Matrix', 'Sparse', 'DataFrame', 'List']:
            for b_np in b_s:
                A = createData(dataType, a, featureNames=['f1', 'f2'])
                b = createData(dataType_b, b_np)
                sol = leastSquaresSolution(A, b)
                A_inv = inverse(A)
                if b.features > 1:
                    b.transpose()
                reference = (A_inv * b)
                reference.transpose()
                reference.setPointNames(['b'])
                assert sol.isApproximatelyEqual(reference)
                assert A.getFeatureNames() == sol.getFeatureNames()
                assert sol.getPointNames() == ['b']
                assert sol.getTypeString() == A.getTypeString()


def testLeastSquareSolution_overdetermined():
    """
        Test success for leastSquareSolution overdetermined system.
    """
    a = numpy.array([[1, 2], [4, 5], [3, 4]])
    b_s = [numpy.array([1, 2, 3]), numpy.array([[1], [2], [3]])]
    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        for dataType_b in ['Matrix', 'Sparse', 'DataFrame', 'List']:
            for b_np in b_s:
                A_orig = createData(dataType, a, featureNames=['f1', 'f2'])
                A = createData(dataType, a, featureNames=['f1', 'f2'])
                b_orig = createData(dataType_b, b_np)
                b = createData(dataType_b, b_np)
                sol = leastSquaresSolution(A, b)

                assert A == A_orig
                assert b == b_orig

                A_pinv = pseudoInverse(A)
                if b.features > 1:
                    b.transpose()
                reference = (A_pinv * b)
                reference.transpose()
                reference.setPointNames(['b'])
                assert sol.isApproximatelyEqual(reference)
                assert A.getFeatureNames() == sol.getFeatureNames()
                assert sol.getPointNames() == ['b']
                assert sol.getTypeString() == A.getTypeString()


def testLeastSquareSolution_underdetermined():
    """
        Test success for leastSquareSolution underdetermined system.
    """
    a = numpy.array([[1, 2, 3], [4, 5, 6]])
    b_s = [numpy.array([1, 2]), numpy.array([[1], [2]])]
    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        for dataType_b in ['Matrix', 'Sparse', 'DataFrame', 'List']:
            for b_np in b_s:
                A_orig = createData(
                    dataType, a, featureNames=['f1', 'f2', 'f3'])
                A = createData(dataType, a, featureNames=['f1', 'f2', 'f3'])
                b_orig = createData(dataType_b, b_np)
                b = createData(dataType_b, b_np)
                sol = leastSquaresSolution(A, b)

                assert A == A_orig
                assert b == b_orig

                A_pinv = pseudoInverse(A)
                if b.features > 1:
                    b.transpose()
                reference = (A_pinv * b)
                reference.transpose()
                reference.setPointNames(['b'])
                assert sol.isApproximatelyEqual(reference)
                assert A.getFeatureNames() == sol.getFeatureNames()
                assert sol.getPointNames() == ['b']
                assert sol.getTypeString() == A.getTypeString()
