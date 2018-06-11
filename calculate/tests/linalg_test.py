from __future__ import absolute_import
import numpy

from nose.tools import *
import UML
from UML import createData
from UML.exceptions import ArgumentException
from UML.calculate import inverse
from UML.calculate import pseudoInverse


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

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        identity_obj = UML.identity(
            dataType, 3, pointNames=pnames, featureNames=pnames)
        orig_obj = createData(
            dataType, data, pointNames=pnames, featureNames=fnames)
        obj = createData(dataType, data, pointNames=pnames,
                         featureNames=fnames)
        obj_inv = inverse(obj)

        assert obj_inv.getPointNames() == obj.getFeatureNames()
        assert obj_inv.getFeatureNames() == obj.getPointNames()
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


def testInverseNonSquareObject():
    """
        Test inverse for non square object.
    """
    data = [[1, 2, 3], [4, 5, 6]]

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        obj = createData(dataType, data)
        try:
            obj_inv = inverse(obj)
        except ArgumentException as e:
            pass
        else:
            raise AssertionError
        
                
def testNonInvertibleObject():
    """
        Test inverse for non invertible object.
    """
    data = [[1, 1], [1, 1]]

    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        obj = createData(dataType, data)
        try:
            obj_inv = inverse(obj)
        except ArgumentException as e:
            pass
        else:
            raise AssertionError

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

        data = [[1,2]]
        obj = createData(dataType, data)
        objs_list.append(obj)

        return objs_list

    def _pseudoInverseTest_implementation(obj, method):
        orig_obj = obj.copy()
        identity = UML.identity(obj.getTypeString(),
                                min(obj.points, obj.features))
        obj_pinv = pseudoInverse(obj, method=method)
        if obj.points <= obj.features:
            identity_from_pinv = obj * obj_pinv
        else:
            identity_from_pinv = obj_pinv * obj
        assert obj_pinv.getPointNames() == obj.getFeatureNames()
        assert obj_pinv.getFeatureNames() == obj.getPointNames()
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
        
        