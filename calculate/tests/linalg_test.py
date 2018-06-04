from __future__ import absolute_import
import numpy

from nose.tools import *
import UML
from UML import createData
from UML.exceptions import ArgumentException
from UML.calculate import inverse
from UML.calculate import pseudoInverse

from UML.data.tests.baseObject import DataTestObjec

###########
# inverse #
###########

def testInverseSquareObject():
    """
        Check that inverse works correctly for a square object.
    """
    data = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
    pnames = ['p1', 'p2', 'p3']
    fnames = ['f1', 'f2', 'f3']
    
    for dataType in ['Matrix', 'Sparse', 'DataFrame', 'List']:
        obj = createData(dataType, data, pointNames=pnames, featureNames=fnames)
        obj_inv = inverse(obj)
        print(obj ** obj_inv)