from __future__ import absolute_import
import UML

from nose.tools import *
import numpy.testing

from UML.data import Matrix


def testRidgeRegressionShapes():
    """ Test ridge regression by checking the shapes of the inputs and outputs """

    data = [[0, 0, 0], [4, 3, 1], [12, 15, -3], ]
    trainObj = UML.createData('Matrix', data)

    data2 = [[5.5, 5], [20, -3]]
    testObj = UML.createData('Matrix', data2)

    name = 'Custom.RidgeRegression'
    ret = UML.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj, arguments={'lamb': 0})

    assert ret.pts == 2
    assert ret.fts == 1
    numpy.testing.assert_approx_equal(ret[0, 0], 10.5, significant=3)
    numpy.testing.assert_approx_equal(ret[1, 0], 18, significant=2)

