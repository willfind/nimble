from __future__ import absolute_import

from nose.tools import *
import numpy.testing

import nimble
from nimble.data import Matrix


def testRidgeRegressionShapes():
    """ Test ridge regression by checking the shapes of the inputs and outputs """

    data = [[0, 0, 0], [4, 3, 1], [12, 15, -3], ]
    trainObj = nimble.createData('Matrix', data)

    data2 = [[5.5, 5], [20, -3]]
    testObj = nimble.createData('Matrix', data2)

    name = 'Custom.RidgeRegression'
    ret = nimble.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj,
                               arguments={'lamb': 0})

    assert len(ret.points) == 2
    assert len(ret.features) == 1
    numpy.testing.assert_approx_equal(ret[0, 0], 10.5, significant=3)
    numpy.testing.assert_approx_equal(ret[1, 0], 18, significant=2)

