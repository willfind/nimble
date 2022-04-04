import numpy as np

import nimble
from nimble.learners import RidgeRegression


def testRidgeRegressionShapes():
    """ Test ridge regression by checking the shapes of the inputs and outputs """

    data = [[0, 0, 0], [4, 3, 1], [12, 15, -3], ]
    trainObj = nimble.data(data)

    data2 = [[5.5, 5], [20, -3]]
    testObj = nimble.data(data2)

    for value in ['nimble.RidgeRegression', RidgeRegression]:
        tl = nimble.train(value, trainX=trainObj, trainY=0)
        assert tl.learnerType == 'regression'
        ret = nimble.trainAndApply(value, trainX=trainObj, trainY=0,
                                   testX=testObj, arguments={'lamb': 0})

        assert len(ret.points) == 2
        assert len(ret.features) == 1
        np.testing.assert_approx_equal(ret[0, 0], 10.5, significant=3)
        np.testing.assert_approx_equal(ret[1, 0], 18, significant=2)
