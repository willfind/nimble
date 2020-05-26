import nimble
from nimble.learners import MultiOutputRidgeRegression

# test for failure to import?


def test_MultiOutputWrapper_simple():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2]]
    trainX = nimble.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222]]
    trainY = nimble.createData('Matrix', data)

    trainY0 = trainY.features.copy(0)
    trainY1 = trainY.features.copy(1)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = nimble.createData('Matrix', data)

    wrappedName = 'nimble.RidgeRegression'
    ret0 = nimble.trainAndApply(wrappedName, trainX=trainX, trainY=trainY0,
                                testX=testX, lamb=1)
    ret1 = nimble.trainAndApply(wrappedName, trainX=trainX, trainY=trainY1,
                                testX=testX, lamb=1)

    for value in ['nimble.MultiOutputRidgeRegression',
                  MultiOutputRidgeRegression]:
        retMulti = nimble.trainAndApply(value, trainX=trainX, trainY=trainY,
                                        testX=testX, lamb=1)

        assert retMulti[0, 0] == ret0[0]
        assert retMulti[0, 1] == ret1[0]
        assert retMulti[1, 0] == ret0[1]
        assert retMulti[1, 1] == ret1[1]
