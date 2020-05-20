from nose.tools import *

import nimble


def testMeanConstantSimple():
    """ Test MeanConstant by checking the ouput given simple hand made inputs """

    dataX = [[0, 0, 0], [1, 10, 10], [0, -1, 4], [1, 0, 20], [0, 1, 0], [1, 2, 3]]
    trainX = nimble.createData('Matrix', dataX)

    dataY = [[0], [1], [0], [1], [0], [1]]
    trainY = nimble.createData('Matrix', dataY)

    name = 'nimble.MeanConstant'
    ret = nimble.trainAndApply(name, trainX=trainX, trainY=trainY, testX=trainX)

    assert len(ret.points) == 6
    assert len(ret.features) == 1

    assert ret[0] == .5
    assert ret[1] == .5
    assert ret[2] == .5
    assert ret[3] == .5
    assert ret[4] == .5
    assert ret[5] == .5
