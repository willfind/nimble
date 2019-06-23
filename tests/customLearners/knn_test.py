from __future__ import absolute_import
from __future__ import print_function

import nimble


def testKNNClassificationSimple():
    """ Test KNN classification by checking the ouput given simple hand made inputs """

    data = [[0, 0, 0], [1, 10, 10], [0, -1, 4], [1, 0, 20]]
    trainObj = nimble.createData('Matrix', data)

    data2 = [[2, 2], [20, 20]]
    testObj = nimble.createData('Matrix', data2)

    name = 'Custom.KNNClassifier'
    ret = nimble.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj, k=3)

    assert ret[0, 0] == 0
    assert ret[1, 0] == 1


def testKNNClassificationSimpleScores():
    """ Test KNN classification by checking the shapes of the inputs and outputs """

    data = [[0, 0, 0], [1, 10, 10], [0, -1, 4], [1, 0, 20]]
    trainObj = nimble.createData('Matrix', data)

    data2 = [[2, 2], [20, 20]]
    testObj = nimble.createData('Matrix', data2)

    name = 'Custom.KNNClassifier'
    tl = nimble.train(name, trainX=trainObj, trainY=0, k=3)

    ret = tl.getScores(testObj)

    assert ret[0, 0] == 2
    assert ret[0, 1] == 1
    assert ret[1, 0] == 1
    assert ret[1, 1] == 2


def testKNNClassificationTie():
    """ Test KNN classification uses k=1 value when scores are tied """

    data = [[0, 0, 0], [1, 10, 10], [0, -1, 4], [1, 0, 20]]
    trainObj = nimble.createData('Matrix', data)

    data2 = [[2, 2], [20, 20]]
    testObj = nimble.createData('Matrix', data2)

    name = 'Custom.KNNClassifier'
    tl = nimble.train(name, trainObj, 0, k=4)
    predK4 = tl.apply(testObj)
    scores = tl.getScores(testObj)
    predK1 = nimble.trainAndApply(name, trainObj, 0, testObj, k=1)

    # check tie occurred
    assert scores[0, 0] == scores[0, 1]
    assert scores[1, 0] == scores[1, 1]
    # check predictions are k=1
    assert predK4 == predK1


def testKNNClassificationGetScores():
    """ Test KNN classification getScores outputs match expectations """

    data = [[0, 0, 0], [0, -1, 4], [1, 10, 10], [1, 0, 20], [2, -94, -56], [2, -22, -44], [2, -32, -87]]
    trainObj = nimble.createData('Matrix', data)

    data2 = [[2, 2], [20, 20], [-99, -99]]
    testObj = nimble.createData('Matrix', data2)

    name = 'Custom.KNNClassifier'
    tl = nimble.train(name, trainX=trainObj, trainY=0, k=3)

    ret = tl.getScores(testObj)

    assert ret[0, 0] == 2
    assert ret[0, 1] == 1
    assert ret[0, 2] == 0
    assert ret[1, 0] == 1
    assert ret[1, 1] == 2
    assert ret[1, 2] == 0
    assert ret[2, 0] == 0
    assert ret[2, 1] == 0
    assert ret[2, 2] == 3
