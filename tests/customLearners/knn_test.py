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


def testKNNClassification_NearestBreaksTie():
    """ Test KNN classification uses nearest value to break tie """

    data = [[0, 0, 0], [1, 10, 10], [1, 11, 11], [2, -9, -9], [2, -10, -10]]
    trainObj = nimble.createData('Matrix', data)

    data2 = [[1, 1], [-1, -1]]
    testObj = nimble.createData('Matrix', data2)

    name = 'Custom.KNNClassifier'

    tl = nimble.train(name, trainObj, 0, k=5)
    predK5 = tl.apply(testObj)
    scores = tl.getScores(testObj)

    # nearest neighbor has label 0 for each point in testObj, but when
    # k=5, 0 receives only one vote and 1 and 2 are tied at 2 votes
    assert scores[0, 0] == 1
    assert scores[1, 0] == 1
    assert scores[0, 1] == 2 and scores[0, 2] == 2
    assert scores[1, 1] == 2 and scores[1, 2] == 2
    # for first point in testObj, 1 is the nearest label of the tied votes
    assert predK5[0] == 1
    # for the second point in testObj, 2 is the nearest label of the tied votes
    assert predK5[1] == 2


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
