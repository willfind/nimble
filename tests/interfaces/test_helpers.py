"""
Functions that could be useful accross multple interface test suites
"""

import numpy as np

import nimble
from nimble.core.data import Base
from nimble.core.interfaces._interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
from nimble.core.interfaces._interface_helpers import extractWinningPredictionLabel
from nimble.core.interfaces._interface_helpers import generateAllPairs


def test_OvOTournament():
    """ Test calculateSingleLabelScoresFromOneVsOneScores() on simple handmade input """

    scores = [0.5, 1.2, -0.3, 0.6, 0.7, 0.2]

    ret = calculateSingleLabelScoresFromOneVsOneScores(scores, 4)
    desired = [2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    assert np.allclose(ret, desired)


def checkLabelOrderingAndScoreAssociations(allLabels, bestScores, allScores):
    """
    Given the output of the 'bestScores' and 'allScores' scoreMode flag of trainAndApply(),
    do some checks to make sure the results match each other.

    """
    if isinstance(bestScores, Base):
        bestScores = bestScores._data
    if isinstance(allScores, Base):
        allScores = allScores._data

    assert len(bestScores) == len(allScores)
    for i in range(len(bestScores)):
        currBest = np.array(bestScores[i]).flatten()
        currAll = np.array(allScores[i]).flatten()
        #score in bestScore matches winning score's slot in allScores
        for j in range(len(allLabels)):
            if currBest[0] == allLabels[j]:
                index = j
                break
        assert currBest[1] == currAll[index]

        #score in bestScore >= every score in allScores
        for value in currAll:
            assert value <= currBest[1]


def testExtractWinningPredictionLabel():
    """
    Unit test for extractWinningPrediction function in runner.py
    """
    predictionData = [[1, 3, 3, 2, 3, 2], [2, 3, 3, 2, 2, 2], [1, 1, 1, 1, 1, 1], [4, 4, 4, 3, 3, 3]]
    BaseObj = nimble.data('Matrix', predictionData)
    BaseObj.transpose()
    predictions = BaseObj.features.calculate(extractWinningPredictionLabel)
    listPredictions = predictions.copy(to="python list")

    assert listPredictions[0][0] - 3 == 0.0
    assert listPredictions[0][1] - 2 == 0.0
    assert listPredictions[0][2] - 1 == 0.0
    assert (listPredictions[0][3] - 4 == 0.0) or (listPredictions[0][3] - 3 == 0.0)


def testGenerateAllPairs():
    """
    Unit test function for testGenerateAllPairs
    """
    testList1 = [1, 2, 3, 4]
    testPairs = generateAllPairs(testList1)

    assert len(testPairs) == 6
    assert ((1, 2) in testPairs) or ((2, 1) in testPairs)
    assert not (((1, 2) in testPairs) and ((2, 1) in testPairs))
    assert ((1, 3) in testPairs) or ((3, 1) in testPairs)
    assert not (((1, 3) in testPairs) and ((3, 1) in testPairs))
    assert ((1, 4) in testPairs) or ((4, 1) in testPairs)
    assert not (((1, 4) in testPairs) and ((4, 1) in testPairs))
    assert ((2, 3) in testPairs) or ((3, 2) in testPairs)
    assert not (((2, 3) in testPairs) and ((3, 2) in testPairs))
    assert ((2, 4) in testPairs) or ((4, 2) in testPairs)
    assert not (((2, 4) in testPairs) and ((4, 2) in testPairs))
    assert ((3, 4) in testPairs) or ((4, 3) in testPairs)
    assert not (((3, 4) in testPairs) and ((4, 3) in testPairs))

    testList2 = []
    testPairs2 = generateAllPairs(testList2)
    assert testPairs2 is None
