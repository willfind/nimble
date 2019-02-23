"""
Functions that could be useful accross multple interface test suites

"""

from __future__ import absolute_import
import numpy

from UML.data import Base
from six.moves import range
#from UML.interfaces.interface_helpers import valueFromOneVOneData

from .interface_helpers import calculateSingleLabelScoresFromOneVsOneScores


def test_OvOTournament():
    """ Test calculateSingleLabelScoresFromOneVsOneScores() on simple handmade input """

    scores = [0.5, 1.2, -0.3, 0.6, 0.7, 0.2]

    ret = calculateSingleLabelScoresFromOneVsOneScores(scores, 4)
    desired = [2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    assert numpy.allclose(ret, desired)


def checkLabelOrderingAndScoreAssociations(allLabels, bestScores, allScores):
    """
    Given the output of the 'bestScores' and 'allScores' scoreMode flag of trainAndApply(),
    do some checks to make sure the results match each other.

    """
    if isinstance(bestScores, Base):
        bestScores = bestScores.data
    if isinstance(allScores, Base):
        allScores = allScores.data

    assert len(bestScores) == len(allScores)
    for i in range(len(bestScores)):
        currBest = numpy.array(bestScores[i]).flatten()
        currAll = numpy.array(allScores[i]).flatten()
        #score in bestScore matches winning score's slot in allScores
        for j in range(len(allLabels)):
            if currBest[0] == allLabels[j]:
                index = j
                break
        assert currBest[1] == currAll[index]

        #score in bestScore >= every score in allScores
        for value in currAll:
            if not value <= currBest[1]:
                assert value <= currBest[1]

