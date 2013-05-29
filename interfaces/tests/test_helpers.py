"""
Functions that could be useful accross multple interface test suites

"""

import numpy

from ..interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
#from ..interface_helpers import valueFromOneVOneData

def test_OvOTournament():
	""" Test calculateSingleLabelScoresFromOneVsOneScores() on simple handmade input """ 

	scores = [0.5, 1.2, -0.3, 0.6, 0.7, 0.2]

	ret = calculateSingleLabelScoresFromOneVsOneScores(scores, 4)
	desired = [2.0/3.0, 2.0/3.0, 1.0/3.0, 1.0/3.0]

	assert numpy.allclose(ret, desired)

