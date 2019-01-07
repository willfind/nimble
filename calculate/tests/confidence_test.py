from __future__ import absolute_import
import numpy
from nose.tools import *

import UML
from UML.calculate import confidenceIntervalHelper


######################
# confidenceInterval #
######################

def testSimpleConfidenceInverval():
    """Test of confidenceInterval using example from wikipedia """
    predRaw = [252.7, 247.7] * 12
    predRaw.append(250.2)
    pred = UML.createData("Matrix", predRaw)
    pred.transpose()

    assert len(pred.points) == 25
    assert len(pred.features) == 1
    mean = pred.features.statistics('mean')[0, 0]
    numpy.testing.assert_approx_equal(mean, 250.2)
    std = pred.features.statistics('samplestandarddeviation')[0, 0]
    numpy.testing.assert_approx_equal(std, 2.5)

    (low, high) = confidenceIntervalHelper(pred, None, 0.95)

    numpy.testing.assert_approx_equal(low, 249.22)
    numpy.testing.assert_approx_equal(high, 251.18)
