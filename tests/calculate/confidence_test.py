from __future__ import absolute_import
try:
    from unittest import mock #python >=3.3
except:
    import mock

import numpy
from nose.tools import *

import UML
from UML.calculate import confidenceIntervalHelper
from UML.exceptions import PackageException, ImproperObjectAction
from ..logHelpers import noLogEntryExpected

def getPredictions():
    predRaw = [252.7, 247.7] * 12
    predRaw.append(250.2)
    pred = UML.createData("Matrix", predRaw, useLog=False)
    pred.transpose(useLog=False)

    assert len(pred.points) == 25
    assert len(pred.features) == 1
    mean = pred.features.statistics('mean')[0, 0]
    numpy.testing.assert_approx_equal(mean, 250.2)
    std = pred.features.statistics('samplestandarddeviation')[0, 0]
    numpy.testing.assert_approx_equal(std, 2.5)

    return pred

######################
# confidenceInterval #
######################
@noLogEntryExpected
def testSimpleConfidenceInverval():
    """Test of confidenceInterval using example from wikipedia """
    pred = getPredictions()
    (low, high) = confidenceIntervalHelper(pred, None, 0.95)

    numpy.testing.assert_approx_equal(low, 249.22)
    numpy.testing.assert_approx_equal(high, 251.18)

@raises(PackageException)
@mock.patch('UML.calculate.confidence.scipy', new=None)
def testCannotImportSciPy():
    pred = getPredictions()
    (low, high) = confidenceIntervalHelper(pred, None, 0.95)

@raises(ImproperObjectAction)
def testPredictionsInvalidShape():
    pred = getPredictions()
    toAdd = UML.createData('Matrix', numpy.ones((len(pred.points), 1)))
    pred.features.add(toAdd)
    assert len(pred.features) == 2

    (low, high) = confidenceIntervalHelper(pred, None, 0.95)
