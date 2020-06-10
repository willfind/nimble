"""
Tests for nimble.calculate.statistics
"""

# Many of the functions in nimble.calculate.statitic are tested not directly
# in this module, but through the functions that call them: featureReport
# in nimble.core.logger.tests.data_set_analyzier_tests and in the data
# hierarchy in nimble.core.data.tests.query_backend

try:
    from unittest import mock #python >=3.3
except ImportError:
    import mock

import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import raises
from nose.tools import assert_almost_equal

import nimble
from nimble.calculate import standardDeviation
from nimble.calculate import quartiles
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination, PackageException
from tests.helpers import generateRegressionData
from tests.helpers import noLogEntryExpected

def testStDev():
    dataArr = np.array([[1], [1], [3], [4], [2], [6], [12], [0]])
    testRowList = nimble.data('List', source=dataArr, featureNames=['nums'])
    stDevContainer = testRowList.features.calculate(standardDeviation)
    stDev = stDevContainer[0, 0]
    assert_almost_equal(stDev, 3.8891, 3)

@noLogEntryExpected
def testQuartilesAPI():
    raw = [5]
    obj = nimble.data("Matrix", raw, useLog=False)
    ret = quartiles(obj.pointView(0))
    assert ret == (5, 5, 5)

    raw = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    obj = nimble.data("List", raw, useLog=False)
    ret = quartiles(obj)
    assert ret == (2, 4, 6)

#the following tests will test both None/NaN ignoring and calculation correctness
testDataTypes = nimble.core.data.available

@noLogEntryExpected
def testProportionMissing():
    raw = [[1, 2, np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.proportionMissing
    for dataType in testDataTypes:
        objl = nimble.data(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = nimble.data(dataType, [1. / 3, 0, 1. / 3], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = nimble.data(dataType, [[1. / 3], [1. / 3], [0]],
                                   useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testProportionZero():
    raw = [[1, 2, np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.proportionZero
    for dataType in testDataTypes:
        objl = nimble.data(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = nimble.data(dataType, [0, 1. / 3, 0], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = nimble.data(dataType, [[0], [0], [1. / 3]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testMinimum():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.minimum
    for dataType in testDataTypes:
        objl = nimble.data(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = nimble.data(dataType, [1, None, 6], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = nimble.data(dataType, [[None], [5], [0]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testMaximum():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.maximum
    for dataType in testDataTypes:
        objl = nimble.data(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = nimble.data(dataType, [7, None, 9], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = nimble.data(dataType, [[None], [6], [9]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testMean():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.mean
    for dataType in testDataTypes:
        objl = nimble.data(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = nimble.data(dataType, [4, None, 7.5], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = nimble.data(dataType, [[None], [5.5], [16. / 3]],
                                   useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testMedian():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.median
    for dataType in testDataTypes:
        objl = nimble.data(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = nimble.data(dataType, [4, None, 7.5], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = nimble.data(dataType, [[None], [5.5], [7]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testStandardDeviation():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.standardDeviation
    for dataType in testDataTypes:
        objl = nimble.data(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfData = [4.242640687119285, None, 2.1213203435596424]
        retlfCorrect = nimble.data(dataType, retlfData, useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpData = [[None], [0.7071067811865476], [4.725815626252609]]
        retlpCorrect = nimble.data(dataType, retlpData, useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testMedianAbsoluteDeviation():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9], [3, 9, 15]]
    func = nimble.calculate.statistic.medianAbsoluteDeviation
    for dataType in testDataTypes:
        objl = nimble.data(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = nimble.data(dataType, [2, None, 3], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = nimble.data(dataType, [[None], [0.5], [2], [6]],
                                   useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testUniqueCount():
    raw = [[1, 'a', np.nan], [5, None, 6], [7.0, 0, 9]]
    func = nimble.calculate.statistic.uniqueCount
    for dataType in ['List', 'DataFrame']:
        objl = nimble.data(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = nimble.data(dataType, [3, 2, 2], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = nimble.data(dataType, [[2], [2], [3]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testQuartiles():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9], [2, 2, 3], [10, 10, 10]]
    func = nimble.calculate.statistic.quartiles
    for dataType in testDataTypes:
        objl = nimble.data(dataType, raw)

        retlf = objl.features.calculate(func)
        retlfCorrect = nimble.data(dataType, [[1.750, None, 5.250], [4.500, None, 7.500], [7.750, None, 9.250]])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func)
        retlpCorrect = nimble.data(dataType, [[None, None, None], [5.250, 5.500, 5.750], [3.500, 7.000, 8.000],
                                             [2.000, 2.000, 2.500], [10.000, 10.000, 10.000]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testNonMissing():
    raw = [1, 2.0, 3, np.nan, None, 'a']
    func = nimble.calculate.statistic.nonMissing
    ret = [func(i) for i in raw]
    retCorrect = [True, True, True, False, False, True]
    assert all(act == exp for act, exp in zip(ret, retCorrect))

def testNonMissingNonZero():
    raw = [0, 1, 2.0, 3, np.nan, None, 'a', 0]
    func = nimble.calculate.statistic.nonMissingNonZero
    ret = [func(i) for i in raw]
    retCorrect = [False, True, True, True, False, False, True, False]
    assert all(act == exp for act, exp in zip(ret, retCorrect))

#############
# residuals #
#############
@raises(PackageException)
@mock.patch('nimble.calculate.statistic.scipy.nimbleAccessible', new=lambda: False)
def test_residuals_exception_sciPyNotInstalled():
    pred = nimble.data("Matrix", [[2],[3],[4]])
    control = nimble.data("Matrix", [[2],[3],[4]])
    nimble.calculate.residuals(pred, control)

# L not nimble object
@raises(InvalidArgumentType)
def test_residuals_exception_toPredictNotNimble():
    pred = [[1],[2],[3]]
    control = nimble.data("Matrix", [[2],[3],[4]])
    nimble.calculate.residuals(pred, control)

# R not nimble object
@raises(InvalidArgumentType)
def test_residuals_exception_controlVarsNotNimble():
    pred = nimble.data("Matrix", [[2],[3],[4]])
    control = [[1],[2],[3]]
    nimble.calculate.residuals(pred, control)

# diff number of points
@raises(InvalidArgumentValueCombination)
def test_residauls_exception_differentNumberOfPoints():
    pred = nimble.data("Matrix", [[2],[3],[4]])
    control = nimble.data("Matrix", [[2],[3],[4],[5]])
    nimble.calculate.residuals(pred, control)

# zero points or zero features
def test_residuals_exception_zeroAxisOnParam():
    predOrig = nimble.data("Matrix", [[2],[3],[4]])
    controlOrig = nimble.data("Matrix", [[2,2],[3,3],[4,4]])

    try:
        pred = predOrig.copy().points.extract(lambda x: False)
        control = controlOrig.copy().points.extract(lambda x: False)
        nimble.calculate.residuals(pred, control)
        assert False  # expected InvalidArgumentValue
    except InvalidArgumentValue as ae:
#        print ae
        pass

    try:
        pred = predOrig.copy().features.extract(lambda x: False)
        nimble.calculate.residuals(pred, controlOrig)
        assert False  # expected InvalidArgumentValue
    except InvalidArgumentValue as ae:
#        print ae
        pass

    try:
        control = controlOrig.copy().features.extract(lambda x: False)
        nimble.calculate.residuals(predOrig, control)
        assert False  # expected InvalidArgumentValue
    except InvalidArgumentValue as ae:
#        print ae
        pass

#compare to same func in scikitlearn
@noLogEntryExpected
def test_residuals_matches_SKL():
    try:
        nimble.core._learnHelpers.findBestInterface("scikitlearn")
    except InvalidArgumentValue:
        return

    # with handmade data
    pred = nimble.data("Matrix", [[0],[2],[4]], useLog=False)
    control = nimble.data("Matrix", [[1],[2],[3]], useLog=False)
    nimbleRet = nimble.calculate.residuals(pred, control)
    tl = nimble.train("scikitlearn.LinearRegression", control, pred, useLog=False)
    sklRet = pred - tl.apply(control, useLog=False)

    assert sklRet.isApproximatelyEqual(nimbleRet)
    assert_array_almost_equal(nimbleRet.copy(to="numpy array"), sklRet.copy(to="numpy array"), 14)

    # with generated data
    (control, pred), (ignore1,ignore2) = generateRegressionData(2, 10, 3)
    nimbleRet = nimble.calculate.residuals(pred, control)
    tl = nimble.train("scikitlearn.LinearRegression", control, pred, useLog=False)
    sklRet = pred - tl.apply(control, useLog=False)

    assert sklRet.isApproximatelyEqual(nimbleRet)
    assert_array_almost_equal(nimbleRet.copy(to="numpy array"), sklRet.copy(to="numpy array"), 15)
