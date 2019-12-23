"""
Tests for nimble.calculate.statistics
"""

# Many of the functions in nimble.calculate.statitic are tested not directly
# in this module, but through the functions that call them: featureReport
# in nimble.logger.tests.data_set_analyzier_tests and in the data
# hierarchy in nimble.data.tests.query_backend


from __future__ import absolute_import
try:
    from unittest import mock #python >=3.3
except:
    import mock

import numpy as np
from numpy.testing import assert_array_almost_equal
from six.moves import range
from nose.tools import raises
from nose.tools import assert_almost_equal

import nimble
from nimble import createData
from nimble.calculate import standardDeviation
from nimble.calculate import quartiles
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination, PackageException
from nimble.helpers import generateRegressionData
from ..assertionHelpers import noLogEntryExpected

def testStDev():
    dataArr = np.array([[1], [1], [3], [4], [2], [6], [12], [0]])
    testRowList = createData('List', data=dataArr, featureNames=['nums'])
    stDevContainer = testRowList.features.calculate(standardDeviation)
    stDev = stDevContainer.copy(to="python list")[0][0]
    assert_almost_equal(stDev, 3.6379, 3)

@noLogEntryExpected
def testQuartilesAPI():
    raw = [5]
    obj = createData("Matrix", raw, useLog=False)
    ret = quartiles(obj.pointView(0))
    assert ret == (5, 5, 5)

    raw = [1, 1, 3, 3]
    ret = quartiles(raw)
    assert ret == (1, 2, 3)

    raw = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    obj = createData("List", raw, useLog=False)
    ret = quartiles(obj)
    assert ret == (2, 4, 6)

#the following tests will test both None/NaN ignoring and calculation correctness
testDataTypes = nimble.data.available

@noLogEntryExpected
def testProportionMissing():
    raw = [[1, 2, np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.proportionMissing
    for dataType in testDataTypes:
        objl = createData(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = createData(dataType, [1. / 3, 0, 1. / 3], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = createData(dataType, [[1. / 3], [1. / 3], [0]],
                                  useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testProportionZero():
    raw = [[1, 2, np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.proportionZero
    for dataType in testDataTypes:
        objl = createData(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = createData(dataType, [0, 1. / 3, 0], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = createData(dataType, [[0], [0], [1. / 3]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testMinimum():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.minimum
    for dataType in testDataTypes:
        objl = createData(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = createData(dataType, [1, None, 6], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = createData(dataType, [[None], [5], [0]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testMaximum():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.maximum
    for dataType in testDataTypes:
        objl = createData(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = createData(dataType, [7, None, 9], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = createData(dataType, [[None], [6], [9]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testMean():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.mean
    for dataType in testDataTypes:
        objl = createData(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = createData(dataType, [4, None, 7.5], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = createData(dataType, [[None], [5.5], [16. / 3]],
                                  useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testMedian():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.median
    for dataType in testDataTypes:
        objl = createData(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = createData(dataType, [4, None, 7.5], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = createData(dataType, [[None], [5.5], [7]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testStandardDeviation():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = nimble.calculate.statistic.standardDeviation
    for dataType in testDataTypes:
        objl = createData(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = createData(dataType, [3, None, 1.5], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = createData(dataType, [[None], [0.5], [3.8586123009300755]],
                                  useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)

@noLogEntryExpected
def testUniqueCount():
    raw = [[1, 'a', np.nan], [5, None, 6], [7.0, 0, 9]]
    func = nimble.calculate.statistic.uniqueCount
    for dataType in ['List', 'DataFrame']:
        objl = createData(dataType, raw, useLog=False)

        retlf = objl.features.calculate(func, useLog=False)
        retlfCorrect = createData(dataType, [3, 2, 2], useLog=False)
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func, useLog=False)
        retlpCorrect = createData(dataType, [[2], [2], [3]], useLog=False)
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testFeatureType():
    raw = [[1, 'a', np.nan], [5, None, 6], [7.0, 0, 9]]
    func = nimble.calculate.statistic.featureType
    for dataType in ['List', 'DataFrame']:
        objl = createData(dataType, raw)

        retlf = objl.features.calculate(func)
        retlfCorrect = createData(dataType, ['Mixed', 'Mixed', 'int'])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func)
        retlpCorrect = createData(dataType, [['Mixed'], ['int'], ['Mixed']])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testQuartiles():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9], [2, 2, 3], [10, 10, 10]]
    func = nimble.calculate.statistic.quartiles
    for dataType in testDataTypes:
        objl = createData(dataType, raw)

        retlf = objl.features.calculate(func)
        retlfCorrect = createData(dataType, [[1.750, None, 5.250], [4.500, None, 7.500], [7.750, None, 9.250]])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.points.calculate(func)
        retlpCorrect = createData(dataType, [[None, None, None], [5.250, 5.500, 5.750], [3.500, 7.000, 8.000],
                                             [2.000, 2.000, 2.500], [10.000, 10.000, 10.000]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testIsMissing():
    raw = [1, 2.0, 3, np.nan, None, 'a']
    func = nimble.calculate.statistic._isMissing
    ret = [func(i) for i in raw]
    retCorrect = [False, False, False, True, True, False]
    assert all([ret[i] == retCorrect[i] for i in range(len(raw))])


def testIsNumericalFeatureGuesser():
    func = nimble.calculate.statistic._isNumericalFeatureGuesser
    raw = [1, 2.0, 3, np.nan, None]
    assert func(raw)
    raw = [1, 2.0, 3, np.nan, None, 'a']
    assert ~func(raw)
    raw = [1, 2.0, 3, np.nan, None, np.complex(1, 1)]
    assert ~func(raw)


def testIsNumericalPoint():
    func = nimble.calculate.statistic._isNumericalPoint
    assert func(1) and func(2.0) and func(3)
    assert ~func(np.nan)
    assert ~func(None)
    assert ~func('a')


#############
# residuals #
#############
@raises(PackageException)
@mock.patch('nimble.calculate.statistic.scipy.nimbleAccessible', new=lambda: False)
def test_residuals_exception_sciPyNotInstalled():
    pred = nimble.createData("Matrix", [[2],[3],[4]])
    control = nimble.createData("Matrix", [[2],[3],[4]])
    nimble.calculate.residuals(pred, control)

# L not nimble object
@raises(InvalidArgumentType)
def test_residuals_exception_toPredictNotNimble():
    pred = [[1],[2],[3]]
    control = nimble.createData("Matrix", [[2],[3],[4]])
    nimble.calculate.residuals(pred, control)

# R not nimble object
@raises(InvalidArgumentType)
def test_residuals_exception_controlVarsNotNimble():
    pred = nimble.createData("Matrix", [[2],[3],[4]])
    control = [[1],[2],[3]]
    nimble.calculate.residuals(pred, control)

# diff number of points
@raises(InvalidArgumentValueCombination)
def test_residauls_exception_differentNumberOfPoints():
    pred = nimble.createData("Matrix", [[2],[3],[4]])
    control = nimble.createData("Matrix", [[2],[3],[4],[5]])
    nimble.calculate.residuals(pred, control)

# zero points or zero features
def test_residuals_exception_zeroAxisOnParam():
    predOrig = nimble.createData("Matrix", [[2],[3],[4]])
    controlOrig = nimble.createData("Matrix", [[2,2],[3,3],[4,4]])

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
        nimble.helpers.findBestInterface("scikitlearn")
    except InvalidArgumentValue:
        return

    # with handmade data
    pred = nimble.createData("Matrix", [[0],[2],[4]], useLog=False)
    control = nimble.createData("Matrix", [[1],[2],[3]], useLog=False)
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
