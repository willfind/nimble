"""
Tests for UML.calculate.statistics

"""

# Many of the functions in UML.calculate.statitic are tested not directly
# in this module, but through the functions that call them: featureReport
# in UML.logger.tests.data_set_analyzier_tests and in the data
# hierarchy in UML.data.tests.query_backend


import numpy as np
from numpy.testing import assert_array_almost_equal

from nose.tools import raises
from nose.tools import assert_almost_equal

import UML
from UML import createData
from UML.calculate import standardDeviation
from UML.calculate import quartiles
from UML.exceptions import ArgumentException
from UML.helpers import generateRegressionData


def testStDev():
    dataArr = np.array([[1], [1], [3], [4], [2], [6], [12], [0]])
    testRowList = createData('List', data=dataArr, featureNames=['nums'])
    stDevContainer = testRowList.calculateForEachFeature(standardDeviation)
    stDev = stDevContainer.copyAs(format="python list")[0][0]
    assert_almost_equal(stDev, 3.6379, 3)


def testQuartilesAPI():
    raw = [5]
    obj = createData("Matrix", raw)
    ret = quartiles(obj.pointView(0))
    assert ret == (5, 5, 5)

    raw = [1, 1, 3, 3]
    ret = quartiles(raw)
    assert ret == (1, 2, 3)

    raw = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    obj = createData("List", raw)
    ret = quartiles(obj)
    assert ret == (2, 4, 6)

#the following tests will test both None/NaN ignoring and calculation correctness
testDataTypes = ['List', 'DataFrame']#'Matrix','Sparse'


def testProportionMissing():
    raw = [[1, 2, np.nan], [None, 5, 6], [7, 0, 9]]
    func = UML.calculate.statistic.proportionMissing
    for dataType in testDataTypes:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, [1. / 3, 0, 1. / 3])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [[1. / 3], [1. / 3], [0]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testProportionZero():
    raw = [[1, 2, np.nan], [None, 5, 6], [7, 0, 9]]
    func = UML.calculate.statistic.proportionZero
    for dataType in testDataTypes:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, [0, 1. / 3, 0])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [[0], [0], [1. / 3]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testMinimum():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = UML.calculate.statistic.minimum
    for dataType in testDataTypes:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, [1, None, 6])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [[None], [5], [0]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testMaximum():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = UML.calculate.statistic.maximum
    for dataType in testDataTypes:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, [7, None, 9])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [[None], [6], [9]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testMean():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = UML.calculate.statistic.mean
    for dataType in testDataTypes:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, [4, None, 7.5])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [[None], [5.5], [16. / 3]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testMedian():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = UML.calculate.statistic.median
    for dataType in testDataTypes:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, [4, None, 7.5])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [[None], [5.5], [7]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testStandardDeviation():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9]]
    func = UML.calculate.statistic.standardDeviation
    for dataType in testDataTypes:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, [3, None, 1.5])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [[None], [0.5], [3.8586123009300755]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testUniqueCount():
    raw = [[1, 'a', np.nan], [5, None, 6], [7.0, 0, 9]]
    func = UML.calculate.statistic.uniqueCount
    for dataType in ['List', 'DataFrame']:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, [3, 2, 2])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [[2], [2], [3]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testFeatureType():
    raw = [[1, 'a', np.nan], [5, None, 6], [7.0, 0, 9]]
    func = UML.calculate.statistic.featureType
    for dataType in ['List', 'DataFrame']:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, ['Mixed', 'Mixed', 'int'])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [['Mixed'], ['int'], ['Mixed']])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testQuartiles():
    raw = [[1, 'a', np.nan], [None, 5, 6], [7, 0, 9], [2, 2, 3], [10, 10, 10]]
    func = UML.calculate.statistic.quartiles
    for dataType in testDataTypes:
        objl = createData(dataType, raw)

        retlf = objl.calculateForEachFeature(func)
        retlfCorrect = createData(dataType, [[1.750, None, 5.250], [4.500, None, 7.500], [7.750, None, 9.250]])
        assert retlf.isIdentical(retlfCorrect)
        assert retlfCorrect.isIdentical(retlf)

        retlp = objl.calculateForEachPoint(func)
        retlpCorrect = createData(dataType, [[None, None, None], [5.250, 5.500, 5.750], [3.500, 7.000, 8.000],
                                             [2.000, 2.000, 2.500], [10.000, 10.000, 10.000]])
        assert retlp.isIdentical(retlpCorrect)
        assert retlpCorrect.isIdentical(retlp)


def testIsMissing():
    raw = [1, 2.0, 3L, np.nan, None, 'a']
    func = UML.calculate.statistic._isMissing
    ret = [func(i) for i in raw]
    retCorrect = [False, False, False, True, True, False]
    assert all([ret[i] == retCorrect[i] for i in xrange(len(raw))])


def testIsNumericalFeatureGuesser():
    func = UML.calculate.statistic._isNumericalFeatureGuesser
    raw = [1, 2.0, 3L, np.nan, None]
    assert func(raw)
    raw = [1, 2.0, 3L, np.nan, None, 'a']
    assert ~func(raw)
    raw = [1, 2.0, 3L, np.nan, None, np.complex(1, 1)]
    assert ~func(raw)


def testIsNumericalPoint():
    func = UML.calculate.statistic._isNumericalPoint
    assert func(1) and func(2.0) and func(3L)
    assert ~func(np.nan)
    assert ~func(None)
    assert ~func('a')


#############
# residuals #
#############


# L not uml object
@raises(ArgumentException)
def test_residuals_exception_toPredictNotUML():
    pred = [[1],[2],[3]]
    control = UML.createData("Matrix", [[2],[3],[4]])
    UML.calculate.residuals(pred, control)

# R not uml object
@raises(ArgumentException)
def test_residuals_exception_controlVarsNotUML():
    pred = UML.createData("Matrix", [[2],[3],[4]])
    control = [[1],[2],[3]]
    UML.calculate.residuals(pred, control)

# diff number of points
@raises(ArgumentException)
def test_residauls_exception_differentNumberOfPoints():
    pred = UML.createData("Matrix", [[2],[3],[4]])
    control = UML.createData("Matrix", [[2],[3],[4],[5]])
    UML.calculate.residuals(pred, control)

# zero points or zero features
def test_residuals_exception_zeroAxisOnParam():
    predOrig = UML.createData("Matrix", [[2],[3],[4]])
    controlOrig = UML.createData("Matrix", [[2,2],[3,3],[4,4]])

    try:
        pred = predOrig.copy().extractPoints(lambda x: False)
        control = controlOrig.copy().extractPoints(lambda x: False)
        UML.calculate.residuals(pred, control)
        assert False  # expected ArgumentException
    except ArgumentException as ae:
#        print ae
        pass

    try:
        pred = predOrig.copy().extractFeatures(lambda x: False)
        UML.calculate.residuals(pred, controlOrig)
        assert False  # expected ArgumentException
    except ArgumentException as ae:
#        print ae
        pass

    try:
        control = controlOrig.copy().extractFeatures(lambda x: False)
        UML.calculate.residuals(predOrig, control)
        assert False  # expected ArgumentException
    except ArgumentException as ae:
#        print ae
        pass

#compare to same func in scikitlearn
def test_residuals_matches_SKL():
    try:
        UML.helpers.findBestInterface("scikitlearn")
    except ArgumentException:
        return

    # with handmade data
    pred = UML.createData("Matrix", [[0],[2],[4]])
    control = UML.createData("Matrix", [[1],[2],[3]])
    umlRet = UML.calculate.residuals(pred, control)
    tl = UML.train("scikitlearn.LinearRegression", control, pred)
    sklRet = pred - tl.apply(control)

    assert sklRet.isApproximatelyEqual(umlRet)
    assert_array_almost_equal(umlRet.copyAs("numpy array"), sklRet.copyAs("numpy array"), 14)

    # with generated data
    (control, pred), (ignore1,ignore2) = generateRegressionData(2, 10, 3)
    umlRet = UML.calculate.residuals(pred, control)
    tl = UML.train("scikitlearn.LinearRegression", control, pred)
    sklRet = pred - tl.apply(control)

    assert sklRet.isApproximatelyEqual(umlRet)
    assert_array_almost_equal(umlRet.copyAs("numpy array"), sklRet.copyAs("numpy array"), 15)
