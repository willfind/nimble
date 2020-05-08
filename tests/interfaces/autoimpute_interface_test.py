"""
Tests for autoimpute interface.
"""

import numpy
from nose.plugins.attrib import attr
from nose.tools import raises

import nimble
from nimble import match
from nimble import fill
from nimble.exceptions import InvalidArgumentValue
from nimble.calculate import rootMeanSquareError, fractionCorrect

from .skipTestDecorator import SkipMissing

autoimputeSkipDec = SkipMissing('autoimpute')

def getDataWithMissing(retType, assignNames=True):
    mu = numpy.array([5.0, 0.0])
    r = numpy.array([
            [  3.40, -2.75],
            [ -2.75,  5.50],])

    # Generate the random samples.
    num = 100
    d = numpy.random.multivariate_normal(mu, r, size=num)
    x = d[:, 0]
    y = d[:, 1]
    # insert missing values
    rm1 = numpy.random.random_sample(num) > 0.85
    x[rm1] = numpy.nan

    data = nimble.createData(retType, {"y": y, "x": x})
    if not assignNames:
        data.points.setNames(None)
        data.features.setNames(None)
    # remove points with only missing values
    data.points.delete(match.allMissing)

    return data

@autoimputeSkipDec
def backend_imputation(learnerName, **kwargs):
    for t in nimble.data.available:
        data = getDataWithMissing(t)
        orig = data.copy()
        matches = data.matchingElements(match.missing)
        nimble.fillMatching(learnerName, matches, data, **kwargs)

        for dFt, mFt, oFt in zip(data.features, matches.features, orig.features):
            for i in range(len(dFt)):
                if mFt[i]:
                    assert dFt[i] != oFt[i]
                else:
                    assert dFt[i] == oFt[i]

def test_autoimpute_SingleImputer():
    backend_imputation('autoimpute.SingleImputer', strategy='least squares')

@raises(InvalidArgumentValue)
def test_autoimpute_SingleImputer_exception_noStrategy():
    # this would work by default, testing that we override to require strategy argument
    backend_imputation('autoimpute.SingleImputer')

def test_autoimpute_MultipleImputer():
    backend_imputation('autoimpute.MultipleImputer', strategy={'x': 'mean', 'y': 'random'})

@raises(InvalidArgumentValue)
def test_autoimpute_MultipleImputer_exception_noStrategy():
    # this would work by default, testing that we override to require strategy argument
    backend_imputation('autoimpute.MultipleImputer')

@autoimputeSkipDec
def test_autoimpute_MiLinearRegression():
    for t in nimble.data.available:
        data = getDataWithMissing(t)
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
        # test data cannot have missing values
        testX.features.fillMatching(fill.mean, match.missing)

        rmse = nimble.trainAndTest('autoimpute.MiLinearRegression', trainX,
                                   trainY, testX, testY, rootMeanSquareError,
                                   mi_kwgs={'n': 1, 'strategy': {'x': 'mean'}})

        nimble.fillMatching('autoimpute.SingleImputer', match.missing, trainX,
                            strategy='mean')

        exp = nimble.trainAndTest('skl.LinearRegression', trainX, trainY,
                                  testX, testY, rootMeanSquareError)

        numpy.testing.assert_almost_equal(rmse, exp)

@autoimputeSkipDec
def test_autoimpute_MiLinearRegression_noNames():
    for t in nimble.data.available:
        data = getDataWithMissing(t, False)
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels=0)
        testX.features.fillMatching(fill.mean, match.missing)

        rmse = nimble.trainAndTest('autoimpute.MiLinearRegression', trainX,
                                   trainY, testX, testY, rootMeanSquareError,
                                   mi_kwgs={'n': 1, 'strategy':'mode'})

        nimble.fillMatching('autoimpute.SingleImputer', match.missing, trainX,
                            strategy='mode')

        exp = nimble.trainAndTest('skl.LinearRegression', trainX, trainY,
                                  testX, testY, rootMeanSquareError)

        numpy.testing.assert_almost_equal(rmse, exp)

@autoimputeSkipDec
@raises(InvalidArgumentValue)
def test_autoimpute_MiLinearRegression_exception_noStrategy():
    data = getDataWithMissing('Matrix')
    trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
    testX.features.fillMatching(fill.mean, match.missing)

    nimble.trainAndTest('autoimpute.MiLinearRegression', trainX, trainY,
                        testX, testY, rootMeanSquareError, mi_kwgs={'n': 1})

@autoimputeSkipDec
def test_autoimpute_MiLogisticRegression():
    for t in nimble.data.available:
        data = getDataWithMissing(t)
        # make y binary
        data.features.transform(lambda ft: numpy.random.choice(2, len(ft.points)),
                                features='y')
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
        # test data cannot have missing values
        testX.features.fillMatching(fill.mean, match.missing)
        fc = nimble.trainAndTest('autoimpute.MiLogisticRegression', trainX,
                                 trainY, testX, testY, fractionCorrect,
                                 model_lib='sklearn',
                                 mi_kwgs={'n': 1, 'strategy': {'x': 'mean'},
                                          'seed': 0})

        nimble.fillMatching('autoimpute.SingleImputer', match.missing, trainX,
                            strategy='mean')

        exp = nimble.trainAndTest('skl.LogisticRegression', trainX, trainY,
                                  testX, testY, fractionCorrect, random_state=0)

        numpy.testing.assert_almost_equal(fc, exp)

@autoimputeSkipDec
def test_autoimpute_MiLogisticRegression_directMultipleImputer():
    for t in nimble.data.available:
        data = getDataWithMissing(t)
        # make y binary
        data.features.transform(lambda ft: numpy.random.choice(2, len(ft.points)),
                                features='y')
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
        # test data cannot have missing values
        testX.features.fillMatching(fill.mean, match.missing)
        fc = nimble.trainAndTest('autoimpute.MiLogisticRegression', trainX,
                                 trainY, testX, testY, fractionCorrect,
                                 model_lib='sklearn',
                                 mi=nimble.Init('MultipleImputer', n=1,
                                            strategy='interpolate'))

        nimble.fillMatching('autoimpute.MultipleImputer', match.missing, trainX,
                            n=1, strategy='interpolate')

        exp = nimble.trainAndTest('skl.LogisticRegression', trainX, trainY,
                                  testX, testY, fractionCorrect)

        numpy.testing.assert_almost_equal(fc, exp)

@autoimputeSkipDec
def test_autoimpute_MiLogisticRegression_noNames():
    for t in nimble.data.available:
        data = getDataWithMissing(t, False)
        data.features.transform(lambda ft: numpy.random.choice(2, len(ft.points)),
                                features=0)
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels=0)

        testX.features.fillMatching(fill.mean, match.missing)
        fc = nimble.trainAndTest('autoimpute.MiLogisticRegression', trainX,
                                 trainY, testX, testY, fractionCorrect,
                                 model_lib='sklearn',
                                 mi_kwgs={'n': 1, 'strategy': 'median'})

        nimble.fillMatching('autoimpute.SingleImputer', match.missing, trainX,
                            strategy='median')

        exp = nimble.trainAndTest('skl.LogisticRegression', trainX, trainY,
                                  testX, testY, fractionCorrect)

        numpy.testing.assert_almost_equal(fc, exp)

@autoimputeSkipDec
@raises(InvalidArgumentValue)
def test_autoimpute_MiLogisticRegression_exception_noStrategy():
    data = getDataWithMissing('Matrix')
    data.features.transform(lambda ft: numpy.random.choice(2, len(ft.points)),
                            features='y')
    trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
    testX.features.fillMatching(fill.mean, match.missing)
    nimble.trainAndTest('autoimpute.MiLogisticRegression', trainX, trainY,
                        testX, testY, fractionCorrect)

@autoimputeSkipDec
@raises(InvalidArgumentValue)
def test_autoimpute_MiLogisticRegression_exception_directMultipleImputerNoStrategy():
    data = getDataWithMissing('Matrix')
    data.features.transform(lambda ft: numpy.random.choice(2, len(ft.points)),
                            features='y')
    trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')

    testX.features.fillMatching(fill.mean, match.missing)
    nimble.trainAndTest('autoimpute.MiLogisticRegression', trainX, trainY,
                        testX, testY, fractionCorrect,
                        mi=nimble.Init('MultipleImputer', n=1))
