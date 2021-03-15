
import numpy

import nimble
from nimble.calculate import meanNormalize, meanStandardDeviationNormalize
from nimble.calculate import range0to1Normalize
from nimble.calculate import percentileNormalize
from nimble.calculate.normalize import rangeNormalize
from tests.helpers import noLogEntryExpected
from tests.helpers import getDataConstructors

@noLogEntryExpected
def assertExpected(func, raw1, raw2, exp1, exp2):
    for shape in [(1, -1), (-1, 1)]:
        raw1 = numpy.array(raw1).reshape(shape)
        raw2 = numpy.array(raw2).reshape(shape)
        exp1 = numpy.array(exp1).reshape(shape)
        exp2 = numpy.array(exp2).reshape(shape)
        for con1 in getDataConstructors():
            for con2 in getDataConstructors():
                data1 = con1(raw1, useLog=False)
                expData1 = con1(exp1, useLog=False)
                data2 = con2(raw2, useLog=False)
                expData2 = con2(exp2, useLog=False)

                out1 = func(data1)
                assert out1 == expData1

                out2 = func(data1, data2)
                assert isinstance(out2, tuple)
                assert out2[0] == expData1
                assert out2[1] == expData2

def test_meanNormalize():
    data1 = [3, -1, 1, -1, 3]
    data2 = [-2, 0, 1, 6]

    exp1 = [2, -2, 0, -2, 2]
    exp2 = [-3, -1, 0, 5]

    assertExpected(meanNormalize, data1, data2, exp1, exp2)

def test_meanNormalize_withNan():
    data1 = [3, -1, numpy.nan, 3, -1]
    data2 = [-2, 0, numpy.nan, 6]

    exp1 = [2, -2, numpy.nan, 2, -2]
    exp2 = [-3, -1, numpy.nan, 5]

    assertExpected(meanNormalize, data1, data2, exp1, exp2)

def test_meanStandardDeviationNormalize():
    data1 = [0, 1, -3, 5, 4, 5]
    data2 = [-2, 0, 1, 2, 6]
    mean = numpy.mean(data1)
    std = numpy.std(data1)

    def zScore(v):
        return (v - mean) / std

    exp1 = list(map(zScore, data1))
    exp2 = list(map(zScore, data2))

    assertExpected(meanStandardDeviationNormalize, data1, data2, exp1, exp2)

def test_meanStandardDeviationNormalize_withNan():
    data1 = [-3, 0, 1, numpy.nan, 5, 5]
    data2 = [0, numpy.nan, 2, -2, 6]
    mean = numpy.nanmean(data1)
    std = numpy.nanstd(data1)

    def zScoreNan(v):
        if numpy.isnan(v):
            return v
        return (v - mean) / std

    exp1 = list(map(zScoreNan, data1))
    exp2 = list(map(zScoreNan, data2))

    assertExpected(meanStandardDeviationNormalize, data1, data2, exp1, exp2)

def test_meanStandardDeviationNormalize_allEqual():
    data1 = [7, 7, 7, 7, 7]
    data2 = [7, 7, 6, 9]
    mean = numpy.mean(data1)
    std = numpy.std(data1)

    def zScoreStdZero(v):
        assert std == 0
        return v - mean

    exp1 = list(map(zScoreStdZero, data1))
    exp2 = list(map(zScoreStdZero, data2))

    assertExpected(meanStandardDeviationNormalize, data1, data2, exp1, exp2)

def test_range0to1Normalize():
    data1 = [-4, 3, -1, 1, 6]
    data2 = [1, -5, -3, 7]

    exp1 = [0, 0.7, 0.3, 0.5, 1]
    exp2 = [0.5, -0.1, 0.1, 1.1]

    assertExpected(range0to1Normalize, data1, data2, exp1, exp2)

def test_range0to1Normalize_withNan():
    data1 = [-4, -1, 1, 6, numpy.nan]
    data2 = [-5, -3, numpy.nan, 7]

    exp1 = [0, 0.3, 0.5, 1, numpy.nan]
    exp2 = [-0.1, 0.1, numpy.nan, 1.1]

    assertExpected(range0to1Normalize, data1, data2, exp1, exp2)

def test_range0to1Normalize_allEqual():
    data1 = [-1, -1, -1, -1, -1]
    data2 = [3, -1, -2, -1]

    exp1 = [0, 0, 0, 0, 0]
    exp2 = [4, 0, -1, 0]

    assertExpected(range0to1Normalize, data1, data2, exp1, exp2)

def test_rangeNormalize():
    data1 = [-4, 3, -1, 2, 6, 1]
    data2 = [7, -5, -3, 1, 2]

    exp1 = [1, 3.8, 2.2, 3.4, 5, 3]
    exp2 = [5.4, 0.6, 1.4, 3, 3.4]

    def range1To5(values1, values2=None):
        return rangeNormalize(values1, values2, start=1, end=5)

    assertExpected(range1To5, data1, data2, exp1, exp2)

def test_rangeNormalize_withNan():
    data1 = [-4, 3, -1, numpy.nan, 6, 1]
    data2 = [7, -5, numpy.nan, 1, 2]

    exp1 = [1, 1.7, 1.3, numpy.nan, 2, 1.5]
    exp2 = [2.1, 0.9, numpy.nan, 1.5, 1.6]

    def range1To2(values1, values2=None):
        return rangeNormalize(values1, values2, start=1, end=2)

    assertExpected(range1To2, data1, data2, exp1, exp2)

def test_rangeNormalize_allEqual():
    data1 = [-1, -1, -1, -1, -1]
    data2 = [3, -1, -2, -1]

    exp1 = [1, 1, 1, 1, 1]
    exp2 = [17, 1, -3, 1]

    def range1To5(values1, values2=None):
        return rangeNormalize(values1, values2, start=1, end=5)

    assertExpected(range1To5, data1, data2, exp1, exp2)

def test_percentileNormalize():
    data1 = [-4, 3, -1, 6, 1, 1]
    data2 = [7, -5, -3, 1, 0]

    # both 1's should be same percentile
    exp1 = [0.0, 0.8, 0.2, 1.0, 0.5, 0.5]
    # -3: 1/3 of the way to covering distance between -4 (0) and -1 (0.2)
    expNeg3Percentile = 0 + (1/3) * (0.2 - 0)
    # 0: 1/2 of the way to covering distance between -1 (0.2) and 1 (0.5)
    expZeroPercentile = 0.2 + (1/2) * (0.5 - 0.2)
    exp2 = [1, 0, expNeg3Percentile, 0.5, expZeroPercentile]

    assertExpected(percentileNormalize, data1, data2, exp1, exp2)

def test_percentileNormalize_withNan():
    data1 = [-4, 3, -1, numpy.nan, 6, 1, 1]
    data2 = [7, -5, -3, 1, 0, numpy.nan]

    # both 1's should be same percentile
    exp1 = [0.0, 0.8, 0.2, numpy.nan, 1.0, 0.5, 0.5]
    # -3 is 1/3 of the way to covering distance between -4 (0) and -1 (0.2)
    expNeg3Percentile = 0 + (1/3) * (0.2 - 0)
    # 0 is 1/2 of the way to covering distance between -1 (0.2) and 1 (0.5)
    expZeroPercentile = 0.2 + (1/2) * (0.5 - 0.2)
    exp2 = [1, 0, expNeg3Percentile, 0.5, expZeroPercentile, numpy.nan]
    assertExpected(percentileNormalize, data1, data2, exp1, exp2)

def test_percentileNormalize_allEqual():
    data1 = [3, 3, 3, 3, 3, 3, 3]
    data2 = [3, 2, 1, 4, 3, 9, 3]

    exp1 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    exp2 = [0.5, 0, 0, 1, 0.5, 1, 0.5]

    assertExpected(percentileNormalize, data1, data2, exp1, exp2)
