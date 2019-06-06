from __future__ import absolute_import

import numpy

import nimble
from nimble import match
from ..assertionHelpers import noLogEntryExpected

@noLogEntryExpected
def backend_match_value(toMatch, true, false):
    """backend for match functions that accept a value"""
    for t in true:
        assert toMatch(t)
    for f in false:
        assert not toMatch(f)

misingValues = [None, float('nan'), numpy.nan]
stringValues = ['a', str(1)]
zeroValues = [0, float(0), numpy.int(0), numpy.float(0)]
positiveValues = [3, float(3), numpy.int(3), numpy.float(3)]
negativeValues = [-3, float(-3), numpy.int(-3), numpy.float(-3)]
numericValues = positiveValues + negativeValues + zeroValues + [numpy.nan]

def test_match_missing():
    true = misingValues
    false = numericValues[:-1] + stringValues
    backend_match_value(match.missing, true, false)

def test_match_numeric():
    true = numericValues
    false = stringValues + [None]
    backend_match_value(match.numeric, true, false)

def test_match_nonNumeric():
    true = stringValues + [None]
    false = numericValues
    backend_match_value(match.nonNumeric, true, false)

def test_match_zero():
    true = zeroValues
    false = positiveValues + negativeValues + stringValues + misingValues
    backend_match_value(match.zero, true, false)

def test_match_nonZero():
    true = positiveValues + negativeValues + stringValues + misingValues
    false = zeroValues
    backend_match_value(match.nonZero, true, false)

def test_match_positive():
    true = positiveValues
    false = negativeValues + zeroValues + stringValues + misingValues
    backend_match_value(match.positive, true, false)

def test_match_negative():
    true = negativeValues
    false = positiveValues + zeroValues + stringValues + misingValues
    backend_match_value(match.negative, true, false)

@noLogEntryExpected
def backend_match_anyAll(anyOrAll, func, data):
    """backend for match functions accepting 1D and 2D data and testing for any or all"""
    data = numpy.array(data, dtype=numpy.object_)
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        # test whole matrix
        if anyOrAll == 'any':
            assert func(toTest)
        else:
            allMatching = toTest[:,2]
            allMatching.features.add(allMatching, useLog=False)
            assert func(allMatching)
        # test by feature
        for i, feature in enumerate(toTest.features):
            # index 0 never contains any matching values
            if i == 0:
                assert not func(feature)
            # index 1 contains matching value, but not all matching
            elif i == 1 and anyOrAll == 'any':
                assert func(feature)
            elif i == 1:
                assert not func(feature)
            # index 2 contains all matching values
            else:
                assert func(feature)
        # test by point
        toTest = nimble.createData(t, data.T, useLog=False)
        for i, point in enumerate(toTest.points):
            # index 0 never contains any matching values
            if i == 0:
                assert not func(point)
            # index 1 contains matching value, but not all matching
            elif i == 1 and anyOrAll == 'any':
                assert func(point)
            elif i == 1:
                assert not func(point)
            # index 2 contains all matching values
            else:
                assert func(point)


def test_match_anyMissing():
    fill = numpy.nan
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll('any', match.anyMissing, data)

def test_match_allMissing():
    fill = numpy.nan
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll('all', match.allMissing, data)

def test_match_anyNumeric():
    fill = 3
    data = [['a','b',fill], ['c','d',fill], ['e',fill,fill]]
    backend_match_anyAll('any', match.anyNumeric, data)

def test_match_allNumeric():
    fill = -3.0
    data = [['a','b',fill], ['c','d',fill], ['e',fill,fill]]
    backend_match_anyAll('all', match.allNumeric, data)

def test_match_anyNonNumeric():
    fill = 'n/a'
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll('any', match.anyNonNumeric, data)

def test_match_allNonNumeric():
    fill = 'missing'
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll('all', match.allNonNumeric, data)

def test_match_anyZero():
    fill = 0
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll('any', match.anyZero, data)

def test_match_allZero():
    fill = 0.0
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll('all', match.allZero, data)

def test_match_anyNonZero():
    fill = 1
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('any', match.anyNonZero, data)

def test_match_allNonZero():
    fill = 'a'
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('all', match.allNonZero, data)

def test_match_anyPositive():
    fill = 1
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('any', match.anyPositive, data)

def test_match_allPositive():
    fill = 2.4
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('all', match.allPositive, data)

def test_match_anyNegative():
    fill = -1
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('any', match.anyNegative, data)

def test_match_allNegative():
    fill = -2.4
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('all', match.allNegative, data)

def test_match_anyValues_int():
    fill = 3
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('any', match.anyValues(fill), data)

def test_match_allValues_int():
    fill = 3
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('all', match.allValues(fill), data)

def test_match_anyValues_str():
    fill = 'a'
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('any', match.anyValues(fill), data)

def test_match_allValues_str():
    fill = 'a'
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll('all', match.allValues(fill), data)

def test_match_anyValues_list():
    data = [[0,0,1], [0,0,2], [0,'a','a']]
    backend_match_anyAll('any', match.anyValues([1, 2, 'a']), data)

def test_match_allValues_list():
    data = [[0,0,1], [0,0,2], [0,'a','a']]
    backend_match_anyAll('all', match.allValues([1, 2, 'a']), data)

def test_match_anyValues_func():
    data = [[0,0,1], [0,0,2], [0,3,3]]
    backend_match_anyAll('any', match.anyValues(lambda x: x > 0), data)

def test_match_allValues_func():
    data = [[0,0,1], [0,0,2], [0,3,3]]
    backend_match_anyAll('all', match.allValues(lambda x: x > 0), data)
