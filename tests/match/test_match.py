import numpy as np

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentValue
from tests.helpers import noLogEntryExpected
from tests.helpers import getDataConstructors
from tests.helpers import raises

@noLogEntryExpected
def backend_match_value(toMatch, true, false):
    """backend for match functions that accept a value"""
    for t in true:
        assert toMatch(t)
    for f in false:
        assert not toMatch(f)

missingValues = [None, float('nan'), np.nan]
stringValues = ['a', str(1)]
zeroValues = [0, float(0), np.int_(0), np.float_(0)]
positiveValues = [3, float(3), np.int_(3), np.float_(3)]
negativeValues = [-3, float(-3), np.int_(-3), np.float_(-3)]
infinityValues = [float('inf'), -float('inf'), np.inf, -np.inf]
trueValues = [True, np.bool_(True)]
falseValues = [False, np.bool_(False)]
boolValues = trueValues + falseValues

numericValues = (positiveValues + negativeValues + zeroValues
                 + infinityValues + missingValues[1:])

sparseUnsafeMatches = [match.missing, match.anyMissing, match.allMissing, 
                       match.nonMissing, match.anyNonMissing, match.nonNumeric,]

def test_match_missing():
    true = missingValues
    false = numericValues[:-2] + stringValues + boolValues
    backend_match_value(match.missing, true, false)

def test_match_nonMissing():
    true = numericValues[:-2] + stringValues + boolValues
    false = missingValues
    backend_match_value(match.nonMissing, true, false)

def test_match_numeric():
    true = numericValues
    false = stringValues + boolValues + [None]
    backend_match_value(match.numeric, true, false)

def test_match_nonNumeric():
    true = stringValues + boolValues + [None]
    false = numericValues
    backend_match_value(match.nonNumeric, true, false)

def test_match_zero():
    true = zeroValues + falseValues
    false = positiveValues + negativeValues + stringValues + missingValues + trueValues
    backend_match_value(match.zero, true, false)

def test_match_nonZero():
    true = positiveValues + negativeValues + stringValues + missingValues + trueValues
    false = zeroValues + falseValues
    backend_match_value(match.nonZero, true, false)

def test_match_positive():
    true = positiveValues + trueValues
    false = negativeValues + zeroValues + stringValues + missingValues + falseValues
    backend_match_value(match.positive, true, false)

def test_match_negative():
    true = negativeValues
    false = positiveValues + zeroValues + stringValues + missingValues + boolValues
    backend_match_value(match.negative, true, false)

def test_match_infinity():
    true = infinityValues
    false = [v for v in numericValues if v not in infinityValues] + stringValues + boolValues
    backend_match_value(match.infinity, true, false)

def test_match_boolean():
    true = boolValues
    false = [1, np.int_(1), 0, np.int_(0), 1.0, np.float_(1), 0.0, np.float_(0)]
    backend_match_value(match.boolean, true, false)

def test_match_integer():
    true = [1, np.int_(1), -1, np.int_(-1), True, False]
    false = [1.0, np.float_(1), -1.0, np.float_(-1)]
    backend_match_value(match.integer, true, false)

def test_match_floating():
    true = [1.0, np.float_(1), -1.0, np.float_(-1)]
    false = [True, False, 1, np.int_(1), -1, np.int_(-1)]
    backend_match_value(match.floating, true, false)

@noLogEntryExpected
def backend_match_anyAll(anyOrAll, func, data):
    """backend for match functions accepting 1D and 2D data and testing for any or all"""
    sparseSafe = True
    if func not in sparseUnsafeMatches:
        sparseSafe = False 
        data = np.array(data, dtype=np.object_)
    else:
        data = np.array(data)
    for constructor in getDataConstructors(includeSparse=sparseSafe):
        toTest = constructor(data, useLog=False)
        # test whole matrix
        if anyOrAll == 'any':
            assert func(toTest)
        else:
            allMatching = toTest[:,2]
            allMatching.features.append(allMatching, useLog=False)
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
        toTest = toTest.T
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
    fill = np.nan
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll('any', match.anyMissing, data)

def test_match_allMissing():
    fill = np.nan
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll('all', match.allMissing, data)

def test_match_anyNonMissing():
    fill = np.nan
    data = [[fill,fill,3], [fill,5,6], [fill,8,9]]
    backend_match_anyAll('any', match.anyNonMissing, data)

def test_match_allNonMissing():
    fill = np.nan
    data = [[fill,fill,3], [fill,5,6], [fill,8,9]]
    backend_match_anyAll('all', match.allNonMissing, data)

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
    fill = 1
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

@noLogEntryExpected
def test_match_QueryString():
    match.QueryString("x == 0")
    match.QueryString("isAFeature is missing")
    match.QueryString("> 3")

def test_match_QueryString_elementQueryIsSet():
    elem = match.QueryString('== feature one > 3', elementQuery=True)
    assert elem('feature one > 3')

    axis = match.QueryString('== feature one > 3', elementQuery=False)
    fnames = ['== feature one', '== feature two', '== feature three']
    data = nimble.data([[1, 3, 5], [4, 6, 8]], featureNames=fnames)
    assert axis(data.points[1])
    assert not axis(data.points[0])

    with raises(InvalidArgumentValue):
        match.QueryString('> 3', elementQuery=False)

    with raises(InvalidArgumentValue):
        match.QueryString('ft2 > 3', elementQuery=True)

def test_match_QueryString_invalid():
    with raises(InvalidArgumentValue):
        match.QueryString("x == 3 == 0")
    with raises(InvalidArgumentValue):
        match.QueryString("is not there")
    with raises(InvalidArgumentValue):
        match.QueryString("is 1")
    with raises(InvalidArgumentValue):
        match.QueryString("< feature > 3")
