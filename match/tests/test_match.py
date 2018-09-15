from __future__ import absolute_import
import numpy

import UML
from UML import match


def backend_match_value(func, true, false):
    for t in true:
        assert func(t)
    for f in false:
        assert not func(f)

missingTypes = [None, float('nan'), numpy.nan]
stringTypes = ['a', str(1)]
zeroTypes = [0, float(0), numpy.int(0), numpy.float(0)]
positiveTypes = [3, float(3), numpy.int(3), numpy.float(3)]
negativeTypes = [-3, float(-3), numpy.int(-3), numpy.float(-3)]
numericTypes = positiveTypes + negativeTypes + zeroTypes + [numpy.nan]

def test_match_missing():
    true = missingTypes
    false = numericTypes[:-1] + stringTypes
    backend_match_value(match.missing, true, false)

def test_match_numeric():
    true = numericTypes
    false = stringTypes + [None]
    backend_match_value(match.numeric, true, false)

def test_match_nonNumeric():
    true = stringTypes + [None]
    false = numericTypes
    backend_match_value(match.nonNumeric, true, false)

def test_match_zero():
    true = zeroTypes
    false = positiveTypes + negativeTypes + stringTypes + missingTypes
    backend_match_value(match.zero, true, false)

def test_match_nonZero():
    true = positiveTypes + negativeTypes + stringTypes + missingTypes
    false = zeroTypes
    backend_match_value(match.nonZero, true, false)

def test_match_positive():
    true = positiveTypes
    false = negativeTypes + zeroTypes + stringTypes + missingTypes
    backend_match_value(match.positive, true, false)

def test_match_negative():
    true = negativeTypes
    false = positiveTypes + zeroTypes + stringTypes + missingTypes
    backend_match_value(match.negative, true, false)

def backend_match_anyAll(func, data):
    any = func.__name__[:3] == 'any'
    data = numpy.array(data, dtype=numpy.object_)
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        # test whole matrix
        if any:
            assert func(toTest)
        else:
            allMatching = toTest[:,2]
            allMatching.appendFeatures(allMatching)
            assert func(allMatching)
        # test by feature
        for i, feature in enumerate(toTest.featureIterator()):
            if i == 0:
                assert not func(feature)
            elif i == 1 and any:
                assert func(feature)
            elif i == 1:
                assert not func(feature)
            else:
                assert func(feature)
        # test by point
        toTest = UML.createData(t, data.T)
        for i, point in enumerate(toTest.pointIterator()):
            if i == 0:
                assert not func(point)
            elif i == 1 and any:
                assert func(point)
            elif i == 1:
                assert not func(point)
            else:
                assert func(point)


def test_match_anyValuesMissing():
    fill = numpy.nan
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll(match.anyValuesMissing, data)

def test_match_allValuesMissing():
    fill = numpy.nan
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll(match.allValuesMissing, data)

def test_match_anyValuesNumeric():
    fill = 3
    data = [['a','b',fill], ['c','d',fill], ['e',fill,fill]]
    backend_match_anyAll(match.anyValuesNumeric, data)

def test_match_allValuesNumeric():
    fill = -3.0
    data = [['a','b',fill], ['c','d',fill], ['e',fill,fill]]
    backend_match_anyAll(match.allValuesNumeric, data)

def test_match_anyValuesNonNumeric():
    fill = 'n/a'
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll(match.anyValuesNonNumeric, data)

def test_match_allValuesNonNumeric():
    fill = 'missing'
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll(match.allValuesNonNumeric, data)

def test_match_anyValuesZero():
    fill = 0
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll(match.anyValuesZero, data)

def test_match_allValuesZero():
    fill = 0.0
    data = [[1,2,fill], [4,5,fill], [7,fill,fill]]
    backend_match_anyAll(match.allValuesZero, data)

def test_match_anyValuesNonZero():
    fill = 1
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.anyValuesNonZero, data)

def test_match_allValuesNonZero():
    fill = 'a'
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.allValuesNonZero, data)

def test_match_anyValuesPositive():
    fill = 1
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.anyValuesPositive, data)

def test_match_allValuesPositive():
    fill = 2.4
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.allValuesPositive, data)

def test_match_anyValuesNegative():
    fill = -1
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.allValuesNegative, data)

def test_match_allValuesNegative():
    fill = -2.4
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.allValuesNegative, data)

def test_match_anyValues_int():
    fill = 3
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.anyValues(fill), data)

def test_match_allValues_int():
    fill = 3
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.allValues(fill), data)

def test_match_anyValues_str():
    fill = 'a'
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.anyValues(fill), data)

def test_match_allValues_str():
    fill = 'a'
    data = [[0,0,fill], [0,0,fill], [0,fill,fill]]
    backend_match_anyAll(match.allValues(fill), data)

def test_match_anyValues_list():
    data = [[0,0,1], [0,0,2], [0,'a','a']]
    backend_match_anyAll(match.anyValues([1, 2, 'a']), data)

def test_match_allValues_list():
    data = [[0,0,1], [0,0,2], [0,'a','a']]
    backend_match_anyAll(match.allValues([1, 2, 'a']), data)

def test_match_anyValues_func():
    data = [[0,0,1], [0,0,2], [0,3,3]]
    backend_match_anyAll(match.anyValues(lambda x: x > 0), data)

def test_match_allValues_func():
    data = [[0,0,1], [0,0,2], [0,3,3]]
    backend_match_anyAll(match.allValues(lambda x: x > 0), data)
