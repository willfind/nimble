from __future__ import absolute_import

import UML
from UML import fill
from UML import match
from UML.exceptions import ArgumentException


def test_fillFactory_matchNumeric_fillNumeric():
    func = fill.factory(1, 0)
    data = [1, 1, 2]
    exp = [0, 0, 2]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert func(toTest) == exp

def test_fillFactory_matchString_fillString():
    func = fill.factory('a', 'b')
    data = ['a', 'a', 'c']
    exp = ['b', 'b', 'c']
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert func(toTest) == exp

def test_fillFactory_matchString_fillNumeric():
    func = fill.factory('a', 0)
    data = ['a', 'a', 'c']
    exp = [0, 0, 'c']
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert func(toTest) == exp

def test_fillFactory_matchNumeric_fillString():
    func = fill.factory(0, 'a')
    data = [0, 0, 1]
    exp = ['a', 'a', 1]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert func(toTest) == exp

def test_fillFactory_matchNumeric_fillNone():
    func = fill.factory(1, None)
    data = [1, 1, 0]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert func(toTest)[0] != func(toTest)[0]
        assert func(toTest)[1] != func(toTest)[1]
        assert func(toTest)[2] == func(toTest)[2]

def test_fillFactory_matchString_fillNone():
    func = fill.factory('a', None)
    data = ['a', 'a', 0]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert func(toTest)[0] != func(toTest)[0]
        assert func(toTest)[1] != func(toTest)[1]
        assert func(toTest)[2] == func(toTest)[2]

def test_fillFactory_matchList_fillConstant():
    func = fill.factory([1, 2], 0)
    data = [1, 2, 1, 2]
    exp = [0, 0, 0, 0]
    assert func(data) == exp

def test_fillFactory_matchList_fillFunction():
    func = fill.factory([1, 2], fill.mean)
    # 1, 2 should be ignored for mean calculation
    data = [1, 2, 3]
    exp = [3, 3, 3]
    assert func(data) == exp

def test_fillFactory_matchFunction_fillConstant():
    func = fill.factory(match.missing, 0)
    data = [None, None, None]
    exp = [0, 0, 0]
    assert func(data) == exp

def test_fillFactory_matchFunction_fillFunction():
    func = fill.factory(match.missing, fill.mean)
    data = [1, None, 5]
    exp = [1, 3, 5]
    assert func(data) == exp

def test_constant_noMatches():
    data = [1, 2, 2, 9]
    match = lambda x: False
    constant = 100
    expected = [1, 2, 2, 9]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert fill.constant(toTest, match, constant) == expected

def test_constant_number_ignoreMatches():
    data = [1, 2, 2, 9]
    match = lambda x: x == 2
    constant = 100
    expected = [1, 100, 100, 9]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert fill.constant(toTest, match, constant) == expected

def test_constant_string_ignoreMatches():
    data = [1, 2, 2, 9]
    match = lambda x: x == 2
    constant = ""
    expected = [1, "", "", 9]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert fill.constant(toTest, match, constant) == expected

def test_constant_allMatches():
    data = [1, 2, 2, 9]
    match = lambda x: x in [1, 2, 9]
    constant = 100
    expected = [100, 100, 100, 100]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert fill.constant(toTest, match, constant) == expected

def backend_fill(func, data, match, expected=None):
    "backend for fill functions that do not require additional arguments"
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert func(toTest, match) == expected

def backend_fill_exception(func, data, match, exceptionType):
    "backend for fill functions when testing exception raising"
    for t in UML.data.available:
        try:
            toTest = UML.createData(t, data)
            func(toTest, match)
            assert False  # Expected an exception
        except exceptionType as et:
#            print et
            pass  # if we get the right thing, carry on.

def test_mean_noMatches():
    data = [1, 2, 2, 9]
    match = lambda x: False
    expected = [1, 2, 2, 9]
    backend_fill(fill.mean, data, match, expected)

def test_mean_ignoreMatch():
    data = [1, 2, 2, 9]
    match = lambda x: x == 2
    expected = [1, 5, 5, 9]
    backend_fill(fill.mean, data, match, expected)

def test_mean_allMatches_exception():
    data = [1, 2, 2, 9]
    match = lambda x: x in [1, 2, 2, 9]
    backend_fill_exception(fill.mean, data, match, ArgumentException)

def test_mean_cannotCalculate_exception():
    data = ['a', 'b', 3, 4]
    match = lambda x: x == 'b'
    backend_fill_exception(fill.mean, data, match, ArgumentException)

def test_median_noMatches():
    data = [1, 2, 9, 2]
    match = lambda x: False
    expected = [1, 2, 9, 2]
    backend_fill(fill.median, data, match, expected)

def test_median_ignoreMatch():
    data = [1, 2, 9, 2]
    match = lambda x: x == 2
    expected = [1, 5, 9, 5]
    backend_fill(fill.median, data, match, expected)

def test_median_allMatches_exception():
    data = [1, 2, 9, 2]
    match = lambda x: x in [1, 2, 9]
    backend_fill_exception(fill.median, data, match, ArgumentException)

def test_median_cannotCalculate_exception():
    data = ['a', 'b', 3, 4]
    match = lambda x: x == 'b'
    backend_fill_exception(fill.median, data, match, ArgumentException)

def test_mode_noMatches():
    data = [1, 2, 2, 9]
    match = lambda x: False
    expected = [1, 2, 2, 9]
    backend_fill(fill.mode, data, match, expected)

def test_mode_ignoreMatch():
    data = [1, 2, 2, 2, 9, 9]
    match = lambda x: x == 2
    expected = [1, 9, 9, 9, 9, 9]
    backend_fill(fill.mode, data, match, expected)

def test_mode_allMatches_exception():
    data = [1, 2, 2, 9, 9]
    match = lambda x: x in [1, 2, 9]
    backend_fill_exception(fill.mode, data, match, ArgumentException)

def test_forwardFill_noMatches():
    data = [1, 2, 3, 4]
    match = lambda x: False
    expected = data
    backend_fill(fill.forwardFill, data, match, expected)

def test_forwardFill_withMatch():
    data = [1, 2, 3, 4]
    match = lambda x: x == 2
    expected = [1, 1, 3, 4]
    backend_fill(fill.forwardFill, data, match, expected)

def test_forwardFill_consecutiveMatches():
    data = [1, 2, 2, 2, 3, 4, 5]
    match = lambda x: x == 2
    expected = [1, 1, 1, 1, 3, 4, 5]
    backend_fill(fill.forwardFill, data, match, expected)

def test_forwardFill_InitialContainsMatch_exception():
    data = [1, 2, 3, 4]
    match = lambda x: x == 1
    backend_fill_exception(fill.forwardFill, data, match, ArgumentException)

def test_backwardFill_noMatches():
    data = [1, 2, 3, 4]
    match = lambda x: False
    expected = data
    backend_fill(fill.backwardFill, data, match, expected)

def test_backwardFill_withMatch():
    data = [1, 2, 3, 4]
    match = lambda x: x == 2
    expected = [1, 3, 3, 4]
    backend_fill(fill.backwardFill, data, match, expected)

def test_backwardFill_consecutiveMatches():
    data = [1, 2, 2, 2, 3, 4, 5]
    match = lambda x: x == 2
    expected = [1, 3, 3, 3, 3, 4, 5]
    backend_fill(fill.backwardFill, data, match, expected)

def test_backwardFill_FinalContainsMatch_exception():
    data = [1, 2, 3, 4]
    match = lambda x: x == 4
    backend_fill_exception(fill.backwardFill, data, match, ArgumentException)

def test_interpolate_noMatches():
    data = [1, 2, 2, 10]
    match = lambda x: False
    expected = data
    backend_fill(fill.interpolate, data, match, expected)

def test_interpolate_withMatch():
    data = [1, 2, 2, 10]
    match = lambda x: x == 2
    # linear function y = x + 3
    expected = [1, 4, 7, 10]
    backend_fill(fill.interpolate, data, match, expected)

def test_interpolate_withArguments():
    data = [1, "na", "na", 5]
    arguments = {}
    # linear function y = 2x + 5
    arguments['xp'] = [0, 4, 8]
    arguments['fp'] = [5, 13, 21]
    match = lambda x: x == "na"
    expected = [1, 7, 9, 5]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert fill.interpolate(toTest, match, **arguments) == expected

def test_interpolate_xKwargIncluded_exception():
    data = [1, "na", "na", 5]
    arguments = {}
    # linear function y = 2x + 5
    arguments['xp'] = [0, 4, 8]
    arguments['fp'] = [5, 13, 21]
    arguments['x'] = [1]  # disallowed argument
    match = lambda x: x == "na"
    for t in UML.data.available:
        try:
            toTest = UML.createData(t, data)
            ret = fill.interpolate(toTest, match, **arguments)
            assert False  # expected ArgumentException
        except ArgumentException:
            pass

def test_kNeighborsRegressor_noMatches():
    data = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]
    match = lambda x: False
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = toTest.copy()
        assert fill.kNeighborsRegressor(toTest, match) == expTest

def test_kNeighborsRegressor_withMatch_K1():
    data = [[1, 1, 1], [1, 1, None], [2, 2, 2], [2, 2, 2], [None, 3, 3], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 1}
    expData = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = UML.createData(t, expData)
        assert fill.kNeighborsRegressor(toTest, match, **arguments) == expTest

def test_kNeighborsRegressor_withMatch_K3():
    data = [[1, 1, 5], [1, 1, None], [2, 2, 2], [2, 2, 2], [None, 3, 3], [5, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    expData = [[1, 1, 5], [1, 1, 3], [2, 2, 2], [2, 2, 2], [3, 3, 3], [5, 3, 3]]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = UML.createData(t, expData)
        assert fill.kNeighborsRegressor(toTest, match, **arguments) == expTest

def test_kNeighborsRegressor_multipleMatch_K1():
    data = [[1, 1, 1], [1, None, None], [2, 2, 2], [2, 2, 2], [None, 3, None], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 1}
    expData = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = UML.createData(t, expData)
        assert fill.kNeighborsRegressor(toTest, match, **arguments) == expTest

def test_kNeighborsRegressor_multipleMatch_K3():
    data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = UML.createData(t, expData)
        assert fill.kNeighborsRegressor(toTest, match, **arguments) == expTest

def test_kNeighborsClassifier_noMatches():
    data = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]
    match = lambda x: False
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = toTest.copy()
        assert fill.kNeighborsClassifier(toTest, match) == expTest

def test_kNeighborsClassifier_withMatch_K1():
    data = [[1, 1, 1], [1, 1, None], [2, 2, 2], [2, 2, 2], [None, 3, 3], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 1}
    expData = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = UML.createData(t, expData)
        assert fill.kNeighborsClassifier(toTest, match, **arguments) == expTest

def test_kNeighborsClassifier_withMatch_K3():
    data = [[1, 1, 1], [1, 1, None], [2, 2, 2], [2, 2, 2], [None, 3, 3], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    expData = [[1, 1, 1], [1, 1, 2], [2, 2, 2], [2, 2, 2], [2, 3, 3], [3, 3, 3]]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = UML.createData(t, expData)
        assert fill.kNeighborsClassifier(toTest, match, **arguments) == expTest

def test_kNeighborsClassifier_multipleMatch_K1():
    data = [[1, 1, 1], [1, None, None], [2, 2, 2], [2, 2, 2], [None, 3, None], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 1}
    expData = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = UML.createData(t, expData)
        assert fill.kNeighborsClassifier(toTest, match, **arguments) == expTest

def test_kNeighborsClassifier_multipleMatch_K3():
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        expTest = UML.createData(t, expData)
        assert fill.kNeighborsClassifier(toTest, match, **arguments) == expTest
