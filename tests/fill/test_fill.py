import nimble
from nimble import fill
from nimble import match
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from ..assertionHelpers import noLogEntryExpected

@noLogEntryExpected
def test_fillFactory_matchNumeric_fillNumeric():
    func = fill.factory(0, 1)
    data = [1, 1, 2]
    exp = [0, 0, 2]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert func(toTest) == exp

@noLogEntryExpected
def test_fillFactory_matchString_fillString():
    func = fill.factory('b', 'a')
    data = ['a', 'a', 'c']
    exp = ['b', 'b', 'c']
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert func(toTest) == exp

@noLogEntryExpected
def test_fillFactory_matchString_fillNumeric():
    func = fill.factory(0, 'a')
    data = ['a', 'a', 'c']
    exp = [0, 0, 'c']
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert func(toTest) == exp

@noLogEntryExpected
def test_fillFactory_matchNumeric_fillString():
    func = fill.factory('a', 0)
    data = [0, 0, 1]
    exp = ['a', 'a', 1]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert func(toTest) == exp

@noLogEntryExpected
def test_fillFactory_matchNumeric_fillNone():
    func = fill.factory(None, 1)
    data = [1, 1, 0]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert func(toTest)[0] != func(toTest)[0]
        assert func(toTest)[1] != func(toTest)[1]
        assert func(toTest)[2] == func(toTest)[2]

@noLogEntryExpected
def test_fillFactory_matchString_fillNone():
    func = fill.factory(None, 'a')
    data = ['a', 'a', 0]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert func(toTest)[0] != func(toTest)[0]
        assert func(toTest)[1] != func(toTest)[1]
        assert func(toTest)[2] == func(toTest)[2]

@noLogEntryExpected
def test_fillFactory_matchList_fillConstant():
    func = fill.factory(0, [1, 2])
    data = [1, 2, 1, 2]
    exp = [0, 0, 0, 0]
    assert func(data) == exp

@noLogEntryExpected
def test_fillFactory_matchList_fillFunction():
    func = fill.factory(fill.mean, [1, 2])
    # 1, 2 should be ignored for mean calculation
    data = [1, 2, 3]
    exp = [3, 3, 3]
    assert func(data) == exp

@noLogEntryExpected
def test_fillFactory_matchFunction_fillConstant():
    func = fill.factory(0, match.missing)
    data = [None, None, None]
    exp = [0, 0, 0]
    assert func(data) == exp

@noLogEntryExpected
def test_fillFactory_matchFunction_fillFunction():
    func = fill.factory(fill.mean, match.missing)
    data = [1, None, 5]
    exp = [1, 3, 5]
    assert func(data) == exp

@noLogEntryExpected
def test_constant_noMatches():
    data = [1, 2, 2, 9]
    match = lambda x: False
    constant = 100
    expected = [1, 2, 2, 9]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert fill.constant(toTest, match, constant) == expected

@noLogEntryExpected
def test_constant_number_ignoreMatches():
    data = [1, 2, 2, 9]
    match = lambda x: x == 2
    constant = 100
    expected = [1, 100, 100, 9]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert fill.constant(toTest, match, constant) == expected

@noLogEntryExpected
def test_constant_string_ignoreMatches():
    data = [1, 2, 2, 9]
    match = lambda x: x == 2
    constant = ""
    expected = [1, "", "", 9]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert fill.constant(toTest, match, constant) == expected

@noLogEntryExpected
def test_constant_allMatches():
    data = [1, 2, 2, 9]
    match = lambda x: x in [1, 2, 9]
    constant = 100
    expected = [100, 100, 100, 100]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert fill.constant(toTest, match, constant) == expected

@noLogEntryExpected
def backend_fill(func, data, match, expected=None):
    "backend for fill functions that do not require additional arguments"
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert func(toTest, match) == expected

@noLogEntryExpected
def backend_fill_exception(func, data, match, exceptionType):
    "backend for fill functions when testing exception raising"
    for t in nimble.data.available:
        try:
            toTest = nimble.createData(t, data, useLog=False)
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
    backend_fill_exception(fill.mean, data, match, InvalidArgumentValue)

def test_mean_cannotCalculate_exception():
    data = ['a', 'b', 3, 4]
    match = lambda x: x == 'b'
    backend_fill_exception(fill.mean, data, match, InvalidArgumentValue)

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
    backend_fill_exception(fill.median, data, match, InvalidArgumentValue)

def test_median_cannotCalculate_exception():
    data = ['a', 'b', 3, 4]
    match = lambda x: x == 'b'
    backend_fill_exception(fill.median, data, match, InvalidArgumentValue)

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
    backend_fill_exception(fill.mode, data, match, InvalidArgumentValue)

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
    backend_fill_exception(fill.forwardFill, data, match, InvalidArgumentValue)

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
    backend_fill_exception(fill.backwardFill, data, match, InvalidArgumentValue)

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

@noLogEntryExpected
def test_interpolate_withArguments():
    data = [1, "na", "na", 5]
    arguments = {}
    # linear function y = 2x + 5
    arguments['xp'] = [0, 4, 8]
    arguments['fp'] = [5, 13, 21]
    match = lambda x: x == "na"
    expected = [1, 7, 9, 5]
    for t in nimble.data.available:
        toTest = nimble.createData(t, data, useLog=False)
        assert fill.interpolate(toTest, match, **arguments) == expected

@noLogEntryExpected
def test_interpolate_xKwargIncluded_exception():
    data = [1, "na", "na", 5]
    arguments = {}
    # linear function y = 2x + 5
    arguments['xp'] = [0, 4, 8]
    arguments['fp'] = [5, 13, 21]
    arguments['x'] = [1]  # disallowed argument
    match = lambda x: x == "na"
    for t in nimble.data.available:
        try:
            toTest = nimble.createData(t, data, useLog=False)
            ret = fill.interpolate(toTest, match, **arguments)
            assert False  # expected TypeError
        except TypeError:
            pass

@noLogEntryExpected
def test_kNeighborsRegressor_noMatches():
    vect = [[1, 1, 1]]
    data = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]
    match = lambda x: False
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = toTest.copy()
        assert fill.kNeighborsRegressor(toTest, match, dataObj) == expTest

@noLogEntryExpected
def test_kNeighborsRegressor_withMatch_K1():
    vect = [[1, 1, None]]
    data = [[1, 1, 1], [1, 1, None], [2, 2, 2], [2, 2, 2], [None, 3, 3], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 1}
    expVect = [[1, 1, 1]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = nimble.createData(t, expVect, useLog=False)
        assert fill.kNeighborsRegressor(toTest, match, dataObj, **arguments) == expTest

@noLogEntryExpected
def test_kNeighborsRegressor_withMatch_K3():
    vect = [[1, 1, None]]
    data = [[1, 1, 5], [1, 1, None], [2, 2, 2], [2, 2, 2], [None, 3, 3], [5, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    expVect = [[1, 1, 3]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = nimble.createData(t, expVect, useLog=False)
        assert fill.kNeighborsRegressor(toTest, match, dataObj, **arguments) == expTest

@noLogEntryExpected
def test_kNeighborsRegressor_multipleMatch_K1():
    vect = [[1, None, None]]
    data = [[1, 1, 1], [1, None, None], [2, 2, 2], [2, 2, 2], [None, 3, None], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 1}
    expVect = [[1, 1, 1]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = nimble.createData(t, expVect, useLog=False)
        assert fill.kNeighborsRegressor(toTest, match, dataObj, **arguments) == expTest

@noLogEntryExpected
def test_kNeighborsRegressor_multipleMatch_K3():
    vect = [1, None, None]
    data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    expVect = [[1, 2, 6]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = nimble.createData(t, expVect, useLog=False)
        assert fill.kNeighborsRegressor(toTest, match, dataObj, **arguments) == expTest

def test_kNeighborsRegressor_featureProvided():
    vect = [[1], [None], [None]]
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        try:
            fill.kNeighborsRegressor(toTest, match, dataObj, **arguments)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

def test_kNeighborsRegressor_exception_allMatches():
    vect = [None, None, None]
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        try:
            fill.kNeighborsRegressor(toTest, match, dataObj, **arguments)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

@noLogEntryExpected
def test_kNeighborsClassifier_noMatches():
    vect = [[1, 1, 1]]
    data = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]
    match = lambda x: False
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = toTest.copy()
        assert fill.kNeighborsClassifier(toTest, match, dataObj) == expTest

@noLogEntryExpected
def test_kNeighborsClassifier_withMatch_K1():
    vect = [[1, 1, None]]
    data = [[1, 1, 1], [1, 1, None], [2, 2, 2], [2, 2, 2], [None, 3, 3], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 1}
    expVect = [[1, 1, 1]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = nimble.createData(t, expVect, useLog=False)
        assert fill.kNeighborsClassifier(toTest, match, dataObj, **arguments) == expTest

@noLogEntryExpected
def test_kNeighborsClassifier_withMatch_K3():
    vect = [[1, 1, None]]
    data = [[1, 1, 1], [1, 1, None], [2, 2, 2], [2, 2, 2], [None, 3, 3], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    expVect = [[1, 1, 2]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = nimble.createData(t, expVect, useLog=False)
        assert fill.kNeighborsClassifier(toTest, match, dataObj, **arguments) == expTest

@noLogEntryExpected
def test_kNeighborsClassifier_multipleMatch_K1():
    vect = [1, None, None]
    data = [[1, 1, 1], [1, None, None], [2, 2, 2], [2, 2, 2], [None, 3, None], [3, 3, 3]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 1}
    expVect = [[1, 1, 1]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = nimble.createData(t, expVect, useLog=False)
        assert fill.kNeighborsClassifier(toTest, match, dataObj, **arguments) == expTest

@noLogEntryExpected
def test_kNeighborsClassifier_multipleMatch_K3():
    vect = [1, None, None]
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    expVect = [[1, 3, 6]]
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        expTest = nimble.createData(t, expVect, useLog=False)
        assert fill.kNeighborsClassifier(toTest, match, dataObj, **arguments) == expTest

def test_kNeighborsClassifier_exception_featureProvided():
    vect = [[1], [None], [None]]
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        try:
            fill.kNeighborsClassifier(toTest, match, dataObj, **arguments)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

def test_kNeighborsClassifier_exception_allMatches():
    vect = [None, None, None]
    data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
    match = lambda x: x != x
    arguments = {'n_neighbors': 3}
    for t in nimble.data.available:
        toTest = nimble.createData(t, vect, useLog=False)
        dataObj = nimble.createData(t, data, useLog=False)
        try:
            fill.kNeighborsClassifier(toTest, match, dataObj, **arguments)
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass
