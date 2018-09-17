from __future__ import absolute_import
import numpy
from nose.tools import raises

import UML
from UML import fill
from UML import match
from UML.exceptions import ArgumentException


def test_fillFactory_matchConstant_fillConstant():
    func = fill.factory(1, 0)
    data = [1, 1, 1]
    exp = [0, 0, 0]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert func(toTest) == exp

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

def backend_fill(func, data, match, expected=None):
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        if expected:
            assert func(toTest, match) == expected
        else:
            func(toTest, match)

def test_mean_noMatches():
    data = [1, 2, 9]
    match = lambda x: False
    expected = [1, 2, 9]
    backend_fill(fill.mean, data, match, expected)

def test_mean_ignoreMatch():
    data = [1, 2, 9]
    match = lambda x: x == 2
    expected = [1, 5, 9]
    backend_fill(fill.mean, data, match, expected)

@raises(ArgumentException)
def test_mean_allMatches():
    data = [1, 2, 9]
    match = lambda x: x in [1, 2, 9]
    backend_fill(fill.mean, data, match)

def test_median_noMatches():
    data = [1, 9, 2]
    match = lambda x: False
    expected = [1, 9, 2]
    backend_fill(fill.median, data, match, expected)

def test_median_ignoreMatch():
    data = [1, 9, 2]
    match = lambda x: x == 2
    expected = [1, 9, 5]
    backend_fill(fill.median, data, match, expected)

@raises(ArgumentException)
def test_median_allMatches():
    data = [1, 2, 9]
    match = lambda x: x in [1, 2, 9]
    backend_fill(fill.median, data, match)

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

@raises(ArgumentException)
def test_mode_allMatches():
    data = [1, 2, 2, 9, 9]
    match = lambda x: x in [1, 2, 9]
    backend_fill(fill.mode, data, match)

def test_forward_noMatches():
    data = [1, 2, 3]
    match = lambda x: False
    expected = data
    backend_fill(fill.forward, data, match, expected)


def test_forward_withMatch():
    data = [1, 2, 3]
    match = lambda x: x == 2
    expected = [1, 1, 3]
    backend_fill(fill.forward, data, match, expected)

@raises(ArgumentException)
def test_forward_InitialContainsMatch():
    data = [1, 2, 3]
    match = lambda x: x == 1
    backend_fill(fill.forward, data, match)

def test_backward_noMatches():
    data = [1, 2, 3]
    match = lambda x: False
    expected = data
    backend_fill(fill.backward, data, match, expected)

def test_backward_withMatch():
    data = [1, 2, 3]
    match = lambda x: x == 2
    expected = [1, 3, 3]
    backend_fill(fill.backward, data, match, expected)

@raises(ArgumentException)
def test_backward_InitialContainsMatch():
    data = [1, 2, 3]
    match = lambda x: x == 3
    backend_fill(fill.backward, data, match)

def test_interpolate_noMatches():
    data = [1, 2, 5]
    match = lambda x: False
    expected = data
    backend_fill(fill.interpolate, data, match, expected)

def test_interpolate_withMatch():
    data = [1, 2, 5]
    match = lambda x: x == 2
    expected = [1, 3, 5]
    backend_fill(fill.interpolate, data, match, expected)

def test_interpolate_withArguments():
    data = [1,2,5]
    arguments = {}
    arguments['x'] = [1]
    arguments['xp'] = [0, 2]
    arguments['fp'] = [1, 5]
    match = lambda x: x == 2
    expected = [1, 3, 5]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert fill.interpolate(toTest, match, arguments) == expected

@raises(ArgumentException)
def test_interpolate_badArguments():
    data = [1,2,5]
    arguments = 11
    match = lambda x: x == 2
    expected = [1, 3, 5]
    for t in UML.data.available:
        toTest = UML.createData(t, data)
        assert fill.interpolate(toTest, match, arguments) == expected
