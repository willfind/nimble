"""
TODO
"""
from __future__ import absolute_import

import numpy
import six

from UML.exceptions import ArgumentException


def missing(value):
    """
    Determines if a single value is missing
    missing values in UML are None and (python or numpy) nan values
    """
    return value is None or value != value

def numeric(value):
    """
    Determines if a single value is numeric
    numeric values include any values with a python or numpy numeric type
    """
    return isinstance(value, (int, float, complex, numpy.number))

def nonNumeric(value):
    """
    Determines if a single is not numeric
    nonNumeric values include any values without a python or numpy numeric type
    """
    return not isinstance(value, (int, float, complex, numpy.number))

def zero(value):
    """
    Determines if a single value is equal to zero
    zero values include any numeric value equal to zero
    """
    return value == 0

def nonZero(value):
    """
    Determines if a single value is not equal to zero
    nonZero values include any non-numeric value and any numeric value
    not equal to zero
    """
    return value != 0

def positive(value):
    """
    Determines if a single value is greater than zero
    positive values include any numeric value greater than zero
    """
    try:
        # py2 inteprets strings as greater than 0, trying to add zero will
        # raise exception for strings
        value + 0
        return value > 0
    except Exception:
        return False

def negative(value):
    """
    Determines if a single value is less than zero
    negative values include any numeric value less than zero
    """
    try:
        # py2 inteprets None as less than 0, trying to add zero will
        # raise exception for None
        value + 0
        return value < 0
    except Exception:
        return False

def anyAllValuesBackend(quantity, data, match):
    """
    Backend function for determining if the data contains any matching values
    or all matching values

    quantity may only be the function any or the function all
    """
    try:
        # 1D data
        return quantity([match(val) for val in data])
    except ArgumentException:
        # 2D data
        if quantity is any:
            # if any feature contains a match we can return True
            for i in range(data.features):
                if quantity([match(val) for val in data[:, i]]):
                    return True
            return False
        # if any feature does not have all matches we can return False
        for i in range(data.features):
            if not quantity([match(val) for val in data[:, i]]):
                return False
        return True

def convertMatchToFunction(match):
    """
    Convert iterables and constants to functions so that a match can always
    be determined by calling match(value)
    """
    if not hasattr(match, '__call__'):
        # case1: list-like
        if ((hasattr(match, '__iter__') or hasattr(match, '__getitem__'))
                and not isinstance(match, six.string_types)):
            matchList = match
            # if nans in the list, need to include separate check in function
            if not all([val == val for val in matchList]):
                match = lambda x: x != x or x in matchList
            else:
                match = lambda x: x in matchList
        # case2: constant
        else:
            matchConstant = match
            if matchConstant is None or matchConstant != matchConstant:
                match = lambda x: x is None or x != x
            else:
                match = lambda x: x == matchConstant
    return match

def anyValues(match):
    """
    Creates a function which accepts data and returns if any values within the
    data contain a match

    match can be a single value, iterable of values, or a boolean function that
    accepts a single value as input
    """
    match = convertMatchToFunction(match)
    def anyValueFinder(data):
        return anyAllValuesBackend(any, data, match)
    return anyValueFinder

def allValues(match):
    """
    Creates a function which accepts data and returns if all values within the
    data contain a match

    match can be a single value, iterable of values, or a boolean function that
    accepts a single value as input
    """
    match = convertMatchToFunction(match)
    def allValueFinder(data):
        return anyAllValuesBackend(all, data, match)
    return allValueFinder

def anyMissing(data):
    """
    Determines if any values in the data are missing
    """
    return anyAllValuesBackend(any, data, missing)

def allMissing(data):
    """
    Determines if all values in the data are missing
    """
    return anyAllValuesBackend(all, data, missing)

def anyNumeric(data):
    """
    Determines if any values in the data are numeric
    """
    return anyAllValuesBackend(any, data, numeric)

def allNumeric(data):
    """
    Determines if all values in the data are numeric
    """
    return anyAllValuesBackend(all, data, numeric)

def anyNonNumeric(data):
    """
    Determines if any values in the data are non-numeric
    """
    return anyAllValuesBackend(any, data, nonNumeric)

def allNonNumeric(data):
    """
    Determines if all values in the data are non-numeric
    """
    return anyAllValuesBackend(all, data, nonNumeric)

def anyZero(data):
    """
    Determines if any values in the data are equal to zero
    """
    return anyAllValuesBackend(any, data, zero)

def allZero(data):
    """
    Determines if all values in the data are equal to zero
    """
    return anyAllValuesBackend(all, data, zero)

def anyNonZero(data):
    """
    Determines if any values in the data are not equal to zero
    """
    return anyAllValuesBackend(any, data, nonZero)

def allNonZero(data):
    """
    Determines if all values in the data are not equal to zero
    """
    return anyAllValuesBackend(all, data, nonZero)

def anyPositive(data):
    """
    Determines if any values in the data are greater than zero
    """
    return anyAllValuesBackend(any, data, positive)

def allPositive(data):
    """
    Determines if all values in the data are greater than zero
    """
    return anyAllValuesBackend(all, data, positive)

def anyNegative(data):
    """
    Determines if any values in the data are less than zero
    """
    return anyAllValuesBackend(any, data, negative)

def allNegative(data):
    """
    Determines if all values in the data are less than zero
    """
    return anyAllValuesBackend(all, data, negative)
