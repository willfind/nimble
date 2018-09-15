from __future__ import absolute_import
import numpy
import six

from UML.exceptions import ArgumentException


def missing(value):
    return value is None or value != value

def numeric(value):
    return isinstance(value, (int, float, complex, numpy.number))

def nonNumeric(value):
    return not numeric(value)

def zero(value):
    return value == 0

def nonZero(value):
    return not zero(value)

def positive(value):
    if not numeric(value):
        return False
    try:
        return value > 0
    except Exception:
        return False

def negative(value):
    if not numeric(value):
        return False
    try:
        return value < 0
    except Exception:
        return False

def anyAllValuesBackend(anyOrAll, matrix, func):
    if anyOrAll == 'any':
        test = any
    else:
        test = all
    try:
        # 1D matrix
        return test([func(val) for val in matrix])
    except ArgumentException:
        if anyOrAll == 'any':
            for i in range(matrix.features):
                if test([func(val) for val in matrix[:, i]]):
                    return True
            return False
        else:
            for i in range(matrix.features):
                if not test([func(val) for val in matrix[:, i]]):
                    return False
            return True

def anyValuesMissing(matrix):
    return anyAllValuesBackend('any', matrix, missing)

def allValuesMissing(matrix):
    return anyAllValuesBackend('all', matrix, missing)

def anyValuesNumeric(matrix):
    return anyAllValuesBackend('any', matrix, numeric)

def allValuesNumeric(matrix):
    return anyAllValuesBackend('all', matrix, numeric)

def anyValuesNonNumeric(matrix):
    return anyAllValuesBackend('any', matrix, nonNumeric)

def allValuesNonNumeric(matrix):
    return anyAllValuesBackend('all', matrix, nonNumeric)

def anyValuesZero(matrix):
    return anyAllValuesBackend('any', matrix, zero)

def allValuesZero(matrix):
    return anyAllValuesBackend('all', matrix, zero)

def anyValuesNonZero(matrix):
    return anyAllValuesBackend('any', matrix, nonZero)

def allValuesNonZero(matrix):
    return anyAllValuesBackend('all', matrix, nonZero)

def anyValuesPositive(matrix):
    return anyAllValuesBackend('any', matrix, positive)

def allValuesPositive(matrix):
    return anyAllValuesBackend('all', matrix, positive)

def anyValuesNegative(matrix):
    return anyAllValuesBackend('any', matrix, negative)

def allValuesNegative(matrix):
    return anyAllValuesBackend('all', matrix, negative)

def convertMatchToFunction(match):
    if not hasattr(match, '__call__'):
        if ((hasattr(match, '__iter__') or
           hasattr(match, '__getitem__')) and
           not isinstance(match, six.string_types)):
            matchList = match
            match = lambda x: x in matchList
        else:
            matchConstant = match
            match = lambda x: x == matchConstant
    return match

def anyValues(match):
    match = convertMatchToFunction(match)
    def anyValueFinder(matrix):
        return anyAllValuesBackend('any', matrix, match)
    return anyValueFinder

def allValues(match):
    match = convertMatchToFunction(match)
    def allValueFinder(matrix):
        return anyAllValuesBackend('all', matrix, match)
    return allValueFinder
