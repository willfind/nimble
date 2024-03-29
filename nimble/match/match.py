
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Variety of functions to determine if data matches given conditions
"""

import numpy as np

def missing(value):
    """
    Determine if a value is missing.

    Return True if Nimble considers the value to be missing. Missing
    values in Nimble are None and (python or numpy) nan values.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    nonMissing, anyMissing, allMissing

    Examples
    --------
    >>> missing(None)
    True
    >>> missing(float('nan'))
    True
    >>> missing(0)
    False
    >>> missing('nan')
    False
    """
    return value is None or value != value

def nonMissing(value):
    """
    Determine if a value is not a missing value.

    Return True if Nimble does not consider the value to be missing.
    Missing values in Nimble are None and (python or numpy) nan values.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    missing, anyMissing, allMissing

    Examples
    --------
    >>> nonMissing(0)
    True
    >>> nonMissing('nan')
    True
    >>> nonMissing(None)
    False
    >>> nonMissing(float('nan'))
    False
    """
    return not missing(value)

def numeric(value):
    """
    Determine if a value is numeric.

    Return True if the value is numeric, otherwise False.  Numeric
    values include any value with a python or numpy numeric type.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyNumeric, allNumeric

    Examples
    --------
    >>> numeric(0)
    True
    >>> numeric(np.float32(8))
    True
    >>> numeric(float('nan'))
    True
    >>> numeric('abc')
    False
    >>> numeric(None)
    False
    """
    return (isinstance(value, (int, float, complex, np.number))
            and not boolean(value))

def nonNumeric(value):
    """
    Determine if a value is non-numeric.

    Return True if the value is non-numeric, otherwise False.
    Non-numeric values include any value without a python or numpy
    numeric type.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyNonNumeric, allNonNumeric

    Examples
    --------
    >>> nonNumeric('abc')
    True
    >>> nonNumeric(None)
    True
    >>> nonNumeric('8')
    True
    >>> nonNumeric(8)
    False
    >>> nonNumeric(float('nan'))
    False
    """
    return not numeric(value)

def zero(value):
    """
    Determine if a value is equal to zero.

    Return True if the value is zero, otherwise False. Zero values
    include any numeric value equal to zero.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyZero, allZero

    Examples
    --------
    >>> zero(0)
    True
    >>> zero(0.0)
    True
    >>> zero(1)
    False
    >>> zero('0')
    False
    """
    return value == 0

def nonZero(value):
    """
    Determine if a value is not equal to zero.

    Return True if the value is not zero, otherwise False. Non-zero
    values include any value not equal to zero.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyNonZero, allNonZero

    Examples
    --------
    >>> nonZero(1)
    True
    >>> nonZero('0')
    True
    >>> nonZero(0)
    False
    >>> nonZero(0.0)
    False
    """
    return value != 0

def positive(value):
    """
    Determine if a value is greater than zero.

    Return True if the value is a numeric value greater than zero,
    otherwise False.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyPositive, allPositive

    Examples
    --------
    >>> positive(1)
    True
    >>> positive(3.333)
    True
    >>> positive(-1)
    False
    >>> positive(0)
    False
    """
    try:
        return value > 0
    except TypeError:
        return False

def negative(value):
    """
    Determine if a value is less than zero.

    Return True if the value is a numeric value less than zero,
    otherwise False.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyNegative, allNegative

    Examples
    --------
    >>> negative(-1)
    True
    >>> negative(-3.333)
    True
    >>> negative(1)
    False
    >>> negative(0)
    False
    """
    try:
        return value < 0
    except TypeError:
        return False

def infinity(value):
    """
    Determine if a value is a float infinity.

    Return True if the value is positive or negative infinity float
    value, otherwise False.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyFalse, allFalse

    Examples
    --------
    >>> infinity(float('inf'))
    True
    >>> infinity(0)
    False
    >>> infinity('inf')
    False
    """
    try:
        return np.isinf(value)
    except TypeError:
        return False

def boolean(value):
    """
    Determine if a value is a boolean value.

    Return True if the value is a boolean type, otherwise return False.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyTrue, allTrue

    Examples
    --------
    >>> boolean(True)
    True
    >>> true(1) # False because 1 not boolean type
    False
    >>> boolean(False)
    True
    >>> boolean('False')
    False
    """
    return isinstance(value, (bool, np.bool_))

def integer(value):
    """
    Determine if a value is an integer type.

    Return True if the value is an integer type, otherwise return False.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyInteger, allInteger

    Examples
    --------
    >>> integer(3)
    True
    >>> integer(3.0)
    False
    >>> integer(True)
    True
    """
    return isinstance(value, (int, np.integer))

def floating(value):
    """
    Determine if a value is a float type.

    Return True if the value is a float type, otherwise return False.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyFloating, allFloating

    Examples
    --------
    >>> floating(3.0)
    True
    >>> floating(3)
    False
    >>> floating(True)
    False
    """
    return isinstance(value, (float, np.floating))

def true(value):
    """
    Determine if a value is a boolean True value.

    Return True if the value is a boolean type equal to True,
    otherwise return False.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyTrue, allTrue

    Examples
    --------
    >>> true(True)
    True
    >>> true(1) # False because 1 not boolean type
    False
    >>> true(False)
    False
    >>> true('True')
    False
    """
    return boolean(value) and value is True

def false(value):
    """
    Determine if a value is a boolean False value.

    Return True if the value is a boolean type equal to False,
    otherwise return False.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyFalse, allFalse

    Examples
    --------
    >>> false(False)
    True
    >>> false(0) # False because 0 not boolean type
    False
    >>> false(True)
    False
    >>> false('False')
    False
    """
    return boolean(value) and value is False

def anyValues(match):
    """
    Factory for functions which will determine if any values match.

    The returned function is designed to input a Nimble data object and
    output True if one or more values in that object contain a match,
    otherwise False.

    Parameters
    ----------
    match : value, iterable, or function
        * value - The value to match in the data.
        * iterable - A list-like, container of values to match in the
          data.
        * function - Input a value and return True if that value is a
          match.

    Returns
    -------
    bool

    See Also
    --------
    anyMissing, anyNumeric, anyNonNumeric, anyZero, anyNonZero,
    anyPositive, anyNegative

    Examples
    --------
    >>> lst = [[1, 1, 1],
    ...        [1, 1, 1],
    ...        [1, 1, -1]]
    >>> X =nimble.data(lst)
    >>> anyNegativeOne = anyValues(-1)
    >>> anyNegativeOne(X)
    True

    >>> lst = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> X =nimble.data(lst)
    >>> anyLetterD = anyValues('D')
    >>> anyLetterD(X)
    False
    """
    match = _convertMatchToFunction(match)
    def anyValueFinder(data):
        return anyAllValuesBackend(any, data, match)
    return anyValueFinder

def allValues(match):
    """
    Factory for functions which will determine if all values match.

    The returned function is designed to input a Nimble data object and
    output True if every value in that object is a match, otherwise
    False.

    Parameters
    ----------
    match : value, iterable, or function
        * value - The value to match in the data.
        * iterable - A list-like container of values to match in the
          data.
        * function - Input a value and return True if that value is a
          match.

    Returns
    -------
    bool

    See Also
    --------
    allMissing, allNumeric, allNonNumeric, allZero, allNonZero,
    allPositive, allNegative

    Examples
    --------
    >>> lst = [[1, 1, 1],
    ...        [1, 1, 1],
    ...        [1, 1, 1]]
    >>> X = nimble.data(lst)
    >>> allOne = allValues(1)
    >>> allOne(X)
    True

    >>> lst = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> X = nimble.data(lst)
    >>> allLetterA = allValues('A')
    >>> allLetterA(X)
    False
    """
    match = _convertMatchToFunction(match)
    def allValueFinder(data):
        return anyAllValuesBackend(all, data, match)
    return allValueFinder

def anyMissing(data):
    """
    Determine if any values in the data are missing.

    Return True if one or more values in the data are considered to be
    missing. Missing values in Nimble are None and (python or numpy) nan
    values.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allMissing, missing

    Examples
    --------
    >>> lst = [[1, 1, 1],
    ...        [1, 1, 1],
    ...        [1, None, 1]]
    >>> X = nimble.data(lst)
    >>> anyMissing(X)
    True

    >>> lst = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> X = nimble.data(lst)
    >>> anyMissing(X)
    False
    """
    return anyAllValuesBackend(any, data, missing)

def allMissing(data):
    """
    Determine if all values in the data are missing.

    Return True if every value in the data is considered to be missing.
    Missing values in Nimble are None and (python or numpy) nan values.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyMissing, missing

    Examples
    --------
    >>> lst = [[float('nan'), None, None],
    ...        [float('nan'), None, None],
    ...        [float('nan'), None, None]]
    >>> X = nimble.data(lst)
    >>> allMissing(X)
    True

    >>> lst = [[float('nan'), None, None],
    ...        [float('nan'), 0, None],
    ...        [float('nan'), None, None]]
    >>> X = nimble.data(lst)
    >>> allMissing(X)
    False
    """
    return anyAllValuesBackend(all, data, missing)

def anyNonMissing(data):
    """
    Determine if any values in the data are not missing.

    Return True if one or more values in the data are not considered to
    be missing. Missing values in Nimble are None and (python or numpy)
    nan values.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allNonMissing, nonMissing

    Examples
    --------
    >>> lst = [[1, 1, None],
    ...        [None, 1, 1],
    ...        [1, None, 1]]
    >>> X = nimble.data(lst)
    >>> anyNonMissing(X)
    True

    >>> lst = [[float('nan'), None, None],
    ...        [float('nan'), None, None],
    ...        [float('nan'), None, None]]
    >>> X = nimble.data(lst)
    >>> anyNonMissing(X)
    False
    """
    return anyAllValuesBackend(any, data, nonMissing)

def allNonMissing(data):
    """
    Determine if all values in the data are missing.

    Return True if every value in the data is considered to be missing.
    Missing values in Nimble are None and (python or numpy) nan values.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyNonMissing, nonMissing

    Examples
    --------
    >>> lst = [[1, 1, 3.0],
    ...        [2, 0, 2.0],
    ...        [3, -1, 1.0]]
    >>> X = nimble.data(lst)
    >>> allNonMissing(X)
    True

    >>> lst = [[1, 1, 3.0],
    ...        [2, float('nan'), 2.0],
    ...        [3, -1, 1.0]]
    >>> X = nimble.data(lst)
    >>> allNonMissing(X)
    False
    """
    return anyAllValuesBackend(all, data, nonMissing)

def anyNumeric(data):
    """
    Determine if any values in the data are numeric.

    Return True if one or more values in the data are numeric. Numeric
    values include any value with a python or numpy numeric type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allNumeric, numeric

    Examples
    --------
    >>> lst = [[1, 'a', None],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> X = nimble.data(lst)
    >>> anyNumeric(X)
    True

    >>> lst = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> X = nimble.data(lst)
    >>> anyNumeric(X)
    False
    """
    return anyAllValuesBackend(any, data, numeric)

def allNumeric(data):
    """
    Determine if all values in the data are numeric.

    Return True if every value in the data is numeric. Numeric values
    include any value with a python or numpy numeric type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyNumeric, numeric

    Examples
    --------
    >>> lst = [[1, 2, 3],
    ...        [4, 5, 6],
    ...        [-1, -2, float('nan')]]
    >>> X = nimble.data(lst)
    >>> allNumeric(X)
    True

    >>> lst = [[1, 2, 3],
    ...        [4, 5, 6],
    ...        [-1, -2, '?']]
    >>> X = nimble.data(lst)
    >>> allNumeric(X)
    False
    """
    return anyAllValuesBackend(all, data, numeric)

def anyNonNumeric(data):
    """
    Determine if any values in the data are non-numeric.

    Return True if one or more values in the data are non-numeric.
    Non-numeric values include any value without a python or numpy
    numeric type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allNonNumeric, nonNumeric

    Examples
    --------
    >>> lst = [[1, 'a', None],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> X = nimble.data(lst)
    >>> anyNonNumeric(X)
    True

    >>> lst = [[1, 2, 3],
    ...        [1, 2, 3],
    ...        [1, 2, 3]]
    >>> X = nimble.data(lst)
    >>> anyNonNumeric(X)
    False
    """
    return anyAllValuesBackend(any, data, nonNumeric)

def allNonNumeric(data):
    """
    Determine if all values in the data are non-numeric.

    Return True if every value in the data is non-numeric. Non-numeric
    values include any value without a python or numpy numeric type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyNonNumeric, nonNumeric

    Examples
    --------
    >>> lst = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> X = nimble.data(lst)
    >>> allNonNumeric(X)
    True

    >>> lst = [[1, 'a', None],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> X = nimble.data(lst)
    >>> allNonNumeric(X)
    False
    """
    return anyAllValuesBackend(all, data, nonNumeric)

def anyZero(data):
    """
    Determine if any values in the data are zero.

    Return True if one or more values in the data are zero. Zero values
    include any numeric value equal to zero.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allZero, zero

    Examples
    --------
    >>> lst = [[1, 'a', 0],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> X = nimble.data(lst)
    >>> anyZero(X)
    True

    >>> lst = [[1, 2, 3],
    ...        [1, 2, 3],
    ...        [1, 2, 3]]
    >>> X = nimble.data(lst)
    >>> anyZero(X)
    False
    """
    return anyAllValuesBackend(any, data, zero)

def allZero(data):
    """
    Determine if all values in the data are zero.

    Return True if every value in the data is zero. Zero values include
    any numeric value equal to zero.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyZero, zero

    Examples
    --------
    >>> lst = [[0, 0.0, 0],
    ...        [0, 0.0, 0],
    ...        [0, 0.0, 0]]
    >>> X = nimble.data(lst)
    >>> allZero(X)
    True

    >>> lst = [[0, 0.0, 0],
    ...        [0, 0.0, 0],
    ...        [0, 0.0, 1]]
    >>> X = nimble.data(lst)
    >>> allZero(X)
    False
    """
    return anyAllValuesBackend(all, data, zero)

def anyNonZero(data):
    """
    Determine if any values in the data are non-zero.

    Return True if one or more values in the data are not zero. Non-zero
    values include any value not equal to zero.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allNonZero, nonZero

    Examples
    --------
    >>> lst = [[1, 'a', 0],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> X = nimble.data(lst)
    >>> anyNonZero(X)
    True

    >>> lst = [[0, 0, 0.0],
    ...        [0, 0, 0.0],
    ...        [0, 0, 0.0]]
    >>> X = nimble.data(lst)
    >>> anyNonZero(X)
    False
    """
    return anyAllValuesBackend(any, data, nonZero)

def allNonZero(data):
    """
    Determine if all values in the data are non-zero.

    Return True if every value in the data is not zero. Non-zero values
    include any value not equal to zero.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyNonZero, nonZero

    Examples
    --------
    >>> lst = [[1, 'a', None],
    ...        [2, 'b', -2],
    ...        [3, 'c', -3.0]]
    >>> X = nimble.data(lst)
    >>> allNonZero(X)
    True

    >>> lst = [[1, 'a', None],
    ...        [2, 'b', -2],
    ...        [3, 'c', 0.0]]
    >>> X = nimble.data(lst)
    >>> allNonZero(X)
    False
    """
    return anyAllValuesBackend(all, data, nonZero)

def anyPositive(data):
    """
    Determine if any values in the data are greater than zero.

    Return True if one or more values in the data are greater than zero.
    Positive values include any numeric value greater than zero.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allPositive, positive

    Examples
    --------
    >>> lst = [[1, 'a', -1],
    ...        [1, 'b', -2],
    ...        [1, 'c', -3]]
    >>> X = nimble.data(lst)
    >>> anyPositive(X)
    True

    >>> lst = [[0, 'a', -1],
    ...        [0, 'b', -2],
    ...        [0, 'c', -3]]
    >>> X = nimble.data(lst)
    >>> anyPositive(X)
    False
    """
    return anyAllValuesBackend(any, data, positive)

def allPositive(data):
    """
    Determine if all values in the data are greater than zero.

    Return True if every value in the data is greater than zero.
    Positive values include any numeric value greater than zero.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyPositive, positive

    Examples
    --------
    >>> lst = [[1, 2, 3],
    ...        [1, 2, 3],
    ...        [1.0, 2.0, 3.0]]
    >>> X = nimble.data(lst)
    >>> allPositive(X)
    True

    >>> lst = [[1, 2, 3],
    ...        [1, 2, 3],
    ...        [0.0, 2.0, 3.0]]
    >>> X = nimble.data(lst)
    >>> allPositive(X)
    False
    """
    return anyAllValuesBackend(all, data, positive)

def anyNegative(data):
    """
    Determine if any values in the data are less than zero.

    Return True if one or more values in the data are less than zero.
    Negative values include any numeric value less than zero.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allNegative, negative

    Examples
    --------
    >>> lst = [[0, 'a', -1],
    ...        [1, 'b', -2],
    ...        [2, 'c', -3]]
    >>> X = nimble.data(lst)
    >>> anyNegative(X)
    True

    >>> lst = [[1, 'a', 0],
    ...        [1, 'b', None],
    ...        [1, 'c', 3]]
    >>> X = nimble.data(lst)
    >>> anyNegative(X)
    False
    """
    return anyAllValuesBackend(any, data, negative)

def allNegative(data):
    """
    Determine if all values in the data are less than zero.

    Return True if every value in the data is less than zero. Negative
    values include any numeric value less than zero.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyNegative, negative

    Examples
    --------
    >>> lst = [[-1, -2, -3],
    ...        [-1, -2, -3],
    ...        [-1.0, -2.0, -3.0]]
    >>> X = nimble.data(lst)
    >>> allNegative(X)
    True

    >>> lst = [[-1, -2, -3],
    ...        [-1, -2, -3],
    ...        [0.0, -2.0, -3.0]]
    >>> X = nimble.data(lst)
    >>> allNegative(X)
    False
    """
    return anyAllValuesBackend(all, data, negative)

def anyInfinity(data):
    """
    Determine if any values in the data are infinity.

    Return True if one or more value in the data is a positive or
    negative infinity float value.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allInfinity, infinity

    Examples
    --------
    >>> lst = [[0, -1.0, float('inf')],
    ...        [1, -2.0, 0],
    ...        [2, -3.0, 0]]
    >>> X = nimble.data(lst)
    >>> anyInfinity(X)
    True

    >>> lst = [[0, -1.0, 'infinity'],
    ...        [1, -2.0, 'zero'],
    ...        [2, -3.0, 'one']]
    >>> X = nimble.data(lst)
    >>> anyInfinity(X)
    False
    """
    return anyAllValuesBackend(any, data, infinity)

def allInfinity(data):
    """
    Determine if all values in the data are infinity.

    Return True if every value in the data is a positive or negative
    infinity float value.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyInfinity, infinity

    Examples
    --------
    >>> lst = [[float('inf'), float('inf'), float('inf')],
    ...        [float('inf'), float('inf'), float('inf')],
    ...        [float('inf'), float('inf'), float('inf')]]
    >>> X = nimble.data(lst)
    >>> allInfinity(X)
    True

    >>> lst = [[float('inf'), float('inf'), float('inf')],
    ...        [float('inf'), 0, float('inf')],
    ...        [float('inf'), float('inf'), float('inf')]]
    >>> X = nimble.data(lst)
    >>> allInfinity(X)
    False
    """
    return anyAllValuesBackend(all, data, infinity)

def anyBoolean(data):
    """
    Determine if any values in the data are a boolean type.

    Return True if one or more value in the data is a boolean type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allBoolean, boolean

    Examples
    --------
    >>> lst = [[0, -1.0, True],
    ...        [1, -2.0, False],
    ...        [2, -3.0, False]]
    >>> X = nimble.data(lst)
    >>> anyBoolean(X)
    True

    >>> lst = [[0, -1.0, 'a'],
    ...        [1, -2.0, 'b'],
    ...        [2, -3.0, 'c']]
    >>> X = nimble.data(lst)
    >>> anyBoolean(X) # Note: 0 and 1 are not boolean type
    False
    """
    return anyAllValuesBackend(any, data, boolean)

def allBoolean(data):
    """
    Determine if all values in the data are a boolean type.

    Return True if every value in the data is boolean type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyBoolean, boolean

    Examples
    --------
    >>> lst = [[True, True, False],
    ...        [True, True, True],
    ...        [False, True, True]]
    >>> X = nimble.data(lst)
    >>> anyBoolean(X)
    True

    >>> lst = [[0, True, False],
    ...        [1, True, True],
    ...        [0, True, True]]
    >>> X = nimble.data(lst)
    >>> allBoolean(X) # Note: 0 and 1 are not boolean type
    False
    """
    return anyAllValuesBackend(all, data, boolean)

def anyInteger(data):
    """
    Determine if any values in the data are an integer type.

    Return True if one or more value in the data is an integer type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allInteger, integer

    Examples
    --------
    >>> lst = [[0, -1.0, 'a'],
    ...        [1, -2.0, 'b'],
    ...        [2, -3.0, 'c']]
    >>> X = nimble.data(lst)
    >>> anyInteger(X)
    True

    >>> lst = [['a', -1.0, None],
    ...        ['b', -2.0, None],
    ...        ['c', -3.0, None]]
    >>> X = nimble.data(lst)
    >>> anyInteger(X)
    False
    """
    return anyAllValuesBackend(any, data, integer)

def allInteger(data):
    """
    Determine if all values in the data are an integer type.

    Return True if every value in the data is an integer type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyInteger, integer

    Examples
    --------
    >>> lst = [[1, 9, True],
    ...        [2, 8, False],
    ...        [3, 7, True]]
    >>> X = nimble.data(lst)
    >>> allInteger(X)
    True

    >>> lst = [[1, 9, -1.1],
    ...        [2, 8, -2.2],
    ...        [3, 7, -3.3]]
    >>> X = nimble.data(lst)
    >>> allInteger(X)
    False
    """
    return anyAllValuesBackend(all, data, integer)

def anyFloating(data):
    """
    Determine if any values in the data are a float type.

    Return True if one or more value in the data is a float type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allFloating, floating

    Examples
    --------
    >>> lst = [[0, -1.0, True],
    ...        [1, -2.0, False],
    ...        [2, -3.0, False]]
    >>> X = nimble.data(lst)
    >>> anyFloating(X)
    True

    >>> lst = [['a', -1, True],
    ...        ['b', -2, False],
    ...        ['c', -3, False]]
    >>> X = nimble.data(lst)
    >>> anyFloating(X)
    False
    """
    return anyAllValuesBackend(any, data, floating)

def allFloating(data):
    """
    Determine if all values in the data are a float type.

    Return True if every value in the data is a float type.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyFloating, floating

    Examples
    --------
    >>> lst = [[1.1, 9.0, 0.1],
    ...        [2.2, 8.0, 0.2],
    ...        [3.3, 7.0, 0.3]]
    >>> X = nimble.data(lst)
    >>> allFloating(X)
    True

    >>> lst = [[1.1, 9.0, 'a'],
    ...        [2.2, 8.0, 'b'],
    ...        [3.3, 7.0, 'c']]
    >>> X = nimble.data(lst)
    >>> allFloating(X)
    False
    """
    return anyAllValuesBackend(all, data, floating)

def anyTrue(data):
    """
    Determine if any values in the data are a boolean True.

    Return True if one or more value in the data is boolean True.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allTrue, true

    Examples
    --------
    >>> lst = [[0, -1.0, True],
    ...        [1, -2.0, False],
    ...        [2, -3.0, False]]
    >>> X = nimble.data(lst)
    >>> anyTrue(X)
    True

    >>> lst = [[0, -1.0, False],
    ...        [1, -2.0, False],
    ...        [2, -3.0, False]]
    >>> X = nimble.data(lst)
    >>> anyTrue(X) # Note: 1 is not boolean type
    False
    """
    return anyAllValuesBackend(any, data, true)

def allTrue(data):
    """
    Determine if all values in the data are a boolean True.

    Return True if every value in the data is boolean True.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyTrue, true

    Examples
    --------
    >>> lst = [[True, True, True],
    ...        [True, True, True],
    ...        [True, True, True]]
    >>> X = nimble.data(lst)
    >>> allTrue(X)
    True

    >>> lst = [[True, True, True],
    ...        [True, False, True],
    ...        [True, True, True]]
    >>> X = nimble.data(lst)
    >>> allTrue(X)
    False
    """
    return anyAllValuesBackend(all, data, true)

def anyFalse(data):
    """
    Determine if any values in the data are a boolean False.

    Return True if one or more value in the data is boolean False.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    allFalse, false

    Examples
    --------
    >>> lst = [[0, -1.0, True],
    ...        [1, -2.0, True],
    ...        [2, -3.0, False]]
    >>> X = nimble.data(lst)
    >>> anyFalse(X)
    True

    >>> lst = [[0, -1.0, True],
    ...        [1, -2.0, True],
    ...        [2, -3.0, True]]
    >>> X = nimble.data(lst)
    >>> anyFalse(X) # Note: 0 is not boolean type
    False
    """
    return anyAllValuesBackend(any, data, false)

def allFalse(data):
    """
    Determine if all values in the data are a boolean False.

    Return True if every value in the data is boolean False.

    Parameters
    ----------
    data : Nimble Base object

    Returns
    -------
    bool

    See Also
    --------
    anyFalse, false

    Examples
    --------
    >>> lst = [[False, False, False],
    ...        [False, False, False],
    ...        [False, False, False]]
    >>> X = nimble.data(lst)
    >>> allFalse(X)
    True

    >>> lst = [[False, False, True],
    ...        [False, False, True],
    ...        [False, True, False]]
    >>> X = nimble.data(lst)
    >>> allFalse(X)
    False
    """
    return anyAllValuesBackend(all, data, false)

def anyAllValuesBackend(quantity, data, match):
    """
    Backend for functions using any and all.

    Capable of handling both 1-dimensional and 2-dimensional data.
    Returns True if every value in the data satisfies the ``quantity``
    function given the ``match`` function. The ``quantity`` parameter
    may only be python's built-in functions any or all.
    """
    try:
        # 1D data
        return quantity([match(val) for val in data])
    except TypeError:
        # 2D data
        if quantity is any:
            # if any feature contains a match we can return True
            for i in range(len(data.features)):
                if quantity([match(val) for val in data[:, i]]):
                    return True
            return False
        # if any feature does not have all matches we can return False
        for i in range(len(data.features)):
            if not quantity([match(val) for val in data[:, i]]):
                return False
        return True

def _convertMatchToFunction(match):
    """
    Factory for functions determing if a value is a match.

    If ``match`` is not callable, it will be converted to a function
    which outputs True if the input value is equal to ``match`` or in
    ``match`` (dependent on its type).  Callable ``match`` parameters
    will be returned unchanged, this serves as a helper when the type of
    input for match is unknown.

    Parameters
    ----------
    match : value, list of values or function
        * value - A value to test equality with other values.
        * list of values - A list of values to test equality with other
          values. Another value is equal if it equal to any of the
          values in the list.
        * function - Input a value and return True if that value is a
          match, otherwise False.

    Returns
    -------
    function

    Examples
    --------
    >>> matchA = _convertMatchToFunction('A')
    >>> matchA('A')
    True
    >>> matchA('B')
    False

    >>> matchList = _convertMatchToFunction([1, 'A'])
    >>> matchList(1)
    True
    >>> matchList('A')
    True
    >>> matchList(2)
    False
    >>> matchList('B')
    False
    """
    if not callable(match):
        # case1: list-like
        if ((hasattr(match, '__iter__') or hasattr(match, '__getitem__'))
                and not isinstance(match, str)):
            matchList = match
            # if nans in the list, need to include separate check in function
            if not all(val == val for val in matchList):
                match = lambda x: x != x or x in matchList
            else:
                match = lambda x: x in matchList
        # case2: constant
        else:
            matchConstant = match
            if matchConstant is None:
                match = lambda x: x is None
            elif matchConstant != matchConstant:
                match = lambda x: x != x
            else:
                match = lambda x: x == matchConstant
    return match
