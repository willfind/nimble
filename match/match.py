"""
Variety of functions to determine if data matches given conditions
"""
from __future__ import absolute_import

import numpy
import six

from UML.exceptions import ArgumentException

def missing(value):
    """
    Determine if a value is missing.

    Return True if UML considers the value to be missing. Missing values
    in UML are None and (python or numpy) nan values.

    Parameters
    ----------
    value

    Returns
    -------
    bool

    See Also
    --------
    anyMissing, allMissing

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
    >>> numeric(numpy.float32(8))
    True
    >>> numeric(float('nan'))
    True
    >>> numeric('abc')
    False
    >>> numeric(None)
    False
    """
    return isinstance(value, (int, float, complex, numpy.number))

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
    True
    """
    return not isinstance(value, (int, float, complex, numpy.number))

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
        # py2 inteprets strings as greater than 0, trying to add zero
        # will raise exception for strings
        value + 0
        return value > 0
    except Exception:
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
        # py2 inteprets None as less than 0, trying to add zero will
        # raise exception for None
        value + 0
        return value < 0
    except Exception:
        return False

def anyValues(match):
    """
    Factory for functions which will determine if any values match.

    The returned function is designed to input a UML data object and
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
    >>> raw = [[1, 1, 1],
    ...        [1, 1, 1],
    ...        [1, 1, -1]]
    >>> data = UML.createData('Matrix', raw)
    >>> anyNegativeOne = anyValues(-1)
    >>> anyNegativeOne(data)
    True

    >>> raw = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> data = UML.createData('List', raw)
    >>> anyLetterD = anyValues('D')
    >>> anyLetterD(data)
    False
    """
    match = convertMatchToFunction(match)
    def anyValueFinder(data):
        return anyAllValuesBackend(any, data, match)
    return anyValueFinder

def allValues(match):
    """
    Factory for functions which will determine if all values match.

    The returned function is designed to input a UML data object and
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
    >>> raw = [[1, 1, 1],
    ...        [1, 1, 1],
    ...        [1, 1, 1]]
    >>> data = UML.createData('Matrix', raw)
    >>> allOne = allValues(1)
    >>> allOne(data)
    True

    >>> raw = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> data = UML.createData('List', raw)
    >>> allLetterA = allValues('A')
    >>> allLetterA(data)
    False
    """
    match = convertMatchToFunction(match)
    def allValueFinder(data):
        return anyAllValuesBackend(all, data, match)
    return allValueFinder

def anyMissing(data):
    """
    Determine if any values in the data are missing.

    Return True if one or more values in the data are considered to be
    missing. Missing values in UML are None and (python or numpy) nan
    values.

    Parameters
    ----------
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    allMissing, missing

    Examples
    --------
    >>> raw = [[1, 1, 1],
    ...        [1, 1, 1],
    ...        [1, None, 1]]
    >>> data = UML.createData('Matrix', raw)
    >>> anyMissing(data)
    True

    >>> raw = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> data = UML.createData('List', raw)
    >>> anyMissing(data)
    False
    """
    return anyAllValuesBackend(any, data, missing)

def allMissing(data):
    """
    Determine if all values in the data are missing.

    Return True if every value in the data is considered to be missing.
    Missing values in UML are None and (python or numpy) nan values.

    Parameters
    ----------
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    anyMissing, missing

    Examples
    --------
    >>> raw = [[float('nan'), None, None],
    ...        [float('nan'), None, None],
    ...        [float('nan'), None, None]]
    >>> data = UML.createData('Matrix', raw)
    >>> allMissing(data)
    True

    >>> raw = [[float('nan'), None, None],
    ...        [float('nan'), 0, None],
    ...        [float('nan'), None, None]]
    >>> data = UML.createData('List', raw)
    >>> allMissing(data)
    False
    """
    return anyAllValuesBackend(all, data, missing)

def anyNumeric(data):
    """
    Determine if any values in the data are numeric.

    Return True if one or more values in the data are numeric. Numeric
    values include any value with a python or numpy numeric type.

    Parameters
    ----------
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    allNumeric, numeric

    Examples
    --------
    >>> raw = [[1, 'a', None],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> data = UML.createData('Matrix', raw)
    >>> anyNumeric(data)
    True

    >>> raw = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> data = UML.createData('List', raw)
    >>> anyNumeric(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    anyNumeric, numeric

    Examples
    --------
    >>> raw = [[1, 2, 3],
    ...        [4, 5, 6],
    ...        [-1, -2, float('nan')]]
    >>> data = UML.createData('Matrix', raw)
    >>> allNumeric(data)
    True

    >>> raw = [[1, 2, 3],
    ...        [4, 5, 6],
    ...        [-1, -2, '?']]
    >>> data = UML.createData('List', raw)
    >>> allNumeric(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    allNonNumeric, nonNumeric

    Examples
    --------
    >>> raw = [[1, 'a', None],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> data = UML.createData('Matrix', raw)
    >>> anyNonNumeric(data)
    True

    >>> raw = [[1, 2, 3],
    ...        [1, 2, 3],
    ...        [1, 2, 3]]
    >>> data = UML.createData('List', raw)
    >>> anyNonNumeric(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    anyNonNumeric, nonNumeric

    Examples
    --------
    >>> raw = [['A', 'B', 'C'],
    ...        ['A', 'B', 'C'],
    ...        ['A', 'B', 'C']]
    >>> data = UML.createData('Matrix', raw)
    >>> allNonNumeric(data)
    True

    >>> raw = [[1, 'a', None],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> data = UML.createData('List', raw)
    >>> allNonNumeric(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    allZero, zero

    Examples
    --------
    >>> raw = [[1, 'a', 0],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> data = UML.createData('Matrix', raw)
    >>> anyZero(data)
    True

    >>> raw = [[1, 2, 3],
    ...        [1, 2, 3],
    ...        [1, 2, 3]]
    >>> data = UML.createData('List', raw)
    >>> anyZero(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    anyZero, zero

    Examples
    --------
    >>> raw = [[0, 0.0, 0],
    ...        [0, 0.0, 0],
    ...        [0, 0.0, 0]]
    >>> data = UML.createData('Matrix', raw)
    >>> allZero(data)
    True

    >>> raw = [[0, 0.0, 0],
    ...        [0, 0.0, 0],
    ...        [0, 0.0, 1]]
    >>> data = UML.createData('List', raw)
    >>> allZero(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    allNonZero, nonZero

    Examples
    --------
    >>> raw = [[1, 'a', 0],
    ...        [1, 'b', 3],
    ...        [1, 'c', None]]
    >>> data = UML.createData('Matrix', raw)
    >>> anyNonZero(data)
    True

    >>> raw = [[1, 2, '0'],
    ...        [1, 2, '1'],
    ...        [1, 2, '2']]
    >>> data = UML.createData('List', raw)
    >>> anyNonZero(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    anyNonZero, nonZero

    Examples
    --------
    >>> raw = [[1, 'a', None],
    ...        [2, 'b', -2],
    ...        [3, 'c', -3.0]]
    >>> data = UML.createData('Matrix', raw)
    >>> allNonZero(data)
    True

    >>> raw = [[1, 'a', None],
    ...        [2, 'b', -2],
    ...        [3, 'c', 0.0]]
    >>> data = UML.createData('List', raw)
    >>> allNonZero(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    allPositive, positive

    Examples
    --------
    >>> raw = [[1, 'a', -1],
    ...        [1, 'b', -2],
    ...        [1, 'c', -3]]
    >>> data = UML.createData('Matrix', raw)
    >>> anyPositive(data)
    True

    >>> raw = [[0, 'a', -1],
    ...        [0, 'b', -2],
    ...        [0, 'c', -3]]
    >>> data = UML.createData('List', raw)
    >>> anyPositive(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    anyPositive, positive

    Examples
    --------
    >>> raw = [[1, 2, 3],
    ...        [1, 2, 3],
    ...        [1.0, 2.0, 3.0]]
    >>> data = UML.createData('Matrix', raw)
    >>> allPositive(data)
    True

    >>> raw = [[1, 2, 3],
    ...        [1, 2, 3],
    ...        [0.0, 2.0, 3.0]]
    >>> data = UML.createData('List', raw)
    >>> allPositive(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    allNegative, negative

    Examples
    --------
    >>> raw = [[0, 'a', -1],
    ...        [1, 'b', -2],
    ...        [2, 'c', -3]]
    >>> data = UML.createData('Matrix', raw)
    >>> anyNegative(data)
    True

    >>> raw = [[1, 'a', 0],
    ...        [1, 'b', None],
    ...        [1, 'c', 3]]
    >>> data = UML.createData('List', raw)
    >>> anyNegative(data)
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
    data : UML object

    Returns
    -------
    bool

    See Also
    --------
    anyNegative, negative

    Examples
    --------
    >>> raw = [[-1, -2, -3],
    ...        [-1, -2, -3],
    ...        [-1.0, -2.0, -3.0]]
    >>> data = UML.createData('Matrix', raw)
    >>> allNegative(data)
    True

    >>> raw = [[-1, -2, -3],
    ...        [-1, -2, -3],
    ...        [0.0, -2.0, -3.0]]
    >>> data = UML.createData('List', raw)
    >>> allNegative(data)
    False
    """
    return anyAllValuesBackend(all, data, negative)

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
    >>> matchA = convertMatchToFunction('A')
    >>> matchA('A')
    True
    >>> matchA('B')
    False

    >>> matchList = convertMatchToFunction([1, 'A'])
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
            if matchConstant is None:
                match = lambda x: x is None
            elif matchConstant != matchConstant:
                match = lambda x: x != x
            else:
                match = lambda x: x == matchConstant
    return match
