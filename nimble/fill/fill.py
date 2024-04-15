
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
Variety of functions to replace values in data with other values
"""

import numpy as np

import nimble
from nimble.match import _convertMatchToFunction
from nimble.exceptions import InvalidArgumentValue

def _booleanElementMatch(vector, match):
    if not isinstance(match, nimble.core.data.Base):
        match = _convertMatchToFunction(match)
        return vector.matchingElements(match, useLog=False)
    return match

def constant(vector, match, constantValue):
    """
    Fill matched values with a constant value.

    Parameters
    ----------
    vector : nimble point or feature
        A nimble Base object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. Nimble offers common use-case functions in its
          match module.
    constantValue : value
        The value which will replace any matching values.

    Returns
    -------
    list
        The vector values with the ``constantValue`` replacing the
        ``match`` values.

    See Also
    --------
    nimble.match

    Examples
    --------
    Match a value.

    >>> lst = [1, 'na', 3, 'na', 5]
    >>> X = nimble.data(lst)
    >>> constant(X, 'na', 0)
    <DataFrame 1pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 1  0  3  0  5
    >

    Match using a function from nimble's match module.

    >>> from nimble import match
    >>> lst = [1, 0, 3, 0, 5]
    >>> X = nimble.data(lst)
    >>> constant(X, match.zero, 99)
    <Matrix 1pt x 5ft
         0  1   2  3   4
       ┌────────────────
     0 │ 1  99  3  99  5
    >
    """
    toFill = _booleanElementMatch(vector, match)

    def filler(vec):
        return [constantValue if fill else val
                for val, fill in zip(vec, toFill)]

    if len(vector.points) == 1:
        return vector.points.calculate(filler, useLog=False)
    return vector.features.calculate(filler, useLog=False)

def mean(vector, match):
    """
    Fill matched values with the mean.

    The calculation of the mean will ignore any matched values, but all
    unmatched values must be numeric. If all the values are a match the
    mean cannot be calculated.

    Parameters
    ----------
    vector : nimble point or feature
        A nimble Base object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. Nimble offers common use-case functions in its
          match module.

    Returns
    -------
    list
        The vector values with the mean replacing the ``match`` values.

    See Also
    --------
    median, mode, nimble.match

    Examples
    --------
    Match a value.

    >>> lst = [1, 'na', 3, 'na', 5]
    >>> X = nimble.data(lst)
    >>> mean(X, 'na')
    <DataFrame 1pt x 5ft
         0    1    2    3    4
       ┌──────────────────────
     0 │ 1  3.000  3  3.000  5
    >

    Match using a function from nimble's match module.

    >>> from nimble import match
    >>> lst = [6, 0, 2, 0, 4]
    >>> X = nimble.data(lst)
    >>> mean(X, match.zero)
    <Matrix 1pt x 5ft
           0      1      2      3      4
       ┌──────────────────────────────────
     0 │ 6.000  4.000  2.000  4.000  4.000
    >
    """
    return statsBackend(vector, match, 'mean', nimble.calculate.mean)

def median(vector, match):
    """
    Fill matched values with the median.

    The calculation of the median will ignore any matched values, but
    all unmatched values must be numeric. If all the values are a match
    the median cannot be calculated.

    Parameters
    ----------
    vector : nimble point or feature
        A nimble Base object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. Nimble offers common use-case functions in its
          match module.

    Returns
    -------
    list
        The vector values with the median replacing the ``match``
        values.

    See Also
    --------
    mean, mode, nimble.match

    Examples
    --------
    Match a value.

    >>> lst = [1, 'na', 3, 'na', 5]
    >>> X = nimble.data(lst)
    >>> median(X, 'na')
    <DataFrame 1pt x 5ft
         0    1    2    3    4
       ┌──────────────────────
     0 │ 1  3.000  3  3.000  5
    >

    Match using a function from nimble's match module.

    >>> from nimble import match
    >>> lst = [6, 0, 2, 0, 4]
    >>> X = nimble.data(lst)
    >>> median(X, match.zero)
    <Matrix 1pt x 5ft
           0      1      2      3      4
       ┌──────────────────────────────────
     0 │ 6.000  4.000  2.000  4.000  4.000
    >
    """
    return statsBackend(vector, match, 'median', nimble.calculate.median)

def mode(vector, match):
    """
    Fill matched values with the mode.

    The calculation of the mode will ignore any matched values. If all
    the values are a match the mean cannot be calculated.

    Parameters
    ----------
    vector : nimble point or feature
        A nimble Base object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. Nimble offers common use-case functions in its
          match module.

    Returns
    -------
    list
        The vector values with the mode replacing the ``match`` values.

    See Also
    --------
    mean, median, nimble.match

    Examples
    --------
    Match a value.

    >>> lst = [1, 'na', 1, 'na', 5]
    >>> X = nimble.data(lst)
    >>> mode(X, 'na')
    <DataFrame 1pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 1  1  1  1  5
    >

    Match using a function from nimble's match module.

    >>> from nimble import match
    >>> lst = [6, 6, 2, 0, 0]
    >>> X = nimble.data(lst)
    >>> mode(X, match.zero)
    <Matrix 1pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 6  6  2  6  6
    >
    """
    return statsBackend(vector, match, 'mode', nimble.calculate.mode)

def forwardFill(vector, match):
    """
    Fill matched values with the previous unmatched value.

    Each matching value will be filled with the first non-matching value
    occurring prior to the matched value. An exception will be raised if
    the first value is a match, since there is not a valid value to
    reference.

    Parameters
    ----------
    vector : nimble point or feature
       A nimble Base object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. Nimble offers common use-case functions in its
          match module.

    Returns
    -------
    list
        The vector values with the forward filled values replacing the
        ``match`` values.

    See Also
    --------
    backwardFill, nimble.match

    Examples
    --------
    Match a value.

    >>> lst = [1, 'na', 3, 'na', 5]
    >>> X = nimble.data(lst)
    >>> forwardFill(X, 'na')
    <DataFrame 1pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 1  1  3  3  5
    >

    Match using a function from nimble's match module.

    >>> from nimble import match
    >>> lst = [6, 0, 2, 0, 4]
    >>> X = nimble.data(lst)
    >>> forwardFill(X, match.zero)
    <Matrix 1pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 6  6  2  2  4
    >
    """
    toFill = _booleanElementMatch(vector, match)
    if toFill[0]:
        msg = directionError('forward fill', vector, 'first')
        raise InvalidArgumentValue(msg)

    def filler(vec):
        ret = []
        for val, fill in zip(vec, toFill):
            if fill:
                ret.append(ret[-1])
            else:
                ret.append(val)
        return ret

    if len(vector.points) == 1:
        return vector.points.calculate(filler, useLog=False)
    return vector.features.calculate(filler, useLog=False)

def backwardFill(vector, match):
    """
    Fill matched values with the next unmatched value.

    Each matching value will be filled with the first non-matching value
    occurring after to the matched value. An exception will be raised if
    the last value is a match, since there is not a valid value to
    reference.

    Parameters
    ----------
    vector : nimble point or feature
        A nimble Base object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. Nimble offers common use-case functions in its
          match module.

    Returns
    -------
    list
        The vector values with the backward filled values replacing the
        ``match`` values.

    See Also
    --------
    forwardFill, nimble.match

    Examples
    --------
    Match a value.

    >>> lst = [1, 'na', 3, 'na', 5]
    >>> X = nimble.data(lst)
    >>> backwardFill(X, 'na')
    <DataFrame 1pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 1  3  3  5  5
    >

    Match using a function from nimble's match module.

    >>> from nimble import match
    >>> lst = [6, 0, 2, 0, 4]
    >>> X = nimble.data(lst)
    >>> backwardFill(X, match.zero)
    <Matrix 1pt x 5ft
         0  1  2  3  4
       ┌──────────────
     0 │ 6  2  2  4  4
    >
    """
    toFill = _booleanElementMatch(vector, match)
    if toFill[-1]:
        msg = directionError('backward fill', vector, 'last')
        raise InvalidArgumentValue(msg)

    def filler(vec):
        ret = np.empty_like(vector, dtype=np.object_)
        numValues = len(vec)
        # pylint: disable=unsupported-assignment-operation
        for i, (val, fill) in enumerate(zip(reversed(vector),
                                            reversed(toFill))):
            idx = numValues - i - 1
            if fill:
                ret[idx] = ret[idx + 1]
            else:
                ret[idx] = val
        return ret

    if len(vector.points) == 1:
        return vector.points.calculate(filler, useLog=False)
    return vector.features.calculate(filler, useLog=False)

def interpolate(vector, match, **kwarguments):
    """
    Fill matched values with the interpolated value.

    The fill value is determined by the piecewise linear interpolant
    returned by numpy.interp. By default, the unmatched values will be
    used as the discrete data points, but additional arguments for
    numpy.interp can be passed as keyword arguments.

    Parameters
    ----------
    vector : nimble point or feature
        A nimble Base object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. Nimble offers common use-case functions in its
          match module.
    kwarguments
        Collection of extra key:value argument pairs to pass to
        numpy.interp.

    Returns
    -------
    list
        The vector values with the interpolated values replacing the
        ``match`` values.

    See Also
    --------
    nimble.match, numpy.interp

    Examples
    --------
    Match a value.

    >>> lst = [1, 'na', 3, 'na', 5]
    >>> X = nimble.data(lst)
    >>> interpolate(X, 'na')
    <DataFrame 1pt x 5ft
         0    1    2    3    4
       ┌──────────────────────
     0 │ 1  2.000  3  4.000  5
    >

    Match using a function from nimble's match module.

    >>> from nimble import match
    >>> lst = [6, 0, 4, 0, 2]
    >>> X = nimble.data(lst)
    >>> interpolate(X, match.zero)
    <Matrix 1pt x 5ft
           0      1      2      3      4
       ┌──────────────────────────────────
     0 │ 6.000  5.000  4.000  3.000  2.000
    >
    """
    toFill = _booleanElementMatch(vector, match)
    if 'x' in kwarguments:
        msg = "'x' is a disallowed keyword argument because it is "
        msg += "determined by the data in the vector."
        raise TypeError(msg)
    matchedLoc = []
    unmatchedLoc = []
    unmatchedVals = []
    for i, (val, fill) in enumerate(zip(vector, toFill)):
        if fill:
            matchedLoc.append(i)
        else:
            unmatchedLoc.append(i)
            unmatchedVals.append(val)

    kwarguments['x'] = matchedLoc
    if 'xp' not in kwarguments:
        kwarguments['xp'] = unmatchedLoc
    if 'fp' not in kwarguments:
        kwarguments['fp'] = unmatchedVals

    tmpV = np.interp(**kwarguments)

    def filler(vec):
        ret = []
        j = 0
        for i, val in enumerate(vec):
            if i in matchedLoc:
                ret.append(tmpV[j])
                j += 1
            else:
                ret.append(val)
        return ret

    if len(vector.points) == 1:
        return vector.points.calculate(filler, useLog=False)
    return vector.features.calculate(filler, useLog=False)

############
# Backends #
############

def statsBackend(vector, match, funcString, statisticsFunction):
    """
    Backend for filling with a statistics function from nimble.calculate.
    """
    toFill = _booleanElementMatch(vector, match)

    def toStat(vec):
        return [val for val, fill in zip(vec, toFill) if not fill]

    if len(vector.points) == 1:
        unmatched = vector.points.calculate(toStat, useLog=False)
    else:
        unmatched = vector.features.calculate(toStat, useLog=False)

    if len(unmatched) == len(vector):
        return vector
    if not unmatched:
        msg = statsExceptionNoMatches(funcString, vector)
        raise InvalidArgumentValue(msg)

    stat = statisticsFunction(unmatched)
    if stat != stat:
        msg = statsExceptionInvalidInput(funcString, vector)
        raise InvalidArgumentValue(msg)

    return constant(vector, match, stat)

###########
# Helpers #
###########

def getAxis(vector):
    """
    Helper function to determine if the vector is a point or feature.
    """
    return 'point' if vector.points == 1 else 'feature'

def getNameAndIndex(axis, vector):
    """
    Helper function to find the name and index of the vector.
    """
    name = None
    index = 0
    if axis == 'point':
        if vector.points._namesCreated():
            name = vector.points.getName(0)
        if isinstance(vector, nimble.core.data.BaseView):
            index = vector._pStart
    else:
        if vector.features._namesCreated():
            name = vector.features.getName(0)
        if isinstance(vector, nimble.core.data.BaseView):
            index = vector._fStart

    return name, index

def getLocationMsg(name, index):
    """
    Helper function to format the error message with either a name or index.
    """
    if name is not None:
        location = f"'{name}'"
    else:
        location = f"at index '{index}'"

    return location

def errorMsgFormatter(msg, vector, **kwargs):
    """
    Generic function to format error messages.
    """
    axis = getAxis(vector)
    name, index = getNameAndIndex(axis, vector)
    location = getLocationMsg(name, index)

    return msg.format(axis=axis, location=location, **kwargs)

def statsExceptionNoMatches(funcString, vector):
    """
    Generic message when the statistics function recieves no values.
    """
    msg = "Cannot calculate {funcString}. The {funcString} is calculated "
    msg += "using only unmatched values. All values for the {axis} {location} "
    msg += "returned a match."

    return errorMsgFormatter(msg, vector, **{'funcString':funcString})

def statsExceptionInvalidInput(funcString, vector):
    """
    Generic message when statistics functions are given invalid data.
    """
    msg = "Cannot calculate {funcString}. The {axis} {location} "
    msg += "contains non-numeric values or is all NaN values"

    return errorMsgFormatter(msg, vector, **{'funcString':funcString})

def directionError(funcString, vector, target):
    """
    Generic message for directional fill with a matched inital value.
    """
    msg = "Unable to provide a {funcString} value for the {axis} {location} "
    msg += "because the {target} value is a match"

    return errorMsgFormatter(msg, vector, **{'funcString':funcString,
                                             'target':target})
