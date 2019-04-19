"""
Variety of functions to replace values in data with other values
"""
from __future__ import absolute_import

import numpy

import UML
from UML.match import convertMatchToFunction
from UML.match import anyValues
from UML.exceptions import InvalidArgumentValue

def factory(match, fill, **kwarguments):
    """
    Return a function for modifying a point or feature.

    The returned function accepts a point or feature and returns the
    modified point or feature as a list.  The modifications occur to any
    value in the point or feature that return True for the ``match``
    parameter and the new value is determined based on the ``fill``
    parameter.

    Parameters
    ----------
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.
    fill : value or function
        * value - The value which will replace any matching values.
        * function - Input a value and return the value which will
          replace the input value. UML offers common use-case functions
          in this module.
    kwarguments
        Collection of extra key:value argument pairs to pass to
        fill function.

    Returns
    -------
    function

    See Also
    --------
    UML.match

    Examples
    --------
    Match a value and fill with a different value.

    >>> raw = [1, 'na', 3, 'na', 5]
    >>> data = UML.createData('Matrix', raw)
    >>> transform = factory('na', 0)
    >>> transform(data)
    [1, 0, 3, 0, 5]

    Match using a function from UML's match module and fill using
    another function in this module.

    >>> from UML import match
    >>> raw = [1, 0, 3, 0, 5]
    >>> data = UML.createData('Matrix', raw)
    >>> transform = factory(match.zero, backwardFill)
    >>> transform(data)
    [1.0, 3.0, 3.0, 5.0, 5.0]
    """
    match = convertMatchToFunction(match)
    if not hasattr(fill, '__call__'):
        value = fill
        # for consistency use numpy.nan for None and nans
        if value is None or value != value:
            value = numpy.nan
        fill = constant
        kwarguments['constantValue'] = value
    if kwarguments:
        def fillFunction(vector):
            return fill(vector, match, **kwarguments)
    else:
        def fillFunction(vector):
            return fill(vector, match)

    return fillFunction

def constant(vector, match, constantValue):
    """
    Fill matched values with a constant value

    Parameters
    ----------
    vector : UML point or feature
        A UML object or UML view object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.
    constantValue : value
        The value which will replace any matching values.

    Returns
    -------
    list
        The vector values with the ``constantValue`` replacing the
        ``match`` values.

    See Also
    --------
    UML.match

    Examples
    --------
    Match a value.

    >>> raw = [1, 'na', 3, 'na', 5]
    >>> data = UML.createData('Matrix', raw)
    >>> constant(data, 'na', 0)
    [1, 0, 3, 0, 5]

    Match using a function from UML's match module.

    >>> from UML import match
    >>> raw = [1, 0, 3, 0, 5]
    >>> data = UML.createData('Matrix', raw)
    >>> constant(data, match.zero, 99)
    [1.0, 99, 3.0, 99, 5.0]
    """
    match = convertMatchToFunction(match)
    return [constantValue if match(val) else val for val in vector]

def mean(vector, match):
    """
    Fill matched values with the mean.

    The calculation of the mean will ignore any matched values, but all
    unmatched values must be numeric. If all the values are a match the
    mean cannot be calculated.

    Parameters
    ----------
    vector : UML point or feature
        A UML object or UML view object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.

    Returns
    -------
    list
        The vector values with the mean replacing the ``match`` values.

    See Also
    --------
    median, mode, UML.match

    Examples
    --------
    Match a value.

    >>> raw = [1, 'na', 3, 'na', 5]
    >>> data = UML.createData('Matrix', raw)
    >>> mean(data, 'na')
    [1, 3.0, 3, 3.0, 5]

    Match using a function from UML's match module.

    >>> from UML import match
    >>> raw = [6, 0, 2, 0, 4]
    >>> data = UML.createData('Matrix', raw)
    >>> mean(data, match.zero)
    [6.0, 4.0, 2.0, 4.0, 4.0]
    """
    return statsBackend(vector, match, 'mean', UML.calculate.mean)

def median(vector, match):
    """
    Fill matched values with the median.

    The calculation of the median will ignore any matched values, but
    all unmatched values must be numeric. If all the values are a match
    the median cannot be calculated.

    Parameters
    ----------
    vector : UML point or feature
        A UML object or UML view object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.

    Returns
    -------
    list
        The vector values with the median replacing the ``match``
        values.

    See Also
    --------
    mean, mode, UML.match

    Examples
    --------
    Match a value.

    >>> raw = [1, 'na', 3, 'na', 5]
    >>> data = UML.createData('Matrix', raw)
    >>> median(data, 'na')
    [1, 3.0, 3, 3.0, 5]

    Match using a function from UML's match module.

    >>> from UML import match
    >>> raw = [6, 0, 2, 0, 4]
    >>> data = UML.createData('Matrix', raw)
    >>> median(data, match.zero)
    [6.0, 4.0, 2.0, 4.0, 4.0]
    """
    return statsBackend(vector, match, 'median', UML.calculate.median)

def mode(vector, match):
    """
    Fill matched values with the mode.

    The calculation of the mode will ignore any matched values. If all
    the values are a match the mean cannot be calculated.

    Parameters
    ----------
    vector : UML point or feature
        A UML object or UML view object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.

    Returns
    -------
    list
        The vector values with the mode replacing the ``match`` values.

    See Also
    --------
    mean, median, UML.match

    Examples
    --------
    Match a value.

    >>> raw = [1, 'na', 1, 'na', 5]
    >>> data = UML.createData('Matrix', raw)
    >>> mode(data, 'na')
    [1, 1.0, 1, 1.0, 5]

    Match using a function from UML's match module.

    >>> from UML import match
    >>> raw = [6, 6, 2, 0, 0]
    >>> data = UML.createData('Matrix', raw)
    >>> mode(data, match.zero)
    [6.0, 6.0, 2.0, 6.0, 6.0]
    """
    return statsBackend(vector, match, 'mode', UML.calculate.mode)

def forwardFill(vector, match):
    """
    Fill matched values with the previous unmatched value.

    Each matching value will be filled with the first non-matching value
    occurring prior to the matched value. An exception will be raised if
    the first value is a match, since there is not a valid value to
    reference.

    Parameters
    ----------
    vector : UML point or feature
       A UML object or UML view object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.

    Returns
    -------
    list
        The vector values with the forward filled values replacing the
        ``match`` values.

    See Also
    --------
    backwardFill, UML.match

    Examples
    --------
    Match a value.

    >>> raw = [1, 'na', 3, 'na', 5]
    >>> data = UML.createData('Matrix', raw)
    >>> forwardFill(data, 'na')
    [1, 1, 3, 3, 5]

    Match using a function from UML's match module.

    >>> from UML import match
    >>> raw = [6, 0, 2, 0, 4]
    >>> data = UML.createData('Matrix', raw)
    >>> forwardFill(data, match.zero)
    [6.0, 6.0, 2.0, 2.0, 4.0]
    """
    match = convertMatchToFunction(match)
    if match(vector[0]):
        msg = directionError('forward fill', vector, 'first')
        raise InvalidArgumentValue(msg)
    ret = []
    for val in vector:
        if match(val):
            ret.append(ret[-1])
        else:
            ret.append(val)
    return ret

def backwardFill(vector, match):
    """
    Fill matched values with the next unmatched value.

    Each matching value will be filled with the first non-matching value
    occurring after to the matched value. An exception will be raised if
    the last value is a match, since there is not a valid value to
    reference.

    Parameters
    ----------
    vector : UML point or feature
        A UML object or UML view object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.

    Returns
    -------
    list
        The vector values with the backward filled values replacing the
        ``match`` values.

    See Also
    --------
    forwardFill, UML.match

    Examples
    --------
    Match a value.

    >>> raw = [1, 'na', 3, 'na', 5]
    >>> data = UML.createData('Matrix', raw)
    >>> backwardFill(data, 'na')
    [1, 3, 3, 5, 5]

    Match using a function from UML's match module.

    >>> from UML import match
    >>> raw = [6, 0, 2, 0, 4]
    >>> data = UML.createData('Matrix', raw)
    >>> backwardFill(data, match.zero)
    [6.0, 2.0, 2.0, 4.0, 4.0]
    """
    match = convertMatchToFunction(match)
    if match(vector[-1]):
        msg = directionError('backward fill', vector, 'last')
        raise InvalidArgumentValue(msg)
    ret = numpy.empty_like(vector, dtype=numpy.object_)
    numValues = len(vector)
    for i, val in enumerate(reversed(vector)):
        idx = numValues - i - 1
        if match(val):
            ret[idx] = ret[idx + 1]
        else:
            ret[idx] = val
    return ret.tolist()

def interpolate(vector, match, **kwarguments):
    """
    Fill matched values with the interpolated value

    The fill value is determined by the piecewise linear interpolant
    returned by numpy.interp. By default, the unmatched values will be
    used as the discrete data points, but additional arguments for
    numpy.interp can be passed as keyword arguments.

    Parameters
    ----------
    vector : UML point or feature
        A UML object or UML view object containing one point or feature.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.
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
    UML.match, numpy.interp

    Examples
    --------
    Match a value.

    >>> raw = [1, 'na', 3, 'na', 5]
    >>> data = UML.createData('Matrix', raw)
    >>> interpolate(data, 'na')
    [1, 2.0, 3, 4.0, 5]

    Match using a function from UML's match module.

    >>> from UML import match
    >>> raw = [6, 0, 4, 0, 2]
    >>> data = UML.createData('Matrix', raw)
    >>> interpolate(data, match.zero)
    [6.0, 5.0, 4.0, 3.0, 2.0]
    """
    match = convertMatchToFunction(match)
    if 'x' in kwarguments:
        msg = "'x' is a disallowed keyword argument because it is "
        msg += "determined by the data in the vector."
        raise TypeError(msg)
    matchedLoc = [i for i, val in enumerate(vector) if match(val)]
    kwarguments['x'] = matchedLoc
    if 'xp' not in kwarguments:
        unmatchedLoc = [i for i, val in enumerate(vector) if not match(val)]
        kwarguments['xp'] = unmatchedLoc
    if 'fp' not in kwarguments:
        unmatchedVals = [val for i, val in enumerate(vector) if not match(val)]
        kwarguments['fp'] = unmatchedVals

    tmpV = numpy.interp(**kwarguments)

    ret = []
    j = 0
    for i, val in enumerate(vector):
        if i in matchedLoc:
            ret.append(tmpV[j])
            j += 1
        else:
            ret.append(val)
    return ret

def kNeighborsRegressor(data, match, **kwarguments):
    """
    Fill matched values with value from skl.kNeighborsRegressor

    The k nearest neighbors are determined by analyzing all points with
    the same unmatched features as the point with missing data. The
    values for the matched feature at those k points are averaged to
    fill the matched value. By default, k=5. This and other parameters
    for skl.kNeighborsRegressor can be adjusted using keyword arguments.

    Parameters
    ----------
    data : UML point or feature
        A UML object or UML view object containing the data.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.
    kwarguments
        Collection of extra key:value argument pairs to pass to
        skl.kNeighborsRegressor.

    Returns
    -------
    list
        The vector values with the kNeighborsRegressor values replacing
        the ``match`` values.

    See Also
    --------
    UML.match

    Examples
    --------
    >>> raw = [[1, 1, 1],
    ...        [1, 1, 1],
    ...        [1, 1, 'na'],
    ...        [2, 2, 2],
    ...        ['na', 2, 2]]
    >>> data = UML.createData('Matrix', raw)
    >>> kNeighborsRegressor(data, 'na', arguments={'n_neighbors': 3})
    Matrix(
        [[  1   1   1  ]
         [  1   1   1  ]
         [  1   1 1.333]
         [  2   2   2  ]
         [1.333 2   2  ]]
        )
    """
    return kNeighborsBackend("skl.KNeighborsRegressor", data, match,
                             **kwarguments)

def kNeighborsClassifier(data, match, **kwarguments):
    """
    Fill matched values with value from skl.kNeighborsClassifier

    The k nearest neighbors are determined by analyzing all points with
    the same unmatched features as the point with missing data. The
    values for the matched feature at those k points are averaged to
    fill the matched value. By default, k=5. This and other parameters
    for skl.kNeighborsClassifier can be adjusted using ``arguments``.

    Parameters
    ----------
    data : UML point or feature
        A UML object or UML view object containing the data.
    match : value or function
        * value - The value which should be filled if it occurs in the
          data.
        * function - Input a value and return True if that value should
          be filled. UML offers common use-case functions in its match
          module.
    kwarguments
        Collection of extra key:value argument pairs to pass to
        skl.kNeighborsClassifier.

    Returns
    -------
    list
        The vector values with the kNeighborsClassifier values replacing
        the ``match`` values.

    See Also
    --------
    UML.match

    Examples
    --------
    >>> raw = [[1, 1, 1],
    ...        [1, 1, 1],
    ...        [1, 1, 'na'],
    ...        [2, 2, 2],
    ...        ['na', 2, 2]]
    >>> data = UML.createData('Matrix', raw)
    >>> kNeighborsClassifier(data, 'na', arguments={'n_neighbors': 3})
    Matrix(
        [[  1   1   1  ]
         [  1   1   1  ]
         [  1   1 1.000]
         [  2   2   2  ]
         [1.000 2   2  ]]
        )
    """
    return kNeighborsBackend("skl.KNeighborsClassifier", data, match,
                             **kwarguments)

############
# Backends #
############

def kNeighborsBackend(method, data, match, **kwarguments):
    """
    Backend for filling using skl kNeighbors functions.
    """
    match = convertMatchToFunction(match)
    tmpDict = {}#store idx, col and values for matching values
    for pID, pt in enumerate(data.points):
        # find matching values in the point
        if anyValues(match)(pt):
            notMatchFts = []
            matchFts = []
            for idx, val in enumerate(pt):
                if match(val):
                    matchFts.append(idx)
                else:
                    notMatchFts.append(idx)
            predictData = data[pID, notMatchFts]

            # make prediction for each feature in the point with matching value
            for fID in matchFts:
                # training data includes not matching features and this feature
                notMatchFts.append(fID)
                trainingData = data[:, notMatchFts]
                # training data includes only points that have valid data at
                # each feature this will also remove the point we are
                # evaluating from the training data
                trainingData.points.delete(anyValues(match), useLog=False)
                pred = UML.trainAndApply(method, trainingData, -1, predictData,
                                         useLog=False, **kwarguments)
                pred = pred[0]
                tmpDict[pID, fID] = pred
                # remove this feature so next prediction will not include it
                del notMatchFts[-1]

    def transform(value, i, j):
        try:
            return tmpDict[(i, j)]
        except KeyError:
            return value

    data.elements.transform(transform, useLog=False)

    return data

def statsBackend(vector, match, funcString, statisticsFunction):
    """
    Backend for filling with a statistics function from UML.calculate.
    """
    match = convertMatchToFunction(match)
    unmatched = [val for val in vector if not match(val)]
    if len(unmatched) == len(vector):
        return list(vector)
    if not unmatched:
        msg = statsExceptionNoMatches(funcString, vector)
        raise InvalidArgumentValue(msg)
    unmatched = UML.createData('List', unmatched, useLog=False)
    stat = statisticsFunction(unmatched)
    if stat is None:
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
        if vector._pointNamesCreated():
            name = vector.points.getName(0)
        if isinstance(vector, UML.data.BaseView):
            index = vector._pStart
    else:
        if vector._featureNamesCreated():
            name = vector.features.getName(0)
        if isinstance(vector, UML.data.BaseView):
            index = vector._fStart

    return name, index

def getLocationMsg(name, index):
    """
    Helper function to format the error message with either a name or index.
    """
    if name is not None:
        location = "'{0}'".format(name)
    else:
        location = "at index '{0}'".format(index)

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
    Generic message when the statisitcs function recieves no values.
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
