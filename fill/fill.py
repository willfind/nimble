"""
TODO
"""
from __future__ import absolute_import

import numpy

import UML
from UML.match import convertMatchToFunction
from UML.match import anyValues
from UML.exceptions import ArgumentException

def factory(match, fill, arguments=None):
    """
    Uses parameters to return a function that accepts a point or feature and
    returns the modified point or feature
    """
    if not hasattr(match, '__call__'):
        match = convertMatchToFunction(match)
    if not hasattr(fill, '__call__'):
        value = fill
        # for consistency use numpy.nan for None and nans
        if value is None or value != value:
            value = numpy.nan
        fill = constant
        arguments = value
    if arguments is None:
        def fillFunction(vector):
            return fill(vector, match)
    else:
        def fillFunction(vector):
            return fill(vector, match, arguments)
    return fillFunction

def constant(vector, match, constantValue):
    """
    Fill matching values with a constant value
    """
    return [constantValue if match(val) else val for val in vector]

def mean(vector, match):
    """
    Calculates the mean of the point or feature, ignoring any matching values,
    and fills matching values with the mean
    """
    return _statsBackend(vector, match, 'mean', UML.calculate.mean)

def median(vector, match):
    """
    Calculates the median of the point or feature, ignoring any matching
    values, and fills matching values with the median
    """
    return _statsBackend(vector, match, 'median', UML.calculate.median)

def mode(vector, match):
    """
    Calculates the mode of the point or feature, ignoring any matching values,
    and fills matching values with the mode
    """
    return _statsBackend(vector, match, 'mode', UML.calculate.mode)

def forwardFill(vector, match):
    """
    Fill matching values with the previous known unmatched value in the point
    or feature
    """
    if match(vector[0]):
        msg = _directionError('forward fill', vector)
        raise ArgumentException(msg)
    ret = []
    for val in vector:
        if match(val):
            ret.append(ret[-1])
        else:
            ret.append(val)
    return ret

def backwardFill(vector, match):
    """
    Fill matched values with the next known unmatched value in the point or
    feature
    """
    if match(vector[-1]):
        msg = _directionError('backward fill', vector)
        raise ArgumentException(msg)
    ret = []
    for val in reversed(vector):
        # prepend since we are working backward
        if match(val):
            ret.insert(0, ret[0])
        else:
            ret.insert(0, val)
    return ret

def interpolate(vector, match, arguments=None):
    """
    Fill matched values with the piecewise linear interpolant returned by
    numpy.interp
    When arguments is None, the unmatched values will be used as the discrete
    data points. Otherwise the arguments for numpy.interp can be passed as
    dictionary
    """
    x = [i for i, val in enumerate(vector) if match(val)]
    if arguments is not None:
        try:
            tmpArguments = arguments.copy()
            tmpArguments['x'] = x
        except Exception:
            msg = 'for fill.interpolate, arguments must be None or a dict.'
            raise ArgumentException(msg)
    else:
        xp = [i for i, val in enumerate(vector) if not match(val)]
        fp = [val for i, val in enumerate(vector) if not match(val)]
        tmpArguments = {'x': x, 'xp': xp, 'fp': fp}
    tmpV = numpy.interp(**tmpArguments)
    ret = []
    j = 0
    for i, val in enumerate(vector):
        if i in x:
            ret.append(tmpV[j])
            j += 1
        else:
            ret.append(val)
    return ret

def kNeighborsRegressor(data, match, arguments=None):
    return kNeighborsBackend("skl.KNeighborsRegressor", data, match, arguments)

def kNeighborsClassifier(data, match, arguments=None):
    return kNeighborsBackend("skl.KNeighborsClassifier", data, match, arguments)

def kNeighborsBackend(method, data, match, arguments):
    if arguments is None:
        arguments = {}

    tmpDict = {}#store idx, col and values for matching values
    for pID, pt in enumerate(data.pointIterator()):
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
                trainingData.deletePoints(anyValues(match))
                pred = UML.trainAndApply(method, trainingData, -1, predictData,
                                         arguments=arguments)
                pred = pred[0]
                tmpDict[pID, fID] = pred
                # remove this feature so next prediction will not include it
                del notMatchFts[-1]

    def transform(value, i, j):
        try:
            return tmpDict[(i, j)]
        except KeyError:
            return value

    data.transformEachElement(transform)

    return data

###########
# Helpers #
###########

def _statsBackend(vector, match, funcString, statisticsFunction):
    unmatched = [val for val in vector if not match(val)]
    if len(unmatched) == len(vector):
        return list(vector)
    if not unmatched:
        msg = _statsError1(funcString, vector)
        raise ArgumentException(msg)
    unmatched = UML.createData('List', unmatched)
    stat = statisticsFunction(unmatched)
    if stat is None:
        msg = _statsError2(funcString, vector)
        raise ArgumentException(msg)

    return constant(vector, match, stat)

def _getAxis(vector):
    return 'point' if vector.points == 1 else 'feature'

def _getNameAndIndex(axis, vector):
    name = None
    index = 0
    if axis == 'point':
        if vector._pointNamesCreated():
            name = vector.getPointName(0)
        if isinstance(vector, UML.data.BaseView):
            index = vector._pStart
    else:
        if vector._featureNamesCreated():
            name = vector.getFeatureName(0)
        if isinstance(vector, UML.data.BaseView):
            index = vector._fStart

    return name, index

def _getLocationMsg(name, index):
    if name is not None:
        location = "'{0}'".format(name)
    else:
        location = "at index '{0}'".format(index)

    return location

def _errorMsgFormatter(msg, funcString, vector):
    axis = _getAxis(vector)
    name, index = _getNameAndIndex(axis, vector)
    location = _getLocationMsg(name, index)

    return msg.format(funcString=funcString, axis=axis, location=location)

def _statsError1(funcString, vector):
    msg = "Cannot calculate {funcString}. The {funcString} is calculated "
    msg += "using only unmatched values. All values for the {axis} {location} "
    msg += "returned a match."

    return _errorMsgFormatter(msg, funcString, vector)

def _statsError2(funcString, vector):
    msg = "Cannot calculate {funcString}. The {axis} {location} "
    msg += "contains non-numeric values or is all NaN values"

    return _errorMsgFormatter(msg, funcString, vector)

def _directionError(funcString, vector):
    msg = "Unable to provide a {funcString} value for the {axis} {location} "
    msg += "because the first value is a match"

    return _errorMsgFormatter(msg, funcString, vector)
