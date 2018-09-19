import UML
import numpy

from UML.match import convertMatchToFunction
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
    unmatched = UML.createData('List', [val for val in vector if not match(val)])
    if len(unmatched) == len(vector):
        return vector
    if len(unmatched) == 0:
        msg = "Cannot calculate mean. The mean is calculated based on "
        msg += "unmatched values and all values for the {0} at index {1} "
        msg += "returned a match"
        # return so msg can be formatted before being raised
        return ArgumentException(msg)
    mean = UML.calculate.mean(unmatched)
    return [mean if match(val) else val for val in vector]


def median(vector, match):
    """
    Calculates the median of the point or feature, ignoring any matching
    values, and fills matching values with the median
    """
    unmatched = UML.createData('List', [val for val in vector if not match(val)])
    if len(unmatched) == len(vector):
        return vector
    if len(unmatched) == 0:
        msg = "Cannot calculate median. The median is calculated based on "
        msg += "unmatched values and all values for the {0} at index {1} "
        msg += "returned a match"
        # return so msg can be formatted before being raised
        return ArgumentException(msg)
    median = UML.calculate.median(unmatched)
    return [median if match(val) else val for val in vector]

def mode(vector, match):
    """
    Calculates the mode of the point or feature, ignoring any matching values,
    and fills matching values with the mode
    """
    unmatched = UML.createData('List', [val for val in vector if not match(val)])
    if len(unmatched) == len(vector):
        return vector
    if len(unmatched) == 0:
        msg = "Cannot calculate mode. The mode is calculated based on "
        msg += "unmatched values and all values for the {0} at index {1} "
        msg += "returned a match"
        # return so msg can be formatted before being raised
        return ArgumentException(msg)
    mode = UML.calculate.mode(unmatched)
    return [mode if match(val) else val for val in vector]

def forward(vector, match):
    """
    Fill matching values with the previous known unmatched value in the point
    or feature
    """
    if match(vector[0]):
        msg = "Unable to provide a forward fill value for the {0} at "
        msg += "index {1} because the first value is a match"
        # return so msg can be formatted before being raised
        return ArgumentException(msg)
    ret = []
    for v in vector:
        if match(v):
            ret.append(ret[-1])
        else:
            ret.append(v)
    return ret

def backward(vector, match):
    """
    Fill matched values with the next known unmatched value in the point or
    feature
    """
    if match(vector[len(vector) - 1]):
        msg = "Unable to provide a backward fill value for the {0} at "
        msg += "index {1} because the last value is a match"
        # return so msg can be formatted before being raised
        return ArgumentException(msg)
    ret = []
    # reverse to work backward
    for v in reversed(vector):
        # prepend since we are working backward
        if match(v):
            ret.insert(0, ret[0])
        else:
            ret.insert(0, v)
    return ret

def interpolate(vector, match, arguments=None):
    """
    Fill matched values with the piecewise linear interpolant returned by
    numpy.interp
    When arguments is None, the unmatched values will be used as the discrete
    data points. Otherwise the arguments for numpy.interp can be passed as
    dictionary
    """
    x = [i for i,v in enumerate(vector) if match(v)]
    if arguments is not None:
        try:
            tmpArguments = arguments.copy()
            tmpArguments['x'] = x
        except Exception:
            msg = 'for fill.interpolate, arguments must be None or a dict.'
            raise ArgumentException(msg)
    else:
        xp = [i for i,v in enumerate(vector) if not match(v)]
        fp = [v for i,v in enumerate(vector) if not match(v)]
        tmpArguments = {'x': x, 'xp': xp, 'fp': fp}
    tmpV = numpy.interp(**tmpArguments)
    ret = []
    j = 0
    for i in range(len(vector)):
        if i in x:
            ret.append(tmpV[j])
            j += 1
        else:
            ret.append(vector[i])
    return ret
